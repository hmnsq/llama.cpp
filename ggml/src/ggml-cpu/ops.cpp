#અન્તર્ભાવ "ops.h"

#અન્તર્ભાવ "ggml-cpu.h"
#અન્તર્ભાવ "ggml-impl.h"
#અન્તર્ભાવ "binary-ops.h"
#અન્તર્ભાવ "simd-gemm.h"
#અન્તર્ભાવ "ggml.h"
#અન્તર્ભાવ "unary-ops.h"
#અન્તર્ભાવ "vec.h"

#અન્તર્ભાવ <algorithm>
#અન્તર્ભાવ <cfloat>
#અન્તર્ભાવ <cmath>

// ggml_compute_forward_dup

સ્થિર શૂન્ય ggml_compute_forward_dup_same_cont(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));
    GGML_ASSERT(src0->type == dst->type);

    અચલ size_t nb0 = ggml_type_size(src0->type);

    અચલ પૂર્ણાંક ith = params->ith; // thread index
    અચલ પૂર્ણાંક nth = params->nth; // number of threads

    // parallelize by blocks
    અચલ પૂર્ણાંક nk = ggml_nelements(src0)/ggml_blck_size(src0->type);
    અચલ પૂર્ણાંક dr = (nk + nth - 1) / nth;
    અચલ પૂર્ણાંક k0 = dr * ith;
    અચલ પૂર્ણાંક k1 = MIN(k0 + dr, nk);

    જો (k0 < k1) {
        memcpy(
            ((અક્ષર *)  dst->data + k0*nb0),
            ((અક્ષર *) src0->data + k0*nb0),
            (k1 - k0) * nb0);
    }
}

template<typename src_t, typename dst_t>
સ્થિર શૂન્ય ggml_compute_forward_dup_flt(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(!ggml_is_quantized(src0->type) && !ggml_is_quantized(dst->type));

    GGML_TENSOR_UNARY_OP_LOCALS

    અચલ પૂર્ણાંક ith = params->ith; // thread index
    અચલ પૂર્ણાંક nth = params->nth; // number of threads

    // parallelize by rows
    અચલ પૂર્ણાંક nr = ne01;
    // number of rows per thread
    અચલ પૂર્ણાંક dr = (nr + nth - 1) / nth;
    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr * ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    // case: type & row size equal
    જો (src0->type == dst->type &&
        ne00 == ne0 &&
        nb00 == ggml_type_size(src0->type) && nb0 == ggml_type_size(dst->type)) {
        // copy by rows
        અચલ size_t rs = ne00*nb00;
        માટે (int64_t i03 = 0; i03 < ne03; i03++) {
            માટે (int64_t i02 = 0; i02 < ne02; i02++) {
                માટે (int64_t i01 = ir0; i01 < ir1; i01++) {
                    memcpy(
                        ((અક્ષર *)  dst->data + i01*nb1  + i02*nb2  + i03*nb3),
                        ((અક્ષર *) src0->data + i01*nb01 + i02*nb02 + i03*nb03),
                        rs);
                }
            }
        }
        પાછા;
    }

    // case: dst tensor is contiguous
    જો (ggml_is_contiguous(dst)) {
        જો (nb00 == કદ(src_t)) {
            જો constexpr (std::is_same_v<dst_t, src_t>) {
                // same type
                size_t id = 0;
                અચલ size_t rs = ne00 * nb00;
                અક્ષર * dst_ptr = (અક્ષર *) dst->data;

                માટે (પૂર્ણાંક i03 = 0; i03 < ne03; i03++) {
                    માટે (પૂર્ણાંક i02 = 0; i02 < ne02; i02++) {
                        id += rs * ir0;
                        માટે (પૂર્ણાંક i01 = ir0; i01 < ir1; i01++) {
                            અચલ અક્ષર * src0_ptr = (અક્ષર *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
                            memcpy(dst_ptr + id, src0_ptr, rs);
                            id += rs;
                        }
                        id += rs * (ne01 - ir1);
                    }
                }
            } નહીંતર {
                // casting between non-quantized types
                size_t id = 0;
                dst_t * dst_ptr = (dst_t *) dst->data;

                માટે (પૂર્ણાંક i03 = 0; i03 < ne03; i03++) {
                    માટે (પૂર્ણાંક i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        માટે (પૂર્ણાંક i01 = ir0; i01 < ir1; i01++) {
                            અચલ src_t * src0_ptr = (src_t *) ((અક્ષર *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
                            માટે (પૂર્ણાંક i00 = 0; i00 < ne00; i00++) {
                                દશાંશ tmp = type_conversion_table<src_t>::to_f32(src0_ptr[i00]);
                                dst_ptr[id] = type_conversion_table<dst_t>::from_f32(tmp);
                                id++;
                            }
                        }
                        id += ne00 * (ne01 - ir1);
                    }
                }
            }
        } નહીંતર {
            //printf("%s: this is not optimal - fix me\n", __func__);

            size_t id = 0;
            dst_t * dst_ptr = (dst_t *) dst->data;

            માટે (પૂર્ણાંક i03 = 0; i03 < ne03; i03++) {
                માટે (પૂર્ણાંક i02 = 0; i02 < ne02; i02++) {
                    id += ne00 * ir0;
                    માટે (પૂર્ણાંક i01 = ir0; i01 < ir1; i01++) {
                        માટે (પૂર્ણાંક i00 = 0; i00 < ne00; i00++) {
                            અચલ src_t * src0_ptr = (src_t *) ((અક્ષર *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);

                            દશાંશ tmp = type_conversion_table<src_t>::to_f32(*src0_ptr);
                            dst_ptr[id] = type_conversion_table<dst_t>::from_f32(tmp);
                            id++;
                        }
                    }
                    id += ne00 * (ne01 - ir1);
                }
            }
        }
        પાછા;
    }

    // dst counters
    int64_t i10 = 0;
    int64_t i11 = 0;
    int64_t i12 = 0;
    int64_t i13 = 0;

    જો constexpr (std::is_same_v<dst_t, src_t>) {
        માટે (int64_t i03 = 0; i03 < ne03; i03++) {
            માટે (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                જ્યારે (i10 >= ne0) {
                    i10 -= ne0;
                    જો (++i11 == ne1) {
                        i11 = 0;
                        જો (++i12 == ne2) {
                            i12 = 0;
                            જો (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                માટે (int64_t i01 = ir0; i01 < ir1; i01++) {
                    માટે (int64_t i00 = 0; i00 < ne00; i00++) {
                        અચલ અક્ષર * src0_ptr = ((અક્ષર *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              અક્ષર * dst_ptr  = ((અક્ષર *)  dst->data + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                        memcpy(dst_ptr, src0_ptr, કદ(dst_t));

                        જો (++i10 == ne00) {
                            i10 = 0;
                            જો (++i11 == ne01) {
                                i11 = 0;
                                જો (++i12 == ne02) {
                                    i12 = 0;
                                    જો (++i13 == ne03) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                જ્યારે (i10 >= ne0) {
                    i10 -= ne0;
                    જો (++i11 == ne1) {
                        i11 = 0;
                        જો (++i12 == ne2) {
                            i12 = 0;
                            જો (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
            }
        }

    } નહીંતર {
        માટે (int64_t i03 = 0; i03 < ne03; i03++) {
            માટે (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                જ્યારે (i10 >= ne0) {
                    i10 -= ne0;
                    જો (++i11 == ne1) {
                        i11 = 0;
                        જો (++i12 == ne2) {
                            i12 = 0;
                            જો (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                માટે (int64_t i01 = ir0; i01 < ir1; i01++) {
                    માટે (int64_t i00 = 0; i00 < ne00; i00++) {
                        અચલ અક્ષર * src0_ptr = ((અક્ષર *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              અક્ષર * dst_ptr  = ((અક્ષર *)  dst->data + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                        દશાંશ tmp = type_conversion_table<src_t>::to_f32(*(અચલ src_t *) src0_ptr);
                        *(dst_t *) dst_ptr = type_conversion_table<dst_t>::from_f32(tmp);

                        જો (++i10 == ne0) {
                            i10 = 0;
                            જો (++i11 == ne1) {
                                i11 = 0;
                                જો (++i12 == ne2) {
                                    i12 = 0;
                                    જો (++i13 == ne3) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                જ્યારે (i10 >= ne0) {
                    i10 -= ne0;
                    જો (++i11 == ne1) {
                        i11 = 0;
                        જો (++i12 == ne2) {
                            i12 = 0;
                            જો (++i13 == ne3) {
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
સ્થિર શૂન્ય ggml_compute_forward_dup_to_q(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(!ggml_is_quantized(src0->type));

    GGML_TENSOR_UNARY_OP_LOCALS

    અચલ પૂર્ણાંક ith = params->ith; // thread index
    અચલ પૂર્ણાંક nth = params->nth; // number of threads

    // parallelize by rows
    અચલ પૂર્ણાંક nr = ne01;
    // number of rows per thread
    અચલ પૂર્ણાંક dr = (nr + nth - 1) / nth;
    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr * ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    જો (ggml_is_contiguous(dst) &&
            nb00 == કદ(src_t) &&
            ggml_get_type_traits_cpu(dst->type)->from_float) {
        // casting non-quantized types --> intermediate f32 --> quantized
        ggml_from_float_t અચલ quantize_row_q = ggml_get_type_traits_cpu(dst->type)->from_float;
        દશાંશ * src0_f32 = (દશાંશ *) params->wdata + (ne00 + CACHE_લીટી_SIZE_F32) * ith;

        size_t id = 0;
        size_t rs = nb0 * (ne00 / ggml_blck_size(dst->type));
        અક્ષર * dst_ptr = (અક્ષર *) dst->data;

        માટે (પૂર્ણાંક i03 = 0; i03 < ne03; i03++) {
            માટે (પૂર્ણાંક i02 = 0; i02 < ne02; i02++) {
                id += rs * ir0;
                માટે (પૂર્ણાંક i01 = ir0; i01 < ir1; i01++) {
                    અચલ src_t * src0_ptr = (src_t *) ((અક્ષર *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);

                    માટે (પૂર્ણાંક i00 = 0; i00 < ne00; i00++) {
                        src0_f32[i00] = type_conversion_table<src_t>::to_f32(src0_ptr[i00]);
                    }

                    quantize_row_q(src0_f32, dst_ptr + id, ne00);
                    id += rs;
                }
                id += rs * (ne01 - ir1);
            }
        }
    } નહીંતર {
        // printf("%s %s\n", ggml_type_name(src0->type), ggml_type_name(dst->type));
        GGML_ABORT("not implemented");
    }
}

// A simplified version of ggml_compute_forward_dup that doesn't do float upcasting, and just plain old memcpy.
સ્થિર શૂન્ય ggml_compute_forward_dup_bytes(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {
    અચલ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(src0->type == dst->type);

    GGML_TENSOR_UNARY_OP_LOCALS;

    જો (ggml_is_contiguous(src0) && ggml_is_contiguous(dst)) {
        ggml_compute_forward_dup_same_cont(params, dst);
        પાછા;
    }

    અચલ size_t type_size = ggml_type_size(src0->type);

    અચલ પૂર્ણાંક ith = params->ith; // thread index
    અચલ પૂર્ણાંક nth = params->nth; // number of threads

    // parallelize by rows
    અચલ પૂર્ણાંક nr = ne01;
    // number of rows per thread
    અચલ પૂર્ણાંક dr = (nr + nth - 1) / nth;
    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr * ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    જો (src0->type == dst->type &&
        ggml_are_same_shape(src0, dst) &&
        nb00 == type_size && nb0 == type_size) {
        // copy by rows
        અચલ size_t rs = ggml_row_size(src0->type, ne00);
        માટે (int64_t i03 = 0; i03 < ne03; i03++) {
            માટે (int64_t i02 = 0; i02 < ne02; i02++) {
                માટે (int64_t i01 = ir0; i01 < ir1; i01++) {
                    memcpy(
                        ((અક્ષર *)  dst->data + i01*nb1  + i02*nb2  + i03*nb3),
                        ((અક્ષર *) src0->data + i01*nb01 + i02*nb02 + i03*nb03),
                        rs);
                }
            }
        }
        પાછા;
    }

    જો (ggml_is_contiguous(dst)) {
        size_t id = 0;
        અક્ષર * dst_ptr = (અક્ષર *) dst->data;
        અચલ size_t rs = ne00 * type_size;

        જો (nb00 == type_size) {
            // src0 is contiguous on first dimension, copy by rows
            માટે (int64_t i03 = 0; i03 < ne03; i03++) {
                માટે (int64_t i02 = 0; i02 < ne02; i02++) {
                    id += rs * ir0;
                    માટે (int64_t i01 = ir0; i01 < ir1; i01++) {
                        અચલ અક્ષર * src0_ptr = (અક્ષર *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
                        memcpy(dst_ptr + id, src0_ptr, rs);
                        id += rs;
                    }
                    id += rs * (ne01 - ir1);
                }
            }
        } નહીંતર {
            //printf("%s: this is not optimal - fix me\n", __func__);

            માટે (int64_t i03 = 0; i03 < ne03; i03++) {
                માટે (int64_t i02 = 0; i02 < ne02; i02++) {
                    id += rs * ir0;
                    માટે (int64_t i01 = ir0; i01 < ir1; i01++) {
                        માટે (int64_t i00 = 0; i00 < ne00; i00++) {
                            અચલ અક્ષર * src0_ptr = (અક્ષર *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03;
                            memcpy(dst_ptr + id, src0_ptr, type_size);

                            id += type_size;
                        }
                    }
                    id += rs * (ne01 - ir1);
                }
            }
        }

        પાછા;
    }

    // dst counters
    int64_t k10 = 0;
    int64_t i11 = 0;
    int64_t i12 = 0;
    int64_t i13 = 0;

    // number of blocks in a row
    અચલ int64_t nk00 = ne00 / ggml_blck_size(src0->type);
    અચલ int64_t nk0  = ne0  / ggml_blck_size(dst->type);

    માટે (int64_t i03 = 0; i03 < ne03; i03++) {
        માટે (int64_t i02 = 0; i02 < ne02; i02++) {
            k10 += nk00 * ir0;
            જ્યારે (k10 >= nk0) {
                k10 -= nk0;
                જો (++i11 == ne1) {
                    i11 = 0;
                    જો (++i12 == ne2) {
                        i12 = 0;
                        જો (++i13 == ne3) {
                            i13 = 0;
                        }
                    }
                }
            }
            માટે (int64_t i01 = ir0; i01 < ir1; i01++) {
                માટે (int64_t k00 = 0; k00 < nk00; k00++) {
                    અચલ અક્ષર * src0_ptr = ((અક્ષર *) src0->data + k00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                          અક્ષર * dst_ptr  = ((અક્ષર *)  dst->data + k10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                    memcpy(dst_ptr, src0_ptr, type_size);

                    જો (++k10 == nk0) {
                        k10 = 0;
                        જો (++i11 == ne1) {
                            i11 = 0;
                            જો (++i12 == ne2) {
                                i12 = 0;
                                જો (++i13 == ne3) {
                                    i13 = 0;
                                }
                            }
                        }
                    }
                }
            }
            k10 += nk00 * (ne01 - ir1);
            જ્યારે (k10 >= nk0) {
                k10 -= nk0;
                જો (++i11 == ne1) {
                    i11 = 0;
                    જો (++i12 == ne2) {
                        i12 = 0;
                        જો (++i13 == ne3) {
                            i13 = 0;
                        }
                    }
                }
            }
        }
    }
}

સ્થિર શૂન્ય ggml_compute_forward_dup_from_q(
        અચલ ggml_compute_params * params,
              ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_દ્વિઅંકી_OP_LOCALS

    અચલ ggml_type type = src0->type;
    ggml_to_float_t અચલ dequantize_row_q = ggml_get_type_traits(type)->to_float;

    size_t qk = ggml_blck_size(type);
    અચલ int64_t nr = ggml_nelements(src1) / qk;

    // destination must be contiguous in the first dimension
    GGML_ASSERT(nb10 == ggml_type_size(dst->type));
    // must either have first dimension large enough to hold a row, or fully contiguous
    GGML_ASSERT((ne10 % qk) == 0 || ggml_is_contiguous(dst));

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr*ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    માટે (int64_t ir = ir0; ir < ir1; ++ir) {

        uint32_t i = ir * qk;

        અચલ int64_t i03 = i/(ne00 * ne01 * ne02);
        અચલ int64_t i02 = (i - i03*ne00*ne01*ne02 )/ (ne00*ne01);
        અચલ int64_t i01 = (i - i03*ne00*ne01*ne02  -  i02*ne01*ne00) / ne00;
        અચલ int64_t i00 = i - i03*ne00*ne01*ne02 - i02*ne01*ne00 - i01*ne00;
        અચલ int64_t x_offset = (i00/qk)*nb00 + i01*nb01 + i02*nb02 + i03 * nb03;

        અચલ int64_t i13 = i/(ne10 * ne11 * ne12);
        અચલ int64_t i12 = (i - i13*ne10*ne11*ne12) / (ne10*ne11);
        અચલ int64_t i11 = (i - i13*ne10*ne11*ne12 - i12*ne10*ne11) / ne10;
        અચલ int64_t i10 = i - i13*ne10*ne11*ne12 - i12*ne10*ne11 - i11*ne10;
        અચલ int64_t dst_offset = i10*nb10 + i11*nb11 + i12*nb12 + i13*nb13;

        dequantize_row_q(
                (અચલ શૂન્ય *) ((અક્ષર *) src0->data + x_offset),
                     (દશાંશ *) ((અક્ષર *)  dst->data + dst_offset), qk);
    }
}

શૂન્ય ggml_compute_forward_dup(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    જો (src0->type == dst->type) {
        ggml_compute_forward_dup_bytes(params, dst);
        પાછા;
    }

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F16:
            {
                /**/ જો (dst->type == GGML_TYPE_F16)  ggml_compute_forward_dup_flt<ggml_fp16_t, ggml_fp16_t>(params, dst);
                નહીંતર જો (dst->type == GGML_TYPE_BF16) ggml_compute_forward_dup_flt<ggml_fp16_t, ggml_bf16_t>(params, dst);
                નહીંતર જો (dst->type == GGML_TYPE_F32)  ggml_compute_forward_dup_flt<ggml_fp16_t, દશાંશ      >(params, dst);
                નહીંતર ggml_compute_forward_dup_to_q<ggml_fp16_t>(params, dst);
            } તોડો;
        સ્થિતિ GGML_TYPE_BF16:
            {
                /**/ જો (dst->type == GGML_TYPE_F16)  ggml_compute_forward_dup_flt<ggml_bf16_t, ggml_fp16_t>(params, dst);
                નહીંતર જો (dst->type == GGML_TYPE_BF16) ggml_compute_forward_dup_flt<ggml_bf16_t, ggml_bf16_t>(params, dst);
                નહીંતર જો (dst->type == GGML_TYPE_F32)  ggml_compute_forward_dup_flt<ggml_bf16_t, દશાંશ      >(params, dst);
                નહીંતર ggml_compute_forward_dup_to_q<ggml_bf16_t>(params, dst);
            } તોડો;
        સ્થિતિ GGML_TYPE_F32:
            {
                /**/ જો (dst->type == GGML_TYPE_F16)  ggml_compute_forward_dup_flt<દશાંશ, ggml_fp16_t>(params, dst);
                નહીંતર જો (dst->type == GGML_TYPE_BF16) ggml_compute_forward_dup_flt<દશાંશ, ggml_bf16_t>(params, dst);
                નહીંતર જો (dst->type == GGML_TYPE_F32)  ggml_compute_forward_dup_flt<દશાંશ, દશાંશ      >(params, dst);
                નહીંતર જો (dst->type == GGML_TYPE_I32)  ggml_compute_forward_dup_flt<દશાંશ, int32_t    >(params, dst);
                નહીંતર ggml_compute_forward_dup_to_q<દશાંશ>(params, dst);
            } તોડો;
        સ્થિતિ GGML_TYPE_I32:
            {
                જો (dst->type == GGML_TYPE_F32) ggml_compute_forward_dup_flt<int32_t, દશાંશ>(params, dst);
                નહીંતર GGML_ABORT("not implemented");
            } તોડો;
        મૂળભૂત:
            {
                જો (ggml_is_quantized(src0->type) && dst->type == GGML_TYPE_F32) {
                    ggml_compute_forward_dup_from_q(params, dst);
                    તોડો;
                }
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_add

સ્થિર શૂન્ય ggml_compute_forward_add_q_f32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

    અચલ પૂર્ણાંક nr  = ggml_nrows(src0);

    GGML_TENSOR_દ્વિઅંકી_OP_LOCALS

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ ggml_type type = src0->type;
    અચલ ggml_type dtype = dst->type;
    ggml_to_float_t અચલ dequantize_row_q = ggml_get_type_traits(type)->to_float;
    ggml_from_float_t અચલ quantize_row_q = ggml_get_type_traits_cpu(dtype)->from_float;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == કદ(દશાંશ));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ggml_is_quantized(src0->type));
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    // rows per thread
    અચલ પૂર્ણાંક dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr*ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    દશાંશ * wdata = (દશાંશ *) params->wdata + (ne00 + CACHE_લીટી_SIZE_F32) * ith;

    માટે (પૂર્ણાંક ir = ir0; ir < ir1; ++ir) {
        // src0 indices
        અચલ પૂર્ણાંક i03 = ir/(ne02*ne01);
        અચલ પૂર્ણાંક i02 = (ir - i03*ne02*ne01)/ne01;
        અચલ પૂર્ણાંક i01 = (ir - i03*ne02*ne01 - i02*ne01);

        // src1 and dst are same shape as src0 => same indices
        અચલ પૂર્ણાંક i13 = i03;
        અચલ પૂર્ણાંક i12 = i02;
        અચલ પૂર્ણાંક i11 = i01;

        અચલ પૂર્ણાંક i3 = i03;
        અચલ પૂર્ણાંક i2 = i02;
        અચલ પૂર્ણાંક i1 = i01;

        શૂન્ય  * src0_row = (શૂન્ય *) ((અક્ષર *) src0->data + (i01*nb01 + i02*nb02 + i03*nb03));
        દશાંશ * src1_row = (દશાંશ *)((અક્ષર *) src1->data + (i11*nb11 + i12*nb12 + i13*nb13));
        શૂન્ય  * dst_row  = (શૂન્ય *) ((અક્ષર *)  dst->data + ( i1*nb1  +  i2*nb2  +  i3*nb3));

        પ્રતિજ્ઞા(ne00 % 32 == 0);

        // unquantize row from src0 to temp buffer
        dequantize_row_q(src0_row, wdata, ne00);
        // add src1
        ggml_vec_acc_f32(ne00, wdata, src1_row);
        // quantize row to dst
        જો (quantize_row_q != ખાલી) {
            quantize_row_q(wdata, dst_row, ne00);
        } નહીંતર {
            memcpy(dst_row, wdata, ne0*nb0);
        }
    }
}

શૂન્ય ggml_compute_forward_add(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
        સ્થિતિ GGML_TYPE_F16:
        સ્થિતિ GGML_TYPE_BF16:
            {
                ggml_compute_forward_add_non_quantized(params, dst);
            } તોડો;
        સ્થિતિ GGML_TYPE_Q1_0:
        સ્થિતિ GGML_TYPE_Q2_0:
        સ્થિતિ GGML_TYPE_Q4_0:
        સ્થિતિ GGML_TYPE_Q4_1:
        સ્થિતિ GGML_TYPE_Q5_0:
        સ્થિતિ GGML_TYPE_Q5_1:
        સ્થિતિ GGML_TYPE_Q8_0:
        સ્થિતિ GGML_TYPE_MXFP4:
        સ્થિતિ GGML_TYPE_NVFP4:
        સ્થિતિ GGML_TYPE_Q2_K:
        સ્થિતિ GGML_TYPE_Q3_K:
        સ્થિતિ GGML_TYPE_Q4_K:
        સ્થિતિ GGML_TYPE_Q5_K:
        સ્થિતિ GGML_TYPE_Q6_K:
        સ્થિતિ GGML_TYPE_TQ1_0:
        સ્થિતિ GGML_TYPE_TQ2_0:
        સ્થિતિ GGML_TYPE_IQ2_XXS:
        સ્થિતિ GGML_TYPE_IQ2_XS:
        સ્થિતિ GGML_TYPE_IQ3_XXS:
        સ્થિતિ GGML_TYPE_IQ1_S:
        સ્થિતિ GGML_TYPE_IQ1_M:
        સ્થિતિ GGML_TYPE_IQ4_NL:
        સ્થિતિ GGML_TYPE_IQ4_XS:
        સ્થિતિ GGML_TYPE_IQ3_S:
        સ્થિતિ GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_add_q_f32(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_add_id

સ્થિર શૂન્ય ggml_compute_forward_add_id_f32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];
    અચલ ggml_tensor * src2 = dst->src[2];

    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src2->type == GGML_TYPE_I32);

    GGML_ASSERT(src0->nb[0] == કદ(દશાંશ));
    GGML_ASSERT(src1->nb[0] == કદ(દશાંશ));

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક nr  = ggml_nrows(src0);

    GGML_TENSOR_TERNARY_OP_LOCALS

    GGML_ASSERT( nb0 == કદ(દશાંશ));
    GGML_ASSERT(nb10 == કદ(દશાંશ));

    // rows per thread
    અચલ પૂર્ણાંક dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr*ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    માટે (પૂર્ણાંક ir = ir0; ir < ir1; ++ir) {
        // src0 indices
        અચલ પૂર્ણાંક i3 = ir/(ne2*ne1);
        અચલ પૂર્ણાંક i2 = (ir - i3*ne2*ne1)/ne1;
        અચલ પૂર્ણાંક i1 = (ir - i3*ne2*ne1 - i2*ne1);

        // src1 indices
        અચલ પૂર્ણાંક i11 = *(int32_t *) ((અક્ષર *) src2->data + i1*nb20 + i2*nb21);

        GGML_ASSERT(i11 >= 0 && i11 < ne11);

        ggml_vec_add_f32(ne0,
                (દશાંશ *) ((અક્ષર *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 ),
                (દશાંશ *) ((અક્ષર *) src0->data + i3*nb03 + i2*nb02 + i1*nb01),
                (દશાંશ *) ((અક્ષર *) src1->data + i11*nb11));
    }
}

શૂન્ય ggml_compute_forward_add_id(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_add_id_f32(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("unsupported type for ggml_compute_forward_add_id: %s", ggml_type_name(src0->type));
            }
    }
}

// ggml_compute_forward_add1

સ્થિર શૂન્ય ggml_compute_forward_add1_f32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == કદ(દશાંશ));
    GGML_ASSERT(nb00 == કદ(દશાંશ));

    // rows per thread
    અચલ પૂર્ણાંક dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr*ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    માટે (પૂર્ણાંક ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        અચલ પૂર્ણાંક i3 = ir/(ne2*ne1);
        અચલ પૂર્ણાંક i2 = (ir - i3*ne2*ne1)/ne1;
        અચલ પૂર્ણાંક i1 = (ir - i3*ne2*ne1 - i2*ne1);

#ifdef GGML_USE_ACCELERATE
        GGML_UNUSED(ggml_vec_add1_f32);

        vDSP_vadd(
                (દશાંશ *) ((અક્ષર *) src0->data + i3*nb03 + i2*nb02 + i1*nb01), 1,
                (દશાંશ *) ((અક્ષર *) src1->data), 0,
                (દશાંશ *) ((અક્ષર *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 ), 1,
                ne0);
#નહીંતર
        ggml_vec_add1_f32(ne0,
                (દશાંશ *) ((અક્ષર *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 ),
                (દશાંશ *) ((અક્ષર *) src0->data + i3*nb03 + i2*nb02 + i1*nb01),
               *(દશાંશ *) src1->data);
#endif
    }
}

સ્થિર શૂન્ય ggml_compute_forward_add1_f16_f32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    અચલ દશાંશ v = *(દશાંશ *) src1->data;

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F16);

    GGML_ASSERT( nb0 == કદ(ggml_fp16_t));
    GGML_ASSERT(nb00 == કદ(ggml_fp16_t));

    // rows per thread
    અચલ પૂર્ણાંક dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr*ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    માટે (પૂર્ણાંક ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        અચલ પૂર્ણાંક i3 = ir/(ne2*ne1);
        અચલ પૂર્ણાંક i2 = (ir - i3*ne2*ne1)/ne1;
        અચલ પૂર્ણાંક i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((અક્ષર *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((અક્ષર *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
        માટે (પૂર્ણાંક i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(src0_ptr[i]) + v);
        }
    }
}

સ્થિર શૂન્ય ggml_compute_forward_add1_f16_f16(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    અચલ દશાંશ v = GGML_CPU_FP16_TO_FP32(*(ggml_fp16_t *) src1->data);

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type  == GGML_TYPE_F16);

    GGML_ASSERT( nb0 == કદ(ggml_fp16_t));
    GGML_ASSERT(nb00 == કદ(ggml_fp16_t));

    // rows per thread
    અચલ પૂર્ણાંક dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr*ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    માટે (પૂર્ણાંક ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        અચલ પૂર્ણાંક i3 = ir/(ne2*ne1);
        અચલ પૂર્ણાંક i2 = (ir - i3*ne2*ne1)/ne1;
        અચલ પૂર્ણાંક i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((અક્ષર *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((અક્ષર *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
        માટે (પૂર્ણાંક i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(src0_ptr[i]) + v);
        }
    }
}

સ્થિર શૂન્ય ggml_compute_forward_add1_q_f32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    અચલ દશાંશ v = *(દશાંશ *) src1->data;

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    અચલ ggml_type type = src0->type;
    ggml_to_float_t અચલ dequantize_row_q = ggml_get_type_traits(type)->to_float;
    ggml_from_float_t અચલ quantize_row_q = ggml_get_type_traits_cpu(type)->from_float;

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
    અચલ પૂર્ણાંક dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr*ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    દશાંશ * wdata = (દશાંશ *) params->wdata + (ne0 + CACHE_લીટી_SIZE_F32) * ith;

    માટે (પૂર્ણાંક ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        અચલ પૂર્ણાંક i3 = ir/(ne2*ne1);
        અચલ પૂર્ણાંક i2 = (ir - i3*ne2*ne1)/ne1;
        અચલ પૂર્ણાંક i1 = (ir - i3*ne2*ne1 - i2*ne1);

        શૂન્ય  * src0_row = (શૂન્ય *) ((અક્ષર *) src0->data + (i1*nb01 + i2*nb02 + i3*nb03));
        શૂન્ય  * dst_row  = (શૂન્ય *) ((અક્ષર *)  dst->data + (i1*nb1  + i2*nb2  + i3*nb0 ));

        પ્રતિજ્ઞા(ne0 % 32 == 0);

        // unquantize row from src0 to temp buffer
        dequantize_row_q(src0_row, wdata, ne0);
        // add src1
        ggml_vec_acc1_f32(ne0, wdata, v);
        // quantize row to dst
        quantize_row_q(wdata, dst_row, ne0);
    }
}

સ્થિર શૂન્ય ggml_compute_forward_add1_bf16_f32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    અચલ દશાંશ v = *(દશાંશ *) src1->data;

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_BF16);

    GGML_ASSERT( nb0 == કદ(ggml_bf16_t));
    GGML_ASSERT(nb00 == કદ(ggml_bf16_t));

    // rows per thread
    અચલ પૂર્ણાંક dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr*ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    માટે (પૂર્ણાંક ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        અચલ પૂર્ણાંક i3 = ir/(ne2*ne1);
        અચલ પૂર્ણાંક i2 = (ir - i3*ne2*ne1)/ne1;
        અચલ પૂર્ણાંક i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_bf16_t * dst_ptr  = (ggml_bf16_t *) ((અક્ષર *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_bf16_t * src0_ptr = (ggml_bf16_t *) ((અક્ષર *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
        માટે (પૂર્ણાંક i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(src0_ptr[i]) + v);
        }
    }
}

સ્થિર શૂન્ય ggml_compute_forward_add1_bf16_bf16(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    અચલ દશાંશ v = GGML_BF16_TO_FP32(*(ggml_bf16_t *) src1->data);

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(src1->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->type  == GGML_TYPE_BF16);

    GGML_ASSERT( nb0 == કદ(ggml_bf16_t));
    GGML_ASSERT(nb00 == કદ(ggml_bf16_t));

    // rows per thread
    અચલ પૂર્ણાંક dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr*ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    માટે (પૂર્ણાંક ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        અચલ પૂર્ણાંક i3 = ir/(ne2*ne1);
        અચલ પૂર્ણાંક i2 = (ir - i3*ne2*ne1)/ne1;
        અચલ પૂર્ણાંક i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_bf16_t * dst_ptr  = (ggml_bf16_t *) ((અક્ષર *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_bf16_t * src0_ptr = (ggml_bf16_t *) ((અક્ષર *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
        માટે (પૂર્ણાંક i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(src0_ptr[i]) + v);
        }
    }
}

શૂન્ય ggml_compute_forward_add1(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_add1_f32(params, dst);
            } તોડો;
        સ્થિતિ GGML_TYPE_F16:
            {
                જો (src1->type == GGML_TYPE_F16) {
                    ggml_compute_forward_add1_f16_f16(params, dst);
                }
                નહીંતર જો (src1->type == GGML_TYPE_F32) {
                    ggml_compute_forward_add1_f16_f32(params, dst);
                }
                નહીંતર {
                    GGML_ABORT("fatal error");
                }
            } તોડો;
        સ્થિતિ GGML_TYPE_BF16:
            {
                જો (src1->type == GGML_TYPE_BF16) {
                    ggml_compute_forward_add1_bf16_bf16(params, dst);
                }
                નહીંતર જો (src1->type == GGML_TYPE_F32) {
                    ggml_compute_forward_add1_bf16_f32(params, dst);
                }
                નહીંતર {
                    GGML_ABORT("fatal error");
                }
            } તોડો;
        સ્થિતિ GGML_TYPE_Q1_0:
        સ્થિતિ GGML_TYPE_Q2_0:
        સ્થિતિ GGML_TYPE_Q4_0:
        સ્થિતિ GGML_TYPE_Q4_1:
        સ્થિતિ GGML_TYPE_Q5_0:
        સ્થિતિ GGML_TYPE_Q5_1:
        સ્થિતિ GGML_TYPE_Q8_0:
        સ્થિતિ GGML_TYPE_Q8_1:
        સ્થિતિ GGML_TYPE_MXFP4:
        સ્થિતિ GGML_TYPE_NVFP4:
        સ્થિતિ GGML_TYPE_Q2_K:
        સ્થિતિ GGML_TYPE_Q3_K:
        સ્થિતિ GGML_TYPE_Q4_K:
        સ્થિતિ GGML_TYPE_Q5_K:
        સ્થિતિ GGML_TYPE_Q6_K:
        સ્થિતિ GGML_TYPE_TQ1_0:
        સ્થિતિ GGML_TYPE_TQ2_0:
        સ્થિતિ GGML_TYPE_IQ2_XXS:
        સ્થિતિ GGML_TYPE_IQ2_XS:
        સ્થિતિ GGML_TYPE_IQ3_XXS:
        સ્થિતિ GGML_TYPE_IQ1_S:
        સ્થિતિ GGML_TYPE_IQ1_M:
        સ્થિતિ GGML_TYPE_IQ4_NL:
        સ્થિતિ GGML_TYPE_IQ4_XS:
        સ્થિતિ GGML_TYPE_IQ3_S:
        સ્થિતિ GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_add1_q_f32(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_acc

સ્થિર શૂન્ય ggml_compute_forward_acc_f32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));

    // view src0 and dst with these strides and data offset inbytes during acc
    // nb0 is implicitly element_size because src0 and dst are contiguous
    size_t nb1     = ((int32_t *) dst->op_params)[0];
    size_t nb2     = ((int32_t *) dst->op_params)[1];
    size_t nb3     = ((int32_t *) dst->op_params)[2];
    size_t offset  = ((int32_t *) dst->op_params)[3];
    તાર્કિક   inplace = (તાર્કિક) ((int32_t *) dst->op_params)[4];

    જો (!inplace) {
        જો (params->ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            memcpy(
                ((અક્ષર *)  dst->data),
                ((અક્ષર *) src0->data),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક nr = ggml_nrows(src1);
    અચલ પૂર્ણાંક nc = src1->ne[0];

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne)
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb)

    // src0 and dst as viewed during acc
    અચલ size_t nb0 = ggml_element_size(src0);

    અચલ size_t nb00 = nb0;
    અચલ size_t nb01 = nb1;
    અચલ size_t nb02 = nb2;
    અચલ size_t nb03 = nb3;

    GGML_ASSERT(offset + (ne10 == 0 ? 0 : ne10-1)*nb0  + (ne11 == 0 ? 0 : ne11-1)*nb1  + (ne12 == 0 ? 0 : ne12-1)*nb2  + (ne13 == 0 ? 0 : ne13-1)*nb3  < ggml_nbytes(dst));
    GGML_ASSERT(offset + (ne10 == 0 ? 0 : ne10-1)*nb00 + (ne11 == 0 ? 0 : ne11-1)*nb01 + (ne12 == 0 ? 0 : ne12-1)*nb02 + (ne13 == 0 ? 0 : ne13-1)*nb03 < ggml_nbytes(src0));

    GGML_ASSERT(nb10 == કદ(દશાંશ));

    // rows per thread
    અચલ પૂર્ણાંક dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr*ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    માટે (પૂર્ણાંક ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are viewed with shape of src1 and offset
        // => same indices
        અચલ પૂર્ણાંક i3 = ir/(ne12*ne11);
        અચલ પૂર્ણાંક i2 = (ir - i3*ne12*ne11)/ne11;
        અચલ પૂર્ણાંક i1 = (ir - i3*ne12*ne11 - i2*ne11);

#ifdef GGML_USE_ACCELERATE
        vDSP_vadd(
                (દશાંશ *) ((અક્ષર *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + offset), 1,
                (દશાંશ *) ((અક્ષર *) src1->data + i3*nb13 + i2*nb12 + i1*nb11), 1,
                (દશાંશ *) ((અક્ષર *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1  + offset), 1, nc);
#નહીંતર
        ggml_vec_add_f32(nc,
                (દશાંશ *) ((અક્ષર *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + offset),
                (દશાંશ *) ((અક્ષર *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + offset),
                (દશાંશ *) ((અક્ષર *) src1->data + i3*nb13 + i2*nb12 + i1*nb11));
#endif
    }
}

શૂન્ય ggml_compute_forward_acc(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_acc_f32(params, dst);
            } તોડો;
        સ્થિતિ GGML_TYPE_F16:
        સ્થિતિ GGML_TYPE_BF16:
        સ્થિતિ GGML_TYPE_Q1_0:
        સ્થિતિ GGML_TYPE_Q2_0:
        સ્થિતિ GGML_TYPE_Q4_0:
        સ્થિતિ GGML_TYPE_Q4_1:
        સ્થિતિ GGML_TYPE_Q5_0:
        સ્થિતિ GGML_TYPE_Q5_1:
        સ્થિતિ GGML_TYPE_Q8_0:
        સ્થિતિ GGML_TYPE_Q8_1:
        સ્થિતિ GGML_TYPE_MXFP4:
        સ્થિતિ GGML_TYPE_NVFP4:
        સ્થિતિ GGML_TYPE_Q2_K:
        સ્થિતિ GGML_TYPE_Q3_K:
        સ્થિતિ GGML_TYPE_Q4_K:
        સ્થિતિ GGML_TYPE_Q5_K:
        સ્થિતિ GGML_TYPE_Q6_K:
        સ્થિતિ GGML_TYPE_TQ1_0:
        સ્થિતિ GGML_TYPE_TQ2_0:
        સ્થિતિ GGML_TYPE_IQ2_XXS:
        સ્થિતિ GGML_TYPE_IQ2_XS:
        સ્થિતિ GGML_TYPE_IQ3_XXS:
        સ્થિતિ GGML_TYPE_IQ1_S:
        સ્થિતિ GGML_TYPE_IQ1_M:
        સ્થિતિ GGML_TYPE_IQ4_NL:
        સ્થિતિ GGML_TYPE_IQ4_XS:
        સ્થિતિ GGML_TYPE_IQ3_S:
        સ્થિતિ GGML_TYPE_IQ2_S:
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_sum

સ્થિર શૂન્ય ggml_compute_forward_sum_f32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    જો (params->ith != 0) {
        પાછા;
    }

    પ્રતિજ્ઞા(ggml_is_scalar(dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)

    ggml_float sum = 0;

    જો (nb00 == કદ(દશાંશ)) {
        ggml_float row_sum = 0;
        માટે (int64_t i03 = 0; i03 < ne03; i03++) {
            માટે (int64_t i02 = 0; i02 < ne02; i02++) {
                માટે (int64_t i01 = 0; i01 < ne01; i01++) {
                    ggml_vec_sum_f32_ggf(ne00,
                            &row_sum,
                            (દશાંશ *) ((અક્ષર *) src0->data + i01*nb01 + i02*nb02 + i03*nb03));
                    sum += row_sum;
                }
            }
        }
    } નહીંતર {
        માટે (int64_t i03 = 0; i03 < ne03; i03++) {
            માટે (int64_t i02 = 0; i02 < ne02; i02++) {
                માટે (int64_t i01 = 0; i01 < ne01; i01++) {
                    અચલ અક્ષર * row = (અચલ અક્ષર *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
                    માટે (int64_t i00 = 0; i00 < ne00; i00++) {
                        sum += (ggml_float) *(અચલ દશાંશ *) (row + i00*nb00);
                    }
                }
            }
        }
    }

    ((દશાંશ *) dst->data)[0] = sum;
}

સ્થિર શૂન્ય ggml_compute_forward_sum_f16(
    અચલ ggml_compute_params * params,
          ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    જો (params->ith != 0) {
        પાછા;
    }

    પ્રતિજ્ઞા(ggml_is_scalar(dst));

    પ્રતિજ્ઞા(src0->nb[0] == કદ(ggml_fp16_t));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)

    દશાંશ sum = 0;
    દશાંશ row_sum = 0;

    માટે (int64_t i03 = 0; i03 < ne03; i03++) {
        માટે (int64_t i02 = 0; i02 < ne02; i02++) {
            માટે (int64_t i01 = 0; i01 < ne01; i01++) {
                ggml_vec_sum_f16_ggf(ne00,
                    &row_sum,
                    (ggml_fp16_t *) ((અક્ષર *) src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03));
                sum += row_sum;
            }
        }
    }
    ((ggml_fp16_t *) dst->data)[0] = GGML_CPU_FP32_TO_FP16(sum);
}

સ્થિર શૂન્ય ggml_compute_forward_sum_bf16(
    અચલ ggml_compute_params * params,
          ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    જો (params->ith != 0) {
        પાછા;
    }

    પ્રતિજ્ઞા(ggml_is_scalar(dst));

    પ્રતિજ્ઞા(src0->nb[0] == કદ(ggml_bf16_t));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)

    દશાંશ sum = 0;
    દશાંશ row_sum = 0;

    માટે (int64_t i03 = 0; i03 < ne03; i03++) {
        માટે (int64_t i02 = 0; i02 < ne02; i02++) {
            માટે (int64_t i01 = 0; i01 < ne01; i01++) {
                ggml_vec_sum_bf16_ggf(ne00,
                    &row_sum,
                    (ggml_bf16_t *) ((અક્ષર *) src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03));
                sum += row_sum;
            }
        }
    }
    ((ggml_bf16_t *) dst->data)[0] = GGML_FP32_TO_BF16(sum);
}

શૂન્ય ggml_compute_forward_sum(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_sum_f32(params, dst);
            } તોડો;
        સ્થિતિ GGML_TYPE_F16:
            {
                ggml_compute_forward_sum_f16(params, dst);
            } તોડો;
        સ્થિતિ GGML_TYPE_BF16:
            {
                ggml_compute_forward_sum_bf16(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_cumsum

સ્થિર શૂન્ય ggml_compute_forward_cumsum_f32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->nb[0] == કદ(દશાંશ));
    GGML_ASSERT(dst->nb[0] == કદ(દશાંશ));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne01);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    અચલ auto [ir0, ir1] = get_thread_range(params, src0);

    માટે (int64_t ir = ir0; ir < ir1; ++ir) {
        અચલ int64_t i03 = ir/(ne02*ne01);
        અચલ int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        અચલ int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        દશાંશ * src_row = (દશાંશ *) ((અક્ષર *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
        દશાંશ * dst_row = (દશાંશ *) ((અક્ષર *) dst->data  + i01*nb1  + i02*nb2  + i03*nb3);

        ggml_vec_cumsum_f32(ne00, dst_row, src_row);
    }
}

શૂન્ય ggml_compute_forward_cumsum(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_cumsum_f32(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_sum_rows

સ્થિર શૂન્ય ggml_compute_forward_sum_rows_f32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    જો (params->ith != 0) {
        પાછા;
    }

    GGML_ASSERT(src0->nb[0] == કદ(દશાંશ));
    GGML_ASSERT(dst->nb[0] == કદ(દશાંશ));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(ne0 == 1);
    GGML_ASSERT(ne1 == ne01);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    માટે (int64_t i3 = 0; i3 < ne03; i3++) {
        માટે (int64_t i2 = 0; i2 < ne02; i2++) {
            માટે (int64_t i1 = 0; i1 < ne01; i1++) {
                દશાંશ * src_row = (દશાંશ *) ((અક્ષર *) src0->data + i1*nb01 + i2*nb02 + i3*nb03);
                દશાંશ * dst_row = (દશાંશ *) ((અક્ષર *) dst->data  + i1*nb1  + i2*nb2  + i3*nb3);
                દશાંશ row_sum = 0;
                ggml_vec_sum_f32(ne00, &row_sum, src_row);
                dst_row[0] = row_sum;
            }
        }
    }
}

શૂન્ય ggml_compute_forward_sum_rows(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_sum_rows_f32(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_mean

સ્થિર શૂન્ય ggml_compute_forward_mean_f32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    જો (params->ith != 0) {
        પાછા;
    }

    પ્રતિજ્ઞા(src0->nb[0] == કદ(દશાંશ));

    GGML_TENSOR_UNARY_OP_LOCALS

    પ્રતિજ્ઞા(ne0 == 1);
    પ્રતિજ્ઞા(ne1 == ne01);
    પ્રતિજ્ઞા(ne2 == ne02);
    પ્રતિજ્ઞા(ne3 == ne03);

    GGML_UNUSED(ne0);
    GGML_UNUSED(ne1);
    GGML_UNUSED(ne2);
    GGML_UNUSED(ne3);

    માટે (int64_t i03 = 0; i03 < ne03; i03++) {
        માટે (int64_t i02 = 0; i02 < ne02; i02++) {
            માટે (int64_t i01 = 0; i01 < ne01; i01++) {
                ggml_vec_sum_f32(ne00,
                        (દશાંશ *) ((અક્ષર *)  dst->data + i01*nb1  + i02*nb2  + i03*nb3),
                        (દશાંશ *) ((અક્ષર *) src0->data + i01*nb01 + i02*nb02 + i03*nb03));

                *(દશાંશ *) ((અક્ષર *) dst->data + i01*nb1 + i02*nb2 + i03*nb3) /= (દશાંશ) ne00;
            }
        }
    }
}

શૂન્ય ggml_compute_forward_mean(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_mean_f32(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_argmax

સ્થિર શૂન્ય ggml_compute_forward_argmax_f32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    જો (params->ith != 0) {
        પાછા;
    }

    પ્રતિજ્ઞા(src0->nb[0] == કદ(દશાંશ));
    પ્રતિજ્ઞા(dst->nb[0] == કદ(દશાંશ));

    અચલ int64_t ne00 = src0->ne[0];
    અચલ int64_t ne01 = src0->ne[1];

    અચલ size_t nb01 = src0->nb[1];
    અચલ size_t nb0 = dst->nb[0];

    માટે (int64_t i1 = 0; i1 < ne01; i1++) {
        દશાંશ * src = (દશાંશ *) ((અક્ષર *) src0->data + i1*nb01);
        int32_t * dst_ = (int32_t *) ((અક્ષર *)  dst->data + i1*nb0);
        પૂર્ણાંક v = 0;
        ggml_vec_argmax_f32(ne00, &v, src);
        dst_[0] = v;
    }
}

શૂન્ય ggml_compute_forward_argmax(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_argmax_f32(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_count_equal

સ્થિર શૂન્ય ggml_compute_forward_ગણતરી_equal_i32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_દ્વિઅંકી_OP_LOCALS;

    GGML_ASSERT(src0->type == GGML_TYPE_I32);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));
    GGML_ASSERT(dst->type == GGML_TYPE_I64);

    અચલ int64_t nr = ggml_nrows(src0);

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    int64_t * sums = (int64_t *) params->wdata;
    int64_t sum_thread = 0;

    // rows per thread
    અચલ int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ int64_t ir0 = dr*ith;
    અચલ int64_t ir1 = MIN(ir0 + dr, nr);

    માટે (int64_t ir = ir0; ir < ir1; ++ir) {
        અચલ int64_t i03 =  ir                        / (ne02*ne01);
        અચલ int64_t i02 = (ir - i03*ne03)            /       ne01;
        અચલ int64_t i01 =  ir - i03*ne03 - i02*ne02;

        અચલ અક્ષર * data0 = (અચલ અક્ષર *) src0->data + i03*nb03 + i02*nb02 + i01*nb01;
        અચલ અક્ષર * data1 = (અચલ અક્ષર *) src1->data + i03*nb13 + i02*nb12 + i01*nb11;

        માટે (int64_t i00 = 0; i00 < ne00; ++i00) {
            અચલ int32_t val0 = *((અચલ int32_t *) (data0 + i00*nb00));
            અચલ int32_t val1 = *((અચલ int32_t *) (data1 + i00*nb10));

            sum_thread += val0 == val1;
        }
    }
    જો (ith != 0) {
        sums[ith] = sum_thread;
    }
    ggml_barrier(params->threadpool);

    જો (ith != 0) {
        પાછા;
    }

    માટે (પૂર્ણાંક ith_other = 1; ith_other < nth; ++ith_other) {
        sum_thread += sums[ith_other];
    }
    *((int64_t *) dst->data) = sum_thread;
}

શૂન્ય ggml_compute_forward_ગણતરી_equal(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_I32:
            {
                ggml_compute_forward_ગણતરી_equal_i32(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_repeat

સ્થિર શૂન્ય ggml_compute_forward_repeat_f32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    જો (params->ith != 0) {
        પાછા;
    }

    GGML_ASSERT(ggml_can_repeat(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    // guaranteed to be an integer due to the check in ggml_can_repeat
    અચલ પૂર્ણાંક nr0 = (પૂર્ણાંક)(ne0/ne00);
    અચલ પૂર્ણાંક nr1 = (પૂર્ણાંક)(ne1/ne01);
    અચલ પૂર્ણાંક nr2 = (પૂર્ણાંક)(ne2/ne02);
    અચલ પૂર્ણાંક nr3 = (પૂર્ણાંક)(ne3/ne03);

    // TODO: support for transposed / permuted tensors
    GGML_ASSERT(nb0  == કદ(દશાંશ));
    GGML_ASSERT(nb00 == કદ(દશાંશ));

    // TODO: maybe this is not optimal?
    માટે                         (પૂર્ણાંક i3 = 0; i3 < nr3;  i3++) {
        માટે                     (પૂર્ણાંક k3 = 0; k3 < ne03; k3++) {
            માટે                 (પૂર્ણાંક i2 = 0; i2 < nr2;  i2++) {
                માટે             (પૂર્ણાંક k2 = 0; k2 < ne02; k2++) {
                    માટે         (પૂર્ણાંક i1 = 0; i1 < nr1;  i1++) {
                        માટે     (પૂર્ણાંક k1 = 0; k1 < ne01; k1++) {
                            માટે (પૂર્ણાંક i0 = 0; i0 < nr0;  i0++) {
                                ggml_vec_cpy_f32(ne00,
                                        (દશાંશ *) ((અક્ષર *)  dst->data + (i3*ne03 + k3)*nb3  + (i2*ne02 + k2)*nb2  + (i1*ne01 + k1)*nb1  + (i0*ne00)*nb0),
                                        (દશાંશ *) ((અક્ષર *) src0->data + (          k3)*nb03 + (          k2)*nb02 + (          k1)*nb01));
                            }
                        }
                    }
                }
            }
        }
    }
}

સ્થિર શૂન્ય ggml_compute_forward_repeat_f16(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    જો (params->ith != 0) {
        પાછા;
    }

    GGML_ASSERT(ggml_can_repeat(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    // guaranteed to be an integer due to the check in ggml_can_repeat
    અચલ પૂર્ણાંક nr0 = (પૂર્ણાંક)(ne0/ne00);
    અચલ પૂર્ણાંક nr1 = (પૂર્ણાંક)(ne1/ne01);
    અચલ પૂર્ણાંક nr2 = (પૂર્ણાંક)(ne2/ne02);
    અચલ પૂર્ણાંક nr3 = (પૂર્ણાંક)(ne3/ne03);

    // TODO: support for transposed / permuted tensors
    GGML_ASSERT(nb0  == કદ(ggml_fp16_t));
    GGML_ASSERT(nb00 == કદ(ggml_fp16_t));

    // TODO: maybe this is not optimal?
    માટે                         (પૂર્ણાંક i3 = 0; i3 < nr3;  i3++) {
        માટે                     (પૂર્ણાંક k3 = 0; k3 < ne03; k3++) {
            માટે                 (પૂર્ણાંક i2 = 0; i2 < nr2;  i2++) {
                માટે             (પૂર્ણાંક k2 = 0; k2 < ne02; k2++) {
                    માટે         (પૂર્ણાંક i1 = 0; i1 < nr1;  i1++) {
                        માટે     (પૂર્ણાંક k1 = 0; k1 < ne01; k1++) {
                            માટે (પૂર્ણાંક i0 = 0; i0 < nr0;  i0++) {
                                ggml_fp16_t * y = (ggml_fp16_t *) ((અક્ષર *)  dst->data + (i3*ne03 + k3)*nb3  + (i2*ne02 + k2)*nb2  + (i1*ne01 + k1)*nb1  + (i0*ne00)*nb0);
                                ggml_fp16_t * x = (ggml_fp16_t *) ((અક્ષર *) src0->data + (          k3)*nb03 + (          k2)*nb02 + (          k1)*nb01);
                                // ggml_vec_cpy_f16(ne00, y, x)
                                માટે (પૂર્ણાંક i = 0; i < ne00; ++i) {
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

શૂન્ય ggml_compute_forward_repeat(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F16:
        સ્થિતિ GGML_TYPE_BF16:
        સ્થિતિ GGML_TYPE_I16:
            {
                ggml_compute_forward_repeat_f16(params, dst);
            } તોડો;
        સ્થિતિ GGML_TYPE_F32:
        સ્થિતિ GGML_TYPE_I32:
            {
                ggml_compute_forward_repeat_f32(params, dst);
            } તોડો;
        // TODO: templateify the implementation and support for I64
        //       ref https://github.com/ggml-org/llama.cpp/pull/14274#discussion_r2169492225
        //case GGML_TYPE_I64:
        //    {
        //        ggml_compute_forward_repeat_i64(params, dst);
        //    } break;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_repeat_back

સ્થિર શૂન્ય ggml_compute_forward_repeat_back_f32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    જો (params->ith != 0) {
        પાછા;
    }

    GGML_ASSERT(ggml_can_repeat(dst, src0));

    GGML_TENSOR_UNARY_OP_LOCALS

    // guaranteed to be an integer due to the check in ggml_can_repeat
    અચલ પૂર્ણાંક nr0 = (પૂર્ણાંક)(ne00/ne0);
    અચલ પૂર્ણાંક nr1 = (પૂર્ણાંક)(ne01/ne1);
    અચલ પૂર્ણાંક nr2 = (પૂર્ણાંક)(ne02/ne2);
    અચલ પૂર્ણાંક nr3 = (પૂર્ણાંક)(ne03/ne3);

    // TODO: support for transposed / permuted tensors
    GGML_ASSERT(nb0  == કદ(દશાંશ));
    GGML_ASSERT(nb00 == કદ(દશાંશ));

    જો (ggml_is_contiguous(dst)) {
        ggml_vec_set_f32(ne0*ne1*ne2*ne3, (દશાંશ *)dst->data, 0);
    } નહીંતર {
        માટે         (પૂર્ણાંક k3 = 0; k3 < ne3; k3++) {
            માટે     (પૂર્ણાંક k2 = 0; k2 < ne2; k2++) {
                માટે (પૂર્ણાંક k1 = 0; k1 < ne1; k1++) {
                    ggml_vec_set_f32(ne0,
                        (દશાંશ *) ((અક્ષર *) dst->data + k1*nb1 + k2*nb2 + k3*nb3),
                        0);
                }
            }
        }
    }

    // TODO: maybe this is not optimal?
    માટે                         (પૂર્ણાંક i3 = 0; i3 < nr3; i3++) {
        માટે                     (પૂર્ણાંક k3 = 0; k3 < ne3; k3++) {
            માટે                 (પૂર્ણાંક i2 = 0; i2 < nr2; i2++) {
                માટે             (પૂર્ણાંક k2 = 0; k2 < ne2; k2++) {
                    માટે         (પૂર્ણાંક i1 = 0; i1 < nr1; i1++) {
                        માટે     (પૂર્ણાંક k1 = 0; k1 < ne1; k1++) {
                            માટે (પૂર્ણાંક i0 = 0; i0 < nr0; i0++) {
                                ggml_vec_acc_f32(ne0,
                                        (દશાંશ *) ((અક્ષર *)  dst->data + (         k3)*nb3  + (         k2)*nb2  + (         k1)*nb1),
                                        (દશાંશ *) ((અક્ષર *) src0->data + (i3*ne3 + k3)*nb03 + (i2*ne2 + k2)*nb02 + (i1*ne1 + k1)*nb01 + (i0*ne0)*nb00));
                            }
                        }
                    }
                }
            }
        }
    }
}

શૂન્ય ggml_compute_forward_repeat_back(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_repeat_back_f32(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_concat
સ્થિર શૂન્ય ggml_compute_forward_concat_any(
    અચલ ggml_compute_params * params,
    ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    GGML_TENSOR_દ્વિઅંકી_OP_LOCALS

    અચલ int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);
    GGML_ASSERT(ggml_is_contiguous_rows(src0));
    GGML_ASSERT(ggml_is_contiguous_rows(src1));

    int64_t o[4] = {0, 0, 0, 0};

    જો (dim == 0) {
        GGML_ASSERT(src0->ne[0] % ggml_blck_size(src0->type) == 0);
        GGML_ASSERT(src1->ne[0] % ggml_blck_size(src1->type) == 0);

        o[dim] = src0->ne[dim]/ggml_blck_size(src0->type);
    } નહીંતર {
        o[dim] = src0->ne[dim];
    }

    // Region 1: copy rows from src0
    માટે (પૂર્ણાંક i3 = 0; i3 < ne03; i3++) {
        માટે (પૂર્ણાંક i2 = ith; i2 < ne02; i2 += nth) {
            માટે (પૂર્ણાંક i1 = 0; i1 < ne01; i1++) {
                અચલ અક્ષર * x = (અચલ અક્ષર *) src0->data + i1*nb01 + i2*nb02 + i3*nb03;
                      અક્ષર * y = (      અક્ષર *) dst->data  + i1*nb1  + i2*nb2  + i3*nb3;
                memcpy(y, x, ggml_row_size(src0->type, ne00));
            }
        }
    }

    // Region 2: copy rows from src1, offset into dst by o[]
    માટે (પૂર્ણાંક i3 = 0; i3 < ne13; i3++) {
        માટે (પૂર્ણાંક i2 = ith; i2 < ne12; i2 += nth) {
            માટે (પૂર્ણાંક i1 = 0; i1 < ne11; i1++) {
                અચલ અક્ષર * x = (અચલ અક્ષર *) src1->data + i1*nb11         + i2*nb12         + i3*nb13;
                      અક્ષર * y = (      અક્ષર *)  dst->data + (i1 + o[1])*nb1 + (i2 + o[2])*nb2 + (i3 + o[3])*nb3 + o[0]*nb0;
                memcpy(y, x, ggml_row_size(src1->type, ne10));
            }
        }
    }
}

સ્થિર શૂન્ય ggml_compute_forward_concat_i8(
    અચલ ggml_compute_params * params,
    ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_type_size(src0->type) == કદ(int8_t));

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    GGML_TENSOR_દ્વિઅંકી_OP_LOCALS

    અચલ int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);

    int64_t o[4] = {0, 0, 0, 0};
    o[dim] = src0->ne[dim];

    અચલ int8_t * x;

    // TODO: smarter multi-theading
    માટે (પૂર્ણાંક i3 = 0; i3 < ne3; i3++) {
        માટે (પૂર્ણાંક i2 = ith; i2 < ne2; i2 += nth) {
            માટે (પૂર્ણાંક i1 = 0; i1 < ne1; i1++) {
                માટે (પૂર્ણાંક i0 = 0; i0 < ne0; i0++) {
                    જો (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                        x = (અચલ int8_t *) ((અચલ અક્ષર *)src0->data + (i0       )*nb00 + (i1       )*nb01 + (i2       )*nb02 + (i3       )*nb03);
                    } નહીંતર {
                        x = (અચલ int8_t *) ((અચલ અક્ષર *)src1->data + (i0 - o[0])*nb10 + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13);
                    }

                    int8_t * y = (int8_t *)((અક્ષર *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);

                    *y = *x;
                }
            }
        }
    }
}

સ્થિર શૂન્ય ggml_compute_forward_concat_f16(
    અચલ ggml_compute_params * params,
    ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_type_size(src0->type) == કદ(ggml_fp16_t));

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    GGML_TENSOR_દ્વિઅંકી_OP_LOCALS

    અચલ int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);

    int64_t o[4] = {0, 0, 0, 0};
    o[dim] = src0->ne[dim];

    અચલ ggml_fp16_t * x;

    // TODO: smarter multi-theading
    માટે (પૂર્ણાંક i3 = 0; i3 < ne3; i3++) {
        માટે (પૂર્ણાંક i2 = ith; i2 < ne2; i2 += nth) {
            માટે (પૂર્ણાંક i1 = 0; i1 < ne1; i1++) {
                માટે (પૂર્ણાંક i0 = 0; i0 < ne0; i0++) {
                    જો (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                        x = (અચલ ggml_fp16_t *) ((અચલ અક્ષર *)src0->data + (i0       )*nb00 + (i1       )*nb01 + (i2       )*nb02 + (i3       )*nb03);
                    } નહીંતર {
                        x = (અચલ ggml_fp16_t *) ((અચલ અક્ષર *)src1->data + (i0 - o[0])*nb10 + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13);
                    }

                    ggml_fp16_t * y = (ggml_fp16_t *)((અક્ષર *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);

                    *y = *x;
                }
            }
        }
    }
}

સ્થિર શૂન્ય ggml_compute_forward_concat_f32(
    અચલ ggml_compute_params * params,
    ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_type_size(src0->type) == કદ(દશાંશ));

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    GGML_TENSOR_દ્વિઅંકી_OP_LOCALS

    અચલ int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);

    int64_t o[4] = {0, 0, 0, 0};
    o[dim] = src0->ne[dim];

    અચલ દશાંશ * x;

    // TODO: smarter multi-theading
    માટે (પૂર્ણાંક i3 = 0; i3 < ne3; i3++) {
        માટે (પૂર્ણાંક i2 = ith; i2 < ne2; i2 += nth) {
            માટે (પૂર્ણાંક i1 = 0; i1 < ne1; i1++) {
                માટે (પૂર્ણાંક i0 = 0; i0 < ne0; i0++) {
                    જો (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                        x = (અચલ દશાંશ *) ((અચલ અક્ષર *)src0->data + (i0       )*nb00 + (i1       )*nb01 + (i2       )*nb02 + (i3       )*nb03);
                    } નહીંતર {
                        x = (અચલ દશાંશ *) ((અચલ અક્ષર *)src1->data + (i0 - o[0])*nb10 + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13);
                    }

                    દશાંશ * y = (દશાંશ *)((અક્ષર *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);

                    *y = *x;
                }
            }
        }
    }
}

શૂન્ય ggml_compute_forward_concat(
    અચલ ggml_compute_params * params,
    ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F16:
        સ્થિતિ GGML_TYPE_BF16:
        સ્થિતિ GGML_TYPE_I16:
            {
                ggml_compute_forward_concat_f16(params, dst);
            } તોડો;
        સ્થિતિ GGML_TYPE_I8:
            {
                ggml_compute_forward_concat_i8(params, dst);
            } તોડો;
        સ્થિતિ GGML_TYPE_F32:
        સ્થિતિ GGML_TYPE_I32:
            {
                ggml_compute_forward_concat_f32(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                ggml_compute_forward_concat_any(params, dst);
            }
    }
}

// row loop shared by the BF16 unary ops
template <શૂન્ય (*vec_op)(અચલ પૂર્ણાંક, ggml_bf16_t *, અચલ ggml_bf16_t *)>
સ્થિર શૂન્ય ggml_compute_forward_unary_bf16_rows(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પ્રતિજ્ઞા(ggml_is_contiguous_rows(src0));
    પ્રતિજ્ઞા(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક nc = src0->ne[0];
    અચલ પૂર્ણાંક nr = ggml_nrows(src0);

    // rows per thread
    અચલ પૂર્ણાંક dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr*ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    માટે (પૂર્ણાંક ir = ir0; ir < ir1; ++ir) {
        અચલ પૂર્ણાંક i3 = ir/(ne02*ne01);
        અચલ પૂર્ણાંક i2 = (ir - i3*ne02*ne01)/ne01;
        અચલ પૂર્ણાંક i1 = (ir - i3*ne02*ne01 - i2*ne01);

        vec_op(nc,
                (ggml_bf16_t *) ((અક્ષર *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_bf16_t *) ((અક્ષર *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));
    }
}

// ggml_compute_forward_gelu

સ્થિર શૂન્ય ggml_compute_forward_gelu_f32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પ્રતિજ્ઞા(ggml_is_contiguous_rows(src0));
    પ્રતિજ્ઞા(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક nc = src0->ne[0];
    અચલ પૂર્ણાંક nr = ggml_nrows(src0);

    // rows per thread
    અચલ પૂર્ણાંક dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr*ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    માટે (પૂર્ણાંક ir = ir0; ir < ir1; ++ir) {
        અચલ પૂર્ણાંક i3 = ir/(ne02*ne01);
        અચલ પૂર્ણાંક i2 = (ir - i3*ne02*ne01)/ne01;
        અચલ પૂર્ણાંક i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_f32(nc,
                (દશાંશ *) ((અક્ષર *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (દશાંશ *) ((અક્ષર *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        માટે (પૂર્ણાંક k = 0; k < nc; k++) {
            અચલ દશાંશ x = ((દશાંશ *) ((અક્ષર *) dst->data + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_UNUSED(x);
            પ્રતિજ્ઞા(!isnan(x));
            પ્રતિજ્ઞા(!isinf(x));
        }
#endif // NDEBUG
    }
}

સ્થિર શૂન્ય ggml_compute_forward_gelu_f16(
    અચલ ggml_compute_params * params,
    ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પ્રતિજ્ઞા(ggml_is_contiguous_rows(src0));
    પ્રતિજ્ઞા(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક nc = src0->ne[0];
    અચલ પૂર્ણાંક nr = ggml_nrows(src0);

    // rows per thread
    અચલ પૂર્ણાંક dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr*ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    માટે (પૂર્ણાંક ir = ir0; ir < ir1; ++ir) {
        અચલ પૂર્ણાંક i3 = ir/(ne02*ne01);
        અચલ પૂર્ણાંક i2 = (ir - i3*ne02*ne01)/ne01;
        અચલ પૂર્ણાંક i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_f16(nc,
                (ggml_fp16_t *) ((અક્ષર *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((અક્ષર *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        માટે (પૂર્ણાંક k = 0; k < nc; k++) {
            અચલ ggml_fp16_t x = ((ggml_fp16_t *) ((અક્ષર *) dst->data + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            અચલ દશાંશ v = GGML_CPU_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            પ્રતિજ્ઞા(!isnan(v));
            પ્રતિજ્ઞા(!isinf(v));
        }
#endif // NDEBUG
    }
}

સ્થિર શૂન્ય ggml_compute_forward_gelu_bf16(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_rows<ggml_vec_gelu_bf16>(params, dst);
}

સ્થિર શૂન્ય ggml_compute_forward_gelu(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_gelu_f32(params, dst);
            } તોડો;
        સ્થિતિ GGML_TYPE_F16:
            {
                ggml_compute_forward_gelu_f16(params, dst);
            } તોડો;
        સ્થિતિ GGML_TYPE_BF16:
            {
                ggml_compute_forward_gelu_bf16(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_fill

સ્થિર શૂન્ય ggml_compute_forward_fill_f32(અચલ ggml_compute_params * params, ggml_tensor * dst) {
    અચલ દશાંશ c = ggml_get_op_params_f32(dst, 0);

    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne);
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb);

    અચલ auto [ir0, ir1] = get_thread_range(params, dst);

    માટે (int64_t ir = ir0; ir < ir1; ++ir) {
        અચલ int64_t i03 = ir/(ne2*ne1);
        અચલ int64_t i02 = (ir - i03*ne2*ne1)/ne1;
        અચલ int64_t i01 = (ir - i03*ne2*ne1 - i02*ne1);

        દશાંશ * dst_ptr  = (દશાંશ *) ((અક્ષર *) dst->data + i03*nb3 + i02*nb2 + i01*nb1);

        ggml_vec_set_f32(ne0, dst_ptr, c);
    }
}

સ્થિર શૂન્ય ggml_compute_forward_fill_f16(અચલ ggml_compute_params * params, ggml_tensor * dst) {
    અચલ ggml_fp16_t c = GGML_CPU_FP32_TO_FP16(ggml_get_op_params_f32(dst, 0));

    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne);
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb);

    અચલ auto [ir0, ir1] = get_thread_range(params, dst);

    માટે (int64_t ir = ir0; ir < ir1; ++ir) {
        અચલ int64_t i03 = ir/(ne2*ne1);
        અચલ int64_t i02 = (ir - i03*ne2*ne1)/ne1;
        અચલ int64_t i01 = (ir - i03*ne2*ne1 - i02*ne1);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((અક્ષર *) dst->data + i03*nb3 + i02*nb2 + i01*nb1);

        ggml_vec_set_f16(ne0, dst_ptr, c);
    }
}

શૂન્ય ggml_compute_forward_fill(અચલ ggml_compute_params * params, ggml_tensor * dst) {
    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_fill_f32(params, dst);
            } તોડો;
        સ્થિતિ GGML_TYPE_F16:
            {
                ggml_compute_forward_fill_f16(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("unsupported type for ggml_compute_forward_fill: %s", ggml_type_name(src0->type));
            }
    }
}

// ggml_compute_tri

સ્થિર શૂન્ય ggml_compute_forward_tri_f32(અચલ ggml_compute_params * params, ggml_tensor * dst) {
    અચલ ggml_tensor * src0 = dst->src[0];

    અચલ ggml_tri_type ttype = (ggml_tri_type) ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_TENSOR_UNARY_OP_LOCALS

    અચલ auto [ir0, ir1] = get_thread_range(params, src0);

    તાર્કિક (*bipred)(પૂર્ણાંક, પૂર્ણાંક);

    પસંદગી (ttype) {
        સ્થિતિ GGML_TRI_TYPE_LOWER:      bipred = [](પૂર્ણાંક i, પૂર્ણાંક r) { પાછા i <  r; }; તોડો;
        સ્થિતિ GGML_TRI_TYPE_LOWER_DIAG: bipred = [](પૂર્ણાંક i, પૂર્ણાંક r) { પાછા i <= r; }; તોડો;
        સ્થિતિ GGML_TRI_TYPE_UPPER:      bipred = [](પૂર્ણાંક i, પૂર્ણાંક r) { પાછા i >  r; }; તોડો;
        સ્થિતિ GGML_TRI_TYPE_UPPER_DIAG: bipred = [](પૂર્ણાંક i, પૂર્ણાંક r) { પાછા i >= r; }; તોડો;
        મૂળભૂત: GGML_ABORT("invalid tri type");
    }

    માટે (int64_t ir = ir0; ir < ir1; ++ir) {
        અચલ int64_t i03 = ir/(ne02*ne01);
        અચલ int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        અચલ int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        અચલ દશાંશ * src_ptr = (અચલ દશાંશ  *) ((અચલ અક્ષર *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);
              દશાંશ * dst_ptr = (      દશાંશ  *) ((      અક્ષર *) dst->data  + i03*nb3  + i02*nb2  + i01*nb1);

        માટે (પૂર્ણાંક i0 = 0; i0 < ne0; ++i0) {
            dst_ptr[i0] = bipred(i0, i01) ? src_ptr[i0] : 0.0f;
        }
    }
}

શૂન્ય ggml_compute_forward_tri(અચલ ggml_compute_params * params, ggml_tensor * dst) {
    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_tri_f32(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_gelu_erf

સ્થિર શૂન્ય ggml_compute_forward_gelu_erf_f32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પ્રતિજ્ઞા(ggml_is_contiguous_rows(src0));
    પ્રતિજ્ઞા(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક nc = src0->ne[0];
    અચલ પૂર્ણાંક nr = ggml_nrows(src0);

    // rows per thread
    અચલ પૂર્ણાંક dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr*ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    માટે (પૂર્ણાંક ir = ir0; ir < ir1; ++ir) {
        અચલ પૂર્ણાંક i3 = ir/(ne02*ne01);
        અચલ પૂર્ણાંક i2 = (ir - i3*ne02*ne01)/ne01;
        અચલ પૂર્ણાંક i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_erf_f32(nc,
                (દશાંશ *) ((અક્ષર *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (દશાંશ *) ((અક્ષર *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        માટે (પૂર્ણાંક k = 0; k < nc; k++) {
            અચલ દશાંશ x = ((દશાંશ *) ((અક્ષર *) dst->data + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_UNUSED(x);
            પ્રતિજ્ઞા(!isnan(x));
            પ્રતિજ્ઞા(!isinf(x));
        }
#endif // NDEBUG
    }
}

સ્થિર શૂન્ય ggml_compute_forward_gelu_erf_f16(
    અચલ ggml_compute_params * params,
    ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પ્રતિજ્ઞા(ggml_is_contiguous_rows(src0));
    પ્રતિજ્ઞા(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક nc = src0->ne[0];
    અચલ પૂર્ણાંક nr = ggml_nrows(src0);

    // rows per thread
    અચલ પૂર્ણાંક dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr*ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    માટે (પૂર્ણાંક ir = ir0; ir < ir1; ++ir) {
        અચલ પૂર્ણાંક i3 = ir/(ne02*ne01);
        અચલ પૂર્ણાંક i2 = (ir - i3*ne02*ne01)/ne01;
        અચલ પૂર્ણાંક i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_erf_f16(nc,
                (ggml_fp16_t *) ((અક્ષર *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((અક્ષર *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        માટે (પૂર્ણાંક k = 0; k < nc; k++) {
            અચલ ggml_fp16_t x = ((ggml_fp16_t *) ((અક્ષર *) dst->data + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            અચલ દશાંશ v = GGML_CPU_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            પ્રતિજ્ઞા(!isnan(v));
            પ્રતિજ્ઞા(!isinf(v));
        }
#endif // NDEBUG
    }
}

સ્થિર શૂન્ય ggml_compute_forward_gelu_erf_bf16(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_rows<ggml_vec_gelu_erf_bf16>(params, dst);
}

સ્થિર શૂન્ય ggml_compute_forward_gelu_erf(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_gelu_erf_f32(params, dst);
            } તોડો;
        સ્થિતિ GGML_TYPE_F16:
            {
                ggml_compute_forward_gelu_erf_f16(params, dst);
            } તોડો;
        સ્થિતિ GGML_TYPE_BF16:
            {
                ggml_compute_forward_gelu_erf_bf16(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_gelu_quick

સ્થિર શૂન્ય ggml_compute_forward_gelu_quick_f32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પ્રતિજ્ઞા(ggml_is_contiguous_rows(src0));
    પ્રતિજ્ઞા(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક nc = src0->ne[0];
    અચલ પૂર્ણાંક nr = ggml_nrows(src0);

    // rows per thread
    અચલ પૂર્ણાંક dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr*ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    માટે (પૂર્ણાંક ir = ir0; ir < ir1; ++ir) {
        અચલ પૂર્ણાંક i3 = ir/(ne02*ne01);
        અચલ પૂર્ણાંક i2 = (ir - i3*ne02*ne01)/ne01;
        અચલ પૂર્ણાંક i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_quick_f32(nc,
                (દશાંશ *) ((અક્ષર *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (દશાંશ *) ((અક્ષર *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        માટે (પૂર્ણાંક k = 0; k < nc; k++) {
            અચલ દશાંશ x = ((દશાંશ *) ((અક્ષર *) dst->data + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_UNUSED(x);
            પ્રતિજ્ઞા(!isnan(x));
            પ્રતિજ્ઞા(!isinf(x));
        }
#endif // NDEBUG
    }
}

સ્થિર શૂન્ય ggml_compute_forward_gelu_quick_f16(
    અચલ ggml_compute_params * params,
    ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પ્રતિજ્ઞા(ggml_is_contiguous_rows(src0));
    પ્રતિજ્ઞા(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક nc = src0->ne[0];
    અચલ પૂર્ણાંક nr = ggml_nrows(src0);

    // rows per thread
    અચલ પૂર્ણાંક dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr*ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    માટે (પૂર્ણાંક ir = ir0; ir < ir1; ++ir) {
        અચલ પૂર્ણાંક i3 = ir/(ne02*ne01);
        અચલ પૂર્ણાંક i2 = (ir - i3*ne02*ne01)/ne01;
        અચલ પૂર્ણાંક i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_quick_f16(nc,
                (ggml_fp16_t *) ((અક્ષર *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((અક્ષર *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        માટે (પૂર્ણાંક k = 0; k < nc; k++) {
            અચલ ggml_fp16_t x = ((ggml_fp16_t *) ((અક્ષર *) dst->data + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            અચલ દશાંશ v = GGML_CPU_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            પ્રતિજ્ઞા(!isnan(v));
            પ્રતિજ્ઞા(!isinf(v));
        }
#endif // NDEBUG
    }
}

સ્થિર શૂન્ય ggml_compute_forward_gelu_quick_bf16(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_rows<ggml_vec_gelu_quick_bf16>(params, dst);
}

સ્થિર શૂન્ય ggml_compute_forward_gelu_quick(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_gelu_quick_f32(params, dst);
            } તોડો;
        સ્થિતિ GGML_TYPE_F16:
            {
                ggml_compute_forward_gelu_quick_f16(params, dst);
            } તોડો;
        સ્થિતિ GGML_TYPE_BF16:
            {
                ggml_compute_forward_gelu_quick_bf16(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_silu

સ્થિર શૂન્ય ggml_compute_forward_silu_f32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પ્રતિજ્ઞા(ggml_is_contiguous_rows(src0));
    પ્રતિજ્ઞા(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક nc = src0->ne[0];
    અચલ પૂર્ણાંક nr = ggml_nrows(src0);

    // rows per thread
    અચલ પૂર્ણાંક dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr*ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    માટે (પૂર્ણાંક ir = ir0; ir < ir1; ++ir) {
        અચલ પૂર્ણાંક i3 = ir/(ne02*ne01);
        અચલ પૂર્ણાંક i2 = (ir - i3*ne02*ne01)/ne01;
        અચલ પૂર્ણાંક i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_silu_f32(nc,
                (દશાંશ *) ((અક્ષર *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (દશાંશ *) ((અક્ષર *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        માટે (પૂર્ણાંક k = 0; k < nc; k++) {
            અચલ દશાંશ x = ((દશાંશ *) ((અક્ષર *) dst->data + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_UNUSED(x);
            પ્રતિજ્ઞા(!isnan(x));
            પ્રતિજ્ઞા(!isinf(x));
        }
#endif // NDEBUG
    }
}

સ્થિર શૂન્ય ggml_compute_forward_silu_f16(
    અચલ ggml_compute_params * params,
    ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પ્રતિજ્ઞા(ggml_is_contiguous_rows(src0));
    પ્રતિજ્ઞા(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક nc = src0->ne[0];
    અચલ પૂર્ણાંક nr = ggml_nrows(src0);

    // rows per thread
    અચલ પૂર્ણાંક dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr*ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    માટે (પૂર્ણાંક ir = ir0; ir < ir1; ++ir) {
        અચલ પૂર્ણાંક i3 = ir/(ne02*ne01);
        અચલ પૂર્ણાંક i2 = (ir - i3*ne02*ne01)/ne01;
        અચલ પૂર્ણાંક i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_silu_f16(nc,
                (ggml_fp16_t *) ((અક્ષર *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((અક્ષર *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        માટે (પૂર્ણાંક k = 0; k < nc; k++) {
            અચલ ggml_fp16_t x = ((ggml_fp16_t *) ((અક્ષર *) dst->data + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            અચલ દશાંશ v = GGML_CPU_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            પ્રતિજ્ઞા(!isnan(v));
            પ્રતિજ્ઞા(!isinf(v));
        }
#endif // NDEBUG
    }
}

સ્થિર શૂન્ય ggml_compute_forward_silu_bf16(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_rows<ggml_vec_silu_bf16>(params, dst);
}

સ્થિર શૂન્ય ggml_compute_forward_silu(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_silu_f32(params, dst);
            } તોડો;
        સ્થિતિ GGML_TYPE_F16:
            {
                ggml_compute_forward_silu_f16(params, dst);
            } તોડો;
        સ્થિતિ GGML_TYPE_BF16:
            {
                ggml_compute_forward_silu_bf16(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}
// ggml_compute_forward_leaky_relu

સ્થિર શૂન્ય ggml_compute_forward_leaky_relu_f32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    જો (params->ith != 0) {
        પાછા;
    }

    પ્રતિજ્ઞા(ggml_is_contiguous_1(src0));
    પ્રતિજ્ઞા(ggml_is_contiguous_1(dst));
    પ્રતિજ્ઞા(ggml_are_same_shape(src0, dst));

    અચલ પૂર્ણાંક n  = ggml_nrows(src0);
    અચલ પૂર્ણાંક nc = src0->ne[0];

    દશાંશ negative_slope;
    memcpy(&negative_slope, dst->op_params, કદ(દશાંશ));

    પ્રતિજ્ઞા(dst->nb[0]  == કદ(દશાંશ));
    પ્રતિજ્ઞા(src0->nb[0] == કદ(દશાંશ));

    માટે (પૂર્ણાંક i = 0; i < n; i++) {
        ggml_vec_leaky_relu_f32(nc,
                (દશાંશ *) ((અક્ષર *) dst->data  + i*( dst->nb[1])),
                (દશાંશ *) ((અક્ષર *) src0->data + i*(src0->nb[1])), negative_slope);
    }
}

સ્થિર શૂન્ય ggml_compute_forward_leaky_relu_f16(
    અચલ ggml_compute_params * params,
    ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    જો (params->ith != 0) {
        પાછા;
    }

    પ્રતિજ્ઞા(ggml_is_contiguous_1(src0));
    પ્રતિજ્ઞા(ggml_is_contiguous_1(dst));
    પ્રતિજ્ઞા(ggml_are_same_shape(src0, dst));

    અચલ પૂર્ણાંક n  = ggml_nrows(src0);
    અચલ પૂર્ણાંક nc = src0->ne[0];

    દશાંશ negative_slope;
    memcpy(&negative_slope, dst->op_params, કદ(દશાંશ));

    પ્રતિજ્ઞા(dst->nb[0]  == કદ(ggml_fp16_t));
    પ્રતિજ્ઞા(src0->nb[0] == કદ(ggml_fp16_t));

    માટે (પૂર્ણાંક i = 0; i < n; i++) {
        ggml_vec_leaky_relu_f16(nc,
                (ggml_fp16_t *) ((અક્ષર *) dst->data  + i*( dst->nb[1])),
                (ggml_fp16_t *) ((અક્ષર *) src0->data + i*(src0->nb[1])), negative_slope);
    }
}

શૂન્ય ggml_compute_forward_leaky_relu(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_leaky_relu_f32(params, dst);
            } તોડો;
        સ્થિતિ GGML_TYPE_F16:
            {
                ggml_compute_forward_leaky_relu_f16(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_silu_back

સ્થિર શૂન્ય ggml_compute_forward_silu_back_f32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * grad = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];

    પ્રતિજ્ઞા(ggml_is_contiguous_1(grad));
    પ્રતિજ્ઞા(ggml_is_contiguous_1(src1));
    પ્રતિજ્ઞા(ggml_is_contiguous_1(dst));
    પ્રતિજ્ઞા(ggml_are_same_shape(src1, dst));
    પ્રતિજ્ઞા(ggml_are_same_shape(src1, grad));

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક nc = src1->ne[0];
    અચલ પૂર્ણાંક nr = ggml_nrows(src1);

    // rows per thread
    અચલ પૂર્ણાંક dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr*ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    માટે (પૂર્ણાંક i1 = ir0; i1 < ir1; i1++) {
        ggml_vec_silu_backward_f32(nc,
                (દશાંશ *) ((અક્ષર *) dst->data  + i1*( dst->nb[1])),
                (દશાંશ *) ((અક્ષર *) src1->data + i1*(src1->nb[1])),
                (દશાંશ *) ((અક્ષર *) grad->data + i1*(grad->nb[1])));

#ifndef NDEBUG
        માટે (પૂર્ણાંક k = 0; k < nc; k++) {
            અચલ દશાંશ x = ((દશાંશ *) ((અક્ષર *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            પ્રતિજ્ઞા(!isnan(x));
            પ્રતિજ્ઞા(!isinf(x));
        }
#endif // NDEBUG
    }
}

સ્થિર શૂન્ય ggml_compute_forward_silu_back_f16(
    અચલ ggml_compute_params * params,
    ggml_tensor * dst) {

    અચલ ggml_tensor * grad = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];

    પ્રતિજ્ઞા(ggml_is_contiguous_1(grad));
    પ્રતિજ્ઞા(ggml_is_contiguous_1(src1));
    પ્રતિજ્ઞા(ggml_is_contiguous_1(dst));
    પ્રતિજ્ઞા(ggml_are_same_shape(src1, dst));
    પ્રતિજ્ઞા(ggml_are_same_shape(src1, grad));

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક nc = src1->ne[0];
    અચલ પૂર્ણાંક nr = ggml_nrows(src1);

    // rows per thread
    અચલ પૂર્ણાંક dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr*ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    માટે (પૂર્ણાંક i1 = ir0; i1 < ir1; i1++) {
        ggml_vec_silu_backward_f16(nc,
                (ggml_fp16_t *) ((અક્ષર *) dst->data  + i1*( dst->nb[1])),
                (ggml_fp16_t *) ((અક્ષર *) src1->data + i1*(src1->nb[1])),
                (ggml_fp16_t *) ((અક્ષર *) grad->data + i1*(grad->nb[1])));

#ifndef NDEBUG
        માટે (પૂર્ણાંક k = 0; k < nc; k++) {
            અચલ દશાંશ x = ((ggml_fp16_t *) ((અક્ષર *) dst->data + i1*( dst->nb[1])))[k];
            અચલ દશાંશ v = GGML_CPU_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            પ્રતિજ્ઞા(!isnan(v));
            પ્રતિજ્ઞા(!isinf(v));
        }
#endif // NDEBUG
    }
}

શૂન્ય ggml_compute_forward_silu_back(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_silu_back_f32(params, dst);
            } તોડો;
        સ્થિતિ GGML_TYPE_F16:
            {
                ggml_compute_forward_silu_back_f16(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_reglu

// row loop shared by the BF16 GLU ops
template <શૂન્ય (*vec_op)(અચલ પૂર્ણાંક, ggml_bf16_t *, અચલ ggml_bf16_t *, અચલ ggml_bf16_t *)>
સ્થિર શૂન્ય ggml_compute_forward_glu_bf16_rows(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];
    અક્ષર * src0_d = (અક્ષર *) src0->data;
    અક્ષર * src1_d = (અક્ષર *) (src1 ? src1->data : src0->data);
    અચલ size_t src0_o = src0->nb[1];
    અચલ size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    જો (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    અચલ પૂર્ણાંક nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    અચલ int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    અચલ પૂર્ણાંક dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr*ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    માટે (પૂર્ણાંક i1 = ir0; i1 < ir1; i1++) {
        અચલ ggml_bf16_t * src0_p = (અચલ ggml_bf16_t *) (src0_d + i1*src0_o);
        અચલ ggml_bf16_t * src1_p = (અચલ ggml_bf16_t *) (src1_d + i1*src1_o);
        ggml_bf16_t       * dst_p  = (ggml_bf16_t *) ((અક્ષર *) dst->data + i1*(dst->nb[1]));

        જો (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        vec_op(nc, dst_p, src0_p, src1_p);
    }
}

સ્થિર શૂન્ય ggml_compute_forward_reglu_f32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];
    અક્ષર * src0_d = (અક્ષર *) src0->data;
    અક્ષર * src1_d = (અક્ષર *) (src1 ? src1->data : src0->data);
    અચલ size_t src0_o = src0->nb[1];
    અચલ size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    જો (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    અચલ પૂર્ણાંક nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    અચલ int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    અચલ પૂર્ણાંક dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr*ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    માટે (પૂર્ણાંક i1 = ir0; i1 < ir1; i1++) {
        દશાંશ * src0_p = (દશાંશ *) (src0_d + i1*src0_o);
        દશાંશ * src1_p = (દશાંશ *) (src1_d + i1*src1_o);

        જો (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_reglu_f32(nc, (દશાંશ *) ((અક્ષર *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        માટે (પૂર્ણાંક k = 0; k < nc; k++) {
            અચલ દશાંશ x = ((દશાંશ *) ((અક્ષર *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            પ્રતિજ્ઞા(!isnan(x));
            પ્રતિજ્ઞા(!isinf(x));
        }
#endif // NDEBUG
    }
}

સ્થિર શૂન્ય ggml_compute_forward_reglu_f16(
    અચલ ggml_compute_params * params,
    ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];
    અક્ષર * src0_d = (અક્ષર *) src0->data;
    અક્ષર * src1_d = (અક્ષર *) (src1 ? src1->data : src0->data);
    અચલ size_t src0_o = src0->nb[1];
    અચલ size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    જો (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    અચલ પૂર્ણાંક nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    અચલ int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    અચલ પૂર્ણાંક dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr*ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    માટે (પૂર્ણાંક i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        જો (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_reglu_f16(nc, (ggml_fp16_t *) ((અક્ષર *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        માટે (પૂર્ણાંક k = 0; k < nc; k++) {
            અચલ ggml_fp16_t x = ((ggml_fp16_t *) ((અક્ષર *) dst->data + i1*( dst->nb[1])))[k];
            અચલ દશાંશ v = GGML_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            પ્રતિજ્ઞા(!isnan(v));
            પ્રતિજ્ઞા(!isinf(v));
        }
#endif // NDEBUG
    }
}

સ્થિર શૂન્ય ggml_compute_forward_reglu_bf16(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_rows<ggml_vec_reglu_bf16>(params, dst);
}

સ્થિર શૂન્ય ggml_compute_forward_reglu(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_reglu_f32(params, dst);
            } તોડો;
        સ્થિતિ GGML_TYPE_F16:
            {
                ggml_compute_forward_reglu_f16(params, dst);
            } તોડો;
        સ્થિતિ GGML_TYPE_BF16:
            {
                ggml_compute_forward_reglu_bf16(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_geglu

સ્થિર શૂન્ય ggml_compute_forward_geglu_f32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];
    અક્ષર * src0_d = (અક્ષર *) src0->data;
    અક્ષર * src1_d = (અક્ષર *) (src1 ? src1->data : src0->data);
    અચલ size_t src0_o = src0->nb[1];
    અચલ size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    જો (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    અચલ પૂર્ણાંક nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    અચલ int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    અચલ પૂર્ણાંક dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr*ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    માટે (પૂર્ણાંક i1 = ir0; i1 < ir1; i1++) {
        દશાંશ * src0_p = (દશાંશ *) (src0_d + i1*src0_o);
        દશાંશ * src1_p = (દશાંશ *) (src1_d + i1*src1_o);

        જો (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_f32(nc, (દશાંશ *) ((અક્ષર *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        માટે (પૂર્ણાંક k = 0; k < nc; k++) {
            અચલ દશાંશ x = ((દશાંશ *) ((અક્ષર *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            પ્રતિજ્ઞા(!isnan(x));
            પ્રતિજ્ઞા(!isinf(x));
        }
#endif // NDEBUG
    }
}

સ્થિર શૂન્ય ggml_compute_forward_geglu_f16(
    અચલ ggml_compute_params * params,
    ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];
    અક્ષર * src0_d = (અક્ષર *) src0->data;
    અક્ષર * src1_d = (અક્ષર *) (src1 ? src1->data : src0->data);
    અચલ size_t src0_o = src0->nb[1];
    અચલ size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    જો (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    અચલ પૂર્ણાંક nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    અચલ int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    અચલ પૂર્ણાંક dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr*ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    માટે (પૂર્ણાંક i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        જો (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_f16(nc, (ggml_fp16_t *) ((અક્ષર *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        માટે (પૂર્ણાંક k = 0; k < nc; k++) {
            અચલ ggml_fp16_t x = ((ggml_fp16_t *) ((અક્ષર *) dst->data + i1*( dst->nb[1])))[k];
            અચલ દશાંશ v = GGML_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            પ્રતિજ્ઞા(!isnan(v));
            પ્રતિજ્ઞા(!isinf(v));
        }
#endif // NDEBUG
    }
}

સ્થિર શૂન્ય ggml_compute_forward_geglu_bf16(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_rows<ggml_vec_geglu_bf16>(params, dst);
}

સ્થિર શૂન્ય ggml_compute_forward_geglu(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_geglu_f32(params, dst);
            } તોડો;
        સ્થિતિ GGML_TYPE_F16:
            {
                ggml_compute_forward_geglu_f16(params, dst);
            } તોડો;
        સ્થિતિ GGML_TYPE_BF16:
            {
                ggml_compute_forward_geglu_bf16(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_swiglu

સ્થિર શૂન્ય ggml_compute_forward_swiglu_f32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];
    અક્ષર * src0_d = (અક્ષર *) src0->data;
    અક્ષર * src1_d = (અક્ષર *) (src1 ? src1->data : src0->data);
    અચલ size_t src0_o = src0->nb[1];
    અચલ size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    જો (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    અચલ પૂર્ણાંક nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    અચલ int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    અચલ પૂર્ણાંક dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr*ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    માટે (પૂર્ણાંક i1 = ir0; i1 < ir1; i1++) {
        દશાંશ * src0_p = (દશાંશ *) (src0_d + i1*src0_o);
        દશાંશ * src1_p = (દશાંશ *) (src1_d + i1*src1_o);

        જો (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_swiglu_f32(nc, (દશાંશ *) ((અક્ષર *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        માટે (પૂર્ણાંક k = 0; k < nc; k++) {
            અચલ દશાંશ x = ((દશાંશ *) ((અક્ષર *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            પ્રતિજ્ઞા(!isnan(x));
            પ્રતિજ્ઞા(!isinf(x));
        }
#endif // NDEBUG
    }
}

સ્થિર શૂન્ય ggml_compute_forward_swiglu_f16(
    અચલ ggml_compute_params * params,
    ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];
    અક્ષર * src0_d = (અક્ષર *) src0->data;
    અક્ષર * src1_d = (અક્ષર *) (src1 ? src1->data : src0->data);
    અચલ size_t src0_o = src0->nb[1];
    અચલ size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    જો (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    અચલ પૂર્ણાંક nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    અચલ int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    અચલ પૂર્ણાંક dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr*ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    માટે (પૂર્ણાંક i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        જો (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_swiglu_f16(nc, (ggml_fp16_t *) ((અક્ષર *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        માટે (પૂર્ણાંક k = 0; k < nc; k++) {
            અચલ ggml_fp16_t x = ((ggml_fp16_t *) ((અક્ષર *) dst->data + i1*( dst->nb[1])))[k];
            અચલ દશાંશ v = GGML_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            પ્રતિજ્ઞા(!isnan(v));
            પ્રતિજ્ઞા(!isinf(v));
        }
#endif // NDEBUG
    }
}

સ્થિર શૂન્ય ggml_compute_forward_swiglu_bf16(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_rows<ggml_vec_swiglu_bf16>(params, dst);
}

સ્થિર શૂન્ય ggml_compute_forward_swiglu(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_swiglu_f32(params, dst);
            } તોડો;
        સ્થિતિ GGML_TYPE_F16:
            {
                ggml_compute_forward_swiglu_f16(params, dst);
            } તોડો;
        સ્થિતિ GGML_TYPE_BF16:
            {
                ggml_compute_forward_swiglu_bf16(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_swiglu_oai

સ્થિર શૂન્ય ggml_compute_forward_swiglu_oai_f32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];
    અક્ષર * src0_d = (અક્ષર *) src0->data;
    અક્ષર * src1_d = (અક્ષર *) (src1 ? src1->data : src0->data);
    અચલ size_t src0_o = src0->nb[1];
    અચલ size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    જો (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    અચલ પૂર્ણાંક nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    અચલ int32_t swapped = ggml_get_op_params_i32(dst, 1);
    અચલ દશાંશ alpha = ggml_get_op_params_f32(dst, 2);
    અચલ દશાંશ limit = ggml_get_op_params_f32(dst, 3);

    // rows per thread
    અચલ પૂર્ણાંક dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr*ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    માટે (પૂર્ણાંક i1 = ir0; i1 < ir1; i1++) {
        દશાંશ * src0_p = (દશાંશ *) (src0_d + i1*src0_o);
        દશાંશ * src1_p = (દશાંશ *) (src1_d + i1*src1_o);
        દશાંશ * dst_p  = (દશાંશ *) ((અક્ષર *) dst->data + i1*(dst->nb[1]));

        જો (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        માટે (પૂર્ણાંક k = 0; k < nc; k++) {
            અચલ દશાંશ x = std::min(src0_p[k], limit);
            અચલ દશાંશ y = std::clamp(src1_p[k], -limit, limit);
            અચલ દશાંશ out_glu = x / (1.f + expf(alpha * (-x)));
            dst_p[k] = out_glu * (y + 1.f);
        }

#ifndef NDEBUG
        માટે (પૂર્ણાંક k = 0; k < nc; k++) {
            અચલ દશાંશ x = dst_p[k];
            GGML_UNUSED(x);
            પ્રતિજ્ઞા(!isnan(x));
            પ્રતિજ્ઞા(!isinf(x));
        }
#endif // NDEBUG
    }
}

સ્થિર શૂન્ય ggml_compute_forward_swiglu_oai_f16(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];
    અક્ષર * src0_d = (અક્ષર *) src0->data;
    અક્ષર * src1_d = (અક્ષર *) (src1 ? src1->data : src0->data);
    અચલ size_t src0_o = src0->nb[1];
    અચલ size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    જો (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    અચલ પૂર્ણાંક nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    અચલ int32_t swapped = ggml_get_op_params_i32(dst, 1);
    અચલ દશાંશ alpha = ggml_get_op_params_f32(dst, 2);
    અચલ દશાંશ limit = ggml_get_op_params_f32(dst, 3);

    // rows per thread
    અચલ પૂર્ણાંક dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr*ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    માટે (પૂર્ણાંક i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);
        ggml_fp16_t * dst_p  = (ggml_fp16_t *) ((અક્ષર *) dst->data + i1*(dst->nb[1]));

        જો (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        માટે (પૂર્ણાંક k = 0; k < nc; k++) {
            અચલ દશાંશ x = std::min(GGML_CPU_FP16_TO_FP32(src0_p[k]), limit);
            અચલ દશાંશ y = std::clamp(GGML_CPU_FP16_TO_FP32(src1_p[k]), -limit, limit);
            અચલ દશાંશ out_glu = x / (1.f + expf(alpha * (-x)));
            dst_p[k] = GGML_CPU_FP32_TO_FP16(out_glu * (y + 1.f));
        }

#ifndef NDEBUG
        માટે (પૂર્ણાંક k = 0; k < nc; k++) {
            અચલ દશાંશ x = GGML_CPU_FP16_TO_FP32(dst_p[k]);
            GGML_UNUSED(x);
            પ્રતિજ્ઞા(!isnan(x));
            પ્રતિજ્ઞા(!isinf(x));
        }
#endif // NDEBUG
    }
}

સ્થિર શૂન્ય ggml_compute_forward_swiglu_oai(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_swiglu_oai_f32(params, dst);
            } તોડો;
        સ્થિતિ GGML_TYPE_F16:
            {
                ggml_compute_forward_swiglu_oai_f16(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_swiglu_clamp

સ્થિર શૂન્ય ggml_compute_forward_swiglu_clamp_f32(અચલ ggml_compute_params * params, ggml_tensor * dst) {
    અચલ ggml_tensor * src0   = dst->src[0];
    અચલ ggml_tensor * src1   = dst->src[1];
    અક્ષર *              src0_d = (અક્ષર *) src0->data;
    અક્ષર *              src1_d = (અક્ષર *) (src1 ? src1->data : src0->data);
    અચલ size_t        src0_o = src0->nb[1];
    અચલ size_t        src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    જો (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    અચલ પૂર્ણાંક nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    અચલ int32_t swapped = ggml_get_op_params_i32(dst, 1);
    અચલ દશાંશ   limit   = ggml_get_op_params_f32(dst, 3);

    અચલ પૂર્ણાંક dr  = (nr + nth - 1) / nth;
    અચલ પૂર્ણાંક ir0 = dr * ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    માટે (પૂર્ણાંક i1 = ir0; i1 < ir1; i1++) {
        દશાંશ * src0_p = (દશાંશ *) (src0_d + i1 * src0_o);
        દશાંશ * src1_p = (દશાંશ *) (src1_d + i1 * src1_o);
        દશાંશ * dst_p  = (દશાંશ *) ((અક્ષર *) dst->data + i1 * (dst->nb[1]));

        જો (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        માટે (પૂર્ણાંક k = 0; k < nc; k++) {
            અચલ દશાંશ gate = std::min(src0_p[k], limit);
            અચલ દશાંશ up   = std::clamp(src1_p[k], -limit, limit);
            dst_p[k]         = gate / (1.f + expf(-gate)) * up;
        }

#ifndef NDEBUG
        માટે (પૂર્ણાંક k = 0; k < nc; k++) {
            અચલ દશાંશ x = dst_p[k];
            GGML_UNUSED(x);
            પ્રતિજ્ઞા(!isnan(x));
            પ્રતિજ્ઞા(!isinf(x));
        }
#endif  // NDEBUG
    }
}

સ્થિર શૂન્ય ggml_compute_forward_swiglu_clamp_f16(અચલ ggml_compute_params * params, ggml_tensor * dst) {
    અચલ ggml_tensor * src0   = dst->src[0];
    અચલ ggml_tensor * src1   = dst->src[1];
    અક્ષર *              src0_d = (અક્ષર *) src0->data;
    અક્ષર *              src1_d = (અક્ષર *) (src1 ? src1->data : src0->data);
    અચલ size_t        src0_o = src0->nb[1];
    અચલ size_t        src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    જો (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    અચલ પૂર્ણાંક nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    અચલ int32_t swapped = ggml_get_op_params_i32(dst, 1);
    અચલ દશાંશ   limit   = ggml_get_op_params_f32(dst, 3);

    અચલ પૂર્ણાંક dr  = (nr + nth - 1) / nth;
    અચલ પૂર્ણાંક ir0 = dr * ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    માટે (પૂર્ણાંક i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1 * src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1 * src1_o);
        ggml_fp16_t * dst_p  = (ggml_fp16_t *) ((અક્ષર *) dst->data + i1 * (dst->nb[1]));

        જો (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        માટે (પૂર્ણાંક k = 0; k < nc; k++) {
            અચલ દશાંશ gate = std::min(GGML_FP16_TO_FP32(src0_p[k]), limit);
            અચલ દશાંશ up   = std::clamp(GGML_FP16_TO_FP32(src1_p[k]), -limit, limit);
            dst_p[k]         = GGML_FP32_TO_FP16(gate / (1.f + expf(-gate)) * up);
        }

#ifndef NDEBUG
        માટે (પૂર્ણાંક k = 0; k < nc; k++) {
            અચલ દશાંશ x = GGML_FP16_TO_FP32(dst_p[k]);
            GGML_UNUSED(x);
            પ્રતિજ્ઞા(!isnan(x));
            પ્રતિજ્ઞા(!isinf(x));
        }
#endif  // NDEBUG
    }
}

સ્થિર શૂન્ય ggml_compute_forward_swiglu_clamp(અચલ ggml_compute_params * params, ggml_tensor * dst) {
    પસંદગી (dst->src[0]->type) {
        સ્થિતિ GGML_TYPE_F32:
            ggml_compute_forward_swiglu_clamp_f32(params, dst);
            તોડો;
        સ્થિતિ GGML_TYPE_F16:
            ggml_compute_forward_swiglu_clamp_f16(params, dst);
            તોડો;
        મૂળભૂત:
            GGML_ABORT("fatal error");
    }
}

// ggml_compute_forward_geglu_erf

સ્થિર શૂન્ય ggml_compute_forward_geglu_erf_f32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];
    અક્ષર * src0_d = (અક્ષર *) src0->data;
    અક્ષર * src1_d = (અક્ષર *) (src1 ? src1->data : src0->data);
    અચલ size_t src0_o = src0->nb[1];
    અચલ size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    જો (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    અચલ પૂર્ણાંક nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    અચલ int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    અચલ પૂર્ણાંક dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr*ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    માટે (પૂર્ણાંક i1 = ir0; i1 < ir1; i1++) {
        દશાંશ * src0_p = (દશાંશ *) (src0_d + i1*src0_o);
        દશાંશ * src1_p = (દશાંશ *) (src1_d + i1*src1_o);

        જો (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_erf_f32(nc, (દશાંશ *) ((અક્ષર *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        માટે (પૂર્ણાંક k = 0; k < nc; k++) {
            અચલ દશાંશ x = ((દશાંશ *) ((અક્ષર *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            પ્રતિજ્ઞા(!isnan(x));
            પ્રતિજ્ઞા(!isinf(x));
        }
#endif // NDEBUG
    }
}

સ્થિર શૂન્ય ggml_compute_forward_geglu_erf_f16(
    અચલ ggml_compute_params * params,
    ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];
    અક્ષર * src0_d = (અક્ષર *) src0->data;
    અક્ષર * src1_d = (અક્ષર *) (src1 ? src1->data : src0->data);
    અચલ size_t src0_o = src0->nb[1];
    અચલ size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    જો (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    અચલ પૂર્ણાંક nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    અચલ int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    અચલ પૂર્ણાંક dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr*ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    માટે (પૂર્ણાંક i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        જો (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_erf_f16(nc, (ggml_fp16_t *) ((અક્ષર *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        માટે (પૂર્ણાંક k = 0; k < nc; k++) {
            અચલ ggml_fp16_t x = ((ggml_fp16_t *) ((અક્ષર *) dst->data + i1*( dst->nb[1])))[k];
            અચલ દશાંશ v = GGML_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            પ્રતિજ્ઞા(!isnan(v));
            પ્રતિજ્ઞા(!isinf(v));
        }
#endif // NDEBUG
    }
}

સ્થિર શૂન્ય ggml_compute_forward_geglu_erf_bf16(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_rows<ggml_vec_geglu_erf_bf16>(params, dst);
}

સ્થિર શૂન્ય ggml_compute_forward_geglu_erf(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_geglu_erf_f32(params, dst);
            } તોડો;
        સ્થિતિ GGML_TYPE_F16:
            {
                ggml_compute_forward_geglu_erf_f16(params, dst);
            } તોડો;
        સ્થિતિ GGML_TYPE_BF16:
            {
                ggml_compute_forward_geglu_erf_bf16(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_geglu_quick

સ્થિર શૂન્ય ggml_compute_forward_geglu_quick_f32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];
    અક્ષર * src0_d = (અક્ષર *) src0->data;
    અક્ષર * src1_d = (અક્ષર *) (src1 ? src1->data : src0->data);
    અચલ size_t src0_o = src0->nb[1];
    અચલ size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    જો (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    અચલ પૂર્ણાંક nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    અચલ int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    અચલ પૂર્ણાંક dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr*ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    માટે (પૂર્ણાંક i1 = ir0; i1 < ir1; i1++) {
        દશાંશ * src0_p = (દશાંશ *) (src0_d + i1*src0_o);
        દશાંશ * src1_p = (દશાંશ *) (src1_d + i1*src1_o);

        જો (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_quick_f32(nc, (દશાંશ *) ((અક્ષર *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        માટે (પૂર્ણાંક k = 0; k < nc; k++) {
            અચલ દશાંશ x = ((દશાંશ *) ((અક્ષર *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            પ્રતિજ્ઞા(!isnan(x));
            પ્રતિજ્ઞા(!isinf(x));
        }
#endif // NDEBUG
    }
}

સ્થિર શૂન્ય ggml_compute_forward_geglu_quick_f16(
    અચલ ggml_compute_params * params,
    ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];
    અક્ષર * src0_d = (અક્ષર *) src0->data;
    અક્ષર * src1_d = (અક્ષર *) (src1 ? src1->data : src0->data);
    અચલ size_t src0_o = src0->nb[1];
    અચલ size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    જો (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    અચલ પૂર્ણાંક nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    અચલ int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    અચલ પૂર્ણાંક dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr*ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    માટે (પૂર્ણાંક i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        જો (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_quick_f16(nc, (ggml_fp16_t *) ((અક્ષર *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        માટે (પૂર્ણાંક k = 0; k < nc; k++) {
            અચલ ggml_fp16_t x = ((ggml_fp16_t *) ((અક્ષર *) dst->data + i1*( dst->nb[1])))[k];
            અચલ દશાંશ v = GGML_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            પ્રતિજ્ઞા(!isnan(v));
            પ્રતિજ્ઞા(!isinf(v));
        }
#endif // NDEBUG
    }
}

સ્થિર શૂન્ય ggml_compute_forward_geglu_quick_bf16(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_rows<ggml_vec_geglu_quick_bf16>(params, dst);
}

સ્થિર શૂન્ય ggml_compute_forward_geglu_quick(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_geglu_quick_f32(params, dst);
            } તોડો;
        સ્થિતિ GGML_TYPE_F16:
            {
                ggml_compute_forward_geglu_quick_f16(params, dst);
            } તોડો;
        સ્થિતિ GGML_TYPE_BF16:
            {
                ggml_compute_forward_geglu_quick_bf16(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_norm

સ્થિર શૂન્ય ggml_compute_forward_norm_f32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    દશાંશ eps;
    memcpy(&eps, dst->op_params, કદ(દશાંશ));

    GGML_ASSERT(eps >= 0.0f);

    માટે (int64_t i03 = 0; i03 < ne03; i03++) {
        માટે (int64_t i02 = 0; i02 < ne02; i02++) {
            માટે (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                અચલ અક્ષર * x = (અચલ અક્ષર *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
                અક્ષર * y = (અક્ષર *) dst->data + i01*nb1 + i02*nb2 + i03*nb3;

                જો (nb00 == કદ(દશાંશ) && nb0 == કદ(દશાંશ)) {
                    અચલ દશાંશ * xf = (અચલ દશાંશ *) x;

                    દશાંશ sum = 0.0;
                    ggml_vec_sum_f32(ne00, &sum, xf);
                    દશાંશ mean = sum/ne00;

                    દશાંશ * yf = (દશાંશ *) y;
                    દશાંશ variance = 0;

#ifdef GGML_USE_ACCELERATE
                    mean = -mean;
                    vDSP_vsadd(xf, 1, &mean, yf, 1, ne00);
                    vDSP_measqv(yf, 1, &variance, ne00);
#નહીંતર
                    variance = ggml_vec_cvar_f32(ne00, yf, xf, mean);
#endif //GGML_USE_ACCELERATE

                    અચલ દશાંશ scale = 1.0f/sqrtf(variance + eps);
                    ggml_vec_scale_f32(ne00, yf, scale);
                } નહીંતર {
                    દશાંશ sum = 0.0;
                    માટે (int64_t i00 = 0; i00 < ne00; i00++) {
                        sum += *(અચલ દશાંશ *) (x + i00*nb00);
                    }
                    અચલ દશાંશ mean = sum/ne00;

                    દશાંશ variance = 0.0f;
                    માટે (int64_t i00 = 0; i00 < ne00; i00++) {
                        અચલ દશાંશ v = *(અચલ દશાંશ *) (x + i00*nb00) - mean;
                        *(દશાંશ *) (y + i00*nb0) = v;
                        variance += v * v;
                    }
                    variance /= ne00;

                    અચલ દશાંશ scale = 1.0f/sqrtf(variance + eps);
                    માટે (int64_t i00 = 0; i00 < ne00; i00++) {
                        *(દશાંશ *) (y + i00*nb0) *= scale;
                    }
                }
            }
        }
    }
}

શૂન્ય ggml_compute_forward_norm(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_norm_f32(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_group_rms_norm

// fusion kinds that can be combined with the rms_norm computation in a single pass.
// extend this enum when adding new fused variants (e.g. FUSE_ADD, FUSE_MUL_ADD, ...).
ગણના ggml_rms_norm_fuse_op {
    GGML_RMS_NORM_FUSE_OP_NONE,
    GGML_RMS_NORM_FUSE_OP_MUL,
};

template <ggml_rms_norm_fuse_op FUSE_OP>
સ્થિર શૂન્ય ggml_compute_forward_rms_norm_f32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst_rms_norm,
        ggml_tensor * dst_fused = nullptr) {

    અચલ ggml_tensor * src0 = dst_rms_norm->src[0];
    અચલ ggml_tensor * src1 = nullptr;
    ggml_tensor       * dst  = dst_rms_norm;

    જો constexpr (FUSE_OP == GGML_RMS_NORM_FUSE_OP_MUL) {
        src1 = (dst_fused->src[0] == dst_rms_norm) ? dst_fused->src[1] : dst_fused->src[0];
        dst  = dst_fused;
    }

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    GGML_ASSERT(src0->nb[0] == કદ(દશાંશ));

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    GGML_TENSOR_દ્વિઅંકી_OP_LOCALS

    દશાંશ eps;
    memcpy(&eps, dst_rms_norm->op_params, કદ(દશાંશ));
    GGML_ASSERT(eps >= 0.0f);

    // TODO: optimize
    માટે (int64_t i03 = 0; i03 < ne03; i03++) {
        માટે (int64_t i02 = 0; i02 < ne02; i02++) {
            માટે (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                અચલ દશાંશ * x = (દશાંશ *) ((અક્ષર *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);

                ggml_float sum = 0.0;
                // worth switching to explicit SIMD?
                માટે (int64_t i00 = 0; i00 < ne00; i00++) {
                    sum += (ggml_float)(x[i00] * x[i00]);
                }

                અચલ દશાંશ mean  = sum/ne00;
                અચલ દશાંશ scale = 1.0f/sqrtf(mean + eps);

                // if you hit this, likely you got an inf somewhere earlier
                પ્રતિજ્ઞા(scale > 0.0f);

                દશાંશ * y = (દશાંશ *) ((અક્ષર *) dst->data + i01*nb1 + i02*nb2 + i03*nb3);

                જો constexpr (FUSE_OP == GGML_RMS_NORM_FUSE_OP_MUL) {
                    અચલ int64_t i11 = i01 % ne11;
                    અચલ int64_t i12 = i02 % ne12;
                    અચલ int64_t i13 = i03 % ne13;
                    અચલ દશાંશ * w = (દશાંશ *) ((અક્ષર *) src1->data + i11*nb11 + i12*nb12 + i13*nb13);

                    માટે (int64_t i00 = 0; i00 < ne00; i00++) {
                        y[i00] = x[i00] * scale * w[i00];
                    }
                } નહીંતર {
                    memcpy(y, x, ne00 * કદ(દશાંશ));
                    ggml_vec_scale_f32(ne00, y, scale);
                }
            }
        }
    }
}

શૂન્ય ggml_compute_forward_rms_norm(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_rms_norm_f32<GGML_RMS_NORM_FUSE_OP_NONE>(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// Fused RMS_NORM + MUL: computes dst = rms_norm(src0) * src1 in a single pass.
// This avoids materializing the intermediate rms_norm result in memory.
શૂન્ય ggml_compute_forward_rms_norm_mul_fused(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst_rms_norm,
        ggml_tensor * dst_mul) {

    GGML_ASSERT(dst_mul != nullptr);
    GGML_ASSERT(dst_mul->src[0] == dst_rms_norm || dst_mul->src[1] == dst_rms_norm);

    અચલ ggml_tensor * src0 = dst_rms_norm->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_rms_norm_f32<GGML_RMS_NORM_FUSE_OP_MUL>(params, dst_rms_norm, dst_mul);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

સ્થિર શૂન્ય ggml_compute_forward_rms_norm_back_f32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0]; // gradients from forward pass output
    અચલ ggml_tensor * src1 = dst->src[1]; // src1 from forward pass

    GGML_ASSERT(ggml_are_same_shape(src0, dst) && ggml_are_same_shape(src0, src1));

    GGML_ASSERT(src0->nb[0] == કદ(દશાંશ));
    GGML_ASSERT(src1->nb[0] == કદ(દશાંશ));

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    GGML_TENSOR_દ્વિઅંકી_OP_LOCALS

    દશાંશ eps;
    memcpy(&eps, dst->op_params, કદ(દશાંશ));

    // TODO: optimize
    માટે (int64_t i03 = 0; i03 < ne03; i03++) {
        માટે (int64_t i02 = 0; i02 < ne02; i02++) {
            માટે (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                // src1 is same shape as src0 => same indices
                અચલ int64_t i11 = i01;
                અચલ int64_t i12 = i02;
                અચલ int64_t i13 = i03;

                અચલ દશાંશ * dz = (દશાંશ *) ((અક્ષર *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
                અચલ દશાંશ * x  = (દશાંશ *) ((અક્ષર *) src1->data + i11*nb11 + i12*nb12 + i13*nb13);

                ggml_float sum_xx  = 0.0;
                ggml_float sum_xdz = 0.0;

                માટે (int64_t i00 = 0; i00 < ne00; i00++) {
                    sum_xx  += (ggml_float)(x[i00] * x[i00]);
                    sum_xdz += (ggml_float)(x[i00] * dz[i00]);
                }

                //const float mean     = (float)(sum_xx)/ne00;
                અચલ દશાંશ mean_eps = (દશાંશ)(sum_xx)/ne00 + eps;
                અચલ દશાંશ sum_eps  = (દશાંશ)(sum_xx) + eps*ne00;
                //const float mean_xdz = (float)(sum_xdz)/ne00;
                // we could cache rms from forward pass to improve performance.
                // to do this implement ggml_rms and compose ggml_rms_norm using ggml_rms.
                //const float rms      = sqrtf(mean_eps);
                અચલ દશાંશ rrms     = 1.0f / sqrtf(mean_eps);
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
                દશાંશ * dx = (દશાંશ *) ((અક્ષર *) dst->data + i01*nb1 + i02*nb2 + i03*nb3);

                // dx[i00] = (dz + x*(-sum_xdz/sum_eps)) * rrms
                // note: https://github.com/ggml-org/ggml/issues/1491
                અચલ દશાંશ scale_x = (દશાંશ) (-sum_xdz) / sum_eps;
                માટે (int64_t i00 = 0; i00 < ne00; i00++) {
                    dx[i00] = (dz[i00] + x[i00] * scale_x) * rrms;
                }
            }
        }
    }
}

શૂન્ય ggml_compute_forward_rms_norm_back(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_rms_norm_back_f32(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_group_norm

સ્થિર શૂન્ય ggml_compute_forward_group_norm_f32(
    અચલ ggml_compute_params * params,
    ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    GGML_ASSERT(src0->nb[0] == કદ(દશાંશ));

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    // TODO: optimize

    દશાંશ eps;
    memcpy(&eps, dst->op_params + 1, કદ(દશાંશ));

    પૂર્ણાંક n_channels = src0->ne[2];
    પૂર્ણાંક n_groups = dst->op_params[0];
    પૂર્ણાંક n_channels_per_group = (n_channels + n_groups - 1) / n_groups;
    માટે (પૂર્ણાંક i = ith; i < n_groups; i += nth) {
        પૂર્ણાંક start = i * n_channels_per_group;
        પૂર્ણાંક end = start + n_channels_per_group;
        જો (end > n_channels) {
            end = n_channels;
        }
        પૂર્ણાંક step = end - start;

        માટે (int64_t i03 = 0; i03 < ne03; i03++) {
            ggml_float sum = 0.0;
            માટે (int64_t i02 = start; i02 < end; i02++) {
                માટે (int64_t i01 = 0; i01 < ne01; i01++) {
                    અચલ દશાંશ * x = (દશાંશ *)((અક્ષર *) src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);

                    ggml_float sumr = 0.0;
                    માટે (int64_t i00 = 0; i00 < ne00; i00++) {
                        sumr += (ggml_float)x[i00];
                    }
                    sum += sumr;
                }
            }
            અચલ દશાંશ mean = sum / (ne00 * ne01 * step);

            ggml_float sum2 = 0.0;
            માટે (int64_t i02 = start; i02 < end; i02++) {
                માટે (int64_t i01 = 0; i01 < ne01; i01++) {
                    અચલ દશાંશ * x = (દશાંશ *)((અક્ષર *) src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);

                    દશાંશ * y = (દશાંશ *)((અક્ષર *) dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3);

                    ggml_float sumr = 0.0;
                    માટે (int64_t i00 = 0; i00 < ne00; i00++) {
                        દશાંશ v = x[i00] - mean;
                        y[i00] = v;
                        sumr += (ggml_float)(v * v);
                    }
                    sum2 += sumr;
                }
            }
            અચલ દશાંશ variance = sum2 / (ne00 * ne01 * step);
            અચલ દશાંશ scale = 1.0f / sqrtf(variance + eps);

            માટે (int64_t i02 = start; i02 < end; i02++) {
                માટે (int64_t i01 = 0; i01 < ne01; i01++) {
                    દશાંશ * y = (દશાંશ *)((અક્ષર *) dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3);
                    ggml_vec_scale_f32(ne00, y, scale);
                }
            }
        }
    }
}

શૂન્ય ggml_compute_forward_group_norm(
    અચલ ggml_compute_params * params,
    ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_group_norm_f32(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_l2_norm

સ્થિર શૂન્ય ggml_compute_forward_l2_norm_f32(
    અચલ ggml_compute_params * params,
    ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    દશાંશ eps;
    memcpy(&eps, dst->op_params, કદ(દશાંશ));

    GGML_ASSERT(eps >= 0.0f);

    // TODO: optimize
    માટે (int64_t i03 = 0; i03 < ne03; i03++) {
        માટે (int64_t i02 = 0; i02 < ne02; i02++) {
            માટે (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                અચલ અક્ષર * x = (અચલ અક્ષર *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;

                ggml_float sum = 0.0;
                માટે (int64_t i00 = 0; i00 < ne00; i00++) {
                    અચલ દશાંશ xi = *(અચલ દશાંશ *) (x + i00*nb00);
                    sum += (ggml_float)(xi * xi);
                }

                અચલ દશાંશ scale = 1.0f/fmaxf(sqrtf(sum), eps);

                અક્ષર * y = (અક્ષર *) dst->data + i01*nb1 + i02*nb2 + i03*nb3;

                જો (nb00 == કદ(દશાંશ) && nb0 == કદ(દશાંશ)) {
                    memcpy(y, x, ne00 * કદ(દશાંશ));
                    ggml_vec_scale_f32(ne00, (દશાંશ *) y, scale);
                } નહીંતર {
                    માટે (int64_t i00 = 0; i00 < ne00; i00++) {
                        અચલ દશાંશ xi = *(અચલ દશાંશ *) (x + i00*nb00);
                        *(દશાંશ *) (y + i00*nb0) = xi * scale;
                    }
                }
            }
        }
    }
}

શૂન્ય ggml_compute_forward_l2_norm(
    અચલ ggml_compute_params * params,
    ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_l2_norm_f32(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_out_prod

સ્થિર શૂન્ય ggml_compute_forward_out_prod_f32(
        અચલ ggml_compute_params * params,
              ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_દ્વિઅંકી_OP_LOCALS

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    GGML_ASSERT(ne2 % ne02 == 0);
    GGML_ASSERT(ne3 % ne03 == 0);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == કદ(દશાંશ));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == કદ(દશાંશ));
    // GGML_ASSERT(nb0 <= nb1);
    // GGML_ASSERT(nb1 <= nb2);
    // GGML_ASSERT(nb2 <= nb3);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    જો (ith == 0) {
        ggml_vec_set_f32(ne0*ne1*ne2*ne3, (દશાંશ *)dst->data, 0);
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
    અચલ int64_t nr = ne1*ne2*ne3;

    // rows per thread
    અચલ int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ int64_t ir0 = dr*ith;
    અચલ int64_t ir1 = MIN(ir0 + dr, nr);

    // block-tiling attempt
    અચલ int64_t blck_0 = MAX(GGML_VEC_MAD_UNROLL, 32);
    અચલ int64_t blck_1 = 16;

    // dps == dst per src0, used for group query attention
    અચલ int64_t dps2 = ne2 / ne02;
    અચલ int64_t dps3 = ne3 / ne03;

    માટે (int64_t bir = ir0; bir < ir1; bir += blck_1) {
        અચલ int64_t bir1 = MIN(bir + blck_1, ir1);
        માટે (int64_t bi01 = 0; bi01 < ne01; bi01 += blck_0) {
            અચલ int64_t bne01 = MIN(bi01 + blck_0, ne01);
            માટે (int64_t ir = bir; ir < bir1; ++ir) {
                // dst indices
                અચલ int64_t i3 = ir/(ne2*ne1);
                અચલ int64_t i2 = (ir - i3*ne2*ne1)/ne1;
                અચલ int64_t i1 = (ir - i3*ne2*ne1 - i2*ne1);

                અચલ int64_t i02 = i2 / dps2;
                અચલ int64_t i03 = i3 / dps3;

                //const int64_t i10 = i1;
                અચલ int64_t i12 = i2;
                અચલ int64_t i13 = i3;

#જો GGML_VEC_MAD_UNROLL > 2
                અચલ int64_t bne01_unroll = bne01 - (bne01 % GGML_VEC_MAD_UNROLL);
                માટે (int64_t i01 = bi01; i01 < bne01_unroll; i01 += GGML_VEC_MAD_UNROLL) {
                    અચલ int64_t i11 = i01;

                    દશાંશ * s0 = (દશાંશ *) ((અક્ષર *) src0->data + (          i01*nb01 + i02*nb02 + i03*nb03));
                    દશાંશ * s1 = (દશાંશ *) ((અક્ષર *) src1->data + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
                    દશાંશ * d  = (દશાંશ *) ((અક્ષર *)  dst->data + (          i1*nb1   + i2*nb2   + i3*nb3));

                    ggml_vec_mad_f32_unroll(ne0, nb01, nb11, d, s0, s1);
                }
                માટે (int64_t i01 = bne01_unroll; i01 < bne01; ++i01) {
                    અચલ int64_t i11 = i01;

                    દશાંશ * s0 = (દશાંશ *) ((અક્ષર *) src0->data + (          i01*nb01 + i02*nb02 + i03*nb03));
                    દશાંશ * s1 = (દશાંશ *) ((અક્ષર *) src1->data + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
                    દશાંશ * d  = (દશાંશ *) ((અક્ષર *)  dst->data + (          i1*nb1   + i2*nb2   + i3*nb3));

                    ggml_vec_mad_f32(ne0, d, s0, *s1);
                }
#નહીંતર
                માટે (int64_t i01 = bi01; i01 < bne01; ++i01) {
                    અચલ int64_t i11 = i01;

                    દશાંશ * s0 = (દશાંશ *) ((અક્ષર *) src0->data + (          i01*nb01 + i02*nb02 + i03*nb03));
                    દશાંશ * s1 = (દશાંશ *) ((અક્ષર *) src1->data + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
                    દશાંશ * d  = (દશાંશ *) ((અક્ષર *)  dst->data + (          i1*nb1 + i2*nb2 + i3*nb3));

                    ggml_vec_mad_f32(ne0, d, s0, *s1);
                }
#endif
            }
        }
    }
}

સ્થિર શૂન્ય ggml_compute_forward_out_prod_q_f32(
        અચલ ggml_compute_params * params,
              ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_દ્વિઅંકી_OP_LOCALS;

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ ggml_type type = src0->type;
    ggml_to_float_t અચલ dequantize_row_q = ggml_get_type_traits(type)->to_float;

    GGML_ASSERT(ne02 == ne12);
    GGML_ASSERT(ne03 == ne13);
    GGML_ASSERT(ne2  == ne12);
    GGML_ASSERT(ne3  == ne13);

    // we don't support permuted src0 dim0
    GGML_ASSERT(nb00 == ggml_type_size(type));

    // dst dim0 cannot be transposed or permuted
    GGML_ASSERT(nb0 == કદ(દશાંશ));
    // GGML_ASSERT(nb0 <= nb1);
    // GGML_ASSERT(nb1 <= nb2);
    // GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    જો (ith == 0) {
        ggml_vec_set_f32(ne0*ne1*ne2*ne3, (દશાંશ *)dst->data, 0);
    }
    ggml_barrier(params->threadpool);

    // parallelize by last three dimensions

    // total rows in dst
    અચલ int64_t nr = ne1*ne2*ne3;

    // rows per thread
    અચલ int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ int64_t ir0 = dr*ith;
    અચલ int64_t ir1 = MIN(ir0 + dr, nr);

    // dst[:,:,:,:] = 0
    // for i2,i3:
    //   for i1:
    //     for i01:
    //       for i0:
    //         dst[i0,i1,i2,i3] += src0[i0,i01,i2,i3] * src1[i1,i01,i2,i3]

    દશાંશ * wdata = (દશાંશ *) params->wdata + (ne0 + CACHE_લીટી_SIZE_F32) * ith;

    માટે (int64_t ir = ir0; ir < ir1; ++ir) {
        // dst indices
        અચલ int64_t i3 = ir/(ne2*ne1);
        અચલ int64_t i2 = (ir - i3*ne2*ne1)/ne1;
        અચલ int64_t i1 = (ir - i3*ne2*ne1 - i2*ne1);

        અચલ int64_t i02 = i2;
        અચલ int64_t i03 = i3;

        //const int64_t i10 = i1;
        અચલ int64_t i12 = i2;
        અચલ int64_t i13 = i3;

        માટે (int64_t i01 = 0; i01 < ne01; ++i01) {
            અચલ int64_t i11 = i01;

            દશાંશ * s0 = (દશાંશ *) ((અક્ષર *) src0->data + (          i01*nb01 + i02*nb02 + i03*nb03));
            દશાંશ * s1 = (દશાંશ *) ((અક્ષર *) src1->data + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
            દશાંશ * d  = (દશાંશ *) ((અક્ષર *)  dst->data + (          i1*nb1 + i2*nb2 + i3*nb3));

            dequantize_row_q(s0, wdata, ne0);
            ggml_vec_mad_f32(ne0, d, wdata, *s1);
        }
    }
}

સ્થિર શૂન્ય ggml_compute_forward_out_prod_f16_f32(
        અચલ ggml_compute_params * params,
              ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_દ્વિઅંકી_OP_LOCALS;

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ne02 == ne12);
    GGML_ASSERT(ne03 == ne13);
    GGML_ASSERT(ne2  == ne12);
    GGML_ASSERT(ne3  == ne13);

    GGML_ASSERT(nb00 == કદ(ggml_fp16_t));
    GGML_ASSERT(nb0  == કદ(દશાંશ));

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    જો (ith == 0) {
        ggml_vec_set_f32(ne0*ne1*ne2*ne3, (દશાંશ *)dst->data, 0);
    }
    ggml_barrier(params->threadpool);

    અચલ int64_t nr = ne1*ne2*ne3;
    અચલ int64_t dr = (nr + nth - 1)/nth;
    અચલ int64_t ir0 = dr*ith;
    અચલ int64_t ir1 = MIN(ir0 + dr, nr);

    દશાંશ * wdata = (દશાંશ *) params->wdata + (ne0 + CACHE_લીટી_SIZE_F32) * ith;

    માટે (int64_t ir = ir0; ir < ir1; ++ir) {
        અચલ int64_t i3 = ir/(ne2*ne1);
        અચલ int64_t i2 = (ir - i3*ne2*ne1)/ne1;
        અચલ int64_t i1 = (ir - i3*ne2*ne1 - i2*ne1);

        અચલ int64_t i02 = i2;
        અચલ int64_t i03 = i3;

        અચલ int64_t i12 = i2;
        અચલ int64_t i13 = i3;

        દશાંશ * d = (દશાંશ *) ((અક્ષર *) dst->data + (i1*nb1 + i2*nb2 + i3*nb3));

        માટે (int64_t i01 = 0; i01 < ne01; ++i01) {
            અચલ int64_t i11 = i01;
            ggml_fp16_t * s0 = (ggml_fp16_t *) ((અક્ષર *) src0->data + (i01*nb01 + i02*nb02 + i03*nb03));
            દશાંશ * s1 = (દશાંશ *) ((અક્ષર *) src1->data + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
            ggml_fp16_to_fp32_row(s0, wdata, ne0);
            ggml_vec_mad_f32(ne0, d, wdata, *s1);
        }
    }
}

શૂન્ય ggml_compute_forward_out_prod(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_Q1_0:
        સ્થિતિ GGML_TYPE_Q2_0:
        સ્થિતિ GGML_TYPE_Q4_0:
        સ્થિતિ GGML_TYPE_Q4_1:
        સ્થિતિ GGML_TYPE_Q5_0:
        સ્થિતિ GGML_TYPE_Q5_1:
        સ્થિતિ GGML_TYPE_Q8_0:
        સ્થિતિ GGML_TYPE_MXFP4:
        સ્થિતિ GGML_TYPE_NVFP4:
        સ્થિતિ GGML_TYPE_Q2_K:
        સ્થિતિ GGML_TYPE_Q3_K:
        સ્થિતિ GGML_TYPE_Q4_K:
        સ્થિતિ GGML_TYPE_Q5_K:
        સ્થિતિ GGML_TYPE_Q6_K:
        સ્થિતિ GGML_TYPE_TQ1_0:
        સ્થિતિ GGML_TYPE_TQ2_0:
        સ્થિતિ GGML_TYPE_IQ2_XXS:
        સ્થિતિ GGML_TYPE_IQ2_XS:
        સ્થિતિ GGML_TYPE_IQ3_XXS:
        સ્થિતિ GGML_TYPE_IQ1_S:
        સ્થિતિ GGML_TYPE_IQ1_M:
        સ્થિતિ GGML_TYPE_IQ4_NL:
        સ્થિતિ GGML_TYPE_IQ4_XS:
        સ્થિતિ GGML_TYPE_IQ3_S:
        સ્થિતિ GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_out_prod_q_f32(params, dst);
            } તોડો;
        સ્થિતિ GGML_TYPE_F16:
            {
                ggml_compute_forward_out_prod_f16_f32(params, dst);
            } તોડો;
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_out_prod_f32(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_scale

સ્થિર શૂન્ય ggml_compute_forward_scale_f32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    દશાંશ s; // scale factor
    દશાંશ b; // bias

    memcpy(&s, (દશાંશ *) dst->op_params + 0, કદ(દશાંશ));
    memcpy(&b, (દશાંશ *) dst->op_params + 1, કદ(દશાંશ));

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક nc = src0->ne[0];
    અચલ પૂર્ણાંક nr = ggml_nrows(src0);

    // rows per thread
    અચલ પૂર્ણાંક dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr*ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    અચલ size_t nb01 = src0->nb[1];

    અચલ size_t nb1 = dst->nb[1];

    જો (b == 0.0f) {
        માટે (પૂર્ણાંક i1 = ir0; i1 < ir1; i1++) {
            જો (dst->data != src0->data) {
                // src0 is same shape as dst => same indices
                // TODO: add x parameter to ggml_vec_scale_f32 and remove this memcpy
                memcpy((અક્ષર *)dst->data + i1*nb1, (અક્ષર *)src0->data + i1*nb01, nc * કદ(દશાંશ));
            }
            ggml_vec_scale_f32(nc, (દશાંશ *) ((અક્ષર *) dst->data + i1*nb1), s);
        }
    } નહીંતર {
        માટે (પૂર્ણાંક i1 = ir0; i1 < ir1; i1++) {
            ggml_vec_mad1_f32(nc,
                (દશાંશ *) ((અક્ષર *) dst->data  + i1*nb1),
                (દશાંશ *) ((અક્ષર *) src0->data + i1*nb1),
                s, b);
        }
    }
}

સ્થિર શૂન્ય ggml_compute_forward_scale_bf16(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    દશાંશ s; // scale factor
    દશાંશ b; // bias

    memcpy(&s, (દશાંશ *) dst->op_params + 0, કદ(દશાંશ));
    memcpy(&b, (દશાંશ *) dst->op_params + 1, કદ(દશાંશ));

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક nc = src0->ne[0];
    અચલ પૂર્ણાંક nr = ggml_nrows(src0);

    // rows per thread
    અચલ પૂર્ણાંક dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr*ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    માટે (પૂર્ણાંક i1 = ir0; i1 < ir1; i1++) {
        અચલ ggml_bf16_t * x = (અચલ ggml_bf16_t *) ((અચલ અક્ષર *) src0->data + i1*src0->nb[1]);
        ggml_bf16_t       * y = (ggml_bf16_t *)       ((અક્ષર *)       dst->data  + i1*dst->nb[1]);
        માટે (પૂર્ણાંક k = 0; k < nc; k++) {
            y[k] = GGML_FP32_TO_BF16(s*GGML_BF16_TO_FP32(x[k]) + b);
        }
    }
}

શૂન્ય ggml_compute_forward_scale(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_scale_f32(params, dst);
            } તોડો;
        સ્થિતિ GGML_TYPE_BF16:
            {
                ggml_compute_forward_scale_bf16(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_set

સ્થિર શૂન્ય ggml_compute_forward_set_f32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));

    // view src0 and dst with these strides and data offset inbytes during set
    // nb0 is implicitly element_size because src0 and dst are contiguous
    size_t nb1     = ((int32_t *) dst->op_params)[0];
    size_t nb2     = ((int32_t *) dst->op_params)[1];
    size_t nb3     = ((int32_t *) dst->op_params)[2];
    size_t offset  = ((int32_t *) dst->op_params)[3];
    તાર્કિક   inplace = (તાર્કિક) ((int32_t *) dst->op_params)[4];

    જો (!inplace) {
        જો (params->ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            memcpy(
                ((અક્ષર *)  dst->data),
                ((અક્ષર *) src0->data),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક nr = ggml_nrows(src1);
    અચલ પૂર્ણાંક nc = src1->ne[0];

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne)
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb)

    // src0 and dst as viewed during set
    અચલ size_t nb0 = ggml_element_size(src0);

    અચલ પૂર્ણાંક im0 = (ne10 == 0 ? 0 : ne10-1);
    અચલ પૂર્ણાંક im1 = (ne11 == 0 ? 0 : ne11-1);
    અચલ પૂર્ણાંક im2 = (ne12 == 0 ? 0 : ne12-1);
    અચલ પૂર્ણાંક im3 = (ne13 == 0 ? 0 : ne13-1);

    GGML_ASSERT(offset + im0*nb0  + im1*nb1  + im2*nb2  + im3*nb3  <= ggml_nbytes(dst));

    GGML_ASSERT(nb10 == કદ(દશાંશ));

    // rows per thread
    અચલ પૂર્ણાંક dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr*ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    માટે (પૂર્ણાંક ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are viewed with shape of src1 and offset
        // => same indices
        અચલ પૂર્ણાંક i3 = ir/(ne12*ne11);
        અચલ પૂર્ણાંક i2 = (ir - i3*ne12*ne11)/ne11;
        અચલ પૂર્ણાંક i1 = (ir - i3*ne12*ne11 - i2*ne11);

        ggml_vec_cpy_f32(nc,
                (દશાંશ *) ((અક્ષર *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + offset),
                (દશાંશ *) ((અક્ષર *) src1->data + i3*nb13 + i2*nb12 + i1*nb11));
    }
}

સ્થિર શૂન્ય ggml_compute_forward_set_i32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));

    // view src0 and dst with these strides and data offset inbytes during set
    // nb0 is implicitly element_size because src0 and dst are contiguous
    size_t nb1     = ((int32_t *) dst->op_params)[0];
    size_t nb2     = ((int32_t *) dst->op_params)[1];
    size_t nb3     = ((int32_t *) dst->op_params)[2];
    size_t offset  = ((int32_t *) dst->op_params)[3];
    તાર્કિક   inplace = (તાર્કિક) ((int32_t *) dst->op_params)[4];

    જો (!inplace) {
        જો (params->ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            memcpy(
                ((અક્ષર *)  dst->data),
                ((અક્ષર *) src0->data),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક nr = ggml_nrows(src1);
    અચલ પૂર્ણાંક nc = src1->ne[0];

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne)
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb)

    // src0 and dst as viewed during set
    અચલ size_t nb0 = ggml_element_size(src0);

    અચલ પૂર્ણાંક im0 = (ne10 == 0 ? 0 : ne10-1);
    અચલ પૂર્ણાંક im1 = (ne11 == 0 ? 0 : ne11-1);
    અચલ પૂર્ણાંક im2 = (ne12 == 0 ? 0 : ne12-1);
    અચલ પૂર્ણાંક im3 = (ne13 == 0 ? 0 : ne13-1);

    GGML_ASSERT(offset + im0*nb0  + im1*nb1  + im2*nb2  + im3*nb3  <= ggml_nbytes(dst));

    GGML_ASSERT(nb10 == કદ(int32_t));

    // rows per thread
    અચલ પૂર્ણાંક dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr*ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    માટે (પૂર્ણાંક ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are viewed with shape of src1 and offset
        // => same indices
        અચલ પૂર્ણાંક i3 = ir/(ne12*ne11);
        અચલ પૂર્ણાંક i2 = (ir - i3*ne12*ne11)/ne11;
        અચલ પૂર્ણાંક i1 = (ir - i3*ne12*ne11 - i2*ne11);

        ggml_vec_cpy_i32(nc,
                (int32_t *) ((અક્ષર *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + offset),
                (int32_t *) ((અક્ષર *) src1->data + i3*nb13 + i2*nb12 + i1*nb11));
    }
}

શૂન્ય ggml_compute_forward_set(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_set_f32(params, dst);
            } તોડો;
        સ્થિતિ GGML_TYPE_I32:
            {
                ggml_compute_forward_set_i32(params, dst);
            } તોડો;
        સ્થિતિ GGML_TYPE_F16:
        સ્થિતિ GGML_TYPE_BF16:
        સ્થિતિ GGML_TYPE_Q1_0:
        સ્થિતિ GGML_TYPE_Q2_0:
        સ્થિતિ GGML_TYPE_Q4_0:
        સ્થિતિ GGML_TYPE_Q4_1:
        સ્થિતિ GGML_TYPE_Q5_0:
        સ્થિતિ GGML_TYPE_Q5_1:
        સ્થિતિ GGML_TYPE_Q8_0:
        સ્થિતિ GGML_TYPE_Q8_1:
        સ્થિતિ GGML_TYPE_MXFP4:
        સ્થિતિ GGML_TYPE_NVFP4:
        સ્થિતિ GGML_TYPE_Q2_K:
        સ્થિતિ GGML_TYPE_Q3_K:
        સ્થિતિ GGML_TYPE_Q4_K:
        સ્થિતિ GGML_TYPE_Q5_K:
        સ્થિતિ GGML_TYPE_Q6_K:
        સ્થિતિ GGML_TYPE_TQ1_0:
        સ્થિતિ GGML_TYPE_TQ2_0:
        સ્થિતિ GGML_TYPE_IQ2_XXS:
        સ્થિતિ GGML_TYPE_IQ2_XS:
        સ્થિતિ GGML_TYPE_IQ3_XXS:
        સ્થિતિ GGML_TYPE_IQ1_S:
        સ્થિતિ GGML_TYPE_IQ1_M:
        સ્થિતિ GGML_TYPE_IQ4_NL:
        સ્થિતિ GGML_TYPE_IQ4_XS:
        સ્થિતિ GGML_TYPE_IQ3_S:
        સ્થિતિ GGML_TYPE_IQ2_S:
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_cpy

શૂન્ય ggml_compute_forward_cpy(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_dup(params, dst);
}

// ggml_compute_forward_cont

શૂન્ય ggml_compute_forward_cont(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_dup(params, dst);
}

// ggml_compute_forward_get_rows

સ્થિર શૂન્ય ggml_compute_forward_get_rows_q(
        અચલ ggml_compute_params * params,
              ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_દ્વિઅંકી_OP_LOCALS

    અચલ int64_t nc = ne00;
    અચલ int64_t nr = ggml_nelements(src1);

    અચલ ggml_type type = src0->type;
    ggml_to_float_t અચલ dequantize_row_q = ggml_get_type_traits(type)->to_float;

    પ્રતિજ્ઞા(ne0  == nc);
    પ્રતિજ્ઞા(ne02 == ne11);
    પ્રતિજ્ઞા(nb00 == ggml_type_size(type));
    પ્રતિજ્ઞા(ggml_nrows(dst) == nr);

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    // rows per thread
    અચલ પૂર્ણાંક dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr*ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    માટે (int64_t i = ir0; i < ir1; ++i) {
        અચલ int64_t i12 = i/(ne11*ne10);
        અચલ int64_t i11 = (i - i12*ne11*ne10)/ne10;
        અચલ int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        અચલ int64_t i01 = *(int32_t *) ((અક્ષર *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        dequantize_row_q(
                (અચલ શૂન્ય *) ((અક્ષર *) src0->data + i01*nb01 + i11*nb02 + i12*nb03),
                     (દશાંશ *) ((અક્ષર *)  dst->data + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

સ્થિર શૂન્ય ggml_compute_forward_get_rows_f16(
        અચલ ggml_compute_params * params,
              ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_દ્વિઅંકી_OP_LOCALS

    અચલ int64_t nc = ne00;
    અચલ int64_t nr = ggml_nelements(src1);

    પ્રતિજ્ઞા(ne0  == nc);
    પ્રતિજ્ઞા(ne02 == ne11);
    પ્રતિજ્ઞા(nb00 == કદ(ggml_fp16_t));
    પ્રતિજ્ઞા(ggml_nrows(dst) == nr);

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    // rows per thread
    અચલ પૂર્ણાંક dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr*ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    માટે (int64_t i = ir0; i < ir1; ++i) {
        અચલ int64_t i12 = i/(ne11*ne10);
        અચલ int64_t i11 = (i - i12*ne11*ne10)/ne10;
        અચલ int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        અચલ int64_t i01 = *(int32_t *) ((અક્ષર *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        ggml_cpu_fp16_to_fp32(
            (અચલ ggml_fp16_t*) ((અક્ષર *) src0->data + i01*nb01 + i11*nb02 + i12*nb03),
                       (દશાંશ *) ((અક્ષર *)  dst->data + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

સ્થિર શૂન્ય ggml_compute_forward_get_rows_bf16(
        અચલ ggml_compute_params * params,
              ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_દ્વિઅંકી_OP_LOCALS

    અચલ int64_t nc = ne00;
    અચલ int64_t nr = ggml_nelements(src1);

    પ્રતિજ્ઞા(ne0  == nc);
    પ્રતિજ્ઞા(ne02 == ne11);
    પ્રતિજ્ઞા(nb00 == કદ(ggml_bf16_t));
    પ્રતિજ્ઞા(ggml_nrows(dst) == nr);

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    // rows per thread
    અચલ પૂર્ણાંક dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr*ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    માટે (int64_t i = ir0; i < ir1; ++i) {
        અચલ int64_t i12 = i/(ne11*ne10);
        અચલ int64_t i11 = (i - i12*ne11*ne10)/ne10;
        અચલ int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        અચલ int64_t i01 = *(int32_t *) ((અક્ષર *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        ggml_cpu_bf16_to_fp32(
            (અચલ ggml_bf16_t *) ((અક્ષર *) src0->data + i01*nb01 + i11*nb02 + i12*nb03),
                        (દશાંશ *) ((અક્ષર *)  dst->data + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

સ્થિર શૂન્ય ggml_compute_forward_get_rows_f32(
        અચલ ggml_compute_params * params,
              ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_દ્વિઅંકી_OP_LOCALS

    અચલ int64_t nc = ne00;
    અચલ int64_t nr = ggml_nelements(src1);

    પ્રતિજ્ઞા(ne0  == nc);
    પ્રતિજ્ઞા(ne02 == ne11);
    પ્રતિજ્ઞા(nb00 == કદ(દશાંશ));
    પ્રતિજ્ઞા(ggml_nrows(dst) == nr);

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    // rows per thread
    અચલ પૂર્ણાંક dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr*ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    માટે (int64_t i = ir0; i < ir1; ++i) {
        અચલ int64_t i12 = i/(ne11*ne10);
        અચલ int64_t i11 = (i - i12*ne11*ne10)/ne10;
        અચલ int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        અચલ int64_t i01 = *(int32_t *) ((અક્ષર *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        ggml_vec_cpy_f32(nc,
                (દશાંશ *) ((અક્ષર *)  dst->data + i10*nb1  + i11*nb2  + i12*nb3),
                (દશાંશ *) ((અક્ષર *) src0->data + i01*nb01 + i11*nb02 + i12*nb03));
    }
}

શૂન્ય ggml_compute_forward_get_rows(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_Q1_0:
        સ્થિતિ GGML_TYPE_Q2_0:
        સ્થિતિ GGML_TYPE_Q4_0:
        સ્થિતિ GGML_TYPE_Q4_1:
        સ્થિતિ GGML_TYPE_Q5_0:
        સ્થિતિ GGML_TYPE_Q5_1:
        સ્થિતિ GGML_TYPE_Q8_0:
        સ્થિતિ GGML_TYPE_Q8_1:
        સ્થિતિ GGML_TYPE_MXFP4:
        સ્થિતિ GGML_TYPE_NVFP4:
        સ્થિતિ GGML_TYPE_Q2_K:
        સ્થિતિ GGML_TYPE_Q3_K:
        સ્થિતિ GGML_TYPE_Q4_K:
        સ્થિતિ GGML_TYPE_Q5_K:
        સ્થિતિ GGML_TYPE_Q6_K:
        સ્થિતિ GGML_TYPE_TQ1_0:
        સ્થિતિ GGML_TYPE_TQ2_0:
        સ્થિતિ GGML_TYPE_IQ2_XXS:
        સ્થિતિ GGML_TYPE_IQ2_XS:
        સ્થિતિ GGML_TYPE_IQ3_XXS:
        સ્થિતિ GGML_TYPE_IQ1_S:
        સ્થિતિ GGML_TYPE_IQ1_M:
        સ્થિતિ GGML_TYPE_IQ4_NL:
        સ્થિતિ GGML_TYPE_IQ4_XS:
        સ્થિતિ GGML_TYPE_IQ3_S:
        સ્થિતિ GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_get_rows_q(params, dst);
            } તોડો;
        સ્થિતિ GGML_TYPE_F16:
            {
                ggml_compute_forward_get_rows_f16(params, dst);
            } તોડો;
        સ્થિતિ GGML_TYPE_BF16:
            {
                ggml_compute_forward_get_rows_bf16(params, dst);
            } તોડો;
        સ્થિતિ GGML_TYPE_F32:
        સ્થિતિ GGML_TYPE_I32:
            {
                ggml_compute_forward_get_rows_f32(params, dst);
            } તોડો;
        મૂળભૂત:
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
સ્થિર શૂન્ય ggml_compute_forward_set_rows_impl(
        અચલ ggml_compute_params * params,
              ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_દ્વિઅંકી_OP_LOCALS

    અચલ int64_t nc = ne00;
    અચલ int64_t nr = ne01;

    પ્રતિજ્ઞા(ne0  == nc);
    પ્રતિજ્ઞા(ne2  == ne02);
    પ્રતિજ્ઞા(ne3  == ne03);
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    પ્રતિજ્ઞા(ne02 % ne11 == 0);
    પ્રતિજ્ઞા(ne03 % ne12 == 0);

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    // rows per thread
    અચલ int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ int64_t ir0 = dr*ith;
    અચલ int64_t ir1 = std::min(ir0 + dr, nr);

    અચલ size_t rs = ggml_row_size(src0->type, nc);

    ggml_from_float_t અચલ from_float = ggml_get_type_traits_cpu(dst->type)->from_float;

    માટે (int64_t i03 = 0; i03 < ne03; ++i03) {
        માટે (int64_t i02 = 0; i02 < ne02; ++i02) {
            માટે (int64_t i = ir0; i < ir1; ++i) {
                અચલ int64_t i12 = i03%ne12;
                અચલ int64_t i11 = i02%ne11;
                અચલ int64_t i10 = i;

                અચલ int64_t i1 = *(idx_t *) ((અક્ષર *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);

                GGML_ASSERT(i1 >= 0 && i1 < ne1);

                જો constexpr (std::is_same_v<src_t, દશાંશ>) {
                    from_float(
                            (અચલ દશાંશ *) ((અક્ષર *) src0->data +  i*nb01 + i02*nb02 + i03*nb03),
                                            ((અક્ષર *)  dst->data + i1*nb1  + i02*nb2  + i03*nb3), nc);
                } નહીંતર જો constexpr (std::is_same_v<src_t, ggml_fp16_t>) {
                    જો (dst->type == GGML_TYPE_F16) {
                        memcpy(
                                            ((અક્ષર *)  dst->data + i1*nb1  + i02*nb2  + i03*nb3),
                                            ((અક્ષર *) src0->data +  i*nb01 + i02*nb02 + i03*nb03),
                                            rs);
                    } નહીંતર {
                        દશાંશ * wdata = (દશાંશ *) params->wdata + (nc + CACHE_લીટી_SIZE_F32) * ith;
                        ggml_fp16_to_fp32_row(
                                (અચલ ggml_fp16_t *) ((અક્ષર *) src0->data + i*nb01 + i02*nb02 + i03*nb03),
                                wdata, nc);
                        from_float(wdata,
                                ((અક્ષર *) dst->data + i1*nb1 + i02*nb2 + i03*nb3), nc);
                    }
                } નહીંતર {
                    GGML_ABORT("src0->type = %d (%s) not supported", src0->type, ggml_type_name(src0->type));
                }
            }
        }
    }
}

શૂન્ય ggml_compute_forward_set_rows(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                જો (src1->type == GGML_TYPE_I64) {
                    ggml_compute_forward_set_rows_impl<દશાંશ, int64_t>(params, dst);
                } નહીંતર જો (src1->type == GGML_TYPE_I32) {
                    ggml_compute_forward_set_rows_impl<દશાંશ, int32_t>(params, dst);
                } નહીંતર {
                    GGML_ABORT("src1->type = %d (%s) not supported", src1->type, ggml_type_name(src1->type));
                }
            } તોડો;
        સ્થિતિ GGML_TYPE_F16:
            {
                જો (src1->type == GGML_TYPE_I64) {
                    ggml_compute_forward_set_rows_impl<ggml_fp16_t, int64_t>(params, dst);
                } નહીંતર જો (src1->type == GGML_TYPE_I32) {
                    ggml_compute_forward_set_rows_impl<ggml_fp16_t, int32_t>(params, dst);
                } નહીંતર {
                    GGML_ABORT("src1->type = %d (%s) not supported", src1->type, ggml_type_name(src1->type));
                }
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("src0->type = %d (%s) not supported", src0->type, ggml_type_name(src0->type));
            }
    }
}

// ggml_compute_forward_get_rows_back

સ્થિર શૂન્ય ggml_compute_forward_get_rows_back_f32_f16(
        અચલ ggml_compute_params * params,
              ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];

    જો (params->ith != 0) {
        પાછા;
    }

    GGML_ASSERT(ggml_is_contiguous(dst));

    // ggml_compute_forward_dup_same_cont(params, opt0, dst);

    memset(dst->data, 0, ggml_nbytes(dst));

    અચલ પૂર્ણાંક nc = src0->ne[0];
    અચલ પૂર્ણાંક nr = ggml_nelements(src1);

    GGML_ASSERT( dst->ne[0] == nc);
    GGML_ASSERT(src0->nb[0] == કદ(ggml_fp16_t));

    માટે (પૂર્ણાંક i = 0; i < nr; ++i) {
        અચલ પૂર્ણાંક r = ((int32_t *) src1->data)[i];

        GGML_ASSERT(r >= 0 && r < dst->ne[1]);

        માટે (પૂર્ણાંક j = 0; j < nc; ++j) {
            ggml_fp16_t v = ((ggml_fp16_t *) ((અક્ષર *) src0->data + i*src0->nb[1]))[j];
            ((દશાંશ *) ((અક્ષર *) dst->data + r*dst->nb[1]))[j] += GGML_CPU_FP16_TO_FP32(v);
        }
    }
}

સ્થિર શૂન્ય ggml_compute_forward_get_rows_back_f32(
        અચલ ggml_compute_params * params,
              ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];

    જો (params->ith != 0) {
        પાછા;
    }

    GGML_ASSERT(ggml_is_contiguous(dst));

    // ggml_compute_forward_dup_same_cont(params, opt0, dst);

    memset(dst->data, 0, ggml_nbytes(dst));

    અચલ પૂર્ણાંક nc = src0->ne[0];
    અચલ પૂર્ણાંક nr = ggml_nelements(src1);

    GGML_ASSERT( dst->ne[0] == nc);
    GGML_ASSERT(src0->nb[0] == કદ(દશાંશ));

    માટે (પૂર્ણાંક i = 0; i < nr; ++i) {
        અચલ પૂર્ણાંક r = ((int32_t *) src1->data)[i];

        GGML_ASSERT(r >= 0 && r < dst->ne[1]);

        ggml_vec_add_f32(nc,
                (દશાંશ *) ((અક્ષર *)  dst->data + r*dst->nb[1]),
                (દશાંશ *) ((અક્ષર *)  dst->data + r*dst->nb[1]),
                (દશાંશ *) ((અક્ષર *) src0->data + i*src0->nb[1]));
    }
}

શૂન્ય ggml_compute_forward_get_rows_back(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F16:
            {
                ggml_compute_forward_get_rows_back_f32_f16(params, dst);
            } તોડો;
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_get_rows_back_f32(params, dst);
            } તોડો;
        મૂળભૂત:
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

સ્થિર શૂન્ય ggml_compute_forward_diag_f32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    જો (params->ith != 0) {
        પાછા;
    }

    // TODO: handle transposed/permuted matrices

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(ne00 == ne0);
    GGML_ASSERT(ne00 == ne1);
    GGML_ASSERT(ne01 == 1);
    GGML_ASSERT(ne02 == ne2);
    GGML_ASSERT(ne03 == ne3);

    GGML_ASSERT(nb00 == કદ(દશાંશ));
    GGML_ASSERT(nb0  == કદ(દશાંશ));

    માટે (પૂર્ણાંક i3 = 0; i3 < ne3; i3++) {
        માટે (પૂર્ણાંક i2 = 0; i2 < ne2; i2++) {
            માટે (પૂર્ણાંક i1 = 0; i1 < ne1; i1++) {
                દશાંશ * d = (દશાંશ *)((અક્ષર *)  dst->data + i3*nb3  + i2*nb2 + i1*nb1);
                દશાંશ * s = (દશાંશ *)((અક્ષર *) src0->data + i3*nb03 + i2*nb02);
                માટે (પૂર્ણાંક i0 = 0; i0 < i1; i0++) {
                    d[i0] = 0;
                }
                d[i1] = s[i1];
                માટે (પૂર્ણાંક i0 = i1+1; i0 < ne0; i0++) {
                    d[i0] = 0;
                }
            }
        }
    }
}

શૂન્ય ggml_compute_forward_diag(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_diag_f32(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_diag_mask_inf

સ્થિર શૂન્ય ggml_compute_forward_diag_mask_f32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst,
        અચલ દશાંશ value) {

    અચલ ggml_tensor * src0 = dst->src[0];

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક  n_past  = ((int32_t *) dst->op_params)[0];
    અચલ તાર્કિક inplace = src0->data == dst->data;

    GGML_ASSERT(n_past >= 0);

    જો (!inplace) {
        જો (ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
            GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));
            memcpy(
                ((અક્ષર *)  dst->data),
                ((અક્ષર *) src0->data),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    // TODO: handle transposed/permuted matrices

    અચલ પૂર્ણાંક n  = ggml_nrows(src0);
    અચલ પૂર્ણાંક nc = src0->ne[0];
    અચલ પૂર્ણાંક nr = src0->ne[1];
    અચલ પૂર્ણાંક nz = n/nr;

    GGML_ASSERT( dst->nb[0] == કદ(દશાંશ));
    GGML_ASSERT(src0->nb[0] == કદ(દશાંશ));

    માટે (પૂર્ણાંક k = 0; k < nz; k++) {
        માટે (પૂર્ણાંક j = ith; j < nr; j += nth) {
            માટે (પૂર્ણાંક i = n_past; i < nc; i++) {
                જો (i > n_past + j) {
                    *(દશાંશ *)((અક્ષર *) dst->data + k*dst->nb[2] + j*dst->nb[1] + i*dst->nb[0]) = value;
                }
            }
        }
    }
}

શૂન્ય ggml_compute_forward_diag_mask_inf(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_diag_mask_f32(params, dst, -INFINITY);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

શૂન્ય ggml_compute_forward_diag_mask_zero(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_diag_mask_f32(params, dst, 0);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_soft_max

સ્થિર શૂન્ય ggml_compute_forward_soft_max_f32(
        અચલ ggml_compute_params * params,
              ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];
    અચલ ggml_tensor * src2 = dst->src[2];

    પ્રતિજ્ઞા(ggml_is_contiguous(dst));
    પ્રતિજ્ઞા(ggml_are_same_shape(src0, dst));

    દશાંશ scale    = 1.0f;
    દશાંશ max_bias = 0.0f;

    memcpy(&scale,    (દશાંશ *) dst->op_params + 0, કદ(દશાંશ));
    memcpy(&max_bias, (દશાંશ *) dst->op_params + 1, કદ(દશાંશ));

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    અચલ int64_t nb11 = src1 ? src1->nb[1] : 1;
    અચલ int64_t nb12 = src1 ? src1->nb[2] : 1;
    અચલ int64_t nb13 = src1 ? src1->nb[3] : 1;

    અચલ int64_t ne12 = src1 ? src1->ne[2] : 1;
    અચલ int64_t ne13 = src1 ? src1->ne[3] : 1;

    // TODO: is this supposed to be ceil instead of floor?
    //       https://huggingface.co/mosaicml/mpt-7b/blob/main/attention.py#L370
    અચલ uint32_t n_head      = ne02;
    અચલ uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));

    અચલ દશાંશ m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    અચલ દશાંશ m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    દશાંશ * wp = (દશાંશ *) params->wdata + (ne00 + CACHE_લીટી_SIZE_F32) * ith;

    અચલ તાર્કિક use_f16 = (src1 && src1->type == GGML_TYPE_F16);

    // sinks
    અચલ દશાંશ * sk = src2 ? (દશાંશ *)((અક્ષર *) src2->data) : nullptr;

    માટે (int64_t i03 = 0; i03 < ne03; i03++) {
        માટે (int64_t i02 = 0; i02 < ne02; i02++) {
            માટે (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                અચલ int64_t i11 = i01;
                અચલ int64_t i12 = i02%ne12;
                અચલ int64_t i13 = i03%ne13;

                // ALiBi
                અચલ uint32_t h = i02; // head
                અચલ દશાંશ slope = (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1) : 1.0f;

                દશાંશ * sp = (દશાંશ *)((અક્ષર *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
                દશાંશ * dp = (દશાંશ *)((અક્ષર *)  dst->data + i01*nb1  + i02*nb2  + i03*nb3);

                // broadcast the mask across rows
                ggml_fp16_t * mp_f16 = src1 ? (ggml_fp16_t *)((અક્ષર *) src1->data + i11*nb11 + i12*nb12 + i13*nb13) : ખાલી;
                દશાંશ       * mp_f32 = src1 ? (દશાંશ       *)((અક્ષર *) src1->data + i11*nb11 + i12*nb12 + i13*nb13) : ખાલી;

                ggml_vec_cpy_f32  (ne00, wp, sp);
                ggml_vec_scale_f32(ne00, wp, scale);
                જો (mp_f32) {
                    જો (use_f16) {
                        માટે (પૂર્ણાંક i = 0; i < ne00; ++i) {
                            wp[i] += slope*GGML_CPU_FP16_TO_FP32(mp_f16[i]);
                        }
                    } નહીંતર {
                        માટે (પૂર્ણાંક i = 0; i < ne00; ++i) {
                            wp[i] += slope*mp_f32[i];
                        }
                    }
                }

#ifndef NDEBUG
                માટે (પૂર્ણાંક i = 0; i < ne00; ++i) {
                    //printf("p[%d] = %f\n", i, p[i]);
                    પ્રતિજ્ઞા(!isnan(wp[i]));
                }
#endif // NDEBUG

                દશાંશ max = -INFINITY;
                ggml_vec_max_f32(ne00, &max, wp);

                // if we have sinks, make a correction as if they were included in the softmax
                જો (sk) {
                    max = MAX(max, sk[i02]);
                }

                ggml_float sum = ggml_vec_soft_max_f32(ne00, dp, wp, max);
                પ્રતિજ્ઞા(sum > 0.0);

                જો (sk) {
                    sum += (ggml_float) expf(sk[i02] - max);
                }

                sum = 1.0/sum;
                ggml_vec_scale_f32(ne00, dp, sum);

#ifndef NDEBUG
                માટે (પૂર્ણાંક i = 0; i < ne00; ++i) {
                    પ્રતિજ્ઞા(!isnan(dp[i]));
                    પ્રતિજ્ઞા(!isinf(dp[i]));
                }
#endif // NDEBUG
            }
        }
    }
}

શૂન્ય ggml_compute_forward_soft_max(
        અચલ ggml_compute_params * params,
              ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_soft_max_f32(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}


// ggml_compute_forward_soft_max_ext_back

સ્થિર શૂન્ય ggml_compute_forward_soft_max_ext_back_f32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_are_same_shape(src1, dst));

    દશાંશ scale    = 1.0f;
    દશાંશ max_bias = 0.0f;

    memcpy(&scale,    (અચલ દશાંશ *) dst->op_params + 0, કદ(દશાંશ));
    memcpy(&max_bias, (અચલ દશાંશ *) dst->op_params + 1, કદ(દશાંશ));

    GGML_ASSERT(max_bias == 0.0f);

    // TODO: handle transposed/permuted matrices

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક nc = src0->ne[0];
    અચલ પૂર્ણાંક nr = ggml_nrows(src0);

    // rows per thread
    અચલ પૂર્ણાંક dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr*ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    માટે (પૂર્ણાંક i1 = ir0; i1 < ir1; i1++) {
        દશાંશ *dy = (દશાંશ *)((અક્ષર *) src0->data + i1*src0->nb[1]);
        દશાંશ *y  = (દશાંશ *)((અક્ષર *) src1->data + i1*src1->nb[1]);
        દશાંશ *dx = (દશાંશ *)((અક્ષર *) dst->data  + i1*dst->nb[1]);

#ifndef NDEBUG
        માટે (પૂર્ણાંક i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            પ્રતિજ્ઞા(!isnan(dy[i]));
            પ્રતિજ્ઞા(!isnan(y[i]));
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
        દશાંશ dot_y_dy = 0;
        ggml_vec_dot_f32(nc, &dot_y_dy, 0, y, 0, dy, 0, 1);
        માટે (પૂર્ણાંક i = 0; i < nc; i++) {
            dx[i] = scale * (dy[i] - dot_y_dy) * y[i];
        }

#ifndef NDEBUG
        માટે (પૂર્ણાંક i = 0; i < nc; ++i) {
            પ્રતિજ્ઞા(!isnan(dx[i]));
            પ્રતિજ્ઞા(!isinf(dx[i]));
        }
#endif // NDEBUG
    }
}

શૂન્ય ggml_compute_forward_soft_max_ext_back(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_soft_max_ext_back_f32(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_clamp

સ્થિર શૂન્ય ggml_compute_forward_clamp_f32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    દશાંશ min;
    દશાંશ max;
    memcpy(&min, (દશાંશ *) dst->op_params + 0, કદ(દશાંશ));
    memcpy(&max, (દશાંશ *) dst->op_params + 1, કદ(દશાંશ));

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક n  = ggml_nrows(src0);
    અચલ પૂર્ણાંક nc = src0->ne[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == કદ(દશાંશ));
    GGML_ASSERT(nb00 == કદ(દશાંશ));

    માટે (પૂર્ણાંક j = ith; j < n; j += nth) {
        અચલ int64_t i1 = j % ne01;
        અચલ int64_t i2 = (j / ne01) % ne02;
        અચલ int64_t i3 = j / (ne01*ne02);

        દશાંશ * dst_ptr  = (દશાંશ *) ((અક્ષર *)  dst->data + i1*nb1  + i2*nb2  + i3*nb3);
        દશાંશ * src0_ptr = (દશાંશ *) ((અક્ષર *) src0->data + i1*nb01 + i2*nb02 + i3*nb03);

        માટે (પૂર્ણાંક i = 0; i < nc; i++) {
            dst_ptr[i] = MAX(MIN(src0_ptr[i], max), min);
        }
    }
}

સ્થિર શૂન્ય ggml_compute_forward_clamp_f16(
    અચલ ggml_compute_params * params,
    ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    દશાંશ min;
    દશાંશ max;
    memcpy(&min, (દશાંશ *) dst->op_params + 0, કદ(દશાંશ));
    memcpy(&max, (દશાંશ *) dst->op_params + 1, કદ(દશાંશ));

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક n  = ggml_nrows(src0);
    અચલ પૂર્ણાંક nc = src0->ne[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == કદ(ggml_fp16_t));
    GGML_ASSERT(nb00 == કદ(ggml_fp16_t));

    માટે (પૂર્ણાંક j = ith; j < n; j += nth) {
        અચલ int64_t i1 = j % ne01;
        અચલ int64_t i2 = (j / ne01) % ne02;
        અચલ int64_t i3 = j / (ne01*ne02);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((અક્ષર *)  dst->data + i1*nb1  + i2*nb2  + i3*nb3);
        ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((અક્ષર *) src0->data + i1*nb01 + i2*nb02 + i3*nb03);

        માટે (પૂર્ણાંક i = 0; i < nc; i++) {
            દશાંશ v = GGML_CPU_FP16_TO_FP32(src0_ptr[i]);
            dst_ptr[i] = GGML_CPU_FP32_TO_FP16(MAX(MIN(v, max), min));
        }
    }
}

શૂન્ય ggml_compute_forward_clamp(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_clamp_f32(params, dst);
            } તોડો;
        સ્થિતિ GGML_TYPE_F16:
            {
                ggml_compute_forward_clamp_f16(params, dst);
            } તોડો;
        સ્થિતિ GGML_TYPE_BF16:
        સ્થિતિ GGML_TYPE_Q1_0:
        સ્થિતિ GGML_TYPE_Q2_0:
        સ્થિતિ GGML_TYPE_Q4_0:
        સ્થિતિ GGML_TYPE_Q4_1:
        સ્થિતિ GGML_TYPE_Q5_0:
        સ્થિતિ GGML_TYPE_Q5_1:
        સ્થિતિ GGML_TYPE_Q8_0:
        સ્થિતિ GGML_TYPE_Q8_1:
        સ્થિતિ GGML_TYPE_MXFP4:
        સ્થિતિ GGML_TYPE_NVFP4:
        સ્થિતિ GGML_TYPE_Q2_K:
        સ્થિતિ GGML_TYPE_Q3_K:
        સ્થિતિ GGML_TYPE_Q4_K:
        સ્થિતિ GGML_TYPE_Q5_K:
        સ્થિતિ GGML_TYPE_Q6_K:
        સ્થિતિ GGML_TYPE_TQ1_0:
        સ્થિતિ GGML_TYPE_TQ2_0:
        સ્થિતિ GGML_TYPE_IQ2_XXS:
        સ્થિતિ GGML_TYPE_IQ2_XS:
        સ્થિતિ GGML_TYPE_IQ3_XXS:
        સ્થિતિ GGML_TYPE_IQ1_S:
        સ્થિતિ GGML_TYPE_IQ1_M:
        સ્થિતિ GGML_TYPE_IQ4_NL:
        સ્થિતિ GGML_TYPE_IQ4_XS:
        સ્થિતિ GGML_TYPE_IQ3_S:
        સ્થિતિ GGML_TYPE_IQ2_S:
        સ્થિતિ GGML_TYPE_Q8_K:
        સ્થિતિ GGML_TYPE_I8:
        સ્થિતિ GGML_TYPE_I16:
        સ્થિતિ GGML_TYPE_I32:
        સ્થિતિ GGML_TYPE_I64:
        સ્થિતિ GGML_TYPE_F64:
        સ્થિતિ GGML_TYPE_ગણતરી:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_rope

સ્થિર દશાંશ rope_yarn_ramp(અચલ દશાંશ નીચલું, અચલ દશાંશ ઉપલું, અચલ પૂર્ણાંક i0) {
    અચલ દશાંશ y = (i0 / 2 - નીચલું) / MAX(0.001f, ઉપલું - નીચલું);
    પાછા 1 - MIN(1, MAX(0, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
સ્થિર શૂન્ય rope_yarn(
    દશાંશ theta_extrap, દશાંશ freq_scale, દશાંશ corr_dims[2], int64_t i0, દશાંશ ext_factor, દશાંશ mscale,
    દશાંશ * cos_theta, દશાંશ * sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    દશાંશ theta_interp = freq_scale * theta_extrap;
    દશાંશ theta = theta_interp;
    જો (ext_factor != 0.0f) {
        દશાંશ ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
}

સ્થિર શૂન્ય ggml_rope_cache_init(
     દશાંશ theta_base, દશાંશ freq_scale, અચલ દશાંશ * freq_factors, દશાંશ corr_dims[2], int64_t ne0, દશાંશ ext_factor, દશાંશ mscale,
     દશાંશ * cache, દશાંશ sin_sign, દશાંશ theta_scale) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    દશાંશ theta = theta_base;
    માટે (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        અચલ દશાંશ ff = freq_factors ? freq_factors[i0/2] : 1.0f;
        rope_yarn(
            theta/ff, freq_scale, corr_dims, i0, ext_factor, mscale, &cache[i0 + 0], &cache[i0 + 1]
        );
        cache[i0 + 1] *= sin_sign;

        theta *= theta_scale;
    }
}

સ્થિર શૂન્ય ggml_mrope_cache_init(
     દશાંશ theta_base_t, દશાંશ theta_base_h, દશાંશ theta_base_w, દશાંશ theta_base_e, પૂર્ણાંક sections[4], તાર્કિક is_imrope, તાર્કિક indep_sects,
     દશાંશ freq_scale, અચલ દશાંશ * freq_factors, દશાંશ corr_dims[2], int64_t ne0, દશાંશ ext_factor, દશાંશ mscale,
     દશાંશ * cache, દશાંશ sin_sign, દશાંશ theta_scale) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    દશાંશ theta_t = theta_base_t;
    દશાંશ theta_h = theta_base_h;
    દશાંશ theta_w = theta_base_w;
    દશાંશ theta_e = theta_base_e;  // extra position id for vision encoder
    પૂર્ણાંક sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    પૂર્ણાંક sec_w = sections[1] + sections[0];
    પૂર્ણાંક sec_e = sections[2] + sec_w;
    GGML_ASSERT(sect_dims <= ne0);

    માટે (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        અચલ દશાંશ ff = freq_factors ? freq_factors[i0/2] : 1.0f;

        પૂર્ણાંક sector = (i0 / 2) % sect_dims;
        જો (indep_sects) {
            // compute theta independently for each dim sections
            // (i.e. reset corresponding theta when `i0` go from one section to another)
            જો (sector == 0) {
                theta_t = theta_base_t;
            }
            નહીંતર જો (sector == sections[0]) {
                theta_h = theta_base_h;;
            }
            નહીંતર જો (sector == sec_w) {
                theta_w = theta_base_w;
            }
            નહીંતર જો (sector == sec_e) {
                theta_e = theta_base_e;
            }
        }

        દશાંશ theta = theta_t;
        જો (is_imrope) { // qwen3vl apply interleaved mrope
            જો (sector % 3 == 1 && sector < 3 * sections[1]) {
                theta = theta_h;
            } નહીંતર જો (sector % 3 == 2 && sector < 3 * sections[2]) {
                theta = theta_w;
            } નહીંતર જો (sector % 3 == 0 && sector < 3 * sections[0]) {
                theta = theta_t;
            } નહીંતર {
                theta = theta_e;
            }
        } નહીંતર {
            જો (sector >= sections[0] && sector < sec_w) {
                theta = theta_h;
            }
            નહીંતર જો (sector >= sec_w && sector < sec_w + sections[2]) {
                theta = theta_w;
            }
            નહીંતર જો (sector >= sec_w + sections[2]) {
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
સ્થિર શૂન્ય rotate_pairs(અચલ int64_t n, અચલ int64_t n_offset, અચલ દશાંશ * cache, અચલ T * src_data, T * dst_data, અચલ પૂર્ણાંક scale = 2) {
  માટે (int64_t i0 = 0; i0 < n; i0 += 2) {
    અચલ int64_t ic = i0/scale; // hack for GGML_ROPE_TYPE_NORMAL, where we need ic = i0; for all other cases, ic = i0/2

    અચલ દશાંશ cos_theta = cache[i0 + 0];
    અચલ દશાંશ sin_theta = cache[i0 + 1];

    અચલ T * અચલ src = src_data + ic;
    T * dst             = dst_data + ic;

    અચલ દશાંશ x0 = type_conversion_table<T>::to_f32(src[0]);
    અચલ દશાંશ x1 = type_conversion_table<T>::to_f32(src[n_offset]);

    dst[0]        = type_conversion_table<T>::from_f32(x0*cos_theta - x1*sin_theta);
    dst[n_offset] = type_conversion_table<T>::from_f32(x0*sin_theta + x1*cos_theta);
  }
}

template<typename T> //float or ggml_fp16_t
સ્થિર શૂન્ય ggml_compute_forward_rope_flt(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst,
        અચલ તાર્કિક forward) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];
    અચલ ggml_tensor * src2 = dst->src[2];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);

    દશાંશ freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    પૂર્ણાંક sections[4];

    //const int n_past     = ((int32_t *) dst->op_params)[0];
    અચલ પૂર્ણાંક n_dims     = ((int32_t *) dst->op_params)[1];
    અચલ પૂર્ણાંક mode       = ((int32_t *) dst->op_params)[2];
    //const int n_ctx      = ((int32_t *) dst->op_params)[3];
    અચલ પૂર્ણાંક n_ctx_orig = ((int32_t *) dst->op_params)[4];

    memcpy(&freq_base,   (int32_t *) dst->op_params +  5, કદ(દશાંશ));
    memcpy(&freq_scale,  (int32_t *) dst->op_params +  6, કદ(દશાંશ));
    memcpy(&ext_factor,  (int32_t *) dst->op_params +  7, કદ(દશાંશ));
    memcpy(&attn_factor, (int32_t *) dst->op_params +  8, કદ(દશાંશ));
    memcpy(&beta_fast,   (int32_t *) dst->op_params +  9, કદ(દશાંશ));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, કદ(દશાંશ));
    memcpy(&sections,    (int32_t *) dst->op_params + 11, કદ(પૂર્ણાંક)*4);

    અચલ પૂર્ણાંક n_offs = ((int32_t *) dst->op_params)[15];

    GGML_TENSOR_UNARY_OP_LOCALS

    //printf("ne0: %d, ne1: %d, ne2: %d, ne3: %d\n", ne0, ne1, ne2, ne3);
    //printf("n_past = %d, ne2 = %d\n", n_past, ne2);

    GGML_ASSERT(nb0 == nb00);
    GGML_ASSERT(nb0 == કદ(T));

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક nr = ggml_nrows(dst);

    GGML_ASSERT(n_dims <= ne0);
    GGML_ASSERT(n_dims % 2 == 0);

    GGML_ASSERT(n_offs >= 0);
    GGML_ASSERT(n_offs % 2 == 0);
    GGML_ASSERT(n_offs + n_dims <= ne0);

    // rows per thread
    અચલ પૂર્ણાંક dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr*ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    // row index used to determine which thread to use
    પૂર્ણાંક ir = 0;

    અચલ દશાંશ theta_scale = powf(freq_base, -2.0f/n_dims);

    દશાંશ corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    અચલ તાર્કિક is_imrope = mode == GGML_ROPE_TYPE_IMROPE; // qwen3vl apply interleaved mrope
    અચલ તાર્કિક mrope_used = mode & GGML_ROPE_TYPE_MROPE;  // ggml_rope_multi, note: also true for vision (24 & 8 == true) and for imrope
    અચલ તાર્કિક is_vision = mode == GGML_ROPE_TYPE_VISION;

    જો (mrope_used) {
        GGML_ASSERT(sections[0] > 0 || sections[1] > 0 || sections[2] > 0);
    }

    જો (is_vision) {
        GGML_ASSERT(n_dims == ne0/2);
        GGML_ASSERT(n_offs == 0);
    }

    અચલ દશાંશ * freq_factors = ખાલી;
    જો (src2 != ખાલી) {
        GGML_ASSERT(src2->type == GGML_TYPE_F32);
        GGML_ASSERT(src2->ne[0] >= n_dims / 2);
        freq_factors = (અચલ દશાંશ *) src2->data;
    }

    // backward process uses inverse rotation by cos and sin.
    // cos and sin build a rotation matrix, where the inverse is the transpose.
    // this essentially just switches the sign of sin.
    અચલ દશાંશ sin_sign = forward ? 1.0f : -1.0f;

    અચલ int32_t * pos = (અચલ int32_t *) src1->data;

    int64_t last_i2 = -1;

    માટે (int64_t i3 = 0; i3 < ne3; i3++) { // batch
        માટે (int64_t i2 = 0; i2 < ne2; i2++) { // seq-len
            માટે (int64_t i1 = 0; i1 < ne1; i1++) { // attn-heads
                જો (ir++ < ir0) ચાલુ; // skip rows mapped to other threads
                જો (ir   > ir1) તોડો;

                દશાંશ * cache = (દશાંશ *) params->wdata + (ne0 + CACHE_લીટી_SIZE_F32)*ith;
                જો (last_i2 != i2) {
                    જો (!mrope_used) {
                        અચલ int64_t p = pos[i2];
                        ggml_rope_cache_init(p, freq_scale, freq_factors, corr_dims, ne0, ext_factor, attn_factor, cache, sin_sign, theta_scale);
                    }
                    નહીંતર {
                        અચલ int64_t p_t = pos[i2];
                        અચલ int64_t p_h = pos[i2 + ne2];
                        અચલ int64_t p_w = pos[i2 + ne2 * 2];
                        અચલ int64_t p_e = pos[i2 + ne2 * 3];
                        ggml_mrope_cache_init(
                            p_t, p_h, p_w, p_e, sections, is_imrope, is_vision,
                            freq_scale, freq_factors, corr_dims, ne0, ext_factor, attn_factor, cache, sin_sign, theta_scale);
                    }

                    last_i2 = i2;
                }

                T * src = (T *)((અક્ષર *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
                T * dst_data  = (T *)((અક્ષર *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1);

                પસંદગી (mode) {
                    સ્થિતિ GGML_ROPE_TYPE_NORMAL:
                        rotate_pairs<T>(n_dims, 1, cache, src + n_offs, dst_data + n_offs, 1);
                        તોડો;
                    સ્થિતિ GGML_ROPE_TYPE_NEOX:
                    સ્થિતિ GGML_ROPE_TYPE_MROPE:
                    સ્થિતિ GGML_ROPE_TYPE_IMROPE:
                        rotate_pairs<T>(n_dims, n_dims/2, cache, src + n_offs, dst_data + n_offs);
                        તોડો;
                    સ્થિતિ GGML_ROPE_TYPE_VISION:
                        rotate_pairs<T>(ne0, n_dims, cache, src, dst_data);
                        તોડો;
                    મૂળભૂત:
                        GGML_ABORT("rope type not supported");
                }

                જો (!is_vision) {
                    // fill the remain channels with data from src tensor
                    માટે (int64_t i0 = 0; i0 < ne0; i0 += 2) {
                        જો (i0 == n_offs) {
                            i0 += n_dims - 2; // skip the rotated channels
                            ચાલુ;
                        }
                        અચલ T * અચલ src = (T *)((અક્ષર *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                        T * dst_data  = (T *)((અક્ષર *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

                        dst_data[0] = src[0];
                        dst_data[1] = src[1];
                    }
                }
            } //attn-heads
        }
    }
}

શૂન્ય ggml_compute_forward_rope(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F16:
            {
                ggml_compute_forward_rope_flt<ggml_fp16_t>(params, dst, સાચું);
            } તોડો;
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_rope_flt<દશાંશ>(params, dst, સાચું);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_rope_back

શૂન્ય ggml_compute_forward_rope_back(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F16:
            {
                ggml_compute_forward_rope_flt<ggml_fp16_t>(params, dst, ખોટું);
            } તોડો;
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_rope_flt<દશાંશ>(params, dst, ખોટું);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_conv_transpose_1d

સ્થિર શૂન્ય ggml_compute_forward_conv_transpose_1d_f16_f32(
        અચલ ggml_compute_params * params,
              ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_દ્વિઅંકી_OP_LOCALS

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક nk = ne00*ne01*ne02;

    GGML_ASSERT(nb00 == કદ(ggml_fp16_t));
    GGML_ASSERT(nb10 == કદ(દશાંશ));

    જો (ith == 0) {
        memset(params->wdata, 0, params->wsize);

        // permute kernel data (src0) from (K x Cout x Cin) to (Cin x K x Cout)
        {
            ggml_fp16_t * અચલ wdata = (ggml_fp16_t *) params->wdata + 0;

            માટે (int64_t i02 = 0; i02 < ne02; i02++) {
                માટે (int64_t i01 = 0; i01 < ne01; i01++) {
                    અચલ ggml_fp16_t * અચલ src = (ggml_fp16_t *)((અક્ષર *) src0->data + i02*nb02 + i01*nb01);
                    ggml_fp16_t * dst_data = wdata + i01*ne00*ne02;
                    માટે (int64_t i00 = 0; i00 < ne00; i00++) {
                        dst_data[i00*ne02 + i02] = src[i00];
                    }
                }
            }
        }

        // permute source data (src1) from (L x Cin) to (Cin x L)
        {
            ggml_fp16_t * અચલ wdata = (ggml_fp16_t *) params->wdata + nk;
            ggml_fp16_t * dst_data = wdata;

            માટે (int64_t i11 = 0; i11 < ne11; i11++) {
                અચલ દશાંશ * અચલ src = (દશાંશ *)((અક્ષર *) src1->data + i11*nb11);
                માટે (int64_t i10 = 0; i10 < ne10; i10++) {
                    dst_data[i10*ne11 + i11] = GGML_CPU_FP32_TO_FP16(src[i10]);
                }
            }
        }

        // need to zero dst since we are accumulating into it
        memset(dst->data, 0, ggml_nbytes(dst));
    }
    ggml_barrier(params->threadpool);

    અચલ int32_t s0 = ((અચલ int32_t*)(dst->op_params))[0];

    // total rows in dst
    અચલ પૂર્ણાંક nr = ne1;

    // rows per thread
    અચલ પૂર્ણાંક dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr*ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    ggml_fp16_t * અચલ wdata     = (ggml_fp16_t *) params->wdata + 0;
    ggml_fp16_t * અચલ wdata_src = wdata + nk;

    માટે (પૂર્ણાંક i1 = ir0; i1 < ir1; i1++) {
        દશાંશ * dst_data = (દશાંશ *)((અક્ષર *) dst->data + i1*nb1);
        ggml_fp16_t * wdata_kernel = wdata + i1*ne02*ne00;
        માટે (પૂર્ણાંક i10 = 0; i10 < ne10; i10++) {
            અચલ પૂર્ણાંક i1n = i10*ne11;
            માટે (પૂર્ણાંક i00 = 0; i00 < ne00; i00++) {
                દશાંશ v = 0;
                ggml_vec_dot_f16(ne02, &v, 0,
                        (ggml_fp16_t *)    wdata_src + i1n, 0,
                        (ggml_fp16_t *) wdata_kernel + i00*ne02, 0, 1);
                dst_data[i10*s0 + i00] += v;
            }
        }
    }
}

સ્થિર શૂન્ય ggml_compute_forward_conv_transpose_1d_f32(
        અચલ ggml_compute_params * params,
              ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_દ્વિઅંકી_OP_LOCALS

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક nk = ne00*ne01*ne02;

    GGML_ASSERT(nb00 == કદ(દશાંશ));
    GGML_ASSERT(nb10 == કદ(દશાંશ));

    જો (ith == 0) {
        memset(params->wdata, 0, params->wsize);

        // prepare kernel data (src0) from (K x Cout x Cin) to (Cin x K x Cout)
        {
            દશાંશ * અચલ wdata = (દશાંશ *) params->wdata + 0;

            માટે (int64_t i02 = 0; i02 < ne02; i02++) {
                માટે (int64_t i01 = 0; i01 < ne01; i01++) {
                    અચલ દશાંશ * અચલ src = (દશાંશ *)((અક્ષર *) src0->data + i02*nb02 + i01*nb01);
                    દશાંશ * dst_data = wdata + i01*ne00*ne02;
                    માટે (int64_t i00 = 0; i00 < ne00; i00++) {
                        dst_data[i00*ne02 + i02] = src[i00];
                    }
                }
            }
        }

        // prepare source data (src1)
        {
            દશાંશ * અચલ wdata = (દશાંશ *) params->wdata + nk;
            દશાંશ * dst_data = wdata;

            માટે (int64_t i11 = 0; i11 < ne11; i11++) {
                અચલ દશાંશ * અચલ src = (દશાંશ *)((અક્ષર *) src1->data + i11*nb11);
                માટે (int64_t i10 = 0; i10 < ne10; i10++) {
                    dst_data[i10*ne11 + i11] = src[i10];
                }
            }
        }

        // need to zero dst since we are accumulating into it
        memset(dst->data, 0, ggml_nbytes(dst));
    }
    ggml_barrier(params->threadpool);

    અચલ int32_t s0 = ((અચલ int32_t*)(dst->op_params))[0];

    // total rows in dst
    અચલ પૂર્ણાંક nr = ne1;

    // rows per thread
    અચલ પૂર્ણાંક dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr*ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    દશાંશ * અચલ wdata     = (દશાંશ *) params->wdata + 0;
    દશાંશ * અચલ wdata_src = wdata + nk;

    માટે (પૂર્ણાંક i1 = ir0; i1 < ir1; i1++) {
        દશાંશ * dst_data = (દશાંશ *)((અક્ષર *) dst->data + i1*nb1);
        દશાંશ * wdata_kernel = wdata + i1*ne02*ne00;
        માટે (પૂર્ણાંક i10 = 0; i10 < ne10; i10++) {
            અચલ પૂર્ણાંક i1n = i10*ne11;
            માટે (પૂર્ણાંક i00 = 0; i00 < ne00; i00++) {
                દશાંશ v = 0;
                ggml_vec_dot_f32(ne02, &v, 0,
                        wdata_src + i1n, 0,
                        wdata_kernel + i00*ne02, 0, 1);
                dst_data[i10*s0 + i00] += v;
            }
        }
    }
}

શૂન્ય ggml_compute_forward_conv_transpose_1d(
        અચલ ggml_compute_params * params,
              ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F16:
            {
                ggml_compute_forward_conv_transpose_1d_f16_f32(params, dst);
            } તોડો;
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_conv_transpose_1d_f32(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_im2col_f32
// src0: kernel [OC, IC, KH, KW]
// src1: image [N, IC, IH, IW]
// dst:  result [N, OH, OW, IC*KH*KW]
સ્થિર શૂન્ય ggml_compute_forward_im2col_f32(
        અચલ ggml_compute_params * params,
              ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_દ્વિઅંકી_OP_LOCALS;

    અચલ int32_t s0 = ((અચલ int32_t *)(dst->op_params))[0];
    અચલ int32_t s1 = ((અચલ int32_t *)(dst->op_params))[1];
    અચલ int32_t p0 = ((અચલ int32_t *)(dst->op_params))[2];
    અચલ int32_t p1 = ((અચલ int32_t *)(dst->op_params))[3];
    અચલ int32_t d0 = ((અચલ int32_t *)(dst->op_params))[4];
    અચલ int32_t d1 = ((અચલ int32_t *)(dst->op_params))[5];
    અચલ તાર્કિક is_2D = ((અચલ int32_t *)(dst->op_params))[6] == 1;

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ int64_t N  = is_2D ? ne13 : ne12;
    અચલ int64_t IC = is_2D ? ne12 : ne11;
    અચલ int64_t IH = is_2D ? ne11 : 1;
    અચલ int64_t IW = ne10;

    અચલ int64_t KH = is_2D ? ne01 : 1;
    અચલ int64_t KW = ne00;

    અચલ int64_t OH = is_2D ? ne2 : 1;
    અચલ int64_t OW = ne1;

    પૂર્ણાંક ofs0 = is_2D ? nb13 : nb12;
    પૂર્ણાંક ofs1 = is_2D ? nb12 : nb11;

    GGML_ASSERT(nb10 == કદ(દશાંશ));

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    {
        દશાંશ * અચલ wdata = (દશાંશ *) dst->data;

        માટે (int64_t in = 0; in < N; in++) {
            માટે (int64_t ioh = 0; ioh < OH; ioh++) { // 1
                માટે (int64_t iow = 0; iow < OW; iow++) {
                    માટે (int64_t iic = ith; iic < IC; iic += nth) {

                        // micro kernel
                        દશાંશ * dst_data = wdata + (in*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
                        અચલ દશાંશ * અચલ src_data = (દશાંશ *)((અક્ષર *) src1->data + in*ofs0 + iic*ofs1); // [IH, IW]

                        માટે (int64_t ikh = 0; ikh < KH; ikh++) {  // 1
                            માટે (int64_t ikw = 0; ikw < KW; ikw++) {
                                અચલ int64_t iiw = iow*s0 + ikw*d0 - p0;
                                અચલ int64_t iih = ioh*s1 + ikh*d1 - p1;

                                જો (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                    dst_data[iic*(KH*KW) + ikh*KW + ikw] = 0;
                                } નહીંતર {
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
સ્થિર શૂન્ય ggml_compute_forward_im2col_f16(
        અચલ ggml_compute_params * params,
              ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16);

    GGML_TENSOR_દ્વિઅંકી_OP_LOCALS;

    અચલ int32_t s0 = ((અચલ int32_t *)(dst->op_params))[0];
    અચલ int32_t s1 = ((અચલ int32_t *)(dst->op_params))[1];
    અચલ int32_t p0 = ((અચલ int32_t *)(dst->op_params))[2];
    અચલ int32_t p1 = ((અચલ int32_t *)(dst->op_params))[3];
    અચલ int32_t d0 = ((અચલ int32_t *)(dst->op_params))[4];
    અચલ int32_t d1 = ((અચલ int32_t *)(dst->op_params))[5];
    અચલ તાર્કિક is_2D = ((અચલ int32_t *)(dst->op_params))[6] == 1;

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ int64_t N  = is_2D ? ne13 : ne12;
    અચલ int64_t IC = is_2D ? ne12 : ne11;
    અચલ int64_t IH = is_2D ? ne11 : 1;
    અચલ int64_t IW = ne10;

    અચલ int64_t KH = is_2D ? ne01 : 1;
    અચલ int64_t KW = ne00;

    અચલ int64_t OH = is_2D ? ne2 : 1;
    અચલ int64_t OW = ne1;

    પૂર્ણાંક ofs0 = is_2D ? nb13 : nb12;
    પૂર્ણાંક ofs1 = is_2D ? nb12 : nb11;

    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    {
        ggml_fp16_t * અચલ wdata = (ggml_fp16_t *) dst->data;

        માટે (int64_t in = 0; in < N; in++) {
            માટે (int64_t ioh = 0; ioh < OH; ioh++) { // 1
                માટે (int64_t iow = 0; iow < OW; iow++) {
                    માટે (int64_t iic = ith; iic < IC; iic += nth) {

                        // micro kernel
                        ggml_fp16_t * dst_data = wdata + (in*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
                        અચલ દશાંશ * અચલ src_data_f32 = src1->type == GGML_TYPE_F32
                            ? (અચલ દશાંશ *)((અચલ અક્ષર *) src1->data + in*ofs0 + iic*ofs1)
                            : nullptr; // [IH, IW]
                        અચલ ggml_fp16_t * અચલ src_data_f16 = src1->type == GGML_TYPE_F16
                            ? (અચલ ggml_fp16_t *)((અચલ અક્ષર *) src1->data + in*ofs0 + iic*ofs1)
                            : nullptr; // [IH, IW]

                        માટે (int64_t ikh = 0; ikh < KH; ikh++) {  // 1
                            માટે (int64_t ikw = 0; ikw < KW; ikw++) {
                                અચલ int64_t iiw = iow*s0 + ikw*d0 - p0;
                                અચલ int64_t iih = ioh*s1 + ikh*d1 - p1;

                                જો (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                    dst_data[iic*(KH*KW) + ikh*KW + ikw] = 0;
                                } નહીંતર {
                                    જો (src_data_f32 != nullptr) {
                                        dst_data[iic*(KH*KW) + ikh*KW + ikw] = GGML_CPU_FP32_TO_FP16(src_data_f32[iih*IW + iiw]);
                                    } નહીંતર {
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

શૂન્ય ggml_compute_forward_im2col(
        અચલ ggml_compute_params * params,
              ggml_tensor * dst) {
    પસંદગી (dst->type) {
        સ્થિતિ GGML_TYPE_F16:
            {
                ggml_compute_forward_im2col_f16(params, dst);
            } તોડો;
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_im2col_f32(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_im2col_back_f32

શૂન્ય ggml_compute_forward_im2col_back_f32(
        અચલ ggml_compute_params * params,
              ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0]; // gradients of forward pass output
    અચલ ggml_tensor * src1 = dst->src[1]; // convolution kernel

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_દ્વિઅંકી_OP_LOCALS;

    અચલ int32_t s0 = ((અચલ int32_t *)(dst->op_params))[0];
    અચલ int32_t s1 = ((અચલ int32_t *)(dst->op_params))[1];
    અચલ int32_t p0 = ((અચલ int32_t *)(dst->op_params))[2];
    અચલ int32_t p1 = ((અચલ int32_t *)(dst->op_params))[3];
    અચલ int32_t d0 = ((અચલ int32_t *)(dst->op_params))[4];
    અચલ int32_t d1 = ((અચલ int32_t *)(dst->op_params))[5];
    અચલ તાર્કિક is_2D = ((અચલ int32_t *)(dst->op_params))[6] == 1;

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ int64_t N  = is_2D ? ne3 : ne2;
    અચલ int64_t IC = is_2D ? ne2 : ne1;
    અચલ int64_t IH = is_2D ? ne1 : 1;
    અચલ int64_t IW = ne0;

    અચલ int64_t KH = is_2D ? ne11 : 1;
    અચલ int64_t KW = ne10;

    અચલ int64_t OH = is_2D ? ne02 : 1;
    અચલ int64_t OW = ne01;

    પૂર્ણાંક ofs0 = is_2D ? nb3 : nb2;
    પૂર્ણાંક ofs1 = is_2D ? nb2 : nb1;

    GGML_ASSERT(nb0  == કદ(દશાંશ));

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    {
        દશાંશ * અચલ wdata = (દશાંશ *) dst->data;

        માટે (int64_t in = 0; in < N; in++) {
            માટે (int64_t iic = ith; iic < IC; iic += nth) {
                માટે (int64_t iih = 0; iih < IH; iih++) {
                    માટે (int64_t iiw = 0; iiw < IW; iiw++) {

                        // micro kernel
                        દશાંશ grad = 0.0f;
                        માટે (int64_t ikh = 0; ikh < KH; ikh++) {
                            માટે (int64_t ikw = 0; ikw < KW; ikw++) {
                                // For s0 > 1 some values were skipped over in the forward pass.
                                // These values have tmpw % s0 != 0 and need to be skipped in the backwards pass as well.
                                અચલ int64_t tmpw = (iiw + p0 - ikw*d0);
                                જો (tmpw % s0 != 0) {
                                    ચાલુ;
                                }
                                અચલ int64_t iow = tmpw / s0;

                                // Equivalent logic as above except for s1.
                                int64_t ioh;
                                જો (is_2D) {
                                    અચલ int64_t tmph = iih + p1 - ikh*d1;

                                    જો (tmph % s1 != 0) {
                                        ચાલુ;
                                    }

                                    ioh = tmph / s1;
                                } નહીંતર {
                                    ioh = 0;
                                }

                                જો (iow < 0 || iow >= OW || ioh < 0 || ioh >= OH) {
                                    ચાલુ;
                                }

                                અચલ દશાંશ * અચલ grad_in = (અચલ દશાંશ *) src0->data
                                    + (in*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
                                grad += grad_in[iic*(KH*KW) + ikh*KW + ikw];
                            }
                        }
                        દશાંશ * dst_data = (દશાંશ *)((અક્ષર *) wdata + (in*ofs0 + iic*ofs1)); // [IH, IW]
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
સ્થિર શૂન્ય ggml_compute_forward_im2col_3d_f16(
        અચલ ggml_compute_params * params,
              ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16);

    GGML_TENSOR_દ્વિઅંકી_OP_LOCALS;

    અચલ int32_t s0 = ((અચલ int32_t *)(dst->op_params))[0];
    અચલ int32_t s1 = ((અચલ int32_t *)(dst->op_params))[1];
    અચલ int32_t s2 = ((અચલ int32_t *)(dst->op_params))[2];
    અચલ int32_t p0 = ((અચલ int32_t *)(dst->op_params))[3];
    અચલ int32_t p1 = ((અચલ int32_t *)(dst->op_params))[4];
    અચલ int32_t p2 = ((અચલ int32_t *)(dst->op_params))[5];
    અચલ int32_t d0 = ((અચલ int32_t *)(dst->op_params))[6];
    અચલ int32_t d1 = ((અચલ int32_t *)(dst->op_params))[7];
    અચલ int32_t d2 = ((અચલ int32_t *)(dst->op_params))[8];
    અચલ int32_t IC = ((અચલ int32_t *)(dst->op_params))[9];


    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ int64_t N  = ne13 / IC;
    અચલ int64_t ID = ne12;
    અચલ int64_t IH = ne11;
    અચલ int64_t IW = ne10;

    અચલ int64_t OC = ne03 / IC;
    GGML_UNUSED(OC);
    અચલ int64_t KD = ne02;
    અચલ int64_t KH = ne01;
    અચલ int64_t KW = ne00;

    અચલ int64_t OD = ne3 / N;
    અચલ int64_t OH = ne2;
    અચલ int64_t OW = ne1;
    અચલ int64_t OH_OW = OH*OW;
    અચલ int64_t KD_KH_KW = KD*KH*KW;
    અચલ int64_t KH_KW = KH*KW;
    અચલ int64_t IC_KD_KH_KW = IC*KD*KH*KW;

    GGML_ASSERT(nb10 == કદ(દશાંશ));

    // im2col: [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
    {
        ggml_fp16_t * અચલ wdata = (ggml_fp16_t *) dst->data;

        માટે (int64_t in = 0; in < N; in++) {
            માટે (int64_t iod = 0; iod < OD; iod++) {
                માટે (int64_t ioh = 0; ioh < OH; ioh++) {
                    માટે (int64_t iow = 0; iow < OW; iow++) {
                        માટે (int64_t iic = ith; iic < IC; iic += nth) {

                            // micro kernel
                            ggml_fp16_t * dst_data = wdata + (in*OD*OH_OW + iod*OH_OW + ioh*OW + iow)*IC_KD_KH_KW; // [IC, KD, KH, KW]
                            અચલ દશાંશ * અચલ src_data = (અચલ દશાંશ *) ((અચલ અક્ષર *)src1->data + (in*IC + iic)*nb13); // [ID, IH, IW]

                            માટે (int64_t ikd = 0; ikd < KD; ikd++) {
                                માટે (int64_t ikh = 0; ikh < KH; ikh++) {
                                    માટે (int64_t ikw = 0; ikw < KW; ikw++) {
                                        અચલ int64_t iiw = iow*s0 + ikw*d0 - p0;
                                        અચલ int64_t iih = ioh*s1 + ikh*d1 - p1;
                                        અચલ int64_t iid = iod*s2 + ikd*d2 - p2;

                                        જો (iid < 0 || iid >= ID || iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                            dst_data[iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw] = 0;
                                        } નહીંતર {
                                            અચલ દશાંશ * અચલ s = (અચલ દશાંશ *) ((અચલ અક્ષર *)src_data + iid*nb12 + iih*nb11 + iiw*nb10); // [ID, IH, IW]
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
સ્થિર શૂન્ય ggml_compute_forward_im2col_3d_f32(
        અચલ ggml_compute_params * params,
              ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_દ્વિઅંકી_OP_LOCALS;

    અચલ int32_t s0 = ((અચલ int32_t *)(dst->op_params))[0];
    અચલ int32_t s1 = ((અચલ int32_t *)(dst->op_params))[1];
    અચલ int32_t s2 = ((અચલ int32_t *)(dst->op_params))[2];
    અચલ int32_t p0 = ((અચલ int32_t *)(dst->op_params))[3];
    અચલ int32_t p1 = ((અચલ int32_t *)(dst->op_params))[4];
    અચલ int32_t p2 = ((અચલ int32_t *)(dst->op_params))[5];
    અચલ int32_t d0 = ((અચલ int32_t *)(dst->op_params))[6];
    અચલ int32_t d1 = ((અચલ int32_t *)(dst->op_params))[7];
    અચલ int32_t d2 = ((અચલ int32_t *)(dst->op_params))[8];
    અચલ int32_t IC = ((અચલ int32_t *)(dst->op_params))[9];


    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ int64_t N  = ne13 / IC;
    અચલ int64_t ID = ne12;
    અચલ int64_t IH = ne11;
    અચલ int64_t IW = ne10;

    અચલ int64_t OC = ne03 / IC;
    GGML_UNUSED(OC);
    અચલ int64_t KD = ne02;
    અચલ int64_t KH = ne01;
    અચલ int64_t KW = ne00;

    અચલ int64_t OD = ne3 / N;
    અચલ int64_t OH = ne2;
    અચલ int64_t OW = ne1;

    અચલ int64_t OH_OW = OH*OW;
    અચલ int64_t KD_KH_KW = KD*KH*KW;
    અચલ int64_t KH_KW = KH*KW;
    અચલ int64_t IC_KD_KH_KW = IC*KD*KH*KW;

    GGML_ASSERT(nb10 == કદ(દશાંશ));

    // im2col: [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
    {
        દશાંશ * અચલ wdata = (દશાંશ *) dst->data;

        માટે (int64_t in = 0; in < N; in++) {
            માટે (int64_t iod = 0; iod < OD; iod++) {
                માટે (int64_t ioh = 0; ioh < OH; ioh++) {
                    માટે (int64_t iow = 0; iow < OW; iow++) {
                        માટે (int64_t iic = ith; iic < IC; iic += nth) {

                            // micro kernel
                            દશાંશ * dst_data = wdata + (in*OD*OH_OW + iod*OH_OW + ioh*OW + iow)*IC_KD_KH_KW; // [IC, KD, KH, KW]
                            અચલ દશાંશ * અચલ src_data = (અચલ દશાંશ *) ((અચલ અક્ષર *)src1->data + (in*IC + iic)*nb13); // [ID, IH, IW]

                            માટે (int64_t ikd = 0; ikd < KD; ikd++) {
                                માટે (int64_t ikh = 0; ikh < KH; ikh++) {
                                    માટે (int64_t ikw = 0; ikw < KW; ikw++) {
                                        અચલ int64_t iiw = iow*s0 + ikw*d0 - p0;
                                        અચલ int64_t iih = ioh*s1 + ikh*d1 - p1;
                                        અચલ int64_t iid = iod*s2 + ikd*d2 - p2;

                                        જો (iid < 0 || iid >= ID || iih < 0 || iih >= IH || iiw < 0 || iiw >= IW || iid < 0 || iid >= ID) {
                                            dst_data[iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw] = 0;
                                        } નહીંતર {
                                            અચલ દશાંશ * અચલ s = (અચલ દશાંશ *) ((અચલ અક્ષર *)src_data + iid*nb12 + iih*nb11 + iiw*nb10); // [ID, IH, IW]
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


શૂન્ય ggml_compute_forward_im2col_3d(
        અચલ ggml_compute_params * params,
              ggml_tensor * dst) {
    પસંદગી (dst->type) {
        સ્થિતિ GGML_TYPE_F16:
            {
                ggml_compute_forward_im2col_3d_f16(params, dst);
            } તોડો;
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_im2col_3d_f32(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

સ્થિર શૂન્ય ggml_call_mul_mat(ggml_type type, અચલ ggml_compute_params * params, int64_t m, int64_t n, int64_t k,
                              શૂન્ય * a, શૂન્ય * b, દશાંશ * c) {
    અચલ ggml_type_traits * traits = ggml_get_type_traits(type);
    સંરચના ggml_tensor src1 = {};
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

    સંરચના ggml_tensor src0 = {};
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

    સંરચના ggml_tensor dst = {};
    dst.ne[0] = n;
    dst.ne[1] = m;
    dst.ne[2] = 1;
    dst.ne[3] = 1;
    dst.nb[0] = કદ(દશાંશ);
    dst.nb[1] = n * કદ(દશાંશ);
    dst.nb[2] = dst.nb[1];
    dst.nb[3] = dst.nb[2];
    dst.data  = c;
    dst.src[0] = &src0;
    dst.src[1] = &src1;

    ggml_compute_forward_mul_mat(params, &dst);
}

સ્થિર અંતઃરેખ int64_t ggml_wrap_around(int64_t coord, int64_t size) {
    પાછા (coord  + size) % size; // adding size avoids negative number weirdness
}

// ggml_compute_forward_col2im_1d
//
// Scatter-add columns [K*OC, T_in] -> signal [T_out, OC]
// where T_out = (T_in - 1)*s + K - 2*p.  Gather approach: each output reads ceil(K/s) inputs.
// Parallelized over the time axis so the split stays balanced whatever OC is.
// Supports F32, F16, BF16 input/output (same type), F32 accumulator.

template <typename elem_t>
સ્થિર શૂન્ય ggml_compute_forward_col2im_1d_impl(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src = dst->src[0];  // [K*OC, T_in]

    GGML_ASSERT(ggml_is_contiguous(src));
    GGML_ASSERT(ggml_is_contiguous(dst));

    અચલ int32_t s0 = ((અચલ int32_t *)(dst->op_params))[0];
    અચલ int32_t OC = ((અચલ int32_t *)(dst->op_params))[1];
    અચલ int32_t p0 = ((અચલ int32_t *)(dst->op_params))[2];

    અચલ int64_t K_OC = src->ne[0];
    અચલ int64_t T_in = src->ne[1];
    અચલ int64_t K    = K_OC / OC;
    અચલ int64_t T_out = dst->ne[0];

    અચલ elem_t * col_data = (અચલ elem_t *) src->data;
    elem_t       * dst_data = (elem_t *) dst->data;

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    // Parallelize over the time axis: the split stays balanced whatever OC is,
    // down to OC = 1 for mono audio, and threads read disjoint column bands
    અચલ int64_t dr = (T_out + nth - 1) / nth;
    અચલ int64_t it0 = dr * ith;
    અચલ int64_t it1 = it0 + dr < T_out ? it0 + dr : T_out;

    માટે (int64_t oc = 0; oc < OC; oc++) {
        માટે (int64_t t_out = it0; t_out < it1; t_out++) {
            અચલ int64_t t_abs = t_out + p0;  // absolute position in uncropped signal
            // Gather: find all (t_in, k) where t_in * s + k == t_abs, 0 <= k < K
            int64_t t_in_min = (t_abs - K + 1 + s0 - 1) / s0;  // ceil((t_abs-K+1)/s)
            જો (t_in_min < 0) t_in_min = 0;
            int64_t t_in_max = t_abs / s0;
            જો (t_in_max >= T_in) t_in_max = T_in - 1;

            દશાંશ sum = 0.0f;
            માટે (int64_t t_in = t_in_min; t_in <= t_in_max; t_in++) {
                int64_t k = t_abs - t_in * s0;
                જો (k >= 0 && k < K) {
                    // col layout: [K*OC, T_in], element (oc*K+k, t_in)
                    sum += type_conversion_table<elem_t>::to_f32(col_data[(oc * K + k) + t_in * K_OC]);
                }
            }
            // dst layout: [T_out, OC], element (t_out, oc)
            dst_data[t_out + oc * T_out] = type_conversion_table<elem_t>::from_f32(sum);
        }
    }
}

શૂન્ય ggml_compute_forward_col2im_1d(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {
    પસંદગી (dst->src[0]->type) {
        સ્થિતિ GGML_TYPE_F32:  ggml_compute_forward_col2im_1d_impl<દશાંશ>      (params, dst); તોડો;
        સ્થિતિ GGML_TYPE_F16:  ggml_compute_forward_col2im_1d_impl<ggml_fp16_t>(params, dst); તોડો;
        સ્થિતિ GGML_TYPE_BF16: ggml_compute_forward_col2im_1d_impl<ggml_bf16_t>(params, dst); તોડો;
        મૂળભૂત: GGML_ABORT("col2im_1d: unsupported type %d", dst->src[0]->type);
    }
}

// ggml_compute_forward_conv_2d


સ્થિર શૂન્ય ggml_compute_forward_conv_2d_impl(અચલ ggml_compute_params * params,
                                              અચલ ggml_tensor *         kernel,  // [KW, KH, IC, OC]
                                              અચલ ggml_tensor *         src,     // [W, H, C, N]
                                              ggml_tensor *               dst,     // [OW, OH, OC, N]
                                              ggml_type                   kernel_type) {

    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(kernel_type == GGML_TYPE_F16 || kernel_type == GGML_TYPE_F32);
    GGML_ASSERT(kernel->type == kernel_type);

    અચલ ggml_type_traits * traits = ggml_get_type_traits(kernel_type);

    અચલ int32_t stride_x   = dst->op_params[0];
    અચલ int32_t stride_y   = dst->op_params[1];
    અચલ int32_t pad_x      = dst->op_params[2];
    અચલ int32_t pad_y      = dst->op_params[3];
    અચલ int32_t dilation_x = dst->op_params[4];
    અચલ int32_t dilation_y = dst->op_params[5];

    અચલ int64_t c_in  = src->ne[2];
    અચલ int64_t c_out = kernel->ne[3];
    GGML_ASSERT(c_in == kernel->ne[2]);

    અચલ int64_t src_w = src->ne[0];
    અચલ int64_t src_h = src->ne[1];
    અચલ int64_t knl_w = kernel->ne[0];
    અચલ int64_t knl_h = kernel->ne[1];
    અચલ int64_t dst_w = dst->ne[0];
    અચલ int64_t dst_h = dst->ne[1];

    અચલ દશાંશ * src_data = (દશાંશ *) src->data;
    શૂન્ય  * knl_data       = kernel->data;
    દશાંશ * dst_data       = (દશાંશ *) dst->data;

    અચલ int64_t knl_n           = knl_w * knl_h * c_in;
    અચલ int64_t patch_total     = dst->ne[3] * dst_w * dst_h;

    અચલ int64_t space_per_patch   = knl_n * traits->type_size + c_out * કદ(દશાંશ);
    અચલ int64_t batch_size        = params->wsize / space_per_patch;
    અચલ int64_t patches_per_batch = batch_size > 8 ? (batch_size / 8) * 8 : batch_size;
    અચલ int64_t batch_n           = (patch_total + patches_per_batch - 1) / patches_per_batch;

    GGML_ASSERT(patches_per_batch > 0 && batch_size >= 1);

    શૂન્ય * tmp = params->wdata;

    માટે (int64_t batch_i = 0; batch_i < batch_n; ++batch_i) {

        અચલ int64_t patch_start_batch = batch_i * patches_per_batch;
        અચલ int64_t patch_end_batch   = std::min(patch_start_batch + patches_per_batch,
                                              patch_total);
        અચલ int64_t patch_n           = patch_end_batch - patch_start_batch;

        અચલ int64_t patch_per_thread  = (patch_n + params->nth - 1) / params->nth;
        અચલ int64_t patch_start       = patch_start_batch + params->ith * patch_per_thread;
        અચલ int64_t patch_end         = std::min(patch_start + patch_per_thread, patch_end_batch);

        //im2col for a patch
        માટે (int64_t p = patch_start; p < patch_end; ++p) {
            અચલ int64_t  batch_n     =  p / (dst_w * dst_h);
            અચલ int64_t  src_x       = (p / dst_w) % dst_h;
            અચલ int64_t  src_y       =  p % dst_w;

            અચલ દશાંશ * src_base = (અચલ દશાંશ *)((અચલ અક્ષર *)src_data + batch_n * src->nb[3]);
            અક્ષર *        dst_row  = (અક્ષર *) tmp + (p % patches_per_batch) * knl_n * traits->type_size;

            માટે (int64_t ic = 0; ic < c_in; ++ic) {
                માટે (int64_t ky = 0; ky < knl_h; ++ky) {
                    માટે (int64_t kx = 0; kx < knl_w; ++kx) {
                        અચલ int64_t sy = src_x * stride_y + ky * dilation_y - pad_y;
                        અચલ int64_t sx = src_y * stride_x + kx * dilation_x - pad_x;

                        int64_t dst_idx = ic * (knl_h * knl_w) + ky * knl_w + kx;

                        દશાંશ src_val;
                        જો (sy < 0 || sy >= src_h || sx < 0 || sx >= src_w) {
                            src_val = 0.0f;
                        } નહીંતર {
                            અચલ દશાંશ * src_ptr = (અચલ દશાંશ *)((અચલ અક્ષર *)src_base + sx * src->nb[0] + sy * src->nb[1] + ic * src->nb[2]);
                            src_val               = *src_ptr;
                        }

                        અક્ષર * element_ptr = dst_row + dst_idx * traits->type_size;
                        જો (kernel_type == GGML_TYPE_F32) {
                            *(દશાંશ *) element_ptr = src_val;
                        } નહીંતર જો (kernel_type == GGML_TYPE_F16) {
                            *(ggml_fp16_t *) element_ptr = GGML_CPU_FP32_TO_FP16(src_val);
                        }
                    }
                }
            }
        }   // patches handled by this thread

        ggml_barrier(params->threadpool);

        દશાંશ * gemm_output = (દશાંશ *) ((અક્ષર *) tmp + patches_per_batch * knl_n * traits->type_size);

        GGML_ASSERT(gemm_output + patch_n * c_out <= (દશાંશ*)tmp + params->wsize);

        // GEMM: patches[patch_n, knl_n] × kernel[knl_n, c_out] = output[patch_n, c_out]
        ggml_call_mul_mat(kernel_type, params, patch_n, c_out, knl_n, tmp, knl_data, gemm_output);

        ggml_barrier(params->threadpool);


        //permute back [OC, N, OH, OW] to [N, OC, OH, OW]
        અચલ int64_t permute_per_thread = (patch_n + params->nth - 1) / params->nth;
        અચલ int64_t permute_start = params->ith * permute_per_thread;
        અચલ int64_t permute_end = std::min(permute_start + permute_per_thread, patch_n);

        માટે (int64_t i = permute_start; i < permute_end; ++i) {
            અચલ int64_t p       = patch_start_batch + i;
            અચલ int64_t batch_n = p / (dst_w * dst_h);
            અચલ int64_t dst_y   = (p / dst_w) % dst_h;
            અચલ int64_t dst_x   = p % dst_w;

            માટે (int64_t oc = 0; oc < c_out; ++oc) {
                અચલ દશાંશ value = gemm_output[i * c_out + oc];
                દશાંશ * dst_ptr = (દશાંશ *)((અક્ષર *)dst_data + dst_x * dst->nb[0] + dst_y * dst->nb[1] + oc * dst->nb[2] + batch_n * dst->nb[3]);
                *dst_ptr = value;
            }
        }
    }
}

શૂન્ય ggml_compute_forward_conv_2d(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];

    ggml_compute_forward_conv_2d_impl(params, src0, src1, dst, src0->type);
}

// ggml_compute_forward_conv_3d

સ્થિર શૂન્ય ggml_compute_forward_conv_3d_impl(અચલ ggml_compute_params * params,
                                              અચલ ggml_tensor *         kernel,
                                              અચલ ggml_tensor *         src,
                                              ggml_tensor *               dst,
                                              ggml_type                   kernel_type) {

    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(kernel_type == GGML_TYPE_F16 || kernel_type == GGML_TYPE_F32);
    GGML_ASSERT(kernel->type == kernel_type);

    અચલ ggml_type_traits * traits = ggml_get_type_traits(kernel_type);

    અચલ int32_t s0 = dst->op_params[0];
    અચલ int32_t s1 = dst->op_params[1];
    અચલ int32_t s2 = dst->op_params[2];
    અચલ int32_t p0 = dst->op_params[3];
    અચલ int32_t p1 = dst->op_params[4];
    અચલ int32_t p2 = dst->op_params[5];
    અચલ int32_t d0 = dst->op_params[6];
    અચલ int32_t d1 = dst->op_params[7];
    અચલ int32_t d2 = dst->op_params[8];
    અચલ int32_t c  = dst->op_params[9];
    અચલ int32_t n  = dst->op_params[10];
    અચલ int32_t oc = dst->op_params[11];

    અચલ int64_t src_w = src->ne[0];
    અચલ int64_t src_h = src->ne[1];
    અચલ int64_t src_d = src->ne[2];
    અચલ int64_t knl_w = kernel->ne[0];
    અચલ int64_t knl_h = kernel->ne[1];
    અચલ int64_t knl_d = kernel->ne[2];
    અચલ int64_t dst_w = dst->ne[0];
    અચલ int64_t dst_h = dst->ne[1];
    અચલ int64_t dst_d = dst->ne[2];

    અચલ દશાંશ * src_data = (દશાંશ *) src->data;
    શૂન્ય  * knl_data       = kernel->data;
    દશાંશ * dst_data       = (દશાંશ *) dst->data;

    અચલ int64_t knl_n_per_channel = knl_w * knl_h * knl_d;
    અચલ int64_t knl_n_total       = knl_n_per_channel * c;
    અચલ int64_t patch_total       = n * dst_w * dst_h * dst_d;

    અચલ int64_t space_per_patch   = knl_n_total * traits->type_size + oc * કદ(દશાંશ);
    અચલ int64_t batch_size        = params->wsize / space_per_patch;
    અચલ int64_t patches_per_batch = batch_size > 8 ? (batch_size / 8) * 8 : batch_size;
    અચલ int64_t batch_n           = (patch_total + patches_per_batch - 1) / patches_per_batch;

    GGML_ASSERT(patches_per_batch > 0 && batch_size >= 1);

    શૂન્ય * tmp = params->wdata;

    માટે (int64_t batch_i = 0; batch_i < batch_n; ++batch_i) {
        અચલ int64_t patch_start_batch = batch_i * patches_per_batch;
        અચલ int64_t patch_end_batch   = std::min(patch_start_batch + patches_per_batch, patch_total);
        અચલ int64_t patch_n_in_batch  = patch_end_batch - patch_start_batch;

        અચલ int64_t patch_per_thread  = (patch_n_in_batch + params->nth - 1) / params->nth;
        અચલ int64_t patch_start       = patch_start_batch + params->ith * patch_per_thread;
        અચલ int64_t patch_end         = std::min(patch_start + patch_per_thread, patch_end_batch);

        માટે (int64_t p = patch_start; p < patch_end; ++p) {
            અચલ int64_t p_in_batch = p % (dst_w * dst_h * dst_d);
            અચલ int64_t p_in_depth = p_in_batch % (dst_w * dst_h);
            અચલ int64_t batch_idx  = p / (dst_w * dst_h * dst_d);
            અચલ int64_t dst_z      = p_in_batch / (dst_w * dst_h);
            અચલ int64_t dst_y      = p_in_depth / dst_w;
            અચલ int64_t dst_x      = p_in_depth % dst_w;

            અક્ષર * dst_row = (અક્ષર *) tmp + (p % patches_per_batch) * knl_n_total * traits->type_size;

            માટે (int64_t ic = 0; ic < c; ++ic) {
                માટે (int64_t kz = 0; kz < knl_d; ++kz) {
                    માટે (int64_t ky = 0; ky < knl_h; ++ky) {
                        માટે (int64_t kx = 0; kx < knl_w; ++kx) {
                            અચલ int64_t sz = dst_z * s2 + kz * d2 - p2;
                            અચલ int64_t sy = dst_y * s1 + ky * d1 - p1;
                            અચલ int64_t sx = dst_x * s0 + kx * d0 - p0;

                            int64_t dst_idx = ic * knl_n_per_channel + kz * (knl_h * knl_w) + ky * knl_w + kx;

                            દશાંશ src_val;
                            જો (sz < 0 || sz >= src_d || sy < 0 || sy >= src_h || sx < 0 || sx >= src_w) {
                                src_val = 0.0f;
                            } નહીંતર {
                                અચલ int64_t cn_idx = batch_idx * c + ic;
                                અચલ દશાંશ * src_ptr = (અચલ દશાંશ *)((અચલ અક્ષર *)src_data + sx*src->nb[0] + sy*src->nb[1] + sz*src->nb[2] + cn_idx*src->nb[3]);
                                src_val = *src_ptr;
                            }

                            અક્ષર * element_ptr = dst_row + dst_idx * traits->type_size;
                            જો (kernel_type == GGML_TYPE_F32) {
                                *(દશાંશ *)element_ptr = src_val;
                            } નહીંતર જો (kernel_type == GGML_TYPE_F16) {
                                *(ggml_fp16_t *)element_ptr = GGML_CPU_FP32_TO_FP16(src_val);
                            }
                        }
                    }
                }
            }
        }

        ggml_barrier(params->threadpool);

        દશાંશ * gemm_output = (દશાંશ *) ((અક્ષર *) tmp + patches_per_batch * knl_n_total * traits->type_size);
        ggml_call_mul_mat(kernel_type, params, patch_n_in_batch, oc, knl_n_total, tmp, knl_data, gemm_output);

        ggml_barrier(params->threadpool);

        અચલ int64_t permute_per_thread = (patch_n_in_batch + params->nth - 1) / params->nth;
        અચલ int64_t permute_start = params->ith * permute_per_thread;
        અચલ int64_t permute_end = std::min(permute_start + permute_per_thread, patch_n_in_batch);

        માટે (int64_t i = permute_start; i < permute_end; ++i) {
            અચલ int64_t p = patch_start_batch + i;
            અચલ int64_t p_in_batch = p % (dst_w * dst_h * dst_d);
            અચલ int64_t p_in_depth = p_in_batch % (dst_w * dst_h);
            અચલ int64_t batch_idx  = p / (dst_w * dst_h * dst_d);
            અચલ int64_t dst_z      = p_in_batch / (dst_w * dst_h);
            અચલ int64_t dst_y      = p_in_depth / dst_w;
            અચલ int64_t dst_x      = p_in_depth % dst_w;

            માટે (int64_t ioc = 0; ioc < oc; ++ioc) {
                અચલ દશાંશ value = gemm_output[i * oc + ioc];
                અચલ int64_t ocn_idx = batch_idx * oc + ioc;
                દશાંશ * dst_ptr = (દશાંશ *)((અક્ષર *)dst_data + dst_x*dst->nb[0] + dst_y*dst->nb[1] + dst_z*dst->nb[2] + ocn_idx*dst->nb[3]);
                *dst_ptr = value;
            }
        }
    }
}

શૂન્ય ggml_compute_forward_conv_3d(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {
    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];
    ggml_compute_forward_conv_3d_impl(params, src0, src1, dst, src0->type);
}

template <typename kernel_t>
સ્થિર શૂન્ય ggml_compute_forward_conv_transpose_2d_impl(
    અચલ ggml_compute_params * params,
          ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_દ્વિઅંકી_OP_LOCALS

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક nk = ne00*ne01*ne02*ne03;

    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == કદ(દશાંશ));

    જો (ith == 0) {
        memset(params->wdata, 0, params->wsize);

        // permute kernel data (src0) from (Kw x Kh x Cout x Cin) to (Cin x Kw x Kh x Cout)
        {
            kernel_t * અચલ wdata = (kernel_t *) params->wdata + 0;

            માટે (int64_t i03 = 0; i03 < ne03; i03++) {
                માટે (int64_t i02 = 0; i02 < ne02; i02++) {
                    અચલ kernel_t * અચલ src = (kernel_t *)((અક્ષર *) src0->data + i03*nb03 + i02*nb02);
                    kernel_t * dst_data = wdata + i02*ne01*ne00*ne03;
                    માટે (int64_t i01 = 0; i01 < ne01; i01++) {
                        માટે (int64_t i00 = 0; i00 < ne00; i00++) {
                            dst_data[i01*ne00*ne03 + i00*ne03 + i03] = src[i01 * ne00 + i00];
                        }
                    }
                }
            }
        }

        // permute source data (src1) from (Sw x Sh x Cin) to (Cin x Sw x Sh), for all batches
        {
            kernel_t * અચલ wdata = (kernel_t *) params->wdata + nk;
            માટે (પૂર્ણાંક i13 = 0; i13 < ne13; i13++) {
                kernel_t * અચલ wdata_b = wdata + i13*ne10*ne11*ne12;
                માટે (પૂર્ણાંક i12 = 0; i12 < ne12; i12++) {
                    માટે (પૂર્ણાંક i11 = 0; i11 < ne11; i11++) {
                        અચલ દશાંશ * અચલ src = (દશાંશ *)((અક્ષર *) src1->data + i13*nb13 + i12*nb12 + i11*nb11);
                        kernel_t * dst_data = wdata_b + i11*ne10*ne12;
                        માટે (પૂર્ણાંક i10 = 0; i10 < ne10; i10++) {
                            જો constexpr (std::is_same_v<kernel_t, ggml_fp16_t>) {
                                dst_data[i10*ne12 + i12] = GGML_CPU_FP32_TO_FP16(src[i10]);
                            } નહીંતર {
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

    અચલ int32_t stride = ggml_get_op_params_i32(dst, 0);

    // total patches in dst
    અચલ પૂર્ણાંક np = ne2;

    // patches per thread
    અચલ પૂર્ણાંક dp = (np + nth - 1)/nth;

    // patch range for this thread
    અચલ પૂર્ણાંક ip0 = dp*ith;
    અચલ પૂર્ણાંક ip1 = MIN(ip0 + dp, np);

    kernel_t * અચલ wdata = (kernel_t *) params->wdata + 0;
    kernel_t * અચલ wdata_src = wdata + nk;

    માટે (પૂર્ણાંક i2 = ip0; i2 < ip1; i2++) { // Cout
        kernel_t * wdata_kernel = wdata + i2*ne01*ne00*ne03;
        માટે (પૂર્ણાંક i3 = 0; i3 < ne3; i3++) { // batch
            દશાંશ * dst_data = (દશાંશ *)((અક્ષર *) dst->data + i3*nb3 + i2*nb2);
            kernel_t * wdata_src_b = wdata_src + i3*ne10*ne11*ne12;
            માટે (પૂર્ણાંક i11 = 0; i11 < ne11; i11++) {
                માટે (પૂર્ણાંક i10 = 0; i10 < ne10; i10++) {
                    અચલ પૂર્ણાંક i1n = i11*ne10*ne12 + i10*ne12;
                    માટે (પૂર્ણાંક i01 = 0; i01 < ne01; i01++) {
                        માટે (પૂર્ણાંક i00 = 0; i00 < ne00; i00++) {
                            દશાંશ v = 0;
                            જો constexpr (std::is_same_v<kernel_t, ggml_fp16_t>) {
                                ggml_vec_dot_f16(ne03, &v, 0,
                                        wdata_src_b + i1n, 0,
                                        wdata_kernel + i01*ne00*ne03 + i00*ne03, 0, 1);
                            } નહીંતર {
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

શૂન્ય ggml_compute_forward_conv_transpose_2d(
        અચલ ggml_compute_params * params,
              ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F16:
            {
                ggml_compute_forward_conv_transpose_2d_impl<ggml_fp16_t>(params, dst);
            } તોડો;
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_conv_transpose_2d_impl<દશાંશ>(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_conv_2d_dw

સંરચના ggml_conv_2d_dw_params {
    int64_t channels;
    int64_t batch;
    int64_t src_w;
    int64_t src_h;
    int64_t dst_w;
    int64_t dst_h;
    int64_t knl_w;
    int64_t knl_h;
    પૂર્ણાંક stride_x;
    પૂર્ણાંક stride_y;
    પૂર્ણાંક pad_x;
    પૂર્ણાંક pad_y;
    પૂર્ણાંક dilation_x;
    પૂર્ણાંક dilation_y;
};

સ્થિર અંતઃરેખ દશાંશ ggml_conv_2d_dw_knl_f32(અચલ અક્ષર * data, int64_t i, ggml_type type) {
    જો (type == GGML_TYPE_F16) {
        પાછા GGML_FP16_TO_FP32(((અચલ ggml_fp16_t *)data)[i]);
    }
    પાછા ((અચલ દશાંશ *)data)[i];
}

સ્થિર શૂન્ય ggml_compute_forward_conv_2d_dw_cwhn(
        અચલ ggml_compute_params * params,
        અચલ ggml_tensor * src,
        અચલ ggml_tensor * kernel,
        ggml_tensor * dst,
        અચલ ggml_conv_2d_dw_params & p) {

    અચલ int64_t c = p.channels;
    અચલ અક્ષર * knl_data = (અચલ અક્ષર *)kernel->data;
    અચલ ggml_type knl_type = kernel->type;

    અચલ int64_t rows_total = p.dst_h * p.batch;
    અચલ int64_t rows_per_thread = (rows_total + params->nth - 1) / params->nth;
    અચલ int64_t row_start = params->ith * rows_per_thread;
    અચલ int64_t row_end = MIN(row_start + rows_per_thread, rows_total);

#ifdef GGML_SIMD
    int64_t c_pkg_end = 0;
    int64_t pkg_size = GGML_F32_EPR;
    જો (knl_type == GGML_TYPE_F32) {
    #જો defined(__ARM_FEATURE_SVE)
        pkg_size = svcntw();
    #નહીંતર
        pkg_size = GGML_F32_EPR;
    #endif
        c_pkg_end = (c / pkg_size) * pkg_size;
    }
#નહીંતર
    અચલ int64_t c_pkg_end = 0;
#endif

    માટે (int64_t row = row_start; row < row_end; ++row) {
        અચલ int64_t dst_y = row % p.dst_h;
        અચલ દશાંશ * src_data = (અચલ દશાંશ *)src->data + (row / p.dst_h) * p.src_w * p.src_h * c;
        માટે (int64_t dst_x = 0; dst_x < p.dst_w; ++dst_x) {
            દશાંશ * dst_data = (દશાંશ *)dst->data + (row * p.dst_w + dst_x) * c;
            અચલ int64_t src_y_base = dst_y * p.stride_y - p.pad_y;
            અચલ int64_t src_x_base = dst_x * p.stride_x - p.pad_x;

#ifdef GGML_SIMD
            માટે (int64_t c_i = 0; c_i < c_pkg_end; c_i += pkg_size) {
                GGML_F32_VEC sum = GGML_F32_VEC_ZERO;
                માટે (int64_t knl_y = 0; knl_y < p.knl_h; ++knl_y) {
                    અચલ int64_t src_y = src_y_base + knl_y * p.dilation_y;
                    જો (src_y < 0 || src_y >= p.src_h) {
                        ચાલુ;
                    }
                    માટે (int64_t knl_x = 0; knl_x < p.knl_w; ++knl_x) {
                        અચલ int64_t src_x = src_x_base + knl_x * p.dilation_x;
                        જો (src_x < 0 || src_x >= p.src_w) {
                            ચાલુ;
                        }
                        અચલ દશાંશ * kp = (અચલ દશાંશ *)knl_data + (knl_y * p.knl_w + knl_x) * c + c_i;
                        GGML_F32_VEC k = GGML_F32_VEC_LOAD(kp);
                        GGML_F32_VEC s = GGML_F32_VEC_LOAD(src_data + (src_y * p.src_w + src_x) * c + c_i);
                        sum = GGML_F32_VEC_FMA(sum, k, s);
                    }
                }
                GGML_F32_VEC_STORE(dst_data + c_i, sum);
            }
#endif
            માટે (int64_t c_i = c_pkg_end; c_i < c; ++c_i) {
                દશાંશ sum = 0.0f;
                માટે (int64_t knl_y = 0; knl_y < p.knl_h; ++knl_y) {
                    અચલ int64_t src_y = src_y_base + knl_y * p.dilation_y;
                    જો (src_y < 0 || src_y >= p.src_h) {
                        ચાલુ;
                    }
                    માટે (int64_t knl_x = 0; knl_x < p.knl_w; ++knl_x) {
                        અચલ int64_t src_x = src_x_base + knl_x * p.dilation_x;
                        જો (src_x < 0 || src_x >= p.src_w) {
                            ચાલુ;
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

સ્થિર શૂન્ય ggml_compute_forward_conv_2d_dw_whcn(
        અચલ ggml_compute_params * params,
        અચલ ggml_tensor * src,
        અચલ ggml_tensor * kernel,
        ggml_tensor * dst,
        અચલ ggml_conv_2d_dw_params & p) {

    અચલ int64_t n = p.channels * p.batch;
    અચલ int64_t per_thread = (n + params->nth - 1) / params->nth;
    અચલ int64_t start = params->ith * per_thread;
    અચલ int64_t end = MIN(start + per_thread, n);
    અચલ અક્ષર * knl_base = (અચલ અક્ષર *)kernel->data;
    અચલ ggml_type knl_type = kernel->type;

    માટે (int64_t i = start; i < end; ++i) {
        અચલ int64_t knl_offset = (i % p.channels) * p.knl_w * p.knl_h;
        અચલ દશાંશ * src_data = (અચલ દશાંશ *)src->data + i * p.src_w * p.src_h;
        દશાંશ * dst_data = (દશાંશ *)dst->data + i * p.dst_w * p.dst_h;

        માટે (int64_t dst_y = 0; dst_y < p.dst_h; ++dst_y) {
            માટે (int64_t dst_x = 0; dst_x < p.dst_w; ++dst_x) {

                દશાંશ sum = 0.0f;
                માટે (int64_t knl_y = 0; knl_y < p.knl_h; ++knl_y) {
                    અચલ int64_t src_y = dst_y * p.stride_y + knl_y * p.dilation_y - p.pad_y;
                    જો (src_y < 0 || src_y >= p.src_h) {
                        ચાલુ;
                    }
                    માટે (int64_t knl_x = 0; knl_x < p.knl_w; ++knl_x) {
                        અચલ int64_t src_x = dst_x * p.stride_x + knl_x * p.dilation_x - p.pad_x;
                        જો (src_x < 0 || src_x >= p.src_w) {
                            ચાલુ;
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

શૂન્ય ggml_compute_forward_conv_2d_dw(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * kernel = dst->src[0];
    અચલ ggml_tensor * src = dst->src[1];
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

    જો (ggml_is_contiguous(src)) {
        ggml_compute_forward_conv_2d_dw_whcn(params, src, kernel, dst, p);
    } નહીંતર જો (ggml_is_contiguous_channels(src)) {
        GGML_ASSERT(kernel->nb[0] >= kernel->nb[2] && kernel->nb[1] >= kernel->nb[0]);
        ggml_compute_forward_conv_2d_dw_cwhn(params, src, kernel, dst, p);
    } નહીંતર {
        GGML_ABORT("non-contiguous memory layout not supported");
    }
}

// ggml_compute_forward_pool_1d_ksp
સ્થિર શૂન્ય ggml_compute_forward_pool_1d_ksp(
        અચલ ggml_compute_params * params,
        અચલ ggml_op_pool op,
        અચલ પૂર્ણાંક k,
        અચલ પૂર્ણાંક s,
        અચલ પૂર્ણાંક p,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src = dst->src[0];

    પ્રતિજ્ઞા(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16);

    જો (params->ith != 0) {
        પાછા;
    }

    અચલ int64_t IW = src->ne[0];
    અચલ int64_t OW = dst->ne[0];

    અચલ int64_t nr = ggml_nrows(src);

    માટે (int64_t ir = 0; ir < nr; ++ir) {
        અચલ અક્ષર * srow_bytes =            (અચલ અક્ષર *) src->data + ir * src->nb[1];
        દશાંશ      * drow       = (દશાંશ *) ((      અક્ષર *) dst->data + ir * dst->nb[1]);

        માટે (int64_t ow = 0; ow < OW; ++ow) {
            દશાંશ res = 0;
            પસંદગી (op) {
                સ્થિતિ GGML_OP_POOL_AVG: res = 0.0f;     તોડો;
                સ્થિતિ GGML_OP_POOL_MAX: res = -FLT_MAX; તોડો;
                સ્થિતિ GGML_OP_POOL_ગણતરી: GGML_ABORT("fatal error");
            }

            પૂર્ણાંક count = 0;
            અચલ પૂર્ણાંક base = (પૂર્ણાંક) ow * s - p;

            માટે (પૂર્ણાંક ki = 0; ki < k; ++ki) {
                અચલ પૂર્ણાંક j = base + ki;
                જો (j < 0 || j >= (પૂર્ણાંક) IW) {
                    ચાલુ;
                }

                દશાંશ v;
                જો (src->type == GGML_TYPE_F32) {
                    v = ((અચલ દશાંશ *) srow_bytes)[j];
                } નહીંતર {
                    v = GGML_CPU_FP16_TO_FP32(((અચલ ggml_fp16_t *) srow_bytes)[j]);
                }

                પસંદગી (op) {
                    સ્થિતિ GGML_OP_POOL_AVG: res += v;                તોડો;
                    સ્થિતિ GGML_OP_POOL_MAX: res =  std::max(v, res); તોડો;
                    સ્થિતિ GGML_OP_POOL_ગણતરી: GGML_ABORT("fatal error");
                }

                ++count;
            }

            પસંદગી (op) {
                સ્થિતિ GGML_OP_POOL_AVG: res = (count > 0) ? (res / count) : 0.0f; તોડો;
                સ્થિતિ GGML_OP_POOL_MAX:                                           તોડો;
                સ્થિતિ GGML_OP_POOL_ગણતરી: GGML_ABORT("fatal error");
            }

            drow[ow] = res;
        }
    }
}

// ggml_compute_forward_pool_1d

શૂન્ય ggml_compute_forward_pool_1d(
        અચલ ggml_compute_params * params,
              ggml_tensor * dst) {

    અચલ int32_t * opts = (અચલ int32_t *)dst->op_params;
    ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    અચલ પૂર્ણાંક k0 = opts[1];
    અચલ પૂર્ણાંક s0 = opts[2];
    અચલ પૂર્ણાંક p0 = opts[3];

    ggml_compute_forward_pool_1d_ksp(params, op, k0, s0, p0, dst);
}

// ggml_compute_forward_pool_2d

શૂન્ય ggml_compute_forward_pool_2d(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src = dst->src[0];

    પ્રતિજ્ઞા(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16);

    જો (params->ith != 0) {
        પાછા;
    }

    અચલ int32_t * opts = (અચલ int32_t *)dst->op_params;

    ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    અચલ પૂર્ણાંક k0 = opts[1];
    અચલ પૂર્ણાંક k1 = opts[2];
    અચલ પૂર્ણાંક s0 = opts[3];
    અચલ પૂર્ણાંક s1 = opts[4];
    અચલ પૂર્ણાંક p0 = opts[5];
    અચલ પૂર્ણાંક p1 = opts[6];
    અચલ અક્ષર * cdata = (અચલ અક્ષર*)src->data;
    અચલ અક્ષર * અચલ data_end = cdata + ggml_nbytes(src);

    અચલ int64_t px = dst->ne[0];
    અચલ int64_t py = dst->ne[1];
    અચલ int64_t pa = px * py;

    દશાંશ * dplane = (દશાંશ *)dst->data;

    અચલ પૂર્ણાંક ka = k0 * k1;
    અચલ પૂર્ણાંક offset0 = -p0;
    અચલ પૂર્ણાંક offset1 = -p1;

    જ્યારે (cdata < data_end) {
        માટે (પૂર્ણાંક oy = 0; oy < py; ++oy) {
            દશાંશ * અચલ drow = dplane + oy * px;
            દશાંશ * અચલ out  = drow;

            માટે (પૂર્ણાંક ox = 0; ox < px; ++ox) {
                દશાંશ res = 0;
                પસંદગી (op) {
                    સ્થિતિ GGML_OP_POOL_AVG: res = 0;        તોડો;
                    સ્થિતિ GGML_OP_POOL_MAX: res = -FLT_MAX; તોડો;
                    સ્થિતિ GGML_OP_POOL_ગણતરી: GGML_ABORT("fatal error");
                }

                અચલ પૂર્ણાંક ix = offset0 + ox * s0;
                અચલ પૂર્ણાંક iy = offset1 + oy * s1;

                માટે (પૂર્ણાંક ky = 0; ky < k1; ++ky) {
                    જો (iy + ky < 0 || iy + ky >= src->ne[1]) {
                        ચાલુ;
                    }

                    અચલ શૂન્ય * srow = (અચલ શૂન્ય *)(cdata + src->nb[1] * (iy + ky));
                    માટે (પૂર્ણાંક kx = 0; kx < k0; ++kx) {
                        પૂર્ણાંક j = ix + kx;
                        જો (j < 0 || j >= src->ne[0]) {
                            ચાલુ;
                        }

                        અચલ દશાંશ srow_j = (src->type == GGML_TYPE_F32) ? ((અચલ દશાંશ*)srow)[j] : GGML_CPU_FP16_TO_FP32(((અચલ ggml_fp16_t*)srow)[j]);
                        પસંદગી (op) {
                            સ્થિતિ GGML_OP_POOL_AVG: res += srow_j;                તોડો;
                            સ્થિતિ GGML_OP_POOL_MAX: res =  std::max(srow_j, res); તોડો;
                            સ્થિતિ GGML_OP_POOL_ગણતરી:               GGML_ABORT("fatal error");
                        }
                    }
                }
                પસંદગી (op) {
                    સ્થિતિ GGML_OP_POOL_AVG:           res /= ka; તોડો;
                    સ્થિતિ GGML_OP_POOL_MAX:                      તોડો;
                    સ્થિતિ GGML_OP_POOL_ગણતરી: GGML_ABORT("fatal error");
                }

                out[ox] = res;
            }
        }

        cdata  += src->nb[2];
        dplane += pa;
    }
}

// ggml_compute_forward_pool_2d_back

શૂન્ય ggml_compute_forward_pool_2d_back(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src  = dst->src[0];
    અચલ ggml_tensor * dstf = dst->src[1]; // forward tensor of dst

    પ્રતિજ્ઞા(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);

    જો (params->ith != 0) {
        પાછા;
    }

    અચલ int32_t * opts = (અચલ int32_t *)dst->op_params;
    ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    અચલ પૂર્ણાંક k0 = opts[1];
    અચલ પૂર્ણાંક k1 = opts[2];
    અચલ પૂર્ણાંક s0 = opts[3];
    અચલ પૂર્ણાંક s1 = opts[4];
    અચલ પૂર્ણાંક p0 = opts[5];
    અચલ પૂર્ણાંક p1 = opts[6];

    અક્ષર       * cdata  = (અક્ષર       *) dst->data;
    અચલ અક્ષર * cdataf = (અચલ અક્ષર *) dstf->data;
    અચલ અક્ષર * અચલ data_end = cdata + ggml_nbytes(dst);

    GGML_ASSERT(params->ith == 0);
    memset(cdata, 0, ggml_nbytes(dst));

    અચલ int64_t px = src->ne[0];
    અચલ int64_t py = src->ne[1];
    અચલ int64_t pa = px * py;

    અચલ દશાંશ * splane = (અચલ દશાંશ *) src->data;

    અચલ પૂર્ણાંક ka = k0 * k1;
    અચલ પૂર્ણાંક offset0 = -p0;
    અચલ પૂર્ણાંક offset1 = -p1;

    જ્યારે (cdata < data_end) {
        માટે (પૂર્ણાંક oy = 0; oy < py; ++oy) {
            અચલ દશાંશ * અચલ srow = splane + oy * px;
            માટે (પૂર્ણાંક ox = 0; ox < px; ++ox) {
                અચલ દશાંશ grad0 = srow[ox];

                અચલ પૂર્ણાંક ix = offset0 + ox * s0;
                અચલ પૂર્ણાંક iy = offset1 + oy * s1;

                જો (op == GGML_OP_POOL_MAX) {
                    દશાંશ maxval = -FLT_MAX;
                    પૂર્ણાંક kxmax = -1;
                    પૂર્ણાંક kymax = -1;

                    માટે (પૂર્ણાંક ky = 0; ky < k1; ++ky) {
                        જો (iy + ky < 0 || iy + ky >= dst->ne[1]) {
                            ચાલુ;
                        }
                        અચલ શૂન્ય * drowf = (અચલ શૂન્ય *)(cdataf + dst->nb[1] * (iy + ky));
                        માટે (પૂર્ણાંક kx = 0; kx < k0; ++kx) {
                            પૂર્ણાંક j = ix + kx;
                            જો (j < 0 || j >= dst->ne[0]) {
                                ચાલુ;
                            }

                            અચલ દશાંશ val = dst->type == GGML_TYPE_F32 ?
                                ((અચલ દશાંશ *) drowf)[j] : GGML_CPU_FP16_TO_FP32(((અચલ ggml_fp16_t *) drowf)[j]);
                            જો (val <= maxval) {
                                ચાલુ;
                            }

                            maxval = val;
                            kxmax = kx;
                            kymax = ky;
                        }
                    }

                    જો (kxmax == -1 || kymax == -1) {
                        ચાલુ;
                    }

                    શૂન્ય * drow = (શૂન્ય *)(cdata + dst->nb[1] * (iy + kymax));
                    અચલ પૂર્ણાંક j = ix + kxmax;
                    જો (dst->type == GGML_TYPE_F32) {
                        ((દશાંશ *) drow)[j] += grad0;
                    } નહીંતર {
                        ((ggml_fp16_t *) drow)[j] = GGML_CPU_FP32_TO_FP16(grad0 + GGML_CPU_FP16_TO_FP32(((અચલ ggml_fp16_t *) drow)[j]));
                    }
                } નહીંતર જો (op == GGML_OP_POOL_AVG) {
                    અચલ દશાંશ grad = grad0 / ka;

                    માટે (પૂર્ણાંક ky = 0; ky < k1; ++ky) {
                        જો (iy + ky < 0 || iy + ky >= dst->ne[1]) {
                            ચાલુ;
                        }
                        શૂન્ય * drow = (શૂન્ય *)(cdata + dst->nb[1] * (iy + ky));
                        માટે (પૂર્ણાંક kx = 0; kx < k0; ++kx) {
                            પૂર્ણાંક j = ix + kx;
                            જો (j < 0 || j >= dst->ne[0]) {
                                ચાલુ;
                            }

                            જો (dst->type == GGML_TYPE_F32) {
                                ((દશાંશ *) drow)[j] += grad;
                            } નહીંતર {
                                ((ggml_fp16_t *) drow)[j] += GGML_CPU_FP32_TO_FP16(grad);
                            }
                        }
                    }
                } નહીંતર {
                    GGML_ASSERT(ખોટું);
                }
            }
        }

        cdata  += dst->nb[2];
        cdataf += dst->nb[2];
        splane += pa;
    }
}

// ggml_compute_forward_upscale

સ્થિર શૂન્ય ggml_compute_forward_upscale_f32(
    અચલ ggml_compute_params * params,
    ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    દશાંશ sf0 = (દશાંશ)ne0/src0->ne[0];
    દશાંશ sf1 = (દશાંશ)ne1/src0->ne[1];
    દશાંશ sf2 = (દશાંશ)ne2/src0->ne[2];
    દશાંશ sf3 = (દશાંશ)ne3/src0->ne[3];
    દશાંશ pixel_offset = 0.5f;

    અચલ int32_t mode_flags = ggml_get_op_params_i32(dst, 0);
    અચલ ggml_scale_mode mode = (ggml_scale_mode) (mode_flags & 0xFF);

    જો (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        pixel_offset = 0.0f;
        sf0 = ne0 > 1 && ne00 > 1 ? (દશાંશ)(ne0 - 1) / (ne00 - 1) : sf0;
        sf1 = ne1 > 1 && ne01 > 1 ? (દશાંશ)(ne1 - 1) / (ne01 - 1) : sf1;
    }

    જો (mode == GGML_SCALE_MODE_NEAREST) {
        માટે (int64_t i3 = 0; i3 < ne3; i3++) {
            અચલ int64_t i03 = i3 / sf3;
            માટે (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                અચલ int64_t i02 = i2 / sf2;
                માટે (int64_t i1 = 0; i1 < ne1; i1++) {
                    અચલ int64_t i01 = i1 / sf1;
                    માટે (int64_t i0 = 0; i0 < ne0; i0++) {
                        અચલ int64_t i00 = i0 / sf0;

                        અચલ દશાંશ * x = (દશાંશ *)((અક્ષર *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              દશાંશ * y = (દશાંશ *)((અક્ષર *)  dst->data +  i0*nb0  +  i1*nb1  +  i2*nb2  +  i3*nb3);

                        *y = *x;
                    }
                }
            }
        }
    } નહીંતર જો (mode == GGML_SCALE_MODE_BILINEAR && (mode_flags & GGML_SCALE_FLAG_ANTIALIAS)) {
        // Similar to F.interpolate(..., mode="bilinear", align_corners=False, antialias=True)
        // https://github.com/pytorch/pytorch/blob/8871ff29b743948d1225389d5b7068f37b22750b/aten/src/ATen/native/cpu/UpSampleKernel.cpp
        auto triangle_filter = [](દશાંશ x) -> દશાંશ {
            પાછા std::max(1.0f - fabsf(x), 0.0f);
        };

        // support and invscale, minimum 1 pixel for bilinear
        અચલ દશાંશ support1  = std::max(1.0f, 1.0f / sf1);
        અચલ દશાંશ invscale1 = 1.0f / support1;
        અચલ દશાંશ support0  = std::max(1.0f, 1.0f / sf0);
        અચલ દશાંશ invscale0 = 1.0f / support0;

        માટે (int64_t i3 = 0; i3 < ne3; i3++) {
            અચલ int64_t i03 = i3 / sf3;
            માટે (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                અચલ int64_t i02 = i2 / sf2;
                માટે (int64_t i1 = 0; i1 < ne1; i1++) {
                    અચલ દશાંશ y = ((દશાંશ) i1 + pixel_offset) / sf1;
                    માટે (int64_t i0 = 0; i0 < ne0; i0++) {
                        અચલ દશાંશ x = ((દશાંશ) i0 + pixel_offset) / sf0;

                        // the range of source pixels that contribute
                        અચલ int64_t x_min = std::max<int64_t>(x - support0 + pixel_offset, 0);
                        અચલ int64_t x_max = std::min<int64_t>(x + support0 + pixel_offset, ne00);
                        અચલ int64_t y_min = std::max<int64_t>(y - support1 + pixel_offset, 0);
                        અચલ int64_t y_max = std::min<int64_t>(y + support1 + pixel_offset, ne01);

                        // bilinear filter with antialiasing
                        દશાંશ val = 0.0f;
                        દશાંશ total_weight = 0.0f;

                        માટે (int64_t sy = y_min; sy < y_max; sy++) {
                            અચલ દશાંશ weight_y = triangle_filter((sy - y + pixel_offset) * invscale1);

                            માટે (int64_t sx = x_min; sx < x_max; sx++) {
                                અચલ દશાંશ weight_x = triangle_filter((sx - x + pixel_offset) * invscale0);
                                અચલ દશાંશ weight = weight_x * weight_y;

                                જો (weight <= 0.0f) {
                                    ચાલુ;
                                }

                                અચલ દશાંશ pixel = *(અચલ દશાંશ *)((અચલ અક્ષર *)src0->data + sx*nb00 + sy*nb01 + i02*nb02 + i03*nb03);
                                val += pixel * weight;
                                total_weight += weight;
                            }
                        }

                        જો (total_weight > 0.0f) {
                            val /= total_weight;
                        }

                        દશાંશ * dst_ptr = (દશાંશ *)((અક્ષર *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);
                        *dst_ptr = val;
                    }
                }
            }
        }
    } નહીંતર જો (mode == GGML_SCALE_MODE_BILINEAR) {
        માટે (int64_t i3 = 0; i3 < ne3; i3++) {
            અચલ int64_t i03 = i3 / sf3;
            માટે (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                અચલ int64_t i02 = i2 / sf2;
                માટે (int64_t i1 = 0; i1 < ne1; i1++) {
                    અચલ દશાંશ y = ((દશાંશ)i1 + pixel_offset) / sf1 - pixel_offset;
                    int64_t y0 = (int64_t)floorf(y);
                    int64_t y1 = y0 + 1;

                    y0 = std::max(int64_t(0), std::min(y0, ne01 - 1));
                    y1 = std::max(int64_t(0), std::min(y1, ne01 - 1));

                    દશાંશ dy = y - (દશાંશ)y0;
                    dy = std::max(0.0f, std::min(dy, 1.0f));

                    માટે (int64_t i0 = 0; i0 < ne0; i0++) {
                        અચલ દશાંશ x = ((દશાંશ)i0 + pixel_offset) / sf0 - pixel_offset;
                        int64_t x0 = (int64_t)floorf(x);
                        int64_t x1 = x0 + 1;

                        x0 = std::max(int64_t(0), std::min(x0, ne00 - 1));
                        x1 = std::max(int64_t(0), std::min(x1, ne00 - 1));

                        દશાંશ dx = x - (દશાંશ)x0;
                        dx = std::max(0.0f, std::min(dx, 1.0f));

                        // fetch the four surrounding pixel values and interpolate
                        અચલ દશાંશ a = *(અચલ દશાંશ *)((અચલ અક્ષર *)src0->data + x0*nb00 + y0*nb01 + i02*nb02 + i03*nb03);
                        અચલ દશાંશ b = *(અચલ દશાંશ *)((અચલ અક્ષર *)src0->data + x1*nb00 + y0*nb01 + i02*nb02 + i03*nb03);
                        અચલ દશાંશ c = *(અચલ દશાંશ *)((અચલ અક્ષર *)src0->data + x0*nb00 + y1*nb01 + i02*nb02 + i03*nb03);
                        અચલ દશાંશ d = *(અચલ દશાંશ *)((અચલ અક્ષર *)src0->data + x1*nb00 + y1*nb01 + i02*nb02 + i03*nb03);

                        અચલ દશાંશ val = a*(1 - dx)*(1 - dy) + b*dx*(1 - dy) + c*(1 - dx)*dy + d*dx*dy;

                        દશાંશ * y_dst = (દશાંશ *)((અક્ષર *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);
                        *y_dst = val;
                    }
                }
            }
        }
    } નહીંતર જો (mode == GGML_SCALE_MODE_BICUBIC) {
        // https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
        અચલ દશાંશ a = -0.75f; // use alpha = -0.75 (same as PyTorch)
        auto weight1 = [a](દશાંશ x) { પાછા ((a + 2) * x - (a + 3)) * x * x + 1; };
        auto weight2 = [a](દશાંશ x) { પાછા ((a * x - 5 * a) * x + 8 * a) * x - 4 * a; };
        auto bicubic = [=](દશાંશ p0, દશાંશ p1, દશાંશ p2, દશાંશ p3, દશાંશ x) {
            અચલ દશાંશ w0 = weight2(x + 1);
            અચલ દશાંશ w1 = weight1(x + 0);
            અચલ દશાંશ w2 = weight1(1 - x);
            અચલ દશાંશ w3 = weight2(2 - x);
            પાછા p0*w0 + p1*w1 + p2*w2 + p3*w3;
        };

        માટે (int64_t i3 = 0; i3 < ne3; i3++) {
            અચલ int64_t i03 = i3 / sf3;
            માટે (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                અચલ int64_t i02 = i2 / sf2;
                માટે (int64_t i1 = 0; i1 < ne1; i1++) {
                    અચલ દશાંશ y = ((દશાંશ)i1 + pixel_offset) / sf1 - pixel_offset;
                    અચલ int64_t y0 = (int64_t)floorf(y);
                    અચલ દશાંશ dy = y - (દશાંશ)y0;

                    માટે (int64_t i0 = 0; i0 < ne0; i0++) {
                        અચલ દશાંશ x = ((દશાંશ)i0 + pixel_offset) / sf0 - pixel_offset;
                        અચલ int64_t x0 = (int64_t)floorf(x);
                        અચલ દશાંશ dx = x - (દશાંશ)x0;

                        auto p = [=](int64_t x_off, int64_t y_off) -> દશાંશ {
                            int64_t i00 = std::max(int64_t(0), std::min(x0 + x_off, ne00 - 1));
                            int64_t i01 = std::max(int64_t(0), std::min(y0 + y_off, ne01 - 1));
                            પાછા *(અચલ દશાંશ *)((અચલ અક્ષર *)src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                        };

                        અચલ દશાંશ val = bicubic(
                            bicubic(p(-1,-1), p(0,-1), p(1,-1), p(2,-1), dx),
                            bicubic(p(-1, 0), p(0, 0), p(1, 0), p(2, 0), dx),
                            bicubic(p(-1, 1), p(0, 1), p(1, 1), p(2, 1), dx),
                            bicubic(p(-1, 2), p(0, 2), p(1, 2), p(2, 2), dx), dy);

                        દશાંશ * y_dst = (દશાંશ *)((અક્ષર *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);
                        *y_dst = val;
                    }
                }
            }
        }
    } નહીંતર {
        GGML_ABORT("unsupported upscale mode");
    }
}

શૂન્ય ggml_compute_forward_upscale(
    અચલ ggml_compute_params * params,
    ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_upscale_f32(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}


// ggml_compute_forward_pad

template<તાર્કિક circular_t>
સ્થિર શૂન્ય ggml_compute_forward_pad_f32(
    અચલ ggml_compute_params * params,
          ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પ્રતિજ્ઞા(dst->nb[0] == કદ(દશાંશ));

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    દશાંશ * dst_ptr = (દશાંશ *) dst->data;
    અચલ int32_t lp0 = ggml_get_op_params_i32(dst, 0);
    અચલ int32_t rp0 = ggml_get_op_params_i32(dst, 1);
    અચલ int32_t lp1 = ggml_get_op_params_i32(dst, 2);
    અચલ int32_t rp1 = ggml_get_op_params_i32(dst, 3);
    અચલ int32_t lp2 = ggml_get_op_params_i32(dst, 4);
    અચલ int32_t rp2 = ggml_get_op_params_i32(dst, 5);
    અચલ int32_t lp3 = ggml_get_op_params_i32(dst, 6);
    અચલ int32_t rp3 = ggml_get_op_params_i32(dst, 7);

    અચલ int64_t nrows = ne1 * ne2 * ne3;
    અચલ તાર્કિક src_row_contig = src0->nb[0] == કદ(દશાંશ);

    માટે (int64_t row = ith; row < nrows; row += nth) {
        અચલ int64_t i1 = row % ne1;
        અચલ int64_t i2 = (row / ne1) % ne2;
        અચલ int64_t i3 = row / (ne1 * ne2);

        દશાંશ * dst_row = dst_ptr + row * ne0;

        જો constexpr (circular_t) {
            // circular padding wraps around on a torus
            અચલ int64_t src_i1 = ggml_wrap_around(i1 - lp1, ne01);
            અચલ int64_t src_i2 = ggml_wrap_around(i2 - lp2, ne02);
            અચલ int64_t src_i3 = ggml_wrap_around(i3 - lp3, ne03);

            માટે (int64_t i0 = 0; i0 < ne0; ++i0) {
                અચલ int64_t src_i0 = ggml_wrap_around(i0 - lp0, ne00);
                dst_row[i0] = *(અચલ દશાંશ *)((અચલ અક્ષર *) src0->data + src_i3*nb03 + src_i2*nb02 + src_i1*nb01 + src_i0*nb00);
            }
        } નહીંતર {
            અચલ તાર્કિક interior = (i1 >= lp1 && i1 < ne1 - rp1)
                               && (i2 >= lp2 && i2 < ne2 - rp2)
                               && (i3 >= lp3 && i3 < ne3 - rp3);
            જો (!interior) {
                memset(dst_row, 0, ne0 * કદ(દશાંશ));
                ચાલુ;
            }

            અચલ અક્ષર * src_row = (અચલ અક્ષર *) src0->data + (i3 - lp3)*nb03 + (i2 - lp2)*nb02 + (i1 - lp1)*nb01;

            જો (src_row_contig && lp0 >= 0 && rp0 >= 0) {
                memset(dst_row, 0, lp0 * કદ(દશાંશ));
                memcpy(dst_row + lp0, src_row, ne00 * કદ(દશાંશ));
                memset(dst_row + lp0 + ne00, 0, rp0 * કદ(દશાંશ));
            } નહીંતર {
                માટે (int64_t i0 = 0; i0 < ne0; ++i0) {
                    જો (i0 >= lp0 && i0 < ne0 - rp0) {
                        dst_row[i0] = *(અચલ દશાંશ *)(src_row + (i0 - lp0)*nb00);
                    } નહીંતર {
                        dst_row[i0] = 0.0f;
                    }
                }
            }
        }
    }
}


શૂન્ય ggml_compute_forward_pad(
    અચલ ggml_compute_params * params,
    ggml_tensor * dst) {
    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ તાર્કિક circular = (તાર્કિક) ggml_get_op_params_i32(dst, 8);
    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                જો (circular) {
                    ggml_compute_forward_pad_f32<સાચું>(params, dst);
                } નહીંતર {
                    ggml_compute_forward_pad_f32<ખોટું>(params, dst);
                }
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_pad_reflect_1d

શૂન્ય ggml_compute_forward_pad_reflect_1d(
        અચલ ggml_compute_params * params,
              ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ int32_t * opts = (અચલ int32_t *) dst->op_params;
    અચલ પૂર્ણાંક p0 = opts[0];
    અચલ પૂર્ણાંક p1 = opts[1];

    GGML_TENSOR_UNARY_OP_LOCALS

    માટે (int64_t i3 = 0; i3 < ne3; i3++) {
        માટે (int64_t i2 = 0; i2 < ne2; i2++) {
            માટે (int64_t i1 = ith; i1 < ne1; i1 += nth) {
                દશાંશ * left  = (દશાંશ *) ((અક્ષર *) dst->data + i3*nb3 + i2*nb2 + i1*nb1 +         p0*nb0);
                દશાંશ * right = (દશાંશ *) ((અક્ષર *) dst->data + i3*nb3 + i2*nb2 + i1*nb1 + (ne0-p1-1)*nb0);

                ggml_vec_cpy_f32(ne00, left, (દશાંશ *) ((અક્ષર *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

                માટે (પૂર્ણાંક i0 = 1; i0 <= p0; i0++) { left[-i0] = left[i0];   }
                માટે (પૂર્ણાંક i0 = 1; i0 <= p1; i0++) { right[i0] = right[-i0]; }
            }
        }
    }
}

// ggml_compute_forward_roll

સ્થિર int64_t ggml_wrap_index(int64_t i, int64_t ne) {
    જો (i < 0) {
        પાછા i + ne;
    } નહીંતર જો (i >= ne) {
        પાછા i - ne;
    }
    પાછા i;
}

સ્થિર શૂન્ય ggml_compute_forward_roll_f32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ દશાંશ * src_data = (અચલ દશાંશ *) src0->data;
    દશાંશ * dst_data = (દશાંશ *) dst->data;

    GGML_TENSOR_UNARY_OP_LOCALS

    અચલ પૂર્ણાંક s0 = ggml_get_op_params_i32(dst, 0);
    અચલ પૂર્ણાંક s1 = ggml_get_op_params_i32(dst, 1);
    અચલ પૂર્ણાંક s2 = ggml_get_op_params_i32(dst, 2);
    અચલ પૂર્ણાંક s3 = ggml_get_op_params_i32(dst, 3);

    અચલ int64_t total = ne1 * ne2 * ne3;
    અચલ int64_t per_thread = (total + params->nth) / params->nth;
    અચલ int64_t start = params->ith * per_thread;
    અચલ int64_t end   = std::min(start + per_thread, total);

    માટે (int64_t i = start; i < end; ++i) {
        અચલ int64_t i1 = i % ne1;
        અચલ int64_t i2 = (i / ne1) % ne2;
        અચલ int64_t i3 = i / (ne2 * ne1);
        દશાંશ * dst_row = dst_data + (i3*nb3 + i2*nb2 + i1*nb1) / કદ(દશાંશ);

        અચલ int64_t i01 = ggml_wrap_index(i1 - s1, ne01);
        અચલ int64_t i02 = ggml_wrap_index(i2 - s2, ne02);
        અચલ int64_t i03 = ggml_wrap_index(i3 - s3, ne03);
        અચલ દશાંશ * src_row = src_data + (i03*nb03 + i02*nb02 + i01*nb01) / કદ(દશાંશ);

        અચલ int64_t s = ggml_wrap_index(-s0, ne00);
        અચલ int64_t n = ne00 - s;
        ggml_vec_cpy_f32(n, dst_row,     src_row + s);
        ggml_vec_cpy_f32(s, dst_row + n, src_row);
    }
}

શૂન્ય ggml_compute_forward_roll(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_roll_f32(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_arange

સ્થિર શૂન્ય ggml_compute_forward_arange_f32(
    અચલ ggml_compute_params * params,
    ggml_tensor * dst) {

    GGML_ASSERT(dst->nb[0] == કદ(દશાંશ));

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ દશાંશ start = ggml_get_op_params_f32(dst, 0);
    અચલ દશાંશ stop  = ggml_get_op_params_f32(dst, 1);
    અચલ દશાંશ step  = ggml_get_op_params_f32(dst, 2);

    અચલ int64_t steps = (int64_t) ceilf((stop - start) / step);

    GGML_ASSERT(ggml_nelements(dst) == steps);

    માટે (int64_t i = ith; i < steps; i+= nth) {
        દશાંશ value = start + step * i;
        ((દશાંશ *)dst->data)[i] = value;
    }
}

શૂન્ય ggml_compute_forward_arange(
    અચલ ggml_compute_params * params,
    ggml_tensor * dst) {
    પસંદગી (dst->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_arange_f32(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

સ્થિર શૂન્ય ggml_compute_forward_timestep_embedding_f32(
    અચલ ggml_compute_params * params,
    ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->nb[0] == કદ(દશાંશ));

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    અચલ પૂર્ણાંક dim = ggml_get_op_params_i32(dst, 0);
    અચલ પૂર્ણાંક max_period = ggml_get_op_params_i32(dst, 1);

    પૂર્ણાંક half = dim / 2;

    માટે (int64_t i = 0; i < ne00; i++) {
        દશાંશ * embed_data = (દશાંશ *)((અક્ષર *)  dst->data +  i*nb1);
        માટે (int64_t j = ith; j < half; j += nth) {
            દશાંશ timestep = ((દશાંશ *)src0->data)[i];
            દશાંશ freq = (દશાંશ)expf(-logf(max_period) * j / half);
            દશાંશ arg = timestep * freq;
            embed_data[j] = cosf(arg);
            embed_data[j + half] = sinf(arg);
        }
        જો (dim % 2 != 0 && ith == 0) {
            embed_data[2 * half] = 0.f;
        }
    }
}

શૂન્ય ggml_compute_forward_timestep_embedding(
    અચલ ggml_compute_params * params,
    ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_timestep_embedding_f32(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_argsort

template<ગણના ggml_sort_order order>
સંરચના cmp_argsort {
    અચલ દશાંશ * data;
    તાર્કિક operator()(int32_t a, int32_t b) અચલ {
        જો constexpr (order == GGML_SORT_ORDER_ASC) {
            પાછા data[a] < data[b];
        } નહીંતર {
            પાછા data[a] > data[b];
        }
    }
};

સ્થિર શૂન્ય ggml_compute_forward_argsort_f32(
    અચલ ggml_compute_params * params,
    ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(nb0 == કદ(દશાંશ));

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ int64_t nr = ggml_nrows(src0);

    ggml_sort_order order = (ggml_sort_order) ggml_get_op_params_i32(dst, 0);

    માટે (int64_t i = ith; i < nr; i += nth) {
        અચલ દશાંશ * src_data = (દશાંશ *)((અક્ષર *) src0->data + i*nb01);

        int32_t * dst_data = (int32_t *)((અક્ષર *) dst->data + i*nb1);

        માટે (int64_t j = 0; j < ne0; j++) {
            dst_data[j] = j;
        }

        પસંદગી (order) {
            સ્થિતિ GGML_SORT_ORDER_ASC:
                std::sort(dst_data, dst_data + ne0, cmp_argsort<GGML_SORT_ORDER_ASC>{src_data});
                તોડો;

            સ્થિતિ GGML_SORT_ORDER_DESC:
                std::sort(dst_data, dst_data + ne0, cmp_argsort<GGML_SORT_ORDER_DESC>{src_data});
                તોડો;

            મૂળભૂત:
                GGML_ABORT("invalid sort order");
        }
    }
}

શૂન્ય ggml_compute_forward_argsort(
    અચલ ggml_compute_params * params,
    ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_argsort_f32(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_top_k

સંરચના cmp_ટોચ_k {
    અચલ દશાંશ * data;
    તાર્કિક operator()(int32_t a, int32_t b) અચલ {
        પાછા data[a] > data[b];
    }
};

સ્થિર શૂન્ય ggml_compute_forward_ટોચ_k_f32(
    અચલ ggml_compute_params * params,
    ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(nb0 == કદ(દશાંશ));

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ int64_t nr = ggml_nrows(src0);

    અચલ પૂર્ણાંક ટોચ_k = ne0;

    int32_t * tmp = (int32_t *) params->wdata + (ne00 + CACHE_લીટી_SIZE_F32) * ith;

    માટે (int64_t i = ith; i < nr; i += nth) {
        અચલ દશાંશ * src_data = (દશાંશ *)((અક્ષર *) src0->data + i*nb01);

        માટે (int64_t j = 0; j < ne00; j++) {
            tmp[j] = j;
        }

        std::partial_sort(tmp, tmp + ટોચ_k, tmp + ne00, cmp_ટોચ_k{src_data});

        int32_t * dst_data = (int32_t *)((અક્ષર *) dst->data + i*nb1);

        std::copy(tmp, tmp + ટોચ_k, dst_data);

        // emphasize that the order is not important
        જો (ટોચ_k > 1) {
            std::swap(dst_data[0], dst_data[1]);
        }
    }
}

શૂન્ય ggml_compute_forward_ટોચ_k(
    અચલ ggml_compute_params * params,
    ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_ટોચ_k_f32(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

સ્થિર શૂન્ય ggml_compute_forward_flash_attn_ext_f16_one_chunk(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst,
        પૂર્ણાંક ir0, પૂર્ણાંક ir1,
        int64_t ic_start, int64_t ic_end,
        દશાંશ * partials, int64_t partial_stride) {

    અચલ તાર્કિક write_partials = (partials != nullptr);
    અચલ ggml_tensor * q     = dst->src[0];
    અચલ ggml_tensor * k     = dst->src[1];
    અચલ ggml_tensor * v     = dst->src[2];
    અચલ ggml_tensor * mask  = dst->src[3];
    અચલ ggml_tensor * sinks = dst->src[4];

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    અચલ int64_t DK = nek0;
    અચલ int64_t DV = nev0;
    અચલ int64_t N  = neq1;

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
    GGML_ASSERT(nb0 == કદ(દશાંશ));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    અચલ int64_t rk2 = neq2/nek2;
    અચલ int64_t rk3 = neq3/nek3;

    અચલ int64_t rv2 = neq2/nev2;
    અચલ int64_t rv3 = neq3/nev3;

    // parallelize by q rows using ggml_vec_dot_f32

    દશાંશ scale         = 1.0f;
    દશાંશ max_bias      = 0.0f;
    દશાંશ logit_softcap = 0.0f;

    memcpy(&scale,         (દશાંશ *) dst->op_params + 0, કદ(દશાંશ));
    memcpy(&max_bias,      (દશાંશ *) dst->op_params + 1, કદ(દશાંશ));
    memcpy(&logit_softcap, (દશાંશ *) dst->op_params + 2, કદ(દશાંશ));

    જો (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    અચલ uint32_t n_head      = neq2;
    અચલ uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));

    અચલ દશાંશ m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    અચલ દશાંશ m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    ggml_type         અચલ k_vec_dot_type = ggml_get_type_traits_cpu(k->type)->vec_dot_type;
    ggml_from_float_t અચલ q_to_vec_dot   = ggml_get_type_traits_cpu(k_vec_dot_type)->from_float;
    ggml_vec_dot_t    અચલ kq_vec_dot     = ggml_get_type_traits_cpu(k->type)->vec_dot;
    ggml_to_float_t   અચલ v_to_float     = ggml_get_type_traits(v->type)->to_float;

    GGML_ASSERT((                            q_to_vec_dot) && "fattn: unsupported K-type");
    GGML_ASSERT((v->type == GGML_TYPE_F32 || v_to_float  ) && "fattn: unsupported V-type");

    પૂર્ણાંક ith = params->ith;

    માટે (પૂર્ણાંક ir = ir0; ir < ir1; ++ir) {
        // q indices
        અચલ પૂર્ણાંક iq3 = ir/(neq2*neq1);
        અચલ પૂર્ણાંક iq2 = (ir - iq3*neq2*neq1)/neq1;
        અચલ પૂર્ણાંક iq1 = (ir - iq3*neq2*neq1 - iq2*neq1);

        અચલ uint32_t h = iq2; // head index
        અચલ દશાંશ slope = (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1) : 1.0f;

        દશાંશ S = 0.0f;      // sum
        દશાંશ M = -INFINITY; // maximum KQ value

        દશાંશ       * VKQ32 = (દશાંશ       *) params->wdata + ith*(1*DK + 2*DV + CACHE_લીટી_SIZE_F32); // FP32 VKQ accumulator
        દશાંશ       * V32   =                 (VKQ32 + 1*DV); // (temporary) FP32 V buffer
        ggml_fp16_t * VKQ16 = (ggml_fp16_t *) (VKQ32 + 1*DV); // (temporary) FP16 VKQ accumulator
        ggml_fp16_t * Q_q   = (ggml_fp16_t *) (VKQ32 + 2*DV); // (temporary) buffer for Q converted to quantized/FP16

        જો (v->type == GGML_TYPE_F16) {
            memset(VKQ16, 0, DV*કદ(ggml_fp16_t));
        } નહીંતર {
            memset(VKQ32, 0, DV*કદ(દશાંશ));
        }

        અચલ ggml_fp16_t * mp = mask ? (ggml_fp16_t *)((અક્ષર *) mask->data + iq1*mask->nb[1] + (iq2%mask->ne[2])*mask->nb[2] + (iq3%mask->ne[3])*mask->nb[3]) : ખાલી;

        // k indices
        અચલ પૂર્ણાંક ik3 = iq3 / rk3;
        અચલ પૂર્ણાંક ik2 = iq2 / rk2;

        // v indices
        અચલ પૂર્ણાંક iv3 = iq3 / rv3;
        અચલ પૂર્ણાંક iv2 = iq2 / rv2;

        અચલ દશાંશ * pq = (અચલ દશાંશ *) ((અક્ષર *) q->data + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3));
        q_to_vec_dot(pq, Q_q, DK);

        // online softmax / attention
        // loop over n_kv and n_head_kv
        // ref: https://arxiv.org/pdf/2112.05682.pdf

        માટે (int64_t ic = ic_start; ic < ic_end; ++ic) {
            અચલ દશાંશ mv = mp ? slope*GGML_CPU_FP16_TO_FP32(mp[ic]) : 0.0f;
            જો (mv == -INFINITY) {
                ચાલુ;
            }

            દશાંશ s; // KQ value

            અચલ અક્ષર * k_data = (અચલ અક્ષર *) k->data + ( ic*nbk1 + ik2*nbk2 + ik3*nbk3);
            kq_vec_dot(DK, &s, 0, k_data, 0, Q_q, 0, 1);

            s = s*scale; // scale KQ value

            જો (logit_softcap != 0.0f) {
                s = logit_softcap*tanhf(s);
            }

            s += mv; // apply mask

            અચલ દશાંશ Mold = M;

            દશાંશ ms = 1.0f; // upon new higher max val, scale VKQ and KQ sum with this value
            દશાંશ vs = 1.0f; // post-softmax KQ value, expf(s - M)

            અચલ અક્ષર * v_data = ((અચલ અક્ષર *) v->data + (ic*nbv1 + iv2*nbv2 + iv3*nbv3));

            જો (v->type == GGML_TYPE_F16) {
                જો (s > M) {
                    // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
                    M = s;
                    ms = expf(Mold - M);

                    // V = V*expf(Mold - M)
                    ggml_vec_scale_f16(DV, VKQ16, ms);
                } નહીંતર {
                    // no new maximum, ms == 1.0f, vs != 1.0f
                    vs = expf(s - M);
                }

                // V += v*expf(s - M)
                ggml_vec_mad_f16(DV, VKQ16, (અચલ ggml_fp16_t *) v_data, vs);
            } નહીંતર {
                જો (s > M) {
                    // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
                    M = s;
                    ms = expf(Mold - M);

                    // V = V*expf(Mold - M)
                    ggml_vec_scale_f32(DV, VKQ32, ms);
                } નહીંતર {
                    // no new maximum, ms == 1.0f, vs != 1.0f
                    vs = expf(s - M);
                }

                // V += v*expf(s - M)
                જો (v_to_float) {
                    v_to_float(v_data, V32, DV);
                    ggml_vec_mad_f32(DV, VKQ32, V32, vs);
                } નહીંતર {
                    // V is F32
                    ggml_vec_mad_f32(DV, VKQ32, (અચલ દશાંશ *) v_data, vs);
                }
            }

            S = S*ms + vs; // scale and increment sum with partial sum
        }

        જો (v->type == GGML_TYPE_F16) {
            માટે (int64_t d = 0; d < DV; ++d) {
                VKQ32[d] = GGML_CPU_FP16_TO_FP32(VKQ16[d]);
            }
        }

        // sinks - apply only on the first kv-chunk
        જો (sinks && ic_start == 0) {
            અચલ દશાંશ s = ((દશાંશ *)((અક્ષર *) sinks->data))[h];

            દશાંશ ms = 1.0f;
            દશાંશ vs = 1.0f;

            જો (s > M) {
                ms = expf(M - s);
                M = s;
                ggml_vec_scale_f32(DV, VKQ32, ms);
            } નહીંતર {
                vs = expf(s - M);
            }

            S = S*ms + vs;
        }

        જો (write_partials) {
            // Write M, S, VKQ to partials for later reduction
            // partials layout: [M, S, VKQ[DV]] per query head
            દશાંશ * partial = partials + ir * partial_stride;
            partial[0] = M;
            partial[1] = S;
            memcpy(partial + 2, VKQ32, DV * કદ(દશાંશ));
        } નહીંતર {
            // V /= S
            અચલ દશાંશ S_inv = S == 0.0f ? 0.0f : 1.0f/S;
            ggml_vec_scale_f32(DV, VKQ32, S_inv);

            // dst indices
            અચલ પૂર્ણાંક i1 = iq1;
            અચલ પૂર્ણાંક i2 = iq2;
            અચલ પૂર્ણાંક i3 = iq3;

            // permute(0, 2, 1, 3)
            memcpy((અક્ષર *) dst->data + (i3*ne2*ne1 + i2 + i1*ne1)*nb1, VKQ32, nb1);
        }
    }
}

સ્થિર શૂન્ય ggml_compute_forward_flash_attn_ext_tiled(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst,
        પૂર્ણાંક ir0, પૂર્ણાંક ir1) {
    અચલ ggml_tensor * q     = dst->src[0];
    અચલ ggml_tensor * k     = dst->src[1];
    અચલ ggml_tensor * v     = dst->src[2];
    અચલ ggml_tensor * mask  = dst->src[3];
    અચલ ggml_tensor * sinks = dst->src[4];

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    અચલ int64_t DK = nek0;
    અચલ int64_t DV = nev0;
    અચલ int64_t N  = neq1;

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
    GGML_ASSERT(nb0 == કદ(દશાંશ));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(k->type == v->type);
    અચલ ggml_type kv_type = k->type;


    // broadcast factors
    અચલ int64_t rk2 = neq2/nek2;
    અચલ int64_t rk3 = neq3/nek3;

    અચલ int64_t rv2 = neq2/nev2;
    અચલ int64_t rv3 = neq3/nev3;

    દશાંશ scale         = 1.0f;
    દશાંશ max_bias      = 0.0f;
    દશાંશ logit_softcap = 0.0f;

    memcpy(&scale,         (દશાંશ *) dst->op_params + 0, કદ(દશાંશ));
    memcpy(&max_bias,      (દશાંશ *) dst->op_params + 1, કદ(દશાંશ));
    memcpy(&logit_softcap, (દશાંશ *) dst->op_params + 2, કદ(દશાંશ));

    જો (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    અચલ uint32_t n_head      = neq2;
    અચલ uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));

    અચલ દશાંશ m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    અચલ દશાંશ m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    પૂર્ણાંક ith = params->ith;

    સ્થિર constexpr પૂર્ણાંક Q_TILE_SZ  = ggml_fa_tile_config::Q;
    સ્થિર constexpr પૂર્ણાંક KV_TILE_SZ = ggml_fa_tile_config::KV;

    પૂર્ણાંક ir = ir0;
    જ્યારે (ir < ir1) {
        // q indices for the start of this tile
        અચલ પૂર્ણાંક iq3 = ir/(neq2*neq1);
        અચલ પૂર્ણાંક iq2 = (ir - iq3*neq2*neq1)/neq1;
        અચલ પૂર્ણાંક iq1 = (ir - iq3*neq2*neq1 - iq2*neq1);

        // Number of valid rows in this tile:
        // - limited by tile size (Q_TILE_SZ)
        // - limited by chunk boundary (ir1 - ir)
        // - limited by head boundary (neq1 - iq1) to avoid crossing into next head
        અચલ પૂર્ણાંક tile_rows = MIN(Q_TILE_SZ, MIN((પૂર્ણાંક)(ir1 - ir), (પૂર્ણાંક)(neq1 - iq1)));
        GGML_ASSERT(tile_rows > 0);

        અચલ uint32_t h = iq2; // head index
        અચલ દશાંશ slope = (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1) : 1.0f;

        દશાંશ S[Q_TILE_SZ];
        દશાંશ M[Q_TILE_SZ];

        માટે (પૂર્ણાંક i = 0 ; i < Q_TILE_SZ; ++i) {
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
        દશાંશ * base  = (દશાંશ *) params->wdata + ith*(Q_TILE_SZ*DK + 2*Q_TILE_SZ*KV_TILE_SZ + Q_TILE_SZ*DV + KV_TILE_SZ*DV + KV_TILE_SZ*DK + CACHE_લીટી_SIZE_F32);

        શૂન્ય  * Q_q    = base;
        દશાંશ * KQ     = (દશાંશ *)((અક્ષર *)base + Q_TILE_SZ * DK * કદ(દશાંશ));
        દશાંશ * mask32 = KQ + Q_TILE_SZ * KV_TILE_SZ;
        દશાંશ * VKQ32  = mask32 + Q_TILE_SZ * KV_TILE_SZ;
        દશાંશ * V32    = VKQ32 + Q_TILE_SZ * DV;
        દશાંશ * K_f32  = V32 + KV_TILE_SZ * DV;

        memset(VKQ32, 0, Q_TILE_SZ * DV * કદ(દશાંશ));
        memset(mask32, 0, Q_TILE_SZ * KV_TILE_SZ * કદ(દશાંશ));

        // k indices
        અચલ પૂર્ણાંક ik3 = iq3 / rk3;
        અચલ પૂર્ણાંક ik2 = iq2 / rk2;

        // v indices
        અચલ પૂર્ણાંક iv3 = iq3 / rv3;
        અચલ પૂર્ણાંક iv2 = iq2 / rv2;

        {
            દશાંશ * Q_f32 = (દશાંશ *)Q_q;
            માટે (પૂર્ણાંક tq = 0; tq < tile_rows; tq++) {
                અચલ દશાંશ * pq = (અચલ દશાંશ *) ((અક્ષર *) q->data + ((iq1 + tq)*nbq1 + iq2*nbq2 + iq3*nbq3));
                memcpy(Q_f32 + tq * DK, pq, DK * કદ(દશાંશ));
            }
            માટે (પૂર્ણાંક tq = tile_rows; tq < Q_TILE_SZ; tq++) {
                memset(Q_f32 + tq * DK, 0, DK * કદ(દશાંશ));
            }
        }

        memset(K_f32, 0, DK * KV_TILE_SZ * કદ(દશાંશ));
        memset(V32,   0, KV_TILE_SZ * DV * કદ(દશાંશ));

        માટે (int64_t ic = 0; ic < nek1; ic += KV_TILE_SZ) {
            અચલ પૂર્ણાંક kv_tile = (પૂર્ણાંક)std::min((int64_t)KV_TILE_SZ, nek1 - ic);

            // skip the tile entirely if all the masks are -inf
            જો (mask) {
                તાર્કિક can_skip = સાચું;
                માટે (પૂર્ણાંક tq = 0; tq < tile_rows; tq++) {
                    અચલ ggml_fp16_t * mp_row = (અચલ ggml_fp16_t *)((અચલ અક્ષર *) mask->data + (iq1 + tq)*mask->nb[1] + (iq2%mask->ne[2])*mask->nb[2] + (iq3%mask->ne[3])*mask->nb[3]);
                    માટે (પૂર્ણાંક tk = 0; tk < kv_tile; tk++) {
                        mask32[tq * KV_TILE_SZ + tk] = slope * GGML_CPU_FP16_TO_FP32(mp_row[ic + tk]);
                        જો (mask32[tq * KV_TILE_SZ + tk] != -INFINITY) {
                            can_skip = ખોટું;
                        }
                    }
                    // Pad remaining mask entries with -inf
                    માટે (પૂર્ણાંક tk = kv_tile; tk < KV_TILE_SZ; tk++) {
                        mask32[tq * KV_TILE_SZ + tk] = -INFINITY;
                    }
                }

                જો (can_skip) {
                    ચાલુ;
                }
            }

            // Pack K tile transposed: K_f32[dk][kv] so KV_TILE is contiguous (SIMD dim)
            // Zero-pad the last tile so the GEMM always operates on KV_TILE_SZ columns
            માટે (પૂર્ણાંક tk = 0; tk < kv_tile; tk++) {
                અચલ અક્ષર * k_data = (અચલ અક્ષર *)k->data + (ic + tk)*nbk1 + ik2*nbk2 + ik3*nbk3;
                જો (kv_type == GGML_TYPE_F16) {
                    અચલ ggml_fp16_t * k_f16 = (અચલ ggml_fp16_t *)k_data;
                    માટે (int64_t dk = 0; dk < DK; dk++) {
                        K_f32[dk * KV_TILE_SZ + tk] = GGML_CPU_FP16_TO_FP32(k_f16[dk]);
                    }
                } નહીંતર {
                    અચલ દશાંશ * k_f32_src = (અચલ દશાંશ *)k_data;
                    માટે (int64_t dk = 0; dk < DK; dk++) {
                        K_f32[dk * KV_TILE_SZ + tk] = k_f32_src[dk];
                    }
                }
            }
            memset(KQ, 0, Q_TILE_SZ * KV_TILE_SZ * કદ(દશાંશ));
            simd_gemm(KQ, (અચલ દશાંશ *)Q_q, K_f32, Q_TILE_SZ, DK, KV_TILE_SZ);
            ggml_vec_scale_f32(Q_TILE_SZ * KV_TILE_SZ, KQ, scale);

            જો (logit_softcap != 0.0f) {
                ggml_vec_tanh_f32(Q_TILE_SZ * KV_TILE_SZ, KQ, KQ);
                ggml_vec_scale_f32(Q_TILE_SZ * KV_TILE_SZ, KQ, logit_softcap);
            }

            // Set padded KQ entries to -inf so softmax gives them zero weight
            જો (kv_tile < KV_TILE_SZ) {
                માટે (પૂર્ણાંક tq = 0; tq < Q_TILE_SZ; tq++) {
                    માટે (પૂર્ણાંક tk = kv_tile; tk < KV_TILE_SZ; tk++) {
                        KQ[tq * KV_TILE_SZ + tk] = -INFINITY;
                    }
                }
            }

            જો (mask) {
                ggml_vec_add_f32(tile_rows * KV_TILE_SZ, KQ, KQ, mask32);
            }

            તાર્કિક skip[Q_TILE_SZ] = {};

            માટે (પૂર્ણાંક tq = 0; tq < Q_TILE_SZ; tq++) {
                દશાંશ * kq_row = KQ + tq * KV_TILE_SZ;

                દશાંશ tile_max;
                ggml_vec_max_f32(KV_TILE_SZ, &tile_max, kq_row);

                જો (tile_max == -INFINITY) {
                    skip[tq] = સાચું;
                    ચાલુ;
                }

                અચલ દશાંશ Mold = M[tq];
                અચલ દશાંશ Mnew = fmaxf(Mold, tile_max);

                જો (Mnew > Mold) {
                    અચલ દશાંશ ms = expf(Mold - Mnew);
                    ggml_vec_scale_f32(DV, VKQ32 + tq * DV, ms);
                    S[tq] *= ms;
                }
                M[tq] = Mnew;


                S[tq] += ggml_vec_soft_max_f32(KV_TILE_SZ, kq_row, kq_row, Mnew);
            }

            // V accumulation: VKQ32 += softmax(KQ) * V
            // Pack V tile to contiguous F32, zero-padded
            માટે (પૂર્ણાંક tk = 0; tk < kv_tile; tk++) {
                અચલ અક્ષર * v_data = (અચલ અક્ષર *)v->data + (ic + tk)*nbv1 + iv2*nbv2 + iv3*nbv3;
                જો (kv_type == GGML_TYPE_F16) {
                    ggml_cpu_fp16_to_fp32((અચલ ggml_fp16_t *)v_data, V32 + tk * DV, DV);
                } નહીંતર {
                    memcpy(V32 + tk * DV, v_data, DV * કદ(દશાંશ));
                }
            }
            માટે (પૂર્ણાંક tq = 0; tq < Q_TILE_SZ; tq++) {
                જો (skip[tq]) {
                    memset(KQ + tq * KV_TILE_SZ, 0, KV_TILE_SZ * કદ(દશાંશ));
                }
            }
            simd_gemm(VKQ32, KQ, V32, Q_TILE_SZ, KV_TILE_SZ, DV);
        }

        // sinks (apply only to valid rows in the tile)
        જો (sinks) {
            અચલ દશાંશ s = ((દશાંશ *)((અક્ષર *) sinks->data))[h];

            માટે (પૂર્ણાંક tq = 0; tq < tile_rows; tq++) {
                દશાંશ ms = 1.0f;
                દશાંશ vs = 1.0f;

                જો (s > M[tq]) {
                    ms = expf(M[tq] - s);
                    ggml_vec_scale_f32(DV, VKQ32 + tq * DV, ms);
                } નહીંતર {
                    vs = expf(s - M[tq]);
                }

                S[tq] = S[tq] * ms + vs;
            }
        }

        માટે (પૂર્ણાંક tq = 0; tq < tile_rows; tq++) {
            // V /= S
            અચલ દશાંશ S_inv = S[tq] == 0.0f ? 0.0f : 1.0f / S[tq];
            ggml_vec_scale_f32(DV, VKQ32 + tq * DV, S_inv);

            // dst indices
            અચલ પૂર્ણાંક i1 = iq1 + tq;
            અચલ પૂર્ણાંક i2 = iq2;
            અચલ પૂર્ણાંક i3 = iq3;

            // permute(0, 2, 1, 3)
            memcpy((અક્ષર *) dst->data + (i3*ne2*ne1 + i2 + i1*ne1)*nb1, VKQ32 + tq * DV, nb1);
        }

        ir += tile_rows;
    }
}

// Reduction function: combines partial results across KV chunks
// Partials layout in wdata: [n_q_heads][n_chunks][2 + DV]
સ્થિર શૂન્ય ggml_flash_attn_ext_reduce_partials(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst,
        અચલ int64_t n_chunks,
        અચલ int64_t chunk_size) {

    અચલ ggml_tensor * q = dst->src[0];
    અચલ ggml_tensor * k = dst->src[1];
    અચલ ggml_tensor * v = dst->src[2];

    અચલ int64_t DK        = k->ne[0];
    અચલ int64_t DV        = v->ne[0];
    અચલ int64_t nek1      = k->ne[1];
    અચલ int64_t n_q_heads = q->ne[2];

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ int64_t wdata_per_thread = DK + 2*DV + CACHE_લીટી_SIZE_F32;
    દશાંશ *       thread_wdata     = (દશાંશ *) params->wdata + ith * wdata_per_thread;

    અચલ int64_t partials_offset  = nth * (DK + 2*DV + CACHE_લીટી_SIZE_F32);
    અચલ int64_t partial_size     = 2 + DV;
    અચલ દશાંશ * partials_base    = (અચલ દશાંશ *) params->wdata + partials_offset;

    // Output layout
    અચલ int64_t ne1 = dst->ne[1];
    અચલ int64_t ne2 = dst->ne[2];
    અચલ size_t  nb1 = dst->nb[1];

    // Each thread reduces a subset of query heads
    માટે (int64_t q_head = ith; q_head < n_q_heads; q_head += nth) {
        દશાંશ   M_final   = -INFINITY;
        દશાંશ   S_final   = 0.0f;
        દશાંશ * VKQ_final = thread_wdata;
        memset(VKQ_final, 0, DV * કદ(દશાંશ));

        // Combine partials from all chunks
        માટે (int64_t chunk_idx = 0; chunk_idx < n_chunks; ++chunk_idx) {
            અચલ int64_t ic_start = chunk_idx * chunk_size;
            જો (ic_start >= nek1) ચાલુ;

            અચલ દશાંશ * partial   = partials_base + (q_head * n_chunks + chunk_idx) * partial_size;
            અચલ દશાંશ   M_chunk   = partial[0];
            અચલ દશાંશ   S_chunk   = partial[1];
            અચલ દશાંશ * VKQ_chunk = partial + 2;

            જો (S_chunk == 0.0f) ચાલુ;

            અચલ દશાંશ M_new     = fmaxf(M_final, M_chunk);
            અચલ દશાંશ scale_old = expf(M_final - M_new);
            અચલ દશાંશ scale_new = expf(M_chunk - M_new);

            માટે (int64_t d = 0; d < DV; ++d) {
                VKQ_final[d] = VKQ_final[d] * scale_old + VKQ_chunk[d] * scale_new;
            }
            S_final = S_final * scale_old + S_chunk * scale_new;
            M_final = M_new;
        }

        // Normalize and write to output
        જો (S_final != 0.0f) {
            અચલ દશાંશ S_inv = 1.0f / S_final;
            ggml_vec_scale_f32(DV, VKQ_final, S_inv);
        }
        // iq1=0, iq3=0 for decode
        memcpy((અક્ષર *) dst->data + (0*ne2*ne1 + q_head + 0*ne1)*nb1, VKQ_final, nb1);
    }
}

સ્થિર શૂન્ય ggml_compute_forward_flash_attn_ext_f16(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * q     = dst->src[0];
    અચલ ggml_tensor * k     = dst->src[1];
    અચલ ggml_tensor * v     = dst->src[2];

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    અચલ int64_t DK = nek0;
    અચલ int64_t DV = nev0;
    અચલ int64_t N  = neq1;


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
    GGML_ASSERT(nb0 == કદ(દશાંશ));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    // When use_ref is set, force the vec-only reference implementation (no tiling, no KV-chunking)
    અચલ તાર્કિક use_ref = params->use_ref;

    અચલ તાર્કિક kv_is_f32_or_f16 = (k->type == GGML_TYPE_F32 || k->type == GGML_TYPE_F16);
    અચલ તાર્કિક use_split_kv_path = !use_ref && (neq1 == 1 && neq3 == 1) && kv_is_f32_or_f16 && (k->type == v->type) && q->type == GGML_TYPE_F32 && nek1 >= 512;

    જો (use_split_kv_path) {
        અચલ int64_t chunk_size = (nek1 + nth - 1) / nth;

        // Partials buffer layout: [q_head][kv_chunk][M, S, VKQ]
        અચલ int64_t partial_size  = 2 + DV;
        દશાંશ *       partials_base = (દશાંશ *) params->wdata + nth * (DK + 2*DV + CACHE_લીટી_SIZE_F32);

        અચલ int64_t ic_start = ith * chunk_size;
        અચલ int64_t ic_end   = std::min(ic_start + chunk_size, nek1);

        અચલ int64_t partial_stride = nth * partial_size;
        દશાંશ *       chunk_partials = partials_base + ith * partial_size;

        જો (ic_start < nek1) {
            માટે (int64_t q_head = 0; q_head < neq2; q_head++) {
                ggml_compute_forward_flash_attn_ext_f16_one_chunk(
                    params, dst, q_head, q_head + 1, ic_start, ic_end,
                    chunk_partials, partial_stride);
            }
        } નહીંતર {
            માટે (int64_t q_head = 0; q_head < neq2; q_head++) {
                દશાંશ * q_partials = chunk_partials + q_head * partial_stride;
                q_partials[0] = -INFINITY;  // M
                q_partials[1] = 0.0f;       // S
            }
        }

        ggml_barrier(params->threadpool);
        ggml_flash_attn_ext_reduce_partials(params, dst, nth, chunk_size);
    } નહીંતર {

        // total rows in q
        અચલ int64_t nr = neq1*neq2*neq3;

        // disable for NUMA
        અચલ તાર્કિક disable_chunking = ggml_is_numa();

        // 4x chunks per thread
        પૂર્ણાંક nth_scaled = nth * 4;
        int64_t chunk_size = (nr + nth_scaled - 1) / nth_scaled;
        int64_t nchunk     = (nr + chunk_size - 1) / chunk_size;

        જો (nth == 1 || nchunk < nth || disable_chunking) {
            nchunk = nth;
        }

        જો (ith == 0) {
            ggml_threadpool_chunk_set(params->threadpool, nth);
        }

        ggml_barrier(params->threadpool);

        અચલ int64_t dr = (nr + nchunk - 1) / nchunk;

        સ્થિર constexpr int64_t Q_TILE_SZ  = ggml_fa_tile_config::Q;
        તાર્કિક use_tiled = !use_ref &&
                               (q->type == GGML_TYPE_F32 &&
                                kv_is_f32_or_f16 &&
                                k->type == v->type &&
                                neq1 >= Q_TILE_SZ);
#જો defined(GGML_SIMD) && !defined(__x86_64__) && !defined(_M_X64)
#જો defined(__ARM_FEATURE_SVE)
        અચલ int64_t f32_epr = svcntw();
#નહીંતર
        અચલ int64_t f32_epr = GGML_F32_EPR;
#endif
        use_tiled &= (DV % f32_epr == 0);
#endif
        પૂર્ણાંક current_chunk = ith;

        જ્યારે (current_chunk < nchunk) {
            અચલ int64_t ir0 = dr * current_chunk;
            અચલ int64_t ir1 = MIN(ir0 + dr, nr);

            જો (use_tiled) {
                ggml_compute_forward_flash_attn_ext_tiled(params, dst, ir0, ir1);
            } નહીંતર {
                ggml_compute_forward_flash_attn_ext_f16_one_chunk(params, dst, ir0, ir1, 0, nek1, nullptr, 0);
            }

            current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
        }
    }
}

શૂન્ય ggml_compute_forward_flash_attn_ext(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {
    પસંદગી (dst->op_params[3]) {
        સ્થિતિ GGML_PREC_DEFAULT:
        સ્થિતિ GGML_PREC_F32:
            {
                // uses F32 accumulators
                ggml_compute_forward_flash_attn_ext_f16(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_flash_attn_back

સ્થિર શૂન્ય ggml_compute_forward_flash_attn_back_f32(
        અચલ ggml_compute_params * params,
        અચલ તાર્કિક masked,
              ggml_tensor * dst) {

    અચલ ggml_tensor * q = dst->src[0];
    અચલ ggml_tensor * k = dst->src[1];
    અચલ ggml_tensor * v = dst->src[2];
    અચલ ggml_tensor * d = dst->src[3];

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

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ int64_t D = neq0;
    અચલ int64_t N = neq1;
    અચલ int64_t P = nek1 - N;
    અચલ int64_t M = P + N;

    અચલ પૂર્ણાંક Mup  = ggml_up(M, GGML_SOFT_MAX_UNROLL);
    અચલ પૂર્ણાંક mxDM = MAX(D, Mup);

    // GGML_ASSERT(ne0 == D);
    // GGML_ASSERT(ne1 == N);
    GGML_ASSERT(P >= 0);

    GGML_ASSERT(nbq0 == કદ(દશાંશ));
    GGML_ASSERT(nbk0 == કદ(દશાંશ));
    GGML_ASSERT(nbv0 == કદ(દશાંશ));

    GGML_ASSERT(neq0 == D);
    GGML_ASSERT(nek0 == D);
    GGML_ASSERT(nev1 == D);
    GGML_ASSERT(ned0 == D);

    GGML_ASSERT(neq1 == N);
    GGML_ASSERT(nek1 == N + P);
    GGML_ASSERT(nev1 == D);
    GGML_ASSERT(ned1 == N);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == કદ(દશાંશ));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    જો (ith == 0) {
        memset(dst->data, 0, nb0*ne0*ne1*ne2*ne3);
    }
    ggml_barrier(params->threadpool);

    અચલ int64_t elem_q = ggml_nelements(q);
    અચલ int64_t elem_k = ggml_nelements(k);

    ggml_type result_type = dst->type;
    GGML_ASSERT(ggml_blck_size(result_type) == 1);
    અચલ size_t tsize = ggml_type_size(result_type);

    અચલ size_t offs_q = 0;
    અચલ size_t offs_k = offs_q + GGML_PAD(elem_q * tsize, GGML_MEM_ALIGN);
    અચલ size_t offs_v = offs_k + GGML_PAD(elem_k * tsize, GGML_MEM_ALIGN);

    શૂન્ય * grad_q = (અક્ષર *) dst->data;
    શૂન્ય * grad_k = (અક્ષર *) dst->data + offs_k;
    શૂન્ય * grad_v = (અક્ષર *) dst->data + offs_v;

    અચલ size_t nbgq1 = nb0*neq0;
    અચલ size_t nbgq2 = nb0*neq0*neq1;
    અચલ size_t nbgq3 = nb0*neq0*neq1*neq2;

    અચલ size_t nbgk1 = nb0*nek0;
    અચલ size_t nbgk2 = nb0*nek0*nek1;
    અચલ size_t nbgk3 = nb0*nek0*nek1*neq2;

    અચલ size_t nbgv1 = nb0*nev0;
    અચલ size_t nbgv2 = nb0*nev0*nev1;
    અચલ size_t nbgv3 = nb0*nev0*nev1*neq2;

    // parallelize by k rows using ggml_vec_dot_f32

    // total rows in k
    અચલ પૂર્ણાંક nr = nek2*nek3;

    // rows per thread
    અચલ પૂર્ણાંક dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr*ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    અચલ દશાંશ scale = 1.0f/sqrtf(D);

    //printf("P=%d N=%d D=%d ir0=%d ir1=%d scale = %f\n", P, N, D, ir0, ir1, scale);

    // how often k2 (and v2) is repeated in q2
    પૂર્ણાંક nrep = neq2/nek2;

    માટે (પૂર્ણાંક ir = ir0; ir < ir1; ++ir) {
        // q indices
        અચલ પૂર્ણાંક ik3 = ir/(nek2);
        અચલ પૂર્ણાંક ik2 = ir - ik3*nek2;

        અચલ પૂર્ણાંક iq3 = ik3;
        અચલ પૂર્ણાંક id3 = ik3;
        અચલ પૂર્ણાંક iv3 = ik3;
        અચલ પૂર્ણાંક iv2 = ik2;

        માટે (પૂર્ણાંક irep = 0; irep < nrep; ++irep) {
            અચલ પૂર્ણાંક iq2 = ik2 + irep*nek2;
            અચલ પૂર્ણાંક id2 = iq2;

            // (ik2 + irep*nek2) % nek2 == ik2
            માટે (પૂર્ણાંક iq1 = 0; iq1 < neq1; ++iq1) {
                અચલ પૂર્ણાંક id1 = iq1;

                // not sure about CACHE_LINE_SIZE_F32..
                // - maybe it must not be multiplied by 2 and excluded from .. in SM 1*(..) offset?
                દશાંશ * S  = (દશાંશ *) params->wdata + ith*2*(mxDM + CACHE_લીટી_SIZE_F32) + 0*(mxDM+CACHE_લીટી_SIZE_F32);
                દશાંશ * SM = (દશાંશ *) params->wdata + ith*2*(mxDM + CACHE_લીટી_SIZE_F32) + 1*(mxDM+CACHE_લીટી_SIZE_F32);

                માટે (પૂર્ણાંક i = M; i < Mup; ++i) {
                    S[i] = -INFINITY;
                }

                અચલ int64_t masked_begin = masked ? (P + iq1 + 1) : M;
                માટે (int64_t ic = 0; ic < masked_begin; ++ic) {
                    // k indices
                    અચલ પૂર્ણાંક ik1 = ic;

                    // S indices
                    અચલ પૂર્ણાંક i1 = ik1;

                    ggml_vec_dot_f32(neq0,
                            S + i1, 0,
                            (દશાંશ *) ((અક્ષર *) k->data + (ik1*nbk1 + ik2*nbk2 + ik3*nbk3)), 0,
                            (દશાંશ *) ((અક્ષર *) q->data + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3)), 0, 1);
                }

                // scale
                ggml_vec_scale_f32(masked_begin, S, scale);

                માટે (int64_t i = masked_begin; i < M; i++) {
                    S[i] = -INFINITY;
                }

                // softmax
                // exclude known -INF S[..] values from max and loop
                // dont forget to set their SM values to zero
                {
                    દશાંશ max = -INFINITY;
                    ggml_vec_max_f32(masked_begin, &max, S);

                    ggml_float sum = 0.0;
                    {
#ifdef GGML_SOFT_MAX_ACCELERATE
                        max = -max;
                        vDSP_vsadd(SM, 1, &max, SM, 1, Mup);
                        vvexpf(SM, SM, &Mup);
                        ggml_vec_sum_f32(Mup, &sum, SM);
#નહીંતર
                        sum = ggml_vec_soft_max_f32(Mup, SM, S, max);
#endif
                    }

                    પ્રતિજ્ઞા(sum > 0.0);

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
                માટે (int64_t ic = 0; ic < D; ++ic) {
                    ggml_vec_mad_f32(masked_begin,
                            S,
                             (દશાંશ *) ((અક્ષર *) v->data + (          ic*nbv1  + iv2*nbv2 + iv3*nbv3)),
                            *(દશાંશ *) ((અક્ષર *) d->data + (ic*nbd0 + id1*nbd1 + id2*nbd2 + id3*nbd3)));
                }

                // S = SM * (S - dot(SM, S))
                દશાંશ dot_SM_gradSM = 0;
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
                માટે (int64_t ic = 0; ic < masked_begin; ++ic) {
                    ggml_vec_mad_f32(D,
                            (દશાંશ *) ((અક્ષર *) grad_q  + (iq1*nbgq1 + iq2*nbgq2  + iq3*nbgq3)),
                            (દશાંશ *) ((અક્ષર *) k->data + (ic*nbk1   + ik2*nbk2   + ik3*nbk3)),
                            S[ic]);
                }

                // grad[k][:D,:M,iq2,iq3] += S.T @ qcur
                // for ic:
                //  grad[k][:D,ic,iq2,iq3] += S.T[0,ic] * qcur[:D,0]
                //  grad[k][:D,ic,iq2,iq3] += S[ic]     * qcur[:D,0]
                // exclude known zero S[..] values from loop
                માટે (int64_t ic = 0; ic < masked_begin; ++ic) {
                    ggml_vec_mad_f32(D,
                            (દશાંશ *) ((અક્ષર *) grad_k  + (ic*nbgk1  + ik2*nbgk2  + ik3*nbgk3)),
                            (દશાંશ *) ((અક્ષર *) q->data + (iq1*nbq1  + iq2*nbq2   + iq3*nbq3)),
                            S[ic]);
                }

                // grad[v][:M,:D,iv2,iv3] += d[:D,id1,id2,id3].T       @ SM
                // for ic:
                //  grad[v][:M,ic,iv2,iv3] += d[:D,id1,id2,id3].T[0,ic] * SM[:M]
                //  grad[v][:M,ic,iv2,iv3] += d[ic,id1,id2,id3]         * SM[:M]
                // exclude known zero SM[..] values from mad
                માટે (int64_t ic = 0; ic < D; ++ic) {
                    ggml_vec_mad_f32(masked_begin,
                            (દશાંશ *) ((અક્ષર *) grad_v   + (          ic*nbgv1 + iv2*nbgv2 + iv3*nbgv3)),
                            SM,
                            *(દશાંશ *) ((અક્ષર *) d->data + (ic*nbd0 + id1*nbd1 + id2*nbd2  + id3*nbd3)));
                }
            }
        }
    }
}

શૂન્ય ggml_compute_forward_flash_attn_back(
        અચલ ggml_compute_params * params,
        અચલ તાર્કિક masked,
        ggml_tensor * dst) {

    અચલ ggml_tensor * q = dst->src[0];

    પસંદગી (q->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_flash_attn_back_f32(params, masked, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_ssm_conv

સ્થિર શૂન્ય ggml_compute_forward_ssm_conv_f32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {
    અચલ ggml_tensor * src0 = dst->src[0]; // conv_x
    અચલ ggml_tensor * src1 = dst->src[1]; // conv1d.weight

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક nc  = src1->ne[0]; // d_conv
    અચલ પૂર્ણાંક ncs = src0->ne[0]; // d_conv - 1 + n_t
    અચલ પૂર્ણાંક nr  = src0->ne[1]; // d_inner
    અચલ પૂર્ણાંક n_t =  dst->ne[1]; // tokens per sequence
    અચલ પૂર્ણાંક n_s =  dst->ne[2]; // number of sequences in the batch

    GGML_ASSERT( dst->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == કદ(દશાંશ));
    GGML_ASSERT(src1->nb[0] == કદ(દશાંશ));
    GGML_ASSERT(src0->nb[1] == src0->ne[0]*કદ(દશાંશ));

    // rows per thread
    અચલ પૂર્ણાંક dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr*ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);
    અચલ પૂર્ણાંક ir  = ir1 - ir0;

    માટે (પૂર્ણાંક i3 = 0; i3 < n_s; ++i3) {
        માટે (પૂર્ણાંક i2 = 0; i2 < n_t; ++i2) {
            // {d_conv - 1 + n_t, d_inner, n_seqs}
            // sliding window
            અચલ દશાંશ * s = (અચલ દશાંશ *) ((અચલ અક્ષર *) src0->data + ir0*(src0->nb[1]) + i2*(src0->nb[0]) + i3*(src0->nb[2])); // {d_conv, d_inner, n_s}
            અચલ દશાંશ * c = (અચલ દશાંશ *) ((અચલ અક્ષર *) src1->data + ir0*(src1->nb[1])); // {d_conv, d_inner}
            દશાંશ * x = (દશાંશ *) ((અક્ષર *) dst->data + ir0*(dst->nb[0]) + i2*(dst->nb[1]) + i3*(dst->nb[2])); // {d_inner, n_t, n_s}

            // TODO: transpose the output for smaller strides for big batches?
            // d_inner
            માટે (પૂર્ણાંક i1 = 0; i1 < ir; ++i1) {
                // rowwise dot product
                // NOTE: not using ggml_vec_dot_f32, because its sum is in double precision
                દશાંશ sumf = 0.0f;

                // d_conv
                માટે (પૂર્ણાંક i0 = 0; i0 < nc; ++i0) {
                    sumf += s[i0 + i1*ncs] * c[i0 + i1*nc];
                }
                x[i1] = sumf;
            }
        }
    }
}

શૂન્ય ggml_compute_forward_ssm_conv(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {
    પસંદગી (dst->src[0]->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_ssm_conv_f32(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_ssm_scan

સ્થિર શૂન્ય ggml_compute_forward_ssm_scan_f32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {
    અચલ ggml_tensor * src0 = dst->src[0]; // s  {d_state, dim, n_head, n_seqs+}
    અચલ ggml_tensor * src1 = dst->src[1]; // x  {dim, n_head, n_seq_tokens, n_seqs}
    અચલ ggml_tensor * src2 = dst->src[2]; // dt {n_head, n_seq_tokens, n_seqs}
    અચલ ggml_tensor * src3 = dst->src[3]; // A  {d_state, n_head} or {1, n_head}
    અચલ ggml_tensor * src4 = dst->src[4]; // B  {d_state, n_group, n_seq_tokens, n_seqs}
    અચલ ggml_tensor * src5 = dst->src[5]; // C  {d_state, n_group, n_seq_tokens, n_seqs}
    અચલ ggml_tensor * src6 = dst->src[6]; // ids {n_seqs}

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ int64_t nc = src0->ne[0]; // d_state
    અચલ int64_t nr = src0->ne[1]; // dim
    અચલ int64_t nh = src1->ne[1]; // n_head
    અચલ int64_t ng = src4->ne[1];
    અચલ int64_t nt = src1->ne[2]; // number of tokens per sequence
    અચલ int64_t ns = src1->ne[3]; // number of sequences in the batch
    અચલ int64_t K  = ggml_get_op_params_i32(dst, 0);

    // can't use ggml_nbytes because src1 is not necessarily contiguous
    અચલ int64_t s_off = ggml_nelements(src1) * ggml_element_size(src1);

    GGML_ASSERT(K >= 1);
    GGML_ASSERT(ggml_nelements(src1) + K*nc*nr*nh*ns == ggml_nelements(dst));
    GGML_ASSERT(src0->nb[0] == કદ(દશાંશ));
    GGML_ASSERT(src1->nb[0] == કદ(દશાંશ));
    GGML_ASSERT(src2->nb[0] == કદ(દશાંશ));
    GGML_ASSERT(src3->nb[0] == કદ(દશાંશ));
    GGML_ASSERT(src4->nb[0] == કદ(દશાંશ));
    GGML_ASSERT(src5->nb[0] == કદ(દશાંશ));
    GGML_ASSERT(src6->nb[0] == કદ(int32_t));
    GGML_ASSERT(nh % ng == 0);
    GGML_ASSERT(src3->ne[0] == 1 || K == 1);

    // heads per thread
    અચલ પૂર્ણાંક dh = (nh + nth - 1)/nth;

    // head range for this thread
    અચલ પૂર્ણાંક ih0 = dh*ith;
    અચલ પૂર્ણાંક ih1 = MIN(ih0 + dh, nh);

    અચલ int32_t * ids = (અચલ int32_t *) src6->data;

    માટે (પૂર્ણાંક i3 = 0; i3 < ns; ++i3) {
        અચલ દશાંશ * s0 = (અચલ દશાંશ *) ((અચલ અક્ષર *) src0->data + ids[i3]*(src0->nb[3])); // {d_state, dim, nh, ns}
              દશાંશ * s  = (      દશાંશ *) ((      અક્ષર *) dst->data  + i3*(src0->nb[3]) + s_off); // {d_state, dim, nh, ns}

        માટે (પૂર્ણાંક i2 = 0; i2 < nt; ++i2) {
            અચલ દશાંશ * x  = (અચલ દશાંશ *) ((અચલ અક્ષર *) src1->data + i2*(src1->nb[2]) + i3*(src1->nb[3])); // {dim, nh, nt, ns}
            અચલ દશાંશ * dt = (અચલ દશાંશ *) ((અચલ અક્ષર *) src2->data + i2*(src2->nb[1]) + i3*(src2->nb[2])); // {nh, nt, ns}
            અચલ દશાંશ * A  = (અચલ દશાંશ *) ((અચલ અક્ષર *) src3->data); // {d_state, nh} or {1, nh}
            અચલ દશાંશ * B  = (અચલ દશાંશ *) ((અચલ અક્ષર *) src4->data + i2*(src4->nb[2]) + i3*(src4->nb[3])); // {d_state, ng, nt, ns}
            અચલ દશાંશ * C  = (અચલ દશાંશ *) ((અચલ અક્ષર *) src5->data + i2*(src5->nb[2]) + i3*(src5->nb[3])); // {d_state, ng, nt, ns}
                  દશાંશ * y  = (      દશાંશ *) ((      અક્ષર *) dst->data + i2*(nh*nr*કદ(દશાંશ)) + i3*(nt*nh*nr*કદ(દશાંશ))); // {dim, nh, nt, ns}

            જો (src3->ne[0] == 1) {
                // Mamba-2 has a scalar decay factor per head; dA can be outside the state-wise loop

                // n_head
                માટે (પૂર્ણાંક h = ih0; h < ih1; ++h) {
                    // ref: https://github.com/state-spaces/mamba/blob/62db608da60f6fc790b8ed9f4b3225e95ca15fde/mamba_ssm/ops/triton/softplus.py#L16
                    અચલ દશાંશ dt_soft_plus = ggml_compute_softplus_f32(dt[h]);
                    અચલ દશાંશ dA = expf(dt_soft_plus * A[h]);
                    અચલ પૂર્ણાંક g = h / (nh / ng); // repeat_interleave

                    // dim
                    માટે (પૂર્ણાંક i1 = 0; i1 < nr; ++i1) {
                        અચલ પૂર્ણાંક ii = i1 + h*nr;
                        અચલ દશાંશ x_dt = x[ii] * dt_soft_plus;
                        દશાંશ sumf = 0.0f;
#જો defined(GGML_SIMD)
    #જો defined(__ARM_FEATURE_SVE)
                        અચલ પૂર્ણાંક ggml_f32_epr = svcntw();
                        અચલ પૂર્ણાંક ggml_f32_step = 1 * ggml_f32_epr;

                        અચલ પૂર્ણાંક np = (nc & ~(ggml_f32_step - 1));

                        GGML_F32_VEC sum = GGML_F32_VEC_ZERO;

                        GGML_F32_VEC adA = GGML_F32_VEC_SET1(dA);
                        GGML_F32_VEC axdt = GGML_F32_VEC_SET1(x_dt);

                        માટે (પૂર્ણાંક i = 0; i < np; i += ggml_f32_step) {
                            // TODO: maybe unroll more?
                            માટે (પૂર્ણાંક j = 0; j < 1; j++) {
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
    #અથવાજો defined(__riscv_v_intrinsic)
                        // todo: RVV implementation
                        અચલ પૂર્ણાંક np = 0;
    #નહીંતર
                        અચલ પૂર્ણાંક np = (nc & ~(GGML_F32_STEP - 1));

                        GGML_F32_VEC sum[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };

                        GGML_F32_VEC adA = GGML_F32_VEC_SET1(dA);
                        GGML_F32_VEC axdt = GGML_F32_VEC_SET1(x_dt);

                        GGML_F32_VEC ax[GGML_F32_ARR];
                        GGML_F32_VEC ay[GGML_F32_ARR];
                        GGML_F32_VEC az[GGML_F32_ARR];

                        માટે (પૂર્ણાંક i = 0; i < np; i += GGML_F32_STEP) {
                            માટે (પૂર્ણાંક j = 0; j < GGML_F32_ARR; j++) {
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
#નહીંતર
                        અચલ પૂર્ણાંક np = 0;
#endif
                        // d_state
                        માટે (પૂર્ણાંક i0 = np; i0 < nc; ++i0) {
                            અચલ પૂર્ણાંક i = i0 + ii*nc;
                            અચલ પૂર્ણાંક ig = i0 + g*nc;
                            // state = prev_state * dA + dB * x
                            અચલ દશાંશ state = (s0[i] * dA) + (B[ig] * x_dt);
                            // y = rowwise_dotprod(state, C)
                            sumf += state * C[ig];
                            s[i] = state;
                        }
                        y[ii] = sumf;
                    }
                }
            } નહીંતર {
                // Mamba-1 has an element-wise decay factor for the states

                // n_head
                માટે (પૂર્ણાંક h = ih0; h < ih1; ++h) {
                    // ref: https://github.com/state-spaces/mamba/blob/62db608da60f6fc790b8ed9f4b3225e95ca15fde/mamba_ssm/ops/triton/softplus.py#L16
                    અચલ દશાંશ dt_soft_plus = ggml_compute_softplus_f32(dt[h]);
                    અચલ પૂર્ણાંક g = h / (nh / ng); // repeat_interleave

                    // dim
                    માટે (પૂર્ણાંક i1 = 0; i1 < nr; ++i1) {
                        અચલ પૂર્ણાંક ii = i1 + h*nr;
                        અચલ દશાંશ x_dt = x[ii] * dt_soft_plus;
#જો defined(__ARM_FEATURE_SVE)
                        svfloat32_t vx_dt = GGML_F32_VEC_SET1(x_dt);
                        svfloat32_t vdt_soft_plus = GGML_F32_VEC_SET1(dt_soft_plus);
                        svfloat32_t r1_vector = GGML_F32_VEC_ZERO;

                        // d_state
                        // TODO: what happens when (d_state % svcntw()) != 0?
                        માટે (int64_t k = 0; k < nc; k += svcntw()) {
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
#નહીંતર
                        દશાંશ sumf = 0.0f;
                        // NOTE: can't really use GGML_SIMD here because d_state is usually 16
                        //       and also because expf is used within the loop.
                        // d_state
                        માટે (પૂર્ણાંક i0 = 0; i0 < nc; ++i0) {
                            અચલ પૂર્ણાંક i = i0 + ii*nc;
                            અચલ પૂર્ણાંક ig = i0 + g*nc;
                            // state = prev_state * dA + dB * x
                            અચલ દશાંશ state = (s0[i] * expf(dt_soft_plus * A[i0 + h*nc])) + (B[ig] * x_dt);
                            // y = rowwise_dotprod(state, C)
                            sumf += state * C[ig];
                            s[i] = state;
                        }
                        y[ii] = sumf;
#endif
                    }
                }
            }
            અચલ int64_t slot = nt - 1 - i2;
            જો (K > 1 && slot > 0 && slot < K) {
                દશાંશ * s_snapshot = (દશાંશ *) ((અક્ષર *) dst->data + s_off + (slot*ns + i3)*(src0->nb[3]));
                માટે (પૂર્ણાંક h = ih0; h < ih1; ++h) {
                    memcpy((અક્ષર *) s_snapshot + h*src0->nb[2], (અક્ષર *) s + h*src0->nb[2], src0->nb[2]);
                }
            }
            // use the output as the source when it's not the first token-wise iteration
            s0 = s;
        }
    }
}

શૂન્ય ggml_compute_forward_ssm_scan(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {
    પસંદગી (dst->src[0]->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_ssm_scan_f32(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_win_part

સ્થિર શૂન્ય ggml_compute_forward_win_part_f32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {
    GGML_UNUSED(params);

    અચલ ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)

    અચલ int32_t nep0 = ((અચલ int32_t *)(dst->op_params))[0];
    અચલ int32_t nep1 = ((અચલ int32_t *)(dst->op_params))[1];
    અચલ int32_t w    = ((અચલ int32_t *)(dst->op_params))[2];

    પ્રતિજ્ઞા(ne00 == ne0);
    પ્રતિજ્ઞા(ne3  == nep0*nep1);

    // TODO: optimize / multi-thread
    માટે (પૂર્ણાંક py = 0; py < nep1; ++py) {
        માટે (પૂર્ણાંક px = 0; px < nep0; ++px) {
            અચલ int64_t i3 = py*nep0 + px;
            માટે (int64_t i2 = 0; i2 < ne2; ++i2) {
                માટે (int64_t i1 = 0; i1 < ne1; ++i1) {
                    માટે (int64_t i0 = 0; i0 < ne0; ++i0) {
                        અચલ int64_t i02 = py*w + i2;
                        અચલ int64_t i01 = px*w + i1;
                        અચલ int64_t i00 = i0;

                        અચલ int64_t i = i3*ne2*ne1*ne0 + i2*ne1*ne0    + i1*ne0   + i0;
                        અચલ int64_t j =                  i02*ne01*ne00 + i01*ne00 + i00;

                        જો (py*w + i2 >= ne02 || px*w + i1 >= ne01) {
                            ((દશાંશ *) dst->data)[i] = 0.0f;
                        } નહીંતર {
                            ((દશાંશ *) dst->data)[i] = ((દશાંશ *) src0->data)[j];
                        }
                    }
                }
            }
        }
    }
}

શૂન્ય ggml_compute_forward_win_part(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_win_part_f32(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_win_unpart

સ્થિર શૂન્ય ggml_compute_forward_win_unpart_f32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {
    GGML_UNUSED(params);

    અચલ ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)

    અચલ int32_t w = ((અચલ int32_t *)(dst->op_params))[0];

    // padding
    અચલ પૂર્ણાંક px = (w - ne1%w)%w;
    //const int py = (w - ne2%w)%w;

    અચલ પૂર્ણાંક npx = (px + ne1)/w;
    //const int npy = (py + ne2)/w;

    પ્રતિજ્ઞા(ne0 == ne00);

    // TODO: optimize / multi-thread
    માટે (int64_t i2 = 0; i2 < ne2; ++i2) {
        માટે (int64_t i1 = 0; i1 < ne1; ++i1) {
            માટે (int64_t i0 = 0; i0 < ne0; ++i0) {
                અચલ પૂર્ણાંક ip2 = i2/w;
                અચલ પૂર્ણાંક ip1 = i1/w;

                અચલ int64_t i02 = i2%w;
                અચલ int64_t i01 = i1%w;
                અચલ int64_t i00 = i0;

                અચલ int64_t i = (ip2*npx + ip1)*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00 + i00;
                અચલ int64_t j =                                  i2*ne1*ne0    + i1*ne0   + i0;

                ((દશાંશ *) dst->data)[j] = ((દશાંશ *) src0->data)[i];
            }
        }
    }
}

શૂન્ય ggml_compute_forward_win_unpart(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_win_unpart_f32(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

//ggml_compute_forward_unary

શૂન્ય ggml_compute_forward_unary(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_unary_op op = ggml_get_unary_op(dst);

    પસંદગી (op) {
        સ્થિતિ GGML_UNARY_OP_ABS:
            {
                ggml_compute_forward_abs(params, dst);
            } તોડો;
        સ્થિતિ GGML_UNARY_OP_SGN:
            {
                ggml_compute_forward_sgn(params, dst);
            } તોડો;
        સ્થિતિ GGML_UNARY_OP_NEG:
            {
                ggml_compute_forward_neg(params, dst);
            } તોડો;
        સ્થિતિ GGML_UNARY_OP_STEP:
            {
                ggml_compute_forward_step(params, dst);
            } તોડો;
        સ્થિતિ GGML_UNARY_OP_TANH:
            {
                ggml_compute_forward_tanh(params, dst);
            } તોડો;
        સ્થિતિ GGML_UNARY_OP_ELU:
            {
                ggml_compute_forward_elu(params, dst);
            } તોડો;
        સ્થિતિ GGML_UNARY_OP_RELU:
            {
                ggml_compute_forward_relu(params, dst);
            } તોડો;
        સ્થિતિ GGML_UNARY_OP_SIGMOID:
            {
                ggml_compute_forward_sigmoid(params, dst);
            } તોડો;
        સ્થિતિ GGML_UNARY_OP_GELU:
            {
                ggml_compute_forward_gelu(params, dst);
            } તોડો;
        સ્થિતિ GGML_UNARY_OP_GELU_ERF:
            {
                ggml_compute_forward_gelu_erf(params, dst);
            } તોડો;
        સ્થિતિ GGML_UNARY_OP_GELU_QUICK:
            {
                ggml_compute_forward_gelu_quick(params, dst);
            } તોડો;
        સ્થિતિ GGML_UNARY_OP_SILU:
            {
                ggml_compute_forward_silu(params, dst);
            } તોડો;
        સ્થિતિ GGML_UNARY_OP_HARDSWISH:
            {
                ggml_compute_forward_hardswish(params, dst);
            } તોડો;
        સ્થિતિ GGML_UNARY_OP_HARDSIGMOID:
            {
                ggml_compute_forward_hardsigmoid(params, dst);
            } તોડો;
        સ્થિતિ GGML_UNARY_OP_EXP:
            {
                ggml_compute_forward_exp(params, dst);
            } તોડો;
        સ્થિતિ GGML_UNARY_OP_FLOOR:
            {
                ggml_compute_forward_floor(params, dst);
            } તોડો;
        સ્થિતિ GGML_UNARY_OP_CEIL:
            {
                ggml_compute_forward_ceil(params, dst);
            } તોડો;
        સ્થિતિ GGML_UNARY_OP_ROUND:
            {
                ggml_compute_forward_round(params, dst);
            } તોડો;
        સ્થિતિ GGML_UNARY_OP_TRUNC:
            {
                ggml_compute_forward_trunc(params, dst);
            } તોડો;
        સ્થિતિ GGML_UNARY_OP_XIELU:
            {
                ggml_compute_forward_xielu(params, dst);
            } તોડો;
        સ્થિતિ GGML_UNARY_OP_EXPM1:
            {
                ggml_compute_forward_expm1(params, dst);
            } તોડો;
        સ્થિતિ GGML_UNARY_OP_SOFTPLUS:
            {
                ggml_compute_forward_softplus(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

//ggml_compute_forward_glu

શૂન્ય ggml_compute_forward_glu(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_glu_op op = ggml_get_glu_op(dst);

    પસંદગી (op) {
        સ્થિતિ GGML_GLU_OP_REGLU:
            {
                ggml_compute_forward_reglu(params, dst);
            } તોડો;
        સ્થિતિ GGML_GLU_OP_GEGLU:
            {
                ggml_compute_forward_geglu(params, dst);
            } તોડો;
        સ્થિતિ GGML_GLU_OP_SWIGLU:
            {
                ggml_compute_forward_swiglu(params, dst);
            } તોડો;
        સ્થિતિ GGML_GLU_OP_SWIGLU_OAI:
            {
                ggml_compute_forward_swiglu_oai(params, dst);
            } તોડો;
        સ્થિતિ GGML_GLU_OP_GEGLU_ERF:
            {
                ggml_compute_forward_geglu_erf(params, dst);
            } તોડો;
        સ્થિતિ GGML_GLU_OP_GEGLU_QUICK:
            {
                ggml_compute_forward_geglu_quick(params, dst);
            } તોડો;
        સ્થિતિ GGML_GLU_OP_SWIGLU_CLAMP:
            {
                ggml_compute_forward_swiglu_clamp(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_get_rel_pos

સ્થિર શૂન્ય ggml_compute_forward_get_rel_pos_f16(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {
    GGML_UNUSED(params);

    અચલ ggml_tensor * src0 = dst->src[0];

    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L292-L322

    GGML_TENSOR_UNARY_OP_LOCALS

    અચલ int64_t w = ne1;

    ggml_fp16_t * src0_data = (ggml_fp16_t *) src0->data;
    ggml_fp16_t * dst_data  = (ggml_fp16_t *) dst->data;

    માટે (int64_t i2 = 0; i2 < ne2; ++i2) {
        માટે (int64_t i1 = 0; i1 < ne1; ++i1) {
            અચલ int64_t pos = (w - i1 - 1) + i2;
            માટે (int64_t i0 = 0; i0 < ne0; ++i0) {
                dst_data[i2*ne1*ne0 + i1*ne0 + i0] = src0_data[pos*ne00 + i0];
            }
        }
    }
}

શૂન્ય ggml_compute_forward_get_rel_pos(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F16:
        સ્થિતિ GGML_TYPE_BF16:
            {
                ggml_compute_forward_get_rel_pos_f16(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_add_rel_pos

સ્થિર શૂન્ય ggml_compute_forward_add_rel_pos_f32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];
    અચલ ggml_tensor * src2 = dst->src[2];

    અચલ તાર્કિક inplace = (તાર્કિક) ((int32_t *) dst->op_params)[0];
    જો (!inplace) {
        જો (params->ith == 0) {
            memcpy((અક્ષર *) dst->data, (અક્ષર *) src0->data, ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }
    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L357-L359

    દશાંશ * src1_data = (દશાંશ *) src1->data;
    દશાંશ * src2_data = (દશાંશ *) src2->data;
    દશાંશ * dst_data  = (દશાંશ *) dst->data;

    અચલ int64_t ne10 = src1->ne[0];
    અચલ int64_t ne11 = src1->ne[1];
    અચલ int64_t ne12 = src1->ne[2];
    અચલ int64_t ne13 = src1->ne[3];

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    // total patches in dst
    અચલ પૂર્ણાંક np = ne13;

    // patches per thread
    અચલ પૂર્ણાંક dp = (np + nth - 1)/nth;

    // patch range for this thread
    અચલ પૂર્ણાંક ip0 = dp*ith;
    અચલ પૂર્ણાંક ip1 = MIN(ip0 + dp, np);

    માટે (int64_t i13 = ip0; i13 < ip1; ++i13) {
        માટે (int64_t i12 = 0; i12 < ne12; ++i12) {
            માટે (int64_t i11 = 0; i11 < ne11; ++i11) {
                અચલ int64_t jp1 = i13*ne12*ne11*ne10 + i12*ne11*ne10 + i11*ne10;
                માટે (int64_t i10 = 0; i10 < ne10; ++i10) {
                    અચલ int64_t jp0  = jp1 + i10;
                    અચલ દશાંશ src1_e = src1_data[jp0];
                    અચલ દશાંશ src2_e = src2_data[jp0];

                    અચલ int64_t jdh = jp0 * ne10;
                    અચલ int64_t jdw = jdh - (ne10 - 1) * i10;

                    માટે (int64_t j = 0; j < ne10; ++j) {
                        dst_data[jdh + j     ] += src2_e;
                        dst_data[jdw + j*ne10] += src1_e;
                    }
                }
            }
        }
    }
}

શૂન્ય ggml_compute_forward_add_rel_pos(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_add_rel_pos_f32(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_rwkv_wkv6

સ્થિર શૂન્ય ggml_compute_forward_rwkv_wkv6_f32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {
    અચલ int64_t T = dst->src[1]->ne[2];
    અચલ int64_t C = dst->ne[0];
    અચલ int64_t HEADS = dst->src[1]->ne[1];
    અચલ int64_t n_seqs = dst->src[5]->ne[1];
    અચલ int64_t head_size = C / HEADS;

    દશાંશ * dst_data = (દશાંશ *) dst->data;
    દશાંશ * state = ((દશાંશ *) dst->data) + C * T;

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક h_start =  (HEADS * (ith    )) / nth;
    અચલ પૂર્ણાંક h_end   = ((HEADS * (ith + 1)) / nth < HEADS) ?
                         (HEADS * (ith + 1)) / nth : HEADS;

    દશાંશ * k =          (દશાંશ *) dst->src[0]->data;
    દશાંશ * v =          (દશાંશ *) dst->src[1]->data;
    દશાંશ * r =          (દશાંશ *) dst->src[2]->data;
    દશાંશ * time_faaaa = (દશાંશ *) dst->src[3]->data;
    દશાંશ * time_decay = (દશાંશ *) dst->src[4]->data;

    size_t t_stride = HEADS * head_size; // Same to C

    size_t h_stride = C / HEADS;
    GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
    size_t h_stride_2d = head_size * head_size;

    જો (ith == 0) {
        memset(dst_data, 0, T * C * કદ(દશાંશ));
    }
    ggml_barrier(params->threadpool);


    #જો defined(__AVX__) && !defined(__AVX512F__)
        #define GGML_F32X GGML_F32x8
        #define GGML_F32X_SET1 GGML_F32x8_SET1
        #define GGML_F32X_LOAD GGML_F32x8_LOAD
        #define GGML_F32X_STORE GGML_F32x8_STORE
        #define GGML_F32X_MUL GGML_F32x8_MUL
        #define GGML_F32X_FMA GGML_F32x8_FMA
        #define WKV_VECTOR_SIZE 8
    #અથવાજો defined(__AVX512F__)
        #define GGML_F32X GGML_F32x16
        #define GGML_F32X_SET1 GGML_F32x16_SET1
        #define GGML_F32X_LOAD GGML_F32x16_LOAD
        #define GGML_F32X_STORE GGML_F32x16_STORE
        #define GGML_F32X_MUL GGML_F32x16_MUL
        #define GGML_F32X_FMA GGML_F32x16_FMA
        #define WKV_VECTOR_SIZE 16
    #અથવાજો defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
        #define GGML_F32X GGML_F32xt
        #define GGML_F32X_SET1 GGML_F32xt_SET1
        #define GGML_F32X_LOAD GGML_F32xt_LOAD
        #define GGML_F32X_STORE GGML_F32xt_STORE
        #define GGML_F32X_MUL GGML_F32xt_MUL
        #define GGML_F32X_FMA GGML_F32xt_FMA
        #define WKV_VECTOR_SIZE 8
    #અથવાજો defined(__ARM_NEON) && defined(__aarch64__)
        #define GGML_F32X GGML_F32x4
        #define GGML_F32X_SET1 GGML_F32x4_SET1
        #define GGML_F32X_LOAD GGML_F32x4_LOAD
        #define GGML_F32X_STORE GGML_F32x4_STORE
        #define GGML_F32X_MUL GGML_F32x4_MUL
        #define GGML_F32X_FMA GGML_F32x4_FMA
        #define WKV_VECTOR_SIZE 4
    #endif

    #ifdef WKV_VECTOR_SIZE
        પૂર્ણાંક wkv_vector_size;
        #જો defined(__ARM_FEATURE_SVE)
            wkv_vector_size = svcntw();
        #નહીંતર
            wkv_vector_size = WKV_VECTOR_SIZE;
        #endif
        અચલ int64_t vec_ગણતરી = head_size / wkv_vector_size;

        માટે (int64_t t = 0; t < T; t++) {
            size_t t_offset = t * t_stride;
            size_t state_offset = head_size * C * (t / (T / n_seqs));
            દશાંશ * state_cur = state + state_offset;
            દશાંશ * state_prev = t % (T / n_seqs) ? state_cur : (દશાંશ*)dst->src[5]->data + state_offset;

            માટે (int64_t h = h_start; h < h_end; h++) {
                size_t h_offset = h * h_stride;
                size_t t_h_offset = t_offset + h_offset;
                size_t h_2d_offset = h * h_stride_2d;

                માટે (int64_t i = 0; i < head_size; i++) {
                    size_t t_h_i_offset = t_h_offset + i;
                    size_t h_i_offset = h_offset + i;
                    size_t h_2d_i_offset = h_2d_offset + i * h_stride;

                    દશાંશ k_val = k[t_h_i_offset];
                    દશાંશ r_val = r[t_h_i_offset];
                    દશાંશ time_faaaa_val = time_faaaa[h_i_offset];
                    દશાંશ time_decay_val = time_decay[t_h_i_offset];

                    // Broadcast scalar values to vectors
                    GGML_F32X k_vec = GGML_F32X_SET1(k_val);
                    GGML_F32X r_vec = GGML_F32X_SET1(r_val);
                    GGML_F32X time_faaaa_vec = GGML_F32X_SET1(time_faaaa_val);
                    GGML_F32X time_decay_vec = GGML_F32X_SET1(time_decay_val);

                    માટે (int64_t j = 0; j < vec_ગણતરી; j++) {
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
                    માટે (int64_t j = vec_ગણતરી * wkv_vector_size; j < head_size; j++) {
                        size_t t_h_j_offset = t_h_offset + j;
                        size_t h_2d_i_j_offset = h_2d_i_offset + j;
                        દશાંશ v_val = v[t_h_j_offset];
                        દશાંશ kv_val = v_val * k_val;
                        દશાંશ prev_state_val = state_prev[h_2d_i_j_offset];
                        દશાંશ temp_val = kv_val * time_faaaa_val + prev_state_val;
                        dst_data[t_h_j_offset] += temp_val * r_val;
                        state_cur[h_2d_i_j_offset] = prev_state_val * time_decay_val + kv_val;
                    }
                }
            }
        }

    #નહીંતર
        // basically fused operations:
        // dst = r @ (time_faaaa * (k @ v) + state),
        // state = time_decay * state + (k @ v),
        // recursive through each token
        માટે (int64_t t = 0; t < T; t++) {
            size_t t_offset = t * t_stride;
            size_t state_offset = head_size * C * (t / (T / n_seqs));
            દશાંશ * state_cur = state + state_offset;
            દશાંશ * state_prev = t % (T / n_seqs) ? state_cur : (દશાંશ*)dst->src[5]->data + state_offset;

            માટે (int64_t h = h_start; h < h_end; h++) {
                size_t h_offset = h * h_stride;
                size_t t_h_offset = t_offset + h_offset;
                size_t h_2d_offset = h * h_stride_2d;

                માટે (int64_t i = 0; i < head_size; i++) {
                    size_t t_h_i_offset = t_h_offset + i;
                    size_t h_i_offset = h_offset + i;
                    size_t h_2d_i_offset = h_2d_offset + i * h_stride;

                    દશાંશ k_val = k[t_h_i_offset];
                    દશાંશ r_val = r[t_h_i_offset];
                    દશાંશ time_faaaa_val = time_faaaa[h_i_offset];
                    // RWKV v6: different time_decay for each token.
                    દશાંશ time_decay_val = time_decay[t_h_i_offset];

                    માટે (int64_t j = 0; j < head_size; j++) {
                        size_t t_h_j_offset = t_h_offset + j;
                        size_t h_2d_i_j_offset = h_2d_i_offset + j;

                        દશાંશ v_val = v[t_h_j_offset];
                        દશાંશ kv_val = v_val * k_val;
                        દશાંશ prev_state_val = state_prev[h_2d_i_j_offset];
                        દશાંશ temp_val = kv_val * time_faaaa_val + prev_state_val;
                        dst_data[t_h_j_offset] += temp_val * r_val;
                        state_cur[h_2d_i_j_offset] = prev_state_val * time_decay_val + kv_val;
                    }
                }
            }
        }
    #endif
}


શૂન્ય ggml_compute_forward_rwkv_wkv6(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_rwkv_wkv6_f32(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_gla

સ્થિર શૂન્ય ggml_compute_forward_gla_f32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {
    અચલ int64_t T = dst->src[1]->ne[2];
    અચલ int64_t C = dst->ne[0];
    અચલ int64_t HEADS = dst->src[1]->ne[1];
    અચલ int64_t n_seqs = dst->src[4]->ne[1];
    અચલ int64_t head_size = C / HEADS;
    અચલ દશાંશ scale = ggml_get_op_params_f32(dst, 0);

    દશાંશ * dst_data = (દશાંશ *) dst->data;
    દશાંશ * state = ((દશાંશ *) dst->data) + C * T;

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક h_start =  (HEADS * (ith    )) / nth;
    અચલ પૂર્ણાંક h_end   = ((HEADS * (ith + 1)) / nth < HEADS) ?
                         (HEADS * (ith + 1)) / nth : HEADS;

    દશાંશ * k = (દશાંશ *) dst->src[0]->data;
    દશાંશ * v = (દશાંશ *) dst->src[1]->data;
    દશાંશ * q = (દશાંશ *) dst->src[2]->data;
    દશાંશ * g = (દશાંશ *) dst->src[3]->data;

    size_t t_stride = HEADS * head_size; // Same to C

    size_t h_stride = C / HEADS;
    GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
    size_t h_stride_2d = head_size * head_size;

    જો (ith == 0) {
        memset(dst_data, 0, T * C * કદ(દશાંશ));
    }
    ggml_barrier(params->threadpool);


    #જો defined(__AVX__) && !defined(__AVX512F__)
        #define GGML_F32X GGML_F32x8
        #define GGML_F32X_SET1 GGML_F32x8_SET1
        #define GGML_F32X_LOAD GGML_F32x8_LOAD
        #define GGML_F32X_STORE GGML_F32x8_STORE
        #define GGML_F32X_MUL GGML_F32x8_MUL
        #define GGML_F32X_FMA GGML_F32x8_FMA
        #define GLA_VECTOR_SIZE 8
    #અથવાજો defined(__AVX512F__)
        #define GGML_F32X GGML_F32x16
        #define GGML_F32X_SET1 GGML_F32x16_SET1
        #define GGML_F32X_LOAD GGML_F32x16_LOAD
        #define GGML_F32X_STORE GGML_F32x16_STORE
        #define GGML_F32X_MUL GGML_F32x16_MUL
        #define GGML_F32X_FMA GGML_F32x16_FMA
        #define GLA_VECTOR_SIZE 16
    #અથવાજો defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
        #define GGML_F32X GGML_F32xt
        #define GGML_F32X_SET1 GGML_F32xt_SET1
        #define GGML_F32X_LOAD GGML_F32xt_LOAD
        #define GGML_F32X_STORE GGML_F32xt_STORE
        #define GGML_F32X_MUL GGML_F32xt_MUL
        #define GGML_F32X_FMA GGML_F32xt_FMA
        #define GLA_VECTOR_SIZE 8
    #અથવાજો defined(__ARM_NEON) && defined(__aarch64__)
        #define GGML_F32X GGML_F32x4
        #define GGML_F32X_SET1 GGML_F32x4_SET1
        #define GGML_F32X_LOAD GGML_F32x4_LOAD
        #define GGML_F32X_STORE GGML_F32x4_STORE
        #define GGML_F32X_MUL GGML_F32x4_MUL
        #define GGML_F32X_FMA GGML_F32x4_FMA
        #define GLA_VECTOR_SIZE 4
    #endif

    #ifdef GLA_VECTOR_SIZE
        પૂર્ણાંક gla_vector_size;
        #જો defined(__ARM_FEATURE_SVE)
            gla_vector_size = svcntw();
        #નહીંતર
            gla_vector_size = GLA_VECTOR_SIZE;
        #endif
        અચલ int64_t vec_ગણતરી = head_size / gla_vector_size;

        માટે (int64_t t = 0; t < T; t++) {
            size_t t_offset = t * t_stride;
            size_t state_offset = head_size * C * (t / (T / n_seqs));
            દશાંશ * state_cur = state + state_offset;
            દશાંશ * state_prev = t % (T / n_seqs) ? state_cur : (દશાંશ*)dst->src[4]->data + state_offset;

            માટે (int64_t h = h_start; h < h_end; h++) {
                size_t h_offset = h * h_stride;
                size_t t_h_offset = t_offset + h_offset;
                size_t h_2d_offset = h * h_stride_2d;

                માટે (int64_t i = 0; i < head_size; i++) {
                    size_t t_h_i_offset = t_h_offset + i;
                    size_t h_2d_i_offset = h_2d_offset + i * h_stride;

                    દશાંશ k_val = k[t_h_i_offset];
                    દશાંશ q_val = q[t_h_i_offset] * scale;
                    દશાંશ g_val = g[t_h_i_offset];

                    // Broadcast scalar values to vectors
                    GGML_F32X k_vec = GGML_F32X_SET1(k_val);
                    GGML_F32X q_vec = GGML_F32X_SET1(q_val);
                    GGML_F32X g_vec = GGML_F32X_SET1(g_val);

                    માટે (int64_t j = 0; j < vec_ગણતરી; j++) {
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
                    માટે (int64_t j = vec_ગણતરી * gla_vector_size; j < head_size; j++) {
                        size_t t_h_j_offset = t_h_offset + j;
                        size_t h_2d_i_j_offset = h_2d_i_offset + j;
                        દશાંશ v_val = v[t_h_j_offset];
                        દશાંશ kv_val = v_val * k_val;
                        દશાંશ prev_state_val = state_prev[h_2d_i_j_offset];
                        દશાંશ temp_val = kv_val + prev_state_val * g_val;
                        dst_data[t_h_j_offset] += temp_val * q_val;
                        state_cur[h_2d_i_j_offset] = temp_val;
                    }
                }
            }
        }

    #નહીંતર
        માટે (int64_t t = 0; t < T; t++) {
            size_t t_offset = t * t_stride;
            size_t state_offset = head_size * C * (t / (T / n_seqs));
            દશાંશ * state_cur = state + state_offset;
            દશાંશ * state_prev = t % (T / n_seqs) ? state_cur : (દશાંશ*)dst->src[4]->data + state_offset;

            માટે (int64_t h = h_start; h < h_end; h++) {
                size_t h_offset = h * h_stride;
                size_t t_h_offset = t_offset + h_offset;
                size_t h_2d_offset = h * h_stride_2d;

                માટે (int64_t i = 0; i < head_size; i++) {
                    size_t t_h_i_offset = t_h_offset + i;
                    size_t h_2d_i_offset = h_2d_offset + i * h_stride;

                    દશાંશ k_val = k[t_h_i_offset];
                    દશાંશ q_val = q[t_h_i_offset] * scale;
                    દશાંશ g_val = g[t_h_i_offset];

                    માટે (int64_t j = 0; j < head_size; j++) {
                        size_t t_h_j_offset = t_h_offset + j;
                        size_t h_2d_i_j_offset = h_2d_i_offset + j;

                        દશાંશ v_val = v[t_h_j_offset];
                        દશાંશ kv_val = v_val * k_val;
                        દશાંશ prev_state_val = state_prev[h_2d_i_j_offset];
                        દશાંશ temp_val = prev_state_val * g_val + kv_val;
                        dst_data[t_h_j_offset] += temp_val * q_val;
                        state_cur[h_2d_i_j_offset] = temp_val;
                    }
                }
            }
        }
    #endif
}


શૂન્ય ggml_compute_forward_gla(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_gla_f32(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

સ્થિર શૂન્ય ggml_compute_forward_solve_tri_f32(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst) {
    અચલ સંરચના ggml_tensor * src0 = dst->src[0];  // A (lower triangular)
    અચલ સંરચના ggml_tensor * src1 = dst->src[1];  // B (RHS)

    GGML_TENSOR_દ્વિઅંકી_OP_LOCALS;

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(ne00 == ne01); // A must be square
    GGML_ASSERT(ne0  == ne10); // solution cols == B cols
    GGML_ASSERT(ne1  == ne11); // solution rows == B rows

    GGML_ASSERT(ne02 == ne12 && ne12 == ne2);
    GGML_ASSERT(ne03 == ne13 && ne13 == ne3);

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ int64_t k = ne10;   // number of RHS columns
    અચલ int64_t n = ne11;   // A is n×n
    અચલ int64_t nr = ne02 * ne03 * k; // we're parallelizing on columns here, so seq x token x column will be the unit

    // chunks per thread
    અચલ int64_t dr = (nr + nth - 1)/nth;

    // chunk range for this thread
    અચલ int64_t ir0 = dr*ith;
    અચલ int64_t ir1 = MIN(ir0 + dr, nr);

    અચલ દશાંશ * A = (અચલ દશાંશ *) src0->data;  // [n, n, B1, B2]
    અચલ દશાંશ * B = (અચલ દશાંશ *) src1->data;  // [n, k, B1, B2]
          દશાંશ * X = (      દશાંશ *) dst->data;   // [n, k, B1, B2]

    માટે (int64_t ir = ir0; ir < ir1; ++ir) {
        અચલ int64_t i03 = ir/(ne02*k);
        અચલ int64_t i02 = (ir - i03*ne02*k)/k;
        અચલ int64_t i01 = (ir - i03*ne02*k - i02*k);

        અચલ દશાંશ * A_batch = A + i02 * nb02 / કદ(દશાંશ) + i03 * nb03 / કદ(દશાંશ);
        અચલ દશાંશ * B_batch = B + i02 * nb12 / કદ(દશાંશ) + i03 * nb13 / કદ(દશાંશ);

        દશાંશ * X_batch = X + i02 * nb2 / કદ(દશાંશ) + i03 * nb3 / કદ(દશાંશ);

        માટે (int64_t i00 = 0; i00 < n; ++i00) {
            દશાંશ sum = 0.0f;
            માટે (int64_t t = 0; t < i00; ++t) {
                sum += A_batch[i00 * n + t] * X_batch[t * k + i01];
            }

            અચલ દશાંશ diag = A_batch[i00 * n + i00];
            પ્રતિજ્ઞા(diag != 0.0f && "Zero diagonal in triangular matrix");

            X_batch[i00 * k + i01] = (B_batch[i00 * k + i01] - sum) / diag;
        }
    }
}

શૂન્ય ggml_compute_forward_solve_tri(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst) {
    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];

    જો (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        ggml_compute_forward_solve_tri_f32(params, dst);
    } નહીંતર {
        GGML_ABORT("fatal error");
    }
}

// ggml_compute_forward_gated_delta_net
સ્થિર શૂન્ય ggml_compute_forward_gated_delta_net_one_chunk(
    અચલ ggml_compute_params * params,
    ggml_tensor * dst,
    int64_t ir0,
    int64_t ir1) {

    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    અચલ int64_t S_v      = src_v->ne[0];
    અચલ int64_t H        = src_v->ne[1];
    અચલ int64_t n_tokens = src_v->ne[2];
    અચલ int64_t n_seqs   = src_v->ne[3];

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

    અચલ તાર્કિક kda = (neg0 == S_v);

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    અચલ int64_t K = ggml_get_op_params_i32(dst, 0);
    GGML_ASSERT(K >= 1);
    // per-seq stride in floats (seq s starts at state + s * seq_stride)
    અચલ int64_t state_seq_stride = src_state->nb[3] / કદ(દશાંશ);

    અચલ int64_t per_thread = S_v + (K > 1 ? S_v * S_v : 0);
    અચલ પૂર્ણાંક ith = params->ith;

    દશાંશ * delta       = (દશાંશ *)params->wdata + ith * per_thread + CACHE_લીટી_SIZE_F32;
    દશાંશ * state_work  = K > 1 ? (delta + S_v) : nullptr;

    // output layout: [attn_scores | new_states]
    // attn_scores: S_v * H * n_tokens * n_seqs    floats
    // new_states:  S_v * S_v * H * n_seqs * K     floats  (K snapshot slots; last min(n_tokens, K))
    અચલ int64_t attn_score_elems    = S_v * H * n_tokens * n_seqs;
    અચલ int64_t state_size_per_snap = S_v * S_v * H * n_seqs;
    દશાંશ * attn_out_base  = (દશાંશ *)dst->data;
    દશાંશ * state_out_base = (દશાંશ *)dst->data + attn_score_elems;

    // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
    // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.

    અચલ દશાંશ * state_in_base = (અચલ દશાંશ *)src_state->data;

  //const int64_t rq1 = nev1 / neq1;
  //const int64_t rk1 = nev1 / nek1;
    અચલ int64_t rq3 = nev3 / neq3;
    અચલ int64_t rk3 = nev3 / nek3;

    અચલ દશાંશ scale = 1.0f / sqrtf((દશાંશ) S_v);

    માટે (int64_t ir = ir0; ir < ir1; ++ir) {
        અચલ int64_t iv1 = ir % H; // head_index
        અચલ int64_t iv3 = ir / H; // sequence

        અચલ int64_t iq1 = iv1 % neq1;
        અચલ int64_t ik1 = iv1 % nek1;

        અચલ int64_t iq3 = iv3 / rq3;
        અચલ int64_t ik3 = iv3 / rk3;

        // For K=1, write directly to the single output slot to avoid an extra memcpy at the end.
        // For K>1, work in scratch and copy out per-token when the slot is in range.
        દશાંશ * s_out = (K > 1)
            ? state_work
            : state_out_base + (iv3 * H + iv1) * S_v * S_v;

        // copy input state into the working buffer and operate in-place
        // state layout [S_v, S_v, H, n_seqs]: seq iv3 starts at iv3 * state_seq_stride.
        અચલ દશાંશ * s_in = state_in_base + iv3 * state_seq_stride + iv1 * S_v * S_v;
        memcpy(s_out, s_in, S_v * S_v * કદ(દશાંશ));

        // attn output pointer for first token of this (head, seq)
        દશાંશ * attn_data = attn_out_base + (iv3 * n_tokens * H + iv1) * S_v;

        માટે (int64_t t = 0; t < n_tokens; t++) {
            અચલ દશાંશ * q_d = (અચલ દશાંશ *)((અચલ અક્ષર *)src_q->data + iq3 * nbq3 + t * nbq2 + iq1 * nbq1);
            અચલ દશાંશ * k_d = (અચલ દશાંશ *)((અચલ અક્ષર *)src_k->data + ik3 * nbk3 + t * nbk2 + ik1 * nbk1);
            અચલ દશાંશ * v_d = (અચલ દશાંશ *)((અચલ અક્ષર *)src_v->data + iv3 * nbv3 + t * nbv2 + iv1 * nbv1);

            અચલ દશાંશ beta_val = *(અચલ દશાંશ *)((અચલ અક્ષર *)src_beta->data + iv3 * nbb3 + t * nbb2 + iv1 * nbb1);
            અચલ દશાંશ * g_d    =  (અચલ દશાંશ *)((અચલ અક્ષર *)src_g->data    + iv3 * nbg3 + t * nbg2 + iv1 * nbg1);

            // state is stored transposed: s_out[j*S_v + i] = S[i][j]
            // so row j of s_out = column j of S (contiguous access)

            જો (kda) {
                // precompute exp(g) into delta scratch (reused below)
                માટે (int64_t i = 0; i < S_v; ++i) {
                    delta[i] = expf(g_d[i]);
                }
                // S[i][:] *= exp(g[i]) => for each row j of M: M[j][i] *= exp(g[i])
                માટે (int64_t j = 0; j < S_v; ++j) {
                    ggml_vec_mul_f32(S_v, &s_out[j * S_v], &s_out[j * S_v], delta);
                }
            } નહીંતર {
                ggml_vec_scale_f32(S_v * S_v, s_out, expf(g_d[0]));
            }

            // delta[j] = sum_i S[i][j] * k[i] = dot(row j of M, k)
            માટે (int64_t j = 0; j < S_v; ++j) {
                દશાંશ sum = 0.0f;
                ggml_vec_dot_f32(S_v, &sum, 0, &s_out[j * S_v], 0, k_d, 0, 1);
                delta[j] = (v_d[j] - sum) * beta_val;
            }

            // outer product: S[i][j] += k[i] * delta[j] => M[j][i] += delta[j] * k[i]
            માટે (int64_t j = 0; j < S_v; ++j) {
                ggml_vec_mad_f32(S_v, &s_out[j * S_v], k_d, delta[j]);
            }

            // attn_out[j] = sum_i S[i][j] * q[i] = dot(row j of M, q)
            માટે (int64_t j = 0; j < S_v; ++j) {
                દશાંશ sum = 0.0f;
                ggml_vec_dot_f32(S_v, &sum, 0, &s_out[j * S_v], 0, q_d, 0, 1);
                attn_data[j] = sum * scale;
            }

            attn_data += S_v * H; // advance to next token

            જો (K > 1) {
                અચલ int64_t target_slot = n_tokens - 1 - t;
                જો (target_slot >= 0 && target_slot < K) {
                    દશાંશ * curr_state_o = state_out_base + target_slot * state_size_per_snap +
                                     (iv3 * H + iv1) * S_v * S_v;
                    memcpy(curr_state_o, s_out, S_v * S_v * કદ(દશાંશ));
                }
            }
        }
    }
}


સ્થિર શૂન્ય ggml_compute_forward_gated_delta_net_f32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    ggml_tensor * V = dst->src[2];
    int64_t nr = V->ne[1] * V->ne[3];

    // disable for NUMA
    અચલ તાર્કિક disable_chunking = ggml_is_numa();

    પૂર્ણાંક nth = params->nth;
    પૂર્ણાંક ith = params->ith;

    // 4x chunks per thread
    પૂર્ણાંક nth_scaled = nth * 4;
    int64_t chunk_size = (nr + nth_scaled - 1) / nth_scaled;
    int64_t nchunk     = (nr + chunk_size - 1) / chunk_size;

    જો (nth == 1 || nchunk < nth || disable_chunking) {
      nchunk = nth;
    }

    જો (ith == 0) {
      ggml_threadpool_chunk_set(params->threadpool, nth);
    }

    ggml_barrier(params->threadpool);

    અચલ int64_t dr = (nr + nchunk - 1) / nchunk;

    પૂર્ણાંક current_chunk = ith;

    જ્યારે (current_chunk < nchunk) {
        અચલ int64_t ir0 = dr * current_chunk;
        અચલ int64_t ir1 = MIN(ir0 + dr, nr);

        ggml_compute_forward_gated_delta_net_one_chunk(params, dst, ir0, ir1);
        current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
    }
}

શૂન્ય ggml_compute_forward_gated_delta_net(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {
    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_gated_delta_net_f32(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}


// ggml_compute_forward_dsv4_hc_comb

સ્થિર શૂન્ય ggml_dsv4_hc_comb_norm_cols(દશાંશ * comb, દશાંશ eps) {
    constexpr int64_t hc = 4;

    માટે (int64_t idst = 0; idst < hc; ++idst) {
        દશાંશ sum = eps;
        માટે (int64_t isrc = 0; isrc < hc; ++isrc) {
            sum += comb[idst + hc*isrc];
        }

        અચલ દશાંશ inv_sum = 1.0f / sum;
        માટે (int64_t isrc = 0; isrc < hc; ++isrc) {
            comb[idst + hc*isrc] *= inv_sum;
        }
    }
}

સ્થિર શૂન્ય ggml_dsv4_hc_comb_norm_rows(દશાંશ * comb, દશાંશ eps) {
    constexpr int64_t hc = 4;

    માટે (int64_t isrc = 0; isrc < hc; ++isrc) {
        દશાંશ sum = eps;
        માટે (int64_t idst = 0; idst < hc; ++idst) {
            sum += comb[idst + hc*isrc];
        }

        અચલ દશાંશ inv_sum = 1.0f / sum;
        માટે (int64_t idst = 0; idst < hc; ++idst) {
            comb[idst + hc*isrc] *= inv_sum;
        }
    }
}

સ્થિર શૂન્ય ggml_compute_forward_dsv4_hc_comb_f32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {
    અચલ ggml_tensor * mixes = dst->src[0];
    અચલ ggml_tensor * scale = dst->src[1];
    અચલ ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    constexpr int64_t hc = 4;
    constexpr int64_t comb_offset = 2*hc;
    constexpr int64_t hc_mix_dim = (2 + hc)*hc;

    અચલ int64_t n_tokens = mixes->ne[1];

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

    અચલ દશાંશ eps = ggml_get_op_params_f32(dst, 0);
    અચલ int32_t n_iter = ggml_get_op_params_i32(dst, 1);
    GGML_ASSERT(n_iter > 0);

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ int64_t dr  = (n_tokens + nth - 1) / nth;
    અચલ int64_t it0 = dr * ith;
    અચલ int64_t it1 = MIN(it0 + dr, n_tokens);

    અચલ દશાંશ scale_comb = *(અચલ દશાંશ *) ((અચલ અક્ષર *) scale->data + 2*nbs0);

    માટે (int64_t it = it0; it < it1; ++it) {
        દશાંશ comb[hc*hc];

        માટે (int64_t isrc = 0; isrc < hc; ++isrc) {
            દશાંશ max = -INFINITY;
            માટે (int64_t idst = 0; idst < hc; ++idst) {
                અચલ int64_t idx = idst + hc*isrc;
                અચલ દશાંશ xv = *(અચલ દશાંશ *) ((અચલ અક્ષર *) mixes->data + (comb_offset + idx)*nbm0 + it*nbm1);
                અચલ દશાંશ bv = *(અચલ દશાંશ *) ((અચલ અક્ષર *) base->data  + (comb_offset + idx)*nbb0);
                અચલ દશાંશ v = xv * scale_comb + bv;
                comb[idx] = v;
                max = MAX(max, v);
            }

            દશાંશ sum = 0.0f;
            માટે (int64_t idst = 0; idst < hc; ++idst) {
                અચલ int64_t idx = idst + hc*isrc;
                અચલ દશાંશ v = expf(comb[idx] - max);
                comb[idx] = v;
                sum += v;
            }

            અચલ દશાંશ inv_sum = 1.0f / sum;
            માટે (int64_t idst = 0; idst < hc; ++idst) {
                અચલ int64_t idx = idst + hc*isrc;
                comb[idx] = comb[idx] * inv_sum + eps;
            }
        }

        ggml_dsv4_hc_comb_norm_cols(comb, eps);
        માટે (int32_t i = 1; i < n_iter; ++i) {
            ggml_dsv4_hc_comb_norm_rows(comb, eps);
            ggml_dsv4_hc_comb_norm_cols(comb, eps);
        }

        માટે (int64_t isrc = 0; isrc < hc; ++isrc) {
            માટે (int64_t idst = 0; idst < hc; ++idst) {
                અચલ int64_t idx = idst + hc*isrc;
                *(દશાંશ *) ((અક્ષર *) dst->data + idst*nbd0 + isrc*nbd1 + it*nbd2) = comb[idx];
            }
        }
    }
}

શૂન્ય ggml_compute_forward_dsv4_hc_comb(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {
    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_dsv4_hc_comb_f32(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_dsv4_hc_pre

સ્થિર શૂન્ય ggml_compute_forward_dsv4_hc_pre_f32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {
    અચલ ggml_tensor * x       = dst->src[0];
    અચલ ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    અચલ int64_t n_embd   = x->ne[0];
    અચલ int64_t hc       = x->ne[1];
    અચલ int64_t n_tokens = x->ne[2];

    અચલ દશાંશ scale = ggml_get_op_params_f32(dst, 0);
    અચલ તાર્કિક  gated = ggml_get_op_params_i32(dst, 1) != 0;

    GGML_ASSERT(dst->ne[0] == n_embd);
    GGML_ASSERT(dst->ne[1] == n_tokens);
    જો (gated) {
        GGML_ASSERT(weights->ne[0] == n_embd);
        GGML_ASSERT(weights->ne[1] == hc);
        GGML_ASSERT(weights->ne[2] == n_tokens);
    } નહીંતર {
        GGML_ASSERT(weights->ne[0] == hc);
        GGML_ASSERT(weights->ne[1] == n_tokens);
    }

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ int64_t nr  = n_embd * n_tokens;
    અચલ int64_t dr  = (nr + nth - 1) / nth;
    અચલ int64_t ir0 = dr * ith;
    અચલ int64_t ir1 = MIN(ir0 + dr, nr);

    માટે (int64_t ir = ir0; ir < ir1; ++ir) {
        અચલ int64_t i0 = ir % n_embd;
        અચલ int64_t it = ir / n_embd;

        દશાંશ sum = 0.0f;
        માટે (int64_t ih = 0; ih < hc; ++ih) {
            અચલ દશાંશ xv = *(અચલ દશાંશ *) ((અચલ અક્ષર *) x->data + i0*nbx0 + ih*nbx1 + it*nbx2);
            દશાંશ wv;
            જો (gated) {
                અચલ દશાંશ gv = *(અચલ દશાંશ *) ((અચલ અક્ષર *) weights->data + i0*nbw0 + ih*nbw1 + it*nbw2);
                wv = 1.0f / (1.0f + expf(-gv));
            } નહીંતર {
                wv = *(અચલ દશાંશ *) ((અચલ અક્ષર *) weights->data + ih*nbw0 + it*nbw1);
            }
            sum += xv * wv;
        }

        *(દશાંશ *) ((અક્ષર *) dst->data + i0*nbd0 + it*nbd1) = scale * sum;
    }
}

શૂન્ય ggml_compute_forward_dsv4_hc_pre(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {
    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_dsv4_hc_pre_f32(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_dsv4_hc_post

સ્થિર શૂન્ય ggml_compute_forward_dsv4_hc_post_f32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {
    અચલ ggml_tensor * x        = dst->src[0];
    અચલ ggml_tensor * residual = dst->src[1];
    અચલ ggml_tensor * post     = dst->src[2];
    અચલ ggml_tensor * comb     = dst->src[3];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    અચલ int64_t n_embd   = x->ne[0];
    અચલ int64_t n_tokens = x->ne[1];
    અચલ int64_t hc       = residual->ne[1];

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
    જો (comb) {
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

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ int64_t nr  = n_embd * hc * n_tokens;
    અચલ int64_t dr  = (nr + nth - 1) / nth;
    અચલ int64_t ir0 = dr * ith;
    અચલ int64_t ir1 = MIN(ir0 + dr, nr);

    માટે (int64_t ir = ir0; ir < ir1; ++ir) {
        અચલ int64_t i0     = ir % n_embd;
        અચલ int64_t idst   = (ir / n_embd) % hc;
        અચલ int64_t it     = ir / (n_embd * hc);

        અચલ દશાંશ xv = *(અચલ દશાંશ *) ((અચલ અક્ષર *) x->data    + i0*nbx0 + it*nbx1);
        અચલ દશાંશ pv = *(અચલ દશાંશ *) ((અચલ અક્ષર *) post->data + idst*nbp0 + it*nbp1);

        દશાંશ sum = xv * pv;
        જો (comb) {
            માટે (int64_t isrc = 0; isrc < hc; ++isrc) {
                અચલ દશાંશ rv = *(અચલ દશાંશ *) ((અચલ અક્ષર *) residual->data + i0*nbr0 + isrc*nbr1 + it*nbr2);
                અચલ દશાંશ cv = *(અચલ દશાંશ *) ((અચલ અક્ષર *) comb->data     + idst*nbc0 + isrc*nbc1 + it*nbc2);
                sum += rv * cv;
            }
        } નહીંતર {
            sum += *(અચલ દશાંશ *) ((અચલ અક્ષર *) residual->data + i0*nbr0 + idst*nbr1 + it*nbr2);
        }

        *(દશાંશ *) ((અક્ષર *) dst->data + i0*nbd0 + idst*nbd1 + it*nbd2) = sum;
    }
}

શૂન્ય ggml_compute_forward_dsv4_hc_post(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {
    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_dsv4_hc_post_f32(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_top_k_mask

શૂન્ય ggml_compute_forward_ટોચ_k_mask(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {
    અચલ ggml_tensor * mask   = dst->src[0];
    અચલ ggml_tensor * ટોચ_k  = dst->src[1];
    અચલ ggml_tensor * score  = dst->src[2];
    અચલ ggml_tensor * blocks = dst->src[3];
    અચલ ggml_tensor * extra  = dst->src[4];

    GGML_ASSERT(dst->type == mask->type);
    GGML_ASSERT(dst->nb[0] == ggml_type_size(dst->type));

    અચલ int64_t n_kv    = mask->ne[0];
    અચલ int64_t ne1     = dst->ne[1];
    અચલ int64_t ne2     = dst->ne[2];
    અચલ int64_t n_rows  = ggml_nrows(dst);
    અચલ int64_t n_ટોચ   = ટોચ_k->ne[0];
    અચલ int64_t n_blk   = blocks ? blocks->ne[0] : 1;
    અચલ int64_t n_pick  = blocks ? blocks->ne[1] : score ? score->ne[0] : n_kv;
    અચલ int64_t n_extra = extra ? extra->ne[0] : 0;

    અચલ size_t ts = ggml_type_size(dst->type);

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    માટે (int64_t ir = ith; ir < n_rows; ir += nth) {
        અચલ int64_t i1 = ir % ne1;
        અચલ int64_t i2 = (ir / ne1) % ne2;
        અચલ int64_t i3 = ir / (ne1*ne2);

        અચલ અક્ષર * mask_row = (અચલ અક્ષર *) mask->data
            + (i1 % mask->ne[1])*mask->nb[1] + (i2 % mask->ne[2])*mask->nb[2] + (i3 % mask->ne[3])*mask->nb[3];
        અક્ષર * dst_row = (અક્ષર *) dst->data + i1*dst->nb[1] + i2*dst->nb[2] + i3*dst->nb[3];

        જો (dst->type == GGML_TYPE_F16) {
            અચલ ggml_fp16_t ninf = GGML_CPU_FP32_TO_FP16(-INFINITY);
            માટે (int64_t c = 0; c < n_kv; ++c) {
                ((ggml_fp16_t *) dst_row)[c] = ninf;
            }
        } નહીંતર {
            માટે (int64_t c = 0; c < n_kv; ++c) {
                ((દશાંશ *) dst_row)[c] = -INFINITY;
            }
        }

        auto keep = [&](int32_t c) {
            જો (c >= 0 && c < n_kv) {
                memcpy(dst_row + c*ts, mask_row + c*ts, ts);
            }
        };

        અચલ અક્ષર * ટોચ_k_row = (અચલ અક્ષર *) ટોચ_k->data + i1*ટોચ_k->nb[1] + i2*ટોચ_k->nb[2] + i3*ટોચ_k->nb[3];

        માટે (int64_t j = 0; j < n_ટોચ; ++j) {
            અચલ int32_t p = *(અચલ int32_t *) (ટોચ_k_row + j*ટોચ_k->nb[0]);
            જો (p < 0 || p >= n_pick) {
                ચાલુ;
            }
            જો (score) {
                અચલ દશાંશ s = *(અચલ દશાંશ *) ((અચલ અક્ષર *) score->data
                    + p*score->nb[0] + i1*score->nb[1] + i2*score->nb[2] + i3*score->nb[3]);
                જો (!(s > -INFINITY)) {
                    ચાલુ;
                }
            }
            જો (!blocks) {
                keep(p);
                ચાલુ;
            }
            માટે (int64_t k = 0; k < n_blk; ++k) {
                keep(*(અચલ int32_t *) ((અચલ અક્ષર *) blocks->data + k*blocks->nb[0] + p*blocks->nb[1]));
            }
        }

        જો (extra) {
            અચલ અક્ષર * extra_row = (અચલ અક્ષર *) extra->data + i1*extra->nb[1] + i2*extra->nb[2] + i3*extra->nb[3];
            માટે (int64_t j = 0; j < n_extra; ++j) {
                keep(*(અચલ int32_t *) (extra_row + j*extra->nb[0]));
            }
        }
    }
}

// ggml_compute_forward_rwkv_wkv7

સ્થિર શૂન્ય ggml_compute_forward_rwkv_wkv7_f32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {
    અચલ int64_t T = dst->src[1]->ne[2];
    અચલ int64_t C = dst->ne[0];
    અચલ int64_t HEADS = dst->src[1]->ne[1];
    અચલ int64_t n_seqs = dst->src[6]->ne[1];
    અચલ int64_t head_size = C / HEADS;

    દશાંશ * dst_data = (દશાંશ *) dst->data;
    દશાંશ * state = ((દશાંશ *) dst->data) + C * T;

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક h_start =  (HEADS * (ith    )) / nth;
    અચલ પૂર્ણાંક h_end   = ((HEADS * (ith + 1)) / nth < HEADS) ?
                         (HEADS * (ith + 1)) / nth : HEADS;

    દશાંશ * r = (દશાંશ *) dst->src[0]->data;
    દશાંશ * w = (દશાંશ *) dst->src[1]->data;
    દશાંશ * k = (દશાંશ *) dst->src[2]->data;
    દશાંશ * v = (દશાંશ *) dst->src[3]->data;
    દશાંશ * a = (દશાંશ *) dst->src[4]->data;
    દશાંશ * b = (દશાંશ *) dst->src[5]->data;

    int64_t t_stride = HEADS * head_size; // Same to C

    int64_t h_stride = C / HEADS;
    GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
    int64_t h_stride_2d = head_size * head_size;

    #જો defined(GGML_SIMD)
        #જો defined(__ARM_FEATURE_SVE) || defined(__riscv_v_intrinsic)
            // scalar Route to scalar implementation       //TODO: Write SVE code and RVV code
            માટે (int64_t t = 0; t < T; t++) {
                int64_t t_offset = t * t_stride;
                int64_t state_offset = head_size * C * (t / (T / n_seqs));
                દશાંશ * state_cur = state + state_offset;
                દશાંશ * state_prev = t % (T / n_seqs) ? state_cur : (દશાંશ*)dst->src[6]->data + state_offset;

                માટે (int64_t h = h_start; h < h_end; h++) {
                    int64_t h_offset = h * h_stride;
                    int64_t t_h_offset = t_offset + h_offset;
                    int64_t h_2d_offset = h * h_stride_2d;

                    માટે (int64_t i = 0; i < head_size; i++) {
                        int64_t t_h_i_offset = t_h_offset + i;
                        int64_t h_2d_i_offset = h_2d_offset + i * h_stride;

                        દશાંશ v_val = v[t_h_i_offset];

                        દશાંશ sa = 0, result = 0;
                        માટે (int64_t j = 0; j < head_size; j++) {
                            sa += a[t_h_offset + j] * state_prev[h_2d_i_offset + j];
                        }

                        માટે (int64_t j = 0; j < head_size; j++) {
                            int64_t t_h_j_offset = t_h_offset + j;
                            int64_t h_2d_i_j_offset = h_2d_i_offset + j;

                            દશાંશ r_val = r[t_h_j_offset];
                            દશાંશ w_val = w[t_h_j_offset];
                            દશાંશ k_val = k[t_h_j_offset];
                            દશાંશ b_val = b[t_h_j_offset];
                            દશાંશ kv_val = v_val * k_val;
                            દશાંશ prev_state_val = state_prev[h_2d_i_j_offset];
                            state_cur[h_2d_i_j_offset] = prev_state_val * w_val + kv_val + sa * b_val;
                            result += state_cur[h_2d_i_j_offset] * r_val;
                        }
                        dst_data[t_h_i_offset] = result;
                    }
                }
            }
        #નહીંતર
            માટે (int64_t t = 0; t < T; t++) {
                int64_t t_offset = t * t_stride;
                int64_t state_offset = head_size * C * (t / (T / n_seqs));
                દશાંશ * state_cur = state + state_offset;
                દશાંશ * state_prev = t % (T / n_seqs) ? state_cur : (દશાંશ*)dst->src[6]->data + state_offset;

                માટે (int64_t h = h_start; h < h_end; h++) {
                    int64_t h_offset = h * h_stride;
                    int64_t t_h_offset = t_offset + h_offset;
                    int64_t h_2d_offset = h * h_stride_2d;

                    માટે (int64_t ii = 0; ii < head_size; ii++) {
                        int64_t t_h_i_offset = t_h_offset + ii;
                        int64_t h_2d_i_offset = h_2d_offset + ii * h_stride;

                        GGML_F32_VEC v_vec = GGML_F32_VEC_SET1(v[t_h_i_offset]);

                        દશાંશ sa = 0;
                        {
                            GGML_F32_VEC sum[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };
                            GGML_F32_VEC ax[GGML_F32_ARR];
                            GGML_F32_VEC ay[GGML_F32_ARR];
                            માટે (int64_t j = 0; j < head_size; j += GGML_F32_STEP) {
                                માટે (int64_t kk = 0; kk < GGML_F32_ARR; kk++) {
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
                        માટે (; j < head_size; j += GGML_F32_STEP) {
                            માટે (int64_t kk = 0; kk < GGML_F32_ARR; kk++) {
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
                        માટે (; j < head_size; j++) {
                            int64_t t_h_j_offset = t_h_offset + j;
                            int64_t h_2d_i_j_offset = h_2d_i_offset + j;

                            દશાંશ r_val = r[t_h_j_offset];
                            દશાંશ w_val = w[t_h_j_offset];
                            દશાંશ k_val = k[t_h_j_offset];
                            દશાંશ b_val = b[t_h_j_offset];
                            દશાંશ kv_val = v[t_h_i_offset] * k_val;

                            દશાંશ prev_state_val = state_prev[h_2d_i_j_offset];
                            state_cur[h_2d_i_j_offset] = prev_state_val * w_val + kv_val + sa * b_val;
                            dst_data[t_h_i_offset] += state_cur[h_2d_i_j_offset] * r_val;
                        }
                    }
                }
            }
        #endif
    #નહીંતર
        માટે (int64_t t = 0; t < T; t++) {
            int64_t t_offset = t * t_stride;
            int64_t state_offset = head_size * C * (t / (T / n_seqs));
            દશાંશ * state_cur = state + state_offset;
            દશાંશ * state_prev = t % (T / n_seqs) ? state_cur : (દશાંશ*)dst->src[6]->data + state_offset;

            માટે (int64_t h = h_start; h < h_end; h++) {
                int64_t h_offset = h * h_stride;
                int64_t t_h_offset = t_offset + h_offset;
                int64_t h_2d_offset = h * h_stride_2d;

                માટે (int64_t i = 0; i < head_size; i++) {
                    int64_t t_h_i_offset = t_h_offset + i;
                    int64_t h_2d_i_offset = h_2d_offset + i * h_stride;

                    દશાંશ v_val = v[t_h_i_offset];

                    દશાંશ sa = 0, result = 0;
                    માટે (int64_t j = 0; j < head_size; j++) {
                        sa += a[t_h_offset + j] * state_prev[h_2d_i_offset + j];
                    }

                    માટે (int64_t j = 0; j < head_size; j++) {
                        int64_t t_h_j_offset = t_h_offset + j;
                        int64_t h_2d_i_j_offset = h_2d_i_offset + j;

                        દશાંશ r_val = r[t_h_j_offset];
                        દશાંશ w_val = w[t_h_j_offset];
                        દશાંશ k_val = k[t_h_j_offset];
                        દશાંશ b_val = b[t_h_j_offset];
                        દશાંશ kv_val = v_val * k_val;
                        દશાંશ prev_state_val = state_prev[h_2d_i_j_offset];
                        state_cur[h_2d_i_j_offset] = prev_state_val * w_val + kv_val + sa * b_val;
                        result += state_cur[h_2d_i_j_offset] * r_val;
                    }
                    dst_data[t_h_i_offset] = result;
                }
            }
        }
    #endif
}


શૂન્ય ggml_compute_forward_rwkv_wkv7(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_rwkv_wkv7_f32(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_map_custom1

શૂન્ય ggml_compute_forward_map_custom1(
        અચલ ggml_compute_params * params,
              ggml_tensor * dst) {

    અચલ ggml_tensor * a = dst->src[0];

    સંરચના ggml_map_custom1_op_params p;
    memcpy(&p, dst->op_params, કદ(p));

    p.fun(dst, a, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_map_custom2

શૂન્ય ggml_compute_forward_map_custom2(
        અચલ ggml_compute_params * params,
              ggml_tensor * dst) {

    અચલ ggml_tensor * a = dst->src[0];
    અચલ ggml_tensor * b = dst->src[1];

    સંરચના ggml_map_custom2_op_params p;
    memcpy(&p, dst->op_params, કદ(p));

    p.fun(dst, a, b, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_map_custom3

શૂન્ય ggml_compute_forward_map_custom3(
        અચલ ggml_compute_params * params,
              ggml_tensor * dst) {

    અચલ ggml_tensor * a = dst->src[0];
    અચલ ggml_tensor * b = dst->src[1];
    અચલ ggml_tensor * c = dst->src[2];

    સંરચના ggml_map_custom3_op_params p;
    memcpy(&p, dst->op_params, કદ(p));

    p.fun(dst, a, b, c, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_custom

શૂન્ય ggml_compute_forward_custom(
    અચલ સંરચના ggml_compute_params * params,
          સંરચના ggml_tensor * dst) {

    સંરચના ggml_custom_op_params p;
    memcpy(&p, dst->op_params, કદ(p));

    p.fun(dst, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_cross_entropy_loss

સ્થિર શૂન્ય ggml_compute_forward_cross_entropy_loss_f32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->nb[0] == ggml_type_size(src0->type));
    GGML_ASSERT(src1->nb[0] == ggml_type_size(src1->type));
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // TODO: handle transposed/permuted matrices
    અચલ int64_t nc = src0->ne[0];
    અચલ int64_t nr = ggml_nrows(src0);

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    દશાંશ * sums =  (દશાંશ *) params->wdata;
    દશાંશ * st   = ((દશાંશ *) params->wdata) + nth + ith*nc;
    દશાંશ sum_thread = 0.0f;

    GGML_ASSERT(params->wsize >= કદ(દશાંશ) * (nth + nth * nc));

    // rows per thread
    અચલ int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ int64_t ir0 = dr*ith;
    અચલ int64_t ir1 = MIN(ir0 + dr, nr);

    માટે (int64_t i1 = ir0; i1 < ir1; ++i1) {
        અચલ દશાંશ * s0 = (અચલ દશાંશ *)((અચલ અક્ષર *) src0->data + i1*src0->nb[1]);
        અચલ દશાંશ * s1 = (અચલ દશાંશ *)((અચલ અક્ષર *) src1->data + i1*src1->nb[1]);

#ifndef NDEBUG
        માટે (int64_t i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            પ્રતિજ્ઞા(!isnan(s0[i]));
            પ્રતિજ્ઞા(!isnan(s1[i]));
        }
#endif // NDEBUG

        દશાંશ max = -INFINITY;
        ggml_vec_max_f32(nc, &max, s0);
        અચલ ggml_float sum_softmax = ggml_vec_log_soft_max_f32(nc, st, s0, max);
        પ્રતિજ્ઞા(sum_softmax >= 0.0);

        ggml_vec_add1_f32(nc, st, st, -sum_softmax);
        ggml_vec_mul_f32(nc, st, st, s1);

        દશાંશ sum_st = 0.0f;
        ggml_vec_sum_f32(nc, &sum_st, st);
        sum_thread += sum_st;

#ifndef NDEBUG
        માટે (int64_t i = 0; i < nc; ++i) {
            પ્રતિજ્ઞા(!isnan(st[i]));
            પ્રતિજ્ઞા(!isinf(st[i]));
        }
#endif // NDEBUG
    }
    sums[ith] = sum_thread;
    ggml_barrier(params->threadpool);

    જો (ith == 0) {
        દશાંશ * dp = (દશાંશ *) dst->data;
        ggml_vec_sum_f32(nth, dp, sums);
        dp[0] *= -1.0f / (દશાંશ) nr;
    }
}

શૂન્ય ggml_compute_forward_cross_entropy_loss(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_cross_entropy_loss_f32(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_cross_entropy_loss_back

સ્થિર શૂન્ય ggml_compute_forward_cross_entropy_loss_back_f32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * grad  = dst->src[0]; // gradient of forward pass output
    અચલ ggml_tensor * src0f = dst->src[1]; // src0 of forward pass
    અચલ ggml_tensor * src1f = dst->src[2]; // src1 of forward pass

    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src0f));
    GGML_ASSERT(ggml_is_contiguous(src1f));
    GGML_ASSERT(ggml_is_contiguous(grad));
    GGML_ASSERT(ggml_are_same_shape(src0f, src1f) && ggml_are_same_shape(src0f, dst));

    અચલ int64_t ith = params->ith;
    અચલ int64_t nth = params->nth;

    // TODO: handle transposed/permuted matrices
    અચલ int64_t nc = src0f->ne[0];
    અચલ int64_t nr = ggml_nrows(src0f);

    // rows per thread
    અચલ int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ int64_t ir0 = dr*ith;
    અચલ int64_t ir1 = MIN(ir0 + dr, nr);

    અચલ દશાંશ d_by_nr = ((અચલ દશાંશ *) grad->data)[0] / (દશાંશ) nr;

    માટે (int64_t i1 = ir0; i1 < ir1; i1++) {
        દશાંશ       * ds0 = (દશાંશ       *)((અક્ષર       *) dst->data   + i1*dst->nb[1]);
        અચલ દશાંશ * s0  = (અચલ દશાંશ *)((અચલ અક્ષર *) src0f->data + i1*src0f->nb[1]);
        અચલ દશાંશ * s1  = (અચલ દશાંશ *)((અચલ અક્ષર *) src1f->data + i1*src1f->nb[1]);

#ifndef NDEBUG
        માટે (int64_t i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            પ્રતિજ્ઞા(!isnan(s0[i]));
            પ્રતિજ્ઞા(!isnan(s1[i]));
        }
#endif // NDEBUG

        // soft_max
        દશાંશ max = -INFINITY;
        ggml_vec_max_f32(nc, &max, s0);
        અચલ ggml_float sum = ggml_vec_soft_max_f32(nc, ds0, s0, max);
        પ્રતિજ્ઞા(sum > 0.0);
        ggml_vec_scale_f32(nc, ds0, 1.0/sum);

        // grad(src0f) = (softmax(src0f) - src1f) * grad(cross_entropy_loss(src0f, src1f)) / nr
        ggml_vec_sub_f32(nc, ds0, ds0, s1);
        ggml_vec_scale_f32(nc, ds0, d_by_nr);

#ifndef NDEBUG
        માટે (int64_t i = 0; i < nc; ++i) {
            પ્રતિજ્ઞા(!isnan(ds0[i]));
            પ્રતિજ્ઞા(!isinf(ds0[i]));
        }
#endif // NDEBUG
    }
}

શૂન્ય ggml_compute_forward_cross_entropy_loss_back(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_cross_entropy_loss_back_f32(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

સ્થિર શૂન્ય ggml_compute_forward_opt_step_adamw_f32(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0         = dst->src[0];
    અચલ ggml_tensor * src0_grad    = dst->src[1];
    અચલ ggml_tensor * src0_grad_m  = dst->src[2];
    અચલ ggml_tensor * src0_grad_v  = dst->src[3];
    અચલ ggml_tensor * adamw_params = dst->src[4];

    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad));
    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad_m));
    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad_v));
    GGML_ASSERT(ggml_nelements(adamw_params) == 7);

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS
    GGML_ASSERT(nb00 == કદ(દશાંશ));

    // rows per thread
    અચલ પૂર્ણાંક dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr*ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    અચલ દશાંશ * adamw_params_ptr = ggml_get_data_f32(adamw_params);

    અચલ દશાંશ alpha  = adamw_params_ptr[0];
    અચલ દશાંશ beta1  = adamw_params_ptr[1];
    અચલ દશાંશ beta2  = adamw_params_ptr[2];
    અચલ દશાંશ eps    = adamw_params_ptr[3];
    અચલ દશાંશ wd     = adamw_params_ptr[4];
    અચલ દશાંશ beta1h = adamw_params_ptr[5];
    અચલ દશાંશ beta2h = adamw_params_ptr[6];
    અચલ દશાંશ keep   = 1.f - alpha * wd;
    માટે (પૂર્ણાંક ir = ir0; ir < ir1; ++ir) {
        અચલ int64_t i03 = ir/(ne02*ne01);
        અચલ int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        અચલ int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        અચલ size_t offset = i03*nb03 + i02*nb02 + i01*nb01;

        દશાંશ       * w = (દશાંશ       *) ((અક્ષર       *) src0->data        + offset); // weight
        અચલ દશાંશ * g = (અચલ દશાંશ *) ((અચલ અક્ષર *) src0_grad->data   + offset); // grad
        દશાંશ       * m = (દશાંશ       *) ((અક્ષર       *) src0_grad_m->data + offset);
        દશાંશ       * v = (દશાંશ       *) ((અક્ષર       *) src0_grad_v->data + offset);

        માટે (પૂર્ણાંક i00 = 0; i00 < ne00; ++i00) {
            m[i00] = m[i00]*beta1 +        g[i00]*(1.0f - beta1);
            v[i00] = v[i00]*beta2 + g[i00]*g[i00]*(1.0f - beta2);

            અચલ દશાંશ mh =       m[i00]*beta1h;
            અચલ દશાંશ vh = sqrtf(v[i00]*beta2h) + eps;

            // The weight decay is applied independently of the Adam momenta m and v.
            // This is NOT equivalent to l2 regularization that adds w[i00]*w[i00] to the loss.
            // See: https://arxiv.org/pdf/1711.05101v3.pdf
            w[i00] = w[i00] * keep - alpha * mh / vh;
        }
    }
}

શૂન્ય ggml_compute_forward_opt_step_adamw(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_opt_step_adamw_f32(params, dst);
            } તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error");
            }
    }
}

સ્થિર શૂન્ય ggml_compute_forward_opt_step_sgd_f32(અચલ ggml_compute_params * params, ggml_tensor * dst) {
    અચલ ggml_tensor * src0       = dst->src[0];
    અચલ ggml_tensor * src0_grad  = dst->src[1];
    અચલ ggml_tensor * sgd_params = dst->src[2];

    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad));
    GGML_ASSERT(ggml_nelements(sgd_params) == 2);

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ પૂર્ણાંક nr = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS
    GGML_ASSERT(nb00 == કદ(દશાંશ));

    // rows per thread
    અચલ પૂર્ણાંક dr = (nr + nth - 1) / nth;

    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr * ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    // using adamw param subset we care about - alpha, wd - could have a separate struct
    અચલ દશાંશ * sgd_params_ptr   = ggml_get_data_f32(sgd_params);
    અચલ દશાંશ   alpha            = sgd_params_ptr[0];
    અચલ દશાંશ   keep             = 1.f - alpha * sgd_params_ptr[1];

    માટે (પૂર્ણાંક ir = ir0; ir < ir1; ++ir) {
        અચલ int64_t i03 = ir / (ne02 * ne01);
        અચલ int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
        અચલ int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

        અચલ size_t offset = i03 * nb03 + i02 * nb02 + i01 * nb01;

        દશાંશ *       w = (દશાંશ *) ((અક્ષર *) src0->data + offset);                   // weight
        અચલ દશાંશ * g = (અચલ દશાંશ *) ((અચલ અક્ષર *) src0_grad->data + offset);  // grad

        માટે (પૂર્ણાંક i00 = 0; i00 < ne00; ++i00) {
            w[i00] = w[i00] * keep - alpha * g[i00];
        }
    }
}

શૂન્ય ggml_compute_forward_opt_step_sgd(અચલ ggml_compute_params * params, ggml_tensor * dst) {
    અચલ ggml_tensor * src0 = dst->src[0];

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_opt_step_sgd_f32(params, dst);
            }
            તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error - sgd is F32 only");
            }
    }
}

સ્થિર અંતઃરેખ દશાંશ ggml_fwht_load(અચલ દશાંશ value) {
    પાછા value;
}

સ્થિર અંતઃરેખ દશાંશ ggml_fwht_load(અચલ ggml_fp16_t value) {
    પાછા ggml_fp16_to_fp32(value);
}

template<typename src_t>
સ્થિર શૂન્ય ggml_compute_forward_fwht_impl(અચલ ggml_compute_params * params, ggml_tensor * dst) {
    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == (std::is_same_v<src_t, દશાંશ> ? GGML_TYPE_F32 : GGML_TYPE_F16));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_દ્વિઅંકી_OP_LOCALS

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ int64_t n = ne10;
    GGML_ASSERT((n & (n - 1)) == 0); // must be power of 2

    અચલ int64_t nr = ne11 * ne12 * ne13;
    અચલ int64_t rows_per_thread = (nr + nth - 1) / nth;
    અચલ int64_t start_row = ith * rows_per_thread;
    અચલ int64_t end_row = MIN(start_row + rows_per_thread, nr);

    અચલ દશાંશ scale = 1.0f / sqrtf((દશાંશ)n);

#જો defined(GGML_SIMD)
    અચલ GGML_F32_VEC v_minus_one = GGML_F32_VEC_SET1(-1.0f);
#endif

    માટે (int64_t r = start_row; r < end_row; r++) {
        અચલ int64_t i13 = r / (ne11 * ne12);
        અચલ int64_t i12 = (r - i13 * ne11 * ne12) / ne11;
        અચલ int64_t i11 = r - i13 * ne11 * ne12 - i12 * ne11;

        અચલ src_t * src_row = (અચલ src_t *) ((અચલ અક્ષર *) src1->data + i11 * nb11 + i12 * nb12 + i13 * nb13);
        દશાંશ * dst_row = (દશાંશ *) ((અક્ષર *) dst->data + i11 * nb1 + i12 * nb2 + i13 * nb3);

        માટે (int64_t j = 0; j < n; j++) {
            dst_row[j] = ggml_fwht_load(src_row[j]) * scale;
        }

        // Scalar passes
#જો defined(GGML_SIMD)
#જો defined(__ARM_FEATURE_SVE)
        અચલ પૂર્ણાંક step = svcntw();
#નહીંતર
        અચલ પૂર્ણાંક step = GGML_F32_EPR;
#endif
#નહીંતર
        અચલ પૂર્ણાંક step = n;
#endif
        માટે (int64_t len = 1; len < step && len < n; len <<= 1) {
            માટે (int64_t i = 0; i < n; i += 2 * len) {
                માટે (int64_t j = 0; j < len; j++) {
                    દશાંશ u = dst_row[i + j];
                    દશાંશ v = dst_row[i + len + j];
                    dst_row[i + j] = u + v;
                    dst_row[i + len + j] = u - v;
                }
            }
        }

        // SIMD passes using GGML_F32_VEC_* macros for multi-architecture support
#જો defined(GGML_SIMD)
        માટે (int64_t len = step; len < n; len <<= 1) {
            માટે (int64_t i = 0; i < n; i += 2 * len) {
                માટે (int64_t j = 0; j < len; j += step) {
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

શૂન્ય ggml_compute_forward_fwht(અચલ ggml_compute_params * params, ggml_tensor * dst) {
    અચલ ggml_tensor * src1 = dst->src[1];

    પસંદગી (src1->type) {
        સ્થિતિ GGML_TYPE_F32:
            {
                ggml_compute_forward_fwht_impl<દશાંશ>(params, dst);
            }
            તોડો;
        સ્થિતિ GGML_TYPE_F16:
            {
                ggml_compute_forward_fwht_impl<ggml_fp16_t>(params, dst);
            }
            તોડો;
        મૂળભૂત:
            {
                GGML_ABORT("fatal error - fwht supports F32 and F16 input");
            }
    }
}

// ggml_compute_forward_lightning_indexer

શૂન્ય ggml_compute_forward_lightning_indexer(
        અચલ ggml_compute_params * params,
        ggml_tensor * dst) {

    અચલ ggml_tensor * q = dst->src[0];
    અચલ ggml_tensor * k = dst->src[1];
    અચલ ggml_tensor * w = dst->src[2]; // weights
    અચલ ggml_tensor * m = dst->src[3]; // mask

    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(   q->type == GGML_TYPE_F32);
    GGML_ASSERT(   w->type == GGML_TYPE_F32);
    GGML_ASSERT(   m->type == GGML_TYPE_F16);

    GGML_TENSOR_LOCALS(int64_t, neq,  q, ne)
    GGML_TENSOR_LOCALS(size_t,  nbq,  q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek,  k, ne)
    GGML_TENSOR_LOCALS(size_t,  nbk,  k, nb)
    GGML_TENSOR_LOCALS(int64_t, નવું,  w, ne)
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

    અચલ પૂર્ણાંક n_embd    = q->ne[0];
    અચલ પૂર્ણાંક n_head    = q->ne[1];
    અચલ પૂર્ણાંક n_tokens  = q->ne[2];
    અચલ પૂર્ણાંક n_stream  = q->ne[3];
    અચલ પૂર્ણાંક n_kv      = k->ne[2];

    ggml_to_float_t અચલ k_to_float = ggml_get_type_traits(k->type)->to_float;
    GGML_ASSERT((k->type == GGML_TYPE_F32 || k_to_float) && "lightning indexer: unsupported K-type");

    અચલ પૂર્ણાંક nr  = n_kv;
    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    // (temporary) buffer for K converted to float
    દશાંશ * k_row_f32 = (દશાંશ *) params->wdata + ith*(1*n_embd + CACHE_લીટી_SIZE_F32);

    // rows per thread
    અચલ પૂર્ણાંક dr = (nr + nth - 1)/nth;

    // row range for this thread
    અચલ પૂર્ણાંક ir0 = dr*ith;
    અચલ પૂર્ણાંક ir1 = MIN(ir0 + dr, nr);

    માટે (પૂર્ણાંક s = 0; s < n_stream; ++s) {
        માટે (પૂર્ણાંક t = 0; t < n_tokens; ++t) {
            અચલ દશાંશ       *   w_row =       (દશાંશ *) ((અક્ષર *)   w->data + t*nbw1 +        s*nbw3);
            અચલ ggml_fp16_t *   m_row = (ggml_fp16_t *) ((અક્ષર *)   m->data + t*nbm1 + (s%nem3)*nbm3);
            દશાંશ             * dst_row =       (દશાંશ *) ((અક્ષર *) dst->data + t*nb1  +        s*nb3 );
            માટે (પૂર્ણાંક ik = ir0; ik < ir1; ++ik) {
                અક્ષર * k_row = (અક્ષર *) k->data + ik*nbk2 + s*nbk3;
                જો (k_to_float) {
                    k_to_float(k_row, k_row_f32, n_embd);
                } નહીંતર {
                    k_row_f32 = (દશાંશ *) k_row;
                }
                દશાંશ score = 0.0f;
                માટે (પૂર્ણાંક h = 0; h < n_head; ++h) {
                    // dot product of q and k for head h
                    દશાંશ qk = 0.0f;
                    અચલ દશાંશ * q_row = (દશાંશ *) ((અક્ષર *) q->data + h*nbq1 + t*nbq2 + s*nbq3);
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
