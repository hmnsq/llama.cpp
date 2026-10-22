#ଅନ୍ତର୍ଭାଵ "ops.h"

#ଅନ୍ତର୍ଭାଵ "ggml-cpu.h"
#ଅନ୍ତର୍ଭାଵ "ggml-impl.h"
#ଅନ୍ତର୍ଭାଵ "binary-ops.h"
#ଅନ୍ତର୍ଭାଵ "simd-gemm.h"
#ଅନ୍ତର୍ଭାଵ "ggml.h"
#ଅନ୍ତର୍ଭାଵ "unary-ops.h"
#ଅନ୍ତର୍ଭାଵ "vec.h"

#ଅନ୍ତର୍ଭାଵ <algorithm>
#ଅନ୍ତର୍ଭାଵ <cfloat>
#ଅନ୍ତର୍ଭାଵ <cmath>

// ggml_compute_forward_dup

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_dup_same_cont(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));
    GGML_ASSERT(src0->type == dst->type);

    ଅଚଳ size_t nb0 = ggml_type_size(src0->type);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith; // thread index
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth; // number of threads

    // parallelize by blocks
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nk = ggml_nelements(src0)/ggml_blck_size(src0->type);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nk + nth - 1) / nth;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା k0 = dr * ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା k1 = MIN(k0 + dr, nk);

    ଯଦି (k0 < k1) {
        memcpy(
            ((ଅକ୍ଷର *)  dst->data + k0*nb0),
            ((ଅକ୍ଷର *) src0->data + k0*nb0),
            (k1 - k0) * nb0);
    }
}

template<typename src_t, typename dst_t>
ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_dup_flt(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(!ggml_is_quantized(src0->type) && !ggml_is_quantized(dst->type));

    GGML_TENSOR_UNARY_OP_LOCALS

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith; // thread index
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth; // number of threads

    // parallelize by rows
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr = ne01;
    // number of rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1) / nth;
    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr * ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    // case: type & row size equal
    ଯଦି (src0->type == dst->type &&
        ne00 == ne0 &&
        nb00 == ggml_type_size(src0->type) && nb0 == ggml_type_size(dst->type)) {
        // copy by rows
        ଅଚଳ size_t rs = ne00*nb00;
        ପାଇଁ (int64_t i03 = 0; i03 < ne03; i03++) {
            ପାଇଁ (int64_t i02 = 0; i02 < ne02; i02++) {
                ପାଇଁ (int64_t i01 = ir0; i01 < ir1; i01++) {
                    memcpy(
                        ((ଅକ୍ଷର *)  dst->data + i01*nb1  + i02*nb2  + i03*nb3),
                        ((ଅକ୍ଷର *) src0->data + i01*nb01 + i02*nb02 + i03*nb03),
                        rs);
                }
            }
        }
        ଫେରନ୍ତୁ;
    }

    // case: dst tensor is contiguous
    ଯଦି (ggml_is_contiguous(dst)) {
        ଯଦି (nb00 == ଆକାର(src_t)) {
            ଯଦି constexpr (std::is_same_v<dst_t, src_t>) {
                // same type
                size_t id = 0;
                ଅଚଳ size_t rs = ne00 * nb00;
                ଅକ୍ଷର * dst_ptr = (ଅକ୍ଷର *) dst->data;

                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i03 = 0; i03 < ne03; i03++) {
                    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i02 = 0; i02 < ne02; i02++) {
                        id += rs * ir0;
                        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i01 = ir0; i01 < ir1; i01++) {
                            ଅଚଳ ଅକ୍ଷର * src0_ptr = (ଅକ୍ଷର *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
                            memcpy(dst_ptr + id, src0_ptr, rs);
                            id += rs;
                        }
                        id += rs * (ne01 - ir1);
                    }
                }
            } ଅନ୍ୟଥା {
                // casting between non-quantized types
                size_t id = 0;
                dst_t * dst_ptr = (dst_t *) dst->data;

                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i03 = 0; i03 < ne03; i03++) {
                    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i01 = ir0; i01 < ir1; i01++) {
                            ଅଚଳ src_t * src0_ptr = (src_t *) ((ଅକ୍ଷର *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
                            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i00 = 0; i00 < ne00; i00++) {
                                ଦଶମିକ tmp = type_conversion_table<src_t>::to_f32(src0_ptr[i00]);
                                dst_ptr[id] = type_conversion_table<dst_t>::from_f32(tmp);
                                id++;
                            }
                        }
                        id += ne00 * (ne01 - ir1);
                    }
                }
            }
        } ଅନ୍ୟଥା {
            //printf("%s: this is not optimal - fix me\n", __func__);

            size_t id = 0;
            dst_t * dst_ptr = (dst_t *) dst->data;

            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i03 = 0; i03 < ne03; i03++) {
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i02 = 0; i02 < ne02; i02++) {
                    id += ne00 * ir0;
                    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i01 = ir0; i01 < ir1; i01++) {
                        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i00 = 0; i00 < ne00; i00++) {
                            ଅଚଳ src_t * src0_ptr = (src_t *) ((ଅକ୍ଷର *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);

                            ଦଶମିକ tmp = type_conversion_table<src_t>::to_f32(*src0_ptr);
                            dst_ptr[id] = type_conversion_table<dst_t>::from_f32(tmp);
                            id++;
                        }
                    }
                    id += ne00 * (ne01 - ir1);
                }
            }
        }
        ଫେରନ୍ତୁ;
    }

    // dst counters
    int64_t i10 = 0;
    int64_t i11 = 0;
    int64_t i12 = 0;
    int64_t i13 = 0;

    ଯଦି constexpr (std::is_same_v<dst_t, src_t>) {
        ପାଇଁ (int64_t i03 = 0; i03 < ne03; i03++) {
            ପାଇଁ (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                ଯେତେବେଳେ (i10 >= ne0) {
                    i10 -= ne0;
                    ଯଦି (++i11 == ne1) {
                        i11 = 0;
                        ଯଦି (++i12 == ne2) {
                            i12 = 0;
                            ଯଦି (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                ପାଇଁ (int64_t i01 = ir0; i01 < ir1; i01++) {
                    ପାଇଁ (int64_t i00 = 0; i00 < ne00; i00++) {
                        ଅଚଳ ଅକ୍ଷର * src0_ptr = ((ଅକ୍ଷର *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              ଅକ୍ଷର * dst_ptr  = ((ଅକ୍ଷର *)  dst->data + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                        memcpy(dst_ptr, src0_ptr, ଆକାର(dst_t));

                        ଯଦି (++i10 == ne00) {
                            i10 = 0;
                            ଯଦି (++i11 == ne01) {
                                i11 = 0;
                                ଯଦି (++i12 == ne02) {
                                    i12 = 0;
                                    ଯଦି (++i13 == ne03) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                ଯେତେବେଳେ (i10 >= ne0) {
                    i10 -= ne0;
                    ଯଦି (++i11 == ne1) {
                        i11 = 0;
                        ଯଦି (++i12 == ne2) {
                            i12 = 0;
                            ଯଦି (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
            }
        }

    } ଅନ୍ୟଥା {
        ପାଇଁ (int64_t i03 = 0; i03 < ne03; i03++) {
            ପାଇଁ (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                ଯେତେବେଳେ (i10 >= ne0) {
                    i10 -= ne0;
                    ଯଦି (++i11 == ne1) {
                        i11 = 0;
                        ଯଦି (++i12 == ne2) {
                            i12 = 0;
                            ଯଦି (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                ପାଇଁ (int64_t i01 = ir0; i01 < ir1; i01++) {
                    ପାଇଁ (int64_t i00 = 0; i00 < ne00; i00++) {
                        ଅଚଳ ଅକ୍ଷର * src0_ptr = ((ଅକ୍ଷର *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              ଅକ୍ଷର * dst_ptr  = ((ଅକ୍ଷର *)  dst->data + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                        ଦଶମିକ tmp = type_conversion_table<src_t>::to_f32(*(ଅଚଳ src_t *) src0_ptr);
                        *(dst_t *) dst_ptr = type_conversion_table<dst_t>::from_f32(tmp);

                        ଯଦି (++i10 == ne0) {
                            i10 = 0;
                            ଯଦି (++i11 == ne1) {
                                i11 = 0;
                                ଯଦି (++i12 == ne2) {
                                    i12 = 0;
                                    ଯଦି (++i13 == ne3) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                ଯେତେବେଳେ (i10 >= ne0) {
                    i10 -= ne0;
                    ଯଦି (++i11 == ne1) {
                        i11 = 0;
                        ଯଦି (++i12 == ne2) {
                            i12 = 0;
                            ଯଦି (++i13 == ne3) {
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
ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_dup_to_q(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(!ggml_is_quantized(src0->type));

    GGML_TENSOR_UNARY_OP_LOCALS

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith; // thread index
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth; // number of threads

    // parallelize by rows
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr = ne01;
    // number of rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1) / nth;
    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr * ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ଯଦି (ggml_is_contiguous(dst) &&
            nb00 == ଆକାର(src_t) &&
            ggml_get_type_traits_cpu(dst->type)->from_float) {
        // casting non-quantized types --> intermediate f32 --> quantized
        ggml_from_float_t ଅଚଳ quantize_row_q = ggml_get_type_traits_cpu(dst->type)->from_float;
        ଦଶମିକ * src0_f32 = (ଦଶମିକ *) params->wdata + (ne00 + CACHE_ଧାଡ଼ି_SIZE_F32) * ith;

        size_t id = 0;
        size_t rs = nb0 * (ne00 / ggml_blck_size(dst->type));
        ଅକ୍ଷର * dst_ptr = (ଅକ୍ଷର *) dst->data;

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i03 = 0; i03 < ne03; i03++) {
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i02 = 0; i02 < ne02; i02++) {
                id += rs * ir0;
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i01 = ir0; i01 < ir1; i01++) {
                    ଅଚଳ src_t * src0_ptr = (src_t *) ((ଅକ୍ଷର *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);

                    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i00 = 0; i00 < ne00; i00++) {
                        src0_f32[i00] = type_conversion_table<src_t>::to_f32(src0_ptr[i00]);
                    }

                    quantize_row_q(src0_f32, dst_ptr + id, ne00);
                    id += rs;
                }
                id += rs * (ne01 - ir1);
            }
        }
    } ଅନ୍ୟଥା {
        // printf("%s %s\n", ggml_type_name(src0->type), ggml_type_name(dst->type));
        GGML_ABORT("not implemented");
    }
}

// A simplified version of ggml_compute_forward_dup that doesn't do float upcasting, and just plain old memcpy.
ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_dup_bytes(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {
    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(src0->type == dst->type);

    GGML_TENSOR_UNARY_OP_LOCALS;

    ଯଦି (ggml_is_contiguous(src0) && ggml_is_contiguous(dst)) {
        ggml_compute_forward_dup_same_cont(params, dst);
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ size_t type_size = ggml_type_size(src0->type);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith; // thread index
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth; // number of threads

    // parallelize by rows
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr = ne01;
    // number of rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1) / nth;
    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr * ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ଯଦି (src0->type == dst->type &&
        ggml_are_same_shape(src0, dst) &&
        nb00 == type_size && nb0 == type_size) {
        // copy by rows
        ଅଚଳ size_t rs = ggml_row_size(src0->type, ne00);
        ପାଇଁ (int64_t i03 = 0; i03 < ne03; i03++) {
            ପାଇଁ (int64_t i02 = 0; i02 < ne02; i02++) {
                ପାଇଁ (int64_t i01 = ir0; i01 < ir1; i01++) {
                    memcpy(
                        ((ଅକ୍ଷର *)  dst->data + i01*nb1  + i02*nb2  + i03*nb3),
                        ((ଅକ୍ଷର *) src0->data + i01*nb01 + i02*nb02 + i03*nb03),
                        rs);
                }
            }
        }
        ଫେରନ୍ତୁ;
    }

    ଯଦି (ggml_is_contiguous(dst)) {
        size_t id = 0;
        ଅକ୍ଷର * dst_ptr = (ଅକ୍ଷର *) dst->data;
        ଅଚଳ size_t rs = ne00 * type_size;

        ଯଦି (nb00 == type_size) {
            // src0 is contiguous on first dimension, copy by rows
            ପାଇଁ (int64_t i03 = 0; i03 < ne03; i03++) {
                ପାଇଁ (int64_t i02 = 0; i02 < ne02; i02++) {
                    id += rs * ir0;
                    ପାଇଁ (int64_t i01 = ir0; i01 < ir1; i01++) {
                        ଅଚଳ ଅକ୍ଷର * src0_ptr = (ଅକ୍ଷର *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
                        memcpy(dst_ptr + id, src0_ptr, rs);
                        id += rs;
                    }
                    id += rs * (ne01 - ir1);
                }
            }
        } ଅନ୍ୟଥା {
            //printf("%s: this is not optimal - fix me\n", __func__);

            ପାଇଁ (int64_t i03 = 0; i03 < ne03; i03++) {
                ପାଇଁ (int64_t i02 = 0; i02 < ne02; i02++) {
                    id += rs * ir0;
                    ପାଇଁ (int64_t i01 = ir0; i01 < ir1; i01++) {
                        ପାଇଁ (int64_t i00 = 0; i00 < ne00; i00++) {
                            ଅଚଳ ଅକ୍ଷର * src0_ptr = (ଅକ୍ଷର *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03;
                            memcpy(dst_ptr + id, src0_ptr, type_size);

                            id += type_size;
                        }
                    }
                    id += rs * (ne01 - ir1);
                }
            }
        }

        ଫେରନ୍ତୁ;
    }

    // dst counters
    int64_t k10 = 0;
    int64_t i11 = 0;
    int64_t i12 = 0;
    int64_t i13 = 0;

    // number of blocks in a row
    ଅଚଳ int64_t nk00 = ne00 / ggml_blck_size(src0->type);
    ଅଚଳ int64_t nk0  = ne0  / ggml_blck_size(dst->type);

    ପାଇଁ (int64_t i03 = 0; i03 < ne03; i03++) {
        ପାଇଁ (int64_t i02 = 0; i02 < ne02; i02++) {
            k10 += nk00 * ir0;
            ଯେତେବେଳେ (k10 >= nk0) {
                k10 -= nk0;
                ଯଦି (++i11 == ne1) {
                    i11 = 0;
                    ଯଦି (++i12 == ne2) {
                        i12 = 0;
                        ଯଦି (++i13 == ne3) {
                            i13 = 0;
                        }
                    }
                }
            }
            ପାଇଁ (int64_t i01 = ir0; i01 < ir1; i01++) {
                ପାଇଁ (int64_t k00 = 0; k00 < nk00; k00++) {
                    ଅଚଳ ଅକ୍ଷର * src0_ptr = ((ଅକ୍ଷର *) src0->data + k00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                          ଅକ୍ଷର * dst_ptr  = ((ଅକ୍ଷର *)  dst->data + k10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                    memcpy(dst_ptr, src0_ptr, type_size);

                    ଯଦି (++k10 == nk0) {
                        k10 = 0;
                        ଯଦି (++i11 == ne1) {
                            i11 = 0;
                            ଯଦି (++i12 == ne2) {
                                i12 = 0;
                                ଯଦି (++i13 == ne3) {
                                    i13 = 0;
                                }
                            }
                        }
                    }
                }
            }
            k10 += nk00 * (ne01 - ir1);
            ଯେତେବେଳେ (k10 >= nk0) {
                k10 -= nk0;
                ଯଦି (++i11 == ne1) {
                    i11 = 0;
                    ଯଦି (++i12 == ne2) {
                        i12 = 0;
                        ଯଦି (++i13 == ne3) {
                            i13 = 0;
                        }
                    }
                }
            }
        }
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_dup_from_q(
        ଅଚଳ ggml_compute_params * params,
              ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ଦ୍ୱିମିକ_OP_LOCALS

    ଅଚଳ ggml_type type = src0->type;
    ggml_to_float_t ଅଚଳ dequantize_row_q = ggml_get_type_traits(type)->to_float;

    size_t qk = ggml_blck_size(type);
    ଅଚଳ int64_t nr = ggml_nelements(src1) / qk;

    // destination must be contiguous in the first dimension
    GGML_ASSERT(nb10 == ggml_type_size(dst->type));
    // must either have first dimension large enough to hold a row, or fully contiguous
    GGML_ASSERT((ne10 % qk) == 0 || ggml_is_contiguous(dst));

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ପାଇଁ (int64_t ir = ir0; ir < ir1; ++ir) {

        uint32_t i = ir * qk;

        ଅଚଳ int64_t i03 = i/(ne00 * ne01 * ne02);
        ଅଚଳ int64_t i02 = (i - i03*ne00*ne01*ne02 )/ (ne00*ne01);
        ଅଚଳ int64_t i01 = (i - i03*ne00*ne01*ne02  -  i02*ne01*ne00) / ne00;
        ଅଚଳ int64_t i00 = i - i03*ne00*ne01*ne02 - i02*ne01*ne00 - i01*ne00;
        ଅଚଳ int64_t x_offset = (i00/qk)*nb00 + i01*nb01 + i02*nb02 + i03 * nb03;

        ଅଚଳ int64_t i13 = i/(ne10 * ne11 * ne12);
        ଅଚଳ int64_t i12 = (i - i13*ne10*ne11*ne12) / (ne10*ne11);
        ଅଚଳ int64_t i11 = (i - i13*ne10*ne11*ne12 - i12*ne10*ne11) / ne10;
        ଅଚଳ int64_t i10 = i - i13*ne10*ne11*ne12 - i12*ne10*ne11 - i11*ne10;
        ଅଚଳ int64_t dst_offset = i10*nb10 + i11*nb11 + i12*nb12 + i13*nb13;

        dequantize_row_q(
                (ଅଚଳ ଶୂନ୍ୟ *) ((ଅକ୍ଷର *) src0->data + x_offset),
                     (ଦଶମିକ *) ((ଅକ୍ଷର *)  dst->data + dst_offset), qk);
    }
}

ଶୂନ୍ୟ ggml_compute_forward_dup(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ଯଦି (src0->type == dst->type) {
        ggml_compute_forward_dup_bytes(params, dst);
        ଫେରନ୍ତୁ;
    }

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F16:
            {
                /**/ ଯଦି (dst->type == GGML_TYPE_F16)  ggml_compute_forward_dup_flt<ggml_fp16_t, ggml_fp16_t>(params, dst);
                ଅନ୍ୟଥା ଯଦି (dst->type == GGML_TYPE_BF16) ggml_compute_forward_dup_flt<ggml_fp16_t, ggml_bf16_t>(params, dst);
                ଅନ୍ୟଥା ଯଦି (dst->type == GGML_TYPE_F32)  ggml_compute_forward_dup_flt<ggml_fp16_t, ଦଶମିକ      >(params, dst);
                ଅନ୍ୟଥା ggml_compute_forward_dup_to_q<ggml_fp16_t>(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_BF16:
            {
                /**/ ଯଦି (dst->type == GGML_TYPE_F16)  ggml_compute_forward_dup_flt<ggml_bf16_t, ggml_fp16_t>(params, dst);
                ଅନ୍ୟଥା ଯଦି (dst->type == GGML_TYPE_BF16) ggml_compute_forward_dup_flt<ggml_bf16_t, ggml_bf16_t>(params, dst);
                ଅନ୍ୟଥା ଯଦି (dst->type == GGML_TYPE_F32)  ggml_compute_forward_dup_flt<ggml_bf16_t, ଦଶମିକ      >(params, dst);
                ଅନ୍ୟଥା ggml_compute_forward_dup_to_q<ggml_bf16_t>(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                /**/ ଯଦି (dst->type == GGML_TYPE_F16)  ggml_compute_forward_dup_flt<ଦଶମିକ, ggml_fp16_t>(params, dst);
                ଅନ୍ୟଥା ଯଦି (dst->type == GGML_TYPE_BF16) ggml_compute_forward_dup_flt<ଦଶମିକ, ggml_bf16_t>(params, dst);
                ଅନ୍ୟଥା ଯଦି (dst->type == GGML_TYPE_F32)  ggml_compute_forward_dup_flt<ଦଶମିକ, ଦଶମିକ      >(params, dst);
                ଅନ୍ୟଥା ଯଦି (dst->type == GGML_TYPE_I32)  ggml_compute_forward_dup_flt<ଦଶମିକ, int32_t    >(params, dst);
                ଅନ୍ୟଥା ggml_compute_forward_dup_to_q<ଦଶମିକ>(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_I32:
            {
                ଯଦି (dst->type == GGML_TYPE_F32) ggml_compute_forward_dup_flt<int32_t, ଦଶମିକ>(params, dst);
                ଅନ୍ୟଥା GGML_ABORT("not implemented");
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                ଯଦି (ggml_is_quantized(src0->type) && dst->type == GGML_TYPE_F32) {
                    ggml_compute_forward_dup_from_q(params, dst);
                    ଭାଙ୍ଗ;
                }
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_add

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_add_q_f32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr  = ggml_nrows(src0);

    GGML_TENSOR_ଦ୍ୱିମିକ_OP_LOCALS

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ggml_type type = src0->type;
    ଅଚଳ ggml_type dtype = dst->type;
    ggml_to_float_t ଅଚଳ dequantize_row_q = ggml_get_type_traits(type)->to_float;
    ggml_from_float_t ଅଚଳ quantize_row_q = ggml_get_type_traits_cpu(dtype)->from_float;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ଆକାର(ଦଶମିକ));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ggml_is_quantized(src0->type));
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    // rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ଦଶମିକ * wdata = (ଦଶମିକ *) params->wdata + (ne00 + CACHE_ଧାଡ଼ି_SIZE_F32) * ith;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir = ir0; ir < ir1; ++ir) {
        // src0 indices
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i03 = ir/(ne02*ne01);
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i02 = (ir - i03*ne02*ne01)/ne01;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i01 = (ir - i03*ne02*ne01 - i02*ne01);

        // src1 and dst are same shape as src0 => same indices
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i13 = i03;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i12 = i02;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i11 = i01;

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i3 = i03;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i2 = i02;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = i01;

        ଶୂନ୍ୟ  * src0_row = (ଶୂନ୍ୟ *) ((ଅକ୍ଷର *) src0->data + (i01*nb01 + i02*nb02 + i03*nb03));
        ଦଶମିକ * src1_row = (ଦଶମିକ *)((ଅକ୍ଷର *) src1->data + (i11*nb11 + i12*nb12 + i13*nb13));
        ଶୂନ୍ୟ  * dst_row  = (ଶୂନ୍ୟ *) ((ଅକ୍ଷର *)  dst->data + ( i1*nb1  +  i2*nb2  +  i3*nb3));

        ପ୍ରତିଜ୍ଞା(ne00 % 32 == 0);

        // unquantize row from src0 to temp buffer
        dequantize_row_q(src0_row, wdata, ne00);
        // add src1
        ggml_vec_acc_f32(ne00, wdata, src1_row);
        // quantize row to dst
        ଯଦି (quantize_row_q != ଖାଲି) {
            quantize_row_q(wdata, dst_row, ne00);
        } ଅନ୍ୟଥା {
            memcpy(dst_row, wdata, ne0*nb0);
        }
    }
}

ଶୂନ୍ୟ ggml_compute_forward_add(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
        ସ୍ଥିତି GGML_TYPE_F16:
        ସ୍ଥିତି GGML_TYPE_BF16:
            {
                ggml_compute_forward_add_non_quantized(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q1_0:
        ସ୍ଥିତି GGML_TYPE_Q2_0:
        ସ୍ଥିତି GGML_TYPE_Q4_0:
        ସ୍ଥିତି GGML_TYPE_Q4_1:
        ସ୍ଥିତି GGML_TYPE_Q5_0:
        ସ୍ଥିତି GGML_TYPE_Q5_1:
        ସ୍ଥିତି GGML_TYPE_Q8_0:
        ସ୍ଥିତି GGML_TYPE_MXFP4:
        ସ୍ଥିତି GGML_TYPE_NVFP4:
        ସ୍ଥିତି GGML_TYPE_Q2_K:
        ସ୍ଥିତି GGML_TYPE_Q3_K:
        ସ୍ଥିତି GGML_TYPE_Q4_K:
        ସ୍ଥିତି GGML_TYPE_Q5_K:
        ସ୍ଥିତି GGML_TYPE_Q6_K:
        ସ୍ଥିତି GGML_TYPE_TQ1_0:
        ସ୍ଥିତି GGML_TYPE_TQ2_0:
        ସ୍ଥିତି GGML_TYPE_IQ2_XXS:
        ସ୍ଥିତି GGML_TYPE_IQ2_XS:
        ସ୍ଥିତି GGML_TYPE_IQ3_XXS:
        ସ୍ଥିତି GGML_TYPE_IQ1_S:
        ସ୍ଥିତି GGML_TYPE_IQ1_M:
        ସ୍ଥିତି GGML_TYPE_IQ4_NL:
        ସ୍ଥିତି GGML_TYPE_IQ4_XS:
        ସ୍ଥିତି GGML_TYPE_IQ3_S:
        ସ୍ଥିତି GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_add_q_f32(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_add_id

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_add_id_f32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];
    ଅଚଳ ggml_tensor * src2 = dst->src[2];

    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src2->type == GGML_TYPE_I32);

    GGML_ASSERT(src0->nb[0] == ଆକାର(ଦଶମିକ));
    GGML_ASSERT(src1->nb[0] == ଆକାର(ଦଶମିକ));

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr  = ggml_nrows(src0);

    GGML_TENSOR_TERNARY_OP_LOCALS

    GGML_ASSERT( nb0 == ଆକାର(ଦଶମିକ));
    GGML_ASSERT(nb10 == ଆକାର(ଦଶମିକ));

    // rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir = ir0; ir < ir1; ++ir) {
        // src0 indices
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i3 = ir/(ne2*ne1);
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i2 = (ir - i3*ne2*ne1)/ne1;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = (ir - i3*ne2*ne1 - i2*ne1);

        // src1 indices
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i11 = *(int32_t *) ((ଅକ୍ଷର *) src2->data + i1*nb20 + i2*nb21);

        GGML_ASSERT(i11 >= 0 && i11 < ne11);

        ggml_vec_add_f32(ne0,
                (ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 ),
                (ଦଶମିକ *) ((ଅକ୍ଷର *) src0->data + i3*nb03 + i2*nb02 + i1*nb01),
                (ଦଶମିକ *) ((ଅକ୍ଷର *) src1->data + i11*nb11));
    }
}

ଶୂନ୍ୟ ggml_compute_forward_add_id(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_add_id_f32(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("unsupported type for ggml_compute_forward_add_id: %s", ggml_type_name(src0->type));
            }
    }
}

// ggml_compute_forward_add1

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_add1_f32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == ଆକାର(ଦଶମିକ));
    GGML_ASSERT(nb00 == ଆକାର(ଦଶମିକ));

    // rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i3 = ir/(ne2*ne1);
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i2 = (ir - i3*ne2*ne1)/ne1;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = (ir - i3*ne2*ne1 - i2*ne1);

#ifdef GGML_USE_ACCELERATE
        GGML_UNUSED(ggml_vec_add1_f32);

        vDSP_vadd(
                (ଦଶମିକ *) ((ଅକ୍ଷର *) src0->data + i3*nb03 + i2*nb02 + i1*nb01), 1,
                (ଦଶମିକ *) ((ଅକ୍ଷର *) src1->data), 0,
                (ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 ), 1,
                ne0);
#ଅନ୍ୟଥା
        ggml_vec_add1_f32(ne0,
                (ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 ),
                (ଦଶମିକ *) ((ଅକ୍ଷର *) src0->data + i3*nb03 + i2*nb02 + i1*nb01),
               *(ଦଶମିକ *) src1->data);
#endif
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_add1_f16_f32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    ଅଚଳ ଦଶମିକ v = *(ଦଶମିକ *) src1->data;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F16);

    GGML_ASSERT( nb0 == ଆକାର(ggml_fp16_t));
    GGML_ASSERT(nb00 == ଆକାର(ggml_fp16_t));

    // rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i3 = ir/(ne2*ne1);
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i2 = (ir - i3*ne2*ne1)/ne1;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((ଅକ୍ଷର *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((ଅକ୍ଷର *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(src0_ptr[i]) + v);
        }
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_add1_f16_f16(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    ଅଚଳ ଦଶମିକ v = GGML_CPU_FP16_TO_FP32(*(ggml_fp16_t *) src1->data);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type  == GGML_TYPE_F16);

    GGML_ASSERT( nb0 == ଆକାର(ggml_fp16_t));
    GGML_ASSERT(nb00 == ଆକାର(ggml_fp16_t));

    // rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i3 = ir/(ne2*ne1);
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i2 = (ir - i3*ne2*ne1)/ne1;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((ଅକ୍ଷର *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((ଅକ୍ଷର *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(src0_ptr[i]) + v);
        }
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_add1_q_f32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    ଅଚଳ ଦଶମିକ v = *(ଦଶମିକ *) src1->data;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    ଅଚଳ ggml_type type = src0->type;
    ggml_to_float_t ଅଚଳ dequantize_row_q = ggml_get_type_traits(type)->to_float;
    ggml_from_float_t ଅଚଳ quantize_row_q = ggml_get_type_traits_cpu(type)->from_float;

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
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ଦଶମିକ * wdata = (ଦଶମିକ *) params->wdata + (ne0 + CACHE_ଧାଡ଼ି_SIZE_F32) * ith;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i3 = ir/(ne2*ne1);
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i2 = (ir - i3*ne2*ne1)/ne1;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ଶୂନ୍ୟ  * src0_row = (ଶୂନ୍ୟ *) ((ଅକ୍ଷର *) src0->data + (i1*nb01 + i2*nb02 + i3*nb03));
        ଶୂନ୍ୟ  * dst_row  = (ଶୂନ୍ୟ *) ((ଅକ୍ଷର *)  dst->data + (i1*nb1  + i2*nb2  + i3*nb0 ));

        ପ୍ରତିଜ୍ଞା(ne0 % 32 == 0);

        // unquantize row from src0 to temp buffer
        dequantize_row_q(src0_row, wdata, ne0);
        // add src1
        ggml_vec_acc1_f32(ne0, wdata, v);
        // quantize row to dst
        quantize_row_q(wdata, dst_row, ne0);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_add1_bf16_f32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    ଅଚଳ ଦଶମିକ v = *(ଦଶମିକ *) src1->data;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_BF16);

    GGML_ASSERT( nb0 == ଆକାର(ggml_bf16_t));
    GGML_ASSERT(nb00 == ଆକାର(ggml_bf16_t));

    // rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i3 = ir/(ne2*ne1);
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i2 = (ir - i3*ne2*ne1)/ne1;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_bf16_t * dst_ptr  = (ggml_bf16_t *) ((ଅକ୍ଷର *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_bf16_t * src0_ptr = (ggml_bf16_t *) ((ଅକ୍ଷର *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(src0_ptr[i]) + v);
        }
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_add1_bf16_bf16(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    ଅଚଳ ଦଶମିକ v = GGML_BF16_TO_FP32(*(ggml_bf16_t *) src1->data);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(src1->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->type  == GGML_TYPE_BF16);

    GGML_ASSERT( nb0 == ଆକାର(ggml_bf16_t));
    GGML_ASSERT(nb00 == ଆକାର(ggml_bf16_t));

    // rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i3 = ir/(ne2*ne1);
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i2 = (ir - i3*ne2*ne1)/ne1;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_bf16_t * dst_ptr  = (ggml_bf16_t *) ((ଅକ୍ଷର *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_bf16_t * src0_ptr = (ggml_bf16_t *) ((ଅକ୍ଷର *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(src0_ptr[i]) + v);
        }
    }
}

ଶୂନ୍ୟ ggml_compute_forward_add1(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_add1_f32(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F16:
            {
                ଯଦି (src1->type == GGML_TYPE_F16) {
                    ggml_compute_forward_add1_f16_f16(params, dst);
                }
                ଅନ୍ୟଥା ଯଦି (src1->type == GGML_TYPE_F32) {
                    ggml_compute_forward_add1_f16_f32(params, dst);
                }
                ଅନ୍ୟଥା {
                    GGML_ABORT("fatal error");
                }
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_BF16:
            {
                ଯଦି (src1->type == GGML_TYPE_BF16) {
                    ggml_compute_forward_add1_bf16_bf16(params, dst);
                }
                ଅନ୍ୟଥା ଯଦି (src1->type == GGML_TYPE_F32) {
                    ggml_compute_forward_add1_bf16_f32(params, dst);
                }
                ଅନ୍ୟଥା {
                    GGML_ABORT("fatal error");
                }
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q1_0:
        ସ୍ଥିତି GGML_TYPE_Q2_0:
        ସ୍ଥିତି GGML_TYPE_Q4_0:
        ସ୍ଥିତି GGML_TYPE_Q4_1:
        ସ୍ଥିତି GGML_TYPE_Q5_0:
        ସ୍ଥିତି GGML_TYPE_Q5_1:
        ସ୍ଥିତି GGML_TYPE_Q8_0:
        ସ୍ଥିତି GGML_TYPE_Q8_1:
        ସ୍ଥିତି GGML_TYPE_MXFP4:
        ସ୍ଥିତି GGML_TYPE_NVFP4:
        ସ୍ଥିତି GGML_TYPE_Q2_K:
        ସ୍ଥିତି GGML_TYPE_Q3_K:
        ସ୍ଥିତି GGML_TYPE_Q4_K:
        ସ୍ଥିତି GGML_TYPE_Q5_K:
        ସ୍ଥିତି GGML_TYPE_Q6_K:
        ସ୍ଥିତି GGML_TYPE_TQ1_0:
        ସ୍ଥିତି GGML_TYPE_TQ2_0:
        ସ୍ଥିତି GGML_TYPE_IQ2_XXS:
        ସ୍ଥିତି GGML_TYPE_IQ2_XS:
        ସ୍ଥିତି GGML_TYPE_IQ3_XXS:
        ସ୍ଥିତି GGML_TYPE_IQ1_S:
        ସ୍ଥିତି GGML_TYPE_IQ1_M:
        ସ୍ଥିତି GGML_TYPE_IQ4_NL:
        ସ୍ଥିତି GGML_TYPE_IQ4_XS:
        ସ୍ଥିତି GGML_TYPE_IQ3_S:
        ସ୍ଥିତି GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_add1_q_f32(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_acc

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_acc_f32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));

    // view src0 and dst with these strides and data offset inbytes during acc
    // nb0 is implicitly element_size because src0 and dst are contiguous
    size_t nb1     = ((int32_t *) dst->op_params)[0];
    size_t nb2     = ((int32_t *) dst->op_params)[1];
    size_t nb3     = ((int32_t *) dst->op_params)[2];
    size_t offset  = ((int32_t *) dst->op_params)[3];
    ତାର୍କିକ   inplace = (ତାର୍କିକ) ((int32_t *) dst->op_params)[4];

    ଯଦି (!inplace) {
        ଯଦି (params->ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            memcpy(
                ((ଅକ୍ଷର *)  dst->data),
                ((ଅକ୍ଷର *) src0->data),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr = ggml_nrows(src1);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nc = src1->ne[0];

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne)
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb)

    // src0 and dst as viewed during acc
    ଅଚଳ size_t nb0 = ggml_element_size(src0);

    ଅଚଳ size_t nb00 = nb0;
    ଅଚଳ size_t nb01 = nb1;
    ଅଚଳ size_t nb02 = nb2;
    ଅଚଳ size_t nb03 = nb3;

    GGML_ASSERT(offset + (ne10 == 0 ? 0 : ne10-1)*nb0  + (ne11 == 0 ? 0 : ne11-1)*nb1  + (ne12 == 0 ? 0 : ne12-1)*nb2  + (ne13 == 0 ? 0 : ne13-1)*nb3  < ggml_nbytes(dst));
    GGML_ASSERT(offset + (ne10 == 0 ? 0 : ne10-1)*nb00 + (ne11 == 0 ? 0 : ne11-1)*nb01 + (ne12 == 0 ? 0 : ne12-1)*nb02 + (ne13 == 0 ? 0 : ne13-1)*nb03 < ggml_nbytes(src0));

    GGML_ASSERT(nb10 == ଆକାର(ଦଶମିକ));

    // rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are viewed with shape of src1 and offset
        // => same indices
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i3 = ir/(ne12*ne11);
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i2 = (ir - i3*ne12*ne11)/ne11;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = (ir - i3*ne12*ne11 - i2*ne11);

#ifdef GGML_USE_ACCELERATE
        vDSP_vadd(
                (ଦଶମିକ *) ((ଅକ୍ଷର *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + offset), 1,
                (ଦଶମିକ *) ((ଅକ୍ଷର *) src1->data + i3*nb13 + i2*nb12 + i1*nb11), 1,
                (ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1  + offset), 1, nc);
#ଅନ୍ୟଥା
        ggml_vec_add_f32(nc,
                (ଦଶମିକ *) ((ଅକ୍ଷର *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + offset),
                (ଦଶମିକ *) ((ଅକ୍ଷର *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + offset),
                (ଦଶମିକ *) ((ଅକ୍ଷର *) src1->data + i3*nb13 + i2*nb12 + i1*nb11));
#endif
    }
}

ଶୂନ୍ୟ ggml_compute_forward_acc(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_acc_f32(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F16:
        ସ୍ଥିତି GGML_TYPE_BF16:
        ସ୍ଥିତି GGML_TYPE_Q1_0:
        ସ୍ଥିତି GGML_TYPE_Q2_0:
        ସ୍ଥିତି GGML_TYPE_Q4_0:
        ସ୍ଥିତି GGML_TYPE_Q4_1:
        ସ୍ଥିତି GGML_TYPE_Q5_0:
        ସ୍ଥିତି GGML_TYPE_Q5_1:
        ସ୍ଥିତି GGML_TYPE_Q8_0:
        ସ୍ଥିତି GGML_TYPE_Q8_1:
        ସ୍ଥିତି GGML_TYPE_MXFP4:
        ସ୍ଥିତି GGML_TYPE_NVFP4:
        ସ୍ଥିତି GGML_TYPE_Q2_K:
        ସ୍ଥିତି GGML_TYPE_Q3_K:
        ସ୍ଥିତି GGML_TYPE_Q4_K:
        ସ୍ଥିତି GGML_TYPE_Q5_K:
        ସ୍ଥିତି GGML_TYPE_Q6_K:
        ସ୍ଥିତି GGML_TYPE_TQ1_0:
        ସ୍ଥିତି GGML_TYPE_TQ2_0:
        ସ୍ଥିତି GGML_TYPE_IQ2_XXS:
        ସ୍ଥିତି GGML_TYPE_IQ2_XS:
        ସ୍ଥିତି GGML_TYPE_IQ3_XXS:
        ସ୍ଥିତି GGML_TYPE_IQ1_S:
        ସ୍ଥିତି GGML_TYPE_IQ1_M:
        ସ୍ଥିତି GGML_TYPE_IQ4_NL:
        ସ୍ଥିତି GGML_TYPE_IQ4_XS:
        ସ୍ଥିତି GGML_TYPE_IQ3_S:
        ସ୍ଥିତି GGML_TYPE_IQ2_S:
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_sum

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_sum_f32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ଯଦି (params->ith != 0) {
        ଫେରନ୍ତୁ;
    }

    ପ୍ରତିଜ୍ଞା(ggml_is_scalar(dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)

    ggml_float sum = 0;

    ଯଦି (nb00 == ଆକାର(ଦଶମିକ)) {
        ggml_float row_sum = 0;
        ପାଇଁ (int64_t i03 = 0; i03 < ne03; i03++) {
            ପାଇଁ (int64_t i02 = 0; i02 < ne02; i02++) {
                ପାଇଁ (int64_t i01 = 0; i01 < ne01; i01++) {
                    ggml_vec_sum_f32_ggf(ne00,
                            &row_sum,
                            (ଦଶମିକ *) ((ଅକ୍ଷର *) src0->data + i01*nb01 + i02*nb02 + i03*nb03));
                    sum += row_sum;
                }
            }
        }
    } ଅନ୍ୟଥା {
        ପାଇଁ (int64_t i03 = 0; i03 < ne03; i03++) {
            ପାଇଁ (int64_t i02 = 0; i02 < ne02; i02++) {
                ପାଇଁ (int64_t i01 = 0; i01 < ne01; i01++) {
                    ଅଚଳ ଅକ୍ଷର * row = (ଅଚଳ ଅକ୍ଷର *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
                    ପାଇଁ (int64_t i00 = 0; i00 < ne00; i00++) {
                        sum += (ggml_float) *(ଅଚଳ ଦଶମିକ *) (row + i00*nb00);
                    }
                }
            }
        }
    }

    ((ଦଶମିକ *) dst->data)[0] = sum;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_sum_f16(
    ଅଚଳ ggml_compute_params * params,
          ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ଯଦି (params->ith != 0) {
        ଫେରନ୍ତୁ;
    }

    ପ୍ରତିଜ୍ଞା(ggml_is_scalar(dst));

    ପ୍ରତିଜ୍ଞା(src0->nb[0] == ଆକାର(ggml_fp16_t));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)

    ଦଶମିକ sum = 0;
    ଦଶମିକ row_sum = 0;

    ପାଇଁ (int64_t i03 = 0; i03 < ne03; i03++) {
        ପାଇଁ (int64_t i02 = 0; i02 < ne02; i02++) {
            ପାଇଁ (int64_t i01 = 0; i01 < ne01; i01++) {
                ggml_vec_sum_f16_ggf(ne00,
                    &row_sum,
                    (ggml_fp16_t *) ((ଅକ୍ଷର *) src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03));
                sum += row_sum;
            }
        }
    }
    ((ggml_fp16_t *) dst->data)[0] = GGML_CPU_FP32_TO_FP16(sum);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_sum_bf16(
    ଅଚଳ ggml_compute_params * params,
          ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ଯଦି (params->ith != 0) {
        ଫେରନ୍ତୁ;
    }

    ପ୍ରତିଜ୍ଞା(ggml_is_scalar(dst));

    ପ୍ରତିଜ୍ଞା(src0->nb[0] == ଆକାର(ggml_bf16_t));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)

    ଦଶମିକ sum = 0;
    ଦଶମିକ row_sum = 0;

    ପାଇଁ (int64_t i03 = 0; i03 < ne03; i03++) {
        ପାଇଁ (int64_t i02 = 0; i02 < ne02; i02++) {
            ପାଇଁ (int64_t i01 = 0; i01 < ne01; i01++) {
                ggml_vec_sum_bf16_ggf(ne00,
                    &row_sum,
                    (ggml_bf16_t *) ((ଅକ୍ଷର *) src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03));
                sum += row_sum;
            }
        }
    }
    ((ggml_bf16_t *) dst->data)[0] = GGML_FP32_TO_BF16(sum);
}

ଶୂନ୍ୟ ggml_compute_forward_sum(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_sum_f32(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F16:
            {
                ggml_compute_forward_sum_f16(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_BF16:
            {
                ggml_compute_forward_sum_bf16(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_cumsum

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_cumsum_f32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->nb[0] == ଆକାର(ଦଶମିକ));
    GGML_ASSERT(dst->nb[0] == ଆକାର(ଦଶମିକ));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne01);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    ଅଚଳ auto [ir0, ir1] = get_thread_range(params, src0);

    ପାଇଁ (int64_t ir = ir0; ir < ir1; ++ir) {
        ଅଚଳ int64_t i03 = ir/(ne02*ne01);
        ଅଚଳ int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        ଅଚଳ int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        ଦଶମିକ * src_row = (ଦଶମିକ *) ((ଅକ୍ଷର *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
        ଦଶମିକ * dst_row = (ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data  + i01*nb1  + i02*nb2  + i03*nb3);

        ggml_vec_cumsum_f32(ne00, dst_row, src_row);
    }
}

ଶୂନ୍ୟ ggml_compute_forward_cumsum(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_cumsum_f32(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_sum_rows

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_sum_rows_f32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ଯଦି (params->ith != 0) {
        ଫେରନ୍ତୁ;
    }

    GGML_ASSERT(src0->nb[0] == ଆକାର(ଦଶମିକ));
    GGML_ASSERT(dst->nb[0] == ଆକାର(ଦଶମିକ));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(ne0 == 1);
    GGML_ASSERT(ne1 == ne01);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    ପାଇଁ (int64_t i3 = 0; i3 < ne03; i3++) {
        ପାଇଁ (int64_t i2 = 0; i2 < ne02; i2++) {
            ପାଇଁ (int64_t i1 = 0; i1 < ne01; i1++) {
                ଦଶମିକ * src_row = (ଦଶମିକ *) ((ଅକ୍ଷର *) src0->data + i1*nb01 + i2*nb02 + i3*nb03);
                ଦଶମିକ * dst_row = (ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data  + i1*nb1  + i2*nb2  + i3*nb3);
                ଦଶମିକ row_sum = 0;
                ggml_vec_sum_f32(ne00, &row_sum, src_row);
                dst_row[0] = row_sum;
            }
        }
    }
}

ଶୂନ୍ୟ ggml_compute_forward_sum_rows(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_sum_rows_f32(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_mean

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_mean_f32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ଯଦି (params->ith != 0) {
        ଫେରନ୍ତୁ;
    }

    ପ୍ରତିଜ୍ଞା(src0->nb[0] == ଆକାର(ଦଶମିକ));

    GGML_TENSOR_UNARY_OP_LOCALS

    ପ୍ରତିଜ୍ଞା(ne0 == 1);
    ପ୍ରତିଜ୍ଞା(ne1 == ne01);
    ପ୍ରତିଜ୍ଞା(ne2 == ne02);
    ପ୍ରତିଜ୍ଞା(ne3 == ne03);

    GGML_UNUSED(ne0);
    GGML_UNUSED(ne1);
    GGML_UNUSED(ne2);
    GGML_UNUSED(ne3);

    ପାଇଁ (int64_t i03 = 0; i03 < ne03; i03++) {
        ପାଇଁ (int64_t i02 = 0; i02 < ne02; i02++) {
            ପାଇଁ (int64_t i01 = 0; i01 < ne01; i01++) {
                ggml_vec_sum_f32(ne00,
                        (ଦଶମିକ *) ((ଅକ୍ଷର *)  dst->data + i01*nb1  + i02*nb2  + i03*nb3),
                        (ଦଶମିକ *) ((ଅକ୍ଷର *) src0->data + i01*nb01 + i02*nb02 + i03*nb03));

                *(ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data + i01*nb1 + i02*nb2 + i03*nb3) /= (ଦଶମିକ) ne00;
            }
        }
    }
}

ଶୂନ୍ୟ ggml_compute_forward_mean(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_mean_f32(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_argmax

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_argmax_f32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ଯଦି (params->ith != 0) {
        ଫେରନ୍ତୁ;
    }

    ପ୍ରତିଜ୍ଞା(src0->nb[0] == ଆକାର(ଦଶମିକ));
    ପ୍ରତିଜ୍ଞା(dst->nb[0] == ଆକାର(ଦଶମିକ));

    ଅଚଳ int64_t ne00 = src0->ne[0];
    ଅଚଳ int64_t ne01 = src0->ne[1];

    ଅଚଳ size_t nb01 = src0->nb[1];
    ଅଚଳ size_t nb0 = dst->nb[0];

    ପାଇଁ (int64_t i1 = 0; i1 < ne01; i1++) {
        ଦଶମିକ * src = (ଦଶମିକ *) ((ଅକ୍ଷର *) src0->data + i1*nb01);
        int32_t * dst_ = (int32_t *) ((ଅକ୍ଷର *)  dst->data + i1*nb0);
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା v = 0;
        ggml_vec_argmax_f32(ne00, &v, src);
        dst_[0] = v;
    }
}

ଶୂନ୍ୟ ggml_compute_forward_argmax(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_argmax_f32(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_count_equal

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_ଗଣନା_equal_i32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ଦ୍ୱିମିକ_OP_LOCALS;

    GGML_ASSERT(src0->type == GGML_TYPE_I32);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));
    GGML_ASSERT(dst->type == GGML_TYPE_I64);

    ଅଚଳ int64_t nr = ggml_nrows(src0);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    int64_t * sums = (int64_t *) params->wdata;
    int64_t sum_thread = 0;

    // rows per thread
    ଅଚଳ int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ int64_t ir0 = dr*ith;
    ଅଚଳ int64_t ir1 = MIN(ir0 + dr, nr);

    ପାଇଁ (int64_t ir = ir0; ir < ir1; ++ir) {
        ଅଚଳ int64_t i03 =  ir                        / (ne02*ne01);
        ଅଚଳ int64_t i02 = (ir - i03*ne03)            /       ne01;
        ଅଚଳ int64_t i01 =  ir - i03*ne03 - i02*ne02;

        ଅଚଳ ଅକ୍ଷର * data0 = (ଅଚଳ ଅକ୍ଷର *) src0->data + i03*nb03 + i02*nb02 + i01*nb01;
        ଅଚଳ ଅକ୍ଷର * data1 = (ଅଚଳ ଅକ୍ଷର *) src1->data + i03*nb13 + i02*nb12 + i01*nb11;

        ପାଇଁ (int64_t i00 = 0; i00 < ne00; ++i00) {
            ଅଚଳ int32_t val0 = *((ଅଚଳ int32_t *) (data0 + i00*nb00));
            ଅଚଳ int32_t val1 = *((ଅଚଳ int32_t *) (data1 + i00*nb10));

            sum_thread += val0 == val1;
        }
    }
    ଯଦି (ith != 0) {
        sums[ith] = sum_thread;
    }
    ggml_barrier(params->threadpool);

    ଯଦି (ith != 0) {
        ଫେରନ୍ତୁ;
    }

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith_other = 1; ith_other < nth; ++ith_other) {
        sum_thread += sums[ith_other];
    }
    *((int64_t *) dst->data) = sum_thread;
}

ଶୂନ୍ୟ ggml_compute_forward_ଗଣନା_equal(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_I32:
            {
                ggml_compute_forward_ଗଣନା_equal_i32(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_repeat

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_repeat_f32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ଯଦି (params->ith != 0) {
        ଫେରନ୍ତୁ;
    }

    GGML_ASSERT(ggml_can_repeat(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    // guaranteed to be an integer due to the check in ggml_can_repeat
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr0 = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)(ne0/ne00);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr1 = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)(ne1/ne01);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr2 = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)(ne2/ne02);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr3 = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)(ne3/ne03);

    // TODO: support for transposed / permuted tensors
    GGML_ASSERT(nb0  == ଆକାର(ଦଶମିକ));
    GGML_ASSERT(nb00 == ଆକାର(ଦଶମିକ));

    // TODO: maybe this is not optimal?
    ପାଇଁ                         (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i3 = 0; i3 < nr3;  i3++) {
        ପାଇଁ                     (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k3 = 0; k3 < ne03; k3++) {
            ପାଇଁ                 (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i2 = 0; i2 < nr2;  i2++) {
                ପାଇଁ             (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k2 = 0; k2 < ne02; k2++) {
                    ପାଇଁ         (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = 0; i1 < nr1;  i1++) {
                        ପାଇଁ     (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k1 = 0; k1 < ne01; k1++) {
                            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0 = 0; i0 < nr0;  i0++) {
                                ggml_vec_cpy_f32(ne00,
                                        (ଦଶମିକ *) ((ଅକ୍ଷର *)  dst->data + (i3*ne03 + k3)*nb3  + (i2*ne02 + k2)*nb2  + (i1*ne01 + k1)*nb1  + (i0*ne00)*nb0),
                                        (ଦଶମିକ *) ((ଅକ୍ଷର *) src0->data + (          k3)*nb03 + (          k2)*nb02 + (          k1)*nb01));
                            }
                        }
                    }
                }
            }
        }
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_repeat_f16(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ଯଦି (params->ith != 0) {
        ଫେରନ୍ତୁ;
    }

    GGML_ASSERT(ggml_can_repeat(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    // guaranteed to be an integer due to the check in ggml_can_repeat
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr0 = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)(ne0/ne00);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr1 = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)(ne1/ne01);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr2 = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)(ne2/ne02);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr3 = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)(ne3/ne03);

    // TODO: support for transposed / permuted tensors
    GGML_ASSERT(nb0  == ଆକାର(ggml_fp16_t));
    GGML_ASSERT(nb00 == ଆକାର(ggml_fp16_t));

    // TODO: maybe this is not optimal?
    ପାଇଁ                         (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i3 = 0; i3 < nr3;  i3++) {
        ପାଇଁ                     (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k3 = 0; k3 < ne03; k3++) {
            ପାଇଁ                 (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i2 = 0; i2 < nr2;  i2++) {
                ପାଇଁ             (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k2 = 0; k2 < ne02; k2++) {
                    ପାଇଁ         (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = 0; i1 < nr1;  i1++) {
                        ପାଇଁ     (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k1 = 0; k1 < ne01; k1++) {
                            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0 = 0; i0 < nr0;  i0++) {
                                ggml_fp16_t * y = (ggml_fp16_t *) ((ଅକ୍ଷର *)  dst->data + (i3*ne03 + k3)*nb3  + (i2*ne02 + k2)*nb2  + (i1*ne01 + k1)*nb1  + (i0*ne00)*nb0);
                                ggml_fp16_t * x = (ggml_fp16_t *) ((ଅକ୍ଷର *) src0->data + (          k3)*nb03 + (          k2)*nb02 + (          k1)*nb01);
                                // ggml_vec_cpy_f16(ne00, y, x)
                                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < ne00; ++i) {
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

ଶୂନ୍ୟ ggml_compute_forward_repeat(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F16:
        ସ୍ଥିତି GGML_TYPE_BF16:
        ସ୍ଥିତି GGML_TYPE_I16:
            {
                ggml_compute_forward_repeat_f16(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F32:
        ସ୍ଥିତି GGML_TYPE_I32:
            {
                ggml_compute_forward_repeat_f32(params, dst);
            } ଭାଙ୍ଗ;
        // TODO: templateify the implementation and support for I64
        //       ref https://github.com/ggml-org/llama.cpp/pull/14274#discussion_r2169492225
        //case GGML_TYPE_I64:
        //    {
        //        ggml_compute_forward_repeat_i64(params, dst);
        //    } break;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_repeat_back

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_repeat_back_f32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ଯଦି (params->ith != 0) {
        ଫେରନ୍ତୁ;
    }

    GGML_ASSERT(ggml_can_repeat(dst, src0));

    GGML_TENSOR_UNARY_OP_LOCALS

    // guaranteed to be an integer due to the check in ggml_can_repeat
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr0 = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)(ne00/ne0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr1 = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)(ne01/ne1);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr2 = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)(ne02/ne2);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr3 = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)(ne03/ne3);

    // TODO: support for transposed / permuted tensors
    GGML_ASSERT(nb0  == ଆକାର(ଦଶମିକ));
    GGML_ASSERT(nb00 == ଆକାର(ଦଶମିକ));

    ଯଦି (ggml_is_contiguous(dst)) {
        ggml_vec_set_f32(ne0*ne1*ne2*ne3, (ଦଶମିକ *)dst->data, 0);
    } ଅନ୍ୟଥା {
        ପାଇଁ         (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k3 = 0; k3 < ne3; k3++) {
            ପାଇଁ     (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k2 = 0; k2 < ne2; k2++) {
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k1 = 0; k1 < ne1; k1++) {
                    ggml_vec_set_f32(ne0,
                        (ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data + k1*nb1 + k2*nb2 + k3*nb3),
                        0);
                }
            }
        }
    }

    // TODO: maybe this is not optimal?
    ପାଇଁ                         (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i3 = 0; i3 < nr3; i3++) {
        ପାଇଁ                     (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k3 = 0; k3 < ne3; k3++) {
            ପାଇଁ                 (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i2 = 0; i2 < nr2; i2++) {
                ପାଇଁ             (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k2 = 0; k2 < ne2; k2++) {
                    ପାଇଁ         (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = 0; i1 < nr1; i1++) {
                        ପାଇଁ     (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k1 = 0; k1 < ne1; k1++) {
                            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0 = 0; i0 < nr0; i0++) {
                                ggml_vec_acc_f32(ne0,
                                        (ଦଶମିକ *) ((ଅକ୍ଷର *)  dst->data + (         k3)*nb3  + (         k2)*nb2  + (         k1)*nb1),
                                        (ଦଶମିକ *) ((ଅକ୍ଷର *) src0->data + (i3*ne3 + k3)*nb03 + (i2*ne2 + k2)*nb02 + (i1*ne1 + k1)*nb01 + (i0*ne0)*nb00));
                            }
                        }
                    }
                }
            }
        }
    }
}

ଶୂନ୍ୟ ggml_compute_forward_repeat_back(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_repeat_back_f32(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_concat
ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_concat_any(
    ଅଚଳ ggml_compute_params * params,
    ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    GGML_TENSOR_ଦ୍ୱିମିକ_OP_LOCALS

    ଅଚଳ int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);
    GGML_ASSERT(ggml_is_contiguous_rows(src0));
    GGML_ASSERT(ggml_is_contiguous_rows(src1));

    int64_t o[4] = {0, 0, 0, 0};

    ଯଦି (dim == 0) {
        GGML_ASSERT(src0->ne[0] % ggml_blck_size(src0->type) == 0);
        GGML_ASSERT(src1->ne[0] % ggml_blck_size(src1->type) == 0);

        o[dim] = src0->ne[dim]/ggml_blck_size(src0->type);
    } ଅନ୍ୟଥା {
        o[dim] = src0->ne[dim];
    }

    // Region 1: copy rows from src0
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i3 = 0; i3 < ne03; i3++) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i2 = ith; i2 < ne02; i2 += nth) {
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = 0; i1 < ne01; i1++) {
                ଅଚଳ ଅକ୍ଷର * x = (ଅଚଳ ଅକ୍ଷର *) src0->data + i1*nb01 + i2*nb02 + i3*nb03;
                      ଅକ୍ଷର * y = (      ଅକ୍ଷର *) dst->data  + i1*nb1  + i2*nb2  + i3*nb3;
                memcpy(y, x, ggml_row_size(src0->type, ne00));
            }
        }
    }

    // Region 2: copy rows from src1, offset into dst by o[]
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i3 = 0; i3 < ne13; i3++) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i2 = ith; i2 < ne12; i2 += nth) {
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = 0; i1 < ne11; i1++) {
                ଅଚଳ ଅକ୍ଷର * x = (ଅଚଳ ଅକ୍ଷର *) src1->data + i1*nb11         + i2*nb12         + i3*nb13;
                      ଅକ୍ଷର * y = (      ଅକ୍ଷର *)  dst->data + (i1 + o[1])*nb1 + (i2 + o[2])*nb2 + (i3 + o[3])*nb3 + o[0]*nb0;
                memcpy(y, x, ggml_row_size(src1->type, ne10));
            }
        }
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_concat_i8(
    ଅଚଳ ggml_compute_params * params,
    ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_type_size(src0->type) == ଆକାର(int8_t));

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    GGML_TENSOR_ଦ୍ୱିମିକ_OP_LOCALS

    ଅଚଳ int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);

    int64_t o[4] = {0, 0, 0, 0};
    o[dim] = src0->ne[dim];

    ଅଚଳ int8_t * x;

    // TODO: smarter multi-theading
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i3 = 0; i3 < ne3; i3++) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i2 = ith; i2 < ne2; i2 += nth) {
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = 0; i1 < ne1; i1++) {
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0 = 0; i0 < ne0; i0++) {
                    ଯଦି (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                        x = (ଅଚଳ int8_t *) ((ଅଚଳ ଅକ୍ଷର *)src0->data + (i0       )*nb00 + (i1       )*nb01 + (i2       )*nb02 + (i3       )*nb03);
                    } ଅନ୍ୟଥା {
                        x = (ଅଚଳ int8_t *) ((ଅଚଳ ଅକ୍ଷର *)src1->data + (i0 - o[0])*nb10 + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13);
                    }

                    int8_t * y = (int8_t *)((ଅକ୍ଷର *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);

                    *y = *x;
                }
            }
        }
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_concat_f16(
    ଅଚଳ ggml_compute_params * params,
    ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_type_size(src0->type) == ଆକାର(ggml_fp16_t));

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    GGML_TENSOR_ଦ୍ୱିମିକ_OP_LOCALS

    ଅଚଳ int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);

    int64_t o[4] = {0, 0, 0, 0};
    o[dim] = src0->ne[dim];

    ଅଚଳ ggml_fp16_t * x;

    // TODO: smarter multi-theading
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i3 = 0; i3 < ne3; i3++) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i2 = ith; i2 < ne2; i2 += nth) {
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = 0; i1 < ne1; i1++) {
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0 = 0; i0 < ne0; i0++) {
                    ଯଦି (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                        x = (ଅଚଳ ggml_fp16_t *) ((ଅଚଳ ଅକ୍ଷର *)src0->data + (i0       )*nb00 + (i1       )*nb01 + (i2       )*nb02 + (i3       )*nb03);
                    } ଅନ୍ୟଥା {
                        x = (ଅଚଳ ggml_fp16_t *) ((ଅଚଳ ଅକ୍ଷର *)src1->data + (i0 - o[0])*nb10 + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13);
                    }

                    ggml_fp16_t * y = (ggml_fp16_t *)((ଅକ୍ଷର *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);

                    *y = *x;
                }
            }
        }
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_concat_f32(
    ଅଚଳ ggml_compute_params * params,
    ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_type_size(src0->type) == ଆକାର(ଦଶମିକ));

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    GGML_TENSOR_ଦ୍ୱିମିକ_OP_LOCALS

    ଅଚଳ int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);

    int64_t o[4] = {0, 0, 0, 0};
    o[dim] = src0->ne[dim];

    ଅଚଳ ଦଶମିକ * x;

    // TODO: smarter multi-theading
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i3 = 0; i3 < ne3; i3++) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i2 = ith; i2 < ne2; i2 += nth) {
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = 0; i1 < ne1; i1++) {
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0 = 0; i0 < ne0; i0++) {
                    ଯଦି (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                        x = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *)src0->data + (i0       )*nb00 + (i1       )*nb01 + (i2       )*nb02 + (i3       )*nb03);
                    } ଅନ୍ୟଥା {
                        x = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *)src1->data + (i0 - o[0])*nb10 + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13);
                    }

                    ଦଶମିକ * y = (ଦଶମିକ *)((ଅକ୍ଷର *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);

                    *y = *x;
                }
            }
        }
    }
}

ଶୂନ୍ୟ ggml_compute_forward_concat(
    ଅଚଳ ggml_compute_params * params,
    ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F16:
        ସ୍ଥିତି GGML_TYPE_BF16:
        ସ୍ଥିତି GGML_TYPE_I16:
            {
                ggml_compute_forward_concat_f16(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_I8:
            {
                ggml_compute_forward_concat_i8(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F32:
        ସ୍ଥିତି GGML_TYPE_I32:
            {
                ggml_compute_forward_concat_f32(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                ggml_compute_forward_concat_any(params, dst);
            }
    }
}

// row loop shared by the BF16 unary ops
template <ଶୂନ୍ୟ (*vec_op)(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା, ggml_bf16_t *, ଅଚଳ ggml_bf16_t *)>
ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_unary_bf16_rows(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ପ୍ରତିଜ୍ଞା(ggml_is_contiguous_rows(src0));
    ପ୍ରତିଜ୍ଞା(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nc = src0->ne[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr = ggml_nrows(src0);

    // rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir = ir0; ir < ir1; ++ir) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i3 = ir/(ne02*ne01);
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i2 = (ir - i3*ne02*ne01)/ne01;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = (ir - i3*ne02*ne01 - i2*ne01);

        vec_op(nc,
                (ggml_bf16_t *) ((ଅକ୍ଷର *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_bf16_t *) ((ଅକ୍ଷର *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));
    }
}

// ggml_compute_forward_gelu

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_gelu_f32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ପ୍ରତିଜ୍ଞା(ggml_is_contiguous_rows(src0));
    ପ୍ରତିଜ୍ଞା(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nc = src0->ne[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr = ggml_nrows(src0);

    // rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir = ir0; ir < ir1; ++ir) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i3 = ir/(ne02*ne01);
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i2 = (ir - i3*ne02*ne01)/ne01;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_f32(nc,
                (ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ଦଶମିକ *) ((ଅକ୍ଷର *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < nc; k++) {
            ଅଚଳ ଦଶମିକ x = ((ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_UNUSED(x);
            ପ୍ରତିଜ୍ଞା(!isnan(x));
            ପ୍ରତିଜ୍ଞା(!isinf(x));
        }
#endif // NDEBUG
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_gelu_f16(
    ଅଚଳ ggml_compute_params * params,
    ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ପ୍ରତିଜ୍ଞା(ggml_is_contiguous_rows(src0));
    ପ୍ରତିଜ୍ଞା(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nc = src0->ne[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr = ggml_nrows(src0);

    // rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir = ir0; ir < ir1; ++ir) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i3 = ir/(ne02*ne01);
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i2 = (ir - i3*ne02*ne01)/ne01;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_f16(nc,
                (ggml_fp16_t *) ((ଅକ୍ଷର *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((ଅକ୍ଷର *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < nc; k++) {
            ଅଚଳ ggml_fp16_t x = ((ggml_fp16_t *) ((ଅକ୍ଷର *) dst->data + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            ଅଚଳ ଦଶମିକ v = GGML_CPU_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            ପ୍ରତିଜ୍ଞା(!isnan(v));
            ପ୍ରତିଜ୍ଞା(!isinf(v));
        }
#endif // NDEBUG
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_gelu_bf16(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_rows<ggml_vec_gelu_bf16>(params, dst);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_gelu(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_gelu_f32(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F16:
            {
                ggml_compute_forward_gelu_f16(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_BF16:
            {
                ggml_compute_forward_gelu_bf16(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_fill

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_fill_f32(ଅଚଳ ggml_compute_params * params, ggml_tensor * dst) {
    ଅଚଳ ଦଶମିକ c = ggml_get_op_params_f32(dst, 0);

    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne);
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb);

    ଅଚଳ auto [ir0, ir1] = get_thread_range(params, dst);

    ପାଇଁ (int64_t ir = ir0; ir < ir1; ++ir) {
        ଅଚଳ int64_t i03 = ir/(ne2*ne1);
        ଅଚଳ int64_t i02 = (ir - i03*ne2*ne1)/ne1;
        ଅଚଳ int64_t i01 = (ir - i03*ne2*ne1 - i02*ne1);

        ଦଶମିକ * dst_ptr  = (ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data + i03*nb3 + i02*nb2 + i01*nb1);

        ggml_vec_set_f32(ne0, dst_ptr, c);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_fill_f16(ଅଚଳ ggml_compute_params * params, ggml_tensor * dst) {
    ଅଚଳ ggml_fp16_t c = GGML_CPU_FP32_TO_FP16(ggml_get_op_params_f32(dst, 0));

    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne);
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb);

    ଅଚଳ auto [ir0, ir1] = get_thread_range(params, dst);

    ପାଇଁ (int64_t ir = ir0; ir < ir1; ++ir) {
        ଅଚଳ int64_t i03 = ir/(ne2*ne1);
        ଅଚଳ int64_t i02 = (ir - i03*ne2*ne1)/ne1;
        ଅଚଳ int64_t i01 = (ir - i03*ne2*ne1 - i02*ne1);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((ଅକ୍ଷର *) dst->data + i03*nb3 + i02*nb2 + i01*nb1);

        ggml_vec_set_f16(ne0, dst_ptr, c);
    }
}

ଶୂନ୍ୟ ggml_compute_forward_fill(ଅଚଳ ggml_compute_params * params, ggml_tensor * dst) {
    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_fill_f32(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F16:
            {
                ggml_compute_forward_fill_f16(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("unsupported type for ggml_compute_forward_fill: %s", ggml_type_name(src0->type));
            }
    }
}

// ggml_compute_tri

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_tri_f32(ଅଚଳ ggml_compute_params * params, ggml_tensor * dst) {
    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ଅଚଳ ggml_tri_type ttype = (ggml_tri_type) ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_TENSOR_UNARY_OP_LOCALS

    ଅଚଳ auto [ir0, ir1] = get_thread_range(params, src0);

    ତାର୍କିକ (*bipred)(ପୂର୍ଣ୍ଣସଂଖ୍ୟା, ପୂର୍ଣ୍ଣସଂଖ୍ୟା);

    ବାଛ (ttype) {
        ସ୍ଥିତି GGML_TRI_TYPE_LOWER:      bipred = [](ପୂର୍ଣ୍ଣସଂଖ୍ୟା i, ପୂର୍ଣ୍ଣସଂଖ୍ୟା r) { ଫେରନ୍ତୁ i <  r; }; ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TRI_TYPE_LOWER_DIAG: bipred = [](ପୂର୍ଣ୍ଣସଂଖ୍ୟା i, ପୂର୍ଣ୍ଣସଂଖ୍ୟା r) { ଫେରନ୍ତୁ i <= r; }; ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TRI_TYPE_UPPER:      bipred = [](ପୂର୍ଣ୍ଣସଂଖ୍ୟା i, ପୂର୍ଣ୍ଣସଂଖ୍ୟା r) { ଫେରନ୍ତୁ i >  r; }; ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TRI_TYPE_UPPER_DIAG: bipred = [](ପୂର୍ଣ୍ଣସଂଖ୍ୟା i, ପୂର୍ଣ୍ଣସଂଖ୍ୟା r) { ଫେରନ୍ତୁ i >= r; }; ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ: GGML_ABORT("invalid tri type");
    }

    ପାଇଁ (int64_t ir = ir0; ir < ir1; ++ir) {
        ଅଚଳ int64_t i03 = ir/(ne02*ne01);
        ଅଚଳ int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        ଅଚଳ int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        ଅଚଳ ଦଶମିକ * src_ptr = (ଅଚଳ ଦଶମିକ  *) ((ଅଚଳ ଅକ୍ଷର *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);
              ଦଶମିକ * dst_ptr = (      ଦଶମିକ  *) ((      ଅକ୍ଷର *) dst->data  + i03*nb3  + i02*nb2  + i01*nb1);

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0 = 0; i0 < ne0; ++i0) {
            dst_ptr[i0] = bipred(i0, i01) ? src_ptr[i0] : 0.0f;
        }
    }
}

ଶୂନ୍ୟ ggml_compute_forward_tri(ଅଚଳ ggml_compute_params * params, ggml_tensor * dst) {
    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_tri_f32(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_gelu_erf

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_gelu_erf_f32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ପ୍ରତିଜ୍ଞା(ggml_is_contiguous_rows(src0));
    ପ୍ରତିଜ୍ଞା(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nc = src0->ne[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr = ggml_nrows(src0);

    // rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir = ir0; ir < ir1; ++ir) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i3 = ir/(ne02*ne01);
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i2 = (ir - i3*ne02*ne01)/ne01;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_erf_f32(nc,
                (ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ଦଶମିକ *) ((ଅକ୍ଷର *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < nc; k++) {
            ଅଚଳ ଦଶମିକ x = ((ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_UNUSED(x);
            ପ୍ରତିଜ୍ଞା(!isnan(x));
            ପ୍ରତିଜ୍ଞା(!isinf(x));
        }
#endif // NDEBUG
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_gelu_erf_f16(
    ଅଚଳ ggml_compute_params * params,
    ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ପ୍ରତିଜ୍ଞା(ggml_is_contiguous_rows(src0));
    ପ୍ରତିଜ୍ଞା(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nc = src0->ne[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr = ggml_nrows(src0);

    // rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir = ir0; ir < ir1; ++ir) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i3 = ir/(ne02*ne01);
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i2 = (ir - i3*ne02*ne01)/ne01;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_erf_f16(nc,
                (ggml_fp16_t *) ((ଅକ୍ଷର *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((ଅକ୍ଷର *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < nc; k++) {
            ଅଚଳ ggml_fp16_t x = ((ggml_fp16_t *) ((ଅକ୍ଷର *) dst->data + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            ଅଚଳ ଦଶମିକ v = GGML_CPU_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            ପ୍ରତିଜ୍ଞା(!isnan(v));
            ପ୍ରତିଜ୍ଞା(!isinf(v));
        }
#endif // NDEBUG
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_gelu_erf_bf16(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_rows<ggml_vec_gelu_erf_bf16>(params, dst);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_gelu_erf(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_gelu_erf_f32(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F16:
            {
                ggml_compute_forward_gelu_erf_f16(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_BF16:
            {
                ggml_compute_forward_gelu_erf_bf16(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_gelu_quick

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_gelu_quick_f32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ପ୍ରତିଜ୍ଞା(ggml_is_contiguous_rows(src0));
    ପ୍ରତିଜ୍ଞା(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nc = src0->ne[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr = ggml_nrows(src0);

    // rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir = ir0; ir < ir1; ++ir) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i3 = ir/(ne02*ne01);
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i2 = (ir - i3*ne02*ne01)/ne01;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_quick_f32(nc,
                (ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ଦଶମିକ *) ((ଅକ୍ଷର *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < nc; k++) {
            ଅଚଳ ଦଶମିକ x = ((ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_UNUSED(x);
            ପ୍ରତିଜ୍ଞା(!isnan(x));
            ପ୍ରତିଜ୍ଞା(!isinf(x));
        }
#endif // NDEBUG
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_gelu_quick_f16(
    ଅଚଳ ggml_compute_params * params,
    ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ପ୍ରତିଜ୍ଞା(ggml_is_contiguous_rows(src0));
    ପ୍ରତିଜ୍ଞା(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nc = src0->ne[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr = ggml_nrows(src0);

    // rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir = ir0; ir < ir1; ++ir) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i3 = ir/(ne02*ne01);
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i2 = (ir - i3*ne02*ne01)/ne01;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_quick_f16(nc,
                (ggml_fp16_t *) ((ଅକ୍ଷର *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((ଅକ୍ଷର *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < nc; k++) {
            ଅଚଳ ggml_fp16_t x = ((ggml_fp16_t *) ((ଅକ୍ଷର *) dst->data + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            ଅଚଳ ଦଶମିକ v = GGML_CPU_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            ପ୍ରତିଜ୍ଞା(!isnan(v));
            ପ୍ରତିଜ୍ଞା(!isinf(v));
        }
#endif // NDEBUG
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_gelu_quick_bf16(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_rows<ggml_vec_gelu_quick_bf16>(params, dst);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_gelu_quick(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_gelu_quick_f32(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F16:
            {
                ggml_compute_forward_gelu_quick_f16(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_BF16:
            {
                ggml_compute_forward_gelu_quick_bf16(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_silu

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_silu_f32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ପ୍ରତିଜ୍ଞା(ggml_is_contiguous_rows(src0));
    ପ୍ରତିଜ୍ଞା(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nc = src0->ne[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr = ggml_nrows(src0);

    // rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir = ir0; ir < ir1; ++ir) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i3 = ir/(ne02*ne01);
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i2 = (ir - i3*ne02*ne01)/ne01;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_silu_f32(nc,
                (ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ଦଶମିକ *) ((ଅକ୍ଷର *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < nc; k++) {
            ଅଚଳ ଦଶମିକ x = ((ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_UNUSED(x);
            ପ୍ରତିଜ୍ଞା(!isnan(x));
            ପ୍ରତିଜ୍ଞା(!isinf(x));
        }
#endif // NDEBUG
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_silu_f16(
    ଅଚଳ ggml_compute_params * params,
    ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ପ୍ରତିଜ୍ଞା(ggml_is_contiguous_rows(src0));
    ପ୍ରତିଜ୍ଞା(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nc = src0->ne[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr = ggml_nrows(src0);

    // rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir = ir0; ir < ir1; ++ir) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i3 = ir/(ne02*ne01);
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i2 = (ir - i3*ne02*ne01)/ne01;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_silu_f16(nc,
                (ggml_fp16_t *) ((ଅକ୍ଷର *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((ଅକ୍ଷର *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < nc; k++) {
            ଅଚଳ ggml_fp16_t x = ((ggml_fp16_t *) ((ଅକ୍ଷର *) dst->data + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            ଅଚଳ ଦଶମିକ v = GGML_CPU_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            ପ୍ରତିଜ୍ଞା(!isnan(v));
            ପ୍ରତିଜ୍ଞା(!isinf(v));
        }
#endif // NDEBUG
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_silu_bf16(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_rows<ggml_vec_silu_bf16>(params, dst);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_silu(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_silu_f32(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F16:
            {
                ggml_compute_forward_silu_f16(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_BF16:
            {
                ggml_compute_forward_silu_bf16(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}
// ggml_compute_forward_leaky_relu

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_leaky_relu_f32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ଯଦି (params->ith != 0) {
        ଫେରନ୍ତୁ;
    }

    ପ୍ରତିଜ୍ଞା(ggml_is_contiguous_1(src0));
    ପ୍ରତିଜ୍ଞା(ggml_is_contiguous_1(dst));
    ପ୍ରତିଜ୍ଞା(ggml_are_same_shape(src0, dst));

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n  = ggml_nrows(src0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nc = src0->ne[0];

    ଦଶମିକ negative_slope;
    memcpy(&negative_slope, dst->op_params, ଆକାର(ଦଶମିକ));

    ପ୍ରତିଜ୍ଞା(dst->nb[0]  == ଆକାର(ଦଶମିକ));
    ପ୍ରତିଜ୍ଞା(src0->nb[0] == ଆକାର(ଦଶମିକ));

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; i++) {
        ggml_vec_leaky_relu_f32(nc,
                (ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data  + i*( dst->nb[1])),
                (ଦଶମିକ *) ((ଅକ୍ଷର *) src0->data + i*(src0->nb[1])), negative_slope);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_leaky_relu_f16(
    ଅଚଳ ggml_compute_params * params,
    ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ଯଦି (params->ith != 0) {
        ଫେରନ୍ତୁ;
    }

    ପ୍ରତିଜ୍ଞା(ggml_is_contiguous_1(src0));
    ପ୍ରତିଜ୍ଞା(ggml_is_contiguous_1(dst));
    ପ୍ରତିଜ୍ଞା(ggml_are_same_shape(src0, dst));

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n  = ggml_nrows(src0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nc = src0->ne[0];

    ଦଶମିକ negative_slope;
    memcpy(&negative_slope, dst->op_params, ଆକାର(ଦଶମିକ));

    ପ୍ରତିଜ୍ଞା(dst->nb[0]  == ଆକାର(ggml_fp16_t));
    ପ୍ରତିଜ୍ଞା(src0->nb[0] == ଆକାର(ggml_fp16_t));

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; i++) {
        ggml_vec_leaky_relu_f16(nc,
                (ggml_fp16_t *) ((ଅକ୍ଷର *) dst->data  + i*( dst->nb[1])),
                (ggml_fp16_t *) ((ଅକ୍ଷର *) src0->data + i*(src0->nb[1])), negative_slope);
    }
}

ଶୂନ୍ୟ ggml_compute_forward_leaky_relu(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_leaky_relu_f32(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F16:
            {
                ggml_compute_forward_leaky_relu_f16(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_silu_back

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_silu_back_f32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * grad = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];

    ପ୍ରତିଜ୍ଞା(ggml_is_contiguous_1(grad));
    ପ୍ରତିଜ୍ଞା(ggml_is_contiguous_1(src1));
    ପ୍ରତିଜ୍ଞା(ggml_is_contiguous_1(dst));
    ପ୍ରତିଜ୍ଞା(ggml_are_same_shape(src1, dst));
    ପ୍ରତିଜ୍ଞା(ggml_are_same_shape(src1, grad));

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nc = src1->ne[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr = ggml_nrows(src1);

    // rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = ir0; i1 < ir1; i1++) {
        ggml_vec_silu_backward_f32(nc,
                (ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data  + i1*( dst->nb[1])),
                (ଦଶମିକ *) ((ଅକ୍ଷର *) src1->data + i1*(src1->nb[1])),
                (ଦଶମିକ *) ((ଅକ୍ଷର *) grad->data + i1*(grad->nb[1])));

#ifndef NDEBUG
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < nc; k++) {
            ଅଚଳ ଦଶମିକ x = ((ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            ପ୍ରତିଜ୍ଞା(!isnan(x));
            ପ୍ରତିଜ୍ଞା(!isinf(x));
        }
#endif // NDEBUG
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_silu_back_f16(
    ଅଚଳ ggml_compute_params * params,
    ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * grad = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];

    ପ୍ରତିଜ୍ଞା(ggml_is_contiguous_1(grad));
    ପ୍ରତିଜ୍ଞା(ggml_is_contiguous_1(src1));
    ପ୍ରତିଜ୍ଞା(ggml_is_contiguous_1(dst));
    ପ୍ରତିଜ୍ଞା(ggml_are_same_shape(src1, dst));
    ପ୍ରତିଜ୍ଞା(ggml_are_same_shape(src1, grad));

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nc = src1->ne[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr = ggml_nrows(src1);

    // rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = ir0; i1 < ir1; i1++) {
        ggml_vec_silu_backward_f16(nc,
                (ggml_fp16_t *) ((ଅକ୍ଷର *) dst->data  + i1*( dst->nb[1])),
                (ggml_fp16_t *) ((ଅକ୍ଷର *) src1->data + i1*(src1->nb[1])),
                (ggml_fp16_t *) ((ଅକ୍ଷର *) grad->data + i1*(grad->nb[1])));

#ifndef NDEBUG
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < nc; k++) {
            ଅଚଳ ଦଶମିକ x = ((ggml_fp16_t *) ((ଅକ୍ଷର *) dst->data + i1*( dst->nb[1])))[k];
            ଅଚଳ ଦଶମିକ v = GGML_CPU_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            ପ୍ରତିଜ୍ଞା(!isnan(v));
            ପ୍ରତିଜ୍ଞା(!isinf(v));
        }
#endif // NDEBUG
    }
}

ଶୂନ୍ୟ ggml_compute_forward_silu_back(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_silu_back_f32(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F16:
            {
                ggml_compute_forward_silu_back_f16(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_reglu

// row loop shared by the BF16 GLU ops
template <ଶୂନ୍ୟ (*vec_op)(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା, ggml_bf16_t *, ଅଚଳ ggml_bf16_t *, ଅଚଳ ggml_bf16_t *)>
ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_glu_bf16_rows(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];
    ଅକ୍ଷର * src0_d = (ଅକ୍ଷର *) src0->data;
    ଅକ୍ଷର * src1_d = (ଅକ୍ଷର *) (src1 ? src1->data : src0->data);
    ଅଚଳ size_t src0_o = src0->nb[1];
    ଅଚଳ size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ଯଦି (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ଅଚଳ int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = ir0; i1 < ir1; i1++) {
        ଅଚଳ ggml_bf16_t * src0_p = (ଅଚଳ ggml_bf16_t *) (src0_d + i1*src0_o);
        ଅଚଳ ggml_bf16_t * src1_p = (ଅଚଳ ggml_bf16_t *) (src1_d + i1*src1_o);
        ggml_bf16_t       * dst_p  = (ggml_bf16_t *) ((ଅକ୍ଷର *) dst->data + i1*(dst->nb[1]));

        ଯଦି (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        vec_op(nc, dst_p, src0_p, src1_p);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_reglu_f32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];
    ଅକ୍ଷର * src0_d = (ଅକ୍ଷର *) src0->data;
    ଅକ୍ଷର * src1_d = (ଅକ୍ଷର *) (src1 ? src1->data : src0->data);
    ଅଚଳ size_t src0_o = src0->nb[1];
    ଅଚଳ size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ଯଦି (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ଅଚଳ int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = ir0; i1 < ir1; i1++) {
        ଦଶମିକ * src0_p = (ଦଶମିକ *) (src0_d + i1*src0_o);
        ଦଶମିକ * src1_p = (ଦଶମିକ *) (src1_d + i1*src1_o);

        ଯଦି (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_reglu_f32(nc, (ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < nc; k++) {
            ଅଚଳ ଦଶମିକ x = ((ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            ପ୍ରତିଜ୍ଞା(!isnan(x));
            ପ୍ରତିଜ୍ଞା(!isinf(x));
        }
#endif // NDEBUG
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_reglu_f16(
    ଅଚଳ ggml_compute_params * params,
    ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];
    ଅକ୍ଷର * src0_d = (ଅକ୍ଷର *) src0->data;
    ଅକ୍ଷର * src1_d = (ଅକ୍ଷର *) (src1 ? src1->data : src0->data);
    ଅଚଳ size_t src0_o = src0->nb[1];
    ଅଚଳ size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ଯଦି (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ଅଚଳ int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        ଯଦି (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_reglu_f16(nc, (ggml_fp16_t *) ((ଅକ୍ଷର *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < nc; k++) {
            ଅଚଳ ggml_fp16_t x = ((ggml_fp16_t *) ((ଅକ୍ଷର *) dst->data + i1*( dst->nb[1])))[k];
            ଅଚଳ ଦଶମିକ v = GGML_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            ପ୍ରତିଜ୍ଞା(!isnan(v));
            ପ୍ରତିଜ୍ଞା(!isinf(v));
        }
#endif // NDEBUG
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_reglu_bf16(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_rows<ggml_vec_reglu_bf16>(params, dst);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_reglu(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_reglu_f32(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F16:
            {
                ggml_compute_forward_reglu_f16(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_BF16:
            {
                ggml_compute_forward_reglu_bf16(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_geglu

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_geglu_f32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];
    ଅକ୍ଷର * src0_d = (ଅକ୍ଷର *) src0->data;
    ଅକ୍ଷର * src1_d = (ଅକ୍ଷର *) (src1 ? src1->data : src0->data);
    ଅଚଳ size_t src0_o = src0->nb[1];
    ଅଚଳ size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ଯଦି (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ଅଚଳ int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = ir0; i1 < ir1; i1++) {
        ଦଶମିକ * src0_p = (ଦଶମିକ *) (src0_d + i1*src0_o);
        ଦଶମିକ * src1_p = (ଦଶମିକ *) (src1_d + i1*src1_o);

        ଯଦି (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_f32(nc, (ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < nc; k++) {
            ଅଚଳ ଦଶମିକ x = ((ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            ପ୍ରତିଜ୍ଞା(!isnan(x));
            ପ୍ରତିଜ୍ଞା(!isinf(x));
        }
#endif // NDEBUG
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_geglu_f16(
    ଅଚଳ ggml_compute_params * params,
    ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];
    ଅକ୍ଷର * src0_d = (ଅକ୍ଷର *) src0->data;
    ଅକ୍ଷର * src1_d = (ଅକ୍ଷର *) (src1 ? src1->data : src0->data);
    ଅଚଳ size_t src0_o = src0->nb[1];
    ଅଚଳ size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ଯଦି (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ଅଚଳ int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        ଯଦି (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_f16(nc, (ggml_fp16_t *) ((ଅକ୍ଷର *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < nc; k++) {
            ଅଚଳ ggml_fp16_t x = ((ggml_fp16_t *) ((ଅକ୍ଷର *) dst->data + i1*( dst->nb[1])))[k];
            ଅଚଳ ଦଶମିକ v = GGML_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            ପ୍ରତିଜ୍ଞା(!isnan(v));
            ପ୍ରତିଜ୍ଞା(!isinf(v));
        }
#endif // NDEBUG
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_geglu_bf16(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_rows<ggml_vec_geglu_bf16>(params, dst);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_geglu(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_geglu_f32(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F16:
            {
                ggml_compute_forward_geglu_f16(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_BF16:
            {
                ggml_compute_forward_geglu_bf16(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_swiglu

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_swiglu_f32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];
    ଅକ୍ଷର * src0_d = (ଅକ୍ଷର *) src0->data;
    ଅକ୍ଷର * src1_d = (ଅକ୍ଷର *) (src1 ? src1->data : src0->data);
    ଅଚଳ size_t src0_o = src0->nb[1];
    ଅଚଳ size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ଯଦି (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ଅଚଳ int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = ir0; i1 < ir1; i1++) {
        ଦଶମିକ * src0_p = (ଦଶମିକ *) (src0_d + i1*src0_o);
        ଦଶମିକ * src1_p = (ଦଶମିକ *) (src1_d + i1*src1_o);

        ଯଦି (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_swiglu_f32(nc, (ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < nc; k++) {
            ଅଚଳ ଦଶମିକ x = ((ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            ପ୍ରତିଜ୍ଞା(!isnan(x));
            ପ୍ରତିଜ୍ଞା(!isinf(x));
        }
#endif // NDEBUG
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_swiglu_f16(
    ଅଚଳ ggml_compute_params * params,
    ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];
    ଅକ୍ଷର * src0_d = (ଅକ୍ଷର *) src0->data;
    ଅକ୍ଷର * src1_d = (ଅକ୍ଷର *) (src1 ? src1->data : src0->data);
    ଅଚଳ size_t src0_o = src0->nb[1];
    ଅଚଳ size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ଯଦି (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ଅଚଳ int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        ଯଦି (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_swiglu_f16(nc, (ggml_fp16_t *) ((ଅକ୍ଷର *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < nc; k++) {
            ଅଚଳ ggml_fp16_t x = ((ggml_fp16_t *) ((ଅକ୍ଷର *) dst->data + i1*( dst->nb[1])))[k];
            ଅଚଳ ଦଶମିକ v = GGML_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            ପ୍ରତିଜ୍ଞା(!isnan(v));
            ପ୍ରତିଜ୍ଞା(!isinf(v));
        }
#endif // NDEBUG
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_swiglu_bf16(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_rows<ggml_vec_swiglu_bf16>(params, dst);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_swiglu(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_swiglu_f32(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F16:
            {
                ggml_compute_forward_swiglu_f16(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_BF16:
            {
                ggml_compute_forward_swiglu_bf16(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_swiglu_oai

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_swiglu_oai_f32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];
    ଅକ୍ଷର * src0_d = (ଅକ୍ଷର *) src0->data;
    ଅକ୍ଷର * src1_d = (ଅକ୍ଷର *) (src1 ? src1->data : src0->data);
    ଅଚଳ size_t src0_o = src0->nb[1];
    ଅଚଳ size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ଯଦି (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ଅଚଳ int32_t swapped = ggml_get_op_params_i32(dst, 1);
    ଅଚଳ ଦଶମିକ alpha = ggml_get_op_params_f32(dst, 2);
    ଅଚଳ ଦଶମିକ limit = ggml_get_op_params_f32(dst, 3);

    // rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = ir0; i1 < ir1; i1++) {
        ଦଶମିକ * src0_p = (ଦଶମିକ *) (src0_d + i1*src0_o);
        ଦଶମିକ * src1_p = (ଦଶମିକ *) (src1_d + i1*src1_o);
        ଦଶମିକ * dst_p  = (ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data + i1*(dst->nb[1]));

        ଯଦି (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < nc; k++) {
            ଅଚଳ ଦଶମିକ x = std::min(src0_p[k], limit);
            ଅଚଳ ଦଶମିକ y = std::clamp(src1_p[k], -limit, limit);
            ଅଚଳ ଦଶମିକ out_glu = x / (1.f + expf(alpha * (-x)));
            dst_p[k] = out_glu * (y + 1.f);
        }

#ifndef NDEBUG
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < nc; k++) {
            ଅଚଳ ଦଶମିକ x = dst_p[k];
            GGML_UNUSED(x);
            ପ୍ରତିଜ୍ଞା(!isnan(x));
            ପ୍ରତିଜ୍ଞା(!isinf(x));
        }
#endif // NDEBUG
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_swiglu_oai_f16(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];
    ଅକ୍ଷର * src0_d = (ଅକ୍ଷର *) src0->data;
    ଅକ୍ଷର * src1_d = (ଅକ୍ଷର *) (src1 ? src1->data : src0->data);
    ଅଚଳ size_t src0_o = src0->nb[1];
    ଅଚଳ size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ଯଦି (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ଅଚଳ int32_t swapped = ggml_get_op_params_i32(dst, 1);
    ଅଚଳ ଦଶମିକ alpha = ggml_get_op_params_f32(dst, 2);
    ଅଚଳ ଦଶମିକ limit = ggml_get_op_params_f32(dst, 3);

    // rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);
        ggml_fp16_t * dst_p  = (ggml_fp16_t *) ((ଅକ୍ଷର *) dst->data + i1*(dst->nb[1]));

        ଯଦି (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < nc; k++) {
            ଅଚଳ ଦଶମିକ x = std::min(GGML_CPU_FP16_TO_FP32(src0_p[k]), limit);
            ଅଚଳ ଦଶମିକ y = std::clamp(GGML_CPU_FP16_TO_FP32(src1_p[k]), -limit, limit);
            ଅଚଳ ଦଶମିକ out_glu = x / (1.f + expf(alpha * (-x)));
            dst_p[k] = GGML_CPU_FP32_TO_FP16(out_glu * (y + 1.f));
        }

#ifndef NDEBUG
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < nc; k++) {
            ଅଚଳ ଦଶମିକ x = GGML_CPU_FP16_TO_FP32(dst_p[k]);
            GGML_UNUSED(x);
            ପ୍ରତିଜ୍ଞା(!isnan(x));
            ପ୍ରତିଜ୍ଞା(!isinf(x));
        }
#endif // NDEBUG
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_swiglu_oai(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_swiglu_oai_f32(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F16:
            {
                ggml_compute_forward_swiglu_oai_f16(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_swiglu_clamp

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_swiglu_clamp_f32(ଅଚଳ ggml_compute_params * params, ggml_tensor * dst) {
    ଅଚଳ ggml_tensor * src0   = dst->src[0];
    ଅଚଳ ggml_tensor * src1   = dst->src[1];
    ଅକ୍ଷର *              src0_d = (ଅକ୍ଷର *) src0->data;
    ଅକ୍ଷର *              src1_d = (ଅକ୍ଷର *) (src1 ? src1->data : src0->data);
    ଅଚଳ size_t        src0_o = src0->nb[1];
    ଅଚଳ size_t        src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ଯଦି (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ଅଚଳ int32_t swapped = ggml_get_op_params_i32(dst, 1);
    ଅଚଳ ଦଶମିକ   limit   = ggml_get_op_params_f32(dst, 3);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr  = (nr + nth - 1) / nth;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr * ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = ir0; i1 < ir1; i1++) {
        ଦଶମିକ * src0_p = (ଦଶମିକ *) (src0_d + i1 * src0_o);
        ଦଶମିକ * src1_p = (ଦଶମିକ *) (src1_d + i1 * src1_o);
        ଦଶମିକ * dst_p  = (ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data + i1 * (dst->nb[1]));

        ଯଦି (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < nc; k++) {
            ଅଚଳ ଦଶମିକ gate = std::min(src0_p[k], limit);
            ଅଚଳ ଦଶମିକ up   = std::clamp(src1_p[k], -limit, limit);
            dst_p[k]         = gate / (1.f + expf(-gate)) * up;
        }

#ifndef NDEBUG
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < nc; k++) {
            ଅଚଳ ଦଶମିକ x = dst_p[k];
            GGML_UNUSED(x);
            ପ୍ରତିଜ୍ଞା(!isnan(x));
            ପ୍ରତିଜ୍ଞା(!isinf(x));
        }
#endif  // NDEBUG
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_swiglu_clamp_f16(ଅଚଳ ggml_compute_params * params, ggml_tensor * dst) {
    ଅଚଳ ggml_tensor * src0   = dst->src[0];
    ଅଚଳ ggml_tensor * src1   = dst->src[1];
    ଅକ୍ଷର *              src0_d = (ଅକ୍ଷର *) src0->data;
    ଅକ୍ଷର *              src1_d = (ଅକ୍ଷର *) (src1 ? src1->data : src0->data);
    ଅଚଳ size_t        src0_o = src0->nb[1];
    ଅଚଳ size_t        src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ଯଦି (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ଅଚଳ int32_t swapped = ggml_get_op_params_i32(dst, 1);
    ଅଚଳ ଦଶମିକ   limit   = ggml_get_op_params_f32(dst, 3);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr  = (nr + nth - 1) / nth;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr * ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1 * src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1 * src1_o);
        ggml_fp16_t * dst_p  = (ggml_fp16_t *) ((ଅକ୍ଷର *) dst->data + i1 * (dst->nb[1]));

        ଯଦି (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < nc; k++) {
            ଅଚଳ ଦଶମିକ gate = std::min(GGML_FP16_TO_FP32(src0_p[k]), limit);
            ଅଚଳ ଦଶମିକ up   = std::clamp(GGML_FP16_TO_FP32(src1_p[k]), -limit, limit);
            dst_p[k]         = GGML_FP32_TO_FP16(gate / (1.f + expf(-gate)) * up);
        }

#ifndef NDEBUG
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < nc; k++) {
            ଅଚଳ ଦଶମିକ x = GGML_FP16_TO_FP32(dst_p[k]);
            GGML_UNUSED(x);
            ପ୍ରତିଜ୍ଞା(!isnan(x));
            ପ୍ରତିଜ୍ଞା(!isinf(x));
        }
#endif  // NDEBUG
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_swiglu_clamp(ଅଚଳ ggml_compute_params * params, ggml_tensor * dst) {
    ବାଛ (dst->src[0]->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            ggml_compute_forward_swiglu_clamp_f32(params, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F16:
            ggml_compute_forward_swiglu_clamp_f16(params, dst);
            ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            GGML_ABORT("fatal error");
    }
}

// ggml_compute_forward_geglu_erf

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_geglu_erf_f32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];
    ଅକ୍ଷର * src0_d = (ଅକ୍ଷର *) src0->data;
    ଅକ୍ଷର * src1_d = (ଅକ୍ଷର *) (src1 ? src1->data : src0->data);
    ଅଚଳ size_t src0_o = src0->nb[1];
    ଅଚଳ size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ଯଦି (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ଅଚଳ int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = ir0; i1 < ir1; i1++) {
        ଦଶମିକ * src0_p = (ଦଶମିକ *) (src0_d + i1*src0_o);
        ଦଶମିକ * src1_p = (ଦଶମିକ *) (src1_d + i1*src1_o);

        ଯଦି (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_erf_f32(nc, (ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < nc; k++) {
            ଅଚଳ ଦଶମିକ x = ((ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            ପ୍ରତିଜ୍ଞା(!isnan(x));
            ପ୍ରତିଜ୍ଞା(!isinf(x));
        }
#endif // NDEBUG
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_geglu_erf_f16(
    ଅଚଳ ggml_compute_params * params,
    ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];
    ଅକ୍ଷର * src0_d = (ଅକ୍ଷର *) src0->data;
    ଅକ୍ଷର * src1_d = (ଅକ୍ଷର *) (src1 ? src1->data : src0->data);
    ଅଚଳ size_t src0_o = src0->nb[1];
    ଅଚଳ size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ଯଦି (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ଅଚଳ int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        ଯଦି (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_erf_f16(nc, (ggml_fp16_t *) ((ଅକ୍ଷର *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < nc; k++) {
            ଅଚଳ ggml_fp16_t x = ((ggml_fp16_t *) ((ଅକ୍ଷର *) dst->data + i1*( dst->nb[1])))[k];
            ଅଚଳ ଦଶମିକ v = GGML_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            ପ୍ରତିଜ୍ଞା(!isnan(v));
            ପ୍ରତିଜ୍ଞା(!isinf(v));
        }
#endif // NDEBUG
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_geglu_erf_bf16(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_rows<ggml_vec_geglu_erf_bf16>(params, dst);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_geglu_erf(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_geglu_erf_f32(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F16:
            {
                ggml_compute_forward_geglu_erf_f16(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_BF16:
            {
                ggml_compute_forward_geglu_erf_bf16(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_geglu_quick

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_geglu_quick_f32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];
    ଅକ୍ଷର * src0_d = (ଅକ୍ଷର *) src0->data;
    ଅକ୍ଷର * src1_d = (ଅକ୍ଷର *) (src1 ? src1->data : src0->data);
    ଅଚଳ size_t src0_o = src0->nb[1];
    ଅଚଳ size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ଯଦି (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ଅଚଳ int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = ir0; i1 < ir1; i1++) {
        ଦଶମିକ * src0_p = (ଦଶମିକ *) (src0_d + i1*src0_o);
        ଦଶମିକ * src1_p = (ଦଶମିକ *) (src1_d + i1*src1_o);

        ଯଦି (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_quick_f32(nc, (ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < nc; k++) {
            ଅଚଳ ଦଶମିକ x = ((ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            ପ୍ରତିଜ୍ଞା(!isnan(x));
            ପ୍ରତିଜ୍ଞା(!isinf(x));
        }
#endif // NDEBUG
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_geglu_quick_f16(
    ଅଚଳ ggml_compute_params * params,
    ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];
    ଅକ୍ଷର * src0_d = (ଅକ୍ଷର *) src0->data;
    ଅକ୍ଷର * src1_d = (ଅକ୍ଷର *) (src1 ? src1->data : src0->data);
    ଅଚଳ size_t src0_o = src0->nb[1];
    ଅଚଳ size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ଯଦି (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ଅଚଳ int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        ଯଦି (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_quick_f16(nc, (ggml_fp16_t *) ((ଅକ୍ଷର *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < nc; k++) {
            ଅଚଳ ggml_fp16_t x = ((ggml_fp16_t *) ((ଅକ୍ଷର *) dst->data + i1*( dst->nb[1])))[k];
            ଅଚଳ ଦଶମିକ v = GGML_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            ପ୍ରତିଜ୍ଞା(!isnan(v));
            ପ୍ରତିଜ୍ଞା(!isinf(v));
        }
#endif // NDEBUG
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_geglu_quick_bf16(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_rows<ggml_vec_geglu_quick_bf16>(params, dst);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_geglu_quick(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_geglu_quick_f32(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F16:
            {
                ggml_compute_forward_geglu_quick_f16(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_BF16:
            {
                ggml_compute_forward_geglu_quick_bf16(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_norm

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_norm_f32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    ଦଶମିକ eps;
    memcpy(&eps, dst->op_params, ଆକାର(ଦଶମିକ));

    GGML_ASSERT(eps >= 0.0f);

    ପାଇଁ (int64_t i03 = 0; i03 < ne03; i03++) {
        ପାଇଁ (int64_t i02 = 0; i02 < ne02; i02++) {
            ପାଇଁ (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                ଅଚଳ ଅକ୍ଷର * x = (ଅଚଳ ଅକ୍ଷର *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
                ଅକ୍ଷର * y = (ଅକ୍ଷର *) dst->data + i01*nb1 + i02*nb2 + i03*nb3;

                ଯଦି (nb00 == ଆକାର(ଦଶମିକ) && nb0 == ଆକାର(ଦଶମିକ)) {
                    ଅଚଳ ଦଶମିକ * xf = (ଅଚଳ ଦଶମିକ *) x;

                    ଦଶମିକ sum = 0.0;
                    ggml_vec_sum_f32(ne00, &sum, xf);
                    ଦଶମିକ mean = sum/ne00;

                    ଦଶମିକ * yf = (ଦଶମିକ *) y;
                    ଦଶମିକ variance = 0;

#ifdef GGML_USE_ACCELERATE
                    mean = -mean;
                    vDSP_vsadd(xf, 1, &mean, yf, 1, ne00);
                    vDSP_measqv(yf, 1, &variance, ne00);
#ଅନ୍ୟଥା
                    variance = ggml_vec_cvar_f32(ne00, yf, xf, mean);
#endif //GGML_USE_ACCELERATE

                    ଅଚଳ ଦଶମିକ scale = 1.0f/sqrtf(variance + eps);
                    ggml_vec_scale_f32(ne00, yf, scale);
                } ଅନ୍ୟଥା {
                    ଦଶମିକ sum = 0.0;
                    ପାଇଁ (int64_t i00 = 0; i00 < ne00; i00++) {
                        sum += *(ଅଚଳ ଦଶମିକ *) (x + i00*nb00);
                    }
                    ଅଚଳ ଦଶମିକ mean = sum/ne00;

                    ଦଶମିକ variance = 0.0f;
                    ପାଇଁ (int64_t i00 = 0; i00 < ne00; i00++) {
                        ଅଚଳ ଦଶମିକ v = *(ଅଚଳ ଦଶମିକ *) (x + i00*nb00) - mean;
                        *(ଦଶମିକ *) (y + i00*nb0) = v;
                        variance += v * v;
                    }
                    variance /= ne00;

                    ଅଚଳ ଦଶମିକ scale = 1.0f/sqrtf(variance + eps);
                    ପାଇଁ (int64_t i00 = 0; i00 < ne00; i00++) {
                        *(ଦଶମିକ *) (y + i00*nb0) *= scale;
                    }
                }
            }
        }
    }
}

ଶୂନ୍ୟ ggml_compute_forward_norm(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_norm_f32(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_group_rms_norm

// fusion kinds that can be combined with the rms_norm computation in a single pass.
// extend this enum when adding new fused variants (e.g. FUSE_ADD, FUSE_MUL_ADD, ...).
ଗଣନା ggml_rms_norm_fuse_op {
    GGML_RMS_NORM_FUSE_OP_NONE,
    GGML_RMS_NORM_FUSE_OP_MUL,
};

template <ggml_rms_norm_fuse_op FUSE_OP>
ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_rms_norm_f32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst_rms_norm,
        ggml_tensor * dst_fused = nullptr) {

    ଅଚଳ ggml_tensor * src0 = dst_rms_norm->src[0];
    ଅଚଳ ggml_tensor * src1 = nullptr;
    ggml_tensor       * dst  = dst_rms_norm;

    ଯଦି constexpr (FUSE_OP == GGML_RMS_NORM_FUSE_OP_MUL) {
        src1 = (dst_fused->src[0] == dst_rms_norm) ? dst_fused->src[1] : dst_fused->src[0];
        dst  = dst_fused;
    }

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    GGML_ASSERT(src0->nb[0] == ଆକାର(ଦଶମିକ));

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    GGML_TENSOR_ଦ୍ୱିମିକ_OP_LOCALS

    ଦଶମିକ eps;
    memcpy(&eps, dst_rms_norm->op_params, ଆକାର(ଦଶମିକ));
    GGML_ASSERT(eps >= 0.0f);

    // TODO: optimize
    ପାଇଁ (int64_t i03 = 0; i03 < ne03; i03++) {
        ପାଇଁ (int64_t i02 = 0; i02 < ne02; i02++) {
            ପାଇଁ (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                ଅଚଳ ଦଶମିକ * x = (ଦଶମିକ *) ((ଅକ୍ଷର *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);

                ggml_float sum = 0.0;
                // worth switching to explicit SIMD?
                ପାଇଁ (int64_t i00 = 0; i00 < ne00; i00++) {
                    sum += (ggml_float)(x[i00] * x[i00]);
                }

                ଅଚଳ ଦଶମିକ mean  = sum/ne00;
                ଅଚଳ ଦଶମିକ scale = 1.0f/sqrtf(mean + eps);

                // if you hit this, likely you got an inf somewhere earlier
                ପ୍ରତିଜ୍ଞା(scale > 0.0f);

                ଦଶମିକ * y = (ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data + i01*nb1 + i02*nb2 + i03*nb3);

                ଯଦି constexpr (FUSE_OP == GGML_RMS_NORM_FUSE_OP_MUL) {
                    ଅଚଳ int64_t i11 = i01 % ne11;
                    ଅଚଳ int64_t i12 = i02 % ne12;
                    ଅଚଳ int64_t i13 = i03 % ne13;
                    ଅଚଳ ଦଶମିକ * w = (ଦଶମିକ *) ((ଅକ୍ଷର *) src1->data + i11*nb11 + i12*nb12 + i13*nb13);

                    ପାଇଁ (int64_t i00 = 0; i00 < ne00; i00++) {
                        y[i00] = x[i00] * scale * w[i00];
                    }
                } ଅନ୍ୟଥା {
                    memcpy(y, x, ne00 * ଆକାର(ଦଶମିକ));
                    ggml_vec_scale_f32(ne00, y, scale);
                }
            }
        }
    }
}

ଶୂନ୍ୟ ggml_compute_forward_rms_norm(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_rms_norm_f32<GGML_RMS_NORM_FUSE_OP_NONE>(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// Fused RMS_NORM + MUL: computes dst = rms_norm(src0) * src1 in a single pass.
// This avoids materializing the intermediate rms_norm result in memory.
ଶୂନ୍ୟ ggml_compute_forward_rms_norm_mul_fused(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst_rms_norm,
        ggml_tensor * dst_mul) {

    GGML_ASSERT(dst_mul != nullptr);
    GGML_ASSERT(dst_mul->src[0] == dst_rms_norm || dst_mul->src[1] == dst_rms_norm);

    ଅଚଳ ggml_tensor * src0 = dst_rms_norm->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_rms_norm_f32<GGML_RMS_NORM_FUSE_OP_MUL>(params, dst_rms_norm, dst_mul);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_rms_norm_back_f32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0]; // gradients from forward pass output
    ଅଚଳ ggml_tensor * src1 = dst->src[1]; // src1 from forward pass

    GGML_ASSERT(ggml_are_same_shape(src0, dst) && ggml_are_same_shape(src0, src1));

    GGML_ASSERT(src0->nb[0] == ଆକାର(ଦଶମିକ));
    GGML_ASSERT(src1->nb[0] == ଆକାର(ଦଶମିକ));

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    GGML_TENSOR_ଦ୍ୱିମିକ_OP_LOCALS

    ଦଶମିକ eps;
    memcpy(&eps, dst->op_params, ଆକାର(ଦଶମିକ));

    // TODO: optimize
    ପାଇଁ (int64_t i03 = 0; i03 < ne03; i03++) {
        ପାଇଁ (int64_t i02 = 0; i02 < ne02; i02++) {
            ପାଇଁ (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                // src1 is same shape as src0 => same indices
                ଅଚଳ int64_t i11 = i01;
                ଅଚଳ int64_t i12 = i02;
                ଅଚଳ int64_t i13 = i03;

                ଅଚଳ ଦଶମିକ * dz = (ଦଶମିକ *) ((ଅକ୍ଷର *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
                ଅଚଳ ଦଶମିକ * x  = (ଦଶମିକ *) ((ଅକ୍ଷର *) src1->data + i11*nb11 + i12*nb12 + i13*nb13);

                ggml_float sum_xx  = 0.0;
                ggml_float sum_xdz = 0.0;

                ପାଇଁ (int64_t i00 = 0; i00 < ne00; i00++) {
                    sum_xx  += (ggml_float)(x[i00] * x[i00]);
                    sum_xdz += (ggml_float)(x[i00] * dz[i00]);
                }

                //const float mean     = (float)(sum_xx)/ne00;
                ଅଚଳ ଦଶମିକ mean_eps = (ଦଶମିକ)(sum_xx)/ne00 + eps;
                ଅଚଳ ଦଶମିକ sum_eps  = (ଦଶମିକ)(sum_xx) + eps*ne00;
                //const float mean_xdz = (float)(sum_xdz)/ne00;
                // we could cache rms from forward pass to improve performance.
                // to do this implement ggml_rms and compose ggml_rms_norm using ggml_rms.
                //const float rms      = sqrtf(mean_eps);
                ଅଚଳ ଦଶମିକ rrms     = 1.0f / sqrtf(mean_eps);
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
                ଦଶମିକ * dx = (ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data + i01*nb1 + i02*nb2 + i03*nb3);

                // dx[i00] = (dz + x*(-sum_xdz/sum_eps)) * rrms
                // note: https://github.com/ggml-org/ggml/issues/1491
                ଅଚଳ ଦଶମିକ scale_x = (ଦଶମିକ) (-sum_xdz) / sum_eps;
                ପାଇଁ (int64_t i00 = 0; i00 < ne00; i00++) {
                    dx[i00] = (dz[i00] + x[i00] * scale_x) * rrms;
                }
            }
        }
    }
}

ଶୂନ୍ୟ ggml_compute_forward_rms_norm_back(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_rms_norm_back_f32(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_group_norm

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_group_norm_f32(
    ଅଚଳ ggml_compute_params * params,
    ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    GGML_ASSERT(src0->nb[0] == ଆକାର(ଦଶମିକ));

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    // TODO: optimize

    ଦଶମିକ eps;
    memcpy(&eps, dst->op_params + 1, ଆକାର(ଦଶମିକ));

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_channels = src0->ne[2];
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_groups = dst->op_params[0];
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_channels_per_group = (n_channels + n_groups - 1) / n_groups;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = ith; i < n_groups; i += nth) {
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା start = i * n_channels_per_group;
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା end = start + n_channels_per_group;
        ଯଦି (end > n_channels) {
            end = n_channels;
        }
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା step = end - start;

        ପାଇଁ (int64_t i03 = 0; i03 < ne03; i03++) {
            ggml_float sum = 0.0;
            ପାଇଁ (int64_t i02 = start; i02 < end; i02++) {
                ପାଇଁ (int64_t i01 = 0; i01 < ne01; i01++) {
                    ଅଚଳ ଦଶମିକ * x = (ଦଶମିକ *)((ଅକ୍ଷର *) src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);

                    ggml_float sumr = 0.0;
                    ପାଇଁ (int64_t i00 = 0; i00 < ne00; i00++) {
                        sumr += (ggml_float)x[i00];
                    }
                    sum += sumr;
                }
            }
            ଅଚଳ ଦଶମିକ mean = sum / (ne00 * ne01 * step);

            ggml_float sum2 = 0.0;
            ପାଇଁ (int64_t i02 = start; i02 < end; i02++) {
                ପାଇଁ (int64_t i01 = 0; i01 < ne01; i01++) {
                    ଅଚଳ ଦଶମିକ * x = (ଦଶମିକ *)((ଅକ୍ଷର *) src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);

                    ଦଶମିକ * y = (ଦଶମିକ *)((ଅକ୍ଷର *) dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3);

                    ggml_float sumr = 0.0;
                    ପାଇଁ (int64_t i00 = 0; i00 < ne00; i00++) {
                        ଦଶମିକ v = x[i00] - mean;
                        y[i00] = v;
                        sumr += (ggml_float)(v * v);
                    }
                    sum2 += sumr;
                }
            }
            ଅଚଳ ଦଶମିକ variance = sum2 / (ne00 * ne01 * step);
            ଅଚଳ ଦଶମିକ scale = 1.0f / sqrtf(variance + eps);

            ପାଇଁ (int64_t i02 = start; i02 < end; i02++) {
                ପାଇଁ (int64_t i01 = 0; i01 < ne01; i01++) {
                    ଦଶମିକ * y = (ଦଶମିକ *)((ଅକ୍ଷର *) dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3);
                    ggml_vec_scale_f32(ne00, y, scale);
                }
            }
        }
    }
}

ଶୂନ୍ୟ ggml_compute_forward_group_norm(
    ଅଚଳ ggml_compute_params * params,
    ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_group_norm_f32(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_l2_norm

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_l2_norm_f32(
    ଅଚଳ ggml_compute_params * params,
    ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    ଦଶମିକ eps;
    memcpy(&eps, dst->op_params, ଆକାର(ଦଶମିକ));

    GGML_ASSERT(eps >= 0.0f);

    // TODO: optimize
    ପାଇଁ (int64_t i03 = 0; i03 < ne03; i03++) {
        ପାଇଁ (int64_t i02 = 0; i02 < ne02; i02++) {
            ପାଇଁ (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                ଅଚଳ ଅକ୍ଷର * x = (ଅଚଳ ଅକ୍ଷର *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;

                ggml_float sum = 0.0;
                ପାଇଁ (int64_t i00 = 0; i00 < ne00; i00++) {
                    ଅଚଳ ଦଶମିକ xi = *(ଅଚଳ ଦଶମିକ *) (x + i00*nb00);
                    sum += (ggml_float)(xi * xi);
                }

                ଅଚଳ ଦଶମିକ scale = 1.0f/fmaxf(sqrtf(sum), eps);

                ଅକ୍ଷର * y = (ଅକ୍ଷର *) dst->data + i01*nb1 + i02*nb2 + i03*nb3;

                ଯଦି (nb00 == ଆକାର(ଦଶମିକ) && nb0 == ଆକାର(ଦଶମିକ)) {
                    memcpy(y, x, ne00 * ଆକାର(ଦଶମିକ));
                    ggml_vec_scale_f32(ne00, (ଦଶମିକ *) y, scale);
                } ଅନ୍ୟଥା {
                    ପାଇଁ (int64_t i00 = 0; i00 < ne00; i00++) {
                        ଅଚଳ ଦଶମିକ xi = *(ଅଚଳ ଦଶମିକ *) (x + i00*nb00);
                        *(ଦଶମିକ *) (y + i00*nb0) = xi * scale;
                    }
                }
            }
        }
    }
}

ଶୂନ୍ୟ ggml_compute_forward_l2_norm(
    ଅଚଳ ggml_compute_params * params,
    ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_l2_norm_f32(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_out_prod

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_out_prod_f32(
        ଅଚଳ ggml_compute_params * params,
              ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ଦ୍ୱିମିକ_OP_LOCALS

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    GGML_ASSERT(ne2 % ne02 == 0);
    GGML_ASSERT(ne3 % ne03 == 0);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ଆକାର(ଦଶମିକ));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == ଆକାର(ଦଶମିକ));
    // GGML_ASSERT(nb0 <= nb1);
    // GGML_ASSERT(nb1 <= nb2);
    // GGML_ASSERT(nb2 <= nb3);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    ଯଦି (ith == 0) {
        ggml_vec_set_f32(ne0*ne1*ne2*ne3, (ଦଶମିକ *)dst->data, 0);
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
    ଅଚଳ int64_t nr = ne1*ne2*ne3;

    // rows per thread
    ଅଚଳ int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ int64_t ir0 = dr*ith;
    ଅଚଳ int64_t ir1 = MIN(ir0 + dr, nr);

    // block-tiling attempt
    ଅଚଳ int64_t blck_0 = MAX(GGML_VEC_MAD_UNROLL, 32);
    ଅଚଳ int64_t blck_1 = 16;

    // dps == dst per src0, used for group query attention
    ଅଚଳ int64_t dps2 = ne2 / ne02;
    ଅଚଳ int64_t dps3 = ne3 / ne03;

    ପାଇଁ (int64_t bir = ir0; bir < ir1; bir += blck_1) {
        ଅଚଳ int64_t bir1 = MIN(bir + blck_1, ir1);
        ପାଇଁ (int64_t bi01 = 0; bi01 < ne01; bi01 += blck_0) {
            ଅଚଳ int64_t bne01 = MIN(bi01 + blck_0, ne01);
            ପାଇଁ (int64_t ir = bir; ir < bir1; ++ir) {
                // dst indices
                ଅଚଳ int64_t i3 = ir/(ne2*ne1);
                ଅଚଳ int64_t i2 = (ir - i3*ne2*ne1)/ne1;
                ଅଚଳ int64_t i1 = (ir - i3*ne2*ne1 - i2*ne1);

                ଅଚଳ int64_t i02 = i2 / dps2;
                ଅଚଳ int64_t i03 = i3 / dps3;

                //const int64_t i10 = i1;
                ଅଚଳ int64_t i12 = i2;
                ଅଚଳ int64_t i13 = i3;

#ଯଦି GGML_VEC_MAD_UNROLL > 2
                ଅଚଳ int64_t bne01_unroll = bne01 - (bne01 % GGML_VEC_MAD_UNROLL);
                ପାଇଁ (int64_t i01 = bi01; i01 < bne01_unroll; i01 += GGML_VEC_MAD_UNROLL) {
                    ଅଚଳ int64_t i11 = i01;

                    ଦଶମିକ * s0 = (ଦଶମିକ *) ((ଅକ୍ଷର *) src0->data + (          i01*nb01 + i02*nb02 + i03*nb03));
                    ଦଶମିକ * s1 = (ଦଶମିକ *) ((ଅକ୍ଷର *) src1->data + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
                    ଦଶମିକ * d  = (ଦଶମିକ *) ((ଅକ୍ଷର *)  dst->data + (          i1*nb1   + i2*nb2   + i3*nb3));

                    ggml_vec_mad_f32_unroll(ne0, nb01, nb11, d, s0, s1);
                }
                ପାଇଁ (int64_t i01 = bne01_unroll; i01 < bne01; ++i01) {
                    ଅଚଳ int64_t i11 = i01;

                    ଦଶମିକ * s0 = (ଦଶମିକ *) ((ଅକ୍ଷର *) src0->data + (          i01*nb01 + i02*nb02 + i03*nb03));
                    ଦଶମିକ * s1 = (ଦଶମିକ *) ((ଅକ୍ଷର *) src1->data + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
                    ଦଶମିକ * d  = (ଦଶମିକ *) ((ଅକ୍ଷର *)  dst->data + (          i1*nb1   + i2*nb2   + i3*nb3));

                    ggml_vec_mad_f32(ne0, d, s0, *s1);
                }
#ଅନ୍ୟଥା
                ପାଇଁ (int64_t i01 = bi01; i01 < bne01; ++i01) {
                    ଅଚଳ int64_t i11 = i01;

                    ଦଶମିକ * s0 = (ଦଶମିକ *) ((ଅକ୍ଷର *) src0->data + (          i01*nb01 + i02*nb02 + i03*nb03));
                    ଦଶମିକ * s1 = (ଦଶମିକ *) ((ଅକ୍ଷର *) src1->data + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
                    ଦଶମିକ * d  = (ଦଶମିକ *) ((ଅକ୍ଷର *)  dst->data + (          i1*nb1 + i2*nb2 + i3*nb3));

                    ggml_vec_mad_f32(ne0, d, s0, *s1);
                }
#endif
            }
        }
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_out_prod_q_f32(
        ଅଚଳ ggml_compute_params * params,
              ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ଦ୍ୱିମିକ_OP_LOCALS;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ggml_type type = src0->type;
    ggml_to_float_t ଅଚଳ dequantize_row_q = ggml_get_type_traits(type)->to_float;

    GGML_ASSERT(ne02 == ne12);
    GGML_ASSERT(ne03 == ne13);
    GGML_ASSERT(ne2  == ne12);
    GGML_ASSERT(ne3  == ne13);

    // we don't support permuted src0 dim0
    GGML_ASSERT(nb00 == ggml_type_size(type));

    // dst dim0 cannot be transposed or permuted
    GGML_ASSERT(nb0 == ଆକାର(ଦଶମିକ));
    // GGML_ASSERT(nb0 <= nb1);
    // GGML_ASSERT(nb1 <= nb2);
    // GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    ଯଦି (ith == 0) {
        ggml_vec_set_f32(ne0*ne1*ne2*ne3, (ଦଶମିକ *)dst->data, 0);
    }
    ggml_barrier(params->threadpool);

    // parallelize by last three dimensions

    // total rows in dst
    ଅଚଳ int64_t nr = ne1*ne2*ne3;

    // rows per thread
    ଅଚଳ int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ int64_t ir0 = dr*ith;
    ଅଚଳ int64_t ir1 = MIN(ir0 + dr, nr);

    // dst[:,:,:,:] = 0
    // for i2,i3:
    //   for i1:
    //     for i01:
    //       for i0:
    //         dst[i0,i1,i2,i3] += src0[i0,i01,i2,i3] * src1[i1,i01,i2,i3]

    ଦଶମିକ * wdata = (ଦଶମିକ *) params->wdata + (ne0 + CACHE_ଧାଡ଼ି_SIZE_F32) * ith;

    ପାଇଁ (int64_t ir = ir0; ir < ir1; ++ir) {
        // dst indices
        ଅଚଳ int64_t i3 = ir/(ne2*ne1);
        ଅଚଳ int64_t i2 = (ir - i3*ne2*ne1)/ne1;
        ଅଚଳ int64_t i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ଅଚଳ int64_t i02 = i2;
        ଅଚଳ int64_t i03 = i3;

        //const int64_t i10 = i1;
        ଅଚଳ int64_t i12 = i2;
        ଅଚଳ int64_t i13 = i3;

        ପାଇଁ (int64_t i01 = 0; i01 < ne01; ++i01) {
            ଅଚଳ int64_t i11 = i01;

            ଦଶମିକ * s0 = (ଦଶମିକ *) ((ଅକ୍ଷର *) src0->data + (          i01*nb01 + i02*nb02 + i03*nb03));
            ଦଶମିକ * s1 = (ଦଶମିକ *) ((ଅକ୍ଷର *) src1->data + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
            ଦଶମିକ * d  = (ଦଶମିକ *) ((ଅକ୍ଷର *)  dst->data + (          i1*nb1 + i2*nb2 + i3*nb3));

            dequantize_row_q(s0, wdata, ne0);
            ggml_vec_mad_f32(ne0, d, wdata, *s1);
        }
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_out_prod_f16_f32(
        ଅଚଳ ggml_compute_params * params,
              ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ଦ୍ୱିମିକ_OP_LOCALS;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ne02 == ne12);
    GGML_ASSERT(ne03 == ne13);
    GGML_ASSERT(ne2  == ne12);
    GGML_ASSERT(ne3  == ne13);

    GGML_ASSERT(nb00 == ଆକାର(ggml_fp16_t));
    GGML_ASSERT(nb0  == ଆକାର(ଦଶମିକ));

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    ଯଦି (ith == 0) {
        ggml_vec_set_f32(ne0*ne1*ne2*ne3, (ଦଶମିକ *)dst->data, 0);
    }
    ggml_barrier(params->threadpool);

    ଅଚଳ int64_t nr = ne1*ne2*ne3;
    ଅଚଳ int64_t dr = (nr + nth - 1)/nth;
    ଅଚଳ int64_t ir0 = dr*ith;
    ଅଚଳ int64_t ir1 = MIN(ir0 + dr, nr);

    ଦଶମିକ * wdata = (ଦଶମିକ *) params->wdata + (ne0 + CACHE_ଧାଡ଼ି_SIZE_F32) * ith;

    ପାଇଁ (int64_t ir = ir0; ir < ir1; ++ir) {
        ଅଚଳ int64_t i3 = ir/(ne2*ne1);
        ଅଚଳ int64_t i2 = (ir - i3*ne2*ne1)/ne1;
        ଅଚଳ int64_t i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ଅଚଳ int64_t i02 = i2;
        ଅଚଳ int64_t i03 = i3;

        ଅଚଳ int64_t i12 = i2;
        ଅଚଳ int64_t i13 = i3;

        ଦଶମିକ * d = (ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data + (i1*nb1 + i2*nb2 + i3*nb3));

        ପାଇଁ (int64_t i01 = 0; i01 < ne01; ++i01) {
            ଅଚଳ int64_t i11 = i01;
            ggml_fp16_t * s0 = (ggml_fp16_t *) ((ଅକ୍ଷର *) src0->data + (i01*nb01 + i02*nb02 + i03*nb03));
            ଦଶମିକ * s1 = (ଦଶମିକ *) ((ଅକ୍ଷର *) src1->data + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
            ggml_fp16_to_fp32_row(s0, wdata, ne0);
            ggml_vec_mad_f32(ne0, d, wdata, *s1);
        }
    }
}

ଶୂନ୍ୟ ggml_compute_forward_out_prod(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_Q1_0:
        ସ୍ଥିତି GGML_TYPE_Q2_0:
        ସ୍ଥିତି GGML_TYPE_Q4_0:
        ସ୍ଥିତି GGML_TYPE_Q4_1:
        ସ୍ଥିତି GGML_TYPE_Q5_0:
        ସ୍ଥିତି GGML_TYPE_Q5_1:
        ସ୍ଥିତି GGML_TYPE_Q8_0:
        ସ୍ଥିତି GGML_TYPE_MXFP4:
        ସ୍ଥିତି GGML_TYPE_NVFP4:
        ସ୍ଥିତି GGML_TYPE_Q2_K:
        ସ୍ଥିତି GGML_TYPE_Q3_K:
        ସ୍ଥିତି GGML_TYPE_Q4_K:
        ସ୍ଥିତି GGML_TYPE_Q5_K:
        ସ୍ଥିତି GGML_TYPE_Q6_K:
        ସ୍ଥିତି GGML_TYPE_TQ1_0:
        ସ୍ଥିତି GGML_TYPE_TQ2_0:
        ସ୍ଥିତି GGML_TYPE_IQ2_XXS:
        ସ୍ଥିତି GGML_TYPE_IQ2_XS:
        ସ୍ଥିତି GGML_TYPE_IQ3_XXS:
        ସ୍ଥିତି GGML_TYPE_IQ1_S:
        ସ୍ଥିତି GGML_TYPE_IQ1_M:
        ସ୍ଥିତି GGML_TYPE_IQ4_NL:
        ସ୍ଥିତି GGML_TYPE_IQ4_XS:
        ସ୍ଥିତି GGML_TYPE_IQ3_S:
        ସ୍ଥିତି GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_out_prod_q_f32(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F16:
            {
                ggml_compute_forward_out_prod_f16_f32(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_out_prod_f32(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_scale

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_scale_f32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    ଦଶମିକ s; // scale factor
    ଦଶମିକ b; // bias

    memcpy(&s, (ଦଶମିକ *) dst->op_params + 0, ଆକାର(ଦଶମିକ));
    memcpy(&b, (ଦଶମିକ *) dst->op_params + 1, ଆକାର(ଦଶମିକ));

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nc = src0->ne[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr = ggml_nrows(src0);

    // rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ଅଚଳ size_t nb01 = src0->nb[1];

    ଅଚଳ size_t nb1 = dst->nb[1];

    ଯଦି (b == 0.0f) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = ir0; i1 < ir1; i1++) {
            ଯଦି (dst->data != src0->data) {
                // src0 is same shape as dst => same indices
                // TODO: add x parameter to ggml_vec_scale_f32 and remove this memcpy
                memcpy((ଅକ୍ଷର *)dst->data + i1*nb1, (ଅକ୍ଷର *)src0->data + i1*nb01, nc * ଆକାର(ଦଶମିକ));
            }
            ggml_vec_scale_f32(nc, (ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data + i1*nb1), s);
        }
    } ଅନ୍ୟଥା {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = ir0; i1 < ir1; i1++) {
            ggml_vec_mad1_f32(nc,
                (ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data  + i1*nb1),
                (ଦଶମିକ *) ((ଅକ୍ଷର *) src0->data + i1*nb1),
                s, b);
        }
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_scale_bf16(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    ଦଶମିକ s; // scale factor
    ଦଶମିକ b; // bias

    memcpy(&s, (ଦଶମିକ *) dst->op_params + 0, ଆକାର(ଦଶମିକ));
    memcpy(&b, (ଦଶମିକ *) dst->op_params + 1, ଆକାର(ଦଶମିକ));

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nc = src0->ne[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr = ggml_nrows(src0);

    // rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = ir0; i1 < ir1; i1++) {
        ଅଚଳ ggml_bf16_t * x = (ଅଚଳ ggml_bf16_t *) ((ଅଚଳ ଅକ୍ଷର *) src0->data + i1*src0->nb[1]);
        ggml_bf16_t       * y = (ggml_bf16_t *)       ((ଅକ୍ଷର *)       dst->data  + i1*dst->nb[1]);
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < nc; k++) {
            y[k] = GGML_FP32_TO_BF16(s*GGML_BF16_TO_FP32(x[k]) + b);
        }
    }
}

ଶୂନ୍ୟ ggml_compute_forward_scale(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_scale_f32(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_BF16:
            {
                ggml_compute_forward_scale_bf16(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_set

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_set_f32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));

    // view src0 and dst with these strides and data offset inbytes during set
    // nb0 is implicitly element_size because src0 and dst are contiguous
    size_t nb1     = ((int32_t *) dst->op_params)[0];
    size_t nb2     = ((int32_t *) dst->op_params)[1];
    size_t nb3     = ((int32_t *) dst->op_params)[2];
    size_t offset  = ((int32_t *) dst->op_params)[3];
    ତାର୍କିକ   inplace = (ତାର୍କିକ) ((int32_t *) dst->op_params)[4];

    ଯଦି (!inplace) {
        ଯଦି (params->ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            memcpy(
                ((ଅକ୍ଷର *)  dst->data),
                ((ଅକ୍ଷର *) src0->data),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr = ggml_nrows(src1);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nc = src1->ne[0];

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne)
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb)

    // src0 and dst as viewed during set
    ଅଚଳ size_t nb0 = ggml_element_size(src0);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା im0 = (ne10 == 0 ? 0 : ne10-1);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା im1 = (ne11 == 0 ? 0 : ne11-1);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା im2 = (ne12 == 0 ? 0 : ne12-1);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା im3 = (ne13 == 0 ? 0 : ne13-1);

    GGML_ASSERT(offset + im0*nb0  + im1*nb1  + im2*nb2  + im3*nb3  <= ggml_nbytes(dst));

    GGML_ASSERT(nb10 == ଆକାର(ଦଶମିକ));

    // rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are viewed with shape of src1 and offset
        // => same indices
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i3 = ir/(ne12*ne11);
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i2 = (ir - i3*ne12*ne11)/ne11;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = (ir - i3*ne12*ne11 - i2*ne11);

        ggml_vec_cpy_f32(nc,
                (ଦଶମିକ *) ((ଅକ୍ଷର *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + offset),
                (ଦଶମିକ *) ((ଅକ୍ଷର *) src1->data + i3*nb13 + i2*nb12 + i1*nb11));
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_set_i32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));

    // view src0 and dst with these strides and data offset inbytes during set
    // nb0 is implicitly element_size because src0 and dst are contiguous
    size_t nb1     = ((int32_t *) dst->op_params)[0];
    size_t nb2     = ((int32_t *) dst->op_params)[1];
    size_t nb3     = ((int32_t *) dst->op_params)[2];
    size_t offset  = ((int32_t *) dst->op_params)[3];
    ତାର୍କିକ   inplace = (ତାର୍କିକ) ((int32_t *) dst->op_params)[4];

    ଯଦି (!inplace) {
        ଯଦି (params->ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            memcpy(
                ((ଅକ୍ଷର *)  dst->data),
                ((ଅକ୍ଷର *) src0->data),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr = ggml_nrows(src1);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nc = src1->ne[0];

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne)
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb)

    // src0 and dst as viewed during set
    ଅଚଳ size_t nb0 = ggml_element_size(src0);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା im0 = (ne10 == 0 ? 0 : ne10-1);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା im1 = (ne11 == 0 ? 0 : ne11-1);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା im2 = (ne12 == 0 ? 0 : ne12-1);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା im3 = (ne13 == 0 ? 0 : ne13-1);

    GGML_ASSERT(offset + im0*nb0  + im1*nb1  + im2*nb2  + im3*nb3  <= ggml_nbytes(dst));

    GGML_ASSERT(nb10 == ଆକାର(int32_t));

    // rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are viewed with shape of src1 and offset
        // => same indices
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i3 = ir/(ne12*ne11);
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i2 = (ir - i3*ne12*ne11)/ne11;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = (ir - i3*ne12*ne11 - i2*ne11);

        ggml_vec_cpy_i32(nc,
                (int32_t *) ((ଅକ୍ଷର *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + offset),
                (int32_t *) ((ଅକ୍ଷର *) src1->data + i3*nb13 + i2*nb12 + i1*nb11));
    }
}

ଶୂନ୍ୟ ggml_compute_forward_set(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_set_f32(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_I32:
            {
                ggml_compute_forward_set_i32(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F16:
        ସ୍ଥିତି GGML_TYPE_BF16:
        ସ୍ଥିତି GGML_TYPE_Q1_0:
        ସ୍ଥିତି GGML_TYPE_Q2_0:
        ସ୍ଥିତି GGML_TYPE_Q4_0:
        ସ୍ଥିତି GGML_TYPE_Q4_1:
        ସ୍ଥିତି GGML_TYPE_Q5_0:
        ସ୍ଥିତି GGML_TYPE_Q5_1:
        ସ୍ଥିତି GGML_TYPE_Q8_0:
        ସ୍ଥିତି GGML_TYPE_Q8_1:
        ସ୍ଥିତି GGML_TYPE_MXFP4:
        ସ୍ଥିତି GGML_TYPE_NVFP4:
        ସ୍ଥିତି GGML_TYPE_Q2_K:
        ସ୍ଥିତି GGML_TYPE_Q3_K:
        ସ୍ଥିତି GGML_TYPE_Q4_K:
        ସ୍ଥିତି GGML_TYPE_Q5_K:
        ସ୍ଥିତି GGML_TYPE_Q6_K:
        ସ୍ଥିତି GGML_TYPE_TQ1_0:
        ସ୍ଥିତି GGML_TYPE_TQ2_0:
        ସ୍ଥିତି GGML_TYPE_IQ2_XXS:
        ସ୍ଥିତି GGML_TYPE_IQ2_XS:
        ସ୍ଥିତି GGML_TYPE_IQ3_XXS:
        ସ୍ଥିତି GGML_TYPE_IQ1_S:
        ସ୍ଥିତି GGML_TYPE_IQ1_M:
        ସ୍ଥିତି GGML_TYPE_IQ4_NL:
        ସ୍ଥିତି GGML_TYPE_IQ4_XS:
        ସ୍ଥିତି GGML_TYPE_IQ3_S:
        ସ୍ଥିତି GGML_TYPE_IQ2_S:
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_cpy

ଶୂନ୍ୟ ggml_compute_forward_cpy(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_dup(params, dst);
}

// ggml_compute_forward_cont

ଶୂନ୍ୟ ggml_compute_forward_cont(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_dup(params, dst);
}

// ggml_compute_forward_get_rows

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_get_rows_q(
        ଅଚଳ ggml_compute_params * params,
              ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ଦ୍ୱିମିକ_OP_LOCALS

    ଅଚଳ int64_t nc = ne00;
    ଅଚଳ int64_t nr = ggml_nelements(src1);

    ଅଚଳ ggml_type type = src0->type;
    ggml_to_float_t ଅଚଳ dequantize_row_q = ggml_get_type_traits(type)->to_float;

    ପ୍ରତିଜ୍ଞା(ne0  == nc);
    ପ୍ରତିଜ୍ଞା(ne02 == ne11);
    ପ୍ରତିଜ୍ଞା(nb00 == ggml_type_size(type));
    ପ୍ରତିଜ୍ଞା(ggml_nrows(dst) == nr);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    // rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ପାଇଁ (int64_t i = ir0; i < ir1; ++i) {
        ଅଚଳ int64_t i12 = i/(ne11*ne10);
        ଅଚଳ int64_t i11 = (i - i12*ne11*ne10)/ne10;
        ଅଚଳ int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        ଅଚଳ int64_t i01 = *(int32_t *) ((ଅକ୍ଷର *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        dequantize_row_q(
                (ଅଚଳ ଶୂନ୍ୟ *) ((ଅକ୍ଷର *) src0->data + i01*nb01 + i11*nb02 + i12*nb03),
                     (ଦଶମିକ *) ((ଅକ୍ଷର *)  dst->data + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_get_rows_f16(
        ଅଚଳ ggml_compute_params * params,
              ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ଦ୍ୱିମିକ_OP_LOCALS

    ଅଚଳ int64_t nc = ne00;
    ଅଚଳ int64_t nr = ggml_nelements(src1);

    ପ୍ରତିଜ୍ଞା(ne0  == nc);
    ପ୍ରତିଜ୍ଞା(ne02 == ne11);
    ପ୍ରତିଜ୍ଞା(nb00 == ଆକାର(ggml_fp16_t));
    ପ୍ରତିଜ୍ଞା(ggml_nrows(dst) == nr);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    // rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ପାଇଁ (int64_t i = ir0; i < ir1; ++i) {
        ଅଚଳ int64_t i12 = i/(ne11*ne10);
        ଅଚଳ int64_t i11 = (i - i12*ne11*ne10)/ne10;
        ଅଚଳ int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        ଅଚଳ int64_t i01 = *(int32_t *) ((ଅକ୍ଷର *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        ggml_cpu_fp16_to_fp32(
            (ଅଚଳ ggml_fp16_t*) ((ଅକ୍ଷର *) src0->data + i01*nb01 + i11*nb02 + i12*nb03),
                       (ଦଶମିକ *) ((ଅକ୍ଷର *)  dst->data + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_get_rows_bf16(
        ଅଚଳ ggml_compute_params * params,
              ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ଦ୍ୱିମିକ_OP_LOCALS

    ଅଚଳ int64_t nc = ne00;
    ଅଚଳ int64_t nr = ggml_nelements(src1);

    ପ୍ରତିଜ୍ଞା(ne0  == nc);
    ପ୍ରତିଜ୍ଞା(ne02 == ne11);
    ପ୍ରତିଜ୍ଞା(nb00 == ଆକାର(ggml_bf16_t));
    ପ୍ରତିଜ୍ଞା(ggml_nrows(dst) == nr);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    // rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ପାଇଁ (int64_t i = ir0; i < ir1; ++i) {
        ଅଚଳ int64_t i12 = i/(ne11*ne10);
        ଅଚଳ int64_t i11 = (i - i12*ne11*ne10)/ne10;
        ଅଚଳ int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        ଅଚଳ int64_t i01 = *(int32_t *) ((ଅକ୍ଷର *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        ggml_cpu_bf16_to_fp32(
            (ଅଚଳ ggml_bf16_t *) ((ଅକ୍ଷର *) src0->data + i01*nb01 + i11*nb02 + i12*nb03),
                        (ଦଶମିକ *) ((ଅକ୍ଷର *)  dst->data + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_get_rows_f32(
        ଅଚଳ ggml_compute_params * params,
              ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ଦ୍ୱିମିକ_OP_LOCALS

    ଅଚଳ int64_t nc = ne00;
    ଅଚଳ int64_t nr = ggml_nelements(src1);

    ପ୍ରତିଜ୍ଞା(ne0  == nc);
    ପ୍ରତିଜ୍ଞା(ne02 == ne11);
    ପ୍ରତିଜ୍ଞା(nb00 == ଆକାର(ଦଶମିକ));
    ପ୍ରତିଜ୍ଞା(ggml_nrows(dst) == nr);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    // rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ପାଇଁ (int64_t i = ir0; i < ir1; ++i) {
        ଅଚଳ int64_t i12 = i/(ne11*ne10);
        ଅଚଳ int64_t i11 = (i - i12*ne11*ne10)/ne10;
        ଅଚଳ int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        ଅଚଳ int64_t i01 = *(int32_t *) ((ଅକ୍ଷର *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        ggml_vec_cpy_f32(nc,
                (ଦଶମିକ *) ((ଅକ୍ଷର *)  dst->data + i10*nb1  + i11*nb2  + i12*nb3),
                (ଦଶମିକ *) ((ଅକ୍ଷର *) src0->data + i01*nb01 + i11*nb02 + i12*nb03));
    }
}

ଶୂନ୍ୟ ggml_compute_forward_get_rows(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_Q1_0:
        ସ୍ଥିତି GGML_TYPE_Q2_0:
        ସ୍ଥିତି GGML_TYPE_Q4_0:
        ସ୍ଥିତି GGML_TYPE_Q4_1:
        ସ୍ଥିତି GGML_TYPE_Q5_0:
        ସ୍ଥିତି GGML_TYPE_Q5_1:
        ସ୍ଥିତି GGML_TYPE_Q8_0:
        ସ୍ଥିତି GGML_TYPE_Q8_1:
        ସ୍ଥିତି GGML_TYPE_MXFP4:
        ସ୍ଥିତି GGML_TYPE_NVFP4:
        ସ୍ଥିତି GGML_TYPE_Q2_K:
        ସ୍ଥିତି GGML_TYPE_Q3_K:
        ସ୍ଥିତି GGML_TYPE_Q4_K:
        ସ୍ଥିତି GGML_TYPE_Q5_K:
        ସ୍ଥିତି GGML_TYPE_Q6_K:
        ସ୍ଥିତି GGML_TYPE_TQ1_0:
        ସ୍ଥିତି GGML_TYPE_TQ2_0:
        ସ୍ଥିତି GGML_TYPE_IQ2_XXS:
        ସ୍ଥିତି GGML_TYPE_IQ2_XS:
        ସ୍ଥିତି GGML_TYPE_IQ3_XXS:
        ସ୍ଥିତି GGML_TYPE_IQ1_S:
        ସ୍ଥିତି GGML_TYPE_IQ1_M:
        ସ୍ଥିତି GGML_TYPE_IQ4_NL:
        ସ୍ଥିତି GGML_TYPE_IQ4_XS:
        ସ୍ଥିତି GGML_TYPE_IQ3_S:
        ସ୍ଥିତି GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_get_rows_q(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F16:
            {
                ggml_compute_forward_get_rows_f16(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_BF16:
            {
                ggml_compute_forward_get_rows_bf16(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F32:
        ସ୍ଥିତି GGML_TYPE_I32:
            {
                ggml_compute_forward_get_rows_f32(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
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
ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_set_rows_impl(
        ଅଚଳ ggml_compute_params * params,
              ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ଦ୍ୱିମିକ_OP_LOCALS

    ଅଚଳ int64_t nc = ne00;
    ଅଚଳ int64_t nr = ne01;

    ପ୍ରତିଜ୍ଞା(ne0  == nc);
    ପ୍ରତିଜ୍ଞା(ne2  == ne02);
    ପ୍ରତିଜ୍ଞା(ne3  == ne03);
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    ପ୍ରତିଜ୍ଞା(ne02 % ne11 == 0);
    ପ୍ରତିଜ୍ଞା(ne03 % ne12 == 0);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    // rows per thread
    ଅଚଳ int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ int64_t ir0 = dr*ith;
    ଅଚଳ int64_t ir1 = std::min(ir0 + dr, nr);

    ଅଚଳ size_t rs = ggml_row_size(src0->type, nc);

    ggml_from_float_t ଅଚଳ from_float = ggml_get_type_traits_cpu(dst->type)->from_float;

    ପାଇଁ (int64_t i03 = 0; i03 < ne03; ++i03) {
        ପାଇଁ (int64_t i02 = 0; i02 < ne02; ++i02) {
            ପାଇଁ (int64_t i = ir0; i < ir1; ++i) {
                ଅଚଳ int64_t i12 = i03%ne12;
                ଅଚଳ int64_t i11 = i02%ne11;
                ଅଚଳ int64_t i10 = i;

                ଅଚଳ int64_t i1 = *(idx_t *) ((ଅକ୍ଷର *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);

                GGML_ASSERT(i1 >= 0 && i1 < ne1);

                ଯଦି constexpr (std::is_same_v<src_t, ଦଶମିକ>) {
                    from_float(
                            (ଅଚଳ ଦଶମିକ *) ((ଅକ୍ଷର *) src0->data +  i*nb01 + i02*nb02 + i03*nb03),
                                            ((ଅକ୍ଷର *)  dst->data + i1*nb1  + i02*nb2  + i03*nb3), nc);
                } ଅନ୍ୟଥା ଯଦି constexpr (std::is_same_v<src_t, ggml_fp16_t>) {
                    ଯଦି (dst->type == GGML_TYPE_F16) {
                        memcpy(
                                            ((ଅକ୍ଷର *)  dst->data + i1*nb1  + i02*nb2  + i03*nb3),
                                            ((ଅକ୍ଷର *) src0->data +  i*nb01 + i02*nb02 + i03*nb03),
                                            rs);
                    } ଅନ୍ୟଥା {
                        ଦଶମିକ * wdata = (ଦଶମିକ *) params->wdata + (nc + CACHE_ଧାଡ଼ି_SIZE_F32) * ith;
                        ggml_fp16_to_fp32_row(
                                (ଅଚଳ ggml_fp16_t *) ((ଅକ୍ଷର *) src0->data + i*nb01 + i02*nb02 + i03*nb03),
                                wdata, nc);
                        from_float(wdata,
                                ((ଅକ୍ଷର *) dst->data + i1*nb1 + i02*nb2 + i03*nb3), nc);
                    }
                } ଅନ୍ୟଥା {
                    GGML_ABORT("src0->type = %d (%s) not supported", src0->type, ggml_type_name(src0->type));
                }
            }
        }
    }
}

ଶୂନ୍ୟ ggml_compute_forward_set_rows(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ଯଦି (src1->type == GGML_TYPE_I64) {
                    ggml_compute_forward_set_rows_impl<ଦଶମିକ, int64_t>(params, dst);
                } ଅନ୍ୟଥା ଯଦି (src1->type == GGML_TYPE_I32) {
                    ggml_compute_forward_set_rows_impl<ଦଶମିକ, int32_t>(params, dst);
                } ଅନ୍ୟଥା {
                    GGML_ABORT("src1->type = %d (%s) not supported", src1->type, ggml_type_name(src1->type));
                }
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F16:
            {
                ଯଦି (src1->type == GGML_TYPE_I64) {
                    ggml_compute_forward_set_rows_impl<ggml_fp16_t, int64_t>(params, dst);
                } ଅନ୍ୟଥା ଯଦି (src1->type == GGML_TYPE_I32) {
                    ggml_compute_forward_set_rows_impl<ggml_fp16_t, int32_t>(params, dst);
                } ଅନ୍ୟଥା {
                    GGML_ABORT("src1->type = %d (%s) not supported", src1->type, ggml_type_name(src1->type));
                }
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("src0->type = %d (%s) not supported", src0->type, ggml_type_name(src0->type));
            }
    }
}

// ggml_compute_forward_get_rows_back

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_get_rows_back_f32_f16(
        ଅଚଳ ggml_compute_params * params,
              ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];

    ଯଦି (params->ith != 0) {
        ଫେରନ୍ତୁ;
    }

    GGML_ASSERT(ggml_is_contiguous(dst));

    // ggml_compute_forward_dup_same_cont(params, opt0, dst);

    memset(dst->data, 0, ggml_nbytes(dst));

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nc = src0->ne[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr = ggml_nelements(src1);

    GGML_ASSERT( dst->ne[0] == nc);
    GGML_ASSERT(src0->nb[0] == ଆକାର(ggml_fp16_t));

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < nr; ++i) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା r = ((int32_t *) src1->data)[i];

        GGML_ASSERT(r >= 0 && r < dst->ne[1]);

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < nc; ++j) {
            ggml_fp16_t v = ((ggml_fp16_t *) ((ଅକ୍ଷର *) src0->data + i*src0->nb[1]))[j];
            ((ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data + r*dst->nb[1]))[j] += GGML_CPU_FP16_TO_FP32(v);
        }
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_get_rows_back_f32(
        ଅଚଳ ggml_compute_params * params,
              ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];

    ଯଦି (params->ith != 0) {
        ଫେରନ୍ତୁ;
    }

    GGML_ASSERT(ggml_is_contiguous(dst));

    // ggml_compute_forward_dup_same_cont(params, opt0, dst);

    memset(dst->data, 0, ggml_nbytes(dst));

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nc = src0->ne[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr = ggml_nelements(src1);

    GGML_ASSERT( dst->ne[0] == nc);
    GGML_ASSERT(src0->nb[0] == ଆକାର(ଦଶମିକ));

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < nr; ++i) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା r = ((int32_t *) src1->data)[i];

        GGML_ASSERT(r >= 0 && r < dst->ne[1]);

        ggml_vec_add_f32(nc,
                (ଦଶମିକ *) ((ଅକ୍ଷର *)  dst->data + r*dst->nb[1]),
                (ଦଶମିକ *) ((ଅକ୍ଷର *)  dst->data + r*dst->nb[1]),
                (ଦଶମିକ *) ((ଅକ୍ଷର *) src0->data + i*src0->nb[1]));
    }
}

ଶୂନ୍ୟ ggml_compute_forward_get_rows_back(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F16:
            {
                ggml_compute_forward_get_rows_back_f32_f16(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_get_rows_back_f32(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
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

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_diag_f32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ଯଦି (params->ith != 0) {
        ଫେରନ୍ତୁ;
    }

    // TODO: handle transposed/permuted matrices

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(ne00 == ne0);
    GGML_ASSERT(ne00 == ne1);
    GGML_ASSERT(ne01 == 1);
    GGML_ASSERT(ne02 == ne2);
    GGML_ASSERT(ne03 == ne3);

    GGML_ASSERT(nb00 == ଆକାର(ଦଶମିକ));
    GGML_ASSERT(nb0  == ଆକାର(ଦଶମିକ));

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i3 = 0; i3 < ne3; i3++) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i2 = 0; i2 < ne2; i2++) {
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = 0; i1 < ne1; i1++) {
                ଦଶମିକ * d = (ଦଶମିକ *)((ଅକ୍ଷର *)  dst->data + i3*nb3  + i2*nb2 + i1*nb1);
                ଦଶମିକ * s = (ଦଶମିକ *)((ଅକ୍ଷର *) src0->data + i3*nb03 + i2*nb02);
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0 = 0; i0 < i1; i0++) {
                    d[i0] = 0;
                }
                d[i1] = s[i1];
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0 = i1+1; i0 < ne0; i0++) {
                    d[i0] = 0;
                }
            }
        }
    }
}

ଶୂନ୍ୟ ggml_compute_forward_diag(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_diag_f32(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_diag_mask_inf

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_diag_mask_f32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst,
        ଅଚଳ ଦଶମିକ value) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା  n_past  = ((int32_t *) dst->op_params)[0];
    ଅଚଳ ତାର୍କିକ inplace = src0->data == dst->data;

    GGML_ASSERT(n_past >= 0);

    ଯଦି (!inplace) {
        ଯଦି (ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
            GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));
            memcpy(
                ((ଅକ୍ଷର *)  dst->data),
                ((ଅକ୍ଷର *) src0->data),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    // TODO: handle transposed/permuted matrices

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n  = ggml_nrows(src0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nc = src0->ne[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr = src0->ne[1];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nz = n/nr;

    GGML_ASSERT( dst->nb[0] == ଆକାର(ଦଶମିକ));
    GGML_ASSERT(src0->nb[0] == ଆକାର(ଦଶମିକ));

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < nz; k++) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = ith; j < nr; j += nth) {
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = n_past; i < nc; i++) {
                ଯଦି (i > n_past + j) {
                    *(ଦଶମିକ *)((ଅକ୍ଷର *) dst->data + k*dst->nb[2] + j*dst->nb[1] + i*dst->nb[0]) = value;
                }
            }
        }
    }
}

ଶୂନ୍ୟ ggml_compute_forward_diag_mask_inf(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_diag_mask_f32(params, dst, -INFINITY);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ଶୂନ୍ୟ ggml_compute_forward_diag_mask_zero(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_diag_mask_f32(params, dst, 0);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_soft_max

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_soft_max_f32(
        ଅଚଳ ggml_compute_params * params,
              ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];
    ଅଚଳ ggml_tensor * src2 = dst->src[2];

    ପ୍ରତିଜ୍ଞା(ggml_is_contiguous(dst));
    ପ୍ରତିଜ୍ଞା(ggml_are_same_shape(src0, dst));

    ଦଶମିକ scale    = 1.0f;
    ଦଶମିକ max_bias = 0.0f;

    memcpy(&scale,    (ଦଶମିକ *) dst->op_params + 0, ଆକାର(ଦଶମିକ));
    memcpy(&max_bias, (ଦଶମିକ *) dst->op_params + 1, ଆକାର(ଦଶମିକ));

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    ଅଚଳ int64_t nb11 = src1 ? src1->nb[1] : 1;
    ଅଚଳ int64_t nb12 = src1 ? src1->nb[2] : 1;
    ଅଚଳ int64_t nb13 = src1 ? src1->nb[3] : 1;

    ଅଚଳ int64_t ne12 = src1 ? src1->ne[2] : 1;
    ଅଚଳ int64_t ne13 = src1 ? src1->ne[3] : 1;

    // TODO: is this supposed to be ceil instead of floor?
    //       https://huggingface.co/mosaicml/mpt-7b/blob/main/attention.py#L370
    ଅଚଳ uint32_t n_head      = ne02;
    ଅଚଳ uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));

    ଅଚଳ ଦଶମିକ m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    ଅଚଳ ଦଶମିକ m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    ଦଶମିକ * wp = (ଦଶମିକ *) params->wdata + (ne00 + CACHE_ଧାଡ଼ି_SIZE_F32) * ith;

    ଅଚଳ ତାର୍କିକ use_f16 = (src1 && src1->type == GGML_TYPE_F16);

    // sinks
    ଅଚଳ ଦଶମିକ * sk = src2 ? (ଦଶମିକ *)((ଅକ୍ଷର *) src2->data) : nullptr;

    ପାଇଁ (int64_t i03 = 0; i03 < ne03; i03++) {
        ପାଇଁ (int64_t i02 = 0; i02 < ne02; i02++) {
            ପାଇଁ (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                ଅଚଳ int64_t i11 = i01;
                ଅଚଳ int64_t i12 = i02%ne12;
                ଅଚଳ int64_t i13 = i03%ne13;

                // ALiBi
                ଅଚଳ uint32_t h = i02; // head
                ଅଚଳ ଦଶମିକ slope = (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1) : 1.0f;

                ଦଶମିକ * sp = (ଦଶମିକ *)((ଅକ୍ଷର *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
                ଦଶମିକ * dp = (ଦଶମିକ *)((ଅକ୍ଷର *)  dst->data + i01*nb1  + i02*nb2  + i03*nb3);

                // broadcast the mask across rows
                ggml_fp16_t * mp_f16 = src1 ? (ggml_fp16_t *)((ଅକ୍ଷର *) src1->data + i11*nb11 + i12*nb12 + i13*nb13) : ଖାଲି;
                ଦଶମିକ       * mp_f32 = src1 ? (ଦଶମିକ       *)((ଅକ୍ଷର *) src1->data + i11*nb11 + i12*nb12 + i13*nb13) : ଖାଲି;

                ggml_vec_cpy_f32  (ne00, wp, sp);
                ggml_vec_scale_f32(ne00, wp, scale);
                ଯଦି (mp_f32) {
                    ଯଦି (use_f16) {
                        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < ne00; ++i) {
                            wp[i] += slope*GGML_CPU_FP16_TO_FP32(mp_f16[i]);
                        }
                    } ଅନ୍ୟଥା {
                        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < ne00; ++i) {
                            wp[i] += slope*mp_f32[i];
                        }
                    }
                }

#ifndef NDEBUG
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < ne00; ++i) {
                    //printf("p[%d] = %f\n", i, p[i]);
                    ପ୍ରତିଜ୍ଞା(!isnan(wp[i]));
                }
#endif // NDEBUG

                ଦଶମିକ max = -INFINITY;
                ggml_vec_max_f32(ne00, &max, wp);

                // if we have sinks, make a correction as if they were included in the softmax
                ଯଦି (sk) {
                    max = MAX(max, sk[i02]);
                }

                ggml_float sum = ggml_vec_soft_max_f32(ne00, dp, wp, max);
                ପ୍ରତିଜ୍ଞା(sum > 0.0);

                ଯଦି (sk) {
                    sum += (ggml_float) expf(sk[i02] - max);
                }

                sum = 1.0/sum;
                ggml_vec_scale_f32(ne00, dp, sum);

#ifndef NDEBUG
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < ne00; ++i) {
                    ପ୍ରତିଜ୍ଞା(!isnan(dp[i]));
                    ପ୍ରତିଜ୍ଞା(!isinf(dp[i]));
                }
#endif // NDEBUG
            }
        }
    }
}

ଶୂନ୍ୟ ggml_compute_forward_soft_max(
        ଅଚଳ ggml_compute_params * params,
              ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_soft_max_f32(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}


// ggml_compute_forward_soft_max_ext_back

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_soft_max_ext_back_f32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_are_same_shape(src1, dst));

    ଦଶମିକ scale    = 1.0f;
    ଦଶମିକ max_bias = 0.0f;

    memcpy(&scale,    (ଅଚଳ ଦଶମିକ *) dst->op_params + 0, ଆକାର(ଦଶମିକ));
    memcpy(&max_bias, (ଅଚଳ ଦଶମିକ *) dst->op_params + 1, ଆକାର(ଦଶମିକ));

    GGML_ASSERT(max_bias == 0.0f);

    // TODO: handle transposed/permuted matrices

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nc = src0->ne[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr = ggml_nrows(src0);

    // rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = ir0; i1 < ir1; i1++) {
        ଦଶମିକ *dy = (ଦଶମିକ *)((ଅକ୍ଷର *) src0->data + i1*src0->nb[1]);
        ଦଶମିକ *y  = (ଦଶମିକ *)((ଅକ୍ଷର *) src1->data + i1*src1->nb[1]);
        ଦଶମିକ *dx = (ଦଶମିକ *)((ଅକ୍ଷର *) dst->data  + i1*dst->nb[1]);

#ifndef NDEBUG
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            ପ୍ରତିଜ୍ଞା(!isnan(dy[i]));
            ପ୍ରତିଜ୍ଞା(!isnan(y[i]));
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
        ଦଶମିକ dot_y_dy = 0;
        ggml_vec_dot_f32(nc, &dot_y_dy, 0, y, 0, dy, 0, 1);
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < nc; i++) {
            dx[i] = scale * (dy[i] - dot_y_dy) * y[i];
        }

#ifndef NDEBUG
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < nc; ++i) {
            ପ୍ରତିଜ୍ଞା(!isnan(dx[i]));
            ପ୍ରତିଜ୍ଞା(!isinf(dx[i]));
        }
#endif // NDEBUG
    }
}

ଶୂନ୍ୟ ggml_compute_forward_soft_max_ext_back(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_soft_max_ext_back_f32(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_clamp

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_clamp_f32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ଦଶମିକ min;
    ଦଶମିକ max;
    memcpy(&min, (ଦଶମିକ *) dst->op_params + 0, ଆକାର(ଦଶମିକ));
    memcpy(&max, (ଦଶମିକ *) dst->op_params + 1, ଆକାର(ଦଶମିକ));

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n  = ggml_nrows(src0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nc = src0->ne[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == ଆକାର(ଦଶମିକ));
    GGML_ASSERT(nb00 == ଆକାର(ଦଶମିକ));

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = ith; j < n; j += nth) {
        ଅଚଳ int64_t i1 = j % ne01;
        ଅଚଳ int64_t i2 = (j / ne01) % ne02;
        ଅଚଳ int64_t i3 = j / (ne01*ne02);

        ଦଶମିକ * dst_ptr  = (ଦଶମିକ *) ((ଅକ୍ଷର *)  dst->data + i1*nb1  + i2*nb2  + i3*nb3);
        ଦଶମିକ * src0_ptr = (ଦଶମିକ *) ((ଅକ୍ଷର *) src0->data + i1*nb01 + i2*nb02 + i3*nb03);

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < nc; i++) {
            dst_ptr[i] = MAX(MIN(src0_ptr[i], max), min);
        }
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_clamp_f16(
    ଅଚଳ ggml_compute_params * params,
    ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ଦଶମିକ min;
    ଦଶମିକ max;
    memcpy(&min, (ଦଶମିକ *) dst->op_params + 0, ଆକାର(ଦଶମିକ));
    memcpy(&max, (ଦଶମିକ *) dst->op_params + 1, ଆକାର(ଦଶମିକ));

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n  = ggml_nrows(src0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nc = src0->ne[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == ଆକାର(ggml_fp16_t));
    GGML_ASSERT(nb00 == ଆକାର(ggml_fp16_t));

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = ith; j < n; j += nth) {
        ଅଚଳ int64_t i1 = j % ne01;
        ଅଚଳ int64_t i2 = (j / ne01) % ne02;
        ଅଚଳ int64_t i3 = j / (ne01*ne02);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((ଅକ୍ଷର *)  dst->data + i1*nb1  + i2*nb2  + i3*nb3);
        ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((ଅକ୍ଷର *) src0->data + i1*nb01 + i2*nb02 + i3*nb03);

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < nc; i++) {
            ଦଶମିକ v = GGML_CPU_FP16_TO_FP32(src0_ptr[i]);
            dst_ptr[i] = GGML_CPU_FP32_TO_FP16(MAX(MIN(v, max), min));
        }
    }
}

ଶୂନ୍ୟ ggml_compute_forward_clamp(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_clamp_f32(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F16:
            {
                ggml_compute_forward_clamp_f16(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_BF16:
        ସ୍ଥିତି GGML_TYPE_Q1_0:
        ସ୍ଥିତି GGML_TYPE_Q2_0:
        ସ୍ଥିତି GGML_TYPE_Q4_0:
        ସ୍ଥିତି GGML_TYPE_Q4_1:
        ସ୍ଥିତି GGML_TYPE_Q5_0:
        ସ୍ଥିତି GGML_TYPE_Q5_1:
        ସ୍ଥିତି GGML_TYPE_Q8_0:
        ସ୍ଥିତି GGML_TYPE_Q8_1:
        ସ୍ଥିତି GGML_TYPE_MXFP4:
        ସ୍ଥିତି GGML_TYPE_NVFP4:
        ସ୍ଥିତି GGML_TYPE_Q2_K:
        ସ୍ଥିତି GGML_TYPE_Q3_K:
        ସ୍ଥିତି GGML_TYPE_Q4_K:
        ସ୍ଥିତି GGML_TYPE_Q5_K:
        ସ୍ଥିତି GGML_TYPE_Q6_K:
        ସ୍ଥିତି GGML_TYPE_TQ1_0:
        ସ୍ଥିତି GGML_TYPE_TQ2_0:
        ସ୍ଥିତି GGML_TYPE_IQ2_XXS:
        ସ୍ଥିତି GGML_TYPE_IQ2_XS:
        ସ୍ଥିତି GGML_TYPE_IQ3_XXS:
        ସ୍ଥିତି GGML_TYPE_IQ1_S:
        ସ୍ଥିତି GGML_TYPE_IQ1_M:
        ସ୍ଥିତି GGML_TYPE_IQ4_NL:
        ସ୍ଥିତି GGML_TYPE_IQ4_XS:
        ସ୍ଥିତି GGML_TYPE_IQ3_S:
        ସ୍ଥିତି GGML_TYPE_IQ2_S:
        ସ୍ଥିତି GGML_TYPE_Q8_K:
        ସ୍ଥିତି GGML_TYPE_I8:
        ସ୍ଥିତି GGML_TYPE_I16:
        ସ୍ଥିତି GGML_TYPE_I32:
        ସ୍ଥିତି GGML_TYPE_I64:
        ସ୍ଥିତି GGML_TYPE_F64:
        ସ୍ଥିତି GGML_TYPE_ଗଣନା:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_rope

ସ୍ଥିର ଦଶମିକ rope_yarn_ramp(ଅଚଳ ଦଶମିକ ତଳ, ଅଚଳ ଦଶମିକ ଉପର, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0) {
    ଅଚଳ ଦଶମିକ y = (i0 / 2 - ତଳ) / MAX(0.001f, ଉପର - ତଳ);
    ଫେରନ୍ତୁ 1 - MIN(1, MAX(0, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
ସ୍ଥିର ଶୂନ୍ୟ rope_yarn(
    ଦଶମିକ theta_extrap, ଦଶମିକ freq_scale, ଦଶମିକ corr_dims[2], int64_t i0, ଦଶମିକ ext_factor, ଦଶମିକ mscale,
    ଦଶମିକ * cos_theta, ଦଶମିକ * sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    ଦଶମିକ theta_interp = freq_scale * theta_extrap;
    ଦଶମିକ theta = theta_interp;
    ଯଦି (ext_factor != 0.0f) {
        ଦଶମିକ ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_rope_cache_init(
     ଦଶମିକ theta_base, ଦଶମିକ freq_scale, ଅଚଳ ଦଶମିକ * freq_factors, ଦଶମିକ corr_dims[2], int64_t ne0, ଦଶମିକ ext_factor, ଦଶମିକ mscale,
     ଦଶମିକ * cache, ଦଶମିକ sin_sign, ଦଶମିକ theta_scale) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    ଦଶମିକ theta = theta_base;
    ପାଇଁ (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        ଅଚଳ ଦଶମିକ ff = freq_factors ? freq_factors[i0/2] : 1.0f;
        rope_yarn(
            theta/ff, freq_scale, corr_dims, i0, ext_factor, mscale, &cache[i0 + 0], &cache[i0 + 1]
        );
        cache[i0 + 1] *= sin_sign;

        theta *= theta_scale;
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_mrope_cache_init(
     ଦଶମିକ theta_base_t, ଦଶମିକ theta_base_h, ଦଶମିକ theta_base_w, ଦଶମିକ theta_base_e, ପୂର୍ଣ୍ଣସଂଖ୍ୟା sections[4], ତାର୍କିକ is_imrope, ତାର୍କିକ indep_sects,
     ଦଶମିକ freq_scale, ଅଚଳ ଦଶମିକ * freq_factors, ଦଶମିକ corr_dims[2], int64_t ne0, ଦଶମିକ ext_factor, ଦଶମିକ mscale,
     ଦଶମିକ * cache, ଦଶମିକ sin_sign, ଦଶମିକ theta_scale) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    ଦଶମିକ theta_t = theta_base_t;
    ଦଶମିକ theta_h = theta_base_h;
    ଦଶମିକ theta_w = theta_base_w;
    ଦଶମିକ theta_e = theta_base_e;  // extra position id for vision encoder
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା sec_w = sections[1] + sections[0];
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା sec_e = sections[2] + sec_w;
    GGML_ASSERT(sect_dims <= ne0);

    ପାଇଁ (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        ଅଚଳ ଦଶମିକ ff = freq_factors ? freq_factors[i0/2] : 1.0f;

        ପୂର୍ଣ୍ଣସଂଖ୍ୟା sector = (i0 / 2) % sect_dims;
        ଯଦି (indep_sects) {
            // compute theta independently for each dim sections
            // (i.e. reset corresponding theta when `i0` go from one section to another)
            ଯଦି (sector == 0) {
                theta_t = theta_base_t;
            }
            ଅନ୍ୟଥା ଯଦି (sector == sections[0]) {
                theta_h = theta_base_h;;
            }
            ଅନ୍ୟଥା ଯଦି (sector == sec_w) {
                theta_w = theta_base_w;
            }
            ଅନ୍ୟଥା ଯଦି (sector == sec_e) {
                theta_e = theta_base_e;
            }
        }

        ଦଶମିକ theta = theta_t;
        ଯଦି (is_imrope) { // qwen3vl apply interleaved mrope
            ଯଦି (sector % 3 == 1 && sector < 3 * sections[1]) {
                theta = theta_h;
            } ଅନ୍ୟଥା ଯଦି (sector % 3 == 2 && sector < 3 * sections[2]) {
                theta = theta_w;
            } ଅନ୍ୟଥା ଯଦି (sector % 3 == 0 && sector < 3 * sections[0]) {
                theta = theta_t;
            } ଅନ୍ୟଥା {
                theta = theta_e;
            }
        } ଅନ୍ୟଥା {
            ଯଦି (sector >= sections[0] && sector < sec_w) {
                theta = theta_h;
            }
            ଅନ୍ୟଥା ଯଦି (sector >= sec_w && sector < sec_w + sections[2]) {
                theta = theta_w;
            }
            ଅନ୍ୟଥା ଯଦି (sector >= sec_w + sections[2]) {
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
ସ୍ଥିର ଶୂନ୍ୟ rotate_pairs(ଅଚଳ int64_t n, ଅଚଳ int64_t n_offset, ଅଚଳ ଦଶମିକ * cache, ଅଚଳ T * src_data, T * dst_data, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା scale = 2) {
  ପାଇଁ (int64_t i0 = 0; i0 < n; i0 += 2) {
    ଅଚଳ int64_t ic = i0/scale; // hack for GGML_ROPE_TYPE_NORMAL, where we need ic = i0; for all other cases, ic = i0/2

    ଅଚଳ ଦଶମିକ cos_theta = cache[i0 + 0];
    ଅଚଳ ଦଶମିକ sin_theta = cache[i0 + 1];

    ଅଚଳ T * ଅଚଳ src = src_data + ic;
    T * dst             = dst_data + ic;

    ଅଚଳ ଦଶମିକ x0 = type_conversion_table<T>::to_f32(src[0]);
    ଅଚଳ ଦଶମିକ x1 = type_conversion_table<T>::to_f32(src[n_offset]);

    dst[0]        = type_conversion_table<T>::from_f32(x0*cos_theta - x1*sin_theta);
    dst[n_offset] = type_conversion_table<T>::from_f32(x0*sin_theta + x1*cos_theta);
  }
}

template<typename T> //float or ggml_fp16_t
ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_rope_flt(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst,
        ଅଚଳ ତାର୍କିକ forward) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];
    ଅଚଳ ggml_tensor * src2 = dst->src[2];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);

    ଦଶମିକ freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା sections[4];

    //const int n_past     = ((int32_t *) dst->op_params)[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_dims     = ((int32_t *) dst->op_params)[1];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା mode       = ((int32_t *) dst->op_params)[2];
    //const int n_ctx      = ((int32_t *) dst->op_params)[3];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_ctx_orig = ((int32_t *) dst->op_params)[4];

    memcpy(&freq_base,   (int32_t *) dst->op_params +  5, ଆକାର(ଦଶମିକ));
    memcpy(&freq_scale,  (int32_t *) dst->op_params +  6, ଆକାର(ଦଶମିକ));
    memcpy(&ext_factor,  (int32_t *) dst->op_params +  7, ଆକାର(ଦଶମିକ));
    memcpy(&attn_factor, (int32_t *) dst->op_params +  8, ଆକାର(ଦଶମିକ));
    memcpy(&beta_fast,   (int32_t *) dst->op_params +  9, ଆକାର(ଦଶମିକ));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, ଆକାର(ଦଶମିକ));
    memcpy(&sections,    (int32_t *) dst->op_params + 11, ଆକାର(ପୂର୍ଣ୍ଣସଂଖ୍ୟା)*4);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_offs = ((int32_t *) dst->op_params)[15];

    GGML_TENSOR_UNARY_OP_LOCALS

    //printf("ne0: %d, ne1: %d, ne2: %d, ne3: %d\n", ne0, ne1, ne2, ne3);
    //printf("n_past = %d, ne2 = %d\n", n_past, ne2);

    GGML_ASSERT(nb0 == nb00);
    GGML_ASSERT(nb0 == ଆକାର(T));

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr = ggml_nrows(dst);

    GGML_ASSERT(n_dims <= ne0);
    GGML_ASSERT(n_dims % 2 == 0);

    GGML_ASSERT(n_offs >= 0);
    GGML_ASSERT(n_offs % 2 == 0);
    GGML_ASSERT(n_offs + n_dims <= ne0);

    // rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    // row index used to determine which thread to use
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir = 0;

    ଅଚଳ ଦଶମିକ theta_scale = powf(freq_base, -2.0f/n_dims);

    ଦଶମିକ corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    ଅଚଳ ତାର୍କିକ is_imrope = mode == GGML_ROPE_TYPE_IMROPE; // qwen3vl apply interleaved mrope
    ଅଚଳ ତାର୍କିକ mrope_used = mode & GGML_ROPE_TYPE_MROPE;  // ggml_rope_multi, note: also true for vision (24 & 8 == true) and for imrope
    ଅଚଳ ତାର୍କିକ is_vision = mode == GGML_ROPE_TYPE_VISION;

    ଯଦି (mrope_used) {
        GGML_ASSERT(sections[0] > 0 || sections[1] > 0 || sections[2] > 0);
    }

    ଯଦି (is_vision) {
        GGML_ASSERT(n_dims == ne0/2);
        GGML_ASSERT(n_offs == 0);
    }

    ଅଚଳ ଦଶମିକ * freq_factors = ଖାଲି;
    ଯଦି (src2 != ଖାଲି) {
        GGML_ASSERT(src2->type == GGML_TYPE_F32);
        GGML_ASSERT(src2->ne[0] >= n_dims / 2);
        freq_factors = (ଅଚଳ ଦଶମିକ *) src2->data;
    }

    // backward process uses inverse rotation by cos and sin.
    // cos and sin build a rotation matrix, where the inverse is the transpose.
    // this essentially just switches the sign of sin.
    ଅଚଳ ଦଶମିକ sin_sign = forward ? 1.0f : -1.0f;

    ଅଚଳ int32_t * pos = (ଅଚଳ int32_t *) src1->data;

    int64_t last_i2 = -1;

    ପାଇଁ (int64_t i3 = 0; i3 < ne3; i3++) { // batch
        ପାଇଁ (int64_t i2 = 0; i2 < ne2; i2++) { // seq-len
            ପାଇଁ (int64_t i1 = 0; i1 < ne1; i1++) { // attn-heads
                ଯଦି (ir++ < ir0) ଜାରି; // skip rows mapped to other threads
                ଯଦି (ir   > ir1) ଭାଙ୍ଗ;

                ଦଶମିକ * cache = (ଦଶମିକ *) params->wdata + (ne0 + CACHE_ଧାଡ଼ି_SIZE_F32)*ith;
                ଯଦି (last_i2 != i2) {
                    ଯଦି (!mrope_used) {
                        ଅଚଳ int64_t p = pos[i2];
                        ggml_rope_cache_init(p, freq_scale, freq_factors, corr_dims, ne0, ext_factor, attn_factor, cache, sin_sign, theta_scale);
                    }
                    ଅନ୍ୟଥା {
                        ଅଚଳ int64_t p_t = pos[i2];
                        ଅଚଳ int64_t p_h = pos[i2 + ne2];
                        ଅଚଳ int64_t p_w = pos[i2 + ne2 * 2];
                        ଅଚଳ int64_t p_e = pos[i2 + ne2 * 3];
                        ggml_mrope_cache_init(
                            p_t, p_h, p_w, p_e, sections, is_imrope, is_vision,
                            freq_scale, freq_factors, corr_dims, ne0, ext_factor, attn_factor, cache, sin_sign, theta_scale);
                    }

                    last_i2 = i2;
                }

                T * src = (T *)((ଅକ୍ଷର *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
                T * dst_data  = (T *)((ଅକ୍ଷର *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1);

                ବାଛ (mode) {
                    ସ୍ଥିତି GGML_ROPE_TYPE_NORMAL:
                        rotate_pairs<T>(n_dims, 1, cache, src + n_offs, dst_data + n_offs, 1);
                        ଭାଙ୍ଗ;
                    ସ୍ଥିତି GGML_ROPE_TYPE_NEOX:
                    ସ୍ଥିତି GGML_ROPE_TYPE_MROPE:
                    ସ୍ଥିତି GGML_ROPE_TYPE_IMROPE:
                        rotate_pairs<T>(n_dims, n_dims/2, cache, src + n_offs, dst_data + n_offs);
                        ଭାଙ୍ଗ;
                    ସ୍ଥିତି GGML_ROPE_TYPE_VISION:
                        rotate_pairs<T>(ne0, n_dims, cache, src, dst_data);
                        ଭାଙ୍ଗ;
                    ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                        GGML_ABORT("rope type not supported");
                }

                ଯଦି (!is_vision) {
                    // fill the remain channels with data from src tensor
                    ପାଇଁ (int64_t i0 = 0; i0 < ne0; i0 += 2) {
                        ଯଦି (i0 == n_offs) {
                            i0 += n_dims - 2; // skip the rotated channels
                            ଜାରି;
                        }
                        ଅଚଳ T * ଅଚଳ src = (T *)((ଅକ୍ଷର *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                        T * dst_data  = (T *)((ଅକ୍ଷର *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

                        dst_data[0] = src[0];
                        dst_data[1] = src[1];
                    }
                }
            } //attn-heads
        }
    }
}

ଶୂନ୍ୟ ggml_compute_forward_rope(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F16:
            {
                ggml_compute_forward_rope_flt<ggml_fp16_t>(params, dst, ସତ୍ୟ);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_rope_flt<ଦଶମିକ>(params, dst, ସତ୍ୟ);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_rope_back

ଶୂନ୍ୟ ggml_compute_forward_rope_back(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F16:
            {
                ggml_compute_forward_rope_flt<ggml_fp16_t>(params, dst, ମିଥ୍ୟା);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_rope_flt<ଦଶମିକ>(params, dst, ମିଥ୍ୟା);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_conv_transpose_1d

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_conv_transpose_1d_f16_f32(
        ଅଚଳ ggml_compute_params * params,
              ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ଦ୍ୱିମିକ_OP_LOCALS

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nk = ne00*ne01*ne02;

    GGML_ASSERT(nb00 == ଆକାର(ggml_fp16_t));
    GGML_ASSERT(nb10 == ଆକାର(ଦଶମିକ));

    ଯଦି (ith == 0) {
        memset(params->wdata, 0, params->wsize);

        // permute kernel data (src0) from (K x Cout x Cin) to (Cin x K x Cout)
        {
            ggml_fp16_t * ଅଚଳ wdata = (ggml_fp16_t *) params->wdata + 0;

            ପାଇଁ (int64_t i02 = 0; i02 < ne02; i02++) {
                ପାଇଁ (int64_t i01 = 0; i01 < ne01; i01++) {
                    ଅଚଳ ggml_fp16_t * ଅଚଳ src = (ggml_fp16_t *)((ଅକ୍ଷର *) src0->data + i02*nb02 + i01*nb01);
                    ggml_fp16_t * dst_data = wdata + i01*ne00*ne02;
                    ପାଇଁ (int64_t i00 = 0; i00 < ne00; i00++) {
                        dst_data[i00*ne02 + i02] = src[i00];
                    }
                }
            }
        }

        // permute source data (src1) from (L x Cin) to (Cin x L)
        {
            ggml_fp16_t * ଅଚଳ wdata = (ggml_fp16_t *) params->wdata + nk;
            ggml_fp16_t * dst_data = wdata;

            ପାଇଁ (int64_t i11 = 0; i11 < ne11; i11++) {
                ଅଚଳ ଦଶମିକ * ଅଚଳ src = (ଦଶମିକ *)((ଅକ୍ଷର *) src1->data + i11*nb11);
                ପାଇଁ (int64_t i10 = 0; i10 < ne10; i10++) {
                    dst_data[i10*ne11 + i11] = GGML_CPU_FP32_TO_FP16(src[i10]);
                }
            }
        }

        // need to zero dst since we are accumulating into it
        memset(dst->data, 0, ggml_nbytes(dst));
    }
    ggml_barrier(params->threadpool);

    ଅଚଳ int32_t s0 = ((ଅଚଳ int32_t*)(dst->op_params))[0];

    // total rows in dst
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr = ne1;

    // rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ggml_fp16_t * ଅଚଳ wdata     = (ggml_fp16_t *) params->wdata + 0;
    ggml_fp16_t * ଅଚଳ wdata_src = wdata + nk;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = ir0; i1 < ir1; i1++) {
        ଦଶମିକ * dst_data = (ଦଶମିକ *)((ଅକ୍ଷର *) dst->data + i1*nb1);
        ggml_fp16_t * wdata_kernel = wdata + i1*ne02*ne00;
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i10 = 0; i10 < ne10; i10++) {
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1n = i10*ne11;
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i00 = 0; i00 < ne00; i00++) {
                ଦଶମିକ v = 0;
                ggml_vec_dot_f16(ne02, &v, 0,
                        (ggml_fp16_t *)    wdata_src + i1n, 0,
                        (ggml_fp16_t *) wdata_kernel + i00*ne02, 0, 1);
                dst_data[i10*s0 + i00] += v;
            }
        }
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_conv_transpose_1d_f32(
        ଅଚଳ ggml_compute_params * params,
              ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ଦ୍ୱିମିକ_OP_LOCALS

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nk = ne00*ne01*ne02;

    GGML_ASSERT(nb00 == ଆକାର(ଦଶମିକ));
    GGML_ASSERT(nb10 == ଆକାର(ଦଶମିକ));

    ଯଦି (ith == 0) {
        memset(params->wdata, 0, params->wsize);

        // prepare kernel data (src0) from (K x Cout x Cin) to (Cin x K x Cout)
        {
            ଦଶମିକ * ଅଚଳ wdata = (ଦଶମିକ *) params->wdata + 0;

            ପାଇଁ (int64_t i02 = 0; i02 < ne02; i02++) {
                ପାଇଁ (int64_t i01 = 0; i01 < ne01; i01++) {
                    ଅଚଳ ଦଶମିକ * ଅଚଳ src = (ଦଶମିକ *)((ଅକ୍ଷର *) src0->data + i02*nb02 + i01*nb01);
                    ଦଶମିକ * dst_data = wdata + i01*ne00*ne02;
                    ପାଇଁ (int64_t i00 = 0; i00 < ne00; i00++) {
                        dst_data[i00*ne02 + i02] = src[i00];
                    }
                }
            }
        }

        // prepare source data (src1)
        {
            ଦଶମିକ * ଅଚଳ wdata = (ଦଶମିକ *) params->wdata + nk;
            ଦଶମିକ * dst_data = wdata;

            ପାଇଁ (int64_t i11 = 0; i11 < ne11; i11++) {
                ଅଚଳ ଦଶମିକ * ଅଚଳ src = (ଦଶମିକ *)((ଅକ୍ଷର *) src1->data + i11*nb11);
                ପାଇଁ (int64_t i10 = 0; i10 < ne10; i10++) {
                    dst_data[i10*ne11 + i11] = src[i10];
                }
            }
        }

        // need to zero dst since we are accumulating into it
        memset(dst->data, 0, ggml_nbytes(dst));
    }
    ggml_barrier(params->threadpool);

    ଅଚଳ int32_t s0 = ((ଅଚଳ int32_t*)(dst->op_params))[0];

    // total rows in dst
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr = ne1;

    // rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ଦଶମିକ * ଅଚଳ wdata     = (ଦଶମିକ *) params->wdata + 0;
    ଦଶମିକ * ଅଚଳ wdata_src = wdata + nk;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = ir0; i1 < ir1; i1++) {
        ଦଶମିକ * dst_data = (ଦଶମିକ *)((ଅକ୍ଷର *) dst->data + i1*nb1);
        ଦଶମିକ * wdata_kernel = wdata + i1*ne02*ne00;
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i10 = 0; i10 < ne10; i10++) {
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1n = i10*ne11;
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i00 = 0; i00 < ne00; i00++) {
                ଦଶମିକ v = 0;
                ggml_vec_dot_f32(ne02, &v, 0,
                        wdata_src + i1n, 0,
                        wdata_kernel + i00*ne02, 0, 1);
                dst_data[i10*s0 + i00] += v;
            }
        }
    }
}

ଶୂନ୍ୟ ggml_compute_forward_conv_transpose_1d(
        ଅଚଳ ggml_compute_params * params,
              ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F16:
            {
                ggml_compute_forward_conv_transpose_1d_f16_f32(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_conv_transpose_1d_f32(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_im2col_f32
// src0: kernel [OC, IC, KH, KW]
// src1: image [N, IC, IH, IW]
// dst:  result [N, OH, OW, IC*KH*KW]
ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_im2col_f32(
        ଅଚଳ ggml_compute_params * params,
              ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ଦ୍ୱିମିକ_OP_LOCALS;

    ଅଚଳ int32_t s0 = ((ଅଚଳ int32_t *)(dst->op_params))[0];
    ଅଚଳ int32_t s1 = ((ଅଚଳ int32_t *)(dst->op_params))[1];
    ଅଚଳ int32_t p0 = ((ଅଚଳ int32_t *)(dst->op_params))[2];
    ଅଚଳ int32_t p1 = ((ଅଚଳ int32_t *)(dst->op_params))[3];
    ଅଚଳ int32_t d0 = ((ଅଚଳ int32_t *)(dst->op_params))[4];
    ଅଚଳ int32_t d1 = ((ଅଚଳ int32_t *)(dst->op_params))[5];
    ଅଚଳ ତାର୍କିକ is_2D = ((ଅଚଳ int32_t *)(dst->op_params))[6] == 1;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ int64_t N  = is_2D ? ne13 : ne12;
    ଅଚଳ int64_t IC = is_2D ? ne12 : ne11;
    ଅଚଳ int64_t IH = is_2D ? ne11 : 1;
    ଅଚଳ int64_t IW = ne10;

    ଅଚଳ int64_t KH = is_2D ? ne01 : 1;
    ଅଚଳ int64_t KW = ne00;

    ଅଚଳ int64_t OH = is_2D ? ne2 : 1;
    ଅଚଳ int64_t OW = ne1;

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ofs0 = is_2D ? nb13 : nb12;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ofs1 = is_2D ? nb12 : nb11;

    GGML_ASSERT(nb10 == ଆକାର(ଦଶମିକ));

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    {
        ଦଶମିକ * ଅଚଳ wdata = (ଦଶମିକ *) dst->data;

        ପାଇଁ (int64_t in = 0; in < N; in++) {
            ପାଇଁ (int64_t ioh = 0; ioh < OH; ioh++) { // 1
                ପାଇଁ (int64_t iow = 0; iow < OW; iow++) {
                    ପାଇଁ (int64_t iic = ith; iic < IC; iic += nth) {

                        // micro kernel
                        ଦଶମିକ * dst_data = wdata + (in*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
                        ଅଚଳ ଦଶମିକ * ଅଚଳ src_data = (ଦଶମିକ *)((ଅକ୍ଷର *) src1->data + in*ofs0 + iic*ofs1); // [IH, IW]

                        ପାଇଁ (int64_t ikh = 0; ikh < KH; ikh++) {  // 1
                            ପାଇଁ (int64_t ikw = 0; ikw < KW; ikw++) {
                                ଅଚଳ int64_t iiw = iow*s0 + ikw*d0 - p0;
                                ଅଚଳ int64_t iih = ioh*s1 + ikh*d1 - p1;

                                ଯଦି (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                    dst_data[iic*(KH*KW) + ikh*KW + ikw] = 0;
                                } ଅନ୍ୟଥା {
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
ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_im2col_f16(
        ଅଚଳ ggml_compute_params * params,
              ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16);

    GGML_TENSOR_ଦ୍ୱିମିକ_OP_LOCALS;

    ଅଚଳ int32_t s0 = ((ଅଚଳ int32_t *)(dst->op_params))[0];
    ଅଚଳ int32_t s1 = ((ଅଚଳ int32_t *)(dst->op_params))[1];
    ଅଚଳ int32_t p0 = ((ଅଚଳ int32_t *)(dst->op_params))[2];
    ଅଚଳ int32_t p1 = ((ଅଚଳ int32_t *)(dst->op_params))[3];
    ଅଚଳ int32_t d0 = ((ଅଚଳ int32_t *)(dst->op_params))[4];
    ଅଚଳ int32_t d1 = ((ଅଚଳ int32_t *)(dst->op_params))[5];
    ଅଚଳ ତାର୍କିକ is_2D = ((ଅଚଳ int32_t *)(dst->op_params))[6] == 1;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ int64_t N  = is_2D ? ne13 : ne12;
    ଅଚଳ int64_t IC = is_2D ? ne12 : ne11;
    ଅଚଳ int64_t IH = is_2D ? ne11 : 1;
    ଅଚଳ int64_t IW = ne10;

    ଅଚଳ int64_t KH = is_2D ? ne01 : 1;
    ଅଚଳ int64_t KW = ne00;

    ଅଚଳ int64_t OH = is_2D ? ne2 : 1;
    ଅଚଳ int64_t OW = ne1;

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ofs0 = is_2D ? nb13 : nb12;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ofs1 = is_2D ? nb12 : nb11;

    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    {
        ggml_fp16_t * ଅଚଳ wdata = (ggml_fp16_t *) dst->data;

        ପାଇଁ (int64_t in = 0; in < N; in++) {
            ପାଇଁ (int64_t ioh = 0; ioh < OH; ioh++) { // 1
                ପାଇଁ (int64_t iow = 0; iow < OW; iow++) {
                    ପାଇଁ (int64_t iic = ith; iic < IC; iic += nth) {

                        // micro kernel
                        ggml_fp16_t * dst_data = wdata + (in*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
                        ଅଚଳ ଦଶମିକ * ଅଚଳ src_data_f32 = src1->type == GGML_TYPE_F32
                            ? (ଅଚଳ ଦଶମିକ *)((ଅଚଳ ଅକ୍ଷର *) src1->data + in*ofs0 + iic*ofs1)
                            : nullptr; // [IH, IW]
                        ଅଚଳ ggml_fp16_t * ଅଚଳ src_data_f16 = src1->type == GGML_TYPE_F16
                            ? (ଅଚଳ ggml_fp16_t *)((ଅଚଳ ଅକ୍ଷର *) src1->data + in*ofs0 + iic*ofs1)
                            : nullptr; // [IH, IW]

                        ପାଇଁ (int64_t ikh = 0; ikh < KH; ikh++) {  // 1
                            ପାଇଁ (int64_t ikw = 0; ikw < KW; ikw++) {
                                ଅଚଳ int64_t iiw = iow*s0 + ikw*d0 - p0;
                                ଅଚଳ int64_t iih = ioh*s1 + ikh*d1 - p1;

                                ଯଦି (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                    dst_data[iic*(KH*KW) + ikh*KW + ikw] = 0;
                                } ଅନ୍ୟଥା {
                                    ଯଦି (src_data_f32 != nullptr) {
                                        dst_data[iic*(KH*KW) + ikh*KW + ikw] = GGML_CPU_FP32_TO_FP16(src_data_f32[iih*IW + iiw]);
                                    } ଅନ୍ୟଥା {
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

ଶୂନ୍ୟ ggml_compute_forward_im2col(
        ଅଚଳ ggml_compute_params * params,
              ggml_tensor * dst) {
    ବାଛ (dst->type) {
        ସ୍ଥିତି GGML_TYPE_F16:
            {
                ggml_compute_forward_im2col_f16(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_im2col_f32(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_im2col_back_f32

ଶୂନ୍ୟ ggml_compute_forward_im2col_back_f32(
        ଅଚଳ ggml_compute_params * params,
              ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0]; // gradients of forward pass output
    ଅଚଳ ggml_tensor * src1 = dst->src[1]; // convolution kernel

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ଦ୍ୱିମିକ_OP_LOCALS;

    ଅଚଳ int32_t s0 = ((ଅଚଳ int32_t *)(dst->op_params))[0];
    ଅଚଳ int32_t s1 = ((ଅଚଳ int32_t *)(dst->op_params))[1];
    ଅଚଳ int32_t p0 = ((ଅଚଳ int32_t *)(dst->op_params))[2];
    ଅଚଳ int32_t p1 = ((ଅଚଳ int32_t *)(dst->op_params))[3];
    ଅଚଳ int32_t d0 = ((ଅଚଳ int32_t *)(dst->op_params))[4];
    ଅଚଳ int32_t d1 = ((ଅଚଳ int32_t *)(dst->op_params))[5];
    ଅଚଳ ତାର୍କିକ is_2D = ((ଅଚଳ int32_t *)(dst->op_params))[6] == 1;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ int64_t N  = is_2D ? ne3 : ne2;
    ଅଚଳ int64_t IC = is_2D ? ne2 : ne1;
    ଅଚଳ int64_t IH = is_2D ? ne1 : 1;
    ଅଚଳ int64_t IW = ne0;

    ଅଚଳ int64_t KH = is_2D ? ne11 : 1;
    ଅଚଳ int64_t KW = ne10;

    ଅଚଳ int64_t OH = is_2D ? ne02 : 1;
    ଅଚଳ int64_t OW = ne01;

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ofs0 = is_2D ? nb3 : nb2;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ofs1 = is_2D ? nb2 : nb1;

    GGML_ASSERT(nb0  == ଆକାର(ଦଶମିକ));

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    {
        ଦଶମିକ * ଅଚଳ wdata = (ଦଶମିକ *) dst->data;

        ପାଇଁ (int64_t in = 0; in < N; in++) {
            ପାଇଁ (int64_t iic = ith; iic < IC; iic += nth) {
                ପାଇଁ (int64_t iih = 0; iih < IH; iih++) {
                    ପାଇଁ (int64_t iiw = 0; iiw < IW; iiw++) {

                        // micro kernel
                        ଦଶମିକ grad = 0.0f;
                        ପାଇଁ (int64_t ikh = 0; ikh < KH; ikh++) {
                            ପାଇଁ (int64_t ikw = 0; ikw < KW; ikw++) {
                                // For s0 > 1 some values were skipped over in the forward pass.
                                // These values have tmpw % s0 != 0 and need to be skipped in the backwards pass as well.
                                ଅଚଳ int64_t tmpw = (iiw + p0 - ikw*d0);
                                ଯଦି (tmpw % s0 != 0) {
                                    ଜାରି;
                                }
                                ଅଚଳ int64_t iow = tmpw / s0;

                                // Equivalent logic as above except for s1.
                                int64_t ioh;
                                ଯଦି (is_2D) {
                                    ଅଚଳ int64_t tmph = iih + p1 - ikh*d1;

                                    ଯଦି (tmph % s1 != 0) {
                                        ଜାରି;
                                    }

                                    ioh = tmph / s1;
                                } ଅନ୍ୟଥା {
                                    ioh = 0;
                                }

                                ଯଦି (iow < 0 || iow >= OW || ioh < 0 || ioh >= OH) {
                                    ଜାରି;
                                }

                                ଅଚଳ ଦଶମିକ * ଅଚଳ grad_in = (ଅଚଳ ଦଶମିକ *) src0->data
                                    + (in*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
                                grad += grad_in[iic*(KH*KW) + ikh*KW + ikw];
                            }
                        }
                        ଦଶମିକ * dst_data = (ଦଶମିକ *)((ଅକ୍ଷର *) wdata + (in*ofs0 + iic*ofs1)); // [IH, IW]
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
ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_im2col_3d_f16(
        ଅଚଳ ggml_compute_params * params,
              ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16);

    GGML_TENSOR_ଦ୍ୱିମିକ_OP_LOCALS;

    ଅଚଳ int32_t s0 = ((ଅଚଳ int32_t *)(dst->op_params))[0];
    ଅଚଳ int32_t s1 = ((ଅଚଳ int32_t *)(dst->op_params))[1];
    ଅଚଳ int32_t s2 = ((ଅଚଳ int32_t *)(dst->op_params))[2];
    ଅଚଳ int32_t p0 = ((ଅଚଳ int32_t *)(dst->op_params))[3];
    ଅଚଳ int32_t p1 = ((ଅଚଳ int32_t *)(dst->op_params))[4];
    ଅଚଳ int32_t p2 = ((ଅଚଳ int32_t *)(dst->op_params))[5];
    ଅଚଳ int32_t d0 = ((ଅଚଳ int32_t *)(dst->op_params))[6];
    ଅଚଳ int32_t d1 = ((ଅଚଳ int32_t *)(dst->op_params))[7];
    ଅଚଳ int32_t d2 = ((ଅଚଳ int32_t *)(dst->op_params))[8];
    ଅଚଳ int32_t IC = ((ଅଚଳ int32_t *)(dst->op_params))[9];


    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ int64_t N  = ne13 / IC;
    ଅଚଳ int64_t ID = ne12;
    ଅଚଳ int64_t IH = ne11;
    ଅଚଳ int64_t IW = ne10;

    ଅଚଳ int64_t OC = ne03 / IC;
    GGML_UNUSED(OC);
    ଅଚଳ int64_t KD = ne02;
    ଅଚଳ int64_t KH = ne01;
    ଅଚଳ int64_t KW = ne00;

    ଅଚଳ int64_t OD = ne3 / N;
    ଅଚଳ int64_t OH = ne2;
    ଅଚଳ int64_t OW = ne1;
    ଅଚଳ int64_t OH_OW = OH*OW;
    ଅଚଳ int64_t KD_KH_KW = KD*KH*KW;
    ଅଚଳ int64_t KH_KW = KH*KW;
    ଅଚଳ int64_t IC_KD_KH_KW = IC*KD*KH*KW;

    GGML_ASSERT(nb10 == ଆକାର(ଦଶମିକ));

    // im2col: [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
    {
        ggml_fp16_t * ଅଚଳ wdata = (ggml_fp16_t *) dst->data;

        ପାଇଁ (int64_t in = 0; in < N; in++) {
            ପାଇଁ (int64_t iod = 0; iod < OD; iod++) {
                ପାଇଁ (int64_t ioh = 0; ioh < OH; ioh++) {
                    ପାଇଁ (int64_t iow = 0; iow < OW; iow++) {
                        ପାଇଁ (int64_t iic = ith; iic < IC; iic += nth) {

                            // micro kernel
                            ggml_fp16_t * dst_data = wdata + (in*OD*OH_OW + iod*OH_OW + ioh*OW + iow)*IC_KD_KH_KW; // [IC, KD, KH, KW]
                            ଅଚଳ ଦଶମିକ * ଅଚଳ src_data = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *)src1->data + (in*IC + iic)*nb13); // [ID, IH, IW]

                            ପାଇଁ (int64_t ikd = 0; ikd < KD; ikd++) {
                                ପାଇଁ (int64_t ikh = 0; ikh < KH; ikh++) {
                                    ପାଇଁ (int64_t ikw = 0; ikw < KW; ikw++) {
                                        ଅଚଳ int64_t iiw = iow*s0 + ikw*d0 - p0;
                                        ଅଚଳ int64_t iih = ioh*s1 + ikh*d1 - p1;
                                        ଅଚଳ int64_t iid = iod*s2 + ikd*d2 - p2;

                                        ଯଦି (iid < 0 || iid >= ID || iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                            dst_data[iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw] = 0;
                                        } ଅନ୍ୟଥା {
                                            ଅଚଳ ଦଶମିକ * ଅଚଳ s = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *)src_data + iid*nb12 + iih*nb11 + iiw*nb10); // [ID, IH, IW]
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
ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_im2col_3d_f32(
        ଅଚଳ ggml_compute_params * params,
              ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ଦ୍ୱିମିକ_OP_LOCALS;

    ଅଚଳ int32_t s0 = ((ଅଚଳ int32_t *)(dst->op_params))[0];
    ଅଚଳ int32_t s1 = ((ଅଚଳ int32_t *)(dst->op_params))[1];
    ଅଚଳ int32_t s2 = ((ଅଚଳ int32_t *)(dst->op_params))[2];
    ଅଚଳ int32_t p0 = ((ଅଚଳ int32_t *)(dst->op_params))[3];
    ଅଚଳ int32_t p1 = ((ଅଚଳ int32_t *)(dst->op_params))[4];
    ଅଚଳ int32_t p2 = ((ଅଚଳ int32_t *)(dst->op_params))[5];
    ଅଚଳ int32_t d0 = ((ଅଚଳ int32_t *)(dst->op_params))[6];
    ଅଚଳ int32_t d1 = ((ଅଚଳ int32_t *)(dst->op_params))[7];
    ଅଚଳ int32_t d2 = ((ଅଚଳ int32_t *)(dst->op_params))[8];
    ଅଚଳ int32_t IC = ((ଅଚଳ int32_t *)(dst->op_params))[9];


    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ int64_t N  = ne13 / IC;
    ଅଚଳ int64_t ID = ne12;
    ଅଚଳ int64_t IH = ne11;
    ଅଚଳ int64_t IW = ne10;

    ଅଚଳ int64_t OC = ne03 / IC;
    GGML_UNUSED(OC);
    ଅଚଳ int64_t KD = ne02;
    ଅଚଳ int64_t KH = ne01;
    ଅଚଳ int64_t KW = ne00;

    ଅଚଳ int64_t OD = ne3 / N;
    ଅଚଳ int64_t OH = ne2;
    ଅଚଳ int64_t OW = ne1;

    ଅଚଳ int64_t OH_OW = OH*OW;
    ଅଚଳ int64_t KD_KH_KW = KD*KH*KW;
    ଅଚଳ int64_t KH_KW = KH*KW;
    ଅଚଳ int64_t IC_KD_KH_KW = IC*KD*KH*KW;

    GGML_ASSERT(nb10 == ଆକାର(ଦଶମିକ));

    // im2col: [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
    {
        ଦଶମିକ * ଅଚଳ wdata = (ଦଶମିକ *) dst->data;

        ପାଇଁ (int64_t in = 0; in < N; in++) {
            ପାଇଁ (int64_t iod = 0; iod < OD; iod++) {
                ପାଇଁ (int64_t ioh = 0; ioh < OH; ioh++) {
                    ପାଇଁ (int64_t iow = 0; iow < OW; iow++) {
                        ପାଇଁ (int64_t iic = ith; iic < IC; iic += nth) {

                            // micro kernel
                            ଦଶମିକ * dst_data = wdata + (in*OD*OH_OW + iod*OH_OW + ioh*OW + iow)*IC_KD_KH_KW; // [IC, KD, KH, KW]
                            ଅଚଳ ଦଶମିକ * ଅଚଳ src_data = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *)src1->data + (in*IC + iic)*nb13); // [ID, IH, IW]

                            ପାଇଁ (int64_t ikd = 0; ikd < KD; ikd++) {
                                ପାଇଁ (int64_t ikh = 0; ikh < KH; ikh++) {
                                    ପାଇଁ (int64_t ikw = 0; ikw < KW; ikw++) {
                                        ଅଚଳ int64_t iiw = iow*s0 + ikw*d0 - p0;
                                        ଅଚଳ int64_t iih = ioh*s1 + ikh*d1 - p1;
                                        ଅଚଳ int64_t iid = iod*s2 + ikd*d2 - p2;

                                        ଯଦି (iid < 0 || iid >= ID || iih < 0 || iih >= IH || iiw < 0 || iiw >= IW || iid < 0 || iid >= ID) {
                                            dst_data[iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw] = 0;
                                        } ଅନ୍ୟଥା {
                                            ଅଚଳ ଦଶମିକ * ଅଚଳ s = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *)src_data + iid*nb12 + iih*nb11 + iiw*nb10); // [ID, IH, IW]
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


ଶୂନ୍ୟ ggml_compute_forward_im2col_3d(
        ଅଚଳ ggml_compute_params * params,
              ggml_tensor * dst) {
    ବାଛ (dst->type) {
        ସ୍ଥିତି GGML_TYPE_F16:
            {
                ggml_compute_forward_im2col_3d_f16(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_im2col_3d_f32(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_call_mul_mat(ggml_type type, ଅଚଳ ggml_compute_params * params, int64_t m, int64_t n, int64_t k,
                              ଶୂନ୍ୟ * a, ଶୂନ୍ୟ * b, ଦଶମିକ * c) {
    ଅଚଳ ggml_type_traits * traits = ggml_get_type_traits(type);
    ସଂରଚନା ggml_tensor src1 = {};
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

    ସଂରଚନା ggml_tensor src0 = {};
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

    ସଂରଚନା ggml_tensor dst = {};
    dst.ne[0] = n;
    dst.ne[1] = m;
    dst.ne[2] = 1;
    dst.ne[3] = 1;
    dst.nb[0] = ଆକାର(ଦଶମିକ);
    dst.nb[1] = n * ଆକାର(ଦଶମିକ);
    dst.nb[2] = dst.nb[1];
    dst.nb[3] = dst.nb[2];
    dst.data  = c;
    dst.src[0] = &src0;
    dst.src[1] = &src1;

    ggml_compute_forward_mul_mat(params, &dst);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ int64_t ggml_wrap_around(int64_t coord, int64_t size) {
    ଫେରନ୍ତୁ (coord  + size) % size; // adding size avoids negative number weirdness
}

// ggml_compute_forward_col2im_1d
//
// Scatter-add columns [K*OC, T_in] -> signal [T_out, OC]
// where T_out = (T_in - 1)*s + K - 2*p.  Gather approach: each output reads ceil(K/s) inputs.
// Parallelized over the time axis so the split stays balanced whatever OC is.
// Supports F32, F16, BF16 input/output (same type), F32 accumulator.

template <typename elem_t>
ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_col2im_1d_impl(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src = dst->src[0];  // [K*OC, T_in]

    GGML_ASSERT(ggml_is_contiguous(src));
    GGML_ASSERT(ggml_is_contiguous(dst));

    ଅଚଳ int32_t s0 = ((ଅଚଳ int32_t *)(dst->op_params))[0];
    ଅଚଳ int32_t OC = ((ଅଚଳ int32_t *)(dst->op_params))[1];
    ଅଚଳ int32_t p0 = ((ଅଚଳ int32_t *)(dst->op_params))[2];

    ଅଚଳ int64_t K_OC = src->ne[0];
    ଅଚଳ int64_t T_in = src->ne[1];
    ଅଚଳ int64_t K    = K_OC / OC;
    ଅଚଳ int64_t T_out = dst->ne[0];

    ଅଚଳ elem_t * col_data = (ଅଚଳ elem_t *) src->data;
    elem_t       * dst_data = (elem_t *) dst->data;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    // Parallelize over the time axis: the split stays balanced whatever OC is,
    // down to OC = 1 for mono audio, and threads read disjoint column bands
    ଅଚଳ int64_t dr = (T_out + nth - 1) / nth;
    ଅଚଳ int64_t it0 = dr * ith;
    ଅଚଳ int64_t it1 = it0 + dr < T_out ? it0 + dr : T_out;

    ପାଇଁ (int64_t oc = 0; oc < OC; oc++) {
        ପାଇଁ (int64_t t_out = it0; t_out < it1; t_out++) {
            ଅଚଳ int64_t t_abs = t_out + p0;  // absolute position in uncropped signal
            // Gather: find all (t_in, k) where t_in * s + k == t_abs, 0 <= k < K
            int64_t t_in_min = (t_abs - K + 1 + s0 - 1) / s0;  // ceil((t_abs-K+1)/s)
            ଯଦି (t_in_min < 0) t_in_min = 0;
            int64_t t_in_max = t_abs / s0;
            ଯଦି (t_in_max >= T_in) t_in_max = T_in - 1;

            ଦଶମିକ sum = 0.0f;
            ପାଇଁ (int64_t t_in = t_in_min; t_in <= t_in_max; t_in++) {
                int64_t k = t_abs - t_in * s0;
                ଯଦି (k >= 0 && k < K) {
                    // col layout: [K*OC, T_in], element (oc*K+k, t_in)
                    sum += type_conversion_table<elem_t>::to_f32(col_data[(oc * K + k) + t_in * K_OC]);
                }
            }
            // dst layout: [T_out, OC], element (t_out, oc)
            dst_data[t_out + oc * T_out] = type_conversion_table<elem_t>::from_f32(sum);
        }
    }
}

ଶୂନ୍ୟ ggml_compute_forward_col2im_1d(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {
    ବାଛ (dst->src[0]->type) {
        ସ୍ଥିତି GGML_TYPE_F32:  ggml_compute_forward_col2im_1d_impl<ଦଶମିକ>      (params, dst); ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F16:  ggml_compute_forward_col2im_1d_impl<ggml_fp16_t>(params, dst); ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_BF16: ggml_compute_forward_col2im_1d_impl<ggml_bf16_t>(params, dst); ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ: GGML_ABORT("col2im_1d: unsupported type %d", dst->src[0]->type);
    }
}

// ggml_compute_forward_conv_2d


ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_conv_2d_impl(ଅଚଳ ggml_compute_params * params,
                                              ଅଚଳ ggml_tensor *         kernel,  // [KW, KH, IC, OC]
                                              ଅଚଳ ggml_tensor *         src,     // [W, H, C, N]
                                              ggml_tensor *               dst,     // [OW, OH, OC, N]
                                              ggml_type                   kernel_type) {

    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(kernel_type == GGML_TYPE_F16 || kernel_type == GGML_TYPE_F32);
    GGML_ASSERT(kernel->type == kernel_type);

    ଅଚଳ ggml_type_traits * traits = ggml_get_type_traits(kernel_type);

    ଅଚଳ int32_t stride_x   = dst->op_params[0];
    ଅଚଳ int32_t stride_y   = dst->op_params[1];
    ଅଚଳ int32_t pad_x      = dst->op_params[2];
    ଅଚଳ int32_t pad_y      = dst->op_params[3];
    ଅଚଳ int32_t dilation_x = dst->op_params[4];
    ଅଚଳ int32_t dilation_y = dst->op_params[5];

    ଅଚଳ int64_t c_in  = src->ne[2];
    ଅଚଳ int64_t c_out = kernel->ne[3];
    GGML_ASSERT(c_in == kernel->ne[2]);

    ଅଚଳ int64_t src_w = src->ne[0];
    ଅଚଳ int64_t src_h = src->ne[1];
    ଅଚଳ int64_t knl_w = kernel->ne[0];
    ଅଚଳ int64_t knl_h = kernel->ne[1];
    ଅଚଳ int64_t dst_w = dst->ne[0];
    ଅଚଳ int64_t dst_h = dst->ne[1];

    ଅଚଳ ଦଶମିକ * src_data = (ଦଶମିକ *) src->data;
    ଶୂନ୍ୟ  * knl_data       = kernel->data;
    ଦଶମିକ * dst_data       = (ଦଶମିକ *) dst->data;

    ଅଚଳ int64_t knl_n           = knl_w * knl_h * c_in;
    ଅଚଳ int64_t patch_total     = dst->ne[3] * dst_w * dst_h;

    ଅଚଳ int64_t space_per_patch   = knl_n * traits->type_size + c_out * ଆକାର(ଦଶମିକ);
    ଅଚଳ int64_t batch_size        = params->wsize / space_per_patch;
    ଅଚଳ int64_t patches_per_batch = batch_size > 8 ? (batch_size / 8) * 8 : batch_size;
    ଅଚଳ int64_t batch_n           = (patch_total + patches_per_batch - 1) / patches_per_batch;

    GGML_ASSERT(patches_per_batch > 0 && batch_size >= 1);

    ଶୂନ୍ୟ * tmp = params->wdata;

    ପାଇଁ (int64_t batch_i = 0; batch_i < batch_n; ++batch_i) {

        ଅଚଳ int64_t patch_start_batch = batch_i * patches_per_batch;
        ଅଚଳ int64_t patch_end_batch   = std::min(patch_start_batch + patches_per_batch,
                                              patch_total);
        ଅଚଳ int64_t patch_n           = patch_end_batch - patch_start_batch;

        ଅଚଳ int64_t patch_per_thread  = (patch_n + params->nth - 1) / params->nth;
        ଅଚଳ int64_t patch_start       = patch_start_batch + params->ith * patch_per_thread;
        ଅଚଳ int64_t patch_end         = std::min(patch_start + patch_per_thread, patch_end_batch);

        //im2col for a patch
        ପାଇଁ (int64_t p = patch_start; p < patch_end; ++p) {
            ଅଚଳ int64_t  batch_n     =  p / (dst_w * dst_h);
            ଅଚଳ int64_t  src_x       = (p / dst_w) % dst_h;
            ଅଚଳ int64_t  src_y       =  p % dst_w;

            ଅଚଳ ଦଶମିକ * src_base = (ଅଚଳ ଦଶମିକ *)((ଅଚଳ ଅକ୍ଷର *)src_data + batch_n * src->nb[3]);
            ଅକ୍ଷର *        dst_row  = (ଅକ୍ଷର *) tmp + (p % patches_per_batch) * knl_n * traits->type_size;

            ପାଇଁ (int64_t ic = 0; ic < c_in; ++ic) {
                ପାଇଁ (int64_t ky = 0; ky < knl_h; ++ky) {
                    ପାଇଁ (int64_t kx = 0; kx < knl_w; ++kx) {
                        ଅଚଳ int64_t sy = src_x * stride_y + ky * dilation_y - pad_y;
                        ଅଚଳ int64_t sx = src_y * stride_x + kx * dilation_x - pad_x;

                        int64_t dst_idx = ic * (knl_h * knl_w) + ky * knl_w + kx;

                        ଦଶମିକ src_val;
                        ଯଦି (sy < 0 || sy >= src_h || sx < 0 || sx >= src_w) {
                            src_val = 0.0f;
                        } ଅନ୍ୟଥା {
                            ଅଚଳ ଦଶମିକ * src_ptr = (ଅଚଳ ଦଶମିକ *)((ଅଚଳ ଅକ୍ଷର *)src_base + sx * src->nb[0] + sy * src->nb[1] + ic * src->nb[2]);
                            src_val               = *src_ptr;
                        }

                        ଅକ୍ଷର * element_ptr = dst_row + dst_idx * traits->type_size;
                        ଯଦି (kernel_type == GGML_TYPE_F32) {
                            *(ଦଶମିକ *) element_ptr = src_val;
                        } ଅନ୍ୟଥା ଯଦି (kernel_type == GGML_TYPE_F16) {
                            *(ggml_fp16_t *) element_ptr = GGML_CPU_FP32_TO_FP16(src_val);
                        }
                    }
                }
            }
        }   // patches handled by this thread

        ggml_barrier(params->threadpool);

        ଦଶମିକ * gemm_output = (ଦଶମିକ *) ((ଅକ୍ଷର *) tmp + patches_per_batch * knl_n * traits->type_size);

        GGML_ASSERT(gemm_output + patch_n * c_out <= (ଦଶମିକ*)tmp + params->wsize);

        // GEMM: patches[patch_n, knl_n] × kernel[knl_n, c_out] = output[patch_n, c_out]
        ggml_call_mul_mat(kernel_type, params, patch_n, c_out, knl_n, tmp, knl_data, gemm_output);

        ggml_barrier(params->threadpool);


        //permute back [OC, N, OH, OW] to [N, OC, OH, OW]
        ଅଚଳ int64_t permute_per_thread = (patch_n + params->nth - 1) / params->nth;
        ଅଚଳ int64_t permute_start = params->ith * permute_per_thread;
        ଅଚଳ int64_t permute_end = std::min(permute_start + permute_per_thread, patch_n);

        ପାଇଁ (int64_t i = permute_start; i < permute_end; ++i) {
            ଅଚଳ int64_t p       = patch_start_batch + i;
            ଅଚଳ int64_t batch_n = p / (dst_w * dst_h);
            ଅଚଳ int64_t dst_y   = (p / dst_w) % dst_h;
            ଅଚଳ int64_t dst_x   = p % dst_w;

            ପାଇଁ (int64_t oc = 0; oc < c_out; ++oc) {
                ଅଚଳ ଦଶମିକ value = gemm_output[i * c_out + oc];
                ଦଶମିକ * dst_ptr = (ଦଶମିକ *)((ଅକ୍ଷର *)dst_data + dst_x * dst->nb[0] + dst_y * dst->nb[1] + oc * dst->nb[2] + batch_n * dst->nb[3]);
                *dst_ptr = value;
            }
        }
    }
}

ଶୂନ୍ୟ ggml_compute_forward_conv_2d(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];

    ggml_compute_forward_conv_2d_impl(params, src0, src1, dst, src0->type);
}

// ggml_compute_forward_conv_3d

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_conv_3d_impl(ଅଚଳ ggml_compute_params * params,
                                              ଅଚଳ ggml_tensor *         kernel,
                                              ଅଚଳ ggml_tensor *         src,
                                              ggml_tensor *               dst,
                                              ggml_type                   kernel_type) {

    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(kernel_type == GGML_TYPE_F16 || kernel_type == GGML_TYPE_F32);
    GGML_ASSERT(kernel->type == kernel_type);

    ଅଚଳ ggml_type_traits * traits = ggml_get_type_traits(kernel_type);

    ଅଚଳ int32_t s0 = dst->op_params[0];
    ଅଚଳ int32_t s1 = dst->op_params[1];
    ଅଚଳ int32_t s2 = dst->op_params[2];
    ଅଚଳ int32_t p0 = dst->op_params[3];
    ଅଚଳ int32_t p1 = dst->op_params[4];
    ଅଚଳ int32_t p2 = dst->op_params[5];
    ଅଚଳ int32_t d0 = dst->op_params[6];
    ଅଚଳ int32_t d1 = dst->op_params[7];
    ଅଚଳ int32_t d2 = dst->op_params[8];
    ଅଚଳ int32_t c  = dst->op_params[9];
    ଅଚଳ int32_t n  = dst->op_params[10];
    ଅଚଳ int32_t oc = dst->op_params[11];

    ଅଚଳ int64_t src_w = src->ne[0];
    ଅଚଳ int64_t src_h = src->ne[1];
    ଅଚଳ int64_t src_d = src->ne[2];
    ଅଚଳ int64_t knl_w = kernel->ne[0];
    ଅଚଳ int64_t knl_h = kernel->ne[1];
    ଅଚଳ int64_t knl_d = kernel->ne[2];
    ଅଚଳ int64_t dst_w = dst->ne[0];
    ଅଚଳ int64_t dst_h = dst->ne[1];
    ଅଚଳ int64_t dst_d = dst->ne[2];

    ଅଚଳ ଦଶମିକ * src_data = (ଦଶମିକ *) src->data;
    ଶୂନ୍ୟ  * knl_data       = kernel->data;
    ଦଶମିକ * dst_data       = (ଦଶମିକ *) dst->data;

    ଅଚଳ int64_t knl_n_per_channel = knl_w * knl_h * knl_d;
    ଅଚଳ int64_t knl_n_total       = knl_n_per_channel * c;
    ଅଚଳ int64_t patch_total       = n * dst_w * dst_h * dst_d;

    ଅଚଳ int64_t space_per_patch   = knl_n_total * traits->type_size + oc * ଆକାର(ଦଶମିକ);
    ଅଚଳ int64_t batch_size        = params->wsize / space_per_patch;
    ଅଚଳ int64_t patches_per_batch = batch_size > 8 ? (batch_size / 8) * 8 : batch_size;
    ଅଚଳ int64_t batch_n           = (patch_total + patches_per_batch - 1) / patches_per_batch;

    GGML_ASSERT(patches_per_batch > 0 && batch_size >= 1);

    ଶୂନ୍ୟ * tmp = params->wdata;

    ପାଇଁ (int64_t batch_i = 0; batch_i < batch_n; ++batch_i) {
        ଅଚଳ int64_t patch_start_batch = batch_i * patches_per_batch;
        ଅଚଳ int64_t patch_end_batch   = std::min(patch_start_batch + patches_per_batch, patch_total);
        ଅଚଳ int64_t patch_n_in_batch  = patch_end_batch - patch_start_batch;

        ଅଚଳ int64_t patch_per_thread  = (patch_n_in_batch + params->nth - 1) / params->nth;
        ଅଚଳ int64_t patch_start       = patch_start_batch + params->ith * patch_per_thread;
        ଅଚଳ int64_t patch_end         = std::min(patch_start + patch_per_thread, patch_end_batch);

        ପାଇଁ (int64_t p = patch_start; p < patch_end; ++p) {
            ଅଚଳ int64_t p_in_batch = p % (dst_w * dst_h * dst_d);
            ଅଚଳ int64_t p_in_depth = p_in_batch % (dst_w * dst_h);
            ଅଚଳ int64_t batch_idx  = p / (dst_w * dst_h * dst_d);
            ଅଚଳ int64_t dst_z      = p_in_batch / (dst_w * dst_h);
            ଅଚଳ int64_t dst_y      = p_in_depth / dst_w;
            ଅଚଳ int64_t dst_x      = p_in_depth % dst_w;

            ଅକ୍ଷର * dst_row = (ଅକ୍ଷର *) tmp + (p % patches_per_batch) * knl_n_total * traits->type_size;

            ପାଇଁ (int64_t ic = 0; ic < c; ++ic) {
                ପାଇଁ (int64_t kz = 0; kz < knl_d; ++kz) {
                    ପାଇଁ (int64_t ky = 0; ky < knl_h; ++ky) {
                        ପାଇଁ (int64_t kx = 0; kx < knl_w; ++kx) {
                            ଅଚଳ int64_t sz = dst_z * s2 + kz * d2 - p2;
                            ଅଚଳ int64_t sy = dst_y * s1 + ky * d1 - p1;
                            ଅଚଳ int64_t sx = dst_x * s0 + kx * d0 - p0;

                            int64_t dst_idx = ic * knl_n_per_channel + kz * (knl_h * knl_w) + ky * knl_w + kx;

                            ଦଶମିକ src_val;
                            ଯଦି (sz < 0 || sz >= src_d || sy < 0 || sy >= src_h || sx < 0 || sx >= src_w) {
                                src_val = 0.0f;
                            } ଅନ୍ୟଥା {
                                ଅଚଳ int64_t cn_idx = batch_idx * c + ic;
                                ଅଚଳ ଦଶମିକ * src_ptr = (ଅଚଳ ଦଶମିକ *)((ଅଚଳ ଅକ୍ଷର *)src_data + sx*src->nb[0] + sy*src->nb[1] + sz*src->nb[2] + cn_idx*src->nb[3]);
                                src_val = *src_ptr;
                            }

                            ଅକ୍ଷର * element_ptr = dst_row + dst_idx * traits->type_size;
                            ଯଦି (kernel_type == GGML_TYPE_F32) {
                                *(ଦଶମିକ *)element_ptr = src_val;
                            } ଅନ୍ୟଥା ଯଦି (kernel_type == GGML_TYPE_F16) {
                                *(ggml_fp16_t *)element_ptr = GGML_CPU_FP32_TO_FP16(src_val);
                            }
                        }
                    }
                }
            }
        }

        ggml_barrier(params->threadpool);

        ଦଶମିକ * gemm_output = (ଦଶମିକ *) ((ଅକ୍ଷର *) tmp + patches_per_batch * knl_n_total * traits->type_size);
        ggml_call_mul_mat(kernel_type, params, patch_n_in_batch, oc, knl_n_total, tmp, knl_data, gemm_output);

        ggml_barrier(params->threadpool);

        ଅଚଳ int64_t permute_per_thread = (patch_n_in_batch + params->nth - 1) / params->nth;
        ଅଚଳ int64_t permute_start = params->ith * permute_per_thread;
        ଅଚଳ int64_t permute_end = std::min(permute_start + permute_per_thread, patch_n_in_batch);

        ପାଇଁ (int64_t i = permute_start; i < permute_end; ++i) {
            ଅଚଳ int64_t p = patch_start_batch + i;
            ଅଚଳ int64_t p_in_batch = p % (dst_w * dst_h * dst_d);
            ଅଚଳ int64_t p_in_depth = p_in_batch % (dst_w * dst_h);
            ଅଚଳ int64_t batch_idx  = p / (dst_w * dst_h * dst_d);
            ଅଚଳ int64_t dst_z      = p_in_batch / (dst_w * dst_h);
            ଅଚଳ int64_t dst_y      = p_in_depth / dst_w;
            ଅଚଳ int64_t dst_x      = p_in_depth % dst_w;

            ପାଇଁ (int64_t ioc = 0; ioc < oc; ++ioc) {
                ଅଚଳ ଦଶମିକ value = gemm_output[i * oc + ioc];
                ଅଚଳ int64_t ocn_idx = batch_idx * oc + ioc;
                ଦଶମିକ * dst_ptr = (ଦଶମିକ *)((ଅକ୍ଷର *)dst_data + dst_x*dst->nb[0] + dst_y*dst->nb[1] + dst_z*dst->nb[2] + ocn_idx*dst->nb[3]);
                *dst_ptr = value;
            }
        }
    }
}

ଶୂନ୍ୟ ggml_compute_forward_conv_3d(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {
    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];
    ggml_compute_forward_conv_3d_impl(params, src0, src1, dst, src0->type);
}

template <typename kernel_t>
ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_conv_transpose_2d_impl(
    ଅଚଳ ggml_compute_params * params,
          ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ଦ୍ୱିମିକ_OP_LOCALS

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nk = ne00*ne01*ne02*ne03;

    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == ଆକାର(ଦଶମିକ));

    ଯଦି (ith == 0) {
        memset(params->wdata, 0, params->wsize);

        // permute kernel data (src0) from (Kw x Kh x Cout x Cin) to (Cin x Kw x Kh x Cout)
        {
            kernel_t * ଅଚଳ wdata = (kernel_t *) params->wdata + 0;

            ପାଇଁ (int64_t i03 = 0; i03 < ne03; i03++) {
                ପାଇଁ (int64_t i02 = 0; i02 < ne02; i02++) {
                    ଅଚଳ kernel_t * ଅଚଳ src = (kernel_t *)((ଅକ୍ଷର *) src0->data + i03*nb03 + i02*nb02);
                    kernel_t * dst_data = wdata + i02*ne01*ne00*ne03;
                    ପାଇଁ (int64_t i01 = 0; i01 < ne01; i01++) {
                        ପାଇଁ (int64_t i00 = 0; i00 < ne00; i00++) {
                            dst_data[i01*ne00*ne03 + i00*ne03 + i03] = src[i01 * ne00 + i00];
                        }
                    }
                }
            }
        }

        // permute source data (src1) from (Sw x Sh x Cin) to (Cin x Sw x Sh), for all batches
        {
            kernel_t * ଅଚଳ wdata = (kernel_t *) params->wdata + nk;
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i13 = 0; i13 < ne13; i13++) {
                kernel_t * ଅଚଳ wdata_b = wdata + i13*ne10*ne11*ne12;
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i12 = 0; i12 < ne12; i12++) {
                    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i11 = 0; i11 < ne11; i11++) {
                        ଅଚଳ ଦଶମିକ * ଅଚଳ src = (ଦଶମିକ *)((ଅକ୍ଷର *) src1->data + i13*nb13 + i12*nb12 + i11*nb11);
                        kernel_t * dst_data = wdata_b + i11*ne10*ne12;
                        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i10 = 0; i10 < ne10; i10++) {
                            ଯଦି constexpr (std::is_same_v<kernel_t, ggml_fp16_t>) {
                                dst_data[i10*ne12 + i12] = GGML_CPU_FP32_TO_FP16(src[i10]);
                            } ଅନ୍ୟଥା {
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

    ଅଚଳ int32_t stride = ggml_get_op_params_i32(dst, 0);

    // total patches in dst
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା np = ne2;

    // patches per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dp = (np + nth - 1)/nth;

    // patch range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ip0 = dp*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ip1 = MIN(ip0 + dp, np);

    kernel_t * ଅଚଳ wdata = (kernel_t *) params->wdata + 0;
    kernel_t * ଅଚଳ wdata_src = wdata + nk;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i2 = ip0; i2 < ip1; i2++) { // Cout
        kernel_t * wdata_kernel = wdata + i2*ne01*ne00*ne03;
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i3 = 0; i3 < ne3; i3++) { // batch
            ଦଶମିକ * dst_data = (ଦଶମିକ *)((ଅକ୍ଷର *) dst->data + i3*nb3 + i2*nb2);
            kernel_t * wdata_src_b = wdata_src + i3*ne10*ne11*ne12;
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i11 = 0; i11 < ne11; i11++) {
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i10 = 0; i10 < ne10; i10++) {
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1n = i11*ne10*ne12 + i10*ne12;
                    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i01 = 0; i01 < ne01; i01++) {
                        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i00 = 0; i00 < ne00; i00++) {
                            ଦଶମିକ v = 0;
                            ଯଦି constexpr (std::is_same_v<kernel_t, ggml_fp16_t>) {
                                ggml_vec_dot_f16(ne03, &v, 0,
                                        wdata_src_b + i1n, 0,
                                        wdata_kernel + i01*ne00*ne03 + i00*ne03, 0, 1);
                            } ଅନ୍ୟଥା {
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

ଶୂନ୍ୟ ggml_compute_forward_conv_transpose_2d(
        ଅଚଳ ggml_compute_params * params,
              ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F16:
            {
                ggml_compute_forward_conv_transpose_2d_impl<ggml_fp16_t>(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_conv_transpose_2d_impl<ଦଶମିକ>(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_conv_2d_dw

ସଂରଚନା ggml_conv_2d_dw_params {
    int64_t channels;
    int64_t batch;
    int64_t src_w;
    int64_t src_h;
    int64_t dst_w;
    int64_t dst_h;
    int64_t knl_w;
    int64_t knl_h;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_x;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_y;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା pad_x;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା pad_y;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା dilation_x;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା dilation_y;
};

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ ggml_conv_2d_dw_knl_f32(ଅଚଳ ଅକ୍ଷର * data, int64_t i, ggml_type type) {
    ଯଦି (type == GGML_TYPE_F16) {
        ଫେରନ୍ତୁ GGML_FP16_TO_FP32(((ଅଚଳ ggml_fp16_t *)data)[i]);
    }
    ଫେରନ୍ତୁ ((ଅଚଳ ଦଶମିକ *)data)[i];
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_conv_2d_dw_cwhn(
        ଅଚଳ ggml_compute_params * params,
        ଅଚଳ ggml_tensor * src,
        ଅଚଳ ggml_tensor * kernel,
        ggml_tensor * dst,
        ଅଚଳ ggml_conv_2d_dw_params & p) {

    ଅଚଳ int64_t c = p.channels;
    ଅଚଳ ଅକ୍ଷର * knl_data = (ଅଚଳ ଅକ୍ଷର *)kernel->data;
    ଅଚଳ ggml_type knl_type = kernel->type;

    ଅଚଳ int64_t rows_total = p.dst_h * p.batch;
    ଅଚଳ int64_t rows_per_thread = (rows_total + params->nth - 1) / params->nth;
    ଅଚଳ int64_t row_start = params->ith * rows_per_thread;
    ଅଚଳ int64_t row_end = MIN(row_start + rows_per_thread, rows_total);

#ifdef GGML_SIMD
    int64_t c_pkg_end = 0;
    int64_t pkg_size = GGML_F32_EPR;
    ଯଦି (knl_type == GGML_TYPE_F32) {
    #ଯଦି defined(__ARM_FEATURE_SVE)
        pkg_size = svcntw();
    #ଅନ୍ୟଥା
        pkg_size = GGML_F32_EPR;
    #endif
        c_pkg_end = (c / pkg_size) * pkg_size;
    }
#ଅନ୍ୟଥା
    ଅଚଳ int64_t c_pkg_end = 0;
#endif

    ପାଇଁ (int64_t row = row_start; row < row_end; ++row) {
        ଅଚଳ int64_t dst_y = row % p.dst_h;
        ଅଚଳ ଦଶମିକ * src_data = (ଅଚଳ ଦଶମିକ *)src->data + (row / p.dst_h) * p.src_w * p.src_h * c;
        ପାଇଁ (int64_t dst_x = 0; dst_x < p.dst_w; ++dst_x) {
            ଦଶମିକ * dst_data = (ଦଶମିକ *)dst->data + (row * p.dst_w + dst_x) * c;
            ଅଚଳ int64_t src_y_base = dst_y * p.stride_y - p.pad_y;
            ଅଚଳ int64_t src_x_base = dst_x * p.stride_x - p.pad_x;

#ifdef GGML_SIMD
            ପାଇଁ (int64_t c_i = 0; c_i < c_pkg_end; c_i += pkg_size) {
                GGML_F32_VEC sum = GGML_F32_VEC_ZERO;
                ପାଇଁ (int64_t knl_y = 0; knl_y < p.knl_h; ++knl_y) {
                    ଅଚଳ int64_t src_y = src_y_base + knl_y * p.dilation_y;
                    ଯଦି (src_y < 0 || src_y >= p.src_h) {
                        ଜାରି;
                    }
                    ପାଇଁ (int64_t knl_x = 0; knl_x < p.knl_w; ++knl_x) {
                        ଅଚଳ int64_t src_x = src_x_base + knl_x * p.dilation_x;
                        ଯଦି (src_x < 0 || src_x >= p.src_w) {
                            ଜାରି;
                        }
                        ଅଚଳ ଦଶମିକ * kp = (ଅଚଳ ଦଶମିକ *)knl_data + (knl_y * p.knl_w + knl_x) * c + c_i;
                        GGML_F32_VEC k = GGML_F32_VEC_LOAD(kp);
                        GGML_F32_VEC s = GGML_F32_VEC_LOAD(src_data + (src_y * p.src_w + src_x) * c + c_i);
                        sum = GGML_F32_VEC_FMA(sum, k, s);
                    }
                }
                GGML_F32_VEC_STORE(dst_data + c_i, sum);
            }
#endif
            ପାଇଁ (int64_t c_i = c_pkg_end; c_i < c; ++c_i) {
                ଦଶମିକ sum = 0.0f;
                ପାଇଁ (int64_t knl_y = 0; knl_y < p.knl_h; ++knl_y) {
                    ଅଚଳ int64_t src_y = src_y_base + knl_y * p.dilation_y;
                    ଯଦି (src_y < 0 || src_y >= p.src_h) {
                        ଜାରି;
                    }
                    ପାଇଁ (int64_t knl_x = 0; knl_x < p.knl_w; ++knl_x) {
                        ଅଚଳ int64_t src_x = src_x_base + knl_x * p.dilation_x;
                        ଯଦି (src_x < 0 || src_x >= p.src_w) {
                            ଜାରି;
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

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_conv_2d_dw_whcn(
        ଅଚଳ ggml_compute_params * params,
        ଅଚଳ ggml_tensor * src,
        ଅଚଳ ggml_tensor * kernel,
        ggml_tensor * dst,
        ଅଚଳ ggml_conv_2d_dw_params & p) {

    ଅଚଳ int64_t n = p.channels * p.batch;
    ଅଚଳ int64_t per_thread = (n + params->nth - 1) / params->nth;
    ଅଚଳ int64_t start = params->ith * per_thread;
    ଅଚଳ int64_t end = MIN(start + per_thread, n);
    ଅଚଳ ଅକ୍ଷର * knl_base = (ଅଚଳ ଅକ୍ଷର *)kernel->data;
    ଅଚଳ ggml_type knl_type = kernel->type;

    ପାଇଁ (int64_t i = start; i < end; ++i) {
        ଅଚଳ int64_t knl_offset = (i % p.channels) * p.knl_w * p.knl_h;
        ଅଚଳ ଦଶମିକ * src_data = (ଅଚଳ ଦଶମିକ *)src->data + i * p.src_w * p.src_h;
        ଦଶମିକ * dst_data = (ଦଶମିକ *)dst->data + i * p.dst_w * p.dst_h;

        ପାଇଁ (int64_t dst_y = 0; dst_y < p.dst_h; ++dst_y) {
            ପାଇଁ (int64_t dst_x = 0; dst_x < p.dst_w; ++dst_x) {

                ଦଶମିକ sum = 0.0f;
                ପାଇଁ (int64_t knl_y = 0; knl_y < p.knl_h; ++knl_y) {
                    ଅଚଳ int64_t src_y = dst_y * p.stride_y + knl_y * p.dilation_y - p.pad_y;
                    ଯଦି (src_y < 0 || src_y >= p.src_h) {
                        ଜାରି;
                    }
                    ପାଇଁ (int64_t knl_x = 0; knl_x < p.knl_w; ++knl_x) {
                        ଅଚଳ int64_t src_x = dst_x * p.stride_x + knl_x * p.dilation_x - p.pad_x;
                        ଯଦି (src_x < 0 || src_x >= p.src_w) {
                            ଜାରି;
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

ଶୂନ୍ୟ ggml_compute_forward_conv_2d_dw(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * kernel = dst->src[0];
    ଅଚଳ ggml_tensor * src = dst->src[1];
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

    ଯଦି (ggml_is_contiguous(src)) {
        ggml_compute_forward_conv_2d_dw_whcn(params, src, kernel, dst, p);
    } ଅନ୍ୟଥା ଯଦି (ggml_is_contiguous_channels(src)) {
        GGML_ASSERT(kernel->nb[0] >= kernel->nb[2] && kernel->nb[1] >= kernel->nb[0]);
        ggml_compute_forward_conv_2d_dw_cwhn(params, src, kernel, dst, p);
    } ଅନ୍ୟଥା {
        GGML_ABORT("non-contiguous memory layout not supported");
    }
}

// ggml_compute_forward_pool_1d_ksp
ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_pool_1d_ksp(
        ଅଚଳ ggml_compute_params * params,
        ଅଚଳ ggml_op_pool op,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା k,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା p,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src = dst->src[0];

    ପ୍ରତିଜ୍ଞା(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16);

    ଯଦି (params->ith != 0) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ int64_t IW = src->ne[0];
    ଅଚଳ int64_t OW = dst->ne[0];

    ଅଚଳ int64_t nr = ggml_nrows(src);

    ପାଇଁ (int64_t ir = 0; ir < nr; ++ir) {
        ଅଚଳ ଅକ୍ଷର * srow_bytes =            (ଅଚଳ ଅକ୍ଷର *) src->data + ir * src->nb[1];
        ଦଶମିକ      * drow       = (ଦଶମିକ *) ((      ଅକ୍ଷର *) dst->data + ir * dst->nb[1]);

        ପାଇଁ (int64_t ow = 0; ow < OW; ++ow) {
            ଦଶମିକ res = 0;
            ବାଛ (op) {
                ସ୍ଥିତି GGML_OP_POOL_AVG: res = 0.0f;     ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_OP_POOL_MAX: res = -FLT_MAX; ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_OP_POOL_ଗଣନା: GGML_ABORT("fatal error");
            }

            ପୂର୍ଣ୍ଣସଂଖ୍ୟା count = 0;
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା base = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ow * s - p;

            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ki = 0; ki < k; ++ki) {
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = base + ki;
                ଯଦି (j < 0 || j >= (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) IW) {
                    ଜାରି;
                }

                ଦଶମିକ v;
                ଯଦି (src->type == GGML_TYPE_F32) {
                    v = ((ଅଚଳ ଦଶମିକ *) srow_bytes)[j];
                } ଅନ୍ୟଥା {
                    v = GGML_CPU_FP16_TO_FP32(((ଅଚଳ ggml_fp16_t *) srow_bytes)[j]);
                }

                ବାଛ (op) {
                    ସ୍ଥିତି GGML_OP_POOL_AVG: res += v;                ଭାଙ୍ଗ;
                    ସ୍ଥିତି GGML_OP_POOL_MAX: res =  std::max(v, res); ଭାଙ୍ଗ;
                    ସ୍ଥିତି GGML_OP_POOL_ଗଣନା: GGML_ABORT("fatal error");
                }

                ++count;
            }

            ବାଛ (op) {
                ସ୍ଥିତି GGML_OP_POOL_AVG: res = (count > 0) ? (res / count) : 0.0f; ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_OP_POOL_MAX:                                           ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_OP_POOL_ଗଣନା: GGML_ABORT("fatal error");
            }

            drow[ow] = res;
        }
    }
}

// ggml_compute_forward_pool_1d

ଶୂନ୍ୟ ggml_compute_forward_pool_1d(
        ଅଚଳ ggml_compute_params * params,
              ggml_tensor * dst) {

    ଅଚଳ int32_t * opts = (ଅଚଳ int32_t *)dst->op_params;
    ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା k0 = opts[1];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s0 = opts[2];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା p0 = opts[3];

    ggml_compute_forward_pool_1d_ksp(params, op, k0, s0, p0, dst);
}

// ggml_compute_forward_pool_2d

ଶୂନ୍ୟ ggml_compute_forward_pool_2d(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src = dst->src[0];

    ପ୍ରତିଜ୍ଞା(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16);

    ଯଦି (params->ith != 0) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ int32_t * opts = (ଅଚଳ int32_t *)dst->op_params;

    ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା k0 = opts[1];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା k1 = opts[2];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s0 = opts[3];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s1 = opts[4];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା p0 = opts[5];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା p1 = opts[6];
    ଅଚଳ ଅକ୍ଷର * cdata = (ଅଚଳ ଅକ୍ଷର*)src->data;
    ଅଚଳ ଅକ୍ଷର * ଅଚଳ data_end = cdata + ggml_nbytes(src);

    ଅଚଳ int64_t px = dst->ne[0];
    ଅଚଳ int64_t py = dst->ne[1];
    ଅଚଳ int64_t pa = px * py;

    ଦଶମିକ * dplane = (ଦଶମିକ *)dst->data;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ka = k0 * k1;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା offset0 = -p0;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା offset1 = -p1;

    ଯେତେବେଳେ (cdata < data_end) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା oy = 0; oy < py; ++oy) {
            ଦଶମିକ * ଅଚଳ drow = dplane + oy * px;
            ଦଶମିକ * ଅଚଳ out  = drow;

            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ox = 0; ox < px; ++ox) {
                ଦଶମିକ res = 0;
                ବାଛ (op) {
                    ସ୍ଥିତି GGML_OP_POOL_AVG: res = 0;        ଭାଙ୍ଗ;
                    ସ୍ଥିତି GGML_OP_POOL_MAX: res = -FLT_MAX; ଭାଙ୍ଗ;
                    ସ୍ଥିତି GGML_OP_POOL_ଗଣନା: GGML_ABORT("fatal error");
                }

                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ix = offset0 + ox * s0;
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iy = offset1 + oy * s1;

                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ky = 0; ky < k1; ++ky) {
                    ଯଦି (iy + ky < 0 || iy + ky >= src->ne[1]) {
                        ଜାରି;
                    }

                    ଅଚଳ ଶୂନ୍ୟ * srow = (ଅଚଳ ଶୂନ୍ୟ *)(cdata + src->nb[1] * (iy + ky));
                    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା kx = 0; kx < k0; ++kx) {
                        ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = ix + kx;
                        ଯଦି (j < 0 || j >= src->ne[0]) {
                            ଜାରି;
                        }

                        ଅଚଳ ଦଶମିକ srow_j = (src->type == GGML_TYPE_F32) ? ((ଅଚଳ ଦଶମିକ*)srow)[j] : GGML_CPU_FP16_TO_FP32(((ଅଚଳ ggml_fp16_t*)srow)[j]);
                        ବାଛ (op) {
                            ସ୍ଥିତି GGML_OP_POOL_AVG: res += srow_j;                ଭାଙ୍ଗ;
                            ସ୍ଥିତି GGML_OP_POOL_MAX: res =  std::max(srow_j, res); ଭାଙ୍ଗ;
                            ସ୍ଥିତି GGML_OP_POOL_ଗଣନା:               GGML_ABORT("fatal error");
                        }
                    }
                }
                ବାଛ (op) {
                    ସ୍ଥିତି GGML_OP_POOL_AVG:           res /= ka; ଭାଙ୍ଗ;
                    ସ୍ଥିତି GGML_OP_POOL_MAX:                      ଭାଙ୍ଗ;
                    ସ୍ଥିତି GGML_OP_POOL_ଗଣନା: GGML_ABORT("fatal error");
                }

                out[ox] = res;
            }
        }

        cdata  += src->nb[2];
        dplane += pa;
    }
}

// ggml_compute_forward_pool_2d_back

ଶୂନ୍ୟ ggml_compute_forward_pool_2d_back(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src  = dst->src[0];
    ଅଚଳ ggml_tensor * dstf = dst->src[1]; // forward tensor of dst

    ପ୍ରତିଜ୍ଞା(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);

    ଯଦି (params->ith != 0) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ int32_t * opts = (ଅଚଳ int32_t *)dst->op_params;
    ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା k0 = opts[1];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା k1 = opts[2];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s0 = opts[3];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s1 = opts[4];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା p0 = opts[5];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା p1 = opts[6];

    ଅକ୍ଷର       * cdata  = (ଅକ୍ଷର       *) dst->data;
    ଅଚଳ ଅକ୍ଷର * cdataf = (ଅଚଳ ଅକ୍ଷର *) dstf->data;
    ଅଚଳ ଅକ୍ଷର * ଅଚଳ data_end = cdata + ggml_nbytes(dst);

    GGML_ASSERT(params->ith == 0);
    memset(cdata, 0, ggml_nbytes(dst));

    ଅଚଳ int64_t px = src->ne[0];
    ଅଚଳ int64_t py = src->ne[1];
    ଅଚଳ int64_t pa = px * py;

    ଅଚଳ ଦଶମିକ * splane = (ଅଚଳ ଦଶମିକ *) src->data;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ka = k0 * k1;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା offset0 = -p0;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା offset1 = -p1;

    ଯେତେବେଳେ (cdata < data_end) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା oy = 0; oy < py; ++oy) {
            ଅଚଳ ଦଶମିକ * ଅଚଳ srow = splane + oy * px;
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ox = 0; ox < px; ++ox) {
                ଅଚଳ ଦଶମିକ grad0 = srow[ox];

                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ix = offset0 + ox * s0;
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iy = offset1 + oy * s1;

                ଯଦି (op == GGML_OP_POOL_MAX) {
                    ଦଶମିକ maxval = -FLT_MAX;
                    ପୂର୍ଣ୍ଣସଂଖ୍ୟା kxmax = -1;
                    ପୂର୍ଣ୍ଣସଂଖ୍ୟା kymax = -1;

                    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ky = 0; ky < k1; ++ky) {
                        ଯଦି (iy + ky < 0 || iy + ky >= dst->ne[1]) {
                            ଜାରି;
                        }
                        ଅଚଳ ଶୂନ୍ୟ * drowf = (ଅଚଳ ଶୂନ୍ୟ *)(cdataf + dst->nb[1] * (iy + ky));
                        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା kx = 0; kx < k0; ++kx) {
                            ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = ix + kx;
                            ଯଦି (j < 0 || j >= dst->ne[0]) {
                                ଜାରି;
                            }

                            ଅଚଳ ଦଶମିକ val = dst->type == GGML_TYPE_F32 ?
                                ((ଅଚଳ ଦଶମିକ *) drowf)[j] : GGML_CPU_FP16_TO_FP32(((ଅଚଳ ggml_fp16_t *) drowf)[j]);
                            ଯଦି (val <= maxval) {
                                ଜାରି;
                            }

                            maxval = val;
                            kxmax = kx;
                            kymax = ky;
                        }
                    }

                    ଯଦି (kxmax == -1 || kymax == -1) {
                        ଜାରି;
                    }

                    ଶୂନ୍ୟ * drow = (ଶୂନ୍ୟ *)(cdata + dst->nb[1] * (iy + kymax));
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = ix + kxmax;
                    ଯଦି (dst->type == GGML_TYPE_F32) {
                        ((ଦଶମିକ *) drow)[j] += grad0;
                    } ଅନ୍ୟଥା {
                        ((ggml_fp16_t *) drow)[j] = GGML_CPU_FP32_TO_FP16(grad0 + GGML_CPU_FP16_TO_FP32(((ଅଚଳ ggml_fp16_t *) drow)[j]));
                    }
                } ଅନ୍ୟଥା ଯଦି (op == GGML_OP_POOL_AVG) {
                    ଅଚଳ ଦଶମିକ grad = grad0 / ka;

                    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ky = 0; ky < k1; ++ky) {
                        ଯଦି (iy + ky < 0 || iy + ky >= dst->ne[1]) {
                            ଜାରି;
                        }
                        ଶୂନ୍ୟ * drow = (ଶୂନ୍ୟ *)(cdata + dst->nb[1] * (iy + ky));
                        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା kx = 0; kx < k0; ++kx) {
                            ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = ix + kx;
                            ଯଦି (j < 0 || j >= dst->ne[0]) {
                                ଜାରି;
                            }

                            ଯଦି (dst->type == GGML_TYPE_F32) {
                                ((ଦଶମିକ *) drow)[j] += grad;
                            } ଅନ୍ୟଥା {
                                ((ggml_fp16_t *) drow)[j] += GGML_CPU_FP32_TO_FP16(grad);
                            }
                        }
                    }
                } ଅନ୍ୟଥା {
                    GGML_ASSERT(ମିଥ୍ୟା);
                }
            }
        }

        cdata  += dst->nb[2];
        cdataf += dst->nb[2];
        splane += pa;
    }
}

// ggml_compute_forward_upscale

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_upscale_f32(
    ଅଚଳ ggml_compute_params * params,
    ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    ଦଶମିକ sf0 = (ଦଶମିକ)ne0/src0->ne[0];
    ଦଶମିକ sf1 = (ଦଶମିକ)ne1/src0->ne[1];
    ଦଶମିକ sf2 = (ଦଶମିକ)ne2/src0->ne[2];
    ଦଶମିକ sf3 = (ଦଶମିକ)ne3/src0->ne[3];
    ଦଶମିକ pixel_offset = 0.5f;

    ଅଚଳ int32_t mode_flags = ggml_get_op_params_i32(dst, 0);
    ଅଚଳ ggml_scale_mode mode = (ggml_scale_mode) (mode_flags & 0xFF);

    ଯଦି (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        pixel_offset = 0.0f;
        sf0 = ne0 > 1 && ne00 > 1 ? (ଦଶମିକ)(ne0 - 1) / (ne00 - 1) : sf0;
        sf1 = ne1 > 1 && ne01 > 1 ? (ଦଶମିକ)(ne1 - 1) / (ne01 - 1) : sf1;
    }

    ଯଦି (mode == GGML_SCALE_MODE_NEAREST) {
        ପାଇଁ (int64_t i3 = 0; i3 < ne3; i3++) {
            ଅଚଳ int64_t i03 = i3 / sf3;
            ପାଇଁ (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                ଅଚଳ int64_t i02 = i2 / sf2;
                ପାଇଁ (int64_t i1 = 0; i1 < ne1; i1++) {
                    ଅଚଳ int64_t i01 = i1 / sf1;
                    ପାଇଁ (int64_t i0 = 0; i0 < ne0; i0++) {
                        ଅଚଳ int64_t i00 = i0 / sf0;

                        ଅଚଳ ଦଶମିକ * x = (ଦଶମିକ *)((ଅକ୍ଷର *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              ଦଶମିକ * y = (ଦଶମିକ *)((ଅକ୍ଷର *)  dst->data +  i0*nb0  +  i1*nb1  +  i2*nb2  +  i3*nb3);

                        *y = *x;
                    }
                }
            }
        }
    } ଅନ୍ୟଥା ଯଦି (mode == GGML_SCALE_MODE_BILINEAR && (mode_flags & GGML_SCALE_FLAG_ANTIALIAS)) {
        // Similar to F.interpolate(..., mode="bilinear", align_corners=False, antialias=True)
        // https://github.com/pytorch/pytorch/blob/8871ff29b743948d1225389d5b7068f37b22750b/aten/src/ATen/native/cpu/UpSampleKernel.cpp
        auto triangle_filter = [](ଦଶମିକ x) -> ଦଶମିକ {
            ଫେରନ୍ତୁ std::max(1.0f - fabsf(x), 0.0f);
        };

        // support and invscale, minimum 1 pixel for bilinear
        ଅଚଳ ଦଶମିକ support1  = std::max(1.0f, 1.0f / sf1);
        ଅଚଳ ଦଶମିକ invscale1 = 1.0f / support1;
        ଅଚଳ ଦଶମିକ support0  = std::max(1.0f, 1.0f / sf0);
        ଅଚଳ ଦଶମିକ invscale0 = 1.0f / support0;

        ପାଇଁ (int64_t i3 = 0; i3 < ne3; i3++) {
            ଅଚଳ int64_t i03 = i3 / sf3;
            ପାଇଁ (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                ଅଚଳ int64_t i02 = i2 / sf2;
                ପାଇଁ (int64_t i1 = 0; i1 < ne1; i1++) {
                    ଅଚଳ ଦଶମିକ y = ((ଦଶମିକ) i1 + pixel_offset) / sf1;
                    ପାଇଁ (int64_t i0 = 0; i0 < ne0; i0++) {
                        ଅଚଳ ଦଶମିକ x = ((ଦଶମିକ) i0 + pixel_offset) / sf0;

                        // the range of source pixels that contribute
                        ଅଚଳ int64_t x_min = std::max<int64_t>(x - support0 + pixel_offset, 0);
                        ଅଚଳ int64_t x_max = std::min<int64_t>(x + support0 + pixel_offset, ne00);
                        ଅଚଳ int64_t y_min = std::max<int64_t>(y - support1 + pixel_offset, 0);
                        ଅଚଳ int64_t y_max = std::min<int64_t>(y + support1 + pixel_offset, ne01);

                        // bilinear filter with antialiasing
                        ଦଶମିକ val = 0.0f;
                        ଦଶମିକ total_weight = 0.0f;

                        ପାଇଁ (int64_t sy = y_min; sy < y_max; sy++) {
                            ଅଚଳ ଦଶମିକ weight_y = triangle_filter((sy - y + pixel_offset) * invscale1);

                            ପାଇଁ (int64_t sx = x_min; sx < x_max; sx++) {
                                ଅଚଳ ଦଶମିକ weight_x = triangle_filter((sx - x + pixel_offset) * invscale0);
                                ଅଚଳ ଦଶମିକ weight = weight_x * weight_y;

                                ଯଦି (weight <= 0.0f) {
                                    ଜାରି;
                                }

                                ଅଚଳ ଦଶମିକ pixel = *(ଅଚଳ ଦଶମିକ *)((ଅଚଳ ଅକ୍ଷର *)src0->data + sx*nb00 + sy*nb01 + i02*nb02 + i03*nb03);
                                val += pixel * weight;
                                total_weight += weight;
                            }
                        }

                        ଯଦି (total_weight > 0.0f) {
                            val /= total_weight;
                        }

                        ଦଶମିକ * dst_ptr = (ଦଶମିକ *)((ଅକ୍ଷର *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);
                        *dst_ptr = val;
                    }
                }
            }
        }
    } ଅନ୍ୟଥା ଯଦି (mode == GGML_SCALE_MODE_BILINEAR) {
        ପାଇଁ (int64_t i3 = 0; i3 < ne3; i3++) {
            ଅଚଳ int64_t i03 = i3 / sf3;
            ପାଇଁ (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                ଅଚଳ int64_t i02 = i2 / sf2;
                ପାଇଁ (int64_t i1 = 0; i1 < ne1; i1++) {
                    ଅଚଳ ଦଶମିକ y = ((ଦଶମିକ)i1 + pixel_offset) / sf1 - pixel_offset;
                    int64_t y0 = (int64_t)floorf(y);
                    int64_t y1 = y0 + 1;

                    y0 = std::max(int64_t(0), std::min(y0, ne01 - 1));
                    y1 = std::max(int64_t(0), std::min(y1, ne01 - 1));

                    ଦଶମିକ dy = y - (ଦଶମିକ)y0;
                    dy = std::max(0.0f, std::min(dy, 1.0f));

                    ପାଇଁ (int64_t i0 = 0; i0 < ne0; i0++) {
                        ଅଚଳ ଦଶମିକ x = ((ଦଶମିକ)i0 + pixel_offset) / sf0 - pixel_offset;
                        int64_t x0 = (int64_t)floorf(x);
                        int64_t x1 = x0 + 1;

                        x0 = std::max(int64_t(0), std::min(x0, ne00 - 1));
                        x1 = std::max(int64_t(0), std::min(x1, ne00 - 1));

                        ଦଶମିକ dx = x - (ଦଶମିକ)x0;
                        dx = std::max(0.0f, std::min(dx, 1.0f));

                        // fetch the four surrounding pixel values and interpolate
                        ଅଚଳ ଦଶମିକ a = *(ଅଚଳ ଦଶମିକ *)((ଅଚଳ ଅକ୍ଷର *)src0->data + x0*nb00 + y0*nb01 + i02*nb02 + i03*nb03);
                        ଅଚଳ ଦଶମିକ b = *(ଅଚଳ ଦଶମିକ *)((ଅଚଳ ଅକ୍ଷର *)src0->data + x1*nb00 + y0*nb01 + i02*nb02 + i03*nb03);
                        ଅଚଳ ଦଶମିକ c = *(ଅଚଳ ଦଶମିକ *)((ଅଚଳ ଅକ୍ଷର *)src0->data + x0*nb00 + y1*nb01 + i02*nb02 + i03*nb03);
                        ଅଚଳ ଦଶମିକ d = *(ଅଚଳ ଦଶମିକ *)((ଅଚଳ ଅକ୍ଷର *)src0->data + x1*nb00 + y1*nb01 + i02*nb02 + i03*nb03);

                        ଅଚଳ ଦଶମିକ val = a*(1 - dx)*(1 - dy) + b*dx*(1 - dy) + c*(1 - dx)*dy + d*dx*dy;

                        ଦଶମିକ * y_dst = (ଦଶମିକ *)((ଅକ୍ଷର *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);
                        *y_dst = val;
                    }
                }
            }
        }
    } ଅନ୍ୟଥା ଯଦି (mode == GGML_SCALE_MODE_BICUBIC) {
        // https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
        ଅଚଳ ଦଶମିକ a = -0.75f; // use alpha = -0.75 (same as PyTorch)
        auto weight1 = [a](ଦଶମିକ x) { ଫେରନ୍ତୁ ((a + 2) * x - (a + 3)) * x * x + 1; };
        auto weight2 = [a](ଦଶମିକ x) { ଫେରନ୍ତୁ ((a * x - 5 * a) * x + 8 * a) * x - 4 * a; };
        auto bicubic = [=](ଦଶମିକ p0, ଦଶମିକ p1, ଦଶମିକ p2, ଦଶମିକ p3, ଦଶମିକ x) {
            ଅଚଳ ଦଶମିକ w0 = weight2(x + 1);
            ଅଚଳ ଦଶମିକ w1 = weight1(x + 0);
            ଅଚଳ ଦଶମିକ w2 = weight1(1 - x);
            ଅଚଳ ଦଶମିକ w3 = weight2(2 - x);
            ଫେରନ୍ତୁ p0*w0 + p1*w1 + p2*w2 + p3*w3;
        };

        ପାଇଁ (int64_t i3 = 0; i3 < ne3; i3++) {
            ଅଚଳ int64_t i03 = i3 / sf3;
            ପାଇଁ (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                ଅଚଳ int64_t i02 = i2 / sf2;
                ପାଇଁ (int64_t i1 = 0; i1 < ne1; i1++) {
                    ଅଚଳ ଦଶମିକ y = ((ଦଶମିକ)i1 + pixel_offset) / sf1 - pixel_offset;
                    ଅଚଳ int64_t y0 = (int64_t)floorf(y);
                    ଅଚଳ ଦଶମିକ dy = y - (ଦଶମିକ)y0;

                    ପାଇଁ (int64_t i0 = 0; i0 < ne0; i0++) {
                        ଅଚଳ ଦଶମିକ x = ((ଦଶମିକ)i0 + pixel_offset) / sf0 - pixel_offset;
                        ଅଚଳ int64_t x0 = (int64_t)floorf(x);
                        ଅଚଳ ଦଶମିକ dx = x - (ଦଶମିକ)x0;

                        auto p = [=](int64_t x_off, int64_t y_off) -> ଦଶମିକ {
                            int64_t i00 = std::max(int64_t(0), std::min(x0 + x_off, ne00 - 1));
                            int64_t i01 = std::max(int64_t(0), std::min(y0 + y_off, ne01 - 1));
                            ଫେରନ୍ତୁ *(ଅଚଳ ଦଶମିକ *)((ଅଚଳ ଅକ୍ଷର *)src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                        };

                        ଅଚଳ ଦଶମିକ val = bicubic(
                            bicubic(p(-1,-1), p(0,-1), p(1,-1), p(2,-1), dx),
                            bicubic(p(-1, 0), p(0, 0), p(1, 0), p(2, 0), dx),
                            bicubic(p(-1, 1), p(0, 1), p(1, 1), p(2, 1), dx),
                            bicubic(p(-1, 2), p(0, 2), p(1, 2), p(2, 2), dx), dy);

                        ଦଶମିକ * y_dst = (ଦଶମିକ *)((ଅକ୍ଷର *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);
                        *y_dst = val;
                    }
                }
            }
        }
    } ଅନ୍ୟଥା {
        GGML_ABORT("unsupported upscale mode");
    }
}

ଶୂନ୍ୟ ggml_compute_forward_upscale(
    ଅଚଳ ggml_compute_params * params,
    ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_upscale_f32(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}


// ggml_compute_forward_pad

template<ତାର୍କିକ circular_t>
ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_pad_f32(
    ଅଚଳ ggml_compute_params * params,
          ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ପ୍ରତିଜ୍ଞା(dst->nb[0] == ଆକାର(ଦଶମିକ));

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    ଦଶମିକ * dst_ptr = (ଦଶମିକ *) dst->data;
    ଅଚଳ int32_t lp0 = ggml_get_op_params_i32(dst, 0);
    ଅଚଳ int32_t rp0 = ggml_get_op_params_i32(dst, 1);
    ଅଚଳ int32_t lp1 = ggml_get_op_params_i32(dst, 2);
    ଅଚଳ int32_t rp1 = ggml_get_op_params_i32(dst, 3);
    ଅଚଳ int32_t lp2 = ggml_get_op_params_i32(dst, 4);
    ଅଚଳ int32_t rp2 = ggml_get_op_params_i32(dst, 5);
    ଅଚଳ int32_t lp3 = ggml_get_op_params_i32(dst, 6);
    ଅଚଳ int32_t rp3 = ggml_get_op_params_i32(dst, 7);

    ଅଚଳ int64_t nrows = ne1 * ne2 * ne3;
    ଅଚଳ ତାର୍କିକ src_row_contig = src0->nb[0] == ଆକାର(ଦଶମିକ);

    ପାଇଁ (int64_t row = ith; row < nrows; row += nth) {
        ଅଚଳ int64_t i1 = row % ne1;
        ଅଚଳ int64_t i2 = (row / ne1) % ne2;
        ଅଚଳ int64_t i3 = row / (ne1 * ne2);

        ଦଶମିକ * dst_row = dst_ptr + row * ne0;

        ଯଦି constexpr (circular_t) {
            // circular padding wraps around on a torus
            ଅଚଳ int64_t src_i1 = ggml_wrap_around(i1 - lp1, ne01);
            ଅଚଳ int64_t src_i2 = ggml_wrap_around(i2 - lp2, ne02);
            ଅଚଳ int64_t src_i3 = ggml_wrap_around(i3 - lp3, ne03);

            ପାଇଁ (int64_t i0 = 0; i0 < ne0; ++i0) {
                ଅଚଳ int64_t src_i0 = ggml_wrap_around(i0 - lp0, ne00);
                dst_row[i0] = *(ଅଚଳ ଦଶମିକ *)((ଅଚଳ ଅକ୍ଷର *) src0->data + src_i3*nb03 + src_i2*nb02 + src_i1*nb01 + src_i0*nb00);
            }
        } ଅନ୍ୟଥା {
            ଅଚଳ ତାର୍କିକ interior = (i1 >= lp1 && i1 < ne1 - rp1)
                               && (i2 >= lp2 && i2 < ne2 - rp2)
                               && (i3 >= lp3 && i3 < ne3 - rp3);
            ଯଦି (!interior) {
                memset(dst_row, 0, ne0 * ଆକାର(ଦଶମିକ));
                ଜାରି;
            }

            ଅଚଳ ଅକ୍ଷର * src_row = (ଅଚଳ ଅକ୍ଷର *) src0->data + (i3 - lp3)*nb03 + (i2 - lp2)*nb02 + (i1 - lp1)*nb01;

            ଯଦି (src_row_contig && lp0 >= 0 && rp0 >= 0) {
                memset(dst_row, 0, lp0 * ଆକାର(ଦଶମିକ));
                memcpy(dst_row + lp0, src_row, ne00 * ଆକାର(ଦଶମିକ));
                memset(dst_row + lp0 + ne00, 0, rp0 * ଆକାର(ଦଶମିକ));
            } ଅନ୍ୟଥା {
                ପାଇଁ (int64_t i0 = 0; i0 < ne0; ++i0) {
                    ଯଦି (i0 >= lp0 && i0 < ne0 - rp0) {
                        dst_row[i0] = *(ଅଚଳ ଦଶମିକ *)(src_row + (i0 - lp0)*nb00);
                    } ଅନ୍ୟଥା {
                        dst_row[i0] = 0.0f;
                    }
                }
            }
        }
    }
}


ଶୂନ୍ୟ ggml_compute_forward_pad(
    ଅଚଳ ggml_compute_params * params,
    ggml_tensor * dst) {
    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ତାର୍କିକ circular = (ତାର୍କିକ) ggml_get_op_params_i32(dst, 8);
    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ଯଦି (circular) {
                    ggml_compute_forward_pad_f32<ସତ୍ୟ>(params, dst);
                } ଅନ୍ୟଥା {
                    ggml_compute_forward_pad_f32<ମିଥ୍ୟା>(params, dst);
                }
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_pad_reflect_1d

ଶୂନ୍ୟ ggml_compute_forward_pad_reflect_1d(
        ଅଚଳ ggml_compute_params * params,
              ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ int32_t * opts = (ଅଚଳ int32_t *) dst->op_params;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା p0 = opts[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା p1 = opts[1];

    GGML_TENSOR_UNARY_OP_LOCALS

    ପାଇଁ (int64_t i3 = 0; i3 < ne3; i3++) {
        ପାଇଁ (int64_t i2 = 0; i2 < ne2; i2++) {
            ପାଇଁ (int64_t i1 = ith; i1 < ne1; i1 += nth) {
                ଦଶମିକ * left  = (ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data + i3*nb3 + i2*nb2 + i1*nb1 +         p0*nb0);
                ଦଶମିକ * right = (ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data + i3*nb3 + i2*nb2 + i1*nb1 + (ne0-p1-1)*nb0);

                ggml_vec_cpy_f32(ne00, left, (ଦଶମିକ *) ((ଅକ୍ଷର *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0 = 1; i0 <= p0; i0++) { left[-i0] = left[i0];   }
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0 = 1; i0 <= p1; i0++) { right[i0] = right[-i0]; }
            }
        }
    }
}

// ggml_compute_forward_roll

ସ୍ଥିର int64_t ggml_wrap_index(int64_t i, int64_t ne) {
    ଯଦି (i < 0) {
        ଫେରନ୍ତୁ i + ne;
    } ଅନ୍ୟଥା ଯଦି (i >= ne) {
        ଫେରନ୍ତୁ i - ne;
    }
    ଫେରନ୍ତୁ i;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_roll_f32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ଦଶମିକ * src_data = (ଅଚଳ ଦଶମିକ *) src0->data;
    ଦଶମିକ * dst_data = (ଦଶମିକ *) dst->data;

    GGML_TENSOR_UNARY_OP_LOCALS

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s0 = ggml_get_op_params_i32(dst, 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s1 = ggml_get_op_params_i32(dst, 1);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s2 = ggml_get_op_params_i32(dst, 2);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s3 = ggml_get_op_params_i32(dst, 3);

    ଅଚଳ int64_t total = ne1 * ne2 * ne3;
    ଅଚଳ int64_t per_thread = (total + params->nth) / params->nth;
    ଅଚଳ int64_t start = params->ith * per_thread;
    ଅଚଳ int64_t end   = std::min(start + per_thread, total);

    ପାଇଁ (int64_t i = start; i < end; ++i) {
        ଅଚଳ int64_t i1 = i % ne1;
        ଅଚଳ int64_t i2 = (i / ne1) % ne2;
        ଅଚଳ int64_t i3 = i / (ne2 * ne1);
        ଦଶମିକ * dst_row = dst_data + (i3*nb3 + i2*nb2 + i1*nb1) / ଆକାର(ଦଶମିକ);

        ଅଚଳ int64_t i01 = ggml_wrap_index(i1 - s1, ne01);
        ଅଚଳ int64_t i02 = ggml_wrap_index(i2 - s2, ne02);
        ଅଚଳ int64_t i03 = ggml_wrap_index(i3 - s3, ne03);
        ଅଚଳ ଦଶମିକ * src_row = src_data + (i03*nb03 + i02*nb02 + i01*nb01) / ଆକାର(ଦଶମିକ);

        ଅଚଳ int64_t s = ggml_wrap_index(-s0, ne00);
        ଅଚଳ int64_t n = ne00 - s;
        ggml_vec_cpy_f32(n, dst_row,     src_row + s);
        ggml_vec_cpy_f32(s, dst_row + n, src_row);
    }
}

ଶୂନ୍ୟ ggml_compute_forward_roll(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_roll_f32(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_arange

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_arange_f32(
    ଅଚଳ ggml_compute_params * params,
    ggml_tensor * dst) {

    GGML_ASSERT(dst->nb[0] == ଆକାର(ଦଶମିକ));

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ଦଶମିକ start = ggml_get_op_params_f32(dst, 0);
    ଅଚଳ ଦଶମିକ stop  = ggml_get_op_params_f32(dst, 1);
    ଅଚଳ ଦଶମିକ step  = ggml_get_op_params_f32(dst, 2);

    ଅଚଳ int64_t steps = (int64_t) ceilf((stop - start) / step);

    GGML_ASSERT(ggml_nelements(dst) == steps);

    ପାଇଁ (int64_t i = ith; i < steps; i+= nth) {
        ଦଶମିକ value = start + step * i;
        ((ଦଶମିକ *)dst->data)[i] = value;
    }
}

ଶୂନ୍ୟ ggml_compute_forward_arange(
    ଅଚଳ ggml_compute_params * params,
    ggml_tensor * dst) {
    ବାଛ (dst->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_arange_f32(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_timestep_embedding_f32(
    ଅଚଳ ggml_compute_params * params,
    ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->nb[0] == ଆକାର(ଦଶମିକ));

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dim = ggml_get_op_params_i32(dst, 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା max_period = ggml_get_op_params_i32(dst, 1);

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା half = dim / 2;

    ପାଇଁ (int64_t i = 0; i < ne00; i++) {
        ଦଶମିକ * embed_data = (ଦଶମିକ *)((ଅକ୍ଷର *)  dst->data +  i*nb1);
        ପାଇଁ (int64_t j = ith; j < half; j += nth) {
            ଦଶମିକ timestep = ((ଦଶମିକ *)src0->data)[i];
            ଦଶମିକ freq = (ଦଶମିକ)expf(-logf(max_period) * j / half);
            ଦଶମିକ arg = timestep * freq;
            embed_data[j] = cosf(arg);
            embed_data[j + half] = sinf(arg);
        }
        ଯଦି (dim % 2 != 0 && ith == 0) {
            embed_data[2 * half] = 0.f;
        }
    }
}

ଶୂନ୍ୟ ggml_compute_forward_timestep_embedding(
    ଅଚଳ ggml_compute_params * params,
    ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_timestep_embedding_f32(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_argsort

template<ଗଣନା ggml_sort_order order>
ସଂରଚନା cmp_argsort {
    ଅଚଳ ଦଶମିକ * data;
    ତାର୍କିକ operator()(int32_t a, int32_t b) ଅଚଳ {
        ଯଦି constexpr (order == GGML_SORT_ORDER_ASC) {
            ଫେରନ୍ତୁ data[a] < data[b];
        } ଅନ୍ୟଥା {
            ଫେରନ୍ତୁ data[a] > data[b];
        }
    }
};

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_argsort_f32(
    ଅଚଳ ggml_compute_params * params,
    ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(nb0 == ଆକାର(ଦଶମିକ));

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ int64_t nr = ggml_nrows(src0);

    ggml_sort_order order = (ggml_sort_order) ggml_get_op_params_i32(dst, 0);

    ପାଇଁ (int64_t i = ith; i < nr; i += nth) {
        ଅଚଳ ଦଶମିକ * src_data = (ଦଶମିକ *)((ଅକ୍ଷର *) src0->data + i*nb01);

        int32_t * dst_data = (int32_t *)((ଅକ୍ଷର *) dst->data + i*nb1);

        ପାଇଁ (int64_t j = 0; j < ne0; j++) {
            dst_data[j] = j;
        }

        ବାଛ (order) {
            ସ୍ଥିତି GGML_SORT_ORDER_ASC:
                std::sort(dst_data, dst_data + ne0, cmp_argsort<GGML_SORT_ORDER_ASC>{src_data});
                ଭାଙ୍ଗ;

            ସ୍ଥିତି GGML_SORT_ORDER_DESC:
                std::sort(dst_data, dst_data + ne0, cmp_argsort<GGML_SORT_ORDER_DESC>{src_data});
                ଭାଙ୍ଗ;

            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                GGML_ABORT("invalid sort order");
        }
    }
}

ଶୂନ୍ୟ ggml_compute_forward_argsort(
    ଅଚଳ ggml_compute_params * params,
    ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_argsort_f32(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_top_k

ସଂରଚନା cmp_ଶୀର୍ଷ_k {
    ଅଚଳ ଦଶମିକ * data;
    ତାର୍କିକ operator()(int32_t a, int32_t b) ଅଚଳ {
        ଫେରନ୍ତୁ data[a] > data[b];
    }
};

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_ଶୀର୍ଷ_k_f32(
    ଅଚଳ ggml_compute_params * params,
    ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(nb0 == ଆକାର(ଦଶମିକ));

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ int64_t nr = ggml_nrows(src0);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଶୀର୍ଷ_k = ne0;

    int32_t * tmp = (int32_t *) params->wdata + (ne00 + CACHE_ଧାଡ଼ି_SIZE_F32) * ith;

    ପାଇଁ (int64_t i = ith; i < nr; i += nth) {
        ଅଚଳ ଦଶମିକ * src_data = (ଦଶମିକ *)((ଅକ୍ଷର *) src0->data + i*nb01);

        ପାଇଁ (int64_t j = 0; j < ne00; j++) {
            tmp[j] = j;
        }

        std::partial_sort(tmp, tmp + ଶୀର୍ଷ_k, tmp + ne00, cmp_ଶୀର୍ଷ_k{src_data});

        int32_t * dst_data = (int32_t *)((ଅକ୍ଷର *) dst->data + i*nb1);

        std::copy(tmp, tmp + ଶୀର୍ଷ_k, dst_data);

        // emphasize that the order is not important
        ଯଦି (ଶୀର୍ଷ_k > 1) {
            std::swap(dst_data[0], dst_data[1]);
        }
    }
}

ଶୂନ୍ୟ ggml_compute_forward_ଶୀର୍ଷ_k(
    ଅଚଳ ggml_compute_params * params,
    ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_ଶୀର୍ଷ_k_f32(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_flash_attn_ext_f16_one_chunk(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1,
        int64_t ic_start, int64_t ic_end,
        ଦଶମିକ * partials, int64_t partial_stride) {

    ଅଚଳ ତାର୍କିକ write_partials = (partials != nullptr);
    ଅଚଳ ggml_tensor * q     = dst->src[0];
    ଅଚଳ ggml_tensor * k     = dst->src[1];
    ଅଚଳ ggml_tensor * v     = dst->src[2];
    ଅଚଳ ggml_tensor * mask  = dst->src[3];
    ଅଚଳ ggml_tensor * sinks = dst->src[4];

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    ଅଚଳ int64_t DK = nek0;
    ଅଚଳ int64_t DV = nev0;
    ଅଚଳ int64_t N  = neq1;

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
    GGML_ASSERT(nb0 == ଆକାର(ଦଶମିକ));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    ଅଚଳ int64_t rk2 = neq2/nek2;
    ଅଚଳ int64_t rk3 = neq3/nek3;

    ଅଚଳ int64_t rv2 = neq2/nev2;
    ଅଚଳ int64_t rv3 = neq3/nev3;

    // parallelize by q rows using ggml_vec_dot_f32

    ଦଶମିକ scale         = 1.0f;
    ଦଶମିକ max_bias      = 0.0f;
    ଦଶମିକ logit_softcap = 0.0f;

    memcpy(&scale,         (ଦଶମିକ *) dst->op_params + 0, ଆକାର(ଦଶମିକ));
    memcpy(&max_bias,      (ଦଶମିକ *) dst->op_params + 1, ଆକାର(ଦଶମିକ));
    memcpy(&logit_softcap, (ଦଶମିକ *) dst->op_params + 2, ଆକାର(ଦଶମିକ));

    ଯଦି (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    ଅଚଳ uint32_t n_head      = neq2;
    ଅଚଳ uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));

    ଅଚଳ ଦଶମିକ m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    ଅଚଳ ଦଶମିକ m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    ggml_type         ଅଚଳ k_vec_dot_type = ggml_get_type_traits_cpu(k->type)->vec_dot_type;
    ggml_from_float_t ଅଚଳ q_to_vec_dot   = ggml_get_type_traits_cpu(k_vec_dot_type)->from_float;
    ggml_vec_dot_t    ଅଚଳ kq_vec_dot     = ggml_get_type_traits_cpu(k->type)->vec_dot;
    ggml_to_float_t   ଅଚଳ v_to_float     = ggml_get_type_traits(v->type)->to_float;

    GGML_ASSERT((                            q_to_vec_dot) && "fattn: unsupported K-type");
    GGML_ASSERT((v->type == GGML_TYPE_F32 || v_to_float  ) && "fattn: unsupported V-type");

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir = ir0; ir < ir1; ++ir) {
        // q indices
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iq3 = ir/(neq2*neq1);
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iq2 = (ir - iq3*neq2*neq1)/neq1;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iq1 = (ir - iq3*neq2*neq1 - iq2*neq1);

        ଅଚଳ uint32_t h = iq2; // head index
        ଅଚଳ ଦଶମିକ slope = (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1) : 1.0f;

        ଦଶମିକ S = 0.0f;      // sum
        ଦଶମିକ M = -INFINITY; // maximum KQ value

        ଦଶମିକ       * VKQ32 = (ଦଶମିକ       *) params->wdata + ith*(1*DK + 2*DV + CACHE_ଧାଡ଼ି_SIZE_F32); // FP32 VKQ accumulator
        ଦଶମିକ       * V32   =                 (VKQ32 + 1*DV); // (temporary) FP32 V buffer
        ggml_fp16_t * VKQ16 = (ggml_fp16_t *) (VKQ32 + 1*DV); // (temporary) FP16 VKQ accumulator
        ggml_fp16_t * Q_q   = (ggml_fp16_t *) (VKQ32 + 2*DV); // (temporary) buffer for Q converted to quantized/FP16

        ଯଦି (v->type == GGML_TYPE_F16) {
            memset(VKQ16, 0, DV*ଆକାର(ggml_fp16_t));
        } ଅନ୍ୟଥା {
            memset(VKQ32, 0, DV*ଆକାର(ଦଶମିକ));
        }

        ଅଚଳ ggml_fp16_t * mp = mask ? (ggml_fp16_t *)((ଅକ୍ଷର *) mask->data + iq1*mask->nb[1] + (iq2%mask->ne[2])*mask->nb[2] + (iq3%mask->ne[3])*mask->nb[3]) : ଖାଲି;

        // k indices
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ik3 = iq3 / rk3;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ik2 = iq2 / rk2;

        // v indices
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iv3 = iq3 / rv3;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iv2 = iq2 / rv2;

        ଅଚଳ ଦଶମିକ * pq = (ଅଚଳ ଦଶମିକ *) ((ଅକ୍ଷର *) q->data + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3));
        q_to_vec_dot(pq, Q_q, DK);

        // online softmax / attention
        // loop over n_kv and n_head_kv
        // ref: https://arxiv.org/pdf/2112.05682.pdf

        ପାଇଁ (int64_t ic = ic_start; ic < ic_end; ++ic) {
            ଅଚଳ ଦଶମିକ mv = mp ? slope*GGML_CPU_FP16_TO_FP32(mp[ic]) : 0.0f;
            ଯଦି (mv == -INFINITY) {
                ଜାରି;
            }

            ଦଶମିକ s; // KQ value

            ଅଚଳ ଅକ୍ଷର * k_data = (ଅଚଳ ଅକ୍ଷର *) k->data + ( ic*nbk1 + ik2*nbk2 + ik3*nbk3);
            kq_vec_dot(DK, &s, 0, k_data, 0, Q_q, 0, 1);

            s = s*scale; // scale KQ value

            ଯଦି (logit_softcap != 0.0f) {
                s = logit_softcap*tanhf(s);
            }

            s += mv; // apply mask

            ଅଚଳ ଦଶମିକ Mold = M;

            ଦଶମିକ ms = 1.0f; // upon new higher max val, scale VKQ and KQ sum with this value
            ଦଶମିକ vs = 1.0f; // post-softmax KQ value, expf(s - M)

            ଅଚଳ ଅକ୍ଷର * v_data = ((ଅଚଳ ଅକ୍ଷର *) v->data + (ic*nbv1 + iv2*nbv2 + iv3*nbv3));

            ଯଦି (v->type == GGML_TYPE_F16) {
                ଯଦି (s > M) {
                    // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
                    M = s;
                    ms = expf(Mold - M);

                    // V = V*expf(Mold - M)
                    ggml_vec_scale_f16(DV, VKQ16, ms);
                } ଅନ୍ୟଥା {
                    // no new maximum, ms == 1.0f, vs != 1.0f
                    vs = expf(s - M);
                }

                // V += v*expf(s - M)
                ggml_vec_mad_f16(DV, VKQ16, (ଅଚଳ ggml_fp16_t *) v_data, vs);
            } ଅନ୍ୟଥା {
                ଯଦି (s > M) {
                    // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
                    M = s;
                    ms = expf(Mold - M);

                    // V = V*expf(Mold - M)
                    ggml_vec_scale_f32(DV, VKQ32, ms);
                } ଅନ୍ୟଥା {
                    // no new maximum, ms == 1.0f, vs != 1.0f
                    vs = expf(s - M);
                }

                // V += v*expf(s - M)
                ଯଦି (v_to_float) {
                    v_to_float(v_data, V32, DV);
                    ggml_vec_mad_f32(DV, VKQ32, V32, vs);
                } ଅନ୍ୟଥା {
                    // V is F32
                    ggml_vec_mad_f32(DV, VKQ32, (ଅଚଳ ଦଶମିକ *) v_data, vs);
                }
            }

            S = S*ms + vs; // scale and increment sum with partial sum
        }

        ଯଦି (v->type == GGML_TYPE_F16) {
            ପାଇଁ (int64_t d = 0; d < DV; ++d) {
                VKQ32[d] = GGML_CPU_FP16_TO_FP32(VKQ16[d]);
            }
        }

        // sinks - apply only on the first kv-chunk
        ଯଦି (sinks && ic_start == 0) {
            ଅଚଳ ଦଶମିକ s = ((ଦଶମିକ *)((ଅକ୍ଷର *) sinks->data))[h];

            ଦଶମିକ ms = 1.0f;
            ଦଶମିକ vs = 1.0f;

            ଯଦି (s > M) {
                ms = expf(M - s);
                M = s;
                ggml_vec_scale_f32(DV, VKQ32, ms);
            } ଅନ୍ୟଥା {
                vs = expf(s - M);
            }

            S = S*ms + vs;
        }

        ଯଦି (write_partials) {
            // Write M, S, VKQ to partials for later reduction
            // partials layout: [M, S, VKQ[DV]] per query head
            ଦଶମିକ * partial = partials + ir * partial_stride;
            partial[0] = M;
            partial[1] = S;
            memcpy(partial + 2, VKQ32, DV * ଆକାର(ଦଶମିକ));
        } ଅନ୍ୟଥା {
            // V /= S
            ଅଚଳ ଦଶମିକ S_inv = S == 0.0f ? 0.0f : 1.0f/S;
            ggml_vec_scale_f32(DV, VKQ32, S_inv);

            // dst indices
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = iq1;
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i2 = iq2;
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i3 = iq3;

            // permute(0, 2, 1, 3)
            memcpy((ଅକ୍ଷର *) dst->data + (i3*ne2*ne1 + i2 + i1*ne1)*nb1, VKQ32, nb1);
        }
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_flash_attn_ext_tiled(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1) {
    ଅଚଳ ggml_tensor * q     = dst->src[0];
    ଅଚଳ ggml_tensor * k     = dst->src[1];
    ଅଚଳ ggml_tensor * v     = dst->src[2];
    ଅଚଳ ggml_tensor * mask  = dst->src[3];
    ଅଚଳ ggml_tensor * sinks = dst->src[4];

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    ଅଚଳ int64_t DK = nek0;
    ଅଚଳ int64_t DV = nev0;
    ଅଚଳ int64_t N  = neq1;

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
    GGML_ASSERT(nb0 == ଆକାର(ଦଶମିକ));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(k->type == v->type);
    ଅଚଳ ggml_type kv_type = k->type;


    // broadcast factors
    ଅଚଳ int64_t rk2 = neq2/nek2;
    ଅଚଳ int64_t rk3 = neq3/nek3;

    ଅଚଳ int64_t rv2 = neq2/nev2;
    ଅଚଳ int64_t rv3 = neq3/nev3;

    ଦଶମିକ scale         = 1.0f;
    ଦଶମିକ max_bias      = 0.0f;
    ଦଶମିକ logit_softcap = 0.0f;

    memcpy(&scale,         (ଦଶମିକ *) dst->op_params + 0, ଆକାର(ଦଶମିକ));
    memcpy(&max_bias,      (ଦଶମିକ *) dst->op_params + 1, ଆକାର(ଦଶମିକ));
    memcpy(&logit_softcap, (ଦଶମିକ *) dst->op_params + 2, ଆକାର(ଦଶମିକ));

    ଯଦି (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    ଅଚଳ uint32_t n_head      = neq2;
    ଅଚଳ uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));

    ଅଚଳ ଦଶମିକ m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    ଅଚଳ ଦଶମିକ m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;

    ସ୍ଥିର constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା Q_TILE_SZ  = ggml_fa_tile_config::Q;
    ସ୍ଥିର constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା KV_TILE_SZ = ggml_fa_tile_config::KV;

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir = ir0;
    ଯେତେବେଳେ (ir < ir1) {
        // q indices for the start of this tile
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iq3 = ir/(neq2*neq1);
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iq2 = (ir - iq3*neq2*neq1)/neq1;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iq1 = (ir - iq3*neq2*neq1 - iq2*neq1);

        // Number of valid rows in this tile:
        // - limited by tile size (Q_TILE_SZ)
        // - limited by chunk boundary (ir1 - ir)
        // - limited by head boundary (neq1 - iq1) to avoid crossing into next head
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା tile_rows = MIN(Q_TILE_SZ, MIN((ପୂର୍ଣ୍ଣସଂଖ୍ୟା)(ir1 - ir), (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)(neq1 - iq1)));
        GGML_ASSERT(tile_rows > 0);

        ଅଚଳ uint32_t h = iq2; // head index
        ଅଚଳ ଦଶମିକ slope = (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1) : 1.0f;

        ଦଶମିକ S[Q_TILE_SZ];
        ଦଶମିକ M[Q_TILE_SZ];

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0 ; i < Q_TILE_SZ; ++i) {
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
        ଦଶମିକ * base  = (ଦଶମିକ *) params->wdata + ith*(Q_TILE_SZ*DK + 2*Q_TILE_SZ*KV_TILE_SZ + Q_TILE_SZ*DV + KV_TILE_SZ*DV + KV_TILE_SZ*DK + CACHE_ଧାଡ଼ି_SIZE_F32);

        ଶୂନ୍ୟ  * Q_q    = base;
        ଦଶମିକ * KQ     = (ଦଶମିକ *)((ଅକ୍ଷର *)base + Q_TILE_SZ * DK * ଆକାର(ଦଶମିକ));
        ଦଶମିକ * mask32 = KQ + Q_TILE_SZ * KV_TILE_SZ;
        ଦଶମିକ * VKQ32  = mask32 + Q_TILE_SZ * KV_TILE_SZ;
        ଦଶମିକ * V32    = VKQ32 + Q_TILE_SZ * DV;
        ଦଶମିକ * K_f32  = V32 + KV_TILE_SZ * DV;

        memset(VKQ32, 0, Q_TILE_SZ * DV * ଆକାର(ଦଶମିକ));
        memset(mask32, 0, Q_TILE_SZ * KV_TILE_SZ * ଆକାର(ଦଶମିକ));

        // k indices
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ik3 = iq3 / rk3;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ik2 = iq2 / rk2;

        // v indices
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iv3 = iq3 / rv3;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iv2 = iq2 / rv2;

        {
            ଦଶମିକ * Q_f32 = (ଦଶମିକ *)Q_q;
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା tq = 0; tq < tile_rows; tq++) {
                ଅଚଳ ଦଶମିକ * pq = (ଅଚଳ ଦଶମିକ *) ((ଅକ୍ଷର *) q->data + ((iq1 + tq)*nbq1 + iq2*nbq2 + iq3*nbq3));
                memcpy(Q_f32 + tq * DK, pq, DK * ଆକାର(ଦଶମିକ));
            }
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା tq = tile_rows; tq < Q_TILE_SZ; tq++) {
                memset(Q_f32 + tq * DK, 0, DK * ଆକାର(ଦଶମିକ));
            }
        }

        memset(K_f32, 0, DK * KV_TILE_SZ * ଆକାର(ଦଶମିକ));
        memset(V32,   0, KV_TILE_SZ * DV * ଆକାର(ଦଶମିକ));

        ପାଇଁ (int64_t ic = 0; ic < nek1; ic += KV_TILE_SZ) {
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା kv_tile = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)std::min((int64_t)KV_TILE_SZ, nek1 - ic);

            // skip the tile entirely if all the masks are -inf
            ଯଦି (mask) {
                ତାର୍କିକ can_skip = ସତ୍ୟ;
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା tq = 0; tq < tile_rows; tq++) {
                    ଅଚଳ ggml_fp16_t * mp_row = (ଅଚଳ ggml_fp16_t *)((ଅଚଳ ଅକ୍ଷର *) mask->data + (iq1 + tq)*mask->nb[1] + (iq2%mask->ne[2])*mask->nb[2] + (iq3%mask->ne[3])*mask->nb[3]);
                    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା tk = 0; tk < kv_tile; tk++) {
                        mask32[tq * KV_TILE_SZ + tk] = slope * GGML_CPU_FP16_TO_FP32(mp_row[ic + tk]);
                        ଯଦି (mask32[tq * KV_TILE_SZ + tk] != -INFINITY) {
                            can_skip = ମିଥ୍ୟା;
                        }
                    }
                    // Pad remaining mask entries with -inf
                    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା tk = kv_tile; tk < KV_TILE_SZ; tk++) {
                        mask32[tq * KV_TILE_SZ + tk] = -INFINITY;
                    }
                }

                ଯଦି (can_skip) {
                    ଜାରି;
                }
            }

            // Pack K tile transposed: K_f32[dk][kv] so KV_TILE is contiguous (SIMD dim)
            // Zero-pad the last tile so the GEMM always operates on KV_TILE_SZ columns
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା tk = 0; tk < kv_tile; tk++) {
                ଅଚଳ ଅକ୍ଷର * k_data = (ଅଚଳ ଅକ୍ଷର *)k->data + (ic + tk)*nbk1 + ik2*nbk2 + ik3*nbk3;
                ଯଦି (kv_type == GGML_TYPE_F16) {
                    ଅଚଳ ggml_fp16_t * k_f16 = (ଅଚଳ ggml_fp16_t *)k_data;
                    ପାଇଁ (int64_t dk = 0; dk < DK; dk++) {
                        K_f32[dk * KV_TILE_SZ + tk] = GGML_CPU_FP16_TO_FP32(k_f16[dk]);
                    }
                } ଅନ୍ୟଥା {
                    ଅଚଳ ଦଶମିକ * k_f32_src = (ଅଚଳ ଦଶମିକ *)k_data;
                    ପାଇଁ (int64_t dk = 0; dk < DK; dk++) {
                        K_f32[dk * KV_TILE_SZ + tk] = k_f32_src[dk];
                    }
                }
            }
            memset(KQ, 0, Q_TILE_SZ * KV_TILE_SZ * ଆକାର(ଦଶମିକ));
            simd_gemm(KQ, (ଅଚଳ ଦଶମିକ *)Q_q, K_f32, Q_TILE_SZ, DK, KV_TILE_SZ);
            ggml_vec_scale_f32(Q_TILE_SZ * KV_TILE_SZ, KQ, scale);

            ଯଦି (logit_softcap != 0.0f) {
                ggml_vec_tanh_f32(Q_TILE_SZ * KV_TILE_SZ, KQ, KQ);
                ggml_vec_scale_f32(Q_TILE_SZ * KV_TILE_SZ, KQ, logit_softcap);
            }

            // Set padded KQ entries to -inf so softmax gives them zero weight
            ଯଦି (kv_tile < KV_TILE_SZ) {
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା tq = 0; tq < Q_TILE_SZ; tq++) {
                    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା tk = kv_tile; tk < KV_TILE_SZ; tk++) {
                        KQ[tq * KV_TILE_SZ + tk] = -INFINITY;
                    }
                }
            }

            ଯଦି (mask) {
                ggml_vec_add_f32(tile_rows * KV_TILE_SZ, KQ, KQ, mask32);
            }

            ତାର୍କିକ skip[Q_TILE_SZ] = {};

            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା tq = 0; tq < Q_TILE_SZ; tq++) {
                ଦଶମିକ * kq_row = KQ + tq * KV_TILE_SZ;

                ଦଶମିକ tile_max;
                ggml_vec_max_f32(KV_TILE_SZ, &tile_max, kq_row);

                ଯଦି (tile_max == -INFINITY) {
                    skip[tq] = ସତ୍ୟ;
                    ଜାରି;
                }

                ଅଚଳ ଦଶମିକ Mold = M[tq];
                ଅଚଳ ଦଶମିକ Mnew = fmaxf(Mold, tile_max);

                ଯଦି (Mnew > Mold) {
                    ଅଚଳ ଦଶମିକ ms = expf(Mold - Mnew);
                    ggml_vec_scale_f32(DV, VKQ32 + tq * DV, ms);
                    S[tq] *= ms;
                }
                M[tq] = Mnew;


                S[tq] += ggml_vec_soft_max_f32(KV_TILE_SZ, kq_row, kq_row, Mnew);
            }

            // V accumulation: VKQ32 += softmax(KQ) * V
            // Pack V tile to contiguous F32, zero-padded
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା tk = 0; tk < kv_tile; tk++) {
                ଅଚଳ ଅକ୍ଷର * v_data = (ଅଚଳ ଅକ୍ଷର *)v->data + (ic + tk)*nbv1 + iv2*nbv2 + iv3*nbv3;
                ଯଦି (kv_type == GGML_TYPE_F16) {
                    ggml_cpu_fp16_to_fp32((ଅଚଳ ggml_fp16_t *)v_data, V32 + tk * DV, DV);
                } ଅନ୍ୟଥା {
                    memcpy(V32 + tk * DV, v_data, DV * ଆକାର(ଦଶମିକ));
                }
            }
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା tq = 0; tq < Q_TILE_SZ; tq++) {
                ଯଦି (skip[tq]) {
                    memset(KQ + tq * KV_TILE_SZ, 0, KV_TILE_SZ * ଆକାର(ଦଶମିକ));
                }
            }
            simd_gemm(VKQ32, KQ, V32, Q_TILE_SZ, KV_TILE_SZ, DV);
        }

        // sinks (apply only to valid rows in the tile)
        ଯଦି (sinks) {
            ଅଚଳ ଦଶମିକ s = ((ଦଶମିକ *)((ଅକ୍ଷର *) sinks->data))[h];

            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା tq = 0; tq < tile_rows; tq++) {
                ଦଶମିକ ms = 1.0f;
                ଦଶମିକ vs = 1.0f;

                ଯଦି (s > M[tq]) {
                    ms = expf(M[tq] - s);
                    ggml_vec_scale_f32(DV, VKQ32 + tq * DV, ms);
                } ଅନ୍ୟଥା {
                    vs = expf(s - M[tq]);
                }

                S[tq] = S[tq] * ms + vs;
            }
        }

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା tq = 0; tq < tile_rows; tq++) {
            // V /= S
            ଅଚଳ ଦଶମିକ S_inv = S[tq] == 0.0f ? 0.0f : 1.0f / S[tq];
            ggml_vec_scale_f32(DV, VKQ32 + tq * DV, S_inv);

            // dst indices
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = iq1 + tq;
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i2 = iq2;
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i3 = iq3;

            // permute(0, 2, 1, 3)
            memcpy((ଅକ୍ଷର *) dst->data + (i3*ne2*ne1 + i2 + i1*ne1)*nb1, VKQ32 + tq * DV, nb1);
        }

        ir += tile_rows;
    }
}

// Reduction function: combines partial results across KV chunks
// Partials layout in wdata: [n_q_heads][n_chunks][2 + DV]
ସ୍ଥିର ଶୂନ୍ୟ ggml_flash_attn_ext_reduce_partials(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst,
        ଅଚଳ int64_t n_chunks,
        ଅଚଳ int64_t chunk_size) {

    ଅଚଳ ggml_tensor * q = dst->src[0];
    ଅଚଳ ggml_tensor * k = dst->src[1];
    ଅଚଳ ggml_tensor * v = dst->src[2];

    ଅଚଳ int64_t DK        = k->ne[0];
    ଅଚଳ int64_t DV        = v->ne[0];
    ଅଚଳ int64_t nek1      = k->ne[1];
    ଅଚଳ int64_t n_q_heads = q->ne[2];

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ int64_t wdata_per_thread = DK + 2*DV + CACHE_ଧାଡ଼ି_SIZE_F32;
    ଦଶମିକ *       thread_wdata     = (ଦଶମିକ *) params->wdata + ith * wdata_per_thread;

    ଅଚଳ int64_t partials_offset  = nth * (DK + 2*DV + CACHE_ଧାଡ଼ି_SIZE_F32);
    ଅଚଳ int64_t partial_size     = 2 + DV;
    ଅଚଳ ଦଶମିକ * partials_base    = (ଅଚଳ ଦଶମିକ *) params->wdata + partials_offset;

    // Output layout
    ଅଚଳ int64_t ne1 = dst->ne[1];
    ଅଚଳ int64_t ne2 = dst->ne[2];
    ଅଚଳ size_t  nb1 = dst->nb[1];

    // Each thread reduces a subset of query heads
    ପାଇଁ (int64_t q_head = ith; q_head < n_q_heads; q_head += nth) {
        ଦଶମିକ   M_final   = -INFINITY;
        ଦଶମିକ   S_final   = 0.0f;
        ଦଶମିକ * VKQ_final = thread_wdata;
        memset(VKQ_final, 0, DV * ଆକାର(ଦଶମିକ));

        // Combine partials from all chunks
        ପାଇଁ (int64_t chunk_idx = 0; chunk_idx < n_chunks; ++chunk_idx) {
            ଅଚଳ int64_t ic_start = chunk_idx * chunk_size;
            ଯଦି (ic_start >= nek1) ଜାରି;

            ଅଚଳ ଦଶମିକ * partial   = partials_base + (q_head * n_chunks + chunk_idx) * partial_size;
            ଅଚଳ ଦଶମିକ   M_chunk   = partial[0];
            ଅଚଳ ଦଶମିକ   S_chunk   = partial[1];
            ଅଚଳ ଦଶମିକ * VKQ_chunk = partial + 2;

            ଯଦି (S_chunk == 0.0f) ଜାରି;

            ଅଚଳ ଦଶମିକ M_new     = fmaxf(M_final, M_chunk);
            ଅଚଳ ଦଶମିକ scale_old = expf(M_final - M_new);
            ଅଚଳ ଦଶମିକ scale_new = expf(M_chunk - M_new);

            ପାଇଁ (int64_t d = 0; d < DV; ++d) {
                VKQ_final[d] = VKQ_final[d] * scale_old + VKQ_chunk[d] * scale_new;
            }
            S_final = S_final * scale_old + S_chunk * scale_new;
            M_final = M_new;
        }

        // Normalize and write to output
        ଯଦି (S_final != 0.0f) {
            ଅଚଳ ଦଶମିକ S_inv = 1.0f / S_final;
            ggml_vec_scale_f32(DV, VKQ_final, S_inv);
        }
        // iq1=0, iq3=0 for decode
        memcpy((ଅକ୍ଷର *) dst->data + (0*ne2*ne1 + q_head + 0*ne1)*nb1, VKQ_final, nb1);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_flash_attn_ext_f16(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * q     = dst->src[0];
    ଅଚଳ ggml_tensor * k     = dst->src[1];
    ଅଚଳ ggml_tensor * v     = dst->src[2];

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    ଅଚଳ int64_t DK = nek0;
    ଅଚଳ int64_t DV = nev0;
    ଅଚଳ int64_t N  = neq1;


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
    GGML_ASSERT(nb0 == ଆକାର(ଦଶମିକ));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    // When use_ref is set, force the vec-only reference implementation (no tiling, no KV-chunking)
    ଅଚଳ ତାର୍କିକ use_ref = params->use_ref;

    ଅଚଳ ତାର୍କିକ kv_is_f32_or_f16 = (k->type == GGML_TYPE_F32 || k->type == GGML_TYPE_F16);
    ଅଚଳ ତାର୍କିକ use_split_kv_path = !use_ref && (neq1 == 1 && neq3 == 1) && kv_is_f32_or_f16 && (k->type == v->type) && q->type == GGML_TYPE_F32 && nek1 >= 512;

    ଯଦି (use_split_kv_path) {
        ଅଚଳ int64_t chunk_size = (nek1 + nth - 1) / nth;

        // Partials buffer layout: [q_head][kv_chunk][M, S, VKQ]
        ଅଚଳ int64_t partial_size  = 2 + DV;
        ଦଶମିକ *       partials_base = (ଦଶମିକ *) params->wdata + nth * (DK + 2*DV + CACHE_ଧାଡ଼ି_SIZE_F32);

        ଅଚଳ int64_t ic_start = ith * chunk_size;
        ଅଚଳ int64_t ic_end   = std::min(ic_start + chunk_size, nek1);

        ଅଚଳ int64_t partial_stride = nth * partial_size;
        ଦଶମିକ *       chunk_partials = partials_base + ith * partial_size;

        ଯଦି (ic_start < nek1) {
            ପାଇଁ (int64_t q_head = 0; q_head < neq2; q_head++) {
                ggml_compute_forward_flash_attn_ext_f16_one_chunk(
                    params, dst, q_head, q_head + 1, ic_start, ic_end,
                    chunk_partials, partial_stride);
            }
        } ଅନ୍ୟଥା {
            ପାଇଁ (int64_t q_head = 0; q_head < neq2; q_head++) {
                ଦଶମିକ * q_partials = chunk_partials + q_head * partial_stride;
                q_partials[0] = -INFINITY;  // M
                q_partials[1] = 0.0f;       // S
            }
        }

        ggml_barrier(params->threadpool);
        ggml_flash_attn_ext_reduce_partials(params, dst, nth, chunk_size);
    } ଅନ୍ୟଥା {

        // total rows in q
        ଅଚଳ int64_t nr = neq1*neq2*neq3;

        // disable for NUMA
        ଅଚଳ ତାର୍କିକ disable_chunking = ggml_is_numa();

        // 4x chunks per thread
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth_scaled = nth * 4;
        int64_t chunk_size = (nr + nth_scaled - 1) / nth_scaled;
        int64_t nchunk     = (nr + chunk_size - 1) / chunk_size;

        ଯଦି (nth == 1 || nchunk < nth || disable_chunking) {
            nchunk = nth;
        }

        ଯଦି (ith == 0) {
            ggml_threadpool_chunk_set(params->threadpool, nth);
        }

        ggml_barrier(params->threadpool);

        ଅଚଳ int64_t dr = (nr + nchunk - 1) / nchunk;

        ସ୍ଥିର constexpr int64_t Q_TILE_SZ  = ggml_fa_tile_config::Q;
        ତାର୍କିକ use_tiled = !use_ref &&
                               (q->type == GGML_TYPE_F32 &&
                                kv_is_f32_or_f16 &&
                                k->type == v->type &&
                                neq1 >= Q_TILE_SZ);
#ଯଦି defined(GGML_SIMD) && !defined(__x86_64__) && !defined(_M_X64)
#ଯଦି defined(__ARM_FEATURE_SVE)
        ଅଚଳ int64_t f32_epr = svcntw();
#ଅନ୍ୟଥା
        ଅଚଳ int64_t f32_epr = GGML_F32_EPR;
#endif
        use_tiled &= (DV % f32_epr == 0);
#endif
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା current_chunk = ith;

        ଯେତେବେଳେ (current_chunk < nchunk) {
            ଅଚଳ int64_t ir0 = dr * current_chunk;
            ଅଚଳ int64_t ir1 = MIN(ir0 + dr, nr);

            ଯଦି (use_tiled) {
                ggml_compute_forward_flash_attn_ext_tiled(params, dst, ir0, ir1);
            } ଅନ୍ୟଥା {
                ggml_compute_forward_flash_attn_ext_f16_one_chunk(params, dst, ir0, ir1, 0, nek1, nullptr, 0);
            }

            current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
        }
    }
}

ଶୂନ୍ୟ ggml_compute_forward_flash_attn_ext(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {
    ବାଛ (dst->op_params[3]) {
        ସ୍ଥିତି GGML_PREC_DEFAULT:
        ସ୍ଥିତି GGML_PREC_F32:
            {
                // uses F32 accumulators
                ggml_compute_forward_flash_attn_ext_f16(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_flash_attn_back

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_flash_attn_back_f32(
        ଅଚଳ ggml_compute_params * params,
        ଅଚଳ ତାର୍କିକ masked,
              ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * q = dst->src[0];
    ଅଚଳ ggml_tensor * k = dst->src[1];
    ଅଚଳ ggml_tensor * v = dst->src[2];
    ଅଚଳ ggml_tensor * d = dst->src[3];

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

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ int64_t D = neq0;
    ଅଚଳ int64_t N = neq1;
    ଅଚଳ int64_t P = nek1 - N;
    ଅଚଳ int64_t M = P + N;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା Mup  = ggml_up(M, GGML_SOFT_MAX_UNROLL);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା mxDM = MAX(D, Mup);

    // GGML_ASSERT(ne0 == D);
    // GGML_ASSERT(ne1 == N);
    GGML_ASSERT(P >= 0);

    GGML_ASSERT(nbq0 == ଆକାର(ଦଶମିକ));
    GGML_ASSERT(nbk0 == ଆକାର(ଦଶମିକ));
    GGML_ASSERT(nbv0 == ଆକାର(ଦଶମିକ));

    GGML_ASSERT(neq0 == D);
    GGML_ASSERT(nek0 == D);
    GGML_ASSERT(nev1 == D);
    GGML_ASSERT(ned0 == D);

    GGML_ASSERT(neq1 == N);
    GGML_ASSERT(nek1 == N + P);
    GGML_ASSERT(nev1 == D);
    GGML_ASSERT(ned1 == N);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == ଆକାର(ଦଶମିକ));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    ଯଦି (ith == 0) {
        memset(dst->data, 0, nb0*ne0*ne1*ne2*ne3);
    }
    ggml_barrier(params->threadpool);

    ଅଚଳ int64_t elem_q = ggml_nelements(q);
    ଅଚଳ int64_t elem_k = ggml_nelements(k);

    ggml_type result_type = dst->type;
    GGML_ASSERT(ggml_blck_size(result_type) == 1);
    ଅଚଳ size_t tsize = ggml_type_size(result_type);

    ଅଚଳ size_t offs_q = 0;
    ଅଚଳ size_t offs_k = offs_q + GGML_PAD(elem_q * tsize, GGML_MEM_ALIGN);
    ଅଚଳ size_t offs_v = offs_k + GGML_PAD(elem_k * tsize, GGML_MEM_ALIGN);

    ଶୂନ୍ୟ * grad_q = (ଅକ୍ଷର *) dst->data;
    ଶୂନ୍ୟ * grad_k = (ଅକ୍ଷର *) dst->data + offs_k;
    ଶୂନ୍ୟ * grad_v = (ଅକ୍ଷର *) dst->data + offs_v;

    ଅଚଳ size_t nbgq1 = nb0*neq0;
    ଅଚଳ size_t nbgq2 = nb0*neq0*neq1;
    ଅଚଳ size_t nbgq3 = nb0*neq0*neq1*neq2;

    ଅଚଳ size_t nbgk1 = nb0*nek0;
    ଅଚଳ size_t nbgk2 = nb0*nek0*nek1;
    ଅଚଳ size_t nbgk3 = nb0*nek0*nek1*neq2;

    ଅଚଳ size_t nbgv1 = nb0*nev0;
    ଅଚଳ size_t nbgv2 = nb0*nev0*nev1;
    ଅଚଳ size_t nbgv3 = nb0*nev0*nev1*neq2;

    // parallelize by k rows using ggml_vec_dot_f32

    // total rows in k
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr = nek2*nek3;

    // rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ଅଚଳ ଦଶମିକ scale = 1.0f/sqrtf(D);

    //printf("P=%d N=%d D=%d ir0=%d ir1=%d scale = %f\n", P, N, D, ir0, ir1, scale);

    // how often k2 (and v2) is repeated in q2
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrep = neq2/nek2;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir = ir0; ir < ir1; ++ir) {
        // q indices
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ik3 = ir/(nek2);
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ik2 = ir - ik3*nek2;

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iq3 = ik3;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା id3 = ik3;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iv3 = ik3;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iv2 = ik2;

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା irep = 0; irep < nrep; ++irep) {
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iq2 = ik2 + irep*nek2;
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା id2 = iq2;

            // (ik2 + irep*nek2) % nek2 == ik2
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା iq1 = 0; iq1 < neq1; ++iq1) {
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା id1 = iq1;

                // not sure about CACHE_LINE_SIZE_F32..
                // - maybe it must not be multiplied by 2 and excluded from .. in SM 1*(..) offset?
                ଦଶମିକ * S  = (ଦଶମିକ *) params->wdata + ith*2*(mxDM + CACHE_ଧାଡ଼ି_SIZE_F32) + 0*(mxDM+CACHE_ଧାଡ଼ି_SIZE_F32);
                ଦଶମିକ * SM = (ଦଶମିକ *) params->wdata + ith*2*(mxDM + CACHE_ଧାଡ଼ି_SIZE_F32) + 1*(mxDM+CACHE_ଧାଡ଼ି_SIZE_F32);

                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = M; i < Mup; ++i) {
                    S[i] = -INFINITY;
                }

                ଅଚଳ int64_t masked_begin = masked ? (P + iq1 + 1) : M;
                ପାଇଁ (int64_t ic = 0; ic < masked_begin; ++ic) {
                    // k indices
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ik1 = ic;

                    // S indices
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = ik1;

                    ggml_vec_dot_f32(neq0,
                            S + i1, 0,
                            (ଦଶମିକ *) ((ଅକ୍ଷର *) k->data + (ik1*nbk1 + ik2*nbk2 + ik3*nbk3)), 0,
                            (ଦଶମିକ *) ((ଅକ୍ଷର *) q->data + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3)), 0, 1);
                }

                // scale
                ggml_vec_scale_f32(masked_begin, S, scale);

                ପାଇଁ (int64_t i = masked_begin; i < M; i++) {
                    S[i] = -INFINITY;
                }

                // softmax
                // exclude known -INF S[..] values from max and loop
                // dont forget to set their SM values to zero
                {
                    ଦଶମିକ max = -INFINITY;
                    ggml_vec_max_f32(masked_begin, &max, S);

                    ggml_float sum = 0.0;
                    {
#ifdef GGML_SOFT_MAX_ACCELERATE
                        max = -max;
                        vDSP_vsadd(SM, 1, &max, SM, 1, Mup);
                        vvexpf(SM, SM, &Mup);
                        ggml_vec_sum_f32(Mup, &sum, SM);
#ଅନ୍ୟଥା
                        sum = ggml_vec_soft_max_f32(Mup, SM, S, max);
#endif
                    }

                    ପ୍ରତିଜ୍ଞା(sum > 0.0);

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
                ପାଇଁ (int64_t ic = 0; ic < D; ++ic) {
                    ggml_vec_mad_f32(masked_begin,
                            S,
                             (ଦଶମିକ *) ((ଅକ୍ଷର *) v->data + (          ic*nbv1  + iv2*nbv2 + iv3*nbv3)),
                            *(ଦଶମିକ *) ((ଅକ୍ଷର *) d->data + (ic*nbd0 + id1*nbd1 + id2*nbd2 + id3*nbd3)));
                }

                // S = SM * (S - dot(SM, S))
                ଦଶମିକ dot_SM_gradSM = 0;
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
                ପାଇଁ (int64_t ic = 0; ic < masked_begin; ++ic) {
                    ggml_vec_mad_f32(D,
                            (ଦଶମିକ *) ((ଅକ୍ଷର *) grad_q  + (iq1*nbgq1 + iq2*nbgq2  + iq3*nbgq3)),
                            (ଦଶମିକ *) ((ଅକ୍ଷର *) k->data + (ic*nbk1   + ik2*nbk2   + ik3*nbk3)),
                            S[ic]);
                }

                // grad[k][:D,:M,iq2,iq3] += S.T @ qcur
                // for ic:
                //  grad[k][:D,ic,iq2,iq3] += S.T[0,ic] * qcur[:D,0]
                //  grad[k][:D,ic,iq2,iq3] += S[ic]     * qcur[:D,0]
                // exclude known zero S[..] values from loop
                ପାଇଁ (int64_t ic = 0; ic < masked_begin; ++ic) {
                    ggml_vec_mad_f32(D,
                            (ଦଶମିକ *) ((ଅକ୍ଷର *) grad_k  + (ic*nbgk1  + ik2*nbgk2  + ik3*nbgk3)),
                            (ଦଶମିକ *) ((ଅକ୍ଷର *) q->data + (iq1*nbq1  + iq2*nbq2   + iq3*nbq3)),
                            S[ic]);
                }

                // grad[v][:M,:D,iv2,iv3] += d[:D,id1,id2,id3].T       @ SM
                // for ic:
                //  grad[v][:M,ic,iv2,iv3] += d[:D,id1,id2,id3].T[0,ic] * SM[:M]
                //  grad[v][:M,ic,iv2,iv3] += d[ic,id1,id2,id3]         * SM[:M]
                // exclude known zero SM[..] values from mad
                ପାଇଁ (int64_t ic = 0; ic < D; ++ic) {
                    ggml_vec_mad_f32(masked_begin,
                            (ଦଶମିକ *) ((ଅକ୍ଷର *) grad_v   + (          ic*nbgv1 + iv2*nbgv2 + iv3*nbgv3)),
                            SM,
                            *(ଦଶମିକ *) ((ଅକ୍ଷର *) d->data + (ic*nbd0 + id1*nbd1 + id2*nbd2  + id3*nbd3)));
                }
            }
        }
    }
}

ଶୂନ୍ୟ ggml_compute_forward_flash_attn_back(
        ଅଚଳ ggml_compute_params * params,
        ଅଚଳ ତାର୍କିକ masked,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * q = dst->src[0];

    ବାଛ (q->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_flash_attn_back_f32(params, masked, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_ssm_conv

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_ssm_conv_f32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {
    ଅଚଳ ggml_tensor * src0 = dst->src[0]; // conv_x
    ଅଚଳ ggml_tensor * src1 = dst->src[1]; // conv1d.weight

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nc  = src1->ne[0]; // d_conv
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncs = src0->ne[0]; // d_conv - 1 + n_t
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr  = src0->ne[1]; // d_inner
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_t =  dst->ne[1]; // tokens per sequence
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_s =  dst->ne[2]; // number of sequences in the batch

    GGML_ASSERT( dst->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == ଆକାର(ଦଶମିକ));
    GGML_ASSERT(src1->nb[0] == ଆକାର(ଦଶମିକ));
    GGML_ASSERT(src0->nb[1] == src0->ne[0]*ଆକାର(ଦଶମିକ));

    // rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir  = ir1 - ir0;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i3 = 0; i3 < n_s; ++i3) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i2 = 0; i2 < n_t; ++i2) {
            // {d_conv - 1 + n_t, d_inner, n_seqs}
            // sliding window
            ଅଚଳ ଦଶମିକ * s = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) src0->data + ir0*(src0->nb[1]) + i2*(src0->nb[0]) + i3*(src0->nb[2])); // {d_conv, d_inner, n_s}
            ଅଚଳ ଦଶମିକ * c = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) src1->data + ir0*(src1->nb[1])); // {d_conv, d_inner}
            ଦଶମିକ * x = (ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data + ir0*(dst->nb[0]) + i2*(dst->nb[1]) + i3*(dst->nb[2])); // {d_inner, n_t, n_s}

            // TODO: transpose the output for smaller strides for big batches?
            // d_inner
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = 0; i1 < ir; ++i1) {
                // rowwise dot product
                // NOTE: not using ggml_vec_dot_f32, because its sum is in double precision
                ଦଶମିକ sumf = 0.0f;

                // d_conv
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0 = 0; i0 < nc; ++i0) {
                    sumf += s[i0 + i1*ncs] * c[i0 + i1*nc];
                }
                x[i1] = sumf;
            }
        }
    }
}

ଶୂନ୍ୟ ggml_compute_forward_ssm_conv(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {
    ବାଛ (dst->src[0]->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_ssm_conv_f32(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_ssm_scan

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_ssm_scan_f32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {
    ଅଚଳ ggml_tensor * src0 = dst->src[0]; // s  {d_state, dim, n_head, n_seqs+}
    ଅଚଳ ggml_tensor * src1 = dst->src[1]; // x  {dim, n_head, n_seq_tokens, n_seqs}
    ଅଚଳ ggml_tensor * src2 = dst->src[2]; // dt {n_head, n_seq_tokens, n_seqs}
    ଅଚଳ ggml_tensor * src3 = dst->src[3]; // A  {d_state, n_head} or {1, n_head}
    ଅଚଳ ggml_tensor * src4 = dst->src[4]; // B  {d_state, n_group, n_seq_tokens, n_seqs}
    ଅଚଳ ggml_tensor * src5 = dst->src[5]; // C  {d_state, n_group, n_seq_tokens, n_seqs}
    ଅଚଳ ggml_tensor * src6 = dst->src[6]; // ids {n_seqs}

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ int64_t nc = src0->ne[0]; // d_state
    ଅଚଳ int64_t nr = src0->ne[1]; // dim
    ଅଚଳ int64_t nh = src1->ne[1]; // n_head
    ଅଚଳ int64_t ng = src4->ne[1];
    ଅଚଳ int64_t nt = src1->ne[2]; // number of tokens per sequence
    ଅଚଳ int64_t ns = src1->ne[3]; // number of sequences in the batch
    ଅଚଳ int64_t K  = ggml_get_op_params_i32(dst, 0);

    // can't use ggml_nbytes because src1 is not necessarily contiguous
    ଅଚଳ int64_t s_off = ggml_nelements(src1) * ggml_element_size(src1);

    GGML_ASSERT(K >= 1);
    GGML_ASSERT(ggml_nelements(src1) + K*nc*nr*nh*ns == ggml_nelements(dst));
    GGML_ASSERT(src0->nb[0] == ଆକାର(ଦଶମିକ));
    GGML_ASSERT(src1->nb[0] == ଆକାର(ଦଶମିକ));
    GGML_ASSERT(src2->nb[0] == ଆକାର(ଦଶମିକ));
    GGML_ASSERT(src3->nb[0] == ଆକାର(ଦଶମିକ));
    GGML_ASSERT(src4->nb[0] == ଆକାର(ଦଶମିକ));
    GGML_ASSERT(src5->nb[0] == ଆକାର(ଦଶମିକ));
    GGML_ASSERT(src6->nb[0] == ଆକାର(int32_t));
    GGML_ASSERT(nh % ng == 0);
    GGML_ASSERT(src3->ne[0] == 1 || K == 1);

    // heads per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dh = (nh + nth - 1)/nth;

    // head range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ih0 = dh*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ih1 = MIN(ih0 + dh, nh);

    ଅଚଳ int32_t * ids = (ଅଚଳ int32_t *) src6->data;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i3 = 0; i3 < ns; ++i3) {
        ଅଚଳ ଦଶମିକ * s0 = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) src0->data + ids[i3]*(src0->nb[3])); // {d_state, dim, nh, ns}
              ଦଶମିକ * s  = (      ଦଶମିକ *) ((      ଅକ୍ଷର *) dst->data  + i3*(src0->nb[3]) + s_off); // {d_state, dim, nh, ns}

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i2 = 0; i2 < nt; ++i2) {
            ଅଚଳ ଦଶମିକ * x  = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) src1->data + i2*(src1->nb[2]) + i3*(src1->nb[3])); // {dim, nh, nt, ns}
            ଅଚଳ ଦଶମିକ * dt = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) src2->data + i2*(src2->nb[1]) + i3*(src2->nb[2])); // {nh, nt, ns}
            ଅଚଳ ଦଶମିକ * A  = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) src3->data); // {d_state, nh} or {1, nh}
            ଅଚଳ ଦଶମିକ * B  = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) src4->data + i2*(src4->nb[2]) + i3*(src4->nb[3])); // {d_state, ng, nt, ns}
            ଅଚଳ ଦଶମିକ * C  = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) src5->data + i2*(src5->nb[2]) + i3*(src5->nb[3])); // {d_state, ng, nt, ns}
                  ଦଶମିକ * y  = (      ଦଶମିକ *) ((      ଅକ୍ଷର *) dst->data + i2*(nh*nr*ଆକାର(ଦଶମିକ)) + i3*(nt*nh*nr*ଆକାର(ଦଶମିକ))); // {dim, nh, nt, ns}

            ଯଦି (src3->ne[0] == 1) {
                // Mamba-2 has a scalar decay factor per head; dA can be outside the state-wise loop

                // n_head
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା h = ih0; h < ih1; ++h) {
                    // ref: https://github.com/state-spaces/mamba/blob/62db608da60f6fc790b8ed9f4b3225e95ca15fde/mamba_ssm/ops/triton/softplus.py#L16
                    ଅଚଳ ଦଶମିକ dt_soft_plus = ggml_compute_softplus_f32(dt[h]);
                    ଅଚଳ ଦଶମିକ dA = expf(dt_soft_plus * A[h]);
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା g = h / (nh / ng); // repeat_interleave

                    // dim
                    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = 0; i1 < nr; ++i1) {
                        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ii = i1 + h*nr;
                        ଅଚଳ ଦଶମିକ x_dt = x[ii] * dt_soft_plus;
                        ଦଶମିକ sumf = 0.0f;
#ଯଦି defined(GGML_SIMD)
    #ଯଦି defined(__ARM_FEATURE_SVE)
                        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_f32_epr = svcntw();
                        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_f32_step = 1 * ggml_f32_epr;

                        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା np = (nc & ~(ggml_f32_step - 1));

                        GGML_F32_VEC sum = GGML_F32_VEC_ZERO;

                        GGML_F32_VEC adA = GGML_F32_VEC_SET1(dA);
                        GGML_F32_VEC axdt = GGML_F32_VEC_SET1(x_dt);

                        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < np; i += ggml_f32_step) {
                            // TODO: maybe unroll more?
                            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < 1; j++) {
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
    #ନଚେତଯଦି defined(__riscv_v_intrinsic)
                        // todo: RVV implementation
                        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା np = 0;
    #ଅନ୍ୟଥା
                        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା np = (nc & ~(GGML_F32_STEP - 1));

                        GGML_F32_VEC sum[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };

                        GGML_F32_VEC adA = GGML_F32_VEC_SET1(dA);
                        GGML_F32_VEC axdt = GGML_F32_VEC_SET1(x_dt);

                        GGML_F32_VEC ax[GGML_F32_ARR];
                        GGML_F32_VEC ay[GGML_F32_ARR];
                        GGML_F32_VEC az[GGML_F32_ARR];

                        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < np; i += GGML_F32_STEP) {
                            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < GGML_F32_ARR; j++) {
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
#ଅନ୍ୟଥା
                        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା np = 0;
#endif
                        // d_state
                        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0 = np; i0 < nc; ++i0) {
                            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = i0 + ii*nc;
                            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ig = i0 + g*nc;
                            // state = prev_state * dA + dB * x
                            ଅଚଳ ଦଶମିକ state = (s0[i] * dA) + (B[ig] * x_dt);
                            // y = rowwise_dotprod(state, C)
                            sumf += state * C[ig];
                            s[i] = state;
                        }
                        y[ii] = sumf;
                    }
                }
            } ଅନ୍ୟଥା {
                // Mamba-1 has an element-wise decay factor for the states

                // n_head
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା h = ih0; h < ih1; ++h) {
                    // ref: https://github.com/state-spaces/mamba/blob/62db608da60f6fc790b8ed9f4b3225e95ca15fde/mamba_ssm/ops/triton/softplus.py#L16
                    ଅଚଳ ଦଶମିକ dt_soft_plus = ggml_compute_softplus_f32(dt[h]);
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା g = h / (nh / ng); // repeat_interleave

                    // dim
                    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = 0; i1 < nr; ++i1) {
                        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ii = i1 + h*nr;
                        ଅଚଳ ଦଶମିକ x_dt = x[ii] * dt_soft_plus;
#ଯଦି defined(__ARM_FEATURE_SVE)
                        svfloat32_t vx_dt = GGML_F32_VEC_SET1(x_dt);
                        svfloat32_t vdt_soft_plus = GGML_F32_VEC_SET1(dt_soft_plus);
                        svfloat32_t r1_vector = GGML_F32_VEC_ZERO;

                        // d_state
                        // TODO: what happens when (d_state % svcntw()) != 0?
                        ପାଇଁ (int64_t k = 0; k < nc; k += svcntw()) {
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
#ଅନ୍ୟଥା
                        ଦଶମିକ sumf = 0.0f;
                        // NOTE: can't really use GGML_SIMD here because d_state is usually 16
                        //       and also because expf is used within the loop.
                        // d_state
                        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0 = 0; i0 < nc; ++i0) {
                            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = i0 + ii*nc;
                            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ig = i0 + g*nc;
                            // state = prev_state * dA + dB * x
                            ଅଚଳ ଦଶମିକ state = (s0[i] * expf(dt_soft_plus * A[i0 + h*nc])) + (B[ig] * x_dt);
                            // y = rowwise_dotprod(state, C)
                            sumf += state * C[ig];
                            s[i] = state;
                        }
                        y[ii] = sumf;
#endif
                    }
                }
            }
            ଅଚଳ int64_t slot = nt - 1 - i2;
            ଯଦି (K > 1 && slot > 0 && slot < K) {
                ଦଶମିକ * s_snapshot = (ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data + s_off + (slot*ns + i3)*(src0->nb[3]));
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା h = ih0; h < ih1; ++h) {
                    memcpy((ଅକ୍ଷର *) s_snapshot + h*src0->nb[2], (ଅକ୍ଷର *) s + h*src0->nb[2], src0->nb[2]);
                }
            }
            // use the output as the source when it's not the first token-wise iteration
            s0 = s;
        }
    }
}

ଶୂନ୍ୟ ggml_compute_forward_ssm_scan(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {
    ବାଛ (dst->src[0]->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_ssm_scan_f32(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_win_part

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_win_part_f32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {
    GGML_UNUSED(params);

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)

    ଅଚଳ int32_t nep0 = ((ଅଚଳ int32_t *)(dst->op_params))[0];
    ଅଚଳ int32_t nep1 = ((ଅଚଳ int32_t *)(dst->op_params))[1];
    ଅଚଳ int32_t w    = ((ଅଚଳ int32_t *)(dst->op_params))[2];

    ପ୍ରତିଜ୍ଞା(ne00 == ne0);
    ପ୍ରତିଜ୍ଞା(ne3  == nep0*nep1);

    // TODO: optimize / multi-thread
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା py = 0; py < nep1; ++py) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା px = 0; px < nep0; ++px) {
            ଅଚଳ int64_t i3 = py*nep0 + px;
            ପାଇଁ (int64_t i2 = 0; i2 < ne2; ++i2) {
                ପାଇଁ (int64_t i1 = 0; i1 < ne1; ++i1) {
                    ପାଇଁ (int64_t i0 = 0; i0 < ne0; ++i0) {
                        ଅଚଳ int64_t i02 = py*w + i2;
                        ଅଚଳ int64_t i01 = px*w + i1;
                        ଅଚଳ int64_t i00 = i0;

                        ଅଚଳ int64_t i = i3*ne2*ne1*ne0 + i2*ne1*ne0    + i1*ne0   + i0;
                        ଅଚଳ int64_t j =                  i02*ne01*ne00 + i01*ne00 + i00;

                        ଯଦି (py*w + i2 >= ne02 || px*w + i1 >= ne01) {
                            ((ଦଶମିକ *) dst->data)[i] = 0.0f;
                        } ଅନ୍ୟଥା {
                            ((ଦଶମିକ *) dst->data)[i] = ((ଦଶମିକ *) src0->data)[j];
                        }
                    }
                }
            }
        }
    }
}

ଶୂନ୍ୟ ggml_compute_forward_win_part(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_win_part_f32(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_win_unpart

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_win_unpart_f32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {
    GGML_UNUSED(params);

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)

    ଅଚଳ int32_t w = ((ଅଚଳ int32_t *)(dst->op_params))[0];

    // padding
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା px = (w - ne1%w)%w;
    //const int py = (w - ne2%w)%w;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା npx = (px + ne1)/w;
    //const int npy = (py + ne2)/w;

    ପ୍ରତିଜ୍ଞା(ne0 == ne00);

    // TODO: optimize / multi-thread
    ପାଇଁ (int64_t i2 = 0; i2 < ne2; ++i2) {
        ପାଇଁ (int64_t i1 = 0; i1 < ne1; ++i1) {
            ପାଇଁ (int64_t i0 = 0; i0 < ne0; ++i0) {
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ip2 = i2/w;
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ip1 = i1/w;

                ଅଚଳ int64_t i02 = i2%w;
                ଅଚଳ int64_t i01 = i1%w;
                ଅଚଳ int64_t i00 = i0;

                ଅଚଳ int64_t i = (ip2*npx + ip1)*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00 + i00;
                ଅଚଳ int64_t j =                                  i2*ne1*ne0    + i1*ne0   + i0;

                ((ଦଶମିକ *) dst->data)[j] = ((ଦଶମିକ *) src0->data)[i];
            }
        }
    }
}

ଶୂନ୍ୟ ggml_compute_forward_win_unpart(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_win_unpart_f32(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

//ggml_compute_forward_unary

ଶୂନ୍ୟ ggml_compute_forward_unary(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_unary_op op = ggml_get_unary_op(dst);

    ବାଛ (op) {
        ସ୍ଥିତି GGML_UNARY_OP_ABS:
            {
                ggml_compute_forward_abs(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_UNARY_OP_SGN:
            {
                ggml_compute_forward_sgn(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_UNARY_OP_NEG:
            {
                ggml_compute_forward_neg(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_UNARY_OP_STEP:
            {
                ggml_compute_forward_step(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_UNARY_OP_TANH:
            {
                ggml_compute_forward_tanh(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_UNARY_OP_ELU:
            {
                ggml_compute_forward_elu(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_UNARY_OP_RELU:
            {
                ggml_compute_forward_relu(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_UNARY_OP_SIGMOID:
            {
                ggml_compute_forward_sigmoid(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_UNARY_OP_GELU:
            {
                ggml_compute_forward_gelu(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_UNARY_OP_GELU_ERF:
            {
                ggml_compute_forward_gelu_erf(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_UNARY_OP_GELU_QUICK:
            {
                ggml_compute_forward_gelu_quick(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_UNARY_OP_SILU:
            {
                ggml_compute_forward_silu(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_UNARY_OP_HARDSWISH:
            {
                ggml_compute_forward_hardswish(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_UNARY_OP_HARDSIGMOID:
            {
                ggml_compute_forward_hardsigmoid(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_UNARY_OP_EXP:
            {
                ggml_compute_forward_exp(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_UNARY_OP_FLOOR:
            {
                ggml_compute_forward_floor(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_UNARY_OP_CEIL:
            {
                ggml_compute_forward_ceil(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_UNARY_OP_ROUND:
            {
                ggml_compute_forward_round(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_UNARY_OP_TRUNC:
            {
                ggml_compute_forward_trunc(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_UNARY_OP_XIELU:
            {
                ggml_compute_forward_xielu(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_UNARY_OP_EXPM1:
            {
                ggml_compute_forward_expm1(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_UNARY_OP_SOFTPLUS:
            {
                ggml_compute_forward_softplus(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

//ggml_compute_forward_glu

ଶୂନ୍ୟ ggml_compute_forward_glu(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_glu_op op = ggml_get_glu_op(dst);

    ବାଛ (op) {
        ସ୍ଥିତି GGML_GLU_OP_REGLU:
            {
                ggml_compute_forward_reglu(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_GLU_OP_GEGLU:
            {
                ggml_compute_forward_geglu(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_GLU_OP_SWIGLU:
            {
                ggml_compute_forward_swiglu(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_GLU_OP_SWIGLU_OAI:
            {
                ggml_compute_forward_swiglu_oai(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_GLU_OP_GEGLU_ERF:
            {
                ggml_compute_forward_geglu_erf(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_GLU_OP_GEGLU_QUICK:
            {
                ggml_compute_forward_geglu_quick(params, dst);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_GLU_OP_SWIGLU_CLAMP:
            {
                ggml_compute_forward_swiglu_clamp(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_get_rel_pos

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_get_rel_pos_f16(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {
    GGML_UNUSED(params);

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L292-L322

    GGML_TENSOR_UNARY_OP_LOCALS

    ଅଚଳ int64_t w = ne1;

    ggml_fp16_t * src0_data = (ggml_fp16_t *) src0->data;
    ggml_fp16_t * dst_data  = (ggml_fp16_t *) dst->data;

    ପାଇଁ (int64_t i2 = 0; i2 < ne2; ++i2) {
        ପାଇଁ (int64_t i1 = 0; i1 < ne1; ++i1) {
            ଅଚଳ int64_t pos = (w - i1 - 1) + i2;
            ପାଇଁ (int64_t i0 = 0; i0 < ne0; ++i0) {
                dst_data[i2*ne1*ne0 + i1*ne0 + i0] = src0_data[pos*ne00 + i0];
            }
        }
    }
}

ଶୂନ୍ୟ ggml_compute_forward_get_rel_pos(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F16:
        ସ୍ଥିତି GGML_TYPE_BF16:
            {
                ggml_compute_forward_get_rel_pos_f16(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_add_rel_pos

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_add_rel_pos_f32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];
    ଅଚଳ ggml_tensor * src2 = dst->src[2];

    ଅଚଳ ତାର୍କିକ inplace = (ତାର୍କିକ) ((int32_t *) dst->op_params)[0];
    ଯଦି (!inplace) {
        ଯଦି (params->ith == 0) {
            memcpy((ଅକ୍ଷର *) dst->data, (ଅକ୍ଷର *) src0->data, ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }
    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L357-L359

    ଦଶମିକ * src1_data = (ଦଶମିକ *) src1->data;
    ଦଶମିକ * src2_data = (ଦଶମିକ *) src2->data;
    ଦଶମିକ * dst_data  = (ଦଶମିକ *) dst->data;

    ଅଚଳ int64_t ne10 = src1->ne[0];
    ଅଚଳ int64_t ne11 = src1->ne[1];
    ଅଚଳ int64_t ne12 = src1->ne[2];
    ଅଚଳ int64_t ne13 = src1->ne[3];

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    // total patches in dst
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା np = ne13;

    // patches per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dp = (np + nth - 1)/nth;

    // patch range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ip0 = dp*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ip1 = MIN(ip0 + dp, np);

    ପାଇଁ (int64_t i13 = ip0; i13 < ip1; ++i13) {
        ପାଇଁ (int64_t i12 = 0; i12 < ne12; ++i12) {
            ପାଇଁ (int64_t i11 = 0; i11 < ne11; ++i11) {
                ଅଚଳ int64_t jp1 = i13*ne12*ne11*ne10 + i12*ne11*ne10 + i11*ne10;
                ପାଇଁ (int64_t i10 = 0; i10 < ne10; ++i10) {
                    ଅଚଳ int64_t jp0  = jp1 + i10;
                    ଅଚଳ ଦଶମିକ src1_e = src1_data[jp0];
                    ଅଚଳ ଦଶମିକ src2_e = src2_data[jp0];

                    ଅଚଳ int64_t jdh = jp0 * ne10;
                    ଅଚଳ int64_t jdw = jdh - (ne10 - 1) * i10;

                    ପାଇଁ (int64_t j = 0; j < ne10; ++j) {
                        dst_data[jdh + j     ] += src2_e;
                        dst_data[jdw + j*ne10] += src1_e;
                    }
                }
            }
        }
    }
}

ଶୂନ୍ୟ ggml_compute_forward_add_rel_pos(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_add_rel_pos_f32(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_rwkv_wkv6

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_rwkv_wkv6_f32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {
    ଅଚଳ int64_t T = dst->src[1]->ne[2];
    ଅଚଳ int64_t C = dst->ne[0];
    ଅଚଳ int64_t HEADS = dst->src[1]->ne[1];
    ଅଚଳ int64_t n_seqs = dst->src[5]->ne[1];
    ଅଚଳ int64_t head_size = C / HEADS;

    ଦଶମିକ * dst_data = (ଦଶମିକ *) dst->data;
    ଦଶମିକ * state = ((ଦଶମିକ *) dst->data) + C * T;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା h_start =  (HEADS * (ith    )) / nth;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା h_end   = ((HEADS * (ith + 1)) / nth < HEADS) ?
                         (HEADS * (ith + 1)) / nth : HEADS;

    ଦଶମିକ * k =          (ଦଶମିକ *) dst->src[0]->data;
    ଦଶମିକ * v =          (ଦଶମିକ *) dst->src[1]->data;
    ଦଶମିକ * r =          (ଦଶମିକ *) dst->src[2]->data;
    ଦଶମିକ * time_faaaa = (ଦଶମିକ *) dst->src[3]->data;
    ଦଶମିକ * time_decay = (ଦଶମିକ *) dst->src[4]->data;

    size_t t_stride = HEADS * head_size; // Same to C

    size_t h_stride = C / HEADS;
    GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
    size_t h_stride_2d = head_size * head_size;

    ଯଦି (ith == 0) {
        memset(dst_data, 0, T * C * ଆକାର(ଦଶମିକ));
    }
    ggml_barrier(params->threadpool);


    #ଯଦି defined(__AVX__) && !defined(__AVX512F__)
        #define GGML_F32X GGML_F32x8
        #define GGML_F32X_SET1 GGML_F32x8_SET1
        #define GGML_F32X_LOAD GGML_F32x8_LOAD
        #define GGML_F32X_STORE GGML_F32x8_STORE
        #define GGML_F32X_MUL GGML_F32x8_MUL
        #define GGML_F32X_FMA GGML_F32x8_FMA
        #define WKV_VECTOR_SIZE 8
    #ନଚେତଯଦି defined(__AVX512F__)
        #define GGML_F32X GGML_F32x16
        #define GGML_F32X_SET1 GGML_F32x16_SET1
        #define GGML_F32X_LOAD GGML_F32x16_LOAD
        #define GGML_F32X_STORE GGML_F32x16_STORE
        #define GGML_F32X_MUL GGML_F32x16_MUL
        #define GGML_F32X_FMA GGML_F32x16_FMA
        #define WKV_VECTOR_SIZE 16
    #ନଚେତଯଦି defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
        #define GGML_F32X GGML_F32xt
        #define GGML_F32X_SET1 GGML_F32xt_SET1
        #define GGML_F32X_LOAD GGML_F32xt_LOAD
        #define GGML_F32X_STORE GGML_F32xt_STORE
        #define GGML_F32X_MUL GGML_F32xt_MUL
        #define GGML_F32X_FMA GGML_F32xt_FMA
        #define WKV_VECTOR_SIZE 8
    #ନଚେତଯଦି defined(__ARM_NEON) && defined(__aarch64__)
        #define GGML_F32X GGML_F32x4
        #define GGML_F32X_SET1 GGML_F32x4_SET1
        #define GGML_F32X_LOAD GGML_F32x4_LOAD
        #define GGML_F32X_STORE GGML_F32x4_STORE
        #define GGML_F32X_MUL GGML_F32x4_MUL
        #define GGML_F32X_FMA GGML_F32x4_FMA
        #define WKV_VECTOR_SIZE 4
    #endif

    #ifdef WKV_VECTOR_SIZE
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା wkv_vector_size;
        #ଯଦି defined(__ARM_FEATURE_SVE)
            wkv_vector_size = svcntw();
        #ଅନ୍ୟଥା
            wkv_vector_size = WKV_VECTOR_SIZE;
        #endif
        ଅଚଳ int64_t vec_ଗଣନା = head_size / wkv_vector_size;

        ପାଇଁ (int64_t t = 0; t < T; t++) {
            size_t t_offset = t * t_stride;
            size_t state_offset = head_size * C * (t / (T / n_seqs));
            ଦଶମିକ * state_cur = state + state_offset;
            ଦଶମିକ * state_prev = t % (T / n_seqs) ? state_cur : (ଦଶମିକ*)dst->src[5]->data + state_offset;

            ପାଇଁ (int64_t h = h_start; h < h_end; h++) {
                size_t h_offset = h * h_stride;
                size_t t_h_offset = t_offset + h_offset;
                size_t h_2d_offset = h * h_stride_2d;

                ପାଇଁ (int64_t i = 0; i < head_size; i++) {
                    size_t t_h_i_offset = t_h_offset + i;
                    size_t h_i_offset = h_offset + i;
                    size_t h_2d_i_offset = h_2d_offset + i * h_stride;

                    ଦଶମିକ k_val = k[t_h_i_offset];
                    ଦଶମିକ r_val = r[t_h_i_offset];
                    ଦଶମିକ time_faaaa_val = time_faaaa[h_i_offset];
                    ଦଶମିକ time_decay_val = time_decay[t_h_i_offset];

                    // Broadcast scalar values to vectors
                    GGML_F32X k_vec = GGML_F32X_SET1(k_val);
                    GGML_F32X r_vec = GGML_F32X_SET1(r_val);
                    GGML_F32X time_faaaa_vec = GGML_F32X_SET1(time_faaaa_val);
                    GGML_F32X time_decay_vec = GGML_F32X_SET1(time_decay_val);

                    ପାଇଁ (int64_t j = 0; j < vec_ଗଣନା; j++) {
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
                    ପାଇଁ (int64_t j = vec_ଗଣନା * wkv_vector_size; j < head_size; j++) {
                        size_t t_h_j_offset = t_h_offset + j;
                        size_t h_2d_i_j_offset = h_2d_i_offset + j;
                        ଦଶମିକ v_val = v[t_h_j_offset];
                        ଦଶମିକ kv_val = v_val * k_val;
                        ଦଶମିକ prev_state_val = state_prev[h_2d_i_j_offset];
                        ଦଶମିକ temp_val = kv_val * time_faaaa_val + prev_state_val;
                        dst_data[t_h_j_offset] += temp_val * r_val;
                        state_cur[h_2d_i_j_offset] = prev_state_val * time_decay_val + kv_val;
                    }
                }
            }
        }

    #ଅନ୍ୟଥା
        // basically fused operations:
        // dst = r @ (time_faaaa * (k @ v) + state),
        // state = time_decay * state + (k @ v),
        // recursive through each token
        ପାଇଁ (int64_t t = 0; t < T; t++) {
            size_t t_offset = t * t_stride;
            size_t state_offset = head_size * C * (t / (T / n_seqs));
            ଦଶମିକ * state_cur = state + state_offset;
            ଦଶମିକ * state_prev = t % (T / n_seqs) ? state_cur : (ଦଶମିକ*)dst->src[5]->data + state_offset;

            ପାଇଁ (int64_t h = h_start; h < h_end; h++) {
                size_t h_offset = h * h_stride;
                size_t t_h_offset = t_offset + h_offset;
                size_t h_2d_offset = h * h_stride_2d;

                ପାଇଁ (int64_t i = 0; i < head_size; i++) {
                    size_t t_h_i_offset = t_h_offset + i;
                    size_t h_i_offset = h_offset + i;
                    size_t h_2d_i_offset = h_2d_offset + i * h_stride;

                    ଦଶମିକ k_val = k[t_h_i_offset];
                    ଦଶମିକ r_val = r[t_h_i_offset];
                    ଦଶମିକ time_faaaa_val = time_faaaa[h_i_offset];
                    // RWKV v6: different time_decay for each token.
                    ଦଶମିକ time_decay_val = time_decay[t_h_i_offset];

                    ପାଇଁ (int64_t j = 0; j < head_size; j++) {
                        size_t t_h_j_offset = t_h_offset + j;
                        size_t h_2d_i_j_offset = h_2d_i_offset + j;

                        ଦଶମିକ v_val = v[t_h_j_offset];
                        ଦଶମିକ kv_val = v_val * k_val;
                        ଦଶମିକ prev_state_val = state_prev[h_2d_i_j_offset];
                        ଦଶମିକ temp_val = kv_val * time_faaaa_val + prev_state_val;
                        dst_data[t_h_j_offset] += temp_val * r_val;
                        state_cur[h_2d_i_j_offset] = prev_state_val * time_decay_val + kv_val;
                    }
                }
            }
        }
    #endif
}


ଶୂନ୍ୟ ggml_compute_forward_rwkv_wkv6(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_rwkv_wkv6_f32(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_gla

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_gla_f32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {
    ଅଚଳ int64_t T = dst->src[1]->ne[2];
    ଅଚଳ int64_t C = dst->ne[0];
    ଅଚଳ int64_t HEADS = dst->src[1]->ne[1];
    ଅଚଳ int64_t n_seqs = dst->src[4]->ne[1];
    ଅଚଳ int64_t head_size = C / HEADS;
    ଅଚଳ ଦଶମିକ scale = ggml_get_op_params_f32(dst, 0);

    ଦଶମିକ * dst_data = (ଦଶମିକ *) dst->data;
    ଦଶମିକ * state = ((ଦଶମିକ *) dst->data) + C * T;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା h_start =  (HEADS * (ith    )) / nth;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା h_end   = ((HEADS * (ith + 1)) / nth < HEADS) ?
                         (HEADS * (ith + 1)) / nth : HEADS;

    ଦଶମିକ * k = (ଦଶମିକ *) dst->src[0]->data;
    ଦଶମିକ * v = (ଦଶମିକ *) dst->src[1]->data;
    ଦଶମିକ * q = (ଦଶମିକ *) dst->src[2]->data;
    ଦଶମିକ * g = (ଦଶମିକ *) dst->src[3]->data;

    size_t t_stride = HEADS * head_size; // Same to C

    size_t h_stride = C / HEADS;
    GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
    size_t h_stride_2d = head_size * head_size;

    ଯଦି (ith == 0) {
        memset(dst_data, 0, T * C * ଆକାର(ଦଶମିକ));
    }
    ggml_barrier(params->threadpool);


    #ଯଦି defined(__AVX__) && !defined(__AVX512F__)
        #define GGML_F32X GGML_F32x8
        #define GGML_F32X_SET1 GGML_F32x8_SET1
        #define GGML_F32X_LOAD GGML_F32x8_LOAD
        #define GGML_F32X_STORE GGML_F32x8_STORE
        #define GGML_F32X_MUL GGML_F32x8_MUL
        #define GGML_F32X_FMA GGML_F32x8_FMA
        #define GLA_VECTOR_SIZE 8
    #ନଚେତଯଦି defined(__AVX512F__)
        #define GGML_F32X GGML_F32x16
        #define GGML_F32X_SET1 GGML_F32x16_SET1
        #define GGML_F32X_LOAD GGML_F32x16_LOAD
        #define GGML_F32X_STORE GGML_F32x16_STORE
        #define GGML_F32X_MUL GGML_F32x16_MUL
        #define GGML_F32X_FMA GGML_F32x16_FMA
        #define GLA_VECTOR_SIZE 16
    #ନଚେତଯଦି defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
        #define GGML_F32X GGML_F32xt
        #define GGML_F32X_SET1 GGML_F32xt_SET1
        #define GGML_F32X_LOAD GGML_F32xt_LOAD
        #define GGML_F32X_STORE GGML_F32xt_STORE
        #define GGML_F32X_MUL GGML_F32xt_MUL
        #define GGML_F32X_FMA GGML_F32xt_FMA
        #define GLA_VECTOR_SIZE 8
    #ନଚେତଯଦି defined(__ARM_NEON) && defined(__aarch64__)
        #define GGML_F32X GGML_F32x4
        #define GGML_F32X_SET1 GGML_F32x4_SET1
        #define GGML_F32X_LOAD GGML_F32x4_LOAD
        #define GGML_F32X_STORE GGML_F32x4_STORE
        #define GGML_F32X_MUL GGML_F32x4_MUL
        #define GGML_F32X_FMA GGML_F32x4_FMA
        #define GLA_VECTOR_SIZE 4
    #endif

    #ifdef GLA_VECTOR_SIZE
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା gla_vector_size;
        #ଯଦି defined(__ARM_FEATURE_SVE)
            gla_vector_size = svcntw();
        #ଅନ୍ୟଥା
            gla_vector_size = GLA_VECTOR_SIZE;
        #endif
        ଅଚଳ int64_t vec_ଗଣନା = head_size / gla_vector_size;

        ପାଇଁ (int64_t t = 0; t < T; t++) {
            size_t t_offset = t * t_stride;
            size_t state_offset = head_size * C * (t / (T / n_seqs));
            ଦଶମିକ * state_cur = state + state_offset;
            ଦଶମିକ * state_prev = t % (T / n_seqs) ? state_cur : (ଦଶମିକ*)dst->src[4]->data + state_offset;

            ପାଇଁ (int64_t h = h_start; h < h_end; h++) {
                size_t h_offset = h * h_stride;
                size_t t_h_offset = t_offset + h_offset;
                size_t h_2d_offset = h * h_stride_2d;

                ପାଇଁ (int64_t i = 0; i < head_size; i++) {
                    size_t t_h_i_offset = t_h_offset + i;
                    size_t h_2d_i_offset = h_2d_offset + i * h_stride;

                    ଦଶମିକ k_val = k[t_h_i_offset];
                    ଦଶମିକ q_val = q[t_h_i_offset] * scale;
                    ଦଶମିକ g_val = g[t_h_i_offset];

                    // Broadcast scalar values to vectors
                    GGML_F32X k_vec = GGML_F32X_SET1(k_val);
                    GGML_F32X q_vec = GGML_F32X_SET1(q_val);
                    GGML_F32X g_vec = GGML_F32X_SET1(g_val);

                    ପାଇଁ (int64_t j = 0; j < vec_ଗଣନା; j++) {
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
                    ପାଇଁ (int64_t j = vec_ଗଣନା * gla_vector_size; j < head_size; j++) {
                        size_t t_h_j_offset = t_h_offset + j;
                        size_t h_2d_i_j_offset = h_2d_i_offset + j;
                        ଦଶମିକ v_val = v[t_h_j_offset];
                        ଦଶମିକ kv_val = v_val * k_val;
                        ଦଶମିକ prev_state_val = state_prev[h_2d_i_j_offset];
                        ଦଶମିକ temp_val = kv_val + prev_state_val * g_val;
                        dst_data[t_h_j_offset] += temp_val * q_val;
                        state_cur[h_2d_i_j_offset] = temp_val;
                    }
                }
            }
        }

    #ଅନ୍ୟଥା
        ପାଇଁ (int64_t t = 0; t < T; t++) {
            size_t t_offset = t * t_stride;
            size_t state_offset = head_size * C * (t / (T / n_seqs));
            ଦଶମିକ * state_cur = state + state_offset;
            ଦଶମିକ * state_prev = t % (T / n_seqs) ? state_cur : (ଦଶମିକ*)dst->src[4]->data + state_offset;

            ପାଇଁ (int64_t h = h_start; h < h_end; h++) {
                size_t h_offset = h * h_stride;
                size_t t_h_offset = t_offset + h_offset;
                size_t h_2d_offset = h * h_stride_2d;

                ପାଇଁ (int64_t i = 0; i < head_size; i++) {
                    size_t t_h_i_offset = t_h_offset + i;
                    size_t h_2d_i_offset = h_2d_offset + i * h_stride;

                    ଦଶମିକ k_val = k[t_h_i_offset];
                    ଦଶମିକ q_val = q[t_h_i_offset] * scale;
                    ଦଶମିକ g_val = g[t_h_i_offset];

                    ପାଇଁ (int64_t j = 0; j < head_size; j++) {
                        size_t t_h_j_offset = t_h_offset + j;
                        size_t h_2d_i_j_offset = h_2d_i_offset + j;

                        ଦଶମିକ v_val = v[t_h_j_offset];
                        ଦଶମିକ kv_val = v_val * k_val;
                        ଦଶମିକ prev_state_val = state_prev[h_2d_i_j_offset];
                        ଦଶମିକ temp_val = prev_state_val * g_val + kv_val;
                        dst_data[t_h_j_offset] += temp_val * q_val;
                        state_cur[h_2d_i_j_offset] = temp_val;
                    }
                }
            }
        }
    #endif
}


ଶୂନ୍ୟ ggml_compute_forward_gla(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_gla_f32(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_solve_tri_f32(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst) {
    ଅଚଳ ସଂରଚନା ggml_tensor * src0 = dst->src[0];  // A (lower triangular)
    ଅଚଳ ସଂରଚନା ggml_tensor * src1 = dst->src[1];  // B (RHS)

    GGML_TENSOR_ଦ୍ୱିମିକ_OP_LOCALS;

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(ne00 == ne01); // A must be square
    GGML_ASSERT(ne0  == ne10); // solution cols == B cols
    GGML_ASSERT(ne1  == ne11); // solution rows == B rows

    GGML_ASSERT(ne02 == ne12 && ne12 == ne2);
    GGML_ASSERT(ne03 == ne13 && ne13 == ne3);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ int64_t k = ne10;   // number of RHS columns
    ଅଚଳ int64_t n = ne11;   // A is n×n
    ଅଚଳ int64_t nr = ne02 * ne03 * k; // we're parallelizing on columns here, so seq x token x column will be the unit

    // chunks per thread
    ଅଚଳ int64_t dr = (nr + nth - 1)/nth;

    // chunk range for this thread
    ଅଚଳ int64_t ir0 = dr*ith;
    ଅଚଳ int64_t ir1 = MIN(ir0 + dr, nr);

    ଅଚଳ ଦଶମିକ * A = (ଅଚଳ ଦଶମିକ *) src0->data;  // [n, n, B1, B2]
    ଅଚଳ ଦଶମିକ * B = (ଅଚଳ ଦଶମିକ *) src1->data;  // [n, k, B1, B2]
          ଦଶମିକ * X = (      ଦଶମିକ *) dst->data;   // [n, k, B1, B2]

    ପାଇଁ (int64_t ir = ir0; ir < ir1; ++ir) {
        ଅଚଳ int64_t i03 = ir/(ne02*k);
        ଅଚଳ int64_t i02 = (ir - i03*ne02*k)/k;
        ଅଚଳ int64_t i01 = (ir - i03*ne02*k - i02*k);

        ଅଚଳ ଦଶମିକ * A_batch = A + i02 * nb02 / ଆକାର(ଦଶମିକ) + i03 * nb03 / ଆକାର(ଦଶମିକ);
        ଅଚଳ ଦଶମିକ * B_batch = B + i02 * nb12 / ଆକାର(ଦଶମିକ) + i03 * nb13 / ଆକାର(ଦଶମିକ);

        ଦଶମିକ * X_batch = X + i02 * nb2 / ଆକାର(ଦଶମିକ) + i03 * nb3 / ଆକାର(ଦଶମିକ);

        ପାଇଁ (int64_t i00 = 0; i00 < n; ++i00) {
            ଦଶମିକ sum = 0.0f;
            ପାଇଁ (int64_t t = 0; t < i00; ++t) {
                sum += A_batch[i00 * n + t] * X_batch[t * k + i01];
            }

            ଅଚଳ ଦଶମିକ diag = A_batch[i00 * n + i00];
            ପ୍ରତିଜ୍ଞା(diag != 0.0f && "Zero diagonal in triangular matrix");

            X_batch[i00 * k + i01] = (B_batch[i00 * k + i01] - sum) / diag;
        }
    }
}

ଶୂନ୍ୟ ggml_compute_forward_solve_tri(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst) {
    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];

    ଯଦି (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        ggml_compute_forward_solve_tri_f32(params, dst);
    } ଅନ୍ୟଥା {
        GGML_ABORT("fatal error");
    }
}

// ggml_compute_forward_gated_delta_net
ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_gated_delta_net_one_chunk(
    ଅଚଳ ggml_compute_params * params,
    ggml_tensor * dst,
    int64_t ir0,
    int64_t ir1) {

    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    ଅଚଳ int64_t S_v      = src_v->ne[0];
    ଅଚଳ int64_t H        = src_v->ne[1];
    ଅଚଳ int64_t n_tokens = src_v->ne[2];
    ଅଚଳ int64_t n_seqs   = src_v->ne[3];

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

    ଅଚଳ ତାର୍କିକ kda = (neg0 == S_v);

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    ଅଚଳ int64_t K = ggml_get_op_params_i32(dst, 0);
    GGML_ASSERT(K >= 1);
    // per-seq stride in floats (seq s starts at state + s * seq_stride)
    ଅଚଳ int64_t state_seq_stride = src_state->nb[3] / ଆକାର(ଦଶମିକ);

    ଅଚଳ int64_t per_thread = S_v + (K > 1 ? S_v * S_v : 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;

    ଦଶମିକ * delta       = (ଦଶମିକ *)params->wdata + ith * per_thread + CACHE_ଧାଡ଼ି_SIZE_F32;
    ଦଶମିକ * state_work  = K > 1 ? (delta + S_v) : nullptr;

    // output layout: [attn_scores | new_states]
    // attn_scores: S_v * H * n_tokens * n_seqs    floats
    // new_states:  S_v * S_v * H * n_seqs * K     floats  (K snapshot slots; last min(n_tokens, K))
    ଅଚଳ int64_t attn_score_elems    = S_v * H * n_tokens * n_seqs;
    ଅଚଳ int64_t state_size_per_snap = S_v * S_v * H * n_seqs;
    ଦଶମିକ * attn_out_base  = (ଦଶମିକ *)dst->data;
    ଦଶମିକ * state_out_base = (ଦଶମିକ *)dst->data + attn_score_elems;

    // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
    // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.

    ଅଚଳ ଦଶମିକ * state_in_base = (ଅଚଳ ଦଶମିକ *)src_state->data;

  //const int64_t rq1 = nev1 / neq1;
  //const int64_t rk1 = nev1 / nek1;
    ଅଚଳ int64_t rq3 = nev3 / neq3;
    ଅଚଳ int64_t rk3 = nev3 / nek3;

    ଅଚଳ ଦଶମିକ scale = 1.0f / sqrtf((ଦଶମିକ) S_v);

    ପାଇଁ (int64_t ir = ir0; ir < ir1; ++ir) {
        ଅଚଳ int64_t iv1 = ir % H; // head_index
        ଅଚଳ int64_t iv3 = ir / H; // sequence

        ଅଚଳ int64_t iq1 = iv1 % neq1;
        ଅଚଳ int64_t ik1 = iv1 % nek1;

        ଅଚଳ int64_t iq3 = iv3 / rq3;
        ଅଚଳ int64_t ik3 = iv3 / rk3;

        // For K=1, write directly to the single output slot to avoid an extra memcpy at the end.
        // For K>1, work in scratch and copy out per-token when the slot is in range.
        ଦଶମିକ * s_out = (K > 1)
            ? state_work
            : state_out_base + (iv3 * H + iv1) * S_v * S_v;

        // copy input state into the working buffer and operate in-place
        // state layout [S_v, S_v, H, n_seqs]: seq iv3 starts at iv3 * state_seq_stride.
        ଅଚଳ ଦଶମିକ * s_in = state_in_base + iv3 * state_seq_stride + iv1 * S_v * S_v;
        memcpy(s_out, s_in, S_v * S_v * ଆକାର(ଦଶମିକ));

        // attn output pointer for first token of this (head, seq)
        ଦଶମିକ * attn_data = attn_out_base + (iv3 * n_tokens * H + iv1) * S_v;

        ପାଇଁ (int64_t t = 0; t < n_tokens; t++) {
            ଅଚଳ ଦଶମିକ * q_d = (ଅଚଳ ଦଶମିକ *)((ଅଚଳ ଅକ୍ଷର *)src_q->data + iq3 * nbq3 + t * nbq2 + iq1 * nbq1);
            ଅଚଳ ଦଶମିକ * k_d = (ଅଚଳ ଦଶମିକ *)((ଅଚଳ ଅକ୍ଷର *)src_k->data + ik3 * nbk3 + t * nbk2 + ik1 * nbk1);
            ଅଚଳ ଦଶମିକ * v_d = (ଅଚଳ ଦଶମିକ *)((ଅଚଳ ଅକ୍ଷର *)src_v->data + iv3 * nbv3 + t * nbv2 + iv1 * nbv1);

            ଅଚଳ ଦଶମିକ beta_val = *(ଅଚଳ ଦଶମିକ *)((ଅଚଳ ଅକ୍ଷର *)src_beta->data + iv3 * nbb3 + t * nbb2 + iv1 * nbb1);
            ଅଚଳ ଦଶମିକ * g_d    =  (ଅଚଳ ଦଶମିକ *)((ଅଚଳ ଅକ୍ଷର *)src_g->data    + iv3 * nbg3 + t * nbg2 + iv1 * nbg1);

            // state is stored transposed: s_out[j*S_v + i] = S[i][j]
            // so row j of s_out = column j of S (contiguous access)

            ଯଦି (kda) {
                // precompute exp(g) into delta scratch (reused below)
                ପାଇଁ (int64_t i = 0; i < S_v; ++i) {
                    delta[i] = expf(g_d[i]);
                }
                // S[i][:] *= exp(g[i]) => for each row j of M: M[j][i] *= exp(g[i])
                ପାଇଁ (int64_t j = 0; j < S_v; ++j) {
                    ggml_vec_mul_f32(S_v, &s_out[j * S_v], &s_out[j * S_v], delta);
                }
            } ଅନ୍ୟଥା {
                ggml_vec_scale_f32(S_v * S_v, s_out, expf(g_d[0]));
            }

            // delta[j] = sum_i S[i][j] * k[i] = dot(row j of M, k)
            ପାଇଁ (int64_t j = 0; j < S_v; ++j) {
                ଦଶମିକ sum = 0.0f;
                ggml_vec_dot_f32(S_v, &sum, 0, &s_out[j * S_v], 0, k_d, 0, 1);
                delta[j] = (v_d[j] - sum) * beta_val;
            }

            // outer product: S[i][j] += k[i] * delta[j] => M[j][i] += delta[j] * k[i]
            ପାଇଁ (int64_t j = 0; j < S_v; ++j) {
                ggml_vec_mad_f32(S_v, &s_out[j * S_v], k_d, delta[j]);
            }

            // attn_out[j] = sum_i S[i][j] * q[i] = dot(row j of M, q)
            ପାଇଁ (int64_t j = 0; j < S_v; ++j) {
                ଦଶମିକ sum = 0.0f;
                ggml_vec_dot_f32(S_v, &sum, 0, &s_out[j * S_v], 0, q_d, 0, 1);
                attn_data[j] = sum * scale;
            }

            attn_data += S_v * H; // advance to next token

            ଯଦି (K > 1) {
                ଅଚଳ int64_t target_slot = n_tokens - 1 - t;
                ଯଦି (target_slot >= 0 && target_slot < K) {
                    ଦଶମିକ * curr_state_o = state_out_base + target_slot * state_size_per_snap +
                                     (iv3 * H + iv1) * S_v * S_v;
                    memcpy(curr_state_o, s_out, S_v * S_v * ଆକାର(ଦଶମିକ));
                }
            }
        }
    }
}


ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_gated_delta_net_f32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ggml_tensor * V = dst->src[2];
    int64_t nr = V->ne[1] * V->ne[3];

    // disable for NUMA
    ଅଚଳ ତାର୍କିକ disable_chunking = ggml_is_numa();

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;

    // 4x chunks per thread
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth_scaled = nth * 4;
    int64_t chunk_size = (nr + nth_scaled - 1) / nth_scaled;
    int64_t nchunk     = (nr + chunk_size - 1) / chunk_size;

    ଯଦି (nth == 1 || nchunk < nth || disable_chunking) {
      nchunk = nth;
    }

    ଯଦି (ith == 0) {
      ggml_threadpool_chunk_set(params->threadpool, nth);
    }

    ggml_barrier(params->threadpool);

    ଅଚଳ int64_t dr = (nr + nchunk - 1) / nchunk;

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା current_chunk = ith;

    ଯେତେବେଳେ (current_chunk < nchunk) {
        ଅଚଳ int64_t ir0 = dr * current_chunk;
        ଅଚଳ int64_t ir1 = MIN(ir0 + dr, nr);

        ggml_compute_forward_gated_delta_net_one_chunk(params, dst, ir0, ir1);
        current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
    }
}

ଶୂନ୍ୟ ggml_compute_forward_gated_delta_net(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {
    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_gated_delta_net_f32(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}


// ggml_compute_forward_dsv4_hc_comb

ସ୍ଥିର ଶୂନ୍ୟ ggml_dsv4_hc_comb_norm_cols(ଦଶମିକ * comb, ଦଶମିକ eps) {
    constexpr int64_t hc = 4;

    ପାଇଁ (int64_t idst = 0; idst < hc; ++idst) {
        ଦଶମିକ sum = eps;
        ପାଇଁ (int64_t isrc = 0; isrc < hc; ++isrc) {
            sum += comb[idst + hc*isrc];
        }

        ଅଚଳ ଦଶମିକ inv_sum = 1.0f / sum;
        ପାଇଁ (int64_t isrc = 0; isrc < hc; ++isrc) {
            comb[idst + hc*isrc] *= inv_sum;
        }
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_dsv4_hc_comb_norm_rows(ଦଶମିକ * comb, ଦଶମିକ eps) {
    constexpr int64_t hc = 4;

    ପାଇଁ (int64_t isrc = 0; isrc < hc; ++isrc) {
        ଦଶମିକ sum = eps;
        ପାଇଁ (int64_t idst = 0; idst < hc; ++idst) {
            sum += comb[idst + hc*isrc];
        }

        ଅଚଳ ଦଶମିକ inv_sum = 1.0f / sum;
        ପାଇଁ (int64_t idst = 0; idst < hc; ++idst) {
            comb[idst + hc*isrc] *= inv_sum;
        }
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_dsv4_hc_comb_f32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {
    ଅଚଳ ggml_tensor * mixes = dst->src[0];
    ଅଚଳ ggml_tensor * scale = dst->src[1];
    ଅଚଳ ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    constexpr int64_t hc = 4;
    constexpr int64_t comb_offset = 2*hc;
    constexpr int64_t hc_mix_dim = (2 + hc)*hc;

    ଅଚଳ int64_t n_tokens = mixes->ne[1];

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

    ଅଚଳ ଦଶମିକ eps = ggml_get_op_params_f32(dst, 0);
    ଅଚଳ int32_t n_iter = ggml_get_op_params_i32(dst, 1);
    GGML_ASSERT(n_iter > 0);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ int64_t dr  = (n_tokens + nth - 1) / nth;
    ଅଚଳ int64_t it0 = dr * ith;
    ଅଚଳ int64_t it1 = MIN(it0 + dr, n_tokens);

    ଅଚଳ ଦଶମିକ scale_comb = *(ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) scale->data + 2*nbs0);

    ପାଇଁ (int64_t it = it0; it < it1; ++it) {
        ଦଶମିକ comb[hc*hc];

        ପାଇଁ (int64_t isrc = 0; isrc < hc; ++isrc) {
            ଦଶମିକ max = -INFINITY;
            ପାଇଁ (int64_t idst = 0; idst < hc; ++idst) {
                ଅଚଳ int64_t idx = idst + hc*isrc;
                ଅଚଳ ଦଶମିକ xv = *(ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) mixes->data + (comb_offset + idx)*nbm0 + it*nbm1);
                ଅଚଳ ଦଶମିକ bv = *(ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) base->data  + (comb_offset + idx)*nbb0);
                ଅଚଳ ଦଶମିକ v = xv * scale_comb + bv;
                comb[idx] = v;
                max = MAX(max, v);
            }

            ଦଶମିକ sum = 0.0f;
            ପାଇଁ (int64_t idst = 0; idst < hc; ++idst) {
                ଅଚଳ int64_t idx = idst + hc*isrc;
                ଅଚଳ ଦଶମିକ v = expf(comb[idx] - max);
                comb[idx] = v;
                sum += v;
            }

            ଅଚଳ ଦଶମିକ inv_sum = 1.0f / sum;
            ପାଇଁ (int64_t idst = 0; idst < hc; ++idst) {
                ଅଚଳ int64_t idx = idst + hc*isrc;
                comb[idx] = comb[idx] * inv_sum + eps;
            }
        }

        ggml_dsv4_hc_comb_norm_cols(comb, eps);
        ପାଇଁ (int32_t i = 1; i < n_iter; ++i) {
            ggml_dsv4_hc_comb_norm_rows(comb, eps);
            ggml_dsv4_hc_comb_norm_cols(comb, eps);
        }

        ପାଇଁ (int64_t isrc = 0; isrc < hc; ++isrc) {
            ପାଇଁ (int64_t idst = 0; idst < hc; ++idst) {
                ଅଚଳ int64_t idx = idst + hc*isrc;
                *(ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data + idst*nbd0 + isrc*nbd1 + it*nbd2) = comb[idx];
            }
        }
    }
}

ଶୂନ୍ୟ ggml_compute_forward_dsv4_hc_comb(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {
    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_dsv4_hc_comb_f32(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_dsv4_hc_pre

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_dsv4_hc_pre_f32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {
    ଅଚଳ ggml_tensor * x       = dst->src[0];
    ଅଚଳ ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ଅଚଳ int64_t n_embd   = x->ne[0];
    ଅଚଳ int64_t hc       = x->ne[1];
    ଅଚଳ int64_t n_tokens = x->ne[2];

    ଅଚଳ ଦଶମିକ scale = ggml_get_op_params_f32(dst, 0);
    ଅଚଳ ତାର୍କିକ  gated = ggml_get_op_params_i32(dst, 1) != 0;

    GGML_ASSERT(dst->ne[0] == n_embd);
    GGML_ASSERT(dst->ne[1] == n_tokens);
    ଯଦି (gated) {
        GGML_ASSERT(weights->ne[0] == n_embd);
        GGML_ASSERT(weights->ne[1] == hc);
        GGML_ASSERT(weights->ne[2] == n_tokens);
    } ଅନ୍ୟଥା {
        GGML_ASSERT(weights->ne[0] == hc);
        GGML_ASSERT(weights->ne[1] == n_tokens);
    }

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ int64_t nr  = n_embd * n_tokens;
    ଅଚଳ int64_t dr  = (nr + nth - 1) / nth;
    ଅଚଳ int64_t ir0 = dr * ith;
    ଅଚଳ int64_t ir1 = MIN(ir0 + dr, nr);

    ପାଇଁ (int64_t ir = ir0; ir < ir1; ++ir) {
        ଅଚଳ int64_t i0 = ir % n_embd;
        ଅଚଳ int64_t it = ir / n_embd;

        ଦଶମିକ sum = 0.0f;
        ପାଇଁ (int64_t ih = 0; ih < hc; ++ih) {
            ଅଚଳ ଦଶମିକ xv = *(ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) x->data + i0*nbx0 + ih*nbx1 + it*nbx2);
            ଦଶମିକ wv;
            ଯଦି (gated) {
                ଅଚଳ ଦଶମିକ gv = *(ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) weights->data + i0*nbw0 + ih*nbw1 + it*nbw2);
                wv = 1.0f / (1.0f + expf(-gv));
            } ଅନ୍ୟଥା {
                wv = *(ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) weights->data + ih*nbw0 + it*nbw1);
            }
            sum += xv * wv;
        }

        *(ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data + i0*nbd0 + it*nbd1) = scale * sum;
    }
}

ଶୂନ୍ୟ ggml_compute_forward_dsv4_hc_pre(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {
    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_dsv4_hc_pre_f32(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_dsv4_hc_post

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_dsv4_hc_post_f32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {
    ଅଚଳ ggml_tensor * x        = dst->src[0];
    ଅଚଳ ggml_tensor * residual = dst->src[1];
    ଅଚଳ ggml_tensor * post     = dst->src[2];
    ଅଚଳ ggml_tensor * comb     = dst->src[3];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ଅଚଳ int64_t n_embd   = x->ne[0];
    ଅଚଳ int64_t n_tokens = x->ne[1];
    ଅଚଳ int64_t hc       = residual->ne[1];

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
    ଯଦି (comb) {
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

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ int64_t nr  = n_embd * hc * n_tokens;
    ଅଚଳ int64_t dr  = (nr + nth - 1) / nth;
    ଅଚଳ int64_t ir0 = dr * ith;
    ଅଚଳ int64_t ir1 = MIN(ir0 + dr, nr);

    ପାଇଁ (int64_t ir = ir0; ir < ir1; ++ir) {
        ଅଚଳ int64_t i0     = ir % n_embd;
        ଅଚଳ int64_t idst   = (ir / n_embd) % hc;
        ଅଚଳ int64_t it     = ir / (n_embd * hc);

        ଅଚଳ ଦଶମିକ xv = *(ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) x->data    + i0*nbx0 + it*nbx1);
        ଅଚଳ ଦଶମିକ pv = *(ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) post->data + idst*nbp0 + it*nbp1);

        ଦଶମିକ sum = xv * pv;
        ଯଦି (comb) {
            ପାଇଁ (int64_t isrc = 0; isrc < hc; ++isrc) {
                ଅଚଳ ଦଶମିକ rv = *(ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) residual->data + i0*nbr0 + isrc*nbr1 + it*nbr2);
                ଅଚଳ ଦଶମିକ cv = *(ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) comb->data     + idst*nbc0 + isrc*nbc1 + it*nbc2);
                sum += rv * cv;
            }
        } ଅନ୍ୟଥା {
            sum += *(ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) residual->data + i0*nbr0 + idst*nbr1 + it*nbr2);
        }

        *(ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data + i0*nbd0 + idst*nbd1 + it*nbd2) = sum;
    }
}

ଶୂନ୍ୟ ggml_compute_forward_dsv4_hc_post(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {
    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_dsv4_hc_post_f32(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_top_k_mask

ଶୂନ୍ୟ ggml_compute_forward_ଶୀର୍ଷ_k_mask(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {
    ଅଚଳ ggml_tensor * mask   = dst->src[0];
    ଅଚଳ ggml_tensor * ଶୀର୍ଷ_k  = dst->src[1];
    ଅଚଳ ggml_tensor * score  = dst->src[2];
    ଅଚଳ ggml_tensor * blocks = dst->src[3];
    ଅଚଳ ggml_tensor * extra  = dst->src[4];

    GGML_ASSERT(dst->type == mask->type);
    GGML_ASSERT(dst->nb[0] == ggml_type_size(dst->type));

    ଅଚଳ int64_t n_kv    = mask->ne[0];
    ଅଚଳ int64_t ne1     = dst->ne[1];
    ଅଚଳ int64_t ne2     = dst->ne[2];
    ଅଚଳ int64_t n_rows  = ggml_nrows(dst);
    ଅଚଳ int64_t n_ଶୀର୍ଷ   = ଶୀର୍ଷ_k->ne[0];
    ଅଚଳ int64_t n_blk   = blocks ? blocks->ne[0] : 1;
    ଅଚଳ int64_t n_pick  = blocks ? blocks->ne[1] : score ? score->ne[0] : n_kv;
    ଅଚଳ int64_t n_extra = extra ? extra->ne[0] : 0;

    ଅଚଳ size_t ts = ggml_type_size(dst->type);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ପାଇଁ (int64_t ir = ith; ir < n_rows; ir += nth) {
        ଅଚଳ int64_t i1 = ir % ne1;
        ଅଚଳ int64_t i2 = (ir / ne1) % ne2;
        ଅଚଳ int64_t i3 = ir / (ne1*ne2);

        ଅଚଳ ଅକ୍ଷର * mask_row = (ଅଚଳ ଅକ୍ଷର *) mask->data
            + (i1 % mask->ne[1])*mask->nb[1] + (i2 % mask->ne[2])*mask->nb[2] + (i3 % mask->ne[3])*mask->nb[3];
        ଅକ୍ଷର * dst_row = (ଅକ୍ଷର *) dst->data + i1*dst->nb[1] + i2*dst->nb[2] + i3*dst->nb[3];

        ଯଦି (dst->type == GGML_TYPE_F16) {
            ଅଚଳ ggml_fp16_t ninf = GGML_CPU_FP32_TO_FP16(-INFINITY);
            ପାଇଁ (int64_t c = 0; c < n_kv; ++c) {
                ((ggml_fp16_t *) dst_row)[c] = ninf;
            }
        } ଅନ୍ୟଥା {
            ପାଇଁ (int64_t c = 0; c < n_kv; ++c) {
                ((ଦଶମିକ *) dst_row)[c] = -INFINITY;
            }
        }

        auto keep = [&](int32_t c) {
            ଯଦି (c >= 0 && c < n_kv) {
                memcpy(dst_row + c*ts, mask_row + c*ts, ts);
            }
        };

        ଅଚଳ ଅକ୍ଷର * ଶୀର୍ଷ_k_row = (ଅଚଳ ଅକ୍ଷର *) ଶୀର୍ଷ_k->data + i1*ଶୀର୍ଷ_k->nb[1] + i2*ଶୀର୍ଷ_k->nb[2] + i3*ଶୀର୍ଷ_k->nb[3];

        ପାଇଁ (int64_t j = 0; j < n_ଶୀର୍ଷ; ++j) {
            ଅଚଳ int32_t p = *(ଅଚଳ int32_t *) (ଶୀର୍ଷ_k_row + j*ଶୀର୍ଷ_k->nb[0]);
            ଯଦି (p < 0 || p >= n_pick) {
                ଜାରି;
            }
            ଯଦି (score) {
                ଅଚଳ ଦଶମିକ s = *(ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) score->data
                    + p*score->nb[0] + i1*score->nb[1] + i2*score->nb[2] + i3*score->nb[3]);
                ଯଦି (!(s > -INFINITY)) {
                    ଜାରି;
                }
            }
            ଯଦି (!blocks) {
                keep(p);
                ଜାରି;
            }
            ପାଇଁ (int64_t k = 0; k < n_blk; ++k) {
                keep(*(ଅଚଳ int32_t *) ((ଅଚଳ ଅକ୍ଷର *) blocks->data + k*blocks->nb[0] + p*blocks->nb[1]));
            }
        }

        ଯଦି (extra) {
            ଅଚଳ ଅକ୍ଷର * extra_row = (ଅଚଳ ଅକ୍ଷର *) extra->data + i1*extra->nb[1] + i2*extra->nb[2] + i3*extra->nb[3];
            ପାଇଁ (int64_t j = 0; j < n_extra; ++j) {
                keep(*(ଅଚଳ int32_t *) (extra_row + j*extra->nb[0]));
            }
        }
    }
}

// ggml_compute_forward_rwkv_wkv7

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_rwkv_wkv7_f32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {
    ଅଚଳ int64_t T = dst->src[1]->ne[2];
    ଅଚଳ int64_t C = dst->ne[0];
    ଅଚଳ int64_t HEADS = dst->src[1]->ne[1];
    ଅଚଳ int64_t n_seqs = dst->src[6]->ne[1];
    ଅଚଳ int64_t head_size = C / HEADS;

    ଦଶମିକ * dst_data = (ଦଶମିକ *) dst->data;
    ଦଶମିକ * state = ((ଦଶମିକ *) dst->data) + C * T;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା h_start =  (HEADS * (ith    )) / nth;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା h_end   = ((HEADS * (ith + 1)) / nth < HEADS) ?
                         (HEADS * (ith + 1)) / nth : HEADS;

    ଦଶମିକ * r = (ଦଶମିକ *) dst->src[0]->data;
    ଦଶମିକ * w = (ଦଶମିକ *) dst->src[1]->data;
    ଦଶମିକ * k = (ଦଶମିକ *) dst->src[2]->data;
    ଦଶମିକ * v = (ଦଶମିକ *) dst->src[3]->data;
    ଦଶମିକ * a = (ଦଶମିକ *) dst->src[4]->data;
    ଦଶମିକ * b = (ଦଶମିକ *) dst->src[5]->data;

    int64_t t_stride = HEADS * head_size; // Same to C

    int64_t h_stride = C / HEADS;
    GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
    int64_t h_stride_2d = head_size * head_size;

    #ଯଦି defined(GGML_SIMD)
        #ଯଦି defined(__ARM_FEATURE_SVE) || defined(__riscv_v_intrinsic)
            // scalar Route to scalar implementation       //TODO: Write SVE code and RVV code
            ପାଇଁ (int64_t t = 0; t < T; t++) {
                int64_t t_offset = t * t_stride;
                int64_t state_offset = head_size * C * (t / (T / n_seqs));
                ଦଶମିକ * state_cur = state + state_offset;
                ଦଶମିକ * state_prev = t % (T / n_seqs) ? state_cur : (ଦଶମିକ*)dst->src[6]->data + state_offset;

                ପାଇଁ (int64_t h = h_start; h < h_end; h++) {
                    int64_t h_offset = h * h_stride;
                    int64_t t_h_offset = t_offset + h_offset;
                    int64_t h_2d_offset = h * h_stride_2d;

                    ପାଇଁ (int64_t i = 0; i < head_size; i++) {
                        int64_t t_h_i_offset = t_h_offset + i;
                        int64_t h_2d_i_offset = h_2d_offset + i * h_stride;

                        ଦଶମିକ v_val = v[t_h_i_offset];

                        ଦଶମିକ sa = 0, result = 0;
                        ପାଇଁ (int64_t j = 0; j < head_size; j++) {
                            sa += a[t_h_offset + j] * state_prev[h_2d_i_offset + j];
                        }

                        ପାଇଁ (int64_t j = 0; j < head_size; j++) {
                            int64_t t_h_j_offset = t_h_offset + j;
                            int64_t h_2d_i_j_offset = h_2d_i_offset + j;

                            ଦଶମିକ r_val = r[t_h_j_offset];
                            ଦଶମିକ w_val = w[t_h_j_offset];
                            ଦଶମିକ k_val = k[t_h_j_offset];
                            ଦଶମିକ b_val = b[t_h_j_offset];
                            ଦଶମିକ kv_val = v_val * k_val;
                            ଦଶମିକ prev_state_val = state_prev[h_2d_i_j_offset];
                            state_cur[h_2d_i_j_offset] = prev_state_val * w_val + kv_val + sa * b_val;
                            result += state_cur[h_2d_i_j_offset] * r_val;
                        }
                        dst_data[t_h_i_offset] = result;
                    }
                }
            }
        #ଅନ୍ୟଥା
            ପାଇଁ (int64_t t = 0; t < T; t++) {
                int64_t t_offset = t * t_stride;
                int64_t state_offset = head_size * C * (t / (T / n_seqs));
                ଦଶମିକ * state_cur = state + state_offset;
                ଦଶମିକ * state_prev = t % (T / n_seqs) ? state_cur : (ଦଶମିକ*)dst->src[6]->data + state_offset;

                ପାଇଁ (int64_t h = h_start; h < h_end; h++) {
                    int64_t h_offset = h * h_stride;
                    int64_t t_h_offset = t_offset + h_offset;
                    int64_t h_2d_offset = h * h_stride_2d;

                    ପାଇଁ (int64_t ii = 0; ii < head_size; ii++) {
                        int64_t t_h_i_offset = t_h_offset + ii;
                        int64_t h_2d_i_offset = h_2d_offset + ii * h_stride;

                        GGML_F32_VEC v_vec = GGML_F32_VEC_SET1(v[t_h_i_offset]);

                        ଦଶମିକ sa = 0;
                        {
                            GGML_F32_VEC sum[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };
                            GGML_F32_VEC ax[GGML_F32_ARR];
                            GGML_F32_VEC ay[GGML_F32_ARR];
                            ପାଇଁ (int64_t j = 0; j < head_size; j += GGML_F32_STEP) {
                                ପାଇଁ (int64_t kk = 0; kk < GGML_F32_ARR; kk++) {
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
                        ପାଇଁ (; j < head_size; j += GGML_F32_STEP) {
                            ପାଇଁ (int64_t kk = 0; kk < GGML_F32_ARR; kk++) {
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
                        ପାଇଁ (; j < head_size; j++) {
                            int64_t t_h_j_offset = t_h_offset + j;
                            int64_t h_2d_i_j_offset = h_2d_i_offset + j;

                            ଦଶମିକ r_val = r[t_h_j_offset];
                            ଦଶମିକ w_val = w[t_h_j_offset];
                            ଦଶମିକ k_val = k[t_h_j_offset];
                            ଦଶମିକ b_val = b[t_h_j_offset];
                            ଦଶମିକ kv_val = v[t_h_i_offset] * k_val;

                            ଦଶମିକ prev_state_val = state_prev[h_2d_i_j_offset];
                            state_cur[h_2d_i_j_offset] = prev_state_val * w_val + kv_val + sa * b_val;
                            dst_data[t_h_i_offset] += state_cur[h_2d_i_j_offset] * r_val;
                        }
                    }
                }
            }
        #endif
    #ଅନ୍ୟଥା
        ପାଇଁ (int64_t t = 0; t < T; t++) {
            int64_t t_offset = t * t_stride;
            int64_t state_offset = head_size * C * (t / (T / n_seqs));
            ଦଶମିକ * state_cur = state + state_offset;
            ଦଶମିକ * state_prev = t % (T / n_seqs) ? state_cur : (ଦଶମିକ*)dst->src[6]->data + state_offset;

            ପାଇଁ (int64_t h = h_start; h < h_end; h++) {
                int64_t h_offset = h * h_stride;
                int64_t t_h_offset = t_offset + h_offset;
                int64_t h_2d_offset = h * h_stride_2d;

                ପାଇଁ (int64_t i = 0; i < head_size; i++) {
                    int64_t t_h_i_offset = t_h_offset + i;
                    int64_t h_2d_i_offset = h_2d_offset + i * h_stride;

                    ଦଶମିକ v_val = v[t_h_i_offset];

                    ଦଶମିକ sa = 0, result = 0;
                    ପାଇଁ (int64_t j = 0; j < head_size; j++) {
                        sa += a[t_h_offset + j] * state_prev[h_2d_i_offset + j];
                    }

                    ପାଇଁ (int64_t j = 0; j < head_size; j++) {
                        int64_t t_h_j_offset = t_h_offset + j;
                        int64_t h_2d_i_j_offset = h_2d_i_offset + j;

                        ଦଶମିକ r_val = r[t_h_j_offset];
                        ଦଶମିକ w_val = w[t_h_j_offset];
                        ଦଶମିକ k_val = k[t_h_j_offset];
                        ଦଶମିକ b_val = b[t_h_j_offset];
                        ଦଶମିକ kv_val = v_val * k_val;
                        ଦଶମିକ prev_state_val = state_prev[h_2d_i_j_offset];
                        state_cur[h_2d_i_j_offset] = prev_state_val * w_val + kv_val + sa * b_val;
                        result += state_cur[h_2d_i_j_offset] * r_val;
                    }
                    dst_data[t_h_i_offset] = result;
                }
            }
        }
    #endif
}


ଶୂନ୍ୟ ggml_compute_forward_rwkv_wkv7(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_rwkv_wkv7_f32(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_map_custom1

ଶୂନ୍ୟ ggml_compute_forward_map_custom1(
        ଅଚଳ ggml_compute_params * params,
              ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * a = dst->src[0];

    ସଂରଚନା ggml_map_custom1_op_params p;
    memcpy(&p, dst->op_params, ଆକାର(p));

    p.fun(dst, a, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_map_custom2

ଶୂନ୍ୟ ggml_compute_forward_map_custom2(
        ଅଚଳ ggml_compute_params * params,
              ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * a = dst->src[0];
    ଅଚଳ ggml_tensor * b = dst->src[1];

    ସଂରଚନା ggml_map_custom2_op_params p;
    memcpy(&p, dst->op_params, ଆକାର(p));

    p.fun(dst, a, b, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_map_custom3

ଶୂନ୍ୟ ggml_compute_forward_map_custom3(
        ଅଚଳ ggml_compute_params * params,
              ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * a = dst->src[0];
    ଅଚଳ ggml_tensor * b = dst->src[1];
    ଅଚଳ ggml_tensor * c = dst->src[2];

    ସଂରଚନା ggml_map_custom3_op_params p;
    memcpy(&p, dst->op_params, ଆକାର(p));

    p.fun(dst, a, b, c, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_custom

ଶୂନ୍ୟ ggml_compute_forward_custom(
    ଅଚଳ ସଂରଚନା ggml_compute_params * params,
          ସଂରଚନା ggml_tensor * dst) {

    ସଂରଚନା ggml_custom_op_params p;
    memcpy(&p, dst->op_params, ଆକାର(p));

    p.fun(dst, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_cross_entropy_loss

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_cross_entropy_loss_f32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->nb[0] == ggml_type_size(src0->type));
    GGML_ASSERT(src1->nb[0] == ggml_type_size(src1->type));
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // TODO: handle transposed/permuted matrices
    ଅଚଳ int64_t nc = src0->ne[0];
    ଅଚଳ int64_t nr = ggml_nrows(src0);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଦଶମିକ * sums =  (ଦଶମିକ *) params->wdata;
    ଦଶମିକ * st   = ((ଦଶମିକ *) params->wdata) + nth + ith*nc;
    ଦଶମିକ sum_thread = 0.0f;

    GGML_ASSERT(params->wsize >= ଆକାର(ଦଶମିକ) * (nth + nth * nc));

    // rows per thread
    ଅଚଳ int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ int64_t ir0 = dr*ith;
    ଅଚଳ int64_t ir1 = MIN(ir0 + dr, nr);

    ପାଇଁ (int64_t i1 = ir0; i1 < ir1; ++i1) {
        ଅଚଳ ଦଶମିକ * s0 = (ଅଚଳ ଦଶମିକ *)((ଅଚଳ ଅକ୍ଷର *) src0->data + i1*src0->nb[1]);
        ଅଚଳ ଦଶମିକ * s1 = (ଅଚଳ ଦଶମିକ *)((ଅଚଳ ଅକ୍ଷର *) src1->data + i1*src1->nb[1]);

#ifndef NDEBUG
        ପାଇଁ (int64_t i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            ପ୍ରତିଜ୍ଞା(!isnan(s0[i]));
            ପ୍ରତିଜ୍ଞା(!isnan(s1[i]));
        }
#endif // NDEBUG

        ଦଶମିକ max = -INFINITY;
        ggml_vec_max_f32(nc, &max, s0);
        ଅଚଳ ggml_float sum_softmax = ggml_vec_log_soft_max_f32(nc, st, s0, max);
        ପ୍ରତିଜ୍ଞା(sum_softmax >= 0.0);

        ggml_vec_add1_f32(nc, st, st, -sum_softmax);
        ggml_vec_mul_f32(nc, st, st, s1);

        ଦଶମିକ sum_st = 0.0f;
        ggml_vec_sum_f32(nc, &sum_st, st);
        sum_thread += sum_st;

#ifndef NDEBUG
        ପାଇଁ (int64_t i = 0; i < nc; ++i) {
            ପ୍ରତିଜ୍ଞା(!isnan(st[i]));
            ପ୍ରତିଜ୍ଞା(!isinf(st[i]));
        }
#endif // NDEBUG
    }
    sums[ith] = sum_thread;
    ggml_barrier(params->threadpool);

    ଯଦି (ith == 0) {
        ଦଶମିକ * dp = (ଦଶମିକ *) dst->data;
        ggml_vec_sum_f32(nth, dp, sums);
        dp[0] *= -1.0f / (ଦଶମିକ) nr;
    }
}

ଶୂନ୍ୟ ggml_compute_forward_cross_entropy_loss(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_cross_entropy_loss_f32(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_cross_entropy_loss_back

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_cross_entropy_loss_back_f32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * grad  = dst->src[0]; // gradient of forward pass output
    ଅଚଳ ggml_tensor * src0f = dst->src[1]; // src0 of forward pass
    ଅଚଳ ggml_tensor * src1f = dst->src[2]; // src1 of forward pass

    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src0f));
    GGML_ASSERT(ggml_is_contiguous(src1f));
    GGML_ASSERT(ggml_is_contiguous(grad));
    GGML_ASSERT(ggml_are_same_shape(src0f, src1f) && ggml_are_same_shape(src0f, dst));

    ଅଚଳ int64_t ith = params->ith;
    ଅଚଳ int64_t nth = params->nth;

    // TODO: handle transposed/permuted matrices
    ଅଚଳ int64_t nc = src0f->ne[0];
    ଅଚଳ int64_t nr = ggml_nrows(src0f);

    // rows per thread
    ଅଚଳ int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ int64_t ir0 = dr*ith;
    ଅଚଳ int64_t ir1 = MIN(ir0 + dr, nr);

    ଅଚଳ ଦଶମିକ d_by_nr = ((ଅଚଳ ଦଶମିକ *) grad->data)[0] / (ଦଶମିକ) nr;

    ପାଇଁ (int64_t i1 = ir0; i1 < ir1; i1++) {
        ଦଶମିକ       * ds0 = (ଦଶମିକ       *)((ଅକ୍ଷର       *) dst->data   + i1*dst->nb[1]);
        ଅଚଳ ଦଶମିକ * s0  = (ଅଚଳ ଦଶମିକ *)((ଅଚଳ ଅକ୍ଷର *) src0f->data + i1*src0f->nb[1]);
        ଅଚଳ ଦଶମିକ * s1  = (ଅଚଳ ଦଶମିକ *)((ଅଚଳ ଅକ୍ଷର *) src1f->data + i1*src1f->nb[1]);

#ifndef NDEBUG
        ପାଇଁ (int64_t i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            ପ୍ରତିଜ୍ଞା(!isnan(s0[i]));
            ପ୍ରତିଜ୍ଞା(!isnan(s1[i]));
        }
#endif // NDEBUG

        // soft_max
        ଦଶମିକ max = -INFINITY;
        ggml_vec_max_f32(nc, &max, s0);
        ଅଚଳ ggml_float sum = ggml_vec_soft_max_f32(nc, ds0, s0, max);
        ପ୍ରତିଜ୍ଞା(sum > 0.0);
        ggml_vec_scale_f32(nc, ds0, 1.0/sum);

        // grad(src0f) = (softmax(src0f) - src1f) * grad(cross_entropy_loss(src0f, src1f)) / nr
        ggml_vec_sub_f32(nc, ds0, ds0, s1);
        ggml_vec_scale_f32(nc, ds0, d_by_nr);

#ifndef NDEBUG
        ପାଇଁ (int64_t i = 0; i < nc; ++i) {
            ପ୍ରତିଜ୍ଞା(!isnan(ds0[i]));
            ପ୍ରତିଜ୍ଞା(!isinf(ds0[i]));
        }
#endif // NDEBUG
    }
}

ଶୂନ୍ୟ ggml_compute_forward_cross_entropy_loss_back(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_cross_entropy_loss_back_f32(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_opt_step_adamw_f32(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0         = dst->src[0];
    ଅଚଳ ggml_tensor * src0_grad    = dst->src[1];
    ଅଚଳ ggml_tensor * src0_grad_m  = dst->src[2];
    ଅଚଳ ggml_tensor * src0_grad_v  = dst->src[3];
    ଅଚଳ ggml_tensor * adamw_params = dst->src[4];

    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad));
    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad_m));
    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad_v));
    GGML_ASSERT(ggml_nelements(adamw_params) == 7);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS
    GGML_ASSERT(nb00 == ଆକାର(ଦଶମିକ));

    // rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ଅଚଳ ଦଶମିକ * adamw_params_ptr = ggml_get_data_f32(adamw_params);

    ଅଚଳ ଦଶମିକ alpha  = adamw_params_ptr[0];
    ଅଚଳ ଦଶମିକ beta1  = adamw_params_ptr[1];
    ଅଚଳ ଦଶମିକ beta2  = adamw_params_ptr[2];
    ଅଚଳ ଦଶମିକ eps    = adamw_params_ptr[3];
    ଅଚଳ ଦଶମିକ wd     = adamw_params_ptr[4];
    ଅଚଳ ଦଶମିକ beta1h = adamw_params_ptr[5];
    ଅଚଳ ଦଶମିକ beta2h = adamw_params_ptr[6];
    ଅଚଳ ଦଶମିକ keep   = 1.f - alpha * wd;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir = ir0; ir < ir1; ++ir) {
        ଅଚଳ int64_t i03 = ir/(ne02*ne01);
        ଅଚଳ int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        ଅଚଳ int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        ଅଚଳ size_t offset = i03*nb03 + i02*nb02 + i01*nb01;

        ଦଶମିକ       * w = (ଦଶମିକ       *) ((ଅକ୍ଷର       *) src0->data        + offset); // weight
        ଅଚଳ ଦଶମିକ * g = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) src0_grad->data   + offset); // grad
        ଦଶମିକ       * m = (ଦଶମିକ       *) ((ଅକ୍ଷର       *) src0_grad_m->data + offset);
        ଦଶମିକ       * v = (ଦଶମିକ       *) ((ଅକ୍ଷର       *) src0_grad_v->data + offset);

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i00 = 0; i00 < ne00; ++i00) {
            m[i00] = m[i00]*beta1 +        g[i00]*(1.0f - beta1);
            v[i00] = v[i00]*beta2 + g[i00]*g[i00]*(1.0f - beta2);

            ଅଚଳ ଦଶମିକ mh =       m[i00]*beta1h;
            ଅଚଳ ଦଶମିକ vh = sqrtf(v[i00]*beta2h) + eps;

            // The weight decay is applied independently of the Adam momenta m and v.
            // This is NOT equivalent to l2 regularization that adds w[i00]*w[i00] to the loss.
            // See: https://arxiv.org/pdf/1711.05101v3.pdf
            w[i00] = w[i00] * keep - alpha * mh / vh;
        }
    }
}

ଶୂନ୍ୟ ggml_compute_forward_opt_step_adamw(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_opt_step_adamw_f32(params, dst);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_opt_step_sgd_f32(ଅଚଳ ggml_compute_params * params, ggml_tensor * dst) {
    ଅଚଳ ggml_tensor * src0       = dst->src[0];
    ଅଚଳ ggml_tensor * src0_grad  = dst->src[1];
    ଅଚଳ ggml_tensor * sgd_params = dst->src[2];

    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad));
    GGML_ASSERT(ggml_nelements(sgd_params) == 2);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS
    GGML_ASSERT(nb00 == ଆକାର(ଦଶମିକ));

    // rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1) / nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr * ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    // using adamw param subset we care about - alpha, wd - could have a separate struct
    ଅଚଳ ଦଶମିକ * sgd_params_ptr   = ggml_get_data_f32(sgd_params);
    ଅଚଳ ଦଶମିକ   alpha            = sgd_params_ptr[0];
    ଅଚଳ ଦଶମିକ   keep             = 1.f - alpha * sgd_params_ptr[1];

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir = ir0; ir < ir1; ++ir) {
        ଅଚଳ int64_t i03 = ir / (ne02 * ne01);
        ଅଚଳ int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
        ଅଚଳ int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

        ଅଚଳ size_t offset = i03 * nb03 + i02 * nb02 + i01 * nb01;

        ଦଶମିକ *       w = (ଦଶମିକ *) ((ଅକ୍ଷର *) src0->data + offset);                   // weight
        ଅଚଳ ଦଶମିକ * g = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) src0_grad->data + offset);  // grad

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i00 = 0; i00 < ne00; ++i00) {
            w[i00] = w[i00] * keep - alpha * g[i00];
        }
    }
}

ଶୂନ୍ୟ ggml_compute_forward_opt_step_sgd(ଅଚଳ ggml_compute_params * params, ggml_tensor * dst) {
    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_opt_step_sgd_f32(params, dst);
            }
            ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error - sgd is F32 only");
            }
    }
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ ggml_fwht_load(ଅଚଳ ଦଶମିକ value) {
    ଫେରନ୍ତୁ value;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ ggml_fwht_load(ଅଚଳ ggml_fp16_t value) {
    ଫେରନ୍ତୁ ggml_fp16_to_fp32(value);
}

template<typename src_t>
ସ୍ଥିର ଶୂନ୍ୟ ggml_compute_forward_fwht_impl(ଅଚଳ ggml_compute_params * params, ggml_tensor * dst) {
    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == (std::is_same_v<src_t, ଦଶମିକ> ? GGML_TYPE_F32 : GGML_TYPE_F16));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ଦ୍ୱିମିକ_OP_LOCALS

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    ଅଚଳ int64_t n = ne10;
    GGML_ASSERT((n & (n - 1)) == 0); // must be power of 2

    ଅଚଳ int64_t nr = ne11 * ne12 * ne13;
    ଅଚଳ int64_t rows_per_thread = (nr + nth - 1) / nth;
    ଅଚଳ int64_t start_row = ith * rows_per_thread;
    ଅଚଳ int64_t end_row = MIN(start_row + rows_per_thread, nr);

    ଅଚଳ ଦଶମିକ scale = 1.0f / sqrtf((ଦଶମିକ)n);

#ଯଦି defined(GGML_SIMD)
    ଅଚଳ GGML_F32_VEC v_minus_one = GGML_F32_VEC_SET1(-1.0f);
#endif

    ପାଇଁ (int64_t r = start_row; r < end_row; r++) {
        ଅଚଳ int64_t i13 = r / (ne11 * ne12);
        ଅଚଳ int64_t i12 = (r - i13 * ne11 * ne12) / ne11;
        ଅଚଳ int64_t i11 = r - i13 * ne11 * ne12 - i12 * ne11;

        ଅଚଳ src_t * src_row = (ଅଚଳ src_t *) ((ଅଚଳ ଅକ୍ଷର *) src1->data + i11 * nb11 + i12 * nb12 + i13 * nb13);
        ଦଶମିକ * dst_row = (ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data + i11 * nb1 + i12 * nb2 + i13 * nb3);

        ପାଇଁ (int64_t j = 0; j < n; j++) {
            dst_row[j] = ggml_fwht_load(src_row[j]) * scale;
        }

        // Scalar passes
#ଯଦି defined(GGML_SIMD)
#ଯଦି defined(__ARM_FEATURE_SVE)
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା step = svcntw();
#ଅନ୍ୟଥା
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା step = GGML_F32_EPR;
#endif
#ଅନ୍ୟଥା
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା step = n;
#endif
        ପାଇଁ (int64_t len = 1; len < step && len < n; len <<= 1) {
            ପାଇଁ (int64_t i = 0; i < n; i += 2 * len) {
                ପାଇଁ (int64_t j = 0; j < len; j++) {
                    ଦଶମିକ u = dst_row[i + j];
                    ଦଶମିକ v = dst_row[i + len + j];
                    dst_row[i + j] = u + v;
                    dst_row[i + len + j] = u - v;
                }
            }
        }

        // SIMD passes using GGML_F32_VEC_* macros for multi-architecture support
#ଯଦି defined(GGML_SIMD)
        ପାଇଁ (int64_t len = step; len < n; len <<= 1) {
            ପାଇଁ (int64_t i = 0; i < n; i += 2 * len) {
                ପାଇଁ (int64_t j = 0; j < len; j += step) {
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

ଶୂନ୍ୟ ggml_compute_forward_fwht(ଅଚଳ ggml_compute_params * params, ggml_tensor * dst) {
    ଅଚଳ ggml_tensor * src1 = dst->src[1];

    ବାଛ (src1->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ggml_compute_forward_fwht_impl<ଦଶମିକ>(params, dst);
            }
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F16:
            {
                ggml_compute_forward_fwht_impl<ggml_fp16_t>(params, dst);
            }
            ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            {
                GGML_ABORT("fatal error - fwht supports F32 and F16 input");
            }
    }
}

// ggml_compute_forward_lightning_indexer

ଶୂନ୍ୟ ggml_compute_forward_lightning_indexer(
        ଅଚଳ ggml_compute_params * params,
        ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * q = dst->src[0];
    ଅଚଳ ggml_tensor * k = dst->src[1];
    ଅଚଳ ggml_tensor * w = dst->src[2]; // weights
    ଅଚଳ ggml_tensor * m = dst->src[3]; // mask

    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(   q->type == GGML_TYPE_F32);
    GGML_ASSERT(   w->type == GGML_TYPE_F32);
    GGML_ASSERT(   m->type == GGML_TYPE_F16);

    GGML_TENSOR_LOCALS(int64_t, neq,  q, ne)
    GGML_TENSOR_LOCALS(size_t,  nbq,  q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek,  k, ne)
    GGML_TENSOR_LOCALS(size_t,  nbk,  k, nb)
    GGML_TENSOR_LOCALS(int64_t, ନୂତନ,  w, ne)
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

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_embd    = q->ne[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_head    = q->ne[1];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_tokens  = q->ne[2];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_stream  = q->ne[3];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_kv      = k->ne[2];

    ggml_to_float_t ଅଚଳ k_to_float = ggml_get_type_traits(k->type)->to_float;
    GGML_ASSERT((k->type == GGML_TYPE_F32 || k_to_float) && "lightning indexer: unsupported K-type");

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr  = n_kv;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = params->ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = params->nth;

    // (temporary) buffer for K converted to float
    ଦଶମିକ * k_row_f32 = (ଦଶମିକ *) params->wdata + ith*(1*n_embd + CACHE_ଧାଡ଼ି_SIZE_F32);

    // rows per thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dr = (nr + nth - 1)/nth;

    // row range for this thread
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir0 = dr*ith;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ir1 = MIN(ir0 + dr, nr);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା s = 0; s < n_stream; ++s) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା t = 0; t < n_tokens; ++t) {
            ଅଚଳ ଦଶମିକ       *   w_row =       (ଦଶମିକ *) ((ଅକ୍ଷର *)   w->data + t*nbw1 +        s*nbw3);
            ଅଚଳ ggml_fp16_t *   m_row = (ggml_fp16_t *) ((ଅକ୍ଷର *)   m->data + t*nbm1 + (s%nem3)*nbm3);
            ଦଶମିକ             * dst_row =       (ଦଶମିକ *) ((ଅକ୍ଷର *) dst->data + t*nb1  +        s*nb3 );
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ik = ir0; ik < ir1; ++ik) {
                ଅକ୍ଷର * k_row = (ଅକ୍ଷର *) k->data + ik*nbk2 + s*nbk3;
                ଯଦି (k_to_float) {
                    k_to_float(k_row, k_row_f32, n_embd);
                } ଅନ୍ୟଥା {
                    k_row_f32 = (ଦଶମିକ *) k_row;
                }
                ଦଶମିକ score = 0.0f;
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା h = 0; h < n_head; ++h) {
                    // dot product of q and k for head h
                    ଦଶମିକ qk = 0.0f;
                    ଅଚଳ ଦଶମିକ * q_row = (ଦଶମିକ *) ((ଅକ୍ଷର *) q->data + h*nbq1 + t*nbq2 + s*nbq3);
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
