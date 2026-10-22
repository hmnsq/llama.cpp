#অন্তর্ভাব "ops.h"

#অন্তর্ভাব "ggml-cpu.h"
#অন্তর্ভাব "ggml-impl.h"
#অন্তর্ভাব "binary-ops.h"
#অন্তর্ভাব "simd-gemm.h"
#অন্তর্ভাব "ggml.h"
#অন্তর্ভাব "unary-ops.h"
#অন্তর্ভাব "vec.h"

#অন্তর্ভাব <algorithm>
#অন্তর্ভাব <cfloat>
#অন্তর্ভাব <cmath>

// ggml_compute_forward_dup

স্থির শূন্য ggml_compute_forward_dup_same_cont(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));
    GGML_ASSERT(src0->type == dst->type);

    ধ্রুব size_t nb0 = ggml_type_size(src0->type);

    ধ্রুব পূর্ণাঙ্ক ith = params->ith; // thread index
    ধ্রুব পূর্ণাঙ্ক nth = params->nth; // number of threads

    // parallelize by blocks
    ধ্রুব পূর্ণাঙ্ক nk = ggml_nelements(src0)/ggml_blck_size(src0->type);
    ধ্রুব পূর্ণাঙ্ক dr = (nk + nth - 1) / nth;
    ধ্রুব পূর্ণাঙ্ক k0 = dr * ith;
    ধ্রুব পূর্ণাঙ্ক k1 = MIN(k0 + dr, nk);

    যদি (k0 < k1) {
        memcpy(
            ((বর্ণ *)  dst->data + k0*nb0),
            ((বর্ণ *) src0->data + k0*nb0),
            (k1 - k0) * nb0);
    }
}

template<typename src_t, typename dst_t>
স্থির শূন্য ggml_compute_forward_dup_flt(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(!ggml_is_quantized(src0->type) && !ggml_is_quantized(dst->type));

    GGML_TENSOR_UNARY_OP_LOCALS

    ধ্রুব পূর্ণাঙ্ক ith = params->ith; // thread index
    ধ্রুব পূর্ণাঙ্ক nth = params->nth; // number of threads

    // parallelize by rows
    ধ্রুব পূর্ণাঙ্ক nr = ne01;
    // number of rows per thread
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1) / nth;
    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr * ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    // case: type & row size equal
    যদি (src0->type == dst->type &&
        ne00 == ne0 &&
        nb00 == ggml_type_size(src0->type) && nb0 == ggml_type_size(dst->type)) {
        // copy by rows
        ধ্রুব size_t rs = ne00*nb00;
        প্রতি (int64_t i03 = 0; i03 < ne03; i03++) {
            প্রতি (int64_t i02 = 0; i02 < ne02; i02++) {
                প্রতি (int64_t i01 = ir0; i01 < ir1; i01++) {
                    memcpy(
                        ((বর্ণ *)  dst->data + i01*nb1  + i02*nb2  + i03*nb3),
                        ((বর্ণ *) src0->data + i01*nb01 + i02*nb02 + i03*nb03),
                        rs);
                }
            }
        }
        প্রত্যাগম;
    }

    // case: dst tensor is contiguous
    যদি (ggml_is_contiguous(dst)) {
        যদি (nb00 == পরিমাণ(src_t)) {
            যদি constexpr (std::is_same_v<dst_t, src_t>) {
                // same type
                size_t id = 0;
                ধ্রুব size_t rs = ne00 * nb00;
                বর্ণ * dst_ptr = (বর্ণ *) dst->data;

                প্রতি (পূর্ণাঙ্ক i03 = 0; i03 < ne03; i03++) {
                    প্রতি (পূর্ণাঙ্ক i02 = 0; i02 < ne02; i02++) {
                        id += rs * ir0;
                        প্রতি (পূর্ণাঙ্ক i01 = ir0; i01 < ir1; i01++) {
                            ধ্রুব বর্ণ * src0_ptr = (বর্ণ *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
                            memcpy(dst_ptr + id, src0_ptr, rs);
                            id += rs;
                        }
                        id += rs * (ne01 - ir1);
                    }
                }
            } অন্যথা {
                // casting between non-quantized types
                size_t id = 0;
                dst_t * dst_ptr = (dst_t *) dst->data;

                প্রতি (পূর্ণাঙ্ক i03 = 0; i03 < ne03; i03++) {
                    প্রতি (পূর্ণাঙ্ক i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        প্রতি (পূর্ণাঙ্ক i01 = ir0; i01 < ir1; i01++) {
                            ধ্রুব src_t * src0_ptr = (src_t *) ((বর্ণ *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
                            প্রতি (পূর্ণাঙ্ক i00 = 0; i00 < ne00; i00++) {
                                ভিন্ন tmp = type_conversion_table<src_t>::to_f32(src0_ptr[i00]);
                                dst_ptr[id] = type_conversion_table<dst_t>::from_f32(tmp);
                                id++;
                            }
                        }
                        id += ne00 * (ne01 - ir1);
                    }
                }
            }
        } অন্যথা {
            //printf("%s: this is not optimal - fix me\n", __func__);

            size_t id = 0;
            dst_t * dst_ptr = (dst_t *) dst->data;

            প্রতি (পূর্ণাঙ্ক i03 = 0; i03 < ne03; i03++) {
                প্রতি (পূর্ণাঙ্ক i02 = 0; i02 < ne02; i02++) {
                    id += ne00 * ir0;
                    প্রতি (পূর্ণাঙ্ক i01 = ir0; i01 < ir1; i01++) {
                        প্রতি (পূর্ণাঙ্ক i00 = 0; i00 < ne00; i00++) {
                            ধ্রুব src_t * src0_ptr = (src_t *) ((বর্ণ *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);

                            ভিন্ন tmp = type_conversion_table<src_t>::to_f32(*src0_ptr);
                            dst_ptr[id] = type_conversion_table<dst_t>::from_f32(tmp);
                            id++;
                        }
                    }
                    id += ne00 * (ne01 - ir1);
                }
            }
        }
        প্রত্যাগম;
    }

    // dst counters
    int64_t i10 = 0;
    int64_t i11 = 0;
    int64_t i12 = 0;
    int64_t i13 = 0;

    যদি constexpr (std::is_same_v<dst_t, src_t>) {
        প্রতি (int64_t i03 = 0; i03 < ne03; i03++) {
            প্রতি (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                যাবত্ (i10 >= ne0) {
                    i10 -= ne0;
                    যদি (++i11 == ne1) {
                        i11 = 0;
                        যদি (++i12 == ne2) {
                            i12 = 0;
                            যদি (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                প্রতি (int64_t i01 = ir0; i01 < ir1; i01++) {
                    প্রতি (int64_t i00 = 0; i00 < ne00; i00++) {
                        ধ্রুব বর্ণ * src0_ptr = ((বর্ণ *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              বর্ণ * dst_ptr  = ((বর্ণ *)  dst->data + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                        memcpy(dst_ptr, src0_ptr, পরিমাণ(dst_t));

                        যদি (++i10 == ne00) {
                            i10 = 0;
                            যদি (++i11 == ne01) {
                                i11 = 0;
                                যদি (++i12 == ne02) {
                                    i12 = 0;
                                    যদি (++i13 == ne03) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                যাবত্ (i10 >= ne0) {
                    i10 -= ne0;
                    যদি (++i11 == ne1) {
                        i11 = 0;
                        যদি (++i12 == ne2) {
                            i12 = 0;
                            যদি (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
            }
        }

    } অন্যথা {
        প্রতি (int64_t i03 = 0; i03 < ne03; i03++) {
            প্রতি (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                যাবত্ (i10 >= ne0) {
                    i10 -= ne0;
                    যদি (++i11 == ne1) {
                        i11 = 0;
                        যদি (++i12 == ne2) {
                            i12 = 0;
                            যদি (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                প্রতি (int64_t i01 = ir0; i01 < ir1; i01++) {
                    প্রতি (int64_t i00 = 0; i00 < ne00; i00++) {
                        ধ্রুব বর্ণ * src0_ptr = ((বর্ণ *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              বর্ণ * dst_ptr  = ((বর্ণ *)  dst->data + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                        ভিন্ন tmp = type_conversion_table<src_t>::to_f32(*(ধ্রুব src_t *) src0_ptr);
                        *(dst_t *) dst_ptr = type_conversion_table<dst_t>::from_f32(tmp);

                        যদি (++i10 == ne0) {
                            i10 = 0;
                            যদি (++i11 == ne1) {
                                i11 = 0;
                                যদি (++i12 == ne2) {
                                    i12 = 0;
                                    যদি (++i13 == ne3) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                যাবত্ (i10 >= ne0) {
                    i10 -= ne0;
                    যদি (++i11 == ne1) {
                        i11 = 0;
                        যদি (++i12 == ne2) {
                            i12 = 0;
                            যদি (++i13 == ne3) {
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
স্থির শূন্য ggml_compute_forward_dup_to_q(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(!ggml_is_quantized(src0->type));

    GGML_TENSOR_UNARY_OP_LOCALS

    ধ্রুব পূর্ণাঙ্ক ith = params->ith; // thread index
    ধ্রুব পূর্ণাঙ্ক nth = params->nth; // number of threads

    // parallelize by rows
    ধ্রুব পূর্ণাঙ্ক nr = ne01;
    // number of rows per thread
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1) / nth;
    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr * ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    যদি (ggml_is_contiguous(dst) &&
            nb00 == পরিমাণ(src_t) &&
            ggml_get_type_traits_cpu(dst->type)->from_float) {
        // casting non-quantized types --> intermediate f32 --> quantized
        ggml_from_float_t ধ্রুব quantize_row_q = ggml_get_type_traits_cpu(dst->type)->from_float;
        ভিন্ন * src0_f32 = (ভিন্ন *) params->wdata + (ne00 + CACHE_সারি_SIZE_F32) * ith;

        size_t id = 0;
        size_t rs = nb0 * (ne00 / ggml_blck_size(dst->type));
        বর্ণ * dst_ptr = (বর্ণ *) dst->data;

        প্রতি (পূর্ণাঙ্ক i03 = 0; i03 < ne03; i03++) {
            প্রতি (পূর্ণাঙ্ক i02 = 0; i02 < ne02; i02++) {
                id += rs * ir0;
                প্রতি (পূর্ণাঙ্ক i01 = ir0; i01 < ir1; i01++) {
                    ধ্রুব src_t * src0_ptr = (src_t *) ((বর্ণ *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);

                    প্রতি (পূর্ণাঙ্ক i00 = 0; i00 < ne00; i00++) {
                        src0_f32[i00] = type_conversion_table<src_t>::to_f32(src0_ptr[i00]);
                    }

                    quantize_row_q(src0_f32, dst_ptr + id, ne00);
                    id += rs;
                }
                id += rs * (ne01 - ir1);
            }
        }
    } অন্যথা {
        // printf("%s %s\n", ggml_type_name(src0->type), ggml_type_name(dst->type));
        GGML_ABORT("not implemented");
    }
}

// A simplified version of ggml_compute_forward_dup that doesn't do float upcasting, and just plain old memcpy.
স্থির শূন্য ggml_compute_forward_dup_bytes(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {
    ধ্রুব ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(src0->type == dst->type);

    GGML_TENSOR_UNARY_OP_LOCALS;

    যদি (ggml_is_contiguous(src0) && ggml_is_contiguous(dst)) {
        ggml_compute_forward_dup_same_cont(params, dst);
        প্রত্যাগম;
    }

    ধ্রুব size_t type_size = ggml_type_size(src0->type);

    ধ্রুব পূর্ণাঙ্ক ith = params->ith; // thread index
    ধ্রুব পূর্ণাঙ্ক nth = params->nth; // number of threads

    // parallelize by rows
    ধ্রুব পূর্ণাঙ্ক nr = ne01;
    // number of rows per thread
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1) / nth;
    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr * ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    যদি (src0->type == dst->type &&
        ggml_are_same_shape(src0, dst) &&
        nb00 == type_size && nb0 == type_size) {
        // copy by rows
        ধ্রুব size_t rs = ggml_row_size(src0->type, ne00);
        প্রতি (int64_t i03 = 0; i03 < ne03; i03++) {
            প্রতি (int64_t i02 = 0; i02 < ne02; i02++) {
                প্রতি (int64_t i01 = ir0; i01 < ir1; i01++) {
                    memcpy(
                        ((বর্ণ *)  dst->data + i01*nb1  + i02*nb2  + i03*nb3),
                        ((বর্ণ *) src0->data + i01*nb01 + i02*nb02 + i03*nb03),
                        rs);
                }
            }
        }
        প্রত্যাগম;
    }

    যদি (ggml_is_contiguous(dst)) {
        size_t id = 0;
        বর্ণ * dst_ptr = (বর্ণ *) dst->data;
        ধ্রুব size_t rs = ne00 * type_size;

        যদি (nb00 == type_size) {
            // src0 is contiguous on first dimension, copy by rows
            প্রতি (int64_t i03 = 0; i03 < ne03; i03++) {
                প্রতি (int64_t i02 = 0; i02 < ne02; i02++) {
                    id += rs * ir0;
                    প্রতি (int64_t i01 = ir0; i01 < ir1; i01++) {
                        ধ্রুব বর্ণ * src0_ptr = (বর্ণ *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
                        memcpy(dst_ptr + id, src0_ptr, rs);
                        id += rs;
                    }
                    id += rs * (ne01 - ir1);
                }
            }
        } অন্যথা {
            //printf("%s: this is not optimal - fix me\n", __func__);

            প্রতি (int64_t i03 = 0; i03 < ne03; i03++) {
                প্রতি (int64_t i02 = 0; i02 < ne02; i02++) {
                    id += rs * ir0;
                    প্রতি (int64_t i01 = ir0; i01 < ir1; i01++) {
                        প্রতি (int64_t i00 = 0; i00 < ne00; i00++) {
                            ধ্রুব বর্ণ * src0_ptr = (বর্ণ *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03;
                            memcpy(dst_ptr + id, src0_ptr, type_size);

                            id += type_size;
                        }
                    }
                    id += rs * (ne01 - ir1);
                }
            }
        }

        প্রত্যাগম;
    }

    // dst counters
    int64_t k10 = 0;
    int64_t i11 = 0;
    int64_t i12 = 0;
    int64_t i13 = 0;

    // number of blocks in a row
    ধ্রুব int64_t nk00 = ne00 / ggml_blck_size(src0->type);
    ধ্রুব int64_t nk0  = ne0  / ggml_blck_size(dst->type);

    প্রতি (int64_t i03 = 0; i03 < ne03; i03++) {
        প্রতি (int64_t i02 = 0; i02 < ne02; i02++) {
            k10 += nk00 * ir0;
            যাবত্ (k10 >= nk0) {
                k10 -= nk0;
                যদি (++i11 == ne1) {
                    i11 = 0;
                    যদি (++i12 == ne2) {
                        i12 = 0;
                        যদি (++i13 == ne3) {
                            i13 = 0;
                        }
                    }
                }
            }
            প্রতি (int64_t i01 = ir0; i01 < ir1; i01++) {
                প্রতি (int64_t k00 = 0; k00 < nk00; k00++) {
                    ধ্রুব বর্ণ * src0_ptr = ((বর্ণ *) src0->data + k00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                          বর্ণ * dst_ptr  = ((বর্ণ *)  dst->data + k10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                    memcpy(dst_ptr, src0_ptr, type_size);

                    যদি (++k10 == nk0) {
                        k10 = 0;
                        যদি (++i11 == ne1) {
                            i11 = 0;
                            যদি (++i12 == ne2) {
                                i12 = 0;
                                যদি (++i13 == ne3) {
                                    i13 = 0;
                                }
                            }
                        }
                    }
                }
            }
            k10 += nk00 * (ne01 - ir1);
            যাবত্ (k10 >= nk0) {
                k10 -= nk0;
                যদি (++i11 == ne1) {
                    i11 = 0;
                    যদি (++i12 == ne2) {
                        i12 = 0;
                        যদি (++i13 == ne3) {
                            i13 = 0;
                        }
                    }
                }
            }
        }
    }
}

স্থির শূন্য ggml_compute_forward_dup_from_q(
        ধ্রুব ggml_compute_params * params,
              ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_দ্বিমিক_OP_LOCALS

    ধ্রুব ggml_type type = src0->type;
    ggml_to_float_t ধ্রুব dequantize_row_q = ggml_get_type_traits(type)->to_float;

    size_t qk = ggml_blck_size(type);
    ধ্রুব int64_t nr = ggml_nelements(src1) / qk;

    // destination must be contiguous in the first dimension
    GGML_ASSERT(nb10 == ggml_type_size(dst->type));
    // must either have first dimension large enough to hold a row, or fully contiguous
    GGML_ASSERT((ne10 % qk) == 0 || ggml_is_contiguous(dst));

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr*ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    প্রতি (int64_t ir = ir0; ir < ir1; ++ir) {

        uint32_t i = ir * qk;

        ধ্রুব int64_t i03 = i/(ne00 * ne01 * ne02);
        ধ্রুব int64_t i02 = (i - i03*ne00*ne01*ne02 )/ (ne00*ne01);
        ধ্রুব int64_t i01 = (i - i03*ne00*ne01*ne02  -  i02*ne01*ne00) / ne00;
        ধ্রুব int64_t i00 = i - i03*ne00*ne01*ne02 - i02*ne01*ne00 - i01*ne00;
        ধ্রুব int64_t x_offset = (i00/qk)*nb00 + i01*nb01 + i02*nb02 + i03 * nb03;

        ধ্রুব int64_t i13 = i/(ne10 * ne11 * ne12);
        ধ্রুব int64_t i12 = (i - i13*ne10*ne11*ne12) / (ne10*ne11);
        ধ্রুব int64_t i11 = (i - i13*ne10*ne11*ne12 - i12*ne10*ne11) / ne10;
        ধ্রুব int64_t i10 = i - i13*ne10*ne11*ne12 - i12*ne10*ne11 - i11*ne10;
        ধ্রুব int64_t dst_offset = i10*nb10 + i11*nb11 + i12*nb12 + i13*nb13;

        dequantize_row_q(
                (ধ্রুব শূন্য *) ((বর্ণ *) src0->data + x_offset),
                     (ভিন্ন *) ((বর্ণ *)  dst->data + dst_offset), qk);
    }
}

শূন্য ggml_compute_forward_dup(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    যদি (src0->type == dst->type) {
        ggml_compute_forward_dup_bytes(params, dst);
        প্রত্যাগম;
    }

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F16:
            {
                /**/ যদি (dst->type == GGML_TYPE_F16)  ggml_compute_forward_dup_flt<ggml_fp16_t, ggml_fp16_t>(params, dst);
                অন্যথা যদি (dst->type == GGML_TYPE_BF16) ggml_compute_forward_dup_flt<ggml_fp16_t, ggml_bf16_t>(params, dst);
                অন্যথা যদি (dst->type == GGML_TYPE_F32)  ggml_compute_forward_dup_flt<ggml_fp16_t, ভিন্ন      >(params, dst);
                অন্যথা ggml_compute_forward_dup_to_q<ggml_fp16_t>(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_BF16:
            {
                /**/ যদি (dst->type == GGML_TYPE_F16)  ggml_compute_forward_dup_flt<ggml_bf16_t, ggml_fp16_t>(params, dst);
                অন্যথা যদি (dst->type == GGML_TYPE_BF16) ggml_compute_forward_dup_flt<ggml_bf16_t, ggml_bf16_t>(params, dst);
                অন্যথা যদি (dst->type == GGML_TYPE_F32)  ggml_compute_forward_dup_flt<ggml_bf16_t, ভিন্ন      >(params, dst);
                অন্যথা ggml_compute_forward_dup_to_q<ggml_bf16_t>(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_F32:
            {
                /**/ যদি (dst->type == GGML_TYPE_F16)  ggml_compute_forward_dup_flt<ভিন্ন, ggml_fp16_t>(params, dst);
                অন্যথা যদি (dst->type == GGML_TYPE_BF16) ggml_compute_forward_dup_flt<ভিন্ন, ggml_bf16_t>(params, dst);
                অন্যথা যদি (dst->type == GGML_TYPE_F32)  ggml_compute_forward_dup_flt<ভিন্ন, ভিন্ন      >(params, dst);
                অন্যথা যদি (dst->type == GGML_TYPE_I32)  ggml_compute_forward_dup_flt<ভিন্ন, int32_t    >(params, dst);
                অন্যথা ggml_compute_forward_dup_to_q<ভিন্ন>(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_I32:
            {
                যদি (dst->type == GGML_TYPE_F32) ggml_compute_forward_dup_flt<int32_t, ভিন্ন>(params, dst);
                অন্যথা GGML_ABORT("not implemented");
            } ভঙ্গ;
        স্বভাব:
            {
                যদি (ggml_is_quantized(src0->type) && dst->type == GGML_TYPE_F32) {
                    ggml_compute_forward_dup_from_q(params, dst);
                    ভঙ্গ;
                }
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_add

স্থির শূন্য ggml_compute_forward_add_q_f32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

    ধ্রুব পূর্ণাঙ্ক nr  = ggml_nrows(src0);

    GGML_TENSOR_দ্বিমিক_OP_LOCALS

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব ggml_type type = src0->type;
    ধ্রুব ggml_type dtype = dst->type;
    ggml_to_float_t ধ্রুব dequantize_row_q = ggml_get_type_traits(type)->to_float;
    ggml_from_float_t ধ্রুব quantize_row_q = ggml_get_type_traits_cpu(dtype)->from_float;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == পরিমাণ(ভিন্ন));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ggml_is_quantized(src0->type));
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    // rows per thread
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr*ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    ভিন্ন * wdata = (ভিন্ন *) params->wdata + (ne00 + CACHE_সারি_SIZE_F32) * ith;

    প্রতি (পূর্ণাঙ্ক ir = ir0; ir < ir1; ++ir) {
        // src0 indices
        ধ্রুব পূর্ণাঙ্ক i03 = ir/(ne02*ne01);
        ধ্রুব পূর্ণাঙ্ক i02 = (ir - i03*ne02*ne01)/ne01;
        ধ্রুব পূর্ণাঙ্ক i01 = (ir - i03*ne02*ne01 - i02*ne01);

        // src1 and dst are same shape as src0 => same indices
        ধ্রুব পূর্ণাঙ্ক i13 = i03;
        ধ্রুব পূর্ণাঙ্ক i12 = i02;
        ধ্রুব পূর্ণাঙ্ক i11 = i01;

        ধ্রুব পূর্ণাঙ্ক i3 = i03;
        ধ্রুব পূর্ণাঙ্ক i2 = i02;
        ধ্রুব পূর্ণাঙ্ক i1 = i01;

        শূন্য  * src0_row = (শূন্য *) ((বর্ণ *) src0->data + (i01*nb01 + i02*nb02 + i03*nb03));
        ভিন্ন * src1_row = (ভিন্ন *)((বর্ণ *) src1->data + (i11*nb11 + i12*nb12 + i13*nb13));
        শূন্য  * dst_row  = (শূন্য *) ((বর্ণ *)  dst->data + ( i1*nb1  +  i2*nb2  +  i3*nb3));

        প্রতিজ্ঞা(ne00 % 32 == 0);

        // unquantize row from src0 to temp buffer
        dequantize_row_q(src0_row, wdata, ne00);
        // add src1
        ggml_vec_acc_f32(ne00, wdata, src1_row);
        // quantize row to dst
        যদি (quantize_row_q != রিক্ত) {
            quantize_row_q(wdata, dst_row, ne00);
        } অন্যথা {
            memcpy(dst_row, wdata, ne0*nb0);
        }
    }
}

শূন্য ggml_compute_forward_add(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
        পক্ষ GGML_TYPE_F16:
        পক্ষ GGML_TYPE_BF16:
            {
                ggml_compute_forward_add_non_quantized(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_Q1_0:
        পক্ষ GGML_TYPE_Q2_0:
        পক্ষ GGML_TYPE_Q4_0:
        পক্ষ GGML_TYPE_Q4_1:
        পক্ষ GGML_TYPE_Q5_0:
        পক্ষ GGML_TYPE_Q5_1:
        পক্ষ GGML_TYPE_Q8_0:
        পক্ষ GGML_TYPE_MXFP4:
        পক্ষ GGML_TYPE_NVFP4:
        পক্ষ GGML_TYPE_Q2_K:
        পক্ষ GGML_TYPE_Q3_K:
        পক্ষ GGML_TYPE_Q4_K:
        পক্ষ GGML_TYPE_Q5_K:
        পক্ষ GGML_TYPE_Q6_K:
        পক্ষ GGML_TYPE_TQ1_0:
        পক্ষ GGML_TYPE_TQ2_0:
        পক্ষ GGML_TYPE_IQ2_XXS:
        পক্ষ GGML_TYPE_IQ2_XS:
        পক্ষ GGML_TYPE_IQ3_XXS:
        পক্ষ GGML_TYPE_IQ1_S:
        পক্ষ GGML_TYPE_IQ1_M:
        পক্ষ GGML_TYPE_IQ4_NL:
        পক্ষ GGML_TYPE_IQ4_XS:
        পক্ষ GGML_TYPE_IQ3_S:
        পক্ষ GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_add_q_f32(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_add_id

স্থির শূন্য ggml_compute_forward_add_id_f32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];
    ধ্রুব ggml_tensor * src2 = dst->src[2];

    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src2->type == GGML_TYPE_I32);

    GGML_ASSERT(src0->nb[0] == পরিমাণ(ভিন্ন));
    GGML_ASSERT(src1->nb[0] == পরিমাণ(ভিন্ন));

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক nr  = ggml_nrows(src0);

    GGML_TENSOR_TERNARY_OP_LOCALS

    GGML_ASSERT( nb0 == পরিমাণ(ভিন্ন));
    GGML_ASSERT(nb10 == পরিমাণ(ভিন্ন));

    // rows per thread
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr*ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    প্রতি (পূর্ণাঙ্ক ir = ir0; ir < ir1; ++ir) {
        // src0 indices
        ধ্রুব পূর্ণাঙ্ক i3 = ir/(ne2*ne1);
        ধ্রুব পূর্ণাঙ্ক i2 = (ir - i3*ne2*ne1)/ne1;
        ধ্রুব পূর্ণাঙ্ক i1 = (ir - i3*ne2*ne1 - i2*ne1);

        // src1 indices
        ধ্রুব পূর্ণাঙ্ক i11 = *(int32_t *) ((বর্ণ *) src2->data + i1*nb20 + i2*nb21);

        GGML_ASSERT(i11 >= 0 && i11 < ne11);

        ggml_vec_add_f32(ne0,
                (ভিন্ন *) ((বর্ণ *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 ),
                (ভিন্ন *) ((বর্ণ *) src0->data + i3*nb03 + i2*nb02 + i1*nb01),
                (ভিন্ন *) ((বর্ণ *) src1->data + i11*nb11));
    }
}

শূন্য ggml_compute_forward_add_id(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_add_id_f32(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("unsupported type for ggml_compute_forward_add_id: %s", ggml_type_name(src0->type));
            }
    }
}

// ggml_compute_forward_add1

স্থির শূন্য ggml_compute_forward_add1_f32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == পরিমাণ(ভিন্ন));
    GGML_ASSERT(nb00 == পরিমাণ(ভিন্ন));

    // rows per thread
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr*ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    প্রতি (পূর্ণাঙ্ক ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        ধ্রুব পূর্ণাঙ্ক i3 = ir/(ne2*ne1);
        ধ্রুব পূর্ণাঙ্ক i2 = (ir - i3*ne2*ne1)/ne1;
        ধ্রুব পূর্ণাঙ্ক i1 = (ir - i3*ne2*ne1 - i2*ne1);

#ifdef GGML_USE_ACCELERATE
        GGML_UNUSED(ggml_vec_add1_f32);

        vDSP_vadd(
                (ভিন্ন *) ((বর্ণ *) src0->data + i3*nb03 + i2*nb02 + i1*nb01), 1,
                (ভিন্ন *) ((বর্ণ *) src1->data), 0,
                (ভিন্ন *) ((বর্ণ *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 ), 1,
                ne0);
#অন্যথা
        ggml_vec_add1_f32(ne0,
                (ভিন্ন *) ((বর্ণ *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 ),
                (ভিন্ন *) ((বর্ণ *) src0->data + i3*nb03 + i2*nb02 + i1*nb01),
               *(ভিন্ন *) src1->data);
#endif
    }
}

স্থির শূন্য ggml_compute_forward_add1_f16_f32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    ধ্রুব ভিন্ন v = *(ভিন্ন *) src1->data;

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F16);

    GGML_ASSERT( nb0 == পরিমাণ(ggml_fp16_t));
    GGML_ASSERT(nb00 == পরিমাণ(ggml_fp16_t));

    // rows per thread
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr*ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    প্রতি (পূর্ণাঙ্ক ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        ধ্রুব পূর্ণাঙ্ক i3 = ir/(ne2*ne1);
        ধ্রুব পূর্ণাঙ্ক i2 = (ir - i3*ne2*ne1)/ne1;
        ধ্রুব পূর্ণাঙ্ক i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((বর্ণ *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((বর্ণ *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
        প্রতি (পূর্ণাঙ্ক i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(src0_ptr[i]) + v);
        }
    }
}

স্থির শূন্য ggml_compute_forward_add1_f16_f16(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    ধ্রুব ভিন্ন v = GGML_CPU_FP16_TO_FP32(*(ggml_fp16_t *) src1->data);

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type  == GGML_TYPE_F16);

    GGML_ASSERT( nb0 == পরিমাণ(ggml_fp16_t));
    GGML_ASSERT(nb00 == পরিমাণ(ggml_fp16_t));

    // rows per thread
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr*ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    প্রতি (পূর্ণাঙ্ক ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        ধ্রুব পূর্ণাঙ্ক i3 = ir/(ne2*ne1);
        ধ্রুব পূর্ণাঙ্ক i2 = (ir - i3*ne2*ne1)/ne1;
        ধ্রুব পূর্ণাঙ্ক i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((বর্ণ *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((বর্ণ *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
        প্রতি (পূর্ণাঙ্ক i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(src0_ptr[i]) + v);
        }
    }
}

স্থির শূন্য ggml_compute_forward_add1_q_f32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    ধ্রুব ভিন্ন v = *(ভিন্ন *) src1->data;

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    ধ্রুব ggml_type type = src0->type;
    ggml_to_float_t ধ্রুব dequantize_row_q = ggml_get_type_traits(type)->to_float;
    ggml_from_float_t ধ্রুব quantize_row_q = ggml_get_type_traits_cpu(type)->from_float;

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
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr*ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    ভিন্ন * wdata = (ভিন্ন *) params->wdata + (ne0 + CACHE_সারি_SIZE_F32) * ith;

    প্রতি (পূর্ণাঙ্ক ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        ধ্রুব পূর্ণাঙ্ক i3 = ir/(ne2*ne1);
        ধ্রুব পূর্ণাঙ্ক i2 = (ir - i3*ne2*ne1)/ne1;
        ধ্রুব পূর্ণাঙ্ক i1 = (ir - i3*ne2*ne1 - i2*ne1);

        শূন্য  * src0_row = (শূন্য *) ((বর্ণ *) src0->data + (i1*nb01 + i2*nb02 + i3*nb03));
        শূন্য  * dst_row  = (শূন্য *) ((বর্ণ *)  dst->data + (i1*nb1  + i2*nb2  + i3*nb0 ));

        প্রতিজ্ঞা(ne0 % 32 == 0);

        // unquantize row from src0 to temp buffer
        dequantize_row_q(src0_row, wdata, ne0);
        // add src1
        ggml_vec_acc1_f32(ne0, wdata, v);
        // quantize row to dst
        quantize_row_q(wdata, dst_row, ne0);
    }
}

স্থির শূন্য ggml_compute_forward_add1_bf16_f32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    ধ্রুব ভিন্ন v = *(ভিন্ন *) src1->data;

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_BF16);

    GGML_ASSERT( nb0 == পরিমাণ(ggml_bf16_t));
    GGML_ASSERT(nb00 == পরিমাণ(ggml_bf16_t));

    // rows per thread
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr*ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    প্রতি (পূর্ণাঙ্ক ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        ধ্রুব পূর্ণাঙ্ক i3 = ir/(ne2*ne1);
        ধ্রুব পূর্ণাঙ্ক i2 = (ir - i3*ne2*ne1)/ne1;
        ধ্রুব পূর্ণাঙ্ক i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_bf16_t * dst_ptr  = (ggml_bf16_t *) ((বর্ণ *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_bf16_t * src0_ptr = (ggml_bf16_t *) ((বর্ণ *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
        প্রতি (পূর্ণাঙ্ক i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(src0_ptr[i]) + v);
        }
    }
}

স্থির শূন্য ggml_compute_forward_add1_bf16_bf16(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    ধ্রুব ভিন্ন v = GGML_BF16_TO_FP32(*(ggml_bf16_t *) src1->data);

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(src1->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->type  == GGML_TYPE_BF16);

    GGML_ASSERT( nb0 == পরিমাণ(ggml_bf16_t));
    GGML_ASSERT(nb00 == পরিমাণ(ggml_bf16_t));

    // rows per thread
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr*ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    প্রতি (পূর্ণাঙ্ক ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        ধ্রুব পূর্ণাঙ্ক i3 = ir/(ne2*ne1);
        ধ্রুব পূর্ণাঙ্ক i2 = (ir - i3*ne2*ne1)/ne1;
        ধ্রুব পূর্ণাঙ্ক i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_bf16_t * dst_ptr  = (ggml_bf16_t *) ((বর্ণ *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_bf16_t * src0_ptr = (ggml_bf16_t *) ((বর্ণ *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
        প্রতি (পূর্ণাঙ্ক i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(src0_ptr[i]) + v);
        }
    }
}

শূন্য ggml_compute_forward_add1(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_add1_f32(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_F16:
            {
                যদি (src1->type == GGML_TYPE_F16) {
                    ggml_compute_forward_add1_f16_f16(params, dst);
                }
                অন্যথা যদি (src1->type == GGML_TYPE_F32) {
                    ggml_compute_forward_add1_f16_f32(params, dst);
                }
                অন্যথা {
                    GGML_ABORT("fatal error");
                }
            } ভঙ্গ;
        পক্ষ GGML_TYPE_BF16:
            {
                যদি (src1->type == GGML_TYPE_BF16) {
                    ggml_compute_forward_add1_bf16_bf16(params, dst);
                }
                অন্যথা যদি (src1->type == GGML_TYPE_F32) {
                    ggml_compute_forward_add1_bf16_f32(params, dst);
                }
                অন্যথা {
                    GGML_ABORT("fatal error");
                }
            } ভঙ্গ;
        পক্ষ GGML_TYPE_Q1_0:
        পক্ষ GGML_TYPE_Q2_0:
        পক্ষ GGML_TYPE_Q4_0:
        পক্ষ GGML_TYPE_Q4_1:
        পক্ষ GGML_TYPE_Q5_0:
        পক্ষ GGML_TYPE_Q5_1:
        পক্ষ GGML_TYPE_Q8_0:
        পক্ষ GGML_TYPE_Q8_1:
        পক্ষ GGML_TYPE_MXFP4:
        পক্ষ GGML_TYPE_NVFP4:
        পক্ষ GGML_TYPE_Q2_K:
        পক্ষ GGML_TYPE_Q3_K:
        পক্ষ GGML_TYPE_Q4_K:
        পক্ষ GGML_TYPE_Q5_K:
        পক্ষ GGML_TYPE_Q6_K:
        পক্ষ GGML_TYPE_TQ1_0:
        পক্ষ GGML_TYPE_TQ2_0:
        পক্ষ GGML_TYPE_IQ2_XXS:
        পক্ষ GGML_TYPE_IQ2_XS:
        পক্ষ GGML_TYPE_IQ3_XXS:
        পক্ষ GGML_TYPE_IQ1_S:
        পক্ষ GGML_TYPE_IQ1_M:
        পক্ষ GGML_TYPE_IQ4_NL:
        পক্ষ GGML_TYPE_IQ4_XS:
        পক্ষ GGML_TYPE_IQ3_S:
        পক্ষ GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_add1_q_f32(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_acc

স্থির শূন্য ggml_compute_forward_acc_f32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));

    // view src0 and dst with these strides and data offset inbytes during acc
    // nb0 is implicitly element_size because src0 and dst are contiguous
    size_t nb1     = ((int32_t *) dst->op_params)[0];
    size_t nb2     = ((int32_t *) dst->op_params)[1];
    size_t nb3     = ((int32_t *) dst->op_params)[2];
    size_t offset  = ((int32_t *) dst->op_params)[3];
    তার্কিক   inplace = (তার্কিক) ((int32_t *) dst->op_params)[4];

    যদি (!inplace) {
        যদি (params->ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            memcpy(
                ((বর্ণ *)  dst->data),
                ((বর্ণ *) src0->data),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক nr = ggml_nrows(src1);
    ধ্রুব পূর্ণাঙ্ক nc = src1->ne[0];

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne)
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb)

    // src0 and dst as viewed during acc
    ধ্রুব size_t nb0 = ggml_element_size(src0);

    ধ্রুব size_t nb00 = nb0;
    ধ্রুব size_t nb01 = nb1;
    ধ্রুব size_t nb02 = nb2;
    ধ্রুব size_t nb03 = nb3;

    GGML_ASSERT(offset + (ne10 == 0 ? 0 : ne10-1)*nb0  + (ne11 == 0 ? 0 : ne11-1)*nb1  + (ne12 == 0 ? 0 : ne12-1)*nb2  + (ne13 == 0 ? 0 : ne13-1)*nb3  < ggml_nbytes(dst));
    GGML_ASSERT(offset + (ne10 == 0 ? 0 : ne10-1)*nb00 + (ne11 == 0 ? 0 : ne11-1)*nb01 + (ne12 == 0 ? 0 : ne12-1)*nb02 + (ne13 == 0 ? 0 : ne13-1)*nb03 < ggml_nbytes(src0));

    GGML_ASSERT(nb10 == পরিমাণ(ভিন্ন));

    // rows per thread
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr*ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    প্রতি (পূর্ণাঙ্ক ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are viewed with shape of src1 and offset
        // => same indices
        ধ্রুব পূর্ণাঙ্ক i3 = ir/(ne12*ne11);
        ধ্রুব পূর্ণাঙ্ক i2 = (ir - i3*ne12*ne11)/ne11;
        ধ্রুব পূর্ণাঙ্ক i1 = (ir - i3*ne12*ne11 - i2*ne11);

#ifdef GGML_USE_ACCELERATE
        vDSP_vadd(
                (ভিন্ন *) ((বর্ণ *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + offset), 1,
                (ভিন্ন *) ((বর্ণ *) src1->data + i3*nb13 + i2*nb12 + i1*nb11), 1,
                (ভিন্ন *) ((বর্ণ *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1  + offset), 1, nc);
#অন্যথা
        ggml_vec_add_f32(nc,
                (ভিন্ন *) ((বর্ণ *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + offset),
                (ভিন্ন *) ((বর্ণ *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + offset),
                (ভিন্ন *) ((বর্ণ *) src1->data + i3*nb13 + i2*nb12 + i1*nb11));
#endif
    }
}

শূন্য ggml_compute_forward_acc(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_acc_f32(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_F16:
        পক্ষ GGML_TYPE_BF16:
        পক্ষ GGML_TYPE_Q1_0:
        পক্ষ GGML_TYPE_Q2_0:
        পক্ষ GGML_TYPE_Q4_0:
        পক্ষ GGML_TYPE_Q4_1:
        পক্ষ GGML_TYPE_Q5_0:
        পক্ষ GGML_TYPE_Q5_1:
        পক্ষ GGML_TYPE_Q8_0:
        পক্ষ GGML_TYPE_Q8_1:
        পক্ষ GGML_TYPE_MXFP4:
        পক্ষ GGML_TYPE_NVFP4:
        পক্ষ GGML_TYPE_Q2_K:
        পক্ষ GGML_TYPE_Q3_K:
        পক্ষ GGML_TYPE_Q4_K:
        পক্ষ GGML_TYPE_Q5_K:
        পক্ষ GGML_TYPE_Q6_K:
        পক্ষ GGML_TYPE_TQ1_0:
        পক্ষ GGML_TYPE_TQ2_0:
        পক্ষ GGML_TYPE_IQ2_XXS:
        পক্ষ GGML_TYPE_IQ2_XS:
        পক্ষ GGML_TYPE_IQ3_XXS:
        পক্ষ GGML_TYPE_IQ1_S:
        পক্ষ GGML_TYPE_IQ1_M:
        পক্ষ GGML_TYPE_IQ4_NL:
        পক্ষ GGML_TYPE_IQ4_XS:
        পক্ষ GGML_TYPE_IQ3_S:
        পক্ষ GGML_TYPE_IQ2_S:
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_sum

স্থির শূন্য ggml_compute_forward_sum_f32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    যদি (params->ith != 0) {
        প্রত্যাগম;
    }

    প্রতিজ্ঞা(ggml_is_scalar(dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)

    ggml_float sum = 0;

    যদি (nb00 == পরিমাণ(ভিন্ন)) {
        ggml_float row_sum = 0;
        প্রতি (int64_t i03 = 0; i03 < ne03; i03++) {
            প্রতি (int64_t i02 = 0; i02 < ne02; i02++) {
                প্রতি (int64_t i01 = 0; i01 < ne01; i01++) {
                    ggml_vec_sum_f32_ggf(ne00,
                            &row_sum,
                            (ভিন্ন *) ((বর্ণ *) src0->data + i01*nb01 + i02*nb02 + i03*nb03));
                    sum += row_sum;
                }
            }
        }
    } অন্যথা {
        প্রতি (int64_t i03 = 0; i03 < ne03; i03++) {
            প্রতি (int64_t i02 = 0; i02 < ne02; i02++) {
                প্রতি (int64_t i01 = 0; i01 < ne01; i01++) {
                    ধ্রুব বর্ণ * row = (ধ্রুব বর্ণ *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
                    প্রতি (int64_t i00 = 0; i00 < ne00; i00++) {
                        sum += (ggml_float) *(ধ্রুব ভিন্ন *) (row + i00*nb00);
                    }
                }
            }
        }
    }

    ((ভিন্ন *) dst->data)[0] = sum;
}

স্থির শূন্য ggml_compute_forward_sum_f16(
    ধ্রুব ggml_compute_params * params,
          ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    যদি (params->ith != 0) {
        প্রত্যাগম;
    }

    প্রতিজ্ঞা(ggml_is_scalar(dst));

    প্রতিজ্ঞা(src0->nb[0] == পরিমাণ(ggml_fp16_t));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)

    ভিন্ন sum = 0;
    ভিন্ন row_sum = 0;

    প্রতি (int64_t i03 = 0; i03 < ne03; i03++) {
        প্রতি (int64_t i02 = 0; i02 < ne02; i02++) {
            প্রতি (int64_t i01 = 0; i01 < ne01; i01++) {
                ggml_vec_sum_f16_ggf(ne00,
                    &row_sum,
                    (ggml_fp16_t *) ((বর্ণ *) src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03));
                sum += row_sum;
            }
        }
    }
    ((ggml_fp16_t *) dst->data)[0] = GGML_CPU_FP32_TO_FP16(sum);
}

স্থির শূন্য ggml_compute_forward_sum_bf16(
    ধ্রুব ggml_compute_params * params,
          ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    যদি (params->ith != 0) {
        প্রত্যাগম;
    }

    প্রতিজ্ঞা(ggml_is_scalar(dst));

    প্রতিজ্ঞা(src0->nb[0] == পরিমাণ(ggml_bf16_t));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)

    ভিন্ন sum = 0;
    ভিন্ন row_sum = 0;

    প্রতি (int64_t i03 = 0; i03 < ne03; i03++) {
        প্রতি (int64_t i02 = 0; i02 < ne02; i02++) {
            প্রতি (int64_t i01 = 0; i01 < ne01; i01++) {
                ggml_vec_sum_bf16_ggf(ne00,
                    &row_sum,
                    (ggml_bf16_t *) ((বর্ণ *) src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03));
                sum += row_sum;
            }
        }
    }
    ((ggml_bf16_t *) dst->data)[0] = GGML_FP32_TO_BF16(sum);
}

শূন্য ggml_compute_forward_sum(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_sum_f32(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_F16:
            {
                ggml_compute_forward_sum_f16(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_BF16:
            {
                ggml_compute_forward_sum_bf16(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_cumsum

স্থির শূন্য ggml_compute_forward_cumsum_f32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->nb[0] == পরিমাণ(ভিন্ন));
    GGML_ASSERT(dst->nb[0] == পরিমাণ(ভিন্ন));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne01);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    ধ্রুব auto [ir0, ir1] = get_thread_range(params, src0);

    প্রতি (int64_t ir = ir0; ir < ir1; ++ir) {
        ধ্রুব int64_t i03 = ir/(ne02*ne01);
        ধ্রুব int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        ধ্রুব int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        ভিন্ন * src_row = (ভিন্ন *) ((বর্ণ *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
        ভিন্ন * dst_row = (ভিন্ন *) ((বর্ণ *) dst->data  + i01*nb1  + i02*nb2  + i03*nb3);

        ggml_vec_cumsum_f32(ne00, dst_row, src_row);
    }
}

শূন্য ggml_compute_forward_cumsum(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_cumsum_f32(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_sum_rows

স্থির শূন্য ggml_compute_forward_sum_rows_f32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    যদি (params->ith != 0) {
        প্রত্যাগম;
    }

    GGML_ASSERT(src0->nb[0] == পরিমাণ(ভিন্ন));
    GGML_ASSERT(dst->nb[0] == পরিমাণ(ভিন্ন));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(ne0 == 1);
    GGML_ASSERT(ne1 == ne01);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    প্রতি (int64_t i3 = 0; i3 < ne03; i3++) {
        প্রতি (int64_t i2 = 0; i2 < ne02; i2++) {
            প্রতি (int64_t i1 = 0; i1 < ne01; i1++) {
                ভিন্ন * src_row = (ভিন্ন *) ((বর্ণ *) src0->data + i1*nb01 + i2*nb02 + i3*nb03);
                ভিন্ন * dst_row = (ভিন্ন *) ((বর্ণ *) dst->data  + i1*nb1  + i2*nb2  + i3*nb3);
                ভিন্ন row_sum = 0;
                ggml_vec_sum_f32(ne00, &row_sum, src_row);
                dst_row[0] = row_sum;
            }
        }
    }
}

শূন্য ggml_compute_forward_sum_rows(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_sum_rows_f32(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_mean

স্থির শূন্য ggml_compute_forward_mean_f32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    যদি (params->ith != 0) {
        প্রত্যাগম;
    }

    প্রতিজ্ঞা(src0->nb[0] == পরিমাণ(ভিন্ন));

    GGML_TENSOR_UNARY_OP_LOCALS

    প্রতিজ্ঞা(ne0 == 1);
    প্রতিজ্ঞা(ne1 == ne01);
    প্রতিজ্ঞা(ne2 == ne02);
    প্রতিজ্ঞা(ne3 == ne03);

    GGML_UNUSED(ne0);
    GGML_UNUSED(ne1);
    GGML_UNUSED(ne2);
    GGML_UNUSED(ne3);

    প্রতি (int64_t i03 = 0; i03 < ne03; i03++) {
        প্রতি (int64_t i02 = 0; i02 < ne02; i02++) {
            প্রতি (int64_t i01 = 0; i01 < ne01; i01++) {
                ggml_vec_sum_f32(ne00,
                        (ভিন্ন *) ((বর্ণ *)  dst->data + i01*nb1  + i02*nb2  + i03*nb3),
                        (ভিন্ন *) ((বর্ণ *) src0->data + i01*nb01 + i02*nb02 + i03*nb03));

                *(ভিন্ন *) ((বর্ণ *) dst->data + i01*nb1 + i02*nb2 + i03*nb3) /= (ভিন্ন) ne00;
            }
        }
    }
}

শূন্য ggml_compute_forward_mean(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_mean_f32(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_argmax

স্থির শূন্য ggml_compute_forward_argmax_f32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    যদি (params->ith != 0) {
        প্রত্যাগম;
    }

    প্রতিজ্ঞা(src0->nb[0] == পরিমাণ(ভিন্ন));
    প্রতিজ্ঞা(dst->nb[0] == পরিমাণ(ভিন্ন));

    ধ্রুব int64_t ne00 = src0->ne[0];
    ধ্রুব int64_t ne01 = src0->ne[1];

    ধ্রুব size_t nb01 = src0->nb[1];
    ধ্রুব size_t nb0 = dst->nb[0];

    প্রতি (int64_t i1 = 0; i1 < ne01; i1++) {
        ভিন্ন * src = (ভিন্ন *) ((বর্ণ *) src0->data + i1*nb01);
        int32_t * dst_ = (int32_t *) ((বর্ণ *)  dst->data + i1*nb0);
        পূর্ণাঙ্ক v = 0;
        ggml_vec_argmax_f32(ne00, &v, src);
        dst_[0] = v;
    }
}

শূন্য ggml_compute_forward_argmax(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_argmax_f32(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_count_equal

স্থির শূন্য ggml_compute_forward_গুনতি_equal_i32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_দ্বিমিক_OP_LOCALS;

    GGML_ASSERT(src0->type == GGML_TYPE_I32);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));
    GGML_ASSERT(dst->type == GGML_TYPE_I64);

    ধ্রুব int64_t nr = ggml_nrows(src0);

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    int64_t * sums = (int64_t *) params->wdata;
    int64_t sum_thread = 0;

    // rows per thread
    ধ্রুব int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব int64_t ir0 = dr*ith;
    ধ্রুব int64_t ir1 = MIN(ir0 + dr, nr);

    প্রতি (int64_t ir = ir0; ir < ir1; ++ir) {
        ধ্রুব int64_t i03 =  ir                        / (ne02*ne01);
        ধ্রুব int64_t i02 = (ir - i03*ne03)            /       ne01;
        ধ্রুব int64_t i01 =  ir - i03*ne03 - i02*ne02;

        ধ্রুব বর্ণ * data0 = (ধ্রুব বর্ণ *) src0->data + i03*nb03 + i02*nb02 + i01*nb01;
        ধ্রুব বর্ণ * data1 = (ধ্রুব বর্ণ *) src1->data + i03*nb13 + i02*nb12 + i01*nb11;

        প্রতি (int64_t i00 = 0; i00 < ne00; ++i00) {
            ধ্রুব int32_t val0 = *((ধ্রুব int32_t *) (data0 + i00*nb00));
            ধ্রুব int32_t val1 = *((ধ্রুব int32_t *) (data1 + i00*nb10));

            sum_thread += val0 == val1;
        }
    }
    যদি (ith != 0) {
        sums[ith] = sum_thread;
    }
    ggml_barrier(params->threadpool);

    যদি (ith != 0) {
        প্রত্যাগম;
    }

    প্রতি (পূর্ণাঙ্ক ith_other = 1; ith_other < nth; ++ith_other) {
        sum_thread += sums[ith_other];
    }
    *((int64_t *) dst->data) = sum_thread;
}

শূন্য ggml_compute_forward_গুনতি_equal(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_I32:
            {
                ggml_compute_forward_গুনতি_equal_i32(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_repeat

স্থির শূন্য ggml_compute_forward_repeat_f32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    যদি (params->ith != 0) {
        প্রত্যাগম;
    }

    GGML_ASSERT(ggml_can_repeat(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    // guaranteed to be an integer due to the check in ggml_can_repeat
    ধ্রুব পূর্ণাঙ্ক nr0 = (পূর্ণাঙ্ক)(ne0/ne00);
    ধ্রুব পূর্ণাঙ্ক nr1 = (পূর্ণাঙ্ক)(ne1/ne01);
    ধ্রুব পূর্ণাঙ্ক nr2 = (পূর্ণাঙ্ক)(ne2/ne02);
    ধ্রুব পূর্ণাঙ্ক nr3 = (পূর্ণাঙ্ক)(ne3/ne03);

    // TODO: support for transposed / permuted tensors
    GGML_ASSERT(nb0  == পরিমাণ(ভিন্ন));
    GGML_ASSERT(nb00 == পরিমাণ(ভিন্ন));

    // TODO: maybe this is not optimal?
    প্রতি                         (পূর্ণাঙ্ক i3 = 0; i3 < nr3;  i3++) {
        প্রতি                     (পূর্ণাঙ্ক k3 = 0; k3 < ne03; k3++) {
            প্রতি                 (পূর্ণাঙ্ক i2 = 0; i2 < nr2;  i2++) {
                প্রতি             (পূর্ণাঙ্ক k2 = 0; k2 < ne02; k2++) {
                    প্রতি         (পূর্ণাঙ্ক i1 = 0; i1 < nr1;  i1++) {
                        প্রতি     (পূর্ণাঙ্ক k1 = 0; k1 < ne01; k1++) {
                            প্রতি (পূর্ণাঙ্ক i0 = 0; i0 < nr0;  i0++) {
                                ggml_vec_cpy_f32(ne00,
                                        (ভিন্ন *) ((বর্ণ *)  dst->data + (i3*ne03 + k3)*nb3  + (i2*ne02 + k2)*nb2  + (i1*ne01 + k1)*nb1  + (i0*ne00)*nb0),
                                        (ভিন্ন *) ((বর্ণ *) src0->data + (          k3)*nb03 + (          k2)*nb02 + (          k1)*nb01));
                            }
                        }
                    }
                }
            }
        }
    }
}

স্থির শূন্য ggml_compute_forward_repeat_f16(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    যদি (params->ith != 0) {
        প্রত্যাগম;
    }

    GGML_ASSERT(ggml_can_repeat(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    // guaranteed to be an integer due to the check in ggml_can_repeat
    ধ্রুব পূর্ণাঙ্ক nr0 = (পূর্ণাঙ্ক)(ne0/ne00);
    ধ্রুব পূর্ণাঙ্ক nr1 = (পূর্ণাঙ্ক)(ne1/ne01);
    ধ্রুব পূর্ণাঙ্ক nr2 = (পূর্ণাঙ্ক)(ne2/ne02);
    ধ্রুব পূর্ণাঙ্ক nr3 = (পূর্ণাঙ্ক)(ne3/ne03);

    // TODO: support for transposed / permuted tensors
    GGML_ASSERT(nb0  == পরিমাণ(ggml_fp16_t));
    GGML_ASSERT(nb00 == পরিমাণ(ggml_fp16_t));

    // TODO: maybe this is not optimal?
    প্রতি                         (পূর্ণাঙ্ক i3 = 0; i3 < nr3;  i3++) {
        প্রতি                     (পূর্ণাঙ্ক k3 = 0; k3 < ne03; k3++) {
            প্রতি                 (পূর্ণাঙ্ক i2 = 0; i2 < nr2;  i2++) {
                প্রতি             (পূর্ণাঙ্ক k2 = 0; k2 < ne02; k2++) {
                    প্রতি         (পূর্ণাঙ্ক i1 = 0; i1 < nr1;  i1++) {
                        প্রতি     (পূর্ণাঙ্ক k1 = 0; k1 < ne01; k1++) {
                            প্রতি (পূর্ণাঙ্ক i0 = 0; i0 < nr0;  i0++) {
                                ggml_fp16_t * y = (ggml_fp16_t *) ((বর্ণ *)  dst->data + (i3*ne03 + k3)*nb3  + (i2*ne02 + k2)*nb2  + (i1*ne01 + k1)*nb1  + (i0*ne00)*nb0);
                                ggml_fp16_t * x = (ggml_fp16_t *) ((বর্ণ *) src0->data + (          k3)*nb03 + (          k2)*nb02 + (          k1)*nb01);
                                // ggml_vec_cpy_f16(ne00, y, x)
                                প্রতি (পূর্ণাঙ্ক i = 0; i < ne00; ++i) {
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

শূন্য ggml_compute_forward_repeat(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F16:
        পক্ষ GGML_TYPE_BF16:
        পক্ষ GGML_TYPE_I16:
            {
                ggml_compute_forward_repeat_f16(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_F32:
        পক্ষ GGML_TYPE_I32:
            {
                ggml_compute_forward_repeat_f32(params, dst);
            } ভঙ্গ;
        // TODO: templateify the implementation and support for I64
        //       ref https://github.com/ggml-org/llama.cpp/pull/14274#discussion_r2169492225
        //case GGML_TYPE_I64:
        //    {
        //        ggml_compute_forward_repeat_i64(params, dst);
        //    } break;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_repeat_back

স্থির শূন্য ggml_compute_forward_repeat_back_f32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    যদি (params->ith != 0) {
        প্রত্যাগম;
    }

    GGML_ASSERT(ggml_can_repeat(dst, src0));

    GGML_TENSOR_UNARY_OP_LOCALS

    // guaranteed to be an integer due to the check in ggml_can_repeat
    ধ্রুব পূর্ণাঙ্ক nr0 = (পূর্ণাঙ্ক)(ne00/ne0);
    ধ্রুব পূর্ণাঙ্ক nr1 = (পূর্ণাঙ্ক)(ne01/ne1);
    ধ্রুব পূর্ণাঙ্ক nr2 = (পূর্ণাঙ্ক)(ne02/ne2);
    ধ্রুব পূর্ণাঙ্ক nr3 = (পূর্ণাঙ্ক)(ne03/ne3);

    // TODO: support for transposed / permuted tensors
    GGML_ASSERT(nb0  == পরিমাণ(ভিন্ন));
    GGML_ASSERT(nb00 == পরিমাণ(ভিন্ন));

    যদি (ggml_is_contiguous(dst)) {
        ggml_vec_set_f32(ne0*ne1*ne2*ne3, (ভিন্ন *)dst->data, 0);
    } অন্যথা {
        প্রতি         (পূর্ণাঙ্ক k3 = 0; k3 < ne3; k3++) {
            প্রতি     (পূর্ণাঙ্ক k2 = 0; k2 < ne2; k2++) {
                প্রতি (পূর্ণাঙ্ক k1 = 0; k1 < ne1; k1++) {
                    ggml_vec_set_f32(ne0,
                        (ভিন্ন *) ((বর্ণ *) dst->data + k1*nb1 + k2*nb2 + k3*nb3),
                        0);
                }
            }
        }
    }

    // TODO: maybe this is not optimal?
    প্রতি                         (পূর্ণাঙ্ক i3 = 0; i3 < nr3; i3++) {
        প্রতি                     (পূর্ণাঙ্ক k3 = 0; k3 < ne3; k3++) {
            প্রতি                 (পূর্ণাঙ্ক i2 = 0; i2 < nr2; i2++) {
                প্রতি             (পূর্ণাঙ্ক k2 = 0; k2 < ne2; k2++) {
                    প্রতি         (পূর্ণাঙ্ক i1 = 0; i1 < nr1; i1++) {
                        প্রতি     (পূর্ণাঙ্ক k1 = 0; k1 < ne1; k1++) {
                            প্রতি (পূর্ণাঙ্ক i0 = 0; i0 < nr0; i0++) {
                                ggml_vec_acc_f32(ne0,
                                        (ভিন্ন *) ((বর্ণ *)  dst->data + (         k3)*nb3  + (         k2)*nb2  + (         k1)*nb1),
                                        (ভিন্ন *) ((বর্ণ *) src0->data + (i3*ne3 + k3)*nb03 + (i2*ne2 + k2)*nb02 + (i1*ne1 + k1)*nb01 + (i0*ne0)*nb00));
                            }
                        }
                    }
                }
            }
        }
    }
}

শূন্য ggml_compute_forward_repeat_back(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_repeat_back_f32(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_concat
স্থির শূন্য ggml_compute_forward_concat_any(
    ধ্রুব ggml_compute_params * params,
    ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    GGML_TENSOR_দ্বিমিক_OP_LOCALS

    ধ্রুব int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);
    GGML_ASSERT(ggml_is_contiguous_rows(src0));
    GGML_ASSERT(ggml_is_contiguous_rows(src1));

    int64_t o[4] = {0, 0, 0, 0};

    যদি (dim == 0) {
        GGML_ASSERT(src0->ne[0] % ggml_blck_size(src0->type) == 0);
        GGML_ASSERT(src1->ne[0] % ggml_blck_size(src1->type) == 0);

        o[dim] = src0->ne[dim]/ggml_blck_size(src0->type);
    } অন্যথা {
        o[dim] = src0->ne[dim];
    }

    // Region 1: copy rows from src0
    প্রতি (পূর্ণাঙ্ক i3 = 0; i3 < ne03; i3++) {
        প্রতি (পূর্ণাঙ্ক i2 = ith; i2 < ne02; i2 += nth) {
            প্রতি (পূর্ণাঙ্ক i1 = 0; i1 < ne01; i1++) {
                ধ্রুব বর্ণ * x = (ধ্রুব বর্ণ *) src0->data + i1*nb01 + i2*nb02 + i3*nb03;
                      বর্ণ * y = (      বর্ণ *) dst->data  + i1*nb1  + i2*nb2  + i3*nb3;
                memcpy(y, x, ggml_row_size(src0->type, ne00));
            }
        }
    }

    // Region 2: copy rows from src1, offset into dst by o[]
    প্রতি (পূর্ণাঙ্ক i3 = 0; i3 < ne13; i3++) {
        প্রতি (পূর্ণাঙ্ক i2 = ith; i2 < ne12; i2 += nth) {
            প্রতি (পূর্ণাঙ্ক i1 = 0; i1 < ne11; i1++) {
                ধ্রুব বর্ণ * x = (ধ্রুব বর্ণ *) src1->data + i1*nb11         + i2*nb12         + i3*nb13;
                      বর্ণ * y = (      বর্ণ *)  dst->data + (i1 + o[1])*nb1 + (i2 + o[2])*nb2 + (i3 + o[3])*nb3 + o[0]*nb0;
                memcpy(y, x, ggml_row_size(src1->type, ne10));
            }
        }
    }
}

স্থির শূন্য ggml_compute_forward_concat_i8(
    ধ্রুব ggml_compute_params * params,
    ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_type_size(src0->type) == পরিমাণ(int8_t));

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    GGML_TENSOR_দ্বিমিক_OP_LOCALS

    ধ্রুব int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);

    int64_t o[4] = {0, 0, 0, 0};
    o[dim] = src0->ne[dim];

    ধ্রুব int8_t * x;

    // TODO: smarter multi-theading
    প্রতি (পূর্ণাঙ্ক i3 = 0; i3 < ne3; i3++) {
        প্রতি (পূর্ণাঙ্ক i2 = ith; i2 < ne2; i2 += nth) {
            প্রতি (পূর্ণাঙ্ক i1 = 0; i1 < ne1; i1++) {
                প্রতি (পূর্ণাঙ্ক i0 = 0; i0 < ne0; i0++) {
                    যদি (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                        x = (ধ্রুব int8_t *) ((ধ্রুব বর্ণ *)src0->data + (i0       )*nb00 + (i1       )*nb01 + (i2       )*nb02 + (i3       )*nb03);
                    } অন্যথা {
                        x = (ধ্রুব int8_t *) ((ধ্রুব বর্ণ *)src1->data + (i0 - o[0])*nb10 + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13);
                    }

                    int8_t * y = (int8_t *)((বর্ণ *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);

                    *y = *x;
                }
            }
        }
    }
}

স্থির শূন্য ggml_compute_forward_concat_f16(
    ধ্রুব ggml_compute_params * params,
    ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_type_size(src0->type) == পরিমাণ(ggml_fp16_t));

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    GGML_TENSOR_দ্বিমিক_OP_LOCALS

    ধ্রুব int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);

    int64_t o[4] = {0, 0, 0, 0};
    o[dim] = src0->ne[dim];

    ধ্রুব ggml_fp16_t * x;

    // TODO: smarter multi-theading
    প্রতি (পূর্ণাঙ্ক i3 = 0; i3 < ne3; i3++) {
        প্রতি (পূর্ণাঙ্ক i2 = ith; i2 < ne2; i2 += nth) {
            প্রতি (পূর্ণাঙ্ক i1 = 0; i1 < ne1; i1++) {
                প্রতি (পূর্ণাঙ্ক i0 = 0; i0 < ne0; i0++) {
                    যদি (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                        x = (ধ্রুব ggml_fp16_t *) ((ধ্রুব বর্ণ *)src0->data + (i0       )*nb00 + (i1       )*nb01 + (i2       )*nb02 + (i3       )*nb03);
                    } অন্যথা {
                        x = (ধ্রুব ggml_fp16_t *) ((ধ্রুব বর্ণ *)src1->data + (i0 - o[0])*nb10 + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13);
                    }

                    ggml_fp16_t * y = (ggml_fp16_t *)((বর্ণ *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);

                    *y = *x;
                }
            }
        }
    }
}

স্থির শূন্য ggml_compute_forward_concat_f32(
    ধ্রুব ggml_compute_params * params,
    ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_type_size(src0->type) == পরিমাণ(ভিন্ন));

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    GGML_TENSOR_দ্বিমিক_OP_LOCALS

    ধ্রুব int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);

    int64_t o[4] = {0, 0, 0, 0};
    o[dim] = src0->ne[dim];

    ধ্রুব ভিন্ন * x;

    // TODO: smarter multi-theading
    প্রতি (পূর্ণাঙ্ক i3 = 0; i3 < ne3; i3++) {
        প্রতি (পূর্ণাঙ্ক i2 = ith; i2 < ne2; i2 += nth) {
            প্রতি (পূর্ণাঙ্ক i1 = 0; i1 < ne1; i1++) {
                প্রতি (পূর্ণাঙ্ক i0 = 0; i0 < ne0; i0++) {
                    যদি (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                        x = (ধ্রুব ভিন্ন *) ((ধ্রুব বর্ণ *)src0->data + (i0       )*nb00 + (i1       )*nb01 + (i2       )*nb02 + (i3       )*nb03);
                    } অন্যথা {
                        x = (ধ্রুব ভিন্ন *) ((ধ্রুব বর্ণ *)src1->data + (i0 - o[0])*nb10 + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13);
                    }

                    ভিন্ন * y = (ভিন্ন *)((বর্ণ *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);

                    *y = *x;
                }
            }
        }
    }
}

শূন্য ggml_compute_forward_concat(
    ধ্রুব ggml_compute_params * params,
    ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F16:
        পক্ষ GGML_TYPE_BF16:
        পক্ষ GGML_TYPE_I16:
            {
                ggml_compute_forward_concat_f16(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_I8:
            {
                ggml_compute_forward_concat_i8(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_F32:
        পক্ষ GGML_TYPE_I32:
            {
                ggml_compute_forward_concat_f32(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                ggml_compute_forward_concat_any(params, dst);
            }
    }
}

// row loop shared by the BF16 unary ops
template <শূন্য (*vec_op)(ধ্রুব পূর্ণাঙ্ক, ggml_bf16_t *, ধ্রুব ggml_bf16_t *)>
স্থির শূন্য ggml_compute_forward_unary_bf16_rows(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    প্রতিজ্ঞা(ggml_is_contiguous_rows(src0));
    প্রতিজ্ঞা(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক nc = src0->ne[0];
    ধ্রুব পূর্ণাঙ্ক nr = ggml_nrows(src0);

    // rows per thread
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr*ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    প্রতি (পূর্ণাঙ্ক ir = ir0; ir < ir1; ++ir) {
        ধ্রুব পূর্ণাঙ্ক i3 = ir/(ne02*ne01);
        ধ্রুব পূর্ণাঙ্ক i2 = (ir - i3*ne02*ne01)/ne01;
        ধ্রুব পূর্ণাঙ্ক i1 = (ir - i3*ne02*ne01 - i2*ne01);

        vec_op(nc,
                (ggml_bf16_t *) ((বর্ণ *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_bf16_t *) ((বর্ণ *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));
    }
}

// ggml_compute_forward_gelu

স্থির শূন্য ggml_compute_forward_gelu_f32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    প্রতিজ্ঞা(ggml_is_contiguous_rows(src0));
    প্রতিজ্ঞা(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক nc = src0->ne[0];
    ধ্রুব পূর্ণাঙ্ক nr = ggml_nrows(src0);

    // rows per thread
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr*ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    প্রতি (পূর্ণাঙ্ক ir = ir0; ir < ir1; ++ir) {
        ধ্রুব পূর্ণাঙ্ক i3 = ir/(ne02*ne01);
        ধ্রুব পূর্ণাঙ্ক i2 = (ir - i3*ne02*ne01)/ne01;
        ধ্রুব পূর্ণাঙ্ক i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_f32(nc,
                (ভিন্ন *) ((বর্ণ *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ভিন্ন *) ((বর্ণ *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        প্রতি (পূর্ণাঙ্ক k = 0; k < nc; k++) {
            ধ্রুব ভিন্ন x = ((ভিন্ন *) ((বর্ণ *) dst->data + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_UNUSED(x);
            প্রতিজ্ঞা(!isnan(x));
            প্রতিজ্ঞা(!isinf(x));
        }
#endif // NDEBUG
    }
}

স্থির শূন্য ggml_compute_forward_gelu_f16(
    ধ্রুব ggml_compute_params * params,
    ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    প্রতিজ্ঞা(ggml_is_contiguous_rows(src0));
    প্রতিজ্ঞা(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক nc = src0->ne[0];
    ধ্রুব পূর্ণাঙ্ক nr = ggml_nrows(src0);

    // rows per thread
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr*ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    প্রতি (পূর্ণাঙ্ক ir = ir0; ir < ir1; ++ir) {
        ধ্রুব পূর্ণাঙ্ক i3 = ir/(ne02*ne01);
        ধ্রুব পূর্ণাঙ্ক i2 = (ir - i3*ne02*ne01)/ne01;
        ধ্রুব পূর্ণাঙ্ক i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_f16(nc,
                (ggml_fp16_t *) ((বর্ণ *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((বর্ণ *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        প্রতি (পূর্ণাঙ্ক k = 0; k < nc; k++) {
            ধ্রুব ggml_fp16_t x = ((ggml_fp16_t *) ((বর্ণ *) dst->data + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            ধ্রুব ভিন্ন v = GGML_CPU_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            প্রতিজ্ঞা(!isnan(v));
            প্রতিজ্ঞা(!isinf(v));
        }
#endif // NDEBUG
    }
}

স্থির শূন্য ggml_compute_forward_gelu_bf16(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_rows<ggml_vec_gelu_bf16>(params, dst);
}

স্থির শূন্য ggml_compute_forward_gelu(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_gelu_f32(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_F16:
            {
                ggml_compute_forward_gelu_f16(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_BF16:
            {
                ggml_compute_forward_gelu_bf16(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_fill

স্থির শূন্য ggml_compute_forward_fill_f32(ধ্রুব ggml_compute_params * params, ggml_tensor * dst) {
    ধ্রুব ভিন্ন c = ggml_get_op_params_f32(dst, 0);

    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne);
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb);

    ধ্রুব auto [ir0, ir1] = get_thread_range(params, dst);

    প্রতি (int64_t ir = ir0; ir < ir1; ++ir) {
        ধ্রুব int64_t i03 = ir/(ne2*ne1);
        ধ্রুব int64_t i02 = (ir - i03*ne2*ne1)/ne1;
        ধ্রুব int64_t i01 = (ir - i03*ne2*ne1 - i02*ne1);

        ভিন্ন * dst_ptr  = (ভিন্ন *) ((বর্ণ *) dst->data + i03*nb3 + i02*nb2 + i01*nb1);

        ggml_vec_set_f32(ne0, dst_ptr, c);
    }
}

স্থির শূন্য ggml_compute_forward_fill_f16(ধ্রুব ggml_compute_params * params, ggml_tensor * dst) {
    ধ্রুব ggml_fp16_t c = GGML_CPU_FP32_TO_FP16(ggml_get_op_params_f32(dst, 0));

    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne);
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb);

    ধ্রুব auto [ir0, ir1] = get_thread_range(params, dst);

    প্রতি (int64_t ir = ir0; ir < ir1; ++ir) {
        ধ্রুব int64_t i03 = ir/(ne2*ne1);
        ধ্রুব int64_t i02 = (ir - i03*ne2*ne1)/ne1;
        ধ্রুব int64_t i01 = (ir - i03*ne2*ne1 - i02*ne1);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((বর্ণ *) dst->data + i03*nb3 + i02*nb2 + i01*nb1);

        ggml_vec_set_f16(ne0, dst_ptr, c);
    }
}

শূন্য ggml_compute_forward_fill(ধ্রুব ggml_compute_params * params, ggml_tensor * dst) {
    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_fill_f32(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_F16:
            {
                ggml_compute_forward_fill_f16(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("unsupported type for ggml_compute_forward_fill: %s", ggml_type_name(src0->type));
            }
    }
}

// ggml_compute_tri

স্থির শূন্য ggml_compute_forward_tri_f32(ধ্রুব ggml_compute_params * params, ggml_tensor * dst) {
    ধ্রুব ggml_tensor * src0 = dst->src[0];

    ধ্রুব ggml_tri_type ttype = (ggml_tri_type) ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_TENSOR_UNARY_OP_LOCALS

    ধ্রুব auto [ir0, ir1] = get_thread_range(params, src0);

    তার্কিক (*bipred)(পূর্ণাঙ্ক, পূর্ণাঙ্ক);

    বিকল্প (ttype) {
        পক্ষ GGML_TRI_TYPE_LOWER:      bipred = [](পূর্ণাঙ্ক i, পূর্ণাঙ্ক r) { প্রত্যাগম i <  r; }; ভঙ্গ;
        পক্ষ GGML_TRI_TYPE_LOWER_DIAG: bipred = [](পূর্ণাঙ্ক i, পূর্ণাঙ্ক r) { প্রত্যাগম i <= r; }; ভঙ্গ;
        পক্ষ GGML_TRI_TYPE_UPPER:      bipred = [](পূর্ণাঙ্ক i, পূর্ণাঙ্ক r) { প্রত্যাগম i >  r; }; ভঙ্গ;
        পক্ষ GGML_TRI_TYPE_UPPER_DIAG: bipred = [](পূর্ণাঙ্ক i, পূর্ণাঙ্ক r) { প্রত্যাগম i >= r; }; ভঙ্গ;
        স্বভাব: GGML_ABORT("invalid tri type");
    }

    প্রতি (int64_t ir = ir0; ir < ir1; ++ir) {
        ধ্রুব int64_t i03 = ir/(ne02*ne01);
        ধ্রুব int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        ধ্রুব int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        ধ্রুব ভিন্ন * src_ptr = (ধ্রুব ভিন্ন  *) ((ধ্রুব বর্ণ *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);
              ভিন্ন * dst_ptr = (      ভিন্ন  *) ((      বর্ণ *) dst->data  + i03*nb3  + i02*nb2  + i01*nb1);

        প্রতি (পূর্ণাঙ্ক i0 = 0; i0 < ne0; ++i0) {
            dst_ptr[i0] = bipred(i0, i01) ? src_ptr[i0] : 0.0f;
        }
    }
}

শূন্য ggml_compute_forward_tri(ধ্রুব ggml_compute_params * params, ggml_tensor * dst) {
    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_tri_f32(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_gelu_erf

স্থির শূন্য ggml_compute_forward_gelu_erf_f32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    প্রতিজ্ঞা(ggml_is_contiguous_rows(src0));
    প্রতিজ্ঞা(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক nc = src0->ne[0];
    ধ্রুব পূর্ণাঙ্ক nr = ggml_nrows(src0);

    // rows per thread
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr*ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    প্রতি (পূর্ণাঙ্ক ir = ir0; ir < ir1; ++ir) {
        ধ্রুব পূর্ণাঙ্ক i3 = ir/(ne02*ne01);
        ধ্রুব পূর্ণাঙ্ক i2 = (ir - i3*ne02*ne01)/ne01;
        ধ্রুব পূর্ণাঙ্ক i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_erf_f32(nc,
                (ভিন্ন *) ((বর্ণ *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ভিন্ন *) ((বর্ণ *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        প্রতি (পূর্ণাঙ্ক k = 0; k < nc; k++) {
            ধ্রুব ভিন্ন x = ((ভিন্ন *) ((বর্ণ *) dst->data + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_UNUSED(x);
            প্রতিজ্ঞা(!isnan(x));
            প্রতিজ্ঞা(!isinf(x));
        }
#endif // NDEBUG
    }
}

স্থির শূন্য ggml_compute_forward_gelu_erf_f16(
    ধ্রুব ggml_compute_params * params,
    ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    প্রতিজ্ঞা(ggml_is_contiguous_rows(src0));
    প্রতিজ্ঞা(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক nc = src0->ne[0];
    ধ্রুব পূর্ণাঙ্ক nr = ggml_nrows(src0);

    // rows per thread
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr*ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    প্রতি (পূর্ণাঙ্ক ir = ir0; ir < ir1; ++ir) {
        ধ্রুব পূর্ণাঙ্ক i3 = ir/(ne02*ne01);
        ধ্রুব পূর্ণাঙ্ক i2 = (ir - i3*ne02*ne01)/ne01;
        ধ্রুব পূর্ণাঙ্ক i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_erf_f16(nc,
                (ggml_fp16_t *) ((বর্ণ *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((বর্ণ *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        প্রতি (পূর্ণাঙ্ক k = 0; k < nc; k++) {
            ধ্রুব ggml_fp16_t x = ((ggml_fp16_t *) ((বর্ণ *) dst->data + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            ধ্রুব ভিন্ন v = GGML_CPU_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            প্রতিজ্ঞা(!isnan(v));
            প্রতিজ্ঞা(!isinf(v));
        }
#endif // NDEBUG
    }
}

স্থির শূন্য ggml_compute_forward_gelu_erf_bf16(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_rows<ggml_vec_gelu_erf_bf16>(params, dst);
}

স্থির শূন্য ggml_compute_forward_gelu_erf(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_gelu_erf_f32(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_F16:
            {
                ggml_compute_forward_gelu_erf_f16(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_BF16:
            {
                ggml_compute_forward_gelu_erf_bf16(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_gelu_quick

স্থির শূন্য ggml_compute_forward_gelu_quick_f32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    প্রতিজ্ঞা(ggml_is_contiguous_rows(src0));
    প্রতিজ্ঞা(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক nc = src0->ne[0];
    ধ্রুব পূর্ণাঙ্ক nr = ggml_nrows(src0);

    // rows per thread
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr*ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    প্রতি (পূর্ণাঙ্ক ir = ir0; ir < ir1; ++ir) {
        ধ্রুব পূর্ণাঙ্ক i3 = ir/(ne02*ne01);
        ধ্রুব পূর্ণাঙ্ক i2 = (ir - i3*ne02*ne01)/ne01;
        ধ্রুব পূর্ণাঙ্ক i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_quick_f32(nc,
                (ভিন্ন *) ((বর্ণ *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ভিন্ন *) ((বর্ণ *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        প্রতি (পূর্ণাঙ্ক k = 0; k < nc; k++) {
            ধ্রুব ভিন্ন x = ((ভিন্ন *) ((বর্ণ *) dst->data + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_UNUSED(x);
            প্রতিজ্ঞা(!isnan(x));
            প্রতিজ্ঞা(!isinf(x));
        }
#endif // NDEBUG
    }
}

স্থির শূন্য ggml_compute_forward_gelu_quick_f16(
    ধ্রুব ggml_compute_params * params,
    ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    প্রতিজ্ঞা(ggml_is_contiguous_rows(src0));
    প্রতিজ্ঞা(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক nc = src0->ne[0];
    ধ্রুব পূর্ণাঙ্ক nr = ggml_nrows(src0);

    // rows per thread
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr*ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    প্রতি (পূর্ণাঙ্ক ir = ir0; ir < ir1; ++ir) {
        ধ্রুব পূর্ণাঙ্ক i3 = ir/(ne02*ne01);
        ধ্রুব পূর্ণাঙ্ক i2 = (ir - i3*ne02*ne01)/ne01;
        ধ্রুব পূর্ণাঙ্ক i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_quick_f16(nc,
                (ggml_fp16_t *) ((বর্ণ *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((বর্ণ *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        প্রতি (পূর্ণাঙ্ক k = 0; k < nc; k++) {
            ধ্রুব ggml_fp16_t x = ((ggml_fp16_t *) ((বর্ণ *) dst->data + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            ধ্রুব ভিন্ন v = GGML_CPU_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            প্রতিজ্ঞা(!isnan(v));
            প্রতিজ্ঞা(!isinf(v));
        }
#endif // NDEBUG
    }
}

স্থির শূন্য ggml_compute_forward_gelu_quick_bf16(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_rows<ggml_vec_gelu_quick_bf16>(params, dst);
}

স্থির শূন্য ggml_compute_forward_gelu_quick(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_gelu_quick_f32(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_F16:
            {
                ggml_compute_forward_gelu_quick_f16(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_BF16:
            {
                ggml_compute_forward_gelu_quick_bf16(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_silu

স্থির শূন্য ggml_compute_forward_silu_f32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    প্রতিজ্ঞা(ggml_is_contiguous_rows(src0));
    প্রতিজ্ঞা(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক nc = src0->ne[0];
    ধ্রুব পূর্ণাঙ্ক nr = ggml_nrows(src0);

    // rows per thread
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr*ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    প্রতি (পূর্ণাঙ্ক ir = ir0; ir < ir1; ++ir) {
        ধ্রুব পূর্ণাঙ্ক i3 = ir/(ne02*ne01);
        ধ্রুব পূর্ণাঙ্ক i2 = (ir - i3*ne02*ne01)/ne01;
        ধ্রুব পূর্ণাঙ্ক i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_silu_f32(nc,
                (ভিন্ন *) ((বর্ণ *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ভিন্ন *) ((বর্ণ *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        প্রতি (পূর্ণাঙ্ক k = 0; k < nc; k++) {
            ধ্রুব ভিন্ন x = ((ভিন্ন *) ((বর্ণ *) dst->data + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_UNUSED(x);
            প্রতিজ্ঞা(!isnan(x));
            প্রতিজ্ঞা(!isinf(x));
        }
#endif // NDEBUG
    }
}

স্থির শূন্য ggml_compute_forward_silu_f16(
    ধ্রুব ggml_compute_params * params,
    ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    প্রতিজ্ঞা(ggml_is_contiguous_rows(src0));
    প্রতিজ্ঞা(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক nc = src0->ne[0];
    ধ্রুব পূর্ণাঙ্ক nr = ggml_nrows(src0);

    // rows per thread
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr*ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    প্রতি (পূর্ণাঙ্ক ir = ir0; ir < ir1; ++ir) {
        ধ্রুব পূর্ণাঙ্ক i3 = ir/(ne02*ne01);
        ধ্রুব পূর্ণাঙ্ক i2 = (ir - i3*ne02*ne01)/ne01;
        ধ্রুব পূর্ণাঙ্ক i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_silu_f16(nc,
                (ggml_fp16_t *) ((বর্ণ *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((বর্ণ *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        প্রতি (পূর্ণাঙ্ক k = 0; k < nc; k++) {
            ধ্রুব ggml_fp16_t x = ((ggml_fp16_t *) ((বর্ণ *) dst->data + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            ধ্রুব ভিন্ন v = GGML_CPU_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            প্রতিজ্ঞা(!isnan(v));
            প্রতিজ্ঞা(!isinf(v));
        }
#endif // NDEBUG
    }
}

স্থির শূন্য ggml_compute_forward_silu_bf16(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_rows<ggml_vec_silu_bf16>(params, dst);
}

স্থির শূন্য ggml_compute_forward_silu(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_silu_f32(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_F16:
            {
                ggml_compute_forward_silu_f16(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_BF16:
            {
                ggml_compute_forward_silu_bf16(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}
// ggml_compute_forward_leaky_relu

স্থির শূন্য ggml_compute_forward_leaky_relu_f32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    যদি (params->ith != 0) {
        প্রত্যাগম;
    }

    প্রতিজ্ঞা(ggml_is_contiguous_1(src0));
    প্রতিজ্ঞা(ggml_is_contiguous_1(dst));
    প্রতিজ্ঞা(ggml_are_same_shape(src0, dst));

    ধ্রুব পূর্ণাঙ্ক n  = ggml_nrows(src0);
    ধ্রুব পূর্ণাঙ্ক nc = src0->ne[0];

    ভিন্ন negative_slope;
    memcpy(&negative_slope, dst->op_params, পরিমাণ(ভিন্ন));

    প্রতিজ্ঞা(dst->nb[0]  == পরিমাণ(ভিন্ন));
    প্রতিজ্ঞা(src0->nb[0] == পরিমাণ(ভিন্ন));

    প্রতি (পূর্ণাঙ্ক i = 0; i < n; i++) {
        ggml_vec_leaky_relu_f32(nc,
                (ভিন্ন *) ((বর্ণ *) dst->data  + i*( dst->nb[1])),
                (ভিন্ন *) ((বর্ণ *) src0->data + i*(src0->nb[1])), negative_slope);
    }
}

স্থির শূন্য ggml_compute_forward_leaky_relu_f16(
    ধ্রুব ggml_compute_params * params,
    ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    যদি (params->ith != 0) {
        প্রত্যাগম;
    }

    প্রতিজ্ঞা(ggml_is_contiguous_1(src0));
    প্রতিজ্ঞা(ggml_is_contiguous_1(dst));
    প্রতিজ্ঞা(ggml_are_same_shape(src0, dst));

    ধ্রুব পূর্ণাঙ্ক n  = ggml_nrows(src0);
    ধ্রুব পূর্ণাঙ্ক nc = src0->ne[0];

    ভিন্ন negative_slope;
    memcpy(&negative_slope, dst->op_params, পরিমাণ(ভিন্ন));

    প্রতিজ্ঞা(dst->nb[0]  == পরিমাণ(ggml_fp16_t));
    প্রতিজ্ঞা(src0->nb[0] == পরিমাণ(ggml_fp16_t));

    প্রতি (পূর্ণাঙ্ক i = 0; i < n; i++) {
        ggml_vec_leaky_relu_f16(nc,
                (ggml_fp16_t *) ((বর্ণ *) dst->data  + i*( dst->nb[1])),
                (ggml_fp16_t *) ((বর্ণ *) src0->data + i*(src0->nb[1])), negative_slope);
    }
}

শূন্য ggml_compute_forward_leaky_relu(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_leaky_relu_f32(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_F16:
            {
                ggml_compute_forward_leaky_relu_f16(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_silu_back

স্থির শূন্য ggml_compute_forward_silu_back_f32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * grad = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];

    প্রতিজ্ঞা(ggml_is_contiguous_1(grad));
    প্রতিজ্ঞা(ggml_is_contiguous_1(src1));
    প্রতিজ্ঞা(ggml_is_contiguous_1(dst));
    প্রতিজ্ঞা(ggml_are_same_shape(src1, dst));
    প্রতিজ্ঞা(ggml_are_same_shape(src1, grad));

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক nc = src1->ne[0];
    ধ্রুব পূর্ণাঙ্ক nr = ggml_nrows(src1);

    // rows per thread
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr*ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    প্রতি (পূর্ণাঙ্ক i1 = ir0; i1 < ir1; i1++) {
        ggml_vec_silu_backward_f32(nc,
                (ভিন্ন *) ((বর্ণ *) dst->data  + i1*( dst->nb[1])),
                (ভিন্ন *) ((বর্ণ *) src1->data + i1*(src1->nb[1])),
                (ভিন্ন *) ((বর্ণ *) grad->data + i1*(grad->nb[1])));

#ifndef NDEBUG
        প্রতি (পূর্ণাঙ্ক k = 0; k < nc; k++) {
            ধ্রুব ভিন্ন x = ((ভিন্ন *) ((বর্ণ *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            প্রতিজ্ঞা(!isnan(x));
            প্রতিজ্ঞা(!isinf(x));
        }
#endif // NDEBUG
    }
}

স্থির শূন্য ggml_compute_forward_silu_back_f16(
    ধ্রুব ggml_compute_params * params,
    ggml_tensor * dst) {

    ধ্রুব ggml_tensor * grad = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];

    প্রতিজ্ঞা(ggml_is_contiguous_1(grad));
    প্রতিজ্ঞা(ggml_is_contiguous_1(src1));
    প্রতিজ্ঞা(ggml_is_contiguous_1(dst));
    প্রতিজ্ঞা(ggml_are_same_shape(src1, dst));
    প্রতিজ্ঞা(ggml_are_same_shape(src1, grad));

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক nc = src1->ne[0];
    ধ্রুব পূর্ণাঙ্ক nr = ggml_nrows(src1);

    // rows per thread
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr*ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    প্রতি (পূর্ণাঙ্ক i1 = ir0; i1 < ir1; i1++) {
        ggml_vec_silu_backward_f16(nc,
                (ggml_fp16_t *) ((বর্ণ *) dst->data  + i1*( dst->nb[1])),
                (ggml_fp16_t *) ((বর্ণ *) src1->data + i1*(src1->nb[1])),
                (ggml_fp16_t *) ((বর্ণ *) grad->data + i1*(grad->nb[1])));

#ifndef NDEBUG
        প্রতি (পূর্ণাঙ্ক k = 0; k < nc; k++) {
            ধ্রুব ভিন্ন x = ((ggml_fp16_t *) ((বর্ণ *) dst->data + i1*( dst->nb[1])))[k];
            ধ্রুব ভিন্ন v = GGML_CPU_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            প্রতিজ্ঞা(!isnan(v));
            প্রতিজ্ঞা(!isinf(v));
        }
#endif // NDEBUG
    }
}

শূন্য ggml_compute_forward_silu_back(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_silu_back_f32(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_F16:
            {
                ggml_compute_forward_silu_back_f16(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_reglu

// row loop shared by the BF16 GLU ops
template <শূন্য (*vec_op)(ধ্রুব পূর্ণাঙ্ক, ggml_bf16_t *, ধ্রুব ggml_bf16_t *, ধ্রুব ggml_bf16_t *)>
স্থির শূন্য ggml_compute_forward_glu_bf16_rows(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];
    বর্ণ * src0_d = (বর্ণ *) src0->data;
    বর্ণ * src1_d = (বর্ণ *) (src1 ? src1->data : src0->data);
    ধ্রুব size_t src0_o = src0->nb[1];
    ধ্রুব size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    যদি (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ধ্রুব পূর্ণাঙ্ক nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ধ্রুব int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr*ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    প্রতি (পূর্ণাঙ্ক i1 = ir0; i1 < ir1; i1++) {
        ধ্রুব ggml_bf16_t * src0_p = (ধ্রুব ggml_bf16_t *) (src0_d + i1*src0_o);
        ধ্রুব ggml_bf16_t * src1_p = (ধ্রুব ggml_bf16_t *) (src1_d + i1*src1_o);
        ggml_bf16_t       * dst_p  = (ggml_bf16_t *) ((বর্ণ *) dst->data + i1*(dst->nb[1]));

        যদি (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        vec_op(nc, dst_p, src0_p, src1_p);
    }
}

স্থির শূন্য ggml_compute_forward_reglu_f32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];
    বর্ণ * src0_d = (বর্ণ *) src0->data;
    বর্ণ * src1_d = (বর্ণ *) (src1 ? src1->data : src0->data);
    ধ্রুব size_t src0_o = src0->nb[1];
    ধ্রুব size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    যদি (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ধ্রুব পূর্ণাঙ্ক nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ধ্রুব int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr*ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    প্রতি (পূর্ণাঙ্ক i1 = ir0; i1 < ir1; i1++) {
        ভিন্ন * src0_p = (ভিন্ন *) (src0_d + i1*src0_o);
        ভিন্ন * src1_p = (ভিন্ন *) (src1_d + i1*src1_o);

        যদি (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_reglu_f32(nc, (ভিন্ন *) ((বর্ণ *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        প্রতি (পূর্ণাঙ্ক k = 0; k < nc; k++) {
            ধ্রুব ভিন্ন x = ((ভিন্ন *) ((বর্ণ *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            প্রতিজ্ঞা(!isnan(x));
            প্রতিজ্ঞা(!isinf(x));
        }
#endif // NDEBUG
    }
}

স্থির শূন্য ggml_compute_forward_reglu_f16(
    ধ্রুব ggml_compute_params * params,
    ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];
    বর্ণ * src0_d = (বর্ণ *) src0->data;
    বর্ণ * src1_d = (বর্ণ *) (src1 ? src1->data : src0->data);
    ধ্রুব size_t src0_o = src0->nb[1];
    ধ্রুব size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    যদি (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ধ্রুব পূর্ণাঙ্ক nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ধ্রুব int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr*ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    প্রতি (পূর্ণাঙ্ক i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        যদি (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_reglu_f16(nc, (ggml_fp16_t *) ((বর্ণ *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        প্রতি (পূর্ণাঙ্ক k = 0; k < nc; k++) {
            ধ্রুব ggml_fp16_t x = ((ggml_fp16_t *) ((বর্ণ *) dst->data + i1*( dst->nb[1])))[k];
            ধ্রুব ভিন্ন v = GGML_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            প্রতিজ্ঞা(!isnan(v));
            প্রতিজ্ঞা(!isinf(v));
        }
#endif // NDEBUG
    }
}

স্থির শূন্য ggml_compute_forward_reglu_bf16(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_rows<ggml_vec_reglu_bf16>(params, dst);
}

স্থির শূন্য ggml_compute_forward_reglu(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_reglu_f32(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_F16:
            {
                ggml_compute_forward_reglu_f16(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_BF16:
            {
                ggml_compute_forward_reglu_bf16(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_geglu

স্থির শূন্য ggml_compute_forward_geglu_f32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];
    বর্ণ * src0_d = (বর্ণ *) src0->data;
    বর্ণ * src1_d = (বর্ণ *) (src1 ? src1->data : src0->data);
    ধ্রুব size_t src0_o = src0->nb[1];
    ধ্রুব size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    যদি (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ধ্রুব পূর্ণাঙ্ক nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ধ্রুব int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr*ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    প্রতি (পূর্ণাঙ্ক i1 = ir0; i1 < ir1; i1++) {
        ভিন্ন * src0_p = (ভিন্ন *) (src0_d + i1*src0_o);
        ভিন্ন * src1_p = (ভিন্ন *) (src1_d + i1*src1_o);

        যদি (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_f32(nc, (ভিন্ন *) ((বর্ণ *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        প্রতি (পূর্ণাঙ্ক k = 0; k < nc; k++) {
            ধ্রুব ভিন্ন x = ((ভিন্ন *) ((বর্ণ *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            প্রতিজ্ঞা(!isnan(x));
            প্রতিজ্ঞা(!isinf(x));
        }
#endif // NDEBUG
    }
}

স্থির শূন্য ggml_compute_forward_geglu_f16(
    ধ্রুব ggml_compute_params * params,
    ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];
    বর্ণ * src0_d = (বর্ণ *) src0->data;
    বর্ণ * src1_d = (বর্ণ *) (src1 ? src1->data : src0->data);
    ধ্রুব size_t src0_o = src0->nb[1];
    ধ্রুব size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    যদি (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ধ্রুব পূর্ণাঙ্ক nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ধ্রুব int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr*ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    প্রতি (পূর্ণাঙ্ক i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        যদি (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_f16(nc, (ggml_fp16_t *) ((বর্ণ *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        প্রতি (পূর্ণাঙ্ক k = 0; k < nc; k++) {
            ধ্রুব ggml_fp16_t x = ((ggml_fp16_t *) ((বর্ণ *) dst->data + i1*( dst->nb[1])))[k];
            ধ্রুব ভিন্ন v = GGML_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            প্রতিজ্ঞা(!isnan(v));
            প্রতিজ্ঞা(!isinf(v));
        }
#endif // NDEBUG
    }
}

স্থির শূন্য ggml_compute_forward_geglu_bf16(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_rows<ggml_vec_geglu_bf16>(params, dst);
}

স্থির শূন্য ggml_compute_forward_geglu(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_geglu_f32(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_F16:
            {
                ggml_compute_forward_geglu_f16(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_BF16:
            {
                ggml_compute_forward_geglu_bf16(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_swiglu

স্থির শূন্য ggml_compute_forward_swiglu_f32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];
    বর্ণ * src0_d = (বর্ণ *) src0->data;
    বর্ণ * src1_d = (বর্ণ *) (src1 ? src1->data : src0->data);
    ধ্রুব size_t src0_o = src0->nb[1];
    ধ্রুব size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    যদি (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ধ্রুব পূর্ণাঙ্ক nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ধ্রুব int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr*ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    প্রতি (পূর্ণাঙ্ক i1 = ir0; i1 < ir1; i1++) {
        ভিন্ন * src0_p = (ভিন্ন *) (src0_d + i1*src0_o);
        ভিন্ন * src1_p = (ভিন্ন *) (src1_d + i1*src1_o);

        যদি (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_swiglu_f32(nc, (ভিন্ন *) ((বর্ণ *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        প্রতি (পূর্ণাঙ্ক k = 0; k < nc; k++) {
            ধ্রুব ভিন্ন x = ((ভিন্ন *) ((বর্ণ *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            প্রতিজ্ঞা(!isnan(x));
            প্রতিজ্ঞা(!isinf(x));
        }
#endif // NDEBUG
    }
}

স্থির শূন্য ggml_compute_forward_swiglu_f16(
    ধ্রুব ggml_compute_params * params,
    ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];
    বর্ণ * src0_d = (বর্ণ *) src0->data;
    বর্ণ * src1_d = (বর্ণ *) (src1 ? src1->data : src0->data);
    ধ্রুব size_t src0_o = src0->nb[1];
    ধ্রুব size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    যদি (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ধ্রুব পূর্ণাঙ্ক nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ধ্রুব int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr*ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    প্রতি (পূর্ণাঙ্ক i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        যদি (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_swiglu_f16(nc, (ggml_fp16_t *) ((বর্ণ *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        প্রতি (পূর্ণাঙ্ক k = 0; k < nc; k++) {
            ধ্রুব ggml_fp16_t x = ((ggml_fp16_t *) ((বর্ণ *) dst->data + i1*( dst->nb[1])))[k];
            ধ্রুব ভিন্ন v = GGML_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            প্রতিজ্ঞা(!isnan(v));
            প্রতিজ্ঞা(!isinf(v));
        }
#endif // NDEBUG
    }
}

স্থির শূন্য ggml_compute_forward_swiglu_bf16(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_rows<ggml_vec_swiglu_bf16>(params, dst);
}

স্থির শূন্য ggml_compute_forward_swiglu(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_swiglu_f32(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_F16:
            {
                ggml_compute_forward_swiglu_f16(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_BF16:
            {
                ggml_compute_forward_swiglu_bf16(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_swiglu_oai

স্থির শূন্য ggml_compute_forward_swiglu_oai_f32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];
    বর্ণ * src0_d = (বর্ণ *) src0->data;
    বর্ণ * src1_d = (বর্ণ *) (src1 ? src1->data : src0->data);
    ধ্রুব size_t src0_o = src0->nb[1];
    ধ্রুব size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    যদি (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ধ্রুব পূর্ণাঙ্ক nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ধ্রুব int32_t swapped = ggml_get_op_params_i32(dst, 1);
    ধ্রুব ভিন্ন alpha = ggml_get_op_params_f32(dst, 2);
    ধ্রুব ভিন্ন limit = ggml_get_op_params_f32(dst, 3);

    // rows per thread
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr*ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    প্রতি (পূর্ণাঙ্ক i1 = ir0; i1 < ir1; i1++) {
        ভিন্ন * src0_p = (ভিন্ন *) (src0_d + i1*src0_o);
        ভিন্ন * src1_p = (ভিন্ন *) (src1_d + i1*src1_o);
        ভিন্ন * dst_p  = (ভিন্ন *) ((বর্ণ *) dst->data + i1*(dst->nb[1]));

        যদি (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        প্রতি (পূর্ণাঙ্ক k = 0; k < nc; k++) {
            ধ্রুব ভিন্ন x = std::min(src0_p[k], limit);
            ধ্রুব ভিন্ন y = std::clamp(src1_p[k], -limit, limit);
            ধ্রুব ভিন্ন out_glu = x / (1.f + expf(alpha * (-x)));
            dst_p[k] = out_glu * (y + 1.f);
        }

#ifndef NDEBUG
        প্রতি (পূর্ণাঙ্ক k = 0; k < nc; k++) {
            ধ্রুব ভিন্ন x = dst_p[k];
            GGML_UNUSED(x);
            প্রতিজ্ঞা(!isnan(x));
            প্রতিজ্ঞা(!isinf(x));
        }
#endif // NDEBUG
    }
}

স্থির শূন্য ggml_compute_forward_swiglu_oai_f16(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];
    বর্ণ * src0_d = (বর্ণ *) src0->data;
    বর্ণ * src1_d = (বর্ণ *) (src1 ? src1->data : src0->data);
    ধ্রুব size_t src0_o = src0->nb[1];
    ধ্রুব size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    যদি (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ধ্রুব পূর্ণাঙ্ক nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ধ্রুব int32_t swapped = ggml_get_op_params_i32(dst, 1);
    ধ্রুব ভিন্ন alpha = ggml_get_op_params_f32(dst, 2);
    ধ্রুব ভিন্ন limit = ggml_get_op_params_f32(dst, 3);

    // rows per thread
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr*ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    প্রতি (পূর্ণাঙ্ক i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);
        ggml_fp16_t * dst_p  = (ggml_fp16_t *) ((বর্ণ *) dst->data + i1*(dst->nb[1]));

        যদি (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        প্রতি (পূর্ণাঙ্ক k = 0; k < nc; k++) {
            ধ্রুব ভিন্ন x = std::min(GGML_CPU_FP16_TO_FP32(src0_p[k]), limit);
            ধ্রুব ভিন্ন y = std::clamp(GGML_CPU_FP16_TO_FP32(src1_p[k]), -limit, limit);
            ধ্রুব ভিন্ন out_glu = x / (1.f + expf(alpha * (-x)));
            dst_p[k] = GGML_CPU_FP32_TO_FP16(out_glu * (y + 1.f));
        }

#ifndef NDEBUG
        প্রতি (পূর্ণাঙ্ক k = 0; k < nc; k++) {
            ধ্রুব ভিন্ন x = GGML_CPU_FP16_TO_FP32(dst_p[k]);
            GGML_UNUSED(x);
            প্রতিজ্ঞা(!isnan(x));
            প্রতিজ্ঞা(!isinf(x));
        }
#endif // NDEBUG
    }
}

স্থির শূন্য ggml_compute_forward_swiglu_oai(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_swiglu_oai_f32(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_F16:
            {
                ggml_compute_forward_swiglu_oai_f16(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_swiglu_clamp

স্থির শূন্য ggml_compute_forward_swiglu_clamp_f32(ধ্রুব ggml_compute_params * params, ggml_tensor * dst) {
    ধ্রুব ggml_tensor * src0   = dst->src[0];
    ধ্রুব ggml_tensor * src1   = dst->src[1];
    বর্ণ *              src0_d = (বর্ণ *) src0->data;
    বর্ণ *              src1_d = (বর্ণ *) (src1 ? src1->data : src0->data);
    ধ্রুব size_t        src0_o = src0->nb[1];
    ধ্রুব size_t        src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    যদি (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ধ্রুব পূর্ণাঙ্ক nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ধ্রুব int32_t swapped = ggml_get_op_params_i32(dst, 1);
    ধ্রুব ভিন্ন   limit   = ggml_get_op_params_f32(dst, 3);

    ধ্রুব পূর্ণাঙ্ক dr  = (nr + nth - 1) / nth;
    ধ্রুব পূর্ণাঙ্ক ir0 = dr * ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    প্রতি (পূর্ণাঙ্ক i1 = ir0; i1 < ir1; i1++) {
        ভিন্ন * src0_p = (ভিন্ন *) (src0_d + i1 * src0_o);
        ভিন্ন * src1_p = (ভিন্ন *) (src1_d + i1 * src1_o);
        ভিন্ন * dst_p  = (ভিন্ন *) ((বর্ণ *) dst->data + i1 * (dst->nb[1]));

        যদি (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        প্রতি (পূর্ণাঙ্ক k = 0; k < nc; k++) {
            ধ্রুব ভিন্ন gate = std::min(src0_p[k], limit);
            ধ্রুব ভিন্ন up   = std::clamp(src1_p[k], -limit, limit);
            dst_p[k]         = gate / (1.f + expf(-gate)) * up;
        }

#ifndef NDEBUG
        প্রতি (পূর্ণাঙ্ক k = 0; k < nc; k++) {
            ধ্রুব ভিন্ন x = dst_p[k];
            GGML_UNUSED(x);
            প্রতিজ্ঞা(!isnan(x));
            প্রতিজ্ঞা(!isinf(x));
        }
#endif  // NDEBUG
    }
}

স্থির শূন্য ggml_compute_forward_swiglu_clamp_f16(ধ্রুব ggml_compute_params * params, ggml_tensor * dst) {
    ধ্রুব ggml_tensor * src0   = dst->src[0];
    ধ্রুব ggml_tensor * src1   = dst->src[1];
    বর্ণ *              src0_d = (বর্ণ *) src0->data;
    বর্ণ *              src1_d = (বর্ণ *) (src1 ? src1->data : src0->data);
    ধ্রুব size_t        src0_o = src0->nb[1];
    ধ্রুব size_t        src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    যদি (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ধ্রুব পূর্ণাঙ্ক nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ধ্রুব int32_t swapped = ggml_get_op_params_i32(dst, 1);
    ধ্রুব ভিন্ন   limit   = ggml_get_op_params_f32(dst, 3);

    ধ্রুব পূর্ণাঙ্ক dr  = (nr + nth - 1) / nth;
    ধ্রুব পূর্ণাঙ্ক ir0 = dr * ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    প্রতি (পূর্ণাঙ্ক i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1 * src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1 * src1_o);
        ggml_fp16_t * dst_p  = (ggml_fp16_t *) ((বর্ণ *) dst->data + i1 * (dst->nb[1]));

        যদি (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        প্রতি (পূর্ণাঙ্ক k = 0; k < nc; k++) {
            ধ্রুব ভিন্ন gate = std::min(GGML_FP16_TO_FP32(src0_p[k]), limit);
            ধ্রুব ভিন্ন up   = std::clamp(GGML_FP16_TO_FP32(src1_p[k]), -limit, limit);
            dst_p[k]         = GGML_FP32_TO_FP16(gate / (1.f + expf(-gate)) * up);
        }

#ifndef NDEBUG
        প্রতি (পূর্ণাঙ্ক k = 0; k < nc; k++) {
            ধ্রুব ভিন্ন x = GGML_FP16_TO_FP32(dst_p[k]);
            GGML_UNUSED(x);
            প্রতিজ্ঞা(!isnan(x));
            প্রতিজ্ঞা(!isinf(x));
        }
#endif  // NDEBUG
    }
}

স্থির শূন্য ggml_compute_forward_swiglu_clamp(ধ্রুব ggml_compute_params * params, ggml_tensor * dst) {
    বিকল্প (dst->src[0]->type) {
        পক্ষ GGML_TYPE_F32:
            ggml_compute_forward_swiglu_clamp_f32(params, dst);
            ভঙ্গ;
        পক্ষ GGML_TYPE_F16:
            ggml_compute_forward_swiglu_clamp_f16(params, dst);
            ভঙ্গ;
        স্বভাব:
            GGML_ABORT("fatal error");
    }
}

// ggml_compute_forward_geglu_erf

স্থির শূন্য ggml_compute_forward_geglu_erf_f32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];
    বর্ণ * src0_d = (বর্ণ *) src0->data;
    বর্ণ * src1_d = (বর্ণ *) (src1 ? src1->data : src0->data);
    ধ্রুব size_t src0_o = src0->nb[1];
    ধ্রুব size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    যদি (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ধ্রুব পূর্ণাঙ্ক nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ধ্রুব int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr*ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    প্রতি (পূর্ণাঙ্ক i1 = ir0; i1 < ir1; i1++) {
        ভিন্ন * src0_p = (ভিন্ন *) (src0_d + i1*src0_o);
        ভিন্ন * src1_p = (ভিন্ন *) (src1_d + i1*src1_o);

        যদি (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_erf_f32(nc, (ভিন্ন *) ((বর্ণ *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        প্রতি (পূর্ণাঙ্ক k = 0; k < nc; k++) {
            ধ্রুব ভিন্ন x = ((ভিন্ন *) ((বর্ণ *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            প্রতিজ্ঞা(!isnan(x));
            প্রতিজ্ঞা(!isinf(x));
        }
#endif // NDEBUG
    }
}

স্থির শূন্য ggml_compute_forward_geglu_erf_f16(
    ধ্রুব ggml_compute_params * params,
    ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];
    বর্ণ * src0_d = (বর্ণ *) src0->data;
    বর্ণ * src1_d = (বর্ণ *) (src1 ? src1->data : src0->data);
    ধ্রুব size_t src0_o = src0->nb[1];
    ধ্রুব size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    যদি (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ধ্রুব পূর্ণাঙ্ক nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ধ্রুব int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr*ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    প্রতি (পূর্ণাঙ্ক i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        যদি (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_erf_f16(nc, (ggml_fp16_t *) ((বর্ণ *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        প্রতি (পূর্ণাঙ্ক k = 0; k < nc; k++) {
            ধ্রুব ggml_fp16_t x = ((ggml_fp16_t *) ((বর্ণ *) dst->data + i1*( dst->nb[1])))[k];
            ধ্রুব ভিন্ন v = GGML_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            প্রতিজ্ঞা(!isnan(v));
            প্রতিজ্ঞা(!isinf(v));
        }
#endif // NDEBUG
    }
}

স্থির শূন্য ggml_compute_forward_geglu_erf_bf16(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_rows<ggml_vec_geglu_erf_bf16>(params, dst);
}

স্থির শূন্য ggml_compute_forward_geglu_erf(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_geglu_erf_f32(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_F16:
            {
                ggml_compute_forward_geglu_erf_f16(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_BF16:
            {
                ggml_compute_forward_geglu_erf_bf16(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_geglu_quick

স্থির শূন্য ggml_compute_forward_geglu_quick_f32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];
    বর্ণ * src0_d = (বর্ণ *) src0->data;
    বর্ণ * src1_d = (বর্ণ *) (src1 ? src1->data : src0->data);
    ধ্রুব size_t src0_o = src0->nb[1];
    ধ্রুব size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    যদি (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ধ্রুব পূর্ণাঙ্ক nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ধ্রুব int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr*ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    প্রতি (পূর্ণাঙ্ক i1 = ir0; i1 < ir1; i1++) {
        ভিন্ন * src0_p = (ভিন্ন *) (src0_d + i1*src0_o);
        ভিন্ন * src1_p = (ভিন্ন *) (src1_d + i1*src1_o);

        যদি (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_quick_f32(nc, (ভিন্ন *) ((বর্ণ *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        প্রতি (পূর্ণাঙ্ক k = 0; k < nc; k++) {
            ধ্রুব ভিন্ন x = ((ভিন্ন *) ((বর্ণ *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            প্রতিজ্ঞা(!isnan(x));
            প্রতিজ্ঞা(!isinf(x));
        }
#endif // NDEBUG
    }
}

স্থির শূন্য ggml_compute_forward_geglu_quick_f16(
    ধ্রুব ggml_compute_params * params,
    ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];
    বর্ণ * src0_d = (বর্ণ *) src0->data;
    বর্ণ * src1_d = (বর্ণ *) (src1 ? src1->data : src0->data);
    ধ্রুব size_t src0_o = src0->nb[1];
    ধ্রুব size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    যদি (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ধ্রুব পূর্ণাঙ্ক nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ধ্রুব int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr*ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    প্রতি (পূর্ণাঙ্ক i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        যদি (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_quick_f16(nc, (ggml_fp16_t *) ((বর্ণ *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        প্রতি (পূর্ণাঙ্ক k = 0; k < nc; k++) {
            ধ্রুব ggml_fp16_t x = ((ggml_fp16_t *) ((বর্ণ *) dst->data + i1*( dst->nb[1])))[k];
            ধ্রুব ভিন্ন v = GGML_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            প্রতিজ্ঞা(!isnan(v));
            প্রতিজ্ঞা(!isinf(v));
        }
#endif // NDEBUG
    }
}

স্থির শূন্য ggml_compute_forward_geglu_quick_bf16(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_rows<ggml_vec_geglu_quick_bf16>(params, dst);
}

স্থির শূন্য ggml_compute_forward_geglu_quick(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_geglu_quick_f32(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_F16:
            {
                ggml_compute_forward_geglu_quick_f16(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_BF16:
            {
                ggml_compute_forward_geglu_quick_bf16(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_norm

স্থির শূন্য ggml_compute_forward_norm_f32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    ভিন্ন eps;
    memcpy(&eps, dst->op_params, পরিমাণ(ভিন্ন));

    GGML_ASSERT(eps >= 0.0f);

    প্রতি (int64_t i03 = 0; i03 < ne03; i03++) {
        প্রতি (int64_t i02 = 0; i02 < ne02; i02++) {
            প্রতি (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                ধ্রুব বর্ণ * x = (ধ্রুব বর্ণ *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
                বর্ণ * y = (বর্ণ *) dst->data + i01*nb1 + i02*nb2 + i03*nb3;

                যদি (nb00 == পরিমাণ(ভিন্ন) && nb0 == পরিমাণ(ভিন্ন)) {
                    ধ্রুব ভিন্ন * xf = (ধ্রুব ভিন্ন *) x;

                    ভিন্ন sum = 0.0;
                    ggml_vec_sum_f32(ne00, &sum, xf);
                    ভিন্ন mean = sum/ne00;

                    ভিন্ন * yf = (ভিন্ন *) y;
                    ভিন্ন variance = 0;

#ifdef GGML_USE_ACCELERATE
                    mean = -mean;
                    vDSP_vsadd(xf, 1, &mean, yf, 1, ne00);
                    vDSP_measqv(yf, 1, &variance, ne00);
#অন্যথা
                    variance = ggml_vec_cvar_f32(ne00, yf, xf, mean);
#endif //GGML_USE_ACCELERATE

                    ধ্রুব ভিন্ন scale = 1.0f/sqrtf(variance + eps);
                    ggml_vec_scale_f32(ne00, yf, scale);
                } অন্যথা {
                    ভিন্ন sum = 0.0;
                    প্রতি (int64_t i00 = 0; i00 < ne00; i00++) {
                        sum += *(ধ্রুব ভিন্ন *) (x + i00*nb00);
                    }
                    ধ্রুব ভিন্ন mean = sum/ne00;

                    ভিন্ন variance = 0.0f;
                    প্রতি (int64_t i00 = 0; i00 < ne00; i00++) {
                        ধ্রুব ভিন্ন v = *(ধ্রুব ভিন্ন *) (x + i00*nb00) - mean;
                        *(ভিন্ন *) (y + i00*nb0) = v;
                        variance += v * v;
                    }
                    variance /= ne00;

                    ধ্রুব ভিন্ন scale = 1.0f/sqrtf(variance + eps);
                    প্রতি (int64_t i00 = 0; i00 < ne00; i00++) {
                        *(ভিন্ন *) (y + i00*nb0) *= scale;
                    }
                }
            }
        }
    }
}

শূন্য ggml_compute_forward_norm(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_norm_f32(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_group_rms_norm

// fusion kinds that can be combined with the rms_norm computation in a single pass.
// extend this enum when adding new fused variants (e.g. FUSE_ADD, FUSE_MUL_ADD, ...).
গণনা ggml_rms_norm_fuse_op {
    GGML_RMS_NORM_FUSE_OP_NONE,
    GGML_RMS_NORM_FUSE_OP_MUL,
};

template <ggml_rms_norm_fuse_op FUSE_OP>
স্থির শূন্য ggml_compute_forward_rms_norm_f32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst_rms_norm,
        ggml_tensor * dst_fused = nullptr) {

    ধ্রুব ggml_tensor * src0 = dst_rms_norm->src[0];
    ধ্রুব ggml_tensor * src1 = nullptr;
    ggml_tensor       * dst  = dst_rms_norm;

    যদি constexpr (FUSE_OP == GGML_RMS_NORM_FUSE_OP_MUL) {
        src1 = (dst_fused->src[0] == dst_rms_norm) ? dst_fused->src[1] : dst_fused->src[0];
        dst  = dst_fused;
    }

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    GGML_ASSERT(src0->nb[0] == পরিমাণ(ভিন্ন));

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    GGML_TENSOR_দ্বিমিক_OP_LOCALS

    ভিন্ন eps;
    memcpy(&eps, dst_rms_norm->op_params, পরিমাণ(ভিন্ন));
    GGML_ASSERT(eps >= 0.0f);

    // TODO: optimize
    প্রতি (int64_t i03 = 0; i03 < ne03; i03++) {
        প্রতি (int64_t i02 = 0; i02 < ne02; i02++) {
            প্রতি (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                ধ্রুব ভিন্ন * x = (ভিন্ন *) ((বর্ণ *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);

                ggml_float sum = 0.0;
                // worth switching to explicit SIMD?
                প্রতি (int64_t i00 = 0; i00 < ne00; i00++) {
                    sum += (ggml_float)(x[i00] * x[i00]);
                }

                ধ্রুব ভিন্ন mean  = sum/ne00;
                ধ্রুব ভিন্ন scale = 1.0f/sqrtf(mean + eps);

                // if you hit this, likely you got an inf somewhere earlier
                প্রতিজ্ঞা(scale > 0.0f);

                ভিন্ন * y = (ভিন্ন *) ((বর্ণ *) dst->data + i01*nb1 + i02*nb2 + i03*nb3);

                যদি constexpr (FUSE_OP == GGML_RMS_NORM_FUSE_OP_MUL) {
                    ধ্রুব int64_t i11 = i01 % ne11;
                    ধ্রুব int64_t i12 = i02 % ne12;
                    ধ্রুব int64_t i13 = i03 % ne13;
                    ধ্রুব ভিন্ন * w = (ভিন্ন *) ((বর্ণ *) src1->data + i11*nb11 + i12*nb12 + i13*nb13);

                    প্রতি (int64_t i00 = 0; i00 < ne00; i00++) {
                        y[i00] = x[i00] * scale * w[i00];
                    }
                } অন্যথা {
                    memcpy(y, x, ne00 * পরিমাণ(ভিন্ন));
                    ggml_vec_scale_f32(ne00, y, scale);
                }
            }
        }
    }
}

শূন্য ggml_compute_forward_rms_norm(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_rms_norm_f32<GGML_RMS_NORM_FUSE_OP_NONE>(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// Fused RMS_NORM + MUL: computes dst = rms_norm(src0) * src1 in a single pass.
// This avoids materializing the intermediate rms_norm result in memory.
শূন্য ggml_compute_forward_rms_norm_mul_fused(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst_rms_norm,
        ggml_tensor * dst_mul) {

    GGML_ASSERT(dst_mul != nullptr);
    GGML_ASSERT(dst_mul->src[0] == dst_rms_norm || dst_mul->src[1] == dst_rms_norm);

    ধ্রুব ggml_tensor * src0 = dst_rms_norm->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_rms_norm_f32<GGML_RMS_NORM_FUSE_OP_MUL>(params, dst_rms_norm, dst_mul);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

স্থির শূন্য ggml_compute_forward_rms_norm_back_f32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0]; // gradients from forward pass output
    ধ্রুব ggml_tensor * src1 = dst->src[1]; // src1 from forward pass

    GGML_ASSERT(ggml_are_same_shape(src0, dst) && ggml_are_same_shape(src0, src1));

    GGML_ASSERT(src0->nb[0] == পরিমাণ(ভিন্ন));
    GGML_ASSERT(src1->nb[0] == পরিমাণ(ভিন্ন));

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    GGML_TENSOR_দ্বিমিক_OP_LOCALS

    ভিন্ন eps;
    memcpy(&eps, dst->op_params, পরিমাণ(ভিন্ন));

    // TODO: optimize
    প্রতি (int64_t i03 = 0; i03 < ne03; i03++) {
        প্রতি (int64_t i02 = 0; i02 < ne02; i02++) {
            প্রতি (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                // src1 is same shape as src0 => same indices
                ধ্রুব int64_t i11 = i01;
                ধ্রুব int64_t i12 = i02;
                ধ্রুব int64_t i13 = i03;

                ধ্রুব ভিন্ন * dz = (ভিন্ন *) ((বর্ণ *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
                ধ্রুব ভিন্ন * x  = (ভিন্ন *) ((বর্ণ *) src1->data + i11*nb11 + i12*nb12 + i13*nb13);

                ggml_float sum_xx  = 0.0;
                ggml_float sum_xdz = 0.0;

                প্রতি (int64_t i00 = 0; i00 < ne00; i00++) {
                    sum_xx  += (ggml_float)(x[i00] * x[i00]);
                    sum_xdz += (ggml_float)(x[i00] * dz[i00]);
                }

                //const float mean     = (float)(sum_xx)/ne00;
                ধ্রুব ভিন্ন mean_eps = (ভিন্ন)(sum_xx)/ne00 + eps;
                ধ্রুব ভিন্ন sum_eps  = (ভিন্ন)(sum_xx) + eps*ne00;
                //const float mean_xdz = (float)(sum_xdz)/ne00;
                // we could cache rms from forward pass to improve performance.
                // to do this implement ggml_rms and compose ggml_rms_norm using ggml_rms.
                //const float rms      = sqrtf(mean_eps);
                ধ্রুব ভিন্ন rrms     = 1.0f / sqrtf(mean_eps);
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
                ভিন্ন * dx = (ভিন্ন *) ((বর্ণ *) dst->data + i01*nb1 + i02*nb2 + i03*nb3);

                // dx[i00] = (dz + x*(-sum_xdz/sum_eps)) * rrms
                // note: https://github.com/ggml-org/ggml/issues/1491
                ধ্রুব ভিন্ন scale_x = (ভিন্ন) (-sum_xdz) / sum_eps;
                প্রতি (int64_t i00 = 0; i00 < ne00; i00++) {
                    dx[i00] = (dz[i00] + x[i00] * scale_x) * rrms;
                }
            }
        }
    }
}

শূন্য ggml_compute_forward_rms_norm_back(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_rms_norm_back_f32(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_group_norm

স্থির শূন্য ggml_compute_forward_group_norm_f32(
    ধ্রুব ggml_compute_params * params,
    ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    GGML_ASSERT(src0->nb[0] == পরিমাণ(ভিন্ন));

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    // TODO: optimize

    ভিন্ন eps;
    memcpy(&eps, dst->op_params + 1, পরিমাণ(ভিন্ন));

    পূর্ণাঙ্ক n_channels = src0->ne[2];
    পূর্ণাঙ্ক n_groups = dst->op_params[0];
    পূর্ণাঙ্ক n_channels_per_group = (n_channels + n_groups - 1) / n_groups;
    প্রতি (পূর্ণাঙ্ক i = ith; i < n_groups; i += nth) {
        পূর্ণাঙ্ক start = i * n_channels_per_group;
        পূর্ণাঙ্ক end = start + n_channels_per_group;
        যদি (end > n_channels) {
            end = n_channels;
        }
        পূর্ণাঙ্ক step = end - start;

        প্রতি (int64_t i03 = 0; i03 < ne03; i03++) {
            ggml_float sum = 0.0;
            প্রতি (int64_t i02 = start; i02 < end; i02++) {
                প্রতি (int64_t i01 = 0; i01 < ne01; i01++) {
                    ধ্রুব ভিন্ন * x = (ভিন্ন *)((বর্ণ *) src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);

                    ggml_float sumr = 0.0;
                    প্রতি (int64_t i00 = 0; i00 < ne00; i00++) {
                        sumr += (ggml_float)x[i00];
                    }
                    sum += sumr;
                }
            }
            ধ্রুব ভিন্ন mean = sum / (ne00 * ne01 * step);

            ggml_float sum2 = 0.0;
            প্রতি (int64_t i02 = start; i02 < end; i02++) {
                প্রতি (int64_t i01 = 0; i01 < ne01; i01++) {
                    ধ্রুব ভিন্ন * x = (ভিন্ন *)((বর্ণ *) src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);

                    ভিন্ন * y = (ভিন্ন *)((বর্ণ *) dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3);

                    ggml_float sumr = 0.0;
                    প্রতি (int64_t i00 = 0; i00 < ne00; i00++) {
                        ভিন্ন v = x[i00] - mean;
                        y[i00] = v;
                        sumr += (ggml_float)(v * v);
                    }
                    sum2 += sumr;
                }
            }
            ধ্রুব ভিন্ন variance = sum2 / (ne00 * ne01 * step);
            ধ্রুব ভিন্ন scale = 1.0f / sqrtf(variance + eps);

            প্রতি (int64_t i02 = start; i02 < end; i02++) {
                প্রতি (int64_t i01 = 0; i01 < ne01; i01++) {
                    ভিন্ন * y = (ভিন্ন *)((বর্ণ *) dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3);
                    ggml_vec_scale_f32(ne00, y, scale);
                }
            }
        }
    }
}

শূন্য ggml_compute_forward_group_norm(
    ধ্রুব ggml_compute_params * params,
    ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_group_norm_f32(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_l2_norm

স্থির শূন্য ggml_compute_forward_l2_norm_f32(
    ধ্রুব ggml_compute_params * params,
    ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    ভিন্ন eps;
    memcpy(&eps, dst->op_params, পরিমাণ(ভিন্ন));

    GGML_ASSERT(eps >= 0.0f);

    // TODO: optimize
    প্রতি (int64_t i03 = 0; i03 < ne03; i03++) {
        প্রতি (int64_t i02 = 0; i02 < ne02; i02++) {
            প্রতি (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                ধ্রুব বর্ণ * x = (ধ্রুব বর্ণ *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;

                ggml_float sum = 0.0;
                প্রতি (int64_t i00 = 0; i00 < ne00; i00++) {
                    ধ্রুব ভিন্ন xi = *(ধ্রুব ভিন্ন *) (x + i00*nb00);
                    sum += (ggml_float)(xi * xi);
                }

                ধ্রুব ভিন্ন scale = 1.0f/fmaxf(sqrtf(sum), eps);

                বর্ণ * y = (বর্ণ *) dst->data + i01*nb1 + i02*nb2 + i03*nb3;

                যদি (nb00 == পরিমাণ(ভিন্ন) && nb0 == পরিমাণ(ভিন্ন)) {
                    memcpy(y, x, ne00 * পরিমাণ(ভিন্ন));
                    ggml_vec_scale_f32(ne00, (ভিন্ন *) y, scale);
                } অন্যথা {
                    প্রতি (int64_t i00 = 0; i00 < ne00; i00++) {
                        ধ্রুব ভিন্ন xi = *(ধ্রুব ভিন্ন *) (x + i00*nb00);
                        *(ভিন্ন *) (y + i00*nb0) = xi * scale;
                    }
                }
            }
        }
    }
}

শূন্য ggml_compute_forward_l2_norm(
    ধ্রুব ggml_compute_params * params,
    ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_l2_norm_f32(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_out_prod

স্থির শূন্য ggml_compute_forward_out_prod_f32(
        ধ্রুব ggml_compute_params * params,
              ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_দ্বিমিক_OP_LOCALS

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    GGML_ASSERT(ne2 % ne02 == 0);
    GGML_ASSERT(ne3 % ne03 == 0);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == পরিমাণ(ভিন্ন));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == পরিমাণ(ভিন্ন));
    // GGML_ASSERT(nb0 <= nb1);
    // GGML_ASSERT(nb1 <= nb2);
    // GGML_ASSERT(nb2 <= nb3);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    যদি (ith == 0) {
        ggml_vec_set_f32(ne0*ne1*ne2*ne3, (ভিন্ন *)dst->data, 0);
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
    ধ্রুব int64_t nr = ne1*ne2*ne3;

    // rows per thread
    ধ্রুব int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব int64_t ir0 = dr*ith;
    ধ্রুব int64_t ir1 = MIN(ir0 + dr, nr);

    // block-tiling attempt
    ধ্রুব int64_t blck_0 = MAX(GGML_VEC_MAD_UNROLL, 32);
    ধ্রুব int64_t blck_1 = 16;

    // dps == dst per src0, used for group query attention
    ধ্রুব int64_t dps2 = ne2 / ne02;
    ধ্রুব int64_t dps3 = ne3 / ne03;

    প্রতি (int64_t bir = ir0; bir < ir1; bir += blck_1) {
        ধ্রুব int64_t bir1 = MIN(bir + blck_1, ir1);
        প্রতি (int64_t bi01 = 0; bi01 < ne01; bi01 += blck_0) {
            ধ্রুব int64_t bne01 = MIN(bi01 + blck_0, ne01);
            প্রতি (int64_t ir = bir; ir < bir1; ++ir) {
                // dst indices
                ধ্রুব int64_t i3 = ir/(ne2*ne1);
                ধ্রুব int64_t i2 = (ir - i3*ne2*ne1)/ne1;
                ধ্রুব int64_t i1 = (ir - i3*ne2*ne1 - i2*ne1);

                ধ্রুব int64_t i02 = i2 / dps2;
                ধ্রুব int64_t i03 = i3 / dps3;

                //const int64_t i10 = i1;
                ধ্রুব int64_t i12 = i2;
                ধ্রুব int64_t i13 = i3;

#যদি GGML_VEC_MAD_UNROLL > 2
                ধ্রুব int64_t bne01_unroll = bne01 - (bne01 % GGML_VEC_MAD_UNROLL);
                প্রতি (int64_t i01 = bi01; i01 < bne01_unroll; i01 += GGML_VEC_MAD_UNROLL) {
                    ধ্রুব int64_t i11 = i01;

                    ভিন্ন * s0 = (ভিন্ন *) ((বর্ণ *) src0->data + (          i01*nb01 + i02*nb02 + i03*nb03));
                    ভিন্ন * s1 = (ভিন্ন *) ((বর্ণ *) src1->data + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
                    ভিন্ন * d  = (ভিন্ন *) ((বর্ণ *)  dst->data + (          i1*nb1   + i2*nb2   + i3*nb3));

                    ggml_vec_mad_f32_unroll(ne0, nb01, nb11, d, s0, s1);
                }
                প্রতি (int64_t i01 = bne01_unroll; i01 < bne01; ++i01) {
                    ধ্রুব int64_t i11 = i01;

                    ভিন্ন * s0 = (ভিন্ন *) ((বর্ণ *) src0->data + (          i01*nb01 + i02*nb02 + i03*nb03));
                    ভিন্ন * s1 = (ভিন্ন *) ((বর্ণ *) src1->data + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
                    ভিন্ন * d  = (ভিন্ন *) ((বর্ণ *)  dst->data + (          i1*nb1   + i2*nb2   + i3*nb3));

                    ggml_vec_mad_f32(ne0, d, s0, *s1);
                }
#অন্যথা
                প্রতি (int64_t i01 = bi01; i01 < bne01; ++i01) {
                    ধ্রুব int64_t i11 = i01;

                    ভিন্ন * s0 = (ভিন্ন *) ((বর্ণ *) src0->data + (          i01*nb01 + i02*nb02 + i03*nb03));
                    ভিন্ন * s1 = (ভিন্ন *) ((বর্ণ *) src1->data + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
                    ভিন্ন * d  = (ভিন্ন *) ((বর্ণ *)  dst->data + (          i1*nb1 + i2*nb2 + i3*nb3));

                    ggml_vec_mad_f32(ne0, d, s0, *s1);
                }
#endif
            }
        }
    }
}

স্থির শূন্য ggml_compute_forward_out_prod_q_f32(
        ধ্রুব ggml_compute_params * params,
              ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_দ্বিমিক_OP_LOCALS;

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব ggml_type type = src0->type;
    ggml_to_float_t ধ্রুব dequantize_row_q = ggml_get_type_traits(type)->to_float;

    GGML_ASSERT(ne02 == ne12);
    GGML_ASSERT(ne03 == ne13);
    GGML_ASSERT(ne2  == ne12);
    GGML_ASSERT(ne3  == ne13);

    // we don't support permuted src0 dim0
    GGML_ASSERT(nb00 == ggml_type_size(type));

    // dst dim0 cannot be transposed or permuted
    GGML_ASSERT(nb0 == পরিমাণ(ভিন্ন));
    // GGML_ASSERT(nb0 <= nb1);
    // GGML_ASSERT(nb1 <= nb2);
    // GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    যদি (ith == 0) {
        ggml_vec_set_f32(ne0*ne1*ne2*ne3, (ভিন্ন *)dst->data, 0);
    }
    ggml_barrier(params->threadpool);

    // parallelize by last three dimensions

    // total rows in dst
    ধ্রুব int64_t nr = ne1*ne2*ne3;

    // rows per thread
    ধ্রুব int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব int64_t ir0 = dr*ith;
    ধ্রুব int64_t ir1 = MIN(ir0 + dr, nr);

    // dst[:,:,:,:] = 0
    // for i2,i3:
    //   for i1:
    //     for i01:
    //       for i0:
    //         dst[i0,i1,i2,i3] += src0[i0,i01,i2,i3] * src1[i1,i01,i2,i3]

    ভিন্ন * wdata = (ভিন্ন *) params->wdata + (ne0 + CACHE_সারি_SIZE_F32) * ith;

    প্রতি (int64_t ir = ir0; ir < ir1; ++ir) {
        // dst indices
        ধ্রুব int64_t i3 = ir/(ne2*ne1);
        ধ্রুব int64_t i2 = (ir - i3*ne2*ne1)/ne1;
        ধ্রুব int64_t i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ধ্রুব int64_t i02 = i2;
        ধ্রুব int64_t i03 = i3;

        //const int64_t i10 = i1;
        ধ্রুব int64_t i12 = i2;
        ধ্রুব int64_t i13 = i3;

        প্রতি (int64_t i01 = 0; i01 < ne01; ++i01) {
            ধ্রুব int64_t i11 = i01;

            ভিন্ন * s0 = (ভিন্ন *) ((বর্ণ *) src0->data + (          i01*nb01 + i02*nb02 + i03*nb03));
            ভিন্ন * s1 = (ভিন্ন *) ((বর্ণ *) src1->data + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
            ভিন্ন * d  = (ভিন্ন *) ((বর্ণ *)  dst->data + (          i1*nb1 + i2*nb2 + i3*nb3));

            dequantize_row_q(s0, wdata, ne0);
            ggml_vec_mad_f32(ne0, d, wdata, *s1);
        }
    }
}

স্থির শূন্য ggml_compute_forward_out_prod_f16_f32(
        ধ্রুব ggml_compute_params * params,
              ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_দ্বিমিক_OP_LOCALS;

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ne02 == ne12);
    GGML_ASSERT(ne03 == ne13);
    GGML_ASSERT(ne2  == ne12);
    GGML_ASSERT(ne3  == ne13);

    GGML_ASSERT(nb00 == পরিমাণ(ggml_fp16_t));
    GGML_ASSERT(nb0  == পরিমাণ(ভিন্ন));

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    যদি (ith == 0) {
        ggml_vec_set_f32(ne0*ne1*ne2*ne3, (ভিন্ন *)dst->data, 0);
    }
    ggml_barrier(params->threadpool);

    ধ্রুব int64_t nr = ne1*ne2*ne3;
    ধ্রুব int64_t dr = (nr + nth - 1)/nth;
    ধ্রুব int64_t ir0 = dr*ith;
    ধ্রুব int64_t ir1 = MIN(ir0 + dr, nr);

    ভিন্ন * wdata = (ভিন্ন *) params->wdata + (ne0 + CACHE_সারি_SIZE_F32) * ith;

    প্রতি (int64_t ir = ir0; ir < ir1; ++ir) {
        ধ্রুব int64_t i3 = ir/(ne2*ne1);
        ধ্রুব int64_t i2 = (ir - i3*ne2*ne1)/ne1;
        ধ্রুব int64_t i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ধ্রুব int64_t i02 = i2;
        ধ্রুব int64_t i03 = i3;

        ধ্রুব int64_t i12 = i2;
        ধ্রুব int64_t i13 = i3;

        ভিন্ন * d = (ভিন্ন *) ((বর্ণ *) dst->data + (i1*nb1 + i2*nb2 + i3*nb3));

        প্রতি (int64_t i01 = 0; i01 < ne01; ++i01) {
            ধ্রুব int64_t i11 = i01;
            ggml_fp16_t * s0 = (ggml_fp16_t *) ((বর্ণ *) src0->data + (i01*nb01 + i02*nb02 + i03*nb03));
            ভিন্ন * s1 = (ভিন্ন *) ((বর্ণ *) src1->data + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
            ggml_fp16_to_fp32_row(s0, wdata, ne0);
            ggml_vec_mad_f32(ne0, d, wdata, *s1);
        }
    }
}

শূন্য ggml_compute_forward_out_prod(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_Q1_0:
        পক্ষ GGML_TYPE_Q2_0:
        পক্ষ GGML_TYPE_Q4_0:
        পক্ষ GGML_TYPE_Q4_1:
        পক্ষ GGML_TYPE_Q5_0:
        পক্ষ GGML_TYPE_Q5_1:
        পক্ষ GGML_TYPE_Q8_0:
        পক্ষ GGML_TYPE_MXFP4:
        পক্ষ GGML_TYPE_NVFP4:
        পক্ষ GGML_TYPE_Q2_K:
        পক্ষ GGML_TYPE_Q3_K:
        পক্ষ GGML_TYPE_Q4_K:
        পক্ষ GGML_TYPE_Q5_K:
        পক্ষ GGML_TYPE_Q6_K:
        পক্ষ GGML_TYPE_TQ1_0:
        পক্ষ GGML_TYPE_TQ2_0:
        পক্ষ GGML_TYPE_IQ2_XXS:
        পক্ষ GGML_TYPE_IQ2_XS:
        পক্ষ GGML_TYPE_IQ3_XXS:
        পক্ষ GGML_TYPE_IQ1_S:
        পক্ষ GGML_TYPE_IQ1_M:
        পক্ষ GGML_TYPE_IQ4_NL:
        পক্ষ GGML_TYPE_IQ4_XS:
        পক্ষ GGML_TYPE_IQ3_S:
        পক্ষ GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_out_prod_q_f32(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_F16:
            {
                ggml_compute_forward_out_prod_f16_f32(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_out_prod_f32(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_scale

স্থির শূন্য ggml_compute_forward_scale_f32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    ভিন্ন s; // scale factor
    ভিন্ন b; // bias

    memcpy(&s, (ভিন্ন *) dst->op_params + 0, পরিমাণ(ভিন্ন));
    memcpy(&b, (ভিন্ন *) dst->op_params + 1, পরিমাণ(ভিন্ন));

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক nc = src0->ne[0];
    ধ্রুব পূর্ণাঙ্ক nr = ggml_nrows(src0);

    // rows per thread
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr*ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    ধ্রুব size_t nb01 = src0->nb[1];

    ধ্রুব size_t nb1 = dst->nb[1];

    যদি (b == 0.0f) {
        প্রতি (পূর্ণাঙ্ক i1 = ir0; i1 < ir1; i1++) {
            যদি (dst->data != src0->data) {
                // src0 is same shape as dst => same indices
                // TODO: add x parameter to ggml_vec_scale_f32 and remove this memcpy
                memcpy((বর্ণ *)dst->data + i1*nb1, (বর্ণ *)src0->data + i1*nb01, nc * পরিমাণ(ভিন্ন));
            }
            ggml_vec_scale_f32(nc, (ভিন্ন *) ((বর্ণ *) dst->data + i1*nb1), s);
        }
    } অন্যথা {
        প্রতি (পূর্ণাঙ্ক i1 = ir0; i1 < ir1; i1++) {
            ggml_vec_mad1_f32(nc,
                (ভিন্ন *) ((বর্ণ *) dst->data  + i1*nb1),
                (ভিন্ন *) ((বর্ণ *) src0->data + i1*nb1),
                s, b);
        }
    }
}

স্থির শূন্য ggml_compute_forward_scale_bf16(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    ভিন্ন s; // scale factor
    ভিন্ন b; // bias

    memcpy(&s, (ভিন্ন *) dst->op_params + 0, পরিমাণ(ভিন্ন));
    memcpy(&b, (ভিন্ন *) dst->op_params + 1, পরিমাণ(ভিন্ন));

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক nc = src0->ne[0];
    ধ্রুব পূর্ণাঙ্ক nr = ggml_nrows(src0);

    // rows per thread
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr*ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    প্রতি (পূর্ণাঙ্ক i1 = ir0; i1 < ir1; i1++) {
        ধ্রুব ggml_bf16_t * x = (ধ্রুব ggml_bf16_t *) ((ধ্রুব বর্ণ *) src0->data + i1*src0->nb[1]);
        ggml_bf16_t       * y = (ggml_bf16_t *)       ((বর্ণ *)       dst->data  + i1*dst->nb[1]);
        প্রতি (পূর্ণাঙ্ক k = 0; k < nc; k++) {
            y[k] = GGML_FP32_TO_BF16(s*GGML_BF16_TO_FP32(x[k]) + b);
        }
    }
}

শূন্য ggml_compute_forward_scale(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_scale_f32(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_BF16:
            {
                ggml_compute_forward_scale_bf16(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_set

স্থির শূন্য ggml_compute_forward_set_f32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));

    // view src0 and dst with these strides and data offset inbytes during set
    // nb0 is implicitly element_size because src0 and dst are contiguous
    size_t nb1     = ((int32_t *) dst->op_params)[0];
    size_t nb2     = ((int32_t *) dst->op_params)[1];
    size_t nb3     = ((int32_t *) dst->op_params)[2];
    size_t offset  = ((int32_t *) dst->op_params)[3];
    তার্কিক   inplace = (তার্কিক) ((int32_t *) dst->op_params)[4];

    যদি (!inplace) {
        যদি (params->ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            memcpy(
                ((বর্ণ *)  dst->data),
                ((বর্ণ *) src0->data),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক nr = ggml_nrows(src1);
    ধ্রুব পূর্ণাঙ্ক nc = src1->ne[0];

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne)
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb)

    // src0 and dst as viewed during set
    ধ্রুব size_t nb0 = ggml_element_size(src0);

    ধ্রুব পূর্ণাঙ্ক im0 = (ne10 == 0 ? 0 : ne10-1);
    ধ্রুব পূর্ণাঙ্ক im1 = (ne11 == 0 ? 0 : ne11-1);
    ধ্রুব পূর্ণাঙ্ক im2 = (ne12 == 0 ? 0 : ne12-1);
    ধ্রুব পূর্ণাঙ্ক im3 = (ne13 == 0 ? 0 : ne13-1);

    GGML_ASSERT(offset + im0*nb0  + im1*nb1  + im2*nb2  + im3*nb3  <= ggml_nbytes(dst));

    GGML_ASSERT(nb10 == পরিমাণ(ভিন্ন));

    // rows per thread
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr*ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    প্রতি (পূর্ণাঙ্ক ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are viewed with shape of src1 and offset
        // => same indices
        ধ্রুব পূর্ণাঙ্ক i3 = ir/(ne12*ne11);
        ধ্রুব পূর্ণাঙ্ক i2 = (ir - i3*ne12*ne11)/ne11;
        ধ্রুব পূর্ণাঙ্ক i1 = (ir - i3*ne12*ne11 - i2*ne11);

        ggml_vec_cpy_f32(nc,
                (ভিন্ন *) ((বর্ণ *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + offset),
                (ভিন্ন *) ((বর্ণ *) src1->data + i3*nb13 + i2*nb12 + i1*nb11));
    }
}

স্থির শূন্য ggml_compute_forward_set_i32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));

    // view src0 and dst with these strides and data offset inbytes during set
    // nb0 is implicitly element_size because src0 and dst are contiguous
    size_t nb1     = ((int32_t *) dst->op_params)[0];
    size_t nb2     = ((int32_t *) dst->op_params)[1];
    size_t nb3     = ((int32_t *) dst->op_params)[2];
    size_t offset  = ((int32_t *) dst->op_params)[3];
    তার্কিক   inplace = (তার্কিক) ((int32_t *) dst->op_params)[4];

    যদি (!inplace) {
        যদি (params->ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            memcpy(
                ((বর্ণ *)  dst->data),
                ((বর্ণ *) src0->data),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক nr = ggml_nrows(src1);
    ধ্রুব পূর্ণাঙ্ক nc = src1->ne[0];

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne)
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb)

    // src0 and dst as viewed during set
    ধ্রুব size_t nb0 = ggml_element_size(src0);

    ধ্রুব পূর্ণাঙ্ক im0 = (ne10 == 0 ? 0 : ne10-1);
    ধ্রুব পূর্ণাঙ্ক im1 = (ne11 == 0 ? 0 : ne11-1);
    ধ্রুব পূর্ণাঙ্ক im2 = (ne12 == 0 ? 0 : ne12-1);
    ধ্রুব পূর্ণাঙ্ক im3 = (ne13 == 0 ? 0 : ne13-1);

    GGML_ASSERT(offset + im0*nb0  + im1*nb1  + im2*nb2  + im3*nb3  <= ggml_nbytes(dst));

    GGML_ASSERT(nb10 == পরিমাণ(int32_t));

    // rows per thread
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr*ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    প্রতি (পূর্ণাঙ্ক ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are viewed with shape of src1 and offset
        // => same indices
        ধ্রুব পূর্ণাঙ্ক i3 = ir/(ne12*ne11);
        ধ্রুব পূর্ণাঙ্ক i2 = (ir - i3*ne12*ne11)/ne11;
        ধ্রুব পূর্ণাঙ্ক i1 = (ir - i3*ne12*ne11 - i2*ne11);

        ggml_vec_cpy_i32(nc,
                (int32_t *) ((বর্ণ *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + offset),
                (int32_t *) ((বর্ণ *) src1->data + i3*nb13 + i2*nb12 + i1*nb11));
    }
}

শূন্য ggml_compute_forward_set(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_set_f32(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_I32:
            {
                ggml_compute_forward_set_i32(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_F16:
        পক্ষ GGML_TYPE_BF16:
        পক্ষ GGML_TYPE_Q1_0:
        পক্ষ GGML_TYPE_Q2_0:
        পক্ষ GGML_TYPE_Q4_0:
        পক্ষ GGML_TYPE_Q4_1:
        পক্ষ GGML_TYPE_Q5_0:
        পক্ষ GGML_TYPE_Q5_1:
        পক্ষ GGML_TYPE_Q8_0:
        পক্ষ GGML_TYPE_Q8_1:
        পক্ষ GGML_TYPE_MXFP4:
        পক্ষ GGML_TYPE_NVFP4:
        পক্ষ GGML_TYPE_Q2_K:
        পক্ষ GGML_TYPE_Q3_K:
        পক্ষ GGML_TYPE_Q4_K:
        পক্ষ GGML_TYPE_Q5_K:
        পক্ষ GGML_TYPE_Q6_K:
        পক্ষ GGML_TYPE_TQ1_0:
        পক্ষ GGML_TYPE_TQ2_0:
        পক্ষ GGML_TYPE_IQ2_XXS:
        পক্ষ GGML_TYPE_IQ2_XS:
        পক্ষ GGML_TYPE_IQ3_XXS:
        পক্ষ GGML_TYPE_IQ1_S:
        পক্ষ GGML_TYPE_IQ1_M:
        পক্ষ GGML_TYPE_IQ4_NL:
        পক্ষ GGML_TYPE_IQ4_XS:
        পক্ষ GGML_TYPE_IQ3_S:
        পক্ষ GGML_TYPE_IQ2_S:
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_cpy

শূন্য ggml_compute_forward_cpy(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_dup(params, dst);
}

// ggml_compute_forward_cont

শূন্য ggml_compute_forward_cont(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_dup(params, dst);
}

// ggml_compute_forward_get_rows

স্থির শূন্য ggml_compute_forward_get_rows_q(
        ধ্রুব ggml_compute_params * params,
              ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_দ্বিমিক_OP_LOCALS

    ধ্রুব int64_t nc = ne00;
    ধ্রুব int64_t nr = ggml_nelements(src1);

    ধ্রুব ggml_type type = src0->type;
    ggml_to_float_t ধ্রুব dequantize_row_q = ggml_get_type_traits(type)->to_float;

    প্রতিজ্ঞা(ne0  == nc);
    প্রতিজ্ঞা(ne02 == ne11);
    প্রতিজ্ঞা(nb00 == ggml_type_size(type));
    প্রতিজ্ঞা(ggml_nrows(dst) == nr);

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    // rows per thread
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr*ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    প্রতি (int64_t i = ir0; i < ir1; ++i) {
        ধ্রুব int64_t i12 = i/(ne11*ne10);
        ধ্রুব int64_t i11 = (i - i12*ne11*ne10)/ne10;
        ধ্রুব int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        ধ্রুব int64_t i01 = *(int32_t *) ((বর্ণ *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        dequantize_row_q(
                (ধ্রুব শূন্য *) ((বর্ণ *) src0->data + i01*nb01 + i11*nb02 + i12*nb03),
                     (ভিন্ন *) ((বর্ণ *)  dst->data + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

স্থির শূন্য ggml_compute_forward_get_rows_f16(
        ধ্রুব ggml_compute_params * params,
              ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_দ্বিমিক_OP_LOCALS

    ধ্রুব int64_t nc = ne00;
    ধ্রুব int64_t nr = ggml_nelements(src1);

    প্রতিজ্ঞা(ne0  == nc);
    প্রতিজ্ঞা(ne02 == ne11);
    প্রতিজ্ঞা(nb00 == পরিমাণ(ggml_fp16_t));
    প্রতিজ্ঞা(ggml_nrows(dst) == nr);

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    // rows per thread
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr*ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    প্রতি (int64_t i = ir0; i < ir1; ++i) {
        ধ্রুব int64_t i12 = i/(ne11*ne10);
        ধ্রুব int64_t i11 = (i - i12*ne11*ne10)/ne10;
        ধ্রুব int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        ধ্রুব int64_t i01 = *(int32_t *) ((বর্ণ *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        ggml_cpu_fp16_to_fp32(
            (ধ্রুব ggml_fp16_t*) ((বর্ণ *) src0->data + i01*nb01 + i11*nb02 + i12*nb03),
                       (ভিন্ন *) ((বর্ণ *)  dst->data + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

স্থির শূন্য ggml_compute_forward_get_rows_bf16(
        ধ্রুব ggml_compute_params * params,
              ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_দ্বিমিক_OP_LOCALS

    ধ্রুব int64_t nc = ne00;
    ধ্রুব int64_t nr = ggml_nelements(src1);

    প্রতিজ্ঞা(ne0  == nc);
    প্রতিজ্ঞা(ne02 == ne11);
    প্রতিজ্ঞা(nb00 == পরিমাণ(ggml_bf16_t));
    প্রতিজ্ঞা(ggml_nrows(dst) == nr);

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    // rows per thread
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr*ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    প্রতি (int64_t i = ir0; i < ir1; ++i) {
        ধ্রুব int64_t i12 = i/(ne11*ne10);
        ধ্রুব int64_t i11 = (i - i12*ne11*ne10)/ne10;
        ধ্রুব int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        ধ্রুব int64_t i01 = *(int32_t *) ((বর্ণ *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        ggml_cpu_bf16_to_fp32(
            (ধ্রুব ggml_bf16_t *) ((বর্ণ *) src0->data + i01*nb01 + i11*nb02 + i12*nb03),
                        (ভিন্ন *) ((বর্ণ *)  dst->data + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

স্থির শূন্য ggml_compute_forward_get_rows_f32(
        ধ্রুব ggml_compute_params * params,
              ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_দ্বিমিক_OP_LOCALS

    ধ্রুব int64_t nc = ne00;
    ধ্রুব int64_t nr = ggml_nelements(src1);

    প্রতিজ্ঞা(ne0  == nc);
    প্রতিজ্ঞা(ne02 == ne11);
    প্রতিজ্ঞা(nb00 == পরিমাণ(ভিন্ন));
    প্রতিজ্ঞা(ggml_nrows(dst) == nr);

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    // rows per thread
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr*ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    প্রতি (int64_t i = ir0; i < ir1; ++i) {
        ধ্রুব int64_t i12 = i/(ne11*ne10);
        ধ্রুব int64_t i11 = (i - i12*ne11*ne10)/ne10;
        ধ্রুব int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        ধ্রুব int64_t i01 = *(int32_t *) ((বর্ণ *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        ggml_vec_cpy_f32(nc,
                (ভিন্ন *) ((বর্ণ *)  dst->data + i10*nb1  + i11*nb2  + i12*nb3),
                (ভিন্ন *) ((বর্ণ *) src0->data + i01*nb01 + i11*nb02 + i12*nb03));
    }
}

শূন্য ggml_compute_forward_get_rows(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_Q1_0:
        পক্ষ GGML_TYPE_Q2_0:
        পক্ষ GGML_TYPE_Q4_0:
        পক্ষ GGML_TYPE_Q4_1:
        পক্ষ GGML_TYPE_Q5_0:
        পক্ষ GGML_TYPE_Q5_1:
        পক্ষ GGML_TYPE_Q8_0:
        পক্ষ GGML_TYPE_Q8_1:
        পক্ষ GGML_TYPE_MXFP4:
        পক্ষ GGML_TYPE_NVFP4:
        পক্ষ GGML_TYPE_Q2_K:
        পক্ষ GGML_TYPE_Q3_K:
        পক্ষ GGML_TYPE_Q4_K:
        পক্ষ GGML_TYPE_Q5_K:
        পক্ষ GGML_TYPE_Q6_K:
        পক্ষ GGML_TYPE_TQ1_0:
        পক্ষ GGML_TYPE_TQ2_0:
        পক্ষ GGML_TYPE_IQ2_XXS:
        পক্ষ GGML_TYPE_IQ2_XS:
        পক্ষ GGML_TYPE_IQ3_XXS:
        পক্ষ GGML_TYPE_IQ1_S:
        পক্ষ GGML_TYPE_IQ1_M:
        পক্ষ GGML_TYPE_IQ4_NL:
        পক্ষ GGML_TYPE_IQ4_XS:
        পক্ষ GGML_TYPE_IQ3_S:
        পক্ষ GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_get_rows_q(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_F16:
            {
                ggml_compute_forward_get_rows_f16(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_BF16:
            {
                ggml_compute_forward_get_rows_bf16(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_F32:
        পক্ষ GGML_TYPE_I32:
            {
                ggml_compute_forward_get_rows_f32(params, dst);
            } ভঙ্গ;
        স্বভাব:
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
স্থির শূন্য ggml_compute_forward_set_rows_impl(
        ধ্রুব ggml_compute_params * params,
              ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_দ্বিমিক_OP_LOCALS

    ধ্রুব int64_t nc = ne00;
    ধ্রুব int64_t nr = ne01;

    প্রতিজ্ঞা(ne0  == nc);
    প্রতিজ্ঞা(ne2  == ne02);
    প্রতিজ্ঞা(ne3  == ne03);
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    প্রতিজ্ঞা(ne02 % ne11 == 0);
    প্রতিজ্ঞা(ne03 % ne12 == 0);

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    // rows per thread
    ধ্রুব int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব int64_t ir0 = dr*ith;
    ধ্রুব int64_t ir1 = std::min(ir0 + dr, nr);

    ধ্রুব size_t rs = ggml_row_size(src0->type, nc);

    ggml_from_float_t ধ্রুব from_float = ggml_get_type_traits_cpu(dst->type)->from_float;

    প্রতি (int64_t i03 = 0; i03 < ne03; ++i03) {
        প্রতি (int64_t i02 = 0; i02 < ne02; ++i02) {
            প্রতি (int64_t i = ir0; i < ir1; ++i) {
                ধ্রুব int64_t i12 = i03%ne12;
                ধ্রুব int64_t i11 = i02%ne11;
                ধ্রুব int64_t i10 = i;

                ধ্রুব int64_t i1 = *(idx_t *) ((বর্ণ *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);

                GGML_ASSERT(i1 >= 0 && i1 < ne1);

                যদি constexpr (std::is_same_v<src_t, ভিন্ন>) {
                    from_float(
                            (ধ্রুব ভিন্ন *) ((বর্ণ *) src0->data +  i*nb01 + i02*nb02 + i03*nb03),
                                            ((বর্ণ *)  dst->data + i1*nb1  + i02*nb2  + i03*nb3), nc);
                } অন্যথা যদি constexpr (std::is_same_v<src_t, ggml_fp16_t>) {
                    যদি (dst->type == GGML_TYPE_F16) {
                        memcpy(
                                            ((বর্ণ *)  dst->data + i1*nb1  + i02*nb2  + i03*nb3),
                                            ((বর্ণ *) src0->data +  i*nb01 + i02*nb02 + i03*nb03),
                                            rs);
                    } অন্যথা {
                        ভিন্ন * wdata = (ভিন্ন *) params->wdata + (nc + CACHE_সারি_SIZE_F32) * ith;
                        ggml_fp16_to_fp32_row(
                                (ধ্রুব ggml_fp16_t *) ((বর্ণ *) src0->data + i*nb01 + i02*nb02 + i03*nb03),
                                wdata, nc);
                        from_float(wdata,
                                ((বর্ণ *) dst->data + i1*nb1 + i02*nb2 + i03*nb3), nc);
                    }
                } অন্যথা {
                    GGML_ABORT("src0->type = %d (%s) not supported", src0->type, ggml_type_name(src0->type));
                }
            }
        }
    }
}

শূন্য ggml_compute_forward_set_rows(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                যদি (src1->type == GGML_TYPE_I64) {
                    ggml_compute_forward_set_rows_impl<ভিন্ন, int64_t>(params, dst);
                } অন্যথা যদি (src1->type == GGML_TYPE_I32) {
                    ggml_compute_forward_set_rows_impl<ভিন্ন, int32_t>(params, dst);
                } অন্যথা {
                    GGML_ABORT("src1->type = %d (%s) not supported", src1->type, ggml_type_name(src1->type));
                }
            } ভঙ্গ;
        পক্ষ GGML_TYPE_F16:
            {
                যদি (src1->type == GGML_TYPE_I64) {
                    ggml_compute_forward_set_rows_impl<ggml_fp16_t, int64_t>(params, dst);
                } অন্যথা যদি (src1->type == GGML_TYPE_I32) {
                    ggml_compute_forward_set_rows_impl<ggml_fp16_t, int32_t>(params, dst);
                } অন্যথা {
                    GGML_ABORT("src1->type = %d (%s) not supported", src1->type, ggml_type_name(src1->type));
                }
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("src0->type = %d (%s) not supported", src0->type, ggml_type_name(src0->type));
            }
    }
}

// ggml_compute_forward_get_rows_back

স্থির শূন্য ggml_compute_forward_get_rows_back_f32_f16(
        ধ্রুব ggml_compute_params * params,
              ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];

    যদি (params->ith != 0) {
        প্রত্যাগম;
    }

    GGML_ASSERT(ggml_is_contiguous(dst));

    // ggml_compute_forward_dup_same_cont(params, opt0, dst);

    memset(dst->data, 0, ggml_nbytes(dst));

    ধ্রুব পূর্ণাঙ্ক nc = src0->ne[0];
    ধ্রুব পূর্ণাঙ্ক nr = ggml_nelements(src1);

    GGML_ASSERT( dst->ne[0] == nc);
    GGML_ASSERT(src0->nb[0] == পরিমাণ(ggml_fp16_t));

    প্রতি (পূর্ণাঙ্ক i = 0; i < nr; ++i) {
        ধ্রুব পূর্ণাঙ্ক r = ((int32_t *) src1->data)[i];

        GGML_ASSERT(r >= 0 && r < dst->ne[1]);

        প্রতি (পূর্ণাঙ্ক j = 0; j < nc; ++j) {
            ggml_fp16_t v = ((ggml_fp16_t *) ((বর্ণ *) src0->data + i*src0->nb[1]))[j];
            ((ভিন্ন *) ((বর্ণ *) dst->data + r*dst->nb[1]))[j] += GGML_CPU_FP16_TO_FP32(v);
        }
    }
}

স্থির শূন্য ggml_compute_forward_get_rows_back_f32(
        ধ্রুব ggml_compute_params * params,
              ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];

    যদি (params->ith != 0) {
        প্রত্যাগম;
    }

    GGML_ASSERT(ggml_is_contiguous(dst));

    // ggml_compute_forward_dup_same_cont(params, opt0, dst);

    memset(dst->data, 0, ggml_nbytes(dst));

    ধ্রুব পূর্ণাঙ্ক nc = src0->ne[0];
    ধ্রুব পূর্ণাঙ্ক nr = ggml_nelements(src1);

    GGML_ASSERT( dst->ne[0] == nc);
    GGML_ASSERT(src0->nb[0] == পরিমাণ(ভিন্ন));

    প্রতি (পূর্ণাঙ্ক i = 0; i < nr; ++i) {
        ধ্রুব পূর্ণাঙ্ক r = ((int32_t *) src1->data)[i];

        GGML_ASSERT(r >= 0 && r < dst->ne[1]);

        ggml_vec_add_f32(nc,
                (ভিন্ন *) ((বর্ণ *)  dst->data + r*dst->nb[1]),
                (ভিন্ন *) ((বর্ণ *)  dst->data + r*dst->nb[1]),
                (ভিন্ন *) ((বর্ণ *) src0->data + i*src0->nb[1]));
    }
}

শূন্য ggml_compute_forward_get_rows_back(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F16:
            {
                ggml_compute_forward_get_rows_back_f32_f16(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_get_rows_back_f32(params, dst);
            } ভঙ্গ;
        স্বভাব:
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

স্থির শূন্য ggml_compute_forward_diag_f32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    যদি (params->ith != 0) {
        প্রত্যাগম;
    }

    // TODO: handle transposed/permuted matrices

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(ne00 == ne0);
    GGML_ASSERT(ne00 == ne1);
    GGML_ASSERT(ne01 == 1);
    GGML_ASSERT(ne02 == ne2);
    GGML_ASSERT(ne03 == ne3);

    GGML_ASSERT(nb00 == পরিমাণ(ভিন্ন));
    GGML_ASSERT(nb0  == পরিমাণ(ভিন্ন));

    প্রতি (পূর্ণাঙ্ক i3 = 0; i3 < ne3; i3++) {
        প্রতি (পূর্ণাঙ্ক i2 = 0; i2 < ne2; i2++) {
            প্রতি (পূর্ণাঙ্ক i1 = 0; i1 < ne1; i1++) {
                ভিন্ন * d = (ভিন্ন *)((বর্ণ *)  dst->data + i3*nb3  + i2*nb2 + i1*nb1);
                ভিন্ন * s = (ভিন্ন *)((বর্ণ *) src0->data + i3*nb03 + i2*nb02);
                প্রতি (পূর্ণাঙ্ক i0 = 0; i0 < i1; i0++) {
                    d[i0] = 0;
                }
                d[i1] = s[i1];
                প্রতি (পূর্ণাঙ্ক i0 = i1+1; i0 < ne0; i0++) {
                    d[i0] = 0;
                }
            }
        }
    }
}

শূন্য ggml_compute_forward_diag(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_diag_f32(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_diag_mask_inf

স্থির শূন্য ggml_compute_forward_diag_mask_f32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst,
        ধ্রুব ভিন্ন value) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক  n_past  = ((int32_t *) dst->op_params)[0];
    ধ্রুব তার্কিক inplace = src0->data == dst->data;

    GGML_ASSERT(n_past >= 0);

    যদি (!inplace) {
        যদি (ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
            GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));
            memcpy(
                ((বর্ণ *)  dst->data),
                ((বর্ণ *) src0->data),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    // TODO: handle transposed/permuted matrices

    ধ্রুব পূর্ণাঙ্ক n  = ggml_nrows(src0);
    ধ্রুব পূর্ণাঙ্ক nc = src0->ne[0];
    ধ্রুব পূর্ণাঙ্ক nr = src0->ne[1];
    ধ্রুব পূর্ণাঙ্ক nz = n/nr;

    GGML_ASSERT( dst->nb[0] == পরিমাণ(ভিন্ন));
    GGML_ASSERT(src0->nb[0] == পরিমাণ(ভিন্ন));

    প্রতি (পূর্ণাঙ্ক k = 0; k < nz; k++) {
        প্রতি (পূর্ণাঙ্ক j = ith; j < nr; j += nth) {
            প্রতি (পূর্ণাঙ্ক i = n_past; i < nc; i++) {
                যদি (i > n_past + j) {
                    *(ভিন্ন *)((বর্ণ *) dst->data + k*dst->nb[2] + j*dst->nb[1] + i*dst->nb[0]) = value;
                }
            }
        }
    }
}

শূন্য ggml_compute_forward_diag_mask_inf(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_diag_mask_f32(params, dst, -INFINITY);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

শূন্য ggml_compute_forward_diag_mask_zero(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_diag_mask_f32(params, dst, 0);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_soft_max

স্থির শূন্য ggml_compute_forward_soft_max_f32(
        ধ্রুব ggml_compute_params * params,
              ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];
    ধ্রুব ggml_tensor * src2 = dst->src[2];

    প্রতিজ্ঞা(ggml_is_contiguous(dst));
    প্রতিজ্ঞা(ggml_are_same_shape(src0, dst));

    ভিন্ন scale    = 1.0f;
    ভিন্ন max_bias = 0.0f;

    memcpy(&scale,    (ভিন্ন *) dst->op_params + 0, পরিমাণ(ভিন্ন));
    memcpy(&max_bias, (ভিন্ন *) dst->op_params + 1, পরিমাণ(ভিন্ন));

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    ধ্রুব int64_t nb11 = src1 ? src1->nb[1] : 1;
    ধ্রুব int64_t nb12 = src1 ? src1->nb[2] : 1;
    ধ্রুব int64_t nb13 = src1 ? src1->nb[3] : 1;

    ধ্রুব int64_t ne12 = src1 ? src1->ne[2] : 1;
    ধ্রুব int64_t ne13 = src1 ? src1->ne[3] : 1;

    // TODO: is this supposed to be ceil instead of floor?
    //       https://huggingface.co/mosaicml/mpt-7b/blob/main/attention.py#L370
    ধ্রুব uint32_t n_head      = ne02;
    ধ্রুব uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));

    ধ্রুব ভিন্ন m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    ধ্রুব ভিন্ন m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    ভিন্ন * wp = (ভিন্ন *) params->wdata + (ne00 + CACHE_সারি_SIZE_F32) * ith;

    ধ্রুব তার্কিক use_f16 = (src1 && src1->type == GGML_TYPE_F16);

    // sinks
    ধ্রুব ভিন্ন * sk = src2 ? (ভিন্ন *)((বর্ণ *) src2->data) : nullptr;

    প্রতি (int64_t i03 = 0; i03 < ne03; i03++) {
        প্রতি (int64_t i02 = 0; i02 < ne02; i02++) {
            প্রতি (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                ধ্রুব int64_t i11 = i01;
                ধ্রুব int64_t i12 = i02%ne12;
                ধ্রুব int64_t i13 = i03%ne13;

                // ALiBi
                ধ্রুব uint32_t h = i02; // head
                ধ্রুব ভিন্ন slope = (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1) : 1.0f;

                ভিন্ন * sp = (ভিন্ন *)((বর্ণ *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
                ভিন্ন * dp = (ভিন্ন *)((বর্ণ *)  dst->data + i01*nb1  + i02*nb2  + i03*nb3);

                // broadcast the mask across rows
                ggml_fp16_t * mp_f16 = src1 ? (ggml_fp16_t *)((বর্ণ *) src1->data + i11*nb11 + i12*nb12 + i13*nb13) : রিক্ত;
                ভিন্ন       * mp_f32 = src1 ? (ভিন্ন       *)((বর্ণ *) src1->data + i11*nb11 + i12*nb12 + i13*nb13) : রিক্ত;

                ggml_vec_cpy_f32  (ne00, wp, sp);
                ggml_vec_scale_f32(ne00, wp, scale);
                যদি (mp_f32) {
                    যদি (use_f16) {
                        প্রতি (পূর্ণাঙ্ক i = 0; i < ne00; ++i) {
                            wp[i] += slope*GGML_CPU_FP16_TO_FP32(mp_f16[i]);
                        }
                    } অন্যথা {
                        প্রতি (পূর্ণাঙ্ক i = 0; i < ne00; ++i) {
                            wp[i] += slope*mp_f32[i];
                        }
                    }
                }

#ifndef NDEBUG
                প্রতি (পূর্ণাঙ্ক i = 0; i < ne00; ++i) {
                    //printf("p[%d] = %f\n", i, p[i]);
                    প্রতিজ্ঞা(!isnan(wp[i]));
                }
#endif // NDEBUG

                ভিন্ন max = -INFINITY;
                ggml_vec_max_f32(ne00, &max, wp);

                // if we have sinks, make a correction as if they were included in the softmax
                যদি (sk) {
                    max = MAX(max, sk[i02]);
                }

                ggml_float sum = ggml_vec_soft_max_f32(ne00, dp, wp, max);
                প্রতিজ্ঞা(sum > 0.0);

                যদি (sk) {
                    sum += (ggml_float) expf(sk[i02] - max);
                }

                sum = 1.0/sum;
                ggml_vec_scale_f32(ne00, dp, sum);

#ifndef NDEBUG
                প্রতি (পূর্ণাঙ্ক i = 0; i < ne00; ++i) {
                    প্রতিজ্ঞা(!isnan(dp[i]));
                    প্রতিজ্ঞা(!isinf(dp[i]));
                }
#endif // NDEBUG
            }
        }
    }
}

শূন্য ggml_compute_forward_soft_max(
        ধ্রুব ggml_compute_params * params,
              ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_soft_max_f32(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}


// ggml_compute_forward_soft_max_ext_back

স্থির শূন্য ggml_compute_forward_soft_max_ext_back_f32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_are_same_shape(src1, dst));

    ভিন্ন scale    = 1.0f;
    ভিন্ন max_bias = 0.0f;

    memcpy(&scale,    (ধ্রুব ভিন্ন *) dst->op_params + 0, পরিমাণ(ভিন্ন));
    memcpy(&max_bias, (ধ্রুব ভিন্ন *) dst->op_params + 1, পরিমাণ(ভিন্ন));

    GGML_ASSERT(max_bias == 0.0f);

    // TODO: handle transposed/permuted matrices

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক nc = src0->ne[0];
    ধ্রুব পূর্ণাঙ্ক nr = ggml_nrows(src0);

    // rows per thread
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr*ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    প্রতি (পূর্ণাঙ্ক i1 = ir0; i1 < ir1; i1++) {
        ভিন্ন *dy = (ভিন্ন *)((বর্ণ *) src0->data + i1*src0->nb[1]);
        ভিন্ন *y  = (ভিন্ন *)((বর্ণ *) src1->data + i1*src1->nb[1]);
        ভিন্ন *dx = (ভিন্ন *)((বর্ণ *) dst->data  + i1*dst->nb[1]);

#ifndef NDEBUG
        প্রতি (পূর্ণাঙ্ক i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            প্রতিজ্ঞা(!isnan(dy[i]));
            প্রতিজ্ঞা(!isnan(y[i]));
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
        ভিন্ন dot_y_dy = 0;
        ggml_vec_dot_f32(nc, &dot_y_dy, 0, y, 0, dy, 0, 1);
        প্রতি (পূর্ণাঙ্ক i = 0; i < nc; i++) {
            dx[i] = scale * (dy[i] - dot_y_dy) * y[i];
        }

#ifndef NDEBUG
        প্রতি (পূর্ণাঙ্ক i = 0; i < nc; ++i) {
            প্রতিজ্ঞা(!isnan(dx[i]));
            প্রতিজ্ঞা(!isinf(dx[i]));
        }
#endif // NDEBUG
    }
}

শূন্য ggml_compute_forward_soft_max_ext_back(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_soft_max_ext_back_f32(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_clamp

স্থির শূন্য ggml_compute_forward_clamp_f32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    ভিন্ন min;
    ভিন্ন max;
    memcpy(&min, (ভিন্ন *) dst->op_params + 0, পরিমাণ(ভিন্ন));
    memcpy(&max, (ভিন্ন *) dst->op_params + 1, পরিমাণ(ভিন্ন));

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক n  = ggml_nrows(src0);
    ধ্রুব পূর্ণাঙ্ক nc = src0->ne[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == পরিমাণ(ভিন্ন));
    GGML_ASSERT(nb00 == পরিমাণ(ভিন্ন));

    প্রতি (পূর্ণাঙ্ক j = ith; j < n; j += nth) {
        ধ্রুব int64_t i1 = j % ne01;
        ধ্রুব int64_t i2 = (j / ne01) % ne02;
        ধ্রুব int64_t i3 = j / (ne01*ne02);

        ভিন্ন * dst_ptr  = (ভিন্ন *) ((বর্ণ *)  dst->data + i1*nb1  + i2*nb2  + i3*nb3);
        ভিন্ন * src0_ptr = (ভিন্ন *) ((বর্ণ *) src0->data + i1*nb01 + i2*nb02 + i3*nb03);

        প্রতি (পূর্ণাঙ্ক i = 0; i < nc; i++) {
            dst_ptr[i] = MAX(MIN(src0_ptr[i], max), min);
        }
    }
}

স্থির শূন্য ggml_compute_forward_clamp_f16(
    ধ্রুব ggml_compute_params * params,
    ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    ভিন্ন min;
    ভিন্ন max;
    memcpy(&min, (ভিন্ন *) dst->op_params + 0, পরিমাণ(ভিন্ন));
    memcpy(&max, (ভিন্ন *) dst->op_params + 1, পরিমাণ(ভিন্ন));

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক n  = ggml_nrows(src0);
    ধ্রুব পূর্ণাঙ্ক nc = src0->ne[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == পরিমাণ(ggml_fp16_t));
    GGML_ASSERT(nb00 == পরিমাণ(ggml_fp16_t));

    প্রতি (পূর্ণাঙ্ক j = ith; j < n; j += nth) {
        ধ্রুব int64_t i1 = j % ne01;
        ধ্রুব int64_t i2 = (j / ne01) % ne02;
        ধ্রুব int64_t i3 = j / (ne01*ne02);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((বর্ণ *)  dst->data + i1*nb1  + i2*nb2  + i3*nb3);
        ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((বর্ণ *) src0->data + i1*nb01 + i2*nb02 + i3*nb03);

        প্রতি (পূর্ণাঙ্ক i = 0; i < nc; i++) {
            ভিন্ন v = GGML_CPU_FP16_TO_FP32(src0_ptr[i]);
            dst_ptr[i] = GGML_CPU_FP32_TO_FP16(MAX(MIN(v, max), min));
        }
    }
}

শূন্য ggml_compute_forward_clamp(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_clamp_f32(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_F16:
            {
                ggml_compute_forward_clamp_f16(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_BF16:
        পক্ষ GGML_TYPE_Q1_0:
        পক্ষ GGML_TYPE_Q2_0:
        পক্ষ GGML_TYPE_Q4_0:
        পক্ষ GGML_TYPE_Q4_1:
        পক্ষ GGML_TYPE_Q5_0:
        পক্ষ GGML_TYPE_Q5_1:
        পক্ষ GGML_TYPE_Q8_0:
        পক্ষ GGML_TYPE_Q8_1:
        পক্ষ GGML_TYPE_MXFP4:
        পক্ষ GGML_TYPE_NVFP4:
        পক্ষ GGML_TYPE_Q2_K:
        পক্ষ GGML_TYPE_Q3_K:
        পক্ষ GGML_TYPE_Q4_K:
        পক্ষ GGML_TYPE_Q5_K:
        পক্ষ GGML_TYPE_Q6_K:
        পক্ষ GGML_TYPE_TQ1_0:
        পক্ষ GGML_TYPE_TQ2_0:
        পক্ষ GGML_TYPE_IQ2_XXS:
        পক্ষ GGML_TYPE_IQ2_XS:
        পক্ষ GGML_TYPE_IQ3_XXS:
        পক্ষ GGML_TYPE_IQ1_S:
        পক্ষ GGML_TYPE_IQ1_M:
        পক্ষ GGML_TYPE_IQ4_NL:
        পক্ষ GGML_TYPE_IQ4_XS:
        পক্ষ GGML_TYPE_IQ3_S:
        পক্ষ GGML_TYPE_IQ2_S:
        পক্ষ GGML_TYPE_Q8_K:
        পক্ষ GGML_TYPE_I8:
        পক্ষ GGML_TYPE_I16:
        পক্ষ GGML_TYPE_I32:
        পক্ষ GGML_TYPE_I64:
        পক্ষ GGML_TYPE_F64:
        পক্ষ GGML_TYPE_গুনতি:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_rope

স্থির ভিন্ন rope_yarn_ramp(ধ্রুব ভিন্ন নিচের, ধ্রুব ভিন্ন উপরের, ধ্রুব পূর্ণাঙ্ক i0) {
    ধ্রুব ভিন্ন y = (i0 / 2 - নিচের) / MAX(0.001f, উপরের - নিচের);
    প্রত্যাগম 1 - MIN(1, MAX(0, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
স্থির শূন্য rope_yarn(
    ভিন্ন theta_extrap, ভিন্ন freq_scale, ভিন্ন corr_dims[2], int64_t i0, ভিন্ন ext_factor, ভিন্ন mscale,
    ভিন্ন * cos_theta, ভিন্ন * sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    ভিন্ন theta_interp = freq_scale * theta_extrap;
    ভিন্ন theta = theta_interp;
    যদি (ext_factor != 0.0f) {
        ভিন্ন ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
}

স্থির শূন্য ggml_rope_cache_init(
     ভিন্ন theta_base, ভিন্ন freq_scale, ধ্রুব ভিন্ন * freq_factors, ভিন্ন corr_dims[2], int64_t ne0, ভিন্ন ext_factor, ভিন্ন mscale,
     ভিন্ন * cache, ভিন্ন sin_sign, ভিন্ন theta_scale) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    ভিন্ন theta = theta_base;
    প্রতি (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        ধ্রুব ভিন্ন ff = freq_factors ? freq_factors[i0/2] : 1.0f;
        rope_yarn(
            theta/ff, freq_scale, corr_dims, i0, ext_factor, mscale, &cache[i0 + 0], &cache[i0 + 1]
        );
        cache[i0 + 1] *= sin_sign;

        theta *= theta_scale;
    }
}

স্থির শূন্য ggml_mrope_cache_init(
     ভিন্ন theta_base_t, ভিন্ন theta_base_h, ভিন্ন theta_base_w, ভিন্ন theta_base_e, পূর্ণাঙ্ক sections[4], তার্কিক is_imrope, তার্কিক indep_sects,
     ভিন্ন freq_scale, ধ্রুব ভিন্ন * freq_factors, ভিন্ন corr_dims[2], int64_t ne0, ভিন্ন ext_factor, ভিন্ন mscale,
     ভিন্ন * cache, ভিন্ন sin_sign, ভিন্ন theta_scale) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    ভিন্ন theta_t = theta_base_t;
    ভিন্ন theta_h = theta_base_h;
    ভিন্ন theta_w = theta_base_w;
    ভিন্ন theta_e = theta_base_e;  // extra position id for vision encoder
    পূর্ণাঙ্ক sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    পূর্ণাঙ্ক sec_w = sections[1] + sections[0];
    পূর্ণাঙ্ক sec_e = sections[2] + sec_w;
    GGML_ASSERT(sect_dims <= ne0);

    প্রতি (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        ধ্রুব ভিন্ন ff = freq_factors ? freq_factors[i0/2] : 1.0f;

        পূর্ণাঙ্ক sector = (i0 / 2) % sect_dims;
        যদি (indep_sects) {
            // compute theta independently for each dim sections
            // (i.e. reset corresponding theta when `i0` go from one section to another)
            যদি (sector == 0) {
                theta_t = theta_base_t;
            }
            অন্যথা যদি (sector == sections[0]) {
                theta_h = theta_base_h;;
            }
            অন্যথা যদি (sector == sec_w) {
                theta_w = theta_base_w;
            }
            অন্যথা যদি (sector == sec_e) {
                theta_e = theta_base_e;
            }
        }

        ভিন্ন theta = theta_t;
        যদি (is_imrope) { // qwen3vl apply interleaved mrope
            যদি (sector % 3 == 1 && sector < 3 * sections[1]) {
                theta = theta_h;
            } অন্যথা যদি (sector % 3 == 2 && sector < 3 * sections[2]) {
                theta = theta_w;
            } অন্যথা যদি (sector % 3 == 0 && sector < 3 * sections[0]) {
                theta = theta_t;
            } অন্যথা {
                theta = theta_e;
            }
        } অন্যথা {
            যদি (sector >= sections[0] && sector < sec_w) {
                theta = theta_h;
            }
            অন্যথা যদি (sector >= sec_w && sector < sec_w + sections[2]) {
                theta = theta_w;
            }
            অন্যথা যদি (sector >= sec_w + sections[2]) {
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
স্থির শূন্য rotate_pairs(ধ্রুব int64_t n, ধ্রুব int64_t n_offset, ধ্রুব ভিন্ন * cache, ধ্রুব T * src_data, T * dst_data, ধ্রুব পূর্ণাঙ্ক scale = 2) {
  প্রতি (int64_t i0 = 0; i0 < n; i0 += 2) {
    ধ্রুব int64_t ic = i0/scale; // hack for GGML_ROPE_TYPE_NORMAL, where we need ic = i0; for all other cases, ic = i0/2

    ধ্রুব ভিন্ন cos_theta = cache[i0 + 0];
    ধ্রুব ভিন্ন sin_theta = cache[i0 + 1];

    ধ্রুব T * ধ্রুব src = src_data + ic;
    T * dst             = dst_data + ic;

    ধ্রুব ভিন্ন x0 = type_conversion_table<T>::to_f32(src[0]);
    ধ্রুব ভিন্ন x1 = type_conversion_table<T>::to_f32(src[n_offset]);

    dst[0]        = type_conversion_table<T>::from_f32(x0*cos_theta - x1*sin_theta);
    dst[n_offset] = type_conversion_table<T>::from_f32(x0*sin_theta + x1*cos_theta);
  }
}

template<typename T> //float or ggml_fp16_t
স্থির শূন্য ggml_compute_forward_rope_flt(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst,
        ধ্রুব তার্কিক forward) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];
    ধ্রুব ggml_tensor * src2 = dst->src[2];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);

    ভিন্ন freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    পূর্ণাঙ্ক sections[4];

    //const int n_past     = ((int32_t *) dst->op_params)[0];
    ধ্রুব পূর্ণাঙ্ক n_dims     = ((int32_t *) dst->op_params)[1];
    ধ্রুব পূর্ণাঙ্ক mode       = ((int32_t *) dst->op_params)[2];
    //const int n_ctx      = ((int32_t *) dst->op_params)[3];
    ধ্রুব পূর্ণাঙ্ক n_ctx_orig = ((int32_t *) dst->op_params)[4];

    memcpy(&freq_base,   (int32_t *) dst->op_params +  5, পরিমাণ(ভিন্ন));
    memcpy(&freq_scale,  (int32_t *) dst->op_params +  6, পরিমাণ(ভিন্ন));
    memcpy(&ext_factor,  (int32_t *) dst->op_params +  7, পরিমাণ(ভিন্ন));
    memcpy(&attn_factor, (int32_t *) dst->op_params +  8, পরিমাণ(ভিন্ন));
    memcpy(&beta_fast,   (int32_t *) dst->op_params +  9, পরিমাণ(ভিন্ন));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, পরিমাণ(ভিন্ন));
    memcpy(&sections,    (int32_t *) dst->op_params + 11, পরিমাণ(পূর্ণাঙ্ক)*4);

    ধ্রুব পূর্ণাঙ্ক n_offs = ((int32_t *) dst->op_params)[15];

    GGML_TENSOR_UNARY_OP_LOCALS

    //printf("ne0: %d, ne1: %d, ne2: %d, ne3: %d\n", ne0, ne1, ne2, ne3);
    //printf("n_past = %d, ne2 = %d\n", n_past, ne2);

    GGML_ASSERT(nb0 == nb00);
    GGML_ASSERT(nb0 == পরিমাণ(T));

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক nr = ggml_nrows(dst);

    GGML_ASSERT(n_dims <= ne0);
    GGML_ASSERT(n_dims % 2 == 0);

    GGML_ASSERT(n_offs >= 0);
    GGML_ASSERT(n_offs % 2 == 0);
    GGML_ASSERT(n_offs + n_dims <= ne0);

    // rows per thread
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr*ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    // row index used to determine which thread to use
    পূর্ণাঙ্ক ir = 0;

    ধ্রুব ভিন্ন theta_scale = powf(freq_base, -2.0f/n_dims);

    ভিন্ন corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    ধ্রুব তার্কিক is_imrope = mode == GGML_ROPE_TYPE_IMROPE; // qwen3vl apply interleaved mrope
    ধ্রুব তার্কিক mrope_used = mode & GGML_ROPE_TYPE_MROPE;  // ggml_rope_multi, note: also true for vision (24 & 8 == true) and for imrope
    ধ্রুব তার্কিক is_vision = mode == GGML_ROPE_TYPE_VISION;

    যদি (mrope_used) {
        GGML_ASSERT(sections[0] > 0 || sections[1] > 0 || sections[2] > 0);
    }

    যদি (is_vision) {
        GGML_ASSERT(n_dims == ne0/2);
        GGML_ASSERT(n_offs == 0);
    }

    ধ্রুব ভিন্ন * freq_factors = রিক্ত;
    যদি (src2 != রিক্ত) {
        GGML_ASSERT(src2->type == GGML_TYPE_F32);
        GGML_ASSERT(src2->ne[0] >= n_dims / 2);
        freq_factors = (ধ্রুব ভিন্ন *) src2->data;
    }

    // backward process uses inverse rotation by cos and sin.
    // cos and sin build a rotation matrix, where the inverse is the transpose.
    // this essentially just switches the sign of sin.
    ধ্রুব ভিন্ন sin_sign = forward ? 1.0f : -1.0f;

    ধ্রুব int32_t * pos = (ধ্রুব int32_t *) src1->data;

    int64_t last_i2 = -1;

    প্রতি (int64_t i3 = 0; i3 < ne3; i3++) { // batch
        প্রতি (int64_t i2 = 0; i2 < ne2; i2++) { // seq-len
            প্রতি (int64_t i1 = 0; i1 < ne1; i1++) { // attn-heads
                যদি (ir++ < ir0) অনুবর্ত; // skip rows mapped to other threads
                যদি (ir   > ir1) ভঙ্গ;

                ভিন্ন * cache = (ভিন্ন *) params->wdata + (ne0 + CACHE_সারি_SIZE_F32)*ith;
                যদি (last_i2 != i2) {
                    যদি (!mrope_used) {
                        ধ্রুব int64_t p = pos[i2];
                        ggml_rope_cache_init(p, freq_scale, freq_factors, corr_dims, ne0, ext_factor, attn_factor, cache, sin_sign, theta_scale);
                    }
                    অন্যথা {
                        ধ্রুব int64_t p_t = pos[i2];
                        ধ্রুব int64_t p_h = pos[i2 + ne2];
                        ধ্রুব int64_t p_w = pos[i2 + ne2 * 2];
                        ধ্রুব int64_t p_e = pos[i2 + ne2 * 3];
                        ggml_mrope_cache_init(
                            p_t, p_h, p_w, p_e, sections, is_imrope, is_vision,
                            freq_scale, freq_factors, corr_dims, ne0, ext_factor, attn_factor, cache, sin_sign, theta_scale);
                    }

                    last_i2 = i2;
                }

                T * src = (T *)((বর্ণ *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
                T * dst_data  = (T *)((বর্ণ *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1);

                বিকল্প (mode) {
                    পক্ষ GGML_ROPE_TYPE_NORMAL:
                        rotate_pairs<T>(n_dims, 1, cache, src + n_offs, dst_data + n_offs, 1);
                        ভঙ্গ;
                    পক্ষ GGML_ROPE_TYPE_NEOX:
                    পক্ষ GGML_ROPE_TYPE_MROPE:
                    পক্ষ GGML_ROPE_TYPE_IMROPE:
                        rotate_pairs<T>(n_dims, n_dims/2, cache, src + n_offs, dst_data + n_offs);
                        ভঙ্গ;
                    পক্ষ GGML_ROPE_TYPE_VISION:
                        rotate_pairs<T>(ne0, n_dims, cache, src, dst_data);
                        ভঙ্গ;
                    স্বভাব:
                        GGML_ABORT("rope type not supported");
                }

                যদি (!is_vision) {
                    // fill the remain channels with data from src tensor
                    প্রতি (int64_t i0 = 0; i0 < ne0; i0 += 2) {
                        যদি (i0 == n_offs) {
                            i0 += n_dims - 2; // skip the rotated channels
                            অনুবর্ত;
                        }
                        ধ্রুব T * ধ্রুব src = (T *)((বর্ণ *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                        T * dst_data  = (T *)((বর্ণ *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

                        dst_data[0] = src[0];
                        dst_data[1] = src[1];
                    }
                }
            } //attn-heads
        }
    }
}

শূন্য ggml_compute_forward_rope(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F16:
            {
                ggml_compute_forward_rope_flt<ggml_fp16_t>(params, dst, সত্য);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_rope_flt<ভিন্ন>(params, dst, সত্য);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_rope_back

শূন্য ggml_compute_forward_rope_back(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F16:
            {
                ggml_compute_forward_rope_flt<ggml_fp16_t>(params, dst, অসত্য);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_rope_flt<ভিন্ন>(params, dst, অসত্য);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_conv_transpose_1d

স্থির শূন্য ggml_compute_forward_conv_transpose_1d_f16_f32(
        ধ্রুব ggml_compute_params * params,
              ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_দ্বিমিক_OP_LOCALS

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক nk = ne00*ne01*ne02;

    GGML_ASSERT(nb00 == পরিমাণ(ggml_fp16_t));
    GGML_ASSERT(nb10 == পরিমাণ(ভিন্ন));

    যদি (ith == 0) {
        memset(params->wdata, 0, params->wsize);

        // permute kernel data (src0) from (K x Cout x Cin) to (Cin x K x Cout)
        {
            ggml_fp16_t * ধ্রুব wdata = (ggml_fp16_t *) params->wdata + 0;

            প্রতি (int64_t i02 = 0; i02 < ne02; i02++) {
                প্রতি (int64_t i01 = 0; i01 < ne01; i01++) {
                    ধ্রুব ggml_fp16_t * ধ্রুব src = (ggml_fp16_t *)((বর্ণ *) src0->data + i02*nb02 + i01*nb01);
                    ggml_fp16_t * dst_data = wdata + i01*ne00*ne02;
                    প্রতি (int64_t i00 = 0; i00 < ne00; i00++) {
                        dst_data[i00*ne02 + i02] = src[i00];
                    }
                }
            }
        }

        // permute source data (src1) from (L x Cin) to (Cin x L)
        {
            ggml_fp16_t * ধ্রুব wdata = (ggml_fp16_t *) params->wdata + nk;
            ggml_fp16_t * dst_data = wdata;

            প্রতি (int64_t i11 = 0; i11 < ne11; i11++) {
                ধ্রুব ভিন্ন * ধ্রুব src = (ভিন্ন *)((বর্ণ *) src1->data + i11*nb11);
                প্রতি (int64_t i10 = 0; i10 < ne10; i10++) {
                    dst_data[i10*ne11 + i11] = GGML_CPU_FP32_TO_FP16(src[i10]);
                }
            }
        }

        // need to zero dst since we are accumulating into it
        memset(dst->data, 0, ggml_nbytes(dst));
    }
    ggml_barrier(params->threadpool);

    ধ্রুব int32_t s0 = ((ধ্রুব int32_t*)(dst->op_params))[0];

    // total rows in dst
    ধ্রুব পূর্ণাঙ্ক nr = ne1;

    // rows per thread
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr*ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    ggml_fp16_t * ধ্রুব wdata     = (ggml_fp16_t *) params->wdata + 0;
    ggml_fp16_t * ধ্রুব wdata_src = wdata + nk;

    প্রতি (পূর্ণাঙ্ক i1 = ir0; i1 < ir1; i1++) {
        ভিন্ন * dst_data = (ভিন্ন *)((বর্ণ *) dst->data + i1*nb1);
        ggml_fp16_t * wdata_kernel = wdata + i1*ne02*ne00;
        প্রতি (পূর্ণাঙ্ক i10 = 0; i10 < ne10; i10++) {
            ধ্রুব পূর্ণাঙ্ক i1n = i10*ne11;
            প্রতি (পূর্ণাঙ্ক i00 = 0; i00 < ne00; i00++) {
                ভিন্ন v = 0;
                ggml_vec_dot_f16(ne02, &v, 0,
                        (ggml_fp16_t *)    wdata_src + i1n, 0,
                        (ggml_fp16_t *) wdata_kernel + i00*ne02, 0, 1);
                dst_data[i10*s0 + i00] += v;
            }
        }
    }
}

স্থির শূন্য ggml_compute_forward_conv_transpose_1d_f32(
        ধ্রুব ggml_compute_params * params,
              ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_দ্বিমিক_OP_LOCALS

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক nk = ne00*ne01*ne02;

    GGML_ASSERT(nb00 == পরিমাণ(ভিন্ন));
    GGML_ASSERT(nb10 == পরিমাণ(ভিন্ন));

    যদি (ith == 0) {
        memset(params->wdata, 0, params->wsize);

        // prepare kernel data (src0) from (K x Cout x Cin) to (Cin x K x Cout)
        {
            ভিন্ন * ধ্রুব wdata = (ভিন্ন *) params->wdata + 0;

            প্রতি (int64_t i02 = 0; i02 < ne02; i02++) {
                প্রতি (int64_t i01 = 0; i01 < ne01; i01++) {
                    ধ্রুব ভিন্ন * ধ্রুব src = (ভিন্ন *)((বর্ণ *) src0->data + i02*nb02 + i01*nb01);
                    ভিন্ন * dst_data = wdata + i01*ne00*ne02;
                    প্রতি (int64_t i00 = 0; i00 < ne00; i00++) {
                        dst_data[i00*ne02 + i02] = src[i00];
                    }
                }
            }
        }

        // prepare source data (src1)
        {
            ভিন্ন * ধ্রুব wdata = (ভিন্ন *) params->wdata + nk;
            ভিন্ন * dst_data = wdata;

            প্রতি (int64_t i11 = 0; i11 < ne11; i11++) {
                ধ্রুব ভিন্ন * ধ্রুব src = (ভিন্ন *)((বর্ণ *) src1->data + i11*nb11);
                প্রতি (int64_t i10 = 0; i10 < ne10; i10++) {
                    dst_data[i10*ne11 + i11] = src[i10];
                }
            }
        }

        // need to zero dst since we are accumulating into it
        memset(dst->data, 0, ggml_nbytes(dst));
    }
    ggml_barrier(params->threadpool);

    ধ্রুব int32_t s0 = ((ধ্রুব int32_t*)(dst->op_params))[0];

    // total rows in dst
    ধ্রুব পূর্ণাঙ্ক nr = ne1;

    // rows per thread
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr*ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    ভিন্ন * ধ্রুব wdata     = (ভিন্ন *) params->wdata + 0;
    ভিন্ন * ধ্রুব wdata_src = wdata + nk;

    প্রতি (পূর্ণাঙ্ক i1 = ir0; i1 < ir1; i1++) {
        ভিন্ন * dst_data = (ভিন্ন *)((বর্ণ *) dst->data + i1*nb1);
        ভিন্ন * wdata_kernel = wdata + i1*ne02*ne00;
        প্রতি (পূর্ণাঙ্ক i10 = 0; i10 < ne10; i10++) {
            ধ্রুব পূর্ণাঙ্ক i1n = i10*ne11;
            প্রতি (পূর্ণাঙ্ক i00 = 0; i00 < ne00; i00++) {
                ভিন্ন v = 0;
                ggml_vec_dot_f32(ne02, &v, 0,
                        wdata_src + i1n, 0,
                        wdata_kernel + i00*ne02, 0, 1);
                dst_data[i10*s0 + i00] += v;
            }
        }
    }
}

শূন্য ggml_compute_forward_conv_transpose_1d(
        ধ্রুব ggml_compute_params * params,
              ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F16:
            {
                ggml_compute_forward_conv_transpose_1d_f16_f32(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_conv_transpose_1d_f32(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_im2col_f32
// src0: kernel [OC, IC, KH, KW]
// src1: image [N, IC, IH, IW]
// dst:  result [N, OH, OW, IC*KH*KW]
স্থির শূন্য ggml_compute_forward_im2col_f32(
        ধ্রুব ggml_compute_params * params,
              ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_দ্বিমিক_OP_LOCALS;

    ধ্রুব int32_t s0 = ((ধ্রুব int32_t *)(dst->op_params))[0];
    ধ্রুব int32_t s1 = ((ধ্রুব int32_t *)(dst->op_params))[1];
    ধ্রুব int32_t p0 = ((ধ্রুব int32_t *)(dst->op_params))[2];
    ধ্রুব int32_t p1 = ((ধ্রুব int32_t *)(dst->op_params))[3];
    ধ্রুব int32_t d0 = ((ধ্রুব int32_t *)(dst->op_params))[4];
    ধ্রুব int32_t d1 = ((ধ্রুব int32_t *)(dst->op_params))[5];
    ধ্রুব তার্কিক is_2D = ((ধ্রুব int32_t *)(dst->op_params))[6] == 1;

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব int64_t N  = is_2D ? ne13 : ne12;
    ধ্রুব int64_t IC = is_2D ? ne12 : ne11;
    ধ্রুব int64_t IH = is_2D ? ne11 : 1;
    ধ্রুব int64_t IW = ne10;

    ধ্রুব int64_t KH = is_2D ? ne01 : 1;
    ধ্রুব int64_t KW = ne00;

    ধ্রুব int64_t OH = is_2D ? ne2 : 1;
    ধ্রুব int64_t OW = ne1;

    পূর্ণাঙ্ক ofs0 = is_2D ? nb13 : nb12;
    পূর্ণাঙ্ক ofs1 = is_2D ? nb12 : nb11;

    GGML_ASSERT(nb10 == পরিমাণ(ভিন্ন));

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    {
        ভিন্ন * ধ্রুব wdata = (ভিন্ন *) dst->data;

        প্রতি (int64_t in = 0; in < N; in++) {
            প্রতি (int64_t ioh = 0; ioh < OH; ioh++) { // 1
                প্রতি (int64_t iow = 0; iow < OW; iow++) {
                    প্রতি (int64_t iic = ith; iic < IC; iic += nth) {

                        // micro kernel
                        ভিন্ন * dst_data = wdata + (in*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
                        ধ্রুব ভিন্ন * ধ্রুব src_data = (ভিন্ন *)((বর্ণ *) src1->data + in*ofs0 + iic*ofs1); // [IH, IW]

                        প্রতি (int64_t ikh = 0; ikh < KH; ikh++) {  // 1
                            প্রতি (int64_t ikw = 0; ikw < KW; ikw++) {
                                ধ্রুব int64_t iiw = iow*s0 + ikw*d0 - p0;
                                ধ্রুব int64_t iih = ioh*s1 + ikh*d1 - p1;

                                যদি (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                    dst_data[iic*(KH*KW) + ikh*KW + ikw] = 0;
                                } অন্যথা {
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
স্থির শূন্য ggml_compute_forward_im2col_f16(
        ধ্রুব ggml_compute_params * params,
              ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16);

    GGML_TENSOR_দ্বিমিক_OP_LOCALS;

    ধ্রুব int32_t s0 = ((ধ্রুব int32_t *)(dst->op_params))[0];
    ধ্রুব int32_t s1 = ((ধ্রুব int32_t *)(dst->op_params))[1];
    ধ্রুব int32_t p0 = ((ধ্রুব int32_t *)(dst->op_params))[2];
    ধ্রুব int32_t p1 = ((ধ্রুব int32_t *)(dst->op_params))[3];
    ধ্রুব int32_t d0 = ((ধ্রুব int32_t *)(dst->op_params))[4];
    ধ্রুব int32_t d1 = ((ধ্রুব int32_t *)(dst->op_params))[5];
    ধ্রুব তার্কিক is_2D = ((ধ্রুব int32_t *)(dst->op_params))[6] == 1;

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব int64_t N  = is_2D ? ne13 : ne12;
    ধ্রুব int64_t IC = is_2D ? ne12 : ne11;
    ধ্রুব int64_t IH = is_2D ? ne11 : 1;
    ধ্রুব int64_t IW = ne10;

    ধ্রুব int64_t KH = is_2D ? ne01 : 1;
    ধ্রুব int64_t KW = ne00;

    ধ্রুব int64_t OH = is_2D ? ne2 : 1;
    ধ্রুব int64_t OW = ne1;

    পূর্ণাঙ্ক ofs0 = is_2D ? nb13 : nb12;
    পূর্ণাঙ্ক ofs1 = is_2D ? nb12 : nb11;

    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    {
        ggml_fp16_t * ধ্রুব wdata = (ggml_fp16_t *) dst->data;

        প্রতি (int64_t in = 0; in < N; in++) {
            প্রতি (int64_t ioh = 0; ioh < OH; ioh++) { // 1
                প্রতি (int64_t iow = 0; iow < OW; iow++) {
                    প্রতি (int64_t iic = ith; iic < IC; iic += nth) {

                        // micro kernel
                        ggml_fp16_t * dst_data = wdata + (in*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
                        ধ্রুব ভিন্ন * ধ্রুব src_data_f32 = src1->type == GGML_TYPE_F32
                            ? (ধ্রুব ভিন্ন *)((ধ্রুব বর্ণ *) src1->data + in*ofs0 + iic*ofs1)
                            : nullptr; // [IH, IW]
                        ধ্রুব ggml_fp16_t * ধ্রুব src_data_f16 = src1->type == GGML_TYPE_F16
                            ? (ধ্রুব ggml_fp16_t *)((ধ্রুব বর্ণ *) src1->data + in*ofs0 + iic*ofs1)
                            : nullptr; // [IH, IW]

                        প্রতি (int64_t ikh = 0; ikh < KH; ikh++) {  // 1
                            প্রতি (int64_t ikw = 0; ikw < KW; ikw++) {
                                ধ্রুব int64_t iiw = iow*s0 + ikw*d0 - p0;
                                ধ্রুব int64_t iih = ioh*s1 + ikh*d1 - p1;

                                যদি (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                    dst_data[iic*(KH*KW) + ikh*KW + ikw] = 0;
                                } অন্যথা {
                                    যদি (src_data_f32 != nullptr) {
                                        dst_data[iic*(KH*KW) + ikh*KW + ikw] = GGML_CPU_FP32_TO_FP16(src_data_f32[iih*IW + iiw]);
                                    } অন্যথা {
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

শূন্য ggml_compute_forward_im2col(
        ধ্রুব ggml_compute_params * params,
              ggml_tensor * dst) {
    বিকল্প (dst->type) {
        পক্ষ GGML_TYPE_F16:
            {
                ggml_compute_forward_im2col_f16(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_im2col_f32(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_im2col_back_f32

শূন্য ggml_compute_forward_im2col_back_f32(
        ধ্রুব ggml_compute_params * params,
              ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0]; // gradients of forward pass output
    ধ্রুব ggml_tensor * src1 = dst->src[1]; // convolution kernel

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_দ্বিমিক_OP_LOCALS;

    ধ্রুব int32_t s0 = ((ধ্রুব int32_t *)(dst->op_params))[0];
    ধ্রুব int32_t s1 = ((ধ্রুব int32_t *)(dst->op_params))[1];
    ধ্রুব int32_t p0 = ((ধ্রুব int32_t *)(dst->op_params))[2];
    ধ্রুব int32_t p1 = ((ধ্রুব int32_t *)(dst->op_params))[3];
    ধ্রুব int32_t d0 = ((ধ্রুব int32_t *)(dst->op_params))[4];
    ধ্রুব int32_t d1 = ((ধ্রুব int32_t *)(dst->op_params))[5];
    ধ্রুব তার্কিক is_2D = ((ধ্রুব int32_t *)(dst->op_params))[6] == 1;

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব int64_t N  = is_2D ? ne3 : ne2;
    ধ্রুব int64_t IC = is_2D ? ne2 : ne1;
    ধ্রুব int64_t IH = is_2D ? ne1 : 1;
    ধ্রুব int64_t IW = ne0;

    ধ্রুব int64_t KH = is_2D ? ne11 : 1;
    ধ্রুব int64_t KW = ne10;

    ধ্রুব int64_t OH = is_2D ? ne02 : 1;
    ধ্রুব int64_t OW = ne01;

    পূর্ণাঙ্ক ofs0 = is_2D ? nb3 : nb2;
    পূর্ণাঙ্ক ofs1 = is_2D ? nb2 : nb1;

    GGML_ASSERT(nb0  == পরিমাণ(ভিন্ন));

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    {
        ভিন্ন * ধ্রুব wdata = (ভিন্ন *) dst->data;

        প্রতি (int64_t in = 0; in < N; in++) {
            প্রতি (int64_t iic = ith; iic < IC; iic += nth) {
                প্রতি (int64_t iih = 0; iih < IH; iih++) {
                    প্রতি (int64_t iiw = 0; iiw < IW; iiw++) {

                        // micro kernel
                        ভিন্ন grad = 0.0f;
                        প্রতি (int64_t ikh = 0; ikh < KH; ikh++) {
                            প্রতি (int64_t ikw = 0; ikw < KW; ikw++) {
                                // For s0 > 1 some values were skipped over in the forward pass.
                                // These values have tmpw % s0 != 0 and need to be skipped in the backwards pass as well.
                                ধ্রুব int64_t tmpw = (iiw + p0 - ikw*d0);
                                যদি (tmpw % s0 != 0) {
                                    অনুবর্ত;
                                }
                                ধ্রুব int64_t iow = tmpw / s0;

                                // Equivalent logic as above except for s1.
                                int64_t ioh;
                                যদি (is_2D) {
                                    ধ্রুব int64_t tmph = iih + p1 - ikh*d1;

                                    যদি (tmph % s1 != 0) {
                                        অনুবর্ত;
                                    }

                                    ioh = tmph / s1;
                                } অন্যথা {
                                    ioh = 0;
                                }

                                যদি (iow < 0 || iow >= OW || ioh < 0 || ioh >= OH) {
                                    অনুবর্ত;
                                }

                                ধ্রুব ভিন্ন * ধ্রুব grad_in = (ধ্রুব ভিন্ন *) src0->data
                                    + (in*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
                                grad += grad_in[iic*(KH*KW) + ikh*KW + ikw];
                            }
                        }
                        ভিন্ন * dst_data = (ভিন্ন *)((বর্ণ *) wdata + (in*ofs0 + iic*ofs1)); // [IH, IW]
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
স্থির শূন্য ggml_compute_forward_im2col_3d_f16(
        ধ্রুব ggml_compute_params * params,
              ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16);

    GGML_TENSOR_দ্বিমিক_OP_LOCALS;

    ধ্রুব int32_t s0 = ((ধ্রুব int32_t *)(dst->op_params))[0];
    ধ্রুব int32_t s1 = ((ধ্রুব int32_t *)(dst->op_params))[1];
    ধ্রুব int32_t s2 = ((ধ্রুব int32_t *)(dst->op_params))[2];
    ধ্রুব int32_t p0 = ((ধ্রুব int32_t *)(dst->op_params))[3];
    ধ্রুব int32_t p1 = ((ধ্রুব int32_t *)(dst->op_params))[4];
    ধ্রুব int32_t p2 = ((ধ্রুব int32_t *)(dst->op_params))[5];
    ধ্রুব int32_t d0 = ((ধ্রুব int32_t *)(dst->op_params))[6];
    ধ্রুব int32_t d1 = ((ধ্রুব int32_t *)(dst->op_params))[7];
    ধ্রুব int32_t d2 = ((ধ্রুব int32_t *)(dst->op_params))[8];
    ধ্রুব int32_t IC = ((ধ্রুব int32_t *)(dst->op_params))[9];


    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব int64_t N  = ne13 / IC;
    ধ্রুব int64_t ID = ne12;
    ধ্রুব int64_t IH = ne11;
    ধ্রুব int64_t IW = ne10;

    ধ্রুব int64_t OC = ne03 / IC;
    GGML_UNUSED(OC);
    ধ্রুব int64_t KD = ne02;
    ধ্রুব int64_t KH = ne01;
    ধ্রুব int64_t KW = ne00;

    ধ্রুব int64_t OD = ne3 / N;
    ধ্রুব int64_t OH = ne2;
    ধ্রুব int64_t OW = ne1;
    ধ্রুব int64_t OH_OW = OH*OW;
    ধ্রুব int64_t KD_KH_KW = KD*KH*KW;
    ধ্রুব int64_t KH_KW = KH*KW;
    ধ্রুব int64_t IC_KD_KH_KW = IC*KD*KH*KW;

    GGML_ASSERT(nb10 == পরিমাণ(ভিন্ন));

    // im2col: [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
    {
        ggml_fp16_t * ধ্রুব wdata = (ggml_fp16_t *) dst->data;

        প্রতি (int64_t in = 0; in < N; in++) {
            প্রতি (int64_t iod = 0; iod < OD; iod++) {
                প্রতি (int64_t ioh = 0; ioh < OH; ioh++) {
                    প্রতি (int64_t iow = 0; iow < OW; iow++) {
                        প্রতি (int64_t iic = ith; iic < IC; iic += nth) {

                            // micro kernel
                            ggml_fp16_t * dst_data = wdata + (in*OD*OH_OW + iod*OH_OW + ioh*OW + iow)*IC_KD_KH_KW; // [IC, KD, KH, KW]
                            ধ্রুব ভিন্ন * ধ্রুব src_data = (ধ্রুব ভিন্ন *) ((ধ্রুব বর্ণ *)src1->data + (in*IC + iic)*nb13); // [ID, IH, IW]

                            প্রতি (int64_t ikd = 0; ikd < KD; ikd++) {
                                প্রতি (int64_t ikh = 0; ikh < KH; ikh++) {
                                    প্রতি (int64_t ikw = 0; ikw < KW; ikw++) {
                                        ধ্রুব int64_t iiw = iow*s0 + ikw*d0 - p0;
                                        ধ্রুব int64_t iih = ioh*s1 + ikh*d1 - p1;
                                        ধ্রুব int64_t iid = iod*s2 + ikd*d2 - p2;

                                        যদি (iid < 0 || iid >= ID || iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                            dst_data[iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw] = 0;
                                        } অন্যথা {
                                            ধ্রুব ভিন্ন * ধ্রুব s = (ধ্রুব ভিন্ন *) ((ধ্রুব বর্ণ *)src_data + iid*nb12 + iih*nb11 + iiw*nb10); // [ID, IH, IW]
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
স্থির শূন্য ggml_compute_forward_im2col_3d_f32(
        ধ্রুব ggml_compute_params * params,
              ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_দ্বিমিক_OP_LOCALS;

    ধ্রুব int32_t s0 = ((ধ্রুব int32_t *)(dst->op_params))[0];
    ধ্রুব int32_t s1 = ((ধ্রুব int32_t *)(dst->op_params))[1];
    ধ্রুব int32_t s2 = ((ধ্রুব int32_t *)(dst->op_params))[2];
    ধ্রুব int32_t p0 = ((ধ্রুব int32_t *)(dst->op_params))[3];
    ধ্রুব int32_t p1 = ((ধ্রুব int32_t *)(dst->op_params))[4];
    ধ্রুব int32_t p2 = ((ধ্রুব int32_t *)(dst->op_params))[5];
    ধ্রুব int32_t d0 = ((ধ্রুব int32_t *)(dst->op_params))[6];
    ধ্রুব int32_t d1 = ((ধ্রুব int32_t *)(dst->op_params))[7];
    ধ্রুব int32_t d2 = ((ধ্রুব int32_t *)(dst->op_params))[8];
    ধ্রুব int32_t IC = ((ধ্রুব int32_t *)(dst->op_params))[9];


    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব int64_t N  = ne13 / IC;
    ধ্রুব int64_t ID = ne12;
    ধ্রুব int64_t IH = ne11;
    ধ্রুব int64_t IW = ne10;

    ধ্রুব int64_t OC = ne03 / IC;
    GGML_UNUSED(OC);
    ধ্রুব int64_t KD = ne02;
    ধ্রুব int64_t KH = ne01;
    ধ্রুব int64_t KW = ne00;

    ধ্রুব int64_t OD = ne3 / N;
    ধ্রুব int64_t OH = ne2;
    ধ্রুব int64_t OW = ne1;

    ধ্রুব int64_t OH_OW = OH*OW;
    ধ্রুব int64_t KD_KH_KW = KD*KH*KW;
    ধ্রুব int64_t KH_KW = KH*KW;
    ধ্রুব int64_t IC_KD_KH_KW = IC*KD*KH*KW;

    GGML_ASSERT(nb10 == পরিমাণ(ভিন্ন));

    // im2col: [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
    {
        ভিন্ন * ধ্রুব wdata = (ভিন্ন *) dst->data;

        প্রতি (int64_t in = 0; in < N; in++) {
            প্রতি (int64_t iod = 0; iod < OD; iod++) {
                প্রতি (int64_t ioh = 0; ioh < OH; ioh++) {
                    প্রতি (int64_t iow = 0; iow < OW; iow++) {
                        প্রতি (int64_t iic = ith; iic < IC; iic += nth) {

                            // micro kernel
                            ভিন্ন * dst_data = wdata + (in*OD*OH_OW + iod*OH_OW + ioh*OW + iow)*IC_KD_KH_KW; // [IC, KD, KH, KW]
                            ধ্রুব ভিন্ন * ধ্রুব src_data = (ধ্রুব ভিন্ন *) ((ধ্রুব বর্ণ *)src1->data + (in*IC + iic)*nb13); // [ID, IH, IW]

                            প্রতি (int64_t ikd = 0; ikd < KD; ikd++) {
                                প্রতি (int64_t ikh = 0; ikh < KH; ikh++) {
                                    প্রতি (int64_t ikw = 0; ikw < KW; ikw++) {
                                        ধ্রুব int64_t iiw = iow*s0 + ikw*d0 - p0;
                                        ধ্রুব int64_t iih = ioh*s1 + ikh*d1 - p1;
                                        ধ্রুব int64_t iid = iod*s2 + ikd*d2 - p2;

                                        যদি (iid < 0 || iid >= ID || iih < 0 || iih >= IH || iiw < 0 || iiw >= IW || iid < 0 || iid >= ID) {
                                            dst_data[iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw] = 0;
                                        } অন্যথা {
                                            ধ্রুব ভিন্ন * ধ্রুব s = (ধ্রুব ভিন্ন *) ((ধ্রুব বর্ণ *)src_data + iid*nb12 + iih*nb11 + iiw*nb10); // [ID, IH, IW]
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


শূন্য ggml_compute_forward_im2col_3d(
        ধ্রুব ggml_compute_params * params,
              ggml_tensor * dst) {
    বিকল্প (dst->type) {
        পক্ষ GGML_TYPE_F16:
            {
                ggml_compute_forward_im2col_3d_f16(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_im2col_3d_f32(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

স্থির শূন্য ggml_call_mul_mat(ggml_type type, ধ্রুব ggml_compute_params * params, int64_t m, int64_t n, int64_t k,
                              শূন্য * a, শূন্য * b, ভিন্ন * c) {
    ধ্রুব ggml_type_traits * traits = ggml_get_type_traits(type);
    সংরচনা ggml_tensor src1 = {};
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

    সংরচনা ggml_tensor src0 = {};
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

    সংরচনা ggml_tensor dst = {};
    dst.ne[0] = n;
    dst.ne[1] = m;
    dst.ne[2] = 1;
    dst.ne[3] = 1;
    dst.nb[0] = পরিমাণ(ভিন্ন);
    dst.nb[1] = n * পরিমাণ(ভিন্ন);
    dst.nb[2] = dst.nb[1];
    dst.nb[3] = dst.nb[2];
    dst.data  = c;
    dst.src[0] = &src0;
    dst.src[1] = &src1;

    ggml_compute_forward_mul_mat(params, &dst);
}

স্থির অন্তঃস্থ int64_t ggml_wrap_around(int64_t coord, int64_t size) {
    প্রত্যাগম (coord  + size) % size; // adding size avoids negative number weirdness
}

// ggml_compute_forward_col2im_1d
//
// Scatter-add columns [K*OC, T_in] -> signal [T_out, OC]
// where T_out = (T_in - 1)*s + K - 2*p.  Gather approach: each output reads ceil(K/s) inputs.
// Parallelized over the time axis so the split stays balanced whatever OC is.
// Supports F32, F16, BF16 input/output (same type), F32 accumulator.

template <typename elem_t>
স্থির শূন্য ggml_compute_forward_col2im_1d_impl(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src = dst->src[0];  // [K*OC, T_in]

    GGML_ASSERT(ggml_is_contiguous(src));
    GGML_ASSERT(ggml_is_contiguous(dst));

    ধ্রুব int32_t s0 = ((ধ্রুব int32_t *)(dst->op_params))[0];
    ধ্রুব int32_t OC = ((ধ্রুব int32_t *)(dst->op_params))[1];
    ধ্রুব int32_t p0 = ((ধ্রুব int32_t *)(dst->op_params))[2];

    ধ্রুব int64_t K_OC = src->ne[0];
    ধ্রুব int64_t T_in = src->ne[1];
    ধ্রুব int64_t K    = K_OC / OC;
    ধ্রুব int64_t T_out = dst->ne[0];

    ধ্রুব elem_t * col_data = (ধ্রুব elem_t *) src->data;
    elem_t       * dst_data = (elem_t *) dst->data;

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    // Parallelize over the time axis: the split stays balanced whatever OC is,
    // down to OC = 1 for mono audio, and threads read disjoint column bands
    ধ্রুব int64_t dr = (T_out + nth - 1) / nth;
    ধ্রুব int64_t it0 = dr * ith;
    ধ্রুব int64_t it1 = it0 + dr < T_out ? it0 + dr : T_out;

    প্রতি (int64_t oc = 0; oc < OC; oc++) {
        প্রতি (int64_t t_out = it0; t_out < it1; t_out++) {
            ধ্রুব int64_t t_abs = t_out + p0;  // absolute position in uncropped signal
            // Gather: find all (t_in, k) where t_in * s + k == t_abs, 0 <= k < K
            int64_t t_in_min = (t_abs - K + 1 + s0 - 1) / s0;  // ceil((t_abs-K+1)/s)
            যদি (t_in_min < 0) t_in_min = 0;
            int64_t t_in_max = t_abs / s0;
            যদি (t_in_max >= T_in) t_in_max = T_in - 1;

            ভিন্ন sum = 0.0f;
            প্রতি (int64_t t_in = t_in_min; t_in <= t_in_max; t_in++) {
                int64_t k = t_abs - t_in * s0;
                যদি (k >= 0 && k < K) {
                    // col layout: [K*OC, T_in], element (oc*K+k, t_in)
                    sum += type_conversion_table<elem_t>::to_f32(col_data[(oc * K + k) + t_in * K_OC]);
                }
            }
            // dst layout: [T_out, OC], element (t_out, oc)
            dst_data[t_out + oc * T_out] = type_conversion_table<elem_t>::from_f32(sum);
        }
    }
}

শূন্য ggml_compute_forward_col2im_1d(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {
    বিকল্প (dst->src[0]->type) {
        পক্ষ GGML_TYPE_F32:  ggml_compute_forward_col2im_1d_impl<ভিন্ন>      (params, dst); ভঙ্গ;
        পক্ষ GGML_TYPE_F16:  ggml_compute_forward_col2im_1d_impl<ggml_fp16_t>(params, dst); ভঙ্গ;
        পক্ষ GGML_TYPE_BF16: ggml_compute_forward_col2im_1d_impl<ggml_bf16_t>(params, dst); ভঙ্গ;
        স্বভাব: GGML_ABORT("col2im_1d: unsupported type %d", dst->src[0]->type);
    }
}

// ggml_compute_forward_conv_2d


স্থির শূন্য ggml_compute_forward_conv_2d_impl(ধ্রুব ggml_compute_params * params,
                                              ধ্রুব ggml_tensor *         kernel,  // [KW, KH, IC, OC]
                                              ধ্রুব ggml_tensor *         src,     // [W, H, C, N]
                                              ggml_tensor *               dst,     // [OW, OH, OC, N]
                                              ggml_type                   kernel_type) {

    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(kernel_type == GGML_TYPE_F16 || kernel_type == GGML_TYPE_F32);
    GGML_ASSERT(kernel->type == kernel_type);

    ধ্রুব ggml_type_traits * traits = ggml_get_type_traits(kernel_type);

    ধ্রুব int32_t stride_x   = dst->op_params[0];
    ধ্রুব int32_t stride_y   = dst->op_params[1];
    ধ্রুব int32_t pad_x      = dst->op_params[2];
    ধ্রুব int32_t pad_y      = dst->op_params[3];
    ধ্রুব int32_t dilation_x = dst->op_params[4];
    ধ্রুব int32_t dilation_y = dst->op_params[5];

    ধ্রুব int64_t c_in  = src->ne[2];
    ধ্রুব int64_t c_out = kernel->ne[3];
    GGML_ASSERT(c_in == kernel->ne[2]);

    ধ্রুব int64_t src_w = src->ne[0];
    ধ্রুব int64_t src_h = src->ne[1];
    ধ্রুব int64_t knl_w = kernel->ne[0];
    ধ্রুব int64_t knl_h = kernel->ne[1];
    ধ্রুব int64_t dst_w = dst->ne[0];
    ধ্রুব int64_t dst_h = dst->ne[1];

    ধ্রুব ভিন্ন * src_data = (ভিন্ন *) src->data;
    শূন্য  * knl_data       = kernel->data;
    ভিন্ন * dst_data       = (ভিন্ন *) dst->data;

    ধ্রুব int64_t knl_n           = knl_w * knl_h * c_in;
    ধ্রুব int64_t patch_total     = dst->ne[3] * dst_w * dst_h;

    ধ্রুব int64_t space_per_patch   = knl_n * traits->type_size + c_out * পরিমাণ(ভিন্ন);
    ধ্রুব int64_t batch_size        = params->wsize / space_per_patch;
    ধ্রুব int64_t patches_per_batch = batch_size > 8 ? (batch_size / 8) * 8 : batch_size;
    ধ্রুব int64_t batch_n           = (patch_total + patches_per_batch - 1) / patches_per_batch;

    GGML_ASSERT(patches_per_batch > 0 && batch_size >= 1);

    শূন্য * tmp = params->wdata;

    প্রতি (int64_t batch_i = 0; batch_i < batch_n; ++batch_i) {

        ধ্রুব int64_t patch_start_batch = batch_i * patches_per_batch;
        ধ্রুব int64_t patch_end_batch   = std::min(patch_start_batch + patches_per_batch,
                                              patch_total);
        ধ্রুব int64_t patch_n           = patch_end_batch - patch_start_batch;

        ধ্রুব int64_t patch_per_thread  = (patch_n + params->nth - 1) / params->nth;
        ধ্রুব int64_t patch_start       = patch_start_batch + params->ith * patch_per_thread;
        ধ্রুব int64_t patch_end         = std::min(patch_start + patch_per_thread, patch_end_batch);

        //im2col for a patch
        প্রতি (int64_t p = patch_start; p < patch_end; ++p) {
            ধ্রুব int64_t  batch_n     =  p / (dst_w * dst_h);
            ধ্রুব int64_t  src_x       = (p / dst_w) % dst_h;
            ধ্রুব int64_t  src_y       =  p % dst_w;

            ধ্রুব ভিন্ন * src_base = (ধ্রুব ভিন্ন *)((ধ্রুব বর্ণ *)src_data + batch_n * src->nb[3]);
            বর্ণ *        dst_row  = (বর্ণ *) tmp + (p % patches_per_batch) * knl_n * traits->type_size;

            প্রতি (int64_t ic = 0; ic < c_in; ++ic) {
                প্রতি (int64_t ky = 0; ky < knl_h; ++ky) {
                    প্রতি (int64_t kx = 0; kx < knl_w; ++kx) {
                        ধ্রুব int64_t sy = src_x * stride_y + ky * dilation_y - pad_y;
                        ধ্রুব int64_t sx = src_y * stride_x + kx * dilation_x - pad_x;

                        int64_t dst_idx = ic * (knl_h * knl_w) + ky * knl_w + kx;

                        ভিন্ন src_val;
                        যদি (sy < 0 || sy >= src_h || sx < 0 || sx >= src_w) {
                            src_val = 0.0f;
                        } অন্যথা {
                            ধ্রুব ভিন্ন * src_ptr = (ধ্রুব ভিন্ন *)((ধ্রুব বর্ণ *)src_base + sx * src->nb[0] + sy * src->nb[1] + ic * src->nb[2]);
                            src_val               = *src_ptr;
                        }

                        বর্ণ * element_ptr = dst_row + dst_idx * traits->type_size;
                        যদি (kernel_type == GGML_TYPE_F32) {
                            *(ভিন্ন *) element_ptr = src_val;
                        } অন্যথা যদি (kernel_type == GGML_TYPE_F16) {
                            *(ggml_fp16_t *) element_ptr = GGML_CPU_FP32_TO_FP16(src_val);
                        }
                    }
                }
            }
        }   // patches handled by this thread

        ggml_barrier(params->threadpool);

        ভিন্ন * gemm_output = (ভিন্ন *) ((বর্ণ *) tmp + patches_per_batch * knl_n * traits->type_size);

        GGML_ASSERT(gemm_output + patch_n * c_out <= (ভিন্ন*)tmp + params->wsize);

        // GEMM: patches[patch_n, knl_n] × kernel[knl_n, c_out] = output[patch_n, c_out]
        ggml_call_mul_mat(kernel_type, params, patch_n, c_out, knl_n, tmp, knl_data, gemm_output);

        ggml_barrier(params->threadpool);


        //permute back [OC, N, OH, OW] to [N, OC, OH, OW]
        ধ্রুব int64_t permute_per_thread = (patch_n + params->nth - 1) / params->nth;
        ধ্রুব int64_t permute_start = params->ith * permute_per_thread;
        ধ্রুব int64_t permute_end = std::min(permute_start + permute_per_thread, patch_n);

        প্রতি (int64_t i = permute_start; i < permute_end; ++i) {
            ধ্রুব int64_t p       = patch_start_batch + i;
            ধ্রুব int64_t batch_n = p / (dst_w * dst_h);
            ধ্রুব int64_t dst_y   = (p / dst_w) % dst_h;
            ধ্রুব int64_t dst_x   = p % dst_w;

            প্রতি (int64_t oc = 0; oc < c_out; ++oc) {
                ধ্রুব ভিন্ন value = gemm_output[i * c_out + oc];
                ভিন্ন * dst_ptr = (ভিন্ন *)((বর্ণ *)dst_data + dst_x * dst->nb[0] + dst_y * dst->nb[1] + oc * dst->nb[2] + batch_n * dst->nb[3]);
                *dst_ptr = value;
            }
        }
    }
}

শূন্য ggml_compute_forward_conv_2d(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];

    ggml_compute_forward_conv_2d_impl(params, src0, src1, dst, src0->type);
}

// ggml_compute_forward_conv_3d

স্থির শূন্য ggml_compute_forward_conv_3d_impl(ধ্রুব ggml_compute_params * params,
                                              ধ্রুব ggml_tensor *         kernel,
                                              ধ্রুব ggml_tensor *         src,
                                              ggml_tensor *               dst,
                                              ggml_type                   kernel_type) {

    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(kernel_type == GGML_TYPE_F16 || kernel_type == GGML_TYPE_F32);
    GGML_ASSERT(kernel->type == kernel_type);

    ধ্রুব ggml_type_traits * traits = ggml_get_type_traits(kernel_type);

    ধ্রুব int32_t s0 = dst->op_params[0];
    ধ্রুব int32_t s1 = dst->op_params[1];
    ধ্রুব int32_t s2 = dst->op_params[2];
    ধ্রুব int32_t p0 = dst->op_params[3];
    ধ্রুব int32_t p1 = dst->op_params[4];
    ধ্রুব int32_t p2 = dst->op_params[5];
    ধ্রুব int32_t d0 = dst->op_params[6];
    ধ্রুব int32_t d1 = dst->op_params[7];
    ধ্রুব int32_t d2 = dst->op_params[8];
    ধ্রুব int32_t c  = dst->op_params[9];
    ধ্রুব int32_t n  = dst->op_params[10];
    ধ্রুব int32_t oc = dst->op_params[11];

    ধ্রুব int64_t src_w = src->ne[0];
    ধ্রুব int64_t src_h = src->ne[1];
    ধ্রুব int64_t src_d = src->ne[2];
    ধ্রুব int64_t knl_w = kernel->ne[0];
    ধ্রুব int64_t knl_h = kernel->ne[1];
    ধ্রুব int64_t knl_d = kernel->ne[2];
    ধ্রুব int64_t dst_w = dst->ne[0];
    ধ্রুব int64_t dst_h = dst->ne[1];
    ধ্রুব int64_t dst_d = dst->ne[2];

    ধ্রুব ভিন্ন * src_data = (ভিন্ন *) src->data;
    শূন্য  * knl_data       = kernel->data;
    ভিন্ন * dst_data       = (ভিন্ন *) dst->data;

    ধ্রুব int64_t knl_n_per_channel = knl_w * knl_h * knl_d;
    ধ্রুব int64_t knl_n_total       = knl_n_per_channel * c;
    ধ্রুব int64_t patch_total       = n * dst_w * dst_h * dst_d;

    ধ্রুব int64_t space_per_patch   = knl_n_total * traits->type_size + oc * পরিমাণ(ভিন্ন);
    ধ্রুব int64_t batch_size        = params->wsize / space_per_patch;
    ধ্রুব int64_t patches_per_batch = batch_size > 8 ? (batch_size / 8) * 8 : batch_size;
    ধ্রুব int64_t batch_n           = (patch_total + patches_per_batch - 1) / patches_per_batch;

    GGML_ASSERT(patches_per_batch > 0 && batch_size >= 1);

    শূন্য * tmp = params->wdata;

    প্রতি (int64_t batch_i = 0; batch_i < batch_n; ++batch_i) {
        ধ্রুব int64_t patch_start_batch = batch_i * patches_per_batch;
        ধ্রুব int64_t patch_end_batch   = std::min(patch_start_batch + patches_per_batch, patch_total);
        ধ্রুব int64_t patch_n_in_batch  = patch_end_batch - patch_start_batch;

        ধ্রুব int64_t patch_per_thread  = (patch_n_in_batch + params->nth - 1) / params->nth;
        ধ্রুব int64_t patch_start       = patch_start_batch + params->ith * patch_per_thread;
        ধ্রুব int64_t patch_end         = std::min(patch_start + patch_per_thread, patch_end_batch);

        প্রতি (int64_t p = patch_start; p < patch_end; ++p) {
            ধ্রুব int64_t p_in_batch = p % (dst_w * dst_h * dst_d);
            ধ্রুব int64_t p_in_depth = p_in_batch % (dst_w * dst_h);
            ধ্রুব int64_t batch_idx  = p / (dst_w * dst_h * dst_d);
            ধ্রুব int64_t dst_z      = p_in_batch / (dst_w * dst_h);
            ধ্রুব int64_t dst_y      = p_in_depth / dst_w;
            ধ্রুব int64_t dst_x      = p_in_depth % dst_w;

            বর্ণ * dst_row = (বর্ণ *) tmp + (p % patches_per_batch) * knl_n_total * traits->type_size;

            প্রতি (int64_t ic = 0; ic < c; ++ic) {
                প্রতি (int64_t kz = 0; kz < knl_d; ++kz) {
                    প্রতি (int64_t ky = 0; ky < knl_h; ++ky) {
                        প্রতি (int64_t kx = 0; kx < knl_w; ++kx) {
                            ধ্রুব int64_t sz = dst_z * s2 + kz * d2 - p2;
                            ধ্রুব int64_t sy = dst_y * s1 + ky * d1 - p1;
                            ধ্রুব int64_t sx = dst_x * s0 + kx * d0 - p0;

                            int64_t dst_idx = ic * knl_n_per_channel + kz * (knl_h * knl_w) + ky * knl_w + kx;

                            ভিন্ন src_val;
                            যদি (sz < 0 || sz >= src_d || sy < 0 || sy >= src_h || sx < 0 || sx >= src_w) {
                                src_val = 0.0f;
                            } অন্যথা {
                                ধ্রুব int64_t cn_idx = batch_idx * c + ic;
                                ধ্রুব ভিন্ন * src_ptr = (ধ্রুব ভিন্ন *)((ধ্রুব বর্ণ *)src_data + sx*src->nb[0] + sy*src->nb[1] + sz*src->nb[2] + cn_idx*src->nb[3]);
                                src_val = *src_ptr;
                            }

                            বর্ণ * element_ptr = dst_row + dst_idx * traits->type_size;
                            যদি (kernel_type == GGML_TYPE_F32) {
                                *(ভিন্ন *)element_ptr = src_val;
                            } অন্যথা যদি (kernel_type == GGML_TYPE_F16) {
                                *(ggml_fp16_t *)element_ptr = GGML_CPU_FP32_TO_FP16(src_val);
                            }
                        }
                    }
                }
            }
        }

        ggml_barrier(params->threadpool);

        ভিন্ন * gemm_output = (ভিন্ন *) ((বর্ণ *) tmp + patches_per_batch * knl_n_total * traits->type_size);
        ggml_call_mul_mat(kernel_type, params, patch_n_in_batch, oc, knl_n_total, tmp, knl_data, gemm_output);

        ggml_barrier(params->threadpool);

        ধ্রুব int64_t permute_per_thread = (patch_n_in_batch + params->nth - 1) / params->nth;
        ধ্রুব int64_t permute_start = params->ith * permute_per_thread;
        ধ্রুব int64_t permute_end = std::min(permute_start + permute_per_thread, patch_n_in_batch);

        প্রতি (int64_t i = permute_start; i < permute_end; ++i) {
            ধ্রুব int64_t p = patch_start_batch + i;
            ধ্রুব int64_t p_in_batch = p % (dst_w * dst_h * dst_d);
            ধ্রুব int64_t p_in_depth = p_in_batch % (dst_w * dst_h);
            ধ্রুব int64_t batch_idx  = p / (dst_w * dst_h * dst_d);
            ধ্রুব int64_t dst_z      = p_in_batch / (dst_w * dst_h);
            ধ্রুব int64_t dst_y      = p_in_depth / dst_w;
            ধ্রুব int64_t dst_x      = p_in_depth % dst_w;

            প্রতি (int64_t ioc = 0; ioc < oc; ++ioc) {
                ধ্রুব ভিন্ন value = gemm_output[i * oc + ioc];
                ধ্রুব int64_t ocn_idx = batch_idx * oc + ioc;
                ভিন্ন * dst_ptr = (ভিন্ন *)((বর্ণ *)dst_data + dst_x*dst->nb[0] + dst_y*dst->nb[1] + dst_z*dst->nb[2] + ocn_idx*dst->nb[3]);
                *dst_ptr = value;
            }
        }
    }
}

শূন্য ggml_compute_forward_conv_3d(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {
    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];
    ggml_compute_forward_conv_3d_impl(params, src0, src1, dst, src0->type);
}

template <typename kernel_t>
স্থির শূন্য ggml_compute_forward_conv_transpose_2d_impl(
    ধ্রুব ggml_compute_params * params,
          ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_দ্বিমিক_OP_LOCALS

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক nk = ne00*ne01*ne02*ne03;

    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == পরিমাণ(ভিন্ন));

    যদি (ith == 0) {
        memset(params->wdata, 0, params->wsize);

        // permute kernel data (src0) from (Kw x Kh x Cout x Cin) to (Cin x Kw x Kh x Cout)
        {
            kernel_t * ধ্রুব wdata = (kernel_t *) params->wdata + 0;

            প্রতি (int64_t i03 = 0; i03 < ne03; i03++) {
                প্রতি (int64_t i02 = 0; i02 < ne02; i02++) {
                    ধ্রুব kernel_t * ধ্রুব src = (kernel_t *)((বর্ণ *) src0->data + i03*nb03 + i02*nb02);
                    kernel_t * dst_data = wdata + i02*ne01*ne00*ne03;
                    প্রতি (int64_t i01 = 0; i01 < ne01; i01++) {
                        প্রতি (int64_t i00 = 0; i00 < ne00; i00++) {
                            dst_data[i01*ne00*ne03 + i00*ne03 + i03] = src[i01 * ne00 + i00];
                        }
                    }
                }
            }
        }

        // permute source data (src1) from (Sw x Sh x Cin) to (Cin x Sw x Sh), for all batches
        {
            kernel_t * ধ্রুব wdata = (kernel_t *) params->wdata + nk;
            প্রতি (পূর্ণাঙ্ক i13 = 0; i13 < ne13; i13++) {
                kernel_t * ধ্রুব wdata_b = wdata + i13*ne10*ne11*ne12;
                প্রতি (পূর্ণাঙ্ক i12 = 0; i12 < ne12; i12++) {
                    প্রতি (পূর্ণাঙ্ক i11 = 0; i11 < ne11; i11++) {
                        ধ্রুব ভিন্ন * ধ্রুব src = (ভিন্ন *)((বর্ণ *) src1->data + i13*nb13 + i12*nb12 + i11*nb11);
                        kernel_t * dst_data = wdata_b + i11*ne10*ne12;
                        প্রতি (পূর্ণাঙ্ক i10 = 0; i10 < ne10; i10++) {
                            যদি constexpr (std::is_same_v<kernel_t, ggml_fp16_t>) {
                                dst_data[i10*ne12 + i12] = GGML_CPU_FP32_TO_FP16(src[i10]);
                            } অন্যথা {
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

    ধ্রুব int32_t stride = ggml_get_op_params_i32(dst, 0);

    // total patches in dst
    ধ্রুব পূর্ণাঙ্ক np = ne2;

    // patches per thread
    ধ্রুব পূর্ণাঙ্ক dp = (np + nth - 1)/nth;

    // patch range for this thread
    ধ্রুব পূর্ণাঙ্ক ip0 = dp*ith;
    ধ্রুব পূর্ণাঙ্ক ip1 = MIN(ip0 + dp, np);

    kernel_t * ধ্রুব wdata = (kernel_t *) params->wdata + 0;
    kernel_t * ধ্রুব wdata_src = wdata + nk;

    প্রতি (পূর্ণাঙ্ক i2 = ip0; i2 < ip1; i2++) { // Cout
        kernel_t * wdata_kernel = wdata + i2*ne01*ne00*ne03;
        প্রতি (পূর্ণাঙ্ক i3 = 0; i3 < ne3; i3++) { // batch
            ভিন্ন * dst_data = (ভিন্ন *)((বর্ণ *) dst->data + i3*nb3 + i2*nb2);
            kernel_t * wdata_src_b = wdata_src + i3*ne10*ne11*ne12;
            প্রতি (পূর্ণাঙ্ক i11 = 0; i11 < ne11; i11++) {
                প্রতি (পূর্ণাঙ্ক i10 = 0; i10 < ne10; i10++) {
                    ধ্রুব পূর্ণাঙ্ক i1n = i11*ne10*ne12 + i10*ne12;
                    প্রতি (পূর্ণাঙ্ক i01 = 0; i01 < ne01; i01++) {
                        প্রতি (পূর্ণাঙ্ক i00 = 0; i00 < ne00; i00++) {
                            ভিন্ন v = 0;
                            যদি constexpr (std::is_same_v<kernel_t, ggml_fp16_t>) {
                                ggml_vec_dot_f16(ne03, &v, 0,
                                        wdata_src_b + i1n, 0,
                                        wdata_kernel + i01*ne00*ne03 + i00*ne03, 0, 1);
                            } অন্যথা {
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

শূন্য ggml_compute_forward_conv_transpose_2d(
        ধ্রুব ggml_compute_params * params,
              ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F16:
            {
                ggml_compute_forward_conv_transpose_2d_impl<ggml_fp16_t>(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_conv_transpose_2d_impl<ভিন্ন>(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_conv_2d_dw

সংরচনা ggml_conv_2d_dw_params {
    int64_t channels;
    int64_t batch;
    int64_t src_w;
    int64_t src_h;
    int64_t dst_w;
    int64_t dst_h;
    int64_t knl_w;
    int64_t knl_h;
    পূর্ণাঙ্ক stride_x;
    পূর্ণাঙ্ক stride_y;
    পূর্ণাঙ্ক pad_x;
    পূর্ণাঙ্ক pad_y;
    পূর্ণাঙ্ক dilation_x;
    পূর্ণাঙ্ক dilation_y;
};

স্থির অন্তঃস্থ ভিন্ন ggml_conv_2d_dw_knl_f32(ধ্রুব বর্ণ * data, int64_t i, ggml_type type) {
    যদি (type == GGML_TYPE_F16) {
        প্রত্যাগম GGML_FP16_TO_FP32(((ধ্রুব ggml_fp16_t *)data)[i]);
    }
    প্রত্যাগম ((ধ্রুব ভিন্ন *)data)[i];
}

স্থির শূন্য ggml_compute_forward_conv_2d_dw_cwhn(
        ধ্রুব ggml_compute_params * params,
        ধ্রুব ggml_tensor * src,
        ধ্রুব ggml_tensor * kernel,
        ggml_tensor * dst,
        ধ্রুব ggml_conv_2d_dw_params & p) {

    ধ্রুব int64_t c = p.channels;
    ধ্রুব বর্ণ * knl_data = (ধ্রুব বর্ণ *)kernel->data;
    ধ্রুব ggml_type knl_type = kernel->type;

    ধ্রুব int64_t rows_total = p.dst_h * p.batch;
    ধ্রুব int64_t rows_per_thread = (rows_total + params->nth - 1) / params->nth;
    ধ্রুব int64_t row_start = params->ith * rows_per_thread;
    ধ্রুব int64_t row_end = MIN(row_start + rows_per_thread, rows_total);

#ifdef GGML_SIMD
    int64_t c_pkg_end = 0;
    int64_t pkg_size = GGML_F32_EPR;
    যদি (knl_type == GGML_TYPE_F32) {
    #যদি defined(__ARM_FEATURE_SVE)
        pkg_size = svcntw();
    #অন্যথা
        pkg_size = GGML_F32_EPR;
    #endif
        c_pkg_end = (c / pkg_size) * pkg_size;
    }
#অন্যথা
    ধ্রুব int64_t c_pkg_end = 0;
#endif

    প্রতি (int64_t row = row_start; row < row_end; ++row) {
        ধ্রুব int64_t dst_y = row % p.dst_h;
        ধ্রুব ভিন্ন * src_data = (ধ্রুব ভিন্ন *)src->data + (row / p.dst_h) * p.src_w * p.src_h * c;
        প্রতি (int64_t dst_x = 0; dst_x < p.dst_w; ++dst_x) {
            ভিন্ন * dst_data = (ভিন্ন *)dst->data + (row * p.dst_w + dst_x) * c;
            ধ্রুব int64_t src_y_base = dst_y * p.stride_y - p.pad_y;
            ধ্রুব int64_t src_x_base = dst_x * p.stride_x - p.pad_x;

#ifdef GGML_SIMD
            প্রতি (int64_t c_i = 0; c_i < c_pkg_end; c_i += pkg_size) {
                GGML_F32_VEC sum = GGML_F32_VEC_ZERO;
                প্রতি (int64_t knl_y = 0; knl_y < p.knl_h; ++knl_y) {
                    ধ্রুব int64_t src_y = src_y_base + knl_y * p.dilation_y;
                    যদি (src_y < 0 || src_y >= p.src_h) {
                        অনুবর্ত;
                    }
                    প্রতি (int64_t knl_x = 0; knl_x < p.knl_w; ++knl_x) {
                        ধ্রুব int64_t src_x = src_x_base + knl_x * p.dilation_x;
                        যদি (src_x < 0 || src_x >= p.src_w) {
                            অনুবর্ত;
                        }
                        ধ্রুব ভিন্ন * kp = (ধ্রুব ভিন্ন *)knl_data + (knl_y * p.knl_w + knl_x) * c + c_i;
                        GGML_F32_VEC k = GGML_F32_VEC_LOAD(kp);
                        GGML_F32_VEC s = GGML_F32_VEC_LOAD(src_data + (src_y * p.src_w + src_x) * c + c_i);
                        sum = GGML_F32_VEC_FMA(sum, k, s);
                    }
                }
                GGML_F32_VEC_STORE(dst_data + c_i, sum);
            }
#endif
            প্রতি (int64_t c_i = c_pkg_end; c_i < c; ++c_i) {
                ভিন্ন sum = 0.0f;
                প্রতি (int64_t knl_y = 0; knl_y < p.knl_h; ++knl_y) {
                    ধ্রুব int64_t src_y = src_y_base + knl_y * p.dilation_y;
                    যদি (src_y < 0 || src_y >= p.src_h) {
                        অনুবর্ত;
                    }
                    প্রতি (int64_t knl_x = 0; knl_x < p.knl_w; ++knl_x) {
                        ধ্রুব int64_t src_x = src_x_base + knl_x * p.dilation_x;
                        যদি (src_x < 0 || src_x >= p.src_w) {
                            অনুবর্ত;
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

স্থির শূন্য ggml_compute_forward_conv_2d_dw_whcn(
        ধ্রুব ggml_compute_params * params,
        ধ্রুব ggml_tensor * src,
        ধ্রুব ggml_tensor * kernel,
        ggml_tensor * dst,
        ধ্রুব ggml_conv_2d_dw_params & p) {

    ধ্রুব int64_t n = p.channels * p.batch;
    ধ্রুব int64_t per_thread = (n + params->nth - 1) / params->nth;
    ধ্রুব int64_t start = params->ith * per_thread;
    ধ্রুব int64_t end = MIN(start + per_thread, n);
    ধ্রুব বর্ণ * knl_base = (ধ্রুব বর্ণ *)kernel->data;
    ধ্রুব ggml_type knl_type = kernel->type;

    প্রতি (int64_t i = start; i < end; ++i) {
        ধ্রুব int64_t knl_offset = (i % p.channels) * p.knl_w * p.knl_h;
        ধ্রুব ভিন্ন * src_data = (ধ্রুব ভিন্ন *)src->data + i * p.src_w * p.src_h;
        ভিন্ন * dst_data = (ভিন্ন *)dst->data + i * p.dst_w * p.dst_h;

        প্রতি (int64_t dst_y = 0; dst_y < p.dst_h; ++dst_y) {
            প্রতি (int64_t dst_x = 0; dst_x < p.dst_w; ++dst_x) {

                ভিন্ন sum = 0.0f;
                প্রতি (int64_t knl_y = 0; knl_y < p.knl_h; ++knl_y) {
                    ধ্রুব int64_t src_y = dst_y * p.stride_y + knl_y * p.dilation_y - p.pad_y;
                    যদি (src_y < 0 || src_y >= p.src_h) {
                        অনুবর্ত;
                    }
                    প্রতি (int64_t knl_x = 0; knl_x < p.knl_w; ++knl_x) {
                        ধ্রুব int64_t src_x = dst_x * p.stride_x + knl_x * p.dilation_x - p.pad_x;
                        যদি (src_x < 0 || src_x >= p.src_w) {
                            অনুবর্ত;
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

শূন্য ggml_compute_forward_conv_2d_dw(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * kernel = dst->src[0];
    ধ্রুব ggml_tensor * src = dst->src[1];
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

    যদি (ggml_is_contiguous(src)) {
        ggml_compute_forward_conv_2d_dw_whcn(params, src, kernel, dst, p);
    } অন্যথা যদি (ggml_is_contiguous_channels(src)) {
        GGML_ASSERT(kernel->nb[0] >= kernel->nb[2] && kernel->nb[1] >= kernel->nb[0]);
        ggml_compute_forward_conv_2d_dw_cwhn(params, src, kernel, dst, p);
    } অন্যথা {
        GGML_ABORT("non-contiguous memory layout not supported");
    }
}

// ggml_compute_forward_pool_1d_ksp
স্থির শূন্য ggml_compute_forward_pool_1d_ksp(
        ধ্রুব ggml_compute_params * params,
        ধ্রুব ggml_op_pool op,
        ধ্রুব পূর্ণাঙ্ক k,
        ধ্রুব পূর্ণাঙ্ক s,
        ধ্রুব পূর্ণাঙ্ক p,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src = dst->src[0];

    প্রতিজ্ঞা(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16);

    যদি (params->ith != 0) {
        প্রত্যাগম;
    }

    ধ্রুব int64_t IW = src->ne[0];
    ধ্রুব int64_t OW = dst->ne[0];

    ধ্রুব int64_t nr = ggml_nrows(src);

    প্রতি (int64_t ir = 0; ir < nr; ++ir) {
        ধ্রুব বর্ণ * srow_bytes =            (ধ্রুব বর্ণ *) src->data + ir * src->nb[1];
        ভিন্ন      * drow       = (ভিন্ন *) ((      বর্ণ *) dst->data + ir * dst->nb[1]);

        প্রতি (int64_t ow = 0; ow < OW; ++ow) {
            ভিন্ন res = 0;
            বিকল্প (op) {
                পক্ষ GGML_OP_POOL_AVG: res = 0.0f;     ভঙ্গ;
                পক্ষ GGML_OP_POOL_MAX: res = -FLT_MAX; ভঙ্গ;
                পক্ষ GGML_OP_POOL_গুনতি: GGML_ABORT("fatal error");
            }

            পূর্ণাঙ্ক count = 0;
            ধ্রুব পূর্ণাঙ্ক base = (পূর্ণাঙ্ক) ow * s - p;

            প্রতি (পূর্ণাঙ্ক ki = 0; ki < k; ++ki) {
                ধ্রুব পূর্ণাঙ্ক j = base + ki;
                যদি (j < 0 || j >= (পূর্ণাঙ্ক) IW) {
                    অনুবর্ত;
                }

                ভিন্ন v;
                যদি (src->type == GGML_TYPE_F32) {
                    v = ((ধ্রুব ভিন্ন *) srow_bytes)[j];
                } অন্যথা {
                    v = GGML_CPU_FP16_TO_FP32(((ধ্রুব ggml_fp16_t *) srow_bytes)[j]);
                }

                বিকল্প (op) {
                    পক্ষ GGML_OP_POOL_AVG: res += v;                ভঙ্গ;
                    পক্ষ GGML_OP_POOL_MAX: res =  std::max(v, res); ভঙ্গ;
                    পক্ষ GGML_OP_POOL_গুনতি: GGML_ABORT("fatal error");
                }

                ++count;
            }

            বিকল্প (op) {
                পক্ষ GGML_OP_POOL_AVG: res = (count > 0) ? (res / count) : 0.0f; ভঙ্গ;
                পক্ষ GGML_OP_POOL_MAX:                                           ভঙ্গ;
                পক্ষ GGML_OP_POOL_গুনতি: GGML_ABORT("fatal error");
            }

            drow[ow] = res;
        }
    }
}

// ggml_compute_forward_pool_1d

শূন্য ggml_compute_forward_pool_1d(
        ধ্রুব ggml_compute_params * params,
              ggml_tensor * dst) {

    ধ্রুব int32_t * opts = (ধ্রুব int32_t *)dst->op_params;
    ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    ধ্রুব পূর্ণাঙ্ক k0 = opts[1];
    ধ্রুব পূর্ণাঙ্ক s0 = opts[2];
    ধ্রুব পূর্ণাঙ্ক p0 = opts[3];

    ggml_compute_forward_pool_1d_ksp(params, op, k0, s0, p0, dst);
}

// ggml_compute_forward_pool_2d

শূন্য ggml_compute_forward_pool_2d(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src = dst->src[0];

    প্রতিজ্ঞা(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16);

    যদি (params->ith != 0) {
        প্রত্যাগম;
    }

    ধ্রুব int32_t * opts = (ধ্রুব int32_t *)dst->op_params;

    ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    ধ্রুব পূর্ণাঙ্ক k0 = opts[1];
    ধ্রুব পূর্ণাঙ্ক k1 = opts[2];
    ধ্রুব পূর্ণাঙ্ক s0 = opts[3];
    ধ্রুব পূর্ণাঙ্ক s1 = opts[4];
    ধ্রুব পূর্ণাঙ্ক p0 = opts[5];
    ধ্রুব পূর্ণাঙ্ক p1 = opts[6];
    ধ্রুব বর্ণ * cdata = (ধ্রুব বর্ণ*)src->data;
    ধ্রুব বর্ণ * ধ্রুব data_end = cdata + ggml_nbytes(src);

    ধ্রুব int64_t px = dst->ne[0];
    ধ্রুব int64_t py = dst->ne[1];
    ধ্রুব int64_t pa = px * py;

    ভিন্ন * dplane = (ভিন্ন *)dst->data;

    ধ্রুব পূর্ণাঙ্ক ka = k0 * k1;
    ধ্রুব পূর্ণাঙ্ক offset0 = -p0;
    ধ্রুব পূর্ণাঙ্ক offset1 = -p1;

    যাবত্ (cdata < data_end) {
        প্রতি (পূর্ণাঙ্ক oy = 0; oy < py; ++oy) {
            ভিন্ন * ধ্রুব drow = dplane + oy * px;
            ভিন্ন * ধ্রুব out  = drow;

            প্রতি (পূর্ণাঙ্ক ox = 0; ox < px; ++ox) {
                ভিন্ন res = 0;
                বিকল্প (op) {
                    পক্ষ GGML_OP_POOL_AVG: res = 0;        ভঙ্গ;
                    পক্ষ GGML_OP_POOL_MAX: res = -FLT_MAX; ভঙ্গ;
                    পক্ষ GGML_OP_POOL_গুনতি: GGML_ABORT("fatal error");
                }

                ধ্রুব পূর্ণাঙ্ক ix = offset0 + ox * s0;
                ধ্রুব পূর্ণাঙ্ক iy = offset1 + oy * s1;

                প্রতি (পূর্ণাঙ্ক ky = 0; ky < k1; ++ky) {
                    যদি (iy + ky < 0 || iy + ky >= src->ne[1]) {
                        অনুবর্ত;
                    }

                    ধ্রুব শূন্য * srow = (ধ্রুব শূন্য *)(cdata + src->nb[1] * (iy + ky));
                    প্রতি (পূর্ণাঙ্ক kx = 0; kx < k0; ++kx) {
                        পূর্ণাঙ্ক j = ix + kx;
                        যদি (j < 0 || j >= src->ne[0]) {
                            অনুবর্ত;
                        }

                        ধ্রুব ভিন্ন srow_j = (src->type == GGML_TYPE_F32) ? ((ধ্রুব ভিন্ন*)srow)[j] : GGML_CPU_FP16_TO_FP32(((ধ্রুব ggml_fp16_t*)srow)[j]);
                        বিকল্প (op) {
                            পক্ষ GGML_OP_POOL_AVG: res += srow_j;                ভঙ্গ;
                            পক্ষ GGML_OP_POOL_MAX: res =  std::max(srow_j, res); ভঙ্গ;
                            পক্ষ GGML_OP_POOL_গুনতি:               GGML_ABORT("fatal error");
                        }
                    }
                }
                বিকল্প (op) {
                    পক্ষ GGML_OP_POOL_AVG:           res /= ka; ভঙ্গ;
                    পক্ষ GGML_OP_POOL_MAX:                      ভঙ্গ;
                    পক্ষ GGML_OP_POOL_গুনতি: GGML_ABORT("fatal error");
                }

                out[ox] = res;
            }
        }

        cdata  += src->nb[2];
        dplane += pa;
    }
}

// ggml_compute_forward_pool_2d_back

শূন্য ggml_compute_forward_pool_2d_back(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src  = dst->src[0];
    ধ্রুব ggml_tensor * dstf = dst->src[1]; // forward tensor of dst

    প্রতিজ্ঞা(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);

    যদি (params->ith != 0) {
        প্রত্যাগম;
    }

    ধ্রুব int32_t * opts = (ধ্রুব int32_t *)dst->op_params;
    ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    ধ্রুব পূর্ণাঙ্ক k0 = opts[1];
    ধ্রুব পূর্ণাঙ্ক k1 = opts[2];
    ধ্রুব পূর্ণাঙ্ক s0 = opts[3];
    ধ্রুব পূর্ণাঙ্ক s1 = opts[4];
    ধ্রুব পূর্ণাঙ্ক p0 = opts[5];
    ধ্রুব পূর্ণাঙ্ক p1 = opts[6];

    বর্ণ       * cdata  = (বর্ণ       *) dst->data;
    ধ্রুব বর্ণ * cdataf = (ধ্রুব বর্ণ *) dstf->data;
    ধ্রুব বর্ণ * ধ্রুব data_end = cdata + ggml_nbytes(dst);

    GGML_ASSERT(params->ith == 0);
    memset(cdata, 0, ggml_nbytes(dst));

    ধ্রুব int64_t px = src->ne[0];
    ধ্রুব int64_t py = src->ne[1];
    ধ্রুব int64_t pa = px * py;

    ধ্রুব ভিন্ন * splane = (ধ্রুব ভিন্ন *) src->data;

    ধ্রুব পূর্ণাঙ্ক ka = k0 * k1;
    ধ্রুব পূর্ণাঙ্ক offset0 = -p0;
    ধ্রুব পূর্ণাঙ্ক offset1 = -p1;

    যাবত্ (cdata < data_end) {
        প্রতি (পূর্ণাঙ্ক oy = 0; oy < py; ++oy) {
            ধ্রুব ভিন্ন * ধ্রুব srow = splane + oy * px;
            প্রতি (পূর্ণাঙ্ক ox = 0; ox < px; ++ox) {
                ধ্রুব ভিন্ন grad0 = srow[ox];

                ধ্রুব পূর্ণাঙ্ক ix = offset0 + ox * s0;
                ধ্রুব পূর্ণাঙ্ক iy = offset1 + oy * s1;

                যদি (op == GGML_OP_POOL_MAX) {
                    ভিন্ন maxval = -FLT_MAX;
                    পূর্ণাঙ্ক kxmax = -1;
                    পূর্ণাঙ্ক kymax = -1;

                    প্রতি (পূর্ণাঙ্ক ky = 0; ky < k1; ++ky) {
                        যদি (iy + ky < 0 || iy + ky >= dst->ne[1]) {
                            অনুবর্ত;
                        }
                        ধ্রুব শূন্য * drowf = (ধ্রুব শূন্য *)(cdataf + dst->nb[1] * (iy + ky));
                        প্রতি (পূর্ণাঙ্ক kx = 0; kx < k0; ++kx) {
                            পূর্ণাঙ্ক j = ix + kx;
                            যদি (j < 0 || j >= dst->ne[0]) {
                                অনুবর্ত;
                            }

                            ধ্রুব ভিন্ন val = dst->type == GGML_TYPE_F32 ?
                                ((ধ্রুব ভিন্ন *) drowf)[j] : GGML_CPU_FP16_TO_FP32(((ধ্রুব ggml_fp16_t *) drowf)[j]);
                            যদি (val <= maxval) {
                                অনুবর্ত;
                            }

                            maxval = val;
                            kxmax = kx;
                            kymax = ky;
                        }
                    }

                    যদি (kxmax == -1 || kymax == -1) {
                        অনুবর্ত;
                    }

                    শূন্য * drow = (শূন্য *)(cdata + dst->nb[1] * (iy + kymax));
                    ধ্রুব পূর্ণাঙ্ক j = ix + kxmax;
                    যদি (dst->type == GGML_TYPE_F32) {
                        ((ভিন্ন *) drow)[j] += grad0;
                    } অন্যথা {
                        ((ggml_fp16_t *) drow)[j] = GGML_CPU_FP32_TO_FP16(grad0 + GGML_CPU_FP16_TO_FP32(((ধ্রুব ggml_fp16_t *) drow)[j]));
                    }
                } অন্যথা যদি (op == GGML_OP_POOL_AVG) {
                    ধ্রুব ভিন্ন grad = grad0 / ka;

                    প্রতি (পূর্ণাঙ্ক ky = 0; ky < k1; ++ky) {
                        যদি (iy + ky < 0 || iy + ky >= dst->ne[1]) {
                            অনুবর্ত;
                        }
                        শূন্য * drow = (শূন্য *)(cdata + dst->nb[1] * (iy + ky));
                        প্রতি (পূর্ণাঙ্ক kx = 0; kx < k0; ++kx) {
                            পূর্ণাঙ্ক j = ix + kx;
                            যদি (j < 0 || j >= dst->ne[0]) {
                                অনুবর্ত;
                            }

                            যদি (dst->type == GGML_TYPE_F32) {
                                ((ভিন্ন *) drow)[j] += grad;
                            } অন্যথা {
                                ((ggml_fp16_t *) drow)[j] += GGML_CPU_FP32_TO_FP16(grad);
                            }
                        }
                    }
                } অন্যথা {
                    GGML_ASSERT(অসত্য);
                }
            }
        }

        cdata  += dst->nb[2];
        cdataf += dst->nb[2];
        splane += pa;
    }
}

// ggml_compute_forward_upscale

স্থির শূন্য ggml_compute_forward_upscale_f32(
    ধ্রুব ggml_compute_params * params,
    ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    ভিন্ন sf0 = (ভিন্ন)ne0/src0->ne[0];
    ভিন্ন sf1 = (ভিন্ন)ne1/src0->ne[1];
    ভিন্ন sf2 = (ভিন্ন)ne2/src0->ne[2];
    ভিন্ন sf3 = (ভিন্ন)ne3/src0->ne[3];
    ভিন্ন pixel_offset = 0.5f;

    ধ্রুব int32_t mode_flags = ggml_get_op_params_i32(dst, 0);
    ধ্রুব ggml_scale_mode mode = (ggml_scale_mode) (mode_flags & 0xFF);

    যদি (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        pixel_offset = 0.0f;
        sf0 = ne0 > 1 && ne00 > 1 ? (ভিন্ন)(ne0 - 1) / (ne00 - 1) : sf0;
        sf1 = ne1 > 1 && ne01 > 1 ? (ভিন্ন)(ne1 - 1) / (ne01 - 1) : sf1;
    }

    যদি (mode == GGML_SCALE_MODE_NEAREST) {
        প্রতি (int64_t i3 = 0; i3 < ne3; i3++) {
            ধ্রুব int64_t i03 = i3 / sf3;
            প্রতি (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                ধ্রুব int64_t i02 = i2 / sf2;
                প্রতি (int64_t i1 = 0; i1 < ne1; i1++) {
                    ধ্রুব int64_t i01 = i1 / sf1;
                    প্রতি (int64_t i0 = 0; i0 < ne0; i0++) {
                        ধ্রুব int64_t i00 = i0 / sf0;

                        ধ্রুব ভিন্ন * x = (ভিন্ন *)((বর্ণ *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              ভিন্ন * y = (ভিন্ন *)((বর্ণ *)  dst->data +  i0*nb0  +  i1*nb1  +  i2*nb2  +  i3*nb3);

                        *y = *x;
                    }
                }
            }
        }
    } অন্যথা যদি (mode == GGML_SCALE_MODE_BILINEAR && (mode_flags & GGML_SCALE_FLAG_ANTIALIAS)) {
        // Similar to F.interpolate(..., mode="bilinear", align_corners=False, antialias=True)
        // https://github.com/pytorch/pytorch/blob/8871ff29b743948d1225389d5b7068f37b22750b/aten/src/ATen/native/cpu/UpSampleKernel.cpp
        auto triangle_filter = [](ভিন্ন x) -> ভিন্ন {
            প্রত্যাগম std::max(1.0f - fabsf(x), 0.0f);
        };

        // support and invscale, minimum 1 pixel for bilinear
        ধ্রুব ভিন্ন support1  = std::max(1.0f, 1.0f / sf1);
        ধ্রুব ভিন্ন invscale1 = 1.0f / support1;
        ধ্রুব ভিন্ন support0  = std::max(1.0f, 1.0f / sf0);
        ধ্রুব ভিন্ন invscale0 = 1.0f / support0;

        প্রতি (int64_t i3 = 0; i3 < ne3; i3++) {
            ধ্রুব int64_t i03 = i3 / sf3;
            প্রতি (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                ধ্রুব int64_t i02 = i2 / sf2;
                প্রতি (int64_t i1 = 0; i1 < ne1; i1++) {
                    ধ্রুব ভিন্ন y = ((ভিন্ন) i1 + pixel_offset) / sf1;
                    প্রতি (int64_t i0 = 0; i0 < ne0; i0++) {
                        ধ্রুব ভিন্ন x = ((ভিন্ন) i0 + pixel_offset) / sf0;

                        // the range of source pixels that contribute
                        ধ্রুব int64_t x_min = std::max<int64_t>(x - support0 + pixel_offset, 0);
                        ধ্রুব int64_t x_max = std::min<int64_t>(x + support0 + pixel_offset, ne00);
                        ধ্রুব int64_t y_min = std::max<int64_t>(y - support1 + pixel_offset, 0);
                        ধ্রুব int64_t y_max = std::min<int64_t>(y + support1 + pixel_offset, ne01);

                        // bilinear filter with antialiasing
                        ভিন্ন val = 0.0f;
                        ভিন্ন total_weight = 0.0f;

                        প্রতি (int64_t sy = y_min; sy < y_max; sy++) {
                            ধ্রুব ভিন্ন weight_y = triangle_filter((sy - y + pixel_offset) * invscale1);

                            প্রতি (int64_t sx = x_min; sx < x_max; sx++) {
                                ধ্রুব ভিন্ন weight_x = triangle_filter((sx - x + pixel_offset) * invscale0);
                                ধ্রুব ভিন্ন weight = weight_x * weight_y;

                                যদি (weight <= 0.0f) {
                                    অনুবর্ত;
                                }

                                ধ্রুব ভিন্ন pixel = *(ধ্রুব ভিন্ন *)((ধ্রুব বর্ণ *)src0->data + sx*nb00 + sy*nb01 + i02*nb02 + i03*nb03);
                                val += pixel * weight;
                                total_weight += weight;
                            }
                        }

                        যদি (total_weight > 0.0f) {
                            val /= total_weight;
                        }

                        ভিন্ন * dst_ptr = (ভিন্ন *)((বর্ণ *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);
                        *dst_ptr = val;
                    }
                }
            }
        }
    } অন্যথা যদি (mode == GGML_SCALE_MODE_BILINEAR) {
        প্রতি (int64_t i3 = 0; i3 < ne3; i3++) {
            ধ্রুব int64_t i03 = i3 / sf3;
            প্রতি (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                ধ্রুব int64_t i02 = i2 / sf2;
                প্রতি (int64_t i1 = 0; i1 < ne1; i1++) {
                    ধ্রুব ভিন্ন y = ((ভিন্ন)i1 + pixel_offset) / sf1 - pixel_offset;
                    int64_t y0 = (int64_t)floorf(y);
                    int64_t y1 = y0 + 1;

                    y0 = std::max(int64_t(0), std::min(y0, ne01 - 1));
                    y1 = std::max(int64_t(0), std::min(y1, ne01 - 1));

                    ভিন্ন dy = y - (ভিন্ন)y0;
                    dy = std::max(0.0f, std::min(dy, 1.0f));

                    প্রতি (int64_t i0 = 0; i0 < ne0; i0++) {
                        ধ্রুব ভিন্ন x = ((ভিন্ন)i0 + pixel_offset) / sf0 - pixel_offset;
                        int64_t x0 = (int64_t)floorf(x);
                        int64_t x1 = x0 + 1;

                        x0 = std::max(int64_t(0), std::min(x0, ne00 - 1));
                        x1 = std::max(int64_t(0), std::min(x1, ne00 - 1));

                        ভিন্ন dx = x - (ভিন্ন)x0;
                        dx = std::max(0.0f, std::min(dx, 1.0f));

                        // fetch the four surrounding pixel values and interpolate
                        ধ্রুব ভিন্ন a = *(ধ্রুব ভিন্ন *)((ধ্রুব বর্ণ *)src0->data + x0*nb00 + y0*nb01 + i02*nb02 + i03*nb03);
                        ধ্রুব ভিন্ন b = *(ধ্রুব ভিন্ন *)((ধ্রুব বর্ণ *)src0->data + x1*nb00 + y0*nb01 + i02*nb02 + i03*nb03);
                        ধ্রুব ভিন্ন c = *(ধ্রুব ভিন্ন *)((ধ্রুব বর্ণ *)src0->data + x0*nb00 + y1*nb01 + i02*nb02 + i03*nb03);
                        ধ্রুব ভিন্ন d = *(ধ্রুব ভিন্ন *)((ধ্রুব বর্ণ *)src0->data + x1*nb00 + y1*nb01 + i02*nb02 + i03*nb03);

                        ধ্রুব ভিন্ন val = a*(1 - dx)*(1 - dy) + b*dx*(1 - dy) + c*(1 - dx)*dy + d*dx*dy;

                        ভিন্ন * y_dst = (ভিন্ন *)((বর্ণ *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);
                        *y_dst = val;
                    }
                }
            }
        }
    } অন্যথা যদি (mode == GGML_SCALE_MODE_BICUBIC) {
        // https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
        ধ্রুব ভিন্ন a = -0.75f; // use alpha = -0.75 (same as PyTorch)
        auto weight1 = [a](ভিন্ন x) { প্রত্যাগম ((a + 2) * x - (a + 3)) * x * x + 1; };
        auto weight2 = [a](ভিন্ন x) { প্রত্যাগম ((a * x - 5 * a) * x + 8 * a) * x - 4 * a; };
        auto bicubic = [=](ভিন্ন p0, ভিন্ন p1, ভিন্ন p2, ভিন্ন p3, ভিন্ন x) {
            ধ্রুব ভিন্ন w0 = weight2(x + 1);
            ধ্রুব ভিন্ন w1 = weight1(x + 0);
            ধ্রুব ভিন্ন w2 = weight1(1 - x);
            ধ্রুব ভিন্ন w3 = weight2(2 - x);
            প্রত্যাগম p0*w0 + p1*w1 + p2*w2 + p3*w3;
        };

        প্রতি (int64_t i3 = 0; i3 < ne3; i3++) {
            ধ্রুব int64_t i03 = i3 / sf3;
            প্রতি (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                ধ্রুব int64_t i02 = i2 / sf2;
                প্রতি (int64_t i1 = 0; i1 < ne1; i1++) {
                    ধ্রুব ভিন্ন y = ((ভিন্ন)i1 + pixel_offset) / sf1 - pixel_offset;
                    ধ্রুব int64_t y0 = (int64_t)floorf(y);
                    ধ্রুব ভিন্ন dy = y - (ভিন্ন)y0;

                    প্রতি (int64_t i0 = 0; i0 < ne0; i0++) {
                        ধ্রুব ভিন্ন x = ((ভিন্ন)i0 + pixel_offset) / sf0 - pixel_offset;
                        ধ্রুব int64_t x0 = (int64_t)floorf(x);
                        ধ্রুব ভিন্ন dx = x - (ভিন্ন)x0;

                        auto p = [=](int64_t x_off, int64_t y_off) -> ভিন্ন {
                            int64_t i00 = std::max(int64_t(0), std::min(x0 + x_off, ne00 - 1));
                            int64_t i01 = std::max(int64_t(0), std::min(y0 + y_off, ne01 - 1));
                            প্রত্যাগম *(ধ্রুব ভিন্ন *)((ধ্রুব বর্ণ *)src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                        };

                        ধ্রুব ভিন্ন val = bicubic(
                            bicubic(p(-1,-1), p(0,-1), p(1,-1), p(2,-1), dx),
                            bicubic(p(-1, 0), p(0, 0), p(1, 0), p(2, 0), dx),
                            bicubic(p(-1, 1), p(0, 1), p(1, 1), p(2, 1), dx),
                            bicubic(p(-1, 2), p(0, 2), p(1, 2), p(2, 2), dx), dy);

                        ভিন্ন * y_dst = (ভিন্ন *)((বর্ণ *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);
                        *y_dst = val;
                    }
                }
            }
        }
    } অন্যথা {
        GGML_ABORT("unsupported upscale mode");
    }
}

শূন্য ggml_compute_forward_upscale(
    ধ্রুব ggml_compute_params * params,
    ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_upscale_f32(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}


// ggml_compute_forward_pad

template<তার্কিক circular_t>
স্থির শূন্য ggml_compute_forward_pad_f32(
    ধ্রুব ggml_compute_params * params,
          ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    প্রতিজ্ঞা(dst->nb[0] == পরিমাণ(ভিন্ন));

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    ভিন্ন * dst_ptr = (ভিন্ন *) dst->data;
    ধ্রুব int32_t lp0 = ggml_get_op_params_i32(dst, 0);
    ধ্রুব int32_t rp0 = ggml_get_op_params_i32(dst, 1);
    ধ্রুব int32_t lp1 = ggml_get_op_params_i32(dst, 2);
    ধ্রুব int32_t rp1 = ggml_get_op_params_i32(dst, 3);
    ধ্রুব int32_t lp2 = ggml_get_op_params_i32(dst, 4);
    ধ্রুব int32_t rp2 = ggml_get_op_params_i32(dst, 5);
    ধ্রুব int32_t lp3 = ggml_get_op_params_i32(dst, 6);
    ধ্রুব int32_t rp3 = ggml_get_op_params_i32(dst, 7);

    ধ্রুব int64_t nrows = ne1 * ne2 * ne3;
    ধ্রুব তার্কিক src_row_contig = src0->nb[0] == পরিমাণ(ভিন্ন);

    প্রতি (int64_t row = ith; row < nrows; row += nth) {
        ধ্রুব int64_t i1 = row % ne1;
        ধ্রুব int64_t i2 = (row / ne1) % ne2;
        ধ্রুব int64_t i3 = row / (ne1 * ne2);

        ভিন্ন * dst_row = dst_ptr + row * ne0;

        যদি constexpr (circular_t) {
            // circular padding wraps around on a torus
            ধ্রুব int64_t src_i1 = ggml_wrap_around(i1 - lp1, ne01);
            ধ্রুব int64_t src_i2 = ggml_wrap_around(i2 - lp2, ne02);
            ধ্রুব int64_t src_i3 = ggml_wrap_around(i3 - lp3, ne03);

            প্রতি (int64_t i0 = 0; i0 < ne0; ++i0) {
                ধ্রুব int64_t src_i0 = ggml_wrap_around(i0 - lp0, ne00);
                dst_row[i0] = *(ধ্রুব ভিন্ন *)((ধ্রুব বর্ণ *) src0->data + src_i3*nb03 + src_i2*nb02 + src_i1*nb01 + src_i0*nb00);
            }
        } অন্যথা {
            ধ্রুব তার্কিক interior = (i1 >= lp1 && i1 < ne1 - rp1)
                               && (i2 >= lp2 && i2 < ne2 - rp2)
                               && (i3 >= lp3 && i3 < ne3 - rp3);
            যদি (!interior) {
                memset(dst_row, 0, ne0 * পরিমাণ(ভিন্ন));
                অনুবর্ত;
            }

            ধ্রুব বর্ণ * src_row = (ধ্রুব বর্ণ *) src0->data + (i3 - lp3)*nb03 + (i2 - lp2)*nb02 + (i1 - lp1)*nb01;

            যদি (src_row_contig && lp0 >= 0 && rp0 >= 0) {
                memset(dst_row, 0, lp0 * পরিমাণ(ভিন্ন));
                memcpy(dst_row + lp0, src_row, ne00 * পরিমাণ(ভিন্ন));
                memset(dst_row + lp0 + ne00, 0, rp0 * পরিমাণ(ভিন্ন));
            } অন্যথা {
                প্রতি (int64_t i0 = 0; i0 < ne0; ++i0) {
                    যদি (i0 >= lp0 && i0 < ne0 - rp0) {
                        dst_row[i0] = *(ধ্রুব ভিন্ন *)(src_row + (i0 - lp0)*nb00);
                    } অন্যথা {
                        dst_row[i0] = 0.0f;
                    }
                }
            }
        }
    }
}


শূন্য ggml_compute_forward_pad(
    ধ্রুব ggml_compute_params * params,
    ggml_tensor * dst) {
    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব তার্কিক circular = (তার্কিক) ggml_get_op_params_i32(dst, 8);
    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                যদি (circular) {
                    ggml_compute_forward_pad_f32<সত্য>(params, dst);
                } অন্যথা {
                    ggml_compute_forward_pad_f32<অসত্য>(params, dst);
                }
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_pad_reflect_1d

শূন্য ggml_compute_forward_pad_reflect_1d(
        ধ্রুব ggml_compute_params * params,
              ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব int32_t * opts = (ধ্রুব int32_t *) dst->op_params;
    ধ্রুব পূর্ণাঙ্ক p0 = opts[0];
    ধ্রুব পূর্ণাঙ্ক p1 = opts[1];

    GGML_TENSOR_UNARY_OP_LOCALS

    প্রতি (int64_t i3 = 0; i3 < ne3; i3++) {
        প্রতি (int64_t i2 = 0; i2 < ne2; i2++) {
            প্রতি (int64_t i1 = ith; i1 < ne1; i1 += nth) {
                ভিন্ন * left  = (ভিন্ন *) ((বর্ণ *) dst->data + i3*nb3 + i2*nb2 + i1*nb1 +         p0*nb0);
                ভিন্ন * right = (ভিন্ন *) ((বর্ণ *) dst->data + i3*nb3 + i2*nb2 + i1*nb1 + (ne0-p1-1)*nb0);

                ggml_vec_cpy_f32(ne00, left, (ভিন্ন *) ((বর্ণ *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

                প্রতি (পূর্ণাঙ্ক i0 = 1; i0 <= p0; i0++) { left[-i0] = left[i0];   }
                প্রতি (পূর্ণাঙ্ক i0 = 1; i0 <= p1; i0++) { right[i0] = right[-i0]; }
            }
        }
    }
}

// ggml_compute_forward_roll

স্থির int64_t ggml_wrap_index(int64_t i, int64_t ne) {
    যদি (i < 0) {
        প্রত্যাগম i + ne;
    } অন্যথা যদি (i >= ne) {
        প্রত্যাগম i - ne;
    }
    প্রত্যাগম i;
}

স্থির শূন্য ggml_compute_forward_roll_f32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ভিন্ন * src_data = (ধ্রুব ভিন্ন *) src0->data;
    ভিন্ন * dst_data = (ভিন্ন *) dst->data;

    GGML_TENSOR_UNARY_OP_LOCALS

    ধ্রুব পূর্ণাঙ্ক s0 = ggml_get_op_params_i32(dst, 0);
    ধ্রুব পূর্ণাঙ্ক s1 = ggml_get_op_params_i32(dst, 1);
    ধ্রুব পূর্ণাঙ্ক s2 = ggml_get_op_params_i32(dst, 2);
    ধ্রুব পূর্ণাঙ্ক s3 = ggml_get_op_params_i32(dst, 3);

    ধ্রুব int64_t total = ne1 * ne2 * ne3;
    ধ্রুব int64_t per_thread = (total + params->nth) / params->nth;
    ধ্রুব int64_t start = params->ith * per_thread;
    ধ্রুব int64_t end   = std::min(start + per_thread, total);

    প্রতি (int64_t i = start; i < end; ++i) {
        ধ্রুব int64_t i1 = i % ne1;
        ধ্রুব int64_t i2 = (i / ne1) % ne2;
        ধ্রুব int64_t i3 = i / (ne2 * ne1);
        ভিন্ন * dst_row = dst_data + (i3*nb3 + i2*nb2 + i1*nb1) / পরিমাণ(ভিন্ন);

        ধ্রুব int64_t i01 = ggml_wrap_index(i1 - s1, ne01);
        ধ্রুব int64_t i02 = ggml_wrap_index(i2 - s2, ne02);
        ধ্রুব int64_t i03 = ggml_wrap_index(i3 - s3, ne03);
        ধ্রুব ভিন্ন * src_row = src_data + (i03*nb03 + i02*nb02 + i01*nb01) / পরিমাণ(ভিন্ন);

        ধ্রুব int64_t s = ggml_wrap_index(-s0, ne00);
        ধ্রুব int64_t n = ne00 - s;
        ggml_vec_cpy_f32(n, dst_row,     src_row + s);
        ggml_vec_cpy_f32(s, dst_row + n, src_row);
    }
}

শূন্য ggml_compute_forward_roll(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_roll_f32(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_arange

স্থির শূন্য ggml_compute_forward_arange_f32(
    ধ্রুব ggml_compute_params * params,
    ggml_tensor * dst) {

    GGML_ASSERT(dst->nb[0] == পরিমাণ(ভিন্ন));

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব ভিন্ন start = ggml_get_op_params_f32(dst, 0);
    ধ্রুব ভিন্ন stop  = ggml_get_op_params_f32(dst, 1);
    ধ্রুব ভিন্ন step  = ggml_get_op_params_f32(dst, 2);

    ধ্রুব int64_t steps = (int64_t) ceilf((stop - start) / step);

    GGML_ASSERT(ggml_nelements(dst) == steps);

    প্রতি (int64_t i = ith; i < steps; i+= nth) {
        ভিন্ন value = start + step * i;
        ((ভিন্ন *)dst->data)[i] = value;
    }
}

শূন্য ggml_compute_forward_arange(
    ধ্রুব ggml_compute_params * params,
    ggml_tensor * dst) {
    বিকল্প (dst->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_arange_f32(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

স্থির শূন্য ggml_compute_forward_timestep_embedding_f32(
    ধ্রুব ggml_compute_params * params,
    ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->nb[0] == পরিমাণ(ভিন্ন));

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    ধ্রুব পূর্ণাঙ্ক dim = ggml_get_op_params_i32(dst, 0);
    ধ্রুব পূর্ণাঙ্ক max_period = ggml_get_op_params_i32(dst, 1);

    পূর্ণাঙ্ক half = dim / 2;

    প্রতি (int64_t i = 0; i < ne00; i++) {
        ভিন্ন * embed_data = (ভিন্ন *)((বর্ণ *)  dst->data +  i*nb1);
        প্রতি (int64_t j = ith; j < half; j += nth) {
            ভিন্ন timestep = ((ভিন্ন *)src0->data)[i];
            ভিন্ন freq = (ভিন্ন)expf(-logf(max_period) * j / half);
            ভিন্ন arg = timestep * freq;
            embed_data[j] = cosf(arg);
            embed_data[j + half] = sinf(arg);
        }
        যদি (dim % 2 != 0 && ith == 0) {
            embed_data[2 * half] = 0.f;
        }
    }
}

শূন্য ggml_compute_forward_timestep_embedding(
    ধ্রুব ggml_compute_params * params,
    ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_timestep_embedding_f32(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_argsort

template<গণনা ggml_sort_order order>
সংরচনা cmp_argsort {
    ধ্রুব ভিন্ন * data;
    তার্কিক operator()(int32_t a, int32_t b) ধ্রুব {
        যদি constexpr (order == GGML_SORT_ORDER_ASC) {
            প্রত্যাগম data[a] < data[b];
        } অন্যথা {
            প্রত্যাগম data[a] > data[b];
        }
    }
};

স্থির শূন্য ggml_compute_forward_argsort_f32(
    ধ্রুব ggml_compute_params * params,
    ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(nb0 == পরিমাণ(ভিন্ন));

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব int64_t nr = ggml_nrows(src0);

    ggml_sort_order order = (ggml_sort_order) ggml_get_op_params_i32(dst, 0);

    প্রতি (int64_t i = ith; i < nr; i += nth) {
        ধ্রুব ভিন্ন * src_data = (ভিন্ন *)((বর্ণ *) src0->data + i*nb01);

        int32_t * dst_data = (int32_t *)((বর্ণ *) dst->data + i*nb1);

        প্রতি (int64_t j = 0; j < ne0; j++) {
            dst_data[j] = j;
        }

        বিকল্প (order) {
            পক্ষ GGML_SORT_ORDER_ASC:
                std::sort(dst_data, dst_data + ne0, cmp_argsort<GGML_SORT_ORDER_ASC>{src_data});
                ভঙ্গ;

            পক্ষ GGML_SORT_ORDER_DESC:
                std::sort(dst_data, dst_data + ne0, cmp_argsort<GGML_SORT_ORDER_DESC>{src_data});
                ভঙ্গ;

            স্বভাব:
                GGML_ABORT("invalid sort order");
        }
    }
}

শূন্য ggml_compute_forward_argsort(
    ধ্রুব ggml_compute_params * params,
    ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_argsort_f32(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_top_k

সংরচনা cmp_শীর্ষ_k {
    ধ্রুব ভিন্ন * data;
    তার্কিক operator()(int32_t a, int32_t b) ধ্রুব {
        প্রত্যাগম data[a] > data[b];
    }
};

স্থির শূন্য ggml_compute_forward_শীর্ষ_k_f32(
    ধ্রুব ggml_compute_params * params,
    ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(nb0 == পরিমাণ(ভিন্ন));

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব int64_t nr = ggml_nrows(src0);

    ধ্রুব পূর্ণাঙ্ক শীর্ষ_k = ne0;

    int32_t * tmp = (int32_t *) params->wdata + (ne00 + CACHE_সারি_SIZE_F32) * ith;

    প্রতি (int64_t i = ith; i < nr; i += nth) {
        ধ্রুব ভিন্ন * src_data = (ভিন্ন *)((বর্ণ *) src0->data + i*nb01);

        প্রতি (int64_t j = 0; j < ne00; j++) {
            tmp[j] = j;
        }

        std::partial_sort(tmp, tmp + শীর্ষ_k, tmp + ne00, cmp_শীর্ষ_k{src_data});

        int32_t * dst_data = (int32_t *)((বর্ণ *) dst->data + i*nb1);

        std::copy(tmp, tmp + শীর্ষ_k, dst_data);

        // emphasize that the order is not important
        যদি (শীর্ষ_k > 1) {
            std::swap(dst_data[0], dst_data[1]);
        }
    }
}

শূন্য ggml_compute_forward_শীর্ষ_k(
    ধ্রুব ggml_compute_params * params,
    ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_শীর্ষ_k_f32(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

স্থির শূন্য ggml_compute_forward_flash_attn_ext_f16_one_chunk(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst,
        পূর্ণাঙ্ক ir0, পূর্ণাঙ্ক ir1,
        int64_t ic_start, int64_t ic_end,
        ভিন্ন * partials, int64_t partial_stride) {

    ধ্রুব তার্কিক write_partials = (partials != nullptr);
    ধ্রুব ggml_tensor * q     = dst->src[0];
    ধ্রুব ggml_tensor * k     = dst->src[1];
    ধ্রুব ggml_tensor * v     = dst->src[2];
    ধ্রুব ggml_tensor * mask  = dst->src[3];
    ধ্রুব ggml_tensor * sinks = dst->src[4];

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    ধ্রুব int64_t DK = nek0;
    ধ্রুব int64_t DV = nev0;
    ধ্রুব int64_t N  = neq1;

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
    GGML_ASSERT(nb0 == পরিমাণ(ভিন্ন));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    ধ্রুব int64_t rk2 = neq2/nek2;
    ধ্রুব int64_t rk3 = neq3/nek3;

    ধ্রুব int64_t rv2 = neq2/nev2;
    ধ্রুব int64_t rv3 = neq3/nev3;

    // parallelize by q rows using ggml_vec_dot_f32

    ভিন্ন scale         = 1.0f;
    ভিন্ন max_bias      = 0.0f;
    ভিন্ন logit_softcap = 0.0f;

    memcpy(&scale,         (ভিন্ন *) dst->op_params + 0, পরিমাণ(ভিন্ন));
    memcpy(&max_bias,      (ভিন্ন *) dst->op_params + 1, পরিমাণ(ভিন্ন));
    memcpy(&logit_softcap, (ভিন্ন *) dst->op_params + 2, পরিমাণ(ভিন্ন));

    যদি (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    ধ্রুব uint32_t n_head      = neq2;
    ধ্রুব uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));

    ধ্রুব ভিন্ন m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    ধ্রুব ভিন্ন m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    ggml_type         ধ্রুব k_vec_dot_type = ggml_get_type_traits_cpu(k->type)->vec_dot_type;
    ggml_from_float_t ধ্রুব q_to_vec_dot   = ggml_get_type_traits_cpu(k_vec_dot_type)->from_float;
    ggml_vec_dot_t    ধ্রুব kq_vec_dot     = ggml_get_type_traits_cpu(k->type)->vec_dot;
    ggml_to_float_t   ধ্রুব v_to_float     = ggml_get_type_traits(v->type)->to_float;

    GGML_ASSERT((                            q_to_vec_dot) && "fattn: unsupported K-type");
    GGML_ASSERT((v->type == GGML_TYPE_F32 || v_to_float  ) && "fattn: unsupported V-type");

    পূর্ণাঙ্ক ith = params->ith;

    প্রতি (পূর্ণাঙ্ক ir = ir0; ir < ir1; ++ir) {
        // q indices
        ধ্রুব পূর্ণাঙ্ক iq3 = ir/(neq2*neq1);
        ধ্রুব পূর্ণাঙ্ক iq2 = (ir - iq3*neq2*neq1)/neq1;
        ধ্রুব পূর্ণাঙ্ক iq1 = (ir - iq3*neq2*neq1 - iq2*neq1);

        ধ্রুব uint32_t h = iq2; // head index
        ধ্রুব ভিন্ন slope = (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1) : 1.0f;

        ভিন্ন S = 0.0f;      // sum
        ভিন্ন M = -INFINITY; // maximum KQ value

        ভিন্ন       * VKQ32 = (ভিন্ন       *) params->wdata + ith*(1*DK + 2*DV + CACHE_সারি_SIZE_F32); // FP32 VKQ accumulator
        ভিন্ন       * V32   =                 (VKQ32 + 1*DV); // (temporary) FP32 V buffer
        ggml_fp16_t * VKQ16 = (ggml_fp16_t *) (VKQ32 + 1*DV); // (temporary) FP16 VKQ accumulator
        ggml_fp16_t * Q_q   = (ggml_fp16_t *) (VKQ32 + 2*DV); // (temporary) buffer for Q converted to quantized/FP16

        যদি (v->type == GGML_TYPE_F16) {
            memset(VKQ16, 0, DV*পরিমাণ(ggml_fp16_t));
        } অন্যথা {
            memset(VKQ32, 0, DV*পরিমাণ(ভিন্ন));
        }

        ধ্রুব ggml_fp16_t * mp = mask ? (ggml_fp16_t *)((বর্ণ *) mask->data + iq1*mask->nb[1] + (iq2%mask->ne[2])*mask->nb[2] + (iq3%mask->ne[3])*mask->nb[3]) : রিক্ত;

        // k indices
        ধ্রুব পূর্ণাঙ্ক ik3 = iq3 / rk3;
        ধ্রুব পূর্ণাঙ্ক ik2 = iq2 / rk2;

        // v indices
        ধ্রুব পূর্ণাঙ্ক iv3 = iq3 / rv3;
        ধ্রুব পূর্ণাঙ্ক iv2 = iq2 / rv2;

        ধ্রুব ভিন্ন * pq = (ধ্রুব ভিন্ন *) ((বর্ণ *) q->data + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3));
        q_to_vec_dot(pq, Q_q, DK);

        // online softmax / attention
        // loop over n_kv and n_head_kv
        // ref: https://arxiv.org/pdf/2112.05682.pdf

        প্রতি (int64_t ic = ic_start; ic < ic_end; ++ic) {
            ধ্রুব ভিন্ন mv = mp ? slope*GGML_CPU_FP16_TO_FP32(mp[ic]) : 0.0f;
            যদি (mv == -INFINITY) {
                অনুবর্ত;
            }

            ভিন্ন s; // KQ value

            ধ্রুব বর্ণ * k_data = (ধ্রুব বর্ণ *) k->data + ( ic*nbk1 + ik2*nbk2 + ik3*nbk3);
            kq_vec_dot(DK, &s, 0, k_data, 0, Q_q, 0, 1);

            s = s*scale; // scale KQ value

            যদি (logit_softcap != 0.0f) {
                s = logit_softcap*tanhf(s);
            }

            s += mv; // apply mask

            ধ্রুব ভিন্ন Mold = M;

            ভিন্ন ms = 1.0f; // upon new higher max val, scale VKQ and KQ sum with this value
            ভিন্ন vs = 1.0f; // post-softmax KQ value, expf(s - M)

            ধ্রুব বর্ণ * v_data = ((ধ্রুব বর্ণ *) v->data + (ic*nbv1 + iv2*nbv2 + iv3*nbv3));

            যদি (v->type == GGML_TYPE_F16) {
                যদি (s > M) {
                    // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
                    M = s;
                    ms = expf(Mold - M);

                    // V = V*expf(Mold - M)
                    ggml_vec_scale_f16(DV, VKQ16, ms);
                } অন্যথা {
                    // no new maximum, ms == 1.0f, vs != 1.0f
                    vs = expf(s - M);
                }

                // V += v*expf(s - M)
                ggml_vec_mad_f16(DV, VKQ16, (ধ্রুব ggml_fp16_t *) v_data, vs);
            } অন্যথা {
                যদি (s > M) {
                    // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
                    M = s;
                    ms = expf(Mold - M);

                    // V = V*expf(Mold - M)
                    ggml_vec_scale_f32(DV, VKQ32, ms);
                } অন্যথা {
                    // no new maximum, ms == 1.0f, vs != 1.0f
                    vs = expf(s - M);
                }

                // V += v*expf(s - M)
                যদি (v_to_float) {
                    v_to_float(v_data, V32, DV);
                    ggml_vec_mad_f32(DV, VKQ32, V32, vs);
                } অন্যথা {
                    // V is F32
                    ggml_vec_mad_f32(DV, VKQ32, (ধ্রুব ভিন্ন *) v_data, vs);
                }
            }

            S = S*ms + vs; // scale and increment sum with partial sum
        }

        যদি (v->type == GGML_TYPE_F16) {
            প্রতি (int64_t d = 0; d < DV; ++d) {
                VKQ32[d] = GGML_CPU_FP16_TO_FP32(VKQ16[d]);
            }
        }

        // sinks - apply only on the first kv-chunk
        যদি (sinks && ic_start == 0) {
            ধ্রুব ভিন্ন s = ((ভিন্ন *)((বর্ণ *) sinks->data))[h];

            ভিন্ন ms = 1.0f;
            ভিন্ন vs = 1.0f;

            যদি (s > M) {
                ms = expf(M - s);
                M = s;
                ggml_vec_scale_f32(DV, VKQ32, ms);
            } অন্যথা {
                vs = expf(s - M);
            }

            S = S*ms + vs;
        }

        যদি (write_partials) {
            // Write M, S, VKQ to partials for later reduction
            // partials layout: [M, S, VKQ[DV]] per query head
            ভিন্ন * partial = partials + ir * partial_stride;
            partial[0] = M;
            partial[1] = S;
            memcpy(partial + 2, VKQ32, DV * পরিমাণ(ভিন্ন));
        } অন্যথা {
            // V /= S
            ধ্রুব ভিন্ন S_inv = S == 0.0f ? 0.0f : 1.0f/S;
            ggml_vec_scale_f32(DV, VKQ32, S_inv);

            // dst indices
            ধ্রুব পূর্ণাঙ্ক i1 = iq1;
            ধ্রুব পূর্ণাঙ্ক i2 = iq2;
            ধ্রুব পূর্ণাঙ্ক i3 = iq3;

            // permute(0, 2, 1, 3)
            memcpy((বর্ণ *) dst->data + (i3*ne2*ne1 + i2 + i1*ne1)*nb1, VKQ32, nb1);
        }
    }
}

স্থির শূন্য ggml_compute_forward_flash_attn_ext_tiled(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst,
        পূর্ণাঙ্ক ir0, পূর্ণাঙ্ক ir1) {
    ধ্রুব ggml_tensor * q     = dst->src[0];
    ধ্রুব ggml_tensor * k     = dst->src[1];
    ধ্রুব ggml_tensor * v     = dst->src[2];
    ধ্রুব ggml_tensor * mask  = dst->src[3];
    ধ্রুব ggml_tensor * sinks = dst->src[4];

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    ধ্রুব int64_t DK = nek0;
    ধ্রুব int64_t DV = nev0;
    ধ্রুব int64_t N  = neq1;

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
    GGML_ASSERT(nb0 == পরিমাণ(ভিন্ন));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(k->type == v->type);
    ধ্রুব ggml_type kv_type = k->type;


    // broadcast factors
    ধ্রুব int64_t rk2 = neq2/nek2;
    ধ্রুব int64_t rk3 = neq3/nek3;

    ধ্রুব int64_t rv2 = neq2/nev2;
    ধ্রুব int64_t rv3 = neq3/nev3;

    ভিন্ন scale         = 1.0f;
    ভিন্ন max_bias      = 0.0f;
    ভিন্ন logit_softcap = 0.0f;

    memcpy(&scale,         (ভিন্ন *) dst->op_params + 0, পরিমাণ(ভিন্ন));
    memcpy(&max_bias,      (ভিন্ন *) dst->op_params + 1, পরিমাণ(ভিন্ন));
    memcpy(&logit_softcap, (ভিন্ন *) dst->op_params + 2, পরিমাণ(ভিন্ন));

    যদি (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    ধ্রুব uint32_t n_head      = neq2;
    ধ্রুব uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));

    ধ্রুব ভিন্ন m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    ধ্রুব ভিন্ন m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    পূর্ণাঙ্ক ith = params->ith;

    স্থির constexpr পূর্ণাঙ্ক Q_TILE_SZ  = ggml_fa_tile_config::Q;
    স্থির constexpr পূর্ণাঙ্ক KV_TILE_SZ = ggml_fa_tile_config::KV;

    পূর্ণাঙ্ক ir = ir0;
    যাবত্ (ir < ir1) {
        // q indices for the start of this tile
        ধ্রুব পূর্ণাঙ্ক iq3 = ir/(neq2*neq1);
        ধ্রুব পূর্ণাঙ্ক iq2 = (ir - iq3*neq2*neq1)/neq1;
        ধ্রুব পূর্ণাঙ্ক iq1 = (ir - iq3*neq2*neq1 - iq2*neq1);

        // Number of valid rows in this tile:
        // - limited by tile size (Q_TILE_SZ)
        // - limited by chunk boundary (ir1 - ir)
        // - limited by head boundary (neq1 - iq1) to avoid crossing into next head
        ধ্রুব পূর্ণাঙ্ক tile_rows = MIN(Q_TILE_SZ, MIN((পূর্ণাঙ্ক)(ir1 - ir), (পূর্ণাঙ্ক)(neq1 - iq1)));
        GGML_ASSERT(tile_rows > 0);

        ধ্রুব uint32_t h = iq2; // head index
        ধ্রুব ভিন্ন slope = (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1) : 1.0f;

        ভিন্ন S[Q_TILE_SZ];
        ভিন্ন M[Q_TILE_SZ];

        প্রতি (পূর্ণাঙ্ক i = 0 ; i < Q_TILE_SZ; ++i) {
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
        ভিন্ন * base  = (ভিন্ন *) params->wdata + ith*(Q_TILE_SZ*DK + 2*Q_TILE_SZ*KV_TILE_SZ + Q_TILE_SZ*DV + KV_TILE_SZ*DV + KV_TILE_SZ*DK + CACHE_সারি_SIZE_F32);

        শূন্য  * Q_q    = base;
        ভিন্ন * KQ     = (ভিন্ন *)((বর্ণ *)base + Q_TILE_SZ * DK * পরিমাণ(ভিন্ন));
        ভিন্ন * mask32 = KQ + Q_TILE_SZ * KV_TILE_SZ;
        ভিন্ন * VKQ32  = mask32 + Q_TILE_SZ * KV_TILE_SZ;
        ভিন্ন * V32    = VKQ32 + Q_TILE_SZ * DV;
        ভিন্ন * K_f32  = V32 + KV_TILE_SZ * DV;

        memset(VKQ32, 0, Q_TILE_SZ * DV * পরিমাণ(ভিন্ন));
        memset(mask32, 0, Q_TILE_SZ * KV_TILE_SZ * পরিমাণ(ভিন্ন));

        // k indices
        ধ্রুব পূর্ণাঙ্ক ik3 = iq3 / rk3;
        ধ্রুব পূর্ণাঙ্ক ik2 = iq2 / rk2;

        // v indices
        ধ্রুব পূর্ণাঙ্ক iv3 = iq3 / rv3;
        ধ্রুব পূর্ণাঙ্ক iv2 = iq2 / rv2;

        {
            ভিন্ন * Q_f32 = (ভিন্ন *)Q_q;
            প্রতি (পূর্ণাঙ্ক tq = 0; tq < tile_rows; tq++) {
                ধ্রুব ভিন্ন * pq = (ধ্রুব ভিন্ন *) ((বর্ণ *) q->data + ((iq1 + tq)*nbq1 + iq2*nbq2 + iq3*nbq3));
                memcpy(Q_f32 + tq * DK, pq, DK * পরিমাণ(ভিন্ন));
            }
            প্রতি (পূর্ণাঙ্ক tq = tile_rows; tq < Q_TILE_SZ; tq++) {
                memset(Q_f32 + tq * DK, 0, DK * পরিমাণ(ভিন্ন));
            }
        }

        memset(K_f32, 0, DK * KV_TILE_SZ * পরিমাণ(ভিন্ন));
        memset(V32,   0, KV_TILE_SZ * DV * পরিমাণ(ভিন্ন));

        প্রতি (int64_t ic = 0; ic < nek1; ic += KV_TILE_SZ) {
            ধ্রুব পূর্ণাঙ্ক kv_tile = (পূর্ণাঙ্ক)std::min((int64_t)KV_TILE_SZ, nek1 - ic);

            // skip the tile entirely if all the masks are -inf
            যদি (mask) {
                তার্কিক can_skip = সত্য;
                প্রতি (পূর্ণাঙ্ক tq = 0; tq < tile_rows; tq++) {
                    ধ্রুব ggml_fp16_t * mp_row = (ধ্রুব ggml_fp16_t *)((ধ্রুব বর্ণ *) mask->data + (iq1 + tq)*mask->nb[1] + (iq2%mask->ne[2])*mask->nb[2] + (iq3%mask->ne[3])*mask->nb[3]);
                    প্রতি (পূর্ণাঙ্ক tk = 0; tk < kv_tile; tk++) {
                        mask32[tq * KV_TILE_SZ + tk] = slope * GGML_CPU_FP16_TO_FP32(mp_row[ic + tk]);
                        যদি (mask32[tq * KV_TILE_SZ + tk] != -INFINITY) {
                            can_skip = অসত্য;
                        }
                    }
                    // Pad remaining mask entries with -inf
                    প্রতি (পূর্ণাঙ্ক tk = kv_tile; tk < KV_TILE_SZ; tk++) {
                        mask32[tq * KV_TILE_SZ + tk] = -INFINITY;
                    }
                }

                যদি (can_skip) {
                    অনুবর্ত;
                }
            }

            // Pack K tile transposed: K_f32[dk][kv] so KV_TILE is contiguous (SIMD dim)
            // Zero-pad the last tile so the GEMM always operates on KV_TILE_SZ columns
            প্রতি (পূর্ণাঙ্ক tk = 0; tk < kv_tile; tk++) {
                ধ্রুব বর্ণ * k_data = (ধ্রুব বর্ণ *)k->data + (ic + tk)*nbk1 + ik2*nbk2 + ik3*nbk3;
                যদি (kv_type == GGML_TYPE_F16) {
                    ধ্রুব ggml_fp16_t * k_f16 = (ধ্রুব ggml_fp16_t *)k_data;
                    প্রতি (int64_t dk = 0; dk < DK; dk++) {
                        K_f32[dk * KV_TILE_SZ + tk] = GGML_CPU_FP16_TO_FP32(k_f16[dk]);
                    }
                } অন্যথা {
                    ধ্রুব ভিন্ন * k_f32_src = (ধ্রুব ভিন্ন *)k_data;
                    প্রতি (int64_t dk = 0; dk < DK; dk++) {
                        K_f32[dk * KV_TILE_SZ + tk] = k_f32_src[dk];
                    }
                }
            }
            memset(KQ, 0, Q_TILE_SZ * KV_TILE_SZ * পরিমাণ(ভিন্ন));
            simd_gemm(KQ, (ধ্রুব ভিন্ন *)Q_q, K_f32, Q_TILE_SZ, DK, KV_TILE_SZ);
            ggml_vec_scale_f32(Q_TILE_SZ * KV_TILE_SZ, KQ, scale);

            যদি (logit_softcap != 0.0f) {
                ggml_vec_tanh_f32(Q_TILE_SZ * KV_TILE_SZ, KQ, KQ);
                ggml_vec_scale_f32(Q_TILE_SZ * KV_TILE_SZ, KQ, logit_softcap);
            }

            // Set padded KQ entries to -inf so softmax gives them zero weight
            যদি (kv_tile < KV_TILE_SZ) {
                প্রতি (পূর্ণাঙ্ক tq = 0; tq < Q_TILE_SZ; tq++) {
                    প্রতি (পূর্ণাঙ্ক tk = kv_tile; tk < KV_TILE_SZ; tk++) {
                        KQ[tq * KV_TILE_SZ + tk] = -INFINITY;
                    }
                }
            }

            যদি (mask) {
                ggml_vec_add_f32(tile_rows * KV_TILE_SZ, KQ, KQ, mask32);
            }

            তার্কিক skip[Q_TILE_SZ] = {};

            প্রতি (পূর্ণাঙ্ক tq = 0; tq < Q_TILE_SZ; tq++) {
                ভিন্ন * kq_row = KQ + tq * KV_TILE_SZ;

                ভিন্ন tile_max;
                ggml_vec_max_f32(KV_TILE_SZ, &tile_max, kq_row);

                যদি (tile_max == -INFINITY) {
                    skip[tq] = সত্য;
                    অনুবর্ত;
                }

                ধ্রুব ভিন্ন Mold = M[tq];
                ধ্রুব ভিন্ন Mnew = fmaxf(Mold, tile_max);

                যদি (Mnew > Mold) {
                    ধ্রুব ভিন্ন ms = expf(Mold - Mnew);
                    ggml_vec_scale_f32(DV, VKQ32 + tq * DV, ms);
                    S[tq] *= ms;
                }
                M[tq] = Mnew;


                S[tq] += ggml_vec_soft_max_f32(KV_TILE_SZ, kq_row, kq_row, Mnew);
            }

            // V accumulation: VKQ32 += softmax(KQ) * V
            // Pack V tile to contiguous F32, zero-padded
            প্রতি (পূর্ণাঙ্ক tk = 0; tk < kv_tile; tk++) {
                ধ্রুব বর্ণ * v_data = (ধ্রুব বর্ণ *)v->data + (ic + tk)*nbv1 + iv2*nbv2 + iv3*nbv3;
                যদি (kv_type == GGML_TYPE_F16) {
                    ggml_cpu_fp16_to_fp32((ধ্রুব ggml_fp16_t *)v_data, V32 + tk * DV, DV);
                } অন্যথা {
                    memcpy(V32 + tk * DV, v_data, DV * পরিমাণ(ভিন্ন));
                }
            }
            প্রতি (পূর্ণাঙ্ক tq = 0; tq < Q_TILE_SZ; tq++) {
                যদি (skip[tq]) {
                    memset(KQ + tq * KV_TILE_SZ, 0, KV_TILE_SZ * পরিমাণ(ভিন্ন));
                }
            }
            simd_gemm(VKQ32, KQ, V32, Q_TILE_SZ, KV_TILE_SZ, DV);
        }

        // sinks (apply only to valid rows in the tile)
        যদি (sinks) {
            ধ্রুব ভিন্ন s = ((ভিন্ন *)((বর্ণ *) sinks->data))[h];

            প্রতি (পূর্ণাঙ্ক tq = 0; tq < tile_rows; tq++) {
                ভিন্ন ms = 1.0f;
                ভিন্ন vs = 1.0f;

                যদি (s > M[tq]) {
                    ms = expf(M[tq] - s);
                    ggml_vec_scale_f32(DV, VKQ32 + tq * DV, ms);
                } অন্যথা {
                    vs = expf(s - M[tq]);
                }

                S[tq] = S[tq] * ms + vs;
            }
        }

        প্রতি (পূর্ণাঙ্ক tq = 0; tq < tile_rows; tq++) {
            // V /= S
            ধ্রুব ভিন্ন S_inv = S[tq] == 0.0f ? 0.0f : 1.0f / S[tq];
            ggml_vec_scale_f32(DV, VKQ32 + tq * DV, S_inv);

            // dst indices
            ধ্রুব পূর্ণাঙ্ক i1 = iq1 + tq;
            ধ্রুব পূর্ণাঙ্ক i2 = iq2;
            ধ্রুব পূর্ণাঙ্ক i3 = iq3;

            // permute(0, 2, 1, 3)
            memcpy((বর্ণ *) dst->data + (i3*ne2*ne1 + i2 + i1*ne1)*nb1, VKQ32 + tq * DV, nb1);
        }

        ir += tile_rows;
    }
}

// Reduction function: combines partial results across KV chunks
// Partials layout in wdata: [n_q_heads][n_chunks][2 + DV]
স্থির শূন্য ggml_flash_attn_ext_reduce_partials(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst,
        ধ্রুব int64_t n_chunks,
        ধ্রুব int64_t chunk_size) {

    ধ্রুব ggml_tensor * q = dst->src[0];
    ধ্রুব ggml_tensor * k = dst->src[1];
    ধ্রুব ggml_tensor * v = dst->src[2];

    ধ্রুব int64_t DK        = k->ne[0];
    ধ্রুব int64_t DV        = v->ne[0];
    ধ্রুব int64_t nek1      = k->ne[1];
    ধ্রুব int64_t n_q_heads = q->ne[2];

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব int64_t wdata_per_thread = DK + 2*DV + CACHE_সারি_SIZE_F32;
    ভিন্ন *       thread_wdata     = (ভিন্ন *) params->wdata + ith * wdata_per_thread;

    ধ্রুব int64_t partials_offset  = nth * (DK + 2*DV + CACHE_সারি_SIZE_F32);
    ধ্রুব int64_t partial_size     = 2 + DV;
    ধ্রুব ভিন্ন * partials_base    = (ধ্রুব ভিন্ন *) params->wdata + partials_offset;

    // Output layout
    ধ্রুব int64_t ne1 = dst->ne[1];
    ধ্রুব int64_t ne2 = dst->ne[2];
    ধ্রুব size_t  nb1 = dst->nb[1];

    // Each thread reduces a subset of query heads
    প্রতি (int64_t q_head = ith; q_head < n_q_heads; q_head += nth) {
        ভিন্ন   M_final   = -INFINITY;
        ভিন্ন   S_final   = 0.0f;
        ভিন্ন * VKQ_final = thread_wdata;
        memset(VKQ_final, 0, DV * পরিমাণ(ভিন্ন));

        // Combine partials from all chunks
        প্রতি (int64_t chunk_idx = 0; chunk_idx < n_chunks; ++chunk_idx) {
            ধ্রুব int64_t ic_start = chunk_idx * chunk_size;
            যদি (ic_start >= nek1) অনুবর্ত;

            ধ্রুব ভিন্ন * partial   = partials_base + (q_head * n_chunks + chunk_idx) * partial_size;
            ধ্রুব ভিন্ন   M_chunk   = partial[0];
            ধ্রুব ভিন্ন   S_chunk   = partial[1];
            ধ্রুব ভিন্ন * VKQ_chunk = partial + 2;

            যদি (S_chunk == 0.0f) অনুবর্ত;

            ধ্রুব ভিন্ন M_new     = fmaxf(M_final, M_chunk);
            ধ্রুব ভিন্ন scale_old = expf(M_final - M_new);
            ধ্রুব ভিন্ন scale_new = expf(M_chunk - M_new);

            প্রতি (int64_t d = 0; d < DV; ++d) {
                VKQ_final[d] = VKQ_final[d] * scale_old + VKQ_chunk[d] * scale_new;
            }
            S_final = S_final * scale_old + S_chunk * scale_new;
            M_final = M_new;
        }

        // Normalize and write to output
        যদি (S_final != 0.0f) {
            ধ্রুব ভিন্ন S_inv = 1.0f / S_final;
            ggml_vec_scale_f32(DV, VKQ_final, S_inv);
        }
        // iq1=0, iq3=0 for decode
        memcpy((বর্ণ *) dst->data + (0*ne2*ne1 + q_head + 0*ne1)*nb1, VKQ_final, nb1);
    }
}

স্থির শূন্য ggml_compute_forward_flash_attn_ext_f16(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * q     = dst->src[0];
    ধ্রুব ggml_tensor * k     = dst->src[1];
    ধ্রুব ggml_tensor * v     = dst->src[2];

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    ধ্রুব int64_t DK = nek0;
    ধ্রুব int64_t DV = nev0;
    ধ্রুব int64_t N  = neq1;


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
    GGML_ASSERT(nb0 == পরিমাণ(ভিন্ন));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    // When use_ref is set, force the vec-only reference implementation (no tiling, no KV-chunking)
    ধ্রুব তার্কিক use_ref = params->use_ref;

    ধ্রুব তার্কিক kv_is_f32_or_f16 = (k->type == GGML_TYPE_F32 || k->type == GGML_TYPE_F16);
    ধ্রুব তার্কিক use_split_kv_path = !use_ref && (neq1 == 1 && neq3 == 1) && kv_is_f32_or_f16 && (k->type == v->type) && q->type == GGML_TYPE_F32 && nek1 >= 512;

    যদি (use_split_kv_path) {
        ধ্রুব int64_t chunk_size = (nek1 + nth - 1) / nth;

        // Partials buffer layout: [q_head][kv_chunk][M, S, VKQ]
        ধ্রুব int64_t partial_size  = 2 + DV;
        ভিন্ন *       partials_base = (ভিন্ন *) params->wdata + nth * (DK + 2*DV + CACHE_সারি_SIZE_F32);

        ধ্রুব int64_t ic_start = ith * chunk_size;
        ধ্রুব int64_t ic_end   = std::min(ic_start + chunk_size, nek1);

        ধ্রুব int64_t partial_stride = nth * partial_size;
        ভিন্ন *       chunk_partials = partials_base + ith * partial_size;

        যদি (ic_start < nek1) {
            প্রতি (int64_t q_head = 0; q_head < neq2; q_head++) {
                ggml_compute_forward_flash_attn_ext_f16_one_chunk(
                    params, dst, q_head, q_head + 1, ic_start, ic_end,
                    chunk_partials, partial_stride);
            }
        } অন্যথা {
            প্রতি (int64_t q_head = 0; q_head < neq2; q_head++) {
                ভিন্ন * q_partials = chunk_partials + q_head * partial_stride;
                q_partials[0] = -INFINITY;  // M
                q_partials[1] = 0.0f;       // S
            }
        }

        ggml_barrier(params->threadpool);
        ggml_flash_attn_ext_reduce_partials(params, dst, nth, chunk_size);
    } অন্যথা {

        // total rows in q
        ধ্রুব int64_t nr = neq1*neq2*neq3;

        // disable for NUMA
        ধ্রুব তার্কিক disable_chunking = ggml_is_numa();

        // 4x chunks per thread
        পূর্ণাঙ্ক nth_scaled = nth * 4;
        int64_t chunk_size = (nr + nth_scaled - 1) / nth_scaled;
        int64_t nchunk     = (nr + chunk_size - 1) / chunk_size;

        যদি (nth == 1 || nchunk < nth || disable_chunking) {
            nchunk = nth;
        }

        যদি (ith == 0) {
            ggml_threadpool_chunk_set(params->threadpool, nth);
        }

        ggml_barrier(params->threadpool);

        ধ্রুব int64_t dr = (nr + nchunk - 1) / nchunk;

        স্থির constexpr int64_t Q_TILE_SZ  = ggml_fa_tile_config::Q;
        তার্কিক use_tiled = !use_ref &&
                               (q->type == GGML_TYPE_F32 &&
                                kv_is_f32_or_f16 &&
                                k->type == v->type &&
                                neq1 >= Q_TILE_SZ);
#যদি defined(GGML_SIMD) && !defined(__x86_64__) && !defined(_M_X64)
#যদি defined(__ARM_FEATURE_SVE)
        ধ্রুব int64_t f32_epr = svcntw();
#অন্যথা
        ধ্রুব int64_t f32_epr = GGML_F32_EPR;
#endif
        use_tiled &= (DV % f32_epr == 0);
#endif
        পূর্ণাঙ্ক current_chunk = ith;

        যাবত্ (current_chunk < nchunk) {
            ধ্রুব int64_t ir0 = dr * current_chunk;
            ধ্রুব int64_t ir1 = MIN(ir0 + dr, nr);

            যদি (use_tiled) {
                ggml_compute_forward_flash_attn_ext_tiled(params, dst, ir0, ir1);
            } অন্যথা {
                ggml_compute_forward_flash_attn_ext_f16_one_chunk(params, dst, ir0, ir1, 0, nek1, nullptr, 0);
            }

            current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
        }
    }
}

শূন্য ggml_compute_forward_flash_attn_ext(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {
    বিকল্প (dst->op_params[3]) {
        পক্ষ GGML_PREC_DEFAULT:
        পক্ষ GGML_PREC_F32:
            {
                // uses F32 accumulators
                ggml_compute_forward_flash_attn_ext_f16(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_flash_attn_back

স্থির শূন্য ggml_compute_forward_flash_attn_back_f32(
        ধ্রুব ggml_compute_params * params,
        ধ্রুব তার্কিক masked,
              ggml_tensor * dst) {

    ধ্রুব ggml_tensor * q = dst->src[0];
    ধ্রুব ggml_tensor * k = dst->src[1];
    ধ্রুব ggml_tensor * v = dst->src[2];
    ধ্রুব ggml_tensor * d = dst->src[3];

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

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব int64_t D = neq0;
    ধ্রুব int64_t N = neq1;
    ধ্রুব int64_t P = nek1 - N;
    ধ্রুব int64_t M = P + N;

    ধ্রুব পূর্ণাঙ্ক Mup  = ggml_up(M, GGML_SOFT_MAX_UNROLL);
    ধ্রুব পূর্ণাঙ্ক mxDM = MAX(D, Mup);

    // GGML_ASSERT(ne0 == D);
    // GGML_ASSERT(ne1 == N);
    GGML_ASSERT(P >= 0);

    GGML_ASSERT(nbq0 == পরিমাণ(ভিন্ন));
    GGML_ASSERT(nbk0 == পরিমাণ(ভিন্ন));
    GGML_ASSERT(nbv0 == পরিমাণ(ভিন্ন));

    GGML_ASSERT(neq0 == D);
    GGML_ASSERT(nek0 == D);
    GGML_ASSERT(nev1 == D);
    GGML_ASSERT(ned0 == D);

    GGML_ASSERT(neq1 == N);
    GGML_ASSERT(nek1 == N + P);
    GGML_ASSERT(nev1 == D);
    GGML_ASSERT(ned1 == N);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == পরিমাণ(ভিন্ন));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    যদি (ith == 0) {
        memset(dst->data, 0, nb0*ne0*ne1*ne2*ne3);
    }
    ggml_barrier(params->threadpool);

    ধ্রুব int64_t elem_q = ggml_nelements(q);
    ধ্রুব int64_t elem_k = ggml_nelements(k);

    ggml_type result_type = dst->type;
    GGML_ASSERT(ggml_blck_size(result_type) == 1);
    ধ্রুব size_t tsize = ggml_type_size(result_type);

    ধ্রুব size_t offs_q = 0;
    ধ্রুব size_t offs_k = offs_q + GGML_PAD(elem_q * tsize, GGML_MEM_ALIGN);
    ধ্রুব size_t offs_v = offs_k + GGML_PAD(elem_k * tsize, GGML_MEM_ALIGN);

    শূন্য * grad_q = (বর্ণ *) dst->data;
    শূন্য * grad_k = (বর্ণ *) dst->data + offs_k;
    শূন্য * grad_v = (বর্ণ *) dst->data + offs_v;

    ধ্রুব size_t nbgq1 = nb0*neq0;
    ধ্রুব size_t nbgq2 = nb0*neq0*neq1;
    ধ্রুব size_t nbgq3 = nb0*neq0*neq1*neq2;

    ধ্রুব size_t nbgk1 = nb0*nek0;
    ধ্রুব size_t nbgk2 = nb0*nek0*nek1;
    ধ্রুব size_t nbgk3 = nb0*nek0*nek1*neq2;

    ধ্রুব size_t nbgv1 = nb0*nev0;
    ধ্রুব size_t nbgv2 = nb0*nev0*nev1;
    ধ্রুব size_t nbgv3 = nb0*nev0*nev1*neq2;

    // parallelize by k rows using ggml_vec_dot_f32

    // total rows in k
    ধ্রুব পূর্ণাঙ্ক nr = nek2*nek3;

    // rows per thread
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr*ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    ধ্রুব ভিন্ন scale = 1.0f/sqrtf(D);

    //printf("P=%d N=%d D=%d ir0=%d ir1=%d scale = %f\n", P, N, D, ir0, ir1, scale);

    // how often k2 (and v2) is repeated in q2
    পূর্ণাঙ্ক nrep = neq2/nek2;

    প্রতি (পূর্ণাঙ্ক ir = ir0; ir < ir1; ++ir) {
        // q indices
        ধ্রুব পূর্ণাঙ্ক ik3 = ir/(nek2);
        ধ্রুব পূর্ণাঙ্ক ik2 = ir - ik3*nek2;

        ধ্রুব পূর্ণাঙ্ক iq3 = ik3;
        ধ্রুব পূর্ণাঙ্ক id3 = ik3;
        ধ্রুব পূর্ণাঙ্ক iv3 = ik3;
        ধ্রুব পূর্ণাঙ্ক iv2 = ik2;

        প্রতি (পূর্ণাঙ্ক irep = 0; irep < nrep; ++irep) {
            ধ্রুব পূর্ণাঙ্ক iq2 = ik2 + irep*nek2;
            ধ্রুব পূর্ণাঙ্ক id2 = iq2;

            // (ik2 + irep*nek2) % nek2 == ik2
            প্রতি (পূর্ণাঙ্ক iq1 = 0; iq1 < neq1; ++iq1) {
                ধ্রুব পূর্ণাঙ্ক id1 = iq1;

                // not sure about CACHE_LINE_SIZE_F32..
                // - maybe it must not be multiplied by 2 and excluded from .. in SM 1*(..) offset?
                ভিন্ন * S  = (ভিন্ন *) params->wdata + ith*2*(mxDM + CACHE_সারি_SIZE_F32) + 0*(mxDM+CACHE_সারি_SIZE_F32);
                ভিন্ন * SM = (ভিন্ন *) params->wdata + ith*2*(mxDM + CACHE_সারি_SIZE_F32) + 1*(mxDM+CACHE_সারি_SIZE_F32);

                প্রতি (পূর্ণাঙ্ক i = M; i < Mup; ++i) {
                    S[i] = -INFINITY;
                }

                ধ্রুব int64_t masked_begin = masked ? (P + iq1 + 1) : M;
                প্রতি (int64_t ic = 0; ic < masked_begin; ++ic) {
                    // k indices
                    ধ্রুব পূর্ণাঙ্ক ik1 = ic;

                    // S indices
                    ধ্রুব পূর্ণাঙ্ক i1 = ik1;

                    ggml_vec_dot_f32(neq0,
                            S + i1, 0,
                            (ভিন্ন *) ((বর্ণ *) k->data + (ik1*nbk1 + ik2*nbk2 + ik3*nbk3)), 0,
                            (ভিন্ন *) ((বর্ণ *) q->data + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3)), 0, 1);
                }

                // scale
                ggml_vec_scale_f32(masked_begin, S, scale);

                প্রতি (int64_t i = masked_begin; i < M; i++) {
                    S[i] = -INFINITY;
                }

                // softmax
                // exclude known -INF S[..] values from max and loop
                // dont forget to set their SM values to zero
                {
                    ভিন্ন max = -INFINITY;
                    ggml_vec_max_f32(masked_begin, &max, S);

                    ggml_float sum = 0.0;
                    {
#ifdef GGML_SOFT_MAX_ACCELERATE
                        max = -max;
                        vDSP_vsadd(SM, 1, &max, SM, 1, Mup);
                        vvexpf(SM, SM, &Mup);
                        ggml_vec_sum_f32(Mup, &sum, SM);
#অন্যথা
                        sum = ggml_vec_soft_max_f32(Mup, SM, S, max);
#endif
                    }

                    প্রতিজ্ঞা(sum > 0.0);

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
                প্রতি (int64_t ic = 0; ic < D; ++ic) {
                    ggml_vec_mad_f32(masked_begin,
                            S,
                             (ভিন্ন *) ((বর্ণ *) v->data + (          ic*nbv1  + iv2*nbv2 + iv3*nbv3)),
                            *(ভিন্ন *) ((বর্ণ *) d->data + (ic*nbd0 + id1*nbd1 + id2*nbd2 + id3*nbd3)));
                }

                // S = SM * (S - dot(SM, S))
                ভিন্ন dot_SM_gradSM = 0;
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
                প্রতি (int64_t ic = 0; ic < masked_begin; ++ic) {
                    ggml_vec_mad_f32(D,
                            (ভিন্ন *) ((বর্ণ *) grad_q  + (iq1*nbgq1 + iq2*nbgq2  + iq3*nbgq3)),
                            (ভিন্ন *) ((বর্ণ *) k->data + (ic*nbk1   + ik2*nbk2   + ik3*nbk3)),
                            S[ic]);
                }

                // grad[k][:D,:M,iq2,iq3] += S.T @ qcur
                // for ic:
                //  grad[k][:D,ic,iq2,iq3] += S.T[0,ic] * qcur[:D,0]
                //  grad[k][:D,ic,iq2,iq3] += S[ic]     * qcur[:D,0]
                // exclude known zero S[..] values from loop
                প্রতি (int64_t ic = 0; ic < masked_begin; ++ic) {
                    ggml_vec_mad_f32(D,
                            (ভিন্ন *) ((বর্ণ *) grad_k  + (ic*nbgk1  + ik2*nbgk2  + ik3*nbgk3)),
                            (ভিন্ন *) ((বর্ণ *) q->data + (iq1*nbq1  + iq2*nbq2   + iq3*nbq3)),
                            S[ic]);
                }

                // grad[v][:M,:D,iv2,iv3] += d[:D,id1,id2,id3].T       @ SM
                // for ic:
                //  grad[v][:M,ic,iv2,iv3] += d[:D,id1,id2,id3].T[0,ic] * SM[:M]
                //  grad[v][:M,ic,iv2,iv3] += d[ic,id1,id2,id3]         * SM[:M]
                // exclude known zero SM[..] values from mad
                প্রতি (int64_t ic = 0; ic < D; ++ic) {
                    ggml_vec_mad_f32(masked_begin,
                            (ভিন্ন *) ((বর্ণ *) grad_v   + (          ic*nbgv1 + iv2*nbgv2 + iv3*nbgv3)),
                            SM,
                            *(ভিন্ন *) ((বর্ণ *) d->data + (ic*nbd0 + id1*nbd1 + id2*nbd2  + id3*nbd3)));
                }
            }
        }
    }
}

শূন্য ggml_compute_forward_flash_attn_back(
        ধ্রুব ggml_compute_params * params,
        ধ্রুব তার্কিক masked,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * q = dst->src[0];

    বিকল্প (q->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_flash_attn_back_f32(params, masked, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_ssm_conv

স্থির শূন্য ggml_compute_forward_ssm_conv_f32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {
    ধ্রুব ggml_tensor * src0 = dst->src[0]; // conv_x
    ধ্রুব ggml_tensor * src1 = dst->src[1]; // conv1d.weight

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক nc  = src1->ne[0]; // d_conv
    ধ্রুব পূর্ণাঙ্ক ncs = src0->ne[0]; // d_conv - 1 + n_t
    ধ্রুব পূর্ণাঙ্ক nr  = src0->ne[1]; // d_inner
    ধ্রুব পূর্ণাঙ্ক n_t =  dst->ne[1]; // tokens per sequence
    ধ্রুব পূর্ণাঙ্ক n_s =  dst->ne[2]; // number of sequences in the batch

    GGML_ASSERT( dst->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == পরিমাণ(ভিন্ন));
    GGML_ASSERT(src1->nb[0] == পরিমাণ(ভিন্ন));
    GGML_ASSERT(src0->nb[1] == src0->ne[0]*পরিমাণ(ভিন্ন));

    // rows per thread
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr*ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);
    ধ্রুব পূর্ণাঙ্ক ir  = ir1 - ir0;

    প্রতি (পূর্ণাঙ্ক i3 = 0; i3 < n_s; ++i3) {
        প্রতি (পূর্ণাঙ্ক i2 = 0; i2 < n_t; ++i2) {
            // {d_conv - 1 + n_t, d_inner, n_seqs}
            // sliding window
            ধ্রুব ভিন্ন * s = (ধ্রুব ভিন্ন *) ((ধ্রুব বর্ণ *) src0->data + ir0*(src0->nb[1]) + i2*(src0->nb[0]) + i3*(src0->nb[2])); // {d_conv, d_inner, n_s}
            ধ্রুব ভিন্ন * c = (ধ্রুব ভিন্ন *) ((ধ্রুব বর্ণ *) src1->data + ir0*(src1->nb[1])); // {d_conv, d_inner}
            ভিন্ন * x = (ভিন্ন *) ((বর্ণ *) dst->data + ir0*(dst->nb[0]) + i2*(dst->nb[1]) + i3*(dst->nb[2])); // {d_inner, n_t, n_s}

            // TODO: transpose the output for smaller strides for big batches?
            // d_inner
            প্রতি (পূর্ণাঙ্ক i1 = 0; i1 < ir; ++i1) {
                // rowwise dot product
                // NOTE: not using ggml_vec_dot_f32, because its sum is in double precision
                ভিন্ন sumf = 0.0f;

                // d_conv
                প্রতি (পূর্ণাঙ্ক i0 = 0; i0 < nc; ++i0) {
                    sumf += s[i0 + i1*ncs] * c[i0 + i1*nc];
                }
                x[i1] = sumf;
            }
        }
    }
}

শূন্য ggml_compute_forward_ssm_conv(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {
    বিকল্প (dst->src[0]->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_ssm_conv_f32(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_ssm_scan

স্থির শূন্য ggml_compute_forward_ssm_scan_f32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {
    ধ্রুব ggml_tensor * src0 = dst->src[0]; // s  {d_state, dim, n_head, n_seqs+}
    ধ্রুব ggml_tensor * src1 = dst->src[1]; // x  {dim, n_head, n_seq_tokens, n_seqs}
    ধ্রুব ggml_tensor * src2 = dst->src[2]; // dt {n_head, n_seq_tokens, n_seqs}
    ধ্রুব ggml_tensor * src3 = dst->src[3]; // A  {d_state, n_head} or {1, n_head}
    ধ্রুব ggml_tensor * src4 = dst->src[4]; // B  {d_state, n_group, n_seq_tokens, n_seqs}
    ধ্রুব ggml_tensor * src5 = dst->src[5]; // C  {d_state, n_group, n_seq_tokens, n_seqs}
    ধ্রুব ggml_tensor * src6 = dst->src[6]; // ids {n_seqs}

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব int64_t nc = src0->ne[0]; // d_state
    ধ্রুব int64_t nr = src0->ne[1]; // dim
    ধ্রুব int64_t nh = src1->ne[1]; // n_head
    ধ্রুব int64_t ng = src4->ne[1];
    ধ্রুব int64_t nt = src1->ne[2]; // number of tokens per sequence
    ধ্রুব int64_t ns = src1->ne[3]; // number of sequences in the batch
    ধ্রুব int64_t K  = ggml_get_op_params_i32(dst, 0);

    // can't use ggml_nbytes because src1 is not necessarily contiguous
    ধ্রুব int64_t s_off = ggml_nelements(src1) * ggml_element_size(src1);

    GGML_ASSERT(K >= 1);
    GGML_ASSERT(ggml_nelements(src1) + K*nc*nr*nh*ns == ggml_nelements(dst));
    GGML_ASSERT(src0->nb[0] == পরিমাণ(ভিন্ন));
    GGML_ASSERT(src1->nb[0] == পরিমাণ(ভিন্ন));
    GGML_ASSERT(src2->nb[0] == পরিমাণ(ভিন্ন));
    GGML_ASSERT(src3->nb[0] == পরিমাণ(ভিন্ন));
    GGML_ASSERT(src4->nb[0] == পরিমাণ(ভিন্ন));
    GGML_ASSERT(src5->nb[0] == পরিমাণ(ভিন্ন));
    GGML_ASSERT(src6->nb[0] == পরিমাণ(int32_t));
    GGML_ASSERT(nh % ng == 0);
    GGML_ASSERT(src3->ne[0] == 1 || K == 1);

    // heads per thread
    ধ্রুব পূর্ণাঙ্ক dh = (nh + nth - 1)/nth;

    // head range for this thread
    ধ্রুব পূর্ণাঙ্ক ih0 = dh*ith;
    ধ্রুব পূর্ণাঙ্ক ih1 = MIN(ih0 + dh, nh);

    ধ্রুব int32_t * ids = (ধ্রুব int32_t *) src6->data;

    প্রতি (পূর্ণাঙ্ক i3 = 0; i3 < ns; ++i3) {
        ধ্রুব ভিন্ন * s0 = (ধ্রুব ভিন্ন *) ((ধ্রুব বর্ণ *) src0->data + ids[i3]*(src0->nb[3])); // {d_state, dim, nh, ns}
              ভিন্ন * s  = (      ভিন্ন *) ((      বর্ণ *) dst->data  + i3*(src0->nb[3]) + s_off); // {d_state, dim, nh, ns}

        প্রতি (পূর্ণাঙ্ক i2 = 0; i2 < nt; ++i2) {
            ধ্রুব ভিন্ন * x  = (ধ্রুব ভিন্ন *) ((ধ্রুব বর্ণ *) src1->data + i2*(src1->nb[2]) + i3*(src1->nb[3])); // {dim, nh, nt, ns}
            ধ্রুব ভিন্ন * dt = (ধ্রুব ভিন্ন *) ((ধ্রুব বর্ণ *) src2->data + i2*(src2->nb[1]) + i3*(src2->nb[2])); // {nh, nt, ns}
            ধ্রুব ভিন্ন * A  = (ধ্রুব ভিন্ন *) ((ধ্রুব বর্ণ *) src3->data); // {d_state, nh} or {1, nh}
            ধ্রুব ভিন্ন * B  = (ধ্রুব ভিন্ন *) ((ধ্রুব বর্ণ *) src4->data + i2*(src4->nb[2]) + i3*(src4->nb[3])); // {d_state, ng, nt, ns}
            ধ্রুব ভিন্ন * C  = (ধ্রুব ভিন্ন *) ((ধ্রুব বর্ণ *) src5->data + i2*(src5->nb[2]) + i3*(src5->nb[3])); // {d_state, ng, nt, ns}
                  ভিন্ন * y  = (      ভিন্ন *) ((      বর্ণ *) dst->data + i2*(nh*nr*পরিমাণ(ভিন্ন)) + i3*(nt*nh*nr*পরিমাণ(ভিন্ন))); // {dim, nh, nt, ns}

            যদি (src3->ne[0] == 1) {
                // Mamba-2 has a scalar decay factor per head; dA can be outside the state-wise loop

                // n_head
                প্রতি (পূর্ণাঙ্ক h = ih0; h < ih1; ++h) {
                    // ref: https://github.com/state-spaces/mamba/blob/62db608da60f6fc790b8ed9f4b3225e95ca15fde/mamba_ssm/ops/triton/softplus.py#L16
                    ধ্রুব ভিন্ন dt_soft_plus = ggml_compute_softplus_f32(dt[h]);
                    ধ্রুব ভিন্ন dA = expf(dt_soft_plus * A[h]);
                    ধ্রুব পূর্ণাঙ্ক g = h / (nh / ng); // repeat_interleave

                    // dim
                    প্রতি (পূর্ণাঙ্ক i1 = 0; i1 < nr; ++i1) {
                        ধ্রুব পূর্ণাঙ্ক ii = i1 + h*nr;
                        ধ্রুব ভিন্ন x_dt = x[ii] * dt_soft_plus;
                        ভিন্ন sumf = 0.0f;
#যদি defined(GGML_SIMD)
    #যদি defined(__ARM_FEATURE_SVE)
                        ধ্রুব পূর্ণাঙ্ক ggml_f32_epr = svcntw();
                        ধ্রুব পূর্ণাঙ্ক ggml_f32_step = 1 * ggml_f32_epr;

                        ধ্রুব পূর্ণাঙ্ক np = (nc & ~(ggml_f32_step - 1));

                        GGML_F32_VEC sum = GGML_F32_VEC_ZERO;

                        GGML_F32_VEC adA = GGML_F32_VEC_SET1(dA);
                        GGML_F32_VEC axdt = GGML_F32_VEC_SET1(x_dt);

                        প্রতি (পূর্ণাঙ্ক i = 0; i < np; i += ggml_f32_step) {
                            // TODO: maybe unroll more?
                            প্রতি (পূর্ণাঙ্ক j = 0; j < 1; j++) {
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
    #নোচেদ্যদি defined(__riscv_v_intrinsic)
                        // todo: RVV implementation
                        ধ্রুব পূর্ণাঙ্ক np = 0;
    #অন্যথা
                        ধ্রুব পূর্ণাঙ্ক np = (nc & ~(GGML_F32_STEP - 1));

                        GGML_F32_VEC sum[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };

                        GGML_F32_VEC adA = GGML_F32_VEC_SET1(dA);
                        GGML_F32_VEC axdt = GGML_F32_VEC_SET1(x_dt);

                        GGML_F32_VEC ax[GGML_F32_ARR];
                        GGML_F32_VEC ay[GGML_F32_ARR];
                        GGML_F32_VEC az[GGML_F32_ARR];

                        প্রতি (পূর্ণাঙ্ক i = 0; i < np; i += GGML_F32_STEP) {
                            প্রতি (পূর্ণাঙ্ক j = 0; j < GGML_F32_ARR; j++) {
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
#অন্যথা
                        ধ্রুব পূর্ণাঙ্ক np = 0;
#endif
                        // d_state
                        প্রতি (পূর্ণাঙ্ক i0 = np; i0 < nc; ++i0) {
                            ধ্রুব পূর্ণাঙ্ক i = i0 + ii*nc;
                            ধ্রুব পূর্ণাঙ্ক ig = i0 + g*nc;
                            // state = prev_state * dA + dB * x
                            ধ্রুব ভিন্ন state = (s0[i] * dA) + (B[ig] * x_dt);
                            // y = rowwise_dotprod(state, C)
                            sumf += state * C[ig];
                            s[i] = state;
                        }
                        y[ii] = sumf;
                    }
                }
            } অন্যথা {
                // Mamba-1 has an element-wise decay factor for the states

                // n_head
                প্রতি (পূর্ণাঙ্ক h = ih0; h < ih1; ++h) {
                    // ref: https://github.com/state-spaces/mamba/blob/62db608da60f6fc790b8ed9f4b3225e95ca15fde/mamba_ssm/ops/triton/softplus.py#L16
                    ধ্রুব ভিন্ন dt_soft_plus = ggml_compute_softplus_f32(dt[h]);
                    ধ্রুব পূর্ণাঙ্ক g = h / (nh / ng); // repeat_interleave

                    // dim
                    প্রতি (পূর্ণাঙ্ক i1 = 0; i1 < nr; ++i1) {
                        ধ্রুব পূর্ণাঙ্ক ii = i1 + h*nr;
                        ধ্রুব ভিন্ন x_dt = x[ii] * dt_soft_plus;
#যদি defined(__ARM_FEATURE_SVE)
                        svfloat32_t vx_dt = GGML_F32_VEC_SET1(x_dt);
                        svfloat32_t vdt_soft_plus = GGML_F32_VEC_SET1(dt_soft_plus);
                        svfloat32_t r1_vector = GGML_F32_VEC_ZERO;

                        // d_state
                        // TODO: what happens when (d_state % svcntw()) != 0?
                        প্রতি (int64_t k = 0; k < nc; k += svcntw()) {
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
#অন্যথা
                        ভিন্ন sumf = 0.0f;
                        // NOTE: can't really use GGML_SIMD here because d_state is usually 16
                        //       and also because expf is used within the loop.
                        // d_state
                        প্রতি (পূর্ণাঙ্ক i0 = 0; i0 < nc; ++i0) {
                            ধ্রুব পূর্ণাঙ্ক i = i0 + ii*nc;
                            ধ্রুব পূর্ণাঙ্ক ig = i0 + g*nc;
                            // state = prev_state * dA + dB * x
                            ধ্রুব ভিন্ন state = (s0[i] * expf(dt_soft_plus * A[i0 + h*nc])) + (B[ig] * x_dt);
                            // y = rowwise_dotprod(state, C)
                            sumf += state * C[ig];
                            s[i] = state;
                        }
                        y[ii] = sumf;
#endif
                    }
                }
            }
            ধ্রুব int64_t slot = nt - 1 - i2;
            যদি (K > 1 && slot > 0 && slot < K) {
                ভিন্ন * s_snapshot = (ভিন্ন *) ((বর্ণ *) dst->data + s_off + (slot*ns + i3)*(src0->nb[3]));
                প্রতি (পূর্ণাঙ্ক h = ih0; h < ih1; ++h) {
                    memcpy((বর্ণ *) s_snapshot + h*src0->nb[2], (বর্ণ *) s + h*src0->nb[2], src0->nb[2]);
                }
            }
            // use the output as the source when it's not the first token-wise iteration
            s0 = s;
        }
    }
}

শূন্য ggml_compute_forward_ssm_scan(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {
    বিকল্প (dst->src[0]->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_ssm_scan_f32(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_win_part

স্থির শূন্য ggml_compute_forward_win_part_f32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {
    GGML_UNUSED(params);

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)

    ধ্রুব int32_t nep0 = ((ধ্রুব int32_t *)(dst->op_params))[0];
    ধ্রুব int32_t nep1 = ((ধ্রুব int32_t *)(dst->op_params))[1];
    ধ্রুব int32_t w    = ((ধ্রুব int32_t *)(dst->op_params))[2];

    প্রতিজ্ঞা(ne00 == ne0);
    প্রতিজ্ঞা(ne3  == nep0*nep1);

    // TODO: optimize / multi-thread
    প্রতি (পূর্ণাঙ্ক py = 0; py < nep1; ++py) {
        প্রতি (পূর্ণাঙ্ক px = 0; px < nep0; ++px) {
            ধ্রুব int64_t i3 = py*nep0 + px;
            প্রতি (int64_t i2 = 0; i2 < ne2; ++i2) {
                প্রতি (int64_t i1 = 0; i1 < ne1; ++i1) {
                    প্রতি (int64_t i0 = 0; i0 < ne0; ++i0) {
                        ধ্রুব int64_t i02 = py*w + i2;
                        ধ্রুব int64_t i01 = px*w + i1;
                        ধ্রুব int64_t i00 = i0;

                        ধ্রুব int64_t i = i3*ne2*ne1*ne0 + i2*ne1*ne0    + i1*ne0   + i0;
                        ধ্রুব int64_t j =                  i02*ne01*ne00 + i01*ne00 + i00;

                        যদি (py*w + i2 >= ne02 || px*w + i1 >= ne01) {
                            ((ভিন্ন *) dst->data)[i] = 0.0f;
                        } অন্যথা {
                            ((ভিন্ন *) dst->data)[i] = ((ভিন্ন *) src0->data)[j];
                        }
                    }
                }
            }
        }
    }
}

শূন্য ggml_compute_forward_win_part(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_win_part_f32(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_win_unpart

স্থির শূন্য ggml_compute_forward_win_unpart_f32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {
    GGML_UNUSED(params);

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)

    ধ্রুব int32_t w = ((ধ্রুব int32_t *)(dst->op_params))[0];

    // padding
    ধ্রুব পূর্ণাঙ্ক px = (w - ne1%w)%w;
    //const int py = (w - ne2%w)%w;

    ধ্রুব পূর্ণাঙ্ক npx = (px + ne1)/w;
    //const int npy = (py + ne2)/w;

    প্রতিজ্ঞা(ne0 == ne00);

    // TODO: optimize / multi-thread
    প্রতি (int64_t i2 = 0; i2 < ne2; ++i2) {
        প্রতি (int64_t i1 = 0; i1 < ne1; ++i1) {
            প্রতি (int64_t i0 = 0; i0 < ne0; ++i0) {
                ধ্রুব পূর্ণাঙ্ক ip2 = i2/w;
                ধ্রুব পূর্ণাঙ্ক ip1 = i1/w;

                ধ্রুব int64_t i02 = i2%w;
                ধ্রুব int64_t i01 = i1%w;
                ধ্রুব int64_t i00 = i0;

                ধ্রুব int64_t i = (ip2*npx + ip1)*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00 + i00;
                ধ্রুব int64_t j =                                  i2*ne1*ne0    + i1*ne0   + i0;

                ((ভিন্ন *) dst->data)[j] = ((ভিন্ন *) src0->data)[i];
            }
        }
    }
}

শূন্য ggml_compute_forward_win_unpart(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_win_unpart_f32(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

//ggml_compute_forward_unary

শূন্য ggml_compute_forward_unary(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_unary_op op = ggml_get_unary_op(dst);

    বিকল্প (op) {
        পক্ষ GGML_UNARY_OP_ABS:
            {
                ggml_compute_forward_abs(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_UNARY_OP_SGN:
            {
                ggml_compute_forward_sgn(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_UNARY_OP_NEG:
            {
                ggml_compute_forward_neg(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_UNARY_OP_STEP:
            {
                ggml_compute_forward_step(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_UNARY_OP_TANH:
            {
                ggml_compute_forward_tanh(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_UNARY_OP_ELU:
            {
                ggml_compute_forward_elu(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_UNARY_OP_RELU:
            {
                ggml_compute_forward_relu(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_UNARY_OP_SIGMOID:
            {
                ggml_compute_forward_sigmoid(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_UNARY_OP_GELU:
            {
                ggml_compute_forward_gelu(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_UNARY_OP_GELU_ERF:
            {
                ggml_compute_forward_gelu_erf(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_UNARY_OP_GELU_QUICK:
            {
                ggml_compute_forward_gelu_quick(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_UNARY_OP_SILU:
            {
                ggml_compute_forward_silu(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_UNARY_OP_HARDSWISH:
            {
                ggml_compute_forward_hardswish(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_UNARY_OP_HARDSIGMOID:
            {
                ggml_compute_forward_hardsigmoid(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_UNARY_OP_EXP:
            {
                ggml_compute_forward_exp(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_UNARY_OP_FLOOR:
            {
                ggml_compute_forward_floor(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_UNARY_OP_CEIL:
            {
                ggml_compute_forward_ceil(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_UNARY_OP_ROUND:
            {
                ggml_compute_forward_round(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_UNARY_OP_TRUNC:
            {
                ggml_compute_forward_trunc(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_UNARY_OP_XIELU:
            {
                ggml_compute_forward_xielu(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_UNARY_OP_EXPM1:
            {
                ggml_compute_forward_expm1(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_UNARY_OP_SOFTPLUS:
            {
                ggml_compute_forward_softplus(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

//ggml_compute_forward_glu

শূন্য ggml_compute_forward_glu(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_glu_op op = ggml_get_glu_op(dst);

    বিকল্প (op) {
        পক্ষ GGML_GLU_OP_REGLU:
            {
                ggml_compute_forward_reglu(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_GLU_OP_GEGLU:
            {
                ggml_compute_forward_geglu(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_GLU_OP_SWIGLU:
            {
                ggml_compute_forward_swiglu(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_GLU_OP_SWIGLU_OAI:
            {
                ggml_compute_forward_swiglu_oai(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_GLU_OP_GEGLU_ERF:
            {
                ggml_compute_forward_geglu_erf(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_GLU_OP_GEGLU_QUICK:
            {
                ggml_compute_forward_geglu_quick(params, dst);
            } ভঙ্গ;
        পক্ষ GGML_GLU_OP_SWIGLU_CLAMP:
            {
                ggml_compute_forward_swiglu_clamp(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_get_rel_pos

স্থির শূন্য ggml_compute_forward_get_rel_pos_f16(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {
    GGML_UNUSED(params);

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L292-L322

    GGML_TENSOR_UNARY_OP_LOCALS

    ধ্রুব int64_t w = ne1;

    ggml_fp16_t * src0_data = (ggml_fp16_t *) src0->data;
    ggml_fp16_t * dst_data  = (ggml_fp16_t *) dst->data;

    প্রতি (int64_t i2 = 0; i2 < ne2; ++i2) {
        প্রতি (int64_t i1 = 0; i1 < ne1; ++i1) {
            ধ্রুব int64_t pos = (w - i1 - 1) + i2;
            প্রতি (int64_t i0 = 0; i0 < ne0; ++i0) {
                dst_data[i2*ne1*ne0 + i1*ne0 + i0] = src0_data[pos*ne00 + i0];
            }
        }
    }
}

শূন্য ggml_compute_forward_get_rel_pos(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F16:
        পক্ষ GGML_TYPE_BF16:
            {
                ggml_compute_forward_get_rel_pos_f16(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_add_rel_pos

স্থির শূন্য ggml_compute_forward_add_rel_pos_f32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];
    ধ্রুব ggml_tensor * src2 = dst->src[2];

    ধ্রুব তার্কিক inplace = (তার্কিক) ((int32_t *) dst->op_params)[0];
    যদি (!inplace) {
        যদি (params->ith == 0) {
            memcpy((বর্ণ *) dst->data, (বর্ণ *) src0->data, ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }
    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L357-L359

    ভিন্ন * src1_data = (ভিন্ন *) src1->data;
    ভিন্ন * src2_data = (ভিন্ন *) src2->data;
    ভিন্ন * dst_data  = (ভিন্ন *) dst->data;

    ধ্রুব int64_t ne10 = src1->ne[0];
    ধ্রুব int64_t ne11 = src1->ne[1];
    ধ্রুব int64_t ne12 = src1->ne[2];
    ধ্রুব int64_t ne13 = src1->ne[3];

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    // total patches in dst
    ধ্রুব পূর্ণাঙ্ক np = ne13;

    // patches per thread
    ধ্রুব পূর্ণাঙ্ক dp = (np + nth - 1)/nth;

    // patch range for this thread
    ধ্রুব পূর্ণাঙ্ক ip0 = dp*ith;
    ধ্রুব পূর্ণাঙ্ক ip1 = MIN(ip0 + dp, np);

    প্রতি (int64_t i13 = ip0; i13 < ip1; ++i13) {
        প্রতি (int64_t i12 = 0; i12 < ne12; ++i12) {
            প্রতি (int64_t i11 = 0; i11 < ne11; ++i11) {
                ধ্রুব int64_t jp1 = i13*ne12*ne11*ne10 + i12*ne11*ne10 + i11*ne10;
                প্রতি (int64_t i10 = 0; i10 < ne10; ++i10) {
                    ধ্রুব int64_t jp0  = jp1 + i10;
                    ধ্রুব ভিন্ন src1_e = src1_data[jp0];
                    ধ্রুব ভিন্ন src2_e = src2_data[jp0];

                    ধ্রুব int64_t jdh = jp0 * ne10;
                    ধ্রুব int64_t jdw = jdh - (ne10 - 1) * i10;

                    প্রতি (int64_t j = 0; j < ne10; ++j) {
                        dst_data[jdh + j     ] += src2_e;
                        dst_data[jdw + j*ne10] += src1_e;
                    }
                }
            }
        }
    }
}

শূন্য ggml_compute_forward_add_rel_pos(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_add_rel_pos_f32(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_rwkv_wkv6

স্থির শূন্য ggml_compute_forward_rwkv_wkv6_f32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {
    ধ্রুব int64_t T = dst->src[1]->ne[2];
    ধ্রুব int64_t C = dst->ne[0];
    ধ্রুব int64_t HEADS = dst->src[1]->ne[1];
    ধ্রুব int64_t n_seqs = dst->src[5]->ne[1];
    ধ্রুব int64_t head_size = C / HEADS;

    ভিন্ন * dst_data = (ভিন্ন *) dst->data;
    ভিন্ন * state = ((ভিন্ন *) dst->data) + C * T;

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক h_start =  (HEADS * (ith    )) / nth;
    ধ্রুব পূর্ণাঙ্ক h_end   = ((HEADS * (ith + 1)) / nth < HEADS) ?
                         (HEADS * (ith + 1)) / nth : HEADS;

    ভিন্ন * k =          (ভিন্ন *) dst->src[0]->data;
    ভিন্ন * v =          (ভিন্ন *) dst->src[1]->data;
    ভিন্ন * r =          (ভিন্ন *) dst->src[2]->data;
    ভিন্ন * time_faaaa = (ভিন্ন *) dst->src[3]->data;
    ভিন্ন * time_decay = (ভিন্ন *) dst->src[4]->data;

    size_t t_stride = HEADS * head_size; // Same to C

    size_t h_stride = C / HEADS;
    GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
    size_t h_stride_2d = head_size * head_size;

    যদি (ith == 0) {
        memset(dst_data, 0, T * C * পরিমাণ(ভিন্ন));
    }
    ggml_barrier(params->threadpool);


    #যদি defined(__AVX__) && !defined(__AVX512F__)
        #define GGML_F32X GGML_F32x8
        #define GGML_F32X_SET1 GGML_F32x8_SET1
        #define GGML_F32X_LOAD GGML_F32x8_LOAD
        #define GGML_F32X_STORE GGML_F32x8_STORE
        #define GGML_F32X_MUL GGML_F32x8_MUL
        #define GGML_F32X_FMA GGML_F32x8_FMA
        #define WKV_VECTOR_SIZE 8
    #নোচেদ্যদি defined(__AVX512F__)
        #define GGML_F32X GGML_F32x16
        #define GGML_F32X_SET1 GGML_F32x16_SET1
        #define GGML_F32X_LOAD GGML_F32x16_LOAD
        #define GGML_F32X_STORE GGML_F32x16_STORE
        #define GGML_F32X_MUL GGML_F32x16_MUL
        #define GGML_F32X_FMA GGML_F32x16_FMA
        #define WKV_VECTOR_SIZE 16
    #নোচেদ্যদি defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
        #define GGML_F32X GGML_F32xt
        #define GGML_F32X_SET1 GGML_F32xt_SET1
        #define GGML_F32X_LOAD GGML_F32xt_LOAD
        #define GGML_F32X_STORE GGML_F32xt_STORE
        #define GGML_F32X_MUL GGML_F32xt_MUL
        #define GGML_F32X_FMA GGML_F32xt_FMA
        #define WKV_VECTOR_SIZE 8
    #নোচেদ্যদি defined(__ARM_NEON) && defined(__aarch64__)
        #define GGML_F32X GGML_F32x4
        #define GGML_F32X_SET1 GGML_F32x4_SET1
        #define GGML_F32X_LOAD GGML_F32x4_LOAD
        #define GGML_F32X_STORE GGML_F32x4_STORE
        #define GGML_F32X_MUL GGML_F32x4_MUL
        #define GGML_F32X_FMA GGML_F32x4_FMA
        #define WKV_VECTOR_SIZE 4
    #endif

    #ifdef WKV_VECTOR_SIZE
        পূর্ণাঙ্ক wkv_vector_size;
        #যদি defined(__ARM_FEATURE_SVE)
            wkv_vector_size = svcntw();
        #অন্যথা
            wkv_vector_size = WKV_VECTOR_SIZE;
        #endif
        ধ্রুব int64_t vec_গুনতি = head_size / wkv_vector_size;

        প্রতি (int64_t t = 0; t < T; t++) {
            size_t t_offset = t * t_stride;
            size_t state_offset = head_size * C * (t / (T / n_seqs));
            ভিন্ন * state_cur = state + state_offset;
            ভিন্ন * state_prev = t % (T / n_seqs) ? state_cur : (ভিন্ন*)dst->src[5]->data + state_offset;

            প্রতি (int64_t h = h_start; h < h_end; h++) {
                size_t h_offset = h * h_stride;
                size_t t_h_offset = t_offset + h_offset;
                size_t h_2d_offset = h * h_stride_2d;

                প্রতি (int64_t i = 0; i < head_size; i++) {
                    size_t t_h_i_offset = t_h_offset + i;
                    size_t h_i_offset = h_offset + i;
                    size_t h_2d_i_offset = h_2d_offset + i * h_stride;

                    ভিন্ন k_val = k[t_h_i_offset];
                    ভিন্ন r_val = r[t_h_i_offset];
                    ভিন্ন time_faaaa_val = time_faaaa[h_i_offset];
                    ভিন্ন time_decay_val = time_decay[t_h_i_offset];

                    // Broadcast scalar values to vectors
                    GGML_F32X k_vec = GGML_F32X_SET1(k_val);
                    GGML_F32X r_vec = GGML_F32X_SET1(r_val);
                    GGML_F32X time_faaaa_vec = GGML_F32X_SET1(time_faaaa_val);
                    GGML_F32X time_decay_vec = GGML_F32X_SET1(time_decay_val);

                    প্রতি (int64_t j = 0; j < vec_গুনতি; j++) {
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
                    প্রতি (int64_t j = vec_গুনতি * wkv_vector_size; j < head_size; j++) {
                        size_t t_h_j_offset = t_h_offset + j;
                        size_t h_2d_i_j_offset = h_2d_i_offset + j;
                        ভিন্ন v_val = v[t_h_j_offset];
                        ভিন্ন kv_val = v_val * k_val;
                        ভিন্ন prev_state_val = state_prev[h_2d_i_j_offset];
                        ভিন্ন temp_val = kv_val * time_faaaa_val + prev_state_val;
                        dst_data[t_h_j_offset] += temp_val * r_val;
                        state_cur[h_2d_i_j_offset] = prev_state_val * time_decay_val + kv_val;
                    }
                }
            }
        }

    #অন্যথা
        // basically fused operations:
        // dst = r @ (time_faaaa * (k @ v) + state),
        // state = time_decay * state + (k @ v),
        // recursive through each token
        প্রতি (int64_t t = 0; t < T; t++) {
            size_t t_offset = t * t_stride;
            size_t state_offset = head_size * C * (t / (T / n_seqs));
            ভিন্ন * state_cur = state + state_offset;
            ভিন্ন * state_prev = t % (T / n_seqs) ? state_cur : (ভিন্ন*)dst->src[5]->data + state_offset;

            প্রতি (int64_t h = h_start; h < h_end; h++) {
                size_t h_offset = h * h_stride;
                size_t t_h_offset = t_offset + h_offset;
                size_t h_2d_offset = h * h_stride_2d;

                প্রতি (int64_t i = 0; i < head_size; i++) {
                    size_t t_h_i_offset = t_h_offset + i;
                    size_t h_i_offset = h_offset + i;
                    size_t h_2d_i_offset = h_2d_offset + i * h_stride;

                    ভিন্ন k_val = k[t_h_i_offset];
                    ভিন্ন r_val = r[t_h_i_offset];
                    ভিন্ন time_faaaa_val = time_faaaa[h_i_offset];
                    // RWKV v6: different time_decay for each token.
                    ভিন্ন time_decay_val = time_decay[t_h_i_offset];

                    প্রতি (int64_t j = 0; j < head_size; j++) {
                        size_t t_h_j_offset = t_h_offset + j;
                        size_t h_2d_i_j_offset = h_2d_i_offset + j;

                        ভিন্ন v_val = v[t_h_j_offset];
                        ভিন্ন kv_val = v_val * k_val;
                        ভিন্ন prev_state_val = state_prev[h_2d_i_j_offset];
                        ভিন্ন temp_val = kv_val * time_faaaa_val + prev_state_val;
                        dst_data[t_h_j_offset] += temp_val * r_val;
                        state_cur[h_2d_i_j_offset] = prev_state_val * time_decay_val + kv_val;
                    }
                }
            }
        }
    #endif
}


শূন্য ggml_compute_forward_rwkv_wkv6(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_rwkv_wkv6_f32(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_gla

স্থির শূন্য ggml_compute_forward_gla_f32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {
    ধ্রুব int64_t T = dst->src[1]->ne[2];
    ধ্রুব int64_t C = dst->ne[0];
    ধ্রুব int64_t HEADS = dst->src[1]->ne[1];
    ধ্রুব int64_t n_seqs = dst->src[4]->ne[1];
    ধ্রুব int64_t head_size = C / HEADS;
    ধ্রুব ভিন্ন scale = ggml_get_op_params_f32(dst, 0);

    ভিন্ন * dst_data = (ভিন্ন *) dst->data;
    ভিন্ন * state = ((ভিন্ন *) dst->data) + C * T;

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক h_start =  (HEADS * (ith    )) / nth;
    ধ্রুব পূর্ণাঙ্ক h_end   = ((HEADS * (ith + 1)) / nth < HEADS) ?
                         (HEADS * (ith + 1)) / nth : HEADS;

    ভিন্ন * k = (ভিন্ন *) dst->src[0]->data;
    ভিন্ন * v = (ভিন্ন *) dst->src[1]->data;
    ভিন্ন * q = (ভিন্ন *) dst->src[2]->data;
    ভিন্ন * g = (ভিন্ন *) dst->src[3]->data;

    size_t t_stride = HEADS * head_size; // Same to C

    size_t h_stride = C / HEADS;
    GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
    size_t h_stride_2d = head_size * head_size;

    যদি (ith == 0) {
        memset(dst_data, 0, T * C * পরিমাণ(ভিন্ন));
    }
    ggml_barrier(params->threadpool);


    #যদি defined(__AVX__) && !defined(__AVX512F__)
        #define GGML_F32X GGML_F32x8
        #define GGML_F32X_SET1 GGML_F32x8_SET1
        #define GGML_F32X_LOAD GGML_F32x8_LOAD
        #define GGML_F32X_STORE GGML_F32x8_STORE
        #define GGML_F32X_MUL GGML_F32x8_MUL
        #define GGML_F32X_FMA GGML_F32x8_FMA
        #define GLA_VECTOR_SIZE 8
    #নোচেদ্যদি defined(__AVX512F__)
        #define GGML_F32X GGML_F32x16
        #define GGML_F32X_SET1 GGML_F32x16_SET1
        #define GGML_F32X_LOAD GGML_F32x16_LOAD
        #define GGML_F32X_STORE GGML_F32x16_STORE
        #define GGML_F32X_MUL GGML_F32x16_MUL
        #define GGML_F32X_FMA GGML_F32x16_FMA
        #define GLA_VECTOR_SIZE 16
    #নোচেদ্যদি defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
        #define GGML_F32X GGML_F32xt
        #define GGML_F32X_SET1 GGML_F32xt_SET1
        #define GGML_F32X_LOAD GGML_F32xt_LOAD
        #define GGML_F32X_STORE GGML_F32xt_STORE
        #define GGML_F32X_MUL GGML_F32xt_MUL
        #define GGML_F32X_FMA GGML_F32xt_FMA
        #define GLA_VECTOR_SIZE 8
    #নোচেদ্যদি defined(__ARM_NEON) && defined(__aarch64__)
        #define GGML_F32X GGML_F32x4
        #define GGML_F32X_SET1 GGML_F32x4_SET1
        #define GGML_F32X_LOAD GGML_F32x4_LOAD
        #define GGML_F32X_STORE GGML_F32x4_STORE
        #define GGML_F32X_MUL GGML_F32x4_MUL
        #define GGML_F32X_FMA GGML_F32x4_FMA
        #define GLA_VECTOR_SIZE 4
    #endif

    #ifdef GLA_VECTOR_SIZE
        পূর্ণাঙ্ক gla_vector_size;
        #যদি defined(__ARM_FEATURE_SVE)
            gla_vector_size = svcntw();
        #অন্যথা
            gla_vector_size = GLA_VECTOR_SIZE;
        #endif
        ধ্রুব int64_t vec_গুনতি = head_size / gla_vector_size;

        প্রতি (int64_t t = 0; t < T; t++) {
            size_t t_offset = t * t_stride;
            size_t state_offset = head_size * C * (t / (T / n_seqs));
            ভিন্ন * state_cur = state + state_offset;
            ভিন্ন * state_prev = t % (T / n_seqs) ? state_cur : (ভিন্ন*)dst->src[4]->data + state_offset;

            প্রতি (int64_t h = h_start; h < h_end; h++) {
                size_t h_offset = h * h_stride;
                size_t t_h_offset = t_offset + h_offset;
                size_t h_2d_offset = h * h_stride_2d;

                প্রতি (int64_t i = 0; i < head_size; i++) {
                    size_t t_h_i_offset = t_h_offset + i;
                    size_t h_2d_i_offset = h_2d_offset + i * h_stride;

                    ভিন্ন k_val = k[t_h_i_offset];
                    ভিন্ন q_val = q[t_h_i_offset] * scale;
                    ভিন্ন g_val = g[t_h_i_offset];

                    // Broadcast scalar values to vectors
                    GGML_F32X k_vec = GGML_F32X_SET1(k_val);
                    GGML_F32X q_vec = GGML_F32X_SET1(q_val);
                    GGML_F32X g_vec = GGML_F32X_SET1(g_val);

                    প্রতি (int64_t j = 0; j < vec_গুনতি; j++) {
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
                    প্রতি (int64_t j = vec_গুনতি * gla_vector_size; j < head_size; j++) {
                        size_t t_h_j_offset = t_h_offset + j;
                        size_t h_2d_i_j_offset = h_2d_i_offset + j;
                        ভিন্ন v_val = v[t_h_j_offset];
                        ভিন্ন kv_val = v_val * k_val;
                        ভিন্ন prev_state_val = state_prev[h_2d_i_j_offset];
                        ভিন্ন temp_val = kv_val + prev_state_val * g_val;
                        dst_data[t_h_j_offset] += temp_val * q_val;
                        state_cur[h_2d_i_j_offset] = temp_val;
                    }
                }
            }
        }

    #অন্যথা
        প্রতি (int64_t t = 0; t < T; t++) {
            size_t t_offset = t * t_stride;
            size_t state_offset = head_size * C * (t / (T / n_seqs));
            ভিন্ন * state_cur = state + state_offset;
            ভিন্ন * state_prev = t % (T / n_seqs) ? state_cur : (ভিন্ন*)dst->src[4]->data + state_offset;

            প্রতি (int64_t h = h_start; h < h_end; h++) {
                size_t h_offset = h * h_stride;
                size_t t_h_offset = t_offset + h_offset;
                size_t h_2d_offset = h * h_stride_2d;

                প্রতি (int64_t i = 0; i < head_size; i++) {
                    size_t t_h_i_offset = t_h_offset + i;
                    size_t h_2d_i_offset = h_2d_offset + i * h_stride;

                    ভিন্ন k_val = k[t_h_i_offset];
                    ভিন্ন q_val = q[t_h_i_offset] * scale;
                    ভিন্ন g_val = g[t_h_i_offset];

                    প্রতি (int64_t j = 0; j < head_size; j++) {
                        size_t t_h_j_offset = t_h_offset + j;
                        size_t h_2d_i_j_offset = h_2d_i_offset + j;

                        ভিন্ন v_val = v[t_h_j_offset];
                        ভিন্ন kv_val = v_val * k_val;
                        ভিন্ন prev_state_val = state_prev[h_2d_i_j_offset];
                        ভিন্ন temp_val = prev_state_val * g_val + kv_val;
                        dst_data[t_h_j_offset] += temp_val * q_val;
                        state_cur[h_2d_i_j_offset] = temp_val;
                    }
                }
            }
        }
    #endif
}


শূন্য ggml_compute_forward_gla(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_gla_f32(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

স্থির শূন্য ggml_compute_forward_solve_tri_f32(ধ্রুব সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst) {
    ধ্রুব সংরচনা ggml_tensor * src0 = dst->src[0];  // A (lower triangular)
    ধ্রুব সংরচনা ggml_tensor * src1 = dst->src[1];  // B (RHS)

    GGML_TENSOR_দ্বিমিক_OP_LOCALS;

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(ne00 == ne01); // A must be square
    GGML_ASSERT(ne0  == ne10); // solution cols == B cols
    GGML_ASSERT(ne1  == ne11); // solution rows == B rows

    GGML_ASSERT(ne02 == ne12 && ne12 == ne2);
    GGML_ASSERT(ne03 == ne13 && ne13 == ne3);

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব int64_t k = ne10;   // number of RHS columns
    ধ্রুব int64_t n = ne11;   // A is n×n
    ধ্রুব int64_t nr = ne02 * ne03 * k; // we're parallelizing on columns here, so seq x token x column will be the unit

    // chunks per thread
    ধ্রুব int64_t dr = (nr + nth - 1)/nth;

    // chunk range for this thread
    ধ্রুব int64_t ir0 = dr*ith;
    ধ্রুব int64_t ir1 = MIN(ir0 + dr, nr);

    ধ্রুব ভিন্ন * A = (ধ্রুব ভিন্ন *) src0->data;  // [n, n, B1, B2]
    ধ্রুব ভিন্ন * B = (ধ্রুব ভিন্ন *) src1->data;  // [n, k, B1, B2]
          ভিন্ন * X = (      ভিন্ন *) dst->data;   // [n, k, B1, B2]

    প্রতি (int64_t ir = ir0; ir < ir1; ++ir) {
        ধ্রুব int64_t i03 = ir/(ne02*k);
        ধ্রুব int64_t i02 = (ir - i03*ne02*k)/k;
        ধ্রুব int64_t i01 = (ir - i03*ne02*k - i02*k);

        ধ্রুব ভিন্ন * A_batch = A + i02 * nb02 / পরিমাণ(ভিন্ন) + i03 * nb03 / পরিমাণ(ভিন্ন);
        ধ্রুব ভিন্ন * B_batch = B + i02 * nb12 / পরিমাণ(ভিন্ন) + i03 * nb13 / পরিমাণ(ভিন্ন);

        ভিন্ন * X_batch = X + i02 * nb2 / পরিমাণ(ভিন্ন) + i03 * nb3 / পরিমাণ(ভিন্ন);

        প্রতি (int64_t i00 = 0; i00 < n; ++i00) {
            ভিন্ন sum = 0.0f;
            প্রতি (int64_t t = 0; t < i00; ++t) {
                sum += A_batch[i00 * n + t] * X_batch[t * k + i01];
            }

            ধ্রুব ভিন্ন diag = A_batch[i00 * n + i00];
            প্রতিজ্ঞা(diag != 0.0f && "Zero diagonal in triangular matrix");

            X_batch[i00 * k + i01] = (B_batch[i00 * k + i01] - sum) / diag;
        }
    }
}

শূন্য ggml_compute_forward_solve_tri(ধ্রুব সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst) {
    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];

    যদি (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        ggml_compute_forward_solve_tri_f32(params, dst);
    } অন্যথা {
        GGML_ABORT("fatal error");
    }
}

// ggml_compute_forward_gated_delta_net
স্থির শূন্য ggml_compute_forward_gated_delta_net_one_chunk(
    ধ্রুব ggml_compute_params * params,
    ggml_tensor * dst,
    int64_t ir0,
    int64_t ir1) {

    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    ধ্রুব int64_t S_v      = src_v->ne[0];
    ধ্রুব int64_t H        = src_v->ne[1];
    ধ্রুব int64_t n_tokens = src_v->ne[2];
    ধ্রুব int64_t n_seqs   = src_v->ne[3];

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

    ধ্রুব তার্কিক kda = (neg0 == S_v);

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    ধ্রুব int64_t K = ggml_get_op_params_i32(dst, 0);
    GGML_ASSERT(K >= 1);
    // per-seq stride in floats (seq s starts at state + s * seq_stride)
    ধ্রুব int64_t state_seq_stride = src_state->nb[3] / পরিমাণ(ভিন্ন);

    ধ্রুব int64_t per_thread = S_v + (K > 1 ? S_v * S_v : 0);
    ধ্রুব পূর্ণাঙ্ক ith = params->ith;

    ভিন্ন * delta       = (ভিন্ন *)params->wdata + ith * per_thread + CACHE_সারি_SIZE_F32;
    ভিন্ন * state_work  = K > 1 ? (delta + S_v) : nullptr;

    // output layout: [attn_scores | new_states]
    // attn_scores: S_v * H * n_tokens * n_seqs    floats
    // new_states:  S_v * S_v * H * n_seqs * K     floats  (K snapshot slots; last min(n_tokens, K))
    ধ্রুব int64_t attn_score_elems    = S_v * H * n_tokens * n_seqs;
    ধ্রুব int64_t state_size_per_snap = S_v * S_v * H * n_seqs;
    ভিন্ন * attn_out_base  = (ভিন্ন *)dst->data;
    ভিন্ন * state_out_base = (ভিন্ন *)dst->data + attn_score_elems;

    // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
    // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.

    ধ্রুব ভিন্ন * state_in_base = (ধ্রুব ভিন্ন *)src_state->data;

  //const int64_t rq1 = nev1 / neq1;
  //const int64_t rk1 = nev1 / nek1;
    ধ্রুব int64_t rq3 = nev3 / neq3;
    ধ্রুব int64_t rk3 = nev3 / nek3;

    ধ্রুব ভিন্ন scale = 1.0f / sqrtf((ভিন্ন) S_v);

    প্রতি (int64_t ir = ir0; ir < ir1; ++ir) {
        ধ্রুব int64_t iv1 = ir % H; // head_index
        ধ্রুব int64_t iv3 = ir / H; // sequence

        ধ্রুব int64_t iq1 = iv1 % neq1;
        ধ্রুব int64_t ik1 = iv1 % nek1;

        ধ্রুব int64_t iq3 = iv3 / rq3;
        ধ্রুব int64_t ik3 = iv3 / rk3;

        // For K=1, write directly to the single output slot to avoid an extra memcpy at the end.
        // For K>1, work in scratch and copy out per-token when the slot is in range.
        ভিন্ন * s_out = (K > 1)
            ? state_work
            : state_out_base + (iv3 * H + iv1) * S_v * S_v;

        // copy input state into the working buffer and operate in-place
        // state layout [S_v, S_v, H, n_seqs]: seq iv3 starts at iv3 * state_seq_stride.
        ধ্রুব ভিন্ন * s_in = state_in_base + iv3 * state_seq_stride + iv1 * S_v * S_v;
        memcpy(s_out, s_in, S_v * S_v * পরিমাণ(ভিন্ন));

        // attn output pointer for first token of this (head, seq)
        ভিন্ন * attn_data = attn_out_base + (iv3 * n_tokens * H + iv1) * S_v;

        প্রতি (int64_t t = 0; t < n_tokens; t++) {
            ধ্রুব ভিন্ন * q_d = (ধ্রুব ভিন্ন *)((ধ্রুব বর্ণ *)src_q->data + iq3 * nbq3 + t * nbq2 + iq1 * nbq1);
            ধ্রুব ভিন্ন * k_d = (ধ্রুব ভিন্ন *)((ধ্রুব বর্ণ *)src_k->data + ik3 * nbk3 + t * nbk2 + ik1 * nbk1);
            ধ্রুব ভিন্ন * v_d = (ধ্রুব ভিন্ন *)((ধ্রুব বর্ণ *)src_v->data + iv3 * nbv3 + t * nbv2 + iv1 * nbv1);

            ধ্রুব ভিন্ন beta_val = *(ধ্রুব ভিন্ন *)((ধ্রুব বর্ণ *)src_beta->data + iv3 * nbb3 + t * nbb2 + iv1 * nbb1);
            ধ্রুব ভিন্ন * g_d    =  (ধ্রুব ভিন্ন *)((ধ্রুব বর্ণ *)src_g->data    + iv3 * nbg3 + t * nbg2 + iv1 * nbg1);

            // state is stored transposed: s_out[j*S_v + i] = S[i][j]
            // so row j of s_out = column j of S (contiguous access)

            যদি (kda) {
                // precompute exp(g) into delta scratch (reused below)
                প্রতি (int64_t i = 0; i < S_v; ++i) {
                    delta[i] = expf(g_d[i]);
                }
                // S[i][:] *= exp(g[i]) => for each row j of M: M[j][i] *= exp(g[i])
                প্রতি (int64_t j = 0; j < S_v; ++j) {
                    ggml_vec_mul_f32(S_v, &s_out[j * S_v], &s_out[j * S_v], delta);
                }
            } অন্যথা {
                ggml_vec_scale_f32(S_v * S_v, s_out, expf(g_d[0]));
            }

            // delta[j] = sum_i S[i][j] * k[i] = dot(row j of M, k)
            প্রতি (int64_t j = 0; j < S_v; ++j) {
                ভিন্ন sum = 0.0f;
                ggml_vec_dot_f32(S_v, &sum, 0, &s_out[j * S_v], 0, k_d, 0, 1);
                delta[j] = (v_d[j] - sum) * beta_val;
            }

            // outer product: S[i][j] += k[i] * delta[j] => M[j][i] += delta[j] * k[i]
            প্রতি (int64_t j = 0; j < S_v; ++j) {
                ggml_vec_mad_f32(S_v, &s_out[j * S_v], k_d, delta[j]);
            }

            // attn_out[j] = sum_i S[i][j] * q[i] = dot(row j of M, q)
            প্রতি (int64_t j = 0; j < S_v; ++j) {
                ভিন্ন sum = 0.0f;
                ggml_vec_dot_f32(S_v, &sum, 0, &s_out[j * S_v], 0, q_d, 0, 1);
                attn_data[j] = sum * scale;
            }

            attn_data += S_v * H; // advance to next token

            যদি (K > 1) {
                ধ্রুব int64_t target_slot = n_tokens - 1 - t;
                যদি (target_slot >= 0 && target_slot < K) {
                    ভিন্ন * curr_state_o = state_out_base + target_slot * state_size_per_snap +
                                     (iv3 * H + iv1) * S_v * S_v;
                    memcpy(curr_state_o, s_out, S_v * S_v * পরিমাণ(ভিন্ন));
                }
            }
        }
    }
}


স্থির শূন্য ggml_compute_forward_gated_delta_net_f32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ggml_tensor * V = dst->src[2];
    int64_t nr = V->ne[1] * V->ne[3];

    // disable for NUMA
    ধ্রুব তার্কিক disable_chunking = ggml_is_numa();

    পূর্ণাঙ্ক nth = params->nth;
    পূর্ণাঙ্ক ith = params->ith;

    // 4x chunks per thread
    পূর্ণাঙ্ক nth_scaled = nth * 4;
    int64_t chunk_size = (nr + nth_scaled - 1) / nth_scaled;
    int64_t nchunk     = (nr + chunk_size - 1) / chunk_size;

    যদি (nth == 1 || nchunk < nth || disable_chunking) {
      nchunk = nth;
    }

    যদি (ith == 0) {
      ggml_threadpool_chunk_set(params->threadpool, nth);
    }

    ggml_barrier(params->threadpool);

    ধ্রুব int64_t dr = (nr + nchunk - 1) / nchunk;

    পূর্ণাঙ্ক current_chunk = ith;

    যাবত্ (current_chunk < nchunk) {
        ধ্রুব int64_t ir0 = dr * current_chunk;
        ধ্রুব int64_t ir1 = MIN(ir0 + dr, nr);

        ggml_compute_forward_gated_delta_net_one_chunk(params, dst, ir0, ir1);
        current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
    }
}

শূন্য ggml_compute_forward_gated_delta_net(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {
    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_gated_delta_net_f32(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}


// ggml_compute_forward_dsv4_hc_comb

স্থির শূন্য ggml_dsv4_hc_comb_norm_cols(ভিন্ন * comb, ভিন্ন eps) {
    constexpr int64_t hc = 4;

    প্রতি (int64_t idst = 0; idst < hc; ++idst) {
        ভিন্ন sum = eps;
        প্রতি (int64_t isrc = 0; isrc < hc; ++isrc) {
            sum += comb[idst + hc*isrc];
        }

        ধ্রুব ভিন্ন inv_sum = 1.0f / sum;
        প্রতি (int64_t isrc = 0; isrc < hc; ++isrc) {
            comb[idst + hc*isrc] *= inv_sum;
        }
    }
}

স্থির শূন্য ggml_dsv4_hc_comb_norm_rows(ভিন্ন * comb, ভিন্ন eps) {
    constexpr int64_t hc = 4;

    প্রতি (int64_t isrc = 0; isrc < hc; ++isrc) {
        ভিন্ন sum = eps;
        প্রতি (int64_t idst = 0; idst < hc; ++idst) {
            sum += comb[idst + hc*isrc];
        }

        ধ্রুব ভিন্ন inv_sum = 1.0f / sum;
        প্রতি (int64_t idst = 0; idst < hc; ++idst) {
            comb[idst + hc*isrc] *= inv_sum;
        }
    }
}

স্থির শূন্য ggml_compute_forward_dsv4_hc_comb_f32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {
    ধ্রুব ggml_tensor * mixes = dst->src[0];
    ধ্রুব ggml_tensor * scale = dst->src[1];
    ধ্রুব ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    constexpr int64_t hc = 4;
    constexpr int64_t comb_offset = 2*hc;
    constexpr int64_t hc_mix_dim = (2 + hc)*hc;

    ধ্রুব int64_t n_tokens = mixes->ne[1];

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

    ধ্রুব ভিন্ন eps = ggml_get_op_params_f32(dst, 0);
    ধ্রুব int32_t n_iter = ggml_get_op_params_i32(dst, 1);
    GGML_ASSERT(n_iter > 0);

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব int64_t dr  = (n_tokens + nth - 1) / nth;
    ধ্রুব int64_t it0 = dr * ith;
    ধ্রুব int64_t it1 = MIN(it0 + dr, n_tokens);

    ধ্রুব ভিন্ন scale_comb = *(ধ্রুব ভিন্ন *) ((ধ্রুব বর্ণ *) scale->data + 2*nbs0);

    প্রতি (int64_t it = it0; it < it1; ++it) {
        ভিন্ন comb[hc*hc];

        প্রতি (int64_t isrc = 0; isrc < hc; ++isrc) {
            ভিন্ন max = -INFINITY;
            প্রতি (int64_t idst = 0; idst < hc; ++idst) {
                ধ্রুব int64_t idx = idst + hc*isrc;
                ধ্রুব ভিন্ন xv = *(ধ্রুব ভিন্ন *) ((ধ্রুব বর্ণ *) mixes->data + (comb_offset + idx)*nbm0 + it*nbm1);
                ধ্রুব ভিন্ন bv = *(ধ্রুব ভিন্ন *) ((ধ্রুব বর্ণ *) base->data  + (comb_offset + idx)*nbb0);
                ধ্রুব ভিন্ন v = xv * scale_comb + bv;
                comb[idx] = v;
                max = MAX(max, v);
            }

            ভিন্ন sum = 0.0f;
            প্রতি (int64_t idst = 0; idst < hc; ++idst) {
                ধ্রুব int64_t idx = idst + hc*isrc;
                ধ্রুব ভিন্ন v = expf(comb[idx] - max);
                comb[idx] = v;
                sum += v;
            }

            ধ্রুব ভিন্ন inv_sum = 1.0f / sum;
            প্রতি (int64_t idst = 0; idst < hc; ++idst) {
                ধ্রুব int64_t idx = idst + hc*isrc;
                comb[idx] = comb[idx] * inv_sum + eps;
            }
        }

        ggml_dsv4_hc_comb_norm_cols(comb, eps);
        প্রতি (int32_t i = 1; i < n_iter; ++i) {
            ggml_dsv4_hc_comb_norm_rows(comb, eps);
            ggml_dsv4_hc_comb_norm_cols(comb, eps);
        }

        প্রতি (int64_t isrc = 0; isrc < hc; ++isrc) {
            প্রতি (int64_t idst = 0; idst < hc; ++idst) {
                ধ্রুব int64_t idx = idst + hc*isrc;
                *(ভিন্ন *) ((বর্ণ *) dst->data + idst*nbd0 + isrc*nbd1 + it*nbd2) = comb[idx];
            }
        }
    }
}

শূন্য ggml_compute_forward_dsv4_hc_comb(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {
    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_dsv4_hc_comb_f32(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_dsv4_hc_pre

স্থির শূন্য ggml_compute_forward_dsv4_hc_pre_f32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {
    ধ্রুব ggml_tensor * x       = dst->src[0];
    ধ্রুব ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ধ্রুব int64_t n_embd   = x->ne[0];
    ধ্রুব int64_t hc       = x->ne[1];
    ধ্রুব int64_t n_tokens = x->ne[2];

    ধ্রুব ভিন্ন scale = ggml_get_op_params_f32(dst, 0);
    ধ্রুব তার্কিক  gated = ggml_get_op_params_i32(dst, 1) != 0;

    GGML_ASSERT(dst->ne[0] == n_embd);
    GGML_ASSERT(dst->ne[1] == n_tokens);
    যদি (gated) {
        GGML_ASSERT(weights->ne[0] == n_embd);
        GGML_ASSERT(weights->ne[1] == hc);
        GGML_ASSERT(weights->ne[2] == n_tokens);
    } অন্যথা {
        GGML_ASSERT(weights->ne[0] == hc);
        GGML_ASSERT(weights->ne[1] == n_tokens);
    }

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব int64_t nr  = n_embd * n_tokens;
    ধ্রুব int64_t dr  = (nr + nth - 1) / nth;
    ধ্রুব int64_t ir0 = dr * ith;
    ধ্রুব int64_t ir1 = MIN(ir0 + dr, nr);

    প্রতি (int64_t ir = ir0; ir < ir1; ++ir) {
        ধ্রুব int64_t i0 = ir % n_embd;
        ধ্রুব int64_t it = ir / n_embd;

        ভিন্ন sum = 0.0f;
        প্রতি (int64_t ih = 0; ih < hc; ++ih) {
            ধ্রুব ভিন্ন xv = *(ধ্রুব ভিন্ন *) ((ধ্রুব বর্ণ *) x->data + i0*nbx0 + ih*nbx1 + it*nbx2);
            ভিন্ন wv;
            যদি (gated) {
                ধ্রুব ভিন্ন gv = *(ধ্রুব ভিন্ন *) ((ধ্রুব বর্ণ *) weights->data + i0*nbw0 + ih*nbw1 + it*nbw2);
                wv = 1.0f / (1.0f + expf(-gv));
            } অন্যথা {
                wv = *(ধ্রুব ভিন্ন *) ((ধ্রুব বর্ণ *) weights->data + ih*nbw0 + it*nbw1);
            }
            sum += xv * wv;
        }

        *(ভিন্ন *) ((বর্ণ *) dst->data + i0*nbd0 + it*nbd1) = scale * sum;
    }
}

শূন্য ggml_compute_forward_dsv4_hc_pre(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {
    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_dsv4_hc_pre_f32(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_dsv4_hc_post

স্থির শূন্য ggml_compute_forward_dsv4_hc_post_f32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {
    ধ্রুব ggml_tensor * x        = dst->src[0];
    ধ্রুব ggml_tensor * residual = dst->src[1];
    ধ্রুব ggml_tensor * post     = dst->src[2];
    ধ্রুব ggml_tensor * comb     = dst->src[3];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ধ্রুব int64_t n_embd   = x->ne[0];
    ধ্রুব int64_t n_tokens = x->ne[1];
    ধ্রুব int64_t hc       = residual->ne[1];

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
    যদি (comb) {
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

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব int64_t nr  = n_embd * hc * n_tokens;
    ধ্রুব int64_t dr  = (nr + nth - 1) / nth;
    ধ্রুব int64_t ir0 = dr * ith;
    ধ্রুব int64_t ir1 = MIN(ir0 + dr, nr);

    প্রতি (int64_t ir = ir0; ir < ir1; ++ir) {
        ধ্রুব int64_t i0     = ir % n_embd;
        ধ্রুব int64_t idst   = (ir / n_embd) % hc;
        ধ্রুব int64_t it     = ir / (n_embd * hc);

        ধ্রুব ভিন্ন xv = *(ধ্রুব ভিন্ন *) ((ধ্রুব বর্ণ *) x->data    + i0*nbx0 + it*nbx1);
        ধ্রুব ভিন্ন pv = *(ধ্রুব ভিন্ন *) ((ধ্রুব বর্ণ *) post->data + idst*nbp0 + it*nbp1);

        ভিন্ন sum = xv * pv;
        যদি (comb) {
            প্রতি (int64_t isrc = 0; isrc < hc; ++isrc) {
                ধ্রুব ভিন্ন rv = *(ধ্রুব ভিন্ন *) ((ধ্রুব বর্ণ *) residual->data + i0*nbr0 + isrc*nbr1 + it*nbr2);
                ধ্রুব ভিন্ন cv = *(ধ্রুব ভিন্ন *) ((ধ্রুব বর্ণ *) comb->data     + idst*nbc0 + isrc*nbc1 + it*nbc2);
                sum += rv * cv;
            }
        } অন্যথা {
            sum += *(ধ্রুব ভিন্ন *) ((ধ্রুব বর্ণ *) residual->data + i0*nbr0 + idst*nbr1 + it*nbr2);
        }

        *(ভিন্ন *) ((বর্ণ *) dst->data + i0*nbd0 + idst*nbd1 + it*nbd2) = sum;
    }
}

শূন্য ggml_compute_forward_dsv4_hc_post(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {
    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_dsv4_hc_post_f32(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_top_k_mask

শূন্য ggml_compute_forward_শীর্ষ_k_mask(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {
    ধ্রুব ggml_tensor * mask   = dst->src[0];
    ধ্রুব ggml_tensor * শীর্ষ_k  = dst->src[1];
    ধ্রুব ggml_tensor * score  = dst->src[2];
    ধ্রুব ggml_tensor * blocks = dst->src[3];
    ধ্রুব ggml_tensor * extra  = dst->src[4];

    GGML_ASSERT(dst->type == mask->type);
    GGML_ASSERT(dst->nb[0] == ggml_type_size(dst->type));

    ধ্রুব int64_t n_kv    = mask->ne[0];
    ধ্রুব int64_t ne1     = dst->ne[1];
    ধ্রুব int64_t ne2     = dst->ne[2];
    ধ্রুব int64_t n_rows  = ggml_nrows(dst);
    ধ্রুব int64_t n_শীর্ষ   = শীর্ষ_k->ne[0];
    ধ্রুব int64_t n_blk   = blocks ? blocks->ne[0] : 1;
    ধ্রুব int64_t n_pick  = blocks ? blocks->ne[1] : score ? score->ne[0] : n_kv;
    ধ্রুব int64_t n_extra = extra ? extra->ne[0] : 0;

    ধ্রুব size_t ts = ggml_type_size(dst->type);

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    প্রতি (int64_t ir = ith; ir < n_rows; ir += nth) {
        ধ্রুব int64_t i1 = ir % ne1;
        ধ্রুব int64_t i2 = (ir / ne1) % ne2;
        ধ্রুব int64_t i3 = ir / (ne1*ne2);

        ধ্রুব বর্ণ * mask_row = (ধ্রুব বর্ণ *) mask->data
            + (i1 % mask->ne[1])*mask->nb[1] + (i2 % mask->ne[2])*mask->nb[2] + (i3 % mask->ne[3])*mask->nb[3];
        বর্ণ * dst_row = (বর্ণ *) dst->data + i1*dst->nb[1] + i2*dst->nb[2] + i3*dst->nb[3];

        যদি (dst->type == GGML_TYPE_F16) {
            ধ্রুব ggml_fp16_t ninf = GGML_CPU_FP32_TO_FP16(-INFINITY);
            প্রতি (int64_t c = 0; c < n_kv; ++c) {
                ((ggml_fp16_t *) dst_row)[c] = ninf;
            }
        } অন্যথা {
            প্রতি (int64_t c = 0; c < n_kv; ++c) {
                ((ভিন্ন *) dst_row)[c] = -INFINITY;
            }
        }

        auto keep = [&](int32_t c) {
            যদি (c >= 0 && c < n_kv) {
                memcpy(dst_row + c*ts, mask_row + c*ts, ts);
            }
        };

        ধ্রুব বর্ণ * শীর্ষ_k_row = (ধ্রুব বর্ণ *) শীর্ষ_k->data + i1*শীর্ষ_k->nb[1] + i2*শীর্ষ_k->nb[2] + i3*শীর্ষ_k->nb[3];

        প্রতি (int64_t j = 0; j < n_শীর্ষ; ++j) {
            ধ্রুব int32_t p = *(ধ্রুব int32_t *) (শীর্ষ_k_row + j*শীর্ষ_k->nb[0]);
            যদি (p < 0 || p >= n_pick) {
                অনুবর্ত;
            }
            যদি (score) {
                ধ্রুব ভিন্ন s = *(ধ্রুব ভিন্ন *) ((ধ্রুব বর্ণ *) score->data
                    + p*score->nb[0] + i1*score->nb[1] + i2*score->nb[2] + i3*score->nb[3]);
                যদি (!(s > -INFINITY)) {
                    অনুবর্ত;
                }
            }
            যদি (!blocks) {
                keep(p);
                অনুবর্ত;
            }
            প্রতি (int64_t k = 0; k < n_blk; ++k) {
                keep(*(ধ্রুব int32_t *) ((ধ্রুব বর্ণ *) blocks->data + k*blocks->nb[0] + p*blocks->nb[1]));
            }
        }

        যদি (extra) {
            ধ্রুব বর্ণ * extra_row = (ধ্রুব বর্ণ *) extra->data + i1*extra->nb[1] + i2*extra->nb[2] + i3*extra->nb[3];
            প্রতি (int64_t j = 0; j < n_extra; ++j) {
                keep(*(ধ্রুব int32_t *) (extra_row + j*extra->nb[0]));
            }
        }
    }
}

// ggml_compute_forward_rwkv_wkv7

স্থির শূন্য ggml_compute_forward_rwkv_wkv7_f32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {
    ধ্রুব int64_t T = dst->src[1]->ne[2];
    ধ্রুব int64_t C = dst->ne[0];
    ধ্রুব int64_t HEADS = dst->src[1]->ne[1];
    ধ্রুব int64_t n_seqs = dst->src[6]->ne[1];
    ধ্রুব int64_t head_size = C / HEADS;

    ভিন্ন * dst_data = (ভিন্ন *) dst->data;
    ভিন্ন * state = ((ভিন্ন *) dst->data) + C * T;

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক h_start =  (HEADS * (ith    )) / nth;
    ধ্রুব পূর্ণাঙ্ক h_end   = ((HEADS * (ith + 1)) / nth < HEADS) ?
                         (HEADS * (ith + 1)) / nth : HEADS;

    ভিন্ন * r = (ভিন্ন *) dst->src[0]->data;
    ভিন্ন * w = (ভিন্ন *) dst->src[1]->data;
    ভিন্ন * k = (ভিন্ন *) dst->src[2]->data;
    ভিন্ন * v = (ভিন্ন *) dst->src[3]->data;
    ভিন্ন * a = (ভিন্ন *) dst->src[4]->data;
    ভিন্ন * b = (ভিন্ন *) dst->src[5]->data;

    int64_t t_stride = HEADS * head_size; // Same to C

    int64_t h_stride = C / HEADS;
    GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
    int64_t h_stride_2d = head_size * head_size;

    #যদি defined(GGML_SIMD)
        #যদি defined(__ARM_FEATURE_SVE) || defined(__riscv_v_intrinsic)
            // scalar Route to scalar implementation       //TODO: Write SVE code and RVV code
            প্রতি (int64_t t = 0; t < T; t++) {
                int64_t t_offset = t * t_stride;
                int64_t state_offset = head_size * C * (t / (T / n_seqs));
                ভিন্ন * state_cur = state + state_offset;
                ভিন্ন * state_prev = t % (T / n_seqs) ? state_cur : (ভিন্ন*)dst->src[6]->data + state_offset;

                প্রতি (int64_t h = h_start; h < h_end; h++) {
                    int64_t h_offset = h * h_stride;
                    int64_t t_h_offset = t_offset + h_offset;
                    int64_t h_2d_offset = h * h_stride_2d;

                    প্রতি (int64_t i = 0; i < head_size; i++) {
                        int64_t t_h_i_offset = t_h_offset + i;
                        int64_t h_2d_i_offset = h_2d_offset + i * h_stride;

                        ভিন্ন v_val = v[t_h_i_offset];

                        ভিন্ন sa = 0, result = 0;
                        প্রতি (int64_t j = 0; j < head_size; j++) {
                            sa += a[t_h_offset + j] * state_prev[h_2d_i_offset + j];
                        }

                        প্রতি (int64_t j = 0; j < head_size; j++) {
                            int64_t t_h_j_offset = t_h_offset + j;
                            int64_t h_2d_i_j_offset = h_2d_i_offset + j;

                            ভিন্ন r_val = r[t_h_j_offset];
                            ভিন্ন w_val = w[t_h_j_offset];
                            ভিন্ন k_val = k[t_h_j_offset];
                            ভিন্ন b_val = b[t_h_j_offset];
                            ভিন্ন kv_val = v_val * k_val;
                            ভিন্ন prev_state_val = state_prev[h_2d_i_j_offset];
                            state_cur[h_2d_i_j_offset] = prev_state_val * w_val + kv_val + sa * b_val;
                            result += state_cur[h_2d_i_j_offset] * r_val;
                        }
                        dst_data[t_h_i_offset] = result;
                    }
                }
            }
        #অন্যথা
            প্রতি (int64_t t = 0; t < T; t++) {
                int64_t t_offset = t * t_stride;
                int64_t state_offset = head_size * C * (t / (T / n_seqs));
                ভিন্ন * state_cur = state + state_offset;
                ভিন্ন * state_prev = t % (T / n_seqs) ? state_cur : (ভিন্ন*)dst->src[6]->data + state_offset;

                প্রতি (int64_t h = h_start; h < h_end; h++) {
                    int64_t h_offset = h * h_stride;
                    int64_t t_h_offset = t_offset + h_offset;
                    int64_t h_2d_offset = h * h_stride_2d;

                    প্রতি (int64_t ii = 0; ii < head_size; ii++) {
                        int64_t t_h_i_offset = t_h_offset + ii;
                        int64_t h_2d_i_offset = h_2d_offset + ii * h_stride;

                        GGML_F32_VEC v_vec = GGML_F32_VEC_SET1(v[t_h_i_offset]);

                        ভিন্ন sa = 0;
                        {
                            GGML_F32_VEC sum[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };
                            GGML_F32_VEC ax[GGML_F32_ARR];
                            GGML_F32_VEC ay[GGML_F32_ARR];
                            প্রতি (int64_t j = 0; j < head_size; j += GGML_F32_STEP) {
                                প্রতি (int64_t kk = 0; kk < GGML_F32_ARR; kk++) {
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
                        প্রতি (; j < head_size; j += GGML_F32_STEP) {
                            প্রতি (int64_t kk = 0; kk < GGML_F32_ARR; kk++) {
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
                        প্রতি (; j < head_size; j++) {
                            int64_t t_h_j_offset = t_h_offset + j;
                            int64_t h_2d_i_j_offset = h_2d_i_offset + j;

                            ভিন্ন r_val = r[t_h_j_offset];
                            ভিন্ন w_val = w[t_h_j_offset];
                            ভিন্ন k_val = k[t_h_j_offset];
                            ভিন্ন b_val = b[t_h_j_offset];
                            ভিন্ন kv_val = v[t_h_i_offset] * k_val;

                            ভিন্ন prev_state_val = state_prev[h_2d_i_j_offset];
                            state_cur[h_2d_i_j_offset] = prev_state_val * w_val + kv_val + sa * b_val;
                            dst_data[t_h_i_offset] += state_cur[h_2d_i_j_offset] * r_val;
                        }
                    }
                }
            }
        #endif
    #অন্যথা
        প্রতি (int64_t t = 0; t < T; t++) {
            int64_t t_offset = t * t_stride;
            int64_t state_offset = head_size * C * (t / (T / n_seqs));
            ভিন্ন * state_cur = state + state_offset;
            ভিন্ন * state_prev = t % (T / n_seqs) ? state_cur : (ভিন্ন*)dst->src[6]->data + state_offset;

            প্রতি (int64_t h = h_start; h < h_end; h++) {
                int64_t h_offset = h * h_stride;
                int64_t t_h_offset = t_offset + h_offset;
                int64_t h_2d_offset = h * h_stride_2d;

                প্রতি (int64_t i = 0; i < head_size; i++) {
                    int64_t t_h_i_offset = t_h_offset + i;
                    int64_t h_2d_i_offset = h_2d_offset + i * h_stride;

                    ভিন্ন v_val = v[t_h_i_offset];

                    ভিন্ন sa = 0, result = 0;
                    প্রতি (int64_t j = 0; j < head_size; j++) {
                        sa += a[t_h_offset + j] * state_prev[h_2d_i_offset + j];
                    }

                    প্রতি (int64_t j = 0; j < head_size; j++) {
                        int64_t t_h_j_offset = t_h_offset + j;
                        int64_t h_2d_i_j_offset = h_2d_i_offset + j;

                        ভিন্ন r_val = r[t_h_j_offset];
                        ভিন্ন w_val = w[t_h_j_offset];
                        ভিন্ন k_val = k[t_h_j_offset];
                        ভিন্ন b_val = b[t_h_j_offset];
                        ভিন্ন kv_val = v_val * k_val;
                        ভিন্ন prev_state_val = state_prev[h_2d_i_j_offset];
                        state_cur[h_2d_i_j_offset] = prev_state_val * w_val + kv_val + sa * b_val;
                        result += state_cur[h_2d_i_j_offset] * r_val;
                    }
                    dst_data[t_h_i_offset] = result;
                }
            }
        }
    #endif
}


শূন্য ggml_compute_forward_rwkv_wkv7(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_rwkv_wkv7_f32(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_map_custom1

শূন্য ggml_compute_forward_map_custom1(
        ধ্রুব ggml_compute_params * params,
              ggml_tensor * dst) {

    ধ্রুব ggml_tensor * a = dst->src[0];

    সংরচনা ggml_map_custom1_op_params p;
    memcpy(&p, dst->op_params, পরিমাণ(p));

    p.fun(dst, a, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_map_custom2

শূন্য ggml_compute_forward_map_custom2(
        ধ্রুব ggml_compute_params * params,
              ggml_tensor * dst) {

    ধ্রুব ggml_tensor * a = dst->src[0];
    ধ্রুব ggml_tensor * b = dst->src[1];

    সংরচনা ggml_map_custom2_op_params p;
    memcpy(&p, dst->op_params, পরিমাণ(p));

    p.fun(dst, a, b, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_map_custom3

শূন্য ggml_compute_forward_map_custom3(
        ধ্রুব ggml_compute_params * params,
              ggml_tensor * dst) {

    ধ্রুব ggml_tensor * a = dst->src[0];
    ধ্রুব ggml_tensor * b = dst->src[1];
    ধ্রুব ggml_tensor * c = dst->src[2];

    সংরচনা ggml_map_custom3_op_params p;
    memcpy(&p, dst->op_params, পরিমাণ(p));

    p.fun(dst, a, b, c, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_custom

শূন্য ggml_compute_forward_custom(
    ধ্রুব সংরচনা ggml_compute_params * params,
          সংরচনা ggml_tensor * dst) {

    সংরচনা ggml_custom_op_params p;
    memcpy(&p, dst->op_params, পরিমাণ(p));

    p.fun(dst, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_cross_entropy_loss

স্থির শূন্য ggml_compute_forward_cross_entropy_loss_f32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->nb[0] == ggml_type_size(src0->type));
    GGML_ASSERT(src1->nb[0] == ggml_type_size(src1->type));
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // TODO: handle transposed/permuted matrices
    ধ্রুব int64_t nc = src0->ne[0];
    ধ্রুব int64_t nr = ggml_nrows(src0);

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ভিন্ন * sums =  (ভিন্ন *) params->wdata;
    ভিন্ন * st   = ((ভিন্ন *) params->wdata) + nth + ith*nc;
    ভিন্ন sum_thread = 0.0f;

    GGML_ASSERT(params->wsize >= পরিমাণ(ভিন্ন) * (nth + nth * nc));

    // rows per thread
    ধ্রুব int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব int64_t ir0 = dr*ith;
    ধ্রুব int64_t ir1 = MIN(ir0 + dr, nr);

    প্রতি (int64_t i1 = ir0; i1 < ir1; ++i1) {
        ধ্রুব ভিন্ন * s0 = (ধ্রুব ভিন্ন *)((ধ্রুব বর্ণ *) src0->data + i1*src0->nb[1]);
        ধ্রুব ভিন্ন * s1 = (ধ্রুব ভিন্ন *)((ধ্রুব বর্ণ *) src1->data + i1*src1->nb[1]);

#ifndef NDEBUG
        প্রতি (int64_t i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            প্রতিজ্ঞা(!isnan(s0[i]));
            প্রতিজ্ঞা(!isnan(s1[i]));
        }
#endif // NDEBUG

        ভিন্ন max = -INFINITY;
        ggml_vec_max_f32(nc, &max, s0);
        ধ্রুব ggml_float sum_softmax = ggml_vec_log_soft_max_f32(nc, st, s0, max);
        প্রতিজ্ঞা(sum_softmax >= 0.0);

        ggml_vec_add1_f32(nc, st, st, -sum_softmax);
        ggml_vec_mul_f32(nc, st, st, s1);

        ভিন্ন sum_st = 0.0f;
        ggml_vec_sum_f32(nc, &sum_st, st);
        sum_thread += sum_st;

#ifndef NDEBUG
        প্রতি (int64_t i = 0; i < nc; ++i) {
            প্রতিজ্ঞা(!isnan(st[i]));
            প্রতিজ্ঞা(!isinf(st[i]));
        }
#endif // NDEBUG
    }
    sums[ith] = sum_thread;
    ggml_barrier(params->threadpool);

    যদি (ith == 0) {
        ভিন্ন * dp = (ভিন্ন *) dst->data;
        ggml_vec_sum_f32(nth, dp, sums);
        dp[0] *= -1.0f / (ভিন্ন) nr;
    }
}

শূন্য ggml_compute_forward_cross_entropy_loss(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_cross_entropy_loss_f32(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_cross_entropy_loss_back

স্থির শূন্য ggml_compute_forward_cross_entropy_loss_back_f32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * grad  = dst->src[0]; // gradient of forward pass output
    ধ্রুব ggml_tensor * src0f = dst->src[1]; // src0 of forward pass
    ধ্রুব ggml_tensor * src1f = dst->src[2]; // src1 of forward pass

    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src0f));
    GGML_ASSERT(ggml_is_contiguous(src1f));
    GGML_ASSERT(ggml_is_contiguous(grad));
    GGML_ASSERT(ggml_are_same_shape(src0f, src1f) && ggml_are_same_shape(src0f, dst));

    ধ্রুব int64_t ith = params->ith;
    ধ্রুব int64_t nth = params->nth;

    // TODO: handle transposed/permuted matrices
    ধ্রুব int64_t nc = src0f->ne[0];
    ধ্রুব int64_t nr = ggml_nrows(src0f);

    // rows per thread
    ধ্রুব int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব int64_t ir0 = dr*ith;
    ধ্রুব int64_t ir1 = MIN(ir0 + dr, nr);

    ধ্রুব ভিন্ন d_by_nr = ((ধ্রুব ভিন্ন *) grad->data)[0] / (ভিন্ন) nr;

    প্রতি (int64_t i1 = ir0; i1 < ir1; i1++) {
        ভিন্ন       * ds0 = (ভিন্ন       *)((বর্ণ       *) dst->data   + i1*dst->nb[1]);
        ধ্রুব ভিন্ন * s0  = (ধ্রুব ভিন্ন *)((ধ্রুব বর্ণ *) src0f->data + i1*src0f->nb[1]);
        ধ্রুব ভিন্ন * s1  = (ধ্রুব ভিন্ন *)((ধ্রুব বর্ণ *) src1f->data + i1*src1f->nb[1]);

#ifndef NDEBUG
        প্রতি (int64_t i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            প্রতিজ্ঞা(!isnan(s0[i]));
            প্রতিজ্ঞা(!isnan(s1[i]));
        }
#endif // NDEBUG

        // soft_max
        ভিন্ন max = -INFINITY;
        ggml_vec_max_f32(nc, &max, s0);
        ধ্রুব ggml_float sum = ggml_vec_soft_max_f32(nc, ds0, s0, max);
        প্রতিজ্ঞা(sum > 0.0);
        ggml_vec_scale_f32(nc, ds0, 1.0/sum);

        // grad(src0f) = (softmax(src0f) - src1f) * grad(cross_entropy_loss(src0f, src1f)) / nr
        ggml_vec_sub_f32(nc, ds0, ds0, s1);
        ggml_vec_scale_f32(nc, ds0, d_by_nr);

#ifndef NDEBUG
        প্রতি (int64_t i = 0; i < nc; ++i) {
            প্রতিজ্ঞা(!isnan(ds0[i]));
            প্রতিজ্ঞা(!isinf(ds0[i]));
        }
#endif // NDEBUG
    }
}

শূন্য ggml_compute_forward_cross_entropy_loss_back(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_cross_entropy_loss_back_f32(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

স্থির শূন্য ggml_compute_forward_opt_step_adamw_f32(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0         = dst->src[0];
    ধ্রুব ggml_tensor * src0_grad    = dst->src[1];
    ধ্রুব ggml_tensor * src0_grad_m  = dst->src[2];
    ধ্রুব ggml_tensor * src0_grad_v  = dst->src[3];
    ধ্রুব ggml_tensor * adamw_params = dst->src[4];

    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad));
    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad_m));
    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad_v));
    GGML_ASSERT(ggml_nelements(adamw_params) == 7);

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS
    GGML_ASSERT(nb00 == পরিমাণ(ভিন্ন));

    // rows per thread
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr*ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    ধ্রুব ভিন্ন * adamw_params_ptr = ggml_get_data_f32(adamw_params);

    ধ্রুব ভিন্ন alpha  = adamw_params_ptr[0];
    ধ্রুব ভিন্ন beta1  = adamw_params_ptr[1];
    ধ্রুব ভিন্ন beta2  = adamw_params_ptr[2];
    ধ্রুব ভিন্ন eps    = adamw_params_ptr[3];
    ধ্রুব ভিন্ন wd     = adamw_params_ptr[4];
    ধ্রুব ভিন্ন beta1h = adamw_params_ptr[5];
    ধ্রুব ভিন্ন beta2h = adamw_params_ptr[6];
    ধ্রুব ভিন্ন keep   = 1.f - alpha * wd;
    প্রতি (পূর্ণাঙ্ক ir = ir0; ir < ir1; ++ir) {
        ধ্রুব int64_t i03 = ir/(ne02*ne01);
        ধ্রুব int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        ধ্রুব int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        ধ্রুব size_t offset = i03*nb03 + i02*nb02 + i01*nb01;

        ভিন্ন       * w = (ভিন্ন       *) ((বর্ণ       *) src0->data        + offset); // weight
        ধ্রুব ভিন্ন * g = (ধ্রুব ভিন্ন *) ((ধ্রুব বর্ণ *) src0_grad->data   + offset); // grad
        ভিন্ন       * m = (ভিন্ন       *) ((বর্ণ       *) src0_grad_m->data + offset);
        ভিন্ন       * v = (ভিন্ন       *) ((বর্ণ       *) src0_grad_v->data + offset);

        প্রতি (পূর্ণাঙ্ক i00 = 0; i00 < ne00; ++i00) {
            m[i00] = m[i00]*beta1 +        g[i00]*(1.0f - beta1);
            v[i00] = v[i00]*beta2 + g[i00]*g[i00]*(1.0f - beta2);

            ধ্রুব ভিন্ন mh =       m[i00]*beta1h;
            ধ্রুব ভিন্ন vh = sqrtf(v[i00]*beta2h) + eps;

            // The weight decay is applied independently of the Adam momenta m and v.
            // This is NOT equivalent to l2 regularization that adds w[i00]*w[i00] to the loss.
            // See: https://arxiv.org/pdf/1711.05101v3.pdf
            w[i00] = w[i00] * keep - alpha * mh / vh;
        }
    }
}

শূন্য ggml_compute_forward_opt_step_adamw(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_opt_step_adamw_f32(params, dst);
            } ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error");
            }
    }
}

স্থির শূন্য ggml_compute_forward_opt_step_sgd_f32(ধ্রুব ggml_compute_params * params, ggml_tensor * dst) {
    ধ্রুব ggml_tensor * src0       = dst->src[0];
    ধ্রুব ggml_tensor * src0_grad  = dst->src[1];
    ধ্রুব ggml_tensor * sgd_params = dst->src[2];

    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad));
    GGML_ASSERT(ggml_nelements(sgd_params) == 2);

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব পূর্ণাঙ্ক nr = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS
    GGML_ASSERT(nb00 == পরিমাণ(ভিন্ন));

    // rows per thread
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1) / nth;

    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr * ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    // using adamw param subset we care about - alpha, wd - could have a separate struct
    ধ্রুব ভিন্ন * sgd_params_ptr   = ggml_get_data_f32(sgd_params);
    ধ্রুব ভিন্ন   alpha            = sgd_params_ptr[0];
    ধ্রুব ভিন্ন   keep             = 1.f - alpha * sgd_params_ptr[1];

    প্রতি (পূর্ণাঙ্ক ir = ir0; ir < ir1; ++ir) {
        ধ্রুব int64_t i03 = ir / (ne02 * ne01);
        ধ্রুব int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
        ধ্রুব int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

        ধ্রুব size_t offset = i03 * nb03 + i02 * nb02 + i01 * nb01;

        ভিন্ন *       w = (ভিন্ন *) ((বর্ণ *) src0->data + offset);                   // weight
        ধ্রুব ভিন্ন * g = (ধ্রুব ভিন্ন *) ((ধ্রুব বর্ণ *) src0_grad->data + offset);  // grad

        প্রতি (পূর্ণাঙ্ক i00 = 0; i00 < ne00; ++i00) {
            w[i00] = w[i00] * keep - alpha * g[i00];
        }
    }
}

শূন্য ggml_compute_forward_opt_step_sgd(ধ্রুব ggml_compute_params * params, ggml_tensor * dst) {
    ধ্রুব ggml_tensor * src0 = dst->src[0];

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_opt_step_sgd_f32(params, dst);
            }
            ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error - sgd is F32 only");
            }
    }
}

স্থির অন্তঃস্থ ভিন্ন ggml_fwht_load(ধ্রুব ভিন্ন value) {
    প্রত্যাগম value;
}

স্থির অন্তঃস্থ ভিন্ন ggml_fwht_load(ধ্রুব ggml_fp16_t value) {
    প্রত্যাগম ggml_fp16_to_fp32(value);
}

template<typename src_t>
স্থির শূন্য ggml_compute_forward_fwht_impl(ধ্রুব ggml_compute_params * params, ggml_tensor * dst) {
    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == (std::is_same_v<src_t, ভিন্ন> ? GGML_TYPE_F32 : GGML_TYPE_F16));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_দ্বিমিক_OP_LOCALS

    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    ধ্রুব int64_t n = ne10;
    GGML_ASSERT((n & (n - 1)) == 0); // must be power of 2

    ধ্রুব int64_t nr = ne11 * ne12 * ne13;
    ধ্রুব int64_t rows_per_thread = (nr + nth - 1) / nth;
    ধ্রুব int64_t start_row = ith * rows_per_thread;
    ধ্রুব int64_t end_row = MIN(start_row + rows_per_thread, nr);

    ধ্রুব ভিন্ন scale = 1.0f / sqrtf((ভিন্ন)n);

#যদি defined(GGML_SIMD)
    ধ্রুব GGML_F32_VEC v_minus_one = GGML_F32_VEC_SET1(-1.0f);
#endif

    প্রতি (int64_t r = start_row; r < end_row; r++) {
        ধ্রুব int64_t i13 = r / (ne11 * ne12);
        ধ্রুব int64_t i12 = (r - i13 * ne11 * ne12) / ne11;
        ধ্রুব int64_t i11 = r - i13 * ne11 * ne12 - i12 * ne11;

        ধ্রুব src_t * src_row = (ধ্রুব src_t *) ((ধ্রুব বর্ণ *) src1->data + i11 * nb11 + i12 * nb12 + i13 * nb13);
        ভিন্ন * dst_row = (ভিন্ন *) ((বর্ণ *) dst->data + i11 * nb1 + i12 * nb2 + i13 * nb3);

        প্রতি (int64_t j = 0; j < n; j++) {
            dst_row[j] = ggml_fwht_load(src_row[j]) * scale;
        }

        // Scalar passes
#যদি defined(GGML_SIMD)
#যদি defined(__ARM_FEATURE_SVE)
        ধ্রুব পূর্ণাঙ্ক step = svcntw();
#অন্যথা
        ধ্রুব পূর্ণাঙ্ক step = GGML_F32_EPR;
#endif
#অন্যথা
        ধ্রুব পূর্ণাঙ্ক step = n;
#endif
        প্রতি (int64_t len = 1; len < step && len < n; len <<= 1) {
            প্রতি (int64_t i = 0; i < n; i += 2 * len) {
                প্রতি (int64_t j = 0; j < len; j++) {
                    ভিন্ন u = dst_row[i + j];
                    ভিন্ন v = dst_row[i + len + j];
                    dst_row[i + j] = u + v;
                    dst_row[i + len + j] = u - v;
                }
            }
        }

        // SIMD passes using GGML_F32_VEC_* macros for multi-architecture support
#যদি defined(GGML_SIMD)
        প্রতি (int64_t len = step; len < n; len <<= 1) {
            প্রতি (int64_t i = 0; i < n; i += 2 * len) {
                প্রতি (int64_t j = 0; j < len; j += step) {
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

শূন্য ggml_compute_forward_fwht(ধ্রুব ggml_compute_params * params, ggml_tensor * dst) {
    ধ্রুব ggml_tensor * src1 = dst->src[1];

    বিকল্প (src1->type) {
        পক্ষ GGML_TYPE_F32:
            {
                ggml_compute_forward_fwht_impl<ভিন্ন>(params, dst);
            }
            ভঙ্গ;
        পক্ষ GGML_TYPE_F16:
            {
                ggml_compute_forward_fwht_impl<ggml_fp16_t>(params, dst);
            }
            ভঙ্গ;
        স্বভাব:
            {
                GGML_ABORT("fatal error - fwht supports F32 and F16 input");
            }
    }
}

// ggml_compute_forward_lightning_indexer

শূন্য ggml_compute_forward_lightning_indexer(
        ধ্রুব ggml_compute_params * params,
        ggml_tensor * dst) {

    ধ্রুব ggml_tensor * q = dst->src[0];
    ধ্রুব ggml_tensor * k = dst->src[1];
    ধ্রুব ggml_tensor * w = dst->src[2]; // weights
    ধ্রুব ggml_tensor * m = dst->src[3]; // mask

    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(   q->type == GGML_TYPE_F32);
    GGML_ASSERT(   w->type == GGML_TYPE_F32);
    GGML_ASSERT(   m->type == GGML_TYPE_F16);

    GGML_TENSOR_LOCALS(int64_t, neq,  q, ne)
    GGML_TENSOR_LOCALS(size_t,  nbq,  q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek,  k, ne)
    GGML_TENSOR_LOCALS(size_t,  nbk,  k, nb)
    GGML_TENSOR_LOCALS(int64_t, নূতন,  w, ne)
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

    ধ্রুব পূর্ণাঙ্ক n_embd    = q->ne[0];
    ধ্রুব পূর্ণাঙ্ক n_head    = q->ne[1];
    ধ্রুব পূর্ণাঙ্ক n_tokens  = q->ne[2];
    ধ্রুব পূর্ণাঙ্ক n_stream  = q->ne[3];
    ধ্রুব পূর্ণাঙ্ক n_kv      = k->ne[2];

    ggml_to_float_t ধ্রুব k_to_float = ggml_get_type_traits(k->type)->to_float;
    GGML_ASSERT((k->type == GGML_TYPE_F32 || k_to_float) && "lightning indexer: unsupported K-type");

    ধ্রুব পূর্ণাঙ্ক nr  = n_kv;
    ধ্রুব পূর্ণাঙ্ক ith = params->ith;
    ধ্রুব পূর্ণাঙ্ক nth = params->nth;

    // (temporary) buffer for K converted to float
    ভিন্ন * k_row_f32 = (ভিন্ন *) params->wdata + ith*(1*n_embd + CACHE_সারি_SIZE_F32);

    // rows per thread
    ধ্রুব পূর্ণাঙ্ক dr = (nr + nth - 1)/nth;

    // row range for this thread
    ধ্রুব পূর্ণাঙ্ক ir0 = dr*ith;
    ধ্রুব পূর্ণাঙ্ক ir1 = MIN(ir0 + dr, nr);

    প্রতি (পূর্ণাঙ্ক s = 0; s < n_stream; ++s) {
        প্রতি (পূর্ণাঙ্ক t = 0; t < n_tokens; ++t) {
            ধ্রুব ভিন্ন       *   w_row =       (ভিন্ন *) ((বর্ণ *)   w->data + t*nbw1 +        s*nbw3);
            ধ্রুব ggml_fp16_t *   m_row = (ggml_fp16_t *) ((বর্ণ *)   m->data + t*nbm1 + (s%nem3)*nbm3);
            ভিন্ন             * dst_row =       (ভিন্ন *) ((বর্ণ *) dst->data + t*nb1  +        s*nb3 );
            প্রতি (পূর্ণাঙ্ক ik = ir0; ik < ir1; ++ik) {
                বর্ণ * k_row = (বর্ণ *) k->data + ik*nbk2 + s*nbk3;
                যদি (k_to_float) {
                    k_to_float(k_row, k_row_f32, n_embd);
                } অন্যথা {
                    k_row_f32 = (ভিন্ন *) k_row;
                }
                ভিন্ন score = 0.0f;
                প্রতি (পূর্ণাঙ্ক h = 0; h < n_head; ++h) {
                    // dot product of q and k for head h
                    ভিন্ন qk = 0.0f;
                    ধ্রুব ভিন্ন * q_row = (ভিন্ন *) ((বর্ণ *) q->data + h*nbq1 + t*nbq2 + s*nbq3);
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
