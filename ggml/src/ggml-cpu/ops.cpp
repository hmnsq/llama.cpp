#അന്തര്ഭാവ "ops.h"

#അന്തര്ഭാവ "ggml-cpu.h"
#അന്തര്ഭാവ "ggml-impl.h"
#അന്തര്ഭാവ "binary-ops.h"
#അന്തര്ഭാവ "simd-gemm.h"
#അന്തര്ഭാവ "ggml.h"
#അന്തര്ഭാവ "unary-ops.h"
#അന്തര്ഭാവ "vec.h"

#അന്തര്ഭാവ <algorithm>
#അന്തര്ഭാവ <cfloat>
#അന്തര്ഭാവ <cmath>

// ggml_compute_forward_dup

സ്ഥിതം ശൂന്യം ggml_compute_forward_dup_same_cont(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));
    GGML_ASSERT(src0->type == dst->type);

    സ്ഥിരം size_t nb0 = ggml_type_size(src0->type);

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith; // thread index
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth; // number of threads

    // parallelize by blocks
    സ്ഥിരം പൂർണ്ണസംഖ്യ nk = ggml_nelements(src0)/ggml_blck_size(src0->type);
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nk + nth - 1) / nth;
    സ്ഥിരം പൂർണ്ണസംഖ്യ k0 = dr * ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ k1 = MIN(k0 + dr, nk);

    എങ്കിൽ (k0 < k1) {
        memcpy(
            ((അക്ഷരം *)  dst->data + k0*nb0),
            ((അക്ഷരം *) src0->data + k0*nb0),
            (k1 - k0) * nb0);
    }
}

template<typename src_t, typename dst_t>
സ്ഥിതം ശൂന്യം ggml_compute_forward_dup_flt(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(!ggml_is_quantized(src0->type) && !ggml_is_quantized(dst->type));

    GGML_TENSOR_UNARY_OP_LOCALS

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith; // thread index
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth; // number of threads

    // parallelize by rows
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr = ne01;
    // number of rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1) / nth;
    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr * ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    // case: type & row size equal
    എങ്കിൽ (src0->type == dst->type &&
        ne00 == ne0 &&
        nb00 == ggml_type_size(src0->type) && nb0 == ggml_type_size(dst->type)) {
        // copy by rows
        സ്ഥിരം size_t rs = ne00*nb00;
        വേണ്ടി (int64_t i03 = 0; i03 < ne03; i03++) {
            വേണ്ടി (int64_t i02 = 0; i02 < ne02; i02++) {
                വേണ്ടി (int64_t i01 = ir0; i01 < ir1; i01++) {
                    memcpy(
                        ((അക്ഷരം *)  dst->data + i01*nb1  + i02*nb2  + i03*nb3),
                        ((അക്ഷരം *) src0->data + i01*nb01 + i02*nb02 + i03*nb03),
                        rs);
                }
            }
        }
        തിരികെ;
    }

    // case: dst tensor is contiguous
    എങ്കിൽ (ggml_is_contiguous(dst)) {
        എങ്കിൽ (nb00 == വലുപ്പം(src_t)) {
            എങ്കിൽ constexpr (std::is_same_v<dst_t, src_t>) {
                // same type
                size_t id = 0;
                സ്ഥിരം size_t rs = ne00 * nb00;
                അക്ഷരം * dst_ptr = (അക്ഷരം *) dst->data;

                വേണ്ടി (പൂർണ്ണസംഖ്യ i03 = 0; i03 < ne03; i03++) {
                    വേണ്ടി (പൂർണ്ണസംഖ്യ i02 = 0; i02 < ne02; i02++) {
                        id += rs * ir0;
                        വേണ്ടി (പൂർണ്ണസംഖ്യ i01 = ir0; i01 < ir1; i01++) {
                            സ്ഥിരം അക്ഷരം * src0_ptr = (അക്ഷരം *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
                            memcpy(dst_ptr + id, src0_ptr, rs);
                            id += rs;
                        }
                        id += rs * (ne01 - ir1);
                    }
                }
            } അല്ലെങ്കിൽ {
                // casting between non-quantized types
                size_t id = 0;
                dst_t * dst_ptr = (dst_t *) dst->data;

                വേണ്ടി (പൂർണ്ണസംഖ്യ i03 = 0; i03 < ne03; i03++) {
                    വേണ്ടി (പൂർണ്ണസംഖ്യ i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        വേണ്ടി (പൂർണ്ണസംഖ്യ i01 = ir0; i01 < ir1; i01++) {
                            സ്ഥിരം src_t * src0_ptr = (src_t *) ((അക്ഷരം *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
                            വേണ്ടി (പൂർണ്ണസംഖ്യ i00 = 0; i00 < ne00; i00++) {
                                ദശാംശം tmp = type_conversion_table<src_t>::to_f32(src0_ptr[i00]);
                                dst_ptr[id] = type_conversion_table<dst_t>::from_f32(tmp);
                                id++;
                            }
                        }
                        id += ne00 * (ne01 - ir1);
                    }
                }
            }
        } അല്ലെങ്കിൽ {
            //printf("%s: this is not optimal - fix me\n", __func__);

            size_t id = 0;
            dst_t * dst_ptr = (dst_t *) dst->data;

            വേണ്ടി (പൂർണ്ണസംഖ്യ i03 = 0; i03 < ne03; i03++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ i02 = 0; i02 < ne02; i02++) {
                    id += ne00 * ir0;
                    വേണ്ടി (പൂർണ്ണസംഖ്യ i01 = ir0; i01 < ir1; i01++) {
                        വേണ്ടി (പൂർണ്ണസംഖ്യ i00 = 0; i00 < ne00; i00++) {
                            സ്ഥിരം src_t * src0_ptr = (src_t *) ((അക്ഷരം *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);

                            ദശാംശം tmp = type_conversion_table<src_t>::to_f32(*src0_ptr);
                            dst_ptr[id] = type_conversion_table<dst_t>::from_f32(tmp);
                            id++;
                        }
                    }
                    id += ne00 * (ne01 - ir1);
                }
            }
        }
        തിരികെ;
    }

    // dst counters
    int64_t i10 = 0;
    int64_t i11 = 0;
    int64_t i12 = 0;
    int64_t i13 = 0;

    എങ്കിൽ constexpr (std::is_same_v<dst_t, src_t>) {
        വേണ്ടി (int64_t i03 = 0; i03 < ne03; i03++) {
            വേണ്ടി (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                ആകുമ്പോൾ (i10 >= ne0) {
                    i10 -= ne0;
                    എങ്കിൽ (++i11 == ne1) {
                        i11 = 0;
                        എങ്കിൽ (++i12 == ne2) {
                            i12 = 0;
                            എങ്കിൽ (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                വേണ്ടി (int64_t i01 = ir0; i01 < ir1; i01++) {
                    വേണ്ടി (int64_t i00 = 0; i00 < ne00; i00++) {
                        സ്ഥിരം അക്ഷരം * src0_ptr = ((അക്ഷരം *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              അക്ഷരം * dst_ptr  = ((അക്ഷരം *)  dst->data + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                        memcpy(dst_ptr, src0_ptr, വലുപ്പം(dst_t));

                        എങ്കിൽ (++i10 == ne00) {
                            i10 = 0;
                            എങ്കിൽ (++i11 == ne01) {
                                i11 = 0;
                                എങ്കിൽ (++i12 == ne02) {
                                    i12 = 0;
                                    എങ്കിൽ (++i13 == ne03) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                ആകുമ്പോൾ (i10 >= ne0) {
                    i10 -= ne0;
                    എങ്കിൽ (++i11 == ne1) {
                        i11 = 0;
                        എങ്കിൽ (++i12 == ne2) {
                            i12 = 0;
                            എങ്കിൽ (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
            }
        }

    } അല്ലെങ്കിൽ {
        വേണ്ടി (int64_t i03 = 0; i03 < ne03; i03++) {
            വേണ്ടി (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                ആകുമ്പോൾ (i10 >= ne0) {
                    i10 -= ne0;
                    എങ്കിൽ (++i11 == ne1) {
                        i11 = 0;
                        എങ്കിൽ (++i12 == ne2) {
                            i12 = 0;
                            എങ്കിൽ (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                വേണ്ടി (int64_t i01 = ir0; i01 < ir1; i01++) {
                    വേണ്ടി (int64_t i00 = 0; i00 < ne00; i00++) {
                        സ്ഥിരം അക്ഷരം * src0_ptr = ((അക്ഷരം *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              അക്ഷരം * dst_ptr  = ((അക്ഷരം *)  dst->data + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                        ദശാംശം tmp = type_conversion_table<src_t>::to_f32(*(സ്ഥിരം src_t *) src0_ptr);
                        *(dst_t *) dst_ptr = type_conversion_table<dst_t>::from_f32(tmp);

                        എങ്കിൽ (++i10 == ne0) {
                            i10 = 0;
                            എങ്കിൽ (++i11 == ne1) {
                                i11 = 0;
                                എങ്കിൽ (++i12 == ne2) {
                                    i12 = 0;
                                    എങ്കിൽ (++i13 == ne3) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                ആകുമ്പോൾ (i10 >= ne0) {
                    i10 -= ne0;
                    എങ്കിൽ (++i11 == ne1) {
                        i11 = 0;
                        എങ്കിൽ (++i12 == ne2) {
                            i12 = 0;
                            എങ്കിൽ (++i13 == ne3) {
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
സ്ഥിതം ശൂന്യം ggml_compute_forward_dup_to_q(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(!ggml_is_quantized(src0->type));

    GGML_TENSOR_UNARY_OP_LOCALS

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith; // thread index
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth; // number of threads

    // parallelize by rows
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr = ne01;
    // number of rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1) / nth;
    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr * ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    എങ്കിൽ (ggml_is_contiguous(dst) &&
            nb00 == വലുപ്പം(src_t) &&
            ggml_get_type_traits_cpu(dst->type)->from_float) {
        // casting non-quantized types --> intermediate f32 --> quantized
        ggml_from_float_t സ്ഥിരം quantize_row_q = ggml_get_type_traits_cpu(dst->type)->from_float;
        ദശാംശം * src0_f32 = (ദശാംശം *) params->wdata + (ne00 + CACHE_വരി_SIZE_F32) * ith;

        size_t id = 0;
        size_t rs = nb0 * (ne00 / ggml_blck_size(dst->type));
        അക്ഷരം * dst_ptr = (അക്ഷരം *) dst->data;

        വേണ്ടി (പൂർണ്ണസംഖ്യ i03 = 0; i03 < ne03; i03++) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ i02 = 0; i02 < ne02; i02++) {
                id += rs * ir0;
                വേണ്ടി (പൂർണ്ണസംഖ്യ i01 = ir0; i01 < ir1; i01++) {
                    സ്ഥിരം src_t * src0_ptr = (src_t *) ((അക്ഷരം *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);

                    വേണ്ടി (പൂർണ്ണസംഖ്യ i00 = 0; i00 < ne00; i00++) {
                        src0_f32[i00] = type_conversion_table<src_t>::to_f32(src0_ptr[i00]);
                    }

                    quantize_row_q(src0_f32, dst_ptr + id, ne00);
                    id += rs;
                }
                id += rs * (ne01 - ir1);
            }
        }
    } അല്ലെങ്കിൽ {
        // printf("%s %s\n", ggml_type_name(src0->type), ggml_type_name(dst->type));
        GGML_ABORT("not implemented");
    }
}

// A simplified version of ggml_compute_forward_dup that doesn't do float upcasting, and just plain old memcpy.
സ്ഥിതം ശൂന്യം ggml_compute_forward_dup_bytes(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {
    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(src0->type == dst->type);

    GGML_TENSOR_UNARY_OP_LOCALS;

    എങ്കിൽ (ggml_is_contiguous(src0) && ggml_is_contiguous(dst)) {
        ggml_compute_forward_dup_same_cont(params, dst);
        തിരികെ;
    }

    സ്ഥിരം size_t type_size = ggml_type_size(src0->type);

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith; // thread index
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth; // number of threads

    // parallelize by rows
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr = ne01;
    // number of rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1) / nth;
    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr * ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    എങ്കിൽ (src0->type == dst->type &&
        ggml_are_same_shape(src0, dst) &&
        nb00 == type_size && nb0 == type_size) {
        // copy by rows
        സ്ഥിരം size_t rs = ggml_row_size(src0->type, ne00);
        വേണ്ടി (int64_t i03 = 0; i03 < ne03; i03++) {
            വേണ്ടി (int64_t i02 = 0; i02 < ne02; i02++) {
                വേണ്ടി (int64_t i01 = ir0; i01 < ir1; i01++) {
                    memcpy(
                        ((അക്ഷരം *)  dst->data + i01*nb1  + i02*nb2  + i03*nb3),
                        ((അക്ഷരം *) src0->data + i01*nb01 + i02*nb02 + i03*nb03),
                        rs);
                }
            }
        }
        തിരികെ;
    }

    എങ്കിൽ (ggml_is_contiguous(dst)) {
        size_t id = 0;
        അക്ഷരം * dst_ptr = (അക്ഷരം *) dst->data;
        സ്ഥിരം size_t rs = ne00 * type_size;

        എങ്കിൽ (nb00 == type_size) {
            // src0 is contiguous on first dimension, copy by rows
            വേണ്ടി (int64_t i03 = 0; i03 < ne03; i03++) {
                വേണ്ടി (int64_t i02 = 0; i02 < ne02; i02++) {
                    id += rs * ir0;
                    വേണ്ടി (int64_t i01 = ir0; i01 < ir1; i01++) {
                        സ്ഥിരം അക്ഷരം * src0_ptr = (അക്ഷരം *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
                        memcpy(dst_ptr + id, src0_ptr, rs);
                        id += rs;
                    }
                    id += rs * (ne01 - ir1);
                }
            }
        } അല്ലെങ്കിൽ {
            //printf("%s: this is not optimal - fix me\n", __func__);

            വേണ്ടി (int64_t i03 = 0; i03 < ne03; i03++) {
                വേണ്ടി (int64_t i02 = 0; i02 < ne02; i02++) {
                    id += rs * ir0;
                    വേണ്ടി (int64_t i01 = ir0; i01 < ir1; i01++) {
                        വേണ്ടി (int64_t i00 = 0; i00 < ne00; i00++) {
                            സ്ഥിരം അക്ഷരം * src0_ptr = (അക്ഷരം *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03;
                            memcpy(dst_ptr + id, src0_ptr, type_size);

                            id += type_size;
                        }
                    }
                    id += rs * (ne01 - ir1);
                }
            }
        }

        തിരികെ;
    }

    // dst counters
    int64_t k10 = 0;
    int64_t i11 = 0;
    int64_t i12 = 0;
    int64_t i13 = 0;

    // number of blocks in a row
    സ്ഥിരം int64_t nk00 = ne00 / ggml_blck_size(src0->type);
    സ്ഥിരം int64_t nk0  = ne0  / ggml_blck_size(dst->type);

    വേണ്ടി (int64_t i03 = 0; i03 < ne03; i03++) {
        വേണ്ടി (int64_t i02 = 0; i02 < ne02; i02++) {
            k10 += nk00 * ir0;
            ആകുമ്പോൾ (k10 >= nk0) {
                k10 -= nk0;
                എങ്കിൽ (++i11 == ne1) {
                    i11 = 0;
                    എങ്കിൽ (++i12 == ne2) {
                        i12 = 0;
                        എങ്കിൽ (++i13 == ne3) {
                            i13 = 0;
                        }
                    }
                }
            }
            വേണ്ടി (int64_t i01 = ir0; i01 < ir1; i01++) {
                വേണ്ടി (int64_t k00 = 0; k00 < nk00; k00++) {
                    സ്ഥിരം അക്ഷരം * src0_ptr = ((അക്ഷരം *) src0->data + k00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                          അക്ഷരം * dst_ptr  = ((അക്ഷരം *)  dst->data + k10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                    memcpy(dst_ptr, src0_ptr, type_size);

                    എങ്കിൽ (++k10 == nk0) {
                        k10 = 0;
                        എങ്കിൽ (++i11 == ne1) {
                            i11 = 0;
                            എങ്കിൽ (++i12 == ne2) {
                                i12 = 0;
                                എങ്കിൽ (++i13 == ne3) {
                                    i13 = 0;
                                }
                            }
                        }
                    }
                }
            }
            k10 += nk00 * (ne01 - ir1);
            ആകുമ്പോൾ (k10 >= nk0) {
                k10 -= nk0;
                എങ്കിൽ (++i11 == ne1) {
                    i11 = 0;
                    എങ്കിൽ (++i12 == ne2) {
                        i12 = 0;
                        എങ്കിൽ (++i13 == ne3) {
                            i13 = 0;
                        }
                    }
                }
            }
        }
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_dup_from_q(
        സ്ഥിരം ggml_compute_params * params,
              ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ദ്വിമാന_OP_LOCALS

    സ്ഥിരം ggml_type type = src0->type;
    ggml_to_float_t സ്ഥിരം dequantize_row_q = ggml_get_type_traits(type)->to_float;

    size_t qk = ggml_blck_size(type);
    സ്ഥിരം int64_t nr = ggml_nelements(src1) / qk;

    // destination must be contiguous in the first dimension
    GGML_ASSERT(nb10 == ggml_type_size(dst->type));
    // must either have first dimension large enough to hold a row, or fully contiguous
    GGML_ASSERT((ne10 % qk) == 0 || ggml_is_contiguous(dst));

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    വേണ്ടി (int64_t ir = ir0; ir < ir1; ++ir) {

        uint32_t i = ir * qk;

        സ്ഥിരം int64_t i03 = i/(ne00 * ne01 * ne02);
        സ്ഥിരം int64_t i02 = (i - i03*ne00*ne01*ne02 )/ (ne00*ne01);
        സ്ഥിരം int64_t i01 = (i - i03*ne00*ne01*ne02  -  i02*ne01*ne00) / ne00;
        സ്ഥിരം int64_t i00 = i - i03*ne00*ne01*ne02 - i02*ne01*ne00 - i01*ne00;
        സ്ഥിരം int64_t x_offset = (i00/qk)*nb00 + i01*nb01 + i02*nb02 + i03 * nb03;

        സ്ഥിരം int64_t i13 = i/(ne10 * ne11 * ne12);
        സ്ഥിരം int64_t i12 = (i - i13*ne10*ne11*ne12) / (ne10*ne11);
        സ്ഥിരം int64_t i11 = (i - i13*ne10*ne11*ne12 - i12*ne10*ne11) / ne10;
        സ്ഥിരം int64_t i10 = i - i13*ne10*ne11*ne12 - i12*ne10*ne11 - i11*ne10;
        സ്ഥിരം int64_t dst_offset = i10*nb10 + i11*nb11 + i12*nb12 + i13*nb13;

        dequantize_row_q(
                (സ്ഥിരം ശൂന്യം *) ((അക്ഷരം *) src0->data + x_offset),
                     (ദശാംശം *) ((അക്ഷരം *)  dst->data + dst_offset), qk);
    }
}

ശൂന്യം ggml_compute_forward_dup(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    എങ്കിൽ (src0->type == dst->type) {
        ggml_compute_forward_dup_bytes(params, dst);
        തിരികെ;
    }

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F16:
            {
                /**/ എങ്കിൽ (dst->type == GGML_TYPE_F16)  ggml_compute_forward_dup_flt<ggml_fp16_t, ggml_fp16_t>(params, dst);
                അല്ലെങ്കിൽ എങ്കിൽ (dst->type == GGML_TYPE_BF16) ggml_compute_forward_dup_flt<ggml_fp16_t, ggml_bf16_t>(params, dst);
                അല്ലെങ്കിൽ എങ്കിൽ (dst->type == GGML_TYPE_F32)  ggml_compute_forward_dup_flt<ggml_fp16_t, ദശാംശം      >(params, dst);
                അല്ലെങ്കിൽ ggml_compute_forward_dup_to_q<ggml_fp16_t>(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_BF16:
            {
                /**/ എങ്കിൽ (dst->type == GGML_TYPE_F16)  ggml_compute_forward_dup_flt<ggml_bf16_t, ggml_fp16_t>(params, dst);
                അല്ലെങ്കിൽ എങ്കിൽ (dst->type == GGML_TYPE_BF16) ggml_compute_forward_dup_flt<ggml_bf16_t, ggml_bf16_t>(params, dst);
                അല്ലെങ്കിൽ എങ്കിൽ (dst->type == GGML_TYPE_F32)  ggml_compute_forward_dup_flt<ggml_bf16_t, ദശാംശം      >(params, dst);
                അല്ലെങ്കിൽ ggml_compute_forward_dup_to_q<ggml_bf16_t>(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F32:
            {
                /**/ എങ്കിൽ (dst->type == GGML_TYPE_F16)  ggml_compute_forward_dup_flt<ദശാംശം, ggml_fp16_t>(params, dst);
                അല്ലെങ്കിൽ എങ്കിൽ (dst->type == GGML_TYPE_BF16) ggml_compute_forward_dup_flt<ദശാംശം, ggml_bf16_t>(params, dst);
                അല്ലെങ്കിൽ എങ്കിൽ (dst->type == GGML_TYPE_F32)  ggml_compute_forward_dup_flt<ദശാംശം, ദശാംശം      >(params, dst);
                അല്ലെങ്കിൽ എങ്കിൽ (dst->type == GGML_TYPE_I32)  ggml_compute_forward_dup_flt<ദശാംശം, int32_t    >(params, dst);
                അല്ലെങ്കിൽ ggml_compute_forward_dup_to_q<ദശാംശം>(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_I32:
            {
                എങ്കിൽ (dst->type == GGML_TYPE_F32) ggml_compute_forward_dup_flt<int32_t, ദശാംശം>(params, dst);
                അല്ലെങ്കിൽ GGML_ABORT("not implemented");
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                എങ്കിൽ (ggml_is_quantized(src0->type) && dst->type == GGML_TYPE_F32) {
                    ggml_compute_forward_dup_from_q(params, dst);
                    നിർത്തുക;
                }
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_add

സ്ഥിതം ശൂന്യം ggml_compute_forward_add_q_f32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

    സ്ഥിരം പൂർണ്ണസംഖ്യ nr  = ggml_nrows(src0);

    GGML_TENSOR_ദ്വിമാന_OP_LOCALS

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം ggml_type type = src0->type;
    സ്ഥിരം ggml_type dtype = dst->type;
    ggml_to_float_t സ്ഥിരം dequantize_row_q = ggml_get_type_traits(type)->to_float;
    ggml_from_float_t സ്ഥിരം quantize_row_q = ggml_get_type_traits_cpu(dtype)->from_float;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == വലുപ്പം(ദശാംശം));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ggml_is_quantized(src0->type));
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    // rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    ദശാംശം * wdata = (ദശാംശം *) params->wdata + (ne00 + CACHE_വരി_SIZE_F32) * ith;

    വേണ്ടി (പൂർണ്ണസംഖ്യ ir = ir0; ir < ir1; ++ir) {
        // src0 indices
        സ്ഥിരം പൂർണ്ണസംഖ്യ i03 = ir/(ne02*ne01);
        സ്ഥിരം പൂർണ്ണസംഖ്യ i02 = (ir - i03*ne02*ne01)/ne01;
        സ്ഥിരം പൂർണ്ണസംഖ്യ i01 = (ir - i03*ne02*ne01 - i02*ne01);

        // src1 and dst are same shape as src0 => same indices
        സ്ഥിരം പൂർണ്ണസംഖ്യ i13 = i03;
        സ്ഥിരം പൂർണ്ണസംഖ്യ i12 = i02;
        സ്ഥിരം പൂർണ്ണസംഖ്യ i11 = i01;

        സ്ഥിരം പൂർണ്ണസംഖ്യ i3 = i03;
        സ്ഥിരം പൂർണ്ണസംഖ്യ i2 = i02;
        സ്ഥിരം പൂർണ്ണസംഖ്യ i1 = i01;

        ശൂന്യം  * src0_row = (ശൂന്യം *) ((അക്ഷരം *) src0->data + (i01*nb01 + i02*nb02 + i03*nb03));
        ദശാംശം * src1_row = (ദശാംശം *)((അക്ഷരം *) src1->data + (i11*nb11 + i12*nb12 + i13*nb13));
        ശൂന്യം  * dst_row  = (ശൂന്യം *) ((അക്ഷരം *)  dst->data + ( i1*nb1  +  i2*nb2  +  i3*nb3));

        പ്രതിജ്ഞാ(ne00 % 32 == 0);

        // unquantize row from src0 to temp buffer
        dequantize_row_q(src0_row, wdata, ne00);
        // add src1
        ggml_vec_acc_f32(ne00, wdata, src1_row);
        // quantize row to dst
        എങ്കിൽ (quantize_row_q != ശൂന്യ) {
            quantize_row_q(wdata, dst_row, ne00);
        } അല്ലെങ്കിൽ {
            memcpy(dst_row, wdata, ne0*nb0);
        }
    }
}

ശൂന്യം ggml_compute_forward_add(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
        അവസ്ഥ GGML_TYPE_F16:
        അവസ്ഥ GGML_TYPE_BF16:
            {
                ggml_compute_forward_add_non_quantized(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_Q1_0:
        അവസ്ഥ GGML_TYPE_Q2_0:
        അവസ്ഥ GGML_TYPE_Q4_0:
        അവസ്ഥ GGML_TYPE_Q4_1:
        അവസ്ഥ GGML_TYPE_Q5_0:
        അവസ്ഥ GGML_TYPE_Q5_1:
        അവസ്ഥ GGML_TYPE_Q8_0:
        അവസ്ഥ GGML_TYPE_MXFP4:
        അവസ്ഥ GGML_TYPE_NVFP4:
        അവസ്ഥ GGML_TYPE_Q2_K:
        അവസ്ഥ GGML_TYPE_Q3_K:
        അവസ്ഥ GGML_TYPE_Q4_K:
        അവസ്ഥ GGML_TYPE_Q5_K:
        അവസ്ഥ GGML_TYPE_Q6_K:
        അവസ്ഥ GGML_TYPE_TQ1_0:
        അവസ്ഥ GGML_TYPE_TQ2_0:
        അവസ്ഥ GGML_TYPE_IQ2_XXS:
        അവസ്ഥ GGML_TYPE_IQ2_XS:
        അവസ്ഥ GGML_TYPE_IQ3_XXS:
        അവസ്ഥ GGML_TYPE_IQ1_S:
        അവസ്ഥ GGML_TYPE_IQ1_M:
        അവസ്ഥ GGML_TYPE_IQ4_NL:
        അവസ്ഥ GGML_TYPE_IQ4_XS:
        അവസ്ഥ GGML_TYPE_IQ3_S:
        അവസ്ഥ GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_add_q_f32(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_add_id

സ്ഥിതം ശൂന്യം ggml_compute_forward_add_id_f32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];
    സ്ഥിരം ggml_tensor * src2 = dst->src[2];

    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src2->type == GGML_TYPE_I32);

    GGML_ASSERT(src0->nb[0] == വലുപ്പം(ദശാംശം));
    GGML_ASSERT(src1->nb[0] == വലുപ്പം(ദശാംശം));

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ nr  = ggml_nrows(src0);

    GGML_TENSOR_TERNARY_OP_LOCALS

    GGML_ASSERT( nb0 == വലുപ്പം(ദശാംശം));
    GGML_ASSERT(nb10 == വലുപ്പം(ദശാംശം));

    // rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    വേണ്ടി (പൂർണ്ണസംഖ്യ ir = ir0; ir < ir1; ++ir) {
        // src0 indices
        സ്ഥിരം പൂർണ്ണസംഖ്യ i3 = ir/(ne2*ne1);
        സ്ഥിരം പൂർണ്ണസംഖ്യ i2 = (ir - i3*ne2*ne1)/ne1;
        സ്ഥിരം പൂർണ്ണസംഖ്യ i1 = (ir - i3*ne2*ne1 - i2*ne1);

        // src1 indices
        സ്ഥിരം പൂർണ്ണസംഖ്യ i11 = *(int32_t *) ((അക്ഷരം *) src2->data + i1*nb20 + i2*nb21);

        GGML_ASSERT(i11 >= 0 && i11 < ne11);

        ggml_vec_add_f32(ne0,
                (ദശാംശം *) ((അക്ഷരം *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 ),
                (ദശാംശം *) ((അക്ഷരം *) src0->data + i3*nb03 + i2*nb02 + i1*nb01),
                (ദശാംശം *) ((അക്ഷരം *) src1->data + i11*nb11));
    }
}

ശൂന്യം ggml_compute_forward_add_id(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_add_id_f32(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("unsupported type for ggml_compute_forward_add_id: %s", ggml_type_name(src0->type));
            }
    }
}

// ggml_compute_forward_add1

സ്ഥിതം ശൂന്യം ggml_compute_forward_add1_f32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == വലുപ്പം(ദശാംശം));
    GGML_ASSERT(nb00 == വലുപ്പം(ദശാംശം));

    // rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    വേണ്ടി (പൂർണ്ണസംഖ്യ ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        സ്ഥിരം പൂർണ്ണസംഖ്യ i3 = ir/(ne2*ne1);
        സ്ഥിരം പൂർണ്ണസംഖ്യ i2 = (ir - i3*ne2*ne1)/ne1;
        സ്ഥിരം പൂർണ്ണസംഖ്യ i1 = (ir - i3*ne2*ne1 - i2*ne1);

#ifdef GGML_USE_ACCELERATE
        GGML_UNUSED(ggml_vec_add1_f32);

        vDSP_vadd(
                (ദശാംശം *) ((അക്ഷരം *) src0->data + i3*nb03 + i2*nb02 + i1*nb01), 1,
                (ദശാംശം *) ((അക്ഷരം *) src1->data), 0,
                (ദശാംശം *) ((അക്ഷരം *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 ), 1,
                ne0);
#അല്ലെങ്കിൽ
        ggml_vec_add1_f32(ne0,
                (ദശാംശം *) ((അക്ഷരം *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 ),
                (ദശാംശം *) ((അക്ഷരം *) src0->data + i3*nb03 + i2*nb02 + i1*nb01),
               *(ദശാംശം *) src1->data);
#endif
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_add1_f16_f32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    സ്ഥിരം ദശാംശം v = *(ദശാംശം *) src1->data;

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F16);

    GGML_ASSERT( nb0 == വലുപ്പം(ggml_fp16_t));
    GGML_ASSERT(nb00 == വലുപ്പം(ggml_fp16_t));

    // rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    വേണ്ടി (പൂർണ്ണസംഖ്യ ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        സ്ഥിരം പൂർണ്ണസംഖ്യ i3 = ir/(ne2*ne1);
        സ്ഥിരം പൂർണ്ണസംഖ്യ i2 = (ir - i3*ne2*ne1)/ne1;
        സ്ഥിരം പൂർണ്ണസംഖ്യ i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((അക്ഷരം *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((അക്ഷരം *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(src0_ptr[i]) + v);
        }
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_add1_f16_f16(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    സ്ഥിരം ദശാംശം v = GGML_CPU_FP16_TO_FP32(*(ggml_fp16_t *) src1->data);

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type  == GGML_TYPE_F16);

    GGML_ASSERT( nb0 == വലുപ്പം(ggml_fp16_t));
    GGML_ASSERT(nb00 == വലുപ്പം(ggml_fp16_t));

    // rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    വേണ്ടി (പൂർണ്ണസംഖ്യ ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        സ്ഥിരം പൂർണ്ണസംഖ്യ i3 = ir/(ne2*ne1);
        സ്ഥിരം പൂർണ്ണസംഖ്യ i2 = (ir - i3*ne2*ne1)/ne1;
        സ്ഥിരം പൂർണ്ണസംഖ്യ i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((അക്ഷരം *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((അക്ഷരം *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(src0_ptr[i]) + v);
        }
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_add1_q_f32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    സ്ഥിരം ദശാംശം v = *(ദശാംശം *) src1->data;

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    സ്ഥിരം ggml_type type = src0->type;
    ggml_to_float_t സ്ഥിരം dequantize_row_q = ggml_get_type_traits(type)->to_float;
    ggml_from_float_t സ്ഥിരം quantize_row_q = ggml_get_type_traits_cpu(type)->from_float;

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
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    ദശാംശം * wdata = (ദശാംശം *) params->wdata + (ne0 + CACHE_വരി_SIZE_F32) * ith;

    വേണ്ടി (പൂർണ്ണസംഖ്യ ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        സ്ഥിരം പൂർണ്ണസംഖ്യ i3 = ir/(ne2*ne1);
        സ്ഥിരം പൂർണ്ണസംഖ്യ i2 = (ir - i3*ne2*ne1)/ne1;
        സ്ഥിരം പൂർണ്ണസംഖ്യ i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ശൂന്യം  * src0_row = (ശൂന്യം *) ((അക്ഷരം *) src0->data + (i1*nb01 + i2*nb02 + i3*nb03));
        ശൂന്യം  * dst_row  = (ശൂന്യം *) ((അക്ഷരം *)  dst->data + (i1*nb1  + i2*nb2  + i3*nb0 ));

        പ്രതിജ്ഞാ(ne0 % 32 == 0);

        // unquantize row from src0 to temp buffer
        dequantize_row_q(src0_row, wdata, ne0);
        // add src1
        ggml_vec_acc1_f32(ne0, wdata, v);
        // quantize row to dst
        quantize_row_q(wdata, dst_row, ne0);
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_add1_bf16_f32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    സ്ഥിരം ദശാംശം v = *(ദശാംശം *) src1->data;

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_BF16);

    GGML_ASSERT( nb0 == വലുപ്പം(ggml_bf16_t));
    GGML_ASSERT(nb00 == വലുപ്പം(ggml_bf16_t));

    // rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    വേണ്ടി (പൂർണ്ണസംഖ്യ ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        സ്ഥിരം പൂർണ്ണസംഖ്യ i3 = ir/(ne2*ne1);
        സ്ഥിരം പൂർണ്ണസംഖ്യ i2 = (ir - i3*ne2*ne1)/ne1;
        സ്ഥിരം പൂർണ്ണസംഖ്യ i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_bf16_t * dst_ptr  = (ggml_bf16_t *) ((അക്ഷരം *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_bf16_t * src0_ptr = (ggml_bf16_t *) ((അക്ഷരം *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(src0_ptr[i]) + v);
        }
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_add1_bf16_bf16(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    സ്ഥിരം ദശാംശം v = GGML_BF16_TO_FP32(*(ggml_bf16_t *) src1->data);

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(src1->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->type  == GGML_TYPE_BF16);

    GGML_ASSERT( nb0 == വലുപ്പം(ggml_bf16_t));
    GGML_ASSERT(nb00 == വലുപ്പം(ggml_bf16_t));

    // rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    വേണ്ടി (പൂർണ്ണസംഖ്യ ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        സ്ഥിരം പൂർണ്ണസംഖ്യ i3 = ir/(ne2*ne1);
        സ്ഥിരം പൂർണ്ണസംഖ്യ i2 = (ir - i3*ne2*ne1)/ne1;
        സ്ഥിരം പൂർണ്ണസംഖ്യ i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_bf16_t * dst_ptr  = (ggml_bf16_t *) ((അക്ഷരം *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_bf16_t * src0_ptr = (ggml_bf16_t *) ((അക്ഷരം *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(src0_ptr[i]) + v);
        }
    }
}

ശൂന്യം ggml_compute_forward_add1(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_add1_f32(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F16:
            {
                എങ്കിൽ (src1->type == GGML_TYPE_F16) {
                    ggml_compute_forward_add1_f16_f16(params, dst);
                }
                അല്ലെങ്കിൽ എങ്കിൽ (src1->type == GGML_TYPE_F32) {
                    ggml_compute_forward_add1_f16_f32(params, dst);
                }
                അല്ലെങ്കിൽ {
                    GGML_ABORT("fatal error");
                }
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_BF16:
            {
                എങ്കിൽ (src1->type == GGML_TYPE_BF16) {
                    ggml_compute_forward_add1_bf16_bf16(params, dst);
                }
                അല്ലെങ്കിൽ എങ്കിൽ (src1->type == GGML_TYPE_F32) {
                    ggml_compute_forward_add1_bf16_f32(params, dst);
                }
                അല്ലെങ്കിൽ {
                    GGML_ABORT("fatal error");
                }
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_Q1_0:
        അവസ്ഥ GGML_TYPE_Q2_0:
        അവസ്ഥ GGML_TYPE_Q4_0:
        അവസ്ഥ GGML_TYPE_Q4_1:
        അവസ്ഥ GGML_TYPE_Q5_0:
        അവസ്ഥ GGML_TYPE_Q5_1:
        അവസ്ഥ GGML_TYPE_Q8_0:
        അവസ്ഥ GGML_TYPE_Q8_1:
        അവസ്ഥ GGML_TYPE_MXFP4:
        അവസ്ഥ GGML_TYPE_NVFP4:
        അവസ്ഥ GGML_TYPE_Q2_K:
        അവസ്ഥ GGML_TYPE_Q3_K:
        അവസ്ഥ GGML_TYPE_Q4_K:
        അവസ്ഥ GGML_TYPE_Q5_K:
        അവസ്ഥ GGML_TYPE_Q6_K:
        അവസ്ഥ GGML_TYPE_TQ1_0:
        അവസ്ഥ GGML_TYPE_TQ2_0:
        അവസ്ഥ GGML_TYPE_IQ2_XXS:
        അവസ്ഥ GGML_TYPE_IQ2_XS:
        അവസ്ഥ GGML_TYPE_IQ3_XXS:
        അവസ്ഥ GGML_TYPE_IQ1_S:
        അവസ്ഥ GGML_TYPE_IQ1_M:
        അവസ്ഥ GGML_TYPE_IQ4_NL:
        അവസ്ഥ GGML_TYPE_IQ4_XS:
        അവസ്ഥ GGML_TYPE_IQ3_S:
        അവസ്ഥ GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_add1_q_f32(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_acc

സ്ഥിതം ശൂന്യം ggml_compute_forward_acc_f32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));

    // view src0 and dst with these strides and data offset inbytes during acc
    // nb0 is implicitly element_size because src0 and dst are contiguous
    size_t nb1     = ((int32_t *) dst->op_params)[0];
    size_t nb2     = ((int32_t *) dst->op_params)[1];
    size_t nb3     = ((int32_t *) dst->op_params)[2];
    size_t offset  = ((int32_t *) dst->op_params)[3];
    തർക്കം   inplace = (തർക്കം) ((int32_t *) dst->op_params)[4];

    എങ്കിൽ (!inplace) {
        എങ്കിൽ (params->ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            memcpy(
                ((അക്ഷരം *)  dst->data),
                ((അക്ഷരം *) src0->data),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ nr = ggml_nrows(src1);
    സ്ഥിരം പൂർണ്ണസംഖ്യ nc = src1->ne[0];

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne)
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb)

    // src0 and dst as viewed during acc
    സ്ഥിരം size_t nb0 = ggml_element_size(src0);

    സ്ഥിരം size_t nb00 = nb0;
    സ്ഥിരം size_t nb01 = nb1;
    സ്ഥിരം size_t nb02 = nb2;
    സ്ഥിരം size_t nb03 = nb3;

    GGML_ASSERT(offset + (ne10 == 0 ? 0 : ne10-1)*nb0  + (ne11 == 0 ? 0 : ne11-1)*nb1  + (ne12 == 0 ? 0 : ne12-1)*nb2  + (ne13 == 0 ? 0 : ne13-1)*nb3  < ggml_nbytes(dst));
    GGML_ASSERT(offset + (ne10 == 0 ? 0 : ne10-1)*nb00 + (ne11 == 0 ? 0 : ne11-1)*nb01 + (ne12 == 0 ? 0 : ne12-1)*nb02 + (ne13 == 0 ? 0 : ne13-1)*nb03 < ggml_nbytes(src0));

    GGML_ASSERT(nb10 == വലുപ്പം(ദശാംശം));

    // rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    വേണ്ടി (പൂർണ്ണസംഖ്യ ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are viewed with shape of src1 and offset
        // => same indices
        സ്ഥിരം പൂർണ്ണസംഖ്യ i3 = ir/(ne12*ne11);
        സ്ഥിരം പൂർണ്ണസംഖ്യ i2 = (ir - i3*ne12*ne11)/ne11;
        സ്ഥിരം പൂർണ്ണസംഖ്യ i1 = (ir - i3*ne12*ne11 - i2*ne11);

#ifdef GGML_USE_ACCELERATE
        vDSP_vadd(
                (ദശാംശം *) ((അക്ഷരം *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + offset), 1,
                (ദശാംശം *) ((അക്ഷരം *) src1->data + i3*nb13 + i2*nb12 + i1*nb11), 1,
                (ദശാംശം *) ((അക്ഷരം *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1  + offset), 1, nc);
#അല്ലെങ്കിൽ
        ggml_vec_add_f32(nc,
                (ദശാംശം *) ((അക്ഷരം *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + offset),
                (ദശാംശം *) ((അക്ഷരം *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + offset),
                (ദശാംശം *) ((അക്ഷരം *) src1->data + i3*nb13 + i2*nb12 + i1*nb11));
#endif
    }
}

ശൂന്യം ggml_compute_forward_acc(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_acc_f32(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F16:
        അവസ്ഥ GGML_TYPE_BF16:
        അവസ്ഥ GGML_TYPE_Q1_0:
        അവസ്ഥ GGML_TYPE_Q2_0:
        അവസ്ഥ GGML_TYPE_Q4_0:
        അവസ്ഥ GGML_TYPE_Q4_1:
        അവസ്ഥ GGML_TYPE_Q5_0:
        അവസ്ഥ GGML_TYPE_Q5_1:
        അവസ്ഥ GGML_TYPE_Q8_0:
        അവസ്ഥ GGML_TYPE_Q8_1:
        അവസ്ഥ GGML_TYPE_MXFP4:
        അവസ്ഥ GGML_TYPE_NVFP4:
        അവസ്ഥ GGML_TYPE_Q2_K:
        അവസ്ഥ GGML_TYPE_Q3_K:
        അവസ്ഥ GGML_TYPE_Q4_K:
        അവസ്ഥ GGML_TYPE_Q5_K:
        അവസ്ഥ GGML_TYPE_Q6_K:
        അവസ്ഥ GGML_TYPE_TQ1_0:
        അവസ്ഥ GGML_TYPE_TQ2_0:
        അവസ്ഥ GGML_TYPE_IQ2_XXS:
        അവസ്ഥ GGML_TYPE_IQ2_XS:
        അവസ്ഥ GGML_TYPE_IQ3_XXS:
        അവസ്ഥ GGML_TYPE_IQ1_S:
        അവസ്ഥ GGML_TYPE_IQ1_M:
        അവസ്ഥ GGML_TYPE_IQ4_NL:
        അവസ്ഥ GGML_TYPE_IQ4_XS:
        അവസ്ഥ GGML_TYPE_IQ3_S:
        അവസ്ഥ GGML_TYPE_IQ2_S:
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_sum

സ്ഥിതം ശൂന്യം ggml_compute_forward_sum_f32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    എങ്കിൽ (params->ith != 0) {
        തിരികെ;
    }

    പ്രതിജ്ഞാ(ggml_is_scalar(dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)

    ggml_float sum = 0;

    എങ്കിൽ (nb00 == വലുപ്പം(ദശാംശം)) {
        ggml_float row_sum = 0;
        വേണ്ടി (int64_t i03 = 0; i03 < ne03; i03++) {
            വേണ്ടി (int64_t i02 = 0; i02 < ne02; i02++) {
                വേണ്ടി (int64_t i01 = 0; i01 < ne01; i01++) {
                    ggml_vec_sum_f32_ggf(ne00,
                            &row_sum,
                            (ദശാംശം *) ((അക്ഷരം *) src0->data + i01*nb01 + i02*nb02 + i03*nb03));
                    sum += row_sum;
                }
            }
        }
    } അല്ലെങ്കിൽ {
        വേണ്ടി (int64_t i03 = 0; i03 < ne03; i03++) {
            വേണ്ടി (int64_t i02 = 0; i02 < ne02; i02++) {
                വേണ്ടി (int64_t i01 = 0; i01 < ne01; i01++) {
                    സ്ഥിരം അക്ഷരം * row = (സ്ഥിരം അക്ഷരം *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
                    വേണ്ടി (int64_t i00 = 0; i00 < ne00; i00++) {
                        sum += (ggml_float) *(സ്ഥിരം ദശാംശം *) (row + i00*nb00);
                    }
                }
            }
        }
    }

    ((ദശാംശം *) dst->data)[0] = sum;
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_sum_f16(
    സ്ഥിരം ggml_compute_params * params,
          ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    എങ്കിൽ (params->ith != 0) {
        തിരികെ;
    }

    പ്രതിജ്ഞാ(ggml_is_scalar(dst));

    പ്രതിജ്ഞാ(src0->nb[0] == വലുപ്പം(ggml_fp16_t));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)

    ദശാംശം sum = 0;
    ദശാംശം row_sum = 0;

    വേണ്ടി (int64_t i03 = 0; i03 < ne03; i03++) {
        വേണ്ടി (int64_t i02 = 0; i02 < ne02; i02++) {
            വേണ്ടി (int64_t i01 = 0; i01 < ne01; i01++) {
                ggml_vec_sum_f16_ggf(ne00,
                    &row_sum,
                    (ggml_fp16_t *) ((അക്ഷരം *) src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03));
                sum += row_sum;
            }
        }
    }
    ((ggml_fp16_t *) dst->data)[0] = GGML_CPU_FP32_TO_FP16(sum);
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_sum_bf16(
    സ്ഥിരം ggml_compute_params * params,
          ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    എങ്കിൽ (params->ith != 0) {
        തിരികെ;
    }

    പ്രതിജ്ഞാ(ggml_is_scalar(dst));

    പ്രതിജ്ഞാ(src0->nb[0] == വലുപ്പം(ggml_bf16_t));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)

    ദശാംശം sum = 0;
    ദശാംശം row_sum = 0;

    വേണ്ടി (int64_t i03 = 0; i03 < ne03; i03++) {
        വേണ്ടി (int64_t i02 = 0; i02 < ne02; i02++) {
            വേണ്ടി (int64_t i01 = 0; i01 < ne01; i01++) {
                ggml_vec_sum_bf16_ggf(ne00,
                    &row_sum,
                    (ggml_bf16_t *) ((അക്ഷരം *) src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03));
                sum += row_sum;
            }
        }
    }
    ((ggml_bf16_t *) dst->data)[0] = GGML_FP32_TO_BF16(sum);
}

ശൂന്യം ggml_compute_forward_sum(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_sum_f32(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F16:
            {
                ggml_compute_forward_sum_f16(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_BF16:
            {
                ggml_compute_forward_sum_bf16(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_cumsum

സ്ഥിതം ശൂന്യം ggml_compute_forward_cumsum_f32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->nb[0] == വലുപ്പം(ദശാംശം));
    GGML_ASSERT(dst->nb[0] == വലുപ്പം(ദശാംശം));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne01);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    സ്ഥിരം auto [ir0, ir1] = get_thread_range(params, src0);

    വേണ്ടി (int64_t ir = ir0; ir < ir1; ++ir) {
        സ്ഥിരം int64_t i03 = ir/(ne02*ne01);
        സ്ഥിരം int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        സ്ഥിരം int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        ദശാംശം * src_row = (ദശാംശം *) ((അക്ഷരം *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
        ദശാംശം * dst_row = (ദശാംശം *) ((അക്ഷരം *) dst->data  + i01*nb1  + i02*nb2  + i03*nb3);

        ggml_vec_cumsum_f32(ne00, dst_row, src_row);
    }
}

ശൂന്യം ggml_compute_forward_cumsum(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_cumsum_f32(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_sum_rows

സ്ഥിതം ശൂന്യം ggml_compute_forward_sum_rows_f32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    എങ്കിൽ (params->ith != 0) {
        തിരികെ;
    }

    GGML_ASSERT(src0->nb[0] == വലുപ്പം(ദശാംശം));
    GGML_ASSERT(dst->nb[0] == വലുപ്പം(ദശാംശം));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(ne0 == 1);
    GGML_ASSERT(ne1 == ne01);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    വേണ്ടി (int64_t i3 = 0; i3 < ne03; i3++) {
        വേണ്ടി (int64_t i2 = 0; i2 < ne02; i2++) {
            വേണ്ടി (int64_t i1 = 0; i1 < ne01; i1++) {
                ദശാംശം * src_row = (ദശാംശം *) ((അക്ഷരം *) src0->data + i1*nb01 + i2*nb02 + i3*nb03);
                ദശാംശം * dst_row = (ദശാംശം *) ((അക്ഷരം *) dst->data  + i1*nb1  + i2*nb2  + i3*nb3);
                ദശാംശം row_sum = 0;
                ggml_vec_sum_f32(ne00, &row_sum, src_row);
                dst_row[0] = row_sum;
            }
        }
    }
}

ശൂന്യം ggml_compute_forward_sum_rows(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_sum_rows_f32(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_mean

സ്ഥിതം ശൂന്യം ggml_compute_forward_mean_f32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    എങ്കിൽ (params->ith != 0) {
        തിരികെ;
    }

    പ്രതിജ്ഞാ(src0->nb[0] == വലുപ്പം(ദശാംശം));

    GGML_TENSOR_UNARY_OP_LOCALS

    പ്രതിജ്ഞാ(ne0 == 1);
    പ്രതിജ്ഞാ(ne1 == ne01);
    പ്രതിജ്ഞാ(ne2 == ne02);
    പ്രതിജ്ഞാ(ne3 == ne03);

    GGML_UNUSED(ne0);
    GGML_UNUSED(ne1);
    GGML_UNUSED(ne2);
    GGML_UNUSED(ne3);

    വേണ്ടി (int64_t i03 = 0; i03 < ne03; i03++) {
        വേണ്ടി (int64_t i02 = 0; i02 < ne02; i02++) {
            വേണ്ടി (int64_t i01 = 0; i01 < ne01; i01++) {
                ggml_vec_sum_f32(ne00,
                        (ദശാംശം *) ((അക്ഷരം *)  dst->data + i01*nb1  + i02*nb2  + i03*nb3),
                        (ദശാംശം *) ((അക്ഷരം *) src0->data + i01*nb01 + i02*nb02 + i03*nb03));

                *(ദശാംശം *) ((അക്ഷരം *) dst->data + i01*nb1 + i02*nb2 + i03*nb3) /= (ദശാംശം) ne00;
            }
        }
    }
}

ശൂന്യം ggml_compute_forward_mean(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_mean_f32(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_argmax

സ്ഥിതം ശൂന്യം ggml_compute_forward_argmax_f32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    എങ്കിൽ (params->ith != 0) {
        തിരികെ;
    }

    പ്രതിജ്ഞാ(src0->nb[0] == വലുപ്പം(ദശാംശം));
    പ്രതിജ്ഞാ(dst->nb[0] == വലുപ്പം(ദശാംശം));

    സ്ഥിരം int64_t ne00 = src0->ne[0];
    സ്ഥിരം int64_t ne01 = src0->ne[1];

    സ്ഥിരം size_t nb01 = src0->nb[1];
    സ്ഥിരം size_t nb0 = dst->nb[0];

    വേണ്ടി (int64_t i1 = 0; i1 < ne01; i1++) {
        ദശാംശം * src = (ദശാംശം *) ((അക്ഷരം *) src0->data + i1*nb01);
        int32_t * dst_ = (int32_t *) ((അക്ഷരം *)  dst->data + i1*nb0);
        പൂർണ്ണസംഖ്യ v = 0;
        ggml_vec_argmax_f32(ne00, &v, src);
        dst_[0] = v;
    }
}

ശൂന്യം ggml_compute_forward_argmax(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_argmax_f32(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_count_equal

സ്ഥിതം ശൂന്യം ggml_compute_forward_എണ്ണം_equal_i32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ദ്വിമാന_OP_LOCALS;

    GGML_ASSERT(src0->type == GGML_TYPE_I32);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));
    GGML_ASSERT(dst->type == GGML_TYPE_I64);

    സ്ഥിരം int64_t nr = ggml_nrows(src0);

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    int64_t * sums = (int64_t *) params->wdata;
    int64_t sum_thread = 0;

    // rows per thread
    സ്ഥിരം int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം int64_t ir0 = dr*ith;
    സ്ഥിരം int64_t ir1 = MIN(ir0 + dr, nr);

    വേണ്ടി (int64_t ir = ir0; ir < ir1; ++ir) {
        സ്ഥിരം int64_t i03 =  ir                        / (ne02*ne01);
        സ്ഥിരം int64_t i02 = (ir - i03*ne03)            /       ne01;
        സ്ഥിരം int64_t i01 =  ir - i03*ne03 - i02*ne02;

        സ്ഥിരം അക്ഷരം * data0 = (സ്ഥിരം അക്ഷരം *) src0->data + i03*nb03 + i02*nb02 + i01*nb01;
        സ്ഥിരം അക്ഷരം * data1 = (സ്ഥിരം അക്ഷരം *) src1->data + i03*nb13 + i02*nb12 + i01*nb11;

        വേണ്ടി (int64_t i00 = 0; i00 < ne00; ++i00) {
            സ്ഥിരം int32_t val0 = *((സ്ഥിരം int32_t *) (data0 + i00*nb00));
            സ്ഥിരം int32_t val1 = *((സ്ഥിരം int32_t *) (data1 + i00*nb10));

            sum_thread += val0 == val1;
        }
    }
    എങ്കിൽ (ith != 0) {
        sums[ith] = sum_thread;
    }
    ggml_barrier(params->threadpool);

    എങ്കിൽ (ith != 0) {
        തിരികെ;
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ ith_other = 1; ith_other < nth; ++ith_other) {
        sum_thread += sums[ith_other];
    }
    *((int64_t *) dst->data) = sum_thread;
}

ശൂന്യം ggml_compute_forward_എണ്ണം_equal(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_I32:
            {
                ggml_compute_forward_എണ്ണം_equal_i32(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_repeat

സ്ഥിതം ശൂന്യം ggml_compute_forward_repeat_f32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    എങ്കിൽ (params->ith != 0) {
        തിരികെ;
    }

    GGML_ASSERT(ggml_can_repeat(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    // guaranteed to be an integer due to the check in ggml_can_repeat
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr0 = (പൂർണ്ണസംഖ്യ)(ne0/ne00);
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr1 = (പൂർണ്ണസംഖ്യ)(ne1/ne01);
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr2 = (പൂർണ്ണസംഖ്യ)(ne2/ne02);
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr3 = (പൂർണ്ണസംഖ്യ)(ne3/ne03);

    // TODO: support for transposed / permuted tensors
    GGML_ASSERT(nb0  == വലുപ്പം(ദശാംശം));
    GGML_ASSERT(nb00 == വലുപ്പം(ദശാംശം));

    // TODO: maybe this is not optimal?
    വേണ്ടി                         (പൂർണ്ണസംഖ്യ i3 = 0; i3 < nr3;  i3++) {
        വേണ്ടി                     (പൂർണ്ണസംഖ്യ k3 = 0; k3 < ne03; k3++) {
            വേണ്ടി                 (പൂർണ്ണസംഖ്യ i2 = 0; i2 < nr2;  i2++) {
                വേണ്ടി             (പൂർണ്ണസംഖ്യ k2 = 0; k2 < ne02; k2++) {
                    വേണ്ടി         (പൂർണ്ണസംഖ്യ i1 = 0; i1 < nr1;  i1++) {
                        വേണ്ടി     (പൂർണ്ണസംഖ്യ k1 = 0; k1 < ne01; k1++) {
                            വേണ്ടി (പൂർണ്ണസംഖ്യ i0 = 0; i0 < nr0;  i0++) {
                                ggml_vec_cpy_f32(ne00,
                                        (ദശാംശം *) ((അക്ഷരം *)  dst->data + (i3*ne03 + k3)*nb3  + (i2*ne02 + k2)*nb2  + (i1*ne01 + k1)*nb1  + (i0*ne00)*nb0),
                                        (ദശാംശം *) ((അക്ഷരം *) src0->data + (          k3)*nb03 + (          k2)*nb02 + (          k1)*nb01));
                            }
                        }
                    }
                }
            }
        }
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_repeat_f16(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    എങ്കിൽ (params->ith != 0) {
        തിരികെ;
    }

    GGML_ASSERT(ggml_can_repeat(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    // guaranteed to be an integer due to the check in ggml_can_repeat
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr0 = (പൂർണ്ണസംഖ്യ)(ne0/ne00);
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr1 = (പൂർണ്ണസംഖ്യ)(ne1/ne01);
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr2 = (പൂർണ്ണസംഖ്യ)(ne2/ne02);
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr3 = (പൂർണ്ണസംഖ്യ)(ne3/ne03);

    // TODO: support for transposed / permuted tensors
    GGML_ASSERT(nb0  == വലുപ്പം(ggml_fp16_t));
    GGML_ASSERT(nb00 == വലുപ്പം(ggml_fp16_t));

    // TODO: maybe this is not optimal?
    വേണ്ടി                         (പൂർണ്ണസംഖ്യ i3 = 0; i3 < nr3;  i3++) {
        വേണ്ടി                     (പൂർണ്ണസംഖ്യ k3 = 0; k3 < ne03; k3++) {
            വേണ്ടി                 (പൂർണ്ണസംഖ്യ i2 = 0; i2 < nr2;  i2++) {
                വേണ്ടി             (പൂർണ്ണസംഖ്യ k2 = 0; k2 < ne02; k2++) {
                    വേണ്ടി         (പൂർണ്ണസംഖ്യ i1 = 0; i1 < nr1;  i1++) {
                        വേണ്ടി     (പൂർണ്ണസംഖ്യ k1 = 0; k1 < ne01; k1++) {
                            വേണ്ടി (പൂർണ്ണസംഖ്യ i0 = 0; i0 < nr0;  i0++) {
                                ggml_fp16_t * y = (ggml_fp16_t *) ((അക്ഷരം *)  dst->data + (i3*ne03 + k3)*nb3  + (i2*ne02 + k2)*nb2  + (i1*ne01 + k1)*nb1  + (i0*ne00)*nb0);
                                ggml_fp16_t * x = (ggml_fp16_t *) ((അക്ഷരം *) src0->data + (          k3)*nb03 + (          k2)*nb02 + (          k1)*nb01);
                                // ggml_vec_cpy_f16(ne00, y, x)
                                വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < ne00; ++i) {
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

ശൂന്യം ggml_compute_forward_repeat(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F16:
        അവസ്ഥ GGML_TYPE_BF16:
        അവസ്ഥ GGML_TYPE_I16:
            {
                ggml_compute_forward_repeat_f16(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F32:
        അവസ്ഥ GGML_TYPE_I32:
            {
                ggml_compute_forward_repeat_f32(params, dst);
            } നിർത്തുക;
        // TODO: templateify the implementation and support for I64
        //       ref https://github.com/ggml-org/llama.cpp/pull/14274#discussion_r2169492225
        //case GGML_TYPE_I64:
        //    {
        //        ggml_compute_forward_repeat_i64(params, dst);
        //    } break;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_repeat_back

സ്ഥിതം ശൂന്യം ggml_compute_forward_repeat_back_f32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    എങ്കിൽ (params->ith != 0) {
        തിരികെ;
    }

    GGML_ASSERT(ggml_can_repeat(dst, src0));

    GGML_TENSOR_UNARY_OP_LOCALS

    // guaranteed to be an integer due to the check in ggml_can_repeat
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr0 = (പൂർണ്ണസംഖ്യ)(ne00/ne0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr1 = (പൂർണ്ണസംഖ്യ)(ne01/ne1);
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr2 = (പൂർണ്ണസംഖ്യ)(ne02/ne2);
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr3 = (പൂർണ്ണസംഖ്യ)(ne03/ne3);

    // TODO: support for transposed / permuted tensors
    GGML_ASSERT(nb0  == വലുപ്പം(ദശാംശം));
    GGML_ASSERT(nb00 == വലുപ്പം(ദശാംശം));

    എങ്കിൽ (ggml_is_contiguous(dst)) {
        ggml_vec_set_f32(ne0*ne1*ne2*ne3, (ദശാംശം *)dst->data, 0);
    } അല്ലെങ്കിൽ {
        വേണ്ടി         (പൂർണ്ണസംഖ്യ k3 = 0; k3 < ne3; k3++) {
            വേണ്ടി     (പൂർണ്ണസംഖ്യ k2 = 0; k2 < ne2; k2++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ k1 = 0; k1 < ne1; k1++) {
                    ggml_vec_set_f32(ne0,
                        (ദശാംശം *) ((അക്ഷരം *) dst->data + k1*nb1 + k2*nb2 + k3*nb3),
                        0);
                }
            }
        }
    }

    // TODO: maybe this is not optimal?
    വേണ്ടി                         (പൂർണ്ണസംഖ്യ i3 = 0; i3 < nr3; i3++) {
        വേണ്ടി                     (പൂർണ്ണസംഖ്യ k3 = 0; k3 < ne3; k3++) {
            വേണ്ടി                 (പൂർണ്ണസംഖ്യ i2 = 0; i2 < nr2; i2++) {
                വേണ്ടി             (പൂർണ്ണസംഖ്യ k2 = 0; k2 < ne2; k2++) {
                    വേണ്ടി         (പൂർണ്ണസംഖ്യ i1 = 0; i1 < nr1; i1++) {
                        വേണ്ടി     (പൂർണ്ണസംഖ്യ k1 = 0; k1 < ne1; k1++) {
                            വേണ്ടി (പൂർണ്ണസംഖ്യ i0 = 0; i0 < nr0; i0++) {
                                ggml_vec_acc_f32(ne0,
                                        (ദശാംശം *) ((അക്ഷരം *)  dst->data + (         k3)*nb3  + (         k2)*nb2  + (         k1)*nb1),
                                        (ദശാംശം *) ((അക്ഷരം *) src0->data + (i3*ne3 + k3)*nb03 + (i2*ne2 + k2)*nb02 + (i1*ne1 + k1)*nb01 + (i0*ne0)*nb00));
                            }
                        }
                    }
                }
            }
        }
    }
}

ശൂന്യം ggml_compute_forward_repeat_back(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_repeat_back_f32(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_concat
സ്ഥിതം ശൂന്യം ggml_compute_forward_concat_any(
    സ്ഥിരം ggml_compute_params * params,
    ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    GGML_TENSOR_ദ്വിമാന_OP_LOCALS

    സ്ഥിരം int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);
    GGML_ASSERT(ggml_is_contiguous_rows(src0));
    GGML_ASSERT(ggml_is_contiguous_rows(src1));

    int64_t o[4] = {0, 0, 0, 0};

    എങ്കിൽ (dim == 0) {
        GGML_ASSERT(src0->ne[0] % ggml_blck_size(src0->type) == 0);
        GGML_ASSERT(src1->ne[0] % ggml_blck_size(src1->type) == 0);

        o[dim] = src0->ne[dim]/ggml_blck_size(src0->type);
    } അല്ലെങ്കിൽ {
        o[dim] = src0->ne[dim];
    }

    // Region 1: copy rows from src0
    വേണ്ടി (പൂർണ്ണസംഖ്യ i3 = 0; i3 < ne03; i3++) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ i2 = ith; i2 < ne02; i2 += nth) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ i1 = 0; i1 < ne01; i1++) {
                സ്ഥിരം അക്ഷരം * x = (സ്ഥിരം അക്ഷരം *) src0->data + i1*nb01 + i2*nb02 + i3*nb03;
                      അക്ഷരം * y = (      അക്ഷരം *) dst->data  + i1*nb1  + i2*nb2  + i3*nb3;
                memcpy(y, x, ggml_row_size(src0->type, ne00));
            }
        }
    }

    // Region 2: copy rows from src1, offset into dst by o[]
    വേണ്ടി (പൂർണ്ണസംഖ്യ i3 = 0; i3 < ne13; i3++) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ i2 = ith; i2 < ne12; i2 += nth) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ i1 = 0; i1 < ne11; i1++) {
                സ്ഥിരം അക്ഷരം * x = (സ്ഥിരം അക്ഷരം *) src1->data + i1*nb11         + i2*nb12         + i3*nb13;
                      അക്ഷരം * y = (      അക്ഷരം *)  dst->data + (i1 + o[1])*nb1 + (i2 + o[2])*nb2 + (i3 + o[3])*nb3 + o[0]*nb0;
                memcpy(y, x, ggml_row_size(src1->type, ne10));
            }
        }
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_concat_i8(
    സ്ഥിരം ggml_compute_params * params,
    ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_type_size(src0->type) == വലുപ്പം(int8_t));

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    GGML_TENSOR_ദ്വിമാന_OP_LOCALS

    സ്ഥിരം int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);

    int64_t o[4] = {0, 0, 0, 0};
    o[dim] = src0->ne[dim];

    സ്ഥിരം int8_t * x;

    // TODO: smarter multi-theading
    വേണ്ടി (പൂർണ്ണസംഖ്യ i3 = 0; i3 < ne3; i3++) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ i2 = ith; i2 < ne2; i2 += nth) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ i1 = 0; i1 < ne1; i1++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ i0 = 0; i0 < ne0; i0++) {
                    എങ്കിൽ (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                        x = (സ്ഥിരം int8_t *) ((സ്ഥിരം അക്ഷരം *)src0->data + (i0       )*nb00 + (i1       )*nb01 + (i2       )*nb02 + (i3       )*nb03);
                    } അല്ലെങ്കിൽ {
                        x = (സ്ഥിരം int8_t *) ((സ്ഥിരം അക്ഷരം *)src1->data + (i0 - o[0])*nb10 + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13);
                    }

                    int8_t * y = (int8_t *)((അക്ഷരം *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);

                    *y = *x;
                }
            }
        }
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_concat_f16(
    സ്ഥിരം ggml_compute_params * params,
    ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_type_size(src0->type) == വലുപ്പം(ggml_fp16_t));

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    GGML_TENSOR_ദ്വിമാന_OP_LOCALS

    സ്ഥിരം int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);

    int64_t o[4] = {0, 0, 0, 0};
    o[dim] = src0->ne[dim];

    സ്ഥിരം ggml_fp16_t * x;

    // TODO: smarter multi-theading
    വേണ്ടി (പൂർണ്ണസംഖ്യ i3 = 0; i3 < ne3; i3++) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ i2 = ith; i2 < ne2; i2 += nth) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ i1 = 0; i1 < ne1; i1++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ i0 = 0; i0 < ne0; i0++) {
                    എങ്കിൽ (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                        x = (സ്ഥിരം ggml_fp16_t *) ((സ്ഥിരം അക്ഷരം *)src0->data + (i0       )*nb00 + (i1       )*nb01 + (i2       )*nb02 + (i3       )*nb03);
                    } അല്ലെങ്കിൽ {
                        x = (സ്ഥിരം ggml_fp16_t *) ((സ്ഥിരം അക്ഷരം *)src1->data + (i0 - o[0])*nb10 + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13);
                    }

                    ggml_fp16_t * y = (ggml_fp16_t *)((അക്ഷരം *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);

                    *y = *x;
                }
            }
        }
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_concat_f32(
    സ്ഥിരം ggml_compute_params * params,
    ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_type_size(src0->type) == വലുപ്പം(ദശാംശം));

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    GGML_TENSOR_ദ്വിമാന_OP_LOCALS

    സ്ഥിരം int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);

    int64_t o[4] = {0, 0, 0, 0};
    o[dim] = src0->ne[dim];

    സ്ഥിരം ദശാംശം * x;

    // TODO: smarter multi-theading
    വേണ്ടി (പൂർണ്ണസംഖ്യ i3 = 0; i3 < ne3; i3++) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ i2 = ith; i2 < ne2; i2 += nth) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ i1 = 0; i1 < ne1; i1++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ i0 = 0; i0 < ne0; i0++) {
                    എങ്കിൽ (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                        x = (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *)src0->data + (i0       )*nb00 + (i1       )*nb01 + (i2       )*nb02 + (i3       )*nb03);
                    } അല്ലെങ്കിൽ {
                        x = (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *)src1->data + (i0 - o[0])*nb10 + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13);
                    }

                    ദശാംശം * y = (ദശാംശം *)((അക്ഷരം *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);

                    *y = *x;
                }
            }
        }
    }
}

ശൂന്യം ggml_compute_forward_concat(
    സ്ഥിരം ggml_compute_params * params,
    ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F16:
        അവസ്ഥ GGML_TYPE_BF16:
        അവസ്ഥ GGML_TYPE_I16:
            {
                ggml_compute_forward_concat_f16(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_I8:
            {
                ggml_compute_forward_concat_i8(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F32:
        അവസ്ഥ GGML_TYPE_I32:
            {
                ggml_compute_forward_concat_f32(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                ggml_compute_forward_concat_any(params, dst);
            }
    }
}

// row loop shared by the BF16 unary ops
template <ശൂന്യം (*vec_op)(സ്ഥിരം പൂർണ്ണസംഖ്യ, ggml_bf16_t *, സ്ഥിരം ggml_bf16_t *)>
സ്ഥിതം ശൂന്യം ggml_compute_forward_unary_bf16_rows(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    പ്രതിജ്ഞാ(ggml_is_contiguous_rows(src0));
    പ്രതിജ്ഞാ(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ nc = src0->ne[0];
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr = ggml_nrows(src0);

    // rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    വേണ്ടി (പൂർണ്ണസംഖ്യ ir = ir0; ir < ir1; ++ir) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ i3 = ir/(ne02*ne01);
        സ്ഥിരം പൂർണ്ണസംഖ്യ i2 = (ir - i3*ne02*ne01)/ne01;
        സ്ഥിരം പൂർണ്ണസംഖ്യ i1 = (ir - i3*ne02*ne01 - i2*ne01);

        vec_op(nc,
                (ggml_bf16_t *) ((അക്ഷരം *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_bf16_t *) ((അക്ഷരം *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));
    }
}

// ggml_compute_forward_gelu

സ്ഥിതം ശൂന്യം ggml_compute_forward_gelu_f32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    പ്രതിജ്ഞാ(ggml_is_contiguous_rows(src0));
    പ്രതിജ്ഞാ(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ nc = src0->ne[0];
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr = ggml_nrows(src0);

    // rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    വേണ്ടി (പൂർണ്ണസംഖ്യ ir = ir0; ir < ir1; ++ir) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ i3 = ir/(ne02*ne01);
        സ്ഥിരം പൂർണ്ണസംഖ്യ i2 = (ir - i3*ne02*ne01)/ne01;
        സ്ഥിരം പൂർണ്ണസംഖ്യ i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_f32(nc,
                (ദശാംശം *) ((അക്ഷരം *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ദശാംശം *) ((അക്ഷരം *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < nc; k++) {
            സ്ഥിരം ദശാംശം x = ((ദശാംശം *) ((അക്ഷരം *) dst->data + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_UNUSED(x);
            പ്രതിജ്ഞാ(!isnan(x));
            പ്രതിജ്ഞാ(!isinf(x));
        }
#endif // NDEBUG
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_gelu_f16(
    സ്ഥിരം ggml_compute_params * params,
    ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    പ്രതിജ്ഞാ(ggml_is_contiguous_rows(src0));
    പ്രതിജ്ഞാ(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ nc = src0->ne[0];
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr = ggml_nrows(src0);

    // rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    വേണ്ടി (പൂർണ്ണസംഖ്യ ir = ir0; ir < ir1; ++ir) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ i3 = ir/(ne02*ne01);
        സ്ഥിരം പൂർണ്ണസംഖ്യ i2 = (ir - i3*ne02*ne01)/ne01;
        സ്ഥിരം പൂർണ്ണസംഖ്യ i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_f16(nc,
                (ggml_fp16_t *) ((അക്ഷരം *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((അക്ഷരം *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < nc; k++) {
            സ്ഥിരം ggml_fp16_t x = ((ggml_fp16_t *) ((അക്ഷരം *) dst->data + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            സ്ഥിരം ദശാംശം v = GGML_CPU_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            പ്രതിജ്ഞാ(!isnan(v));
            പ്രതിജ്ഞാ(!isinf(v));
        }
#endif // NDEBUG
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_gelu_bf16(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_rows<ggml_vec_gelu_bf16>(params, dst);
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_gelu(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_gelu_f32(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F16:
            {
                ggml_compute_forward_gelu_f16(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_BF16:
            {
                ggml_compute_forward_gelu_bf16(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_fill

സ്ഥിതം ശൂന്യം ggml_compute_forward_fill_f32(സ്ഥിരം ggml_compute_params * params, ggml_tensor * dst) {
    സ്ഥിരം ദശാംശം c = ggml_get_op_params_f32(dst, 0);

    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne);
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb);

    സ്ഥിരം auto [ir0, ir1] = get_thread_range(params, dst);

    വേണ്ടി (int64_t ir = ir0; ir < ir1; ++ir) {
        സ്ഥിരം int64_t i03 = ir/(ne2*ne1);
        സ്ഥിരം int64_t i02 = (ir - i03*ne2*ne1)/ne1;
        സ്ഥിരം int64_t i01 = (ir - i03*ne2*ne1 - i02*ne1);

        ദശാംശം * dst_ptr  = (ദശാംശം *) ((അക്ഷരം *) dst->data + i03*nb3 + i02*nb2 + i01*nb1);

        ggml_vec_set_f32(ne0, dst_ptr, c);
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_fill_f16(സ്ഥിരം ggml_compute_params * params, ggml_tensor * dst) {
    സ്ഥിരം ggml_fp16_t c = GGML_CPU_FP32_TO_FP16(ggml_get_op_params_f32(dst, 0));

    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne);
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb);

    സ്ഥിരം auto [ir0, ir1] = get_thread_range(params, dst);

    വേണ്ടി (int64_t ir = ir0; ir < ir1; ++ir) {
        സ്ഥിരം int64_t i03 = ir/(ne2*ne1);
        സ്ഥിരം int64_t i02 = (ir - i03*ne2*ne1)/ne1;
        സ്ഥിരം int64_t i01 = (ir - i03*ne2*ne1 - i02*ne1);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((അക്ഷരം *) dst->data + i03*nb3 + i02*nb2 + i01*nb1);

        ggml_vec_set_f16(ne0, dst_ptr, c);
    }
}

ശൂന്യം ggml_compute_forward_fill(സ്ഥിരം ggml_compute_params * params, ggml_tensor * dst) {
    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_fill_f32(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F16:
            {
                ggml_compute_forward_fill_f16(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("unsupported type for ggml_compute_forward_fill: %s", ggml_type_name(src0->type));
            }
    }
}

// ggml_compute_tri

സ്ഥിതം ശൂന്യം ggml_compute_forward_tri_f32(സ്ഥിരം ggml_compute_params * params, ggml_tensor * dst) {
    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    സ്ഥിരം ggml_tri_type ttype = (ggml_tri_type) ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_TENSOR_UNARY_OP_LOCALS

    സ്ഥിരം auto [ir0, ir1] = get_thread_range(params, src0);

    തർക്കം (*bipred)(പൂർണ്ണസംഖ്യ, പൂർണ്ണസംഖ്യ);

    തിരഞ്ഞെടുപ്പ് (ttype) {
        അവസ്ഥ GGML_TRI_TYPE_LOWER:      bipred = [](പൂർണ്ണസംഖ്യ i, പൂർണ്ണസംഖ്യ r) { തിരികെ i <  r; }; നിർത്തുക;
        അവസ്ഥ GGML_TRI_TYPE_LOWER_DIAG: bipred = [](പൂർണ്ണസംഖ്യ i, പൂർണ്ണസംഖ്യ r) { തിരികെ i <= r; }; നിർത്തുക;
        അവസ്ഥ GGML_TRI_TYPE_UPPER:      bipred = [](പൂർണ്ണസംഖ്യ i, പൂർണ്ണസംഖ്യ r) { തിരികെ i >  r; }; നിർത്തുക;
        അവസ്ഥ GGML_TRI_TYPE_UPPER_DIAG: bipred = [](പൂർണ്ണസംഖ്യ i, പൂർണ്ണസംഖ്യ r) { തിരികെ i >= r; }; നിർത്തുക;
        സ്ഥിരസ്ഥിതി: GGML_ABORT("invalid tri type");
    }

    വേണ്ടി (int64_t ir = ir0; ir < ir1; ++ir) {
        സ്ഥിരം int64_t i03 = ir/(ne02*ne01);
        സ്ഥിരം int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        സ്ഥിരം int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        സ്ഥിരം ദശാംശം * src_ptr = (സ്ഥിരം ദശാംശം  *) ((സ്ഥിരം അക്ഷരം *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);
              ദശാംശം * dst_ptr = (      ദശാംശം  *) ((      അക്ഷരം *) dst->data  + i03*nb3  + i02*nb2  + i01*nb1);

        വേണ്ടി (പൂർണ്ണസംഖ്യ i0 = 0; i0 < ne0; ++i0) {
            dst_ptr[i0] = bipred(i0, i01) ? src_ptr[i0] : 0.0f;
        }
    }
}

ശൂന്യം ggml_compute_forward_tri(സ്ഥിരം ggml_compute_params * params, ggml_tensor * dst) {
    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_tri_f32(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_gelu_erf

സ്ഥിതം ശൂന്യം ggml_compute_forward_gelu_erf_f32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    പ്രതിജ്ഞാ(ggml_is_contiguous_rows(src0));
    പ്രതിജ്ഞാ(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ nc = src0->ne[0];
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr = ggml_nrows(src0);

    // rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    വേണ്ടി (പൂർണ്ണസംഖ്യ ir = ir0; ir < ir1; ++ir) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ i3 = ir/(ne02*ne01);
        സ്ഥിരം പൂർണ്ണസംഖ്യ i2 = (ir - i3*ne02*ne01)/ne01;
        സ്ഥിരം പൂർണ്ണസംഖ്യ i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_erf_f32(nc,
                (ദശാംശം *) ((അക്ഷരം *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ദശാംശം *) ((അക്ഷരം *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < nc; k++) {
            സ്ഥിരം ദശാംശം x = ((ദശാംശം *) ((അക്ഷരം *) dst->data + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_UNUSED(x);
            പ്രതിജ്ഞാ(!isnan(x));
            പ്രതിജ്ഞാ(!isinf(x));
        }
#endif // NDEBUG
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_gelu_erf_f16(
    സ്ഥിരം ggml_compute_params * params,
    ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    പ്രതിജ്ഞാ(ggml_is_contiguous_rows(src0));
    പ്രതിജ്ഞാ(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ nc = src0->ne[0];
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr = ggml_nrows(src0);

    // rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    വേണ്ടി (പൂർണ്ണസംഖ്യ ir = ir0; ir < ir1; ++ir) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ i3 = ir/(ne02*ne01);
        സ്ഥിരം പൂർണ്ണസംഖ്യ i2 = (ir - i3*ne02*ne01)/ne01;
        സ്ഥിരം പൂർണ്ണസംഖ്യ i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_erf_f16(nc,
                (ggml_fp16_t *) ((അക്ഷരം *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((അക്ഷരം *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < nc; k++) {
            സ്ഥിരം ggml_fp16_t x = ((ggml_fp16_t *) ((അക്ഷരം *) dst->data + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            സ്ഥിരം ദശാംശം v = GGML_CPU_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            പ്രതിജ്ഞാ(!isnan(v));
            പ്രതിജ്ഞാ(!isinf(v));
        }
#endif // NDEBUG
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_gelu_erf_bf16(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_rows<ggml_vec_gelu_erf_bf16>(params, dst);
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_gelu_erf(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_gelu_erf_f32(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F16:
            {
                ggml_compute_forward_gelu_erf_f16(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_BF16:
            {
                ggml_compute_forward_gelu_erf_bf16(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_gelu_quick

സ്ഥിതം ശൂന്യം ggml_compute_forward_gelu_quick_f32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    പ്രതിജ്ഞാ(ggml_is_contiguous_rows(src0));
    പ്രതിജ്ഞാ(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ nc = src0->ne[0];
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr = ggml_nrows(src0);

    // rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    വേണ്ടി (പൂർണ്ണസംഖ്യ ir = ir0; ir < ir1; ++ir) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ i3 = ir/(ne02*ne01);
        സ്ഥിരം പൂർണ്ണസംഖ്യ i2 = (ir - i3*ne02*ne01)/ne01;
        സ്ഥിരം പൂർണ്ണസംഖ്യ i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_quick_f32(nc,
                (ദശാംശം *) ((അക്ഷരം *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ദശാംശം *) ((അക്ഷരം *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < nc; k++) {
            സ്ഥിരം ദശാംശം x = ((ദശാംശം *) ((അക്ഷരം *) dst->data + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_UNUSED(x);
            പ്രതിജ്ഞാ(!isnan(x));
            പ്രതിജ്ഞാ(!isinf(x));
        }
#endif // NDEBUG
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_gelu_quick_f16(
    സ്ഥിരം ggml_compute_params * params,
    ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    പ്രതിജ്ഞാ(ggml_is_contiguous_rows(src0));
    പ്രതിജ്ഞാ(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ nc = src0->ne[0];
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr = ggml_nrows(src0);

    // rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    വേണ്ടി (പൂർണ്ണസംഖ്യ ir = ir0; ir < ir1; ++ir) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ i3 = ir/(ne02*ne01);
        സ്ഥിരം പൂർണ്ണസംഖ്യ i2 = (ir - i3*ne02*ne01)/ne01;
        സ്ഥിരം പൂർണ്ണസംഖ്യ i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_quick_f16(nc,
                (ggml_fp16_t *) ((അക്ഷരം *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((അക്ഷരം *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < nc; k++) {
            സ്ഥിരം ggml_fp16_t x = ((ggml_fp16_t *) ((അക്ഷരം *) dst->data + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            സ്ഥിരം ദശാംശം v = GGML_CPU_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            പ്രതിജ്ഞാ(!isnan(v));
            പ്രതിജ്ഞാ(!isinf(v));
        }
#endif // NDEBUG
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_gelu_quick_bf16(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_rows<ggml_vec_gelu_quick_bf16>(params, dst);
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_gelu_quick(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_gelu_quick_f32(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F16:
            {
                ggml_compute_forward_gelu_quick_f16(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_BF16:
            {
                ggml_compute_forward_gelu_quick_bf16(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_silu

സ്ഥിതം ശൂന്യം ggml_compute_forward_silu_f32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    പ്രതിജ്ഞാ(ggml_is_contiguous_rows(src0));
    പ്രതിജ്ഞാ(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ nc = src0->ne[0];
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr = ggml_nrows(src0);

    // rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    വേണ്ടി (പൂർണ്ണസംഖ്യ ir = ir0; ir < ir1; ++ir) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ i3 = ir/(ne02*ne01);
        സ്ഥിരം പൂർണ്ണസംഖ്യ i2 = (ir - i3*ne02*ne01)/ne01;
        സ്ഥിരം പൂർണ്ണസംഖ്യ i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_silu_f32(nc,
                (ദശാംശം *) ((അക്ഷരം *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ദശാംശം *) ((അക്ഷരം *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < nc; k++) {
            സ്ഥിരം ദശാംശം x = ((ദശാംശം *) ((അക്ഷരം *) dst->data + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_UNUSED(x);
            പ്രതിജ്ഞാ(!isnan(x));
            പ്രതിജ്ഞാ(!isinf(x));
        }
#endif // NDEBUG
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_silu_f16(
    സ്ഥിരം ggml_compute_params * params,
    ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    പ്രതിജ്ഞാ(ggml_is_contiguous_rows(src0));
    പ്രതിജ്ഞാ(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ nc = src0->ne[0];
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr = ggml_nrows(src0);

    // rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    വേണ്ടി (പൂർണ്ണസംഖ്യ ir = ir0; ir < ir1; ++ir) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ i3 = ir/(ne02*ne01);
        സ്ഥിരം പൂർണ്ണസംഖ്യ i2 = (ir - i3*ne02*ne01)/ne01;
        സ്ഥിരം പൂർണ്ണസംഖ്യ i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_silu_f16(nc,
                (ggml_fp16_t *) ((അക്ഷരം *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((അക്ഷരം *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < nc; k++) {
            സ്ഥിരം ggml_fp16_t x = ((ggml_fp16_t *) ((അക്ഷരം *) dst->data + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            സ്ഥിരം ദശാംശം v = GGML_CPU_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            പ്രതിജ്ഞാ(!isnan(v));
            പ്രതിജ്ഞാ(!isinf(v));
        }
#endif // NDEBUG
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_silu_bf16(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_rows<ggml_vec_silu_bf16>(params, dst);
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_silu(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_silu_f32(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F16:
            {
                ggml_compute_forward_silu_f16(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_BF16:
            {
                ggml_compute_forward_silu_bf16(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}
// ggml_compute_forward_leaky_relu

സ്ഥിതം ശൂന്യം ggml_compute_forward_leaky_relu_f32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    എങ്കിൽ (params->ith != 0) {
        തിരികെ;
    }

    പ്രതിജ്ഞാ(ggml_is_contiguous_1(src0));
    പ്രതിജ്ഞാ(ggml_is_contiguous_1(dst));
    പ്രതിജ്ഞാ(ggml_are_same_shape(src0, dst));

    സ്ഥിരം പൂർണ്ണസംഖ്യ n  = ggml_nrows(src0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ nc = src0->ne[0];

    ദശാംശം negative_slope;
    memcpy(&negative_slope, dst->op_params, വലുപ്പം(ദശാംശം));

    പ്രതിജ്ഞാ(dst->nb[0]  == വലുപ്പം(ദശാംശം));
    പ്രതിജ്ഞാ(src0->nb[0] == വലുപ്പം(ദശാംശം));

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n; i++) {
        ggml_vec_leaky_relu_f32(nc,
                (ദശാംശം *) ((അക്ഷരം *) dst->data  + i*( dst->nb[1])),
                (ദശാംശം *) ((അക്ഷരം *) src0->data + i*(src0->nb[1])), negative_slope);
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_leaky_relu_f16(
    സ്ഥിരം ggml_compute_params * params,
    ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    എങ്കിൽ (params->ith != 0) {
        തിരികെ;
    }

    പ്രതിജ്ഞാ(ggml_is_contiguous_1(src0));
    പ്രതിജ്ഞാ(ggml_is_contiguous_1(dst));
    പ്രതിജ്ഞാ(ggml_are_same_shape(src0, dst));

    സ്ഥിരം പൂർണ്ണസംഖ്യ n  = ggml_nrows(src0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ nc = src0->ne[0];

    ദശാംശം negative_slope;
    memcpy(&negative_slope, dst->op_params, വലുപ്പം(ദശാംശം));

    പ്രതിജ്ഞാ(dst->nb[0]  == വലുപ്പം(ggml_fp16_t));
    പ്രതിജ്ഞാ(src0->nb[0] == വലുപ്പം(ggml_fp16_t));

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n; i++) {
        ggml_vec_leaky_relu_f16(nc,
                (ggml_fp16_t *) ((അക്ഷരം *) dst->data  + i*( dst->nb[1])),
                (ggml_fp16_t *) ((അക്ഷരം *) src0->data + i*(src0->nb[1])), negative_slope);
    }
}

ശൂന്യം ggml_compute_forward_leaky_relu(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_leaky_relu_f32(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F16:
            {
                ggml_compute_forward_leaky_relu_f16(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_silu_back

സ്ഥിതം ശൂന്യം ggml_compute_forward_silu_back_f32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * grad = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];

    പ്രതിജ്ഞാ(ggml_is_contiguous_1(grad));
    പ്രതിജ്ഞാ(ggml_is_contiguous_1(src1));
    പ്രതിജ്ഞാ(ggml_is_contiguous_1(dst));
    പ്രതിജ്ഞാ(ggml_are_same_shape(src1, dst));
    പ്രതിജ്ഞാ(ggml_are_same_shape(src1, grad));

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ nc = src1->ne[0];
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr = ggml_nrows(src1);

    // rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    വേണ്ടി (പൂർണ്ണസംഖ്യ i1 = ir0; i1 < ir1; i1++) {
        ggml_vec_silu_backward_f32(nc,
                (ദശാംശം *) ((അക്ഷരം *) dst->data  + i1*( dst->nb[1])),
                (ദശാംശം *) ((അക്ഷരം *) src1->data + i1*(src1->nb[1])),
                (ദശാംശം *) ((അക്ഷരം *) grad->data + i1*(grad->nb[1])));

#ifndef NDEBUG
        വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < nc; k++) {
            സ്ഥിരം ദശാംശം x = ((ദശാംശം *) ((അക്ഷരം *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            പ്രതിജ്ഞാ(!isnan(x));
            പ്രതിജ്ഞാ(!isinf(x));
        }
#endif // NDEBUG
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_silu_back_f16(
    സ്ഥിരം ggml_compute_params * params,
    ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * grad = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];

    പ്രതിജ്ഞാ(ggml_is_contiguous_1(grad));
    പ്രതിജ്ഞാ(ggml_is_contiguous_1(src1));
    പ്രതിജ്ഞാ(ggml_is_contiguous_1(dst));
    പ്രതിജ്ഞാ(ggml_are_same_shape(src1, dst));
    പ്രതിജ്ഞാ(ggml_are_same_shape(src1, grad));

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ nc = src1->ne[0];
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr = ggml_nrows(src1);

    // rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    വേണ്ടി (പൂർണ്ണസംഖ്യ i1 = ir0; i1 < ir1; i1++) {
        ggml_vec_silu_backward_f16(nc,
                (ggml_fp16_t *) ((അക്ഷരം *) dst->data  + i1*( dst->nb[1])),
                (ggml_fp16_t *) ((അക്ഷരം *) src1->data + i1*(src1->nb[1])),
                (ggml_fp16_t *) ((അക്ഷരം *) grad->data + i1*(grad->nb[1])));

#ifndef NDEBUG
        വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < nc; k++) {
            സ്ഥിരം ദശാംശം x = ((ggml_fp16_t *) ((അക്ഷരം *) dst->data + i1*( dst->nb[1])))[k];
            സ്ഥിരം ദശാംശം v = GGML_CPU_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            പ്രതിജ്ഞാ(!isnan(v));
            പ്രതിജ്ഞാ(!isinf(v));
        }
#endif // NDEBUG
    }
}

ശൂന്യം ggml_compute_forward_silu_back(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_silu_back_f32(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F16:
            {
                ggml_compute_forward_silu_back_f16(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_reglu

// row loop shared by the BF16 GLU ops
template <ശൂന്യം (*vec_op)(സ്ഥിരം പൂർണ്ണസംഖ്യ, ggml_bf16_t *, സ്ഥിരം ggml_bf16_t *, സ്ഥിരം ggml_bf16_t *)>
സ്ഥിതം ശൂന്യം ggml_compute_forward_glu_bf16_rows(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];
    അക്ഷരം * src0_d = (അക്ഷരം *) src0->data;
    അക്ഷരം * src1_d = (അക്ഷരം *) (src1 ? src1->data : src0->data);
    സ്ഥിരം size_t src0_o = src0->nb[1];
    സ്ഥിരം size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    എങ്കിൽ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    സ്ഥിരം int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    വേണ്ടി (പൂർണ്ണസംഖ്യ i1 = ir0; i1 < ir1; i1++) {
        സ്ഥിരം ggml_bf16_t * src0_p = (സ്ഥിരം ggml_bf16_t *) (src0_d + i1*src0_o);
        സ്ഥിരം ggml_bf16_t * src1_p = (സ്ഥിരം ggml_bf16_t *) (src1_d + i1*src1_o);
        ggml_bf16_t       * dst_p  = (ggml_bf16_t *) ((അക്ഷരം *) dst->data + i1*(dst->nb[1]));

        എങ്കിൽ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        vec_op(nc, dst_p, src0_p, src1_p);
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_reglu_f32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];
    അക്ഷരം * src0_d = (അക്ഷരം *) src0->data;
    അക്ഷരം * src1_d = (അക്ഷരം *) (src1 ? src1->data : src0->data);
    സ്ഥിരം size_t src0_o = src0->nb[1];
    സ്ഥിരം size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    എങ്കിൽ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    സ്ഥിരം int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    വേണ്ടി (പൂർണ്ണസംഖ്യ i1 = ir0; i1 < ir1; i1++) {
        ദശാംശം * src0_p = (ദശാംശം *) (src0_d + i1*src0_o);
        ദശാംശം * src1_p = (ദശാംശം *) (src1_d + i1*src1_o);

        എങ്കിൽ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_reglu_f32(nc, (ദശാംശം *) ((അക്ഷരം *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < nc; k++) {
            സ്ഥിരം ദശാംശം x = ((ദശാംശം *) ((അക്ഷരം *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            പ്രതിജ്ഞാ(!isnan(x));
            പ്രതിജ്ഞാ(!isinf(x));
        }
#endif // NDEBUG
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_reglu_f16(
    സ്ഥിരം ggml_compute_params * params,
    ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];
    അക്ഷരം * src0_d = (അക്ഷരം *) src0->data;
    അക്ഷരം * src1_d = (അക്ഷരം *) (src1 ? src1->data : src0->data);
    സ്ഥിരം size_t src0_o = src0->nb[1];
    സ്ഥിരം size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    എങ്കിൽ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    സ്ഥിരം int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    വേണ്ടി (പൂർണ്ണസംഖ്യ i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        എങ്കിൽ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_reglu_f16(nc, (ggml_fp16_t *) ((അക്ഷരം *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < nc; k++) {
            സ്ഥിരം ggml_fp16_t x = ((ggml_fp16_t *) ((അക്ഷരം *) dst->data + i1*( dst->nb[1])))[k];
            സ്ഥിരം ദശാംശം v = GGML_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            പ്രതിജ്ഞാ(!isnan(v));
            പ്രതിജ്ഞാ(!isinf(v));
        }
#endif // NDEBUG
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_reglu_bf16(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_rows<ggml_vec_reglu_bf16>(params, dst);
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_reglu(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_reglu_f32(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F16:
            {
                ggml_compute_forward_reglu_f16(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_BF16:
            {
                ggml_compute_forward_reglu_bf16(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_geglu

സ്ഥിതം ശൂന്യം ggml_compute_forward_geglu_f32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];
    അക്ഷരം * src0_d = (അക്ഷരം *) src0->data;
    അക്ഷരം * src1_d = (അക്ഷരം *) (src1 ? src1->data : src0->data);
    സ്ഥിരം size_t src0_o = src0->nb[1];
    സ്ഥിരം size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    എങ്കിൽ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    സ്ഥിരം int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    വേണ്ടി (പൂർണ്ണസംഖ്യ i1 = ir0; i1 < ir1; i1++) {
        ദശാംശം * src0_p = (ദശാംശം *) (src0_d + i1*src0_o);
        ദശാംശം * src1_p = (ദശാംശം *) (src1_d + i1*src1_o);

        എങ്കിൽ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_f32(nc, (ദശാംശം *) ((അക്ഷരം *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < nc; k++) {
            സ്ഥിരം ദശാംശം x = ((ദശാംശം *) ((അക്ഷരം *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            പ്രതിജ്ഞാ(!isnan(x));
            പ്രതിജ്ഞാ(!isinf(x));
        }
#endif // NDEBUG
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_geglu_f16(
    സ്ഥിരം ggml_compute_params * params,
    ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];
    അക്ഷരം * src0_d = (അക്ഷരം *) src0->data;
    അക്ഷരം * src1_d = (അക്ഷരം *) (src1 ? src1->data : src0->data);
    സ്ഥിരം size_t src0_o = src0->nb[1];
    സ്ഥിരം size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    എങ്കിൽ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    സ്ഥിരം int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    വേണ്ടി (പൂർണ്ണസംഖ്യ i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        എങ്കിൽ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_f16(nc, (ggml_fp16_t *) ((അക്ഷരം *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < nc; k++) {
            സ്ഥിരം ggml_fp16_t x = ((ggml_fp16_t *) ((അക്ഷരം *) dst->data + i1*( dst->nb[1])))[k];
            സ്ഥിരം ദശാംശം v = GGML_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            പ്രതിജ്ഞാ(!isnan(v));
            പ്രതിജ്ഞാ(!isinf(v));
        }
#endif // NDEBUG
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_geglu_bf16(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_rows<ggml_vec_geglu_bf16>(params, dst);
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_geglu(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_geglu_f32(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F16:
            {
                ggml_compute_forward_geglu_f16(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_BF16:
            {
                ggml_compute_forward_geglu_bf16(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_swiglu

സ്ഥിതം ശൂന്യം ggml_compute_forward_swiglu_f32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];
    അക്ഷരം * src0_d = (അക്ഷരം *) src0->data;
    അക്ഷരം * src1_d = (അക്ഷരം *) (src1 ? src1->data : src0->data);
    സ്ഥിരം size_t src0_o = src0->nb[1];
    സ്ഥിരം size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    എങ്കിൽ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    സ്ഥിരം int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    വേണ്ടി (പൂർണ്ണസംഖ്യ i1 = ir0; i1 < ir1; i1++) {
        ദശാംശം * src0_p = (ദശാംശം *) (src0_d + i1*src0_o);
        ദശാംശം * src1_p = (ദശാംശം *) (src1_d + i1*src1_o);

        എങ്കിൽ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_swiglu_f32(nc, (ദശാംശം *) ((അക്ഷരം *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < nc; k++) {
            സ്ഥിരം ദശാംശം x = ((ദശാംശം *) ((അക്ഷരം *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            പ്രതിജ്ഞാ(!isnan(x));
            പ്രതിജ്ഞാ(!isinf(x));
        }
#endif // NDEBUG
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_swiglu_f16(
    സ്ഥിരം ggml_compute_params * params,
    ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];
    അക്ഷരം * src0_d = (അക്ഷരം *) src0->data;
    അക്ഷരം * src1_d = (അക്ഷരം *) (src1 ? src1->data : src0->data);
    സ്ഥിരം size_t src0_o = src0->nb[1];
    സ്ഥിരം size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    എങ്കിൽ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    സ്ഥിരം int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    വേണ്ടി (പൂർണ്ണസംഖ്യ i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        എങ്കിൽ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_swiglu_f16(nc, (ggml_fp16_t *) ((അക്ഷരം *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < nc; k++) {
            സ്ഥിരം ggml_fp16_t x = ((ggml_fp16_t *) ((അക്ഷരം *) dst->data + i1*( dst->nb[1])))[k];
            സ്ഥിരം ദശാംശം v = GGML_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            പ്രതിജ്ഞാ(!isnan(v));
            പ്രതിജ്ഞാ(!isinf(v));
        }
#endif // NDEBUG
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_swiglu_bf16(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_rows<ggml_vec_swiglu_bf16>(params, dst);
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_swiglu(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_swiglu_f32(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F16:
            {
                ggml_compute_forward_swiglu_f16(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_BF16:
            {
                ggml_compute_forward_swiglu_bf16(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_swiglu_oai

സ്ഥിതം ശൂന്യം ggml_compute_forward_swiglu_oai_f32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];
    അക്ഷരം * src0_d = (അക്ഷരം *) src0->data;
    അക്ഷരം * src1_d = (അക്ഷരം *) (src1 ? src1->data : src0->data);
    സ്ഥിരം size_t src0_o = src0->nb[1];
    സ്ഥിരം size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    എങ്കിൽ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    സ്ഥിരം int32_t swapped = ggml_get_op_params_i32(dst, 1);
    സ്ഥിരം ദശാംശം alpha = ggml_get_op_params_f32(dst, 2);
    സ്ഥിരം ദശാംശം limit = ggml_get_op_params_f32(dst, 3);

    // rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    വേണ്ടി (പൂർണ്ണസംഖ്യ i1 = ir0; i1 < ir1; i1++) {
        ദശാംശം * src0_p = (ദശാംശം *) (src0_d + i1*src0_o);
        ദശാംശം * src1_p = (ദശാംശം *) (src1_d + i1*src1_o);
        ദശാംശം * dst_p  = (ദശാംശം *) ((അക്ഷരം *) dst->data + i1*(dst->nb[1]));

        എങ്കിൽ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < nc; k++) {
            സ്ഥിരം ദശാംശം x = std::min(src0_p[k], limit);
            സ്ഥിരം ദശാംശം y = std::clamp(src1_p[k], -limit, limit);
            സ്ഥിരം ദശാംശം out_glu = x / (1.f + expf(alpha * (-x)));
            dst_p[k] = out_glu * (y + 1.f);
        }

#ifndef NDEBUG
        വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < nc; k++) {
            സ്ഥിരം ദശാംശം x = dst_p[k];
            GGML_UNUSED(x);
            പ്രതിജ്ഞാ(!isnan(x));
            പ്രതിജ്ഞാ(!isinf(x));
        }
#endif // NDEBUG
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_swiglu_oai_f16(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];
    അക്ഷരം * src0_d = (അക്ഷരം *) src0->data;
    അക്ഷരം * src1_d = (അക്ഷരം *) (src1 ? src1->data : src0->data);
    സ്ഥിരം size_t src0_o = src0->nb[1];
    സ്ഥിരം size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    എങ്കിൽ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    സ്ഥിരം int32_t swapped = ggml_get_op_params_i32(dst, 1);
    സ്ഥിരം ദശാംശം alpha = ggml_get_op_params_f32(dst, 2);
    സ്ഥിരം ദശാംശം limit = ggml_get_op_params_f32(dst, 3);

    // rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    വേണ്ടി (പൂർണ്ണസംഖ്യ i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);
        ggml_fp16_t * dst_p  = (ggml_fp16_t *) ((അക്ഷരം *) dst->data + i1*(dst->nb[1]));

        എങ്കിൽ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < nc; k++) {
            സ്ഥിരം ദശാംശം x = std::min(GGML_CPU_FP16_TO_FP32(src0_p[k]), limit);
            സ്ഥിരം ദശാംശം y = std::clamp(GGML_CPU_FP16_TO_FP32(src1_p[k]), -limit, limit);
            സ്ഥിരം ദശാംശം out_glu = x / (1.f + expf(alpha * (-x)));
            dst_p[k] = GGML_CPU_FP32_TO_FP16(out_glu * (y + 1.f));
        }

#ifndef NDEBUG
        വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < nc; k++) {
            സ്ഥിരം ദശാംശം x = GGML_CPU_FP16_TO_FP32(dst_p[k]);
            GGML_UNUSED(x);
            പ്രതിജ്ഞാ(!isnan(x));
            പ്രതിജ്ഞാ(!isinf(x));
        }
#endif // NDEBUG
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_swiglu_oai(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_swiglu_oai_f32(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F16:
            {
                ggml_compute_forward_swiglu_oai_f16(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_swiglu_clamp

സ്ഥിതം ശൂന്യം ggml_compute_forward_swiglu_clamp_f32(സ്ഥിരം ggml_compute_params * params, ggml_tensor * dst) {
    സ്ഥിരം ggml_tensor * src0   = dst->src[0];
    സ്ഥിരം ggml_tensor * src1   = dst->src[1];
    അക്ഷരം *              src0_d = (അക്ഷരം *) src0->data;
    അക്ഷരം *              src1_d = (അക്ഷരം *) (src1 ? src1->data : src0->data);
    സ്ഥിരം size_t        src0_o = src0->nb[1];
    സ്ഥിരം size_t        src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    എങ്കിൽ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    സ്ഥിരം int32_t swapped = ggml_get_op_params_i32(dst, 1);
    സ്ഥിരം ദശാംശം   limit   = ggml_get_op_params_f32(dst, 3);

    സ്ഥിരം പൂർണ്ണസംഖ്യ dr  = (nr + nth - 1) / nth;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr * ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    വേണ്ടി (പൂർണ്ണസംഖ്യ i1 = ir0; i1 < ir1; i1++) {
        ദശാംശം * src0_p = (ദശാംശം *) (src0_d + i1 * src0_o);
        ദശാംശം * src1_p = (ദശാംശം *) (src1_d + i1 * src1_o);
        ദശാംശം * dst_p  = (ദശാംശം *) ((അക്ഷരം *) dst->data + i1 * (dst->nb[1]));

        എങ്കിൽ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < nc; k++) {
            സ്ഥിരം ദശാംശം gate = std::min(src0_p[k], limit);
            സ്ഥിരം ദശാംശം up   = std::clamp(src1_p[k], -limit, limit);
            dst_p[k]         = gate / (1.f + expf(-gate)) * up;
        }

#ifndef NDEBUG
        വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < nc; k++) {
            സ്ഥിരം ദശാംശം x = dst_p[k];
            GGML_UNUSED(x);
            പ്രതിജ്ഞാ(!isnan(x));
            പ്രതിജ്ഞാ(!isinf(x));
        }
#endif  // NDEBUG
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_swiglu_clamp_f16(സ്ഥിരം ggml_compute_params * params, ggml_tensor * dst) {
    സ്ഥിരം ggml_tensor * src0   = dst->src[0];
    സ്ഥിരം ggml_tensor * src1   = dst->src[1];
    അക്ഷരം *              src0_d = (അക്ഷരം *) src0->data;
    അക്ഷരം *              src1_d = (അക്ഷരം *) (src1 ? src1->data : src0->data);
    സ്ഥിരം size_t        src0_o = src0->nb[1];
    സ്ഥിരം size_t        src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    എങ്കിൽ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    സ്ഥിരം int32_t swapped = ggml_get_op_params_i32(dst, 1);
    സ്ഥിരം ദശാംശം   limit   = ggml_get_op_params_f32(dst, 3);

    സ്ഥിരം പൂർണ്ണസംഖ്യ dr  = (nr + nth - 1) / nth;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr * ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    വേണ്ടി (പൂർണ്ണസംഖ്യ i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1 * src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1 * src1_o);
        ggml_fp16_t * dst_p  = (ggml_fp16_t *) ((അക്ഷരം *) dst->data + i1 * (dst->nb[1]));

        എങ്കിൽ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < nc; k++) {
            സ്ഥിരം ദശാംശം gate = std::min(GGML_FP16_TO_FP32(src0_p[k]), limit);
            സ്ഥിരം ദശാംശം up   = std::clamp(GGML_FP16_TO_FP32(src1_p[k]), -limit, limit);
            dst_p[k]         = GGML_FP32_TO_FP16(gate / (1.f + expf(-gate)) * up);
        }

#ifndef NDEBUG
        വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < nc; k++) {
            സ്ഥിരം ദശാംശം x = GGML_FP16_TO_FP32(dst_p[k]);
            GGML_UNUSED(x);
            പ്രതിജ്ഞാ(!isnan(x));
            പ്രതിജ്ഞാ(!isinf(x));
        }
#endif  // NDEBUG
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_swiglu_clamp(സ്ഥിരം ggml_compute_params * params, ggml_tensor * dst) {
    തിരഞ്ഞെടുപ്പ് (dst->src[0]->type) {
        അവസ്ഥ GGML_TYPE_F32:
            ggml_compute_forward_swiglu_clamp_f32(params, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F16:
            ggml_compute_forward_swiglu_clamp_f16(params, dst);
            നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            GGML_ABORT("fatal error");
    }
}

// ggml_compute_forward_geglu_erf

സ്ഥിതം ശൂന്യം ggml_compute_forward_geglu_erf_f32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];
    അക്ഷരം * src0_d = (അക്ഷരം *) src0->data;
    അക്ഷരം * src1_d = (അക്ഷരം *) (src1 ? src1->data : src0->data);
    സ്ഥിരം size_t src0_o = src0->nb[1];
    സ്ഥിരം size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    എങ്കിൽ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    സ്ഥിരം int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    വേണ്ടി (പൂർണ്ണസംഖ്യ i1 = ir0; i1 < ir1; i1++) {
        ദശാംശം * src0_p = (ദശാംശം *) (src0_d + i1*src0_o);
        ദശാംശം * src1_p = (ദശാംശം *) (src1_d + i1*src1_o);

        എങ്കിൽ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_erf_f32(nc, (ദശാംശം *) ((അക്ഷരം *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < nc; k++) {
            സ്ഥിരം ദശാംശം x = ((ദശാംശം *) ((അക്ഷരം *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            പ്രതിജ്ഞാ(!isnan(x));
            പ്രതിജ്ഞാ(!isinf(x));
        }
#endif // NDEBUG
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_geglu_erf_f16(
    സ്ഥിരം ggml_compute_params * params,
    ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];
    അക്ഷരം * src0_d = (അക്ഷരം *) src0->data;
    അക്ഷരം * src1_d = (അക്ഷരം *) (src1 ? src1->data : src0->data);
    സ്ഥിരം size_t src0_o = src0->nb[1];
    സ്ഥിരം size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    എങ്കിൽ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    സ്ഥിരം int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    വേണ്ടി (പൂർണ്ണസംഖ്യ i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        എങ്കിൽ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_erf_f16(nc, (ggml_fp16_t *) ((അക്ഷരം *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < nc; k++) {
            സ്ഥിരം ggml_fp16_t x = ((ggml_fp16_t *) ((അക്ഷരം *) dst->data + i1*( dst->nb[1])))[k];
            സ്ഥിരം ദശാംശം v = GGML_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            പ്രതിജ്ഞാ(!isnan(v));
            പ്രതിജ്ഞാ(!isinf(v));
        }
#endif // NDEBUG
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_geglu_erf_bf16(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_rows<ggml_vec_geglu_erf_bf16>(params, dst);
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_geglu_erf(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_geglu_erf_f32(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F16:
            {
                ggml_compute_forward_geglu_erf_f16(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_BF16:
            {
                ggml_compute_forward_geglu_erf_bf16(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_geglu_quick

സ്ഥിതം ശൂന്യം ggml_compute_forward_geglu_quick_f32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];
    അക്ഷരം * src0_d = (അക്ഷരം *) src0->data;
    അക്ഷരം * src1_d = (അക്ഷരം *) (src1 ? src1->data : src0->data);
    സ്ഥിരം size_t src0_o = src0->nb[1];
    സ്ഥിരം size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    എങ്കിൽ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    സ്ഥിരം int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    വേണ്ടി (പൂർണ്ണസംഖ്യ i1 = ir0; i1 < ir1; i1++) {
        ദശാംശം * src0_p = (ദശാംശം *) (src0_d + i1*src0_o);
        ദശാംശം * src1_p = (ദശാംശം *) (src1_d + i1*src1_o);

        എങ്കിൽ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_quick_f32(nc, (ദശാംശം *) ((അക്ഷരം *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < nc; k++) {
            സ്ഥിരം ദശാംശം x = ((ദശാംശം *) ((അക്ഷരം *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            പ്രതിജ്ഞാ(!isnan(x));
            പ്രതിജ്ഞാ(!isinf(x));
        }
#endif // NDEBUG
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_geglu_quick_f16(
    സ്ഥിരം ggml_compute_params * params,
    ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];
    അക്ഷരം * src0_d = (അക്ഷരം *) src0->data;
    അക്ഷരം * src1_d = (അക്ഷരം *) (src1 ? src1->data : src0->data);
    സ്ഥിരം size_t src0_o = src0->nb[1];
    സ്ഥിരം size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    എങ്കിൽ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    സ്ഥിരം int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    വേണ്ടി (പൂർണ്ണസംഖ്യ i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        എങ്കിൽ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_quick_f16(nc, (ggml_fp16_t *) ((അക്ഷരം *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < nc; k++) {
            സ്ഥിരം ggml_fp16_t x = ((ggml_fp16_t *) ((അക്ഷരം *) dst->data + i1*( dst->nb[1])))[k];
            സ്ഥിരം ദശാംശം v = GGML_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            പ്രതിജ്ഞാ(!isnan(v));
            പ്രതിജ്ഞാ(!isinf(v));
        }
#endif // NDEBUG
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_geglu_quick_bf16(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_rows<ggml_vec_geglu_quick_bf16>(params, dst);
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_geglu_quick(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_geglu_quick_f32(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F16:
            {
                ggml_compute_forward_geglu_quick_f16(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_BF16:
            {
                ggml_compute_forward_geglu_quick_bf16(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_norm

സ്ഥിതം ശൂന്യം ggml_compute_forward_norm_f32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    ദശാംശം eps;
    memcpy(&eps, dst->op_params, വലുപ്പം(ദശാംശം));

    GGML_ASSERT(eps >= 0.0f);

    വേണ്ടി (int64_t i03 = 0; i03 < ne03; i03++) {
        വേണ്ടി (int64_t i02 = 0; i02 < ne02; i02++) {
            വേണ്ടി (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                സ്ഥിരം അക്ഷരം * x = (സ്ഥിരം അക്ഷരം *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
                അക്ഷരം * y = (അക്ഷരം *) dst->data + i01*nb1 + i02*nb2 + i03*nb3;

                എങ്കിൽ (nb00 == വലുപ്പം(ദശാംശം) && nb0 == വലുപ്പം(ദശാംശം)) {
                    സ്ഥിരം ദശാംശം * xf = (സ്ഥിരം ദശാംശം *) x;

                    ദശാംശം sum = 0.0;
                    ggml_vec_sum_f32(ne00, &sum, xf);
                    ദശാംശം mean = sum/ne00;

                    ദശാംശം * yf = (ദശാംശം *) y;
                    ദശാംശം variance = 0;

#ifdef GGML_USE_ACCELERATE
                    mean = -mean;
                    vDSP_vsadd(xf, 1, &mean, yf, 1, ne00);
                    vDSP_measqv(yf, 1, &variance, ne00);
#അല്ലെങ്കിൽ
                    variance = ggml_vec_cvar_f32(ne00, yf, xf, mean);
#endif //GGML_USE_ACCELERATE

                    സ്ഥിരം ദശാംശം scale = 1.0f/sqrtf(variance + eps);
                    ggml_vec_scale_f32(ne00, yf, scale);
                } അല്ലെങ്കിൽ {
                    ദശാംശം sum = 0.0;
                    വേണ്ടി (int64_t i00 = 0; i00 < ne00; i00++) {
                        sum += *(സ്ഥിരം ദശാംശം *) (x + i00*nb00);
                    }
                    സ്ഥിരം ദശാംശം mean = sum/ne00;

                    ദശാംശം variance = 0.0f;
                    വേണ്ടി (int64_t i00 = 0; i00 < ne00; i00++) {
                        സ്ഥിരം ദശാംശം v = *(സ്ഥിരം ദശാംശം *) (x + i00*nb00) - mean;
                        *(ദശാംശം *) (y + i00*nb0) = v;
                        variance += v * v;
                    }
                    variance /= ne00;

                    സ്ഥിരം ദശാംശം scale = 1.0f/sqrtf(variance + eps);
                    വേണ്ടി (int64_t i00 = 0; i00 < ne00; i00++) {
                        *(ദശാംശം *) (y + i00*nb0) *= scale;
                    }
                }
            }
        }
    }
}

ശൂന്യം ggml_compute_forward_norm(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_norm_f32(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_group_rms_norm

// fusion kinds that can be combined with the rms_norm computation in a single pass.
// extend this enum when adding new fused variants (e.g. FUSE_ADD, FUSE_MUL_ADD, ...).
എണ്ണൽ ggml_rms_norm_fuse_op {
    GGML_RMS_NORM_FUSE_OP_NONE,
    GGML_RMS_NORM_FUSE_OP_MUL,
};

template <ggml_rms_norm_fuse_op FUSE_OP>
സ്ഥിതം ശൂന്യം ggml_compute_forward_rms_norm_f32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst_rms_norm,
        ggml_tensor * dst_fused = nullptr) {

    സ്ഥിരം ggml_tensor * src0 = dst_rms_norm->src[0];
    സ്ഥിരം ggml_tensor * src1 = nullptr;
    ggml_tensor       * dst  = dst_rms_norm;

    എങ്കിൽ constexpr (FUSE_OP == GGML_RMS_NORM_FUSE_OP_MUL) {
        src1 = (dst_fused->src[0] == dst_rms_norm) ? dst_fused->src[1] : dst_fused->src[0];
        dst  = dst_fused;
    }

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    GGML_ASSERT(src0->nb[0] == വലുപ്പം(ദശാംശം));

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    GGML_TENSOR_ദ്വിമാന_OP_LOCALS

    ദശാംശം eps;
    memcpy(&eps, dst_rms_norm->op_params, വലുപ്പം(ദശാംശം));
    GGML_ASSERT(eps >= 0.0f);

    // TODO: optimize
    വേണ്ടി (int64_t i03 = 0; i03 < ne03; i03++) {
        വേണ്ടി (int64_t i02 = 0; i02 < ne02; i02++) {
            വേണ്ടി (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                സ്ഥിരം ദശാംശം * x = (ദശാംശം *) ((അക്ഷരം *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);

                ggml_float sum = 0.0;
                // worth switching to explicit SIMD?
                വേണ്ടി (int64_t i00 = 0; i00 < ne00; i00++) {
                    sum += (ggml_float)(x[i00] * x[i00]);
                }

                സ്ഥിരം ദശാംശം mean  = sum/ne00;
                സ്ഥിരം ദശാംശം scale = 1.0f/sqrtf(mean + eps);

                // if you hit this, likely you got an inf somewhere earlier
                പ്രതിജ്ഞാ(scale > 0.0f);

                ദശാംശം * y = (ദശാംശം *) ((അക്ഷരം *) dst->data + i01*nb1 + i02*nb2 + i03*nb3);

                എങ്കിൽ constexpr (FUSE_OP == GGML_RMS_NORM_FUSE_OP_MUL) {
                    സ്ഥിരം int64_t i11 = i01 % ne11;
                    സ്ഥിരം int64_t i12 = i02 % ne12;
                    സ്ഥിരം int64_t i13 = i03 % ne13;
                    സ്ഥിരം ദശാംശം * w = (ദശാംശം *) ((അക്ഷരം *) src1->data + i11*nb11 + i12*nb12 + i13*nb13);

                    വേണ്ടി (int64_t i00 = 0; i00 < ne00; i00++) {
                        y[i00] = x[i00] * scale * w[i00];
                    }
                } അല്ലെങ്കിൽ {
                    memcpy(y, x, ne00 * വലുപ്പം(ദശാംശം));
                    ggml_vec_scale_f32(ne00, y, scale);
                }
            }
        }
    }
}

ശൂന്യം ggml_compute_forward_rms_norm(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_rms_norm_f32<GGML_RMS_NORM_FUSE_OP_NONE>(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// Fused RMS_NORM + MUL: computes dst = rms_norm(src0) * src1 in a single pass.
// This avoids materializing the intermediate rms_norm result in memory.
ശൂന്യം ggml_compute_forward_rms_norm_mul_fused(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst_rms_norm,
        ggml_tensor * dst_mul) {

    GGML_ASSERT(dst_mul != nullptr);
    GGML_ASSERT(dst_mul->src[0] == dst_rms_norm || dst_mul->src[1] == dst_rms_norm);

    സ്ഥിരം ggml_tensor * src0 = dst_rms_norm->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_rms_norm_f32<GGML_RMS_NORM_FUSE_OP_MUL>(params, dst_rms_norm, dst_mul);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_rms_norm_back_f32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0]; // gradients from forward pass output
    സ്ഥിരം ggml_tensor * src1 = dst->src[1]; // src1 from forward pass

    GGML_ASSERT(ggml_are_same_shape(src0, dst) && ggml_are_same_shape(src0, src1));

    GGML_ASSERT(src0->nb[0] == വലുപ്പം(ദശാംശം));
    GGML_ASSERT(src1->nb[0] == വലുപ്പം(ദശാംശം));

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    GGML_TENSOR_ദ്വിമാന_OP_LOCALS

    ദശാംശം eps;
    memcpy(&eps, dst->op_params, വലുപ്പം(ദശാംശം));

    // TODO: optimize
    വേണ്ടി (int64_t i03 = 0; i03 < ne03; i03++) {
        വേണ്ടി (int64_t i02 = 0; i02 < ne02; i02++) {
            വേണ്ടി (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                // src1 is same shape as src0 => same indices
                സ്ഥിരം int64_t i11 = i01;
                സ്ഥിരം int64_t i12 = i02;
                സ്ഥിരം int64_t i13 = i03;

                സ്ഥിരം ദശാംശം * dz = (ദശാംശം *) ((അക്ഷരം *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
                സ്ഥിരം ദശാംശം * x  = (ദശാംശം *) ((അക്ഷരം *) src1->data + i11*nb11 + i12*nb12 + i13*nb13);

                ggml_float sum_xx  = 0.0;
                ggml_float sum_xdz = 0.0;

                വേണ്ടി (int64_t i00 = 0; i00 < ne00; i00++) {
                    sum_xx  += (ggml_float)(x[i00] * x[i00]);
                    sum_xdz += (ggml_float)(x[i00] * dz[i00]);
                }

                //const float mean     = (float)(sum_xx)/ne00;
                സ്ഥിരം ദശാംശം mean_eps = (ദശാംശം)(sum_xx)/ne00 + eps;
                സ്ഥിരം ദശാംശം sum_eps  = (ദശാംശം)(sum_xx) + eps*ne00;
                //const float mean_xdz = (float)(sum_xdz)/ne00;
                // we could cache rms from forward pass to improve performance.
                // to do this implement ggml_rms and compose ggml_rms_norm using ggml_rms.
                //const float rms      = sqrtf(mean_eps);
                സ്ഥിരം ദശാംശം rrms     = 1.0f / sqrtf(mean_eps);
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
                ദശാംശം * dx = (ദശാംശം *) ((അക്ഷരം *) dst->data + i01*nb1 + i02*nb2 + i03*nb3);

                // dx[i00] = (dz + x*(-sum_xdz/sum_eps)) * rrms
                // note: https://github.com/ggml-org/ggml/issues/1491
                സ്ഥിരം ദശാംശം scale_x = (ദശാംശം) (-sum_xdz) / sum_eps;
                വേണ്ടി (int64_t i00 = 0; i00 < ne00; i00++) {
                    dx[i00] = (dz[i00] + x[i00] * scale_x) * rrms;
                }
            }
        }
    }
}

ശൂന്യം ggml_compute_forward_rms_norm_back(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_rms_norm_back_f32(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_group_norm

സ്ഥിതം ശൂന്യം ggml_compute_forward_group_norm_f32(
    സ്ഥിരം ggml_compute_params * params,
    ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    GGML_ASSERT(src0->nb[0] == വലുപ്പം(ദശാംശം));

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    // TODO: optimize

    ദശാംശം eps;
    memcpy(&eps, dst->op_params + 1, വലുപ്പം(ദശാംശം));

    പൂർണ്ണസംഖ്യ n_channels = src0->ne[2];
    പൂർണ്ണസംഖ്യ n_groups = dst->op_params[0];
    പൂർണ്ണസംഖ്യ n_channels_per_group = (n_channels + n_groups - 1) / n_groups;
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = ith; i < n_groups; i += nth) {
        പൂർണ്ണസംഖ്യ start = i * n_channels_per_group;
        പൂർണ്ണസംഖ്യ end = start + n_channels_per_group;
        എങ്കിൽ (end > n_channels) {
            end = n_channels;
        }
        പൂർണ്ണസംഖ്യ step = end - start;

        വേണ്ടി (int64_t i03 = 0; i03 < ne03; i03++) {
            ggml_float sum = 0.0;
            വേണ്ടി (int64_t i02 = start; i02 < end; i02++) {
                വേണ്ടി (int64_t i01 = 0; i01 < ne01; i01++) {
                    സ്ഥിരം ദശാംശം * x = (ദശാംശം *)((അക്ഷരം *) src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);

                    ggml_float sumr = 0.0;
                    വേണ്ടി (int64_t i00 = 0; i00 < ne00; i00++) {
                        sumr += (ggml_float)x[i00];
                    }
                    sum += sumr;
                }
            }
            സ്ഥിരം ദശാംശം mean = sum / (ne00 * ne01 * step);

            ggml_float sum2 = 0.0;
            വേണ്ടി (int64_t i02 = start; i02 < end; i02++) {
                വേണ്ടി (int64_t i01 = 0; i01 < ne01; i01++) {
                    സ്ഥിരം ദശാംശം * x = (ദശാംശം *)((അക്ഷരം *) src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);

                    ദശാംശം * y = (ദശാംശം *)((അക്ഷരം *) dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3);

                    ggml_float sumr = 0.0;
                    വേണ്ടി (int64_t i00 = 0; i00 < ne00; i00++) {
                        ദശാംശം v = x[i00] - mean;
                        y[i00] = v;
                        sumr += (ggml_float)(v * v);
                    }
                    sum2 += sumr;
                }
            }
            സ്ഥിരം ദശാംശം variance = sum2 / (ne00 * ne01 * step);
            സ്ഥിരം ദശാംശം scale = 1.0f / sqrtf(variance + eps);

            വേണ്ടി (int64_t i02 = start; i02 < end; i02++) {
                വേണ്ടി (int64_t i01 = 0; i01 < ne01; i01++) {
                    ദശാംശം * y = (ദശാംശം *)((അക്ഷരം *) dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3);
                    ggml_vec_scale_f32(ne00, y, scale);
                }
            }
        }
    }
}

ശൂന്യം ggml_compute_forward_group_norm(
    സ്ഥിരം ggml_compute_params * params,
    ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_group_norm_f32(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_l2_norm

സ്ഥിതം ശൂന്യം ggml_compute_forward_l2_norm_f32(
    സ്ഥിരം ggml_compute_params * params,
    ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    ദശാംശം eps;
    memcpy(&eps, dst->op_params, വലുപ്പം(ദശാംശം));

    GGML_ASSERT(eps >= 0.0f);

    // TODO: optimize
    വേണ്ടി (int64_t i03 = 0; i03 < ne03; i03++) {
        വേണ്ടി (int64_t i02 = 0; i02 < ne02; i02++) {
            വേണ്ടി (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                സ്ഥിരം അക്ഷരം * x = (സ്ഥിരം അക്ഷരം *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;

                ggml_float sum = 0.0;
                വേണ്ടി (int64_t i00 = 0; i00 < ne00; i00++) {
                    സ്ഥിരം ദശാംശം xi = *(സ്ഥിരം ദശാംശം *) (x + i00*nb00);
                    sum += (ggml_float)(xi * xi);
                }

                സ്ഥിരം ദശാംശം scale = 1.0f/fmaxf(sqrtf(sum), eps);

                അക്ഷരം * y = (അക്ഷരം *) dst->data + i01*nb1 + i02*nb2 + i03*nb3;

                എങ്കിൽ (nb00 == വലുപ്പം(ദശാംശം) && nb0 == വലുപ്പം(ദശാംശം)) {
                    memcpy(y, x, ne00 * വലുപ്പം(ദശാംശം));
                    ggml_vec_scale_f32(ne00, (ദശാംശം *) y, scale);
                } അല്ലെങ്കിൽ {
                    വേണ്ടി (int64_t i00 = 0; i00 < ne00; i00++) {
                        സ്ഥിരം ദശാംശം xi = *(സ്ഥിരം ദശാംശം *) (x + i00*nb00);
                        *(ദശാംശം *) (y + i00*nb0) = xi * scale;
                    }
                }
            }
        }
    }
}

ശൂന്യം ggml_compute_forward_l2_norm(
    സ്ഥിരം ggml_compute_params * params,
    ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_l2_norm_f32(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_out_prod

സ്ഥിതം ശൂന്യം ggml_compute_forward_out_prod_f32(
        സ്ഥിരം ggml_compute_params * params,
              ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ദ്വിമാന_OP_LOCALS

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    GGML_ASSERT(ne2 % ne02 == 0);
    GGML_ASSERT(ne3 % ne03 == 0);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == വലുപ്പം(ദശാംശം));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == വലുപ്പം(ദശാംശം));
    // GGML_ASSERT(nb0 <= nb1);
    // GGML_ASSERT(nb1 <= nb2);
    // GGML_ASSERT(nb2 <= nb3);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    എങ്കിൽ (ith == 0) {
        ggml_vec_set_f32(ne0*ne1*ne2*ne3, (ദശാംശം *)dst->data, 0);
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
    സ്ഥിരം int64_t nr = ne1*ne2*ne3;

    // rows per thread
    സ്ഥിരം int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം int64_t ir0 = dr*ith;
    സ്ഥിരം int64_t ir1 = MIN(ir0 + dr, nr);

    // block-tiling attempt
    സ്ഥിരം int64_t blck_0 = MAX(GGML_VEC_MAD_UNROLL, 32);
    സ്ഥിരം int64_t blck_1 = 16;

    // dps == dst per src0, used for group query attention
    സ്ഥിരം int64_t dps2 = ne2 / ne02;
    സ്ഥിരം int64_t dps3 = ne3 / ne03;

    വേണ്ടി (int64_t bir = ir0; bir < ir1; bir += blck_1) {
        സ്ഥിരം int64_t bir1 = MIN(bir + blck_1, ir1);
        വേണ്ടി (int64_t bi01 = 0; bi01 < ne01; bi01 += blck_0) {
            സ്ഥിരം int64_t bne01 = MIN(bi01 + blck_0, ne01);
            വേണ്ടി (int64_t ir = bir; ir < bir1; ++ir) {
                // dst indices
                സ്ഥിരം int64_t i3 = ir/(ne2*ne1);
                സ്ഥിരം int64_t i2 = (ir - i3*ne2*ne1)/ne1;
                സ്ഥിരം int64_t i1 = (ir - i3*ne2*ne1 - i2*ne1);

                സ്ഥിരം int64_t i02 = i2 / dps2;
                സ്ഥിരം int64_t i03 = i3 / dps3;

                //const int64_t i10 = i1;
                സ്ഥിരം int64_t i12 = i2;
                സ്ഥിരം int64_t i13 = i3;

#എങ്കിൽ GGML_VEC_MAD_UNROLL > 2
                സ്ഥിരം int64_t bne01_unroll = bne01 - (bne01 % GGML_VEC_MAD_UNROLL);
                വേണ്ടി (int64_t i01 = bi01; i01 < bne01_unroll; i01 += GGML_VEC_MAD_UNROLL) {
                    സ്ഥിരം int64_t i11 = i01;

                    ദശാംശം * s0 = (ദശാംശം *) ((അക്ഷരം *) src0->data + (          i01*nb01 + i02*nb02 + i03*nb03));
                    ദശാംശം * s1 = (ദശാംശം *) ((അക്ഷരം *) src1->data + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
                    ദശാംശം * d  = (ദശാംശം *) ((അക്ഷരം *)  dst->data + (          i1*nb1   + i2*nb2   + i3*nb3));

                    ggml_vec_mad_f32_unroll(ne0, nb01, nb11, d, s0, s1);
                }
                വേണ്ടി (int64_t i01 = bne01_unroll; i01 < bne01; ++i01) {
                    സ്ഥിരം int64_t i11 = i01;

                    ദശാംശം * s0 = (ദശാംശം *) ((അക്ഷരം *) src0->data + (          i01*nb01 + i02*nb02 + i03*nb03));
                    ദശാംശം * s1 = (ദശാംശം *) ((അക്ഷരം *) src1->data + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
                    ദശാംശം * d  = (ദശാംശം *) ((അക്ഷരം *)  dst->data + (          i1*nb1   + i2*nb2   + i3*nb3));

                    ggml_vec_mad_f32(ne0, d, s0, *s1);
                }
#അല്ലെങ്കിൽ
                വേണ്ടി (int64_t i01 = bi01; i01 < bne01; ++i01) {
                    സ്ഥിരം int64_t i11 = i01;

                    ദശാംശം * s0 = (ദശാംശം *) ((അക്ഷരം *) src0->data + (          i01*nb01 + i02*nb02 + i03*nb03));
                    ദശാംശം * s1 = (ദശാംശം *) ((അക്ഷരം *) src1->data + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
                    ദശാംശം * d  = (ദശാംശം *) ((അക്ഷരം *)  dst->data + (          i1*nb1 + i2*nb2 + i3*nb3));

                    ggml_vec_mad_f32(ne0, d, s0, *s1);
                }
#endif
            }
        }
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_out_prod_q_f32(
        സ്ഥിരം ggml_compute_params * params,
              ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ദ്വിമാന_OP_LOCALS;

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം ggml_type type = src0->type;
    ggml_to_float_t സ്ഥിരം dequantize_row_q = ggml_get_type_traits(type)->to_float;

    GGML_ASSERT(ne02 == ne12);
    GGML_ASSERT(ne03 == ne13);
    GGML_ASSERT(ne2  == ne12);
    GGML_ASSERT(ne3  == ne13);

    // we don't support permuted src0 dim0
    GGML_ASSERT(nb00 == ggml_type_size(type));

    // dst dim0 cannot be transposed or permuted
    GGML_ASSERT(nb0 == വലുപ്പം(ദശാംശം));
    // GGML_ASSERT(nb0 <= nb1);
    // GGML_ASSERT(nb1 <= nb2);
    // GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    എങ്കിൽ (ith == 0) {
        ggml_vec_set_f32(ne0*ne1*ne2*ne3, (ദശാംശം *)dst->data, 0);
    }
    ggml_barrier(params->threadpool);

    // parallelize by last three dimensions

    // total rows in dst
    സ്ഥിരം int64_t nr = ne1*ne2*ne3;

    // rows per thread
    സ്ഥിരം int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം int64_t ir0 = dr*ith;
    സ്ഥിരം int64_t ir1 = MIN(ir0 + dr, nr);

    // dst[:,:,:,:] = 0
    // for i2,i3:
    //   for i1:
    //     for i01:
    //       for i0:
    //         dst[i0,i1,i2,i3] += src0[i0,i01,i2,i3] * src1[i1,i01,i2,i3]

    ദശാംശം * wdata = (ദശാംശം *) params->wdata + (ne0 + CACHE_വരി_SIZE_F32) * ith;

    വേണ്ടി (int64_t ir = ir0; ir < ir1; ++ir) {
        // dst indices
        സ്ഥിരം int64_t i3 = ir/(ne2*ne1);
        സ്ഥിരം int64_t i2 = (ir - i3*ne2*ne1)/ne1;
        സ്ഥിരം int64_t i1 = (ir - i3*ne2*ne1 - i2*ne1);

        സ്ഥിരം int64_t i02 = i2;
        സ്ഥിരം int64_t i03 = i3;

        //const int64_t i10 = i1;
        സ്ഥിരം int64_t i12 = i2;
        സ്ഥിരം int64_t i13 = i3;

        വേണ്ടി (int64_t i01 = 0; i01 < ne01; ++i01) {
            സ്ഥിരം int64_t i11 = i01;

            ദശാംശം * s0 = (ദശാംശം *) ((അക്ഷരം *) src0->data + (          i01*nb01 + i02*nb02 + i03*nb03));
            ദശാംശം * s1 = (ദശാംശം *) ((അക്ഷരം *) src1->data + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
            ദശാംശം * d  = (ദശാംശം *) ((അക്ഷരം *)  dst->data + (          i1*nb1 + i2*nb2 + i3*nb3));

            dequantize_row_q(s0, wdata, ne0);
            ggml_vec_mad_f32(ne0, d, wdata, *s1);
        }
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_out_prod_f16_f32(
        സ്ഥിരം ggml_compute_params * params,
              ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ദ്വിമാന_OP_LOCALS;

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ne02 == ne12);
    GGML_ASSERT(ne03 == ne13);
    GGML_ASSERT(ne2  == ne12);
    GGML_ASSERT(ne3  == ne13);

    GGML_ASSERT(nb00 == വലുപ്പം(ggml_fp16_t));
    GGML_ASSERT(nb0  == വലുപ്പം(ദശാംശം));

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    എങ്കിൽ (ith == 0) {
        ggml_vec_set_f32(ne0*ne1*ne2*ne3, (ദശാംശം *)dst->data, 0);
    }
    ggml_barrier(params->threadpool);

    സ്ഥിരം int64_t nr = ne1*ne2*ne3;
    സ്ഥിരം int64_t dr = (nr + nth - 1)/nth;
    സ്ഥിരം int64_t ir0 = dr*ith;
    സ്ഥിരം int64_t ir1 = MIN(ir0 + dr, nr);

    ദശാംശം * wdata = (ദശാംശം *) params->wdata + (ne0 + CACHE_വരി_SIZE_F32) * ith;

    വേണ്ടി (int64_t ir = ir0; ir < ir1; ++ir) {
        സ്ഥിരം int64_t i3 = ir/(ne2*ne1);
        സ്ഥിരം int64_t i2 = (ir - i3*ne2*ne1)/ne1;
        സ്ഥിരം int64_t i1 = (ir - i3*ne2*ne1 - i2*ne1);

        സ്ഥിരം int64_t i02 = i2;
        സ്ഥിരം int64_t i03 = i3;

        സ്ഥിരം int64_t i12 = i2;
        സ്ഥിരം int64_t i13 = i3;

        ദശാംശം * d = (ദശാംശം *) ((അക്ഷരം *) dst->data + (i1*nb1 + i2*nb2 + i3*nb3));

        വേണ്ടി (int64_t i01 = 0; i01 < ne01; ++i01) {
            സ്ഥിരം int64_t i11 = i01;
            ggml_fp16_t * s0 = (ggml_fp16_t *) ((അക്ഷരം *) src0->data + (i01*nb01 + i02*nb02 + i03*nb03));
            ദശാംശം * s1 = (ദശാംശം *) ((അക്ഷരം *) src1->data + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
            ggml_fp16_to_fp32_row(s0, wdata, ne0);
            ggml_vec_mad_f32(ne0, d, wdata, *s1);
        }
    }
}

ശൂന്യം ggml_compute_forward_out_prod(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_Q1_0:
        അവസ്ഥ GGML_TYPE_Q2_0:
        അവസ്ഥ GGML_TYPE_Q4_0:
        അവസ്ഥ GGML_TYPE_Q4_1:
        അവസ്ഥ GGML_TYPE_Q5_0:
        അവസ്ഥ GGML_TYPE_Q5_1:
        അവസ്ഥ GGML_TYPE_Q8_0:
        അവസ്ഥ GGML_TYPE_MXFP4:
        അവസ്ഥ GGML_TYPE_NVFP4:
        അവസ്ഥ GGML_TYPE_Q2_K:
        അവസ്ഥ GGML_TYPE_Q3_K:
        അവസ്ഥ GGML_TYPE_Q4_K:
        അവസ്ഥ GGML_TYPE_Q5_K:
        അവസ്ഥ GGML_TYPE_Q6_K:
        അവസ്ഥ GGML_TYPE_TQ1_0:
        അവസ്ഥ GGML_TYPE_TQ2_0:
        അവസ്ഥ GGML_TYPE_IQ2_XXS:
        അവസ്ഥ GGML_TYPE_IQ2_XS:
        അവസ്ഥ GGML_TYPE_IQ3_XXS:
        അവസ്ഥ GGML_TYPE_IQ1_S:
        അവസ്ഥ GGML_TYPE_IQ1_M:
        അവസ്ഥ GGML_TYPE_IQ4_NL:
        അവസ്ഥ GGML_TYPE_IQ4_XS:
        അവസ്ഥ GGML_TYPE_IQ3_S:
        അവസ്ഥ GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_out_prod_q_f32(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F16:
            {
                ggml_compute_forward_out_prod_f16_f32(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_out_prod_f32(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_scale

സ്ഥിതം ശൂന്യം ggml_compute_forward_scale_f32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    ദശാംശം s; // scale factor
    ദശാംശം b; // bias

    memcpy(&s, (ദശാംശം *) dst->op_params + 0, വലുപ്പം(ദശാംശം));
    memcpy(&b, (ദശാംശം *) dst->op_params + 1, വലുപ്പം(ദശാംശം));

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ nc = src0->ne[0];
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr = ggml_nrows(src0);

    // rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    സ്ഥിരം size_t nb01 = src0->nb[1];

    സ്ഥിരം size_t nb1 = dst->nb[1];

    എങ്കിൽ (b == 0.0f) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ i1 = ir0; i1 < ir1; i1++) {
            എങ്കിൽ (dst->data != src0->data) {
                // src0 is same shape as dst => same indices
                // TODO: add x parameter to ggml_vec_scale_f32 and remove this memcpy
                memcpy((അക്ഷരം *)dst->data + i1*nb1, (അക്ഷരം *)src0->data + i1*nb01, nc * വലുപ്പം(ദശാംശം));
            }
            ggml_vec_scale_f32(nc, (ദശാംശം *) ((അക്ഷരം *) dst->data + i1*nb1), s);
        }
    } അല്ലെങ്കിൽ {
        വേണ്ടി (പൂർണ്ണസംഖ്യ i1 = ir0; i1 < ir1; i1++) {
            ggml_vec_mad1_f32(nc,
                (ദശാംശം *) ((അക്ഷരം *) dst->data  + i1*nb1),
                (ദശാംശം *) ((അക്ഷരം *) src0->data + i1*nb1),
                s, b);
        }
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_scale_bf16(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    ദശാംശം s; // scale factor
    ദശാംശം b; // bias

    memcpy(&s, (ദശാംശം *) dst->op_params + 0, വലുപ്പം(ദശാംശം));
    memcpy(&b, (ദശാംശം *) dst->op_params + 1, വലുപ്പം(ദശാംശം));

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ nc = src0->ne[0];
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr = ggml_nrows(src0);

    // rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    വേണ്ടി (പൂർണ്ണസംഖ്യ i1 = ir0; i1 < ir1; i1++) {
        സ്ഥിരം ggml_bf16_t * x = (സ്ഥിരം ggml_bf16_t *) ((സ്ഥിരം അക്ഷരം *) src0->data + i1*src0->nb[1]);
        ggml_bf16_t       * y = (ggml_bf16_t *)       ((അക്ഷരം *)       dst->data  + i1*dst->nb[1]);
        വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < nc; k++) {
            y[k] = GGML_FP32_TO_BF16(s*GGML_BF16_TO_FP32(x[k]) + b);
        }
    }
}

ശൂന്യം ggml_compute_forward_scale(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_scale_f32(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_BF16:
            {
                ggml_compute_forward_scale_bf16(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_set

സ്ഥിതം ശൂന്യം ggml_compute_forward_set_f32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));

    // view src0 and dst with these strides and data offset inbytes during set
    // nb0 is implicitly element_size because src0 and dst are contiguous
    size_t nb1     = ((int32_t *) dst->op_params)[0];
    size_t nb2     = ((int32_t *) dst->op_params)[1];
    size_t nb3     = ((int32_t *) dst->op_params)[2];
    size_t offset  = ((int32_t *) dst->op_params)[3];
    തർക്കം   inplace = (തർക്കം) ((int32_t *) dst->op_params)[4];

    എങ്കിൽ (!inplace) {
        എങ്കിൽ (params->ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            memcpy(
                ((അക്ഷരം *)  dst->data),
                ((അക്ഷരം *) src0->data),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ nr = ggml_nrows(src1);
    സ്ഥിരം പൂർണ്ണസംഖ്യ nc = src1->ne[0];

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne)
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb)

    // src0 and dst as viewed during set
    സ്ഥിരം size_t nb0 = ggml_element_size(src0);

    സ്ഥിരം പൂർണ്ണസംഖ്യ im0 = (ne10 == 0 ? 0 : ne10-1);
    സ്ഥിരം പൂർണ്ണസംഖ്യ im1 = (ne11 == 0 ? 0 : ne11-1);
    സ്ഥിരം പൂർണ്ണസംഖ്യ im2 = (ne12 == 0 ? 0 : ne12-1);
    സ്ഥിരം പൂർണ്ണസംഖ്യ im3 = (ne13 == 0 ? 0 : ne13-1);

    GGML_ASSERT(offset + im0*nb0  + im1*nb1  + im2*nb2  + im3*nb3  <= ggml_nbytes(dst));

    GGML_ASSERT(nb10 == വലുപ്പം(ദശാംശം));

    // rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    വേണ്ടി (പൂർണ്ണസംഖ്യ ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are viewed with shape of src1 and offset
        // => same indices
        സ്ഥിരം പൂർണ്ണസംഖ്യ i3 = ir/(ne12*ne11);
        സ്ഥിരം പൂർണ്ണസംഖ്യ i2 = (ir - i3*ne12*ne11)/ne11;
        സ്ഥിരം പൂർണ്ണസംഖ്യ i1 = (ir - i3*ne12*ne11 - i2*ne11);

        ggml_vec_cpy_f32(nc,
                (ദശാംശം *) ((അക്ഷരം *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + offset),
                (ദശാംശം *) ((അക്ഷരം *) src1->data + i3*nb13 + i2*nb12 + i1*nb11));
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_set_i32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));

    // view src0 and dst with these strides and data offset inbytes during set
    // nb0 is implicitly element_size because src0 and dst are contiguous
    size_t nb1     = ((int32_t *) dst->op_params)[0];
    size_t nb2     = ((int32_t *) dst->op_params)[1];
    size_t nb3     = ((int32_t *) dst->op_params)[2];
    size_t offset  = ((int32_t *) dst->op_params)[3];
    തർക്കം   inplace = (തർക്കം) ((int32_t *) dst->op_params)[4];

    എങ്കിൽ (!inplace) {
        എങ്കിൽ (params->ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            memcpy(
                ((അക്ഷരം *)  dst->data),
                ((അക്ഷരം *) src0->data),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ nr = ggml_nrows(src1);
    സ്ഥിരം പൂർണ്ണസംഖ്യ nc = src1->ne[0];

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne)
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb)

    // src0 and dst as viewed during set
    സ്ഥിരം size_t nb0 = ggml_element_size(src0);

    സ്ഥിരം പൂർണ്ണസംഖ്യ im0 = (ne10 == 0 ? 0 : ne10-1);
    സ്ഥിരം പൂർണ്ണസംഖ്യ im1 = (ne11 == 0 ? 0 : ne11-1);
    സ്ഥിരം പൂർണ്ണസംഖ്യ im2 = (ne12 == 0 ? 0 : ne12-1);
    സ്ഥിരം പൂർണ്ണസംഖ്യ im3 = (ne13 == 0 ? 0 : ne13-1);

    GGML_ASSERT(offset + im0*nb0  + im1*nb1  + im2*nb2  + im3*nb3  <= ggml_nbytes(dst));

    GGML_ASSERT(nb10 == വലുപ്പം(int32_t));

    // rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    വേണ്ടി (പൂർണ്ണസംഖ്യ ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are viewed with shape of src1 and offset
        // => same indices
        സ്ഥിരം പൂർണ്ണസംഖ്യ i3 = ir/(ne12*ne11);
        സ്ഥിരം പൂർണ്ണസംഖ്യ i2 = (ir - i3*ne12*ne11)/ne11;
        സ്ഥിരം പൂർണ്ണസംഖ്യ i1 = (ir - i3*ne12*ne11 - i2*ne11);

        ggml_vec_cpy_i32(nc,
                (int32_t *) ((അക്ഷരം *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + offset),
                (int32_t *) ((അക്ഷരം *) src1->data + i3*nb13 + i2*nb12 + i1*nb11));
    }
}

ശൂന്യം ggml_compute_forward_set(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_set_f32(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_I32:
            {
                ggml_compute_forward_set_i32(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F16:
        അവസ്ഥ GGML_TYPE_BF16:
        അവസ്ഥ GGML_TYPE_Q1_0:
        അവസ്ഥ GGML_TYPE_Q2_0:
        അവസ്ഥ GGML_TYPE_Q4_0:
        അവസ്ഥ GGML_TYPE_Q4_1:
        അവസ്ഥ GGML_TYPE_Q5_0:
        അവസ്ഥ GGML_TYPE_Q5_1:
        അവസ്ഥ GGML_TYPE_Q8_0:
        അവസ്ഥ GGML_TYPE_Q8_1:
        അവസ്ഥ GGML_TYPE_MXFP4:
        അവസ്ഥ GGML_TYPE_NVFP4:
        അവസ്ഥ GGML_TYPE_Q2_K:
        അവസ്ഥ GGML_TYPE_Q3_K:
        അവസ്ഥ GGML_TYPE_Q4_K:
        അവസ്ഥ GGML_TYPE_Q5_K:
        അവസ്ഥ GGML_TYPE_Q6_K:
        അവസ്ഥ GGML_TYPE_TQ1_0:
        അവസ്ഥ GGML_TYPE_TQ2_0:
        അവസ്ഥ GGML_TYPE_IQ2_XXS:
        അവസ്ഥ GGML_TYPE_IQ2_XS:
        അവസ്ഥ GGML_TYPE_IQ3_XXS:
        അവസ്ഥ GGML_TYPE_IQ1_S:
        അവസ്ഥ GGML_TYPE_IQ1_M:
        അവസ്ഥ GGML_TYPE_IQ4_NL:
        അവസ്ഥ GGML_TYPE_IQ4_XS:
        അവസ്ഥ GGML_TYPE_IQ3_S:
        അവസ്ഥ GGML_TYPE_IQ2_S:
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_cpy

ശൂന്യം ggml_compute_forward_cpy(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_dup(params, dst);
}

// ggml_compute_forward_cont

ശൂന്യം ggml_compute_forward_cont(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_dup(params, dst);
}

// ggml_compute_forward_get_rows

സ്ഥിതം ശൂന്യം ggml_compute_forward_get_rows_q(
        സ്ഥിരം ggml_compute_params * params,
              ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ദ്വിമാന_OP_LOCALS

    സ്ഥിരം int64_t nc = ne00;
    സ്ഥിരം int64_t nr = ggml_nelements(src1);

    സ്ഥിരം ggml_type type = src0->type;
    ggml_to_float_t സ്ഥിരം dequantize_row_q = ggml_get_type_traits(type)->to_float;

    പ്രതിജ്ഞാ(ne0  == nc);
    പ്രതിജ്ഞാ(ne02 == ne11);
    പ്രതിജ്ഞാ(nb00 == ggml_type_size(type));
    പ്രതിജ്ഞാ(ggml_nrows(dst) == nr);

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    // rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    വേണ്ടി (int64_t i = ir0; i < ir1; ++i) {
        സ്ഥിരം int64_t i12 = i/(ne11*ne10);
        സ്ഥിരം int64_t i11 = (i - i12*ne11*ne10)/ne10;
        സ്ഥിരം int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        സ്ഥിരം int64_t i01 = *(int32_t *) ((അക്ഷരം *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        dequantize_row_q(
                (സ്ഥിരം ശൂന്യം *) ((അക്ഷരം *) src0->data + i01*nb01 + i11*nb02 + i12*nb03),
                     (ദശാംശം *) ((അക്ഷരം *)  dst->data + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_get_rows_f16(
        സ്ഥിരം ggml_compute_params * params,
              ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ദ്വിമാന_OP_LOCALS

    സ്ഥിരം int64_t nc = ne00;
    സ്ഥിരം int64_t nr = ggml_nelements(src1);

    പ്രതിജ്ഞാ(ne0  == nc);
    പ്രതിജ്ഞാ(ne02 == ne11);
    പ്രതിജ്ഞാ(nb00 == വലുപ്പം(ggml_fp16_t));
    പ്രതിജ്ഞാ(ggml_nrows(dst) == nr);

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    // rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    വേണ്ടി (int64_t i = ir0; i < ir1; ++i) {
        സ്ഥിരം int64_t i12 = i/(ne11*ne10);
        സ്ഥിരം int64_t i11 = (i - i12*ne11*ne10)/ne10;
        സ്ഥിരം int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        സ്ഥിരം int64_t i01 = *(int32_t *) ((അക്ഷരം *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        ggml_cpu_fp16_to_fp32(
            (സ്ഥിരം ggml_fp16_t*) ((അക്ഷരം *) src0->data + i01*nb01 + i11*nb02 + i12*nb03),
                       (ദശാംശം *) ((അക്ഷരം *)  dst->data + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_get_rows_bf16(
        സ്ഥിരം ggml_compute_params * params,
              ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ദ്വിമാന_OP_LOCALS

    സ്ഥിരം int64_t nc = ne00;
    സ്ഥിരം int64_t nr = ggml_nelements(src1);

    പ്രതിജ്ഞാ(ne0  == nc);
    പ്രതിജ്ഞാ(ne02 == ne11);
    പ്രതിജ്ഞാ(nb00 == വലുപ്പം(ggml_bf16_t));
    പ്രതിജ്ഞാ(ggml_nrows(dst) == nr);

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    // rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    വേണ്ടി (int64_t i = ir0; i < ir1; ++i) {
        സ്ഥിരം int64_t i12 = i/(ne11*ne10);
        സ്ഥിരം int64_t i11 = (i - i12*ne11*ne10)/ne10;
        സ്ഥിരം int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        സ്ഥിരം int64_t i01 = *(int32_t *) ((അക്ഷരം *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        ggml_cpu_bf16_to_fp32(
            (സ്ഥിരം ggml_bf16_t *) ((അക്ഷരം *) src0->data + i01*nb01 + i11*nb02 + i12*nb03),
                        (ദശാംശം *) ((അക്ഷരം *)  dst->data + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_get_rows_f32(
        സ്ഥിരം ggml_compute_params * params,
              ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ദ്വിമാന_OP_LOCALS

    സ്ഥിരം int64_t nc = ne00;
    സ്ഥിരം int64_t nr = ggml_nelements(src1);

    പ്രതിജ്ഞാ(ne0  == nc);
    പ്രതിജ്ഞാ(ne02 == ne11);
    പ്രതിജ്ഞാ(nb00 == വലുപ്പം(ദശാംശം));
    പ്രതിജ്ഞാ(ggml_nrows(dst) == nr);

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    // rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    വേണ്ടി (int64_t i = ir0; i < ir1; ++i) {
        സ്ഥിരം int64_t i12 = i/(ne11*ne10);
        സ്ഥിരം int64_t i11 = (i - i12*ne11*ne10)/ne10;
        സ്ഥിരം int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        സ്ഥിരം int64_t i01 = *(int32_t *) ((അക്ഷരം *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        ggml_vec_cpy_f32(nc,
                (ദശാംശം *) ((അക്ഷരം *)  dst->data + i10*nb1  + i11*nb2  + i12*nb3),
                (ദശാംശം *) ((അക്ഷരം *) src0->data + i01*nb01 + i11*nb02 + i12*nb03));
    }
}

ശൂന്യം ggml_compute_forward_get_rows(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_Q1_0:
        അവസ്ഥ GGML_TYPE_Q2_0:
        അവസ്ഥ GGML_TYPE_Q4_0:
        അവസ്ഥ GGML_TYPE_Q4_1:
        അവസ്ഥ GGML_TYPE_Q5_0:
        അവസ്ഥ GGML_TYPE_Q5_1:
        അവസ്ഥ GGML_TYPE_Q8_0:
        അവസ്ഥ GGML_TYPE_Q8_1:
        അവസ്ഥ GGML_TYPE_MXFP4:
        അവസ്ഥ GGML_TYPE_NVFP4:
        അവസ്ഥ GGML_TYPE_Q2_K:
        അവസ്ഥ GGML_TYPE_Q3_K:
        അവസ്ഥ GGML_TYPE_Q4_K:
        അവസ്ഥ GGML_TYPE_Q5_K:
        അവസ്ഥ GGML_TYPE_Q6_K:
        അവസ്ഥ GGML_TYPE_TQ1_0:
        അവസ്ഥ GGML_TYPE_TQ2_0:
        അവസ്ഥ GGML_TYPE_IQ2_XXS:
        അവസ്ഥ GGML_TYPE_IQ2_XS:
        അവസ്ഥ GGML_TYPE_IQ3_XXS:
        അവസ്ഥ GGML_TYPE_IQ1_S:
        അവസ്ഥ GGML_TYPE_IQ1_M:
        അവസ്ഥ GGML_TYPE_IQ4_NL:
        അവസ്ഥ GGML_TYPE_IQ4_XS:
        അവസ്ഥ GGML_TYPE_IQ3_S:
        അവസ്ഥ GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_get_rows_q(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F16:
            {
                ggml_compute_forward_get_rows_f16(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_BF16:
            {
                ggml_compute_forward_get_rows_bf16(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F32:
        അവസ്ഥ GGML_TYPE_I32:
            {
                ggml_compute_forward_get_rows_f32(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
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
സ്ഥിതം ശൂന്യം ggml_compute_forward_set_rows_impl(
        സ്ഥിരം ggml_compute_params * params,
              ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ദ്വിമാന_OP_LOCALS

    സ്ഥിരം int64_t nc = ne00;
    സ്ഥിരം int64_t nr = ne01;

    പ്രതിജ്ഞാ(ne0  == nc);
    പ്രതിജ്ഞാ(ne2  == ne02);
    പ്രതിജ്ഞാ(ne3  == ne03);
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    പ്രതിജ്ഞാ(ne02 % ne11 == 0);
    പ്രതിജ്ഞാ(ne03 % ne12 == 0);

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    // rows per thread
    സ്ഥിരം int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം int64_t ir0 = dr*ith;
    സ്ഥിരം int64_t ir1 = std::min(ir0 + dr, nr);

    സ്ഥിരം size_t rs = ggml_row_size(src0->type, nc);

    ggml_from_float_t സ്ഥിരം from_float = ggml_get_type_traits_cpu(dst->type)->from_float;

    വേണ്ടി (int64_t i03 = 0; i03 < ne03; ++i03) {
        വേണ്ടി (int64_t i02 = 0; i02 < ne02; ++i02) {
            വേണ്ടി (int64_t i = ir0; i < ir1; ++i) {
                സ്ഥിരം int64_t i12 = i03%ne12;
                സ്ഥിരം int64_t i11 = i02%ne11;
                സ്ഥിരം int64_t i10 = i;

                സ്ഥിരം int64_t i1 = *(idx_t *) ((അക്ഷരം *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);

                GGML_ASSERT(i1 >= 0 && i1 < ne1);

                എങ്കിൽ constexpr (std::is_same_v<src_t, ദശാംശം>) {
                    from_float(
                            (സ്ഥിരം ദശാംശം *) ((അക്ഷരം *) src0->data +  i*nb01 + i02*nb02 + i03*nb03),
                                            ((അക്ഷരം *)  dst->data + i1*nb1  + i02*nb2  + i03*nb3), nc);
                } അല്ലെങ്കിൽ എങ്കിൽ constexpr (std::is_same_v<src_t, ggml_fp16_t>) {
                    എങ്കിൽ (dst->type == GGML_TYPE_F16) {
                        memcpy(
                                            ((അക്ഷരം *)  dst->data + i1*nb1  + i02*nb2  + i03*nb3),
                                            ((അക്ഷരം *) src0->data +  i*nb01 + i02*nb02 + i03*nb03),
                                            rs);
                    } അല്ലെങ്കിൽ {
                        ദശാംശം * wdata = (ദശാംശം *) params->wdata + (nc + CACHE_വരി_SIZE_F32) * ith;
                        ggml_fp16_to_fp32_row(
                                (സ്ഥിരം ggml_fp16_t *) ((അക്ഷരം *) src0->data + i*nb01 + i02*nb02 + i03*nb03),
                                wdata, nc);
                        from_float(wdata,
                                ((അക്ഷരം *) dst->data + i1*nb1 + i02*nb2 + i03*nb3), nc);
                    }
                } അല്ലെങ്കിൽ {
                    GGML_ABORT("src0->type = %d (%s) not supported", src0->type, ggml_type_name(src0->type));
                }
            }
        }
    }
}

ശൂന്യം ggml_compute_forward_set_rows(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                എങ്കിൽ (src1->type == GGML_TYPE_I64) {
                    ggml_compute_forward_set_rows_impl<ദശാംശം, int64_t>(params, dst);
                } അല്ലെങ്കിൽ എങ്കിൽ (src1->type == GGML_TYPE_I32) {
                    ggml_compute_forward_set_rows_impl<ദശാംശം, int32_t>(params, dst);
                } അല്ലെങ്കിൽ {
                    GGML_ABORT("src1->type = %d (%s) not supported", src1->type, ggml_type_name(src1->type));
                }
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F16:
            {
                എങ്കിൽ (src1->type == GGML_TYPE_I64) {
                    ggml_compute_forward_set_rows_impl<ggml_fp16_t, int64_t>(params, dst);
                } അല്ലെങ്കിൽ എങ്കിൽ (src1->type == GGML_TYPE_I32) {
                    ggml_compute_forward_set_rows_impl<ggml_fp16_t, int32_t>(params, dst);
                } അല്ലെങ്കിൽ {
                    GGML_ABORT("src1->type = %d (%s) not supported", src1->type, ggml_type_name(src1->type));
                }
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("src0->type = %d (%s) not supported", src0->type, ggml_type_name(src0->type));
            }
    }
}

// ggml_compute_forward_get_rows_back

സ്ഥിതം ശൂന്യം ggml_compute_forward_get_rows_back_f32_f16(
        സ്ഥിരം ggml_compute_params * params,
              ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];

    എങ്കിൽ (params->ith != 0) {
        തിരികെ;
    }

    GGML_ASSERT(ggml_is_contiguous(dst));

    // ggml_compute_forward_dup_same_cont(params, opt0, dst);

    memset(dst->data, 0, ggml_nbytes(dst));

    സ്ഥിരം പൂർണ്ണസംഖ്യ nc = src0->ne[0];
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr = ggml_nelements(src1);

    GGML_ASSERT( dst->ne[0] == nc);
    GGML_ASSERT(src0->nb[0] == വലുപ്പം(ggml_fp16_t));

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < nr; ++i) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ r = ((int32_t *) src1->data)[i];

        GGML_ASSERT(r >= 0 && r < dst->ne[1]);

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < nc; ++j) {
            ggml_fp16_t v = ((ggml_fp16_t *) ((അക്ഷരം *) src0->data + i*src0->nb[1]))[j];
            ((ദശാംശം *) ((അക്ഷരം *) dst->data + r*dst->nb[1]))[j] += GGML_CPU_FP16_TO_FP32(v);
        }
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_get_rows_back_f32(
        സ്ഥിരം ggml_compute_params * params,
              ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];

    എങ്കിൽ (params->ith != 0) {
        തിരികെ;
    }

    GGML_ASSERT(ggml_is_contiguous(dst));

    // ggml_compute_forward_dup_same_cont(params, opt0, dst);

    memset(dst->data, 0, ggml_nbytes(dst));

    സ്ഥിരം പൂർണ്ണസംഖ്യ nc = src0->ne[0];
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr = ggml_nelements(src1);

    GGML_ASSERT( dst->ne[0] == nc);
    GGML_ASSERT(src0->nb[0] == വലുപ്പം(ദശാംശം));

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < nr; ++i) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ r = ((int32_t *) src1->data)[i];

        GGML_ASSERT(r >= 0 && r < dst->ne[1]);

        ggml_vec_add_f32(nc,
                (ദശാംശം *) ((അക്ഷരം *)  dst->data + r*dst->nb[1]),
                (ദശാംശം *) ((അക്ഷരം *)  dst->data + r*dst->nb[1]),
                (ദശാംശം *) ((അക്ഷരം *) src0->data + i*src0->nb[1]));
    }
}

ശൂന്യം ggml_compute_forward_get_rows_back(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F16:
            {
                ggml_compute_forward_get_rows_back_f32_f16(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_get_rows_back_f32(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
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

സ്ഥിതം ശൂന്യം ggml_compute_forward_diag_f32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    എങ്കിൽ (params->ith != 0) {
        തിരികെ;
    }

    // TODO: handle transposed/permuted matrices

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(ne00 == ne0);
    GGML_ASSERT(ne00 == ne1);
    GGML_ASSERT(ne01 == 1);
    GGML_ASSERT(ne02 == ne2);
    GGML_ASSERT(ne03 == ne3);

    GGML_ASSERT(nb00 == വലുപ്പം(ദശാംശം));
    GGML_ASSERT(nb0  == വലുപ്പം(ദശാംശം));

    വേണ്ടി (പൂർണ്ണസംഖ്യ i3 = 0; i3 < ne3; i3++) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ i2 = 0; i2 < ne2; i2++) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ i1 = 0; i1 < ne1; i1++) {
                ദശാംശം * d = (ദശാംശം *)((അക്ഷരം *)  dst->data + i3*nb3  + i2*nb2 + i1*nb1);
                ദശാംശം * s = (ദശാംശം *)((അക്ഷരം *) src0->data + i3*nb03 + i2*nb02);
                വേണ്ടി (പൂർണ്ണസംഖ്യ i0 = 0; i0 < i1; i0++) {
                    d[i0] = 0;
                }
                d[i1] = s[i1];
                വേണ്ടി (പൂർണ്ണസംഖ്യ i0 = i1+1; i0 < ne0; i0++) {
                    d[i0] = 0;
                }
            }
        }
    }
}

ശൂന്യം ggml_compute_forward_diag(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_diag_f32(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_diag_mask_inf

സ്ഥിതം ശൂന്യം ggml_compute_forward_diag_mask_f32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst,
        സ്ഥിരം ദശാംശം value) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ  n_past  = ((int32_t *) dst->op_params)[0];
    സ്ഥിരം തർക്കം inplace = src0->data == dst->data;

    GGML_ASSERT(n_past >= 0);

    എങ്കിൽ (!inplace) {
        എങ്കിൽ (ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
            GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));
            memcpy(
                ((അക്ഷരം *)  dst->data),
                ((അക്ഷരം *) src0->data),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    // TODO: handle transposed/permuted matrices

    സ്ഥിരം പൂർണ്ണസംഖ്യ n  = ggml_nrows(src0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ nc = src0->ne[0];
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr = src0->ne[1];
    സ്ഥിരം പൂർണ്ണസംഖ്യ nz = n/nr;

    GGML_ASSERT( dst->nb[0] == വലുപ്പം(ദശാംശം));
    GGML_ASSERT(src0->nb[0] == വലുപ്പം(ദശാംശം));

    വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < nz; k++) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = ith; j < nr; j += nth) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ i = n_past; i < nc; i++) {
                എങ്കിൽ (i > n_past + j) {
                    *(ദശാംശം *)((അക്ഷരം *) dst->data + k*dst->nb[2] + j*dst->nb[1] + i*dst->nb[0]) = value;
                }
            }
        }
    }
}

ശൂന്യം ggml_compute_forward_diag_mask_inf(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_diag_mask_f32(params, dst, -INFINITY);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ശൂന്യം ggml_compute_forward_diag_mask_zero(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_diag_mask_f32(params, dst, 0);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_soft_max

സ്ഥിതം ശൂന്യം ggml_compute_forward_soft_max_f32(
        സ്ഥിരം ggml_compute_params * params,
              ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];
    സ്ഥിരം ggml_tensor * src2 = dst->src[2];

    പ്രതിജ്ഞാ(ggml_is_contiguous(dst));
    പ്രതിജ്ഞാ(ggml_are_same_shape(src0, dst));

    ദശാംശം scale    = 1.0f;
    ദശാംശം max_bias = 0.0f;

    memcpy(&scale,    (ദശാംശം *) dst->op_params + 0, വലുപ്പം(ദശാംശം));
    memcpy(&max_bias, (ദശാംശം *) dst->op_params + 1, വലുപ്പം(ദശാംശം));

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    സ്ഥിരം int64_t nb11 = src1 ? src1->nb[1] : 1;
    സ്ഥിരം int64_t nb12 = src1 ? src1->nb[2] : 1;
    സ്ഥിരം int64_t nb13 = src1 ? src1->nb[3] : 1;

    സ്ഥിരം int64_t ne12 = src1 ? src1->ne[2] : 1;
    സ്ഥിരം int64_t ne13 = src1 ? src1->ne[3] : 1;

    // TODO: is this supposed to be ceil instead of floor?
    //       https://huggingface.co/mosaicml/mpt-7b/blob/main/attention.py#L370
    സ്ഥിരം uint32_t n_head      = ne02;
    സ്ഥിരം uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));

    സ്ഥിരം ദശാംശം m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    സ്ഥിരം ദശാംശം m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    ദശാംശം * wp = (ദശാംശം *) params->wdata + (ne00 + CACHE_വരി_SIZE_F32) * ith;

    സ്ഥിരം തർക്കം use_f16 = (src1 && src1->type == GGML_TYPE_F16);

    // sinks
    സ്ഥിരം ദശാംശം * sk = src2 ? (ദശാംശം *)((അക്ഷരം *) src2->data) : nullptr;

    വേണ്ടി (int64_t i03 = 0; i03 < ne03; i03++) {
        വേണ്ടി (int64_t i02 = 0; i02 < ne02; i02++) {
            വേണ്ടി (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                സ്ഥിരം int64_t i11 = i01;
                സ്ഥിരം int64_t i12 = i02%ne12;
                സ്ഥിരം int64_t i13 = i03%ne13;

                // ALiBi
                സ്ഥിരം uint32_t h = i02; // head
                സ്ഥിരം ദശാംശം slope = (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1) : 1.0f;

                ദശാംശം * sp = (ദശാംശം *)((അക്ഷരം *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
                ദശാംശം * dp = (ദശാംശം *)((അക്ഷരം *)  dst->data + i01*nb1  + i02*nb2  + i03*nb3);

                // broadcast the mask across rows
                ggml_fp16_t * mp_f16 = src1 ? (ggml_fp16_t *)((അക്ഷരം *) src1->data + i11*nb11 + i12*nb12 + i13*nb13) : ശൂന്യ;
                ദശാംശം       * mp_f32 = src1 ? (ദശാംശം       *)((അക്ഷരം *) src1->data + i11*nb11 + i12*nb12 + i13*nb13) : ശൂന്യ;

                ggml_vec_cpy_f32  (ne00, wp, sp);
                ggml_vec_scale_f32(ne00, wp, scale);
                എങ്കിൽ (mp_f32) {
                    എങ്കിൽ (use_f16) {
                        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < ne00; ++i) {
                            wp[i] += slope*GGML_CPU_FP16_TO_FP32(mp_f16[i]);
                        }
                    } അല്ലെങ്കിൽ {
                        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < ne00; ++i) {
                            wp[i] += slope*mp_f32[i];
                        }
                    }
                }

#ifndef NDEBUG
                വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < ne00; ++i) {
                    //printf("p[%d] = %f\n", i, p[i]);
                    പ്രതിജ്ഞാ(!isnan(wp[i]));
                }
#endif // NDEBUG

                ദശാംശം max = -INFINITY;
                ggml_vec_max_f32(ne00, &max, wp);

                // if we have sinks, make a correction as if they were included in the softmax
                എങ്കിൽ (sk) {
                    max = MAX(max, sk[i02]);
                }

                ggml_float sum = ggml_vec_soft_max_f32(ne00, dp, wp, max);
                പ്രതിജ്ഞാ(sum > 0.0);

                എങ്കിൽ (sk) {
                    sum += (ggml_float) expf(sk[i02] - max);
                }

                sum = 1.0/sum;
                ggml_vec_scale_f32(ne00, dp, sum);

#ifndef NDEBUG
                വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < ne00; ++i) {
                    പ്രതിജ്ഞാ(!isnan(dp[i]));
                    പ്രതിജ്ഞാ(!isinf(dp[i]));
                }
#endif // NDEBUG
            }
        }
    }
}

ശൂന്യം ggml_compute_forward_soft_max(
        സ്ഥിരം ggml_compute_params * params,
              ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_soft_max_f32(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}


// ggml_compute_forward_soft_max_ext_back

സ്ഥിതം ശൂന്യം ggml_compute_forward_soft_max_ext_back_f32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_are_same_shape(src1, dst));

    ദശാംശം scale    = 1.0f;
    ദശാംശം max_bias = 0.0f;

    memcpy(&scale,    (സ്ഥിരം ദശാംശം *) dst->op_params + 0, വലുപ്പം(ദശാംശം));
    memcpy(&max_bias, (സ്ഥിരം ദശാംശം *) dst->op_params + 1, വലുപ്പം(ദശാംശം));

    GGML_ASSERT(max_bias == 0.0f);

    // TODO: handle transposed/permuted matrices

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ nc = src0->ne[0];
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr = ggml_nrows(src0);

    // rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    വേണ്ടി (പൂർണ്ണസംഖ്യ i1 = ir0; i1 < ir1; i1++) {
        ദശാംശം *dy = (ദശാംശം *)((അക്ഷരം *) src0->data + i1*src0->nb[1]);
        ദശാംശം *y  = (ദശാംശം *)((അക്ഷരം *) src1->data + i1*src1->nb[1]);
        ദശാംശം *dx = (ദശാംശം *)((അക്ഷരം *) dst->data  + i1*dst->nb[1]);

#ifndef NDEBUG
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            പ്രതിജ്ഞാ(!isnan(dy[i]));
            പ്രതിജ്ഞാ(!isnan(y[i]));
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
        ദശാംശം dot_y_dy = 0;
        ggml_vec_dot_f32(nc, &dot_y_dy, 0, y, 0, dy, 0, 1);
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < nc; i++) {
            dx[i] = scale * (dy[i] - dot_y_dy) * y[i];
        }

#ifndef NDEBUG
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < nc; ++i) {
            പ്രതിജ്ഞാ(!isnan(dx[i]));
            പ്രതിജ്ഞാ(!isinf(dx[i]));
        }
#endif // NDEBUG
    }
}

ശൂന്യം ggml_compute_forward_soft_max_ext_back(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_soft_max_ext_back_f32(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_clamp

സ്ഥിതം ശൂന്യം ggml_compute_forward_clamp_f32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    ദശാംശം min;
    ദശാംശം max;
    memcpy(&min, (ദശാംശം *) dst->op_params + 0, വലുപ്പം(ദശാംശം));
    memcpy(&max, (ദശാംശം *) dst->op_params + 1, വലുപ്പം(ദശാംശം));

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ n  = ggml_nrows(src0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ nc = src0->ne[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == വലുപ്പം(ദശാംശം));
    GGML_ASSERT(nb00 == വലുപ്പം(ദശാംശം));

    വേണ്ടി (പൂർണ്ണസംഖ്യ j = ith; j < n; j += nth) {
        സ്ഥിരം int64_t i1 = j % ne01;
        സ്ഥിരം int64_t i2 = (j / ne01) % ne02;
        സ്ഥിരം int64_t i3 = j / (ne01*ne02);

        ദശാംശം * dst_ptr  = (ദശാംശം *) ((അക്ഷരം *)  dst->data + i1*nb1  + i2*nb2  + i3*nb3);
        ദശാംശം * src0_ptr = (ദശാംശം *) ((അക്ഷരം *) src0->data + i1*nb01 + i2*nb02 + i3*nb03);

        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < nc; i++) {
            dst_ptr[i] = MAX(MIN(src0_ptr[i], max), min);
        }
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_clamp_f16(
    സ്ഥിരം ggml_compute_params * params,
    ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    ദശാംശം min;
    ദശാംശം max;
    memcpy(&min, (ദശാംശം *) dst->op_params + 0, വലുപ്പം(ദശാംശം));
    memcpy(&max, (ദശാംശം *) dst->op_params + 1, വലുപ്പം(ദശാംശം));

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ n  = ggml_nrows(src0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ nc = src0->ne[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == വലുപ്പം(ggml_fp16_t));
    GGML_ASSERT(nb00 == വലുപ്പം(ggml_fp16_t));

    വേണ്ടി (പൂർണ്ണസംഖ്യ j = ith; j < n; j += nth) {
        സ്ഥിരം int64_t i1 = j % ne01;
        സ്ഥിരം int64_t i2 = (j / ne01) % ne02;
        സ്ഥിരം int64_t i3 = j / (ne01*ne02);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((അക്ഷരം *)  dst->data + i1*nb1  + i2*nb2  + i3*nb3);
        ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((അക്ഷരം *) src0->data + i1*nb01 + i2*nb02 + i3*nb03);

        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < nc; i++) {
            ദശാംശം v = GGML_CPU_FP16_TO_FP32(src0_ptr[i]);
            dst_ptr[i] = GGML_CPU_FP32_TO_FP16(MAX(MIN(v, max), min));
        }
    }
}

ശൂന്യം ggml_compute_forward_clamp(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_clamp_f32(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F16:
            {
                ggml_compute_forward_clamp_f16(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_BF16:
        അവസ്ഥ GGML_TYPE_Q1_0:
        അവസ്ഥ GGML_TYPE_Q2_0:
        അവസ്ഥ GGML_TYPE_Q4_0:
        അവസ്ഥ GGML_TYPE_Q4_1:
        അവസ്ഥ GGML_TYPE_Q5_0:
        അവസ്ഥ GGML_TYPE_Q5_1:
        അവസ്ഥ GGML_TYPE_Q8_0:
        അവസ്ഥ GGML_TYPE_Q8_1:
        അവസ്ഥ GGML_TYPE_MXFP4:
        അവസ്ഥ GGML_TYPE_NVFP4:
        അവസ്ഥ GGML_TYPE_Q2_K:
        അവസ്ഥ GGML_TYPE_Q3_K:
        അവസ്ഥ GGML_TYPE_Q4_K:
        അവസ്ഥ GGML_TYPE_Q5_K:
        അവസ്ഥ GGML_TYPE_Q6_K:
        അവസ്ഥ GGML_TYPE_TQ1_0:
        അവസ്ഥ GGML_TYPE_TQ2_0:
        അവസ്ഥ GGML_TYPE_IQ2_XXS:
        അവസ്ഥ GGML_TYPE_IQ2_XS:
        അവസ്ഥ GGML_TYPE_IQ3_XXS:
        അവസ്ഥ GGML_TYPE_IQ1_S:
        അവസ്ഥ GGML_TYPE_IQ1_M:
        അവസ്ഥ GGML_TYPE_IQ4_NL:
        അവസ്ഥ GGML_TYPE_IQ4_XS:
        അവസ്ഥ GGML_TYPE_IQ3_S:
        അവസ്ഥ GGML_TYPE_IQ2_S:
        അവസ്ഥ GGML_TYPE_Q8_K:
        അവസ്ഥ GGML_TYPE_I8:
        അവസ്ഥ GGML_TYPE_I16:
        അവസ്ഥ GGML_TYPE_I32:
        അവസ്ഥ GGML_TYPE_I64:
        അവസ്ഥ GGML_TYPE_F64:
        അവസ്ഥ GGML_TYPE_എണ്ണം:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_rope

സ്ഥിതം ദശാംശം rope_yarn_ramp(സ്ഥിരം ദശാംശം താഴെയുള്ള, സ്ഥിരം ദശാംശം മുകളിലെ, സ്ഥിരം പൂർണ്ണസംഖ്യ i0) {
    സ്ഥിരം ദശാംശം y = (i0 / 2 - താഴെയുള്ള) / MAX(0.001f, മുകളിലെ - താഴെയുള്ള);
    തിരികെ 1 - MIN(1, MAX(0, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
സ്ഥിതം ശൂന്യം rope_yarn(
    ദശാംശം theta_extrap, ദശാംശം freq_scale, ദശാംശം corr_dims[2], int64_t i0, ദശാംശം ext_factor, ദശാംശം mscale,
    ദശാംശം * cos_theta, ദശാംശം * sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    ദശാംശം theta_interp = freq_scale * theta_extrap;
    ദശാംശം theta = theta_interp;
    എങ്കിൽ (ext_factor != 0.0f) {
        ദശാംശം ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
}

സ്ഥിതം ശൂന്യം ggml_rope_cache_init(
     ദശാംശം theta_base, ദശാംശം freq_scale, സ്ഥിരം ദശാംശം * freq_factors, ദശാംശം corr_dims[2], int64_t ne0, ദശാംശം ext_factor, ദശാംശം mscale,
     ദശാംശം * cache, ദശാംശം sin_sign, ദശാംശം theta_scale) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    ദശാംശം theta = theta_base;
    വേണ്ടി (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        സ്ഥിരം ദശാംശം ff = freq_factors ? freq_factors[i0/2] : 1.0f;
        rope_yarn(
            theta/ff, freq_scale, corr_dims, i0, ext_factor, mscale, &cache[i0 + 0], &cache[i0 + 1]
        );
        cache[i0 + 1] *= sin_sign;

        theta *= theta_scale;
    }
}

സ്ഥിതം ശൂന്യം ggml_mrope_cache_init(
     ദശാംശം theta_base_t, ദശാംശം theta_base_h, ദശാംശം theta_base_w, ദശാംശം theta_base_e, പൂർണ്ണസംഖ്യ sections[4], തർക്കം is_imrope, തർക്കം indep_sects,
     ദശാംശം freq_scale, സ്ഥിരം ദശാംശം * freq_factors, ദശാംശം corr_dims[2], int64_t ne0, ദശാംശം ext_factor, ദശാംശം mscale,
     ദശാംശം * cache, ദശാംശം sin_sign, ദശാംശം theta_scale) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    ദശാംശം theta_t = theta_base_t;
    ദശാംശം theta_h = theta_base_h;
    ദശാംശം theta_w = theta_base_w;
    ദശാംശം theta_e = theta_base_e;  // extra position id for vision encoder
    പൂർണ്ണസംഖ്യ sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    പൂർണ്ണസംഖ്യ sec_w = sections[1] + sections[0];
    പൂർണ്ണസംഖ്യ sec_e = sections[2] + sec_w;
    GGML_ASSERT(sect_dims <= ne0);

    വേണ്ടി (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        സ്ഥിരം ദശാംശം ff = freq_factors ? freq_factors[i0/2] : 1.0f;

        പൂർണ്ണസംഖ്യ sector = (i0 / 2) % sect_dims;
        എങ്കിൽ (indep_sects) {
            // compute theta independently for each dim sections
            // (i.e. reset corresponding theta when `i0` go from one section to another)
            എങ്കിൽ (sector == 0) {
                theta_t = theta_base_t;
            }
            അല്ലെങ്കിൽ എങ്കിൽ (sector == sections[0]) {
                theta_h = theta_base_h;;
            }
            അല്ലെങ്കിൽ എങ്കിൽ (sector == sec_w) {
                theta_w = theta_base_w;
            }
            അല്ലെങ്കിൽ എങ്കിൽ (sector == sec_e) {
                theta_e = theta_base_e;
            }
        }

        ദശാംശം theta = theta_t;
        എങ്കിൽ (is_imrope) { // qwen3vl apply interleaved mrope
            എങ്കിൽ (sector % 3 == 1 && sector < 3 * sections[1]) {
                theta = theta_h;
            } അല്ലെങ്കിൽ എങ്കിൽ (sector % 3 == 2 && sector < 3 * sections[2]) {
                theta = theta_w;
            } അല്ലെങ്കിൽ എങ്കിൽ (sector % 3 == 0 && sector < 3 * sections[0]) {
                theta = theta_t;
            } അല്ലെങ്കിൽ {
                theta = theta_e;
            }
        } അല്ലെങ്കിൽ {
            എങ്കിൽ (sector >= sections[0] && sector < sec_w) {
                theta = theta_h;
            }
            അല്ലെങ്കിൽ എങ്കിൽ (sector >= sec_w && sector < sec_w + sections[2]) {
                theta = theta_w;
            }
            അല്ലെങ്കിൽ എങ്കിൽ (sector >= sec_w + sections[2]) {
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
സ്ഥിതം ശൂന്യം rotate_pairs(സ്ഥിരം int64_t n, സ്ഥിരം int64_t n_offset, സ്ഥിരം ദശാംശം * cache, സ്ഥിരം T * src_data, T * dst_data, സ്ഥിരം പൂർണ്ണസംഖ്യ scale = 2) {
  വേണ്ടി (int64_t i0 = 0; i0 < n; i0 += 2) {
    സ്ഥിരം int64_t ic = i0/scale; // hack for GGML_ROPE_TYPE_NORMAL, where we need ic = i0; for all other cases, ic = i0/2

    സ്ഥിരം ദശാംശം cos_theta = cache[i0 + 0];
    സ്ഥിരം ദശാംശം sin_theta = cache[i0 + 1];

    സ്ഥിരം T * സ്ഥിരം src = src_data + ic;
    T * dst             = dst_data + ic;

    സ്ഥിരം ദശാംശം x0 = type_conversion_table<T>::to_f32(src[0]);
    സ്ഥിരം ദശാംശം x1 = type_conversion_table<T>::to_f32(src[n_offset]);

    dst[0]        = type_conversion_table<T>::from_f32(x0*cos_theta - x1*sin_theta);
    dst[n_offset] = type_conversion_table<T>::from_f32(x0*sin_theta + x1*cos_theta);
  }
}

template<typename T> //float or ggml_fp16_t
സ്ഥിതം ശൂന്യം ggml_compute_forward_rope_flt(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst,
        സ്ഥിരം തർക്കം forward) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];
    സ്ഥിരം ggml_tensor * src2 = dst->src[2];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);

    ദശാംശം freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    പൂർണ്ണസംഖ്യ sections[4];

    //const int n_past     = ((int32_t *) dst->op_params)[0];
    സ്ഥിരം പൂർണ്ണസംഖ്യ n_dims     = ((int32_t *) dst->op_params)[1];
    സ്ഥിരം പൂർണ്ണസംഖ്യ mode       = ((int32_t *) dst->op_params)[2];
    //const int n_ctx      = ((int32_t *) dst->op_params)[3];
    സ്ഥിരം പൂർണ്ണസംഖ്യ n_ctx_orig = ((int32_t *) dst->op_params)[4];

    memcpy(&freq_base,   (int32_t *) dst->op_params +  5, വലുപ്പം(ദശാംശം));
    memcpy(&freq_scale,  (int32_t *) dst->op_params +  6, വലുപ്പം(ദശാംശം));
    memcpy(&ext_factor,  (int32_t *) dst->op_params +  7, വലുപ്പം(ദശാംശം));
    memcpy(&attn_factor, (int32_t *) dst->op_params +  8, വലുപ്പം(ദശാംശം));
    memcpy(&beta_fast,   (int32_t *) dst->op_params +  9, വലുപ്പം(ദശാംശം));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, വലുപ്പം(ദശാംശം));
    memcpy(&sections,    (int32_t *) dst->op_params + 11, വലുപ്പം(പൂർണ്ണസംഖ്യ)*4);

    സ്ഥിരം പൂർണ്ണസംഖ്യ n_offs = ((int32_t *) dst->op_params)[15];

    GGML_TENSOR_UNARY_OP_LOCALS

    //printf("ne0: %d, ne1: %d, ne2: %d, ne3: %d\n", ne0, ne1, ne2, ne3);
    //printf("n_past = %d, ne2 = %d\n", n_past, ne2);

    GGML_ASSERT(nb0 == nb00);
    GGML_ASSERT(nb0 == വലുപ്പം(T));

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ nr = ggml_nrows(dst);

    GGML_ASSERT(n_dims <= ne0);
    GGML_ASSERT(n_dims % 2 == 0);

    GGML_ASSERT(n_offs >= 0);
    GGML_ASSERT(n_offs % 2 == 0);
    GGML_ASSERT(n_offs + n_dims <= ne0);

    // rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    // row index used to determine which thread to use
    പൂർണ്ണസംഖ്യ ir = 0;

    സ്ഥിരം ദശാംശം theta_scale = powf(freq_base, -2.0f/n_dims);

    ദശാംശം corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    സ്ഥിരം തർക്കം is_imrope = mode == GGML_ROPE_TYPE_IMROPE; // qwen3vl apply interleaved mrope
    സ്ഥിരം തർക്കം mrope_used = mode & GGML_ROPE_TYPE_MROPE;  // ggml_rope_multi, note: also true for vision (24 & 8 == true) and for imrope
    സ്ഥിരം തർക്കം is_vision = mode == GGML_ROPE_TYPE_VISION;

    എങ്കിൽ (mrope_used) {
        GGML_ASSERT(sections[0] > 0 || sections[1] > 0 || sections[2] > 0);
    }

    എങ്കിൽ (is_vision) {
        GGML_ASSERT(n_dims == ne0/2);
        GGML_ASSERT(n_offs == 0);
    }

    സ്ഥിരം ദശാംശം * freq_factors = ശൂന്യ;
    എങ്കിൽ (src2 != ശൂന്യ) {
        GGML_ASSERT(src2->type == GGML_TYPE_F32);
        GGML_ASSERT(src2->ne[0] >= n_dims / 2);
        freq_factors = (സ്ഥിരം ദശാംശം *) src2->data;
    }

    // backward process uses inverse rotation by cos and sin.
    // cos and sin build a rotation matrix, where the inverse is the transpose.
    // this essentially just switches the sign of sin.
    സ്ഥിരം ദശാംശം sin_sign = forward ? 1.0f : -1.0f;

    സ്ഥിരം int32_t * pos = (സ്ഥിരം int32_t *) src1->data;

    int64_t last_i2 = -1;

    വേണ്ടി (int64_t i3 = 0; i3 < ne3; i3++) { // batch
        വേണ്ടി (int64_t i2 = 0; i2 < ne2; i2++) { // seq-len
            വേണ്ടി (int64_t i1 = 0; i1 < ne1; i1++) { // attn-heads
                എങ്കിൽ (ir++ < ir0) തുടരുക; // skip rows mapped to other threads
                എങ്കിൽ (ir   > ir1) നിർത്തുക;

                ദശാംശം * cache = (ദശാംശം *) params->wdata + (ne0 + CACHE_വരി_SIZE_F32)*ith;
                എങ്കിൽ (last_i2 != i2) {
                    എങ്കിൽ (!mrope_used) {
                        സ്ഥിരം int64_t p = pos[i2];
                        ggml_rope_cache_init(p, freq_scale, freq_factors, corr_dims, ne0, ext_factor, attn_factor, cache, sin_sign, theta_scale);
                    }
                    അല്ലെങ്കിൽ {
                        സ്ഥിരം int64_t p_t = pos[i2];
                        സ്ഥിരം int64_t p_h = pos[i2 + ne2];
                        സ്ഥിരം int64_t p_w = pos[i2 + ne2 * 2];
                        സ്ഥിരം int64_t p_e = pos[i2 + ne2 * 3];
                        ggml_mrope_cache_init(
                            p_t, p_h, p_w, p_e, sections, is_imrope, is_vision,
                            freq_scale, freq_factors, corr_dims, ne0, ext_factor, attn_factor, cache, sin_sign, theta_scale);
                    }

                    last_i2 = i2;
                }

                T * src = (T *)((അക്ഷരം *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
                T * dst_data  = (T *)((അക്ഷരം *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1);

                തിരഞ്ഞെടുപ്പ് (mode) {
                    അവസ്ഥ GGML_ROPE_TYPE_NORMAL:
                        rotate_pairs<T>(n_dims, 1, cache, src + n_offs, dst_data + n_offs, 1);
                        നിർത്തുക;
                    അവസ്ഥ GGML_ROPE_TYPE_NEOX:
                    അവസ്ഥ GGML_ROPE_TYPE_MROPE:
                    അവസ്ഥ GGML_ROPE_TYPE_IMROPE:
                        rotate_pairs<T>(n_dims, n_dims/2, cache, src + n_offs, dst_data + n_offs);
                        നിർത്തുക;
                    അവസ്ഥ GGML_ROPE_TYPE_VISION:
                        rotate_pairs<T>(ne0, n_dims, cache, src, dst_data);
                        നിർത്തുക;
                    സ്ഥിരസ്ഥിതി:
                        GGML_ABORT("rope type not supported");
                }

                എങ്കിൽ (!is_vision) {
                    // fill the remain channels with data from src tensor
                    വേണ്ടി (int64_t i0 = 0; i0 < ne0; i0 += 2) {
                        എങ്കിൽ (i0 == n_offs) {
                            i0 += n_dims - 2; // skip the rotated channels
                            തുടരുക;
                        }
                        സ്ഥിരം T * സ്ഥിരം src = (T *)((അക്ഷരം *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                        T * dst_data  = (T *)((അക്ഷരം *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

                        dst_data[0] = src[0];
                        dst_data[1] = src[1];
                    }
                }
            } //attn-heads
        }
    }
}

ശൂന്യം ggml_compute_forward_rope(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F16:
            {
                ggml_compute_forward_rope_flt<ggml_fp16_t>(params, dst, സത്യം);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_rope_flt<ദശാംശം>(params, dst, സത്യം);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_rope_back

ശൂന്യം ggml_compute_forward_rope_back(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F16:
            {
                ggml_compute_forward_rope_flt<ggml_fp16_t>(params, dst, അസത്യം);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_rope_flt<ദശാംശം>(params, dst, അസത്യം);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_conv_transpose_1d

സ്ഥിതം ശൂന്യം ggml_compute_forward_conv_transpose_1d_f16_f32(
        സ്ഥിരം ggml_compute_params * params,
              ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ദ്വിമാന_OP_LOCALS

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ nk = ne00*ne01*ne02;

    GGML_ASSERT(nb00 == വലുപ്പം(ggml_fp16_t));
    GGML_ASSERT(nb10 == വലുപ്പം(ദശാംശം));

    എങ്കിൽ (ith == 0) {
        memset(params->wdata, 0, params->wsize);

        // permute kernel data (src0) from (K x Cout x Cin) to (Cin x K x Cout)
        {
            ggml_fp16_t * സ്ഥിരം wdata = (ggml_fp16_t *) params->wdata + 0;

            വേണ്ടി (int64_t i02 = 0; i02 < ne02; i02++) {
                വേണ്ടി (int64_t i01 = 0; i01 < ne01; i01++) {
                    സ്ഥിരം ggml_fp16_t * സ്ഥിരം src = (ggml_fp16_t *)((അക്ഷരം *) src0->data + i02*nb02 + i01*nb01);
                    ggml_fp16_t * dst_data = wdata + i01*ne00*ne02;
                    വേണ്ടി (int64_t i00 = 0; i00 < ne00; i00++) {
                        dst_data[i00*ne02 + i02] = src[i00];
                    }
                }
            }
        }

        // permute source data (src1) from (L x Cin) to (Cin x L)
        {
            ggml_fp16_t * സ്ഥിരം wdata = (ggml_fp16_t *) params->wdata + nk;
            ggml_fp16_t * dst_data = wdata;

            വേണ്ടി (int64_t i11 = 0; i11 < ne11; i11++) {
                സ്ഥിരം ദശാംശം * സ്ഥിരം src = (ദശാംശം *)((അക്ഷരം *) src1->data + i11*nb11);
                വേണ്ടി (int64_t i10 = 0; i10 < ne10; i10++) {
                    dst_data[i10*ne11 + i11] = GGML_CPU_FP32_TO_FP16(src[i10]);
                }
            }
        }

        // need to zero dst since we are accumulating into it
        memset(dst->data, 0, ggml_nbytes(dst));
    }
    ggml_barrier(params->threadpool);

    സ്ഥിരം int32_t s0 = ((സ്ഥിരം int32_t*)(dst->op_params))[0];

    // total rows in dst
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr = ne1;

    // rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    ggml_fp16_t * സ്ഥിരം wdata     = (ggml_fp16_t *) params->wdata + 0;
    ggml_fp16_t * സ്ഥിരം wdata_src = wdata + nk;

    വേണ്ടി (പൂർണ്ണസംഖ്യ i1 = ir0; i1 < ir1; i1++) {
        ദശാംശം * dst_data = (ദശാംശം *)((അക്ഷരം *) dst->data + i1*nb1);
        ggml_fp16_t * wdata_kernel = wdata + i1*ne02*ne00;
        വേണ്ടി (പൂർണ്ണസംഖ്യ i10 = 0; i10 < ne10; i10++) {
            സ്ഥിരം പൂർണ്ണസംഖ്യ i1n = i10*ne11;
            വേണ്ടി (പൂർണ്ണസംഖ്യ i00 = 0; i00 < ne00; i00++) {
                ദശാംശം v = 0;
                ggml_vec_dot_f16(ne02, &v, 0,
                        (ggml_fp16_t *)    wdata_src + i1n, 0,
                        (ggml_fp16_t *) wdata_kernel + i00*ne02, 0, 1);
                dst_data[i10*s0 + i00] += v;
            }
        }
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_conv_transpose_1d_f32(
        സ്ഥിരം ggml_compute_params * params,
              ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ദ്വിമാന_OP_LOCALS

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ nk = ne00*ne01*ne02;

    GGML_ASSERT(nb00 == വലുപ്പം(ദശാംശം));
    GGML_ASSERT(nb10 == വലുപ്പം(ദശാംശം));

    എങ്കിൽ (ith == 0) {
        memset(params->wdata, 0, params->wsize);

        // prepare kernel data (src0) from (K x Cout x Cin) to (Cin x K x Cout)
        {
            ദശാംശം * സ്ഥിരം wdata = (ദശാംശം *) params->wdata + 0;

            വേണ്ടി (int64_t i02 = 0; i02 < ne02; i02++) {
                വേണ്ടി (int64_t i01 = 0; i01 < ne01; i01++) {
                    സ്ഥിരം ദശാംശം * സ്ഥിരം src = (ദശാംശം *)((അക്ഷരം *) src0->data + i02*nb02 + i01*nb01);
                    ദശാംശം * dst_data = wdata + i01*ne00*ne02;
                    വേണ്ടി (int64_t i00 = 0; i00 < ne00; i00++) {
                        dst_data[i00*ne02 + i02] = src[i00];
                    }
                }
            }
        }

        // prepare source data (src1)
        {
            ദശാംശം * സ്ഥിരം wdata = (ദശാംശം *) params->wdata + nk;
            ദശാംശം * dst_data = wdata;

            വേണ്ടി (int64_t i11 = 0; i11 < ne11; i11++) {
                സ്ഥിരം ദശാംശം * സ്ഥിരം src = (ദശാംശം *)((അക്ഷരം *) src1->data + i11*nb11);
                വേണ്ടി (int64_t i10 = 0; i10 < ne10; i10++) {
                    dst_data[i10*ne11 + i11] = src[i10];
                }
            }
        }

        // need to zero dst since we are accumulating into it
        memset(dst->data, 0, ggml_nbytes(dst));
    }
    ggml_barrier(params->threadpool);

    സ്ഥിരം int32_t s0 = ((സ്ഥിരം int32_t*)(dst->op_params))[0];

    // total rows in dst
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr = ne1;

    // rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    ദശാംശം * സ്ഥിരം wdata     = (ദശാംശം *) params->wdata + 0;
    ദശാംശം * സ്ഥിരം wdata_src = wdata + nk;

    വേണ്ടി (പൂർണ്ണസംഖ്യ i1 = ir0; i1 < ir1; i1++) {
        ദശാംശം * dst_data = (ദശാംശം *)((അക്ഷരം *) dst->data + i1*nb1);
        ദശാംശം * wdata_kernel = wdata + i1*ne02*ne00;
        വേണ്ടി (പൂർണ്ണസംഖ്യ i10 = 0; i10 < ne10; i10++) {
            സ്ഥിരം പൂർണ്ണസംഖ്യ i1n = i10*ne11;
            വേണ്ടി (പൂർണ്ണസംഖ്യ i00 = 0; i00 < ne00; i00++) {
                ദശാംശം v = 0;
                ggml_vec_dot_f32(ne02, &v, 0,
                        wdata_src + i1n, 0,
                        wdata_kernel + i00*ne02, 0, 1);
                dst_data[i10*s0 + i00] += v;
            }
        }
    }
}

ശൂന്യം ggml_compute_forward_conv_transpose_1d(
        സ്ഥിരം ggml_compute_params * params,
              ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F16:
            {
                ggml_compute_forward_conv_transpose_1d_f16_f32(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_conv_transpose_1d_f32(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_im2col_f32
// src0: kernel [OC, IC, KH, KW]
// src1: image [N, IC, IH, IW]
// dst:  result [N, OH, OW, IC*KH*KW]
സ്ഥിതം ശൂന്യം ggml_compute_forward_im2col_f32(
        സ്ഥിരം ggml_compute_params * params,
              ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ദ്വിമാന_OP_LOCALS;

    സ്ഥിരം int32_t s0 = ((സ്ഥിരം int32_t *)(dst->op_params))[0];
    സ്ഥിരം int32_t s1 = ((സ്ഥിരം int32_t *)(dst->op_params))[1];
    സ്ഥിരം int32_t p0 = ((സ്ഥിരം int32_t *)(dst->op_params))[2];
    സ്ഥിരം int32_t p1 = ((സ്ഥിരം int32_t *)(dst->op_params))[3];
    സ്ഥിരം int32_t d0 = ((സ്ഥിരം int32_t *)(dst->op_params))[4];
    സ്ഥിരം int32_t d1 = ((സ്ഥിരം int32_t *)(dst->op_params))[5];
    സ്ഥിരം തർക്കം is_2D = ((സ്ഥിരം int32_t *)(dst->op_params))[6] == 1;

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം int64_t N  = is_2D ? ne13 : ne12;
    സ്ഥിരം int64_t IC = is_2D ? ne12 : ne11;
    സ്ഥിരം int64_t IH = is_2D ? ne11 : 1;
    സ്ഥിരം int64_t IW = ne10;

    സ്ഥിരം int64_t KH = is_2D ? ne01 : 1;
    സ്ഥിരം int64_t KW = ne00;

    സ്ഥിരം int64_t OH = is_2D ? ne2 : 1;
    സ്ഥിരം int64_t OW = ne1;

    പൂർണ്ണസംഖ്യ ofs0 = is_2D ? nb13 : nb12;
    പൂർണ്ണസംഖ്യ ofs1 = is_2D ? nb12 : nb11;

    GGML_ASSERT(nb10 == വലുപ്പം(ദശാംശം));

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    {
        ദശാംശം * സ്ഥിരം wdata = (ദശാംശം *) dst->data;

        വേണ്ടി (int64_t in = 0; in < N; in++) {
            വേണ്ടി (int64_t ioh = 0; ioh < OH; ioh++) { // 1
                വേണ്ടി (int64_t iow = 0; iow < OW; iow++) {
                    വേണ്ടി (int64_t iic = ith; iic < IC; iic += nth) {

                        // micro kernel
                        ദശാംശം * dst_data = wdata + (in*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
                        സ്ഥിരം ദശാംശം * സ്ഥിരം src_data = (ദശാംശം *)((അക്ഷരം *) src1->data + in*ofs0 + iic*ofs1); // [IH, IW]

                        വേണ്ടി (int64_t ikh = 0; ikh < KH; ikh++) {  // 1
                            വേണ്ടി (int64_t ikw = 0; ikw < KW; ikw++) {
                                സ്ഥിരം int64_t iiw = iow*s0 + ikw*d0 - p0;
                                സ്ഥിരം int64_t iih = ioh*s1 + ikh*d1 - p1;

                                എങ്കിൽ (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                    dst_data[iic*(KH*KW) + ikh*KW + ikw] = 0;
                                } അല്ലെങ്കിൽ {
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
സ്ഥിതം ശൂന്യം ggml_compute_forward_im2col_f16(
        സ്ഥിരം ggml_compute_params * params,
              ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16);

    GGML_TENSOR_ദ്വിമാന_OP_LOCALS;

    സ്ഥിരം int32_t s0 = ((സ്ഥിരം int32_t *)(dst->op_params))[0];
    സ്ഥിരം int32_t s1 = ((സ്ഥിരം int32_t *)(dst->op_params))[1];
    സ്ഥിരം int32_t p0 = ((സ്ഥിരം int32_t *)(dst->op_params))[2];
    സ്ഥിരം int32_t p1 = ((സ്ഥിരം int32_t *)(dst->op_params))[3];
    സ്ഥിരം int32_t d0 = ((സ്ഥിരം int32_t *)(dst->op_params))[4];
    സ്ഥിരം int32_t d1 = ((സ്ഥിരം int32_t *)(dst->op_params))[5];
    സ്ഥിരം തർക്കം is_2D = ((സ്ഥിരം int32_t *)(dst->op_params))[6] == 1;

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം int64_t N  = is_2D ? ne13 : ne12;
    സ്ഥിരം int64_t IC = is_2D ? ne12 : ne11;
    സ്ഥിരം int64_t IH = is_2D ? ne11 : 1;
    സ്ഥിരം int64_t IW = ne10;

    സ്ഥിരം int64_t KH = is_2D ? ne01 : 1;
    സ്ഥിരം int64_t KW = ne00;

    സ്ഥിരം int64_t OH = is_2D ? ne2 : 1;
    സ്ഥിരം int64_t OW = ne1;

    പൂർണ്ണസംഖ്യ ofs0 = is_2D ? nb13 : nb12;
    പൂർണ്ണസംഖ്യ ofs1 = is_2D ? nb12 : nb11;

    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    {
        ggml_fp16_t * സ്ഥിരം wdata = (ggml_fp16_t *) dst->data;

        വേണ്ടി (int64_t in = 0; in < N; in++) {
            വേണ്ടി (int64_t ioh = 0; ioh < OH; ioh++) { // 1
                വേണ്ടി (int64_t iow = 0; iow < OW; iow++) {
                    വേണ്ടി (int64_t iic = ith; iic < IC; iic += nth) {

                        // micro kernel
                        ggml_fp16_t * dst_data = wdata + (in*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
                        സ്ഥിരം ദശാംശം * സ്ഥിരം src_data_f32 = src1->type == GGML_TYPE_F32
                            ? (സ്ഥിരം ദശാംശം *)((സ്ഥിരം അക്ഷരം *) src1->data + in*ofs0 + iic*ofs1)
                            : nullptr; // [IH, IW]
                        സ്ഥിരം ggml_fp16_t * സ്ഥിരം src_data_f16 = src1->type == GGML_TYPE_F16
                            ? (സ്ഥിരം ggml_fp16_t *)((സ്ഥിരം അക്ഷരം *) src1->data + in*ofs0 + iic*ofs1)
                            : nullptr; // [IH, IW]

                        വേണ്ടി (int64_t ikh = 0; ikh < KH; ikh++) {  // 1
                            വേണ്ടി (int64_t ikw = 0; ikw < KW; ikw++) {
                                സ്ഥിരം int64_t iiw = iow*s0 + ikw*d0 - p0;
                                സ്ഥിരം int64_t iih = ioh*s1 + ikh*d1 - p1;

                                എങ്കിൽ (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                    dst_data[iic*(KH*KW) + ikh*KW + ikw] = 0;
                                } അല്ലെങ്കിൽ {
                                    എങ്കിൽ (src_data_f32 != nullptr) {
                                        dst_data[iic*(KH*KW) + ikh*KW + ikw] = GGML_CPU_FP32_TO_FP16(src_data_f32[iih*IW + iiw]);
                                    } അല്ലെങ്കിൽ {
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

ശൂന്യം ggml_compute_forward_im2col(
        സ്ഥിരം ggml_compute_params * params,
              ggml_tensor * dst) {
    തിരഞ്ഞെടുപ്പ് (dst->type) {
        അവസ്ഥ GGML_TYPE_F16:
            {
                ggml_compute_forward_im2col_f16(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_im2col_f32(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_im2col_back_f32

ശൂന്യം ggml_compute_forward_im2col_back_f32(
        സ്ഥിരം ggml_compute_params * params,
              ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0]; // gradients of forward pass output
    സ്ഥിരം ggml_tensor * src1 = dst->src[1]; // convolution kernel

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ദ്വിമാന_OP_LOCALS;

    സ്ഥിരം int32_t s0 = ((സ്ഥിരം int32_t *)(dst->op_params))[0];
    സ്ഥിരം int32_t s1 = ((സ്ഥിരം int32_t *)(dst->op_params))[1];
    സ്ഥിരം int32_t p0 = ((സ്ഥിരം int32_t *)(dst->op_params))[2];
    സ്ഥിരം int32_t p1 = ((സ്ഥിരം int32_t *)(dst->op_params))[3];
    സ്ഥിരം int32_t d0 = ((സ്ഥിരം int32_t *)(dst->op_params))[4];
    സ്ഥിരം int32_t d1 = ((സ്ഥിരം int32_t *)(dst->op_params))[5];
    സ്ഥിരം തർക്കം is_2D = ((സ്ഥിരം int32_t *)(dst->op_params))[6] == 1;

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം int64_t N  = is_2D ? ne3 : ne2;
    സ്ഥിരം int64_t IC = is_2D ? ne2 : ne1;
    സ്ഥിരം int64_t IH = is_2D ? ne1 : 1;
    സ്ഥിരം int64_t IW = ne0;

    സ്ഥിരം int64_t KH = is_2D ? ne11 : 1;
    സ്ഥിരം int64_t KW = ne10;

    സ്ഥിരം int64_t OH = is_2D ? ne02 : 1;
    സ്ഥിരം int64_t OW = ne01;

    പൂർണ്ണസംഖ്യ ofs0 = is_2D ? nb3 : nb2;
    പൂർണ്ണസംഖ്യ ofs1 = is_2D ? nb2 : nb1;

    GGML_ASSERT(nb0  == വലുപ്പം(ദശാംശം));

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    {
        ദശാംശം * സ്ഥിരം wdata = (ദശാംശം *) dst->data;

        വേണ്ടി (int64_t in = 0; in < N; in++) {
            വേണ്ടി (int64_t iic = ith; iic < IC; iic += nth) {
                വേണ്ടി (int64_t iih = 0; iih < IH; iih++) {
                    വേണ്ടി (int64_t iiw = 0; iiw < IW; iiw++) {

                        // micro kernel
                        ദശാംശം grad = 0.0f;
                        വേണ്ടി (int64_t ikh = 0; ikh < KH; ikh++) {
                            വേണ്ടി (int64_t ikw = 0; ikw < KW; ikw++) {
                                // For s0 > 1 some values were skipped over in the forward pass.
                                // These values have tmpw % s0 != 0 and need to be skipped in the backwards pass as well.
                                സ്ഥിരം int64_t tmpw = (iiw + p0 - ikw*d0);
                                എങ്കിൽ (tmpw % s0 != 0) {
                                    തുടരുക;
                                }
                                സ്ഥിരം int64_t iow = tmpw / s0;

                                // Equivalent logic as above except for s1.
                                int64_t ioh;
                                എങ്കിൽ (is_2D) {
                                    സ്ഥിരം int64_t tmph = iih + p1 - ikh*d1;

                                    എങ്കിൽ (tmph % s1 != 0) {
                                        തുടരുക;
                                    }

                                    ioh = tmph / s1;
                                } അല്ലെങ്കിൽ {
                                    ioh = 0;
                                }

                                എങ്കിൽ (iow < 0 || iow >= OW || ioh < 0 || ioh >= OH) {
                                    തുടരുക;
                                }

                                സ്ഥിരം ദശാംശം * സ്ഥിരം grad_in = (സ്ഥിരം ദശാംശം *) src0->data
                                    + (in*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
                                grad += grad_in[iic*(KH*KW) + ikh*KW + ikw];
                            }
                        }
                        ദശാംശം * dst_data = (ദശാംശം *)((അക്ഷരം *) wdata + (in*ofs0 + iic*ofs1)); // [IH, IW]
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
സ്ഥിതം ശൂന്യം ggml_compute_forward_im2col_3d_f16(
        സ്ഥിരം ggml_compute_params * params,
              ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16);

    GGML_TENSOR_ദ്വിമാന_OP_LOCALS;

    സ്ഥിരം int32_t s0 = ((സ്ഥിരം int32_t *)(dst->op_params))[0];
    സ്ഥിരം int32_t s1 = ((സ്ഥിരം int32_t *)(dst->op_params))[1];
    സ്ഥിരം int32_t s2 = ((സ്ഥിരം int32_t *)(dst->op_params))[2];
    സ്ഥിരം int32_t p0 = ((സ്ഥിരം int32_t *)(dst->op_params))[3];
    സ്ഥിരം int32_t p1 = ((സ്ഥിരം int32_t *)(dst->op_params))[4];
    സ്ഥിരം int32_t p2 = ((സ്ഥിരം int32_t *)(dst->op_params))[5];
    സ്ഥിരം int32_t d0 = ((സ്ഥിരം int32_t *)(dst->op_params))[6];
    സ്ഥിരം int32_t d1 = ((സ്ഥിരം int32_t *)(dst->op_params))[7];
    സ്ഥിരം int32_t d2 = ((സ്ഥിരം int32_t *)(dst->op_params))[8];
    സ്ഥിരം int32_t IC = ((സ്ഥിരം int32_t *)(dst->op_params))[9];


    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം int64_t N  = ne13 / IC;
    സ്ഥിരം int64_t ID = ne12;
    സ്ഥിരം int64_t IH = ne11;
    സ്ഥിരം int64_t IW = ne10;

    സ്ഥിരം int64_t OC = ne03 / IC;
    GGML_UNUSED(OC);
    സ്ഥിരം int64_t KD = ne02;
    സ്ഥിരം int64_t KH = ne01;
    സ്ഥിരം int64_t KW = ne00;

    സ്ഥിരം int64_t OD = ne3 / N;
    സ്ഥിരം int64_t OH = ne2;
    സ്ഥിരം int64_t OW = ne1;
    സ്ഥിരം int64_t OH_OW = OH*OW;
    സ്ഥിരം int64_t KD_KH_KW = KD*KH*KW;
    സ്ഥിരം int64_t KH_KW = KH*KW;
    സ്ഥിരം int64_t IC_KD_KH_KW = IC*KD*KH*KW;

    GGML_ASSERT(nb10 == വലുപ്പം(ദശാംശം));

    // im2col: [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
    {
        ggml_fp16_t * സ്ഥിരം wdata = (ggml_fp16_t *) dst->data;

        വേണ്ടി (int64_t in = 0; in < N; in++) {
            വേണ്ടി (int64_t iod = 0; iod < OD; iod++) {
                വേണ്ടി (int64_t ioh = 0; ioh < OH; ioh++) {
                    വേണ്ടി (int64_t iow = 0; iow < OW; iow++) {
                        വേണ്ടി (int64_t iic = ith; iic < IC; iic += nth) {

                            // micro kernel
                            ggml_fp16_t * dst_data = wdata + (in*OD*OH_OW + iod*OH_OW + ioh*OW + iow)*IC_KD_KH_KW; // [IC, KD, KH, KW]
                            സ്ഥിരം ദശാംശം * സ്ഥിരം src_data = (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *)src1->data + (in*IC + iic)*nb13); // [ID, IH, IW]

                            വേണ്ടി (int64_t ikd = 0; ikd < KD; ikd++) {
                                വേണ്ടി (int64_t ikh = 0; ikh < KH; ikh++) {
                                    വേണ്ടി (int64_t ikw = 0; ikw < KW; ikw++) {
                                        സ്ഥിരം int64_t iiw = iow*s0 + ikw*d0 - p0;
                                        സ്ഥിരം int64_t iih = ioh*s1 + ikh*d1 - p1;
                                        സ്ഥിരം int64_t iid = iod*s2 + ikd*d2 - p2;

                                        എങ്കിൽ (iid < 0 || iid >= ID || iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                            dst_data[iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw] = 0;
                                        } അല്ലെങ്കിൽ {
                                            സ്ഥിരം ദശാംശം * സ്ഥിരം s = (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *)src_data + iid*nb12 + iih*nb11 + iiw*nb10); // [ID, IH, IW]
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
സ്ഥിതം ശൂന്യം ggml_compute_forward_im2col_3d_f32(
        സ്ഥിരം ggml_compute_params * params,
              ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ദ്വിമാന_OP_LOCALS;

    സ്ഥിരം int32_t s0 = ((സ്ഥിരം int32_t *)(dst->op_params))[0];
    സ്ഥിരം int32_t s1 = ((സ്ഥിരം int32_t *)(dst->op_params))[1];
    സ്ഥിരം int32_t s2 = ((സ്ഥിരം int32_t *)(dst->op_params))[2];
    സ്ഥിരം int32_t p0 = ((സ്ഥിരം int32_t *)(dst->op_params))[3];
    സ്ഥിരം int32_t p1 = ((സ്ഥിരം int32_t *)(dst->op_params))[4];
    സ്ഥിരം int32_t p2 = ((സ്ഥിരം int32_t *)(dst->op_params))[5];
    സ്ഥിരം int32_t d0 = ((സ്ഥിരം int32_t *)(dst->op_params))[6];
    സ്ഥിരം int32_t d1 = ((സ്ഥിരം int32_t *)(dst->op_params))[7];
    സ്ഥിരം int32_t d2 = ((സ്ഥിരം int32_t *)(dst->op_params))[8];
    സ്ഥിരം int32_t IC = ((സ്ഥിരം int32_t *)(dst->op_params))[9];


    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം int64_t N  = ne13 / IC;
    സ്ഥിരം int64_t ID = ne12;
    സ്ഥിരം int64_t IH = ne11;
    സ്ഥിരം int64_t IW = ne10;

    സ്ഥിരം int64_t OC = ne03 / IC;
    GGML_UNUSED(OC);
    സ്ഥിരം int64_t KD = ne02;
    സ്ഥിരം int64_t KH = ne01;
    സ്ഥിരം int64_t KW = ne00;

    സ്ഥിരം int64_t OD = ne3 / N;
    സ്ഥിരം int64_t OH = ne2;
    സ്ഥിരം int64_t OW = ne1;

    സ്ഥിരം int64_t OH_OW = OH*OW;
    സ്ഥിരം int64_t KD_KH_KW = KD*KH*KW;
    സ്ഥിരം int64_t KH_KW = KH*KW;
    സ്ഥിരം int64_t IC_KD_KH_KW = IC*KD*KH*KW;

    GGML_ASSERT(nb10 == വലുപ്പം(ദശാംശം));

    // im2col: [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
    {
        ദശാംശം * സ്ഥിരം wdata = (ദശാംശം *) dst->data;

        വേണ്ടി (int64_t in = 0; in < N; in++) {
            വേണ്ടി (int64_t iod = 0; iod < OD; iod++) {
                വേണ്ടി (int64_t ioh = 0; ioh < OH; ioh++) {
                    വേണ്ടി (int64_t iow = 0; iow < OW; iow++) {
                        വേണ്ടി (int64_t iic = ith; iic < IC; iic += nth) {

                            // micro kernel
                            ദശാംശം * dst_data = wdata + (in*OD*OH_OW + iod*OH_OW + ioh*OW + iow)*IC_KD_KH_KW; // [IC, KD, KH, KW]
                            സ്ഥിരം ദശാംശം * സ്ഥിരം src_data = (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *)src1->data + (in*IC + iic)*nb13); // [ID, IH, IW]

                            വേണ്ടി (int64_t ikd = 0; ikd < KD; ikd++) {
                                വേണ്ടി (int64_t ikh = 0; ikh < KH; ikh++) {
                                    വേണ്ടി (int64_t ikw = 0; ikw < KW; ikw++) {
                                        സ്ഥിരം int64_t iiw = iow*s0 + ikw*d0 - p0;
                                        സ്ഥിരം int64_t iih = ioh*s1 + ikh*d1 - p1;
                                        സ്ഥിരം int64_t iid = iod*s2 + ikd*d2 - p2;

                                        എങ്കിൽ (iid < 0 || iid >= ID || iih < 0 || iih >= IH || iiw < 0 || iiw >= IW || iid < 0 || iid >= ID) {
                                            dst_data[iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw] = 0;
                                        } അല്ലെങ്കിൽ {
                                            സ്ഥിരം ദശാംശം * സ്ഥിരം s = (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *)src_data + iid*nb12 + iih*nb11 + iiw*nb10); // [ID, IH, IW]
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


ശൂന്യം ggml_compute_forward_im2col_3d(
        സ്ഥിരം ggml_compute_params * params,
              ggml_tensor * dst) {
    തിരഞ്ഞെടുപ്പ് (dst->type) {
        അവസ്ഥ GGML_TYPE_F16:
            {
                ggml_compute_forward_im2col_3d_f16(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_im2col_3d_f32(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

സ്ഥിതം ശൂന്യം ggml_call_mul_mat(ggml_type type, സ്ഥിരം ggml_compute_params * params, int64_t m, int64_t n, int64_t k,
                              ശൂന്യം * a, ശൂന്യം * b, ദശാംശം * c) {
    സ്ഥിരം ggml_type_traits * traits = ggml_get_type_traits(type);
    ഘടന ggml_tensor src1 = {};
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

    ഘടന ggml_tensor src0 = {};
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

    ഘടന ggml_tensor dst = {};
    dst.ne[0] = n;
    dst.ne[1] = m;
    dst.ne[2] = 1;
    dst.ne[3] = 1;
    dst.nb[0] = വലുപ്പം(ദശാംശം);
    dst.nb[1] = n * വലുപ്പം(ദശാംശം);
    dst.nb[2] = dst.nb[1];
    dst.nb[3] = dst.nb[2];
    dst.data  = c;
    dst.src[0] = &src0;
    dst.src[1] = &src1;

    ggml_compute_forward_mul_mat(params, &dst);
}

സ്ഥിതം വരിയിൽ int64_t ggml_wrap_around(int64_t coord, int64_t size) {
    തിരികെ (coord  + size) % size; // adding size avoids negative number weirdness
}

// ggml_compute_forward_col2im_1d
//
// Scatter-add columns [K*OC, T_in] -> signal [T_out, OC]
// where T_out = (T_in - 1)*s + K - 2*p.  Gather approach: each output reads ceil(K/s) inputs.
// Parallelized over the time axis so the split stays balanced whatever OC is.
// Supports F32, F16, BF16 input/output (same type), F32 accumulator.

template <typename elem_t>
സ്ഥിതം ശൂന്യം ggml_compute_forward_col2im_1d_impl(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src = dst->src[0];  // [K*OC, T_in]

    GGML_ASSERT(ggml_is_contiguous(src));
    GGML_ASSERT(ggml_is_contiguous(dst));

    സ്ഥിരം int32_t s0 = ((സ്ഥിരം int32_t *)(dst->op_params))[0];
    സ്ഥിരം int32_t OC = ((സ്ഥിരം int32_t *)(dst->op_params))[1];
    സ്ഥിരം int32_t p0 = ((സ്ഥിരം int32_t *)(dst->op_params))[2];

    സ്ഥിരം int64_t K_OC = src->ne[0];
    സ്ഥിരം int64_t T_in = src->ne[1];
    സ്ഥിരം int64_t K    = K_OC / OC;
    സ്ഥിരം int64_t T_out = dst->ne[0];

    സ്ഥിരം elem_t * col_data = (സ്ഥിരം elem_t *) src->data;
    elem_t       * dst_data = (elem_t *) dst->data;

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    // Parallelize over the time axis: the split stays balanced whatever OC is,
    // down to OC = 1 for mono audio, and threads read disjoint column bands
    സ്ഥിരം int64_t dr = (T_out + nth - 1) / nth;
    സ്ഥിരം int64_t it0 = dr * ith;
    സ്ഥിരം int64_t it1 = it0 + dr < T_out ? it0 + dr : T_out;

    വേണ്ടി (int64_t oc = 0; oc < OC; oc++) {
        വേണ്ടി (int64_t t_out = it0; t_out < it1; t_out++) {
            സ്ഥിരം int64_t t_abs = t_out + p0;  // absolute position in uncropped signal
            // Gather: find all (t_in, k) where t_in * s + k == t_abs, 0 <= k < K
            int64_t t_in_min = (t_abs - K + 1 + s0 - 1) / s0;  // ceil((t_abs-K+1)/s)
            എങ്കിൽ (t_in_min < 0) t_in_min = 0;
            int64_t t_in_max = t_abs / s0;
            എങ്കിൽ (t_in_max >= T_in) t_in_max = T_in - 1;

            ദശാംശം sum = 0.0f;
            വേണ്ടി (int64_t t_in = t_in_min; t_in <= t_in_max; t_in++) {
                int64_t k = t_abs - t_in * s0;
                എങ്കിൽ (k >= 0 && k < K) {
                    // col layout: [K*OC, T_in], element (oc*K+k, t_in)
                    sum += type_conversion_table<elem_t>::to_f32(col_data[(oc * K + k) + t_in * K_OC]);
                }
            }
            // dst layout: [T_out, OC], element (t_out, oc)
            dst_data[t_out + oc * T_out] = type_conversion_table<elem_t>::from_f32(sum);
        }
    }
}

ശൂന്യം ggml_compute_forward_col2im_1d(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {
    തിരഞ്ഞെടുപ്പ് (dst->src[0]->type) {
        അവസ്ഥ GGML_TYPE_F32:  ggml_compute_forward_col2im_1d_impl<ദശാംശം>      (params, dst); നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F16:  ggml_compute_forward_col2im_1d_impl<ggml_fp16_t>(params, dst); നിർത്തുക;
        അവസ്ഥ GGML_TYPE_BF16: ggml_compute_forward_col2im_1d_impl<ggml_bf16_t>(params, dst); നിർത്തുക;
        സ്ഥിരസ്ഥിതി: GGML_ABORT("col2im_1d: unsupported type %d", dst->src[0]->type);
    }
}

// ggml_compute_forward_conv_2d


സ്ഥിതം ശൂന്യം ggml_compute_forward_conv_2d_impl(സ്ഥിരം ggml_compute_params * params,
                                              സ്ഥിരം ggml_tensor *         kernel,  // [KW, KH, IC, OC]
                                              സ്ഥിരം ggml_tensor *         src,     // [W, H, C, N]
                                              ggml_tensor *               dst,     // [OW, OH, OC, N]
                                              ggml_type                   kernel_type) {

    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(kernel_type == GGML_TYPE_F16 || kernel_type == GGML_TYPE_F32);
    GGML_ASSERT(kernel->type == kernel_type);

    സ്ഥിരം ggml_type_traits * traits = ggml_get_type_traits(kernel_type);

    സ്ഥിരം int32_t stride_x   = dst->op_params[0];
    സ്ഥിരം int32_t stride_y   = dst->op_params[1];
    സ്ഥിരം int32_t pad_x      = dst->op_params[2];
    സ്ഥിരം int32_t pad_y      = dst->op_params[3];
    സ്ഥിരം int32_t dilation_x = dst->op_params[4];
    സ്ഥിരം int32_t dilation_y = dst->op_params[5];

    സ്ഥിരം int64_t c_in  = src->ne[2];
    സ്ഥിരം int64_t c_out = kernel->ne[3];
    GGML_ASSERT(c_in == kernel->ne[2]);

    സ്ഥിരം int64_t src_w = src->ne[0];
    സ്ഥിരം int64_t src_h = src->ne[1];
    സ്ഥിരം int64_t knl_w = kernel->ne[0];
    സ്ഥിരം int64_t knl_h = kernel->ne[1];
    സ്ഥിരം int64_t dst_w = dst->ne[0];
    സ്ഥിരം int64_t dst_h = dst->ne[1];

    സ്ഥിരം ദശാംശം * src_data = (ദശാംശം *) src->data;
    ശൂന്യം  * knl_data       = kernel->data;
    ദശാംശം * dst_data       = (ദശാംശം *) dst->data;

    സ്ഥിരം int64_t knl_n           = knl_w * knl_h * c_in;
    സ്ഥിരം int64_t patch_total     = dst->ne[3] * dst_w * dst_h;

    സ്ഥിരം int64_t space_per_patch   = knl_n * traits->type_size + c_out * വലുപ്പം(ദശാംശം);
    സ്ഥിരം int64_t batch_size        = params->wsize / space_per_patch;
    സ്ഥിരം int64_t patches_per_batch = batch_size > 8 ? (batch_size / 8) * 8 : batch_size;
    സ്ഥിരം int64_t batch_n           = (patch_total + patches_per_batch - 1) / patches_per_batch;

    GGML_ASSERT(patches_per_batch > 0 && batch_size >= 1);

    ശൂന്യം * tmp = params->wdata;

    വേണ്ടി (int64_t batch_i = 0; batch_i < batch_n; ++batch_i) {

        സ്ഥിരം int64_t patch_start_batch = batch_i * patches_per_batch;
        സ്ഥിരം int64_t patch_end_batch   = std::min(patch_start_batch + patches_per_batch,
                                              patch_total);
        സ്ഥിരം int64_t patch_n           = patch_end_batch - patch_start_batch;

        സ്ഥിരം int64_t patch_per_thread  = (patch_n + params->nth - 1) / params->nth;
        സ്ഥിരം int64_t patch_start       = patch_start_batch + params->ith * patch_per_thread;
        സ്ഥിരം int64_t patch_end         = std::min(patch_start + patch_per_thread, patch_end_batch);

        //im2col for a patch
        വേണ്ടി (int64_t p = patch_start; p < patch_end; ++p) {
            സ്ഥിരം int64_t  batch_n     =  p / (dst_w * dst_h);
            സ്ഥിരം int64_t  src_x       = (p / dst_w) % dst_h;
            സ്ഥിരം int64_t  src_y       =  p % dst_w;

            സ്ഥിരം ദശാംശം * src_base = (സ്ഥിരം ദശാംശം *)((സ്ഥിരം അക്ഷരം *)src_data + batch_n * src->nb[3]);
            അക്ഷരം *        dst_row  = (അക്ഷരം *) tmp + (p % patches_per_batch) * knl_n * traits->type_size;

            വേണ്ടി (int64_t ic = 0; ic < c_in; ++ic) {
                വേണ്ടി (int64_t ky = 0; ky < knl_h; ++ky) {
                    വേണ്ടി (int64_t kx = 0; kx < knl_w; ++kx) {
                        സ്ഥിരം int64_t sy = src_x * stride_y + ky * dilation_y - pad_y;
                        സ്ഥിരം int64_t sx = src_y * stride_x + kx * dilation_x - pad_x;

                        int64_t dst_idx = ic * (knl_h * knl_w) + ky * knl_w + kx;

                        ദശാംശം src_val;
                        എങ്കിൽ (sy < 0 || sy >= src_h || sx < 0 || sx >= src_w) {
                            src_val = 0.0f;
                        } അല്ലെങ്കിൽ {
                            സ്ഥിരം ദശാംശം * src_ptr = (സ്ഥിരം ദശാംശം *)((സ്ഥിരം അക്ഷരം *)src_base + sx * src->nb[0] + sy * src->nb[1] + ic * src->nb[2]);
                            src_val               = *src_ptr;
                        }

                        അക്ഷരം * element_ptr = dst_row + dst_idx * traits->type_size;
                        എങ്കിൽ (kernel_type == GGML_TYPE_F32) {
                            *(ദശാംശം *) element_ptr = src_val;
                        } അല്ലെങ്കിൽ എങ്കിൽ (kernel_type == GGML_TYPE_F16) {
                            *(ggml_fp16_t *) element_ptr = GGML_CPU_FP32_TO_FP16(src_val);
                        }
                    }
                }
            }
        }   // patches handled by this thread

        ggml_barrier(params->threadpool);

        ദശാംശം * gemm_output = (ദശാംശം *) ((അക്ഷരം *) tmp + patches_per_batch * knl_n * traits->type_size);

        GGML_ASSERT(gemm_output + patch_n * c_out <= (ദശാംശം*)tmp + params->wsize);

        // GEMM: patches[patch_n, knl_n] × kernel[knl_n, c_out] = output[patch_n, c_out]
        ggml_call_mul_mat(kernel_type, params, patch_n, c_out, knl_n, tmp, knl_data, gemm_output);

        ggml_barrier(params->threadpool);


        //permute back [OC, N, OH, OW] to [N, OC, OH, OW]
        സ്ഥിരം int64_t permute_per_thread = (patch_n + params->nth - 1) / params->nth;
        സ്ഥിരം int64_t permute_start = params->ith * permute_per_thread;
        സ്ഥിരം int64_t permute_end = std::min(permute_start + permute_per_thread, patch_n);

        വേണ്ടി (int64_t i = permute_start; i < permute_end; ++i) {
            സ്ഥിരം int64_t p       = patch_start_batch + i;
            സ്ഥിരം int64_t batch_n = p / (dst_w * dst_h);
            സ്ഥിരം int64_t dst_y   = (p / dst_w) % dst_h;
            സ്ഥിരം int64_t dst_x   = p % dst_w;

            വേണ്ടി (int64_t oc = 0; oc < c_out; ++oc) {
                സ്ഥിരം ദശാംശം value = gemm_output[i * c_out + oc];
                ദശാംശം * dst_ptr = (ദശാംശം *)((അക്ഷരം *)dst_data + dst_x * dst->nb[0] + dst_y * dst->nb[1] + oc * dst->nb[2] + batch_n * dst->nb[3]);
                *dst_ptr = value;
            }
        }
    }
}

ശൂന്യം ggml_compute_forward_conv_2d(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];

    ggml_compute_forward_conv_2d_impl(params, src0, src1, dst, src0->type);
}

// ggml_compute_forward_conv_3d

സ്ഥിതം ശൂന്യം ggml_compute_forward_conv_3d_impl(സ്ഥിരം ggml_compute_params * params,
                                              സ്ഥിരം ggml_tensor *         kernel,
                                              സ്ഥിരം ggml_tensor *         src,
                                              ggml_tensor *               dst,
                                              ggml_type                   kernel_type) {

    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(kernel_type == GGML_TYPE_F16 || kernel_type == GGML_TYPE_F32);
    GGML_ASSERT(kernel->type == kernel_type);

    സ്ഥിരം ggml_type_traits * traits = ggml_get_type_traits(kernel_type);

    സ്ഥിരം int32_t s0 = dst->op_params[0];
    സ്ഥിരം int32_t s1 = dst->op_params[1];
    സ്ഥിരം int32_t s2 = dst->op_params[2];
    സ്ഥിരം int32_t p0 = dst->op_params[3];
    സ്ഥിരം int32_t p1 = dst->op_params[4];
    സ്ഥിരം int32_t p2 = dst->op_params[5];
    സ്ഥിരം int32_t d0 = dst->op_params[6];
    സ്ഥിരം int32_t d1 = dst->op_params[7];
    സ്ഥിരം int32_t d2 = dst->op_params[8];
    സ്ഥിരം int32_t c  = dst->op_params[9];
    സ്ഥിരം int32_t n  = dst->op_params[10];
    സ്ഥിരം int32_t oc = dst->op_params[11];

    സ്ഥിരം int64_t src_w = src->ne[0];
    സ്ഥിരം int64_t src_h = src->ne[1];
    സ്ഥിരം int64_t src_d = src->ne[2];
    സ്ഥിരം int64_t knl_w = kernel->ne[0];
    സ്ഥിരം int64_t knl_h = kernel->ne[1];
    സ്ഥിരം int64_t knl_d = kernel->ne[2];
    സ്ഥിരം int64_t dst_w = dst->ne[0];
    സ്ഥിരം int64_t dst_h = dst->ne[1];
    സ്ഥിരം int64_t dst_d = dst->ne[2];

    സ്ഥിരം ദശാംശം * src_data = (ദശാംശം *) src->data;
    ശൂന്യം  * knl_data       = kernel->data;
    ദശാംശം * dst_data       = (ദശാംശം *) dst->data;

    സ്ഥിരം int64_t knl_n_per_channel = knl_w * knl_h * knl_d;
    സ്ഥിരം int64_t knl_n_total       = knl_n_per_channel * c;
    സ്ഥിരം int64_t patch_total       = n * dst_w * dst_h * dst_d;

    സ്ഥിരം int64_t space_per_patch   = knl_n_total * traits->type_size + oc * വലുപ്പം(ദശാംശം);
    സ്ഥിരം int64_t batch_size        = params->wsize / space_per_patch;
    സ്ഥിരം int64_t patches_per_batch = batch_size > 8 ? (batch_size / 8) * 8 : batch_size;
    സ്ഥിരം int64_t batch_n           = (patch_total + patches_per_batch - 1) / patches_per_batch;

    GGML_ASSERT(patches_per_batch > 0 && batch_size >= 1);

    ശൂന്യം * tmp = params->wdata;

    വേണ്ടി (int64_t batch_i = 0; batch_i < batch_n; ++batch_i) {
        സ്ഥിരം int64_t patch_start_batch = batch_i * patches_per_batch;
        സ്ഥിരം int64_t patch_end_batch   = std::min(patch_start_batch + patches_per_batch, patch_total);
        സ്ഥിരം int64_t patch_n_in_batch  = patch_end_batch - patch_start_batch;

        സ്ഥിരം int64_t patch_per_thread  = (patch_n_in_batch + params->nth - 1) / params->nth;
        സ്ഥിരം int64_t patch_start       = patch_start_batch + params->ith * patch_per_thread;
        സ്ഥിരം int64_t patch_end         = std::min(patch_start + patch_per_thread, patch_end_batch);

        വേണ്ടി (int64_t p = patch_start; p < patch_end; ++p) {
            സ്ഥിരം int64_t p_in_batch = p % (dst_w * dst_h * dst_d);
            സ്ഥിരം int64_t p_in_depth = p_in_batch % (dst_w * dst_h);
            സ്ഥിരം int64_t batch_idx  = p / (dst_w * dst_h * dst_d);
            സ്ഥിരം int64_t dst_z      = p_in_batch / (dst_w * dst_h);
            സ്ഥിരം int64_t dst_y      = p_in_depth / dst_w;
            സ്ഥിരം int64_t dst_x      = p_in_depth % dst_w;

            അക്ഷരം * dst_row = (അക്ഷരം *) tmp + (p % patches_per_batch) * knl_n_total * traits->type_size;

            വേണ്ടി (int64_t ic = 0; ic < c; ++ic) {
                വേണ്ടി (int64_t kz = 0; kz < knl_d; ++kz) {
                    വേണ്ടി (int64_t ky = 0; ky < knl_h; ++ky) {
                        വേണ്ടി (int64_t kx = 0; kx < knl_w; ++kx) {
                            സ്ഥിരം int64_t sz = dst_z * s2 + kz * d2 - p2;
                            സ്ഥിരം int64_t sy = dst_y * s1 + ky * d1 - p1;
                            സ്ഥിരം int64_t sx = dst_x * s0 + kx * d0 - p0;

                            int64_t dst_idx = ic * knl_n_per_channel + kz * (knl_h * knl_w) + ky * knl_w + kx;

                            ദശാംശം src_val;
                            എങ്കിൽ (sz < 0 || sz >= src_d || sy < 0 || sy >= src_h || sx < 0 || sx >= src_w) {
                                src_val = 0.0f;
                            } അല്ലെങ്കിൽ {
                                സ്ഥിരം int64_t cn_idx = batch_idx * c + ic;
                                സ്ഥിരം ദശാംശം * src_ptr = (സ്ഥിരം ദശാംശം *)((സ്ഥിരം അക്ഷരം *)src_data + sx*src->nb[0] + sy*src->nb[1] + sz*src->nb[2] + cn_idx*src->nb[3]);
                                src_val = *src_ptr;
                            }

                            അക്ഷരം * element_ptr = dst_row + dst_idx * traits->type_size;
                            എങ്കിൽ (kernel_type == GGML_TYPE_F32) {
                                *(ദശാംശം *)element_ptr = src_val;
                            } അല്ലെങ്കിൽ എങ്കിൽ (kernel_type == GGML_TYPE_F16) {
                                *(ggml_fp16_t *)element_ptr = GGML_CPU_FP32_TO_FP16(src_val);
                            }
                        }
                    }
                }
            }
        }

        ggml_barrier(params->threadpool);

        ദശാംശം * gemm_output = (ദശാംശം *) ((അക്ഷരം *) tmp + patches_per_batch * knl_n_total * traits->type_size);
        ggml_call_mul_mat(kernel_type, params, patch_n_in_batch, oc, knl_n_total, tmp, knl_data, gemm_output);

        ggml_barrier(params->threadpool);

        സ്ഥിരം int64_t permute_per_thread = (patch_n_in_batch + params->nth - 1) / params->nth;
        സ്ഥിരം int64_t permute_start = params->ith * permute_per_thread;
        സ്ഥിരം int64_t permute_end = std::min(permute_start + permute_per_thread, patch_n_in_batch);

        വേണ്ടി (int64_t i = permute_start; i < permute_end; ++i) {
            സ്ഥിരം int64_t p = patch_start_batch + i;
            സ്ഥിരം int64_t p_in_batch = p % (dst_w * dst_h * dst_d);
            സ്ഥിരം int64_t p_in_depth = p_in_batch % (dst_w * dst_h);
            സ്ഥിരം int64_t batch_idx  = p / (dst_w * dst_h * dst_d);
            സ്ഥിരം int64_t dst_z      = p_in_batch / (dst_w * dst_h);
            സ്ഥിരം int64_t dst_y      = p_in_depth / dst_w;
            സ്ഥിരം int64_t dst_x      = p_in_depth % dst_w;

            വേണ്ടി (int64_t ioc = 0; ioc < oc; ++ioc) {
                സ്ഥിരം ദശാംശം value = gemm_output[i * oc + ioc];
                സ്ഥിരം int64_t ocn_idx = batch_idx * oc + ioc;
                ദശാംശം * dst_ptr = (ദശാംശം *)((അക്ഷരം *)dst_data + dst_x*dst->nb[0] + dst_y*dst->nb[1] + dst_z*dst->nb[2] + ocn_idx*dst->nb[3]);
                *dst_ptr = value;
            }
        }
    }
}

ശൂന്യം ggml_compute_forward_conv_3d(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {
    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];
    ggml_compute_forward_conv_3d_impl(params, src0, src1, dst, src0->type);
}

template <typename kernel_t>
സ്ഥിതം ശൂന്യം ggml_compute_forward_conv_transpose_2d_impl(
    സ്ഥിരം ggml_compute_params * params,
          ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ദ്വിമാന_OP_LOCALS

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ nk = ne00*ne01*ne02*ne03;

    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == വലുപ്പം(ദശാംശം));

    എങ്കിൽ (ith == 0) {
        memset(params->wdata, 0, params->wsize);

        // permute kernel data (src0) from (Kw x Kh x Cout x Cin) to (Cin x Kw x Kh x Cout)
        {
            kernel_t * സ്ഥിരം wdata = (kernel_t *) params->wdata + 0;

            വേണ്ടി (int64_t i03 = 0; i03 < ne03; i03++) {
                വേണ്ടി (int64_t i02 = 0; i02 < ne02; i02++) {
                    സ്ഥിരം kernel_t * സ്ഥിരം src = (kernel_t *)((അക്ഷരം *) src0->data + i03*nb03 + i02*nb02);
                    kernel_t * dst_data = wdata + i02*ne01*ne00*ne03;
                    വേണ്ടി (int64_t i01 = 0; i01 < ne01; i01++) {
                        വേണ്ടി (int64_t i00 = 0; i00 < ne00; i00++) {
                            dst_data[i01*ne00*ne03 + i00*ne03 + i03] = src[i01 * ne00 + i00];
                        }
                    }
                }
            }
        }

        // permute source data (src1) from (Sw x Sh x Cin) to (Cin x Sw x Sh), for all batches
        {
            kernel_t * സ്ഥിരം wdata = (kernel_t *) params->wdata + nk;
            വേണ്ടി (പൂർണ്ണസംഖ്യ i13 = 0; i13 < ne13; i13++) {
                kernel_t * സ്ഥിരം wdata_b = wdata + i13*ne10*ne11*ne12;
                വേണ്ടി (പൂർണ്ണസംഖ്യ i12 = 0; i12 < ne12; i12++) {
                    വേണ്ടി (പൂർണ്ണസംഖ്യ i11 = 0; i11 < ne11; i11++) {
                        സ്ഥിരം ദശാംശം * സ്ഥിരം src = (ദശാംശം *)((അക്ഷരം *) src1->data + i13*nb13 + i12*nb12 + i11*nb11);
                        kernel_t * dst_data = wdata_b + i11*ne10*ne12;
                        വേണ്ടി (പൂർണ്ണസംഖ്യ i10 = 0; i10 < ne10; i10++) {
                            എങ്കിൽ constexpr (std::is_same_v<kernel_t, ggml_fp16_t>) {
                                dst_data[i10*ne12 + i12] = GGML_CPU_FP32_TO_FP16(src[i10]);
                            } അല്ലെങ്കിൽ {
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

    സ്ഥിരം int32_t stride = ggml_get_op_params_i32(dst, 0);

    // total patches in dst
    സ്ഥിരം പൂർണ്ണസംഖ്യ np = ne2;

    // patches per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dp = (np + nth - 1)/nth;

    // patch range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ip0 = dp*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ip1 = MIN(ip0 + dp, np);

    kernel_t * സ്ഥിരം wdata = (kernel_t *) params->wdata + 0;
    kernel_t * സ്ഥിരം wdata_src = wdata + nk;

    വേണ്ടി (പൂർണ്ണസംഖ്യ i2 = ip0; i2 < ip1; i2++) { // Cout
        kernel_t * wdata_kernel = wdata + i2*ne01*ne00*ne03;
        വേണ്ടി (പൂർണ്ണസംഖ്യ i3 = 0; i3 < ne3; i3++) { // batch
            ദശാംശം * dst_data = (ദശാംശം *)((അക്ഷരം *) dst->data + i3*nb3 + i2*nb2);
            kernel_t * wdata_src_b = wdata_src + i3*ne10*ne11*ne12;
            വേണ്ടി (പൂർണ്ണസംഖ്യ i11 = 0; i11 < ne11; i11++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ i10 = 0; i10 < ne10; i10++) {
                    സ്ഥിരം പൂർണ്ണസംഖ്യ i1n = i11*ne10*ne12 + i10*ne12;
                    വേണ്ടി (പൂർണ്ണസംഖ്യ i01 = 0; i01 < ne01; i01++) {
                        വേണ്ടി (പൂർണ്ണസംഖ്യ i00 = 0; i00 < ne00; i00++) {
                            ദശാംശം v = 0;
                            എങ്കിൽ constexpr (std::is_same_v<kernel_t, ggml_fp16_t>) {
                                ggml_vec_dot_f16(ne03, &v, 0,
                                        wdata_src_b + i1n, 0,
                                        wdata_kernel + i01*ne00*ne03 + i00*ne03, 0, 1);
                            } അല്ലെങ്കിൽ {
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

ശൂന്യം ggml_compute_forward_conv_transpose_2d(
        സ്ഥിരം ggml_compute_params * params,
              ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F16:
            {
                ggml_compute_forward_conv_transpose_2d_impl<ggml_fp16_t>(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_conv_transpose_2d_impl<ദശാംശം>(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_conv_2d_dw

ഘടന ggml_conv_2d_dw_params {
    int64_t channels;
    int64_t batch;
    int64_t src_w;
    int64_t src_h;
    int64_t dst_w;
    int64_t dst_h;
    int64_t knl_w;
    int64_t knl_h;
    പൂർണ്ണസംഖ്യ stride_x;
    പൂർണ്ണസംഖ്യ stride_y;
    പൂർണ്ണസംഖ്യ pad_x;
    പൂർണ്ണസംഖ്യ pad_y;
    പൂർണ്ണസംഖ്യ dilation_x;
    പൂർണ്ണസംഖ്യ dilation_y;
};

സ്ഥിതം വരിയിൽ ദശാംശം ggml_conv_2d_dw_knl_f32(സ്ഥിരം അക്ഷരം * data, int64_t i, ggml_type type) {
    എങ്കിൽ (type == GGML_TYPE_F16) {
        തിരികെ GGML_FP16_TO_FP32(((സ്ഥിരം ggml_fp16_t *)data)[i]);
    }
    തിരികെ ((സ്ഥിരം ദശാംശം *)data)[i];
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_conv_2d_dw_cwhn(
        സ്ഥിരം ggml_compute_params * params,
        സ്ഥിരം ggml_tensor * src,
        സ്ഥിരം ggml_tensor * kernel,
        ggml_tensor * dst,
        സ്ഥിരം ggml_conv_2d_dw_params & p) {

    സ്ഥിരം int64_t c = p.channels;
    സ്ഥിരം അക്ഷരം * knl_data = (സ്ഥിരം അക്ഷരം *)kernel->data;
    സ്ഥിരം ggml_type knl_type = kernel->type;

    സ്ഥിരം int64_t rows_total = p.dst_h * p.batch;
    സ്ഥിരം int64_t rows_per_thread = (rows_total + params->nth - 1) / params->nth;
    സ്ഥിരം int64_t row_start = params->ith * rows_per_thread;
    സ്ഥിരം int64_t row_end = MIN(row_start + rows_per_thread, rows_total);

#ifdef GGML_SIMD
    int64_t c_pkg_end = 0;
    int64_t pkg_size = GGML_F32_EPR;
    എങ്കിൽ (knl_type == GGML_TYPE_F32) {
    #എങ്കിൽ defined(__ARM_FEATURE_SVE)
        pkg_size = svcntw();
    #അല്ലെങ്കിൽ
        pkg_size = GGML_F32_EPR;
    #endif
        c_pkg_end = (c / pkg_size) * pkg_size;
    }
#അല്ലെങ്കിൽ
    സ്ഥിരം int64_t c_pkg_end = 0;
#endif

    വേണ്ടി (int64_t row = row_start; row < row_end; ++row) {
        സ്ഥിരം int64_t dst_y = row % p.dst_h;
        സ്ഥിരം ദശാംശം * src_data = (സ്ഥിരം ദശാംശം *)src->data + (row / p.dst_h) * p.src_w * p.src_h * c;
        വേണ്ടി (int64_t dst_x = 0; dst_x < p.dst_w; ++dst_x) {
            ദശാംശം * dst_data = (ദശാംശം *)dst->data + (row * p.dst_w + dst_x) * c;
            സ്ഥിരം int64_t src_y_base = dst_y * p.stride_y - p.pad_y;
            സ്ഥിരം int64_t src_x_base = dst_x * p.stride_x - p.pad_x;

#ifdef GGML_SIMD
            വേണ്ടി (int64_t c_i = 0; c_i < c_pkg_end; c_i += pkg_size) {
                GGML_F32_VEC sum = GGML_F32_VEC_ZERO;
                വേണ്ടി (int64_t knl_y = 0; knl_y < p.knl_h; ++knl_y) {
                    സ്ഥിരം int64_t src_y = src_y_base + knl_y * p.dilation_y;
                    എങ്കിൽ (src_y < 0 || src_y >= p.src_h) {
                        തുടരുക;
                    }
                    വേണ്ടി (int64_t knl_x = 0; knl_x < p.knl_w; ++knl_x) {
                        സ്ഥിരം int64_t src_x = src_x_base + knl_x * p.dilation_x;
                        എങ്കിൽ (src_x < 0 || src_x >= p.src_w) {
                            തുടരുക;
                        }
                        സ്ഥിരം ദശാംശം * kp = (സ്ഥിരം ദശാംശം *)knl_data + (knl_y * p.knl_w + knl_x) * c + c_i;
                        GGML_F32_VEC k = GGML_F32_VEC_LOAD(kp);
                        GGML_F32_VEC s = GGML_F32_VEC_LOAD(src_data + (src_y * p.src_w + src_x) * c + c_i);
                        sum = GGML_F32_VEC_FMA(sum, k, s);
                    }
                }
                GGML_F32_VEC_STORE(dst_data + c_i, sum);
            }
#endif
            വേണ്ടി (int64_t c_i = c_pkg_end; c_i < c; ++c_i) {
                ദശാംശം sum = 0.0f;
                വേണ്ടി (int64_t knl_y = 0; knl_y < p.knl_h; ++knl_y) {
                    സ്ഥിരം int64_t src_y = src_y_base + knl_y * p.dilation_y;
                    എങ്കിൽ (src_y < 0 || src_y >= p.src_h) {
                        തുടരുക;
                    }
                    വേണ്ടി (int64_t knl_x = 0; knl_x < p.knl_w; ++knl_x) {
                        സ്ഥിരം int64_t src_x = src_x_base + knl_x * p.dilation_x;
                        എങ്കിൽ (src_x < 0 || src_x >= p.src_w) {
                            തുടരുക;
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

സ്ഥിതം ശൂന്യം ggml_compute_forward_conv_2d_dw_whcn(
        സ്ഥിരം ggml_compute_params * params,
        സ്ഥിരം ggml_tensor * src,
        സ്ഥിരം ggml_tensor * kernel,
        ggml_tensor * dst,
        സ്ഥിരം ggml_conv_2d_dw_params & p) {

    സ്ഥിരം int64_t n = p.channels * p.batch;
    സ്ഥിരം int64_t per_thread = (n + params->nth - 1) / params->nth;
    സ്ഥിരം int64_t start = params->ith * per_thread;
    സ്ഥിരം int64_t end = MIN(start + per_thread, n);
    സ്ഥിരം അക്ഷരം * knl_base = (സ്ഥിരം അക്ഷരം *)kernel->data;
    സ്ഥിരം ggml_type knl_type = kernel->type;

    വേണ്ടി (int64_t i = start; i < end; ++i) {
        സ്ഥിരം int64_t knl_offset = (i % p.channels) * p.knl_w * p.knl_h;
        സ്ഥിരം ദശാംശം * src_data = (സ്ഥിരം ദശാംശം *)src->data + i * p.src_w * p.src_h;
        ദശാംശം * dst_data = (ദശാംശം *)dst->data + i * p.dst_w * p.dst_h;

        വേണ്ടി (int64_t dst_y = 0; dst_y < p.dst_h; ++dst_y) {
            വേണ്ടി (int64_t dst_x = 0; dst_x < p.dst_w; ++dst_x) {

                ദശാംശം sum = 0.0f;
                വേണ്ടി (int64_t knl_y = 0; knl_y < p.knl_h; ++knl_y) {
                    സ്ഥിരം int64_t src_y = dst_y * p.stride_y + knl_y * p.dilation_y - p.pad_y;
                    എങ്കിൽ (src_y < 0 || src_y >= p.src_h) {
                        തുടരുക;
                    }
                    വേണ്ടി (int64_t knl_x = 0; knl_x < p.knl_w; ++knl_x) {
                        സ്ഥിരം int64_t src_x = dst_x * p.stride_x + knl_x * p.dilation_x - p.pad_x;
                        എങ്കിൽ (src_x < 0 || src_x >= p.src_w) {
                            തുടരുക;
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

ശൂന്യം ggml_compute_forward_conv_2d_dw(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * kernel = dst->src[0];
    സ്ഥിരം ggml_tensor * src = dst->src[1];
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

    എങ്കിൽ (ggml_is_contiguous(src)) {
        ggml_compute_forward_conv_2d_dw_whcn(params, src, kernel, dst, p);
    } അല്ലെങ്കിൽ എങ്കിൽ (ggml_is_contiguous_channels(src)) {
        GGML_ASSERT(kernel->nb[0] >= kernel->nb[2] && kernel->nb[1] >= kernel->nb[0]);
        ggml_compute_forward_conv_2d_dw_cwhn(params, src, kernel, dst, p);
    } അല്ലെങ്കിൽ {
        GGML_ABORT("non-contiguous memory layout not supported");
    }
}

// ggml_compute_forward_pool_1d_ksp
സ്ഥിതം ശൂന്യം ggml_compute_forward_pool_1d_ksp(
        സ്ഥിരം ggml_compute_params * params,
        സ്ഥിരം ggml_op_pool op,
        സ്ഥിരം പൂർണ്ണസംഖ്യ k,
        സ്ഥിരം പൂർണ്ണസംഖ്യ s,
        സ്ഥിരം പൂർണ്ണസംഖ്യ p,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src = dst->src[0];

    പ്രതിജ്ഞാ(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16);

    എങ്കിൽ (params->ith != 0) {
        തിരികെ;
    }

    സ്ഥിരം int64_t IW = src->ne[0];
    സ്ഥിരം int64_t OW = dst->ne[0];

    സ്ഥിരം int64_t nr = ggml_nrows(src);

    വേണ്ടി (int64_t ir = 0; ir < nr; ++ir) {
        സ്ഥിരം അക്ഷരം * srow_bytes =            (സ്ഥിരം അക്ഷരം *) src->data + ir * src->nb[1];
        ദശാംശം      * drow       = (ദശാംശം *) ((      അക്ഷരം *) dst->data + ir * dst->nb[1]);

        വേണ്ടി (int64_t ow = 0; ow < OW; ++ow) {
            ദശാംശം res = 0;
            തിരഞ്ഞെടുപ്പ് (op) {
                അവസ്ഥ GGML_OP_POOL_AVG: res = 0.0f;     നിർത്തുക;
                അവസ്ഥ GGML_OP_POOL_MAX: res = -FLT_MAX; നിർത്തുക;
                അവസ്ഥ GGML_OP_POOL_എണ്ണം: GGML_ABORT("fatal error");
            }

            പൂർണ്ണസംഖ്യ count = 0;
            സ്ഥിരം പൂർണ്ണസംഖ്യ base = (പൂർണ്ണസംഖ്യ) ow * s - p;

            വേണ്ടി (പൂർണ്ണസംഖ്യ ki = 0; ki < k; ++ki) {
                സ്ഥിരം പൂർണ്ണസംഖ്യ j = base + ki;
                എങ്കിൽ (j < 0 || j >= (പൂർണ്ണസംഖ്യ) IW) {
                    തുടരുക;
                }

                ദശാംശം v;
                എങ്കിൽ (src->type == GGML_TYPE_F32) {
                    v = ((സ്ഥിരം ദശാംശം *) srow_bytes)[j];
                } അല്ലെങ്കിൽ {
                    v = GGML_CPU_FP16_TO_FP32(((സ്ഥിരം ggml_fp16_t *) srow_bytes)[j]);
                }

                തിരഞ്ഞെടുപ്പ് (op) {
                    അവസ്ഥ GGML_OP_POOL_AVG: res += v;                നിർത്തുക;
                    അവസ്ഥ GGML_OP_POOL_MAX: res =  std::max(v, res); നിർത്തുക;
                    അവസ്ഥ GGML_OP_POOL_എണ്ണം: GGML_ABORT("fatal error");
                }

                ++count;
            }

            തിരഞ്ഞെടുപ്പ് (op) {
                അവസ്ഥ GGML_OP_POOL_AVG: res = (count > 0) ? (res / count) : 0.0f; നിർത്തുക;
                അവസ്ഥ GGML_OP_POOL_MAX:                                           നിർത്തുക;
                അവസ്ഥ GGML_OP_POOL_എണ്ണം: GGML_ABORT("fatal error");
            }

            drow[ow] = res;
        }
    }
}

// ggml_compute_forward_pool_1d

ശൂന്യം ggml_compute_forward_pool_1d(
        സ്ഥിരം ggml_compute_params * params,
              ggml_tensor * dst) {

    സ്ഥിരം int32_t * opts = (സ്ഥിരം int32_t *)dst->op_params;
    ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    സ്ഥിരം പൂർണ്ണസംഖ്യ k0 = opts[1];
    സ്ഥിരം പൂർണ്ണസംഖ്യ s0 = opts[2];
    സ്ഥിരം പൂർണ്ണസംഖ്യ p0 = opts[3];

    ggml_compute_forward_pool_1d_ksp(params, op, k0, s0, p0, dst);
}

// ggml_compute_forward_pool_2d

ശൂന്യം ggml_compute_forward_pool_2d(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src = dst->src[0];

    പ്രതിജ്ഞാ(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16);

    എങ്കിൽ (params->ith != 0) {
        തിരികെ;
    }

    സ്ഥിരം int32_t * opts = (സ്ഥിരം int32_t *)dst->op_params;

    ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    സ്ഥിരം പൂർണ്ണസംഖ്യ k0 = opts[1];
    സ്ഥിരം പൂർണ്ണസംഖ്യ k1 = opts[2];
    സ്ഥിരം പൂർണ്ണസംഖ്യ s0 = opts[3];
    സ്ഥിരം പൂർണ്ണസംഖ്യ s1 = opts[4];
    സ്ഥിരം പൂർണ്ണസംഖ്യ p0 = opts[5];
    സ്ഥിരം പൂർണ്ണസംഖ്യ p1 = opts[6];
    സ്ഥിരം അക്ഷരം * cdata = (സ്ഥിരം അക്ഷരം*)src->data;
    സ്ഥിരം അക്ഷരം * സ്ഥിരം data_end = cdata + ggml_nbytes(src);

    സ്ഥിരം int64_t px = dst->ne[0];
    സ്ഥിരം int64_t py = dst->ne[1];
    സ്ഥിരം int64_t pa = px * py;

    ദശാംശം * dplane = (ദശാംശം *)dst->data;

    സ്ഥിരം പൂർണ്ണസംഖ്യ ka = k0 * k1;
    സ്ഥിരം പൂർണ്ണസംഖ്യ offset0 = -p0;
    സ്ഥിരം പൂർണ്ണസംഖ്യ offset1 = -p1;

    ആകുമ്പോൾ (cdata < data_end) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ oy = 0; oy < py; ++oy) {
            ദശാംശം * സ്ഥിരം drow = dplane + oy * px;
            ദശാംശം * സ്ഥിരം out  = drow;

            വേണ്ടി (പൂർണ്ണസംഖ്യ ox = 0; ox < px; ++ox) {
                ദശാംശം res = 0;
                തിരഞ്ഞെടുപ്പ് (op) {
                    അവസ്ഥ GGML_OP_POOL_AVG: res = 0;        നിർത്തുക;
                    അവസ്ഥ GGML_OP_POOL_MAX: res = -FLT_MAX; നിർത്തുക;
                    അവസ്ഥ GGML_OP_POOL_എണ്ണം: GGML_ABORT("fatal error");
                }

                സ്ഥിരം പൂർണ്ണസംഖ്യ ix = offset0 + ox * s0;
                സ്ഥിരം പൂർണ്ണസംഖ്യ iy = offset1 + oy * s1;

                വേണ്ടി (പൂർണ്ണസംഖ്യ ky = 0; ky < k1; ++ky) {
                    എങ്കിൽ (iy + ky < 0 || iy + ky >= src->ne[1]) {
                        തുടരുക;
                    }

                    സ്ഥിരം ശൂന്യം * srow = (സ്ഥിരം ശൂന്യം *)(cdata + src->nb[1] * (iy + ky));
                    വേണ്ടി (പൂർണ്ണസംഖ്യ kx = 0; kx < k0; ++kx) {
                        പൂർണ്ണസംഖ്യ j = ix + kx;
                        എങ്കിൽ (j < 0 || j >= src->ne[0]) {
                            തുടരുക;
                        }

                        സ്ഥിരം ദശാംശം srow_j = (src->type == GGML_TYPE_F32) ? ((സ്ഥിരം ദശാംശം*)srow)[j] : GGML_CPU_FP16_TO_FP32(((സ്ഥിരം ggml_fp16_t*)srow)[j]);
                        തിരഞ്ഞെടുപ്പ് (op) {
                            അവസ്ഥ GGML_OP_POOL_AVG: res += srow_j;                നിർത്തുക;
                            അവസ്ഥ GGML_OP_POOL_MAX: res =  std::max(srow_j, res); നിർത്തുക;
                            അവസ്ഥ GGML_OP_POOL_എണ്ണം:               GGML_ABORT("fatal error");
                        }
                    }
                }
                തിരഞ്ഞെടുപ്പ് (op) {
                    അവസ്ഥ GGML_OP_POOL_AVG:           res /= ka; നിർത്തുക;
                    അവസ്ഥ GGML_OP_POOL_MAX:                      നിർത്തുക;
                    അവസ്ഥ GGML_OP_POOL_എണ്ണം: GGML_ABORT("fatal error");
                }

                out[ox] = res;
            }
        }

        cdata  += src->nb[2];
        dplane += pa;
    }
}

// ggml_compute_forward_pool_2d_back

ശൂന്യം ggml_compute_forward_pool_2d_back(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src  = dst->src[0];
    സ്ഥിരം ggml_tensor * dstf = dst->src[1]; // forward tensor of dst

    പ്രതിജ്ഞാ(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);

    എങ്കിൽ (params->ith != 0) {
        തിരികെ;
    }

    സ്ഥിരം int32_t * opts = (സ്ഥിരം int32_t *)dst->op_params;
    ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    സ്ഥിരം പൂർണ്ണസംഖ്യ k0 = opts[1];
    സ്ഥിരം പൂർണ്ണസംഖ്യ k1 = opts[2];
    സ്ഥിരം പൂർണ്ണസംഖ്യ s0 = opts[3];
    സ്ഥിരം പൂർണ്ണസംഖ്യ s1 = opts[4];
    സ്ഥിരം പൂർണ്ണസംഖ്യ p0 = opts[5];
    സ്ഥിരം പൂർണ്ണസംഖ്യ p1 = opts[6];

    അക്ഷരം       * cdata  = (അക്ഷരം       *) dst->data;
    സ്ഥിരം അക്ഷരം * cdataf = (സ്ഥിരം അക്ഷരം *) dstf->data;
    സ്ഥിരം അക്ഷരം * സ്ഥിരം data_end = cdata + ggml_nbytes(dst);

    GGML_ASSERT(params->ith == 0);
    memset(cdata, 0, ggml_nbytes(dst));

    സ്ഥിരം int64_t px = src->ne[0];
    സ്ഥിരം int64_t py = src->ne[1];
    സ്ഥിരം int64_t pa = px * py;

    സ്ഥിരം ദശാംശം * splane = (സ്ഥിരം ദശാംശം *) src->data;

    സ്ഥിരം പൂർണ്ണസംഖ്യ ka = k0 * k1;
    സ്ഥിരം പൂർണ്ണസംഖ്യ offset0 = -p0;
    സ്ഥിരം പൂർണ്ണസംഖ്യ offset1 = -p1;

    ആകുമ്പോൾ (cdata < data_end) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ oy = 0; oy < py; ++oy) {
            സ്ഥിരം ദശാംശം * സ്ഥിരം srow = splane + oy * px;
            വേണ്ടി (പൂർണ്ണസംഖ്യ ox = 0; ox < px; ++ox) {
                സ്ഥിരം ദശാംശം grad0 = srow[ox];

                സ്ഥിരം പൂർണ്ണസംഖ്യ ix = offset0 + ox * s0;
                സ്ഥിരം പൂർണ്ണസംഖ്യ iy = offset1 + oy * s1;

                എങ്കിൽ (op == GGML_OP_POOL_MAX) {
                    ദശാംശം maxval = -FLT_MAX;
                    പൂർണ്ണസംഖ്യ kxmax = -1;
                    പൂർണ്ണസംഖ്യ kymax = -1;

                    വേണ്ടി (പൂർണ്ണസംഖ്യ ky = 0; ky < k1; ++ky) {
                        എങ്കിൽ (iy + ky < 0 || iy + ky >= dst->ne[1]) {
                            തുടരുക;
                        }
                        സ്ഥിരം ശൂന്യം * drowf = (സ്ഥിരം ശൂന്യം *)(cdataf + dst->nb[1] * (iy + ky));
                        വേണ്ടി (പൂർണ്ണസംഖ്യ kx = 0; kx < k0; ++kx) {
                            പൂർണ്ണസംഖ്യ j = ix + kx;
                            എങ്കിൽ (j < 0 || j >= dst->ne[0]) {
                                തുടരുക;
                            }

                            സ്ഥിരം ദശാംശം val = dst->type == GGML_TYPE_F32 ?
                                ((സ്ഥിരം ദശാംശം *) drowf)[j] : GGML_CPU_FP16_TO_FP32(((സ്ഥിരം ggml_fp16_t *) drowf)[j]);
                            എങ്കിൽ (val <= maxval) {
                                തുടരുക;
                            }

                            maxval = val;
                            kxmax = kx;
                            kymax = ky;
                        }
                    }

                    എങ്കിൽ (kxmax == -1 || kymax == -1) {
                        തുടരുക;
                    }

                    ശൂന്യം * drow = (ശൂന്യം *)(cdata + dst->nb[1] * (iy + kymax));
                    സ്ഥിരം പൂർണ്ണസംഖ്യ j = ix + kxmax;
                    എങ്കിൽ (dst->type == GGML_TYPE_F32) {
                        ((ദശാംശം *) drow)[j] += grad0;
                    } അല്ലെങ്കിൽ {
                        ((ggml_fp16_t *) drow)[j] = GGML_CPU_FP32_TO_FP16(grad0 + GGML_CPU_FP16_TO_FP32(((സ്ഥിരം ggml_fp16_t *) drow)[j]));
                    }
                } അല്ലെങ്കിൽ എങ്കിൽ (op == GGML_OP_POOL_AVG) {
                    സ്ഥിരം ദശാംശം grad = grad0 / ka;

                    വേണ്ടി (പൂർണ്ണസംഖ്യ ky = 0; ky < k1; ++ky) {
                        എങ്കിൽ (iy + ky < 0 || iy + ky >= dst->ne[1]) {
                            തുടരുക;
                        }
                        ശൂന്യം * drow = (ശൂന്യം *)(cdata + dst->nb[1] * (iy + ky));
                        വേണ്ടി (പൂർണ്ണസംഖ്യ kx = 0; kx < k0; ++kx) {
                            പൂർണ്ണസംഖ്യ j = ix + kx;
                            എങ്കിൽ (j < 0 || j >= dst->ne[0]) {
                                തുടരുക;
                            }

                            എങ്കിൽ (dst->type == GGML_TYPE_F32) {
                                ((ദശാംശം *) drow)[j] += grad;
                            } അല്ലെങ്കിൽ {
                                ((ggml_fp16_t *) drow)[j] += GGML_CPU_FP32_TO_FP16(grad);
                            }
                        }
                    }
                } അല്ലെങ്കിൽ {
                    GGML_ASSERT(അസത്യം);
                }
            }
        }

        cdata  += dst->nb[2];
        cdataf += dst->nb[2];
        splane += pa;
    }
}

// ggml_compute_forward_upscale

സ്ഥിതം ശൂന്യം ggml_compute_forward_upscale_f32(
    സ്ഥിരം ggml_compute_params * params,
    ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    ദശാംശം sf0 = (ദശാംശം)ne0/src0->ne[0];
    ദശാംശം sf1 = (ദശാംശം)ne1/src0->ne[1];
    ദശാംശം sf2 = (ദശാംശം)ne2/src0->ne[2];
    ദശാംശം sf3 = (ദശാംശം)ne3/src0->ne[3];
    ദശാംശം pixel_offset = 0.5f;

    സ്ഥിരം int32_t mode_flags = ggml_get_op_params_i32(dst, 0);
    സ്ഥിരം ggml_scale_mode mode = (ggml_scale_mode) (mode_flags & 0xFF);

    എങ്കിൽ (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        pixel_offset = 0.0f;
        sf0 = ne0 > 1 && ne00 > 1 ? (ദശാംശം)(ne0 - 1) / (ne00 - 1) : sf0;
        sf1 = ne1 > 1 && ne01 > 1 ? (ദശാംശം)(ne1 - 1) / (ne01 - 1) : sf1;
    }

    എങ്കിൽ (mode == GGML_SCALE_MODE_NEAREST) {
        വേണ്ടി (int64_t i3 = 0; i3 < ne3; i3++) {
            സ്ഥിരം int64_t i03 = i3 / sf3;
            വേണ്ടി (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                സ്ഥിരം int64_t i02 = i2 / sf2;
                വേണ്ടി (int64_t i1 = 0; i1 < ne1; i1++) {
                    സ്ഥിരം int64_t i01 = i1 / sf1;
                    വേണ്ടി (int64_t i0 = 0; i0 < ne0; i0++) {
                        സ്ഥിരം int64_t i00 = i0 / sf0;

                        സ്ഥിരം ദശാംശം * x = (ദശാംശം *)((അക്ഷരം *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              ദശാംശം * y = (ദശാംശം *)((അക്ഷരം *)  dst->data +  i0*nb0  +  i1*nb1  +  i2*nb2  +  i3*nb3);

                        *y = *x;
                    }
                }
            }
        }
    } അല്ലെങ്കിൽ എങ്കിൽ (mode == GGML_SCALE_MODE_BILINEAR && (mode_flags & GGML_SCALE_FLAG_ANTIALIAS)) {
        // Similar to F.interpolate(..., mode="bilinear", align_corners=False, antialias=True)
        // https://github.com/pytorch/pytorch/blob/8871ff29b743948d1225389d5b7068f37b22750b/aten/src/ATen/native/cpu/UpSampleKernel.cpp
        auto triangle_filter = [](ദശാംശം x) -> ദശാംശം {
            തിരികെ std::max(1.0f - fabsf(x), 0.0f);
        };

        // support and invscale, minimum 1 pixel for bilinear
        സ്ഥിരം ദശാംശം support1  = std::max(1.0f, 1.0f / sf1);
        സ്ഥിരം ദശാംശം invscale1 = 1.0f / support1;
        സ്ഥിരം ദശാംശം support0  = std::max(1.0f, 1.0f / sf0);
        സ്ഥിരം ദശാംശം invscale0 = 1.0f / support0;

        വേണ്ടി (int64_t i3 = 0; i3 < ne3; i3++) {
            സ്ഥിരം int64_t i03 = i3 / sf3;
            വേണ്ടി (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                സ്ഥിരം int64_t i02 = i2 / sf2;
                വേണ്ടി (int64_t i1 = 0; i1 < ne1; i1++) {
                    സ്ഥിരം ദശാംശം y = ((ദശാംശം) i1 + pixel_offset) / sf1;
                    വേണ്ടി (int64_t i0 = 0; i0 < ne0; i0++) {
                        സ്ഥിരം ദശാംശം x = ((ദശാംശം) i0 + pixel_offset) / sf0;

                        // the range of source pixels that contribute
                        സ്ഥിരം int64_t x_min = std::max<int64_t>(x - support0 + pixel_offset, 0);
                        സ്ഥിരം int64_t x_max = std::min<int64_t>(x + support0 + pixel_offset, ne00);
                        സ്ഥിരം int64_t y_min = std::max<int64_t>(y - support1 + pixel_offset, 0);
                        സ്ഥിരം int64_t y_max = std::min<int64_t>(y + support1 + pixel_offset, ne01);

                        // bilinear filter with antialiasing
                        ദശാംശം val = 0.0f;
                        ദശാംശം total_weight = 0.0f;

                        വേണ്ടി (int64_t sy = y_min; sy < y_max; sy++) {
                            സ്ഥിരം ദശാംശം weight_y = triangle_filter((sy - y + pixel_offset) * invscale1);

                            വേണ്ടി (int64_t sx = x_min; sx < x_max; sx++) {
                                സ്ഥിരം ദശാംശം weight_x = triangle_filter((sx - x + pixel_offset) * invscale0);
                                സ്ഥിരം ദശാംശം weight = weight_x * weight_y;

                                എങ്കിൽ (weight <= 0.0f) {
                                    തുടരുക;
                                }

                                സ്ഥിരം ദശാംശം pixel = *(സ്ഥിരം ദശാംശം *)((സ്ഥിരം അക്ഷരം *)src0->data + sx*nb00 + sy*nb01 + i02*nb02 + i03*nb03);
                                val += pixel * weight;
                                total_weight += weight;
                            }
                        }

                        എങ്കിൽ (total_weight > 0.0f) {
                            val /= total_weight;
                        }

                        ദശാംശം * dst_ptr = (ദശാംശം *)((അക്ഷരം *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);
                        *dst_ptr = val;
                    }
                }
            }
        }
    } അല്ലെങ്കിൽ എങ്കിൽ (mode == GGML_SCALE_MODE_BILINEAR) {
        വേണ്ടി (int64_t i3 = 0; i3 < ne3; i3++) {
            സ്ഥിരം int64_t i03 = i3 / sf3;
            വേണ്ടി (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                സ്ഥിരം int64_t i02 = i2 / sf2;
                വേണ്ടി (int64_t i1 = 0; i1 < ne1; i1++) {
                    സ്ഥിരം ദശാംശം y = ((ദശാംശം)i1 + pixel_offset) / sf1 - pixel_offset;
                    int64_t y0 = (int64_t)floorf(y);
                    int64_t y1 = y0 + 1;

                    y0 = std::max(int64_t(0), std::min(y0, ne01 - 1));
                    y1 = std::max(int64_t(0), std::min(y1, ne01 - 1));

                    ദശാംശം dy = y - (ദശാംശം)y0;
                    dy = std::max(0.0f, std::min(dy, 1.0f));

                    വേണ്ടി (int64_t i0 = 0; i0 < ne0; i0++) {
                        സ്ഥിരം ദശാംശം x = ((ദശാംശം)i0 + pixel_offset) / sf0 - pixel_offset;
                        int64_t x0 = (int64_t)floorf(x);
                        int64_t x1 = x0 + 1;

                        x0 = std::max(int64_t(0), std::min(x0, ne00 - 1));
                        x1 = std::max(int64_t(0), std::min(x1, ne00 - 1));

                        ദശാംശം dx = x - (ദശാംശം)x0;
                        dx = std::max(0.0f, std::min(dx, 1.0f));

                        // fetch the four surrounding pixel values and interpolate
                        സ്ഥിരം ദശാംശം a = *(സ്ഥിരം ദശാംശം *)((സ്ഥിരം അക്ഷരം *)src0->data + x0*nb00 + y0*nb01 + i02*nb02 + i03*nb03);
                        സ്ഥിരം ദശാംശം b = *(സ്ഥിരം ദശാംശം *)((സ്ഥിരം അക്ഷരം *)src0->data + x1*nb00 + y0*nb01 + i02*nb02 + i03*nb03);
                        സ്ഥിരം ദശാംശം c = *(സ്ഥിരം ദശാംശം *)((സ്ഥിരം അക്ഷരം *)src0->data + x0*nb00 + y1*nb01 + i02*nb02 + i03*nb03);
                        സ്ഥിരം ദശാംശം d = *(സ്ഥിരം ദശാംശം *)((സ്ഥിരം അക്ഷരം *)src0->data + x1*nb00 + y1*nb01 + i02*nb02 + i03*nb03);

                        സ്ഥിരം ദശാംശം val = a*(1 - dx)*(1 - dy) + b*dx*(1 - dy) + c*(1 - dx)*dy + d*dx*dy;

                        ദശാംശം * y_dst = (ദശാംശം *)((അക്ഷരം *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);
                        *y_dst = val;
                    }
                }
            }
        }
    } അല്ലെങ്കിൽ എങ്കിൽ (mode == GGML_SCALE_MODE_BICUBIC) {
        // https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
        സ്ഥിരം ദശാംശം a = -0.75f; // use alpha = -0.75 (same as PyTorch)
        auto weight1 = [a](ദശാംശം x) { തിരികെ ((a + 2) * x - (a + 3)) * x * x + 1; };
        auto weight2 = [a](ദശാംശം x) { തിരികെ ((a * x - 5 * a) * x + 8 * a) * x - 4 * a; };
        auto bicubic = [=](ദശാംശം p0, ദശാംശം p1, ദശാംശം p2, ദശാംശം p3, ദശാംശം x) {
            സ്ഥിരം ദശാംശം w0 = weight2(x + 1);
            സ്ഥിരം ദശാംശം w1 = weight1(x + 0);
            സ്ഥിരം ദശാംശം w2 = weight1(1 - x);
            സ്ഥിരം ദശാംശം w3 = weight2(2 - x);
            തിരികെ p0*w0 + p1*w1 + p2*w2 + p3*w3;
        };

        വേണ്ടി (int64_t i3 = 0; i3 < ne3; i3++) {
            സ്ഥിരം int64_t i03 = i3 / sf3;
            വേണ്ടി (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                സ്ഥിരം int64_t i02 = i2 / sf2;
                വേണ്ടി (int64_t i1 = 0; i1 < ne1; i1++) {
                    സ്ഥിരം ദശാംശം y = ((ദശാംശം)i1 + pixel_offset) / sf1 - pixel_offset;
                    സ്ഥിരം int64_t y0 = (int64_t)floorf(y);
                    സ്ഥിരം ദശാംശം dy = y - (ദശാംശം)y0;

                    വേണ്ടി (int64_t i0 = 0; i0 < ne0; i0++) {
                        സ്ഥിരം ദശാംശം x = ((ദശാംശം)i0 + pixel_offset) / sf0 - pixel_offset;
                        സ്ഥിരം int64_t x0 = (int64_t)floorf(x);
                        സ്ഥിരം ദശാംശം dx = x - (ദശാംശം)x0;

                        auto p = [=](int64_t x_off, int64_t y_off) -> ദശാംശം {
                            int64_t i00 = std::max(int64_t(0), std::min(x0 + x_off, ne00 - 1));
                            int64_t i01 = std::max(int64_t(0), std::min(y0 + y_off, ne01 - 1));
                            തിരികെ *(സ്ഥിരം ദശാംശം *)((സ്ഥിരം അക്ഷരം *)src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                        };

                        സ്ഥിരം ദശാംശം val = bicubic(
                            bicubic(p(-1,-1), p(0,-1), p(1,-1), p(2,-1), dx),
                            bicubic(p(-1, 0), p(0, 0), p(1, 0), p(2, 0), dx),
                            bicubic(p(-1, 1), p(0, 1), p(1, 1), p(2, 1), dx),
                            bicubic(p(-1, 2), p(0, 2), p(1, 2), p(2, 2), dx), dy);

                        ദശാംശം * y_dst = (ദശാംശം *)((അക്ഷരം *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);
                        *y_dst = val;
                    }
                }
            }
        }
    } അല്ലെങ്കിൽ {
        GGML_ABORT("unsupported upscale mode");
    }
}

ശൂന്യം ggml_compute_forward_upscale(
    സ്ഥിരം ggml_compute_params * params,
    ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_upscale_f32(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}


// ggml_compute_forward_pad

template<തർക്കം circular_t>
സ്ഥിതം ശൂന്യം ggml_compute_forward_pad_f32(
    സ്ഥിരം ggml_compute_params * params,
          ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    പ്രതിജ്ഞാ(dst->nb[0] == വലുപ്പം(ദശാംശം));

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    ദശാംശം * dst_ptr = (ദശാംശം *) dst->data;
    സ്ഥിരം int32_t lp0 = ggml_get_op_params_i32(dst, 0);
    സ്ഥിരം int32_t rp0 = ggml_get_op_params_i32(dst, 1);
    സ്ഥിരം int32_t lp1 = ggml_get_op_params_i32(dst, 2);
    സ്ഥിരം int32_t rp1 = ggml_get_op_params_i32(dst, 3);
    സ്ഥിരം int32_t lp2 = ggml_get_op_params_i32(dst, 4);
    സ്ഥിരം int32_t rp2 = ggml_get_op_params_i32(dst, 5);
    സ്ഥിരം int32_t lp3 = ggml_get_op_params_i32(dst, 6);
    സ്ഥിരം int32_t rp3 = ggml_get_op_params_i32(dst, 7);

    സ്ഥിരം int64_t nrows = ne1 * ne2 * ne3;
    സ്ഥിരം തർക്കം src_row_contig = src0->nb[0] == വലുപ്പം(ദശാംശം);

    വേണ്ടി (int64_t row = ith; row < nrows; row += nth) {
        സ്ഥിരം int64_t i1 = row % ne1;
        സ്ഥിരം int64_t i2 = (row / ne1) % ne2;
        സ്ഥിരം int64_t i3 = row / (ne1 * ne2);

        ദശാംശം * dst_row = dst_ptr + row * ne0;

        എങ്കിൽ constexpr (circular_t) {
            // circular padding wraps around on a torus
            സ്ഥിരം int64_t src_i1 = ggml_wrap_around(i1 - lp1, ne01);
            സ്ഥിരം int64_t src_i2 = ggml_wrap_around(i2 - lp2, ne02);
            സ്ഥിരം int64_t src_i3 = ggml_wrap_around(i3 - lp3, ne03);

            വേണ്ടി (int64_t i0 = 0; i0 < ne0; ++i0) {
                സ്ഥിരം int64_t src_i0 = ggml_wrap_around(i0 - lp0, ne00);
                dst_row[i0] = *(സ്ഥിരം ദശാംശം *)((സ്ഥിരം അക്ഷരം *) src0->data + src_i3*nb03 + src_i2*nb02 + src_i1*nb01 + src_i0*nb00);
            }
        } അല്ലെങ്കിൽ {
            സ്ഥിരം തർക്കം interior = (i1 >= lp1 && i1 < ne1 - rp1)
                               && (i2 >= lp2 && i2 < ne2 - rp2)
                               && (i3 >= lp3 && i3 < ne3 - rp3);
            എങ്കിൽ (!interior) {
                memset(dst_row, 0, ne0 * വലുപ്പം(ദശാംശം));
                തുടരുക;
            }

            സ്ഥിരം അക്ഷരം * src_row = (സ്ഥിരം അക്ഷരം *) src0->data + (i3 - lp3)*nb03 + (i2 - lp2)*nb02 + (i1 - lp1)*nb01;

            എങ്കിൽ (src_row_contig && lp0 >= 0 && rp0 >= 0) {
                memset(dst_row, 0, lp0 * വലുപ്പം(ദശാംശം));
                memcpy(dst_row + lp0, src_row, ne00 * വലുപ്പം(ദശാംശം));
                memset(dst_row + lp0 + ne00, 0, rp0 * വലുപ്പം(ദശാംശം));
            } അല്ലെങ്കിൽ {
                വേണ്ടി (int64_t i0 = 0; i0 < ne0; ++i0) {
                    എങ്കിൽ (i0 >= lp0 && i0 < ne0 - rp0) {
                        dst_row[i0] = *(സ്ഥിരം ദശാംശം *)(src_row + (i0 - lp0)*nb00);
                    } അല്ലെങ്കിൽ {
                        dst_row[i0] = 0.0f;
                    }
                }
            }
        }
    }
}


ശൂന്യം ggml_compute_forward_pad(
    സ്ഥിരം ggml_compute_params * params,
    ggml_tensor * dst) {
    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം തർക്കം circular = (തർക്കം) ggml_get_op_params_i32(dst, 8);
    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                എങ്കിൽ (circular) {
                    ggml_compute_forward_pad_f32<സത്യം>(params, dst);
                } അല്ലെങ്കിൽ {
                    ggml_compute_forward_pad_f32<അസത്യം>(params, dst);
                }
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_pad_reflect_1d

ശൂന്യം ggml_compute_forward_pad_reflect_1d(
        സ്ഥിരം ggml_compute_params * params,
              ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം int32_t * opts = (സ്ഥിരം int32_t *) dst->op_params;
    സ്ഥിരം പൂർണ്ണസംഖ്യ p0 = opts[0];
    സ്ഥിരം പൂർണ്ണസംഖ്യ p1 = opts[1];

    GGML_TENSOR_UNARY_OP_LOCALS

    വേണ്ടി (int64_t i3 = 0; i3 < ne3; i3++) {
        വേണ്ടി (int64_t i2 = 0; i2 < ne2; i2++) {
            വേണ്ടി (int64_t i1 = ith; i1 < ne1; i1 += nth) {
                ദശാംശം * left  = (ദശാംശം *) ((അക്ഷരം *) dst->data + i3*nb3 + i2*nb2 + i1*nb1 +         p0*nb0);
                ദശാംശം * right = (ദശാംശം *) ((അക്ഷരം *) dst->data + i3*nb3 + i2*nb2 + i1*nb1 + (ne0-p1-1)*nb0);

                ggml_vec_cpy_f32(ne00, left, (ദശാംശം *) ((അക്ഷരം *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

                വേണ്ടി (പൂർണ്ണസംഖ്യ i0 = 1; i0 <= p0; i0++) { left[-i0] = left[i0];   }
                വേണ്ടി (പൂർണ്ണസംഖ്യ i0 = 1; i0 <= p1; i0++) { right[i0] = right[-i0]; }
            }
        }
    }
}

// ggml_compute_forward_roll

സ്ഥിതം int64_t ggml_wrap_index(int64_t i, int64_t ne) {
    എങ്കിൽ (i < 0) {
        തിരികെ i + ne;
    } അല്ലെങ്കിൽ എങ്കിൽ (i >= ne) {
        തിരികെ i - ne;
    }
    തിരികെ i;
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_roll_f32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ദശാംശം * src_data = (സ്ഥിരം ദശാംശം *) src0->data;
    ദശാംശം * dst_data = (ദശാംശം *) dst->data;

    GGML_TENSOR_UNARY_OP_LOCALS

    സ്ഥിരം പൂർണ്ണസംഖ്യ s0 = ggml_get_op_params_i32(dst, 0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ s1 = ggml_get_op_params_i32(dst, 1);
    സ്ഥിരം പൂർണ്ണസംഖ്യ s2 = ggml_get_op_params_i32(dst, 2);
    സ്ഥിരം പൂർണ്ണസംഖ്യ s3 = ggml_get_op_params_i32(dst, 3);

    സ്ഥിരം int64_t total = ne1 * ne2 * ne3;
    സ്ഥിരം int64_t per_thread = (total + params->nth) / params->nth;
    സ്ഥിരം int64_t start = params->ith * per_thread;
    സ്ഥിരം int64_t end   = std::min(start + per_thread, total);

    വേണ്ടി (int64_t i = start; i < end; ++i) {
        സ്ഥിരം int64_t i1 = i % ne1;
        സ്ഥിരം int64_t i2 = (i / ne1) % ne2;
        സ്ഥിരം int64_t i3 = i / (ne2 * ne1);
        ദശാംശം * dst_row = dst_data + (i3*nb3 + i2*nb2 + i1*nb1) / വലുപ്പം(ദശാംശം);

        സ്ഥിരം int64_t i01 = ggml_wrap_index(i1 - s1, ne01);
        സ്ഥിരം int64_t i02 = ggml_wrap_index(i2 - s2, ne02);
        സ്ഥിരം int64_t i03 = ggml_wrap_index(i3 - s3, ne03);
        സ്ഥിരം ദശാംശം * src_row = src_data + (i03*nb03 + i02*nb02 + i01*nb01) / വലുപ്പം(ദശാംശം);

        സ്ഥിരം int64_t s = ggml_wrap_index(-s0, ne00);
        സ്ഥിരം int64_t n = ne00 - s;
        ggml_vec_cpy_f32(n, dst_row,     src_row + s);
        ggml_vec_cpy_f32(s, dst_row + n, src_row);
    }
}

ശൂന്യം ggml_compute_forward_roll(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_roll_f32(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_arange

സ്ഥിതം ശൂന്യം ggml_compute_forward_arange_f32(
    സ്ഥിരം ggml_compute_params * params,
    ggml_tensor * dst) {

    GGML_ASSERT(dst->nb[0] == വലുപ്പം(ദശാംശം));

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം ദശാംശം start = ggml_get_op_params_f32(dst, 0);
    സ്ഥിരം ദശാംശം stop  = ggml_get_op_params_f32(dst, 1);
    സ്ഥിരം ദശാംശം step  = ggml_get_op_params_f32(dst, 2);

    സ്ഥിരം int64_t steps = (int64_t) ceilf((stop - start) / step);

    GGML_ASSERT(ggml_nelements(dst) == steps);

    വേണ്ടി (int64_t i = ith; i < steps; i+= nth) {
        ദശാംശം value = start + step * i;
        ((ദശാംശം *)dst->data)[i] = value;
    }
}

ശൂന്യം ggml_compute_forward_arange(
    സ്ഥിരം ggml_compute_params * params,
    ggml_tensor * dst) {
    തിരഞ്ഞെടുപ്പ് (dst->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_arange_f32(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_timestep_embedding_f32(
    സ്ഥിരം ggml_compute_params * params,
    ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->nb[0] == വലുപ്പം(ദശാംശം));

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    സ്ഥിരം പൂർണ്ണസംഖ്യ dim = ggml_get_op_params_i32(dst, 0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ max_period = ggml_get_op_params_i32(dst, 1);

    പൂർണ്ണസംഖ്യ half = dim / 2;

    വേണ്ടി (int64_t i = 0; i < ne00; i++) {
        ദശാംശം * embed_data = (ദശാംശം *)((അക്ഷരം *)  dst->data +  i*nb1);
        വേണ്ടി (int64_t j = ith; j < half; j += nth) {
            ദശാംശം timestep = ((ദശാംശം *)src0->data)[i];
            ദശാംശം freq = (ദശാംശം)expf(-logf(max_period) * j / half);
            ദശാംശം arg = timestep * freq;
            embed_data[j] = cosf(arg);
            embed_data[j + half] = sinf(arg);
        }
        എങ്കിൽ (dim % 2 != 0 && ith == 0) {
            embed_data[2 * half] = 0.f;
        }
    }
}

ശൂന്യം ggml_compute_forward_timestep_embedding(
    സ്ഥിരം ggml_compute_params * params,
    ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_timestep_embedding_f32(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_argsort

template<എണ്ണൽ ggml_sort_order order>
ഘടന cmp_argsort {
    സ്ഥിരം ദശാംശം * data;
    തർക്കം operator()(int32_t a, int32_t b) സ്ഥിരം {
        എങ്കിൽ constexpr (order == GGML_SORT_ORDER_ASC) {
            തിരികെ data[a] < data[b];
        } അല്ലെങ്കിൽ {
            തിരികെ data[a] > data[b];
        }
    }
};

സ്ഥിതം ശൂന്യം ggml_compute_forward_argsort_f32(
    സ്ഥിരം ggml_compute_params * params,
    ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(nb0 == വലുപ്പം(ദശാംശം));

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം int64_t nr = ggml_nrows(src0);

    ggml_sort_order order = (ggml_sort_order) ggml_get_op_params_i32(dst, 0);

    വേണ്ടി (int64_t i = ith; i < nr; i += nth) {
        സ്ഥിരം ദശാംശം * src_data = (ദശാംശം *)((അക്ഷരം *) src0->data + i*nb01);

        int32_t * dst_data = (int32_t *)((അക്ഷരം *) dst->data + i*nb1);

        വേണ്ടി (int64_t j = 0; j < ne0; j++) {
            dst_data[j] = j;
        }

        തിരഞ്ഞെടുപ്പ് (order) {
            അവസ്ഥ GGML_SORT_ORDER_ASC:
                std::sort(dst_data, dst_data + ne0, cmp_argsort<GGML_SORT_ORDER_ASC>{src_data});
                നിർത്തുക;

            അവസ്ഥ GGML_SORT_ORDER_DESC:
                std::sort(dst_data, dst_data + ne0, cmp_argsort<GGML_SORT_ORDER_DESC>{src_data});
                നിർത്തുക;

            സ്ഥിരസ്ഥിതി:
                GGML_ABORT("invalid sort order");
        }
    }
}

ശൂന്യം ggml_compute_forward_argsort(
    സ്ഥിരം ggml_compute_params * params,
    ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_argsort_f32(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_top_k

ഘടന cmp_മുകൾ_k {
    സ്ഥിരം ദശാംശം * data;
    തർക്കം operator()(int32_t a, int32_t b) സ്ഥിരം {
        തിരികെ data[a] > data[b];
    }
};

സ്ഥിതം ശൂന്യം ggml_compute_forward_മുകൾ_k_f32(
    സ്ഥിരം ggml_compute_params * params,
    ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(nb0 == വലുപ്പം(ദശാംശം));

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം int64_t nr = ggml_nrows(src0);

    സ്ഥിരം പൂർണ്ണസംഖ്യ മുകൾ_k = ne0;

    int32_t * tmp = (int32_t *) params->wdata + (ne00 + CACHE_വരി_SIZE_F32) * ith;

    വേണ്ടി (int64_t i = ith; i < nr; i += nth) {
        സ്ഥിരം ദശാംശം * src_data = (ദശാംശം *)((അക്ഷരം *) src0->data + i*nb01);

        വേണ്ടി (int64_t j = 0; j < ne00; j++) {
            tmp[j] = j;
        }

        std::partial_sort(tmp, tmp + മുകൾ_k, tmp + ne00, cmp_മുകൾ_k{src_data});

        int32_t * dst_data = (int32_t *)((അക്ഷരം *) dst->data + i*nb1);

        std::copy(tmp, tmp + മുകൾ_k, dst_data);

        // emphasize that the order is not important
        എങ്കിൽ (മുകൾ_k > 1) {
            std::swap(dst_data[0], dst_data[1]);
        }
    }
}

ശൂന്യം ggml_compute_forward_മുകൾ_k(
    സ്ഥിരം ggml_compute_params * params,
    ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_മുകൾ_k_f32(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_flash_attn_ext_f16_one_chunk(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst,
        പൂർണ്ണസംഖ്യ ir0, പൂർണ്ണസംഖ്യ ir1,
        int64_t ic_start, int64_t ic_end,
        ദശാംശം * partials, int64_t partial_stride) {

    സ്ഥിരം തർക്കം write_partials = (partials != nullptr);
    സ്ഥിരം ggml_tensor * q     = dst->src[0];
    സ്ഥിരം ggml_tensor * k     = dst->src[1];
    സ്ഥിരം ggml_tensor * v     = dst->src[2];
    സ്ഥിരം ggml_tensor * mask  = dst->src[3];
    സ്ഥിരം ggml_tensor * sinks = dst->src[4];

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    സ്ഥിരം int64_t DK = nek0;
    സ്ഥിരം int64_t DV = nev0;
    സ്ഥിരം int64_t N  = neq1;

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
    GGML_ASSERT(nb0 == വലുപ്പം(ദശാംശം));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    സ്ഥിരം int64_t rk2 = neq2/nek2;
    സ്ഥിരം int64_t rk3 = neq3/nek3;

    സ്ഥിരം int64_t rv2 = neq2/nev2;
    സ്ഥിരം int64_t rv3 = neq3/nev3;

    // parallelize by q rows using ggml_vec_dot_f32

    ദശാംശം scale         = 1.0f;
    ദശാംശം max_bias      = 0.0f;
    ദശാംശം logit_softcap = 0.0f;

    memcpy(&scale,         (ദശാംശം *) dst->op_params + 0, വലുപ്പം(ദശാംശം));
    memcpy(&max_bias,      (ദശാംശം *) dst->op_params + 1, വലുപ്പം(ദശാംശം));
    memcpy(&logit_softcap, (ദശാംശം *) dst->op_params + 2, വലുപ്പം(ദശാംശം));

    എങ്കിൽ (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    സ്ഥിരം uint32_t n_head      = neq2;
    സ്ഥിരം uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));

    സ്ഥിരം ദശാംശം m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    സ്ഥിരം ദശാംശം m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    ggml_type         സ്ഥിരം k_vec_dot_type = ggml_get_type_traits_cpu(k->type)->vec_dot_type;
    ggml_from_float_t സ്ഥിരം q_to_vec_dot   = ggml_get_type_traits_cpu(k_vec_dot_type)->from_float;
    ggml_vec_dot_t    സ്ഥിരം kq_vec_dot     = ggml_get_type_traits_cpu(k->type)->vec_dot;
    ggml_to_float_t   സ്ഥിരം v_to_float     = ggml_get_type_traits(v->type)->to_float;

    GGML_ASSERT((                            q_to_vec_dot) && "fattn: unsupported K-type");
    GGML_ASSERT((v->type == GGML_TYPE_F32 || v_to_float  ) && "fattn: unsupported V-type");

    പൂർണ്ണസംഖ്യ ith = params->ith;

    വേണ്ടി (പൂർണ്ണസംഖ്യ ir = ir0; ir < ir1; ++ir) {
        // q indices
        സ്ഥിരം പൂർണ്ണസംഖ്യ iq3 = ir/(neq2*neq1);
        സ്ഥിരം പൂർണ്ണസംഖ്യ iq2 = (ir - iq3*neq2*neq1)/neq1;
        സ്ഥിരം പൂർണ്ണസംഖ്യ iq1 = (ir - iq3*neq2*neq1 - iq2*neq1);

        സ്ഥിരം uint32_t h = iq2; // head index
        സ്ഥിരം ദശാംശം slope = (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1) : 1.0f;

        ദശാംശം S = 0.0f;      // sum
        ദശാംശം M = -INFINITY; // maximum KQ value

        ദശാംശം       * VKQ32 = (ദശാംശം       *) params->wdata + ith*(1*DK + 2*DV + CACHE_വരി_SIZE_F32); // FP32 VKQ accumulator
        ദശാംശം       * V32   =                 (VKQ32 + 1*DV); // (temporary) FP32 V buffer
        ggml_fp16_t * VKQ16 = (ggml_fp16_t *) (VKQ32 + 1*DV); // (temporary) FP16 VKQ accumulator
        ggml_fp16_t * Q_q   = (ggml_fp16_t *) (VKQ32 + 2*DV); // (temporary) buffer for Q converted to quantized/FP16

        എങ്കിൽ (v->type == GGML_TYPE_F16) {
            memset(VKQ16, 0, DV*വലുപ്പം(ggml_fp16_t));
        } അല്ലെങ്കിൽ {
            memset(VKQ32, 0, DV*വലുപ്പം(ദശാംശം));
        }

        സ്ഥിരം ggml_fp16_t * mp = mask ? (ggml_fp16_t *)((അക്ഷരം *) mask->data + iq1*mask->nb[1] + (iq2%mask->ne[2])*mask->nb[2] + (iq3%mask->ne[3])*mask->nb[3]) : ശൂന്യ;

        // k indices
        സ്ഥിരം പൂർണ്ണസംഖ്യ ik3 = iq3 / rk3;
        സ്ഥിരം പൂർണ്ണസംഖ്യ ik2 = iq2 / rk2;

        // v indices
        സ്ഥിരം പൂർണ്ണസംഖ്യ iv3 = iq3 / rv3;
        സ്ഥിരം പൂർണ്ണസംഖ്യ iv2 = iq2 / rv2;

        സ്ഥിരം ദശാംശം * pq = (സ്ഥിരം ദശാംശം *) ((അക്ഷരം *) q->data + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3));
        q_to_vec_dot(pq, Q_q, DK);

        // online softmax / attention
        // loop over n_kv and n_head_kv
        // ref: https://arxiv.org/pdf/2112.05682.pdf

        വേണ്ടി (int64_t ic = ic_start; ic < ic_end; ++ic) {
            സ്ഥിരം ദശാംശം mv = mp ? slope*GGML_CPU_FP16_TO_FP32(mp[ic]) : 0.0f;
            എങ്കിൽ (mv == -INFINITY) {
                തുടരുക;
            }

            ദശാംശം s; // KQ value

            സ്ഥിരം അക്ഷരം * k_data = (സ്ഥിരം അക്ഷരം *) k->data + ( ic*nbk1 + ik2*nbk2 + ik3*nbk3);
            kq_vec_dot(DK, &s, 0, k_data, 0, Q_q, 0, 1);

            s = s*scale; // scale KQ value

            എങ്കിൽ (logit_softcap != 0.0f) {
                s = logit_softcap*tanhf(s);
            }

            s += mv; // apply mask

            സ്ഥിരം ദശാംശം Mold = M;

            ദശാംശം ms = 1.0f; // upon new higher max val, scale VKQ and KQ sum with this value
            ദശാംശം vs = 1.0f; // post-softmax KQ value, expf(s - M)

            സ്ഥിരം അക്ഷരം * v_data = ((സ്ഥിരം അക്ഷരം *) v->data + (ic*nbv1 + iv2*nbv2 + iv3*nbv3));

            എങ്കിൽ (v->type == GGML_TYPE_F16) {
                എങ്കിൽ (s > M) {
                    // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
                    M = s;
                    ms = expf(Mold - M);

                    // V = V*expf(Mold - M)
                    ggml_vec_scale_f16(DV, VKQ16, ms);
                } അല്ലെങ്കിൽ {
                    // no new maximum, ms == 1.0f, vs != 1.0f
                    vs = expf(s - M);
                }

                // V += v*expf(s - M)
                ggml_vec_mad_f16(DV, VKQ16, (സ്ഥിരം ggml_fp16_t *) v_data, vs);
            } അല്ലെങ്കിൽ {
                എങ്കിൽ (s > M) {
                    // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
                    M = s;
                    ms = expf(Mold - M);

                    // V = V*expf(Mold - M)
                    ggml_vec_scale_f32(DV, VKQ32, ms);
                } അല്ലെങ്കിൽ {
                    // no new maximum, ms == 1.0f, vs != 1.0f
                    vs = expf(s - M);
                }

                // V += v*expf(s - M)
                എങ്കിൽ (v_to_float) {
                    v_to_float(v_data, V32, DV);
                    ggml_vec_mad_f32(DV, VKQ32, V32, vs);
                } അല്ലെങ്കിൽ {
                    // V is F32
                    ggml_vec_mad_f32(DV, VKQ32, (സ്ഥിരം ദശാംശം *) v_data, vs);
                }
            }

            S = S*ms + vs; // scale and increment sum with partial sum
        }

        എങ്കിൽ (v->type == GGML_TYPE_F16) {
            വേണ്ടി (int64_t d = 0; d < DV; ++d) {
                VKQ32[d] = GGML_CPU_FP16_TO_FP32(VKQ16[d]);
            }
        }

        // sinks - apply only on the first kv-chunk
        എങ്കിൽ (sinks && ic_start == 0) {
            സ്ഥിരം ദശാംശം s = ((ദശാംശം *)((അക്ഷരം *) sinks->data))[h];

            ദശാംശം ms = 1.0f;
            ദശാംശം vs = 1.0f;

            എങ്കിൽ (s > M) {
                ms = expf(M - s);
                M = s;
                ggml_vec_scale_f32(DV, VKQ32, ms);
            } അല്ലെങ്കിൽ {
                vs = expf(s - M);
            }

            S = S*ms + vs;
        }

        എങ്കിൽ (write_partials) {
            // Write M, S, VKQ to partials for later reduction
            // partials layout: [M, S, VKQ[DV]] per query head
            ദശാംശം * partial = partials + ir * partial_stride;
            partial[0] = M;
            partial[1] = S;
            memcpy(partial + 2, VKQ32, DV * വലുപ്പം(ദശാംശം));
        } അല്ലെങ്കിൽ {
            // V /= S
            സ്ഥിരം ദശാംശം S_inv = S == 0.0f ? 0.0f : 1.0f/S;
            ggml_vec_scale_f32(DV, VKQ32, S_inv);

            // dst indices
            സ്ഥിരം പൂർണ്ണസംഖ്യ i1 = iq1;
            സ്ഥിരം പൂർണ്ണസംഖ്യ i2 = iq2;
            സ്ഥിരം പൂർണ്ണസംഖ്യ i3 = iq3;

            // permute(0, 2, 1, 3)
            memcpy((അക്ഷരം *) dst->data + (i3*ne2*ne1 + i2 + i1*ne1)*nb1, VKQ32, nb1);
        }
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_flash_attn_ext_tiled(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst,
        പൂർണ്ണസംഖ്യ ir0, പൂർണ്ണസംഖ്യ ir1) {
    സ്ഥിരം ggml_tensor * q     = dst->src[0];
    സ്ഥിരം ggml_tensor * k     = dst->src[1];
    സ്ഥിരം ggml_tensor * v     = dst->src[2];
    സ്ഥിരം ggml_tensor * mask  = dst->src[3];
    സ്ഥിരം ggml_tensor * sinks = dst->src[4];

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    സ്ഥിരം int64_t DK = nek0;
    സ്ഥിരം int64_t DV = nev0;
    സ്ഥിരം int64_t N  = neq1;

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
    GGML_ASSERT(nb0 == വലുപ്പം(ദശാംശം));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(k->type == v->type);
    സ്ഥിരം ggml_type kv_type = k->type;


    // broadcast factors
    സ്ഥിരം int64_t rk2 = neq2/nek2;
    സ്ഥിരം int64_t rk3 = neq3/nek3;

    സ്ഥിരം int64_t rv2 = neq2/nev2;
    സ്ഥിരം int64_t rv3 = neq3/nev3;

    ദശാംശം scale         = 1.0f;
    ദശാംശം max_bias      = 0.0f;
    ദശാംശം logit_softcap = 0.0f;

    memcpy(&scale,         (ദശാംശം *) dst->op_params + 0, വലുപ്പം(ദശാംശം));
    memcpy(&max_bias,      (ദശാംശം *) dst->op_params + 1, വലുപ്പം(ദശാംശം));
    memcpy(&logit_softcap, (ദശാംശം *) dst->op_params + 2, വലുപ്പം(ദശാംശം));

    എങ്കിൽ (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    സ്ഥിരം uint32_t n_head      = neq2;
    സ്ഥിരം uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));

    സ്ഥിരം ദശാംശം m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    സ്ഥിരം ദശാംശം m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    പൂർണ്ണസംഖ്യ ith = params->ith;

    സ്ഥിതം constexpr പൂർണ്ണസംഖ്യ Q_TILE_SZ  = ggml_fa_tile_config::Q;
    സ്ഥിതം constexpr പൂർണ്ണസംഖ്യ KV_TILE_SZ = ggml_fa_tile_config::KV;

    പൂർണ്ണസംഖ്യ ir = ir0;
    ആകുമ്പോൾ (ir < ir1) {
        // q indices for the start of this tile
        സ്ഥിരം പൂർണ്ണസംഖ്യ iq3 = ir/(neq2*neq1);
        സ്ഥിരം പൂർണ്ണസംഖ്യ iq2 = (ir - iq3*neq2*neq1)/neq1;
        സ്ഥിരം പൂർണ്ണസംഖ്യ iq1 = (ir - iq3*neq2*neq1 - iq2*neq1);

        // Number of valid rows in this tile:
        // - limited by tile size (Q_TILE_SZ)
        // - limited by chunk boundary (ir1 - ir)
        // - limited by head boundary (neq1 - iq1) to avoid crossing into next head
        സ്ഥിരം പൂർണ്ണസംഖ്യ tile_rows = MIN(Q_TILE_SZ, MIN((പൂർണ്ണസംഖ്യ)(ir1 - ir), (പൂർണ്ണസംഖ്യ)(neq1 - iq1)));
        GGML_ASSERT(tile_rows > 0);

        സ്ഥിരം uint32_t h = iq2; // head index
        സ്ഥിരം ദശാംശം slope = (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1) : 1.0f;

        ദശാംശം S[Q_TILE_SZ];
        ദശാംശം M[Q_TILE_SZ];

        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0 ; i < Q_TILE_SZ; ++i) {
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
        ദശാംശം * base  = (ദശാംശം *) params->wdata + ith*(Q_TILE_SZ*DK + 2*Q_TILE_SZ*KV_TILE_SZ + Q_TILE_SZ*DV + KV_TILE_SZ*DV + KV_TILE_SZ*DK + CACHE_വരി_SIZE_F32);

        ശൂന്യം  * Q_q    = base;
        ദശാംശം * KQ     = (ദശാംശം *)((അക്ഷരം *)base + Q_TILE_SZ * DK * വലുപ്പം(ദശാംശം));
        ദശാംശം * mask32 = KQ + Q_TILE_SZ * KV_TILE_SZ;
        ദശാംശം * VKQ32  = mask32 + Q_TILE_SZ * KV_TILE_SZ;
        ദശാംശം * V32    = VKQ32 + Q_TILE_SZ * DV;
        ദശാംശം * K_f32  = V32 + KV_TILE_SZ * DV;

        memset(VKQ32, 0, Q_TILE_SZ * DV * വലുപ്പം(ദശാംശം));
        memset(mask32, 0, Q_TILE_SZ * KV_TILE_SZ * വലുപ്പം(ദശാംശം));

        // k indices
        സ്ഥിരം പൂർണ്ണസംഖ്യ ik3 = iq3 / rk3;
        സ്ഥിരം പൂർണ്ണസംഖ്യ ik2 = iq2 / rk2;

        // v indices
        സ്ഥിരം പൂർണ്ണസംഖ്യ iv3 = iq3 / rv3;
        സ്ഥിരം പൂർണ്ണസംഖ്യ iv2 = iq2 / rv2;

        {
            ദശാംശം * Q_f32 = (ദശാംശം *)Q_q;
            വേണ്ടി (പൂർണ്ണസംഖ്യ tq = 0; tq < tile_rows; tq++) {
                സ്ഥിരം ദശാംശം * pq = (സ്ഥിരം ദശാംശം *) ((അക്ഷരം *) q->data + ((iq1 + tq)*nbq1 + iq2*nbq2 + iq3*nbq3));
                memcpy(Q_f32 + tq * DK, pq, DK * വലുപ്പം(ദശാംശം));
            }
            വേണ്ടി (പൂർണ്ണസംഖ്യ tq = tile_rows; tq < Q_TILE_SZ; tq++) {
                memset(Q_f32 + tq * DK, 0, DK * വലുപ്പം(ദശാംശം));
            }
        }

        memset(K_f32, 0, DK * KV_TILE_SZ * വലുപ്പം(ദശാംശം));
        memset(V32,   0, KV_TILE_SZ * DV * വലുപ്പം(ദശാംശം));

        വേണ്ടി (int64_t ic = 0; ic < nek1; ic += KV_TILE_SZ) {
            സ്ഥിരം പൂർണ്ണസംഖ്യ kv_tile = (പൂർണ്ണസംഖ്യ)std::min((int64_t)KV_TILE_SZ, nek1 - ic);

            // skip the tile entirely if all the masks are -inf
            എങ്കിൽ (mask) {
                തർക്കം can_skip = സത്യം;
                വേണ്ടി (പൂർണ്ണസംഖ്യ tq = 0; tq < tile_rows; tq++) {
                    സ്ഥിരം ggml_fp16_t * mp_row = (സ്ഥിരം ggml_fp16_t *)((സ്ഥിരം അക്ഷരം *) mask->data + (iq1 + tq)*mask->nb[1] + (iq2%mask->ne[2])*mask->nb[2] + (iq3%mask->ne[3])*mask->nb[3]);
                    വേണ്ടി (പൂർണ്ണസംഖ്യ tk = 0; tk < kv_tile; tk++) {
                        mask32[tq * KV_TILE_SZ + tk] = slope * GGML_CPU_FP16_TO_FP32(mp_row[ic + tk]);
                        എങ്കിൽ (mask32[tq * KV_TILE_SZ + tk] != -INFINITY) {
                            can_skip = അസത്യം;
                        }
                    }
                    // Pad remaining mask entries with -inf
                    വേണ്ടി (പൂർണ്ണസംഖ്യ tk = kv_tile; tk < KV_TILE_SZ; tk++) {
                        mask32[tq * KV_TILE_SZ + tk] = -INFINITY;
                    }
                }

                എങ്കിൽ (can_skip) {
                    തുടരുക;
                }
            }

            // Pack K tile transposed: K_f32[dk][kv] so KV_TILE is contiguous (SIMD dim)
            // Zero-pad the last tile so the GEMM always operates on KV_TILE_SZ columns
            വേണ്ടി (പൂർണ്ണസംഖ്യ tk = 0; tk < kv_tile; tk++) {
                സ്ഥിരം അക്ഷരം * k_data = (സ്ഥിരം അക്ഷരം *)k->data + (ic + tk)*nbk1 + ik2*nbk2 + ik3*nbk3;
                എങ്കിൽ (kv_type == GGML_TYPE_F16) {
                    സ്ഥിരം ggml_fp16_t * k_f16 = (സ്ഥിരം ggml_fp16_t *)k_data;
                    വേണ്ടി (int64_t dk = 0; dk < DK; dk++) {
                        K_f32[dk * KV_TILE_SZ + tk] = GGML_CPU_FP16_TO_FP32(k_f16[dk]);
                    }
                } അല്ലെങ്കിൽ {
                    സ്ഥിരം ദശാംശം * k_f32_src = (സ്ഥിരം ദശാംശം *)k_data;
                    വേണ്ടി (int64_t dk = 0; dk < DK; dk++) {
                        K_f32[dk * KV_TILE_SZ + tk] = k_f32_src[dk];
                    }
                }
            }
            memset(KQ, 0, Q_TILE_SZ * KV_TILE_SZ * വലുപ്പം(ദശാംശം));
            simd_gemm(KQ, (സ്ഥിരം ദശാംശം *)Q_q, K_f32, Q_TILE_SZ, DK, KV_TILE_SZ);
            ggml_vec_scale_f32(Q_TILE_SZ * KV_TILE_SZ, KQ, scale);

            എങ്കിൽ (logit_softcap != 0.0f) {
                ggml_vec_tanh_f32(Q_TILE_SZ * KV_TILE_SZ, KQ, KQ);
                ggml_vec_scale_f32(Q_TILE_SZ * KV_TILE_SZ, KQ, logit_softcap);
            }

            // Set padded KQ entries to -inf so softmax gives them zero weight
            എങ്കിൽ (kv_tile < KV_TILE_SZ) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ tq = 0; tq < Q_TILE_SZ; tq++) {
                    വേണ്ടി (പൂർണ്ണസംഖ്യ tk = kv_tile; tk < KV_TILE_SZ; tk++) {
                        KQ[tq * KV_TILE_SZ + tk] = -INFINITY;
                    }
                }
            }

            എങ്കിൽ (mask) {
                ggml_vec_add_f32(tile_rows * KV_TILE_SZ, KQ, KQ, mask32);
            }

            തർക്കം skip[Q_TILE_SZ] = {};

            വേണ്ടി (പൂർണ്ണസംഖ്യ tq = 0; tq < Q_TILE_SZ; tq++) {
                ദശാംശം * kq_row = KQ + tq * KV_TILE_SZ;

                ദശാംശം tile_max;
                ggml_vec_max_f32(KV_TILE_SZ, &tile_max, kq_row);

                എങ്കിൽ (tile_max == -INFINITY) {
                    skip[tq] = സത്യം;
                    തുടരുക;
                }

                സ്ഥിരം ദശാംശം Mold = M[tq];
                സ്ഥിരം ദശാംശം Mnew = fmaxf(Mold, tile_max);

                എങ്കിൽ (Mnew > Mold) {
                    സ്ഥിരം ദശാംശം ms = expf(Mold - Mnew);
                    ggml_vec_scale_f32(DV, VKQ32 + tq * DV, ms);
                    S[tq] *= ms;
                }
                M[tq] = Mnew;


                S[tq] += ggml_vec_soft_max_f32(KV_TILE_SZ, kq_row, kq_row, Mnew);
            }

            // V accumulation: VKQ32 += softmax(KQ) * V
            // Pack V tile to contiguous F32, zero-padded
            വേണ്ടി (പൂർണ്ണസംഖ്യ tk = 0; tk < kv_tile; tk++) {
                സ്ഥിരം അക്ഷരം * v_data = (സ്ഥിരം അക്ഷരം *)v->data + (ic + tk)*nbv1 + iv2*nbv2 + iv3*nbv3;
                എങ്കിൽ (kv_type == GGML_TYPE_F16) {
                    ggml_cpu_fp16_to_fp32((സ്ഥിരം ggml_fp16_t *)v_data, V32 + tk * DV, DV);
                } അല്ലെങ്കിൽ {
                    memcpy(V32 + tk * DV, v_data, DV * വലുപ്പം(ദശാംശം));
                }
            }
            വേണ്ടി (പൂർണ്ണസംഖ്യ tq = 0; tq < Q_TILE_SZ; tq++) {
                എങ്കിൽ (skip[tq]) {
                    memset(KQ + tq * KV_TILE_SZ, 0, KV_TILE_SZ * വലുപ്പം(ദശാംശം));
                }
            }
            simd_gemm(VKQ32, KQ, V32, Q_TILE_SZ, KV_TILE_SZ, DV);
        }

        // sinks (apply only to valid rows in the tile)
        എങ്കിൽ (sinks) {
            സ്ഥിരം ദശാംശം s = ((ദശാംശം *)((അക്ഷരം *) sinks->data))[h];

            വേണ്ടി (പൂർണ്ണസംഖ്യ tq = 0; tq < tile_rows; tq++) {
                ദശാംശം ms = 1.0f;
                ദശാംശം vs = 1.0f;

                എങ്കിൽ (s > M[tq]) {
                    ms = expf(M[tq] - s);
                    ggml_vec_scale_f32(DV, VKQ32 + tq * DV, ms);
                } അല്ലെങ്കിൽ {
                    vs = expf(s - M[tq]);
                }

                S[tq] = S[tq] * ms + vs;
            }
        }

        വേണ്ടി (പൂർണ്ണസംഖ്യ tq = 0; tq < tile_rows; tq++) {
            // V /= S
            സ്ഥിരം ദശാംശം S_inv = S[tq] == 0.0f ? 0.0f : 1.0f / S[tq];
            ggml_vec_scale_f32(DV, VKQ32 + tq * DV, S_inv);

            // dst indices
            സ്ഥിരം പൂർണ്ണസംഖ്യ i1 = iq1 + tq;
            സ്ഥിരം പൂർണ്ണസംഖ്യ i2 = iq2;
            സ്ഥിരം പൂർണ്ണസംഖ്യ i3 = iq3;

            // permute(0, 2, 1, 3)
            memcpy((അക്ഷരം *) dst->data + (i3*ne2*ne1 + i2 + i1*ne1)*nb1, VKQ32 + tq * DV, nb1);
        }

        ir += tile_rows;
    }
}

// Reduction function: combines partial results across KV chunks
// Partials layout in wdata: [n_q_heads][n_chunks][2 + DV]
സ്ഥിതം ശൂന്യം ggml_flash_attn_ext_reduce_partials(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst,
        സ്ഥിരം int64_t n_chunks,
        സ്ഥിരം int64_t chunk_size) {

    സ്ഥിരം ggml_tensor * q = dst->src[0];
    സ്ഥിരം ggml_tensor * k = dst->src[1];
    സ്ഥിരം ggml_tensor * v = dst->src[2];

    സ്ഥിരം int64_t DK        = k->ne[0];
    സ്ഥിരം int64_t DV        = v->ne[0];
    സ്ഥിരം int64_t nek1      = k->ne[1];
    സ്ഥിരം int64_t n_q_heads = q->ne[2];

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം int64_t wdata_per_thread = DK + 2*DV + CACHE_വരി_SIZE_F32;
    ദശാംശം *       thread_wdata     = (ദശാംശം *) params->wdata + ith * wdata_per_thread;

    സ്ഥിരം int64_t partials_offset  = nth * (DK + 2*DV + CACHE_വരി_SIZE_F32);
    സ്ഥിരം int64_t partial_size     = 2 + DV;
    സ്ഥിരം ദശാംശം * partials_base    = (സ്ഥിരം ദശാംശം *) params->wdata + partials_offset;

    // Output layout
    സ്ഥിരം int64_t ne1 = dst->ne[1];
    സ്ഥിരം int64_t ne2 = dst->ne[2];
    സ്ഥിരം size_t  nb1 = dst->nb[1];

    // Each thread reduces a subset of query heads
    വേണ്ടി (int64_t q_head = ith; q_head < n_q_heads; q_head += nth) {
        ദശാംശം   M_final   = -INFINITY;
        ദശാംശം   S_final   = 0.0f;
        ദശാംശം * VKQ_final = thread_wdata;
        memset(VKQ_final, 0, DV * വലുപ്പം(ദശാംശം));

        // Combine partials from all chunks
        വേണ്ടി (int64_t chunk_idx = 0; chunk_idx < n_chunks; ++chunk_idx) {
            സ്ഥിരം int64_t ic_start = chunk_idx * chunk_size;
            എങ്കിൽ (ic_start >= nek1) തുടരുക;

            സ്ഥിരം ദശാംശം * partial   = partials_base + (q_head * n_chunks + chunk_idx) * partial_size;
            സ്ഥിരം ദശാംശം   M_chunk   = partial[0];
            സ്ഥിരം ദശാംശം   S_chunk   = partial[1];
            സ്ഥിരം ദശാംശം * VKQ_chunk = partial + 2;

            എങ്കിൽ (S_chunk == 0.0f) തുടരുക;

            സ്ഥിരം ദശാംശം M_new     = fmaxf(M_final, M_chunk);
            സ്ഥിരം ദശാംശം scale_old = expf(M_final - M_new);
            സ്ഥിരം ദശാംശം scale_new = expf(M_chunk - M_new);

            വേണ്ടി (int64_t d = 0; d < DV; ++d) {
                VKQ_final[d] = VKQ_final[d] * scale_old + VKQ_chunk[d] * scale_new;
            }
            S_final = S_final * scale_old + S_chunk * scale_new;
            M_final = M_new;
        }

        // Normalize and write to output
        എങ്കിൽ (S_final != 0.0f) {
            സ്ഥിരം ദശാംശം S_inv = 1.0f / S_final;
            ggml_vec_scale_f32(DV, VKQ_final, S_inv);
        }
        // iq1=0, iq3=0 for decode
        memcpy((അക്ഷരം *) dst->data + (0*ne2*ne1 + q_head + 0*ne1)*nb1, VKQ_final, nb1);
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_flash_attn_ext_f16(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * q     = dst->src[0];
    സ്ഥിരം ggml_tensor * k     = dst->src[1];
    സ്ഥിരം ggml_tensor * v     = dst->src[2];

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    സ്ഥിരം int64_t DK = nek0;
    സ്ഥിരം int64_t DV = nev0;
    സ്ഥിരം int64_t N  = neq1;


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
    GGML_ASSERT(nb0 == വലുപ്പം(ദശാംശം));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    // When use_ref is set, force the vec-only reference implementation (no tiling, no KV-chunking)
    സ്ഥിരം തർക്കം use_ref = params->use_ref;

    സ്ഥിരം തർക്കം kv_is_f32_or_f16 = (k->type == GGML_TYPE_F32 || k->type == GGML_TYPE_F16);
    സ്ഥിരം തർക്കം use_split_kv_path = !use_ref && (neq1 == 1 && neq3 == 1) && kv_is_f32_or_f16 && (k->type == v->type) && q->type == GGML_TYPE_F32 && nek1 >= 512;

    എങ്കിൽ (use_split_kv_path) {
        സ്ഥിരം int64_t chunk_size = (nek1 + nth - 1) / nth;

        // Partials buffer layout: [q_head][kv_chunk][M, S, VKQ]
        സ്ഥിരം int64_t partial_size  = 2 + DV;
        ദശാംശം *       partials_base = (ദശാംശം *) params->wdata + nth * (DK + 2*DV + CACHE_വരി_SIZE_F32);

        സ്ഥിരം int64_t ic_start = ith * chunk_size;
        സ്ഥിരം int64_t ic_end   = std::min(ic_start + chunk_size, nek1);

        സ്ഥിരം int64_t partial_stride = nth * partial_size;
        ദശാംശം *       chunk_partials = partials_base + ith * partial_size;

        എങ്കിൽ (ic_start < nek1) {
            വേണ്ടി (int64_t q_head = 0; q_head < neq2; q_head++) {
                ggml_compute_forward_flash_attn_ext_f16_one_chunk(
                    params, dst, q_head, q_head + 1, ic_start, ic_end,
                    chunk_partials, partial_stride);
            }
        } അല്ലെങ്കിൽ {
            വേണ്ടി (int64_t q_head = 0; q_head < neq2; q_head++) {
                ദശാംശം * q_partials = chunk_partials + q_head * partial_stride;
                q_partials[0] = -INFINITY;  // M
                q_partials[1] = 0.0f;       // S
            }
        }

        ggml_barrier(params->threadpool);
        ggml_flash_attn_ext_reduce_partials(params, dst, nth, chunk_size);
    } അല്ലെങ്കിൽ {

        // total rows in q
        സ്ഥിരം int64_t nr = neq1*neq2*neq3;

        // disable for NUMA
        സ്ഥിരം തർക്കം disable_chunking = ggml_is_numa();

        // 4x chunks per thread
        പൂർണ്ണസംഖ്യ nth_scaled = nth * 4;
        int64_t chunk_size = (nr + nth_scaled - 1) / nth_scaled;
        int64_t nchunk     = (nr + chunk_size - 1) / chunk_size;

        എങ്കിൽ (nth == 1 || nchunk < nth || disable_chunking) {
            nchunk = nth;
        }

        എങ്കിൽ (ith == 0) {
            ggml_threadpool_chunk_set(params->threadpool, nth);
        }

        ggml_barrier(params->threadpool);

        സ്ഥിരം int64_t dr = (nr + nchunk - 1) / nchunk;

        സ്ഥിതം constexpr int64_t Q_TILE_SZ  = ggml_fa_tile_config::Q;
        തർക്കം use_tiled = !use_ref &&
                               (q->type == GGML_TYPE_F32 &&
                                kv_is_f32_or_f16 &&
                                k->type == v->type &&
                                neq1 >= Q_TILE_SZ);
#എങ്കിൽ defined(GGML_SIMD) && !defined(__x86_64__) && !defined(_M_X64)
#എങ്കിൽ defined(__ARM_FEATURE_SVE)
        സ്ഥിരം int64_t f32_epr = svcntw();
#അല്ലെങ്കിൽ
        സ്ഥിരം int64_t f32_epr = GGML_F32_EPR;
#endif
        use_tiled &= (DV % f32_epr == 0);
#endif
        പൂർണ്ണസംഖ്യ current_chunk = ith;

        ആകുമ്പോൾ (current_chunk < nchunk) {
            സ്ഥിരം int64_t ir0 = dr * current_chunk;
            സ്ഥിരം int64_t ir1 = MIN(ir0 + dr, nr);

            എങ്കിൽ (use_tiled) {
                ggml_compute_forward_flash_attn_ext_tiled(params, dst, ir0, ir1);
            } അല്ലെങ്കിൽ {
                ggml_compute_forward_flash_attn_ext_f16_one_chunk(params, dst, ir0, ir1, 0, nek1, nullptr, 0);
            }

            current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
        }
    }
}

ശൂന്യം ggml_compute_forward_flash_attn_ext(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {
    തിരഞ്ഞെടുപ്പ് (dst->op_params[3]) {
        അവസ്ഥ GGML_PREC_DEFAULT:
        അവസ്ഥ GGML_PREC_F32:
            {
                // uses F32 accumulators
                ggml_compute_forward_flash_attn_ext_f16(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_flash_attn_back

സ്ഥിതം ശൂന്യം ggml_compute_forward_flash_attn_back_f32(
        സ്ഥിരം ggml_compute_params * params,
        സ്ഥിരം തർക്കം masked,
              ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * q = dst->src[0];
    സ്ഥിരം ggml_tensor * k = dst->src[1];
    സ്ഥിരം ggml_tensor * v = dst->src[2];
    സ്ഥിരം ggml_tensor * d = dst->src[3];

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

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം int64_t D = neq0;
    സ്ഥിരം int64_t N = neq1;
    സ്ഥിരം int64_t P = nek1 - N;
    സ്ഥിരം int64_t M = P + N;

    സ്ഥിരം പൂർണ്ണസംഖ്യ Mup  = ggml_up(M, GGML_SOFT_MAX_UNROLL);
    സ്ഥിരം പൂർണ്ണസംഖ്യ mxDM = MAX(D, Mup);

    // GGML_ASSERT(ne0 == D);
    // GGML_ASSERT(ne1 == N);
    GGML_ASSERT(P >= 0);

    GGML_ASSERT(nbq0 == വലുപ്പം(ദശാംശം));
    GGML_ASSERT(nbk0 == വലുപ്പം(ദശാംശം));
    GGML_ASSERT(nbv0 == വലുപ്പം(ദശാംശം));

    GGML_ASSERT(neq0 == D);
    GGML_ASSERT(nek0 == D);
    GGML_ASSERT(nev1 == D);
    GGML_ASSERT(ned0 == D);

    GGML_ASSERT(neq1 == N);
    GGML_ASSERT(nek1 == N + P);
    GGML_ASSERT(nev1 == D);
    GGML_ASSERT(ned1 == N);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == വലുപ്പം(ദശാംശം));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    എങ്കിൽ (ith == 0) {
        memset(dst->data, 0, nb0*ne0*ne1*ne2*ne3);
    }
    ggml_barrier(params->threadpool);

    സ്ഥിരം int64_t elem_q = ggml_nelements(q);
    സ്ഥിരം int64_t elem_k = ggml_nelements(k);

    ggml_type result_type = dst->type;
    GGML_ASSERT(ggml_blck_size(result_type) == 1);
    സ്ഥിരം size_t tsize = ggml_type_size(result_type);

    സ്ഥിരം size_t offs_q = 0;
    സ്ഥിരം size_t offs_k = offs_q + GGML_PAD(elem_q * tsize, GGML_MEM_ALIGN);
    സ്ഥിരം size_t offs_v = offs_k + GGML_PAD(elem_k * tsize, GGML_MEM_ALIGN);

    ശൂന്യം * grad_q = (അക്ഷരം *) dst->data;
    ശൂന്യം * grad_k = (അക്ഷരം *) dst->data + offs_k;
    ശൂന്യം * grad_v = (അക്ഷരം *) dst->data + offs_v;

    സ്ഥിരം size_t nbgq1 = nb0*neq0;
    സ്ഥിരം size_t nbgq2 = nb0*neq0*neq1;
    സ്ഥിരം size_t nbgq3 = nb0*neq0*neq1*neq2;

    സ്ഥിരം size_t nbgk1 = nb0*nek0;
    സ്ഥിരം size_t nbgk2 = nb0*nek0*nek1;
    സ്ഥിരം size_t nbgk3 = nb0*nek0*nek1*neq2;

    സ്ഥിരം size_t nbgv1 = nb0*nev0;
    സ്ഥിരം size_t nbgv2 = nb0*nev0*nev1;
    സ്ഥിരം size_t nbgv3 = nb0*nev0*nev1*neq2;

    // parallelize by k rows using ggml_vec_dot_f32

    // total rows in k
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr = nek2*nek3;

    // rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    സ്ഥിരം ദശാംശം scale = 1.0f/sqrtf(D);

    //printf("P=%d N=%d D=%d ir0=%d ir1=%d scale = %f\n", P, N, D, ir0, ir1, scale);

    // how often k2 (and v2) is repeated in q2
    പൂർണ്ണസംഖ്യ nrep = neq2/nek2;

    വേണ്ടി (പൂർണ്ണസംഖ്യ ir = ir0; ir < ir1; ++ir) {
        // q indices
        സ്ഥിരം പൂർണ്ണസംഖ്യ ik3 = ir/(nek2);
        സ്ഥിരം പൂർണ്ണസംഖ്യ ik2 = ir - ik3*nek2;

        സ്ഥിരം പൂർണ്ണസംഖ്യ iq3 = ik3;
        സ്ഥിരം പൂർണ്ണസംഖ്യ id3 = ik3;
        സ്ഥിരം പൂർണ്ണസംഖ്യ iv3 = ik3;
        സ്ഥിരം പൂർണ്ണസംഖ്യ iv2 = ik2;

        വേണ്ടി (പൂർണ്ണസംഖ്യ irep = 0; irep < nrep; ++irep) {
            സ്ഥിരം പൂർണ്ണസംഖ്യ iq2 = ik2 + irep*nek2;
            സ്ഥിരം പൂർണ്ണസംഖ്യ id2 = iq2;

            // (ik2 + irep*nek2) % nek2 == ik2
            വേണ്ടി (പൂർണ്ണസംഖ്യ iq1 = 0; iq1 < neq1; ++iq1) {
                സ്ഥിരം പൂർണ്ണസംഖ്യ id1 = iq1;

                // not sure about CACHE_LINE_SIZE_F32..
                // - maybe it must not be multiplied by 2 and excluded from .. in SM 1*(..) offset?
                ദശാംശം * S  = (ദശാംശം *) params->wdata + ith*2*(mxDM + CACHE_വരി_SIZE_F32) + 0*(mxDM+CACHE_വരി_SIZE_F32);
                ദശാംശം * SM = (ദശാംശം *) params->wdata + ith*2*(mxDM + CACHE_വരി_SIZE_F32) + 1*(mxDM+CACHE_വരി_SIZE_F32);

                വേണ്ടി (പൂർണ്ണസംഖ്യ i = M; i < Mup; ++i) {
                    S[i] = -INFINITY;
                }

                സ്ഥിരം int64_t masked_begin = masked ? (P + iq1 + 1) : M;
                വേണ്ടി (int64_t ic = 0; ic < masked_begin; ++ic) {
                    // k indices
                    സ്ഥിരം പൂർണ്ണസംഖ്യ ik1 = ic;

                    // S indices
                    സ്ഥിരം പൂർണ്ണസംഖ്യ i1 = ik1;

                    ggml_vec_dot_f32(neq0,
                            S + i1, 0,
                            (ദശാംശം *) ((അക്ഷരം *) k->data + (ik1*nbk1 + ik2*nbk2 + ik3*nbk3)), 0,
                            (ദശാംശം *) ((അക്ഷരം *) q->data + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3)), 0, 1);
                }

                // scale
                ggml_vec_scale_f32(masked_begin, S, scale);

                വേണ്ടി (int64_t i = masked_begin; i < M; i++) {
                    S[i] = -INFINITY;
                }

                // softmax
                // exclude known -INF S[..] values from max and loop
                // dont forget to set their SM values to zero
                {
                    ദശാംശം max = -INFINITY;
                    ggml_vec_max_f32(masked_begin, &max, S);

                    ggml_float sum = 0.0;
                    {
#ifdef GGML_SOFT_MAX_ACCELERATE
                        max = -max;
                        vDSP_vsadd(SM, 1, &max, SM, 1, Mup);
                        vvexpf(SM, SM, &Mup);
                        ggml_vec_sum_f32(Mup, &sum, SM);
#അല്ലെങ്കിൽ
                        sum = ggml_vec_soft_max_f32(Mup, SM, S, max);
#endif
                    }

                    പ്രതിജ്ഞാ(sum > 0.0);

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
                വേണ്ടി (int64_t ic = 0; ic < D; ++ic) {
                    ggml_vec_mad_f32(masked_begin,
                            S,
                             (ദശാംശം *) ((അക്ഷരം *) v->data + (          ic*nbv1  + iv2*nbv2 + iv3*nbv3)),
                            *(ദശാംശം *) ((അക്ഷരം *) d->data + (ic*nbd0 + id1*nbd1 + id2*nbd2 + id3*nbd3)));
                }

                // S = SM * (S - dot(SM, S))
                ദശാംശം dot_SM_gradSM = 0;
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
                വേണ്ടി (int64_t ic = 0; ic < masked_begin; ++ic) {
                    ggml_vec_mad_f32(D,
                            (ദശാംശം *) ((അക്ഷരം *) grad_q  + (iq1*nbgq1 + iq2*nbgq2  + iq3*nbgq3)),
                            (ദശാംശം *) ((അക്ഷരം *) k->data + (ic*nbk1   + ik2*nbk2   + ik3*nbk3)),
                            S[ic]);
                }

                // grad[k][:D,:M,iq2,iq3] += S.T @ qcur
                // for ic:
                //  grad[k][:D,ic,iq2,iq3] += S.T[0,ic] * qcur[:D,0]
                //  grad[k][:D,ic,iq2,iq3] += S[ic]     * qcur[:D,0]
                // exclude known zero S[..] values from loop
                വേണ്ടി (int64_t ic = 0; ic < masked_begin; ++ic) {
                    ggml_vec_mad_f32(D,
                            (ദശാംശം *) ((അക്ഷരം *) grad_k  + (ic*nbgk1  + ik2*nbgk2  + ik3*nbgk3)),
                            (ദശാംശം *) ((അക്ഷരം *) q->data + (iq1*nbq1  + iq2*nbq2   + iq3*nbq3)),
                            S[ic]);
                }

                // grad[v][:M,:D,iv2,iv3] += d[:D,id1,id2,id3].T       @ SM
                // for ic:
                //  grad[v][:M,ic,iv2,iv3] += d[:D,id1,id2,id3].T[0,ic] * SM[:M]
                //  grad[v][:M,ic,iv2,iv3] += d[ic,id1,id2,id3]         * SM[:M]
                // exclude known zero SM[..] values from mad
                വേണ്ടി (int64_t ic = 0; ic < D; ++ic) {
                    ggml_vec_mad_f32(masked_begin,
                            (ദശാംശം *) ((അക്ഷരം *) grad_v   + (          ic*nbgv1 + iv2*nbgv2 + iv3*nbgv3)),
                            SM,
                            *(ദശാംശം *) ((അക്ഷരം *) d->data + (ic*nbd0 + id1*nbd1 + id2*nbd2  + id3*nbd3)));
                }
            }
        }
    }
}

ശൂന്യം ggml_compute_forward_flash_attn_back(
        സ്ഥിരം ggml_compute_params * params,
        സ്ഥിരം തർക്കം masked,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * q = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (q->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_flash_attn_back_f32(params, masked, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_ssm_conv

സ്ഥിതം ശൂന്യം ggml_compute_forward_ssm_conv_f32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {
    സ്ഥിരം ggml_tensor * src0 = dst->src[0]; // conv_x
    സ്ഥിരം ggml_tensor * src1 = dst->src[1]; // conv1d.weight

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ nc  = src1->ne[0]; // d_conv
    സ്ഥിരം പൂർണ്ണസംഖ്യ ncs = src0->ne[0]; // d_conv - 1 + n_t
    സ്ഥിരം പൂർണ്ണസംഖ്യ nr  = src0->ne[1]; // d_inner
    സ്ഥിരം പൂർണ്ണസംഖ്യ n_t =  dst->ne[1]; // tokens per sequence
    സ്ഥിരം പൂർണ്ണസംഖ്യ n_s =  dst->ne[2]; // number of sequences in the batch

    GGML_ASSERT( dst->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == വലുപ്പം(ദശാംശം));
    GGML_ASSERT(src1->nb[0] == വലുപ്പം(ദശാംശം));
    GGML_ASSERT(src0->nb[1] == src0->ne[0]*വലുപ്പം(ദശാംശം));

    // rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir  = ir1 - ir0;

    വേണ്ടി (പൂർണ്ണസംഖ്യ i3 = 0; i3 < n_s; ++i3) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ i2 = 0; i2 < n_t; ++i2) {
            // {d_conv - 1 + n_t, d_inner, n_seqs}
            // sliding window
            സ്ഥിരം ദശാംശം * s = (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) src0->data + ir0*(src0->nb[1]) + i2*(src0->nb[0]) + i3*(src0->nb[2])); // {d_conv, d_inner, n_s}
            സ്ഥിരം ദശാംശം * c = (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) src1->data + ir0*(src1->nb[1])); // {d_conv, d_inner}
            ദശാംശം * x = (ദശാംശം *) ((അക്ഷരം *) dst->data + ir0*(dst->nb[0]) + i2*(dst->nb[1]) + i3*(dst->nb[2])); // {d_inner, n_t, n_s}

            // TODO: transpose the output for smaller strides for big batches?
            // d_inner
            വേണ്ടി (പൂർണ്ണസംഖ്യ i1 = 0; i1 < ir; ++i1) {
                // rowwise dot product
                // NOTE: not using ggml_vec_dot_f32, because its sum is in double precision
                ദശാംശം sumf = 0.0f;

                // d_conv
                വേണ്ടി (പൂർണ്ണസംഖ്യ i0 = 0; i0 < nc; ++i0) {
                    sumf += s[i0 + i1*ncs] * c[i0 + i1*nc];
                }
                x[i1] = sumf;
            }
        }
    }
}

ശൂന്യം ggml_compute_forward_ssm_conv(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {
    തിരഞ്ഞെടുപ്പ് (dst->src[0]->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_ssm_conv_f32(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_ssm_scan

സ്ഥിതം ശൂന്യം ggml_compute_forward_ssm_scan_f32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {
    സ്ഥിരം ggml_tensor * src0 = dst->src[0]; // s  {d_state, dim, n_head, n_seqs+}
    സ്ഥിരം ggml_tensor * src1 = dst->src[1]; // x  {dim, n_head, n_seq_tokens, n_seqs}
    സ്ഥിരം ggml_tensor * src2 = dst->src[2]; // dt {n_head, n_seq_tokens, n_seqs}
    സ്ഥിരം ggml_tensor * src3 = dst->src[3]; // A  {d_state, n_head} or {1, n_head}
    സ്ഥിരം ggml_tensor * src4 = dst->src[4]; // B  {d_state, n_group, n_seq_tokens, n_seqs}
    സ്ഥിരം ggml_tensor * src5 = dst->src[5]; // C  {d_state, n_group, n_seq_tokens, n_seqs}
    സ്ഥിരം ggml_tensor * src6 = dst->src[6]; // ids {n_seqs}

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം int64_t nc = src0->ne[0]; // d_state
    സ്ഥിരം int64_t nr = src0->ne[1]; // dim
    സ്ഥിരം int64_t nh = src1->ne[1]; // n_head
    സ്ഥിരം int64_t ng = src4->ne[1];
    സ്ഥിരം int64_t nt = src1->ne[2]; // number of tokens per sequence
    സ്ഥിരം int64_t ns = src1->ne[3]; // number of sequences in the batch
    സ്ഥിരം int64_t K  = ggml_get_op_params_i32(dst, 0);

    // can't use ggml_nbytes because src1 is not necessarily contiguous
    സ്ഥിരം int64_t s_off = ggml_nelements(src1) * ggml_element_size(src1);

    GGML_ASSERT(K >= 1);
    GGML_ASSERT(ggml_nelements(src1) + K*nc*nr*nh*ns == ggml_nelements(dst));
    GGML_ASSERT(src0->nb[0] == വലുപ്പം(ദശാംശം));
    GGML_ASSERT(src1->nb[0] == വലുപ്പം(ദശാംശം));
    GGML_ASSERT(src2->nb[0] == വലുപ്പം(ദശാംശം));
    GGML_ASSERT(src3->nb[0] == വലുപ്പം(ദശാംശം));
    GGML_ASSERT(src4->nb[0] == വലുപ്പം(ദശാംശം));
    GGML_ASSERT(src5->nb[0] == വലുപ്പം(ദശാംശം));
    GGML_ASSERT(src6->nb[0] == വലുപ്പം(int32_t));
    GGML_ASSERT(nh % ng == 0);
    GGML_ASSERT(src3->ne[0] == 1 || K == 1);

    // heads per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dh = (nh + nth - 1)/nth;

    // head range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ih0 = dh*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ih1 = MIN(ih0 + dh, nh);

    സ്ഥിരം int32_t * ids = (സ്ഥിരം int32_t *) src6->data;

    വേണ്ടി (പൂർണ്ണസംഖ്യ i3 = 0; i3 < ns; ++i3) {
        സ്ഥിരം ദശാംശം * s0 = (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) src0->data + ids[i3]*(src0->nb[3])); // {d_state, dim, nh, ns}
              ദശാംശം * s  = (      ദശാംശം *) ((      അക്ഷരം *) dst->data  + i3*(src0->nb[3]) + s_off); // {d_state, dim, nh, ns}

        വേണ്ടി (പൂർണ്ണസംഖ്യ i2 = 0; i2 < nt; ++i2) {
            സ്ഥിരം ദശാംശം * x  = (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) src1->data + i2*(src1->nb[2]) + i3*(src1->nb[3])); // {dim, nh, nt, ns}
            സ്ഥിരം ദശാംശം * dt = (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) src2->data + i2*(src2->nb[1]) + i3*(src2->nb[2])); // {nh, nt, ns}
            സ്ഥിരം ദശാംശം * A  = (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) src3->data); // {d_state, nh} or {1, nh}
            സ്ഥിരം ദശാംശം * B  = (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) src4->data + i2*(src4->nb[2]) + i3*(src4->nb[3])); // {d_state, ng, nt, ns}
            സ്ഥിരം ദശാംശം * C  = (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) src5->data + i2*(src5->nb[2]) + i3*(src5->nb[3])); // {d_state, ng, nt, ns}
                  ദശാംശം * y  = (      ദശാംശം *) ((      അക്ഷരം *) dst->data + i2*(nh*nr*വലുപ്പം(ദശാംശം)) + i3*(nt*nh*nr*വലുപ്പം(ദശാംശം))); // {dim, nh, nt, ns}

            എങ്കിൽ (src3->ne[0] == 1) {
                // Mamba-2 has a scalar decay factor per head; dA can be outside the state-wise loop

                // n_head
                വേണ്ടി (പൂർണ്ണസംഖ്യ h = ih0; h < ih1; ++h) {
                    // ref: https://github.com/state-spaces/mamba/blob/62db608da60f6fc790b8ed9f4b3225e95ca15fde/mamba_ssm/ops/triton/softplus.py#L16
                    സ്ഥിരം ദശാംശം dt_soft_plus = ggml_compute_softplus_f32(dt[h]);
                    സ്ഥിരം ദശാംശം dA = expf(dt_soft_plus * A[h]);
                    സ്ഥിരം പൂർണ്ണസംഖ്യ g = h / (nh / ng); // repeat_interleave

                    // dim
                    വേണ്ടി (പൂർണ്ണസംഖ്യ i1 = 0; i1 < nr; ++i1) {
                        സ്ഥിരം പൂർണ്ണസംഖ്യ ii = i1 + h*nr;
                        സ്ഥിരം ദശാംശം x_dt = x[ii] * dt_soft_plus;
                        ദശാംശം sumf = 0.0f;
#എങ്കിൽ defined(GGML_SIMD)
    #എങ്കിൽ defined(__ARM_FEATURE_SVE)
                        സ്ഥിരം പൂർണ്ണസംഖ്യ ggml_f32_epr = svcntw();
                        സ്ഥിരം പൂർണ്ണസംഖ്യ ggml_f32_step = 1 * ggml_f32_epr;

                        സ്ഥിരം പൂർണ്ണസംഖ്യ np = (nc & ~(ggml_f32_step - 1));

                        GGML_F32_VEC sum = GGML_F32_VEC_ZERO;

                        GGML_F32_VEC adA = GGML_F32_VEC_SET1(dA);
                        GGML_F32_VEC axdt = GGML_F32_VEC_SET1(x_dt);

                        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < np; i += ggml_f32_step) {
                            // TODO: maybe unroll more?
                            വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < 1; j++) {
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
    #അല്ലെങ്കിലെങ്കിൽ defined(__riscv_v_intrinsic)
                        // todo: RVV implementation
                        സ്ഥിരം പൂർണ്ണസംഖ്യ np = 0;
    #അല്ലെങ്കിൽ
                        സ്ഥിരം പൂർണ്ണസംഖ്യ np = (nc & ~(GGML_F32_STEP - 1));

                        GGML_F32_VEC sum[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };

                        GGML_F32_VEC adA = GGML_F32_VEC_SET1(dA);
                        GGML_F32_VEC axdt = GGML_F32_VEC_SET1(x_dt);

                        GGML_F32_VEC ax[GGML_F32_ARR];
                        GGML_F32_VEC ay[GGML_F32_ARR];
                        GGML_F32_VEC az[GGML_F32_ARR];

                        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < np; i += GGML_F32_STEP) {
                            വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < GGML_F32_ARR; j++) {
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
#അല്ലെങ്കിൽ
                        സ്ഥിരം പൂർണ്ണസംഖ്യ np = 0;
#endif
                        // d_state
                        വേണ്ടി (പൂർണ്ണസംഖ്യ i0 = np; i0 < nc; ++i0) {
                            സ്ഥിരം പൂർണ്ണസംഖ്യ i = i0 + ii*nc;
                            സ്ഥിരം പൂർണ്ണസംഖ്യ ig = i0 + g*nc;
                            // state = prev_state * dA + dB * x
                            സ്ഥിരം ദശാംശം state = (s0[i] * dA) + (B[ig] * x_dt);
                            // y = rowwise_dotprod(state, C)
                            sumf += state * C[ig];
                            s[i] = state;
                        }
                        y[ii] = sumf;
                    }
                }
            } അല്ലെങ്കിൽ {
                // Mamba-1 has an element-wise decay factor for the states

                // n_head
                വേണ്ടി (പൂർണ്ണസംഖ്യ h = ih0; h < ih1; ++h) {
                    // ref: https://github.com/state-spaces/mamba/blob/62db608da60f6fc790b8ed9f4b3225e95ca15fde/mamba_ssm/ops/triton/softplus.py#L16
                    സ്ഥിരം ദശാംശം dt_soft_plus = ggml_compute_softplus_f32(dt[h]);
                    സ്ഥിരം പൂർണ്ണസംഖ്യ g = h / (nh / ng); // repeat_interleave

                    // dim
                    വേണ്ടി (പൂർണ്ണസംഖ്യ i1 = 0; i1 < nr; ++i1) {
                        സ്ഥിരം പൂർണ്ണസംഖ്യ ii = i1 + h*nr;
                        സ്ഥിരം ദശാംശം x_dt = x[ii] * dt_soft_plus;
#എങ്കിൽ defined(__ARM_FEATURE_SVE)
                        svfloat32_t vx_dt = GGML_F32_VEC_SET1(x_dt);
                        svfloat32_t vdt_soft_plus = GGML_F32_VEC_SET1(dt_soft_plus);
                        svfloat32_t r1_vector = GGML_F32_VEC_ZERO;

                        // d_state
                        // TODO: what happens when (d_state % svcntw()) != 0?
                        വേണ്ടി (int64_t k = 0; k < nc; k += svcntw()) {
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
#അല്ലെങ്കിൽ
                        ദശാംശം sumf = 0.0f;
                        // NOTE: can't really use GGML_SIMD here because d_state is usually 16
                        //       and also because expf is used within the loop.
                        // d_state
                        വേണ്ടി (പൂർണ്ണസംഖ്യ i0 = 0; i0 < nc; ++i0) {
                            സ്ഥിരം പൂർണ്ണസംഖ്യ i = i0 + ii*nc;
                            സ്ഥിരം പൂർണ്ണസംഖ്യ ig = i0 + g*nc;
                            // state = prev_state * dA + dB * x
                            സ്ഥിരം ദശാംശം state = (s0[i] * expf(dt_soft_plus * A[i0 + h*nc])) + (B[ig] * x_dt);
                            // y = rowwise_dotprod(state, C)
                            sumf += state * C[ig];
                            s[i] = state;
                        }
                        y[ii] = sumf;
#endif
                    }
                }
            }
            സ്ഥിരം int64_t slot = nt - 1 - i2;
            എങ്കിൽ (K > 1 && slot > 0 && slot < K) {
                ദശാംശം * s_snapshot = (ദശാംശം *) ((അക്ഷരം *) dst->data + s_off + (slot*ns + i3)*(src0->nb[3]));
                വേണ്ടി (പൂർണ്ണസംഖ്യ h = ih0; h < ih1; ++h) {
                    memcpy((അക്ഷരം *) s_snapshot + h*src0->nb[2], (അക്ഷരം *) s + h*src0->nb[2], src0->nb[2]);
                }
            }
            // use the output as the source when it's not the first token-wise iteration
            s0 = s;
        }
    }
}

ശൂന്യം ggml_compute_forward_ssm_scan(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {
    തിരഞ്ഞെടുപ്പ് (dst->src[0]->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_ssm_scan_f32(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_win_part

സ്ഥിതം ശൂന്യം ggml_compute_forward_win_part_f32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {
    GGML_UNUSED(params);

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)

    സ്ഥിരം int32_t nep0 = ((സ്ഥിരം int32_t *)(dst->op_params))[0];
    സ്ഥിരം int32_t nep1 = ((സ്ഥിരം int32_t *)(dst->op_params))[1];
    സ്ഥിരം int32_t w    = ((സ്ഥിരം int32_t *)(dst->op_params))[2];

    പ്രതിജ്ഞാ(ne00 == ne0);
    പ്രതിജ്ഞാ(ne3  == nep0*nep1);

    // TODO: optimize / multi-thread
    വേണ്ടി (പൂർണ്ണസംഖ്യ py = 0; py < nep1; ++py) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ px = 0; px < nep0; ++px) {
            സ്ഥിരം int64_t i3 = py*nep0 + px;
            വേണ്ടി (int64_t i2 = 0; i2 < ne2; ++i2) {
                വേണ്ടി (int64_t i1 = 0; i1 < ne1; ++i1) {
                    വേണ്ടി (int64_t i0 = 0; i0 < ne0; ++i0) {
                        സ്ഥിരം int64_t i02 = py*w + i2;
                        സ്ഥിരം int64_t i01 = px*w + i1;
                        സ്ഥിരം int64_t i00 = i0;

                        സ്ഥിരം int64_t i = i3*ne2*ne1*ne0 + i2*ne1*ne0    + i1*ne0   + i0;
                        സ്ഥിരം int64_t j =                  i02*ne01*ne00 + i01*ne00 + i00;

                        എങ്കിൽ (py*w + i2 >= ne02 || px*w + i1 >= ne01) {
                            ((ദശാംശം *) dst->data)[i] = 0.0f;
                        } അല്ലെങ്കിൽ {
                            ((ദശാംശം *) dst->data)[i] = ((ദശാംശം *) src0->data)[j];
                        }
                    }
                }
            }
        }
    }
}

ശൂന്യം ggml_compute_forward_win_part(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_win_part_f32(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_win_unpart

സ്ഥിതം ശൂന്യം ggml_compute_forward_win_unpart_f32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {
    GGML_UNUSED(params);

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)

    സ്ഥിരം int32_t w = ((സ്ഥിരം int32_t *)(dst->op_params))[0];

    // padding
    സ്ഥിരം പൂർണ്ണസംഖ്യ px = (w - ne1%w)%w;
    //const int py = (w - ne2%w)%w;

    സ്ഥിരം പൂർണ്ണസംഖ്യ npx = (px + ne1)/w;
    //const int npy = (py + ne2)/w;

    പ്രതിജ്ഞാ(ne0 == ne00);

    // TODO: optimize / multi-thread
    വേണ്ടി (int64_t i2 = 0; i2 < ne2; ++i2) {
        വേണ്ടി (int64_t i1 = 0; i1 < ne1; ++i1) {
            വേണ്ടി (int64_t i0 = 0; i0 < ne0; ++i0) {
                സ്ഥിരം പൂർണ്ണസംഖ്യ ip2 = i2/w;
                സ്ഥിരം പൂർണ്ണസംഖ്യ ip1 = i1/w;

                സ്ഥിരം int64_t i02 = i2%w;
                സ്ഥിരം int64_t i01 = i1%w;
                സ്ഥിരം int64_t i00 = i0;

                സ്ഥിരം int64_t i = (ip2*npx + ip1)*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00 + i00;
                സ്ഥിരം int64_t j =                                  i2*ne1*ne0    + i1*ne0   + i0;

                ((ദശാംശം *) dst->data)[j] = ((ദശാംശം *) src0->data)[i];
            }
        }
    }
}

ശൂന്യം ggml_compute_forward_win_unpart(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_win_unpart_f32(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

//ggml_compute_forward_unary

ശൂന്യം ggml_compute_forward_unary(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_unary_op op = ggml_get_unary_op(dst);

    തിരഞ്ഞെടുപ്പ് (op) {
        അവസ്ഥ GGML_UNARY_OP_ABS:
            {
                ggml_compute_forward_abs(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_UNARY_OP_SGN:
            {
                ggml_compute_forward_sgn(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_UNARY_OP_NEG:
            {
                ggml_compute_forward_neg(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_UNARY_OP_STEP:
            {
                ggml_compute_forward_step(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_UNARY_OP_TANH:
            {
                ggml_compute_forward_tanh(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_UNARY_OP_ELU:
            {
                ggml_compute_forward_elu(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_UNARY_OP_RELU:
            {
                ggml_compute_forward_relu(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_UNARY_OP_SIGMOID:
            {
                ggml_compute_forward_sigmoid(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_UNARY_OP_GELU:
            {
                ggml_compute_forward_gelu(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_UNARY_OP_GELU_ERF:
            {
                ggml_compute_forward_gelu_erf(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_UNARY_OP_GELU_QUICK:
            {
                ggml_compute_forward_gelu_quick(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_UNARY_OP_SILU:
            {
                ggml_compute_forward_silu(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_UNARY_OP_HARDSWISH:
            {
                ggml_compute_forward_hardswish(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_UNARY_OP_HARDSIGMOID:
            {
                ggml_compute_forward_hardsigmoid(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_UNARY_OP_EXP:
            {
                ggml_compute_forward_exp(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_UNARY_OP_FLOOR:
            {
                ggml_compute_forward_floor(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_UNARY_OP_CEIL:
            {
                ggml_compute_forward_ceil(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_UNARY_OP_ROUND:
            {
                ggml_compute_forward_round(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_UNARY_OP_TRUNC:
            {
                ggml_compute_forward_trunc(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_UNARY_OP_XIELU:
            {
                ggml_compute_forward_xielu(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_UNARY_OP_EXPM1:
            {
                ggml_compute_forward_expm1(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_UNARY_OP_SOFTPLUS:
            {
                ggml_compute_forward_softplus(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

//ggml_compute_forward_glu

ശൂന്യം ggml_compute_forward_glu(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_glu_op op = ggml_get_glu_op(dst);

    തിരഞ്ഞെടുപ്പ് (op) {
        അവസ്ഥ GGML_GLU_OP_REGLU:
            {
                ggml_compute_forward_reglu(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_GLU_OP_GEGLU:
            {
                ggml_compute_forward_geglu(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_GLU_OP_SWIGLU:
            {
                ggml_compute_forward_swiglu(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_GLU_OP_SWIGLU_OAI:
            {
                ggml_compute_forward_swiglu_oai(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_GLU_OP_GEGLU_ERF:
            {
                ggml_compute_forward_geglu_erf(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_GLU_OP_GEGLU_QUICK:
            {
                ggml_compute_forward_geglu_quick(params, dst);
            } നിർത്തുക;
        അവസ്ഥ GGML_GLU_OP_SWIGLU_CLAMP:
            {
                ggml_compute_forward_swiglu_clamp(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_get_rel_pos

സ്ഥിതം ശൂന്യം ggml_compute_forward_get_rel_pos_f16(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {
    GGML_UNUSED(params);

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L292-L322

    GGML_TENSOR_UNARY_OP_LOCALS

    സ്ഥിരം int64_t w = ne1;

    ggml_fp16_t * src0_data = (ggml_fp16_t *) src0->data;
    ggml_fp16_t * dst_data  = (ggml_fp16_t *) dst->data;

    വേണ്ടി (int64_t i2 = 0; i2 < ne2; ++i2) {
        വേണ്ടി (int64_t i1 = 0; i1 < ne1; ++i1) {
            സ്ഥിരം int64_t pos = (w - i1 - 1) + i2;
            വേണ്ടി (int64_t i0 = 0; i0 < ne0; ++i0) {
                dst_data[i2*ne1*ne0 + i1*ne0 + i0] = src0_data[pos*ne00 + i0];
            }
        }
    }
}

ശൂന്യം ggml_compute_forward_get_rel_pos(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F16:
        അവസ്ഥ GGML_TYPE_BF16:
            {
                ggml_compute_forward_get_rel_pos_f16(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_add_rel_pos

സ്ഥിതം ശൂന്യം ggml_compute_forward_add_rel_pos_f32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];
    സ്ഥിരം ggml_tensor * src2 = dst->src[2];

    സ്ഥിരം തർക്കം inplace = (തർക്കം) ((int32_t *) dst->op_params)[0];
    എങ്കിൽ (!inplace) {
        എങ്കിൽ (params->ith == 0) {
            memcpy((അക്ഷരം *) dst->data, (അക്ഷരം *) src0->data, ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }
    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L357-L359

    ദശാംശം * src1_data = (ദശാംശം *) src1->data;
    ദശാംശം * src2_data = (ദശാംശം *) src2->data;
    ദശാംശം * dst_data  = (ദശാംശം *) dst->data;

    സ്ഥിരം int64_t ne10 = src1->ne[0];
    സ്ഥിരം int64_t ne11 = src1->ne[1];
    സ്ഥിരം int64_t ne12 = src1->ne[2];
    സ്ഥിരം int64_t ne13 = src1->ne[3];

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    // total patches in dst
    സ്ഥിരം പൂർണ്ണസംഖ്യ np = ne13;

    // patches per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dp = (np + nth - 1)/nth;

    // patch range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ip0 = dp*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ip1 = MIN(ip0 + dp, np);

    വേണ്ടി (int64_t i13 = ip0; i13 < ip1; ++i13) {
        വേണ്ടി (int64_t i12 = 0; i12 < ne12; ++i12) {
            വേണ്ടി (int64_t i11 = 0; i11 < ne11; ++i11) {
                സ്ഥിരം int64_t jp1 = i13*ne12*ne11*ne10 + i12*ne11*ne10 + i11*ne10;
                വേണ്ടി (int64_t i10 = 0; i10 < ne10; ++i10) {
                    സ്ഥിരം int64_t jp0  = jp1 + i10;
                    സ്ഥിരം ദശാംശം src1_e = src1_data[jp0];
                    സ്ഥിരം ദശാംശം src2_e = src2_data[jp0];

                    സ്ഥിരം int64_t jdh = jp0 * ne10;
                    സ്ഥിരം int64_t jdw = jdh - (ne10 - 1) * i10;

                    വേണ്ടി (int64_t j = 0; j < ne10; ++j) {
                        dst_data[jdh + j     ] += src2_e;
                        dst_data[jdw + j*ne10] += src1_e;
                    }
                }
            }
        }
    }
}

ശൂന്യം ggml_compute_forward_add_rel_pos(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_add_rel_pos_f32(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_rwkv_wkv6

സ്ഥിതം ശൂന്യം ggml_compute_forward_rwkv_wkv6_f32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {
    സ്ഥിരം int64_t T = dst->src[1]->ne[2];
    സ്ഥിരം int64_t C = dst->ne[0];
    സ്ഥിരം int64_t HEADS = dst->src[1]->ne[1];
    സ്ഥിരം int64_t n_seqs = dst->src[5]->ne[1];
    സ്ഥിരം int64_t head_size = C / HEADS;

    ദശാംശം * dst_data = (ദശാംശം *) dst->data;
    ദശാംശം * state = ((ദശാംശം *) dst->data) + C * T;

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ h_start =  (HEADS * (ith    )) / nth;
    സ്ഥിരം പൂർണ്ണസംഖ്യ h_end   = ((HEADS * (ith + 1)) / nth < HEADS) ?
                         (HEADS * (ith + 1)) / nth : HEADS;

    ദശാംശം * k =          (ദശാംശം *) dst->src[0]->data;
    ദശാംശം * v =          (ദശാംശം *) dst->src[1]->data;
    ദശാംശം * r =          (ദശാംശം *) dst->src[2]->data;
    ദശാംശം * time_faaaa = (ദശാംശം *) dst->src[3]->data;
    ദശാംശം * time_decay = (ദശാംശം *) dst->src[4]->data;

    size_t t_stride = HEADS * head_size; // Same to C

    size_t h_stride = C / HEADS;
    GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
    size_t h_stride_2d = head_size * head_size;

    എങ്കിൽ (ith == 0) {
        memset(dst_data, 0, T * C * വലുപ്പം(ദശാംശം));
    }
    ggml_barrier(params->threadpool);


    #എങ്കിൽ defined(__AVX__) && !defined(__AVX512F__)
        #define GGML_F32X GGML_F32x8
        #define GGML_F32X_SET1 GGML_F32x8_SET1
        #define GGML_F32X_LOAD GGML_F32x8_LOAD
        #define GGML_F32X_STORE GGML_F32x8_STORE
        #define GGML_F32X_MUL GGML_F32x8_MUL
        #define GGML_F32X_FMA GGML_F32x8_FMA
        #define WKV_VECTOR_SIZE 8
    #അല്ലെങ്കിലെങ്കിൽ defined(__AVX512F__)
        #define GGML_F32X GGML_F32x16
        #define GGML_F32X_SET1 GGML_F32x16_SET1
        #define GGML_F32X_LOAD GGML_F32x16_LOAD
        #define GGML_F32X_STORE GGML_F32x16_STORE
        #define GGML_F32X_MUL GGML_F32x16_MUL
        #define GGML_F32X_FMA GGML_F32x16_FMA
        #define WKV_VECTOR_SIZE 16
    #അല്ലെങ്കിലെങ്കിൽ defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
        #define GGML_F32X GGML_F32xt
        #define GGML_F32X_SET1 GGML_F32xt_SET1
        #define GGML_F32X_LOAD GGML_F32xt_LOAD
        #define GGML_F32X_STORE GGML_F32xt_STORE
        #define GGML_F32X_MUL GGML_F32xt_MUL
        #define GGML_F32X_FMA GGML_F32xt_FMA
        #define WKV_VECTOR_SIZE 8
    #അല്ലെങ്കിലെങ്കിൽ defined(__ARM_NEON) && defined(__aarch64__)
        #define GGML_F32X GGML_F32x4
        #define GGML_F32X_SET1 GGML_F32x4_SET1
        #define GGML_F32X_LOAD GGML_F32x4_LOAD
        #define GGML_F32X_STORE GGML_F32x4_STORE
        #define GGML_F32X_MUL GGML_F32x4_MUL
        #define GGML_F32X_FMA GGML_F32x4_FMA
        #define WKV_VECTOR_SIZE 4
    #endif

    #ifdef WKV_VECTOR_SIZE
        പൂർണ്ണസംഖ്യ wkv_vector_size;
        #എങ്കിൽ defined(__ARM_FEATURE_SVE)
            wkv_vector_size = svcntw();
        #അല്ലെങ്കിൽ
            wkv_vector_size = WKV_VECTOR_SIZE;
        #endif
        സ്ഥിരം int64_t vec_എണ്ണം = head_size / wkv_vector_size;

        വേണ്ടി (int64_t t = 0; t < T; t++) {
            size_t t_offset = t * t_stride;
            size_t state_offset = head_size * C * (t / (T / n_seqs));
            ദശാംശം * state_cur = state + state_offset;
            ദശാംശം * state_prev = t % (T / n_seqs) ? state_cur : (ദശാംശം*)dst->src[5]->data + state_offset;

            വേണ്ടി (int64_t h = h_start; h < h_end; h++) {
                size_t h_offset = h * h_stride;
                size_t t_h_offset = t_offset + h_offset;
                size_t h_2d_offset = h * h_stride_2d;

                വേണ്ടി (int64_t i = 0; i < head_size; i++) {
                    size_t t_h_i_offset = t_h_offset + i;
                    size_t h_i_offset = h_offset + i;
                    size_t h_2d_i_offset = h_2d_offset + i * h_stride;

                    ദശാംശം k_val = k[t_h_i_offset];
                    ദശാംശം r_val = r[t_h_i_offset];
                    ദശാംശം time_faaaa_val = time_faaaa[h_i_offset];
                    ദശാംശം time_decay_val = time_decay[t_h_i_offset];

                    // Broadcast scalar values to vectors
                    GGML_F32X k_vec = GGML_F32X_SET1(k_val);
                    GGML_F32X r_vec = GGML_F32X_SET1(r_val);
                    GGML_F32X time_faaaa_vec = GGML_F32X_SET1(time_faaaa_val);
                    GGML_F32X time_decay_vec = GGML_F32X_SET1(time_decay_val);

                    വേണ്ടി (int64_t j = 0; j < vec_എണ്ണം; j++) {
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
                    വേണ്ടി (int64_t j = vec_എണ്ണം * wkv_vector_size; j < head_size; j++) {
                        size_t t_h_j_offset = t_h_offset + j;
                        size_t h_2d_i_j_offset = h_2d_i_offset + j;
                        ദശാംശം v_val = v[t_h_j_offset];
                        ദശാംശം kv_val = v_val * k_val;
                        ദശാംശം prev_state_val = state_prev[h_2d_i_j_offset];
                        ദശാംശം temp_val = kv_val * time_faaaa_val + prev_state_val;
                        dst_data[t_h_j_offset] += temp_val * r_val;
                        state_cur[h_2d_i_j_offset] = prev_state_val * time_decay_val + kv_val;
                    }
                }
            }
        }

    #അല്ലെങ്കിൽ
        // basically fused operations:
        // dst = r @ (time_faaaa * (k @ v) + state),
        // state = time_decay * state + (k @ v),
        // recursive through each token
        വേണ്ടി (int64_t t = 0; t < T; t++) {
            size_t t_offset = t * t_stride;
            size_t state_offset = head_size * C * (t / (T / n_seqs));
            ദശാംശം * state_cur = state + state_offset;
            ദശാംശം * state_prev = t % (T / n_seqs) ? state_cur : (ദശാംശം*)dst->src[5]->data + state_offset;

            വേണ്ടി (int64_t h = h_start; h < h_end; h++) {
                size_t h_offset = h * h_stride;
                size_t t_h_offset = t_offset + h_offset;
                size_t h_2d_offset = h * h_stride_2d;

                വേണ്ടി (int64_t i = 0; i < head_size; i++) {
                    size_t t_h_i_offset = t_h_offset + i;
                    size_t h_i_offset = h_offset + i;
                    size_t h_2d_i_offset = h_2d_offset + i * h_stride;

                    ദശാംശം k_val = k[t_h_i_offset];
                    ദശാംശം r_val = r[t_h_i_offset];
                    ദശാംശം time_faaaa_val = time_faaaa[h_i_offset];
                    // RWKV v6: different time_decay for each token.
                    ദശാംശം time_decay_val = time_decay[t_h_i_offset];

                    വേണ്ടി (int64_t j = 0; j < head_size; j++) {
                        size_t t_h_j_offset = t_h_offset + j;
                        size_t h_2d_i_j_offset = h_2d_i_offset + j;

                        ദശാംശം v_val = v[t_h_j_offset];
                        ദശാംശം kv_val = v_val * k_val;
                        ദശാംശം prev_state_val = state_prev[h_2d_i_j_offset];
                        ദശാംശം temp_val = kv_val * time_faaaa_val + prev_state_val;
                        dst_data[t_h_j_offset] += temp_val * r_val;
                        state_cur[h_2d_i_j_offset] = prev_state_val * time_decay_val + kv_val;
                    }
                }
            }
        }
    #endif
}


ശൂന്യം ggml_compute_forward_rwkv_wkv6(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_rwkv_wkv6_f32(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_gla

സ്ഥിതം ശൂന്യം ggml_compute_forward_gla_f32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {
    സ്ഥിരം int64_t T = dst->src[1]->ne[2];
    സ്ഥിരം int64_t C = dst->ne[0];
    സ്ഥിരം int64_t HEADS = dst->src[1]->ne[1];
    സ്ഥിരം int64_t n_seqs = dst->src[4]->ne[1];
    സ്ഥിരം int64_t head_size = C / HEADS;
    സ്ഥിരം ദശാംശം scale = ggml_get_op_params_f32(dst, 0);

    ദശാംശം * dst_data = (ദശാംശം *) dst->data;
    ദശാംശം * state = ((ദശാംശം *) dst->data) + C * T;

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ h_start =  (HEADS * (ith    )) / nth;
    സ്ഥിരം പൂർണ്ണസംഖ്യ h_end   = ((HEADS * (ith + 1)) / nth < HEADS) ?
                         (HEADS * (ith + 1)) / nth : HEADS;

    ദശാംശം * k = (ദശാംശം *) dst->src[0]->data;
    ദശാംശം * v = (ദശാംശം *) dst->src[1]->data;
    ദശാംശം * q = (ദശാംശം *) dst->src[2]->data;
    ദശാംശം * g = (ദശാംശം *) dst->src[3]->data;

    size_t t_stride = HEADS * head_size; // Same to C

    size_t h_stride = C / HEADS;
    GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
    size_t h_stride_2d = head_size * head_size;

    എങ്കിൽ (ith == 0) {
        memset(dst_data, 0, T * C * വലുപ്പം(ദശാംശം));
    }
    ggml_barrier(params->threadpool);


    #എങ്കിൽ defined(__AVX__) && !defined(__AVX512F__)
        #define GGML_F32X GGML_F32x8
        #define GGML_F32X_SET1 GGML_F32x8_SET1
        #define GGML_F32X_LOAD GGML_F32x8_LOAD
        #define GGML_F32X_STORE GGML_F32x8_STORE
        #define GGML_F32X_MUL GGML_F32x8_MUL
        #define GGML_F32X_FMA GGML_F32x8_FMA
        #define GLA_VECTOR_SIZE 8
    #അല്ലെങ്കിലെങ്കിൽ defined(__AVX512F__)
        #define GGML_F32X GGML_F32x16
        #define GGML_F32X_SET1 GGML_F32x16_SET1
        #define GGML_F32X_LOAD GGML_F32x16_LOAD
        #define GGML_F32X_STORE GGML_F32x16_STORE
        #define GGML_F32X_MUL GGML_F32x16_MUL
        #define GGML_F32X_FMA GGML_F32x16_FMA
        #define GLA_VECTOR_SIZE 16
    #അല്ലെങ്കിലെങ്കിൽ defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
        #define GGML_F32X GGML_F32xt
        #define GGML_F32X_SET1 GGML_F32xt_SET1
        #define GGML_F32X_LOAD GGML_F32xt_LOAD
        #define GGML_F32X_STORE GGML_F32xt_STORE
        #define GGML_F32X_MUL GGML_F32xt_MUL
        #define GGML_F32X_FMA GGML_F32xt_FMA
        #define GLA_VECTOR_SIZE 8
    #അല്ലെങ്കിലെങ്കിൽ defined(__ARM_NEON) && defined(__aarch64__)
        #define GGML_F32X GGML_F32x4
        #define GGML_F32X_SET1 GGML_F32x4_SET1
        #define GGML_F32X_LOAD GGML_F32x4_LOAD
        #define GGML_F32X_STORE GGML_F32x4_STORE
        #define GGML_F32X_MUL GGML_F32x4_MUL
        #define GGML_F32X_FMA GGML_F32x4_FMA
        #define GLA_VECTOR_SIZE 4
    #endif

    #ifdef GLA_VECTOR_SIZE
        പൂർണ്ണസംഖ്യ gla_vector_size;
        #എങ്കിൽ defined(__ARM_FEATURE_SVE)
            gla_vector_size = svcntw();
        #അല്ലെങ്കിൽ
            gla_vector_size = GLA_VECTOR_SIZE;
        #endif
        സ്ഥിരം int64_t vec_എണ്ണം = head_size / gla_vector_size;

        വേണ്ടി (int64_t t = 0; t < T; t++) {
            size_t t_offset = t * t_stride;
            size_t state_offset = head_size * C * (t / (T / n_seqs));
            ദശാംശം * state_cur = state + state_offset;
            ദശാംശം * state_prev = t % (T / n_seqs) ? state_cur : (ദശാംശം*)dst->src[4]->data + state_offset;

            വേണ്ടി (int64_t h = h_start; h < h_end; h++) {
                size_t h_offset = h * h_stride;
                size_t t_h_offset = t_offset + h_offset;
                size_t h_2d_offset = h * h_stride_2d;

                വേണ്ടി (int64_t i = 0; i < head_size; i++) {
                    size_t t_h_i_offset = t_h_offset + i;
                    size_t h_2d_i_offset = h_2d_offset + i * h_stride;

                    ദശാംശം k_val = k[t_h_i_offset];
                    ദശാംശം q_val = q[t_h_i_offset] * scale;
                    ദശാംശം g_val = g[t_h_i_offset];

                    // Broadcast scalar values to vectors
                    GGML_F32X k_vec = GGML_F32X_SET1(k_val);
                    GGML_F32X q_vec = GGML_F32X_SET1(q_val);
                    GGML_F32X g_vec = GGML_F32X_SET1(g_val);

                    വേണ്ടി (int64_t j = 0; j < vec_എണ്ണം; j++) {
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
                    വേണ്ടി (int64_t j = vec_എണ്ണം * gla_vector_size; j < head_size; j++) {
                        size_t t_h_j_offset = t_h_offset + j;
                        size_t h_2d_i_j_offset = h_2d_i_offset + j;
                        ദശാംശം v_val = v[t_h_j_offset];
                        ദശാംശം kv_val = v_val * k_val;
                        ദശാംശം prev_state_val = state_prev[h_2d_i_j_offset];
                        ദശാംശം temp_val = kv_val + prev_state_val * g_val;
                        dst_data[t_h_j_offset] += temp_val * q_val;
                        state_cur[h_2d_i_j_offset] = temp_val;
                    }
                }
            }
        }

    #അല്ലെങ്കിൽ
        വേണ്ടി (int64_t t = 0; t < T; t++) {
            size_t t_offset = t * t_stride;
            size_t state_offset = head_size * C * (t / (T / n_seqs));
            ദശാംശം * state_cur = state + state_offset;
            ദശാംശം * state_prev = t % (T / n_seqs) ? state_cur : (ദശാംശം*)dst->src[4]->data + state_offset;

            വേണ്ടി (int64_t h = h_start; h < h_end; h++) {
                size_t h_offset = h * h_stride;
                size_t t_h_offset = t_offset + h_offset;
                size_t h_2d_offset = h * h_stride_2d;

                വേണ്ടി (int64_t i = 0; i < head_size; i++) {
                    size_t t_h_i_offset = t_h_offset + i;
                    size_t h_2d_i_offset = h_2d_offset + i * h_stride;

                    ദശാംശം k_val = k[t_h_i_offset];
                    ദശാംശം q_val = q[t_h_i_offset] * scale;
                    ദശാംശം g_val = g[t_h_i_offset];

                    വേണ്ടി (int64_t j = 0; j < head_size; j++) {
                        size_t t_h_j_offset = t_h_offset + j;
                        size_t h_2d_i_j_offset = h_2d_i_offset + j;

                        ദശാംശം v_val = v[t_h_j_offset];
                        ദശാംശം kv_val = v_val * k_val;
                        ദശാംശം prev_state_val = state_prev[h_2d_i_j_offset];
                        ദശാംശം temp_val = prev_state_val * g_val + kv_val;
                        dst_data[t_h_j_offset] += temp_val * q_val;
                        state_cur[h_2d_i_j_offset] = temp_val;
                    }
                }
            }
        }
    #endif
}


ശൂന്യം ggml_compute_forward_gla(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_gla_f32(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_solve_tri_f32(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst) {
    സ്ഥിരം ഘടന ggml_tensor * src0 = dst->src[0];  // A (lower triangular)
    സ്ഥിരം ഘടന ggml_tensor * src1 = dst->src[1];  // B (RHS)

    GGML_TENSOR_ദ്വിമാന_OP_LOCALS;

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(ne00 == ne01); // A must be square
    GGML_ASSERT(ne0  == ne10); // solution cols == B cols
    GGML_ASSERT(ne1  == ne11); // solution rows == B rows

    GGML_ASSERT(ne02 == ne12 && ne12 == ne2);
    GGML_ASSERT(ne03 == ne13 && ne13 == ne3);

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം int64_t k = ne10;   // number of RHS columns
    സ്ഥിരം int64_t n = ne11;   // A is n×n
    സ്ഥിരം int64_t nr = ne02 * ne03 * k; // we're parallelizing on columns here, so seq x token x column will be the unit

    // chunks per thread
    സ്ഥിരം int64_t dr = (nr + nth - 1)/nth;

    // chunk range for this thread
    സ്ഥിരം int64_t ir0 = dr*ith;
    സ്ഥിരം int64_t ir1 = MIN(ir0 + dr, nr);

    സ്ഥിരം ദശാംശം * A = (സ്ഥിരം ദശാംശം *) src0->data;  // [n, n, B1, B2]
    സ്ഥിരം ദശാംശം * B = (സ്ഥിരം ദശാംശം *) src1->data;  // [n, k, B1, B2]
          ദശാംശം * X = (      ദശാംശം *) dst->data;   // [n, k, B1, B2]

    വേണ്ടി (int64_t ir = ir0; ir < ir1; ++ir) {
        സ്ഥിരം int64_t i03 = ir/(ne02*k);
        സ്ഥിരം int64_t i02 = (ir - i03*ne02*k)/k;
        സ്ഥിരം int64_t i01 = (ir - i03*ne02*k - i02*k);

        സ്ഥിരം ദശാംശം * A_batch = A + i02 * nb02 / വലുപ്പം(ദശാംശം) + i03 * nb03 / വലുപ്പം(ദശാംശം);
        സ്ഥിരം ദശാംശം * B_batch = B + i02 * nb12 / വലുപ്പം(ദശാംശം) + i03 * nb13 / വലുപ്പം(ദശാംശം);

        ദശാംശം * X_batch = X + i02 * nb2 / വലുപ്പം(ദശാംശം) + i03 * nb3 / വലുപ്പം(ദശാംശം);

        വേണ്ടി (int64_t i00 = 0; i00 < n; ++i00) {
            ദശാംശം sum = 0.0f;
            വേണ്ടി (int64_t t = 0; t < i00; ++t) {
                sum += A_batch[i00 * n + t] * X_batch[t * k + i01];
            }

            സ്ഥിരം ദശാംശം diag = A_batch[i00 * n + i00];
            പ്രതിജ്ഞാ(diag != 0.0f && "Zero diagonal in triangular matrix");

            X_batch[i00 * k + i01] = (B_batch[i00 * k + i01] - sum) / diag;
        }
    }
}

ശൂന്യം ggml_compute_forward_solve_tri(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst) {
    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];

    എങ്കിൽ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        ggml_compute_forward_solve_tri_f32(params, dst);
    } അല്ലെങ്കിൽ {
        GGML_ABORT("fatal error");
    }
}

// ggml_compute_forward_gated_delta_net
സ്ഥിതം ശൂന്യം ggml_compute_forward_gated_delta_net_one_chunk(
    സ്ഥിരം ggml_compute_params * params,
    ggml_tensor * dst,
    int64_t ir0,
    int64_t ir1) {

    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    സ്ഥിരം int64_t S_v      = src_v->ne[0];
    സ്ഥിരം int64_t H        = src_v->ne[1];
    സ്ഥിരം int64_t n_tokens = src_v->ne[2];
    സ്ഥിരം int64_t n_seqs   = src_v->ne[3];

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

    സ്ഥിരം തർക്കം kda = (neg0 == S_v);

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    സ്ഥിരം int64_t K = ggml_get_op_params_i32(dst, 0);
    GGML_ASSERT(K >= 1);
    // per-seq stride in floats (seq s starts at state + s * seq_stride)
    സ്ഥിരം int64_t state_seq_stride = src_state->nb[3] / വലുപ്പം(ദശാംശം);

    സ്ഥിരം int64_t per_thread = S_v + (K > 1 ? S_v * S_v : 0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;

    ദശാംശം * delta       = (ദശാംശം *)params->wdata + ith * per_thread + CACHE_വരി_SIZE_F32;
    ദശാംശം * state_work  = K > 1 ? (delta + S_v) : nullptr;

    // output layout: [attn_scores | new_states]
    // attn_scores: S_v * H * n_tokens * n_seqs    floats
    // new_states:  S_v * S_v * H * n_seqs * K     floats  (K snapshot slots; last min(n_tokens, K))
    സ്ഥിരം int64_t attn_score_elems    = S_v * H * n_tokens * n_seqs;
    സ്ഥിരം int64_t state_size_per_snap = S_v * S_v * H * n_seqs;
    ദശാംശം * attn_out_base  = (ദശാംശം *)dst->data;
    ദശാംശം * state_out_base = (ദശാംശം *)dst->data + attn_score_elems;

    // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
    // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.

    സ്ഥിരം ദശാംശം * state_in_base = (സ്ഥിരം ദശാംശം *)src_state->data;

  //const int64_t rq1 = nev1 / neq1;
  //const int64_t rk1 = nev1 / nek1;
    സ്ഥിരം int64_t rq3 = nev3 / neq3;
    സ്ഥിരം int64_t rk3 = nev3 / nek3;

    സ്ഥിരം ദശാംശം scale = 1.0f / sqrtf((ദശാംശം) S_v);

    വേണ്ടി (int64_t ir = ir0; ir < ir1; ++ir) {
        സ്ഥിരം int64_t iv1 = ir % H; // head_index
        സ്ഥിരം int64_t iv3 = ir / H; // sequence

        സ്ഥിരം int64_t iq1 = iv1 % neq1;
        സ്ഥിരം int64_t ik1 = iv1 % nek1;

        സ്ഥിരം int64_t iq3 = iv3 / rq3;
        സ്ഥിരം int64_t ik3 = iv3 / rk3;

        // For K=1, write directly to the single output slot to avoid an extra memcpy at the end.
        // For K>1, work in scratch and copy out per-token when the slot is in range.
        ദശാംശം * s_out = (K > 1)
            ? state_work
            : state_out_base + (iv3 * H + iv1) * S_v * S_v;

        // copy input state into the working buffer and operate in-place
        // state layout [S_v, S_v, H, n_seqs]: seq iv3 starts at iv3 * state_seq_stride.
        സ്ഥിരം ദശാംശം * s_in = state_in_base + iv3 * state_seq_stride + iv1 * S_v * S_v;
        memcpy(s_out, s_in, S_v * S_v * വലുപ്പം(ദശാംശം));

        // attn output pointer for first token of this (head, seq)
        ദശാംശം * attn_data = attn_out_base + (iv3 * n_tokens * H + iv1) * S_v;

        വേണ്ടി (int64_t t = 0; t < n_tokens; t++) {
            സ്ഥിരം ദശാംശം * q_d = (സ്ഥിരം ദശാംശം *)((സ്ഥിരം അക്ഷരം *)src_q->data + iq3 * nbq3 + t * nbq2 + iq1 * nbq1);
            സ്ഥിരം ദശാംശം * k_d = (സ്ഥിരം ദശാംശം *)((സ്ഥിരം അക്ഷരം *)src_k->data + ik3 * nbk3 + t * nbk2 + ik1 * nbk1);
            സ്ഥിരം ദശാംശം * v_d = (സ്ഥിരം ദശാംശം *)((സ്ഥിരം അക്ഷരം *)src_v->data + iv3 * nbv3 + t * nbv2 + iv1 * nbv1);

            സ്ഥിരം ദശാംശം beta_val = *(സ്ഥിരം ദശാംശം *)((സ്ഥിരം അക്ഷരം *)src_beta->data + iv3 * nbb3 + t * nbb2 + iv1 * nbb1);
            സ്ഥിരം ദശാംശം * g_d    =  (സ്ഥിരം ദശാംശം *)((സ്ഥിരം അക്ഷരം *)src_g->data    + iv3 * nbg3 + t * nbg2 + iv1 * nbg1);

            // state is stored transposed: s_out[j*S_v + i] = S[i][j]
            // so row j of s_out = column j of S (contiguous access)

            എങ്കിൽ (kda) {
                // precompute exp(g) into delta scratch (reused below)
                വേണ്ടി (int64_t i = 0; i < S_v; ++i) {
                    delta[i] = expf(g_d[i]);
                }
                // S[i][:] *= exp(g[i]) => for each row j of M: M[j][i] *= exp(g[i])
                വേണ്ടി (int64_t j = 0; j < S_v; ++j) {
                    ggml_vec_mul_f32(S_v, &s_out[j * S_v], &s_out[j * S_v], delta);
                }
            } അല്ലെങ്കിൽ {
                ggml_vec_scale_f32(S_v * S_v, s_out, expf(g_d[0]));
            }

            // delta[j] = sum_i S[i][j] * k[i] = dot(row j of M, k)
            വേണ്ടി (int64_t j = 0; j < S_v; ++j) {
                ദശാംശം sum = 0.0f;
                ggml_vec_dot_f32(S_v, &sum, 0, &s_out[j * S_v], 0, k_d, 0, 1);
                delta[j] = (v_d[j] - sum) * beta_val;
            }

            // outer product: S[i][j] += k[i] * delta[j] => M[j][i] += delta[j] * k[i]
            വേണ്ടി (int64_t j = 0; j < S_v; ++j) {
                ggml_vec_mad_f32(S_v, &s_out[j * S_v], k_d, delta[j]);
            }

            // attn_out[j] = sum_i S[i][j] * q[i] = dot(row j of M, q)
            വേണ്ടി (int64_t j = 0; j < S_v; ++j) {
                ദശാംശം sum = 0.0f;
                ggml_vec_dot_f32(S_v, &sum, 0, &s_out[j * S_v], 0, q_d, 0, 1);
                attn_data[j] = sum * scale;
            }

            attn_data += S_v * H; // advance to next token

            എങ്കിൽ (K > 1) {
                സ്ഥിരം int64_t target_slot = n_tokens - 1 - t;
                എങ്കിൽ (target_slot >= 0 && target_slot < K) {
                    ദശാംശം * curr_state_o = state_out_base + target_slot * state_size_per_snap +
                                     (iv3 * H + iv1) * S_v * S_v;
                    memcpy(curr_state_o, s_out, S_v * S_v * വലുപ്പം(ദശാംശം));
                }
            }
        }
    }
}


സ്ഥിതം ശൂന്യം ggml_compute_forward_gated_delta_net_f32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    ggml_tensor * V = dst->src[2];
    int64_t nr = V->ne[1] * V->ne[3];

    // disable for NUMA
    സ്ഥിരം തർക്കം disable_chunking = ggml_is_numa();

    പൂർണ്ണസംഖ്യ nth = params->nth;
    പൂർണ്ണസംഖ്യ ith = params->ith;

    // 4x chunks per thread
    പൂർണ്ണസംഖ്യ nth_scaled = nth * 4;
    int64_t chunk_size = (nr + nth_scaled - 1) / nth_scaled;
    int64_t nchunk     = (nr + chunk_size - 1) / chunk_size;

    എങ്കിൽ (nth == 1 || nchunk < nth || disable_chunking) {
      nchunk = nth;
    }

    എങ്കിൽ (ith == 0) {
      ggml_threadpool_chunk_set(params->threadpool, nth);
    }

    ggml_barrier(params->threadpool);

    സ്ഥിരം int64_t dr = (nr + nchunk - 1) / nchunk;

    പൂർണ്ണസംഖ്യ current_chunk = ith;

    ആകുമ്പോൾ (current_chunk < nchunk) {
        സ്ഥിരം int64_t ir0 = dr * current_chunk;
        സ്ഥിരം int64_t ir1 = MIN(ir0 + dr, nr);

        ggml_compute_forward_gated_delta_net_one_chunk(params, dst, ir0, ir1);
        current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
    }
}

ശൂന്യം ggml_compute_forward_gated_delta_net(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {
    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_gated_delta_net_f32(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}


// ggml_compute_forward_dsv4_hc_comb

സ്ഥിതം ശൂന്യം ggml_dsv4_hc_comb_norm_cols(ദശാംശം * comb, ദശാംശം eps) {
    constexpr int64_t hc = 4;

    വേണ്ടി (int64_t idst = 0; idst < hc; ++idst) {
        ദശാംശം sum = eps;
        വേണ്ടി (int64_t isrc = 0; isrc < hc; ++isrc) {
            sum += comb[idst + hc*isrc];
        }

        സ്ഥിരം ദശാംശം inv_sum = 1.0f / sum;
        വേണ്ടി (int64_t isrc = 0; isrc < hc; ++isrc) {
            comb[idst + hc*isrc] *= inv_sum;
        }
    }
}

സ്ഥിതം ശൂന്യം ggml_dsv4_hc_comb_norm_rows(ദശാംശം * comb, ദശാംശം eps) {
    constexpr int64_t hc = 4;

    വേണ്ടി (int64_t isrc = 0; isrc < hc; ++isrc) {
        ദശാംശം sum = eps;
        വേണ്ടി (int64_t idst = 0; idst < hc; ++idst) {
            sum += comb[idst + hc*isrc];
        }

        സ്ഥിരം ദശാംശം inv_sum = 1.0f / sum;
        വേണ്ടി (int64_t idst = 0; idst < hc; ++idst) {
            comb[idst + hc*isrc] *= inv_sum;
        }
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_dsv4_hc_comb_f32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {
    സ്ഥിരം ggml_tensor * mixes = dst->src[0];
    സ്ഥിരം ggml_tensor * scale = dst->src[1];
    സ്ഥിരം ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    constexpr int64_t hc = 4;
    constexpr int64_t comb_offset = 2*hc;
    constexpr int64_t hc_mix_dim = (2 + hc)*hc;

    സ്ഥിരം int64_t n_tokens = mixes->ne[1];

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

    സ്ഥിരം ദശാംശം eps = ggml_get_op_params_f32(dst, 0);
    സ്ഥിരം int32_t n_iter = ggml_get_op_params_i32(dst, 1);
    GGML_ASSERT(n_iter > 0);

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം int64_t dr  = (n_tokens + nth - 1) / nth;
    സ്ഥിരം int64_t it0 = dr * ith;
    സ്ഥിരം int64_t it1 = MIN(it0 + dr, n_tokens);

    സ്ഥിരം ദശാംശം scale_comb = *(സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) scale->data + 2*nbs0);

    വേണ്ടി (int64_t it = it0; it < it1; ++it) {
        ദശാംശം comb[hc*hc];

        വേണ്ടി (int64_t isrc = 0; isrc < hc; ++isrc) {
            ദശാംശം max = -INFINITY;
            വേണ്ടി (int64_t idst = 0; idst < hc; ++idst) {
                സ്ഥിരം int64_t idx = idst + hc*isrc;
                സ്ഥിരം ദശാംശം xv = *(സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) mixes->data + (comb_offset + idx)*nbm0 + it*nbm1);
                സ്ഥിരം ദശാംശം bv = *(സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) base->data  + (comb_offset + idx)*nbb0);
                സ്ഥിരം ദശാംശം v = xv * scale_comb + bv;
                comb[idx] = v;
                max = MAX(max, v);
            }

            ദശാംശം sum = 0.0f;
            വേണ്ടി (int64_t idst = 0; idst < hc; ++idst) {
                സ്ഥിരം int64_t idx = idst + hc*isrc;
                സ്ഥിരം ദശാംശം v = expf(comb[idx] - max);
                comb[idx] = v;
                sum += v;
            }

            സ്ഥിരം ദശാംശം inv_sum = 1.0f / sum;
            വേണ്ടി (int64_t idst = 0; idst < hc; ++idst) {
                സ്ഥിരം int64_t idx = idst + hc*isrc;
                comb[idx] = comb[idx] * inv_sum + eps;
            }
        }

        ggml_dsv4_hc_comb_norm_cols(comb, eps);
        വേണ്ടി (int32_t i = 1; i < n_iter; ++i) {
            ggml_dsv4_hc_comb_norm_rows(comb, eps);
            ggml_dsv4_hc_comb_norm_cols(comb, eps);
        }

        വേണ്ടി (int64_t isrc = 0; isrc < hc; ++isrc) {
            വേണ്ടി (int64_t idst = 0; idst < hc; ++idst) {
                സ്ഥിരം int64_t idx = idst + hc*isrc;
                *(ദശാംശം *) ((അക്ഷരം *) dst->data + idst*nbd0 + isrc*nbd1 + it*nbd2) = comb[idx];
            }
        }
    }
}

ശൂന്യം ggml_compute_forward_dsv4_hc_comb(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {
    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_dsv4_hc_comb_f32(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_dsv4_hc_pre

സ്ഥിതം ശൂന്യം ggml_compute_forward_dsv4_hc_pre_f32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {
    സ്ഥിരം ggml_tensor * x       = dst->src[0];
    സ്ഥിരം ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    സ്ഥിരം int64_t n_embd   = x->ne[0];
    സ്ഥിരം int64_t hc       = x->ne[1];
    സ്ഥിരം int64_t n_tokens = x->ne[2];

    സ്ഥിരം ദശാംശം scale = ggml_get_op_params_f32(dst, 0);
    സ്ഥിരം തർക്കം  gated = ggml_get_op_params_i32(dst, 1) != 0;

    GGML_ASSERT(dst->ne[0] == n_embd);
    GGML_ASSERT(dst->ne[1] == n_tokens);
    എങ്കിൽ (gated) {
        GGML_ASSERT(weights->ne[0] == n_embd);
        GGML_ASSERT(weights->ne[1] == hc);
        GGML_ASSERT(weights->ne[2] == n_tokens);
    } അല്ലെങ്കിൽ {
        GGML_ASSERT(weights->ne[0] == hc);
        GGML_ASSERT(weights->ne[1] == n_tokens);
    }

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം int64_t nr  = n_embd * n_tokens;
    സ്ഥിരം int64_t dr  = (nr + nth - 1) / nth;
    സ്ഥിരം int64_t ir0 = dr * ith;
    സ്ഥിരം int64_t ir1 = MIN(ir0 + dr, nr);

    വേണ്ടി (int64_t ir = ir0; ir < ir1; ++ir) {
        സ്ഥിരം int64_t i0 = ir % n_embd;
        സ്ഥിരം int64_t it = ir / n_embd;

        ദശാംശം sum = 0.0f;
        വേണ്ടി (int64_t ih = 0; ih < hc; ++ih) {
            സ്ഥിരം ദശാംശം xv = *(സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) x->data + i0*nbx0 + ih*nbx1 + it*nbx2);
            ദശാംശം wv;
            എങ്കിൽ (gated) {
                സ്ഥിരം ദശാംശം gv = *(സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) weights->data + i0*nbw0 + ih*nbw1 + it*nbw2);
                wv = 1.0f / (1.0f + expf(-gv));
            } അല്ലെങ്കിൽ {
                wv = *(സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) weights->data + ih*nbw0 + it*nbw1);
            }
            sum += xv * wv;
        }

        *(ദശാംശം *) ((അക്ഷരം *) dst->data + i0*nbd0 + it*nbd1) = scale * sum;
    }
}

ശൂന്യം ggml_compute_forward_dsv4_hc_pre(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {
    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_dsv4_hc_pre_f32(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_dsv4_hc_post

സ്ഥിതം ശൂന്യം ggml_compute_forward_dsv4_hc_post_f32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {
    സ്ഥിരം ggml_tensor * x        = dst->src[0];
    സ്ഥിരം ggml_tensor * residual = dst->src[1];
    സ്ഥിരം ggml_tensor * post     = dst->src[2];
    സ്ഥിരം ggml_tensor * comb     = dst->src[3];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    സ്ഥിരം int64_t n_embd   = x->ne[0];
    സ്ഥിരം int64_t n_tokens = x->ne[1];
    സ്ഥിരം int64_t hc       = residual->ne[1];

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
    എങ്കിൽ (comb) {
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

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം int64_t nr  = n_embd * hc * n_tokens;
    സ്ഥിരം int64_t dr  = (nr + nth - 1) / nth;
    സ്ഥിരം int64_t ir0 = dr * ith;
    സ്ഥിരം int64_t ir1 = MIN(ir0 + dr, nr);

    വേണ്ടി (int64_t ir = ir0; ir < ir1; ++ir) {
        സ്ഥിരം int64_t i0     = ir % n_embd;
        സ്ഥിരം int64_t idst   = (ir / n_embd) % hc;
        സ്ഥിരം int64_t it     = ir / (n_embd * hc);

        സ്ഥിരം ദശാംശം xv = *(സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) x->data    + i0*nbx0 + it*nbx1);
        സ്ഥിരം ദശാംശം pv = *(സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) post->data + idst*nbp0 + it*nbp1);

        ദശാംശം sum = xv * pv;
        എങ്കിൽ (comb) {
            വേണ്ടി (int64_t isrc = 0; isrc < hc; ++isrc) {
                സ്ഥിരം ദശാംശം rv = *(സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) residual->data + i0*nbr0 + isrc*nbr1 + it*nbr2);
                സ്ഥിരം ദശാംശം cv = *(സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) comb->data     + idst*nbc0 + isrc*nbc1 + it*nbc2);
                sum += rv * cv;
            }
        } അല്ലെങ്കിൽ {
            sum += *(സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) residual->data + i0*nbr0 + idst*nbr1 + it*nbr2);
        }

        *(ദശാംശം *) ((അക്ഷരം *) dst->data + i0*nbd0 + idst*nbd1 + it*nbd2) = sum;
    }
}

ശൂന്യം ggml_compute_forward_dsv4_hc_post(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {
    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_dsv4_hc_post_f32(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_top_k_mask

ശൂന്യം ggml_compute_forward_മുകൾ_k_mask(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {
    സ്ഥിരം ggml_tensor * mask   = dst->src[0];
    സ്ഥിരം ggml_tensor * മുകൾ_k  = dst->src[1];
    സ്ഥിരം ggml_tensor * score  = dst->src[2];
    സ്ഥിരം ggml_tensor * blocks = dst->src[3];
    സ്ഥിരം ggml_tensor * extra  = dst->src[4];

    GGML_ASSERT(dst->type == mask->type);
    GGML_ASSERT(dst->nb[0] == ggml_type_size(dst->type));

    സ്ഥിരം int64_t n_kv    = mask->ne[0];
    സ്ഥിരം int64_t ne1     = dst->ne[1];
    സ്ഥിരം int64_t ne2     = dst->ne[2];
    സ്ഥിരം int64_t n_rows  = ggml_nrows(dst);
    സ്ഥിരം int64_t n_മുകൾ   = മുകൾ_k->ne[0];
    സ്ഥിരം int64_t n_blk   = blocks ? blocks->ne[0] : 1;
    സ്ഥിരം int64_t n_pick  = blocks ? blocks->ne[1] : score ? score->ne[0] : n_kv;
    സ്ഥിരം int64_t n_extra = extra ? extra->ne[0] : 0;

    സ്ഥിരം size_t ts = ggml_type_size(dst->type);

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    വേണ്ടി (int64_t ir = ith; ir < n_rows; ir += nth) {
        സ്ഥിരം int64_t i1 = ir % ne1;
        സ്ഥിരം int64_t i2 = (ir / ne1) % ne2;
        സ്ഥിരം int64_t i3 = ir / (ne1*ne2);

        സ്ഥിരം അക്ഷരം * mask_row = (സ്ഥിരം അക്ഷരം *) mask->data
            + (i1 % mask->ne[1])*mask->nb[1] + (i2 % mask->ne[2])*mask->nb[2] + (i3 % mask->ne[3])*mask->nb[3];
        അക്ഷരം * dst_row = (അക്ഷരം *) dst->data + i1*dst->nb[1] + i2*dst->nb[2] + i3*dst->nb[3];

        എങ്കിൽ (dst->type == GGML_TYPE_F16) {
            സ്ഥിരം ggml_fp16_t ninf = GGML_CPU_FP32_TO_FP16(-INFINITY);
            വേണ്ടി (int64_t c = 0; c < n_kv; ++c) {
                ((ggml_fp16_t *) dst_row)[c] = ninf;
            }
        } അല്ലെങ്കിൽ {
            വേണ്ടി (int64_t c = 0; c < n_kv; ++c) {
                ((ദശാംശം *) dst_row)[c] = -INFINITY;
            }
        }

        auto keep = [&](int32_t c) {
            എങ്കിൽ (c >= 0 && c < n_kv) {
                memcpy(dst_row + c*ts, mask_row + c*ts, ts);
            }
        };

        സ്ഥിരം അക്ഷരം * മുകൾ_k_row = (സ്ഥിരം അക്ഷരം *) മുകൾ_k->data + i1*മുകൾ_k->nb[1] + i2*മുകൾ_k->nb[2] + i3*മുകൾ_k->nb[3];

        വേണ്ടി (int64_t j = 0; j < n_മുകൾ; ++j) {
            സ്ഥിരം int32_t p = *(സ്ഥിരം int32_t *) (മുകൾ_k_row + j*മുകൾ_k->nb[0]);
            എങ്കിൽ (p < 0 || p >= n_pick) {
                തുടരുക;
            }
            എങ്കിൽ (score) {
                സ്ഥിരം ദശാംശം s = *(സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) score->data
                    + p*score->nb[0] + i1*score->nb[1] + i2*score->nb[2] + i3*score->nb[3]);
                എങ്കിൽ (!(s > -INFINITY)) {
                    തുടരുക;
                }
            }
            എങ്കിൽ (!blocks) {
                keep(p);
                തുടരുക;
            }
            വേണ്ടി (int64_t k = 0; k < n_blk; ++k) {
                keep(*(സ്ഥിരം int32_t *) ((സ്ഥിരം അക്ഷരം *) blocks->data + k*blocks->nb[0] + p*blocks->nb[1]));
            }
        }

        എങ്കിൽ (extra) {
            സ്ഥിരം അക്ഷരം * extra_row = (സ്ഥിരം അക്ഷരം *) extra->data + i1*extra->nb[1] + i2*extra->nb[2] + i3*extra->nb[3];
            വേണ്ടി (int64_t j = 0; j < n_extra; ++j) {
                keep(*(സ്ഥിരം int32_t *) (extra_row + j*extra->nb[0]));
            }
        }
    }
}

// ggml_compute_forward_rwkv_wkv7

സ്ഥിതം ശൂന്യം ggml_compute_forward_rwkv_wkv7_f32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {
    സ്ഥിരം int64_t T = dst->src[1]->ne[2];
    സ്ഥിരം int64_t C = dst->ne[0];
    സ്ഥിരം int64_t HEADS = dst->src[1]->ne[1];
    സ്ഥിരം int64_t n_seqs = dst->src[6]->ne[1];
    സ്ഥിരം int64_t head_size = C / HEADS;

    ദശാംശം * dst_data = (ദശാംശം *) dst->data;
    ദശാംശം * state = ((ദശാംശം *) dst->data) + C * T;

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ h_start =  (HEADS * (ith    )) / nth;
    സ്ഥിരം പൂർണ്ണസംഖ്യ h_end   = ((HEADS * (ith + 1)) / nth < HEADS) ?
                         (HEADS * (ith + 1)) / nth : HEADS;

    ദശാംശം * r = (ദശാംശം *) dst->src[0]->data;
    ദശാംശം * w = (ദശാംശം *) dst->src[1]->data;
    ദശാംശം * k = (ദശാംശം *) dst->src[2]->data;
    ദശാംശം * v = (ദശാംശം *) dst->src[3]->data;
    ദശാംശം * a = (ദശാംശം *) dst->src[4]->data;
    ദശാംശം * b = (ദശാംശം *) dst->src[5]->data;

    int64_t t_stride = HEADS * head_size; // Same to C

    int64_t h_stride = C / HEADS;
    GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
    int64_t h_stride_2d = head_size * head_size;

    #എങ്കിൽ defined(GGML_SIMD)
        #എങ്കിൽ defined(__ARM_FEATURE_SVE) || defined(__riscv_v_intrinsic)
            // scalar Route to scalar implementation       //TODO: Write SVE code and RVV code
            വേണ്ടി (int64_t t = 0; t < T; t++) {
                int64_t t_offset = t * t_stride;
                int64_t state_offset = head_size * C * (t / (T / n_seqs));
                ദശാംശം * state_cur = state + state_offset;
                ദശാംശം * state_prev = t % (T / n_seqs) ? state_cur : (ദശാംശം*)dst->src[6]->data + state_offset;

                വേണ്ടി (int64_t h = h_start; h < h_end; h++) {
                    int64_t h_offset = h * h_stride;
                    int64_t t_h_offset = t_offset + h_offset;
                    int64_t h_2d_offset = h * h_stride_2d;

                    വേണ്ടി (int64_t i = 0; i < head_size; i++) {
                        int64_t t_h_i_offset = t_h_offset + i;
                        int64_t h_2d_i_offset = h_2d_offset + i * h_stride;

                        ദശാംശം v_val = v[t_h_i_offset];

                        ദശാംശം sa = 0, result = 0;
                        വേണ്ടി (int64_t j = 0; j < head_size; j++) {
                            sa += a[t_h_offset + j] * state_prev[h_2d_i_offset + j];
                        }

                        വേണ്ടി (int64_t j = 0; j < head_size; j++) {
                            int64_t t_h_j_offset = t_h_offset + j;
                            int64_t h_2d_i_j_offset = h_2d_i_offset + j;

                            ദശാംശം r_val = r[t_h_j_offset];
                            ദശാംശം w_val = w[t_h_j_offset];
                            ദശാംശം k_val = k[t_h_j_offset];
                            ദശാംശം b_val = b[t_h_j_offset];
                            ദശാംശം kv_val = v_val * k_val;
                            ദശാംശം prev_state_val = state_prev[h_2d_i_j_offset];
                            state_cur[h_2d_i_j_offset] = prev_state_val * w_val + kv_val + sa * b_val;
                            result += state_cur[h_2d_i_j_offset] * r_val;
                        }
                        dst_data[t_h_i_offset] = result;
                    }
                }
            }
        #അല്ലെങ്കിൽ
            വേണ്ടി (int64_t t = 0; t < T; t++) {
                int64_t t_offset = t * t_stride;
                int64_t state_offset = head_size * C * (t / (T / n_seqs));
                ദശാംശം * state_cur = state + state_offset;
                ദശാംശം * state_prev = t % (T / n_seqs) ? state_cur : (ദശാംശം*)dst->src[6]->data + state_offset;

                വേണ്ടി (int64_t h = h_start; h < h_end; h++) {
                    int64_t h_offset = h * h_stride;
                    int64_t t_h_offset = t_offset + h_offset;
                    int64_t h_2d_offset = h * h_stride_2d;

                    വേണ്ടി (int64_t ii = 0; ii < head_size; ii++) {
                        int64_t t_h_i_offset = t_h_offset + ii;
                        int64_t h_2d_i_offset = h_2d_offset + ii * h_stride;

                        GGML_F32_VEC v_vec = GGML_F32_VEC_SET1(v[t_h_i_offset]);

                        ദശാംശം sa = 0;
                        {
                            GGML_F32_VEC sum[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };
                            GGML_F32_VEC ax[GGML_F32_ARR];
                            GGML_F32_VEC ay[GGML_F32_ARR];
                            വേണ്ടി (int64_t j = 0; j < head_size; j += GGML_F32_STEP) {
                                വേണ്ടി (int64_t kk = 0; kk < GGML_F32_ARR; kk++) {
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
                        വേണ്ടി (; j < head_size; j += GGML_F32_STEP) {
                            വേണ്ടി (int64_t kk = 0; kk < GGML_F32_ARR; kk++) {
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
                        വേണ്ടി (; j < head_size; j++) {
                            int64_t t_h_j_offset = t_h_offset + j;
                            int64_t h_2d_i_j_offset = h_2d_i_offset + j;

                            ദശാംശം r_val = r[t_h_j_offset];
                            ദശാംശം w_val = w[t_h_j_offset];
                            ദശാംശം k_val = k[t_h_j_offset];
                            ദശാംശം b_val = b[t_h_j_offset];
                            ദശാംശം kv_val = v[t_h_i_offset] * k_val;

                            ദശാംശം prev_state_val = state_prev[h_2d_i_j_offset];
                            state_cur[h_2d_i_j_offset] = prev_state_val * w_val + kv_val + sa * b_val;
                            dst_data[t_h_i_offset] += state_cur[h_2d_i_j_offset] * r_val;
                        }
                    }
                }
            }
        #endif
    #അല്ലെങ്കിൽ
        വേണ്ടി (int64_t t = 0; t < T; t++) {
            int64_t t_offset = t * t_stride;
            int64_t state_offset = head_size * C * (t / (T / n_seqs));
            ദശാംശം * state_cur = state + state_offset;
            ദശാംശം * state_prev = t % (T / n_seqs) ? state_cur : (ദശാംശം*)dst->src[6]->data + state_offset;

            വേണ്ടി (int64_t h = h_start; h < h_end; h++) {
                int64_t h_offset = h * h_stride;
                int64_t t_h_offset = t_offset + h_offset;
                int64_t h_2d_offset = h * h_stride_2d;

                വേണ്ടി (int64_t i = 0; i < head_size; i++) {
                    int64_t t_h_i_offset = t_h_offset + i;
                    int64_t h_2d_i_offset = h_2d_offset + i * h_stride;

                    ദശാംശം v_val = v[t_h_i_offset];

                    ദശാംശം sa = 0, result = 0;
                    വേണ്ടി (int64_t j = 0; j < head_size; j++) {
                        sa += a[t_h_offset + j] * state_prev[h_2d_i_offset + j];
                    }

                    വേണ്ടി (int64_t j = 0; j < head_size; j++) {
                        int64_t t_h_j_offset = t_h_offset + j;
                        int64_t h_2d_i_j_offset = h_2d_i_offset + j;

                        ദശാംശം r_val = r[t_h_j_offset];
                        ദശാംശം w_val = w[t_h_j_offset];
                        ദശാംശം k_val = k[t_h_j_offset];
                        ദശാംശം b_val = b[t_h_j_offset];
                        ദശാംശം kv_val = v_val * k_val;
                        ദശാംശം prev_state_val = state_prev[h_2d_i_j_offset];
                        state_cur[h_2d_i_j_offset] = prev_state_val * w_val + kv_val + sa * b_val;
                        result += state_cur[h_2d_i_j_offset] * r_val;
                    }
                    dst_data[t_h_i_offset] = result;
                }
            }
        }
    #endif
}


ശൂന്യം ggml_compute_forward_rwkv_wkv7(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_rwkv_wkv7_f32(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_map_custom1

ശൂന്യം ggml_compute_forward_map_custom1(
        സ്ഥിരം ggml_compute_params * params,
              ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * a = dst->src[0];

    ഘടന ggml_map_custom1_op_params p;
    memcpy(&p, dst->op_params, വലുപ്പം(p));

    p.fun(dst, a, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_map_custom2

ശൂന്യം ggml_compute_forward_map_custom2(
        സ്ഥിരം ggml_compute_params * params,
              ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * a = dst->src[0];
    സ്ഥിരം ggml_tensor * b = dst->src[1];

    ഘടന ggml_map_custom2_op_params p;
    memcpy(&p, dst->op_params, വലുപ്പം(p));

    p.fun(dst, a, b, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_map_custom3

ശൂന്യം ggml_compute_forward_map_custom3(
        സ്ഥിരം ggml_compute_params * params,
              ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * a = dst->src[0];
    സ്ഥിരം ggml_tensor * b = dst->src[1];
    സ്ഥിരം ggml_tensor * c = dst->src[2];

    ഘടന ggml_map_custom3_op_params p;
    memcpy(&p, dst->op_params, വലുപ്പം(p));

    p.fun(dst, a, b, c, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_custom

ശൂന്യം ggml_compute_forward_custom(
    സ്ഥിരം ഘടന ggml_compute_params * params,
          ഘടന ggml_tensor * dst) {

    ഘടന ggml_custom_op_params p;
    memcpy(&p, dst->op_params, വലുപ്പം(p));

    p.fun(dst, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_cross_entropy_loss

സ്ഥിതം ശൂന്യം ggml_compute_forward_cross_entropy_loss_f32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->nb[0] == ggml_type_size(src0->type));
    GGML_ASSERT(src1->nb[0] == ggml_type_size(src1->type));
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // TODO: handle transposed/permuted matrices
    സ്ഥിരം int64_t nc = src0->ne[0];
    സ്ഥിരം int64_t nr = ggml_nrows(src0);

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    ദശാംശം * sums =  (ദശാംശം *) params->wdata;
    ദശാംശം * st   = ((ദശാംശം *) params->wdata) + nth + ith*nc;
    ദശാംശം sum_thread = 0.0f;

    GGML_ASSERT(params->wsize >= വലുപ്പം(ദശാംശം) * (nth + nth * nc));

    // rows per thread
    സ്ഥിരം int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം int64_t ir0 = dr*ith;
    സ്ഥിരം int64_t ir1 = MIN(ir0 + dr, nr);

    വേണ്ടി (int64_t i1 = ir0; i1 < ir1; ++i1) {
        സ്ഥിരം ദശാംശം * s0 = (സ്ഥിരം ദശാംശം *)((സ്ഥിരം അക്ഷരം *) src0->data + i1*src0->nb[1]);
        സ്ഥിരം ദശാംശം * s1 = (സ്ഥിരം ദശാംശം *)((സ്ഥിരം അക്ഷരം *) src1->data + i1*src1->nb[1]);

#ifndef NDEBUG
        വേണ്ടി (int64_t i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            പ്രതിജ്ഞാ(!isnan(s0[i]));
            പ്രതിജ്ഞാ(!isnan(s1[i]));
        }
#endif // NDEBUG

        ദശാംശം max = -INFINITY;
        ggml_vec_max_f32(nc, &max, s0);
        സ്ഥിരം ggml_float sum_softmax = ggml_vec_log_soft_max_f32(nc, st, s0, max);
        പ്രതിജ്ഞാ(sum_softmax >= 0.0);

        ggml_vec_add1_f32(nc, st, st, -sum_softmax);
        ggml_vec_mul_f32(nc, st, st, s1);

        ദശാംശം sum_st = 0.0f;
        ggml_vec_sum_f32(nc, &sum_st, st);
        sum_thread += sum_st;

#ifndef NDEBUG
        വേണ്ടി (int64_t i = 0; i < nc; ++i) {
            പ്രതിജ്ഞാ(!isnan(st[i]));
            പ്രതിജ്ഞാ(!isinf(st[i]));
        }
#endif // NDEBUG
    }
    sums[ith] = sum_thread;
    ggml_barrier(params->threadpool);

    എങ്കിൽ (ith == 0) {
        ദശാംശം * dp = (ദശാംശം *) dst->data;
        ggml_vec_sum_f32(nth, dp, sums);
        dp[0] *= -1.0f / (ദശാംശം) nr;
    }
}

ശൂന്യം ggml_compute_forward_cross_entropy_loss(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_cross_entropy_loss_f32(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_cross_entropy_loss_back

സ്ഥിതം ശൂന്യം ggml_compute_forward_cross_entropy_loss_back_f32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * grad  = dst->src[0]; // gradient of forward pass output
    സ്ഥിരം ggml_tensor * src0f = dst->src[1]; // src0 of forward pass
    സ്ഥിരം ggml_tensor * src1f = dst->src[2]; // src1 of forward pass

    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src0f));
    GGML_ASSERT(ggml_is_contiguous(src1f));
    GGML_ASSERT(ggml_is_contiguous(grad));
    GGML_ASSERT(ggml_are_same_shape(src0f, src1f) && ggml_are_same_shape(src0f, dst));

    സ്ഥിരം int64_t ith = params->ith;
    സ്ഥിരം int64_t nth = params->nth;

    // TODO: handle transposed/permuted matrices
    സ്ഥിരം int64_t nc = src0f->ne[0];
    സ്ഥിരം int64_t nr = ggml_nrows(src0f);

    // rows per thread
    സ്ഥിരം int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം int64_t ir0 = dr*ith;
    സ്ഥിരം int64_t ir1 = MIN(ir0 + dr, nr);

    സ്ഥിരം ദശാംശം d_by_nr = ((സ്ഥിരം ദശാംശം *) grad->data)[0] / (ദശാംശം) nr;

    വേണ്ടി (int64_t i1 = ir0; i1 < ir1; i1++) {
        ദശാംശം       * ds0 = (ദശാംശം       *)((അക്ഷരം       *) dst->data   + i1*dst->nb[1]);
        സ്ഥിരം ദശാംശം * s0  = (സ്ഥിരം ദശാംശം *)((സ്ഥിരം അക്ഷരം *) src0f->data + i1*src0f->nb[1]);
        സ്ഥിരം ദശാംശം * s1  = (സ്ഥിരം ദശാംശം *)((സ്ഥിരം അക്ഷരം *) src1f->data + i1*src1f->nb[1]);

#ifndef NDEBUG
        വേണ്ടി (int64_t i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            പ്രതിജ്ഞാ(!isnan(s0[i]));
            പ്രതിജ്ഞാ(!isnan(s1[i]));
        }
#endif // NDEBUG

        // soft_max
        ദശാംശം max = -INFINITY;
        ggml_vec_max_f32(nc, &max, s0);
        സ്ഥിരം ggml_float sum = ggml_vec_soft_max_f32(nc, ds0, s0, max);
        പ്രതിജ്ഞാ(sum > 0.0);
        ggml_vec_scale_f32(nc, ds0, 1.0/sum);

        // grad(src0f) = (softmax(src0f) - src1f) * grad(cross_entropy_loss(src0f, src1f)) / nr
        ggml_vec_sub_f32(nc, ds0, ds0, s1);
        ggml_vec_scale_f32(nc, ds0, d_by_nr);

#ifndef NDEBUG
        വേണ്ടി (int64_t i = 0; i < nc; ++i) {
            പ്രതിജ്ഞാ(!isnan(ds0[i]));
            പ്രതിജ്ഞാ(!isinf(ds0[i]));
        }
#endif // NDEBUG
    }
}

ശൂന്യം ggml_compute_forward_cross_entropy_loss_back(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_cross_entropy_loss_back_f32(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_opt_step_adamw_f32(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0         = dst->src[0];
    സ്ഥിരം ggml_tensor * src0_grad    = dst->src[1];
    സ്ഥിരം ggml_tensor * src0_grad_m  = dst->src[2];
    സ്ഥിരം ggml_tensor * src0_grad_v  = dst->src[3];
    സ്ഥിരം ggml_tensor * adamw_params = dst->src[4];

    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad));
    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad_m));
    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad_v));
    GGML_ASSERT(ggml_nelements(adamw_params) == 7);

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS
    GGML_ASSERT(nb00 == വലുപ്പം(ദശാംശം));

    // rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    സ്ഥിരം ദശാംശം * adamw_params_ptr = ggml_get_data_f32(adamw_params);

    സ്ഥിരം ദശാംശം alpha  = adamw_params_ptr[0];
    സ്ഥിരം ദശാംശം beta1  = adamw_params_ptr[1];
    സ്ഥിരം ദശാംശം beta2  = adamw_params_ptr[2];
    സ്ഥിരം ദശാംശം eps    = adamw_params_ptr[3];
    സ്ഥിരം ദശാംശം wd     = adamw_params_ptr[4];
    സ്ഥിരം ദശാംശം beta1h = adamw_params_ptr[5];
    സ്ഥിരം ദശാംശം beta2h = adamw_params_ptr[6];
    സ്ഥിരം ദശാംശം keep   = 1.f - alpha * wd;
    വേണ്ടി (പൂർണ്ണസംഖ്യ ir = ir0; ir < ir1; ++ir) {
        സ്ഥിരം int64_t i03 = ir/(ne02*ne01);
        സ്ഥിരം int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        സ്ഥിരം int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        സ്ഥിരം size_t offset = i03*nb03 + i02*nb02 + i01*nb01;

        ദശാംശം       * w = (ദശാംശം       *) ((അക്ഷരം       *) src0->data        + offset); // weight
        സ്ഥിരം ദശാംശം * g = (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) src0_grad->data   + offset); // grad
        ദശാംശം       * m = (ദശാംശം       *) ((അക്ഷരം       *) src0_grad_m->data + offset);
        ദശാംശം       * v = (ദശാംശം       *) ((അക്ഷരം       *) src0_grad_v->data + offset);

        വേണ്ടി (പൂർണ്ണസംഖ്യ i00 = 0; i00 < ne00; ++i00) {
            m[i00] = m[i00]*beta1 +        g[i00]*(1.0f - beta1);
            v[i00] = v[i00]*beta2 + g[i00]*g[i00]*(1.0f - beta2);

            സ്ഥിരം ദശാംശം mh =       m[i00]*beta1h;
            സ്ഥിരം ദശാംശം vh = sqrtf(v[i00]*beta2h) + eps;

            // The weight decay is applied independently of the Adam momenta m and v.
            // This is NOT equivalent to l2 regularization that adds w[i00]*w[i00] to the loss.
            // See: https://arxiv.org/pdf/1711.05101v3.pdf
            w[i00] = w[i00] * keep - alpha * mh / vh;
        }
    }
}

ശൂന്യം ggml_compute_forward_opt_step_adamw(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_opt_step_adamw_f32(params, dst);
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error");
            }
    }
}

സ്ഥിതം ശൂന്യം ggml_compute_forward_opt_step_sgd_f32(സ്ഥിരം ggml_compute_params * params, ggml_tensor * dst) {
    സ്ഥിരം ggml_tensor * src0       = dst->src[0];
    സ്ഥിരം ggml_tensor * src0_grad  = dst->src[1];
    സ്ഥിരം ggml_tensor * sgd_params = dst->src[2];

    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad));
    GGML_ASSERT(ggml_nelements(sgd_params) == 2);

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം പൂർണ്ണസംഖ്യ nr = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS
    GGML_ASSERT(nb00 == വലുപ്പം(ദശാംശം));

    // rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1) / nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr * ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    // using adamw param subset we care about - alpha, wd - could have a separate struct
    സ്ഥിരം ദശാംശം * sgd_params_ptr   = ggml_get_data_f32(sgd_params);
    സ്ഥിരം ദശാംശം   alpha            = sgd_params_ptr[0];
    സ്ഥിരം ദശാംശം   keep             = 1.f - alpha * sgd_params_ptr[1];

    വേണ്ടി (പൂർണ്ണസംഖ്യ ir = ir0; ir < ir1; ++ir) {
        സ്ഥിരം int64_t i03 = ir / (ne02 * ne01);
        സ്ഥിരം int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
        സ്ഥിരം int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

        സ്ഥിരം size_t offset = i03 * nb03 + i02 * nb02 + i01 * nb01;

        ദശാംശം *       w = (ദശാംശം *) ((അക്ഷരം *) src0->data + offset);                   // weight
        സ്ഥിരം ദശാംശം * g = (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) src0_grad->data + offset);  // grad

        വേണ്ടി (പൂർണ്ണസംഖ്യ i00 = 0; i00 < ne00; ++i00) {
            w[i00] = w[i00] * keep - alpha * g[i00];
        }
    }
}

ശൂന്യം ggml_compute_forward_opt_step_sgd(സ്ഥിരം ggml_compute_params * params, ggml_tensor * dst) {
    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_opt_step_sgd_f32(params, dst);
            }
            നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error - sgd is F32 only");
            }
    }
}

സ്ഥിതം വരിയിൽ ദശാംശം ggml_fwht_load(സ്ഥിരം ദശാംശം value) {
    തിരികെ value;
}

സ്ഥിതം വരിയിൽ ദശാംശം ggml_fwht_load(സ്ഥിരം ggml_fp16_t value) {
    തിരികെ ggml_fp16_to_fp32(value);
}

template<typename src_t>
സ്ഥിതം ശൂന്യം ggml_compute_forward_fwht_impl(സ്ഥിരം ggml_compute_params * params, ggml_tensor * dst) {
    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == (std::is_same_v<src_t, ദശാംശം> ? GGML_TYPE_F32 : GGML_TYPE_F16));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ദ്വിമാന_OP_LOCALS

    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    സ്ഥിരം int64_t n = ne10;
    GGML_ASSERT((n & (n - 1)) == 0); // must be power of 2

    സ്ഥിരം int64_t nr = ne11 * ne12 * ne13;
    സ്ഥിരം int64_t rows_per_thread = (nr + nth - 1) / nth;
    സ്ഥിരം int64_t start_row = ith * rows_per_thread;
    സ്ഥിരം int64_t end_row = MIN(start_row + rows_per_thread, nr);

    സ്ഥിരം ദശാംശം scale = 1.0f / sqrtf((ദശാംശം)n);

#എങ്കിൽ defined(GGML_SIMD)
    സ്ഥിരം GGML_F32_VEC v_minus_one = GGML_F32_VEC_SET1(-1.0f);
#endif

    വേണ്ടി (int64_t r = start_row; r < end_row; r++) {
        സ്ഥിരം int64_t i13 = r / (ne11 * ne12);
        സ്ഥിരം int64_t i12 = (r - i13 * ne11 * ne12) / ne11;
        സ്ഥിരം int64_t i11 = r - i13 * ne11 * ne12 - i12 * ne11;

        സ്ഥിരം src_t * src_row = (സ്ഥിരം src_t *) ((സ്ഥിരം അക്ഷരം *) src1->data + i11 * nb11 + i12 * nb12 + i13 * nb13);
        ദശാംശം * dst_row = (ദശാംശം *) ((അക്ഷരം *) dst->data + i11 * nb1 + i12 * nb2 + i13 * nb3);

        വേണ്ടി (int64_t j = 0; j < n; j++) {
            dst_row[j] = ggml_fwht_load(src_row[j]) * scale;
        }

        // Scalar passes
#എങ്കിൽ defined(GGML_SIMD)
#എങ്കിൽ defined(__ARM_FEATURE_SVE)
        സ്ഥിരം പൂർണ്ണസംഖ്യ step = svcntw();
#അല്ലെങ്കിൽ
        സ്ഥിരം പൂർണ്ണസംഖ്യ step = GGML_F32_EPR;
#endif
#അല്ലെങ്കിൽ
        സ്ഥിരം പൂർണ്ണസംഖ്യ step = n;
#endif
        വേണ്ടി (int64_t len = 1; len < step && len < n; len <<= 1) {
            വേണ്ടി (int64_t i = 0; i < n; i += 2 * len) {
                വേണ്ടി (int64_t j = 0; j < len; j++) {
                    ദശാംശം u = dst_row[i + j];
                    ദശാംശം v = dst_row[i + len + j];
                    dst_row[i + j] = u + v;
                    dst_row[i + len + j] = u - v;
                }
            }
        }

        // SIMD passes using GGML_F32_VEC_* macros for multi-architecture support
#എങ്കിൽ defined(GGML_SIMD)
        വേണ്ടി (int64_t len = step; len < n; len <<= 1) {
            വേണ്ടി (int64_t i = 0; i < n; i += 2 * len) {
                വേണ്ടി (int64_t j = 0; j < len; j += step) {
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

ശൂന്യം ggml_compute_forward_fwht(സ്ഥിരം ggml_compute_params * params, ggml_tensor * dst) {
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];

    തിരഞ്ഞെടുപ്പ് (src1->type) {
        അവസ്ഥ GGML_TYPE_F32:
            {
                ggml_compute_forward_fwht_impl<ദശാംശം>(params, dst);
            }
            നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F16:
            {
                ggml_compute_forward_fwht_impl<ggml_fp16_t>(params, dst);
            }
            നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            {
                GGML_ABORT("fatal error - fwht supports F32 and F16 input");
            }
    }
}

// ggml_compute_forward_lightning_indexer

ശൂന്യം ggml_compute_forward_lightning_indexer(
        സ്ഥിരം ggml_compute_params * params,
        ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * q = dst->src[0];
    സ്ഥിരം ggml_tensor * k = dst->src[1];
    സ്ഥിരം ggml_tensor * w = dst->src[2]; // weights
    സ്ഥിരം ggml_tensor * m = dst->src[3]; // mask

    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(   q->type == GGML_TYPE_F32);
    GGML_ASSERT(   w->type == GGML_TYPE_F32);
    GGML_ASSERT(   m->type == GGML_TYPE_F16);

    GGML_TENSOR_LOCALS(int64_t, neq,  q, ne)
    GGML_TENSOR_LOCALS(size_t,  nbq,  q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek,  k, ne)
    GGML_TENSOR_LOCALS(size_t,  nbk,  k, nb)
    GGML_TENSOR_LOCALS(int64_t, പുതിയ,  w, ne)
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

    സ്ഥിരം പൂർണ്ണസംഖ്യ n_embd    = q->ne[0];
    സ്ഥിരം പൂർണ്ണസംഖ്യ n_head    = q->ne[1];
    സ്ഥിരം പൂർണ്ണസംഖ്യ n_tokens  = q->ne[2];
    സ്ഥിരം പൂർണ്ണസംഖ്യ n_stream  = q->ne[3];
    സ്ഥിരം പൂർണ്ണസംഖ്യ n_kv      = k->ne[2];

    ggml_to_float_t സ്ഥിരം k_to_float = ggml_get_type_traits(k->type)->to_float;
    GGML_ASSERT((k->type == GGML_TYPE_F32 || k_to_float) && "lightning indexer: unsupported K-type");

    സ്ഥിരം പൂർണ്ണസംഖ്യ nr  = n_kv;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ith = params->ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nth = params->nth;

    // (temporary) buffer for K converted to float
    ദശാംശം * k_row_f32 = (ദശാംശം *) params->wdata + ith*(1*n_embd + CACHE_വരി_SIZE_F32);

    // rows per thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ dr = (nr + nth - 1)/nth;

    // row range for this thread
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir0 = dr*ith;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir1 = MIN(ir0 + dr, nr);

    വേണ്ടി (പൂർണ്ണസംഖ്യ s = 0; s < n_stream; ++s) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ t = 0; t < n_tokens; ++t) {
            സ്ഥിരം ദശാംശം       *   w_row =       (ദശാംശം *) ((അക്ഷരം *)   w->data + t*nbw1 +        s*nbw3);
            സ്ഥിരം ggml_fp16_t *   m_row = (ggml_fp16_t *) ((അക്ഷരം *)   m->data + t*nbm1 + (s%nem3)*nbm3);
            ദശാംശം             * dst_row =       (ദശാംശം *) ((അക്ഷരം *) dst->data + t*nb1  +        s*nb3 );
            വേണ്ടി (പൂർണ്ണസംഖ്യ ik = ir0; ik < ir1; ++ik) {
                അക്ഷരം * k_row = (അക്ഷരം *) k->data + ik*nbk2 + s*nbk3;
                എങ്കിൽ (k_to_float) {
                    k_to_float(k_row, k_row_f32, n_embd);
                } അല്ലെങ്കിൽ {
                    k_row_f32 = (ദശാംശം *) k_row;
                }
                ദശാംശം score = 0.0f;
                വേണ്ടി (പൂർണ്ണസംഖ്യ h = 0; h < n_head; ++h) {
                    // dot product of q and k for head h
                    ദശാംശം qk = 0.0f;
                    സ്ഥിരം ദശാംശം * q_row = (ദശാംശം *) ((അക്ഷരം *) q->data + h*nbq1 + t*nbq2 + s*nbq3);
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
