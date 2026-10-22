#ਅਨ੍ਤਰ੍ਭਾਵ "ops.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-cpu.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "binary-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "simd-gemm.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "unary-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "vec.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <algorithm>
#ਅਨ੍ਤਰ੍ਭਾਵ <cfloat>
#ਅਨ੍ਤਰ੍ਭਾਵ <cmath>

// ggml_compute_forward_dup

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_dup_same_cont(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));
    GGML_ASSERT(src0->type == dst->type);

    ਅਚਲ size_t nb0 = ggml_type_ਮਾਪ(src0->type);

    ਅਚਲ ਪੂਰਨ ith = params->ith; // thread index
    ਅਚਲ ਪੂਰਨ nth = params->nth; // number of threads

    // parallelize by blocks
    ਅਚਲ ਪੂਰਨ nk = ggml_nelements(src0)/ggml_blck_ਮਾਪ(src0->type);
    ਅਚਲ ਪੂਰਨ dr = (nk + nth - 1) / nth;
    ਅਚਲ ਪੂਰਨ k0 = dr * ith;
    ਅਚਲ ਪੂਰਨ k1 = MIN(k0 + dr, nk);

    ਜੇ (k0 < k1) {
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(
            ((ਅੱਖਰ *)  dst->ਅੰਕੜੇ + k0*nb0),
            ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + k0*nb0),
            (k1 - k0) * nb0);
    }
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ src_t, ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_dup_flt(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(!ggml_is_quantized(src0->type) && !ggml_is_quantized(dst->type));

    GGML_TENSOR_UNARY_OP_LOCALS

    ਅਚਲ ਪੂਰਨ ith = params->ith; // thread index
    ਅਚਲ ਪੂਰਨ nth = params->nth; // number of threads

    // parallelize by rows
    ਅਚਲ ਪੂਰਨ nr = ne01;
    // number of rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1) / nth;
    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr * ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    // case: type & row size equal
    ਜੇ (src0->type == dst->type &&
        ne00 == ne0 &&
        nb00 == ggml_type_ਮਾਪ(src0->type) && nb0 == ggml_type_ਮਾਪ(dst->type)) {
        // copy by rows
        ਅਚਲ size_t rs = ne00*nb00;
        ਲਈ (int64_t i03 = 0; i03 < ne03; i03++) {
            ਲਈ (int64_t i02 = 0; i02 < ne02; i02++) {
                ਲਈ (int64_t i01 = ir0; i01 < ir1; i01++) {
                    ਸਮ੍ਰਿਤੀ_ਨਕਲ(
                        ((ਅੱਖਰ *)  dst->ਅੰਕੜੇ + i01*nb1  + i02*nb2  + i03*nb3),
                        ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i01*nb01 + i02*nb02 + i03*nb03),
                        rs);
                }
            }
        }
        ਵਾਪਸ;
    }

    // case: dst tensor is contiguous
    ਜੇ (ggml_is_contiguous(dst)) {
        ਜੇ (nb00 == ਆਕਾਰ(src_t)) {
            ਜੇ constexpr (std::is_same_v<dst_t, src_t>) {
                // same type
                size_t id = 0;
                ਅਚਲ size_t rs = ne00 * nb00;
                ਅੱਖਰ * dst_ptr = (ਅੱਖਰ *) dst->ਅੰਕੜੇ;

                ਲਈ (ਪੂਰਨ i03 = 0; i03 < ne03; i03++) {
                    ਲਈ (ਪੂਰਨ i02 = 0; i02 < ne02; i02++) {
                        id += rs * ir0;
                        ਲਈ (ਪੂਰਨ i01 = ir0; i01 < ir1; i01++) {
                            ਅਚਲ ਅੱਖਰ * src0_ptr = (ਅੱਖਰ *) src0->ਅੰਕੜੇ + i01*nb01 + i02*nb02 + i03*nb03;
                            ਸਮ੍ਰਿਤੀ_ਨਕਲ(dst_ptr + id, src0_ptr, rs);
                            id += rs;
                        }
                        id += rs * (ne01 - ir1);
                    }
                }
            } ਨਹੀਂਤਾਂ {
                // casting between non-quantized types
                size_t id = 0;
                dst_t * dst_ptr = (dst_t *) dst->ਅੰਕੜੇ;

                ਲਈ (ਪੂਰਨ i03 = 0; i03 < ne03; i03++) {
                    ਲਈ (ਪੂਰਨ i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        ਲਈ (ਪੂਰਨ i01 = ir0; i01 < ir1; i01++) {
                            ਅਚਲ src_t * src0_ptr = (src_t *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i01*nb01 + i02*nb02 + i03*nb03);
                            ਲਈ (ਪੂਰਨ i00 = 0; i00 < ne00; i00++) {
                                ਦਸ਼ਮਲਵ tmp = type_conversion_ਸਾਰਣੀ<src_t>::ਮੰਜ਼ਿਲ_f32(src0_ptr[i00]);
                                dst_ptr[id] = type_conversion_ਸਾਰਣੀ<dst_t>::from_f32(tmp);
                                id++;
                            }
                        }
                        id += ne00 * (ne01 - ir1);
                    }
                }
            }
        } ਨਹੀਂਤਾਂ {
            //printf("%s: this is not optimal - fix me\n", __func__);

            size_t id = 0;
            dst_t * dst_ptr = (dst_t *) dst->ਅੰਕੜੇ;

            ਲਈ (ਪੂਰਨ i03 = 0; i03 < ne03; i03++) {
                ਲਈ (ਪੂਰਨ i02 = 0; i02 < ne02; i02++) {
                    id += ne00 * ir0;
                    ਲਈ (ਪੂਰਨ i01 = ir0; i01 < ir1; i01++) {
                        ਲਈ (ਪੂਰਨ i00 = 0; i00 < ne00; i00++) {
                            ਅਚਲ src_t * src0_ptr = (src_t *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);

                            ਦਸ਼ਮਲਵ tmp = type_conversion_ਸਾਰਣੀ<src_t>::ਮੰਜ਼ਿਲ_f32(*src0_ptr);
                            dst_ptr[id] = type_conversion_ਸਾਰਣੀ<dst_t>::from_f32(tmp);
                            id++;
                        }
                    }
                    id += ne00 * (ne01 - ir1);
                }
            }
        }
        ਵਾਪਸ;
    }

    // dst counters
    int64_t i10 = 0;
    int64_t i11 = 0;
    int64_t i12 = 0;
    int64_t i13 = 0;

    ਜੇ constexpr (std::is_same_v<dst_t, src_t>) {
        ਲਈ (int64_t i03 = 0; i03 < ne03; i03++) {
            ਲਈ (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                ਜਦੋਂ (i10 >= ne0) {
                    i10 -= ne0;
                    ਜੇ (++i11 == ne1) {
                        i11 = 0;
                        ਜੇ (++i12 == ne2) {
                            i12 = 0;
                            ਜੇ (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                ਲਈ (int64_t i01 = ir0; i01 < ir1; i01++) {
                    ਲਈ (int64_t i00 = 0; i00 < ne00; i00++) {
                        ਅਚਲ ਅੱਖਰ * src0_ptr = ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              ਅੱਖਰ * dst_ptr  = ((ਅੱਖਰ *)  dst->ਅੰਕੜੇ + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                        ਸਮ੍ਰਿਤੀ_ਨਕਲ(dst_ptr, src0_ptr, ਆਕਾਰ(dst_t));

                        ਜੇ (++i10 == ne00) {
                            i10 = 0;
                            ਜੇ (++i11 == ne01) {
                                i11 = 0;
                                ਜੇ (++i12 == ne02) {
                                    i12 = 0;
                                    ਜੇ (++i13 == ne03) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                ਜਦੋਂ (i10 >= ne0) {
                    i10 -= ne0;
                    ਜੇ (++i11 == ne1) {
                        i11 = 0;
                        ਜੇ (++i12 == ne2) {
                            i12 = 0;
                            ਜੇ (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
            }
        }

    } ਨਹੀਂਤਾਂ {
        ਲਈ (int64_t i03 = 0; i03 < ne03; i03++) {
            ਲਈ (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                ਜਦੋਂ (i10 >= ne0) {
                    i10 -= ne0;
                    ਜੇ (++i11 == ne1) {
                        i11 = 0;
                        ਜੇ (++i12 == ne2) {
                            i12 = 0;
                            ਜੇ (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                ਲਈ (int64_t i01 = ir0; i01 < ir1; i01++) {
                    ਲਈ (int64_t i00 = 0; i00 < ne00; i00++) {
                        ਅਚਲ ਅੱਖਰ * src0_ptr = ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              ਅੱਖਰ * dst_ptr  = ((ਅੱਖਰ *)  dst->ਅੰਕੜੇ + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                        ਦਸ਼ਮਲਵ tmp = type_conversion_ਸਾਰਣੀ<src_t>::ਮੰਜ਼ਿਲ_f32(*(ਅਚਲ src_t *) src0_ptr);
                        *(dst_t *) dst_ptr = type_conversion_ਸਾਰਣੀ<dst_t>::from_f32(tmp);

                        ਜੇ (++i10 == ne0) {
                            i10 = 0;
                            ਜੇ (++i11 == ne1) {
                                i11 = 0;
                                ਜੇ (++i12 == ne2) {
                                    i12 = 0;
                                    ਜੇ (++i13 == ne3) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                ਜਦੋਂ (i10 >= ne0) {
                    i10 -= ne0;
                    ਜੇ (++i11 == ne1) {
                        i11 = 0;
                        ਜੇ (++i12 == ne2) {
                            i12 = 0;
                            ਜੇ (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
            }
        }
    }
}


ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ src_t>
ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_dup_ਮੰਜ਼ਿਲ_q(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(!ggml_is_quantized(src0->type));

    GGML_TENSOR_UNARY_OP_LOCALS

    ਅਚਲ ਪੂਰਨ ith = params->ith; // thread index
    ਅਚਲ ਪੂਰਨ nth = params->nth; // number of threads

    // parallelize by rows
    ਅਚਲ ਪੂਰਨ nr = ne01;
    // number of rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1) / nth;
    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr * ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਜੇ (ggml_is_contiguous(dst) &&
            nb00 == ਆਕਾਰ(src_t) &&
            ggml_get_type_traits_cpu(dst->type)->from_float) {
        // casting non-quantized types --> intermediate f32 --> quantized
        ggml_from_float_t ਅਚਲ quantize_ਪੰਕਤੀ_q = ggml_get_type_traits_cpu(dst->type)->from_float;
        ਦਸ਼ਮਲਵ * src0_f32 = (ਦਸ਼ਮਲਵ *) params->wdata + (ne00 + CACHE_ਸਤਰ_ਮਾਪ_F32) * ith;

        size_t id = 0;
        size_t rs = nb0 * (ne00 / ggml_blck_ਮਾਪ(dst->type));
        ਅੱਖਰ * dst_ptr = (ਅੱਖਰ *) dst->ਅੰਕੜੇ;

        ਲਈ (ਪੂਰਨ i03 = 0; i03 < ne03; i03++) {
            ਲਈ (ਪੂਰਨ i02 = 0; i02 < ne02; i02++) {
                id += rs * ir0;
                ਲਈ (ਪੂਰਨ i01 = ir0; i01 < ir1; i01++) {
                    ਅਚਲ src_t * src0_ptr = (src_t *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i01*nb01 + i02*nb02 + i03*nb03);

                    ਲਈ (ਪੂਰਨ i00 = 0; i00 < ne00; i00++) {
                        src0_f32[i00] = type_conversion_ਸਾਰਣੀ<src_t>::ਮੰਜ਼ਿਲ_f32(src0_ptr[i00]);
                    }

                    quantize_ਪੰਕਤੀ_q(src0_f32, dst_ptr + id, ne00);
                    id += rs;
                }
                id += rs * (ne01 - ir1);
            }
        }
    } ਨਹੀਂਤਾਂ {
        // printf("%s %s\n", ggml_type_name(src0->type), ggml_type_name(dst->type));
        GGML_ABORT("not implemented");
    }
}

// A simplified version of ggml_compute_forward_dup that doesn't do float upcasting, and just plain old memcpy.
ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_dup_ਬਾਈਟਾਂ(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(src0->type == dst->type);

    GGML_TENSOR_UNARY_OP_LOCALS;

    ਜੇ (ggml_is_contiguous(src0) && ggml_is_contiguous(dst)) {
        ggml_compute_forward_dup_same_cont(params, dst);
        ਵਾਪਸ;
    }

    ਅਚਲ size_t type_ਮਾਪ = ggml_type_ਮਾਪ(src0->type);

    ਅਚਲ ਪੂਰਨ ith = params->ith; // thread index
    ਅਚਲ ਪੂਰਨ nth = params->nth; // number of threads

    // parallelize by rows
    ਅਚਲ ਪੂਰਨ nr = ne01;
    // number of rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1) / nth;
    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr * ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਜੇ (src0->type == dst->type &&
        ggml_are_same_shape(src0, dst) &&
        nb00 == type_ਮਾਪ && nb0 == type_ਮਾਪ) {
        // copy by rows
        ਅਚਲ size_t rs = ggml_ਪੰਕਤੀ_ਮਾਪ(src0->type, ne00);
        ਲਈ (int64_t i03 = 0; i03 < ne03; i03++) {
            ਲਈ (int64_t i02 = 0; i02 < ne02; i02++) {
                ਲਈ (int64_t i01 = ir0; i01 < ir1; i01++) {
                    ਸਮ੍ਰਿਤੀ_ਨਕਲ(
                        ((ਅੱਖਰ *)  dst->ਅੰਕੜੇ + i01*nb1  + i02*nb2  + i03*nb3),
                        ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i01*nb01 + i02*nb02 + i03*nb03),
                        rs);
                }
            }
        }
        ਵਾਪਸ;
    }

    ਜੇ (ggml_is_contiguous(dst)) {
        size_t id = 0;
        ਅੱਖਰ * dst_ptr = (ਅੱਖਰ *) dst->ਅੰਕੜੇ;
        ਅਚਲ size_t rs = ne00 * type_ਮਾਪ;

        ਜੇ (nb00 == type_ਮਾਪ) {
            // src0 is contiguous on first dimension, copy by rows
            ਲਈ (int64_t i03 = 0; i03 < ne03; i03++) {
                ਲਈ (int64_t i02 = 0; i02 < ne02; i02++) {
                    id += rs * ir0;
                    ਲਈ (int64_t i01 = ir0; i01 < ir1; i01++) {
                        ਅਚਲ ਅੱਖਰ * src0_ptr = (ਅੱਖਰ *) src0->ਅੰਕੜੇ + i01*nb01 + i02*nb02 + i03*nb03;
                        ਸਮ੍ਰਿਤੀ_ਨਕਲ(dst_ptr + id, src0_ptr, rs);
                        id += rs;
                    }
                    id += rs * (ne01 - ir1);
                }
            }
        } ਨਹੀਂਤਾਂ {
            //printf("%s: this is not optimal - fix me\n", __func__);

            ਲਈ (int64_t i03 = 0; i03 < ne03; i03++) {
                ਲਈ (int64_t i02 = 0; i02 < ne02; i02++) {
                    id += rs * ir0;
                    ਲਈ (int64_t i01 = ir0; i01 < ir1; i01++) {
                        ਲਈ (int64_t i00 = 0; i00 < ne00; i00++) {
                            ਅਚਲ ਅੱਖਰ * src0_ptr = (ਅੱਖਰ *) src0->ਅੰਕੜੇ + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03;
                            ਸਮ੍ਰਿਤੀ_ਨਕਲ(dst_ptr + id, src0_ptr, type_ਮਾਪ);

                            id += type_ਮਾਪ;
                        }
                    }
                    id += rs * (ne01 - ir1);
                }
            }
        }

        ਵਾਪਸ;
    }

    // dst counters
    int64_t k10 = 0;
    int64_t i11 = 0;
    int64_t i12 = 0;
    int64_t i13 = 0;

    // number of blocks in a row
    ਅਚਲ int64_t nk00 = ne00 / ggml_blck_ਮਾਪ(src0->type);
    ਅਚਲ int64_t nk0  = ne0  / ggml_blck_ਮਾਪ(dst->type);

    ਲਈ (int64_t i03 = 0; i03 < ne03; i03++) {
        ਲਈ (int64_t i02 = 0; i02 < ne02; i02++) {
            k10 += nk00 * ir0;
            ਜਦੋਂ (k10 >= nk0) {
                k10 -= nk0;
                ਜੇ (++i11 == ne1) {
                    i11 = 0;
                    ਜੇ (++i12 == ne2) {
                        i12 = 0;
                        ਜੇ (++i13 == ne3) {
                            i13 = 0;
                        }
                    }
                }
            }
            ਲਈ (int64_t i01 = ir0; i01 < ir1; i01++) {
                ਲਈ (int64_t k00 = 0; k00 < nk00; k00++) {
                    ਅਚਲ ਅੱਖਰ * src0_ptr = ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + k00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                          ਅੱਖਰ * dst_ptr  = ((ਅੱਖਰ *)  dst->ਅੰਕੜੇ + k10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                    ਸਮ੍ਰਿਤੀ_ਨਕਲ(dst_ptr, src0_ptr, type_ਮਾਪ);

                    ਜੇ (++k10 == nk0) {
                        k10 = 0;
                        ਜੇ (++i11 == ne1) {
                            i11 = 0;
                            ਜੇ (++i12 == ne2) {
                                i12 = 0;
                                ਜੇ (++i13 == ne3) {
                                    i13 = 0;
                                }
                            }
                        }
                    }
                }
            }
            k10 += nk00 * (ne01 - ir1);
            ਜਦੋਂ (k10 >= nk0) {
                k10 -= nk0;
                ਜੇ (++i11 == ne1) {
                    i11 = 0;
                    ਜੇ (++i12 == ne2) {
                        i12 = 0;
                        ਜੇ (++i13 == ne3) {
                            i13 = 0;
                        }
                    }
                }
            }
        }
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_dup_from_q(
        ਅਚਲ ggml_compute_params * params,
              ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

    ਅਚਲ ggml_type type = src0->type;
    ggml_ਮੰਜ਼ਿਲ_float_t ਅਚਲ dequantize_ਪੰਕਤੀ_q = ggml_get_type_traits(type)->ਮੰਜ਼ਿਲ_float;

    size_t qk = ggml_blck_ਮਾਪ(type);
    ਅਚਲ int64_t nr = ggml_nelements(src1) / qk;

    // destination must be contiguous in the first dimension
    GGML_ASSERT(nb10 == ggml_type_ਮਾਪ(dst->type));
    // must either have first dimension large enough to hold a row, or fully contiguous
    GGML_ASSERT((ne10 % qk) == 0 || ggml_is_contiguous(dst));

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr*ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਲਈ (int64_t ir = ir0; ir < ir1; ++ir) {

        uint32_t i = ir * qk;

        ਅਚਲ int64_t i03 = i/(ne00 * ne01 * ne02);
        ਅਚਲ int64_t i02 = (i - i03*ne00*ne01*ne02 )/ (ne00*ne01);
        ਅਚਲ int64_t i01 = (i - i03*ne00*ne01*ne02  -  i02*ne01*ne00) / ne00;
        ਅਚਲ int64_t i00 = i - i03*ne00*ne01*ne02 - i02*ne01*ne00 - i01*ne00;
        ਅਚਲ int64_t x_ਵਿੱਥ = (i00/qk)*nb00 + i01*nb01 + i02*nb02 + i03 * nb03;

        ਅਚਲ int64_t i13 = i/(ne10 * ne11 * ne12);
        ਅਚਲ int64_t i12 = (i - i13*ne10*ne11*ne12) / (ne10*ne11);
        ਅਚਲ int64_t i11 = (i - i13*ne10*ne11*ne12 - i12*ne10*ne11) / ne10;
        ਅਚਲ int64_t i10 = i - i13*ne10*ne11*ne12 - i12*ne10*ne11 - i11*ne10;
        ਅਚਲ int64_t dst_ਵਿੱਥ = i10*nb10 + i11*nb11 + i12*nb12 + i13*nb13;

        dequantize_ਪੰਕਤੀ_q(
                (ਅਚਲ ਖਾਲੀ *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + x_ਵਿੱਥ),
                     (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *)  dst->ਅੰਕੜੇ + dst_ਵਿੱਥ), qk);
    }
}

ਖਾਲੀ ggml_compute_forward_dup(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਜੇ (src0->type == dst->type) {
        ggml_compute_forward_dup_ਬਾਈਟਾਂ(params, dst);
        ਵਾਪਸ;
    }

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F16:
            {
                /**/ ਜੇ (dst->type == GGML_TYPE_F16)  ggml_compute_forward_dup_flt<ggml_fp16_t, ggml_fp16_t>(params, dst);
                ਨਹੀਂਤਾਂ ਜੇ (dst->type == GGML_TYPE_BF16) ggml_compute_forward_dup_flt<ggml_fp16_t, ggml_bf16_t>(params, dst);
                ਨਹੀਂਤਾਂ ਜੇ (dst->type == GGML_TYPE_F32)  ggml_compute_forward_dup_flt<ggml_fp16_t, ਦਸ਼ਮਲਵ      >(params, dst);
                ਨਹੀਂਤਾਂ ggml_compute_forward_dup_ਮੰਜ਼ਿਲ_q<ggml_fp16_t>(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_BF16:
            {
                /**/ ਜੇ (dst->type == GGML_TYPE_F16)  ggml_compute_forward_dup_flt<ggml_bf16_t, ggml_fp16_t>(params, dst);
                ਨਹੀਂਤਾਂ ਜੇ (dst->type == GGML_TYPE_BF16) ggml_compute_forward_dup_flt<ggml_bf16_t, ggml_bf16_t>(params, dst);
                ਨਹੀਂਤਾਂ ਜੇ (dst->type == GGML_TYPE_F32)  ggml_compute_forward_dup_flt<ggml_bf16_t, ਦਸ਼ਮਲਵ      >(params, dst);
                ਨਹੀਂਤਾਂ ggml_compute_forward_dup_ਮੰਜ਼ਿਲ_q<ggml_bf16_t>(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F32:
            {
                /**/ ਜੇ (dst->type == GGML_TYPE_F16)  ggml_compute_forward_dup_flt<ਦਸ਼ਮਲਵ, ggml_fp16_t>(params, dst);
                ਨਹੀਂਤਾਂ ਜੇ (dst->type == GGML_TYPE_BF16) ggml_compute_forward_dup_flt<ਦਸ਼ਮਲਵ, ggml_bf16_t>(params, dst);
                ਨਹੀਂਤਾਂ ਜੇ (dst->type == GGML_TYPE_F32)  ggml_compute_forward_dup_flt<ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ      >(params, dst);
                ਨਹੀਂਤਾਂ ਜੇ (dst->type == GGML_TYPE_I32)  ggml_compute_forward_dup_flt<ਦਸ਼ਮਲਵ, int32_t    >(params, dst);
                ਨਹੀਂਤਾਂ ggml_compute_forward_dup_ਮੰਜ਼ਿਲ_q<ਦਸ਼ਮਲਵ>(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_I32:
            {
                ਜੇ (dst->type == GGML_TYPE_F32) ggml_compute_forward_dup_flt<int32_t, ਦਸ਼ਮਲਵ>(params, dst);
                ਨਹੀਂਤਾਂ GGML_ABORT("not implemented");
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                ਜੇ (ggml_is_quantized(src0->type) && dst->type == GGML_TYPE_F32) {
                    ggml_compute_forward_dup_from_q(params, dst);
                    ਰੋਕੋ;
                }
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_add

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_ਜੋੜੋ_q_f32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

    ਅਚਲ ਪੂਰਨ nr  = ggml_nrows(src0);

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ggml_type type = src0->type;
    ਅਚਲ ggml_type dtype = dst->type;
    ggml_ਮੰਜ਼ਿਲ_float_t ਅਚਲ dequantize_ਪੰਕਤੀ_q = ggml_get_type_traits(type)->ਮੰਜ਼ਿਲ_float;
    ggml_from_float_t ਅਚਲ quantize_ਪੰਕਤੀ_q = ggml_get_type_traits_cpu(dtype)->from_float;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_ਮਾਪ(type));
    GGML_ASSERT(nb10 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ggml_is_quantized(src0->type));
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    // rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr*ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਦਸ਼ਮਲਵ * wdata = (ਦਸ਼ਮਲਵ *) params->wdata + (ne00 + CACHE_ਸਤਰ_ਮਾਪ_F32) * ith;

    ਲਈ (ਪੂਰਨ ir = ir0; ir < ir1; ++ir) {
        // src0 indices
        ਅਚਲ ਪੂਰਨ i03 = ir/(ne02*ne01);
        ਅਚਲ ਪੂਰਨ i02 = (ir - i03*ne02*ne01)/ne01;
        ਅਚਲ ਪੂਰਨ i01 = (ir - i03*ne02*ne01 - i02*ne01);

        // src1 and dst are same shape as src0 => same indices
        ਅਚਲ ਪੂਰਨ i13 = i03;
        ਅਚਲ ਪੂਰਨ i12 = i02;
        ਅਚਲ ਪੂਰਨ i11 = i01;

        ਅਚਲ ਪੂਰਨ i3 = i03;
        ਅਚਲ ਪੂਰਨ i2 = i02;
        ਅਚਲ ਪੂਰਨ i1 = i01;

        ਖਾਲੀ  * src0_ਪੰਕਤੀ = (ਖਾਲੀ *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + (i01*nb01 + i02*nb02 + i03*nb03));
        ਦਸ਼ਮਲਵ * src1_ਪੰਕਤੀ = (ਦਸ਼ਮਲਵ *)((ਅੱਖਰ *) src1->ਅੰਕੜੇ + (i11*nb11 + i12*nb12 + i13*nb13));
        ਖਾਲੀ  * dst_ਪੰਕਤੀ  = (ਖਾਲੀ *) ((ਅੱਖਰ *)  dst->ਅੰਕੜੇ + ( i1*nb1  +  i2*nb2  +  i3*nb3));

        ਪ੍ਰਤਿਜ੍ਞਾ(ne00 % 32 == 0);

        // unquantize row from src0 to temp buffer
        dequantize_ਪੰਕਤੀ_q(src0_ਪੰਕਤੀ, wdata, ne00);
        // add src1
        ggml_vec_acc_f32(ne00, wdata, src1_ਪੰਕਤੀ);
        // quantize row to dst
        ਜੇ (quantize_ਪੰਕਤੀ_q != ਨਿਰਾ) {
            quantize_ਪੰਕਤੀ_q(wdata, dst_ਪੰਕਤੀ, ne00);
        } ਨਹੀਂਤਾਂ {
            ਸਮ੍ਰਿਤੀ_ਨਕਲ(dst_ਪੰਕਤੀ, wdata, ne0*nb0);
        }
    }
}

ਖਾਲੀ ggml_compute_forward_ਜੋੜੋ(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
        ਹਾਲਤ GGML_TYPE_F16:
        ਹਾਲਤ GGML_TYPE_BF16:
            {
                ggml_compute_forward_ਜੋੜੋ_non_quantized(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q1_0:
        ਹਾਲਤ GGML_TYPE_Q2_0:
        ਹਾਲਤ GGML_TYPE_Q4_0:
        ਹਾਲਤ GGML_TYPE_Q4_1:
        ਹਾਲਤ GGML_TYPE_Q5_0:
        ਹਾਲਤ GGML_TYPE_Q5_1:
        ਹਾਲਤ GGML_TYPE_Q8_0:
        ਹਾਲਤ GGML_TYPE_MXFP4:
        ਹਾਲਤ GGML_TYPE_NVFP4:
        ਹਾਲਤ GGML_TYPE_Q2_K:
        ਹਾਲਤ GGML_TYPE_Q3_K:
        ਹਾਲਤ GGML_TYPE_Q4_K:
        ਹਾਲਤ GGML_TYPE_Q5_K:
        ਹਾਲਤ GGML_TYPE_Q6_K:
        ਹਾਲਤ GGML_TYPE_TQ1_0:
        ਹਾਲਤ GGML_TYPE_TQ2_0:
        ਹਾਲਤ GGML_TYPE_IQ2_XXS:
        ਹਾਲਤ GGML_TYPE_IQ2_XS:
        ਹਾਲਤ GGML_TYPE_IQ3_XXS:
        ਹਾਲਤ GGML_TYPE_IQ1_S:
        ਹਾਲਤ GGML_TYPE_IQ1_M:
        ਹਾਲਤ GGML_TYPE_IQ4_NL:
        ਹਾਲਤ GGML_TYPE_IQ4_XS:
        ਹਾਲਤ GGML_TYPE_IQ3_S:
        ਹਾਲਤ GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_ਜੋੜੋ_q_f32(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_add_id

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_ਜੋੜੋ_id_f32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];
    ਅਚਲ ggml_tensor * src2 = dst->src[2];

    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src2->type == GGML_TYPE_I32);

    GGML_ASSERT(src0->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    GGML_ASSERT(src1->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ nr  = ggml_nrows(src0);

    GGML_TENSOR_TERNARY_OP_LOCALS

    GGML_ASSERT( nb0 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    GGML_ASSERT(nb10 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    // rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr*ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਲਈ (ਪੂਰਨ ir = ir0; ir < ir1; ++ir) {
        // src0 indices
        ਅਚਲ ਪੂਰਨ i3 = ir/(ne2*ne1);
        ਅਚਲ ਪੂਰਨ i2 = (ir - i3*ne2*ne1)/ne1;
        ਅਚਲ ਪੂਰਨ i1 = (ir - i3*ne2*ne1 - i2*ne1);

        // src1 indices
        ਅਚਲ ਪੂਰਨ i11 = *(int32_t *) ((ਅੱਖਰ *) src2->ਅੰਕੜੇ + i1*nb20 + i2*nb21);

        GGML_ASSERT(i11 >= 0 && i11 < ne11);

        ggml_vec_ਜੋੜੋ_f32(ne0,
                (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ  + i3*nb3  + i2*nb2  + i1*nb1 ),
                (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i3*nb03 + i2*nb02 + i1*nb01),
                (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src1->ਅੰਕੜੇ + i11*nb11));
    }
}

ਖਾਲੀ ggml_compute_forward_ਜੋੜੋ_id(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_ਜੋੜੋ_id_f32(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("unsupported type for ggml_compute_forward_add_id: %s", ggml_type_name(src0->type));
            }
    }
}

// ggml_compute_forward_add1

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_add1_f32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    GGML_ASSERT(nb00 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    // rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr*ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਲਈ (ਪੂਰਨ ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        ਅਚਲ ਪੂਰਨ i3 = ir/(ne2*ne1);
        ਅਚਲ ਪੂਰਨ i2 = (ir - i3*ne2*ne1)/ne1;
        ਅਚਲ ਪੂਰਨ i1 = (ir - i3*ne2*ne1 - i2*ne1);

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_ACCELERATE
        GGML_ਅਣਵਰਤਿਆ(ggml_vec_add1_f32);

        vDSP_vadd(
                (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i3*nb03 + i2*nb02 + i1*nb01), 1,
                (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src1->ਅੰਕੜੇ), 0,
                (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ  + i3*nb3  + i2*nb2  + i1*nb1 ), 1,
                ne0);
#ਨਹੀਂਤਾਂ
        ggml_vec_add1_f32(ne0,
                (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ  + i3*nb3  + i2*nb2  + i1*nb1 ),
                (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i3*nb03 + i2*nb02 + i1*nb01),
               *(ਦਸ਼ਮਲਵ *) src1->ਅੰਕੜੇ);
#ਜੇ_ਅੰਤ
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_add1_f16_f32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    ਅਚਲ ਦਸ਼ਮਲਵ v = *(ਦਸ਼ਮਲਵ *) src1->ਅੰਕੜੇ;

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F16);

    GGML_ASSERT( nb0 == ਆਕਾਰ(ggml_fp16_t));
    GGML_ASSERT(nb00 == ਆਕਾਰ(ggml_fp16_t));

    // rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr*ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਲਈ (ਪੂਰਨ ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        ਅਚਲ ਪੂਰਨ i3 = ir/(ne2*ne1);
        ਅਚਲ ਪੂਰਨ i2 = (ir - i3*ne2*ne1)/ne1;
        ਅਚਲ ਪੂਰਨ i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i3*nb03 + i2*nb02 + i1*nb01);
        ਲਈ (ਪੂਰਨ i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_CPU_FP32_ਮੰਜ਼ਿਲ_FP16(GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(src0_ptr[i]) + v);
        }
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_add1_f16_f16(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    ਅਚਲ ਦਸ਼ਮਲਵ v = GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(*(ggml_fp16_t *) src1->ਅੰਕੜੇ);

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type  == GGML_TYPE_F16);

    GGML_ASSERT( nb0 == ਆਕਾਰ(ggml_fp16_t));
    GGML_ASSERT(nb00 == ਆਕਾਰ(ggml_fp16_t));

    // rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr*ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਲਈ (ਪੂਰਨ ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        ਅਚਲ ਪੂਰਨ i3 = ir/(ne2*ne1);
        ਅਚਲ ਪੂਰਨ i2 = (ir - i3*ne2*ne1)/ne1;
        ਅਚਲ ਪੂਰਨ i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i3*nb03 + i2*nb02 + i1*nb01);
        ਲਈ (ਪੂਰਨ i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_CPU_FP32_ਮੰਜ਼ਿਲ_FP16(GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(src0_ptr[i]) + v);
        }
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_add1_q_f32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    ਅਚਲ ਦਸ਼ਮਲਵ v = *(ਦਸ਼ਮਲਵ *) src1->ਅੰਕੜੇ;

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    ਅਚਲ ggml_type type = src0->type;
    ggml_ਮੰਜ਼ਿਲ_float_t ਅਚਲ dequantize_ਪੰਕਤੀ_q = ggml_get_type_traits(type)->ਮੰਜ਼ਿਲ_float;
    ggml_from_float_t ਅਚਲ quantize_ਪੰਕਤੀ_q = ggml_get_type_traits_cpu(type)->from_float;

    // we don't support permuted src0
    GGML_ASSERT(nb00 == ggml_type_ਮਾਪ(type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ggml_is_quantized(src0->type));
    GGML_ASSERT(dst->type == src0->type);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    // rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr*ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਦਸ਼ਮਲਵ * wdata = (ਦਸ਼ਮਲਵ *) params->wdata + (ne0 + CACHE_ਸਤਰ_ਮਾਪ_F32) * ith;

    ਲਈ (ਪੂਰਨ ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        ਅਚਲ ਪੂਰਨ i3 = ir/(ne2*ne1);
        ਅਚਲ ਪੂਰਨ i2 = (ir - i3*ne2*ne1)/ne1;
        ਅਚਲ ਪੂਰਨ i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ਖਾਲੀ  * src0_ਪੰਕਤੀ = (ਖਾਲੀ *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + (i1*nb01 + i2*nb02 + i3*nb03));
        ਖਾਲੀ  * dst_ਪੰਕਤੀ  = (ਖਾਲੀ *) ((ਅੱਖਰ *)  dst->ਅੰਕੜੇ + (i1*nb1  + i2*nb2  + i3*nb0 ));

        ਪ੍ਰਤਿਜ੍ਞਾ(ne0 % 32 == 0);

        // unquantize row from src0 to temp buffer
        dequantize_ਪੰਕਤੀ_q(src0_ਪੰਕਤੀ, wdata, ne0);
        // add src1
        ggml_vec_acc1_f32(ne0, wdata, v);
        // quantize row to dst
        quantize_ਪੰਕਤੀ_q(wdata, dst_ਪੰਕਤੀ, ne0);
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_add1_bf16_f32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    ਅਚਲ ਦਸ਼ਮਲਵ v = *(ਦਸ਼ਮਲਵ *) src1->ਅੰਕੜੇ;

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_BF16);

    GGML_ASSERT( nb0 == ਆਕਾਰ(ggml_bf16_t));
    GGML_ASSERT(nb00 == ਆਕਾਰ(ggml_bf16_t));

    // rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr*ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਲਈ (ਪੂਰਨ ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        ਅਚਲ ਪੂਰਨ i3 = ir/(ne2*ne1);
        ਅਚਲ ਪੂਰਨ i2 = (ir - i3*ne2*ne1)/ne1;
        ਅਚਲ ਪੂਰਨ i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_bf16_t * dst_ptr  = (ggml_bf16_t *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_bf16_t * src0_ptr = (ggml_bf16_t *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i3*nb03 + i2*nb02 + i1*nb01);
        ਲਈ (ਪੂਰਨ i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_FP32_ਮੰਜ਼ਿਲ_BF16(GGML_BF16_ਮੰਜ਼ਿਲ_FP32(src0_ptr[i]) + v);
        }
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_add1_bf16_bf16(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    ਅਚਲ ਦਸ਼ਮਲਵ v = GGML_BF16_ਮੰਜ਼ਿਲ_FP32(*(ggml_bf16_t *) src1->ਅੰਕੜੇ);

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(src1->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->type  == GGML_TYPE_BF16);

    GGML_ASSERT( nb0 == ਆਕਾਰ(ggml_bf16_t));
    GGML_ASSERT(nb00 == ਆਕਾਰ(ggml_bf16_t));

    // rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr*ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਲਈ (ਪੂਰਨ ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        ਅਚਲ ਪੂਰਨ i3 = ir/(ne2*ne1);
        ਅਚਲ ਪੂਰਨ i2 = (ir - i3*ne2*ne1)/ne1;
        ਅਚਲ ਪੂਰਨ i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_bf16_t * dst_ptr  = (ggml_bf16_t *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_bf16_t * src0_ptr = (ggml_bf16_t *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i3*nb03 + i2*nb02 + i1*nb01);
        ਲਈ (ਪੂਰਨ i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_FP32_ਮੰਜ਼ਿਲ_BF16(GGML_BF16_ਮੰਜ਼ਿਲ_FP32(src0_ptr[i]) + v);
        }
    }
}

ਖਾਲੀ ggml_compute_forward_add1(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_add1_f32(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F16:
            {
                ਜੇ (src1->type == GGML_TYPE_F16) {
                    ggml_compute_forward_add1_f16_f16(params, dst);
                }
                ਨਹੀਂਤਾਂ ਜੇ (src1->type == GGML_TYPE_F32) {
                    ggml_compute_forward_add1_f16_f32(params, dst);
                }
                ਨਹੀਂਤਾਂ {
                    GGML_ABORT("fatal error");
                }
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_BF16:
            {
                ਜੇ (src1->type == GGML_TYPE_BF16) {
                    ggml_compute_forward_add1_bf16_bf16(params, dst);
                }
                ਨਹੀਂਤਾਂ ਜੇ (src1->type == GGML_TYPE_F32) {
                    ggml_compute_forward_add1_bf16_f32(params, dst);
                }
                ਨਹੀਂਤਾਂ {
                    GGML_ABORT("fatal error");
                }
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q1_0:
        ਹਾਲਤ GGML_TYPE_Q2_0:
        ਹਾਲਤ GGML_TYPE_Q4_0:
        ਹਾਲਤ GGML_TYPE_Q4_1:
        ਹਾਲਤ GGML_TYPE_Q5_0:
        ਹਾਲਤ GGML_TYPE_Q5_1:
        ਹਾਲਤ GGML_TYPE_Q8_0:
        ਹਾਲਤ GGML_TYPE_Q8_1:
        ਹਾਲਤ GGML_TYPE_MXFP4:
        ਹਾਲਤ GGML_TYPE_NVFP4:
        ਹਾਲਤ GGML_TYPE_Q2_K:
        ਹਾਲਤ GGML_TYPE_Q3_K:
        ਹਾਲਤ GGML_TYPE_Q4_K:
        ਹਾਲਤ GGML_TYPE_Q5_K:
        ਹਾਲਤ GGML_TYPE_Q6_K:
        ਹਾਲਤ GGML_TYPE_TQ1_0:
        ਹਾਲਤ GGML_TYPE_TQ2_0:
        ਹਾਲਤ GGML_TYPE_IQ2_XXS:
        ਹਾਲਤ GGML_TYPE_IQ2_XS:
        ਹਾਲਤ GGML_TYPE_IQ3_XXS:
        ਹਾਲਤ GGML_TYPE_IQ1_S:
        ਹਾਲਤ GGML_TYPE_IQ1_M:
        ਹਾਲਤ GGML_TYPE_IQ4_NL:
        ਹਾਲਤ GGML_TYPE_IQ4_XS:
        ਹਾਲਤ GGML_TYPE_IQ3_S:
        ਹਾਲਤ GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_add1_q_f32(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_acc

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_acc_f32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));

    // view src0 and dst with these strides and data offset inbytes during acc
    // nb0 is implicitly element_size because src0 and dst are contiguous
    size_t nb1     = ((int32_t *) dst->op_params)[0];
    size_t nb2     = ((int32_t *) dst->op_params)[1];
    size_t nb3     = ((int32_t *) dst->op_params)[2];
    size_t ਵਿੱਥ  = ((int32_t *) dst->op_params)[3];
    ਤਾਰਕਿਕ   inplace = (ਤਾਰਕਿਕ) ((int32_t *) dst->op_params)[4];

    ਜੇ (!inplace) {
        ਜੇ (params->ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            ਸਮ੍ਰਿਤੀ_ਨਕਲ(
                ((ਅੱਖਰ *)  dst->ਅੰਕੜੇ),
                ((ਅੱਖਰ *) src0->ਅੰਕੜੇ),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ nr = ggml_nrows(src1);
    ਅਚਲ ਪੂਰਨ nc = src1->ne[0];

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne)
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb)

    // src0 and dst as viewed during acc
    ਅਚਲ size_t nb0 = ggml_element_ਮਾਪ(src0);

    ਅਚਲ size_t nb00 = nb0;
    ਅਚਲ size_t nb01 = nb1;
    ਅਚਲ size_t nb02 = nb2;
    ਅਚਲ size_t nb03 = nb3;

    GGML_ASSERT(ਵਿੱਥ + (ne10 == 0 ? 0 : ne10-1)*nb0  + (ne11 == 0 ? 0 : ne11-1)*nb1  + (ne12 == 0 ? 0 : ne12-1)*nb2  + (ne13 == 0 ? 0 : ne13-1)*nb3  < ggml_nbytes(dst));
    GGML_ASSERT(ਵਿੱਥ + (ne10 == 0 ? 0 : ne10-1)*nb00 + (ne11 == 0 ? 0 : ne11-1)*nb01 + (ne12 == 0 ? 0 : ne12-1)*nb02 + (ne13 == 0 ? 0 : ne13-1)*nb03 < ggml_nbytes(src0));

    GGML_ASSERT(nb10 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    // rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr*ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਲਈ (ਪੂਰਨ ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are viewed with shape of src1 and offset
        // => same indices
        ਅਚਲ ਪੂਰਨ i3 = ir/(ne12*ne11);
        ਅਚਲ ਪੂਰਨ i2 = (ir - i3*ne12*ne11)/ne11;
        ਅਚਲ ਪੂਰਨ i1 = (ir - i3*ne12*ne11 - i2*ne11);

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_ACCELERATE
        vDSP_vadd(
                (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i3*nb03 + i2*nb02 + i1*nb01 + ਵਿੱਥ), 1,
                (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src1->ਅੰਕੜੇ + i3*nb13 + i2*nb12 + i1*nb11), 1,
                (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ  + i3*nb3  + i2*nb2  + i1*nb1  + ਵਿੱਥ), 1, nc);
#ਨਹੀਂਤਾਂ
        ggml_vec_ਜੋੜੋ_f32(nc,
                (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *)  dst->ਅੰਕੜੇ + i3*nb3  + i2*nb2  + i1*nb1  + ਵਿੱਥ),
                (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i3*nb03 + i2*nb02 + i1*nb01 + ਵਿੱਥ),
                (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src1->ਅੰਕੜੇ + i3*nb13 + i2*nb12 + i1*nb11));
#ਜੇ_ਅੰਤ
    }
}

ਖਾਲੀ ggml_compute_forward_acc(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_acc_f32(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F16:
        ਹਾਲਤ GGML_TYPE_BF16:
        ਹਾਲਤ GGML_TYPE_Q1_0:
        ਹਾਲਤ GGML_TYPE_Q2_0:
        ਹਾਲਤ GGML_TYPE_Q4_0:
        ਹਾਲਤ GGML_TYPE_Q4_1:
        ਹਾਲਤ GGML_TYPE_Q5_0:
        ਹਾਲਤ GGML_TYPE_Q5_1:
        ਹਾਲਤ GGML_TYPE_Q8_0:
        ਹਾਲਤ GGML_TYPE_Q8_1:
        ਹਾਲਤ GGML_TYPE_MXFP4:
        ਹਾਲਤ GGML_TYPE_NVFP4:
        ਹਾਲਤ GGML_TYPE_Q2_K:
        ਹਾਲਤ GGML_TYPE_Q3_K:
        ਹਾਲਤ GGML_TYPE_Q4_K:
        ਹਾਲਤ GGML_TYPE_Q5_K:
        ਹਾਲਤ GGML_TYPE_Q6_K:
        ਹਾਲਤ GGML_TYPE_TQ1_0:
        ਹਾਲਤ GGML_TYPE_TQ2_0:
        ਹਾਲਤ GGML_TYPE_IQ2_XXS:
        ਹਾਲਤ GGML_TYPE_IQ2_XS:
        ਹਾਲਤ GGML_TYPE_IQ3_XXS:
        ਹਾਲਤ GGML_TYPE_IQ1_S:
        ਹਾਲਤ GGML_TYPE_IQ1_M:
        ਹਾਲਤ GGML_TYPE_IQ4_NL:
        ਹਾਲਤ GGML_TYPE_IQ4_XS:
        ਹਾਲਤ GGML_TYPE_IQ3_S:
        ਹਾਲਤ GGML_TYPE_IQ2_S:
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_sum

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_ਜੋੜ_f32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਜੇ (params->ith != 0) {
        ਵਾਪਸ;
    }

    ਪ੍ਰਤਿਜ੍ਞਾ(ggml_is_scalar(dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)

    ggml_float ਜੋੜ = 0;

    ਜੇ (nb00 == ਆਕਾਰ(ਦਸ਼ਮਲਵ)) {
        ggml_float ਪੰਕਤੀ_ਜੋੜ = 0;
        ਲਈ (int64_t i03 = 0; i03 < ne03; i03++) {
            ਲਈ (int64_t i02 = 0; i02 < ne02; i02++) {
                ਲਈ (int64_t i01 = 0; i01 < ne01; i01++) {
                    ggml_vec_ਜੋੜ_f32_ggf(ne00,
                            &ਪੰਕਤੀ_ਜੋੜ,
                            (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i01*nb01 + i02*nb02 + i03*nb03));
                    ਜੋੜ += ਪੰਕਤੀ_ਜੋੜ;
                }
            }
        }
    } ਨਹੀਂਤਾਂ {
        ਲਈ (int64_t i03 = 0; i03 < ne03; i03++) {
            ਲਈ (int64_t i02 = 0; i02 < ne02; i02++) {
                ਲਈ (int64_t i01 = 0; i01 < ne01; i01++) {
                    ਅਚਲ ਅੱਖਰ * ਪੰਕਤੀ = (ਅਚਲ ਅੱਖਰ *) src0->ਅੰਕੜੇ + i01*nb01 + i02*nb02 + i03*nb03;
                    ਲਈ (int64_t i00 = 0; i00 < ne00; i00++) {
                        ਜੋੜ += (ggml_float) *(ਅਚਲ ਦਸ਼ਮਲਵ *) (ਪੰਕਤੀ + i00*nb00);
                    }
                }
            }
        }
    }

    ((ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ)[0] = ਜੋੜ;
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_ਜੋੜ_f16(
    ਅਚਲ ggml_compute_params * params,
          ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਜੇ (params->ith != 0) {
        ਵਾਪਸ;
    }

    ਪ੍ਰਤਿਜ੍ਞਾ(ggml_is_scalar(dst));

    ਪ੍ਰਤਿਜ੍ਞਾ(src0->nb[0] == ਆਕਾਰ(ggml_fp16_t));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)

    ਦਸ਼ਮਲਵ ਜੋੜ = 0;
    ਦਸ਼ਮਲਵ ਪੰਕਤੀ_ਜੋੜ = 0;

    ਲਈ (int64_t i03 = 0; i03 < ne03; i03++) {
        ਲਈ (int64_t i02 = 0; i02 < ne02; i02++) {
            ਲਈ (int64_t i01 = 0; i01 < ne01; i01++) {
                ggml_vec_ਜੋੜ_f16_ggf(ne00,
                    &ਪੰਕਤੀ_ਜੋੜ,
                    (ggml_fp16_t *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i01 * nb01 + i02 * nb02 + i03 * nb03));
                ਜੋੜ += ਪੰਕਤੀ_ਜੋੜ;
            }
        }
    }
    ((ggml_fp16_t *) dst->ਅੰਕੜੇ)[0] = GGML_CPU_FP32_ਮੰਜ਼ਿਲ_FP16(ਜੋੜ);
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_ਜੋੜ_bf16(
    ਅਚਲ ggml_compute_params * params,
          ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਜੇ (params->ith != 0) {
        ਵਾਪਸ;
    }

    ਪ੍ਰਤਿਜ੍ਞਾ(ggml_is_scalar(dst));

    ਪ੍ਰਤਿਜ੍ਞਾ(src0->nb[0] == ਆਕਾਰ(ggml_bf16_t));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)

    ਦਸ਼ਮਲਵ ਜੋੜ = 0;
    ਦਸ਼ਮਲਵ ਪੰਕਤੀ_ਜੋੜ = 0;

    ਲਈ (int64_t i03 = 0; i03 < ne03; i03++) {
        ਲਈ (int64_t i02 = 0; i02 < ne02; i02++) {
            ਲਈ (int64_t i01 = 0; i01 < ne01; i01++) {
                ggml_vec_ਜੋੜ_bf16_ggf(ne00,
                    &ਪੰਕਤੀ_ਜੋੜ,
                    (ggml_bf16_t *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i01 * nb01 + i02 * nb02 + i03 * nb03));
                ਜੋੜ += ਪੰਕਤੀ_ਜੋੜ;
            }
        }
    }
    ((ggml_bf16_t *) dst->ਅੰਕੜੇ)[0] = GGML_FP32_ਮੰਜ਼ਿਲ_BF16(ਜੋੜ);
}

ਖਾਲੀ ggml_compute_forward_ਜੋੜ(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_ਜੋੜ_f32(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F16:
            {
                ggml_compute_forward_ਜੋੜ_f16(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_BF16:
            {
                ggml_compute_forward_ਜੋੜ_bf16(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_cumsum

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_cumsum_f32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    GGML_ASSERT(dst->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne01);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    ਅਚਲ ਸਵੈਚਾਲਿਤ [ir0, ir1] = get_thread_range(params, src0);

    ਲਈ (int64_t ir = ir0; ir < ir1; ++ir) {
        ਅਚਲ int64_t i03 = ir/(ne02*ne01);
        ਅਚਲ int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        ਅਚਲ int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        ਦਸ਼ਮਲਵ * src_ਪੰਕਤੀ = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i01*nb01 + i02*nb02 + i03*nb03);
        ਦਸ਼ਮਲਵ * dst_ਪੰਕਤੀ = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ  + i01*nb1  + i02*nb2  + i03*nb3);

        ggml_vec_cumsum_f32(ne00, dst_ਪੰਕਤੀ, src_ਪੰਕਤੀ);
    }
}

ਖਾਲੀ ggml_compute_forward_cumsum(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_cumsum_f32(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_sum_rows

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_ਜੋੜ_ਪੰਕਤੀਆਂ_f32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਜੇ (params->ith != 0) {
        ਵਾਪਸ;
    }

    GGML_ASSERT(src0->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    GGML_ASSERT(dst->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(ne0 == 1);
    GGML_ASSERT(ne1 == ne01);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    ਲਈ (int64_t i3 = 0; i3 < ne03; i3++) {
        ਲਈ (int64_t i2 = 0; i2 < ne02; i2++) {
            ਲਈ (int64_t i1 = 0; i1 < ne01; i1++) {
                ਦਸ਼ਮਲਵ * src_ਪੰਕਤੀ = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i1*nb01 + i2*nb02 + i3*nb03);
                ਦਸ਼ਮਲਵ * dst_ਪੰਕਤੀ = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ  + i1*nb1  + i2*nb2  + i3*nb3);
                ਦਸ਼ਮਲਵ ਪੰਕਤੀ_ਜੋੜ = 0;
                ggml_vec_ਜੋੜ_f32(ne00, &ਪੰਕਤੀ_ਜੋੜ, src_ਪੰਕਤੀ);
                dst_ਪੰਕਤੀ[0] = ਪੰਕਤੀ_ਜੋੜ;
            }
        }
    }
}

ਖਾਲੀ ggml_compute_forward_ਜੋੜ_ਪੰਕਤੀਆਂ(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_ਜੋੜ_ਪੰਕਤੀਆਂ_f32(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_mean

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_ਔਸਤ_f32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਜੇ (params->ith != 0) {
        ਵਾਪਸ;
    }

    ਪ੍ਰਤਿਜ੍ਞਾ(src0->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    GGML_TENSOR_UNARY_OP_LOCALS

    ਪ੍ਰਤਿਜ੍ਞਾ(ne0 == 1);
    ਪ੍ਰਤਿਜ੍ਞਾ(ne1 == ne01);
    ਪ੍ਰਤਿਜ੍ਞਾ(ne2 == ne02);
    ਪ੍ਰਤਿਜ੍ਞਾ(ne3 == ne03);

    GGML_ਅਣਵਰਤਿਆ(ne0);
    GGML_ਅਣਵਰਤਿਆ(ne1);
    GGML_ਅਣਵਰਤਿਆ(ne2);
    GGML_ਅਣਵਰਤਿਆ(ne3);

    ਲਈ (int64_t i03 = 0; i03 < ne03; i03++) {
        ਲਈ (int64_t i02 = 0; i02 < ne02; i02++) {
            ਲਈ (int64_t i01 = 0; i01 < ne01; i01++) {
                ggml_vec_ਜੋੜ_f32(ne00,
                        (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *)  dst->ਅੰਕੜੇ + i01*nb1  + i02*nb2  + i03*nb3),
                        (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i01*nb01 + i02*nb02 + i03*nb03));

                *(ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i01*nb1 + i02*nb2 + i03*nb3) /= (ਦਸ਼ਮਲਵ) ne00;
            }
        }
    }
}

ਖਾਲੀ ggml_compute_forward_ਔਸਤ(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_ਔਸਤ_f32(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_argmax

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_argmax_f32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਜੇ (params->ith != 0) {
        ਵਾਪਸ;
    }

    ਪ੍ਰਤਿਜ੍ਞਾ(src0->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਪ੍ਰਤਿਜ੍ਞਾ(dst->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਅਚਲ int64_t ne00 = src0->ne[0];
    ਅਚਲ int64_t ne01 = src0->ne[1];

    ਅਚਲ size_t nb01 = src0->nb[1];
    ਅਚਲ size_t nb0 = dst->nb[0];

    ਲਈ (int64_t i1 = 0; i1 < ne01; i1++) {
        ਦਸ਼ਮਲਵ * src = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i1*nb01);
        int32_t * dst_ = (int32_t *) ((ਅੱਖਰ *)  dst->ਅੰਕੜੇ + i1*nb0);
        ਪੂਰਨ v = 0;
        ggml_vec_argmax_f32(ne00, &v, src);
        dst_[0] = v;
    }
}

ਖਾਲੀ ggml_compute_forward_argmax(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_argmax_f32(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_count_equal

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_ਗਿਣਤੀ_equal_i32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS;

    GGML_ASSERT(src0->type == GGML_TYPE_I32);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));
    GGML_ASSERT(dst->type == GGML_TYPE_I64);

    ਅਚਲ int64_t nr = ggml_nrows(src0);

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    int64_t * sums = (int64_t *) params->wdata;
    int64_t ਜੋੜ_thread = 0;

    // rows per thread
    ਅਚਲ int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ int64_t ir0 = dr*ith;
    ਅਚਲ int64_t ir1 = MIN(ir0 + dr, nr);

    ਲਈ (int64_t ir = ir0; ir < ir1; ++ir) {
        ਅਚਲ int64_t i03 =  ir                        / (ne02*ne01);
        ਅਚਲ int64_t i02 = (ir - i03*ne03)            /       ne01;
        ਅਚਲ int64_t i01 =  ir - i03*ne03 - i02*ne02;

        ਅਚਲ ਅੱਖਰ * data0 = (ਅਚਲ ਅੱਖਰ *) src0->ਅੰਕੜੇ + i03*nb03 + i02*nb02 + i01*nb01;
        ਅਚਲ ਅੱਖਰ * data1 = (ਅਚਲ ਅੱਖਰ *) src1->ਅੰਕੜੇ + i03*nb13 + i02*nb12 + i01*nb11;

        ਲਈ (int64_t i00 = 0; i00 < ne00; ++i00) {
            ਅਚਲ int32_t val0 = *((ਅਚਲ int32_t *) (data0 + i00*nb00));
            ਅਚਲ int32_t val1 = *((ਅਚਲ int32_t *) (data1 + i00*nb10));

            ਜੋੜ_thread += val0 == val1;
        }
    }
    ਜੇ (ith != 0) {
        sums[ith] = ਜੋੜ_thread;
    }
    ggml_barrier(params->threadpool);

    ਜੇ (ith != 0) {
        ਵਾਪਸ;
    }

    ਲਈ (ਪੂਰਨ ith_other = 1; ith_other < nth; ++ith_other) {
        ਜੋੜ_thread += sums[ith_other];
    }
    *((int64_t *) dst->ਅੰਕੜੇ) = ਜੋੜ_thread;
}

ਖਾਲੀ ggml_compute_forward_ਗਿਣਤੀ_equal(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_I32:
            {
                ggml_compute_forward_ਗਿਣਤੀ_equal_i32(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_repeat

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_repeat_f32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਜੇ (params->ith != 0) {
        ਵਾਪਸ;
    }

    GGML_ASSERT(ggml_can_repeat(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    // guaranteed to be an integer due to the check in ggml_can_repeat
    ਅਚਲ ਪੂਰਨ nr0 = (ਪੂਰਨ)(ne0/ne00);
    ਅਚਲ ਪੂਰਨ nr1 = (ਪੂਰਨ)(ne1/ne01);
    ਅਚਲ ਪੂਰਨ nr2 = (ਪੂਰਨ)(ne2/ne02);
    ਅਚਲ ਪੂਰਨ nr3 = (ਪੂਰਨ)(ne3/ne03);

    // TODO: support for transposed / permuted tensors
    GGML_ASSERT(nb0  == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    GGML_ASSERT(nb00 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    // TODO: maybe this is not optimal?
    ਲਈ                         (ਪੂਰਨ i3 = 0; i3 < nr3;  i3++) {
        ਲਈ                     (ਪੂਰਨ k3 = 0; k3 < ne03; k3++) {
            ਲਈ                 (ਪੂਰਨ i2 = 0; i2 < nr2;  i2++) {
                ਲਈ             (ਪੂਰਨ k2 = 0; k2 < ne02; k2++) {
                    ਲਈ         (ਪੂਰਨ i1 = 0; i1 < nr1;  i1++) {
                        ਲਈ     (ਪੂਰਨ k1 = 0; k1 < ne01; k1++) {
                            ਲਈ (ਪੂਰਨ i0 = 0; i0 < nr0;  i0++) {
                                ggml_vec_cpy_f32(ne00,
                                        (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *)  dst->ਅੰਕੜੇ + (i3*ne03 + k3)*nb3  + (i2*ne02 + k2)*nb2  + (i1*ne01 + k1)*nb1  + (i0*ne00)*nb0),
                                        (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + (          k3)*nb03 + (          k2)*nb02 + (          k1)*nb01));
                            }
                        }
                    }
                }
            }
        }
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_repeat_f16(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਜੇ (params->ith != 0) {
        ਵਾਪਸ;
    }

    GGML_ASSERT(ggml_can_repeat(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    // guaranteed to be an integer due to the check in ggml_can_repeat
    ਅਚਲ ਪੂਰਨ nr0 = (ਪੂਰਨ)(ne0/ne00);
    ਅਚਲ ਪੂਰਨ nr1 = (ਪੂਰਨ)(ne1/ne01);
    ਅਚਲ ਪੂਰਨ nr2 = (ਪੂਰਨ)(ne2/ne02);
    ਅਚਲ ਪੂਰਨ nr3 = (ਪੂਰਨ)(ne3/ne03);

    // TODO: support for transposed / permuted tensors
    GGML_ASSERT(nb0  == ਆਕਾਰ(ggml_fp16_t));
    GGML_ASSERT(nb00 == ਆਕਾਰ(ggml_fp16_t));

    // TODO: maybe this is not optimal?
    ਲਈ                         (ਪੂਰਨ i3 = 0; i3 < nr3;  i3++) {
        ਲਈ                     (ਪੂਰਨ k3 = 0; k3 < ne03; k3++) {
            ਲਈ                 (ਪੂਰਨ i2 = 0; i2 < nr2;  i2++) {
                ਲਈ             (ਪੂਰਨ k2 = 0; k2 < ne02; k2++) {
                    ਲਈ         (ਪੂਰਨ i1 = 0; i1 < nr1;  i1++) {
                        ਲਈ     (ਪੂਰਨ k1 = 0; k1 < ne01; k1++) {
                            ਲਈ (ਪੂਰਨ i0 = 0; i0 < nr0;  i0++) {
                                ggml_fp16_t * y = (ggml_fp16_t *) ((ਅੱਖਰ *)  dst->ਅੰਕੜੇ + (i3*ne03 + k3)*nb3  + (i2*ne02 + k2)*nb2  + (i1*ne01 + k1)*nb1  + (i0*ne00)*nb0);
                                ggml_fp16_t * x = (ggml_fp16_t *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + (          k3)*nb03 + (          k2)*nb02 + (          k1)*nb01);
                                // ggml_vec_cpy_f16(ne00, y, x)
                                ਲਈ (ਪੂਰਨ i = 0; i < ne00; ++i) {
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

ਖਾਲੀ ggml_compute_forward_repeat(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F16:
        ਹਾਲਤ GGML_TYPE_BF16:
        ਹਾਲਤ GGML_TYPE_I16:
            {
                ggml_compute_forward_repeat_f16(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F32:
        ਹਾਲਤ GGML_TYPE_I32:
            {
                ggml_compute_forward_repeat_f32(params, dst);
            } ਰੋਕੋ;
        // TODO: templateify the implementation and support for I64
        //       ref https://github.com/ggml-org/llama.cpp/pull/14274#discussion_r2169492225
        //case GGML_TYPE_I64:
        //    {
        //        ggml_compute_forward_repeat_i64(params, dst);
        //    } break;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_repeat_back

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_repeat_ਪਿੱਠ_f32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਜੇ (params->ith != 0) {
        ਵਾਪਸ;
    }

    GGML_ASSERT(ggml_can_repeat(dst, src0));

    GGML_TENSOR_UNARY_OP_LOCALS

    // guaranteed to be an integer due to the check in ggml_can_repeat
    ਅਚਲ ਪੂਰਨ nr0 = (ਪੂਰਨ)(ne00/ne0);
    ਅਚਲ ਪੂਰਨ nr1 = (ਪੂਰਨ)(ne01/ne1);
    ਅਚਲ ਪੂਰਨ nr2 = (ਪੂਰਨ)(ne02/ne2);
    ਅਚਲ ਪੂਰਨ nr3 = (ਪੂਰਨ)(ne03/ne3);

    // TODO: support for transposed / permuted tensors
    GGML_ASSERT(nb0  == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    GGML_ASSERT(nb00 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਜੇ (ggml_is_contiguous(dst)) {
        ggml_vec_ਸੈੱਟ_f32(ne0*ne1*ne2*ne3, (ਦਸ਼ਮਲਵ *)dst->ਅੰਕੜੇ, 0);
    } ਨਹੀਂਤਾਂ {
        ਲਈ         (ਪੂਰਨ k3 = 0; k3 < ne3; k3++) {
            ਲਈ     (ਪੂਰਨ k2 = 0; k2 < ne2; k2++) {
                ਲਈ (ਪੂਰਨ k1 = 0; k1 < ne1; k1++) {
                    ggml_vec_ਸੈੱਟ_f32(ne0,
                        (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + k1*nb1 + k2*nb2 + k3*nb3),
                        0);
                }
            }
        }
    }

    // TODO: maybe this is not optimal?
    ਲਈ                         (ਪੂਰਨ i3 = 0; i3 < nr3; i3++) {
        ਲਈ                     (ਪੂਰਨ k3 = 0; k3 < ne3; k3++) {
            ਲਈ                 (ਪੂਰਨ i2 = 0; i2 < nr2; i2++) {
                ਲਈ             (ਪੂਰਨ k2 = 0; k2 < ne2; k2++) {
                    ਲਈ         (ਪੂਰਨ i1 = 0; i1 < nr1; i1++) {
                        ਲਈ     (ਪੂਰਨ k1 = 0; k1 < ne1; k1++) {
                            ਲਈ (ਪੂਰਨ i0 = 0; i0 < nr0; i0++) {
                                ggml_vec_acc_f32(ne0,
                                        (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *)  dst->ਅੰਕੜੇ + (         k3)*nb3  + (         k2)*nb2  + (         k1)*nb1),
                                        (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + (i3*ne3 + k3)*nb03 + (i2*ne2 + k2)*nb02 + (i1*ne1 + k1)*nb01 + (i0*ne0)*nb00));
                            }
                        }
                    }
                }
            }
        }
    }
}

ਖਾਲੀ ggml_compute_forward_repeat_ਪਿੱਠ(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_repeat_ਪਿੱਠ_f32(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_concat
ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_concat_any(
    ਅਚਲ ggml_compute_params * params,
    ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

    ਅਚਲ int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);
    GGML_ASSERT(ggml_is_contiguous_ਪੰਕਤੀਆਂ(src0));
    GGML_ASSERT(ggml_is_contiguous_ਪੰਕਤੀਆਂ(src1));

    int64_t o[4] = {0, 0, 0, 0};

    ਜੇ (dim == 0) {
        GGML_ASSERT(src0->ne[0] % ggml_blck_ਮਾਪ(src0->type) == 0);
        GGML_ASSERT(src1->ne[0] % ggml_blck_ਮਾਪ(src1->type) == 0);

        o[dim] = src0->ne[dim]/ggml_blck_ਮਾਪ(src0->type);
    } ਨਹੀਂਤਾਂ {
        o[dim] = src0->ne[dim];
    }

    // Region 1: copy rows from src0
    ਲਈ (ਪੂਰਨ i3 = 0; i3 < ne03; i3++) {
        ਲਈ (ਪੂਰਨ i2 = ith; i2 < ne02; i2 += nth) {
            ਲਈ (ਪੂਰਨ i1 = 0; i1 < ne01; i1++) {
                ਅਚਲ ਅੱਖਰ * x = (ਅਚਲ ਅੱਖਰ *) src0->ਅੰਕੜੇ + i1*nb01 + i2*nb02 + i3*nb03;
                      ਅੱਖਰ * y = (      ਅੱਖਰ *) dst->ਅੰਕੜੇ  + i1*nb1  + i2*nb2  + i3*nb3;
                ਸਮ੍ਰਿਤੀ_ਨਕਲ(y, x, ggml_ਪੰਕਤੀ_ਮਾਪ(src0->type, ne00));
            }
        }
    }

    // Region 2: copy rows from src1, offset into dst by o[]
    ਲਈ (ਪੂਰਨ i3 = 0; i3 < ne13; i3++) {
        ਲਈ (ਪੂਰਨ i2 = ith; i2 < ne12; i2 += nth) {
            ਲਈ (ਪੂਰਨ i1 = 0; i1 < ne11; i1++) {
                ਅਚਲ ਅੱਖਰ * x = (ਅਚਲ ਅੱਖਰ *) src1->ਅੰਕੜੇ + i1*nb11         + i2*nb12         + i3*nb13;
                      ਅੱਖਰ * y = (      ਅੱਖਰ *)  dst->ਅੰਕੜੇ + (i1 + o[1])*nb1 + (i2 + o[2])*nb2 + (i3 + o[3])*nb3 + o[0]*nb0;
                ਸਮ੍ਰਿਤੀ_ਨਕਲ(y, x, ggml_ਪੰਕਤੀ_ਮਾਪ(src1->type, ne10));
            }
        }
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_concat_i8(
    ਅਚਲ ggml_compute_params * params,
    ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_type_ਮਾਪ(src0->type) == ਆਕਾਰ(int8_t));

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

    ਅਚਲ int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);

    int64_t o[4] = {0, 0, 0, 0};
    o[dim] = src0->ne[dim];

    ਅਚਲ int8_t * x;

    // TODO: smarter multi-theading
    ਲਈ (ਪੂਰਨ i3 = 0; i3 < ne3; i3++) {
        ਲਈ (ਪੂਰਨ i2 = ith; i2 < ne2; i2 += nth) {
            ਲਈ (ਪੂਰਨ i1 = 0; i1 < ne1; i1++) {
                ਲਈ (ਪੂਰਨ i0 = 0; i0 < ne0; i0++) {
                    ਜੇ (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                        x = (ਅਚਲ int8_t *) ((ਅਚਲ ਅੱਖਰ *)src0->ਅੰਕੜੇ + (i0       )*nb00 + (i1       )*nb01 + (i2       )*nb02 + (i3       )*nb03);
                    } ਨਹੀਂਤਾਂ {
                        x = (ਅਚਲ int8_t *) ((ਅਚਲ ਅੱਖਰ *)src1->ਅੰਕੜੇ + (i0 - o[0])*nb10 + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13);
                    }

                    int8_t * y = (int8_t *)((ਅੱਖਰ *)dst->ਅੰਕੜੇ + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);

                    *y = *x;
                }
            }
        }
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_concat_f16(
    ਅਚਲ ggml_compute_params * params,
    ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_type_ਮਾਪ(src0->type) == ਆਕਾਰ(ggml_fp16_t));

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

    ਅਚਲ int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);

    int64_t o[4] = {0, 0, 0, 0};
    o[dim] = src0->ne[dim];

    ਅਚਲ ggml_fp16_t * x;

    // TODO: smarter multi-theading
    ਲਈ (ਪੂਰਨ i3 = 0; i3 < ne3; i3++) {
        ਲਈ (ਪੂਰਨ i2 = ith; i2 < ne2; i2 += nth) {
            ਲਈ (ਪੂਰਨ i1 = 0; i1 < ne1; i1++) {
                ਲਈ (ਪੂਰਨ i0 = 0; i0 < ne0; i0++) {
                    ਜੇ (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                        x = (ਅਚਲ ggml_fp16_t *) ((ਅਚਲ ਅੱਖਰ *)src0->ਅੰਕੜੇ + (i0       )*nb00 + (i1       )*nb01 + (i2       )*nb02 + (i3       )*nb03);
                    } ਨਹੀਂਤਾਂ {
                        x = (ਅਚਲ ggml_fp16_t *) ((ਅਚਲ ਅੱਖਰ *)src1->ਅੰਕੜੇ + (i0 - o[0])*nb10 + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13);
                    }

                    ggml_fp16_t * y = (ggml_fp16_t *)((ਅੱਖਰ *)dst->ਅੰਕੜੇ + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);

                    *y = *x;
                }
            }
        }
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_concat_f32(
    ਅਚਲ ggml_compute_params * params,
    ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_type_ਮਾਪ(src0->type) == ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

    ਅਚਲ int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);

    int64_t o[4] = {0, 0, 0, 0};
    o[dim] = src0->ne[dim];

    ਅਚਲ ਦਸ਼ਮਲਵ * x;

    // TODO: smarter multi-theading
    ਲਈ (ਪੂਰਨ i3 = 0; i3 < ne3; i3++) {
        ਲਈ (ਪੂਰਨ i2 = ith; i2 < ne2; i2 += nth) {
            ਲਈ (ਪੂਰਨ i1 = 0; i1 < ne1; i1++) {
                ਲਈ (ਪੂਰਨ i0 = 0; i0 < ne0; i0++) {
                    ਜੇ (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                        x = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *)src0->ਅੰਕੜੇ + (i0       )*nb00 + (i1       )*nb01 + (i2       )*nb02 + (i3       )*nb03);
                    } ਨਹੀਂਤਾਂ {
                        x = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *)src1->ਅੰਕੜੇ + (i0 - o[0])*nb10 + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13);
                    }

                    ਦਸ਼ਮਲਵ * y = (ਦਸ਼ਮਲਵ *)((ਅੱਖਰ *)dst->ਅੰਕੜੇ + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);

                    *y = *x;
                }
            }
        }
    }
}

ਖਾਲੀ ggml_compute_forward_concat(
    ਅਚਲ ggml_compute_params * params,
    ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F16:
        ਹਾਲਤ GGML_TYPE_BF16:
        ਹਾਲਤ GGML_TYPE_I16:
            {
                ggml_compute_forward_concat_f16(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_I8:
            {
                ggml_compute_forward_concat_i8(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F32:
        ਹਾਲਤ GGML_TYPE_I32:
            {
                ggml_compute_forward_concat_f32(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                ggml_compute_forward_concat_any(params, dst);
            }
    }
}

// row loop shared by the BF16 unary ops
ਸਾਂਚਾ <ਖਾਲੀ (*vec_op)(ਅਚਲ ਪੂਰਨ, ggml_bf16_t *, ਅਚਲ ggml_bf16_t *)>
ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_unary_bf16_ਪੰਕਤੀਆਂ(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਪ੍ਰਤਿਜ੍ਞਾ(ggml_is_contiguous_ਪੰਕਤੀਆਂ(src0));
    ਪ੍ਰਤਿਜ੍ਞਾ(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ nc = src0->ne[0];
    ਅਚਲ ਪੂਰਨ nr = ggml_nrows(src0);

    // rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr*ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਲਈ (ਪੂਰਨ ir = ir0; ir < ir1; ++ir) {
        ਅਚਲ ਪੂਰਨ i3 = ir/(ne02*ne01);
        ਅਚਲ ਪੂਰਨ i2 = (ir - i3*ne02*ne01)/ne01;
        ਅਚਲ ਪੂਰਨ i1 = (ir - i3*ne02*ne01 - i2*ne01);

        vec_op(nc,
                (ggml_bf16_t *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_bf16_t *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i3*nb03 + i2*nb02 + i1*nb01));
    }
}

// ggml_compute_forward_gelu

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_gelu_f32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਪ੍ਰਤਿਜ੍ਞਾ(ggml_is_contiguous_ਪੰਕਤੀਆਂ(src0));
    ਪ੍ਰਤਿਜ੍ਞਾ(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ nc = src0->ne[0];
    ਅਚਲ ਪੂਰਨ nr = ggml_nrows(src0);

    // rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr*ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਲਈ (ਪੂਰਨ ir = ir0; ir < ir1; ++ir) {
        ਅਚਲ ਪੂਰਨ i3 = ir/(ne02*ne01);
        ਅਚਲ ਪੂਰਨ i2 = (ir - i3*ne02*ne01)/ne01;
        ਅਚਲ ਪੂਰਨ i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_f32(nc,
                (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ  + i3*nb3  + i2*nb2  + i1*nb1),
                (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i3*nb03 + i2*nb02 + i1*nb01));

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NDEBUG
        ਲਈ (ਪੂਰਨ k = 0; k < nc; k++) {
            ਅਚਲ ਦਸ਼ਮਲਵ x = ((ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_ਅਣਵਰਤਿਆ(x);
            ਪ੍ਰਤਿਜ੍ਞਾ(!isnan(x));
            ਪ੍ਰਤਿਜ੍ਞਾ(!isinf(x));
        }
#ਜੇ_ਅੰਤ // NDEBUG
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_gelu_f16(
    ਅਚਲ ggml_compute_params * params,
    ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਪ੍ਰਤਿਜ੍ਞਾ(ggml_is_contiguous_ਪੰਕਤੀਆਂ(src0));
    ਪ੍ਰਤਿਜ੍ਞਾ(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ nc = src0->ne[0];
    ਅਚਲ ਪੂਰਨ nr = ggml_nrows(src0);

    // rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr*ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਲਈ (ਪੂਰਨ ir = ir0; ir < ir1; ++ir) {
        ਅਚਲ ਪੂਰਨ i3 = ir/(ne02*ne01);
        ਅਚਲ ਪੂਰਨ i2 = (ir - i3*ne02*ne01)/ne01;
        ਅਚਲ ਪੂਰਨ i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_f16(nc,
                (ggml_fp16_t *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i3*nb03 + i2*nb02 + i1*nb01));

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NDEBUG
        ਲਈ (ਪੂਰਨ k = 0; k < nc; k++) {
            ਅਚਲ ggml_fp16_t x = ((ggml_fp16_t *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            ਅਚਲ ਦਸ਼ਮਲਵ v = GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x);
            GGML_ਅਣਵਰਤਿਆ(v);
            ਪ੍ਰਤਿਜ੍ਞਾ(!isnan(v));
            ਪ੍ਰਤਿਜ੍ਞਾ(!isinf(v));
        }
#ਜੇ_ਅੰਤ // NDEBUG
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_gelu_bf16(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_ਪੰਕਤੀਆਂ<ggml_vec_gelu_bf16>(params, dst);
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_gelu(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_gelu_f32(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F16:
            {
                ggml_compute_forward_gelu_f16(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_BF16:
            {
                ggml_compute_forward_gelu_bf16(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_fill

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_fill_f32(ਅਚਲ ggml_compute_params * params, ggml_tensor * dst) {
    ਅਚਲ ਦਸ਼ਮਲਵ c = ggml_get_op_params_f32(dst, 0);

    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne);
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb);

    ਅਚਲ ਸਵੈਚਾਲਿਤ [ir0, ir1] = get_thread_range(params, dst);

    ਲਈ (int64_t ir = ir0; ir < ir1; ++ir) {
        ਅਚਲ int64_t i03 = ir/(ne2*ne1);
        ਅਚਲ int64_t i02 = (ir - i03*ne2*ne1)/ne1;
        ਅਚਲ int64_t i01 = (ir - i03*ne2*ne1 - i02*ne1);

        ਦਸ਼ਮਲਵ * dst_ptr  = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i03*nb3 + i02*nb2 + i01*nb1);

        ggml_vec_ਸੈੱਟ_f32(ne0, dst_ptr, c);
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_fill_f16(ਅਚਲ ggml_compute_params * params, ggml_tensor * dst) {
    ਅਚਲ ggml_fp16_t c = GGML_CPU_FP32_ਮੰਜ਼ਿਲ_FP16(ggml_get_op_params_f32(dst, 0));

    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne);
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb);

    ਅਚਲ ਸਵੈਚਾਲਿਤ [ir0, ir1] = get_thread_range(params, dst);

    ਲਈ (int64_t ir = ir0; ir < ir1; ++ir) {
        ਅਚਲ int64_t i03 = ir/(ne2*ne1);
        ਅਚਲ int64_t i02 = (ir - i03*ne2*ne1)/ne1;
        ਅਚਲ int64_t i01 = (ir - i03*ne2*ne1 - i02*ne1);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i03*nb3 + i02*nb2 + i01*nb1);

        ggml_vec_ਸੈੱਟ_f16(ne0, dst_ptr, c);
    }
}

ਖਾਲੀ ggml_compute_forward_fill(ਅਚਲ ggml_compute_params * params, ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_fill_f32(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F16:
            {
                ggml_compute_forward_fill_f16(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("unsupported type for ggml_compute_forward_fill: %s", ggml_type_name(src0->type));
            }
    }
}

// ggml_compute_tri

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_tri_f32(ਅਚਲ ggml_compute_params * params, ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਅਚਲ ggml_tri_type ttype = (ggml_tri_type) ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_TENSOR_UNARY_OP_LOCALS

    ਅਚਲ ਸਵੈਚਾਲਿਤ [ir0, ir1] = get_thread_range(params, src0);

    ਤਾਰਕਿਕ (*bipred)(ਪੂਰਨ, ਪੂਰਨ);

    ਚੋਣ (ttype) {
        ਹਾਲਤ GGML_TRI_TYPE_ਹੇਠਲਾ:      bipred = [](ਪੂਰਨ i, ਪੂਰਨ r) { ਵਾਪਸ i <  r; }; ਰੋਕੋ;
        ਹਾਲਤ GGML_TRI_TYPE_ਹੇਠਲਾ_DIAG: bipred = [](ਪੂਰਨ i, ਪੂਰਨ r) { ਵਾਪਸ i <= r; }; ਰੋਕੋ;
        ਹਾਲਤ GGML_TRI_TYPE_ਉੱਤਲਾ:      bipred = [](ਪੂਰਨ i, ਪੂਰਨ r) { ਵਾਪਸ i >  r; }; ਰੋਕੋ;
        ਹਾਲਤ GGML_TRI_TYPE_ਉੱਤਲਾ_DIAG: bipred = [](ਪੂਰਨ i, ਪੂਰਨ r) { ਵਾਪਸ i >= r; }; ਰੋਕੋ;
        ਮੂਲ: GGML_ABORT("invalid tri type");
    }

    ਲਈ (int64_t ir = ir0; ir < ir1; ++ir) {
        ਅਚਲ int64_t i03 = ir/(ne02*ne01);
        ਅਚਲ int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        ਅਚਲ int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        ਅਚਲ ਦਸ਼ਮਲਵ * src_ptr = (ਅਚਲ ਦਸ਼ਮਲਵ  *) ((ਅਚਲ ਅੱਖਰ *) src0->ਅੰਕੜੇ + i03*nb03 + i02*nb02 + i01*nb01);
              ਦਸ਼ਮਲਵ * dst_ptr = (      ਦਸ਼ਮਲਵ  *) ((      ਅੱਖਰ *) dst->ਅੰਕੜੇ  + i03*nb3  + i02*nb2  + i01*nb1);

        ਲਈ (ਪੂਰਨ i0 = 0; i0 < ne0; ++i0) {
            dst_ptr[i0] = bipred(i0, i01) ? src_ptr[i0] : 0.0f;
        }
    }
}

ਖਾਲੀ ggml_compute_forward_tri(ਅਚਲ ggml_compute_params * params, ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_tri_f32(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_gelu_erf

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_gelu_erf_f32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਪ੍ਰਤਿਜ੍ਞਾ(ggml_is_contiguous_ਪੰਕਤੀਆਂ(src0));
    ਪ੍ਰਤਿਜ੍ਞਾ(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ nc = src0->ne[0];
    ਅਚਲ ਪੂਰਨ nr = ggml_nrows(src0);

    // rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr*ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਲਈ (ਪੂਰਨ ir = ir0; ir < ir1; ++ir) {
        ਅਚਲ ਪੂਰਨ i3 = ir/(ne02*ne01);
        ਅਚਲ ਪੂਰਨ i2 = (ir - i3*ne02*ne01)/ne01;
        ਅਚਲ ਪੂਰਨ i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_erf_f32(nc,
                (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ  + i3*nb3  + i2*nb2  + i1*nb1),
                (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i3*nb03 + i2*nb02 + i1*nb01));

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NDEBUG
        ਲਈ (ਪੂਰਨ k = 0; k < nc; k++) {
            ਅਚਲ ਦਸ਼ਮਲਵ x = ((ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_ਅਣਵਰਤਿਆ(x);
            ਪ੍ਰਤਿਜ੍ਞਾ(!isnan(x));
            ਪ੍ਰਤਿਜ੍ਞਾ(!isinf(x));
        }
#ਜੇ_ਅੰਤ // NDEBUG
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_gelu_erf_f16(
    ਅਚਲ ggml_compute_params * params,
    ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਪ੍ਰਤਿਜ੍ਞਾ(ggml_is_contiguous_ਪੰਕਤੀਆਂ(src0));
    ਪ੍ਰਤਿਜ੍ਞਾ(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ nc = src0->ne[0];
    ਅਚਲ ਪੂਰਨ nr = ggml_nrows(src0);

    // rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr*ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਲਈ (ਪੂਰਨ ir = ir0; ir < ir1; ++ir) {
        ਅਚਲ ਪੂਰਨ i3 = ir/(ne02*ne01);
        ਅਚਲ ਪੂਰਨ i2 = (ir - i3*ne02*ne01)/ne01;
        ਅਚਲ ਪੂਰਨ i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_erf_f16(nc,
                (ggml_fp16_t *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i3*nb03 + i2*nb02 + i1*nb01));

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NDEBUG
        ਲਈ (ਪੂਰਨ k = 0; k < nc; k++) {
            ਅਚਲ ggml_fp16_t x = ((ggml_fp16_t *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            ਅਚਲ ਦਸ਼ਮਲਵ v = GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x);
            GGML_ਅਣਵਰਤਿਆ(v);
            ਪ੍ਰਤਿਜ੍ਞਾ(!isnan(v));
            ਪ੍ਰਤਿਜ੍ਞਾ(!isinf(v));
        }
#ਜੇ_ਅੰਤ // NDEBUG
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_gelu_erf_bf16(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_ਪੰਕਤੀਆਂ<ggml_vec_gelu_erf_bf16>(params, dst);
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_gelu_erf(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_gelu_erf_f32(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F16:
            {
                ggml_compute_forward_gelu_erf_f16(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_BF16:
            {
                ggml_compute_forward_gelu_erf_bf16(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_gelu_quick

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_gelu_ਤੇਜ਼_f32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਪ੍ਰਤਿਜ੍ਞਾ(ggml_is_contiguous_ਪੰਕਤੀਆਂ(src0));
    ਪ੍ਰਤਿਜ੍ਞਾ(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ nc = src0->ne[0];
    ਅਚਲ ਪੂਰਨ nr = ggml_nrows(src0);

    // rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr*ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਲਈ (ਪੂਰਨ ir = ir0; ir < ir1; ++ir) {
        ਅਚਲ ਪੂਰਨ i3 = ir/(ne02*ne01);
        ਅਚਲ ਪੂਰਨ i2 = (ir - i3*ne02*ne01)/ne01;
        ਅਚਲ ਪੂਰਨ i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_ਤੇਜ਼_f32(nc,
                (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ  + i3*nb3  + i2*nb2  + i1*nb1),
                (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i3*nb03 + i2*nb02 + i1*nb01));

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NDEBUG
        ਲਈ (ਪੂਰਨ k = 0; k < nc; k++) {
            ਅਚਲ ਦਸ਼ਮਲਵ x = ((ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_ਅਣਵਰਤਿਆ(x);
            ਪ੍ਰਤਿਜ੍ਞਾ(!isnan(x));
            ਪ੍ਰਤਿਜ੍ਞਾ(!isinf(x));
        }
#ਜੇ_ਅੰਤ // NDEBUG
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_gelu_ਤੇਜ਼_f16(
    ਅਚਲ ggml_compute_params * params,
    ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਪ੍ਰਤਿਜ੍ਞਾ(ggml_is_contiguous_ਪੰਕਤੀਆਂ(src0));
    ਪ੍ਰਤਿਜ੍ਞਾ(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ nc = src0->ne[0];
    ਅਚਲ ਪੂਰਨ nr = ggml_nrows(src0);

    // rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr*ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਲਈ (ਪੂਰਨ ir = ir0; ir < ir1; ++ir) {
        ਅਚਲ ਪੂਰਨ i3 = ir/(ne02*ne01);
        ਅਚਲ ਪੂਰਨ i2 = (ir - i3*ne02*ne01)/ne01;
        ਅਚਲ ਪੂਰਨ i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_ਤੇਜ਼_f16(nc,
                (ggml_fp16_t *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i3*nb03 + i2*nb02 + i1*nb01));

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NDEBUG
        ਲਈ (ਪੂਰਨ k = 0; k < nc; k++) {
            ਅਚਲ ggml_fp16_t x = ((ggml_fp16_t *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            ਅਚਲ ਦਸ਼ਮਲਵ v = GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x);
            GGML_ਅਣਵਰਤਿਆ(v);
            ਪ੍ਰਤਿਜ੍ਞਾ(!isnan(v));
            ਪ੍ਰਤਿਜ੍ਞਾ(!isinf(v));
        }
#ਜੇ_ਅੰਤ // NDEBUG
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_gelu_ਤੇਜ਼_bf16(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_ਪੰਕਤੀਆਂ<ggml_vec_gelu_ਤੇਜ਼_bf16>(params, dst);
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_gelu_ਤੇਜ਼(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_gelu_ਤੇਜ਼_f32(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F16:
            {
                ggml_compute_forward_gelu_ਤੇਜ਼_f16(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_BF16:
            {
                ggml_compute_forward_gelu_ਤੇਜ਼_bf16(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_silu

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_silu_f32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਪ੍ਰਤਿਜ੍ਞਾ(ggml_is_contiguous_ਪੰਕਤੀਆਂ(src0));
    ਪ੍ਰਤਿਜ੍ਞਾ(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ nc = src0->ne[0];
    ਅਚਲ ਪੂਰਨ nr = ggml_nrows(src0);

    // rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr*ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਲਈ (ਪੂਰਨ ir = ir0; ir < ir1; ++ir) {
        ਅਚਲ ਪੂਰਨ i3 = ir/(ne02*ne01);
        ਅਚਲ ਪੂਰਨ i2 = (ir - i3*ne02*ne01)/ne01;
        ਅਚਲ ਪੂਰਨ i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_silu_f32(nc,
                (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ  + i3*nb3  + i2*nb2  + i1*nb1),
                (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i3*nb03 + i2*nb02 + i1*nb01));

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NDEBUG
        ਲਈ (ਪੂਰਨ k = 0; k < nc; k++) {
            ਅਚਲ ਦਸ਼ਮਲਵ x = ((ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_ਅਣਵਰਤਿਆ(x);
            ਪ੍ਰਤਿਜ੍ਞਾ(!isnan(x));
            ਪ੍ਰਤਿਜ੍ਞਾ(!isinf(x));
        }
#ਜੇ_ਅੰਤ // NDEBUG
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_silu_f16(
    ਅਚਲ ggml_compute_params * params,
    ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਪ੍ਰਤਿਜ੍ਞਾ(ggml_is_contiguous_ਪੰਕਤੀਆਂ(src0));
    ਪ੍ਰਤਿਜ੍ਞਾ(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ nc = src0->ne[0];
    ਅਚਲ ਪੂਰਨ nr = ggml_nrows(src0);

    // rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr*ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਲਈ (ਪੂਰਨ ir = ir0; ir < ir1; ++ir) {
        ਅਚਲ ਪੂਰਨ i3 = ir/(ne02*ne01);
        ਅਚਲ ਪੂਰਨ i2 = (ir - i3*ne02*ne01)/ne01;
        ਅਚਲ ਪੂਰਨ i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_silu_f16(nc,
                (ggml_fp16_t *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i3*nb03 + i2*nb02 + i1*nb01));

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NDEBUG
        ਲਈ (ਪੂਰਨ k = 0; k < nc; k++) {
            ਅਚਲ ggml_fp16_t x = ((ggml_fp16_t *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            ਅਚਲ ਦਸ਼ਮਲਵ v = GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x);
            GGML_ਅਣਵਰਤਿਆ(v);
            ਪ੍ਰਤਿਜ੍ਞਾ(!isnan(v));
            ਪ੍ਰਤਿਜ੍ਞਾ(!isinf(v));
        }
#ਜੇ_ਅੰਤ // NDEBUG
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_silu_bf16(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_ਪੰਕਤੀਆਂ<ggml_vec_silu_bf16>(params, dst);
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_silu(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_silu_f32(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F16:
            {
                ggml_compute_forward_silu_f16(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_BF16:
            {
                ggml_compute_forward_silu_bf16(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}
// ggml_compute_forward_leaky_relu

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_leaky_relu_f32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਜੇ (params->ith != 0) {
        ਵਾਪਸ;
    }

    ਪ੍ਰਤਿਜ੍ਞਾ(ggml_is_contiguous_1(src0));
    ਪ੍ਰਤਿਜ੍ਞਾ(ggml_is_contiguous_1(dst));
    ਪ੍ਰਤਿਜ੍ਞਾ(ggml_are_same_shape(src0, dst));

    ਅਚਲ ਪੂਰਨ n  = ggml_nrows(src0);
    ਅਚਲ ਪੂਰਨ nc = src0->ne[0];

    ਦਸ਼ਮਲਵ negative_ਢਲਾਨ;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&negative_ਢਲਾਨ, dst->op_params, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਪ੍ਰਤਿਜ੍ਞਾ(dst->nb[0]  == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਪ੍ਰਤਿਜ੍ਞਾ(src0->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਲਈ (ਪੂਰਨ i = 0; i < n; i++) {
        ggml_vec_leaky_relu_f32(nc,
                (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ  + i*( dst->nb[1])),
                (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i*(src0->nb[1])), negative_ਢਲਾਨ);
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_leaky_relu_f16(
    ਅਚਲ ggml_compute_params * params,
    ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਜੇ (params->ith != 0) {
        ਵਾਪਸ;
    }

    ਪ੍ਰਤਿਜ੍ਞਾ(ggml_is_contiguous_1(src0));
    ਪ੍ਰਤਿਜ੍ਞਾ(ggml_is_contiguous_1(dst));
    ਪ੍ਰਤਿਜ੍ਞਾ(ggml_are_same_shape(src0, dst));

    ਅਚਲ ਪੂਰਨ n  = ggml_nrows(src0);
    ਅਚਲ ਪੂਰਨ nc = src0->ne[0];

    ਦਸ਼ਮਲਵ negative_ਢਲਾਨ;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&negative_ਢਲਾਨ, dst->op_params, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਪ੍ਰਤਿਜ੍ਞਾ(dst->nb[0]  == ਆਕਾਰ(ggml_fp16_t));
    ਪ੍ਰਤਿਜ੍ਞਾ(src0->nb[0] == ਆਕਾਰ(ggml_fp16_t));

    ਲਈ (ਪੂਰਨ i = 0; i < n; i++) {
        ggml_vec_leaky_relu_f16(nc,
                (ggml_fp16_t *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ  + i*( dst->nb[1])),
                (ggml_fp16_t *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i*(src0->nb[1])), negative_ਢਲਾਨ);
    }
}

ਖਾਲੀ ggml_compute_forward_leaky_relu(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_leaky_relu_f32(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F16:
            {
                ggml_compute_forward_leaky_relu_f16(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_silu_back

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_silu_ਪਿੱਠ_f32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * grad = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    ਪ੍ਰਤਿਜ੍ਞਾ(ggml_is_contiguous_1(grad));
    ਪ੍ਰਤਿਜ੍ਞਾ(ggml_is_contiguous_1(src1));
    ਪ੍ਰਤਿਜ੍ਞਾ(ggml_is_contiguous_1(dst));
    ਪ੍ਰਤਿਜ੍ਞਾ(ggml_are_same_shape(src1, dst));
    ਪ੍ਰਤਿਜ੍ਞਾ(ggml_are_same_shape(src1, grad));

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ nc = src1->ne[0];
    ਅਚਲ ਪੂਰਨ nr = ggml_nrows(src1);

    // rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr*ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਲਈ (ਪੂਰਨ i1 = ir0; i1 < ir1; i1++) {
        ggml_vec_silu_backward_f32(nc,
                (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ  + i1*( dst->nb[1])),
                (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src1->ਅੰਕੜੇ + i1*(src1->nb[1])),
                (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) grad->ਅੰਕੜੇ + i1*(grad->nb[1])));

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NDEBUG
        ਲਈ (ਪੂਰਨ k = 0; k < nc; k++) {
            ਅਚਲ ਦਸ਼ਮਲਵ x = ((ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i1*( dst->nb[1])))[k];
            GGML_ਅਣਵਰਤਿਆ(x);
            ਪ੍ਰਤਿਜ੍ਞਾ(!isnan(x));
            ਪ੍ਰਤਿਜ੍ਞਾ(!isinf(x));
        }
#ਜੇ_ਅੰਤ // NDEBUG
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_silu_ਪਿੱਠ_f16(
    ਅਚਲ ggml_compute_params * params,
    ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * grad = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    ਪ੍ਰਤਿਜ੍ਞਾ(ggml_is_contiguous_1(grad));
    ਪ੍ਰਤਿਜ੍ਞਾ(ggml_is_contiguous_1(src1));
    ਪ੍ਰਤਿਜ੍ਞਾ(ggml_is_contiguous_1(dst));
    ਪ੍ਰਤਿਜ੍ਞਾ(ggml_are_same_shape(src1, dst));
    ਪ੍ਰਤਿਜ੍ਞਾ(ggml_are_same_shape(src1, grad));

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ nc = src1->ne[0];
    ਅਚਲ ਪੂਰਨ nr = ggml_nrows(src1);

    // rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr*ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਲਈ (ਪੂਰਨ i1 = ir0; i1 < ir1; i1++) {
        ggml_vec_silu_backward_f16(nc,
                (ggml_fp16_t *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ  + i1*( dst->nb[1])),
                (ggml_fp16_t *) ((ਅੱਖਰ *) src1->ਅੰਕੜੇ + i1*(src1->nb[1])),
                (ggml_fp16_t *) ((ਅੱਖਰ *) grad->ਅੰਕੜੇ + i1*(grad->nb[1])));

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NDEBUG
        ਲਈ (ਪੂਰਨ k = 0; k < nc; k++) {
            ਅਚਲ ਦਸ਼ਮਲਵ x = ((ggml_fp16_t *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i1*( dst->nb[1])))[k];
            ਅਚਲ ਦਸ਼ਮਲਵ v = GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x);
            GGML_ਅਣਵਰਤਿਆ(v);
            ਪ੍ਰਤਿਜ੍ਞਾ(!isnan(v));
            ਪ੍ਰਤਿਜ੍ਞਾ(!isinf(v));
        }
#ਜੇ_ਅੰਤ // NDEBUG
    }
}

ਖਾਲੀ ggml_compute_forward_silu_ਪਿੱਠ(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_silu_ਪਿੱਠ_f32(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F16:
            {
                ggml_compute_forward_silu_ਪਿੱਠ_f16(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_reglu

// row loop shared by the BF16 GLU ops
ਸਾਂਚਾ <ਖਾਲੀ (*vec_op)(ਅਚਲ ਪੂਰਨ, ggml_bf16_t *, ਅਚਲ ggml_bf16_t *, ਅਚਲ ggml_bf16_t *)>
ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_glu_bf16_ਪੰਕਤੀਆਂ(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];
    ਅੱਖਰ * src0_d = (ਅੱਖਰ *) src0->ਅੰਕੜੇ;
    ਅੱਖਰ * src1_d = (ਅੱਖਰ *) (src1 ? src1->ਅੰਕੜੇ : src0->ਅੰਕੜੇ);
    ਅਚਲ size_t src0_o = src0->nb[1];
    ਅਚਲ size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ਜੇ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ਅਚਲ ਪੂਰਨ nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ਅਚਲ int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr*ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਲਈ (ਪੂਰਨ i1 = ir0; i1 < ir1; i1++) {
        ਅਚਲ ggml_bf16_t * src0_p = (ਅਚਲ ggml_bf16_t *) (src0_d + i1*src0_o);
        ਅਚਲ ggml_bf16_t * src1_p = (ਅਚਲ ggml_bf16_t *) (src1_d + i1*src1_o);
        ggml_bf16_t       * dst_p  = (ggml_bf16_t *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i1*(dst->nb[1]));

        ਜੇ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        vec_op(nc, dst_p, src0_p, src1_p);
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_reglu_f32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];
    ਅੱਖਰ * src0_d = (ਅੱਖਰ *) src0->ਅੰਕੜੇ;
    ਅੱਖਰ * src1_d = (ਅੱਖਰ *) (src1 ? src1->ਅੰਕੜੇ : src0->ਅੰਕੜੇ);
    ਅਚਲ size_t src0_o = src0->nb[1];
    ਅਚਲ size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ਜੇ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ਅਚਲ ਪੂਰਨ nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ਅਚਲ int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr*ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਲਈ (ਪੂਰਨ i1 = ir0; i1 < ir1; i1++) {
        ਦਸ਼ਮਲਵ * src0_p = (ਦਸ਼ਮਲਵ *) (src0_d + i1*src0_o);
        ਦਸ਼ਮਲਵ * src1_p = (ਦਸ਼ਮਲਵ *) (src1_d + i1*src1_o);

        ਜੇ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_reglu_f32(nc, (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i1*(dst->nb[1])), src0_p, src1_p);

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NDEBUG
        ਲਈ (ਪੂਰਨ k = 0; k < nc; k++) {
            ਅਚਲ ਦਸ਼ਮਲਵ x = ((ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i1*( dst->nb[1])))[k];
            GGML_ਅਣਵਰਤਿਆ(x);
            ਪ੍ਰਤਿਜ੍ਞਾ(!isnan(x));
            ਪ੍ਰਤਿਜ੍ਞਾ(!isinf(x));
        }
#ਜੇ_ਅੰਤ // NDEBUG
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_reglu_f16(
    ਅਚਲ ggml_compute_params * params,
    ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];
    ਅੱਖਰ * src0_d = (ਅੱਖਰ *) src0->ਅੰਕੜੇ;
    ਅੱਖਰ * src1_d = (ਅੱਖਰ *) (src1 ? src1->ਅੰਕੜੇ : src0->ਅੰਕੜੇ);
    ਅਚਲ size_t src0_o = src0->nb[1];
    ਅਚਲ size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ਜੇ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ਅਚਲ ਪੂਰਨ nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ਅਚਲ int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr*ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਲਈ (ਪੂਰਨ i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        ਜੇ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_reglu_f16(nc, (ggml_fp16_t *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i1*(dst->nb[1])), src0_p, src1_p);

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NDEBUG
        ਲਈ (ਪੂਰਨ k = 0; k < nc; k++) {
            ਅਚਲ ggml_fp16_t x = ((ggml_fp16_t *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i1*( dst->nb[1])))[k];
            ਅਚਲ ਦਸ਼ਮਲਵ v = GGML_FP16_ਮੰਜ਼ਿਲ_FP32(x);
            GGML_ਅਣਵਰਤਿਆ(v);
            ਪ੍ਰਤਿਜ੍ਞਾ(!isnan(v));
            ਪ੍ਰਤਿਜ੍ਞਾ(!isinf(v));
        }
#ਜੇ_ਅੰਤ // NDEBUG
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_reglu_bf16(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_ਪੰਕਤੀਆਂ<ggml_vec_reglu_bf16>(params, dst);
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_reglu(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_reglu_f32(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F16:
            {
                ggml_compute_forward_reglu_f16(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_BF16:
            {
                ggml_compute_forward_reglu_bf16(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_geglu

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_geglu_f32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];
    ਅੱਖਰ * src0_d = (ਅੱਖਰ *) src0->ਅੰਕੜੇ;
    ਅੱਖਰ * src1_d = (ਅੱਖਰ *) (src1 ? src1->ਅੰਕੜੇ : src0->ਅੰਕੜੇ);
    ਅਚਲ size_t src0_o = src0->nb[1];
    ਅਚਲ size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ਜੇ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ਅਚਲ ਪੂਰਨ nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ਅਚਲ int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr*ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਲਈ (ਪੂਰਨ i1 = ir0; i1 < ir1; i1++) {
        ਦਸ਼ਮਲਵ * src0_p = (ਦਸ਼ਮਲਵ *) (src0_d + i1*src0_o);
        ਦਸ਼ਮਲਵ * src1_p = (ਦਸ਼ਮਲਵ *) (src1_d + i1*src1_o);

        ਜੇ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_f32(nc, (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i1*(dst->nb[1])), src0_p, src1_p);

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NDEBUG
        ਲਈ (ਪੂਰਨ k = 0; k < nc; k++) {
            ਅਚਲ ਦਸ਼ਮਲਵ x = ((ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i1*( dst->nb[1])))[k];
            GGML_ਅਣਵਰਤਿਆ(x);
            ਪ੍ਰਤਿਜ੍ਞਾ(!isnan(x));
            ਪ੍ਰਤਿਜ੍ਞਾ(!isinf(x));
        }
#ਜੇ_ਅੰਤ // NDEBUG
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_geglu_f16(
    ਅਚਲ ggml_compute_params * params,
    ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];
    ਅੱਖਰ * src0_d = (ਅੱਖਰ *) src0->ਅੰਕੜੇ;
    ਅੱਖਰ * src1_d = (ਅੱਖਰ *) (src1 ? src1->ਅੰਕੜੇ : src0->ਅੰਕੜੇ);
    ਅਚਲ size_t src0_o = src0->nb[1];
    ਅਚਲ size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ਜੇ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ਅਚਲ ਪੂਰਨ nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ਅਚਲ int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr*ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਲਈ (ਪੂਰਨ i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        ਜੇ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_f16(nc, (ggml_fp16_t *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i1*(dst->nb[1])), src0_p, src1_p);

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NDEBUG
        ਲਈ (ਪੂਰਨ k = 0; k < nc; k++) {
            ਅਚਲ ggml_fp16_t x = ((ggml_fp16_t *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i1*( dst->nb[1])))[k];
            ਅਚਲ ਦਸ਼ਮਲਵ v = GGML_FP16_ਮੰਜ਼ਿਲ_FP32(x);
            GGML_ਅਣਵਰਤਿਆ(v);
            ਪ੍ਰਤਿਜ੍ਞਾ(!isnan(v));
            ਪ੍ਰਤਿਜ੍ਞਾ(!isinf(v));
        }
#ਜੇ_ਅੰਤ // NDEBUG
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_geglu_bf16(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_ਪੰਕਤੀਆਂ<ggml_vec_geglu_bf16>(params, dst);
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_geglu(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_geglu_f32(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F16:
            {
                ggml_compute_forward_geglu_f16(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_BF16:
            {
                ggml_compute_forward_geglu_bf16(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_swiglu

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_swiglu_f32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];
    ਅੱਖਰ * src0_d = (ਅੱਖਰ *) src0->ਅੰਕੜੇ;
    ਅੱਖਰ * src1_d = (ਅੱਖਰ *) (src1 ? src1->ਅੰਕੜੇ : src0->ਅੰਕੜੇ);
    ਅਚਲ size_t src0_o = src0->nb[1];
    ਅਚਲ size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ਜੇ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ਅਚਲ ਪੂਰਨ nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ਅਚਲ int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr*ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਲਈ (ਪੂਰਨ i1 = ir0; i1 < ir1; i1++) {
        ਦਸ਼ਮਲਵ * src0_p = (ਦਸ਼ਮਲਵ *) (src0_d + i1*src0_o);
        ਦਸ਼ਮਲਵ * src1_p = (ਦਸ਼ਮਲਵ *) (src1_d + i1*src1_o);

        ਜੇ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_swiglu_f32(nc, (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i1*(dst->nb[1])), src0_p, src1_p);

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NDEBUG
        ਲਈ (ਪੂਰਨ k = 0; k < nc; k++) {
            ਅਚਲ ਦਸ਼ਮਲਵ x = ((ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i1*( dst->nb[1])))[k];
            GGML_ਅਣਵਰਤਿਆ(x);
            ਪ੍ਰਤਿਜ੍ਞਾ(!isnan(x));
            ਪ੍ਰਤਿਜ੍ਞਾ(!isinf(x));
        }
#ਜੇ_ਅੰਤ // NDEBUG
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_swiglu_f16(
    ਅਚਲ ggml_compute_params * params,
    ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];
    ਅੱਖਰ * src0_d = (ਅੱਖਰ *) src0->ਅੰਕੜੇ;
    ਅੱਖਰ * src1_d = (ਅੱਖਰ *) (src1 ? src1->ਅੰਕੜੇ : src0->ਅੰਕੜੇ);
    ਅਚਲ size_t src0_o = src0->nb[1];
    ਅਚਲ size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ਜੇ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ਅਚਲ ਪੂਰਨ nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ਅਚਲ int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr*ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਲਈ (ਪੂਰਨ i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        ਜੇ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_swiglu_f16(nc, (ggml_fp16_t *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i1*(dst->nb[1])), src0_p, src1_p);

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NDEBUG
        ਲਈ (ਪੂਰਨ k = 0; k < nc; k++) {
            ਅਚਲ ggml_fp16_t x = ((ggml_fp16_t *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i1*( dst->nb[1])))[k];
            ਅਚਲ ਦਸ਼ਮਲਵ v = GGML_FP16_ਮੰਜ਼ਿਲ_FP32(x);
            GGML_ਅਣਵਰਤਿਆ(v);
            ਪ੍ਰਤਿਜ੍ਞਾ(!isnan(v));
            ਪ੍ਰਤਿਜ੍ਞਾ(!isinf(v));
        }
#ਜੇ_ਅੰਤ // NDEBUG
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_swiglu_bf16(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_ਪੰਕਤੀਆਂ<ggml_vec_swiglu_bf16>(params, dst);
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_swiglu(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_swiglu_f32(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F16:
            {
                ggml_compute_forward_swiglu_f16(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_BF16:
            {
                ggml_compute_forward_swiglu_bf16(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_swiglu_oai

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_swiglu_oai_f32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];
    ਅੱਖਰ * src0_d = (ਅੱਖਰ *) src0->ਅੰਕੜੇ;
    ਅੱਖਰ * src1_d = (ਅੱਖਰ *) (src1 ? src1->ਅੰਕੜੇ : src0->ਅੰਕੜੇ);
    ਅਚਲ size_t src0_o = src0->nb[1];
    ਅਚਲ size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ਜੇ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ਅਚਲ ਪੂਰਨ nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ਅਚਲ int32_t swapped = ggml_get_op_params_i32(dst, 1);
    ਅਚਲ ਦਸ਼ਮਲਵ alpha = ggml_get_op_params_f32(dst, 2);
    ਅਚਲ ਦਸ਼ਮਲਵ ਹੱਦਬੰਦੀ = ggml_get_op_params_f32(dst, 3);

    // rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr*ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਲਈ (ਪੂਰਨ i1 = ir0; i1 < ir1; i1++) {
        ਦਸ਼ਮਲਵ * src0_p = (ਦਸ਼ਮਲਵ *) (src0_d + i1*src0_o);
        ਦਸ਼ਮਲਵ * src1_p = (ਦਸ਼ਮਲਵ *) (src1_d + i1*src1_o);
        ਦਸ਼ਮਲਵ * dst_p  = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i1*(dst->nb[1]));

        ਜੇ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ਲਈ (ਪੂਰਨ k = 0; k < nc; k++) {
            ਅਚਲ ਦਸ਼ਮਲਵ x = std::min(src0_p[k], ਹੱਦਬੰਦੀ);
            ਅਚਲ ਦਸ਼ਮਲਵ y = std::clamp(src1_p[k], -ਹੱਦਬੰਦੀ, ਹੱਦਬੰਦੀ);
            ਅਚਲ ਦਸ਼ਮਲਵ ਬਾਹਰਲਾ_glu = x / (1.f + expf(alpha * (-x)));
            dst_p[k] = ਬਾਹਰਲਾ_glu * (y + 1.f);
        }

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NDEBUG
        ਲਈ (ਪੂਰਨ k = 0; k < nc; k++) {
            ਅਚਲ ਦਸ਼ਮਲਵ x = dst_p[k];
            GGML_ਅਣਵਰਤਿਆ(x);
            ਪ੍ਰਤਿਜ੍ਞਾ(!isnan(x));
            ਪ੍ਰਤਿਜ੍ਞਾ(!isinf(x));
        }
#ਜੇ_ਅੰਤ // NDEBUG
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_swiglu_oai_f16(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];
    ਅੱਖਰ * src0_d = (ਅੱਖਰ *) src0->ਅੰਕੜੇ;
    ਅੱਖਰ * src1_d = (ਅੱਖਰ *) (src1 ? src1->ਅੰਕੜੇ : src0->ਅੰਕੜੇ);
    ਅਚਲ size_t src0_o = src0->nb[1];
    ਅਚਲ size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ਜੇ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ਅਚਲ ਪੂਰਨ nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ਅਚਲ int32_t swapped = ggml_get_op_params_i32(dst, 1);
    ਅਚਲ ਦਸ਼ਮਲਵ alpha = ggml_get_op_params_f32(dst, 2);
    ਅਚਲ ਦਸ਼ਮਲਵ ਹੱਦਬੰਦੀ = ggml_get_op_params_f32(dst, 3);

    // rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr*ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਲਈ (ਪੂਰਨ i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);
        ggml_fp16_t * dst_p  = (ggml_fp16_t *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i1*(dst->nb[1]));

        ਜੇ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ਲਈ (ਪੂਰਨ k = 0; k < nc; k++) {
            ਅਚਲ ਦਸ਼ਮਲਵ x = std::min(GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(src0_p[k]), ਹੱਦਬੰਦੀ);
            ਅਚਲ ਦਸ਼ਮਲਵ y = std::clamp(GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(src1_p[k]), -ਹੱਦਬੰਦੀ, ਹੱਦਬੰਦੀ);
            ਅਚਲ ਦਸ਼ਮਲਵ ਬਾਹਰਲਾ_glu = x / (1.f + expf(alpha * (-x)));
            dst_p[k] = GGML_CPU_FP32_ਮੰਜ਼ਿਲ_FP16(ਬਾਹਰਲਾ_glu * (y + 1.f));
        }

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NDEBUG
        ਲਈ (ਪੂਰਨ k = 0; k < nc; k++) {
            ਅਚਲ ਦਸ਼ਮਲਵ x = GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(dst_p[k]);
            GGML_ਅਣਵਰਤਿਆ(x);
            ਪ੍ਰਤਿਜ੍ਞਾ(!isnan(x));
            ਪ੍ਰਤਿਜ੍ਞਾ(!isinf(x));
        }
#ਜੇ_ਅੰਤ // NDEBUG
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_swiglu_oai(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_swiglu_oai_f32(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F16:
            {
                ggml_compute_forward_swiglu_oai_f16(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_swiglu_clamp

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_swiglu_clamp_f32(ਅਚਲ ggml_compute_params * params, ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * src0   = dst->src[0];
    ਅਚਲ ggml_tensor * src1   = dst->src[1];
    ਅੱਖਰ *              src0_d = (ਅੱਖਰ *) src0->ਅੰਕੜੇ;
    ਅੱਖਰ *              src1_d = (ਅੱਖਰ *) (src1 ? src1->ਅੰਕੜੇ : src0->ਅੰਕੜੇ);
    ਅਚਲ size_t        src0_o = src0->nb[1];
    ਅਚਲ size_t        src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ਜੇ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ਅਚਲ ਪੂਰਨ nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ਅਚਲ int32_t swapped = ggml_get_op_params_i32(dst, 1);
    ਅਚਲ ਦਸ਼ਮਲਵ   ਹੱਦਬੰਦੀ   = ggml_get_op_params_f32(dst, 3);

    ਅਚਲ ਪੂਰਨ dr  = (nr + nth - 1) / nth;
    ਅਚਲ ਪੂਰਨ ir0 = dr * ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਲਈ (ਪੂਰਨ i1 = ir0; i1 < ir1; i1++) {
        ਦਸ਼ਮਲਵ * src0_p = (ਦਸ਼ਮਲਵ *) (src0_d + i1 * src0_o);
        ਦਸ਼ਮਲਵ * src1_p = (ਦਸ਼ਮਲਵ *) (src1_d + i1 * src1_o);
        ਦਸ਼ਮਲਵ * dst_p  = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i1 * (dst->nb[1]));

        ਜੇ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ਲਈ (ਪੂਰਨ k = 0; k < nc; k++) {
            ਅਚਲ ਦਸ਼ਮਲਵ gate = std::min(src0_p[k], ਹੱਦਬੰਦੀ);
            ਅਚਲ ਦਸ਼ਮਲਵ ਉੱਪਰ   = std::clamp(src1_p[k], -ਹੱਦਬੰਦੀ, ਹੱਦਬੰਦੀ);
            dst_p[k]         = gate / (1.f + expf(-gate)) * ਉੱਪਰ;
        }

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NDEBUG
        ਲਈ (ਪੂਰਨ k = 0; k < nc; k++) {
            ਅਚਲ ਦਸ਼ਮਲਵ x = dst_p[k];
            GGML_ਅਣਵਰਤਿਆ(x);
            ਪ੍ਰਤਿਜ੍ਞਾ(!isnan(x));
            ਪ੍ਰਤਿਜ੍ਞਾ(!isinf(x));
        }
#ਜੇ_ਅੰਤ  // NDEBUG
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_swiglu_clamp_f16(ਅਚਲ ggml_compute_params * params, ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * src0   = dst->src[0];
    ਅਚਲ ggml_tensor * src1   = dst->src[1];
    ਅੱਖਰ *              src0_d = (ਅੱਖਰ *) src0->ਅੰਕੜੇ;
    ਅੱਖਰ *              src1_d = (ਅੱਖਰ *) (src1 ? src1->ਅੰਕੜੇ : src0->ਅੰਕੜੇ);
    ਅਚਲ size_t        src0_o = src0->nb[1];
    ਅਚਲ size_t        src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ਜੇ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ਅਚਲ ਪੂਰਨ nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ਅਚਲ int32_t swapped = ggml_get_op_params_i32(dst, 1);
    ਅਚਲ ਦਸ਼ਮਲਵ   ਹੱਦਬੰਦੀ   = ggml_get_op_params_f32(dst, 3);

    ਅਚਲ ਪੂਰਨ dr  = (nr + nth - 1) / nth;
    ਅਚਲ ਪੂਰਨ ir0 = dr * ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਲਈ (ਪੂਰਨ i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1 * src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1 * src1_o);
        ggml_fp16_t * dst_p  = (ggml_fp16_t *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i1 * (dst->nb[1]));

        ਜੇ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ਲਈ (ਪੂਰਨ k = 0; k < nc; k++) {
            ਅਚਲ ਦਸ਼ਮਲਵ gate = std::min(GGML_FP16_ਮੰਜ਼ਿਲ_FP32(src0_p[k]), ਹੱਦਬੰਦੀ);
            ਅਚਲ ਦਸ਼ਮਲਵ ਉੱਪਰ   = std::clamp(GGML_FP16_ਮੰਜ਼ਿਲ_FP32(src1_p[k]), -ਹੱਦਬੰਦੀ, ਹੱਦਬੰਦੀ);
            dst_p[k]         = GGML_FP32_ਮੰਜ਼ਿਲ_FP16(gate / (1.f + expf(-gate)) * ਉੱਪਰ);
        }

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NDEBUG
        ਲਈ (ਪੂਰਨ k = 0; k < nc; k++) {
            ਅਚਲ ਦਸ਼ਮਲਵ x = GGML_FP16_ਮੰਜ਼ਿਲ_FP32(dst_p[k]);
            GGML_ਅਣਵਰਤਿਆ(x);
            ਪ੍ਰਤਿਜ੍ਞਾ(!isnan(x));
            ਪ੍ਰਤਿਜ੍ਞਾ(!isinf(x));
        }
#ਜੇ_ਅੰਤ  // NDEBUG
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_swiglu_clamp(ਅਚਲ ggml_compute_params * params, ggml_tensor * dst) {
    ਚੋਣ (dst->src[0]->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            ggml_compute_forward_swiglu_clamp_f32(params, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F16:
            ggml_compute_forward_swiglu_clamp_f16(params, dst);
            ਰੋਕੋ;
        ਮੂਲ:
            GGML_ABORT("fatal error");
    }
}

// ggml_compute_forward_geglu_erf

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_geglu_erf_f32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];
    ਅੱਖਰ * src0_d = (ਅੱਖਰ *) src0->ਅੰਕੜੇ;
    ਅੱਖਰ * src1_d = (ਅੱਖਰ *) (src1 ? src1->ਅੰਕੜੇ : src0->ਅੰਕੜੇ);
    ਅਚਲ size_t src0_o = src0->nb[1];
    ਅਚਲ size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ਜੇ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ਅਚਲ ਪੂਰਨ nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ਅਚਲ int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr*ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਲਈ (ਪੂਰਨ i1 = ir0; i1 < ir1; i1++) {
        ਦਸ਼ਮਲਵ * src0_p = (ਦਸ਼ਮਲਵ *) (src0_d + i1*src0_o);
        ਦਸ਼ਮਲਵ * src1_p = (ਦਸ਼ਮਲਵ *) (src1_d + i1*src1_o);

        ਜੇ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_erf_f32(nc, (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i1*(dst->nb[1])), src0_p, src1_p);

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NDEBUG
        ਲਈ (ਪੂਰਨ k = 0; k < nc; k++) {
            ਅਚਲ ਦਸ਼ਮਲਵ x = ((ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i1*( dst->nb[1])))[k];
            GGML_ਅਣਵਰਤਿਆ(x);
            ਪ੍ਰਤਿਜ੍ਞਾ(!isnan(x));
            ਪ੍ਰਤਿਜ੍ਞਾ(!isinf(x));
        }
#ਜੇ_ਅੰਤ // NDEBUG
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_geglu_erf_f16(
    ਅਚਲ ggml_compute_params * params,
    ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];
    ਅੱਖਰ * src0_d = (ਅੱਖਰ *) src0->ਅੰਕੜੇ;
    ਅੱਖਰ * src1_d = (ਅੱਖਰ *) (src1 ? src1->ਅੰਕੜੇ : src0->ਅੰਕੜੇ);
    ਅਚਲ size_t src0_o = src0->nb[1];
    ਅਚਲ size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ਜੇ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ਅਚਲ ਪੂਰਨ nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ਅਚਲ int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr*ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਲਈ (ਪੂਰਨ i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        ਜੇ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_erf_f16(nc, (ggml_fp16_t *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i1*(dst->nb[1])), src0_p, src1_p);

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NDEBUG
        ਲਈ (ਪੂਰਨ k = 0; k < nc; k++) {
            ਅਚਲ ggml_fp16_t x = ((ggml_fp16_t *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i1*( dst->nb[1])))[k];
            ਅਚਲ ਦਸ਼ਮਲਵ v = GGML_FP16_ਮੰਜ਼ਿਲ_FP32(x);
            GGML_ਅਣਵਰਤਿਆ(v);
            ਪ੍ਰਤਿਜ੍ਞਾ(!isnan(v));
            ਪ੍ਰਤਿਜ੍ਞਾ(!isinf(v));
        }
#ਜੇ_ਅੰਤ // NDEBUG
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_geglu_erf_bf16(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_ਪੰਕਤੀਆਂ<ggml_vec_geglu_erf_bf16>(params, dst);
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_geglu_erf(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_geglu_erf_f32(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F16:
            {
                ggml_compute_forward_geglu_erf_f16(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_BF16:
            {
                ggml_compute_forward_geglu_erf_bf16(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_geglu_quick

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_geglu_ਤੇਜ਼_f32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];
    ਅੱਖਰ * src0_d = (ਅੱਖਰ *) src0->ਅੰਕੜੇ;
    ਅੱਖਰ * src1_d = (ਅੱਖਰ *) (src1 ? src1->ਅੰਕੜੇ : src0->ਅੰਕੜੇ);
    ਅਚਲ size_t src0_o = src0->nb[1];
    ਅਚਲ size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ਜੇ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ਅਚਲ ਪੂਰਨ nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ਅਚਲ int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr*ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਲਈ (ਪੂਰਨ i1 = ir0; i1 < ir1; i1++) {
        ਦਸ਼ਮਲਵ * src0_p = (ਦਸ਼ਮਲਵ *) (src0_d + i1*src0_o);
        ਦਸ਼ਮਲਵ * src1_p = (ਦਸ਼ਮਲਵ *) (src1_d + i1*src1_o);

        ਜੇ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_ਤੇਜ਼_f32(nc, (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i1*(dst->nb[1])), src0_p, src1_p);

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NDEBUG
        ਲਈ (ਪੂਰਨ k = 0; k < nc; k++) {
            ਅਚਲ ਦਸ਼ਮਲਵ x = ((ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i1*( dst->nb[1])))[k];
            GGML_ਅਣਵਰਤਿਆ(x);
            ਪ੍ਰਤਿਜ੍ਞਾ(!isnan(x));
            ਪ੍ਰਤਿਜ੍ਞਾ(!isinf(x));
        }
#ਜੇ_ਅੰਤ // NDEBUG
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_geglu_ਤੇਜ਼_f16(
    ਅਚਲ ggml_compute_params * params,
    ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];
    ਅੱਖਰ * src0_d = (ਅੱਖਰ *) src0->ਅੰਕੜੇ;
    ਅੱਖਰ * src1_d = (ਅੱਖਰ *) (src1 ? src1->ਅੰਕੜੇ : src0->ਅੰਕੜੇ);
    ਅਚਲ size_t src0_o = src0->nb[1];
    ਅਚਲ size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ਜੇ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ਅਚਲ ਪੂਰਨ nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ਅਚਲ int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr*ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਲਈ (ਪੂਰਨ i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        ਜੇ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_ਤੇਜ਼_f16(nc, (ggml_fp16_t *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i1*(dst->nb[1])), src0_p, src1_p);

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NDEBUG
        ਲਈ (ਪੂਰਨ k = 0; k < nc; k++) {
            ਅਚਲ ggml_fp16_t x = ((ggml_fp16_t *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i1*( dst->nb[1])))[k];
            ਅਚਲ ਦਸ਼ਮਲਵ v = GGML_FP16_ਮੰਜ਼ਿਲ_FP32(x);
            GGML_ਅਣਵਰਤਿਆ(v);
            ਪ੍ਰਤਿਜ੍ਞਾ(!isnan(v));
            ਪ੍ਰਤਿਜ੍ਞਾ(!isinf(v));
        }
#ਜੇ_ਅੰਤ // NDEBUG
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_geglu_ਤੇਜ਼_bf16(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_ਪੰਕਤੀਆਂ<ggml_vec_geglu_ਤੇਜ਼_bf16>(params, dst);
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_geglu_ਤੇਜ਼(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_geglu_ਤੇਜ਼_f32(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F16:
            {
                ggml_compute_forward_geglu_ਤੇਜ਼_f16(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_BF16:
            {
                ggml_compute_forward_geglu_ਤੇਜ਼_bf16(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_norm

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_norm_f32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    ਦਸ਼ਮਲਵ eps;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&eps, dst->op_params, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    GGML_ASSERT(eps >= 0.0f);

    ਲਈ (int64_t i03 = 0; i03 < ne03; i03++) {
        ਲਈ (int64_t i02 = 0; i02 < ne02; i02++) {
            ਲਈ (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                ਅਚਲ ਅੱਖਰ * x = (ਅਚਲ ਅੱਖਰ *) src0->ਅੰਕੜੇ + i01*nb01 + i02*nb02 + i03*nb03;
                ਅੱਖਰ * y = (ਅੱਖਰ *) dst->ਅੰਕੜੇ + i01*nb1 + i02*nb2 + i03*nb3;

                ਜੇ (nb00 == ਆਕਾਰ(ਦਸ਼ਮਲਵ) && nb0 == ਆਕਾਰ(ਦਸ਼ਮਲਵ)) {
                    ਅਚਲ ਦਸ਼ਮਲਵ * xf = (ਅਚਲ ਦਸ਼ਮਲਵ *) x;

                    ਦਸ਼ਮਲਵ ਜੋੜ = 0.0;
                    ggml_vec_ਜੋੜ_f32(ne00, &ਜੋੜ, xf);
                    ਦਸ਼ਮਲਵ ਔਸਤ_2 = ਜੋੜ/ne00;

                    ਦਸ਼ਮਲਵ * yf = (ਦਸ਼ਮਲਵ *) y;
                    ਦਸ਼ਮਲਵ variance = 0;

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_ACCELERATE
                    ਔਸਤ_2 = -ਔਸਤ_2;
                    vDSP_vsadd(xf, 1, &ਔਸਤ_2, yf, 1, ne00);
                    vDSP_measqv(yf, 1, &variance, ne00);
#ਨਹੀਂਤਾਂ
                    variance = ggml_vec_cvar_f32(ne00, yf, xf, ਔਸਤ_2);
#ਜੇ_ਅੰਤ //GGML_USE_ACCELERATE

                    ਅਚਲ ਦਸ਼ਮਲਵ scale = 1.0f/sqrtf(variance + eps);
                    ggml_vec_scale_f32(ne00, yf, scale);
                } ਨਹੀਂਤਾਂ {
                    ਦਸ਼ਮਲਵ ਜੋੜ = 0.0;
                    ਲਈ (int64_t i00 = 0; i00 < ne00; i00++) {
                        ਜੋੜ += *(ਅਚਲ ਦਸ਼ਮਲਵ *) (x + i00*nb00);
                    }
                    ਅਚਲ ਦਸ਼ਮਲਵ ਔਸਤ_2 = ਜੋੜ/ne00;

                    ਦਸ਼ਮਲਵ variance = 0.0f;
                    ਲਈ (int64_t i00 = 0; i00 < ne00; i00++) {
                        ਅਚਲ ਦਸ਼ਮਲਵ v = *(ਅਚਲ ਦਸ਼ਮਲਵ *) (x + i00*nb00) - ਔਸਤ_2;
                        *(ਦਸ਼ਮਲਵ *) (y + i00*nb0) = v;
                        variance += v * v;
                    }
                    variance /= ne00;

                    ਅਚਲ ਦਸ਼ਮਲਵ scale = 1.0f/sqrtf(variance + eps);
                    ਲਈ (int64_t i00 = 0; i00 < ne00; i00++) {
                        *(ਦਸ਼ਮਲਵ *) (y + i00*nb0) *= scale;
                    }
                }
            }
        }
    }
}

ਖਾਲੀ ggml_compute_forward_norm(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_norm_f32(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_group_rms_norm

// fusion kinds that can be combined with the rms_norm computation in a single pass.
// extend this enum when adding new fused variants (e.g. FUSE_ADD, FUSE_MUL_ADD, ...).
ਗਿਣਤੀ ggml_rms_norm_fuse_op {
    GGML_RMS_NORM_FUSE_OP_NONE,
    GGML_RMS_NORM_FUSE_OP_MUL,
};

ਸਾਂਚਾ <ggml_rms_norm_fuse_op FUSE_OP>
ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_rms_norm_f32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst_rms_norm,
        ggml_tensor * dst_fused = ਖਾਲੀ_ਸੂਚਕ) {

    ਅਚਲ ggml_tensor * src0 = dst_rms_norm->src[0];
    ਅਚਲ ggml_tensor * src1 = ਖਾਲੀ_ਸੂਚਕ;
    ggml_tensor       * dst  = dst_rms_norm;

    ਜੇ constexpr (FUSE_OP == GGML_RMS_NORM_FUSE_OP_MUL) {
        src1 = (dst_fused->src[0] == dst_rms_norm) ? dst_fused->src[1] : dst_fused->src[0];
        dst  = dst_fused;
    }

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    GGML_ASSERT(src0->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

    ਦਸ਼ਮਲਵ eps;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&eps, dst_rms_norm->op_params, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    GGML_ASSERT(eps >= 0.0f);

    // TODO: optimize
    ਲਈ (int64_t i03 = 0; i03 < ne03; i03++) {
        ਲਈ (int64_t i02 = 0; i02 < ne02; i02++) {
            ਲਈ (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                ਅਚਲ ਦਸ਼ਮਲਵ * x = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i01*nb01 + i02*nb02 + i03*nb03);

                ggml_float ਜੋੜ = 0.0;
                // worth switching to explicit SIMD?
                ਲਈ (int64_t i00 = 0; i00 < ne00; i00++) {
                    ਜੋੜ += (ggml_float)(x[i00] * x[i00]);
                }

                ਅਚਲ ਦਸ਼ਮਲਵ ਔਸਤ_2  = ਜੋੜ/ne00;
                ਅਚਲ ਦਸ਼ਮਲਵ scale = 1.0f/sqrtf(ਔਸਤ_2 + eps);

                // if you hit this, likely you got an inf somewhere earlier
                ਪ੍ਰਤਿਜ੍ਞਾ(scale > 0.0f);

                ਦਸ਼ਮਲਵ * y = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i01*nb1 + i02*nb2 + i03*nb3);

                ਜੇ constexpr (FUSE_OP == GGML_RMS_NORM_FUSE_OP_MUL) {
                    ਅਚਲ int64_t i11 = i01 % ne11;
                    ਅਚਲ int64_t i12 = i02 % ne12;
                    ਅਚਲ int64_t i13 = i03 % ne13;
                    ਅਚਲ ਦਸ਼ਮਲਵ * w = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src1->ਅੰਕੜੇ + i11*nb11 + i12*nb12 + i13*nb13);

                    ਲਈ (int64_t i00 = 0; i00 < ne00; i00++) {
                        y[i00] = x[i00] * scale * w[i00];
                    }
                } ਨਹੀਂਤਾਂ {
                    ਸਮ੍ਰਿਤੀ_ਨਕਲ(y, x, ne00 * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                    ggml_vec_scale_f32(ne00, y, scale);
                }
            }
        }
    }
}

ਖਾਲੀ ggml_compute_forward_rms_norm(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_rms_norm_f32<GGML_RMS_NORM_FUSE_OP_NONE>(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// Fused RMS_NORM + MUL: computes dst = rms_norm(src0) * src1 in a single pass.
// This avoids materializing the intermediate rms_norm result in memory.
ਖਾਲੀ ggml_compute_forward_rms_norm_mul_fused(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst_rms_norm,
        ggml_tensor * dst_mul) {

    GGML_ASSERT(dst_mul != ਖਾਲੀ_ਸੂਚਕ);
    GGML_ASSERT(dst_mul->src[0] == dst_rms_norm || dst_mul->src[1] == dst_rms_norm);

    ਅਚਲ ggml_tensor * src0 = dst_rms_norm->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_rms_norm_f32<GGML_RMS_NORM_FUSE_OP_MUL>(params, dst_rms_norm, dst_mul);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_rms_norm_ਪਿੱਠ_f32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0]; // gradients from forward pass output
    ਅਚਲ ggml_tensor * src1 = dst->src[1]; // src1 from forward pass

    GGML_ASSERT(ggml_are_same_shape(src0, dst) && ggml_are_same_shape(src0, src1));

    GGML_ASSERT(src0->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    GGML_ASSERT(src1->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

    ਦਸ਼ਮਲਵ eps;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&eps, dst->op_params, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    // TODO: optimize
    ਲਈ (int64_t i03 = 0; i03 < ne03; i03++) {
        ਲਈ (int64_t i02 = 0; i02 < ne02; i02++) {
            ਲਈ (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                // src1 is same shape as src0 => same indices
                ਅਚਲ int64_t i11 = i01;
                ਅਚਲ int64_t i12 = i02;
                ਅਚਲ int64_t i13 = i03;

                ਅਚਲ ਦਸ਼ਮਲਵ * dz = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i01*nb01 + i02*nb02 + i03*nb03);
                ਅਚਲ ਦਸ਼ਮਲਵ * x  = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src1->ਅੰਕੜੇ + i11*nb11 + i12*nb12 + i13*nb13);

                ggml_float ਜੋੜ_xx  = 0.0;
                ggml_float ਜੋੜ_xdz = 0.0;

                ਲਈ (int64_t i00 = 0; i00 < ne00; i00++) {
                    ਜੋੜ_xx  += (ggml_float)(x[i00] * x[i00]);
                    ਜੋੜ_xdz += (ggml_float)(x[i00] * dz[i00]);
                }

                //const float mean     = (float)(sum_xx)/ne00;
                ਅਚਲ ਦਸ਼ਮਲਵ ਔਸਤ_eps = (ਦਸ਼ਮਲਵ)(ਜੋੜ_xx)/ne00 + eps;
                ਅਚਲ ਦਸ਼ਮਲਵ ਜੋੜ_eps  = (ਦਸ਼ਮਲਵ)(ਜੋੜ_xx) + eps*ne00;
                //const float mean_xdz = (float)(sum_xdz)/ne00;
                // we could cache rms from forward pass to improve performance.
                // to do this implement ggml_rms and compose ggml_rms_norm using ggml_rms.
                //const float rms      = sqrtf(mean_eps);
                ਅਚਲ ਦਸ਼ਮਲਵ rrms     = 1.0f / sqrtf(ਔਸਤ_eps);
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
                ਦਸ਼ਮਲਵ * dx = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i01*nb1 + i02*nb2 + i03*nb3);

                // dx[i00] = (dz + x*(-sum_xdz/sum_eps)) * rrms
                // note: https://github.com/ggml-org/ggml/issues/1491
                ਅਚਲ ਦਸ਼ਮਲਵ scale_x = (ਦਸ਼ਮਲਵ) (-ਜੋੜ_xdz) / ਜੋੜ_eps;
                ਲਈ (int64_t i00 = 0; i00 < ne00; i00++) {
                    dx[i00] = (dz[i00] + x[i00] * scale_x) * rrms;
                }
            }
        }
    }
}

ਖਾਲੀ ggml_compute_forward_rms_norm_ਪਿੱਠ(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_rms_norm_ਪਿੱਠ_f32(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_group_norm

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_ਸਮੂਹ_norm_f32(
    ਅਚਲ ggml_compute_params * params,
    ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    GGML_ASSERT(src0->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    // TODO: optimize

    ਦਸ਼ਮਲਵ eps;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&eps, dst->op_params + 1, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਪੂਰਨ n_channels = src0->ne[2];
    ਪੂਰਨ n_groups = dst->op_params[0];
    ਪੂਰਨ n_channels_ਪ੍ਰਤੀ_ਸਮੂਹ = (n_channels + n_groups - 1) / n_groups;
    ਲਈ (ਪੂਰਨ i = ith; i < n_groups; i += nth) {
        ਪੂਰਨ start = i * n_channels_ਪ੍ਰਤੀ_ਸਮੂਹ;
        ਪੂਰਨ end = start + n_channels_ਪ੍ਰਤੀ_ਸਮੂਹ;
        ਜੇ (end > n_channels) {
            end = n_channels;
        }
        ਪੂਰਨ ਕਦਮ = end - start;

        ਲਈ (int64_t i03 = 0; i03 < ne03; i03++) {
            ggml_float ਜੋੜ = 0.0;
            ਲਈ (int64_t i02 = start; i02 < end; i02++) {
                ਲਈ (int64_t i01 = 0; i01 < ne01; i01++) {
                    ਅਚਲ ਦਸ਼ਮਲਵ * x = (ਦਸ਼ਮਲਵ *)((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i01 * nb01 + i02 * nb02 + i03 * nb03);

                    ggml_float sumr = 0.0;
                    ਲਈ (int64_t i00 = 0; i00 < ne00; i00++) {
                        sumr += (ggml_float)x[i00];
                    }
                    ਜੋੜ += sumr;
                }
            }
            ਅਚਲ ਦਸ਼ਮਲਵ ਔਸਤ_2 = ਜੋੜ / (ne00 * ne01 * ਕਦਮ);

            ggml_float sum2 = 0.0;
            ਲਈ (int64_t i02 = start; i02 < end; i02++) {
                ਲਈ (int64_t i01 = 0; i01 < ne01; i01++) {
                    ਅਚਲ ਦਸ਼ਮਲਵ * x = (ਦਸ਼ਮਲਵ *)((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i01 * nb01 + i02 * nb02 + i03 * nb03);

                    ਦਸ਼ਮਲਵ * y = (ਦਸ਼ਮਲਵ *)((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i01 * nb1 + i02 * nb2 + i03 * nb3);

                    ggml_float sumr = 0.0;
                    ਲਈ (int64_t i00 = 0; i00 < ne00; i00++) {
                        ਦਸ਼ਮਲਵ v = x[i00] - ਔਸਤ_2;
                        y[i00] = v;
                        sumr += (ggml_float)(v * v);
                    }
                    sum2 += sumr;
                }
            }
            ਅਚਲ ਦਸ਼ਮਲਵ variance = sum2 / (ne00 * ne01 * ਕਦਮ);
            ਅਚਲ ਦਸ਼ਮਲਵ scale = 1.0f / sqrtf(variance + eps);

            ਲਈ (int64_t i02 = start; i02 < end; i02++) {
                ਲਈ (int64_t i01 = 0; i01 < ne01; i01++) {
                    ਦਸ਼ਮਲਵ * y = (ਦਸ਼ਮਲਵ *)((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i01 * nb1 + i02 * nb2 + i03 * nb3);
                    ggml_vec_scale_f32(ne00, y, scale);
                }
            }
        }
    }
}

ਖਾਲੀ ggml_compute_forward_ਸਮੂਹ_norm(
    ਅਚਲ ggml_compute_params * params,
    ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_ਸਮੂਹ_norm_f32(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_l2_norm

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_l2_norm_f32(
    ਅਚਲ ggml_compute_params * params,
    ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    ਦਸ਼ਮਲਵ eps;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&eps, dst->op_params, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    GGML_ASSERT(eps >= 0.0f);

    // TODO: optimize
    ਲਈ (int64_t i03 = 0; i03 < ne03; i03++) {
        ਲਈ (int64_t i02 = 0; i02 < ne02; i02++) {
            ਲਈ (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                ਅਚਲ ਅੱਖਰ * x = (ਅਚਲ ਅੱਖਰ *) src0->ਅੰਕੜੇ + i01*nb01 + i02*nb02 + i03*nb03;

                ggml_float ਜੋੜ = 0.0;
                ਲਈ (int64_t i00 = 0; i00 < ne00; i00++) {
                    ਅਚਲ ਦਸ਼ਮਲਵ xi = *(ਅਚਲ ਦਸ਼ਮਲਵ *) (x + i00*nb00);
                    ਜੋੜ += (ggml_float)(xi * xi);
                }

                ਅਚਲ ਦਸ਼ਮਲਵ scale = 1.0f/fmaxf(sqrtf(ਜੋੜ), eps);

                ਅੱਖਰ * y = (ਅੱਖਰ *) dst->ਅੰਕੜੇ + i01*nb1 + i02*nb2 + i03*nb3;

                ਜੇ (nb00 == ਆਕਾਰ(ਦਸ਼ਮਲਵ) && nb0 == ਆਕਾਰ(ਦਸ਼ਮਲਵ)) {
                    ਸਮ੍ਰਿਤੀ_ਨਕਲ(y, x, ne00 * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                    ggml_vec_scale_f32(ne00, (ਦਸ਼ਮਲਵ *) y, scale);
                } ਨਹੀਂਤਾਂ {
                    ਲਈ (int64_t i00 = 0; i00 < ne00; i00++) {
                        ਅਚਲ ਦਸ਼ਮਲਵ xi = *(ਅਚਲ ਦਸ਼ਮਲਵ *) (x + i00*nb00);
                        *(ਦਸ਼ਮਲਵ *) (y + i00*nb0) = xi * scale;
                    }
                }
            }
        }
    }
}

ਖਾਲੀ ggml_compute_forward_l2_norm(
    ਅਚਲ ggml_compute_params * params,
    ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_l2_norm_f32(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_out_prod

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_ਬਾਹਰਲਾ_prod_f32(
        ਅਚਲ ggml_compute_params * params,
              ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    GGML_ASSERT(ne2 % ne02 == 0);
    GGML_ASSERT(ne3 % ne03 == 0);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    // GGML_ASSERT(nb0 <= nb1);
    // GGML_ASSERT(nb1 <= nb2);
    // GGML_ASSERT(nb2 <= nb3);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    ਜੇ (ith == 0) {
        ggml_vec_ਸੈੱਟ_f32(ne0*ne1*ne2*ne3, (ਦਸ਼ਮਲਵ *)dst->ਅੰਕੜੇ, 0);
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
    ਅਚਲ int64_t nr = ne1*ne2*ne3;

    // rows per thread
    ਅਚਲ int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ int64_t ir0 = dr*ith;
    ਅਚਲ int64_t ir1 = MIN(ir0 + dr, nr);

    // block-tiling attempt
    ਅਚਲ int64_t blck_0 = MAX(GGML_VEC_MAD_UNROLL, 32);
    ਅਚਲ int64_t blck_1 = 16;

    // dps == dst per src0, used for group query attention
    ਅਚਲ int64_t dps2 = ne2 / ne02;
    ਅਚਲ int64_t dps3 = ne3 / ne03;

    ਲਈ (int64_t bir = ir0; bir < ir1; bir += blck_1) {
        ਅਚਲ int64_t bir1 = MIN(bir + blck_1, ir1);
        ਲਈ (int64_t bi01 = 0; bi01 < ne01; bi01 += blck_0) {
            ਅਚਲ int64_t bne01 = MIN(bi01 + blck_0, ne01);
            ਲਈ (int64_t ir = bir; ir < bir1; ++ir) {
                // dst indices
                ਅਚਲ int64_t i3 = ir/(ne2*ne1);
                ਅਚਲ int64_t i2 = (ir - i3*ne2*ne1)/ne1;
                ਅਚਲ int64_t i1 = (ir - i3*ne2*ne1 - i2*ne1);

                ਅਚਲ int64_t i02 = i2 / dps2;
                ਅਚਲ int64_t i03 = i3 / dps3;

                //const int64_t i10 = i1;
                ਅਚਲ int64_t i12 = i2;
                ਅਚਲ int64_t i13 = i3;

#ਜੇ GGML_VEC_MAD_UNROLL > 2
                ਅਚਲ int64_t bne01_unroll = bne01 - (bne01 % GGML_VEC_MAD_UNROLL);
                ਲਈ (int64_t i01 = bi01; i01 < bne01_unroll; i01 += GGML_VEC_MAD_UNROLL) {
                    ਅਚਲ int64_t i11 = i01;

                    ਦਸ਼ਮਲਵ * s0 = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + (          i01*nb01 + i02*nb02 + i03*nb03));
                    ਦਸ਼ਮਲਵ * s1 = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src1->ਅੰਕੜੇ + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
                    ਦਸ਼ਮਲਵ * d  = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *)  dst->ਅੰਕੜੇ + (          i1*nb1   + i2*nb2   + i3*nb3));

                    ggml_vec_mad_f32_unroll(ne0, nb01, nb11, d, s0, s1);
                }
                ਲਈ (int64_t i01 = bne01_unroll; i01 < bne01; ++i01) {
                    ਅਚਲ int64_t i11 = i01;

                    ਦਸ਼ਮਲਵ * s0 = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + (          i01*nb01 + i02*nb02 + i03*nb03));
                    ਦਸ਼ਮਲਵ * s1 = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src1->ਅੰਕੜੇ + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
                    ਦਸ਼ਮਲਵ * d  = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *)  dst->ਅੰਕੜੇ + (          i1*nb1   + i2*nb2   + i3*nb3));

                    ggml_vec_mad_f32(ne0, d, s0, *s1);
                }
#ਨਹੀਂਤਾਂ
                ਲਈ (int64_t i01 = bi01; i01 < bne01; ++i01) {
                    ਅਚਲ int64_t i11 = i01;

                    ਦਸ਼ਮਲਵ * s0 = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + (          i01*nb01 + i02*nb02 + i03*nb03));
                    ਦਸ਼ਮਲਵ * s1 = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src1->ਅੰਕੜੇ + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
                    ਦਸ਼ਮਲਵ * d  = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *)  dst->ਅੰਕੜੇ + (          i1*nb1 + i2*nb2 + i3*nb3));

                    ggml_vec_mad_f32(ne0, d, s0, *s1);
                }
#ਜੇ_ਅੰਤ
            }
        }
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_ਬਾਹਰਲਾ_prod_q_f32(
        ਅਚਲ ggml_compute_params * params,
              ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS;

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ggml_type type = src0->type;
    ggml_ਮੰਜ਼ਿਲ_float_t ਅਚਲ dequantize_ਪੰਕਤੀ_q = ggml_get_type_traits(type)->ਮੰਜ਼ਿਲ_float;

    GGML_ASSERT(ne02 == ne12);
    GGML_ASSERT(ne03 == ne13);
    GGML_ASSERT(ne2  == ne12);
    GGML_ASSERT(ne3  == ne13);

    // we don't support permuted src0 dim0
    GGML_ASSERT(nb00 == ggml_type_ਮਾਪ(type));

    // dst dim0 cannot be transposed or permuted
    GGML_ASSERT(nb0 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    // GGML_ASSERT(nb0 <= nb1);
    // GGML_ASSERT(nb1 <= nb2);
    // GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    ਜੇ (ith == 0) {
        ggml_vec_ਸੈੱਟ_f32(ne0*ne1*ne2*ne3, (ਦਸ਼ਮਲਵ *)dst->ਅੰਕੜੇ, 0);
    }
    ggml_barrier(params->threadpool);

    // parallelize by last three dimensions

    // total rows in dst
    ਅਚਲ int64_t nr = ne1*ne2*ne3;

    // rows per thread
    ਅਚਲ int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ int64_t ir0 = dr*ith;
    ਅਚਲ int64_t ir1 = MIN(ir0 + dr, nr);

    // dst[:,:,:,:] = 0
    // for i2,i3:
    //   for i1:
    //     for i01:
    //       for i0:
    //         dst[i0,i1,i2,i3] += src0[i0,i01,i2,i3] * src1[i1,i01,i2,i3]

    ਦਸ਼ਮਲਵ * wdata = (ਦਸ਼ਮਲਵ *) params->wdata + (ne0 + CACHE_ਸਤਰ_ਮਾਪ_F32) * ith;

    ਲਈ (int64_t ir = ir0; ir < ir1; ++ir) {
        // dst indices
        ਅਚਲ int64_t i3 = ir/(ne2*ne1);
        ਅਚਲ int64_t i2 = (ir - i3*ne2*ne1)/ne1;
        ਅਚਲ int64_t i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ਅਚਲ int64_t i02 = i2;
        ਅਚਲ int64_t i03 = i3;

        //const int64_t i10 = i1;
        ਅਚਲ int64_t i12 = i2;
        ਅਚਲ int64_t i13 = i3;

        ਲਈ (int64_t i01 = 0; i01 < ne01; ++i01) {
            ਅਚਲ int64_t i11 = i01;

            ਦਸ਼ਮਲਵ * s0 = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + (          i01*nb01 + i02*nb02 + i03*nb03));
            ਦਸ਼ਮਲਵ * s1 = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src1->ਅੰਕੜੇ + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
            ਦਸ਼ਮਲਵ * d  = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *)  dst->ਅੰਕੜੇ + (          i1*nb1 + i2*nb2 + i3*nb3));

            dequantize_ਪੰਕਤੀ_q(s0, wdata, ne0);
            ggml_vec_mad_f32(ne0, d, wdata, *s1);
        }
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_ਬਾਹਰਲਾ_prod_f16_f32(
        ਅਚਲ ggml_compute_params * params,
              ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS;

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ne02 == ne12);
    GGML_ASSERT(ne03 == ne13);
    GGML_ASSERT(ne2  == ne12);
    GGML_ASSERT(ne3  == ne13);

    GGML_ASSERT(nb00 == ਆਕਾਰ(ggml_fp16_t));
    GGML_ASSERT(nb0  == ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    ਜੇ (ith == 0) {
        ggml_vec_ਸੈੱਟ_f32(ne0*ne1*ne2*ne3, (ਦਸ਼ਮਲਵ *)dst->ਅੰਕੜੇ, 0);
    }
    ggml_barrier(params->threadpool);

    ਅਚਲ int64_t nr = ne1*ne2*ne3;
    ਅਚਲ int64_t dr = (nr + nth - 1)/nth;
    ਅਚਲ int64_t ir0 = dr*ith;
    ਅਚਲ int64_t ir1 = MIN(ir0 + dr, nr);

    ਦਸ਼ਮਲਵ * wdata = (ਦਸ਼ਮਲਵ *) params->wdata + (ne0 + CACHE_ਸਤਰ_ਮਾਪ_F32) * ith;

    ਲਈ (int64_t ir = ir0; ir < ir1; ++ir) {
        ਅਚਲ int64_t i3 = ir/(ne2*ne1);
        ਅਚਲ int64_t i2 = (ir - i3*ne2*ne1)/ne1;
        ਅਚਲ int64_t i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ਅਚਲ int64_t i02 = i2;
        ਅਚਲ int64_t i03 = i3;

        ਅਚਲ int64_t i12 = i2;
        ਅਚਲ int64_t i13 = i3;

        ਦਸ਼ਮਲਵ * d = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + (i1*nb1 + i2*nb2 + i3*nb3));

        ਲਈ (int64_t i01 = 0; i01 < ne01; ++i01) {
            ਅਚਲ int64_t i11 = i01;
            ggml_fp16_t * s0 = (ggml_fp16_t *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + (i01*nb01 + i02*nb02 + i03*nb03));
            ਦਸ਼ਮਲਵ * s1 = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src1->ਅੰਕੜੇ + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
            ggml_fp16_ਮੰਜ਼ਿਲ_fp32_ਪੰਕਤੀ(s0, wdata, ne0);
            ggml_vec_mad_f32(ne0, d, wdata, *s1);
        }
    }
}

ਖਾਲੀ ggml_compute_forward_ਬਾਹਰਲਾ_prod(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_Q1_0:
        ਹਾਲਤ GGML_TYPE_Q2_0:
        ਹਾਲਤ GGML_TYPE_Q4_0:
        ਹਾਲਤ GGML_TYPE_Q4_1:
        ਹਾਲਤ GGML_TYPE_Q5_0:
        ਹਾਲਤ GGML_TYPE_Q5_1:
        ਹਾਲਤ GGML_TYPE_Q8_0:
        ਹਾਲਤ GGML_TYPE_MXFP4:
        ਹਾਲਤ GGML_TYPE_NVFP4:
        ਹਾਲਤ GGML_TYPE_Q2_K:
        ਹਾਲਤ GGML_TYPE_Q3_K:
        ਹਾਲਤ GGML_TYPE_Q4_K:
        ਹਾਲਤ GGML_TYPE_Q5_K:
        ਹਾਲਤ GGML_TYPE_Q6_K:
        ਹਾਲਤ GGML_TYPE_TQ1_0:
        ਹਾਲਤ GGML_TYPE_TQ2_0:
        ਹਾਲਤ GGML_TYPE_IQ2_XXS:
        ਹਾਲਤ GGML_TYPE_IQ2_XS:
        ਹਾਲਤ GGML_TYPE_IQ3_XXS:
        ਹਾਲਤ GGML_TYPE_IQ1_S:
        ਹਾਲਤ GGML_TYPE_IQ1_M:
        ਹਾਲਤ GGML_TYPE_IQ4_NL:
        ਹਾਲਤ GGML_TYPE_IQ4_XS:
        ਹਾਲਤ GGML_TYPE_IQ3_S:
        ਹਾਲਤ GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_ਬਾਹਰਲਾ_prod_q_f32(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F16:
            {
                ggml_compute_forward_ਬਾਹਰਲਾ_prod_f16_f32(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_ਬਾਹਰਲਾ_prod_f32(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_scale

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_scale_f32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    ਦਸ਼ਮਲਵ s; // scale factor
    ਦਸ਼ਮਲਵ b; // bias

    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&s, (ਦਸ਼ਮਲਵ *) dst->op_params + 0, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&b, (ਦਸ਼ਮਲਵ *) dst->op_params + 1, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ nc = src0->ne[0];
    ਅਚਲ ਪੂਰਨ nr = ggml_nrows(src0);

    // rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr*ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਅਚਲ size_t nb01 = src0->nb[1];

    ਅਚਲ size_t nb1 = dst->nb[1];

    ਜੇ (b == 0.0f) {
        ਲਈ (ਪੂਰਨ i1 = ir0; i1 < ir1; i1++) {
            ਜੇ (dst->ਅੰਕੜੇ != src0->ਅੰਕੜੇ) {
                // src0 is same shape as dst => same indices
                // TODO: add x parameter to ggml_vec_scale_f32 and remove this memcpy
                ਸਮ੍ਰਿਤੀ_ਨਕਲ((ਅੱਖਰ *)dst->ਅੰਕੜੇ + i1*nb1, (ਅੱਖਰ *)src0->ਅੰਕੜੇ + i1*nb01, nc * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
            }
            ggml_vec_scale_f32(nc, (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i1*nb1), s);
        }
    } ਨਹੀਂਤਾਂ {
        ਲਈ (ਪੂਰਨ i1 = ir0; i1 < ir1; i1++) {
            ggml_vec_mad1_f32(nc,
                (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ  + i1*nb1),
                (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i1*nb1),
                s, b);
        }
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_scale_bf16(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    ਦਸ਼ਮਲਵ s; // scale factor
    ਦਸ਼ਮਲਵ b; // bias

    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&s, (ਦਸ਼ਮਲਵ *) dst->op_params + 0, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&b, (ਦਸ਼ਮਲਵ *) dst->op_params + 1, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ nc = src0->ne[0];
    ਅਚਲ ਪੂਰਨ nr = ggml_nrows(src0);

    // rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr*ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਲਈ (ਪੂਰਨ i1 = ir0; i1 < ir1; i1++) {
        ਅਚਲ ggml_bf16_t * x = (ਅਚਲ ggml_bf16_t *) ((ਅਚਲ ਅੱਖਰ *) src0->ਅੰਕੜੇ + i1*src0->nb[1]);
        ggml_bf16_t       * y = (ggml_bf16_t *)       ((ਅੱਖਰ *)       dst->ਅੰਕੜੇ  + i1*dst->nb[1]);
        ਲਈ (ਪੂਰਨ k = 0; k < nc; k++) {
            y[k] = GGML_FP32_ਮੰਜ਼ਿਲ_BF16(s*GGML_BF16_ਮੰਜ਼ਿਲ_FP32(x[k]) + b);
        }
    }
}

ਖਾਲੀ ggml_compute_forward_scale(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_scale_f32(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_BF16:
            {
                ggml_compute_forward_scale_bf16(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_set

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_ਸੈੱਟ_f32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));

    // view src0 and dst with these strides and data offset inbytes during set
    // nb0 is implicitly element_size because src0 and dst are contiguous
    size_t nb1     = ((int32_t *) dst->op_params)[0];
    size_t nb2     = ((int32_t *) dst->op_params)[1];
    size_t nb3     = ((int32_t *) dst->op_params)[2];
    size_t ਵਿੱਥ  = ((int32_t *) dst->op_params)[3];
    ਤਾਰਕਿਕ   inplace = (ਤਾਰਕਿਕ) ((int32_t *) dst->op_params)[4];

    ਜੇ (!inplace) {
        ਜੇ (params->ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            ਸਮ੍ਰਿਤੀ_ਨਕਲ(
                ((ਅੱਖਰ *)  dst->ਅੰਕੜੇ),
                ((ਅੱਖਰ *) src0->ਅੰਕੜੇ),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ nr = ggml_nrows(src1);
    ਅਚਲ ਪੂਰਨ nc = src1->ne[0];

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne)
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb)

    // src0 and dst as viewed during set
    ਅਚਲ size_t nb0 = ggml_element_ਮਾਪ(src0);

    ਅਚਲ ਪੂਰਨ im0 = (ne10 == 0 ? 0 : ne10-1);
    ਅਚਲ ਪੂਰਨ im1 = (ne11 == 0 ? 0 : ne11-1);
    ਅਚਲ ਪੂਰਨ im2 = (ne12 == 0 ? 0 : ne12-1);
    ਅਚਲ ਪੂਰਨ im3 = (ne13 == 0 ? 0 : ne13-1);

    GGML_ASSERT(ਵਿੱਥ + im0*nb0  + im1*nb1  + im2*nb2  + im3*nb3  <= ggml_nbytes(dst));

    GGML_ASSERT(nb10 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    // rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr*ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਲਈ (ਪੂਰਨ ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are viewed with shape of src1 and offset
        // => same indices
        ਅਚਲ ਪੂਰਨ i3 = ir/(ne12*ne11);
        ਅਚਲ ਪੂਰਨ i2 = (ir - i3*ne12*ne11)/ne11;
        ਅਚਲ ਪੂਰਨ i1 = (ir - i3*ne12*ne11 - i2*ne11);

        ggml_vec_cpy_f32(nc,
                (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *)  dst->ਅੰਕੜੇ + i3*nb3  + i2*nb2  + i1*nb1  + ਵਿੱਥ),
                (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src1->ਅੰਕੜੇ + i3*nb13 + i2*nb12 + i1*nb11));
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_ਸੈੱਟ_i32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));

    // view src0 and dst with these strides and data offset inbytes during set
    // nb0 is implicitly element_size because src0 and dst are contiguous
    size_t nb1     = ((int32_t *) dst->op_params)[0];
    size_t nb2     = ((int32_t *) dst->op_params)[1];
    size_t nb3     = ((int32_t *) dst->op_params)[2];
    size_t ਵਿੱਥ  = ((int32_t *) dst->op_params)[3];
    ਤਾਰਕਿਕ   inplace = (ਤਾਰਕਿਕ) ((int32_t *) dst->op_params)[4];

    ਜੇ (!inplace) {
        ਜੇ (params->ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            ਸਮ੍ਰਿਤੀ_ਨਕਲ(
                ((ਅੱਖਰ *)  dst->ਅੰਕੜੇ),
                ((ਅੱਖਰ *) src0->ਅੰਕੜੇ),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ nr = ggml_nrows(src1);
    ਅਚਲ ਪੂਰਨ nc = src1->ne[0];

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne)
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb)

    // src0 and dst as viewed during set
    ਅਚਲ size_t nb0 = ggml_element_ਮਾਪ(src0);

    ਅਚਲ ਪੂਰਨ im0 = (ne10 == 0 ? 0 : ne10-1);
    ਅਚਲ ਪੂਰਨ im1 = (ne11 == 0 ? 0 : ne11-1);
    ਅਚਲ ਪੂਰਨ im2 = (ne12 == 0 ? 0 : ne12-1);
    ਅਚਲ ਪੂਰਨ im3 = (ne13 == 0 ? 0 : ne13-1);

    GGML_ASSERT(ਵਿੱਥ + im0*nb0  + im1*nb1  + im2*nb2  + im3*nb3  <= ggml_nbytes(dst));

    GGML_ASSERT(nb10 == ਆਕਾਰ(int32_t));

    // rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr*ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਲਈ (ਪੂਰਨ ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are viewed with shape of src1 and offset
        // => same indices
        ਅਚਲ ਪੂਰਨ i3 = ir/(ne12*ne11);
        ਅਚਲ ਪੂਰਨ i2 = (ir - i3*ne12*ne11)/ne11;
        ਅਚਲ ਪੂਰਨ i1 = (ir - i3*ne12*ne11 - i2*ne11);

        ggml_vec_cpy_i32(nc,
                (int32_t *) ((ਅੱਖਰ *)  dst->ਅੰਕੜੇ + i3*nb3  + i2*nb2  + i1*nb1  + ਵਿੱਥ),
                (int32_t *) ((ਅੱਖਰ *) src1->ਅੰਕੜੇ + i3*nb13 + i2*nb12 + i1*nb11));
    }
}

ਖਾਲੀ ggml_compute_forward_ਸੈੱਟ(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_ਸੈੱਟ_f32(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_I32:
            {
                ggml_compute_forward_ਸੈੱਟ_i32(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F16:
        ਹਾਲਤ GGML_TYPE_BF16:
        ਹਾਲਤ GGML_TYPE_Q1_0:
        ਹਾਲਤ GGML_TYPE_Q2_0:
        ਹਾਲਤ GGML_TYPE_Q4_0:
        ਹਾਲਤ GGML_TYPE_Q4_1:
        ਹਾਲਤ GGML_TYPE_Q5_0:
        ਹਾਲਤ GGML_TYPE_Q5_1:
        ਹਾਲਤ GGML_TYPE_Q8_0:
        ਹਾਲਤ GGML_TYPE_Q8_1:
        ਹਾਲਤ GGML_TYPE_MXFP4:
        ਹਾਲਤ GGML_TYPE_NVFP4:
        ਹਾਲਤ GGML_TYPE_Q2_K:
        ਹਾਲਤ GGML_TYPE_Q3_K:
        ਹਾਲਤ GGML_TYPE_Q4_K:
        ਹਾਲਤ GGML_TYPE_Q5_K:
        ਹਾਲਤ GGML_TYPE_Q6_K:
        ਹਾਲਤ GGML_TYPE_TQ1_0:
        ਹਾਲਤ GGML_TYPE_TQ2_0:
        ਹਾਲਤ GGML_TYPE_IQ2_XXS:
        ਹਾਲਤ GGML_TYPE_IQ2_XS:
        ਹਾਲਤ GGML_TYPE_IQ3_XXS:
        ਹਾਲਤ GGML_TYPE_IQ1_S:
        ਹਾਲਤ GGML_TYPE_IQ1_M:
        ਹਾਲਤ GGML_TYPE_IQ4_NL:
        ਹਾਲਤ GGML_TYPE_IQ4_XS:
        ਹਾਲਤ GGML_TYPE_IQ3_S:
        ਹਾਲਤ GGML_TYPE_IQ2_S:
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_cpy

ਖਾਲੀ ggml_compute_forward_cpy(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_dup(params, dst);
}

// ggml_compute_forward_cont

ਖਾਲੀ ggml_compute_forward_cont(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_dup(params, dst);
}

// ggml_compute_forward_get_rows

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_get_ਪੰਕਤੀਆਂ_q(
        ਅਚਲ ggml_compute_params * params,
              ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

    ਅਚਲ int64_t nc = ne00;
    ਅਚਲ int64_t nr = ggml_nelements(src1);

    ਅਚਲ ggml_type type = src0->type;
    ggml_ਮੰਜ਼ਿਲ_float_t ਅਚਲ dequantize_ਪੰਕਤੀ_q = ggml_get_type_traits(type)->ਮੰਜ਼ਿਲ_float;

    ਪ੍ਰਤਿਜ੍ਞਾ(ne0  == nc);
    ਪ੍ਰਤਿਜ੍ਞਾ(ne02 == ne11);
    ਪ੍ਰਤਿਜ੍ਞਾ(nb00 == ggml_type_ਮਾਪ(type));
    ਪ੍ਰਤਿਜ੍ਞਾ(ggml_nrows(dst) == nr);

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    // rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr*ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਲਈ (int64_t i = ir0; i < ir1; ++i) {
        ਅਚਲ int64_t i12 = i/(ne11*ne10);
        ਅਚਲ int64_t i11 = (i - i12*ne11*ne10)/ne10;
        ਅਚਲ int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        ਅਚਲ int64_t i01 = *(int32_t *) ((ਅੱਖਰ *) src1->ਅੰਕੜੇ + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        dequantize_ਪੰਕਤੀ_q(
                (ਅਚਲ ਖਾਲੀ *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i01*nb01 + i11*nb02 + i12*nb03),
                     (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *)  dst->ਅੰਕੜੇ + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_get_ਪੰਕਤੀਆਂ_f16(
        ਅਚਲ ggml_compute_params * params,
              ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

    ਅਚਲ int64_t nc = ne00;
    ਅਚਲ int64_t nr = ggml_nelements(src1);

    ਪ੍ਰਤਿਜ੍ਞਾ(ne0  == nc);
    ਪ੍ਰਤਿਜ੍ਞਾ(ne02 == ne11);
    ਪ੍ਰਤਿਜ੍ਞਾ(nb00 == ਆਕਾਰ(ggml_fp16_t));
    ਪ੍ਰਤਿਜ੍ਞਾ(ggml_nrows(dst) == nr);

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    // rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr*ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਲਈ (int64_t i = ir0; i < ir1; ++i) {
        ਅਚਲ int64_t i12 = i/(ne11*ne10);
        ਅਚਲ int64_t i11 = (i - i12*ne11*ne10)/ne10;
        ਅਚਲ int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        ਅਚਲ int64_t i01 = *(int32_t *) ((ਅੱਖਰ *) src1->ਅੰਕੜੇ + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        ggml_cpu_fp16_ਮੰਜ਼ਿਲ_fp32(
            (ਅਚਲ ggml_fp16_t*) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i01*nb01 + i11*nb02 + i12*nb03),
                       (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *)  dst->ਅੰਕੜੇ + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_get_ਪੰਕਤੀਆਂ_bf16(
        ਅਚਲ ggml_compute_params * params,
              ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

    ਅਚਲ int64_t nc = ne00;
    ਅਚਲ int64_t nr = ggml_nelements(src1);

    ਪ੍ਰਤਿਜ੍ਞਾ(ne0  == nc);
    ਪ੍ਰਤਿਜ੍ਞਾ(ne02 == ne11);
    ਪ੍ਰਤਿਜ੍ਞਾ(nb00 == ਆਕਾਰ(ggml_bf16_t));
    ਪ੍ਰਤਿਜ੍ਞਾ(ggml_nrows(dst) == nr);

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    // rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr*ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਲਈ (int64_t i = ir0; i < ir1; ++i) {
        ਅਚਲ int64_t i12 = i/(ne11*ne10);
        ਅਚਲ int64_t i11 = (i - i12*ne11*ne10)/ne10;
        ਅਚਲ int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        ਅਚਲ int64_t i01 = *(int32_t *) ((ਅੱਖਰ *) src1->ਅੰਕੜੇ + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        ggml_cpu_bf16_ਮੰਜ਼ਿਲ_fp32(
            (ਅਚਲ ggml_bf16_t *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i01*nb01 + i11*nb02 + i12*nb03),
                        (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *)  dst->ਅੰਕੜੇ + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_get_ਪੰਕਤੀਆਂ_f32(
        ਅਚਲ ggml_compute_params * params,
              ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

    ਅਚਲ int64_t nc = ne00;
    ਅਚਲ int64_t nr = ggml_nelements(src1);

    ਪ੍ਰਤਿਜ੍ਞਾ(ne0  == nc);
    ਪ੍ਰਤਿਜ੍ਞਾ(ne02 == ne11);
    ਪ੍ਰਤਿਜ੍ਞਾ(nb00 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਪ੍ਰਤਿਜ੍ਞਾ(ggml_nrows(dst) == nr);

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    // rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr*ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਲਈ (int64_t i = ir0; i < ir1; ++i) {
        ਅਚਲ int64_t i12 = i/(ne11*ne10);
        ਅਚਲ int64_t i11 = (i - i12*ne11*ne10)/ne10;
        ਅਚਲ int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        ਅਚਲ int64_t i01 = *(int32_t *) ((ਅੱਖਰ *) src1->ਅੰਕੜੇ + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        ggml_vec_cpy_f32(nc,
                (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *)  dst->ਅੰਕੜੇ + i10*nb1  + i11*nb2  + i12*nb3),
                (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i01*nb01 + i11*nb02 + i12*nb03));
    }
}

ਖਾਲੀ ggml_compute_forward_get_ਪੰਕਤੀਆਂ(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_Q1_0:
        ਹਾਲਤ GGML_TYPE_Q2_0:
        ਹਾਲਤ GGML_TYPE_Q4_0:
        ਹਾਲਤ GGML_TYPE_Q4_1:
        ਹਾਲਤ GGML_TYPE_Q5_0:
        ਹਾਲਤ GGML_TYPE_Q5_1:
        ਹਾਲਤ GGML_TYPE_Q8_0:
        ਹਾਲਤ GGML_TYPE_Q8_1:
        ਹਾਲਤ GGML_TYPE_MXFP4:
        ਹਾਲਤ GGML_TYPE_NVFP4:
        ਹਾਲਤ GGML_TYPE_Q2_K:
        ਹਾਲਤ GGML_TYPE_Q3_K:
        ਹਾਲਤ GGML_TYPE_Q4_K:
        ਹਾਲਤ GGML_TYPE_Q5_K:
        ਹਾਲਤ GGML_TYPE_Q6_K:
        ਹਾਲਤ GGML_TYPE_TQ1_0:
        ਹਾਲਤ GGML_TYPE_TQ2_0:
        ਹਾਲਤ GGML_TYPE_IQ2_XXS:
        ਹਾਲਤ GGML_TYPE_IQ2_XS:
        ਹਾਲਤ GGML_TYPE_IQ3_XXS:
        ਹਾਲਤ GGML_TYPE_IQ1_S:
        ਹਾਲਤ GGML_TYPE_IQ1_M:
        ਹਾਲਤ GGML_TYPE_IQ4_NL:
        ਹਾਲਤ GGML_TYPE_IQ4_XS:
        ਹਾਲਤ GGML_TYPE_IQ3_S:
        ਹਾਲਤ GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_get_ਪੰਕਤੀਆਂ_q(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F16:
            {
                ggml_compute_forward_get_ਪੰਕਤੀਆਂ_f16(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_BF16:
            {
                ggml_compute_forward_get_ਪੰਕਤੀਆਂ_bf16(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F32:
        ਹਾਲਤ GGML_TYPE_I32:
            {
                ggml_compute_forward_get_ਪੰਕਤੀਆਂ_f32(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
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

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ src_t, ਕਿਸਮ_ਨਾਂ idx_t>
ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_ਸੈੱਟ_ਪੰਕਤੀਆਂ_impl(
        ਅਚਲ ggml_compute_params * params,
              ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

    ਅਚਲ int64_t nc = ne00;
    ਅਚਲ int64_t nr = ne01;

    ਪ੍ਰਤਿਜ੍ਞਾ(ne0  == nc);
    ਪ੍ਰਤਿਜ੍ਞਾ(ne2  == ne02);
    ਪ੍ਰਤਿਜ੍ਞਾ(ne3  == ne03);
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    ਪ੍ਰਤਿਜ੍ਞਾ(ne02 % ne11 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(ne03 % ne12 == 0);

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    // rows per thread
    ਅਚਲ int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ int64_t ir0 = dr*ith;
    ਅਚਲ int64_t ir1 = std::min(ir0 + dr, nr);

    ਅਚਲ size_t rs = ggml_ਪੰਕਤੀ_ਮਾਪ(src0->type, nc);

    ggml_from_float_t ਅਚਲ from_float = ggml_get_type_traits_cpu(dst->type)->from_float;

    ਲਈ (int64_t i03 = 0; i03 < ne03; ++i03) {
        ਲਈ (int64_t i02 = 0; i02 < ne02; ++i02) {
            ਲਈ (int64_t i = ir0; i < ir1; ++i) {
                ਅਚਲ int64_t i12 = i03%ne12;
                ਅਚਲ int64_t i11 = i02%ne11;
                ਅਚਲ int64_t i10 = i;

                ਅਚਲ int64_t i1 = *(idx_t *) ((ਅੱਖਰ *) src1->ਅੰਕੜੇ + i10*nb10 + i11*nb11 + i12*nb12);

                GGML_ASSERT(i1 >= 0 && i1 < ne1);

                ਜੇ constexpr (std::is_same_v<src_t, ਦਸ਼ਮਲਵ>) {
                    from_float(
                            (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ +  i*nb01 + i02*nb02 + i03*nb03),
                                            ((ਅੱਖਰ *)  dst->ਅੰਕੜੇ + i1*nb1  + i02*nb2  + i03*nb3), nc);
                } ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<src_t, ggml_fp16_t>) {
                    ਜੇ (dst->type == GGML_TYPE_F16) {
                        ਸਮ੍ਰਿਤੀ_ਨਕਲ(
                                            ((ਅੱਖਰ *)  dst->ਅੰਕੜੇ + i1*nb1  + i02*nb2  + i03*nb3),
                                            ((ਅੱਖਰ *) src0->ਅੰਕੜੇ +  i*nb01 + i02*nb02 + i03*nb03),
                                            rs);
                    } ਨਹੀਂਤਾਂ {
                        ਦਸ਼ਮਲਵ * wdata = (ਦਸ਼ਮਲਵ *) params->wdata + (nc + CACHE_ਸਤਰ_ਮਾਪ_F32) * ith;
                        ggml_fp16_ਮੰਜ਼ਿਲ_fp32_ਪੰਕਤੀ(
                                (ਅਚਲ ggml_fp16_t *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i*nb01 + i02*nb02 + i03*nb03),
                                wdata, nc);
                        from_float(wdata,
                                ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i1*nb1 + i02*nb2 + i03*nb3), nc);
                    }
                } ਨਹੀਂਤਾਂ {
                    GGML_ABORT("src0->type = %d (%s) not supported", src0->type, ggml_type_name(src0->type));
                }
            }
        }
    }
}

ਖਾਲੀ ggml_compute_forward_ਸੈੱਟ_ਪੰਕਤੀਆਂ(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ਜੇ (src1->type == GGML_TYPE_I64) {
                    ggml_compute_forward_ਸੈੱਟ_ਪੰਕਤੀਆਂ_impl<ਦਸ਼ਮਲਵ, int64_t>(params, dst);
                } ਨਹੀਂਤਾਂ ਜੇ (src1->type == GGML_TYPE_I32) {
                    ggml_compute_forward_ਸੈੱਟ_ਪੰਕਤੀਆਂ_impl<ਦਸ਼ਮਲਵ, int32_t>(params, dst);
                } ਨਹੀਂਤਾਂ {
                    GGML_ABORT("src1->type = %d (%s) not supported", src1->type, ggml_type_name(src1->type));
                }
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F16:
            {
                ਜੇ (src1->type == GGML_TYPE_I64) {
                    ggml_compute_forward_ਸੈੱਟ_ਪੰਕਤੀਆਂ_impl<ggml_fp16_t, int64_t>(params, dst);
                } ਨਹੀਂਤਾਂ ਜੇ (src1->type == GGML_TYPE_I32) {
                    ggml_compute_forward_ਸੈੱਟ_ਪੰਕਤੀਆਂ_impl<ggml_fp16_t, int32_t>(params, dst);
                } ਨਹੀਂਤਾਂ {
                    GGML_ABORT("src1->type = %d (%s) not supported", src1->type, ggml_type_name(src1->type));
                }
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("src0->type = %d (%s) not supported", src0->type, ggml_type_name(src0->type));
            }
    }
}

// ggml_compute_forward_get_rows_back

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_get_ਪੰਕਤੀਆਂ_ਪਿੱਠ_f32_f16(
        ਅਚਲ ggml_compute_params * params,
              ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    ਜੇ (params->ith != 0) {
        ਵਾਪਸ;
    }

    GGML_ASSERT(ggml_is_contiguous(dst));

    // ggml_compute_forward_dup_same_cont(params, opt0, dst);

    memset(dst->ਅੰਕੜੇ, 0, ggml_nbytes(dst));

    ਅਚਲ ਪੂਰਨ nc = src0->ne[0];
    ਅਚਲ ਪੂਰਨ nr = ggml_nelements(src1);

    GGML_ASSERT( dst->ne[0] == nc);
    GGML_ASSERT(src0->nb[0] == ਆਕਾਰ(ggml_fp16_t));

    ਲਈ (ਪੂਰਨ i = 0; i < nr; ++i) {
        ਅਚਲ ਪੂਰਨ r = ((int32_t *) src1->ਅੰਕੜੇ)[i];

        GGML_ASSERT(r >= 0 && r < dst->ne[1]);

        ਲਈ (ਪੂਰਨ j = 0; j < nc; ++j) {
            ggml_fp16_t v = ((ggml_fp16_t *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i*src0->nb[1]))[j];
            ((ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + r*dst->nb[1]))[j] += GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(v);
        }
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_get_ਪੰਕਤੀਆਂ_ਪਿੱਠ_f32(
        ਅਚਲ ggml_compute_params * params,
              ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    ਜੇ (params->ith != 0) {
        ਵਾਪਸ;
    }

    GGML_ASSERT(ggml_is_contiguous(dst));

    // ggml_compute_forward_dup_same_cont(params, opt0, dst);

    memset(dst->ਅੰਕੜੇ, 0, ggml_nbytes(dst));

    ਅਚਲ ਪੂਰਨ nc = src0->ne[0];
    ਅਚਲ ਪੂਰਨ nr = ggml_nelements(src1);

    GGML_ASSERT( dst->ne[0] == nc);
    GGML_ASSERT(src0->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਲਈ (ਪੂਰਨ i = 0; i < nr; ++i) {
        ਅਚਲ ਪੂਰਨ r = ((int32_t *) src1->ਅੰਕੜੇ)[i];

        GGML_ASSERT(r >= 0 && r < dst->ne[1]);

        ggml_vec_ਜੋੜੋ_f32(nc,
                (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *)  dst->ਅੰਕੜੇ + r*dst->nb[1]),
                (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *)  dst->ਅੰਕੜੇ + r*dst->nb[1]),
                (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i*src0->nb[1]));
    }
}

ਖਾਲੀ ggml_compute_forward_get_ਪੰਕਤੀਆਂ_ਪਿੱਠ(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F16:
            {
                ggml_compute_forward_get_ਪੰਕਤੀਆਂ_ਪਿੱਠ_f32_f16(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_get_ਪੰਕਤੀਆਂ_ਪਿੱਠ_f32(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
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

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_diag_f32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਜੇ (params->ith != 0) {
        ਵਾਪਸ;
    }

    // TODO: handle transposed/permuted matrices

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(ne00 == ne0);
    GGML_ASSERT(ne00 == ne1);
    GGML_ASSERT(ne01 == 1);
    GGML_ASSERT(ne02 == ne2);
    GGML_ASSERT(ne03 == ne3);

    GGML_ASSERT(nb00 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    GGML_ASSERT(nb0  == ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਲਈ (ਪੂਰਨ i3 = 0; i3 < ne3; i3++) {
        ਲਈ (ਪੂਰਨ i2 = 0; i2 < ne2; i2++) {
            ਲਈ (ਪੂਰਨ i1 = 0; i1 < ne1; i1++) {
                ਦਸ਼ਮਲਵ * d = (ਦਸ਼ਮਲਵ *)((ਅੱਖਰ *)  dst->ਅੰਕੜੇ + i3*nb3  + i2*nb2 + i1*nb1);
                ਦਸ਼ਮਲਵ * s = (ਦਸ਼ਮਲਵ *)((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i3*nb03 + i2*nb02);
                ਲਈ (ਪੂਰਨ i0 = 0; i0 < i1; i0++) {
                    d[i0] = 0;
                }
                d[i1] = s[i1];
                ਲਈ (ਪੂਰਨ i0 = i1+1; i0 < ne0; i0++) {
                    d[i0] = 0;
                }
            }
        }
    }
}

ਖਾਲੀ ggml_compute_forward_diag(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_diag_f32(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_diag_mask_inf

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_diag_ਨਕਾਬ_f32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst,
        ਅਚਲ ਦਸ਼ਮਲਵ value) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ  n_past  = ((int32_t *) dst->op_params)[0];
    ਅਚਲ ਤਾਰਕਿਕ inplace = src0->ਅੰਕੜੇ == dst->ਅੰਕੜੇ;

    GGML_ASSERT(n_past >= 0);

    ਜੇ (!inplace) {
        ਜੇ (ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
            GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));
            ਸਮ੍ਰਿਤੀ_ਨਕਲ(
                ((ਅੱਖਰ *)  dst->ਅੰਕੜੇ),
                ((ਅੱਖਰ *) src0->ਅੰਕੜੇ),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    // TODO: handle transposed/permuted matrices

    ਅਚਲ ਪੂਰਨ n  = ggml_nrows(src0);
    ਅਚਲ ਪੂਰਨ nc = src0->ne[0];
    ਅਚਲ ਪੂਰਨ nr = src0->ne[1];
    ਅਚਲ ਪੂਰਨ nz = n/nr;

    GGML_ASSERT( dst->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    GGML_ASSERT(src0->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਲਈ (ਪੂਰਨ k = 0; k < nz; k++) {
        ਲਈ (ਪੂਰਨ j = ith; j < nr; j += nth) {
            ਲਈ (ਪੂਰਨ i = n_past; i < nc; i++) {
                ਜੇ (i > n_past + j) {
                    *(ਦਸ਼ਮਲਵ *)((ਅੱਖਰ *) dst->ਅੰਕੜੇ + k*dst->nb[2] + j*dst->nb[1] + i*dst->nb[0]) = value;
                }
            }
        }
    }
}

ਖਾਲੀ ggml_compute_forward_diag_ਨਕਾਬ_inf(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_diag_ਨਕਾਬ_f32(params, dst, -INFINITY);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ਖਾਲੀ ggml_compute_forward_diag_ਨਕਾਬ_zero(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_diag_ਨਕਾਬ_f32(params, dst, 0);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_soft_max

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_soft_max_f32(
        ਅਚਲ ggml_compute_params * params,
              ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];
    ਅਚਲ ggml_tensor * src2 = dst->src[2];

    ਪ੍ਰਤਿਜ੍ਞਾ(ggml_is_contiguous(dst));
    ਪ੍ਰਤਿਜ੍ਞਾ(ggml_are_same_shape(src0, dst));

    ਦਸ਼ਮਲਵ scale    = 1.0f;
    ਦਸ਼ਮਲਵ max_bias = 0.0f;

    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&scale,    (ਦਸ਼ਮਲਵ *) dst->op_params + 0, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&max_bias, (ਦਸ਼ਮਲਵ *) dst->op_params + 1, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    ਅਚਲ int64_t nb11 = src1 ? src1->nb[1] : 1;
    ਅਚਲ int64_t nb12 = src1 ? src1->nb[2] : 1;
    ਅਚਲ int64_t nb13 = src1 ? src1->nb[3] : 1;

    ਅਚਲ int64_t ne12 = src1 ? src1->ne[2] : 1;
    ਅਚਲ int64_t ne13 = src1 ? src1->ne[3] : 1;

    // TODO: is this supposed to be ceil instead of floor?
    //       https://huggingface.co/mosaicml/mpt-7b/blob/main/attention.py#L370
    ਅਚਲ uint32_t n_ਸਿਰ      = ne02;
    ਅਚਲ uint32_t n_ਸਿਰ_log2 = 1u << (uint32_t) floor(log2(n_ਸਿਰ));

    ਅਚਲ ਦਸ਼ਮਲਵ m0 = powf(2.0f, -(max_bias       ) / n_ਸਿਰ_log2);
    ਅਚਲ ਦਸ਼ਮਲਵ m1 = powf(2.0f, -(max_bias / 2.0f) / n_ਸਿਰ_log2);

    ਦਸ਼ਮਲਵ * wp = (ਦਸ਼ਮਲਵ *) params->wdata + (ne00 + CACHE_ਸਤਰ_ਮਾਪ_F32) * ith;

    ਅਚਲ ਤਾਰਕਿਕ use_f16 = (src1 && src1->type == GGML_TYPE_F16);

    // sinks
    ਅਚਲ ਦਸ਼ਮਲਵ * sk = src2 ? (ਦਸ਼ਮਲਵ *)((ਅੱਖਰ *) src2->ਅੰਕੜੇ) : ਖਾਲੀ_ਸੂਚਕ;

    ਲਈ (int64_t i03 = 0; i03 < ne03; i03++) {
        ਲਈ (int64_t i02 = 0; i02 < ne02; i02++) {
            ਲਈ (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                ਅਚਲ int64_t i11 = i01;
                ਅਚਲ int64_t i12 = i02%ne12;
                ਅਚਲ int64_t i13 = i03%ne13;

                // ALiBi
                ਅਚਲ uint32_t h = i02; //ਸਿਰ
                ਅਚਲ ਦਸ਼ਮਲਵ ਢਲਾਨ = (max_bias > 0.0f) ? h < n_ਸਿਰ_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_ਸਿਰ_log2) + 1) : 1.0f;

                ਦਸ਼ਮਲਵ * sp = (ਦਸ਼ਮਲਵ *)((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i01*nb01 + i02*nb02 + i03*nb03);
                ਦਸ਼ਮਲਵ * dp = (ਦਸ਼ਮਲਵ *)((ਅੱਖਰ *)  dst->ਅੰਕੜੇ + i01*nb1  + i02*nb2  + i03*nb3);

                // broadcast the mask across rows
                ggml_fp16_t * mp_f16 = src1 ? (ggml_fp16_t *)((ਅੱਖਰ *) src1->ਅੰਕੜੇ + i11*nb11 + i12*nb12 + i13*nb13) : ਨਿਰਾ;
                ਦਸ਼ਮਲਵ       * mp_f32 = src1 ? (ਦਸ਼ਮਲਵ       *)((ਅੱਖਰ *) src1->ਅੰਕੜੇ + i11*nb11 + i12*nb12 + i13*nb13) : ਨਿਰਾ;

                ggml_vec_cpy_f32  (ne00, wp, sp);
                ggml_vec_scale_f32(ne00, wp, scale);
                ਜੇ (mp_f32) {
                    ਜੇ (use_f16) {
                        ਲਈ (ਪੂਰਨ i = 0; i < ne00; ++i) {
                            wp[i] += ਢਲਾਨ*GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(mp_f16[i]);
                        }
                    } ਨਹੀਂਤਾਂ {
                        ਲਈ (ਪੂਰਨ i = 0; i < ne00; ++i) {
                            wp[i] += ਢਲਾਨ*mp_f32[i];
                        }
                    }
                }

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NDEBUG
                ਲਈ (ਪੂਰਨ i = 0; i < ne00; ++i) {
                    //printf("p[%d] = %f\n", i, p[i]);
                    ਪ੍ਰਤਿਜ੍ਞਾ(!isnan(wp[i]));
                }
#ਜੇ_ਅੰਤ // NDEBUG

                ਦਸ਼ਮਲਵ max = -INFINITY;
                ggml_vec_max_f32(ne00, &max, wp);

                // if we have sinks, make a correction as if they were included in the softmax
                ਜੇ (sk) {
                    max = MAX(max, sk[i02]);
                }

                ggml_float ਜੋੜ = ggml_vec_soft_max_f32(ne00, dp, wp, max);
                ਪ੍ਰਤਿਜ੍ਞਾ(ਜੋੜ > 0.0);

                ਜੇ (sk) {
                    ਜੋੜ += (ggml_float) expf(sk[i02] - max);
                }

                ਜੋੜ = 1.0/ਜੋੜ;
                ggml_vec_scale_f32(ne00, dp, ਜੋੜ);

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NDEBUG
                ਲਈ (ਪੂਰਨ i = 0; i < ne00; ++i) {
                    ਪ੍ਰਤਿਜ੍ਞਾ(!isnan(dp[i]));
                    ਪ੍ਰਤਿਜ੍ਞਾ(!isinf(dp[i]));
                }
#ਜੇ_ਅੰਤ // NDEBUG
            }
        }
    }
}

ਖਾਲੀ ggml_compute_forward_soft_max(
        ਅਚਲ ggml_compute_params * params,
              ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_soft_max_f32(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}


// ggml_compute_forward_soft_max_ext_back

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_soft_max_ext_ਪਿੱਠ_f32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_are_same_shape(src1, dst));

    ਦਸ਼ਮਲਵ scale    = 1.0f;
    ਦਸ਼ਮਲਵ max_bias = 0.0f;

    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&scale,    (ਅਚਲ ਦਸ਼ਮਲਵ *) dst->op_params + 0, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&max_bias, (ਅਚਲ ਦਸ਼ਮਲਵ *) dst->op_params + 1, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    GGML_ASSERT(max_bias == 0.0f);

    // TODO: handle transposed/permuted matrices

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ nc = src0->ne[0];
    ਅਚਲ ਪੂਰਨ nr = ggml_nrows(src0);

    // rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr*ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਲਈ (ਪੂਰਨ i1 = ir0; i1 < ir1; i1++) {
        ਦਸ਼ਮਲਵ *dy = (ਦਸ਼ਮਲਵ *)((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i1*src0->nb[1]);
        ਦਸ਼ਮਲਵ *y  = (ਦਸ਼ਮਲਵ *)((ਅੱਖਰ *) src1->ਅੰਕੜੇ + i1*src1->nb[1]);
        ਦਸ਼ਮਲਵ *dx = (ਦਸ਼ਮਲਵ *)((ਅੱਖਰ *) dst->ਅੰਕੜੇ  + i1*dst->nb[1]);

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NDEBUG
        ਲਈ (ਪੂਰਨ i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            ਪ੍ਰਤਿਜ੍ਞਾ(!isnan(dy[i]));
            ਪ੍ਰਤਿਜ੍ਞਾ(!isnan(y[i]));
        }
#ਜੇ_ਅੰਤ // NDEBUG
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
        ਦਸ਼ਮਲਵ dot_y_dy = 0;
        ggml_vec_dot_f32(nc, &dot_y_dy, 0, y, 0, dy, 0, 1);
        ਲਈ (ਪੂਰਨ i = 0; i < nc; i++) {
            dx[i] = scale * (dy[i] - dot_y_dy) * y[i];
        }

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NDEBUG
        ਲਈ (ਪੂਰਨ i = 0; i < nc; ++i) {
            ਪ੍ਰਤਿਜ੍ਞਾ(!isnan(dx[i]));
            ਪ੍ਰਤਿਜ੍ਞਾ(!isinf(dx[i]));
        }
#ਜੇ_ਅੰਤ // NDEBUG
    }
}

ਖਾਲੀ ggml_compute_forward_soft_max_ext_ਪਿੱਠ(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_soft_max_ext_ਪਿੱਠ_f32(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_clamp

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_clamp_f32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਦਸ਼ਮਲਵ min;
    ਦਸ਼ਮਲਵ max;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&min, (ਦਸ਼ਮਲਵ *) dst->op_params + 0, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&max, (ਦਸ਼ਮਲਵ *) dst->op_params + 1, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ n  = ggml_nrows(src0);
    ਅਚਲ ਪੂਰਨ nc = src0->ne[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    GGML_ASSERT(nb00 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਲਈ (ਪੂਰਨ j = ith; j < n; j += nth) {
        ਅਚਲ int64_t i1 = j % ne01;
        ਅਚਲ int64_t i2 = (j / ne01) % ne02;
        ਅਚਲ int64_t i3 = j / (ne01*ne02);

        ਦਸ਼ਮਲਵ * dst_ptr  = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *)  dst->ਅੰਕੜੇ + i1*nb1  + i2*nb2  + i3*nb3);
        ਦਸ਼ਮਲਵ * src0_ptr = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i1*nb01 + i2*nb02 + i3*nb03);

        ਲਈ (ਪੂਰਨ i = 0; i < nc; i++) {
            dst_ptr[i] = MAX(MIN(src0_ptr[i], max), min);
        }
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_clamp_f16(
    ਅਚਲ ggml_compute_params * params,
    ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਦਸ਼ਮਲਵ min;
    ਦਸ਼ਮਲਵ max;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&min, (ਦਸ਼ਮਲਵ *) dst->op_params + 0, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&max, (ਦਸ਼ਮਲਵ *) dst->op_params + 1, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ n  = ggml_nrows(src0);
    ਅਚਲ ਪੂਰਨ nc = src0->ne[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == ਆਕਾਰ(ggml_fp16_t));
    GGML_ASSERT(nb00 == ਆਕਾਰ(ggml_fp16_t));

    ਲਈ (ਪੂਰਨ j = ith; j < n; j += nth) {
        ਅਚਲ int64_t i1 = j % ne01;
        ਅਚਲ int64_t i2 = (j / ne01) % ne02;
        ਅਚਲ int64_t i3 = j / (ne01*ne02);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((ਅੱਖਰ *)  dst->ਅੰਕੜੇ + i1*nb1  + i2*nb2  + i3*nb3);
        ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i1*nb01 + i2*nb02 + i3*nb03);

        ਲਈ (ਪੂਰਨ i = 0; i < nc; i++) {
            ਦਸ਼ਮਲਵ v = GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(src0_ptr[i]);
            dst_ptr[i] = GGML_CPU_FP32_ਮੰਜ਼ਿਲ_FP16(MAX(MIN(v, max), min));
        }
    }
}

ਖਾਲੀ ggml_compute_forward_clamp(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_clamp_f32(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F16:
            {
                ggml_compute_forward_clamp_f16(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_BF16:
        ਹਾਲਤ GGML_TYPE_Q1_0:
        ਹਾਲਤ GGML_TYPE_Q2_0:
        ਹਾਲਤ GGML_TYPE_Q4_0:
        ਹਾਲਤ GGML_TYPE_Q4_1:
        ਹਾਲਤ GGML_TYPE_Q5_0:
        ਹਾਲਤ GGML_TYPE_Q5_1:
        ਹਾਲਤ GGML_TYPE_Q8_0:
        ਹਾਲਤ GGML_TYPE_Q8_1:
        ਹਾਲਤ GGML_TYPE_MXFP4:
        ਹਾਲਤ GGML_TYPE_NVFP4:
        ਹਾਲਤ GGML_TYPE_Q2_K:
        ਹਾਲਤ GGML_TYPE_Q3_K:
        ਹਾਲਤ GGML_TYPE_Q4_K:
        ਹਾਲਤ GGML_TYPE_Q5_K:
        ਹਾਲਤ GGML_TYPE_Q6_K:
        ਹਾਲਤ GGML_TYPE_TQ1_0:
        ਹਾਲਤ GGML_TYPE_TQ2_0:
        ਹਾਲਤ GGML_TYPE_IQ2_XXS:
        ਹਾਲਤ GGML_TYPE_IQ2_XS:
        ਹਾਲਤ GGML_TYPE_IQ3_XXS:
        ਹਾਲਤ GGML_TYPE_IQ1_S:
        ਹਾਲਤ GGML_TYPE_IQ1_M:
        ਹਾਲਤ GGML_TYPE_IQ4_NL:
        ਹਾਲਤ GGML_TYPE_IQ4_XS:
        ਹਾਲਤ GGML_TYPE_IQ3_S:
        ਹਾਲਤ GGML_TYPE_IQ2_S:
        ਹਾਲਤ GGML_TYPE_Q8_K:
        ਹਾਲਤ GGML_TYPE_I8:
        ਹਾਲਤ GGML_TYPE_I16:
        ਹਾਲਤ GGML_TYPE_I32:
        ਹਾਲਤ GGML_TYPE_I64:
        ਹਾਲਤ GGML_TYPE_F64:
        ਹਾਲਤ GGML_TYPE_ਗਿਣਤੀ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_rope

ਸਥਿਰ ਦਸ਼ਮਲਵ rope_yarn_ramp(ਅਚਲ ਦਸ਼ਮਲਵ ਨੀਵਾਂ, ਅਚਲ ਦਸ਼ਮਲਵ ਉੱਚਾ, ਅਚਲ ਪੂਰਨ i0) {
    ਅਚਲ ਦਸ਼ਮਲਵ y = (i0 / 2 - ਨੀਵਾਂ) / MAX(0.001f, ਉੱਚਾ - ਨੀਵਾਂ);
    ਵਾਪਸ 1 - MIN(1, MAX(0, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
ਸਥਿਰ ਖਾਲੀ rope_yarn(
    ਦਸ਼ਮਲਵ theta_extrap, ਦਸ਼ਮਲਵ freq_scale, ਦਸ਼ਮਲਵ corr_dims[2], int64_t i0, ਦਸ਼ਮਲਵ ext_ਗੁਣਕ, ਦਸ਼ਮਲਵ mscale,
    ਦਸ਼ਮਲਵ * cos_theta, ਦਸ਼ਮਲਵ * sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    ਦਸ਼ਮਲਵ theta_interp = freq_scale * theta_extrap;
    ਦਸ਼ਮਲਵ theta = theta_interp;
    ਜੇ (ext_ਗੁਣਕ != 0.0f) {
        ਦਸ਼ਮਲਵ ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_ਗੁਣਕ;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
}

ਸਥਿਰ ਖਾਲੀ ggml_rope_cache_init(
     ਦਸ਼ਮਲਵ theta_ਅਧਾਰ, ਦਸ਼ਮਲਵ freq_scale, ਅਚਲ ਦਸ਼ਮਲਵ * freq_factors, ਦਸ਼ਮਲਵ corr_dims[2], int64_t ne0, ਦਸ਼ਮਲਵ ext_ਗੁਣਕ, ਦਸ਼ਮਲਵ mscale,
     ਦਸ਼ਮਲਵ * cache, ਦਸ਼ਮਲਵ sin_sign, ਦਸ਼ਮਲਵ theta_scale) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    ਦਸ਼ਮਲਵ theta = theta_ਅਧਾਰ;
    ਲਈ (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        ਅਚਲ ਦਸ਼ਮਲਵ ff = freq_factors ? freq_factors[i0/2] : 1.0f;
        rope_yarn(
            theta/ff, freq_scale, corr_dims, i0, ext_ਗੁਣਕ, mscale, &cache[i0 + 0], &cache[i0 + 1]
        );
        cache[i0 + 1] *= sin_sign;

        theta *= theta_scale;
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_mrope_cache_init(
     ਦਸ਼ਮਲਵ theta_ਅਧਾਰ_t, ਦਸ਼ਮਲਵ theta_ਅਧਾਰ_h, ਦਸ਼ਮਲਵ theta_ਅਧਾਰ_w, ਦਸ਼ਮਲਵ theta_ਅਧਾਰ_e, ਪੂਰਨ sections[4], ਤਾਰਕਿਕ is_imrope, ਤਾਰਕਿਕ indep_sects,
     ਦਸ਼ਮਲਵ freq_scale, ਅਚਲ ਦਸ਼ਮਲਵ * freq_factors, ਦਸ਼ਮਲਵ corr_dims[2], int64_t ne0, ਦਸ਼ਮਲਵ ext_ਗੁਣਕ, ਦਸ਼ਮਲਵ mscale,
     ਦਸ਼ਮਲਵ * cache, ਦਸ਼ਮਲਵ sin_sign, ਦਸ਼ਮਲਵ theta_scale) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    ਦਸ਼ਮਲਵ theta_t = theta_ਅਧਾਰ_t;
    ਦਸ਼ਮਲਵ theta_h = theta_ਅਧਾਰ_h;
    ਦਸ਼ਮਲਵ theta_w = theta_ਅਧਾਰ_w;
    ਦਸ਼ਮਲਵ theta_e = theta_ਅਧਾਰ_e;  // extra position id for vision encoder
    ਪੂਰਨ sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    ਪੂਰਨ sec_w = sections[1] + sections[0];
    ਪੂਰਨ sec_e = sections[2] + sec_w;
    GGML_ASSERT(sect_dims <= ne0);

    ਲਈ (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        ਅਚਲ ਦਸ਼ਮਲਵ ff = freq_factors ? freq_factors[i0/2] : 1.0f;

        ਪੂਰਨ sector = (i0 / 2) % sect_dims;
        ਜੇ (indep_sects) {
            // compute theta independently for each dim sections
            // (i.e. reset corresponding theta when `i0` go from one section to another)
            ਜੇ (sector == 0) {
                theta_t = theta_ਅਧਾਰ_t;
            }
            ਨਹੀਂਤਾਂ ਜੇ (sector == sections[0]) {
                theta_h = theta_ਅਧਾਰ_h;;
            }
            ਨਹੀਂਤਾਂ ਜੇ (sector == sec_w) {
                theta_w = theta_ਅਧਾਰ_w;
            }
            ਨਹੀਂਤਾਂ ਜੇ (sector == sec_e) {
                theta_e = theta_ਅਧਾਰ_e;
            }
        }

        ਦਸ਼ਮਲਵ theta = theta_t;
        ਜੇ (is_imrope) { // qwen3vl apply interleaved mrope
            ਜੇ (sector % 3 == 1 && sector < 3 * sections[1]) {
                theta = theta_h;
            } ਨਹੀਂਤਾਂ ਜੇ (sector % 3 == 2 && sector < 3 * sections[2]) {
                theta = theta_w;
            } ਨਹੀਂਤਾਂ ਜੇ (sector % 3 == 0 && sector < 3 * sections[0]) {
                theta = theta_t;
            } ਨਹੀਂਤਾਂ {
                theta = theta_e;
            }
        } ਨਹੀਂਤਾਂ {
            ਜੇ (sector >= sections[0] && sector < sec_w) {
                theta = theta_h;
            }
            ਨਹੀਂਤਾਂ ਜੇ (sector >= sec_w && sector < sec_w + sections[2]) {
                theta = theta_w;
            }
            ਨਹੀਂਤਾਂ ਜੇ (sector >= sec_w + sections[2]) {
                theta = theta_e;
            }
        }

        rope_yarn(
            theta/ff, freq_scale, corr_dims, i0, ext_ਗੁਣਕ, mscale, &cache[i0 + 0], &cache[i0 + 1]
        );
        cache[i0 + 1] *= sin_sign;

        theta_t *= theta_scale;
        theta_w *= theta_scale;
        theta_h *= theta_scale;
        theta_e *= theta_scale;
    }
}


ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ ਖਾਲੀ rotate_ਜੋੜੇ(ਅਚਲ int64_t n, ਅਚਲ int64_t n_ਵਿੱਥ, ਅਚਲ ਦਸ਼ਮਲਵ * cache, ਅਚਲ T * src_ਅੰਕੜੇ, T * dst_ਅੰਕੜੇ, ਅਚਲ ਪੂਰਨ scale = 2) {
  ਲਈ (int64_t i0 = 0; i0 < n; i0 += 2) {
    ਅਚਲ int64_t ic = i0/scale; // hack for GGML_ROPE_TYPE_NORMAL, where we need ic = i0; for all other cases, ic = i0/2

    ਅਚਲ ਦਸ਼ਮਲਵ cos_theta = cache[i0 + 0];
    ਅਚਲ ਦਸ਼ਮਲਵ sin_theta = cache[i0 + 1];

    ਅਚਲ T * ਅਚਲ src = src_ਅੰਕੜੇ + ic;
    T * dst             = dst_ਅੰਕੜੇ + ic;

    ਅਚਲ ਦਸ਼ਮਲਵ x0 = type_conversion_ਸਾਰਣੀ<T>::ਮੰਜ਼ਿਲ_f32(src[0]);
    ਅਚਲ ਦਸ਼ਮਲਵ x1 = type_conversion_ਸਾਰਣੀ<T>::ਮੰਜ਼ਿਲ_f32(src[n_ਵਿੱਥ]);

    dst[0]        = type_conversion_ਸਾਰਣੀ<T>::from_f32(x0*cos_theta - x1*sin_theta);
    dst[n_ਵਿੱਥ] = type_conversion_ਸਾਰਣੀ<T>::from_f32(x0*sin_theta + x1*cos_theta);
  }
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T> //float or ggml_fp16_t
ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_rope_flt(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst,
        ਅਚਲ ਤਾਰਕਿਕ forward) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];
    ਅਚਲ ggml_tensor * src2 = dst->src[2];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);

    ਦਸ਼ਮਲਵ freq_ਅਧਾਰ, freq_scale, ext_ਗੁਣਕ, attn_ਗੁਣਕ, beta_fast, beta_slow;
    ਪੂਰਨ sections[4];

    //const int n_past     = ((int32_t *) dst->op_params)[0];
    ਅਚਲ ਪੂਰਨ n_dims     = ((int32_t *) dst->op_params)[1];
    ਅਚਲ ਪੂਰਨ mode       = ((int32_t *) dst->op_params)[2];
    //const int n_ctx      = ((int32_t *) dst->op_params)[3];
    ਅਚਲ ਪੂਰਨ n_ctx_orig = ((int32_t *) dst->op_params)[4];

    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&freq_ਅਧਾਰ,   (int32_t *) dst->op_params +  5, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&freq_scale,  (int32_t *) dst->op_params +  6, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&ext_ਗੁਣਕ,  (int32_t *) dst->op_params +  7, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&attn_ਗੁਣਕ, (int32_t *) dst->op_params +  8, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&beta_fast,   (int32_t *) dst->op_params +  9, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&beta_slow,   (int32_t *) dst->op_params + 10, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&sections,    (int32_t *) dst->op_params + 11, ਆਕਾਰ(ਪੂਰਨ)*4);

    ਅਚਲ ਪੂਰਨ n_offs = ((int32_t *) dst->op_params)[15];

    GGML_TENSOR_UNARY_OP_LOCALS

    //printf("ne0: %d, ne1: %d, ne2: %d, ne3: %d\n", ne0, ne1, ne2, ne3);
    //printf("n_past = %d, ne2 = %d\n", n_past, ne2);

    GGML_ASSERT(nb0 == nb00);
    GGML_ASSERT(nb0 == ਆਕਾਰ(T));

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ nr = ggml_nrows(dst);

    GGML_ASSERT(n_dims <= ne0);
    GGML_ASSERT(n_dims % 2 == 0);

    GGML_ASSERT(n_offs >= 0);
    GGML_ASSERT(n_offs % 2 == 0);
    GGML_ASSERT(n_offs + n_dims <= ne0);

    // rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr*ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    // row index used to determine which thread to use
    ਪੂਰਨ ir = 0;

    ਅਚਲ ਦਸ਼ਮਲਵ theta_scale = powf(freq_ਅਧਾਰ, -2.0f/n_dims);

    ਦਸ਼ਮਲਵ corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_ਅਧਾਰ, beta_fast, beta_slow, corr_dims);

    ਅਚਲ ਤਾਰਕਿਕ is_imrope = mode == GGML_ROPE_TYPE_IMROPE; // qwen3vl apply interleaved mrope
    ਅਚਲ ਤਾਰਕਿਕ mrope_ਵਰਤਿਆ = mode & GGML_ROPE_TYPE_MROPE;  // ggml_rope_multi, note: also true for vision (24 & 8 == true) and for imrope
    ਅਚਲ ਤਾਰਕਿਕ is_vision = mode == GGML_ROPE_TYPE_VISION;

    ਜੇ (mrope_ਵਰਤਿਆ) {
        GGML_ASSERT(sections[0] > 0 || sections[1] > 0 || sections[2] > 0);
    }

    ਜੇ (is_vision) {
        GGML_ASSERT(n_dims == ne0/2);
        GGML_ASSERT(n_offs == 0);
    }

    ਅਚਲ ਦਸ਼ਮਲਵ * freq_factors = ਨਿਰਾ;
    ਜੇ (src2 != ਨਿਰਾ) {
        GGML_ASSERT(src2->type == GGML_TYPE_F32);
        GGML_ASSERT(src2->ne[0] >= n_dims / 2);
        freq_factors = (ਅਚਲ ਦਸ਼ਮਲਵ *) src2->ਅੰਕੜੇ;
    }

    // backward process uses inverse rotation by cos and sin.
    // cos and sin build a rotation matrix, where the inverse is the transpose.
    // this essentially just switches the sign of sin.
    ਅਚਲ ਦਸ਼ਮਲਵ sin_sign = forward ? 1.0f : -1.0f;

    ਅਚਲ int32_t * pos = (ਅਚਲ int32_t *) src1->ਅੰਕੜੇ;

    int64_t ਆਖਰੀ_i2 = -1;

    ਲਈ (int64_t i3 = 0; i3 < ne3; i3++) { // batch
        ਲਈ (int64_t i2 = 0; i2 < ne2; i2++) { // seq-len
            ਲਈ (int64_t i1 = 0; i1 < ne1; i1++) { // attn-heads
                ਜੇ (ir++ < ir0) ਜਾਰੀ; // skip rows mapped to other threads
                ਜੇ (ir   > ir1) ਰੋਕੋ;

                ਦਸ਼ਮਲਵ * cache = (ਦਸ਼ਮਲਵ *) params->wdata + (ne0 + CACHE_ਸਤਰ_ਮਾਪ_F32)*ith;
                ਜੇ (ਆਖਰੀ_i2 != i2) {
                    ਜੇ (!mrope_ਵਰਤਿਆ) {
                        ਅਚਲ int64_t p = pos[i2];
                        ggml_rope_cache_init(p, freq_scale, freq_factors, corr_dims, ne0, ext_ਗੁਣਕ, attn_ਗੁਣਕ, cache, sin_sign, theta_scale);
                    }
                    ਨਹੀਂਤਾਂ {
                        ਅਚਲ int64_t p_t = pos[i2];
                        ਅਚਲ int64_t p_h = pos[i2 + ne2];
                        ਅਚਲ int64_t p_w = pos[i2 + ne2 * 2];
                        ਅਚਲ int64_t p_e = pos[i2 + ne2 * 3];
                        ggml_mrope_cache_init(
                            p_t, p_h, p_w, p_e, sections, is_imrope, is_vision,
                            freq_scale, freq_factors, corr_dims, ne0, ext_ਗੁਣਕ, attn_ਗੁਣਕ, cache, sin_sign, theta_scale);
                    }

                    ਆਖਰੀ_i2 = i2;
                }

                T * src = (T *)((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i3*nb03 + i2*nb02 + i1*nb01);
                T * dst_ਅੰਕੜੇ  = (T *)((ਅੱਖਰ *)  dst->ਅੰਕੜੇ + i3*nb3  + i2*nb2  + i1*nb1);

                ਚੋਣ (mode) {
                    ਹਾਲਤ GGML_ROPE_TYPE_ਸਾਧਾਰਨ:
                        rotate_ਜੋੜੇ<T>(n_dims, 1, cache, src + n_offs, dst_ਅੰਕੜੇ + n_offs, 1);
                        ਰੋਕੋ;
                    ਹਾਲਤ GGML_ROPE_TYPE_NEOX:
                    ਹਾਲਤ GGML_ROPE_TYPE_MROPE:
                    ਹਾਲਤ GGML_ROPE_TYPE_IMROPE:
                        rotate_ਜੋੜੇ<T>(n_dims, n_dims/2, cache, src + n_offs, dst_ਅੰਕੜੇ + n_offs);
                        ਰੋਕੋ;
                    ਹਾਲਤ GGML_ROPE_TYPE_VISION:
                        rotate_ਜੋੜੇ<T>(ne0, n_dims, cache, src, dst_ਅੰਕੜੇ);
                        ਰੋਕੋ;
                    ਮੂਲ:
                        GGML_ABORT("rope type not supported");
                }

                ਜੇ (!is_vision) {
                    // fill the remain channels with data from src tensor
                    ਲਈ (int64_t i0 = 0; i0 < ne0; i0 += 2) {
                        ਜੇ (i0 == n_offs) {
                            i0 += n_dims - 2; // skip the rotated channels
                            ਜਾਰੀ;
                        }
                        ਅਚਲ T * ਅਚਲ src = (T *)((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                        T * dst_ਅੰਕੜੇ  = (T *)((ਅੱਖਰ *)  dst->ਅੰਕੜੇ + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

                        dst_ਅੰਕੜੇ[0] = src[0];
                        dst_ਅੰਕੜੇ[1] = src[1];
                    }
                }
            } //attn-heads
        }
    }
}

ਖਾਲੀ ggml_compute_forward_rope(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F16:
            {
                ggml_compute_forward_rope_flt<ggml_fp16_t>(params, dst, ਸੱਚ);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_rope_flt<ਦਸ਼ਮਲਵ>(params, dst, ਸੱਚ);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_rope_back

ਖਾਲੀ ggml_compute_forward_rope_ਪਿੱਠ(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F16:
            {
                ggml_compute_forward_rope_flt<ggml_fp16_t>(params, dst, ਝੂਠ);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_rope_flt<ਦਸ਼ਮਲਵ>(params, dst, ਝੂਠ);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_conv_transpose_1d

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_conv_transpose_1d_f16_f32(
        ਅਚਲ ggml_compute_params * params,
              ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ nk = ne00*ne01*ne02;

    GGML_ASSERT(nb00 == ਆਕਾਰ(ggml_fp16_t));
    GGML_ASSERT(nb10 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਜੇ (ith == 0) {
        memset(params->wdata, 0, params->wsize);

        // permute kernel data (src0) from (K x Cout x Cin) to (Cin x K x Cout)
        {
            ggml_fp16_t * ਅਚਲ wdata = (ggml_fp16_t *) params->wdata + 0;

            ਲਈ (int64_t i02 = 0; i02 < ne02; i02++) {
                ਲਈ (int64_t i01 = 0; i01 < ne01; i01++) {
                    ਅਚਲ ggml_fp16_t * ਅਚਲ src = (ggml_fp16_t *)((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i02*nb02 + i01*nb01);
                    ggml_fp16_t * dst_ਅੰਕੜੇ = wdata + i01*ne00*ne02;
                    ਲਈ (int64_t i00 = 0; i00 < ne00; i00++) {
                        dst_ਅੰਕੜੇ[i00*ne02 + i02] = src[i00];
                    }
                }
            }
        }

        // permute source data (src1) from (L x Cin) to (Cin x L)
        {
            ggml_fp16_t * ਅਚਲ wdata = (ggml_fp16_t *) params->wdata + nk;
            ggml_fp16_t * dst_ਅੰਕੜੇ = wdata;

            ਲਈ (int64_t i11 = 0; i11 < ne11; i11++) {
                ਅਚਲ ਦਸ਼ਮਲਵ * ਅਚਲ src = (ਦਸ਼ਮਲਵ *)((ਅੱਖਰ *) src1->ਅੰਕੜੇ + i11*nb11);
                ਲਈ (int64_t i10 = 0; i10 < ne10; i10++) {
                    dst_ਅੰਕੜੇ[i10*ne11 + i11] = GGML_CPU_FP32_ਮੰਜ਼ਿਲ_FP16(src[i10]);
                }
            }
        }

        // need to zero dst since we are accumulating into it
        memset(dst->ਅੰਕੜੇ, 0, ggml_nbytes(dst));
    }
    ggml_barrier(params->threadpool);

    ਅਚਲ int32_t s0 = ((ਅਚਲ int32_t*)(dst->op_params))[0];

    // total rows in dst
    ਅਚਲ ਪੂਰਨ nr = ne1;

    // rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr*ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ggml_fp16_t * ਅਚਲ wdata     = (ggml_fp16_t *) params->wdata + 0;
    ggml_fp16_t * ਅਚਲ wdata_src = wdata + nk;

    ਲਈ (ਪੂਰਨ i1 = ir0; i1 < ir1; i1++) {
        ਦਸ਼ਮਲਵ * dst_ਅੰਕੜੇ = (ਦਸ਼ਮਲਵ *)((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i1*nb1);
        ggml_fp16_t * wdata_kernel = wdata + i1*ne02*ne00;
        ਲਈ (ਪੂਰਨ i10 = 0; i10 < ne10; i10++) {
            ਅਚਲ ਪੂਰਨ i1n = i10*ne11;
            ਲਈ (ਪੂਰਨ i00 = 0; i00 < ne00; i00++) {
                ਦਸ਼ਮਲਵ v = 0;
                ggml_vec_dot_f16(ne02, &v, 0,
                        (ggml_fp16_t *)    wdata_src + i1n, 0,
                        (ggml_fp16_t *) wdata_kernel + i00*ne02, 0, 1);
                dst_ਅੰਕੜੇ[i10*s0 + i00] += v;
            }
        }
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_conv_transpose_1d_f32(
        ਅਚਲ ggml_compute_params * params,
              ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ nk = ne00*ne01*ne02;

    GGML_ASSERT(nb00 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    GGML_ASSERT(nb10 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਜੇ (ith == 0) {
        memset(params->wdata, 0, params->wsize);

        // prepare kernel data (src0) from (K x Cout x Cin) to (Cin x K x Cout)
        {
            ਦਸ਼ਮਲਵ * ਅਚਲ wdata = (ਦਸ਼ਮਲਵ *) params->wdata + 0;

            ਲਈ (int64_t i02 = 0; i02 < ne02; i02++) {
                ਲਈ (int64_t i01 = 0; i01 < ne01; i01++) {
                    ਅਚਲ ਦਸ਼ਮਲਵ * ਅਚਲ src = (ਦਸ਼ਮਲਵ *)((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i02*nb02 + i01*nb01);
                    ਦਸ਼ਮਲਵ * dst_ਅੰਕੜੇ = wdata + i01*ne00*ne02;
                    ਲਈ (int64_t i00 = 0; i00 < ne00; i00++) {
                        dst_ਅੰਕੜੇ[i00*ne02 + i02] = src[i00];
                    }
                }
            }
        }

        // prepare source data (src1)
        {
            ਦਸ਼ਮਲਵ * ਅਚਲ wdata = (ਦਸ਼ਮਲਵ *) params->wdata + nk;
            ਦਸ਼ਮਲਵ * dst_ਅੰਕੜੇ = wdata;

            ਲਈ (int64_t i11 = 0; i11 < ne11; i11++) {
                ਅਚਲ ਦਸ਼ਮਲਵ * ਅਚਲ src = (ਦਸ਼ਮਲਵ *)((ਅੱਖਰ *) src1->ਅੰਕੜੇ + i11*nb11);
                ਲਈ (int64_t i10 = 0; i10 < ne10; i10++) {
                    dst_ਅੰਕੜੇ[i10*ne11 + i11] = src[i10];
                }
            }
        }

        // need to zero dst since we are accumulating into it
        memset(dst->ਅੰਕੜੇ, 0, ggml_nbytes(dst));
    }
    ggml_barrier(params->threadpool);

    ਅਚਲ int32_t s0 = ((ਅਚਲ int32_t*)(dst->op_params))[0];

    // total rows in dst
    ਅਚਲ ਪੂਰਨ nr = ne1;

    // rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr*ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਦਸ਼ਮਲਵ * ਅਚਲ wdata     = (ਦਸ਼ਮਲਵ *) params->wdata + 0;
    ਦਸ਼ਮਲਵ * ਅਚਲ wdata_src = wdata + nk;

    ਲਈ (ਪੂਰਨ i1 = ir0; i1 < ir1; i1++) {
        ਦਸ਼ਮਲਵ * dst_ਅੰਕੜੇ = (ਦਸ਼ਮਲਵ *)((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i1*nb1);
        ਦਸ਼ਮਲਵ * wdata_kernel = wdata + i1*ne02*ne00;
        ਲਈ (ਪੂਰਨ i10 = 0; i10 < ne10; i10++) {
            ਅਚਲ ਪੂਰਨ i1n = i10*ne11;
            ਲਈ (ਪੂਰਨ i00 = 0; i00 < ne00; i00++) {
                ਦਸ਼ਮਲਵ v = 0;
                ggml_vec_dot_f32(ne02, &v, 0,
                        wdata_src + i1n, 0,
                        wdata_kernel + i00*ne02, 0, 1);
                dst_ਅੰਕੜੇ[i10*s0 + i00] += v;
            }
        }
    }
}

ਖਾਲੀ ggml_compute_forward_conv_transpose_1d(
        ਅਚਲ ggml_compute_params * params,
              ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F16:
            {
                ggml_compute_forward_conv_transpose_1d_f16_f32(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_conv_transpose_1d_f32(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_im2col_f32
// src0: kernel [OC, IC, KH, KW]
// src1: image [N, IC, IH, IW]
// dst:  result [N, OH, OW, IC*KH*KW]
ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_im2col_f32(
        ਅਚਲ ggml_compute_params * params,
              ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS;

    ਅਚਲ int32_t s0 = ((ਅਚਲ int32_t *)(dst->op_params))[0];
    ਅਚਲ int32_t s1 = ((ਅਚਲ int32_t *)(dst->op_params))[1];
    ਅਚਲ int32_t p0 = ((ਅਚਲ int32_t *)(dst->op_params))[2];
    ਅਚਲ int32_t p1 = ((ਅਚਲ int32_t *)(dst->op_params))[3];
    ਅਚਲ int32_t d0 = ((ਅਚਲ int32_t *)(dst->op_params))[4];
    ਅਚਲ int32_t d1 = ((ਅਚਲ int32_t *)(dst->op_params))[5];
    ਅਚਲ ਤਾਰਕਿਕ is_2D = ((ਅਚਲ int32_t *)(dst->op_params))[6] == 1;

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ int64_t N  = is_2D ? ne13 : ne12;
    ਅਚਲ int64_t IC = is_2D ? ne12 : ne11;
    ਅਚਲ int64_t IH = is_2D ? ne11 : 1;
    ਅਚਲ int64_t IW = ne10;

    ਅਚਲ int64_t KH = is_2D ? ne01 : 1;
    ਅਚਲ int64_t KW = ne00;

    ਅਚਲ int64_t OH = is_2D ? ne2 : 1;
    ਅਚਲ int64_t OW = ne1;

    ਪੂਰਨ ofs0 = is_2D ? nb13 : nb12;
    ਪੂਰਨ ofs1 = is_2D ? nb12 : nb11;

    GGML_ASSERT(nb10 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    {
        ਦਸ਼ਮਲਵ * ਅਚਲ wdata = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;

        ਲਈ (int64_t ਵਿੱਚ = 0; ਵਿੱਚ < N; ਵਿੱਚ++) {
            ਲਈ (int64_t ioh = 0; ioh < OH; ioh++) { // 1
                ਲਈ (int64_t iow = 0; iow < OW; iow++) {
                    ਲਈ (int64_t iic = ith; iic < IC; iic += nth) {

                        // micro kernel
                        ਦਸ਼ਮਲਵ * dst_ਅੰਕੜੇ = wdata + (ਵਿੱਚ*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
                        ਅਚਲ ਦਸ਼ਮਲਵ * ਅਚਲ src_ਅੰਕੜੇ = (ਦਸ਼ਮਲਵ *)((ਅੱਖਰ *) src1->ਅੰਕੜੇ + ਵਿੱਚ*ofs0 + iic*ofs1); // [IH, IW]

                        ਲਈ (int64_t ikh = 0; ikh < KH; ikh++) {  // 1
                            ਲਈ (int64_t ikw = 0; ikw < KW; ikw++) {
                                ਅਚਲ int64_t iiw = iow*s0 + ikw*d0 - p0;
                                ਅਚਲ int64_t iih = ioh*s1 + ikh*d1 - p1;

                                ਜੇ (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                    dst_ਅੰਕੜੇ[iic*(KH*KW) + ikh*KW + ikw] = 0;
                                } ਨਹੀਂਤਾਂ {
                                    dst_ਅੰਕੜੇ[iic*(KH*KW) + ikh*KW + ikw] = (src_ਅੰਕੜੇ[iih*IW + iiw]);
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
ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_im2col_f16(
        ਅਚਲ ggml_compute_params * params,
              ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16);

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS;

    ਅਚਲ int32_t s0 = ((ਅਚਲ int32_t *)(dst->op_params))[0];
    ਅਚਲ int32_t s1 = ((ਅਚਲ int32_t *)(dst->op_params))[1];
    ਅਚਲ int32_t p0 = ((ਅਚਲ int32_t *)(dst->op_params))[2];
    ਅਚਲ int32_t p1 = ((ਅਚਲ int32_t *)(dst->op_params))[3];
    ਅਚਲ int32_t d0 = ((ਅਚਲ int32_t *)(dst->op_params))[4];
    ਅਚਲ int32_t d1 = ((ਅਚਲ int32_t *)(dst->op_params))[5];
    ਅਚਲ ਤਾਰਕਿਕ is_2D = ((ਅਚਲ int32_t *)(dst->op_params))[6] == 1;

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ int64_t N  = is_2D ? ne13 : ne12;
    ਅਚਲ int64_t IC = is_2D ? ne12 : ne11;
    ਅਚਲ int64_t IH = is_2D ? ne11 : 1;
    ਅਚਲ int64_t IW = ne10;

    ਅਚਲ int64_t KH = is_2D ? ne01 : 1;
    ਅਚਲ int64_t KW = ne00;

    ਅਚਲ int64_t OH = is_2D ? ne2 : 1;
    ਅਚਲ int64_t OW = ne1;

    ਪੂਰਨ ofs0 = is_2D ? nb13 : nb12;
    ਪੂਰਨ ofs1 = is_2D ? nb12 : nb11;

    GGML_ASSERT(nb10 == ggml_type_ਮਾਪ(src1->type));

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    {
        ggml_fp16_t * ਅਚਲ wdata = (ggml_fp16_t *) dst->ਅੰਕੜੇ;

        ਲਈ (int64_t ਵਿੱਚ = 0; ਵਿੱਚ < N; ਵਿੱਚ++) {
            ਲਈ (int64_t ioh = 0; ioh < OH; ioh++) { // 1
                ਲਈ (int64_t iow = 0; iow < OW; iow++) {
                    ਲਈ (int64_t iic = ith; iic < IC; iic += nth) {

                        // micro kernel
                        ggml_fp16_t * dst_ਅੰਕੜੇ = wdata + (ਵਿੱਚ*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
                        ਅਚਲ ਦਸ਼ਮਲਵ * ਅਚਲ src_ਅੰਕੜੇ_f32 = src1->type == GGML_TYPE_F32
                            ? (ਅਚਲ ਦਸ਼ਮਲਵ *)((ਅਚਲ ਅੱਖਰ *) src1->ਅੰਕੜੇ + ਵਿੱਚ*ofs0 + iic*ofs1)
                            : ਖਾਲੀ_ਸੂਚਕ; // [IH, IW]
                        ਅਚਲ ggml_fp16_t * ਅਚਲ src_ਅੰਕੜੇ_f16 = src1->type == GGML_TYPE_F16
                            ? (ਅਚਲ ggml_fp16_t *)((ਅਚਲ ਅੱਖਰ *) src1->ਅੰਕੜੇ + ਵਿੱਚ*ofs0 + iic*ofs1)
                            : ਖਾਲੀ_ਸੂਚਕ; // [IH, IW]

                        ਲਈ (int64_t ikh = 0; ikh < KH; ikh++) {  // 1
                            ਲਈ (int64_t ikw = 0; ikw < KW; ikw++) {
                                ਅਚਲ int64_t iiw = iow*s0 + ikw*d0 - p0;
                                ਅਚਲ int64_t iih = ioh*s1 + ikh*d1 - p1;

                                ਜੇ (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                    dst_ਅੰਕੜੇ[iic*(KH*KW) + ikh*KW + ikw] = 0;
                                } ਨਹੀਂਤਾਂ {
                                    ਜੇ (src_ਅੰਕੜੇ_f32 != ਖਾਲੀ_ਸੂਚਕ) {
                                        dst_ਅੰਕੜੇ[iic*(KH*KW) + ikh*KW + ikw] = GGML_CPU_FP32_ਮੰਜ਼ਿਲ_FP16(src_ਅੰਕੜੇ_f32[iih*IW + iiw]);
                                    } ਨਹੀਂਤਾਂ {
                                        dst_ਅੰਕੜੇ[iic*(KH*KW) + ikh*KW + ikw] = src_ਅੰਕੜੇ_f16[iih*IW + iiw];
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

ਖਾਲੀ ggml_compute_forward_im2col(
        ਅਚਲ ggml_compute_params * params,
              ggml_tensor * dst) {
    ਚੋਣ (dst->type) {
        ਹਾਲਤ GGML_TYPE_F16:
            {
                ggml_compute_forward_im2col_f16(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_im2col_f32(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_im2col_back_f32

ਖਾਲੀ ggml_compute_forward_im2col_ਪਿੱਠ_f32(
        ਅਚਲ ggml_compute_params * params,
              ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0]; // gradients of forward pass output
    ਅਚਲ ggml_tensor * src1 = dst->src[1]; // convolution kernel

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS;

    ਅਚਲ int32_t s0 = ((ਅਚਲ int32_t *)(dst->op_params))[0];
    ਅਚਲ int32_t s1 = ((ਅਚਲ int32_t *)(dst->op_params))[1];
    ਅਚਲ int32_t p0 = ((ਅਚਲ int32_t *)(dst->op_params))[2];
    ਅਚਲ int32_t p1 = ((ਅਚਲ int32_t *)(dst->op_params))[3];
    ਅਚਲ int32_t d0 = ((ਅਚਲ int32_t *)(dst->op_params))[4];
    ਅਚਲ int32_t d1 = ((ਅਚਲ int32_t *)(dst->op_params))[5];
    ਅਚਲ ਤਾਰਕਿਕ is_2D = ((ਅਚਲ int32_t *)(dst->op_params))[6] == 1;

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ int64_t N  = is_2D ? ne3 : ne2;
    ਅਚਲ int64_t IC = is_2D ? ne2 : ne1;
    ਅਚਲ int64_t IH = is_2D ? ne1 : 1;
    ਅਚਲ int64_t IW = ne0;

    ਅਚਲ int64_t KH = is_2D ? ne11 : 1;
    ਅਚਲ int64_t KW = ne10;

    ਅਚਲ int64_t OH = is_2D ? ne02 : 1;
    ਅਚਲ int64_t OW = ne01;

    ਪੂਰਨ ofs0 = is_2D ? nb3 : nb2;
    ਪੂਰਨ ofs1 = is_2D ? nb2 : nb1;

    GGML_ASSERT(nb0  == ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    {
        ਦਸ਼ਮਲਵ * ਅਚਲ wdata = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;

        ਲਈ (int64_t ਵਿੱਚ = 0; ਵਿੱਚ < N; ਵਿੱਚ++) {
            ਲਈ (int64_t iic = ith; iic < IC; iic += nth) {
                ਲਈ (int64_t iih = 0; iih < IH; iih++) {
                    ਲਈ (int64_t iiw = 0; iiw < IW; iiw++) {

                        // micro kernel
                        ਦਸ਼ਮਲਵ grad = 0.0f;
                        ਲਈ (int64_t ikh = 0; ikh < KH; ikh++) {
                            ਲਈ (int64_t ikw = 0; ikw < KW; ikw++) {
                                // For s0 > 1 some values were skipped over in the forward pass.
                                // These values have tmpw % s0 != 0 and need to be skipped in the backwards pass as well.
                                ਅਚਲ int64_t tmpw = (iiw + p0 - ikw*d0);
                                ਜੇ (tmpw % s0 != 0) {
                                    ਜਾਰੀ;
                                }
                                ਅਚਲ int64_t iow = tmpw / s0;

                                // Equivalent logic as above except for s1.
                                int64_t ioh;
                                ਜੇ (is_2D) {
                                    ਅਚਲ int64_t tmph = iih + p1 - ikh*d1;

                                    ਜੇ (tmph % s1 != 0) {
                                        ਜਾਰੀ;
                                    }

                                    ioh = tmph / s1;
                                } ਨਹੀਂਤਾਂ {
                                    ioh = 0;
                                }

                                ਜੇ (iow < 0 || iow >= OW || ioh < 0 || ioh >= OH) {
                                    ਜਾਰੀ;
                                }

                                ਅਚਲ ਦਸ਼ਮਲਵ * ਅਚਲ grad_ਵਿੱਚ = (ਅਚਲ ਦਸ਼ਮਲਵ *) src0->ਅੰਕੜੇ
                                    + (ਵਿੱਚ*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
                                grad += grad_ਵਿੱਚ[iic*(KH*KW) + ikh*KW + ikw];
                            }
                        }
                        ਦਸ਼ਮਲਵ * dst_ਅੰਕੜੇ = (ਦਸ਼ਮਲਵ *)((ਅੱਖਰ *) wdata + (ਵਿੱਚ*ofs0 + iic*ofs1)); // [IH, IW]
                        dst_ਅੰਕੜੇ[iih*IW + iiw] = grad;
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
ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_im2col_3d_f16(
        ਅਚਲ ggml_compute_params * params,
              ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16);

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS;

    ਅਚਲ int32_t s0 = ((ਅਚਲ int32_t *)(dst->op_params))[0];
    ਅਚਲ int32_t s1 = ((ਅਚਲ int32_t *)(dst->op_params))[1];
    ਅਚਲ int32_t s2 = ((ਅਚਲ int32_t *)(dst->op_params))[2];
    ਅਚਲ int32_t p0 = ((ਅਚਲ int32_t *)(dst->op_params))[3];
    ਅਚਲ int32_t p1 = ((ਅਚਲ int32_t *)(dst->op_params))[4];
    ਅਚਲ int32_t p2 = ((ਅਚਲ int32_t *)(dst->op_params))[5];
    ਅਚਲ int32_t d0 = ((ਅਚਲ int32_t *)(dst->op_params))[6];
    ਅਚਲ int32_t d1 = ((ਅਚਲ int32_t *)(dst->op_params))[7];
    ਅਚਲ int32_t d2 = ((ਅਚਲ int32_t *)(dst->op_params))[8];
    ਅਚਲ int32_t IC = ((ਅਚਲ int32_t *)(dst->op_params))[9];


    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ int64_t N  = ne13 / IC;
    ਅਚਲ int64_t ID = ne12;
    ਅਚਲ int64_t IH = ne11;
    ਅਚਲ int64_t IW = ne10;

    ਅਚਲ int64_t OC = ne03 / IC;
    GGML_ਅਣਵਰਤਿਆ(OC);
    ਅਚਲ int64_t KD = ne02;
    ਅਚਲ int64_t KH = ne01;
    ਅਚਲ int64_t KW = ne00;

    ਅਚਲ int64_t OD = ne3 / N;
    ਅਚਲ int64_t OH = ne2;
    ਅਚਲ int64_t OW = ne1;
    ਅਚਲ int64_t OH_OW = OH*OW;
    ਅਚਲ int64_t KD_KH_KW = KD*KH*KW;
    ਅਚਲ int64_t KH_KW = KH*KW;
    ਅਚਲ int64_t IC_KD_KH_KW = IC*KD*KH*KW;

    GGML_ASSERT(nb10 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    // im2col: [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
    {
        ggml_fp16_t * ਅਚਲ wdata = (ggml_fp16_t *) dst->ਅੰਕੜੇ;

        ਲਈ (int64_t ਵਿੱਚ = 0; ਵਿੱਚ < N; ਵਿੱਚ++) {
            ਲਈ (int64_t iod = 0; iod < OD; iod++) {
                ਲਈ (int64_t ioh = 0; ioh < OH; ioh++) {
                    ਲਈ (int64_t iow = 0; iow < OW; iow++) {
                        ਲਈ (int64_t iic = ith; iic < IC; iic += nth) {

                            // micro kernel
                            ggml_fp16_t * dst_ਅੰਕੜੇ = wdata + (ਵਿੱਚ*OD*OH_OW + iod*OH_OW + ioh*OW + iow)*IC_KD_KH_KW; // [IC, KD, KH, KW]
                            ਅਚਲ ਦਸ਼ਮਲਵ * ਅਚਲ src_ਅੰਕੜੇ = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *)src1->ਅੰਕੜੇ + (ਵਿੱਚ*IC + iic)*nb13); // [ID, IH, IW]

                            ਲਈ (int64_t ikd = 0; ikd < KD; ikd++) {
                                ਲਈ (int64_t ikh = 0; ikh < KH; ikh++) {
                                    ਲਈ (int64_t ikw = 0; ikw < KW; ikw++) {
                                        ਅਚਲ int64_t iiw = iow*s0 + ikw*d0 - p0;
                                        ਅਚਲ int64_t iih = ioh*s1 + ikh*d1 - p1;
                                        ਅਚਲ int64_t iid = iod*s2 + ikd*d2 - p2;

                                        ਜੇ (iid < 0 || iid >= ID || iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                            dst_ਅੰਕੜੇ[iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw] = 0;
                                        } ਨਹੀਂਤਾਂ {
                                            ਅਚਲ ਦਸ਼ਮਲਵ * ਅਚਲ s = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *)src_ਅੰਕੜੇ + iid*nb12 + iih*nb11 + iiw*nb10); // [ID, IH, IW]
                                            dst_ਅੰਕੜੇ[iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw] = GGML_CPU_FP32_ਮੰਜ਼ਿਲ_FP16(*s);
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
ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_im2col_3d_f32(
        ਅਚਲ ggml_compute_params * params,
              ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS;

    ਅਚਲ int32_t s0 = ((ਅਚਲ int32_t *)(dst->op_params))[0];
    ਅਚਲ int32_t s1 = ((ਅਚਲ int32_t *)(dst->op_params))[1];
    ਅਚਲ int32_t s2 = ((ਅਚਲ int32_t *)(dst->op_params))[2];
    ਅਚਲ int32_t p0 = ((ਅਚਲ int32_t *)(dst->op_params))[3];
    ਅਚਲ int32_t p1 = ((ਅਚਲ int32_t *)(dst->op_params))[4];
    ਅਚਲ int32_t p2 = ((ਅਚਲ int32_t *)(dst->op_params))[5];
    ਅਚਲ int32_t d0 = ((ਅਚਲ int32_t *)(dst->op_params))[6];
    ਅਚਲ int32_t d1 = ((ਅਚਲ int32_t *)(dst->op_params))[7];
    ਅਚਲ int32_t d2 = ((ਅਚਲ int32_t *)(dst->op_params))[8];
    ਅਚਲ int32_t IC = ((ਅਚਲ int32_t *)(dst->op_params))[9];


    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ int64_t N  = ne13 / IC;
    ਅਚਲ int64_t ID = ne12;
    ਅਚਲ int64_t IH = ne11;
    ਅਚਲ int64_t IW = ne10;

    ਅਚਲ int64_t OC = ne03 / IC;
    GGML_ਅਣਵਰਤਿਆ(OC);
    ਅਚਲ int64_t KD = ne02;
    ਅਚਲ int64_t KH = ne01;
    ਅਚਲ int64_t KW = ne00;

    ਅਚਲ int64_t OD = ne3 / N;
    ਅਚਲ int64_t OH = ne2;
    ਅਚਲ int64_t OW = ne1;

    ਅਚਲ int64_t OH_OW = OH*OW;
    ਅਚਲ int64_t KD_KH_KW = KD*KH*KW;
    ਅਚਲ int64_t KH_KW = KH*KW;
    ਅਚਲ int64_t IC_KD_KH_KW = IC*KD*KH*KW;

    GGML_ASSERT(nb10 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    // im2col: [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
    {
        ਦਸ਼ਮਲਵ * ਅਚਲ wdata = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;

        ਲਈ (int64_t ਵਿੱਚ = 0; ਵਿੱਚ < N; ਵਿੱਚ++) {
            ਲਈ (int64_t iod = 0; iod < OD; iod++) {
                ਲਈ (int64_t ioh = 0; ioh < OH; ioh++) {
                    ਲਈ (int64_t iow = 0; iow < OW; iow++) {
                        ਲਈ (int64_t iic = ith; iic < IC; iic += nth) {

                            // micro kernel
                            ਦਸ਼ਮਲਵ * dst_ਅੰਕੜੇ = wdata + (ਵਿੱਚ*OD*OH_OW + iod*OH_OW + ioh*OW + iow)*IC_KD_KH_KW; // [IC, KD, KH, KW]
                            ਅਚਲ ਦਸ਼ਮਲਵ * ਅਚਲ src_ਅੰਕੜੇ = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *)src1->ਅੰਕੜੇ + (ਵਿੱਚ*IC + iic)*nb13); // [ID, IH, IW]

                            ਲਈ (int64_t ikd = 0; ikd < KD; ikd++) {
                                ਲਈ (int64_t ikh = 0; ikh < KH; ikh++) {
                                    ਲਈ (int64_t ikw = 0; ikw < KW; ikw++) {
                                        ਅਚਲ int64_t iiw = iow*s0 + ikw*d0 - p0;
                                        ਅਚਲ int64_t iih = ioh*s1 + ikh*d1 - p1;
                                        ਅਚਲ int64_t iid = iod*s2 + ikd*d2 - p2;

                                        ਜੇ (iid < 0 || iid >= ID || iih < 0 || iih >= IH || iiw < 0 || iiw >= IW || iid < 0 || iid >= ID) {
                                            dst_ਅੰਕੜੇ[iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw] = 0;
                                        } ਨਹੀਂਤਾਂ {
                                            ਅਚਲ ਦਸ਼ਮਲਵ * ਅਚਲ s = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *)src_ਅੰਕੜੇ + iid*nb12 + iih*nb11 + iiw*nb10); // [ID, IH, IW]
                                            dst_ਅੰਕੜੇ[iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw] = *s;
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


ਖਾਲੀ ggml_compute_forward_im2col_3d(
        ਅਚਲ ggml_compute_params * params,
              ggml_tensor * dst) {
    ਚੋਣ (dst->type) {
        ਹਾਲਤ GGML_TYPE_F16:
            {
                ggml_compute_forward_im2col_3d_f16(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_im2col_3d_f32(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_call_mul_mat(ggml_type type, ਅਚਲ ggml_compute_params * params, int64_t m, int64_t n, int64_t k,
                              ਖਾਲੀ * a, ਖਾਲੀ * b, ਦਸ਼ਮਲਵ * c) {
    ਅਚਲ ggml_type_traits * traits = ggml_get_type_traits(type);
    ਬਣਤਰ ggml_tensor src1 = {};
    src1.type  = type;
    src1.ne[0] = k;
    src1.ne[1] = m;
    src1.ne[2] = 1;
    src1.ne[3] = 1;
    src1.nb[0] = traits->type_ਮਾਪ;
    src1.nb[1] = k * traits->type_ਮਾਪ;
    src1.nb[2] = src1.nb[1];
    src1.nb[3] = src1.nb[2];
    src1.ਅੰਕੜੇ  = a;

    ਬਣਤਰ ggml_tensor src0 = {};
    src0.type  = type;
    src0.ne[0] = k;
    src0.ne[1] = n;
    src0.ne[2] = 1;
    src0.ne[3] = 1;
    src0.nb[0] = traits->type_ਮਾਪ;
    src0.nb[1] = k * traits->type_ਮਾਪ;
    src0.nb[2] = src0.nb[1];
    src0.nb[3] = src0.nb[2];
    src0.ਅੰਕੜੇ  = b;

    ਬਣਤਰ ggml_tensor dst = {};
    dst.ne[0] = n;
    dst.ne[1] = m;
    dst.ne[2] = 1;
    dst.ne[3] = 1;
    dst.nb[0] = ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    dst.nb[1] = n * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    dst.nb[2] = dst.nb[1];
    dst.nb[3] = dst.nb[2];
    dst.ਅੰਕੜੇ  = c;
    dst.src[0] = &src0;
    dst.src[1] = &src1;

    ggml_compute_forward_mul_mat(params, &dst);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ int64_t ggml_wrap_around(int64_t coord, int64_t size) {
    ਵਾਪਸ (coord  + size) % size; // adding size avoids negative number weirdness
}

// ggml_compute_forward_col2im_1d
//
// Scatter-add columns [K*OC, T_in] -> signal [T_out, OC]
// where T_out = (T_in - 1)*s + K - 2*p.  Gather approach: each output reads ceil(K/s) inputs.
// Parallelized over the time axis so the split stays balanced whatever OC is.
// Supports F32, F16, BF16 input/output (same type), F32 accumulator.

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ elem_t>
ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_col2im_1d_impl(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src = dst->src[0];  // [K*OC, T_in]

    GGML_ASSERT(ggml_is_contiguous(src));
    GGML_ASSERT(ggml_is_contiguous(dst));

    ਅਚਲ int32_t s0 = ((ਅਚਲ int32_t *)(dst->op_params))[0];
    ਅਚਲ int32_t OC = ((ਅਚਲ int32_t *)(dst->op_params))[1];
    ਅਚਲ int32_t p0 = ((ਅਚਲ int32_t *)(dst->op_params))[2];

    ਅਚਲ int64_t K_OC = src->ne[0];
    ਅਚਲ int64_t T_ਵਿੱਚ = src->ne[1];
    ਅਚਲ int64_t K    = K_OC / OC;
    ਅਚਲ int64_t T_ਬਾਹਰਲਾ = dst->ne[0];

    ਅਚਲ elem_t * ਖਾਨਾ_ਅੰਕੜੇ = (ਅਚਲ elem_t *) src->ਅੰਕੜੇ;
    elem_t       * dst_ਅੰਕੜੇ = (elem_t *) dst->ਅੰਕੜੇ;

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    // Parallelize over the time axis: the split stays balanced whatever OC is,
    // down to OC = 1 for mono audio, and threads read disjoint column bands
    ਅਚਲ int64_t dr = (T_ਬਾਹਰਲਾ + nth - 1) / nth;
    ਅਚਲ int64_t it0 = dr * ith;
    ਅਚਲ int64_t it1 = it0 + dr < T_ਬਾਹਰਲਾ ? it0 + dr : T_ਬਾਹਰਲਾ;

    ਲਈ (int64_t oc = 0; oc < OC; oc++) {
        ਲਈ (int64_t t_ਬਾਹਰਲਾ = it0; t_ਬਾਹਰਲਾ < it1; t_ਬਾਹਰਲਾ++) {
            ਅਚਲ int64_t t_abs = t_ਬਾਹਰਲਾ + p0;  // absolute position in uncropped signal
            // Gather: find all (t_in, k) where t_in * s + k == t_abs, 0 <= k < K
            int64_t t_ਵਿੱਚ_min = (t_abs - K + 1 + s0 - 1) / s0;  // ceil((t_abs-K+1)/s)
            ਜੇ (t_ਵਿੱਚ_min < 0) t_ਵਿੱਚ_min = 0;
            int64_t t_ਵਿੱਚ_max = t_abs / s0;
            ਜੇ (t_ਵਿੱਚ_max >= T_ਵਿੱਚ) t_ਵਿੱਚ_max = T_ਵਿੱਚ - 1;

            ਦਸ਼ਮਲਵ ਜੋੜ = 0.0f;
            ਲਈ (int64_t t_ਵਿੱਚ = t_ਵਿੱਚ_min; t_ਵਿੱਚ <= t_ਵਿੱਚ_max; t_ਵਿੱਚ++) {
                int64_t k = t_abs - t_ਵਿੱਚ * s0;
                ਜੇ (k >= 0 && k < K) {
                    // col layout: [K*OC, T_in], element (oc*K+k, t_in)
                    ਜੋੜ += type_conversion_ਸਾਰਣੀ<elem_t>::ਮੰਜ਼ਿਲ_f32(ਖਾਨਾ_ਅੰਕੜੇ[(oc * K + k) + t_ਵਿੱਚ * K_OC]);
                }
            }
            // dst layout: [T_out, OC], element (t_out, oc)
            dst_ਅੰਕੜੇ[t_ਬਾਹਰਲਾ + oc * T_ਬਾਹਰਲਾ] = type_conversion_ਸਾਰਣੀ<elem_t>::from_f32(ਜੋੜ);
        }
    }
}

ਖਾਲੀ ggml_compute_forward_col2im_1d(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {
    ਚੋਣ (dst->src[0]->type) {
        ਹਾਲਤ GGML_TYPE_F32:  ggml_compute_forward_col2im_1d_impl<ਦਸ਼ਮਲਵ>      (params, dst); ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F16:  ggml_compute_forward_col2im_1d_impl<ggml_fp16_t>(params, dst); ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_BF16: ggml_compute_forward_col2im_1d_impl<ggml_bf16_t>(params, dst); ਰੋਕੋ;
        ਮੂਲ: GGML_ABORT("col2im_1d: unsupported type %d", dst->src[0]->type);
    }
}

// ggml_compute_forward_conv_2d


ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_conv_2d_impl(ਅਚਲ ggml_compute_params * params,
                                              ਅਚਲ ggml_tensor *         kernel,  // [KW, KH, IC, OC]
                                              ਅਚਲ ggml_tensor *         src,     // [W, H, C, N]
                                              ggml_tensor *               dst,     // [OW, OH, OC, N]
                                              ggml_type                   kernel_type) {

    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(kernel_type == GGML_TYPE_F16 || kernel_type == GGML_TYPE_F32);
    GGML_ASSERT(kernel->type == kernel_type);

    ਅਚਲ ggml_type_traits * traits = ggml_get_type_traits(kernel_type);

    ਅਚਲ int32_t stride_x   = dst->op_params[0];
    ਅਚਲ int32_t stride_y   = dst->op_params[1];
    ਅਚਲ int32_t pad_x      = dst->op_params[2];
    ਅਚਲ int32_t pad_y      = dst->op_params[3];
    ਅਚਲ int32_t dilation_x = dst->op_params[4];
    ਅਚਲ int32_t dilation_y = dst->op_params[5];

    ਅਚਲ int64_t c_ਵਿੱਚ  = src->ne[2];
    ਅਚਲ int64_t c_ਬਾਹਰਲਾ = kernel->ne[3];
    GGML_ASSERT(c_ਵਿੱਚ == kernel->ne[2]);

    ਅਚਲ int64_t src_w = src->ne[0];
    ਅਚਲ int64_t src_h = src->ne[1];
    ਅਚਲ int64_t knl_w = kernel->ne[0];
    ਅਚਲ int64_t knl_h = kernel->ne[1];
    ਅਚਲ int64_t dst_w = dst->ne[0];
    ਅਚਲ int64_t dst_h = dst->ne[1];

    ਅਚਲ ਦਸ਼ਮਲਵ * src_ਅੰਕੜੇ = (ਦਸ਼ਮਲਵ *) src->ਅੰਕੜੇ;
    ਖਾਲੀ  * knl_ਅੰਕੜੇ       = kernel->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * dst_ਅੰਕੜੇ       = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;

    ਅਚਲ int64_t knl_n           = knl_w * knl_h * c_ਵਿੱਚ;
    ਅਚਲ int64_t patch_ਕੁੱਲ     = dst->ne[3] * dst_w * dst_h;

    ਅਚਲ int64_t space_ਪ੍ਰਤੀ_patch   = knl_n * traits->type_ਮਾਪ + c_ਬਾਹਰਲਾ * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ int64_t batch_ਮਾਪ        = params->wsize / space_ਪ੍ਰਤੀ_patch;
    ਅਚਲ int64_t patches_ਪ੍ਰਤੀ_batch = batch_ਮਾਪ > 8 ? (batch_ਮਾਪ / 8) * 8 : batch_ਮਾਪ;
    ਅਚਲ int64_t batch_n           = (patch_ਕੁੱਲ + patches_ਪ੍ਰਤੀ_batch - 1) / patches_ਪ੍ਰਤੀ_batch;

    GGML_ASSERT(patches_ਪ੍ਰਤੀ_batch > 0 && batch_ਮਾਪ >= 1);

    ਖਾਲੀ * tmp = params->wdata;

    ਲਈ (int64_t batch_i = 0; batch_i < batch_n; ++batch_i) {

        ਅਚਲ int64_t patch_ਸ਼ੁਰੂ_batch = batch_i * patches_ਪ੍ਰਤੀ_batch;
        ਅਚਲ int64_t patch_ਸਿਰਾ_batch   = std::min(patch_ਸ਼ੁਰੂ_batch + patches_ਪ੍ਰਤੀ_batch,
                                              patch_ਕੁੱਲ);
        ਅਚਲ int64_t patch_n           = patch_ਸਿਰਾ_batch - patch_ਸ਼ੁਰੂ_batch;

        ਅਚਲ int64_t patch_ਪ੍ਰਤੀ_thread  = (patch_n + params->nth - 1) / params->nth;
        ਅਚਲ int64_t patch_ਸ਼ੁਰੂ       = patch_ਸ਼ੁਰੂ_batch + params->ith * patch_ਪ੍ਰਤੀ_thread;
        ਅਚਲ int64_t patch_ਸਿਰਾ         = std::min(patch_ਸ਼ੁਰੂ + patch_ਪ੍ਰਤੀ_thread, patch_ਸਿਰਾ_batch);

        //im2col for a patch
        ਲਈ (int64_t p = patch_ਸ਼ੁਰੂ; p < patch_ਸਿਰਾ; ++p) {
            ਅਚਲ int64_t  batch_n     =  p / (dst_w * dst_h);
            ਅਚਲ int64_t  src_x       = (p / dst_w) % dst_h;
            ਅਚਲ int64_t  src_y       =  p % dst_w;

            ਅਚਲ ਦਸ਼ਮਲਵ * src_ਅਧਾਰ = (ਅਚਲ ਦਸ਼ਮਲਵ *)((ਅਚਲ ਅੱਖਰ *)src_ਅੰਕੜੇ + batch_n * src->nb[3]);
            ਅੱਖਰ *        dst_ਪੰਕਤੀ  = (ਅੱਖਰ *) tmp + (p % patches_ਪ੍ਰਤੀ_batch) * knl_n * traits->type_ਮਾਪ;

            ਲਈ (int64_t ic = 0; ic < c_ਵਿੱਚ; ++ic) {
                ਲਈ (int64_t ky = 0; ky < knl_h; ++ky) {
                    ਲਈ (int64_t kx = 0; kx < knl_w; ++kx) {
                        ਅਚਲ int64_t sy = src_x * stride_y + ky * dilation_y - pad_y;
                        ਅਚਲ int64_t sx = src_y * stride_x + kx * dilation_x - pad_x;

                        int64_t dst_idx = ic * (knl_h * knl_w) + ky * knl_w + kx;

                        ਦਸ਼ਮਲਵ src_val;
                        ਜੇ (sy < 0 || sy >= src_h || sx < 0 || sx >= src_w) {
                            src_val = 0.0f;
                        } ਨਹੀਂਤਾਂ {
                            ਅਚਲ ਦਸ਼ਮਲਵ * src_ptr = (ਅਚਲ ਦਸ਼ਮਲਵ *)((ਅਚਲ ਅੱਖਰ *)src_ਅਧਾਰ + sx * src->nb[0] + sy * src->nb[1] + ic * src->nb[2]);
                            src_val               = *src_ptr;
                        }

                        ਅੱਖਰ * element_ptr = dst_ਪੰਕਤੀ + dst_idx * traits->type_ਮਾਪ;
                        ਜੇ (kernel_type == GGML_TYPE_F32) {
                            *(ਦਸ਼ਮਲਵ *) element_ptr = src_val;
                        } ਨਹੀਂਤਾਂ ਜੇ (kernel_type == GGML_TYPE_F16) {
                            *(ggml_fp16_t *) element_ptr = GGML_CPU_FP32_ਮੰਜ਼ਿਲ_FP16(src_val);
                        }
                    }
                }
            }
        }   // patches handled by this thread

        ggml_barrier(params->threadpool);

        ਦਸ਼ਮਲਵ * gemm_output = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) tmp + patches_ਪ੍ਰਤੀ_batch * knl_n * traits->type_ਮਾਪ);

        GGML_ASSERT(gemm_output + patch_n * c_ਬਾਹਰਲਾ <= (ਦਸ਼ਮਲਵ*)tmp + params->wsize);

        // GEMM: patches[patch_n, knl_n] × kernel[knl_n, c_out] = output[patch_n, c_out]
        ggml_call_mul_mat(kernel_type, params, patch_n, c_ਬਾਹਰਲਾ, knl_n, tmp, knl_ਅੰਕੜੇ, gemm_output);

        ggml_barrier(params->threadpool);


        //permute back [OC, N, OH, OW] to [N, OC, OH, OW]
        ਅਚਲ int64_t ਕ੍ਰਮ_ਬਦਲੋ_ਪ੍ਰਤੀ_thread = (patch_n + params->nth - 1) / params->nth;
        ਅਚਲ int64_t ਕ੍ਰਮ_ਬਦਲੋ_ਸ਼ੁਰੂ = params->ith * ਕ੍ਰਮ_ਬਦਲੋ_ਪ੍ਰਤੀ_thread;
        ਅਚਲ int64_t ਕ੍ਰਮ_ਬਦਲੋ_ਸਿਰਾ = std::min(ਕ੍ਰਮ_ਬਦਲੋ_ਸ਼ੁਰੂ + ਕ੍ਰਮ_ਬਦਲੋ_ਪ੍ਰਤੀ_thread, patch_n);

        ਲਈ (int64_t i = ਕ੍ਰਮ_ਬਦਲੋ_ਸ਼ੁਰੂ; i < ਕ੍ਰਮ_ਬਦਲੋ_ਸਿਰਾ; ++i) {
            ਅਚਲ int64_t p       = patch_ਸ਼ੁਰੂ_batch + i;
            ਅਚਲ int64_t batch_n = p / (dst_w * dst_h);
            ਅਚਲ int64_t dst_y   = (p / dst_w) % dst_h;
            ਅਚਲ int64_t dst_x   = p % dst_w;

            ਲਈ (int64_t oc = 0; oc < c_ਬਾਹਰਲਾ; ++oc) {
                ਅਚਲ ਦਸ਼ਮਲਵ value = gemm_output[i * c_ਬਾਹਰਲਾ + oc];
                ਦਸ਼ਮਲਵ * dst_ptr = (ਦਸ਼ਮਲਵ *)((ਅੱਖਰ *)dst_ਅੰਕੜੇ + dst_x * dst->nb[0] + dst_y * dst->nb[1] + oc * dst->nb[2] + batch_n * dst->nb[3]);
                *dst_ptr = value;
            }
        }
    }
}

ਖਾਲੀ ggml_compute_forward_conv_2d(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    ggml_compute_forward_conv_2d_impl(params, src0, src1, dst, src0->type);
}

// ggml_compute_forward_conv_3d

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_conv_3d_impl(ਅਚਲ ggml_compute_params * params,
                                              ਅਚਲ ggml_tensor *         kernel,
                                              ਅਚਲ ggml_tensor *         src,
                                              ggml_tensor *               dst,
                                              ggml_type                   kernel_type) {

    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(kernel_type == GGML_TYPE_F16 || kernel_type == GGML_TYPE_F32);
    GGML_ASSERT(kernel->type == kernel_type);

    ਅਚਲ ggml_type_traits * traits = ggml_get_type_traits(kernel_type);

    ਅਚਲ int32_t s0 = dst->op_params[0];
    ਅਚਲ int32_t s1 = dst->op_params[1];
    ਅਚਲ int32_t s2 = dst->op_params[2];
    ਅਚਲ int32_t p0 = dst->op_params[3];
    ਅਚਲ int32_t p1 = dst->op_params[4];
    ਅਚਲ int32_t p2 = dst->op_params[5];
    ਅਚਲ int32_t d0 = dst->op_params[6];
    ਅਚਲ int32_t d1 = dst->op_params[7];
    ਅਚਲ int32_t d2 = dst->op_params[8];
    ਅਚਲ int32_t c  = dst->op_params[9];
    ਅਚਲ int32_t n  = dst->op_params[10];
    ਅਚਲ int32_t oc = dst->op_params[11];

    ਅਚਲ int64_t src_w = src->ne[0];
    ਅਚਲ int64_t src_h = src->ne[1];
    ਅਚਲ int64_t src_d = src->ne[2];
    ਅਚਲ int64_t knl_w = kernel->ne[0];
    ਅਚਲ int64_t knl_h = kernel->ne[1];
    ਅਚਲ int64_t knl_d = kernel->ne[2];
    ਅਚਲ int64_t dst_w = dst->ne[0];
    ਅਚਲ int64_t dst_h = dst->ne[1];
    ਅਚਲ int64_t dst_d = dst->ne[2];

    ਅਚਲ ਦਸ਼ਮਲਵ * src_ਅੰਕੜੇ = (ਦਸ਼ਮਲਵ *) src->ਅੰਕੜੇ;
    ਖਾਲੀ  * knl_ਅੰਕੜੇ       = kernel->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * dst_ਅੰਕੜੇ       = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;

    ਅਚਲ int64_t knl_n_ਪ੍ਰਤੀ_channel = knl_w * knl_h * knl_d;
    ਅਚਲ int64_t knl_n_ਕੁੱਲ       = knl_n_ਪ੍ਰਤੀ_channel * c;
    ਅਚਲ int64_t patch_ਕੁੱਲ       = n * dst_w * dst_h * dst_d;

    ਅਚਲ int64_t space_ਪ੍ਰਤੀ_patch   = knl_n_ਕੁੱਲ * traits->type_ਮਾਪ + oc * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ int64_t batch_ਮਾਪ        = params->wsize / space_ਪ੍ਰਤੀ_patch;
    ਅਚਲ int64_t patches_ਪ੍ਰਤੀ_batch = batch_ਮਾਪ > 8 ? (batch_ਮਾਪ / 8) * 8 : batch_ਮਾਪ;
    ਅਚਲ int64_t batch_n           = (patch_ਕੁੱਲ + patches_ਪ੍ਰਤੀ_batch - 1) / patches_ਪ੍ਰਤੀ_batch;

    GGML_ASSERT(patches_ਪ੍ਰਤੀ_batch > 0 && batch_ਮਾਪ >= 1);

    ਖਾਲੀ * tmp = params->wdata;

    ਲਈ (int64_t batch_i = 0; batch_i < batch_n; ++batch_i) {
        ਅਚਲ int64_t patch_ਸ਼ੁਰੂ_batch = batch_i * patches_ਪ੍ਰਤੀ_batch;
        ਅਚਲ int64_t patch_ਸਿਰਾ_batch   = std::min(patch_ਸ਼ੁਰੂ_batch + patches_ਪ੍ਰਤੀ_batch, patch_ਕੁੱਲ);
        ਅਚਲ int64_t patch_n_ਵਿੱਚ_batch  = patch_ਸਿਰਾ_batch - patch_ਸ਼ੁਰੂ_batch;

        ਅਚਲ int64_t patch_ਪ੍ਰਤੀ_thread  = (patch_n_ਵਿੱਚ_batch + params->nth - 1) / params->nth;
        ਅਚਲ int64_t patch_ਸ਼ੁਰੂ       = patch_ਸ਼ੁਰੂ_batch + params->ith * patch_ਪ੍ਰਤੀ_thread;
        ਅਚਲ int64_t patch_ਸਿਰਾ         = std::min(patch_ਸ਼ੁਰੂ + patch_ਪ੍ਰਤੀ_thread, patch_ਸਿਰਾ_batch);

        ਲਈ (int64_t p = patch_ਸ਼ੁਰੂ; p < patch_ਸਿਰਾ; ++p) {
            ਅਚਲ int64_t p_ਵਿੱਚ_batch = p % (dst_w * dst_h * dst_d);
            ਅਚਲ int64_t p_ਵਿੱਚ_ਡੂੰਘਾਈ = p_ਵਿੱਚ_batch % (dst_w * dst_h);
            ਅਚਲ int64_t batch_idx  = p / (dst_w * dst_h * dst_d);
            ਅਚਲ int64_t dst_z      = p_ਵਿੱਚ_batch / (dst_w * dst_h);
            ਅਚਲ int64_t dst_y      = p_ਵਿੱਚ_ਡੂੰਘਾਈ / dst_w;
            ਅਚਲ int64_t dst_x      = p_ਵਿੱਚ_ਡੂੰਘਾਈ % dst_w;

            ਅੱਖਰ * dst_ਪੰਕਤੀ = (ਅੱਖਰ *) tmp + (p % patches_ਪ੍ਰਤੀ_batch) * knl_n_ਕੁੱਲ * traits->type_ਮਾਪ;

            ਲਈ (int64_t ic = 0; ic < c; ++ic) {
                ਲਈ (int64_t kz = 0; kz < knl_d; ++kz) {
                    ਲਈ (int64_t ky = 0; ky < knl_h; ++ky) {
                        ਲਈ (int64_t kx = 0; kx < knl_w; ++kx) {
                            ਅਚਲ int64_t sz = dst_z * s2 + kz * d2 - p2;
                            ਅਚਲ int64_t sy = dst_y * s1 + ky * d1 - p1;
                            ਅਚਲ int64_t sx = dst_x * s0 + kx * d0 - p0;

                            int64_t dst_idx = ic * knl_n_ਪ੍ਰਤੀ_channel + kz * (knl_h * knl_w) + ky * knl_w + kx;

                            ਦਸ਼ਮਲਵ src_val;
                            ਜੇ (sz < 0 || sz >= src_d || sy < 0 || sy >= src_h || sx < 0 || sx >= src_w) {
                                src_val = 0.0f;
                            } ਨਹੀਂਤਾਂ {
                                ਅਚਲ int64_t cn_idx = batch_idx * c + ic;
                                ਅਚਲ ਦਸ਼ਮਲਵ * src_ptr = (ਅਚਲ ਦਸ਼ਮਲਵ *)((ਅਚਲ ਅੱਖਰ *)src_ਅੰਕੜੇ + sx*src->nb[0] + sy*src->nb[1] + sz*src->nb[2] + cn_idx*src->nb[3]);
                                src_val = *src_ptr;
                            }

                            ਅੱਖਰ * element_ptr = dst_ਪੰਕਤੀ + dst_idx * traits->type_ਮਾਪ;
                            ਜੇ (kernel_type == GGML_TYPE_F32) {
                                *(ਦਸ਼ਮਲਵ *)element_ptr = src_val;
                            } ਨਹੀਂਤਾਂ ਜੇ (kernel_type == GGML_TYPE_F16) {
                                *(ggml_fp16_t *)element_ptr = GGML_CPU_FP32_ਮੰਜ਼ਿਲ_FP16(src_val);
                            }
                        }
                    }
                }
            }
        }

        ggml_barrier(params->threadpool);

        ਦਸ਼ਮਲਵ * gemm_output = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) tmp + patches_ਪ੍ਰਤੀ_batch * knl_n_ਕੁੱਲ * traits->type_ਮਾਪ);
        ggml_call_mul_mat(kernel_type, params, patch_n_ਵਿੱਚ_batch, oc, knl_n_ਕੁੱਲ, tmp, knl_ਅੰਕੜੇ, gemm_output);

        ggml_barrier(params->threadpool);

        ਅਚਲ int64_t ਕ੍ਰਮ_ਬਦਲੋ_ਪ੍ਰਤੀ_thread = (patch_n_ਵਿੱਚ_batch + params->nth - 1) / params->nth;
        ਅਚਲ int64_t ਕ੍ਰਮ_ਬਦਲੋ_ਸ਼ੁਰੂ = params->ith * ਕ੍ਰਮ_ਬਦਲੋ_ਪ੍ਰਤੀ_thread;
        ਅਚਲ int64_t ਕ੍ਰਮ_ਬਦਲੋ_ਸਿਰਾ = std::min(ਕ੍ਰਮ_ਬਦਲੋ_ਸ਼ੁਰੂ + ਕ੍ਰਮ_ਬਦਲੋ_ਪ੍ਰਤੀ_thread, patch_n_ਵਿੱਚ_batch);

        ਲਈ (int64_t i = ਕ੍ਰਮ_ਬਦਲੋ_ਸ਼ੁਰੂ; i < ਕ੍ਰਮ_ਬਦਲੋ_ਸਿਰਾ; ++i) {
            ਅਚਲ int64_t p = patch_ਸ਼ੁਰੂ_batch + i;
            ਅਚਲ int64_t p_ਵਿੱਚ_batch = p % (dst_w * dst_h * dst_d);
            ਅਚਲ int64_t p_ਵਿੱਚ_ਡੂੰਘਾਈ = p_ਵਿੱਚ_batch % (dst_w * dst_h);
            ਅਚਲ int64_t batch_idx  = p / (dst_w * dst_h * dst_d);
            ਅਚਲ int64_t dst_z      = p_ਵਿੱਚ_batch / (dst_w * dst_h);
            ਅਚਲ int64_t dst_y      = p_ਵਿੱਚ_ਡੂੰਘਾਈ / dst_w;
            ਅਚਲ int64_t dst_x      = p_ਵਿੱਚ_ਡੂੰਘਾਈ % dst_w;

            ਲਈ (int64_t ioc = 0; ioc < oc; ++ioc) {
                ਅਚਲ ਦਸ਼ਮਲਵ value = gemm_output[i * oc + ioc];
                ਅਚਲ int64_t ocn_idx = batch_idx * oc + ioc;
                ਦਸ਼ਮਲਵ * dst_ptr = (ਦਸ਼ਮਲਵ *)((ਅੱਖਰ *)dst_ਅੰਕੜੇ + dst_x*dst->nb[0] + dst_y*dst->nb[1] + dst_z*dst->nb[2] + ocn_idx*dst->nb[3]);
                *dst_ptr = value;
            }
        }
    }
}

ਖਾਲੀ ggml_compute_forward_conv_3d(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];
    ggml_compute_forward_conv_3d_impl(params, src0, src1, dst, src0->type);
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ kernel_t>
ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_conv_transpose_2d_impl(
    ਅਚਲ ggml_compute_params * params,
          ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ nk = ne00*ne01*ne02*ne03;

    GGML_ASSERT(nb00 == ggml_type_ਮਾਪ(src0->type));
    GGML_ASSERT(nb10 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਜੇ (ith == 0) {
        memset(params->wdata, 0, params->wsize);

        // permute kernel data (src0) from (Kw x Kh x Cout x Cin) to (Cin x Kw x Kh x Cout)
        {
            kernel_t * ਅਚਲ wdata = (kernel_t *) params->wdata + 0;

            ਲਈ (int64_t i03 = 0; i03 < ne03; i03++) {
                ਲਈ (int64_t i02 = 0; i02 < ne02; i02++) {
                    ਅਚਲ kernel_t * ਅਚਲ src = (kernel_t *)((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i03*nb03 + i02*nb02);
                    kernel_t * dst_ਅੰਕੜੇ = wdata + i02*ne01*ne00*ne03;
                    ਲਈ (int64_t i01 = 0; i01 < ne01; i01++) {
                        ਲਈ (int64_t i00 = 0; i00 < ne00; i00++) {
                            dst_ਅੰਕੜੇ[i01*ne00*ne03 + i00*ne03 + i03] = src[i01 * ne00 + i00];
                        }
                    }
                }
            }
        }

        // permute source data (src1) from (Sw x Sh x Cin) to (Cin x Sw x Sh), for all batches
        {
            kernel_t * ਅਚਲ wdata = (kernel_t *) params->wdata + nk;
            ਲਈ (ਪੂਰਨ i13 = 0; i13 < ne13; i13++) {
                kernel_t * ਅਚਲ wdata_b = wdata + i13*ne10*ne11*ne12;
                ਲਈ (ਪੂਰਨ i12 = 0; i12 < ne12; i12++) {
                    ਲਈ (ਪੂਰਨ i11 = 0; i11 < ne11; i11++) {
                        ਅਚਲ ਦਸ਼ਮਲਵ * ਅਚਲ src = (ਦਸ਼ਮਲਵ *)((ਅੱਖਰ *) src1->ਅੰਕੜੇ + i13*nb13 + i12*nb12 + i11*nb11);
                        kernel_t * dst_ਅੰਕੜੇ = wdata_b + i11*ne10*ne12;
                        ਲਈ (ਪੂਰਨ i10 = 0; i10 < ne10; i10++) {
                            ਜੇ constexpr (std::is_same_v<kernel_t, ggml_fp16_t>) {
                                dst_ਅੰਕੜੇ[i10*ne12 + i12] = GGML_CPU_FP32_ਮੰਜ਼ਿਲ_FP16(src[i10]);
                            } ਨਹੀਂਤਾਂ {
                                dst_ਅੰਕੜੇ[i10*ne12 + i12] = src[i10];
                            }
                        }
                    }
                }
            }
        }

        memset(dst->ਅੰਕੜੇ, 0, ggml_nbytes(dst));
    }
    ggml_barrier(params->threadpool);

    ਅਚਲ int32_t stride = ggml_get_op_params_i32(dst, 0);

    // total patches in dst
    ਅਚਲ ਪੂਰਨ np = ne2;

    // patches per thread
    ਅਚਲ ਪੂਰਨ dp = (np + nth - 1)/nth;

    // patch range for this thread
    ਅਚਲ ਪੂਰਨ ip0 = dp*ith;
    ਅਚਲ ਪੂਰਨ ip1 = MIN(ip0 + dp, np);

    kernel_t * ਅਚਲ wdata = (kernel_t *) params->wdata + 0;
    kernel_t * ਅਚਲ wdata_src = wdata + nk;

    ਲਈ (ਪੂਰਨ i2 = ip0; i2 < ip1; i2++) { // Cout
        kernel_t * wdata_kernel = wdata + i2*ne01*ne00*ne03;
        ਲਈ (ਪੂਰਨ i3 = 0; i3 < ne3; i3++) { // batch
            ਦਸ਼ਮਲਵ * dst_ਅੰਕੜੇ = (ਦਸ਼ਮਲਵ *)((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i3*nb3 + i2*nb2);
            kernel_t * wdata_src_b = wdata_src + i3*ne10*ne11*ne12;
            ਲਈ (ਪੂਰਨ i11 = 0; i11 < ne11; i11++) {
                ਲਈ (ਪੂਰਨ i10 = 0; i10 < ne10; i10++) {
                    ਅਚਲ ਪੂਰਨ i1n = i11*ne10*ne12 + i10*ne12;
                    ਲਈ (ਪੂਰਨ i01 = 0; i01 < ne01; i01++) {
                        ਲਈ (ਪੂਰਨ i00 = 0; i00 < ne00; i00++) {
                            ਦਸ਼ਮਲਵ v = 0;
                            ਜੇ constexpr (std::is_same_v<kernel_t, ggml_fp16_t>) {
                                ggml_vec_dot_f16(ne03, &v, 0,
                                        wdata_src_b + i1n, 0,
                                        wdata_kernel + i01*ne00*ne03 + i00*ne03, 0, 1);
                            } ਨਹੀਂਤਾਂ {
                                ggml_vec_dot_f32(ne03, &v, 0,
                                        wdata_src_b + i1n, 0,
                                        wdata_kernel + i01*ne00*ne03 + i00*ne03, 0, 1);
                            }
                            dst_ਅੰਕੜੇ[(i11*stride + i01)*ne0 + i10*stride + i00] += v;
                        }
                    }
                }
            }
        }
    }
}

ਖਾਲੀ ggml_compute_forward_conv_transpose_2d(
        ਅਚਲ ggml_compute_params * params,
              ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F16:
            {
                ggml_compute_forward_conv_transpose_2d_impl<ggml_fp16_t>(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_conv_transpose_2d_impl<ਦਸ਼ਮਲਵ>(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_conv_2d_dw

ਬਣਤਰ ggml_conv_2d_dw_params {
    int64_t channels;
    int64_t batch;
    int64_t src_w;
    int64_t src_h;
    int64_t dst_w;
    int64_t dst_h;
    int64_t knl_w;
    int64_t knl_h;
    ਪੂਰਨ stride_x;
    ਪੂਰਨ stride_y;
    ਪੂਰਨ pad_x;
    ਪੂਰਨ pad_y;
    ਪੂਰਨ dilation_x;
    ਪੂਰਨ dilation_y;
};

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ ggml_conv_2d_dw_knl_f32(ਅਚਲ ਅੱਖਰ * ਅੰਕੜੇ, int64_t i, ggml_type type) {
    ਜੇ (type == GGML_TYPE_F16) {
        ਵਾਪਸ GGML_FP16_ਮੰਜ਼ਿਲ_FP32(((ਅਚਲ ggml_fp16_t *)ਅੰਕੜੇ)[i]);
    }
    ਵਾਪਸ ((ਅਚਲ ਦਸ਼ਮਲਵ *)ਅੰਕੜੇ)[i];
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_conv_2d_dw_cwhn(
        ਅਚਲ ggml_compute_params * params,
        ਅਚਲ ggml_tensor * src,
        ਅਚਲ ggml_tensor * kernel,
        ggml_tensor * dst,
        ਅਚਲ ggml_conv_2d_dw_params & p) {

    ਅਚਲ int64_t c = p.channels;
    ਅਚਲ ਅੱਖਰ * knl_ਅੰਕੜੇ = (ਅਚਲ ਅੱਖਰ *)kernel->ਅੰਕੜੇ;
    ਅਚਲ ggml_type knl_type = kernel->type;

    ਅਚਲ int64_t ਪੰਕਤੀਆਂ_ਕੁੱਲ = p.dst_h * p.batch;
    ਅਚਲ int64_t ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread = (ਪੰਕਤੀਆਂ_ਕੁੱਲ + params->nth - 1) / params->nth;
    ਅਚਲ int64_t ਪੰਕਤੀ_ਸ਼ੁਰੂ = params->ith * ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread;
    ਅਚਲ int64_t ਪੰਕਤੀ_ਸਿਰਾ = MIN(ਪੰਕਤੀ_ਸ਼ੁਰੂ + ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread, ਪੰਕਤੀਆਂ_ਕੁੱਲ);

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SIMD
    int64_t c_pkg_ਸਿਰਾ = 0;
    int64_t pkg_ਮਾਪ = GGML_F32_EPR;
    ਜੇ (knl_type == GGML_TYPE_F32) {
    #ਜੇ defined(__ARM_FEATURE_SVE)
        pkg_ਮਾਪ = svcntw();
    #ਨਹੀਂਤਾਂ
        pkg_ਮਾਪ = GGML_F32_EPR;
    #ਜੇ_ਅੰਤ
        c_pkg_ਸਿਰਾ = (c / pkg_ਮਾਪ) * pkg_ਮਾਪ;
    }
#ਨਹੀਂਤਾਂ
    ਅਚਲ int64_t c_pkg_ਸਿਰਾ = 0;
#ਜੇ_ਅੰਤ

    ਲਈ (int64_t ਪੰਕਤੀ = ਪੰਕਤੀ_ਸ਼ੁਰੂ; ਪੰਕਤੀ < ਪੰਕਤੀ_ਸਿਰਾ; ++ਪੰਕਤੀ) {
        ਅਚਲ int64_t dst_y = ਪੰਕਤੀ % p.dst_h;
        ਅਚਲ ਦਸ਼ਮਲਵ * src_ਅੰਕੜੇ = (ਅਚਲ ਦਸ਼ਮਲਵ *)src->ਅੰਕੜੇ + (ਪੰਕਤੀ / p.dst_h) * p.src_w * p.src_h * c;
        ਲਈ (int64_t dst_x = 0; dst_x < p.dst_w; ++dst_x) {
            ਦਸ਼ਮਲਵ * dst_ਅੰਕੜੇ = (ਦਸ਼ਮਲਵ *)dst->ਅੰਕੜੇ + (ਪੰਕਤੀ * p.dst_w + dst_x) * c;
            ਅਚਲ int64_t src_y_ਅਧਾਰ = dst_y * p.stride_y - p.pad_y;
            ਅਚਲ int64_t src_x_ਅਧਾਰ = dst_x * p.stride_x - p.pad_x;

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SIMD
            ਲਈ (int64_t c_i = 0; c_i < c_pkg_ਸਿਰਾ; c_i += pkg_ਮਾਪ) {
                GGML_F32_VEC ਜੋੜ = GGML_F32_VEC_ZERO;
                ਲਈ (int64_t knl_y = 0; knl_y < p.knl_h; ++knl_y) {
                    ਅਚਲ int64_t src_y = src_y_ਅਧਾਰ + knl_y * p.dilation_y;
                    ਜੇ (src_y < 0 || src_y >= p.src_h) {
                        ਜਾਰੀ;
                    }
                    ਲਈ (int64_t knl_x = 0; knl_x < p.knl_w; ++knl_x) {
                        ਅਚਲ int64_t src_x = src_x_ਅਧਾਰ + knl_x * p.dilation_x;
                        ਜੇ (src_x < 0 || src_x >= p.src_w) {
                            ਜਾਰੀ;
                        }
                        ਅਚਲ ਦਸ਼ਮਲਵ * kp = (ਅਚਲ ਦਸ਼ਮਲਵ *)knl_ਅੰਕੜੇ + (knl_y * p.knl_w + knl_x) * c + c_i;
                        GGML_F32_VEC k = GGML_F32_VEC_ਲੱਦੋ(kp);
                        GGML_F32_VEC s = GGML_F32_VEC_ਲੱਦੋ(src_ਅੰਕੜੇ + (src_y * p.src_w + src_x) * c + c_i);
                        ਜੋੜ = GGML_F32_VEC_FMA(ਜੋੜ, k, s);
                    }
                }
                GGML_F32_VEC_STORE(dst_ਅੰਕੜੇ + c_i, ਜੋੜ);
            }
#ਜੇ_ਅੰਤ
            ਲਈ (int64_t c_i = c_pkg_ਸਿਰਾ; c_i < c; ++c_i) {
                ਦਸ਼ਮਲਵ ਜੋੜ = 0.0f;
                ਲਈ (int64_t knl_y = 0; knl_y < p.knl_h; ++knl_y) {
                    ਅਚਲ int64_t src_y = src_y_ਅਧਾਰ + knl_y * p.dilation_y;
                    ਜੇ (src_y < 0 || src_y >= p.src_h) {
                        ਜਾਰੀ;
                    }
                    ਲਈ (int64_t knl_x = 0; knl_x < p.knl_w; ++knl_x) {
                        ਅਚਲ int64_t src_x = src_x_ਅਧਾਰ + knl_x * p.dilation_x;
                        ਜੇ (src_x < 0 || src_x >= p.src_w) {
                            ਜਾਰੀ;
                        }
                        ਜੋੜ += ggml_conv_2d_dw_knl_f32(knl_ਅੰਕੜੇ, (knl_y * p.knl_w + knl_x) * c + c_i, knl_type)
                             * src_ਅੰਕੜੇ[(src_y * p.src_w + src_x) * c + c_i];
                    }
                }
                dst_ਅੰਕੜੇ[c_i] = ਜੋੜ;
            }
        }
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_conv_2d_dw_whcn(
        ਅਚਲ ggml_compute_params * params,
        ਅਚਲ ggml_tensor * src,
        ਅਚਲ ggml_tensor * kernel,
        ggml_tensor * dst,
        ਅਚਲ ggml_conv_2d_dw_params & p) {

    ਅਚਲ int64_t n = p.channels * p.batch;
    ਅਚਲ int64_t ਪ੍ਰਤੀ_thread = (n + params->nth - 1) / params->nth;
    ਅਚਲ int64_t start = params->ith * ਪ੍ਰਤੀ_thread;
    ਅਚਲ int64_t end = MIN(start + ਪ੍ਰਤੀ_thread, n);
    ਅਚਲ ਅੱਖਰ * knl_ਅਧਾਰ = (ਅਚਲ ਅੱਖਰ *)kernel->ਅੰਕੜੇ;
    ਅਚਲ ggml_type knl_type = kernel->type;

    ਲਈ (int64_t i = start; i < end; ++i) {
        ਅਚਲ int64_t knl_ਵਿੱਥ = (i % p.channels) * p.knl_w * p.knl_h;
        ਅਚਲ ਦਸ਼ਮਲਵ * src_ਅੰਕੜੇ = (ਅਚਲ ਦਸ਼ਮਲਵ *)src->ਅੰਕੜੇ + i * p.src_w * p.src_h;
        ਦਸ਼ਮਲਵ * dst_ਅੰਕੜੇ = (ਦਸ਼ਮਲਵ *)dst->ਅੰਕੜੇ + i * p.dst_w * p.dst_h;

        ਲਈ (int64_t dst_y = 0; dst_y < p.dst_h; ++dst_y) {
            ਲਈ (int64_t dst_x = 0; dst_x < p.dst_w; ++dst_x) {

                ਦਸ਼ਮਲਵ ਜੋੜ = 0.0f;
                ਲਈ (int64_t knl_y = 0; knl_y < p.knl_h; ++knl_y) {
                    ਅਚਲ int64_t src_y = dst_y * p.stride_y + knl_y * p.dilation_y - p.pad_y;
                    ਜੇ (src_y < 0 || src_y >= p.src_h) {
                        ਜਾਰੀ;
                    }
                    ਲਈ (int64_t knl_x = 0; knl_x < p.knl_w; ++knl_x) {
                        ਅਚਲ int64_t src_x = dst_x * p.stride_x + knl_x * p.dilation_x - p.pad_x;
                        ਜੇ (src_x < 0 || src_x >= p.src_w) {
                            ਜਾਰੀ;
                        }
                        ਜੋੜ += ggml_conv_2d_dw_knl_f32(knl_ਅਧਾਰ, knl_ਵਿੱਥ + knl_y * p.knl_w + knl_x, knl_type)
                             * src_ਅੰਕੜੇ[src_y * p.src_w + src_x];
                    }
                }
                dst_ਅੰਕੜੇ[dst_y * p.dst_w + dst_x] = ਜੋੜ;
            }
        }
    }
}

ਖਾਲੀ ggml_compute_forward_conv_2d_dw(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * kernel = dst->src[0];
    ਅਚਲ ggml_tensor * src = dst->src[1];
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

    ਜੇ (ggml_is_contiguous(src)) {
        ggml_compute_forward_conv_2d_dw_whcn(params, src, kernel, dst, p);
    } ਨਹੀਂਤਾਂ ਜੇ (ggml_is_contiguous_channels(src)) {
        GGML_ASSERT(kernel->nb[0] >= kernel->nb[2] && kernel->nb[1] >= kernel->nb[0]);
        ggml_compute_forward_conv_2d_dw_cwhn(params, src, kernel, dst, p);
    } ਨਹੀਂਤਾਂ {
        GGML_ABORT("non-contiguous memory layout not supported");
    }
}

// ggml_compute_forward_pool_1d_ksp
ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_pool_1d_ksp(
        ਅਚਲ ggml_compute_params * params,
        ਅਚਲ ggml_op_pool op,
        ਅਚਲ ਪੂਰਨ k,
        ਅਚਲ ਪੂਰਨ s,
        ਅਚਲ ਪੂਰਨ p,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src = dst->src[0];

    ਪ੍ਰਤਿਜ੍ਞਾ(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16);

    ਜੇ (params->ith != 0) {
        ਵਾਪਸ;
    }

    ਅਚਲ int64_t IW = src->ne[0];
    ਅਚਲ int64_t OW = dst->ne[0];

    ਅਚਲ int64_t nr = ggml_nrows(src);

    ਲਈ (int64_t ir = 0; ir < nr; ++ir) {
        ਅਚਲ ਅੱਖਰ * srow_ਬਾਈਟਾਂ =            (ਅਚਲ ਅੱਖਰ *) src->ਅੰਕੜੇ + ir * src->nb[1];
        ਦਸ਼ਮਲਵ      * drow       = (ਦਸ਼ਮਲਵ *) ((      ਅੱਖਰ *) dst->ਅੰਕੜੇ + ir * dst->nb[1]);

        ਲਈ (int64_t ow = 0; ow < OW; ++ow) {
            ਦਸ਼ਮਲਵ res = 0;
            ਚੋਣ (op) {
                ਹਾਲਤ GGML_OP_POOL_AVG: res = 0.0f;     ਰੋਕੋ;
                ਹਾਲਤ GGML_OP_POOL_MAX: res = -FLT_MAX; ਰੋਕੋ;
                ਹਾਲਤ GGML_OP_POOL_ਗਿਣਤੀ: GGML_ABORT("fatal error");
            }

            ਪੂਰਨ count = 0;
            ਅਚਲ ਪੂਰਨ ਅਧਾਰ = (ਪੂਰਨ) ow * s - p;

            ਲਈ (ਪੂਰਨ ki = 0; ki < k; ++ki) {
                ਅਚਲ ਪੂਰਨ j = ਅਧਾਰ + ki;
                ਜੇ (j < 0 || j >= (ਪੂਰਨ) IW) {
                    ਜਾਰੀ;
                }

                ਦਸ਼ਮਲਵ v;
                ਜੇ (src->type == GGML_TYPE_F32) {
                    v = ((ਅਚਲ ਦਸ਼ਮਲਵ *) srow_ਬਾਈਟਾਂ)[j];
                } ਨਹੀਂਤਾਂ {
                    v = GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(((ਅਚਲ ggml_fp16_t *) srow_ਬਾਈਟਾਂ)[j]);
                }

                ਚੋਣ (op) {
                    ਹਾਲਤ GGML_OP_POOL_AVG: res += v;                ਰੋਕੋ;
                    ਹਾਲਤ GGML_OP_POOL_MAX: res =  std::max(v, res); ਰੋਕੋ;
                    ਹਾਲਤ GGML_OP_POOL_ਗਿਣਤੀ: GGML_ABORT("fatal error");
                }

                ++count;
            }

            ਚੋਣ (op) {
                ਹਾਲਤ GGML_OP_POOL_AVG: res = (count > 0) ? (res / count) : 0.0f; ਰੋਕੋ;
                ਹਾਲਤ GGML_OP_POOL_MAX:                                           ਰੋਕੋ;
                ਹਾਲਤ GGML_OP_POOL_ਗਿਣਤੀ: GGML_ABORT("fatal error");
            }

            drow[ow] = res;
        }
    }
}

// ggml_compute_forward_pool_1d

ਖਾਲੀ ggml_compute_forward_pool_1d(
        ਅਚਲ ggml_compute_params * params,
              ggml_tensor * dst) {

    ਅਚਲ int32_t * opts = (ਅਚਲ int32_t *)dst->op_params;
    ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    ਅਚਲ ਪੂਰਨ k0 = opts[1];
    ਅਚਲ ਪੂਰਨ s0 = opts[2];
    ਅਚਲ ਪੂਰਨ p0 = opts[3];

    ggml_compute_forward_pool_1d_ksp(params, op, k0, s0, p0, dst);
}

// ggml_compute_forward_pool_2d

ਖਾਲੀ ggml_compute_forward_pool_2d(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src = dst->src[0];

    ਪ੍ਰਤਿਜ੍ਞਾ(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16);

    ਜੇ (params->ith != 0) {
        ਵਾਪਸ;
    }

    ਅਚਲ int32_t * opts = (ਅਚਲ int32_t *)dst->op_params;

    ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    ਅਚਲ ਪੂਰਨ k0 = opts[1];
    ਅਚਲ ਪੂਰਨ k1 = opts[2];
    ਅਚਲ ਪੂਰਨ s0 = opts[3];
    ਅਚਲ ਪੂਰਨ s1 = opts[4];
    ਅਚਲ ਪੂਰਨ p0 = opts[5];
    ਅਚਲ ਪੂਰਨ p1 = opts[6];
    ਅਚਲ ਅੱਖਰ * cdata = (ਅਚਲ ਅੱਖਰ*)src->ਅੰਕੜੇ;
    ਅਚਲ ਅੱਖਰ * ਅਚਲ ਅੰਕੜੇ_ਸਿਰਾ = cdata + ggml_nbytes(src);

    ਅਚਲ int64_t px = dst->ne[0];
    ਅਚਲ int64_t py = dst->ne[1];
    ਅਚਲ int64_t pa = px * py;

    ਦਸ਼ਮਲਵ * dplane = (ਦਸ਼ਮਲਵ *)dst->ਅੰਕੜੇ;

    ਅਚਲ ਪੂਰਨ ka = k0 * k1;
    ਅਚਲ ਪੂਰਨ offset0 = -p0;
    ਅਚਲ ਪੂਰਨ offset1 = -p1;

    ਜਦੋਂ (cdata < ਅੰਕੜੇ_ਸਿਰਾ) {
        ਲਈ (ਪੂਰਨ oy = 0; oy < py; ++oy) {
            ਦਸ਼ਮਲਵ * ਅਚਲ drow = dplane + oy * px;
            ਦਸ਼ਮਲਵ * ਅਚਲ ਬਾਹਰਲਾ  = drow;

            ਲਈ (ਪੂਰਨ ox = 0; ox < px; ++ox) {
                ਦਸ਼ਮਲਵ res = 0;
                ਚੋਣ (op) {
                    ਹਾਲਤ GGML_OP_POOL_AVG: res = 0;        ਰੋਕੋ;
                    ਹਾਲਤ GGML_OP_POOL_MAX: res = -FLT_MAX; ਰੋਕੋ;
                    ਹਾਲਤ GGML_OP_POOL_ਗਿਣਤੀ: GGML_ABORT("fatal error");
                }

                ਅਚਲ ਪੂਰਨ ix = offset0 + ox * s0;
                ਅਚਲ ਪੂਰਨ iy = offset1 + oy * s1;

                ਲਈ (ਪੂਰਨ ky = 0; ky < k1; ++ky) {
                    ਜੇ (iy + ky < 0 || iy + ky >= src->ne[1]) {
                        ਜਾਰੀ;
                    }

                    ਅਚਲ ਖਾਲੀ * srow = (ਅਚਲ ਖਾਲੀ *)(cdata + src->nb[1] * (iy + ky));
                    ਲਈ (ਪੂਰਨ kx = 0; kx < k0; ++kx) {
                        ਪੂਰਨ j = ix + kx;
                        ਜੇ (j < 0 || j >= src->ne[0]) {
                            ਜਾਰੀ;
                        }

                        ਅਚਲ ਦਸ਼ਮਲਵ srow_j = (src->type == GGML_TYPE_F32) ? ((ਅਚਲ ਦਸ਼ਮਲਵ*)srow)[j] : GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(((ਅਚਲ ggml_fp16_t*)srow)[j]);
                        ਚੋਣ (op) {
                            ਹਾਲਤ GGML_OP_POOL_AVG: res += srow_j;                ਰੋਕੋ;
                            ਹਾਲਤ GGML_OP_POOL_MAX: res =  std::max(srow_j, res); ਰੋਕੋ;
                            ਹਾਲਤ GGML_OP_POOL_ਗਿਣਤੀ:               GGML_ABORT("fatal error");
                        }
                    }
                }
                ਚੋਣ (op) {
                    ਹਾਲਤ GGML_OP_POOL_AVG:           res /= ka; ਰੋਕੋ;
                    ਹਾਲਤ GGML_OP_POOL_MAX:                      ਰੋਕੋ;
                    ਹਾਲਤ GGML_OP_POOL_ਗਿਣਤੀ: GGML_ABORT("fatal error");
                }

                ਬਾਹਰਲਾ[ox] = res;
            }
        }

        cdata  += src->nb[2];
        dplane += pa;
    }
}

// ggml_compute_forward_pool_2d_back

ਖਾਲੀ ggml_compute_forward_pool_2d_ਪਿੱਠ(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src  = dst->src[0];
    ਅਚਲ ggml_tensor * dstf = dst->src[1]; // forward tensor of dst

    ਪ੍ਰਤਿਜ੍ਞਾ(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);

    ਜੇ (params->ith != 0) {
        ਵਾਪਸ;
    }

    ਅਚਲ int32_t * opts = (ਅਚਲ int32_t *)dst->op_params;
    ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    ਅਚਲ ਪੂਰਨ k0 = opts[1];
    ਅਚਲ ਪੂਰਨ k1 = opts[2];
    ਅਚਲ ਪੂਰਨ s0 = opts[3];
    ਅਚਲ ਪੂਰਨ s1 = opts[4];
    ਅਚਲ ਪੂਰਨ p0 = opts[5];
    ਅਚਲ ਪੂਰਨ p1 = opts[6];

    ਅੱਖਰ       * cdata  = (ਅੱਖਰ       *) dst->ਅੰਕੜੇ;
    ਅਚਲ ਅੱਖਰ * cdataf = (ਅਚਲ ਅੱਖਰ *) dstf->ਅੰਕੜੇ;
    ਅਚਲ ਅੱਖਰ * ਅਚਲ ਅੰਕੜੇ_ਸਿਰਾ = cdata + ggml_nbytes(dst);

    GGML_ASSERT(params->ith == 0);
    memset(cdata, 0, ggml_nbytes(dst));

    ਅਚਲ int64_t px = src->ne[0];
    ਅਚਲ int64_t py = src->ne[1];
    ਅਚਲ int64_t pa = px * py;

    ਅਚਲ ਦਸ਼ਮਲਵ * splane = (ਅਚਲ ਦਸ਼ਮਲਵ *) src->ਅੰਕੜੇ;

    ਅਚਲ ਪੂਰਨ ka = k0 * k1;
    ਅਚਲ ਪੂਰਨ offset0 = -p0;
    ਅਚਲ ਪੂਰਨ offset1 = -p1;

    ਜਦੋਂ (cdata < ਅੰਕੜੇ_ਸਿਰਾ) {
        ਲਈ (ਪੂਰਨ oy = 0; oy < py; ++oy) {
            ਅਚਲ ਦਸ਼ਮਲਵ * ਅਚਲ srow = splane + oy * px;
            ਲਈ (ਪੂਰਨ ox = 0; ox < px; ++ox) {
                ਅਚਲ ਦਸ਼ਮਲਵ grad0 = srow[ox];

                ਅਚਲ ਪੂਰਨ ix = offset0 + ox * s0;
                ਅਚਲ ਪੂਰਨ iy = offset1 + oy * s1;

                ਜੇ (op == GGML_OP_POOL_MAX) {
                    ਦਸ਼ਮਲਵ maxval = -FLT_MAX;
                    ਪੂਰਨ kxmax = -1;
                    ਪੂਰਨ kymax = -1;

                    ਲਈ (ਪੂਰਨ ky = 0; ky < k1; ++ky) {
                        ਜੇ (iy + ky < 0 || iy + ky >= dst->ne[1]) {
                            ਜਾਰੀ;
                        }
                        ਅਚਲ ਖਾਲੀ * drowf = (ਅਚਲ ਖਾਲੀ *)(cdataf + dst->nb[1] * (iy + ky));
                        ਲਈ (ਪੂਰਨ kx = 0; kx < k0; ++kx) {
                            ਪੂਰਨ j = ix + kx;
                            ਜੇ (j < 0 || j >= dst->ne[0]) {
                                ਜਾਰੀ;
                            }

                            ਅਚਲ ਦਸ਼ਮਲਵ val = dst->type == GGML_TYPE_F32 ?
                                ((ਅਚਲ ਦਸ਼ਮਲਵ *) drowf)[j] : GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(((ਅਚਲ ggml_fp16_t *) drowf)[j]);
                            ਜੇ (val <= maxval) {
                                ਜਾਰੀ;
                            }

                            maxval = val;
                            kxmax = kx;
                            kymax = ky;
                        }
                    }

                    ਜੇ (kxmax == -1 || kymax == -1) {
                        ਜਾਰੀ;
                    }

                    ਖਾਲੀ * drow = (ਖਾਲੀ *)(cdata + dst->nb[1] * (iy + kymax));
                    ਅਚਲ ਪੂਰਨ j = ix + kxmax;
                    ਜੇ (dst->type == GGML_TYPE_F32) {
                        ((ਦਸ਼ਮਲਵ *) drow)[j] += grad0;
                    } ਨਹੀਂਤਾਂ {
                        ((ggml_fp16_t *) drow)[j] = GGML_CPU_FP32_ਮੰਜ਼ਿਲ_FP16(grad0 + GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(((ਅਚਲ ggml_fp16_t *) drow)[j]));
                    }
                } ਨਹੀਂਤਾਂ ਜੇ (op == GGML_OP_POOL_AVG) {
                    ਅਚਲ ਦਸ਼ਮਲਵ grad = grad0 / ka;

                    ਲਈ (ਪੂਰਨ ky = 0; ky < k1; ++ky) {
                        ਜੇ (iy + ky < 0 || iy + ky >= dst->ne[1]) {
                            ਜਾਰੀ;
                        }
                        ਖਾਲੀ * drow = (ਖਾਲੀ *)(cdata + dst->nb[1] * (iy + ky));
                        ਲਈ (ਪੂਰਨ kx = 0; kx < k0; ++kx) {
                            ਪੂਰਨ j = ix + kx;
                            ਜੇ (j < 0 || j >= dst->ne[0]) {
                                ਜਾਰੀ;
                            }

                            ਜੇ (dst->type == GGML_TYPE_F32) {
                                ((ਦਸ਼ਮਲਵ *) drow)[j] += grad;
                            } ਨਹੀਂਤਾਂ {
                                ((ggml_fp16_t *) drow)[j] += GGML_CPU_FP32_ਮੰਜ਼ਿਲ_FP16(grad);
                            }
                        }
                    }
                } ਨਹੀਂਤਾਂ {
                    GGML_ASSERT(ਝੂਠ);
                }
            }
        }

        cdata  += dst->nb[2];
        cdataf += dst->nb[2];
        splane += pa;
    }
}

// ggml_compute_forward_upscale

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_upscale_f32(
    ਅਚਲ ggml_compute_params * params,
    ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    ਦਸ਼ਮਲਵ sf0 = (ਦਸ਼ਮਲਵ)ne0/src0->ne[0];
    ਦਸ਼ਮਲਵ sf1 = (ਦਸ਼ਮਲਵ)ne1/src0->ne[1];
    ਦਸ਼ਮਲਵ sf2 = (ਦਸ਼ਮਲਵ)ne2/src0->ne[2];
    ਦਸ਼ਮਲਵ sf3 = (ਦਸ਼ਮਲਵ)ne3/src0->ne[3];
    ਦਸ਼ਮਲਵ pixel_ਵਿੱਥ = 0.5f;

    ਅਚਲ int32_t ਬਹੁਲਕ_flags = ggml_get_op_params_i32(dst, 0);
    ਅਚਲ ggml_scale_ਬਹੁਲਕ mode = (ggml_scale_ਬਹੁਲਕ) (ਬਹੁਲਕ_flags & 0xFF);

    ਜੇ (ਬਹੁਲਕ_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        pixel_ਵਿੱਥ = 0.0f;
        sf0 = ne0 > 1 && ne00 > 1 ? (ਦਸ਼ਮਲਵ)(ne0 - 1) / (ne00 - 1) : sf0;
        sf1 = ne1 > 1 && ne01 > 1 ? (ਦਸ਼ਮਲਵ)(ne1 - 1) / (ne01 - 1) : sf1;
    }

    ਜੇ (mode == GGML_SCALE_ਬਹੁਲਕ_ਸਭ_ਤੋਂ_ਨੇੜੇ) {
        ਲਈ (int64_t i3 = 0; i3 < ne3; i3++) {
            ਅਚਲ int64_t i03 = i3 / sf3;
            ਲਈ (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                ਅਚਲ int64_t i02 = i2 / sf2;
                ਲਈ (int64_t i1 = 0; i1 < ne1; i1++) {
                    ਅਚਲ int64_t i01 = i1 / sf1;
                    ਲਈ (int64_t i0 = 0; i0 < ne0; i0++) {
                        ਅਚਲ int64_t i00 = i0 / sf0;

                        ਅਚਲ ਦਸ਼ਮਲਵ * x = (ਦਸ਼ਮਲਵ *)((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              ਦਸ਼ਮਲਵ * y = (ਦਸ਼ਮਲਵ *)((ਅੱਖਰ *)  dst->ਅੰਕੜੇ +  i0*nb0  +  i1*nb1  +  i2*nb2  +  i3*nb3);

                        *y = *x;
                    }
                }
            }
        }
    } ਨਹੀਂਤਾਂ ਜੇ (mode == GGML_SCALE_ਬਹੁਲਕ_BILINEAR && (ਬਹੁਲਕ_flags & GGML_SCALE_FLAG_ANTIALIAS)) {
        // Similar to F.interpolate(..., mode="bilinear", align_corners=False, antialias=True)
        // https://github.com/pytorch/pytorch/blob/8871ff29b743948d1225389d5b7068f37b22750b/aten/src/ATen/native/cpu/UpSampleKernel.cpp
        ਸਵੈਚਾਲਿਤ triangle_filter = [](ਦਸ਼ਮਲਵ x) -> ਦਸ਼ਮਲਵ {
            ਵਾਪਸ std::max(1.0f - fabsf(x), 0.0f);
        };

        // support and invscale, minimum 1 pixel for bilinear
        ਅਚਲ ਦਸ਼ਮਲਵ support1  = std::max(1.0f, 1.0f / sf1);
        ਅਚਲ ਦਸ਼ਮਲਵ invscale1 = 1.0f / support1;
        ਅਚਲ ਦਸ਼ਮਲਵ support0  = std::max(1.0f, 1.0f / sf0);
        ਅਚਲ ਦਸ਼ਮਲਵ invscale0 = 1.0f / support0;

        ਲਈ (int64_t i3 = 0; i3 < ne3; i3++) {
            ਅਚਲ int64_t i03 = i3 / sf3;
            ਲਈ (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                ਅਚਲ int64_t i02 = i2 / sf2;
                ਲਈ (int64_t i1 = 0; i1 < ne1; i1++) {
                    ਅਚਲ ਦਸ਼ਮਲਵ y = ((ਦਸ਼ਮਲਵ) i1 + pixel_ਵਿੱਥ) / sf1;
                    ਲਈ (int64_t i0 = 0; i0 < ne0; i0++) {
                        ਅਚਲ ਦਸ਼ਮਲਵ x = ((ਦਸ਼ਮਲਵ) i0 + pixel_ਵਿੱਥ) / sf0;

                        // the range of source pixels that contribute
                        ਅਚਲ int64_t x_min = std::max<int64_t>(x - support0 + pixel_ਵਿੱਥ, 0);
                        ਅਚਲ int64_t x_max = std::min<int64_t>(x + support0 + pixel_ਵਿੱਥ, ne00);
                        ਅਚਲ int64_t y_min = std::max<int64_t>(y - support1 + pixel_ਵਿੱਥ, 0);
                        ਅਚਲ int64_t y_max = std::min<int64_t>(y + support1 + pixel_ਵਿੱਥ, ne01);

                        // bilinear filter with antialiasing
                        ਦਸ਼ਮਲਵ val = 0.0f;
                        ਦਸ਼ਮਲਵ ਕੁੱਲ_ਭਾਰ = 0.0f;

                        ਲਈ (int64_t sy = y_min; sy < y_max; sy++) {
                            ਅਚਲ ਦਸ਼ਮਲਵ ਭਾਰ_y = triangle_filter((sy - y + pixel_ਵਿੱਥ) * invscale1);

                            ਲਈ (int64_t sx = x_min; sx < x_max; sx++) {
                                ਅਚਲ ਦਸ਼ਮਲਵ ਭਾਰ_x = triangle_filter((sx - x + pixel_ਵਿੱਥ) * invscale0);
                                ਅਚਲ ਦਸ਼ਮਲਵ ਭਾਰ = ਭਾਰ_x * ਭਾਰ_y;

                                ਜੇ (ਭਾਰ <= 0.0f) {
                                    ਜਾਰੀ;
                                }

                                ਅਚਲ ਦਸ਼ਮਲਵ pixel = *(ਅਚਲ ਦਸ਼ਮਲਵ *)((ਅਚਲ ਅੱਖਰ *)src0->ਅੰਕੜੇ + sx*nb00 + sy*nb01 + i02*nb02 + i03*nb03);
                                val += pixel * ਭਾਰ;
                                ਕੁੱਲ_ਭਾਰ += ਭਾਰ;
                            }
                        }

                        ਜੇ (ਕੁੱਲ_ਭਾਰ > 0.0f) {
                            val /= ਕੁੱਲ_ਭਾਰ;
                        }

                        ਦਸ਼ਮਲਵ * dst_ptr = (ਦਸ਼ਮਲਵ *)((ਅੱਖਰ *)dst->ਅੰਕੜੇ + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);
                        *dst_ptr = val;
                    }
                }
            }
        }
    } ਨਹੀਂਤਾਂ ਜੇ (mode == GGML_SCALE_ਬਹੁਲਕ_BILINEAR) {
        ਲਈ (int64_t i3 = 0; i3 < ne3; i3++) {
            ਅਚਲ int64_t i03 = i3 / sf3;
            ਲਈ (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                ਅਚਲ int64_t i02 = i2 / sf2;
                ਲਈ (int64_t i1 = 0; i1 < ne1; i1++) {
                    ਅਚਲ ਦਸ਼ਮਲਵ y = ((ਦਸ਼ਮਲਵ)i1 + pixel_ਵਿੱਥ) / sf1 - pixel_ਵਿੱਥ;
                    int64_t y0 = (int64_t)floorf(y);
                    int64_t y1 = y0 + 1;

                    y0 = std::max(int64_t(0), std::min(y0, ne01 - 1));
                    y1 = std::max(int64_t(0), std::min(y1, ne01 - 1));

                    ਦਸ਼ਮਲਵ dy = y - (ਦਸ਼ਮਲਵ)y0;
                    dy = std::max(0.0f, std::min(dy, 1.0f));

                    ਲਈ (int64_t i0 = 0; i0 < ne0; i0++) {
                        ਅਚਲ ਦਸ਼ਮਲਵ x = ((ਦਸ਼ਮਲਵ)i0 + pixel_ਵਿੱਥ) / sf0 - pixel_ਵਿੱਥ;
                        int64_t x0 = (int64_t)floorf(x);
                        int64_t x1 = x0 + 1;

                        x0 = std::max(int64_t(0), std::min(x0, ne00 - 1));
                        x1 = std::max(int64_t(0), std::min(x1, ne00 - 1));

                        ਦਸ਼ਮਲਵ dx = x - (ਦਸ਼ਮਲਵ)x0;
                        dx = std::max(0.0f, std::min(dx, 1.0f));

                        // fetch the four surrounding pixel values and interpolate
                        ਅਚਲ ਦਸ਼ਮਲਵ a = *(ਅਚਲ ਦਸ਼ਮਲਵ *)((ਅਚਲ ਅੱਖਰ *)src0->ਅੰਕੜੇ + x0*nb00 + y0*nb01 + i02*nb02 + i03*nb03);
                        ਅਚਲ ਦਸ਼ਮਲਵ b = *(ਅਚਲ ਦਸ਼ਮਲਵ *)((ਅਚਲ ਅੱਖਰ *)src0->ਅੰਕੜੇ + x1*nb00 + y0*nb01 + i02*nb02 + i03*nb03);
                        ਅਚਲ ਦਸ਼ਮਲਵ c = *(ਅਚਲ ਦਸ਼ਮਲਵ *)((ਅਚਲ ਅੱਖਰ *)src0->ਅੰਕੜੇ + x0*nb00 + y1*nb01 + i02*nb02 + i03*nb03);
                        ਅਚਲ ਦਸ਼ਮਲਵ d = *(ਅਚਲ ਦਸ਼ਮਲਵ *)((ਅਚਲ ਅੱਖਰ *)src0->ਅੰਕੜੇ + x1*nb00 + y1*nb01 + i02*nb02 + i03*nb03);

                        ਅਚਲ ਦਸ਼ਮਲਵ val = a*(1 - dx)*(1 - dy) + b*dx*(1 - dy) + c*(1 - dx)*dy + d*dx*dy;

                        ਦਸ਼ਮਲਵ * y_dst = (ਦਸ਼ਮਲਵ *)((ਅੱਖਰ *)dst->ਅੰਕੜੇ + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);
                        *y_dst = val;
                    }
                }
            }
        }
    } ਨਹੀਂਤਾਂ ਜੇ (mode == GGML_SCALE_ਬਹੁਲਕ_BICUBIC) {
        // https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
        ਅਚਲ ਦਸ਼ਮਲਵ a = -0.75f; // use alpha = -0.75 (same as PyTorch)
        ਸਵੈਚਾਲਿਤ weight1 = [a](ਦਸ਼ਮਲਵ x) { ਵਾਪਸ ((a + 2) * x - (a + 3)) * x * x + 1; };
        ਸਵੈਚਾਲਿਤ weight2 = [a](ਦਸ਼ਮਲਵ x) { ਵਾਪਸ ((a * x - 5 * a) * x + 8 * a) * x - 4 * a; };
        ਸਵੈਚਾਲਿਤ bicubic = [=](ਦਸ਼ਮਲਵ p0, ਦਸ਼ਮਲਵ p1, ਦਸ਼ਮਲਵ p2, ਦਸ਼ਮਲਵ p3, ਦਸ਼ਮਲਵ x) {
            ਅਚਲ ਦਸ਼ਮਲਵ w0 = weight2(x + 1);
            ਅਚਲ ਦਸ਼ਮਲਵ w1 = weight1(x + 0);
            ਅਚਲ ਦਸ਼ਮਲਵ w2 = weight1(1 - x);
            ਅਚਲ ਦਸ਼ਮਲਵ w3 = weight2(2 - x);
            ਵਾਪਸ p0*w0 + p1*w1 + p2*w2 + p3*w3;
        };

        ਲਈ (int64_t i3 = 0; i3 < ne3; i3++) {
            ਅਚਲ int64_t i03 = i3 / sf3;
            ਲਈ (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                ਅਚਲ int64_t i02 = i2 / sf2;
                ਲਈ (int64_t i1 = 0; i1 < ne1; i1++) {
                    ਅਚਲ ਦਸ਼ਮਲਵ y = ((ਦਸ਼ਮਲਵ)i1 + pixel_ਵਿੱਥ) / sf1 - pixel_ਵਿੱਥ;
                    ਅਚਲ int64_t y0 = (int64_t)floorf(y);
                    ਅਚਲ ਦਸ਼ਮਲਵ dy = y - (ਦਸ਼ਮਲਵ)y0;

                    ਲਈ (int64_t i0 = 0; i0 < ne0; i0++) {
                        ਅਚਲ ਦਸ਼ਮਲਵ x = ((ਦਸ਼ਮਲਵ)i0 + pixel_ਵਿੱਥ) / sf0 - pixel_ਵਿੱਥ;
                        ਅਚਲ int64_t x0 = (int64_t)floorf(x);
                        ਅਚਲ ਦਸ਼ਮਲਵ dx = x - (ਦਸ਼ਮਲਵ)x0;

                        ਸਵੈਚਾਲਿਤ p = [=](int64_t x_off, int64_t y_off) -> ਦਸ਼ਮਲਵ {
                            int64_t i00 = std::max(int64_t(0), std::min(x0 + x_off, ne00 - 1));
                            int64_t i01 = std::max(int64_t(0), std::min(y0 + y_off, ne01 - 1));
                            ਵਾਪਸ *(ਅਚਲ ਦਸ਼ਮਲਵ *)((ਅਚਲ ਅੱਖਰ *)src0->ਅੰਕੜੇ + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                        };

                        ਅਚਲ ਦਸ਼ਮਲਵ val = bicubic(
                            bicubic(p(-1,-1), p(0,-1), p(1,-1), p(2,-1), dx),
                            bicubic(p(-1, 0), p(0, 0), p(1, 0), p(2, 0), dx),
                            bicubic(p(-1, 1), p(0, 1), p(1, 1), p(2, 1), dx),
                            bicubic(p(-1, 2), p(0, 2), p(1, 2), p(2, 2), dx), dy);

                        ਦਸ਼ਮਲਵ * y_dst = (ਦਸ਼ਮਲਵ *)((ਅੱਖਰ *)dst->ਅੰਕੜੇ + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);
                        *y_dst = val;
                    }
                }
            }
        }
    } ਨਹੀਂਤਾਂ {
        GGML_ABORT("unsupported upscale mode");
    }
}

ਖਾਲੀ ggml_compute_forward_upscale(
    ਅਚਲ ggml_compute_params * params,
    ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_upscale_f32(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}


// ggml_compute_forward_pad

ਸਾਂਚਾ<ਤਾਰਕਿਕ circular_t>
ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_pad_f32(
    ਅਚਲ ggml_compute_params * params,
          ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਪ੍ਰਤਿਜ੍ਞਾ(dst->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    ਦਸ਼ਮਲਵ * dst_ptr = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;
    ਅਚਲ int32_t lp0 = ggml_get_op_params_i32(dst, 0);
    ਅਚਲ int32_t rp0 = ggml_get_op_params_i32(dst, 1);
    ਅਚਲ int32_t lp1 = ggml_get_op_params_i32(dst, 2);
    ਅਚਲ int32_t rp1 = ggml_get_op_params_i32(dst, 3);
    ਅਚਲ int32_t lp2 = ggml_get_op_params_i32(dst, 4);
    ਅਚਲ int32_t rp2 = ggml_get_op_params_i32(dst, 5);
    ਅਚਲ int32_t lp3 = ggml_get_op_params_i32(dst, 6);
    ਅਚਲ int32_t rp3 = ggml_get_op_params_i32(dst, 7);

    ਅਚਲ int64_t nrows = ne1 * ne2 * ne3;
    ਅਚਲ ਤਾਰਕਿਕ src_ਪੰਕਤੀ_contig = src0->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    ਲਈ (int64_t ਪੰਕਤੀ = ith; ਪੰਕਤੀ < nrows; ਪੰਕਤੀ += nth) {
        ਅਚਲ int64_t i1 = ਪੰਕਤੀ % ne1;
        ਅਚਲ int64_t i2 = (ਪੰਕਤੀ / ne1) % ne2;
        ਅਚਲ int64_t i3 = ਪੰਕਤੀ / (ne1 * ne2);

        ਦਸ਼ਮਲਵ * dst_ਪੰਕਤੀ = dst_ptr + ਪੰਕਤੀ * ne0;

        ਜੇ constexpr (circular_t) {
            // circular padding wraps around on a torus
            ਅਚਲ int64_t src_i1 = ggml_wrap_around(i1 - lp1, ne01);
            ਅਚਲ int64_t src_i2 = ggml_wrap_around(i2 - lp2, ne02);
            ਅਚਲ int64_t src_i3 = ggml_wrap_around(i3 - lp3, ne03);

            ਲਈ (int64_t i0 = 0; i0 < ne0; ++i0) {
                ਅਚਲ int64_t src_i0 = ggml_wrap_around(i0 - lp0, ne00);
                dst_ਪੰਕਤੀ[i0] = *(ਅਚਲ ਦਸ਼ਮਲਵ *)((ਅਚਲ ਅੱਖਰ *) src0->ਅੰਕੜੇ + src_i3*nb03 + src_i2*nb02 + src_i1*nb01 + src_i0*nb00);
            }
        } ਨਹੀਂਤਾਂ {
            ਅਚਲ ਤਾਰਕਿਕ interior = (i1 >= lp1 && i1 < ne1 - rp1)
                               && (i2 >= lp2 && i2 < ne2 - rp2)
                               && (i3 >= lp3 && i3 < ne3 - rp3);
            ਜੇ (!interior) {
                memset(dst_ਪੰਕਤੀ, 0, ne0 * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                ਜਾਰੀ;
            }

            ਅਚਲ ਅੱਖਰ * src_ਪੰਕਤੀ = (ਅਚਲ ਅੱਖਰ *) src0->ਅੰਕੜੇ + (i3 - lp3)*nb03 + (i2 - lp2)*nb02 + (i1 - lp1)*nb01;

            ਜੇ (src_ਪੰਕਤੀ_contig && lp0 >= 0 && rp0 >= 0) {
                memset(dst_ਪੰਕਤੀ, 0, lp0 * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                ਸਮ੍ਰਿਤੀ_ਨਕਲ(dst_ਪੰਕਤੀ + lp0, src_ਪੰਕਤੀ, ne00 * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                memset(dst_ਪੰਕਤੀ + lp0 + ne00, 0, rp0 * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
            } ਨਹੀਂਤਾਂ {
                ਲਈ (int64_t i0 = 0; i0 < ne0; ++i0) {
                    ਜੇ (i0 >= lp0 && i0 < ne0 - rp0) {
                        dst_ਪੰਕਤੀ[i0] = *(ਅਚਲ ਦਸ਼ਮਲਵ *)(src_ਪੰਕਤੀ + (i0 - lp0)*nb00);
                    } ਨਹੀਂਤਾਂ {
                        dst_ਪੰਕਤੀ[i0] = 0.0f;
                    }
                }
            }
        }
    }
}


ਖਾਲੀ ggml_compute_forward_pad(
    ਅਚਲ ggml_compute_params * params,
    ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ਤਾਰਕਿਕ circular = (ਤਾਰਕਿਕ) ggml_get_op_params_i32(dst, 8);
    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ਜੇ (circular) {
                    ggml_compute_forward_pad_f32<ਸੱਚ>(params, dst);
                } ਨਹੀਂਤਾਂ {
                    ggml_compute_forward_pad_f32<ਝੂਠ>(params, dst);
                }
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_pad_reflect_1d

ਖਾਲੀ ggml_compute_forward_pad_reflect_1d(
        ਅਚਲ ggml_compute_params * params,
              ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ int32_t * opts = (ਅਚਲ int32_t *) dst->op_params;
    ਅਚਲ ਪੂਰਨ p0 = opts[0];
    ਅਚਲ ਪੂਰਨ p1 = opts[1];

    GGML_TENSOR_UNARY_OP_LOCALS

    ਲਈ (int64_t i3 = 0; i3 < ne3; i3++) {
        ਲਈ (int64_t i2 = 0; i2 < ne2; i2++) {
            ਲਈ (int64_t i1 = ith; i1 < ne1; i1 += nth) {
                ਦਸ਼ਮਲਵ * ਖੱਬੇ  = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i3*nb3 + i2*nb2 + i1*nb1 +         p0*nb0);
                ਦਸ਼ਮਲਵ * ਸੱਜੇ = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i3*nb3 + i2*nb2 + i1*nb1 + (ne0-p1-1)*nb0);

                ggml_vec_cpy_f32(ne00, ਖੱਬੇ, (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i3*nb03 + i2*nb02 + i1*nb01));

                ਲਈ (ਪੂਰਨ i0 = 1; i0 <= p0; i0++) { ਖੱਬੇ[-i0] = ਖੱਬੇ[i0];   }
                ਲਈ (ਪੂਰਨ i0 = 1; i0 <= p1; i0++) { ਸੱਜੇ[i0] = ਸੱਜੇ[-i0]; }
            }
        }
    }
}

// ggml_compute_forward_roll

ਸਥਿਰ int64_t ggml_wrap_ਸੂਚਕਾਂਕ(int64_t i, int64_t ne) {
    ਜੇ (i < 0) {
        ਵਾਪਸ i + ne;
    } ਨਹੀਂਤਾਂ ਜੇ (i >= ne) {
        ਵਾਪਸ i - ne;
    }
    ਵਾਪਸ i;
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_roll_f32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ਦਸ਼ਮਲਵ * src_ਅੰਕੜੇ = (ਅਚਲ ਦਸ਼ਮਲਵ *) src0->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * dst_ਅੰਕੜੇ = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;

    GGML_TENSOR_UNARY_OP_LOCALS

    ਅਚਲ ਪੂਰਨ s0 = ggml_get_op_params_i32(dst, 0);
    ਅਚਲ ਪੂਰਨ s1 = ggml_get_op_params_i32(dst, 1);
    ਅਚਲ ਪੂਰਨ s2 = ggml_get_op_params_i32(dst, 2);
    ਅਚਲ ਪੂਰਨ s3 = ggml_get_op_params_i32(dst, 3);

    ਅਚਲ int64_t ਕੁੱਲ = ne1 * ne2 * ne3;
    ਅਚਲ int64_t ਪ੍ਰਤੀ_thread = (ਕੁੱਲ + params->nth) / params->nth;
    ਅਚਲ int64_t start = params->ith * ਪ੍ਰਤੀ_thread;
    ਅਚਲ int64_t end   = std::min(start + ਪ੍ਰਤੀ_thread, ਕੁੱਲ);

    ਲਈ (int64_t i = start; i < end; ++i) {
        ਅਚਲ int64_t i1 = i % ne1;
        ਅਚਲ int64_t i2 = (i / ne1) % ne2;
        ਅਚਲ int64_t i3 = i / (ne2 * ne1);
        ਦਸ਼ਮਲਵ * dst_ਪੰਕਤੀ = dst_ਅੰਕੜੇ + (i3*nb3 + i2*nb2 + i1*nb1) / ਆਕਾਰ(ਦਸ਼ਮਲਵ);

        ਅਚਲ int64_t i01 = ggml_wrap_ਸੂਚਕਾਂਕ(i1 - s1, ne01);
        ਅਚਲ int64_t i02 = ggml_wrap_ਸੂਚਕਾਂਕ(i2 - s2, ne02);
        ਅਚਲ int64_t i03 = ggml_wrap_ਸੂਚਕਾਂਕ(i3 - s3, ne03);
        ਅਚਲ ਦਸ਼ਮਲਵ * src_ਪੰਕਤੀ = src_ਅੰਕੜੇ + (i03*nb03 + i02*nb02 + i01*nb01) / ਆਕਾਰ(ਦਸ਼ਮਲਵ);

        ਅਚਲ int64_t s = ggml_wrap_ਸੂਚਕਾਂਕ(-s0, ne00);
        ਅਚਲ int64_t n = ne00 - s;
        ggml_vec_cpy_f32(n, dst_ਪੰਕਤੀ,     src_ਪੰਕਤੀ + s);
        ggml_vec_cpy_f32(s, dst_ਪੰਕਤੀ + n, src_ਪੰਕਤੀ);
    }
}

ਖਾਲੀ ggml_compute_forward_roll(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_roll_f32(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_arange

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_arange_f32(
    ਅਚਲ ggml_compute_params * params,
    ggml_tensor * dst) {

    GGML_ASSERT(dst->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਦਸ਼ਮਲਵ start = ggml_get_op_params_f32(dst, 0);
    ਅਚਲ ਦਸ਼ਮਲਵ stop  = ggml_get_op_params_f32(dst, 1);
    ਅਚਲ ਦਸ਼ਮਲਵ ਕਦਮ  = ggml_get_op_params_f32(dst, 2);

    ਅਚਲ int64_t ਕਦਮਾਂ = (int64_t) ceilf((stop - start) / ਕਦਮ);

    GGML_ASSERT(ggml_nelements(dst) == ਕਦਮਾਂ);

    ਲਈ (int64_t i = ith; i < ਕਦਮਾਂ; i+= nth) {
        ਦਸ਼ਮਲਵ value = start + ਕਦਮ * i;
        ((ਦਸ਼ਮਲਵ *)dst->ਅੰਕੜੇ)[i] = value;
    }
}

ਖਾਲੀ ggml_compute_forward_arange(
    ਅਚਲ ggml_compute_params * params,
    ggml_tensor * dst) {
    ਚੋਣ (dst->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_arange_f32(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_timestep_embedding_f32(
    ਅਚਲ ggml_compute_params * params,
    ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    ਅਚਲ ਪੂਰਨ dim = ggml_get_op_params_i32(dst, 0);
    ਅਚਲ ਪੂਰਨ max_period = ggml_get_op_params_i32(dst, 1);

    ਪੂਰਨ half = dim / 2;

    ਲਈ (int64_t i = 0; i < ne00; i++) {
        ਦਸ਼ਮਲਵ * embed_ਅੰਕੜੇ = (ਦਸ਼ਮਲਵ *)((ਅੱਖਰ *)  dst->ਅੰਕੜੇ +  i*nb1);
        ਲਈ (int64_t j = ith; j < half; j += nth) {
            ਦਸ਼ਮਲਵ timestep = ((ਦਸ਼ਮਲਵ *)src0->ਅੰਕੜੇ)[i];
            ਦਸ਼ਮਲਵ freq = (ਦਸ਼ਮਲਵ)expf(-logf(max_period) * j / half);
            ਦਸ਼ਮਲਵ arg = timestep * freq;
            embed_ਅੰਕੜੇ[j] = cosf(arg);
            embed_ਅੰਕੜੇ[j + half] = sinf(arg);
        }
        ਜੇ (dim % 2 != 0 && ith == 0) {
            embed_ਅੰਕੜੇ[2 * half] = 0.f;
        }
    }
}

ਖਾਲੀ ggml_compute_forward_timestep_embedding(
    ਅਚਲ ggml_compute_params * params,
    ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_timestep_embedding_f32(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_argsort

ਸਾਂਚਾ<ਗਿਣਤੀ ggml_ਛਾਂਟੋ_ਕ੍ਰਮ ਕ੍ਰਮ>
ਬਣਤਰ cmp_argsort {
    ਅਚਲ ਦਸ਼ਮਲਵ * ਅੰਕੜੇ;
    ਤਾਰਕਿਕ ਸੰਚਾਲਕ()(int32_t a, int32_t b) ਅਚਲ {
        ਜੇ constexpr (ਕ੍ਰਮ == GGML_ਛਾਂਟੋ_ਕ੍ਰਮ_ASC) {
            ਵਾਪਸ ਅੰਕੜੇ[a] < ਅੰਕੜੇ[b];
        } ਨਹੀਂਤਾਂ {
            ਵਾਪਸ ਅੰਕੜੇ[a] > ਅੰਕੜੇ[b];
        }
    }
};

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_argsort_f32(
    ਅਚਲ ggml_compute_params * params,
    ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(nb0 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ int64_t nr = ggml_nrows(src0);

    ggml_ਛਾਂਟੋ_ਕ੍ਰਮ ਕ੍ਰਮ = (ggml_ਛਾਂਟੋ_ਕ੍ਰਮ) ggml_get_op_params_i32(dst, 0);

    ਲਈ (int64_t i = ith; i < nr; i += nth) {
        ਅਚਲ ਦਸ਼ਮਲਵ * src_ਅੰਕੜੇ = (ਦਸ਼ਮਲਵ *)((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i*nb01);

        int32_t * dst_ਅੰਕੜੇ = (int32_t *)((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i*nb1);

        ਲਈ (int64_t j = 0; j < ne0; j++) {
            dst_ਅੰਕੜੇ[j] = j;
        }

        ਚੋਣ (ਕ੍ਰਮ) {
            ਹਾਲਤ GGML_ਛਾਂਟੋ_ਕ੍ਰਮ_ASC:
                std::sort(dst_ਅੰਕੜੇ, dst_ਅੰਕੜੇ + ne0, cmp_argsort<GGML_ਛਾਂਟੋ_ਕ੍ਰਮ_ASC>{src_ਅੰਕੜੇ});
                ਰੋਕੋ;

            ਹਾਲਤ GGML_ਛਾਂਟੋ_ਕ੍ਰਮ_DESC:
                std::sort(dst_ਅੰਕੜੇ, dst_ਅੰਕੜੇ + ne0, cmp_argsort<GGML_ਛਾਂਟੋ_ਕ੍ਰਮ_DESC>{src_ਅੰਕੜੇ});
                ਰੋਕੋ;

            ਮੂਲ:
                GGML_ABORT("invalid sort order");
        }
    }
}

ਖਾਲੀ ggml_compute_forward_argsort(
    ਅਚਲ ggml_compute_params * params,
    ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_argsort_f32(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_top_k

ਬਣਤਰ cmp_ਸਿਖਰ_k {
    ਅਚਲ ਦਸ਼ਮਲਵ * ਅੰਕੜੇ;
    ਤਾਰਕਿਕ ਸੰਚਾਲਕ()(int32_t a, int32_t b) ਅਚਲ {
        ਵਾਪਸ ਅੰਕੜੇ[a] > ਅੰਕੜੇ[b];
    }
};

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_ਸਿਖਰ_k_f32(
    ਅਚਲ ggml_compute_params * params,
    ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(nb0 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ int64_t nr = ggml_nrows(src0);

    ਅਚਲ ਪੂਰਨ ਸਿਖਰ_k = ne0;

    int32_t * tmp = (int32_t *) params->wdata + (ne00 + CACHE_ਸਤਰ_ਮਾਪ_F32) * ith;

    ਲਈ (int64_t i = ith; i < nr; i += nth) {
        ਅਚਲ ਦਸ਼ਮਲਵ * src_ਅੰਕੜੇ = (ਦਸ਼ਮਲਵ *)((ਅੱਖਰ *) src0->ਅੰਕੜੇ + i*nb01);

        ਲਈ (int64_t j = 0; j < ne00; j++) {
            tmp[j] = j;
        }

        std::partial_sort(tmp, tmp + ਸਿਖਰ_k, tmp + ne00, cmp_ਸਿਖਰ_k{src_ਅੰਕੜੇ});

        int32_t * dst_ਅੰਕੜੇ = (int32_t *)((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i*nb1);

        std::copy(tmp, tmp + ਸਿਖਰ_k, dst_ਅੰਕੜੇ);

        // emphasize that the order is not important
        ਜੇ (ਸਿਖਰ_k > 1) {
            std::swap(dst_ਅੰਕੜੇ[0], dst_ਅੰਕੜੇ[1]);
        }
    }
}

ਖਾਲੀ ggml_compute_forward_ਸਿਖਰ_k(
    ਅਚਲ ggml_compute_params * params,
    ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_ਸਿਖਰ_k_f32(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_flash_attn_ext_f16_one_chunk(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst,
        ਪੂਰਨ ir0, ਪੂਰਨ ir1,
        int64_t ic_ਸ਼ੁਰੂ, int64_t ic_ਸਿਰਾ,
        ਦਸ਼ਮਲਵ * partials, int64_t partial_stride) {

    ਅਚਲ ਤਾਰਕਿਕ write_partials = (partials != ਖਾਲੀ_ਸੂਚਕ);
    ਅਚਲ ggml_tensor * q     = dst->src[0];
    ਅਚਲ ggml_tensor * k     = dst->src[1];
    ਅਚਲ ggml_tensor * v     = dst->src[2];
    ਅਚਲ ggml_tensor * ਨਕਾਬ  = dst->src[3];
    ਅਚਲ ggml_tensor * sinks = dst->src[4];

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    ਅਚਲ int64_t DK = nek0;
    ਅਚਲ int64_t DV = nev0;
    ਅਚਲ int64_t N  = neq1;

    GGML_ASSERT(ne0 == DV);
    GGML_ASSERT(ne2 == N);

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_ਮਾਪ(q->type));
    GGML_ASSERT(nbk0 == ggml_type_ਮਾਪ(k->type));
    GGML_ASSERT(nbv0 == ggml_type_ਮਾਪ(v->type));

    GGML_ASSERT(neq0 == DK);
    GGML_ASSERT(nek0 == DK);
    GGML_ASSERT(nev0 == DV);

    GGML_ASSERT(neq1 == N);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    ਅਚਲ int64_t rk2 = neq2/nek2;
    ਅਚਲ int64_t rk3 = neq3/nek3;

    ਅਚਲ int64_t rv2 = neq2/nev2;
    ਅਚਲ int64_t rv3 = neq3/nev3;

    // parallelize by q rows using ggml_vec_dot_f32

    ਦਸ਼ਮਲਵ scale         = 1.0f;
    ਦਸ਼ਮਲਵ max_bias      = 0.0f;
    ਦਸ਼ਮਲਵ logit_softcap = 0.0f;

    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&scale,         (ਦਸ਼ਮਲਵ *) dst->op_params + 0, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&max_bias,      (ਦਸ਼ਮਲਵ *) dst->op_params + 1, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&logit_softcap, (ਦਸ਼ਮਲਵ *) dst->op_params + 2, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਜੇ (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    ਅਚਲ uint32_t n_ਸਿਰ      = neq2;
    ਅਚਲ uint32_t n_ਸਿਰ_log2 = 1u << (uint32_t) floor(log2(n_ਸਿਰ));

    ਅਚਲ ਦਸ਼ਮਲਵ m0 = powf(2.0f, -(max_bias       ) / n_ਸਿਰ_log2);
    ਅਚਲ ਦਸ਼ਮਲਵ m1 = powf(2.0f, -(max_bias / 2.0f) / n_ਸਿਰ_log2);

    ggml_type         ਅਚਲ k_vec_dot_type = ggml_get_type_traits_cpu(k->type)->vec_dot_type;
    ggml_from_float_t ਅਚਲ q_ਮੰਜ਼ਿਲ_vec_dot   = ggml_get_type_traits_cpu(k_vec_dot_type)->from_float;
    ggml_vec_dot_t    ਅਚਲ kq_vec_dot     = ggml_get_type_traits_cpu(k->type)->vec_dot;
    ggml_ਮੰਜ਼ਿਲ_float_t   ਅਚਲ v_ਮੰਜ਼ਿਲ_float     = ggml_get_type_traits(v->type)->ਮੰਜ਼ਿਲ_float;

    GGML_ASSERT((                            q_ਮੰਜ਼ਿਲ_vec_dot) && "fattn: unsupported K-type");
    GGML_ASSERT((v->type == GGML_TYPE_F32 || v_ਮੰਜ਼ਿਲ_float  ) && "fattn: unsupported V-type");

    ਪੂਰਨ ith = params->ith;

    ਲਈ (ਪੂਰਨ ir = ir0; ir < ir1; ++ir) {
        // q indices
        ਅਚਲ ਪੂਰਨ iq3 = ir/(neq2*neq1);
        ਅਚਲ ਪੂਰਨ iq2 = (ir - iq3*neq2*neq1)/neq1;
        ਅਚਲ ਪੂਰਨ iq1 = (ir - iq3*neq2*neq1 - iq2*neq1);

        ਅਚਲ uint32_t h = iq2; // head index
        ਅਚਲ ਦਸ਼ਮਲਵ ਢਲਾਨ = (max_bias > 0.0f) ? h < n_ਸਿਰ_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_ਸਿਰ_log2) + 1) : 1.0f;

        ਦਸ਼ਮਲਵ S = 0.0f;      //ਜੋੜ
        ਦਸ਼ਮਲਵ M = -INFINITY; // maximum KQ value

        ਦਸ਼ਮਲਵ       * VKQ32 = (ਦਸ਼ਮਲਵ       *) params->wdata + ith*(1*DK + 2*DV + CACHE_ਸਤਰ_ਮਾਪ_F32); // FP32 VKQ accumulator
        ਦਸ਼ਮਲਵ       * V32   =                 (VKQ32 + 1*DV); // (temporary) FP32 V buffer
        ggml_fp16_t * VKQ16 = (ggml_fp16_t *) (VKQ32 + 1*DV); // (temporary) FP16 VKQ accumulator
        ggml_fp16_t * Q_q   = (ggml_fp16_t *) (VKQ32 + 2*DV); // (temporary) buffer for Q converted to quantized/FP16

        ਜੇ (v->type == GGML_TYPE_F16) {
            memset(VKQ16, 0, DV*ਆਕਾਰ(ggml_fp16_t));
        } ਨਹੀਂਤਾਂ {
            memset(VKQ32, 0, DV*ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        }

        ਅਚਲ ggml_fp16_t * mp = ਨਕਾਬ ? (ggml_fp16_t *)((ਅੱਖਰ *) ਨਕਾਬ->ਅੰਕੜੇ + iq1*ਨਕਾਬ->nb[1] + (iq2%ਨਕਾਬ->ne[2])*ਨਕਾਬ->nb[2] + (iq3%ਨਕਾਬ->ne[3])*ਨਕਾਬ->nb[3]) : ਨਿਰਾ;

        // k indices
        ਅਚਲ ਪੂਰਨ ik3 = iq3 / rk3;
        ਅਚਲ ਪੂਰਨ ik2 = iq2 / rk2;

        // v indices
        ਅਚਲ ਪੂਰਨ iv3 = iq3 / rv3;
        ਅਚਲ ਪੂਰਨ iv2 = iq2 / rv2;

        ਅਚਲ ਦਸ਼ਮਲਵ * pq = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) q->ਅੰਕੜੇ + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3));
        q_ਮੰਜ਼ਿਲ_vec_dot(pq, Q_q, DK);

        // online softmax / attention
        // loop over n_kv and n_head_kv
        // ref: https://arxiv.org/pdf/2112.05682.pdf

        ਲਈ (int64_t ic = ic_ਸ਼ੁਰੂ; ic < ic_ਸਿਰਾ; ++ic) {
            ਅਚਲ ਦਸ਼ਮਲਵ mv = mp ? ਢਲਾਨ*GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(mp[ic]) : 0.0f;
            ਜੇ (mv == -INFINITY) {
                ਜਾਰੀ;
            }

            ਦਸ਼ਮਲਵ s; // KQ value

            ਅਚਲ ਅੱਖਰ * k_ਅੰਕੜੇ = (ਅਚਲ ਅੱਖਰ *) k->ਅੰਕੜੇ + ( ic*nbk1 + ik2*nbk2 + ik3*nbk3);
            kq_vec_dot(DK, &s, 0, k_ਅੰਕੜੇ, 0, Q_q, 0, 1);

            s = s*scale; // scale KQ value

            ਜੇ (logit_softcap != 0.0f) {
                s = logit_softcap*tanhf(s);
            }

            s += mv; // apply mask

            ਅਚਲ ਦਸ਼ਮਲਵ Mold = M;

            ਦਸ਼ਮਲਵ ms = 1.0f; // upon new higher max val, scale VKQ and KQ sum with this value
            ਦਸ਼ਮਲਵ vs = 1.0f; // post-softmax KQ value, expf(s - M)

            ਅਚਲ ਅੱਖਰ * v_ਅੰਕੜੇ = ((ਅਚਲ ਅੱਖਰ *) v->ਅੰਕੜੇ + (ic*nbv1 + iv2*nbv2 + iv3*nbv3));

            ਜੇ (v->type == GGML_TYPE_F16) {
                ਜੇ (s > M) {
                    // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
                    M = s;
                    ms = expf(Mold - M);

                    // V = V*expf(Mold - M)
                    ggml_vec_scale_f16(DV, VKQ16, ms);
                } ਨਹੀਂਤਾਂ {
                    // no new maximum, ms == 1.0f, vs != 1.0f
                    vs = expf(s - M);
                }

                // V += v*expf(s - M)
                ggml_vec_mad_f16(DV, VKQ16, (ਅਚਲ ggml_fp16_t *) v_ਅੰਕੜੇ, vs);
            } ਨਹੀਂਤਾਂ {
                ਜੇ (s > M) {
                    // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
                    M = s;
                    ms = expf(Mold - M);

                    // V = V*expf(Mold - M)
                    ggml_vec_scale_f32(DV, VKQ32, ms);
                } ਨਹੀਂਤਾਂ {
                    // no new maximum, ms == 1.0f, vs != 1.0f
                    vs = expf(s - M);
                }

                // V += v*expf(s - M)
                ਜੇ (v_ਮੰਜ਼ਿਲ_float) {
                    v_ਮੰਜ਼ਿਲ_float(v_ਅੰਕੜੇ, V32, DV);
                    ggml_vec_mad_f32(DV, VKQ32, V32, vs);
                } ਨਹੀਂਤਾਂ {
                    // V is F32
                    ggml_vec_mad_f32(DV, VKQ32, (ਅਚਲ ਦਸ਼ਮਲਵ *) v_ਅੰਕੜੇ, vs);
                }
            }

            S = S*ms + vs; // scale and increment sum with partial sum
        }

        ਜੇ (v->type == GGML_TYPE_F16) {
            ਲਈ (int64_t d = 0; d < DV; ++d) {
                VKQ32[d] = GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(VKQ16[d]);
            }
        }

        // sinks - apply only on the first kv-chunk
        ਜੇ (sinks && ic_ਸ਼ੁਰੂ == 0) {
            ਅਚਲ ਦਸ਼ਮਲਵ s = ((ਦਸ਼ਮਲਵ *)((ਅੱਖਰ *) sinks->ਅੰਕੜੇ))[h];

            ਦਸ਼ਮਲਵ ms = 1.0f;
            ਦਸ਼ਮਲਵ vs = 1.0f;

            ਜੇ (s > M) {
                ms = expf(M - s);
                M = s;
                ggml_vec_scale_f32(DV, VKQ32, ms);
            } ਨਹੀਂਤਾਂ {
                vs = expf(s - M);
            }

            S = S*ms + vs;
        }

        ਜੇ (write_partials) {
            // Write M, S, VKQ to partials for later reduction
            // partials layout: [M, S, VKQ[DV]] per query head
            ਦਸ਼ਮਲਵ * partial = partials + ir * partial_stride;
            partial[0] = M;
            partial[1] = S;
            ਸਮ੍ਰਿਤੀ_ਨਕਲ(partial + 2, VKQ32, DV * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        } ਨਹੀਂਤਾਂ {
            // V /= S
            ਅਚਲ ਦਸ਼ਮਲਵ S_inv = S == 0.0f ? 0.0f : 1.0f/S;
            ggml_vec_scale_f32(DV, VKQ32, S_inv);

            // dst indices
            ਅਚਲ ਪੂਰਨ i1 = iq1;
            ਅਚਲ ਪੂਰਨ i2 = iq2;
            ਅਚਲ ਪੂਰਨ i3 = iq3;

            // permute(0, 2, 1, 3)
            ਸਮ੍ਰਿਤੀ_ਨਕਲ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + (i3*ne2*ne1 + i2 + i1*ne1)*nb1, VKQ32, nb1);
        }
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_flash_attn_ext_tiled(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst,
        ਪੂਰਨ ir0, ਪੂਰਨ ir1) {
    ਅਚਲ ggml_tensor * q     = dst->src[0];
    ਅਚਲ ggml_tensor * k     = dst->src[1];
    ਅਚਲ ggml_tensor * v     = dst->src[2];
    ਅਚਲ ggml_tensor * ਨਕਾਬ  = dst->src[3];
    ਅਚਲ ggml_tensor * sinks = dst->src[4];

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    ਅਚਲ int64_t DK = nek0;
    ਅਚਲ int64_t DV = nev0;
    ਅਚਲ int64_t N  = neq1;

    GGML_ASSERT(ne0 == DV);
    GGML_ASSERT(ne2 == N);

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_ਮਾਪ(q->type));
    GGML_ASSERT(nbk0 == ggml_type_ਮਾਪ(k->type));
    GGML_ASSERT(nbv0 == ggml_type_ਮਾਪ(v->type));

    GGML_ASSERT(neq0 == DK);
    GGML_ASSERT(nek0 == DK);
    GGML_ASSERT(nev0 == DV);

    GGML_ASSERT(neq1 == N);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(k->type == v->type);
    ਅਚਲ ggml_type kv_type = k->type;


    // broadcast factors
    ਅਚਲ int64_t rk2 = neq2/nek2;
    ਅਚਲ int64_t rk3 = neq3/nek3;

    ਅਚਲ int64_t rv2 = neq2/nev2;
    ਅਚਲ int64_t rv3 = neq3/nev3;

    ਦਸ਼ਮਲਵ scale         = 1.0f;
    ਦਸ਼ਮਲਵ max_bias      = 0.0f;
    ਦਸ਼ਮਲਵ logit_softcap = 0.0f;

    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&scale,         (ਦਸ਼ਮਲਵ *) dst->op_params + 0, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&max_bias,      (ਦਸ਼ਮਲਵ *) dst->op_params + 1, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&logit_softcap, (ਦਸ਼ਮਲਵ *) dst->op_params + 2, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਜੇ (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    ਅਚਲ uint32_t n_ਸਿਰ      = neq2;
    ਅਚਲ uint32_t n_ਸਿਰ_log2 = 1u << (uint32_t) floor(log2(n_ਸਿਰ));

    ਅਚਲ ਦਸ਼ਮਲਵ m0 = powf(2.0f, -(max_bias       ) / n_ਸਿਰ_log2);
    ਅਚਲ ਦਸ਼ਮਲਵ m1 = powf(2.0f, -(max_bias / 2.0f) / n_ਸਿਰ_log2);

    ਪੂਰਨ ith = params->ith;

    ਸਥਿਰ constexpr ਪੂਰਨ Q_TILE_SZ  = ggml_fa_tile_config::Q;
    ਸਥਿਰ constexpr ਪੂਰਨ KV_TILE_SZ = ggml_fa_tile_config::KV;

    ਪੂਰਨ ir = ir0;
    ਜਦੋਂ (ir < ir1) {
        // q indices for the start of this tile
        ਅਚਲ ਪੂਰਨ iq3 = ir/(neq2*neq1);
        ਅਚਲ ਪੂਰਨ iq2 = (ir - iq3*neq2*neq1)/neq1;
        ਅਚਲ ਪੂਰਨ iq1 = (ir - iq3*neq2*neq1 - iq2*neq1);

        // Number of valid rows in this tile:
        // - limited by tile size (Q_TILE_SZ)
        // - limited by chunk boundary (ir1 - ir)
        // - limited by head boundary (neq1 - iq1) to avoid crossing into next head
        ਅਚਲ ਪੂਰਨ tile_ਪੰਕਤੀਆਂ = MIN(Q_TILE_SZ, MIN((ਪੂਰਨ)(ir1 - ir), (ਪੂਰਨ)(neq1 - iq1)));
        GGML_ASSERT(tile_ਪੰਕਤੀਆਂ > 0);

        ਅਚਲ uint32_t h = iq2; // head index
        ਅਚਲ ਦਸ਼ਮਲਵ ਢਲਾਨ = (max_bias > 0.0f) ? h < n_ਸਿਰ_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_ਸਿਰ_log2) + 1) : 1.0f;

        ਦਸ਼ਮਲਵ S[Q_TILE_SZ];
        ਦਸ਼ਮਲਵ M[Q_TILE_SZ];

        ਲਈ (ਪੂਰਨ i = 0 ; i < Q_TILE_SZ; ++i) {
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
        ਦਸ਼ਮਲਵ * ਅਧਾਰ  = (ਦਸ਼ਮਲਵ *) params->wdata + ith*(Q_TILE_SZ*DK + 2*Q_TILE_SZ*KV_TILE_SZ + Q_TILE_SZ*DV + KV_TILE_SZ*DV + KV_TILE_SZ*DK + CACHE_ਸਤਰ_ਮਾਪ_F32);

        ਖਾਲੀ  * Q_q    = ਅਧਾਰ;
        ਦਸ਼ਮਲਵ * KQ     = (ਦਸ਼ਮਲਵ *)((ਅੱਖਰ *)ਅਧਾਰ + Q_TILE_SZ * DK * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        ਦਸ਼ਮਲਵ * mask32 = KQ + Q_TILE_SZ * KV_TILE_SZ;
        ਦਸ਼ਮਲਵ * VKQ32  = mask32 + Q_TILE_SZ * KV_TILE_SZ;
        ਦਸ਼ਮਲਵ * V32    = VKQ32 + Q_TILE_SZ * DV;
        ਦਸ਼ਮਲਵ * K_f32  = V32 + KV_TILE_SZ * DV;

        memset(VKQ32, 0, Q_TILE_SZ * DV * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        memset(mask32, 0, Q_TILE_SZ * KV_TILE_SZ * ਆਕਾਰ(ਦਸ਼ਮਲਵ));

        // k indices
        ਅਚਲ ਪੂਰਨ ik3 = iq3 / rk3;
        ਅਚਲ ਪੂਰਨ ik2 = iq2 / rk2;

        // v indices
        ਅਚਲ ਪੂਰਨ iv3 = iq3 / rv3;
        ਅਚਲ ਪੂਰਨ iv2 = iq2 / rv2;

        {
            ਦਸ਼ਮਲਵ * Q_f32 = (ਦਸ਼ਮਲਵ *)Q_q;
            ਲਈ (ਪੂਰਨ tq = 0; tq < tile_ਪੰਕਤੀਆਂ; tq++) {
                ਅਚਲ ਦਸ਼ਮਲਵ * pq = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) q->ਅੰਕੜੇ + ((iq1 + tq)*nbq1 + iq2*nbq2 + iq3*nbq3));
                ਸਮ੍ਰਿਤੀ_ਨਕਲ(Q_f32 + tq * DK, pq, DK * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
            }
            ਲਈ (ਪੂਰਨ tq = tile_ਪੰਕਤੀਆਂ; tq < Q_TILE_SZ; tq++) {
                memset(Q_f32 + tq * DK, 0, DK * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
            }
        }

        memset(K_f32, 0, DK * KV_TILE_SZ * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        memset(V32,   0, KV_TILE_SZ * DV * ਆਕਾਰ(ਦਸ਼ਮਲਵ));

        ਲਈ (int64_t ic = 0; ic < nek1; ic += KV_TILE_SZ) {
            ਅਚਲ ਪੂਰਨ kv_tile = (ਪੂਰਨ)std::min((int64_t)KV_TILE_SZ, nek1 - ic);

            // skip the tile entirely if all the masks are -inf
            ਜੇ (ਨਕਾਬ) {
                ਤਾਰਕਿਕ can_skip = ਸੱਚ;
                ਲਈ (ਪੂਰਨ tq = 0; tq < tile_ਪੰਕਤੀਆਂ; tq++) {
                    ਅਚਲ ggml_fp16_t * mp_ਪੰਕਤੀ = (ਅਚਲ ggml_fp16_t *)((ਅਚਲ ਅੱਖਰ *) ਨਕਾਬ->ਅੰਕੜੇ + (iq1 + tq)*ਨਕਾਬ->nb[1] + (iq2%ਨਕਾਬ->ne[2])*ਨਕਾਬ->nb[2] + (iq3%ਨਕਾਬ->ne[3])*ਨਕਾਬ->nb[3]);
                    ਲਈ (ਪੂਰਨ tk = 0; tk < kv_tile; tk++) {
                        mask32[tq * KV_TILE_SZ + tk] = ਢਲਾਨ * GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(mp_ਪੰਕਤੀ[ic + tk]);
                        ਜੇ (mask32[tq * KV_TILE_SZ + tk] != -INFINITY) {
                            can_skip = ਝੂਠ;
                        }
                    }
                    // Pad remaining mask entries with -inf
                    ਲਈ (ਪੂਰਨ tk = kv_tile; tk < KV_TILE_SZ; tk++) {
                        mask32[tq * KV_TILE_SZ + tk] = -INFINITY;
                    }
                }

                ਜੇ (can_skip) {
                    ਜਾਰੀ;
                }
            }

            // Pack K tile transposed: K_f32[dk][kv] so KV_TILE is contiguous (SIMD dim)
            // Zero-pad the last tile so the GEMM always operates on KV_TILE_SZ columns
            ਲਈ (ਪੂਰਨ tk = 0; tk < kv_tile; tk++) {
                ਅਚਲ ਅੱਖਰ * k_ਅੰਕੜੇ = (ਅਚਲ ਅੱਖਰ *)k->ਅੰਕੜੇ + (ic + tk)*nbk1 + ik2*nbk2 + ik3*nbk3;
                ਜੇ (kv_type == GGML_TYPE_F16) {
                    ਅਚਲ ggml_fp16_t * k_f16 = (ਅਚਲ ggml_fp16_t *)k_ਅੰਕੜੇ;
                    ਲਈ (int64_t dk = 0; dk < DK; dk++) {
                        K_f32[dk * KV_TILE_SZ + tk] = GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(k_f16[dk]);
                    }
                } ਨਹੀਂਤਾਂ {
                    ਅਚਲ ਦਸ਼ਮਲਵ * k_f32_src = (ਅਚਲ ਦਸ਼ਮਲਵ *)k_ਅੰਕੜੇ;
                    ਲਈ (int64_t dk = 0; dk < DK; dk++) {
                        K_f32[dk * KV_TILE_SZ + tk] = k_f32_src[dk];
                    }
                }
            }
            memset(KQ, 0, Q_TILE_SZ * KV_TILE_SZ * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
            simd_gemm(KQ, (ਅਚਲ ਦਸ਼ਮਲਵ *)Q_q, K_f32, Q_TILE_SZ, DK, KV_TILE_SZ);
            ggml_vec_scale_f32(Q_TILE_SZ * KV_TILE_SZ, KQ, scale);

            ਜੇ (logit_softcap != 0.0f) {
                ggml_vec_tanh_f32(Q_TILE_SZ * KV_TILE_SZ, KQ, KQ);
                ggml_vec_scale_f32(Q_TILE_SZ * KV_TILE_SZ, KQ, logit_softcap);
            }

            // Set padded KQ entries to -inf so softmax gives them zero weight
            ਜੇ (kv_tile < KV_TILE_SZ) {
                ਲਈ (ਪੂਰਨ tq = 0; tq < Q_TILE_SZ; tq++) {
                    ਲਈ (ਪੂਰਨ tk = kv_tile; tk < KV_TILE_SZ; tk++) {
                        KQ[tq * KV_TILE_SZ + tk] = -INFINITY;
                    }
                }
            }

            ਜੇ (ਨਕਾਬ) {
                ggml_vec_ਜੋੜੋ_f32(tile_ਪੰਕਤੀਆਂ * KV_TILE_SZ, KQ, KQ, mask32);
            }

            ਤਾਰਕਿਕ skip[Q_TILE_SZ] = {};

            ਲਈ (ਪੂਰਨ tq = 0; tq < Q_TILE_SZ; tq++) {
                ਦਸ਼ਮਲਵ * kq_ਪੰਕਤੀ = KQ + tq * KV_TILE_SZ;

                ਦਸ਼ਮਲਵ tile_max;
                ggml_vec_max_f32(KV_TILE_SZ, &tile_max, kq_ਪੰਕਤੀ);

                ਜੇ (tile_max == -INFINITY) {
                    skip[tq] = ਸੱਚ;
                    ਜਾਰੀ;
                }

                ਅਚਲ ਦਸ਼ਮਲਵ Mold = M[tq];
                ਅਚਲ ਦਸ਼ਮਲਵ Mnew = fmaxf(Mold, tile_max);

                ਜੇ (Mnew > Mold) {
                    ਅਚਲ ਦਸ਼ਮਲਵ ms = expf(Mold - Mnew);
                    ggml_vec_scale_f32(DV, VKQ32 + tq * DV, ms);
                    S[tq] *= ms;
                }
                M[tq] = Mnew;


                S[tq] += ggml_vec_soft_max_f32(KV_TILE_SZ, kq_ਪੰਕਤੀ, kq_ਪੰਕਤੀ, Mnew);
            }

            // V accumulation: VKQ32 += softmax(KQ) * V
            // Pack V tile to contiguous F32, zero-padded
            ਲਈ (ਪੂਰਨ tk = 0; tk < kv_tile; tk++) {
                ਅਚਲ ਅੱਖਰ * v_ਅੰਕੜੇ = (ਅਚਲ ਅੱਖਰ *)v->ਅੰਕੜੇ + (ic + tk)*nbv1 + iv2*nbv2 + iv3*nbv3;
                ਜੇ (kv_type == GGML_TYPE_F16) {
                    ggml_cpu_fp16_ਮੰਜ਼ਿਲ_fp32((ਅਚਲ ggml_fp16_t *)v_ਅੰਕੜੇ, V32 + tk * DV, DV);
                } ਨਹੀਂਤਾਂ {
                    ਸਮ੍ਰਿਤੀ_ਨਕਲ(V32 + tk * DV, v_ਅੰਕੜੇ, DV * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                }
            }
            ਲਈ (ਪੂਰਨ tq = 0; tq < Q_TILE_SZ; tq++) {
                ਜੇ (skip[tq]) {
                    memset(KQ + tq * KV_TILE_SZ, 0, KV_TILE_SZ * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                }
            }
            simd_gemm(VKQ32, KQ, V32, Q_TILE_SZ, KV_TILE_SZ, DV);
        }

        // sinks (apply only to valid rows in the tile)
        ਜੇ (sinks) {
            ਅਚਲ ਦਸ਼ਮਲਵ s = ((ਦਸ਼ਮਲਵ *)((ਅੱਖਰ *) sinks->ਅੰਕੜੇ))[h];

            ਲਈ (ਪੂਰਨ tq = 0; tq < tile_ਪੰਕਤੀਆਂ; tq++) {
                ਦਸ਼ਮਲਵ ms = 1.0f;
                ਦਸ਼ਮਲਵ vs = 1.0f;

                ਜੇ (s > M[tq]) {
                    ms = expf(M[tq] - s);
                    ggml_vec_scale_f32(DV, VKQ32 + tq * DV, ms);
                } ਨਹੀਂਤਾਂ {
                    vs = expf(s - M[tq]);
                }

                S[tq] = S[tq] * ms + vs;
            }
        }

        ਲਈ (ਪੂਰਨ tq = 0; tq < tile_ਪੰਕਤੀਆਂ; tq++) {
            // V /= S
            ਅਚਲ ਦਸ਼ਮਲਵ S_inv = S[tq] == 0.0f ? 0.0f : 1.0f / S[tq];
            ggml_vec_scale_f32(DV, VKQ32 + tq * DV, S_inv);

            // dst indices
            ਅਚਲ ਪੂਰਨ i1 = iq1 + tq;
            ਅਚਲ ਪੂਰਨ i2 = iq2;
            ਅਚਲ ਪੂਰਨ i3 = iq3;

            // permute(0, 2, 1, 3)
            ਸਮ੍ਰਿਤੀ_ਨਕਲ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + (i3*ne2*ne1 + i2 + i1*ne1)*nb1, VKQ32 + tq * DV, nb1);
        }

        ir += tile_ਪੰਕਤੀਆਂ;
    }
}

// Reduction function: combines partial results across KV chunks
// Partials layout in wdata: [n_q_heads][n_chunks][2 + DV]
ਸਥਿਰ ਖਾਲੀ ggml_flash_attn_ext_reduce_partials(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst,
        ਅਚਲ int64_t n_chunks,
        ਅਚਲ int64_t chunk_ਮਾਪ) {

    ਅਚਲ ggml_tensor * q = dst->src[0];
    ਅਚਲ ggml_tensor * k = dst->src[1];
    ਅਚਲ ggml_tensor * v = dst->src[2];

    ਅਚਲ int64_t DK        = k->ne[0];
    ਅਚਲ int64_t DV        = v->ne[0];
    ਅਚਲ int64_t nek1      = k->ne[1];
    ਅਚਲ int64_t n_q_heads = q->ne[2];

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ int64_t wdata_ਪ੍ਰਤੀ_thread = DK + 2*DV + CACHE_ਸਤਰ_ਮਾਪ_F32;
    ਦਸ਼ਮਲਵ *       thread_wdata     = (ਦਸ਼ਮਲਵ *) params->wdata + ith * wdata_ਪ੍ਰਤੀ_thread;

    ਅਚਲ int64_t partials_ਵਿੱਥ  = nth * (DK + 2*DV + CACHE_ਸਤਰ_ਮਾਪ_F32);
    ਅਚਲ int64_t partial_ਮਾਪ     = 2 + DV;
    ਅਚਲ ਦਸ਼ਮਲਵ * partials_ਅਧਾਰ    = (ਅਚਲ ਦਸ਼ਮਲਵ *) params->wdata + partials_ਵਿੱਥ;

    // Output layout
    ਅਚਲ int64_t ne1 = dst->ne[1];
    ਅਚਲ int64_t ne2 = dst->ne[2];
    ਅਚਲ size_t  nb1 = dst->nb[1];

    // Each thread reduces a subset of query heads
    ਲਈ (int64_t q_ਸਿਰ = ith; q_ਸਿਰ < n_q_heads; q_ਸਿਰ += nth) {
        ਦਸ਼ਮਲਵ   M_final   = -INFINITY;
        ਦਸ਼ਮਲਵ   S_final   = 0.0f;
        ਦਸ਼ਮਲਵ * VKQ_final = thread_wdata;
        memset(VKQ_final, 0, DV * ਆਕਾਰ(ਦਸ਼ਮਲਵ));

        // Combine partials from all chunks
        ਲਈ (int64_t chunk_idx = 0; chunk_idx < n_chunks; ++chunk_idx) {
            ਅਚਲ int64_t ic_ਸ਼ੁਰੂ = chunk_idx * chunk_ਮਾਪ;
            ਜੇ (ic_ਸ਼ੁਰੂ >= nek1) ਜਾਰੀ;

            ਅਚਲ ਦਸ਼ਮਲਵ * partial   = partials_ਅਧਾਰ + (q_ਸਿਰ * n_chunks + chunk_idx) * partial_ਮਾਪ;
            ਅਚਲ ਦਸ਼ਮਲਵ   M_chunk   = partial[0];
            ਅਚਲ ਦਸ਼ਮਲਵ   S_chunk   = partial[1];
            ਅਚਲ ਦਸ਼ਮਲਵ * VKQ_chunk = partial + 2;

            ਜੇ (S_chunk == 0.0f) ਜਾਰੀ;

            ਅਚਲ ਦਸ਼ਮਲਵ M_new     = fmaxf(M_final, M_chunk);
            ਅਚਲ ਦਸ਼ਮਲਵ scale_old = expf(M_final - M_new);
            ਅਚਲ ਦਸ਼ਮਲਵ scale_new = expf(M_chunk - M_new);

            ਲਈ (int64_t d = 0; d < DV; ++d) {
                VKQ_final[d] = VKQ_final[d] * scale_old + VKQ_chunk[d] * scale_new;
            }
            S_final = S_final * scale_old + S_chunk * scale_new;
            M_final = M_new;
        }

        // Normalize and write to output
        ਜੇ (S_final != 0.0f) {
            ਅਚਲ ਦਸ਼ਮਲਵ S_inv = 1.0f / S_final;
            ggml_vec_scale_f32(DV, VKQ_final, S_inv);
        }
        // iq1=0, iq3=0 for decode
        ਸਮ੍ਰਿਤੀ_ਨਕਲ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + (0*ne2*ne1 + q_ਸਿਰ + 0*ne1)*nb1, VKQ_final, nb1);
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_flash_attn_ext_f16(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * q     = dst->src[0];
    ਅਚਲ ggml_tensor * k     = dst->src[1];
    ਅਚਲ ggml_tensor * v     = dst->src[2];

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    ਅਚਲ int64_t DK = nek0;
    ਅਚਲ int64_t DV = nev0;
    ਅਚਲ int64_t N  = neq1;


    GGML_ASSERT(ne0 == DV);
    GGML_ASSERT(ne2 == N);

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_ਮਾਪ(q->type));
    GGML_ASSERT(nbk0 == ggml_type_ਮਾਪ(k->type));
    GGML_ASSERT(nbv0 == ggml_type_ਮਾਪ(v->type));

    GGML_ASSERT(neq0 == DK);
    GGML_ASSERT(nek0 == DK);
    GGML_ASSERT(nev0 == DV);

    GGML_ASSERT(neq1 == N);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    // When use_ref is set, force the vec-only reference implementation (no tiling, no KV-chunking)
    ਅਚਲ ਤਾਰਕਿਕ use_ref = params->use_ref;

    ਅਚਲ ਤਾਰਕਿਕ kv_is_f32_or_f16 = (k->type == GGML_TYPE_F32 || k->type == GGML_TYPE_F16);
    ਅਚਲ ਤਾਰਕਿਕ use_ਵੰਡੋ_kv_ਰਸਤਾ = !use_ref && (neq1 == 1 && neq3 == 1) && kv_is_f32_or_f16 && (k->type == v->type) && q->type == GGML_TYPE_F32 && nek1 >= 512;

    ਜੇ (use_ਵੰਡੋ_kv_ਰਸਤਾ) {
        ਅਚਲ int64_t chunk_ਮਾਪ = (nek1 + nth - 1) / nth;

        // Partials buffer layout: [q_head][kv_chunk][M, S, VKQ]
        ਅਚਲ int64_t partial_ਮਾਪ  = 2 + DV;
        ਦਸ਼ਮਲਵ *       partials_ਅਧਾਰ = (ਦਸ਼ਮਲਵ *) params->wdata + nth * (DK + 2*DV + CACHE_ਸਤਰ_ਮਾਪ_F32);

        ਅਚਲ int64_t ic_ਸ਼ੁਰੂ = ith * chunk_ਮਾਪ;
        ਅਚਲ int64_t ic_ਸਿਰਾ   = std::min(ic_ਸ਼ੁਰੂ + chunk_ਮਾਪ, nek1);

        ਅਚਲ int64_t partial_stride = nth * partial_ਮਾਪ;
        ਦਸ਼ਮਲਵ *       chunk_partials = partials_ਅਧਾਰ + ith * partial_ਮਾਪ;

        ਜੇ (ic_ਸ਼ੁਰੂ < nek1) {
            ਲਈ (int64_t q_ਸਿਰ = 0; q_ਸਿਰ < neq2; q_ਸਿਰ++) {
                ggml_compute_forward_flash_attn_ext_f16_one_chunk(
                    params, dst, q_ਸਿਰ, q_ਸਿਰ + 1, ic_ਸ਼ੁਰੂ, ic_ਸਿਰਾ,
                    chunk_partials, partial_stride);
            }
        } ਨਹੀਂਤਾਂ {
            ਲਈ (int64_t q_ਸਿਰ = 0; q_ਸਿਰ < neq2; q_ਸਿਰ++) {
                ਦਸ਼ਮਲਵ * q_partials = chunk_partials + q_ਸਿਰ * partial_stride;
                q_partials[0] = -INFINITY;  // M
                q_partials[1] = 0.0f;       // S
            }
        }

        ggml_barrier(params->threadpool);
        ggml_flash_attn_ext_reduce_partials(params, dst, nth, chunk_ਮਾਪ);
    } ਨਹੀਂਤਾਂ {

        // total rows in q
        ਅਚਲ int64_t nr = neq1*neq2*neq3;

        // disable for NUMA
        ਅਚਲ ਤਾਰਕਿਕ disable_chunking = ggml_is_numa();

        // 4x chunks per thread
        ਪੂਰਨ nth_scaled = nth * 4;
        int64_t chunk_ਮਾਪ = (nr + nth_scaled - 1) / nth_scaled;
        int64_t nchunk     = (nr + chunk_ਮਾਪ - 1) / chunk_ਮਾਪ;

        ਜੇ (nth == 1 || nchunk < nth || disable_chunking) {
            nchunk = nth;
        }

        ਜੇ (ith == 0) {
            ggml_threadpool_chunk_ਸੈੱਟ(params->threadpool, nth);
        }

        ggml_barrier(params->threadpool);

        ਅਚਲ int64_t dr = (nr + nchunk - 1) / nchunk;

        ਸਥਿਰ constexpr int64_t Q_TILE_SZ  = ggml_fa_tile_config::Q;
        ਤਾਰਕਿਕ use_tiled = !use_ref &&
                               (q->type == GGML_TYPE_F32 &&
                                kv_is_f32_or_f16 &&
                                k->type == v->type &&
                                neq1 >= Q_TILE_SZ);
#ਜੇ defined(GGML_SIMD) && !defined(__x86_64__) && !defined(_M_X64)
#ਜੇ defined(__ARM_FEATURE_SVE)
        ਅਚਲ int64_t f32_epr = svcntw();
#ਨਹੀਂਤਾਂ
        ਅਚਲ int64_t f32_epr = GGML_F32_EPR;
#ਜੇ_ਅੰਤ
        use_tiled &= (DV % f32_epr == 0);
#ਜੇ_ਅੰਤ
        ਪੂਰਨ current_chunk = ith;

        ਜਦੋਂ (current_chunk < nchunk) {
            ਅਚਲ int64_t ir0 = dr * current_chunk;
            ਅਚਲ int64_t ir1 = MIN(ir0 + dr, nr);

            ਜੇ (use_tiled) {
                ggml_compute_forward_flash_attn_ext_tiled(params, dst, ir0, ir1);
            } ਨਹੀਂਤਾਂ {
                ggml_compute_forward_flash_attn_ext_f16_one_chunk(params, dst, ir0, ir1, 0, nek1, ਖਾਲੀ_ਸੂਚਕ, 0);
            }

            current_chunk = ggml_threadpool_chunk_ਜੋੜੋ(params->threadpool, 1);
        }
    }
}

ਖਾਲੀ ggml_compute_forward_flash_attn_ext(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {
    ਚੋਣ (dst->op_params[3]) {
        ਹਾਲਤ GGML_PREC_DEFAULT:
        ਹਾਲਤ GGML_PREC_F32:
            {
                // uses F32 accumulators
                ggml_compute_forward_flash_attn_ext_f16(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_flash_attn_back

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_flash_attn_ਪਿੱਠ_f32(
        ਅਚਲ ggml_compute_params * params,
        ਅਚਲ ਤਾਰਕਿਕ masked,
              ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * q = dst->src[0];
    ਅਚਲ ggml_tensor * k = dst->src[1];
    ਅਚਲ ggml_tensor * v = dst->src[2];
    ਅਚਲ ggml_tensor * d = dst->src[3];

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

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ int64_t D = neq0;
    ਅਚਲ int64_t N = neq1;
    ਅਚਲ int64_t P = nek1 - N;
    ਅਚਲ int64_t M = P + N;

    ਅਚਲ ਪੂਰਨ Mup  = ggml_ਉੱਪਰ(M, GGML_SOFT_MAX_UNROLL);
    ਅਚਲ ਪੂਰਨ mxDM = MAX(D, Mup);

    // GGML_ASSERT(ne0 == D);
    // GGML_ASSERT(ne1 == N);
    GGML_ASSERT(P >= 0);

    GGML_ASSERT(nbq0 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    GGML_ASSERT(nbk0 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    GGML_ASSERT(nbv0 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    GGML_ASSERT(neq0 == D);
    GGML_ASSERT(nek0 == D);
    GGML_ASSERT(nev1 == D);
    GGML_ASSERT(ned0 == D);

    GGML_ASSERT(neq1 == N);
    GGML_ASSERT(nek1 == N + P);
    GGML_ASSERT(nev1 == D);
    GGML_ASSERT(ned1 == N);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    ਜੇ (ith == 0) {
        memset(dst->ਅੰਕੜੇ, 0, nb0*ne0*ne1*ne2*ne3);
    }
    ggml_barrier(params->threadpool);

    ਅਚਲ int64_t elem_q = ggml_nelements(q);
    ਅਚਲ int64_t elem_k = ggml_nelements(k);

    ggml_type result_type = dst->type;
    GGML_ASSERT(ggml_blck_ਮਾਪ(result_type) == 1);
    ਅਚਲ size_t tsize = ggml_type_ਮਾਪ(result_type);

    ਅਚਲ size_t offs_q = 0;
    ਅਚਲ size_t offs_k = offs_q + GGML_PAD(elem_q * tsize, GGML_MEM_ALIGN);
    ਅਚਲ size_t offs_v = offs_k + GGML_PAD(elem_k * tsize, GGML_MEM_ALIGN);

    ਖਾਲੀ * grad_q = (ਅੱਖਰ *) dst->ਅੰਕੜੇ;
    ਖਾਲੀ * grad_k = (ਅੱਖਰ *) dst->ਅੰਕੜੇ + offs_k;
    ਖਾਲੀ * grad_v = (ਅੱਖਰ *) dst->ਅੰਕੜੇ + offs_v;

    ਅਚਲ size_t nbgq1 = nb0*neq0;
    ਅਚਲ size_t nbgq2 = nb0*neq0*neq1;
    ਅਚਲ size_t nbgq3 = nb0*neq0*neq1*neq2;

    ਅਚਲ size_t nbgk1 = nb0*nek0;
    ਅਚਲ size_t nbgk2 = nb0*nek0*nek1;
    ਅਚਲ size_t nbgk3 = nb0*nek0*nek1*neq2;

    ਅਚਲ size_t nbgv1 = nb0*nev0;
    ਅਚਲ size_t nbgv2 = nb0*nev0*nev1;
    ਅਚਲ size_t nbgv3 = nb0*nev0*nev1*neq2;

    // parallelize by k rows using ggml_vec_dot_f32

    // total rows in k
    ਅਚਲ ਪੂਰਨ nr = nek2*nek3;

    // rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr*ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਅਚਲ ਦਸ਼ਮਲਵ scale = 1.0f/sqrtf(D);

    //printf("P=%d N=%d D=%d ir0=%d ir1=%d scale = %f\n", P, N, D, ir0, ir1, scale);

    // how often k2 (and v2) is repeated in q2
    ਪੂਰਨ nrep = neq2/nek2;

    ਲਈ (ਪੂਰਨ ir = ir0; ir < ir1; ++ir) {
        // q indices
        ਅਚਲ ਪੂਰਨ ik3 = ir/(nek2);
        ਅਚਲ ਪੂਰਨ ik2 = ir - ik3*nek2;

        ਅਚਲ ਪੂਰਨ iq3 = ik3;
        ਅਚਲ ਪੂਰਨ id3 = ik3;
        ਅਚਲ ਪੂਰਨ iv3 = ik3;
        ਅਚਲ ਪੂਰਨ iv2 = ik2;

        ਲਈ (ਪੂਰਨ irep = 0; irep < nrep; ++irep) {
            ਅਚਲ ਪੂਰਨ iq2 = ik2 + irep*nek2;
            ਅਚਲ ਪੂਰਨ id2 = iq2;

            // (ik2 + irep*nek2) % nek2 == ik2
            ਲਈ (ਪੂਰਨ iq1 = 0; iq1 < neq1; ++iq1) {
                ਅਚਲ ਪੂਰਨ id1 = iq1;

                // not sure about CACHE_LINE_SIZE_F32..
                // - maybe it must not be multiplied by 2 and excluded from .. in SM 1*(..) offset?
                ਦਸ਼ਮਲਵ * S  = (ਦਸ਼ਮਲਵ *) params->wdata + ith*2*(mxDM + CACHE_ਸਤਰ_ਮਾਪ_F32) + 0*(mxDM+CACHE_ਸਤਰ_ਮਾਪ_F32);
                ਦਸ਼ਮਲਵ * SM = (ਦਸ਼ਮਲਵ *) params->wdata + ith*2*(mxDM + CACHE_ਸਤਰ_ਮਾਪ_F32) + 1*(mxDM+CACHE_ਸਤਰ_ਮਾਪ_F32);

                ਲਈ (ਪੂਰਨ i = M; i < Mup; ++i) {
                    S[i] = -INFINITY;
                }

                ਅਚਲ int64_t masked_begin = masked ? (P + iq1 + 1) : M;
                ਲਈ (int64_t ic = 0; ic < masked_begin; ++ic) {
                    // k indices
                    ਅਚਲ ਪੂਰਨ ik1 = ic;

                    // S indices
                    ਅਚਲ ਪੂਰਨ i1 = ik1;

                    ggml_vec_dot_f32(neq0,
                            S + i1, 0,
                            (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) k->ਅੰਕੜੇ + (ik1*nbk1 + ik2*nbk2 + ik3*nbk3)), 0,
                            (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) q->ਅੰਕੜੇ + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3)), 0, 1);
                }

                // scale
                ggml_vec_scale_f32(masked_begin, S, scale);

                ਲਈ (int64_t i = masked_begin; i < M; i++) {
                    S[i] = -INFINITY;
                }

                // softmax
                // exclude known -INF S[..] values from max and loop
                // dont forget to set their SM values to zero
                {
                    ਦਸ਼ਮਲਵ max = -INFINITY;
                    ggml_vec_max_f32(masked_begin, &max, S);

                    ggml_float ਜੋੜ = 0.0;
                    {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SOFT_MAX_ACCELERATE
                        max = -max;
                        vDSP_vsadd(SM, 1, &max, SM, 1, Mup);
                        vvexpf(SM, SM, &Mup);
                        ggml_vec_ਜੋੜ_f32(Mup, &ਜੋੜ, SM);
#ਨਹੀਂਤਾਂ
                        ਜੋੜ = ggml_vec_soft_max_f32(Mup, SM, S, max);
#ਜੇ_ਅੰਤ
                    }

                    ਪ੍ਰਤਿਜ੍ਞਾ(ਜੋੜ > 0.0);

                    ਜੋੜ = 1.0/ਜੋੜ;
                    ggml_vec_scale_f32(masked_begin, SM, ਜੋੜ);

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
                ggml_vec_ਸੈੱਟ_f32(masked_begin, S, 0);
                ਲਈ (int64_t ic = 0; ic < D; ++ic) {
                    ggml_vec_mad_f32(masked_begin,
                            S,
                             (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) v->ਅੰਕੜੇ + (          ic*nbv1  + iv2*nbv2 + iv3*nbv3)),
                            *(ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) d->ਅੰਕੜੇ + (ic*nbd0 + id1*nbd1 + id2*nbd2 + id3*nbd3)));
                }

                // S = SM * (S - dot(SM, S))
                ਦਸ਼ਮਲਵ dot_SM_gradSM = 0;
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
                ਲਈ (int64_t ic = 0; ic < masked_begin; ++ic) {
                    ggml_vec_mad_f32(D,
                            (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) grad_q  + (iq1*nbgq1 + iq2*nbgq2  + iq3*nbgq3)),
                            (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) k->ਅੰਕੜੇ + (ic*nbk1   + ik2*nbk2   + ik3*nbk3)),
                            S[ic]);
                }

                // grad[k][:D,:M,iq2,iq3] += S.T @ qcur
                // for ic:
                //  grad[k][:D,ic,iq2,iq3] += S.T[0,ic] * qcur[:D,0]
                //  grad[k][:D,ic,iq2,iq3] += S[ic]     * qcur[:D,0]
                // exclude known zero S[..] values from loop
                ਲਈ (int64_t ic = 0; ic < masked_begin; ++ic) {
                    ggml_vec_mad_f32(D,
                            (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) grad_k  + (ic*nbgk1  + ik2*nbgk2  + ik3*nbgk3)),
                            (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) q->ਅੰਕੜੇ + (iq1*nbq1  + iq2*nbq2   + iq3*nbq3)),
                            S[ic]);
                }

                // grad[v][:M,:D,iv2,iv3] += d[:D,id1,id2,id3].T       @ SM
                // for ic:
                //  grad[v][:M,ic,iv2,iv3] += d[:D,id1,id2,id3].T[0,ic] * SM[:M]
                //  grad[v][:M,ic,iv2,iv3] += d[ic,id1,id2,id3]         * SM[:M]
                // exclude known zero SM[..] values from mad
                ਲਈ (int64_t ic = 0; ic < D; ++ic) {
                    ggml_vec_mad_f32(masked_begin,
                            (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) grad_v   + (          ic*nbgv1 + iv2*nbgv2 + iv3*nbgv3)),
                            SM,
                            *(ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) d->ਅੰਕੜੇ + (ic*nbd0 + id1*nbd1 + id2*nbd2  + id3*nbd3)));
                }
            }
        }
    }
}

ਖਾਲੀ ggml_compute_forward_flash_attn_ਪਿੱਠ(
        ਅਚਲ ggml_compute_params * params,
        ਅਚਲ ਤਾਰਕਿਕ masked,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * q = dst->src[0];

    ਚੋਣ (q->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_flash_attn_ਪਿੱਠ_f32(params, masked, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_ssm_conv

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_ssm_conv_f32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * src0 = dst->src[0]; // conv_x
    ਅਚਲ ggml_tensor * src1 = dst->src[1]; // conv1d.weight

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ nc  = src1->ne[0]; // d_conv
    ਅਚਲ ਪੂਰਨ ncs = src0->ne[0]; // d_conv - 1 + n_t
    ਅਚਲ ਪੂਰਨ nr  = src0->ne[1]; // d_inner
    ਅਚਲ ਪੂਰਨ n_t =  dst->ne[1]; // tokens per sequence
    ਅਚਲ ਪੂਰਨ n_s =  dst->ne[2]; // number of sequences in the batch

    GGML_ASSERT( dst->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    GGML_ASSERT(src1->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    GGML_ASSERT(src0->nb[1] == src0->ne[0]*ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    // rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr*ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);
    ਅਚਲ ਪੂਰਨ ir  = ir1 - ir0;

    ਲਈ (ਪੂਰਨ i3 = 0; i3 < n_s; ++i3) {
        ਲਈ (ਪੂਰਨ i2 = 0; i2 < n_t; ++i2) {
            // {d_conv - 1 + n_t, d_inner, n_seqs}
            // sliding window
            ਅਚਲ ਦਸ਼ਮਲਵ * s = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src0->ਅੰਕੜੇ + ir0*(src0->nb[1]) + i2*(src0->nb[0]) + i3*(src0->nb[2])); // {d_conv, d_inner, n_s}
            ਅਚਲ ਦਸ਼ਮਲਵ * c = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src1->ਅੰਕੜੇ + ir0*(src1->nb[1])); // {d_conv, d_inner}
            ਦਸ਼ਮਲਵ * x = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + ir0*(dst->nb[0]) + i2*(dst->nb[1]) + i3*(dst->nb[2])); // {d_inner, n_t, n_s}

            // TODO: transpose the output for smaller strides for big batches?
            // d_inner
            ਲਈ (ਪੂਰਨ i1 = 0; i1 < ir; ++i1) {
                // rowwise dot product
                // NOTE: not using ggml_vec_dot_f32, because its sum is in double precision
                ਦਸ਼ਮਲਵ sumf = 0.0f;

                // d_conv
                ਲਈ (ਪੂਰਨ i0 = 0; i0 < nc; ++i0) {
                    sumf += s[i0 + i1*ncs] * c[i0 + i1*nc];
                }
                x[i1] = sumf;
            }
        }
    }
}

ਖਾਲੀ ggml_compute_forward_ssm_conv(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {
    ਚੋਣ (dst->src[0]->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_ssm_conv_f32(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_ssm_scan

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_ssm_ਜਾਂਚ_f32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * src0 = dst->src[0]; // s  {d_state, dim, n_head, n_seqs+}
    ਅਚਲ ggml_tensor * src1 = dst->src[1]; // x  {dim, n_head, n_seq_tokens, n_seqs}
    ਅਚਲ ggml_tensor * src2 = dst->src[2]; // dt {n_head, n_seq_tokens, n_seqs}
    ਅਚਲ ggml_tensor * src3 = dst->src[3]; // A  {d_state, n_head} or {1, n_head}
    ਅਚਲ ggml_tensor * src4 = dst->src[4]; // B  {d_state, n_group, n_seq_tokens, n_seqs}
    ਅਚਲ ggml_tensor * src5 = dst->src[5]; // C  {d_state, n_group, n_seq_tokens, n_seqs}
    ਅਚਲ ggml_tensor * src6 = dst->src[6]; // ids {n_seqs}

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ int64_t nc = src0->ne[0]; // d_state
    ਅਚਲ int64_t nr = src0->ne[1]; // dim
    ਅਚਲ int64_t nh = src1->ne[1]; // n_head
    ਅਚਲ int64_t ng = src4->ne[1];
    ਅਚਲ int64_t nt = src1->ne[2]; // number of tokens per sequence
    ਅਚਲ int64_t ns = src1->ne[3]; // number of sequences in the batch
    ਅਚਲ int64_t K  = ggml_get_op_params_i32(dst, 0);

    // can't use ggml_nbytes because src1 is not necessarily contiguous
    ਅਚਲ int64_t s_off = ggml_nelements(src1) * ggml_element_ਮਾਪ(src1);

    GGML_ASSERT(K >= 1);
    GGML_ASSERT(ggml_nelements(src1) + K*nc*nr*nh*ns == ggml_nelements(dst));
    GGML_ASSERT(src0->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    GGML_ASSERT(src1->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    GGML_ASSERT(src2->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    GGML_ASSERT(src3->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    GGML_ASSERT(src4->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    GGML_ASSERT(src5->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    GGML_ASSERT(src6->nb[0] == ਆਕਾਰ(int32_t));
    GGML_ASSERT(nh % ng == 0);
    GGML_ASSERT(src3->ne[0] == 1 || K == 1);

    // heads per thread
    ਅਚਲ ਪੂਰਨ dh = (nh + nth - 1)/nth;

    // head range for this thread
    ਅਚਲ ਪੂਰਨ ih0 = dh*ith;
    ਅਚਲ ਪੂਰਨ ih1 = MIN(ih0 + dh, nh);

    ਅਚਲ int32_t * ids = (ਅਚਲ int32_t *) src6->ਅੰਕੜੇ;

    ਲਈ (ਪੂਰਨ i3 = 0; i3 < ns; ++i3) {
        ਅਚਲ ਦਸ਼ਮਲਵ * s0 = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src0->ਅੰਕੜੇ + ids[i3]*(src0->nb[3])); // {d_state, dim, nh, ns}
              ਦਸ਼ਮਲਵ * s  = (      ਦਸ਼ਮਲਵ *) ((      ਅੱਖਰ *) dst->ਅੰਕੜੇ  + i3*(src0->nb[3]) + s_off); // {d_state, dim, nh, ns}

        ਲਈ (ਪੂਰਨ i2 = 0; i2 < nt; ++i2) {
            ਅਚਲ ਦਸ਼ਮਲਵ * x  = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src1->ਅੰਕੜੇ + i2*(src1->nb[2]) + i3*(src1->nb[3])); // {dim, nh, nt, ns}
            ਅਚਲ ਦਸ਼ਮਲਵ * dt = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src2->ਅੰਕੜੇ + i2*(src2->nb[1]) + i3*(src2->nb[2])); // {nh, nt, ns}
            ਅਚਲ ਦਸ਼ਮਲਵ * A  = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src3->ਅੰਕੜੇ); // {d_state, nh} or {1, nh}
            ਅਚਲ ਦਸ਼ਮਲਵ * B  = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src4->ਅੰਕੜੇ + i2*(src4->nb[2]) + i3*(src4->nb[3])); // {d_state, ng, nt, ns}
            ਅਚਲ ਦਸ਼ਮਲਵ * C  = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src5->ਅੰਕੜੇ + i2*(src5->nb[2]) + i3*(src5->nb[3])); // {d_state, ng, nt, ns}
                  ਦਸ਼ਮਲਵ * y  = (      ਦਸ਼ਮਲਵ *) ((      ਅੱਖਰ *) dst->ਅੰਕੜੇ + i2*(nh*nr*ਆਕਾਰ(ਦਸ਼ਮਲਵ)) + i3*(nt*nh*nr*ਆਕਾਰ(ਦਸ਼ਮਲਵ))); // {dim, nh, nt, ns}

            ਜੇ (src3->ne[0] == 1) {
                // Mamba-2 has a scalar decay factor per head; dA can be outside the state-wise loop

                // n_head
                ਲਈ (ਪੂਰਨ h = ih0; h < ih1; ++h) {
                    // ref: https://github.com/state-spaces/mamba/blob/62db608da60f6fc790b8ed9f4b3225e95ca15fde/mamba_ssm/ops/triton/softplus.py#L16
                    ਅਚਲ ਦਸ਼ਮਲਵ dt_soft_plus = ggml_compute_softplus_f32(dt[h]);
                    ਅਚਲ ਦਸ਼ਮਲਵ dA = expf(dt_soft_plus * A[h]);
                    ਅਚਲ ਪੂਰਨ g = h / (nh / ng); // repeat_interleave

                    // dim
                    ਲਈ (ਪੂਰਨ i1 = 0; i1 < nr; ++i1) {
                        ਅਚਲ ਪੂਰਨ ii = i1 + h*nr;
                        ਅਚਲ ਦਸ਼ਮਲਵ x_dt = x[ii] * dt_soft_plus;
                        ਦਸ਼ਮਲਵ sumf = 0.0f;
#ਜੇ defined(GGML_SIMD)
    #ਜੇ defined(__ARM_FEATURE_SVE)
                        ਅਚਲ ਪੂਰਨ ggml_f32_epr = svcntw();
                        ਅਚਲ ਪੂਰਨ ggml_f32_ਕਦਮ = 1 * ggml_f32_epr;

                        ਅਚਲ ਪੂਰਨ np = (nc & ~(ggml_f32_ਕਦਮ - 1));

                        GGML_F32_VEC ਜੋੜ = GGML_F32_VEC_ZERO;

                        GGML_F32_VEC adA = GGML_F32_VEC_SET1(dA);
                        GGML_F32_VEC axdt = GGML_F32_VEC_SET1(x_dt);

                        ਲਈ (ਪੂਰਨ i = 0; i < np; i += ggml_f32_ਕਦਮ) {
                            // TODO: maybe unroll more?
                            ਲਈ (ਪੂਰਨ j = 0; j < 1; j++) {
                                GGML_F32_VEC t0 = GGML_F32_VEC_ਲੱਦੋ(s0 + i + j*ggml_f32_epr + ii*nc);
                                GGML_F32_VEC t1 = GGML_F32_VEC_ਲੱਦੋ(B + i + j*ggml_f32_epr + g*nc);
                                GGML_F32_VEC t2 = GGML_F32_VEC_ਲੱਦੋ(C + i + j*ggml_f32_epr + g*nc);

                                t0 = GGML_F32_VEC_MUL(t0, adA);
                                t1 = GGML_F32_VEC_MUL(t1, axdt);

                                t0 = GGML_F32_VEC_ਜੋੜੋ(t0, t1);

                                ਜੋੜ = GGML_F32_VEC_FMA(ਜੋੜ, t0, t2);

                                GGML_F32_VEC_STORE(s + i + j*ggml_f32_epr + ii*nc, t0);
                            }
                        }

                        sumf = GGML_F32xt_REDUCE_ONE(ਜੋੜ);
    #ਜਾਂਜੇ defined(__riscv_v_intrinsic)
                        // todo: RVV implementation
                        ਅਚਲ ਪੂਰਨ np = 0;
    #ਨਹੀਂਤਾਂ
                        ਅਚਲ ਪੂਰਨ np = (nc & ~(GGML_F32_ਕਦਮ - 1));

                        GGML_F32_VEC ਜੋੜ[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };

                        GGML_F32_VEC adA = GGML_F32_VEC_SET1(dA);
                        GGML_F32_VEC axdt = GGML_F32_VEC_SET1(x_dt);

                        GGML_F32_VEC ax[GGML_F32_ARR];
                        GGML_F32_VEC ay[GGML_F32_ARR];
                        GGML_F32_VEC az[GGML_F32_ARR];

                        ਲਈ (ਪੂਰਨ i = 0; i < np; i += GGML_F32_ਕਦਮ) {
                            ਲਈ (ਪੂਰਨ j = 0; j < GGML_F32_ARR; j++) {
                                ax[j] = GGML_F32_VEC_ਲੱਦੋ(s0 + i + j*GGML_F32_EPR + ii*nc);
                                ay[j] = GGML_F32_VEC_ਲੱਦੋ(B + i + j*GGML_F32_EPR + g*nc);
                                az[j] = GGML_F32_VEC_ਲੱਦੋ(C + i + j*GGML_F32_EPR + g*nc);

                                ax[j] = GGML_F32_VEC_MUL(ax[j], adA);
                                ay[j] = GGML_F32_VEC_MUL(ay[j], axdt);

                                ax[j] = GGML_F32_VEC_ਜੋੜੋ(ax[j], ay[j]);

                                ਜੋੜ[j] = GGML_F32_VEC_FMA(ਜੋੜ[j], ax[j], az[j]);

                                GGML_F32_VEC_STORE(s + i + j*GGML_F32_EPR + ii*nc, ax[j]);
                            }
                        }

                        // reduce sum0..sum3 to sum0
                        GGML_F32_VEC_REDUCE(sumf, ਜੋੜ);
    #ਜੇ_ਅੰਤ
#ਨਹੀਂਤਾਂ
                        ਅਚਲ ਪੂਰਨ np = 0;
#ਜੇ_ਅੰਤ
                        // d_state
                        ਲਈ (ਪੂਰਨ i0 = np; i0 < nc; ++i0) {
                            ਅਚਲ ਪੂਰਨ i = i0 + ii*nc;
                            ਅਚਲ ਪੂਰਨ ig = i0 + g*nc;
                            // state = prev_state * dA + dB * x
                            ਅਚਲ ਦਸ਼ਮਲਵ state = (s0[i] * dA) + (B[ig] * x_dt);
                            // y = rowwise_dotprod(state, C)
                            sumf += state * C[ig];
                            s[i] = state;
                        }
                        y[ii] = sumf;
                    }
                }
            } ਨਹੀਂਤਾਂ {
                // Mamba-1 has an element-wise decay factor for the states

                // n_head
                ਲਈ (ਪੂਰਨ h = ih0; h < ih1; ++h) {
                    // ref: https://github.com/state-spaces/mamba/blob/62db608da60f6fc790b8ed9f4b3225e95ca15fde/mamba_ssm/ops/triton/softplus.py#L16
                    ਅਚਲ ਦਸ਼ਮਲਵ dt_soft_plus = ggml_compute_softplus_f32(dt[h]);
                    ਅਚਲ ਪੂਰਨ g = h / (nh / ng); // repeat_interleave

                    // dim
                    ਲਈ (ਪੂਰਨ i1 = 0; i1 < nr; ++i1) {
                        ਅਚਲ ਪੂਰਨ ii = i1 + h*nr;
                        ਅਚਲ ਦਸ਼ਮਲਵ x_dt = x[ii] * dt_soft_plus;
#ਜੇ defined(__ARM_FEATURE_SVE)
                        svfloat32_t vx_dt = GGML_F32_VEC_SET1(x_dt);
                        svfloat32_t vdt_soft_plus = GGML_F32_VEC_SET1(dt_soft_plus);
                        svfloat32_t r1_ਸਦਿਸ਼ = GGML_F32_VEC_ZERO;

                        // d_state
                        // TODO: what happens when (d_state % svcntw()) != 0?
                        ਲਈ (int64_t k = 0; k < nc; k += svcntw()) {
                            svfloat32_t vA = GGML_F32_VEC_ਲੱਦੋ(&A[h*nc + k]);
                            svfloat32_t vB = GGML_F32_VEC_ਲੱਦੋ(&B[k + g*nc]);
                            svfloat32_t vC = GGML_F32_VEC_ਲੱਦੋ(&C[k + g*nc]);
                            svfloat32_t vs0 = GGML_F32_VEC_ਲੱਦੋ(&s0[ii*nc + k]);

                            svfloat32_t t1 = GGML_F32_VEC_MUL(vdt_soft_plus, vA);
                            t1 = exp_ps_sve(svptrue_b32(), t1);
                            svfloat32_t t2 = GGML_F32_VEC_MUL(vx_dt, vB);

                            vs0 = GGML_F32_VEC_FMA(t2, vs0, t1);
                            r1_ਸਦਿਸ਼ = GGML_F32_VEC_ਜੋੜੋ(GGML_F32_VEC_MUL(vs0, vC), r1_ਸਦਿਸ਼);

                            GGML_F32_VEC_STORE(&s[ii*nc + k], vs0);
                        }
                        y[ii] = GGML_F32xt_REDUCE_ONE(r1_ਸਦਿਸ਼);
#ਨਹੀਂਤਾਂ
                        ਦਸ਼ਮਲਵ sumf = 0.0f;
                        // NOTE: can't really use GGML_SIMD here because d_state is usually 16
                        //       and also because expf is used within the loop.
                        // d_state
                        ਲਈ (ਪੂਰਨ i0 = 0; i0 < nc; ++i0) {
                            ਅਚਲ ਪੂਰਨ i = i0 + ii*nc;
                            ਅਚਲ ਪੂਰਨ ig = i0 + g*nc;
                            // state = prev_state * dA + dB * x
                            ਅਚਲ ਦਸ਼ਮਲਵ state = (s0[i] * expf(dt_soft_plus * A[i0 + h*nc])) + (B[ig] * x_dt);
                            // y = rowwise_dotprod(state, C)
                            sumf += state * C[ig];
                            s[i] = state;
                        }
                        y[ii] = sumf;
#ਜੇ_ਅੰਤ
                    }
                }
            }
            ਅਚਲ int64_t ਥਾਂ = nt - 1 - i2;
            ਜੇ (K > 1 && ਥਾਂ > 0 && ਥਾਂ < K) {
                ਦਸ਼ਮਲਵ * s_snapshot = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + s_off + (ਥਾਂ*ns + i3)*(src0->nb[3]));
                ਲਈ (ਪੂਰਨ h = ih0; h < ih1; ++h) {
                    ਸਮ੍ਰਿਤੀ_ਨਕਲ((ਅੱਖਰ *) s_snapshot + h*src0->nb[2], (ਅੱਖਰ *) s + h*src0->nb[2], src0->nb[2]);
                }
            }
            // use the output as the source when it's not the first token-wise iteration
            s0 = s;
        }
    }
}

ਖਾਲੀ ggml_compute_forward_ssm_ਜਾਂਚ(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {
    ਚੋਣ (dst->src[0]->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_ssm_ਜਾਂਚ_f32(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_win_part

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_win_part_f32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {
    GGML_ਅਣਵਰਤਿਆ(params);

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)

    ਅਚਲ int32_t nep0 = ((ਅਚਲ int32_t *)(dst->op_params))[0];
    ਅਚਲ int32_t nep1 = ((ਅਚਲ int32_t *)(dst->op_params))[1];
    ਅਚਲ int32_t w    = ((ਅਚਲ int32_t *)(dst->op_params))[2];

    ਪ੍ਰਤਿਜ੍ਞਾ(ne00 == ne0);
    ਪ੍ਰਤਿਜ੍ਞਾ(ne3  == nep0*nep1);

    // TODO: optimize / multi-thread
    ਲਈ (ਪੂਰਨ py = 0; py < nep1; ++py) {
        ਲਈ (ਪੂਰਨ px = 0; px < nep0; ++px) {
            ਅਚਲ int64_t i3 = py*nep0 + px;
            ਲਈ (int64_t i2 = 0; i2 < ne2; ++i2) {
                ਲਈ (int64_t i1 = 0; i1 < ne1; ++i1) {
                    ਲਈ (int64_t i0 = 0; i0 < ne0; ++i0) {
                        ਅਚਲ int64_t i02 = py*w + i2;
                        ਅਚਲ int64_t i01 = px*w + i1;
                        ਅਚਲ int64_t i00 = i0;

                        ਅਚਲ int64_t i = i3*ne2*ne1*ne0 + i2*ne1*ne0    + i1*ne0   + i0;
                        ਅਚਲ int64_t j =                  i02*ne01*ne00 + i01*ne00 + i00;

                        ਜੇ (py*w + i2 >= ne02 || px*w + i1 >= ne01) {
                            ((ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ)[i] = 0.0f;
                        } ਨਹੀਂਤਾਂ {
                            ((ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ)[i] = ((ਦਸ਼ਮਲਵ *) src0->ਅੰਕੜੇ)[j];
                        }
                    }
                }
            }
        }
    }
}

ਖਾਲੀ ggml_compute_forward_win_part(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_win_part_f32(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_win_unpart

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_win_unpart_f32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {
    GGML_ਅਣਵਰਤਿਆ(params);

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)

    ਅਚਲ int32_t w = ((ਅਚਲ int32_t *)(dst->op_params))[0];

    // padding
    ਅਚਲ ਪੂਰਨ px = (w - ne1%w)%w;
    //const int py = (w - ne2%w)%w;

    ਅਚਲ ਪੂਰਨ npx = (px + ne1)/w;
    //const int npy = (py + ne2)/w;

    ਪ੍ਰਤਿਜ੍ਞਾ(ne0 == ne00);

    // TODO: optimize / multi-thread
    ਲਈ (int64_t i2 = 0; i2 < ne2; ++i2) {
        ਲਈ (int64_t i1 = 0; i1 < ne1; ++i1) {
            ਲਈ (int64_t i0 = 0; i0 < ne0; ++i0) {
                ਅਚਲ ਪੂਰਨ ip2 = i2/w;
                ਅਚਲ ਪੂਰਨ ip1 = i1/w;

                ਅਚਲ int64_t i02 = i2%w;
                ਅਚਲ int64_t i01 = i1%w;
                ਅਚਲ int64_t i00 = i0;

                ਅਚਲ int64_t i = (ip2*npx + ip1)*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00 + i00;
                ਅਚਲ int64_t j =                                  i2*ne1*ne0    + i1*ne0   + i0;

                ((ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ)[j] = ((ਦਸ਼ਮਲਵ *) src0->ਅੰਕੜੇ)[i];
            }
        }
    }
}

ਖਾਲੀ ggml_compute_forward_win_unpart(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_win_unpart_f32(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

//ggml_compute_forward_unary

ਖਾਲੀ ggml_compute_forward_unary(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_unary_op op = ggml_get_unary_op(dst);

    ਚੋਣ (op) {
        ਹਾਲਤ GGML_UNARY_OP_ABS:
            {
                ggml_compute_forward_abs(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_UNARY_OP_SGN:
            {
                ggml_compute_forward_sgn(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_UNARY_OP_NEG:
            {
                ggml_compute_forward_neg(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_UNARY_OP_ਕਦਮ:
            {
                ggml_compute_forward_ਕਦਮ(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_UNARY_OP_TANH:
            {
                ggml_compute_forward_tanh(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_UNARY_OP_ELU:
            {
                ggml_compute_forward_elu(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_UNARY_OP_RELU:
            {
                ggml_compute_forward_relu(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_UNARY_OP_SIGMOID:
            {
                ggml_compute_forward_sigmoid(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_UNARY_OP_GELU:
            {
                ggml_compute_forward_gelu(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_UNARY_OP_GELU_ERF:
            {
                ggml_compute_forward_gelu_erf(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_UNARY_OP_GELU_ਤੇਜ਼:
            {
                ggml_compute_forward_gelu_ਤੇਜ਼(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_UNARY_OP_SILU:
            {
                ggml_compute_forward_silu(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_UNARY_OP_HARDSWISH:
            {
                ggml_compute_forward_hardswish(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_UNARY_OP_HARDSIGMOID:
            {
                ggml_compute_forward_hardsigmoid(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_UNARY_OP_EXP:
            {
                ggml_compute_forward_exp(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_UNARY_OP_FLOOR:
            {
                ggml_compute_forward_floor(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_UNARY_OP_CEIL:
            {
                ggml_compute_forward_ceil(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_UNARY_OP_ਗੇੜਾ:
            {
                ggml_compute_forward_ਗੇੜਾ(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_UNARY_OP_TRUNC:
            {
                ggml_compute_forward_trunc(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_UNARY_OP_XIELU:
            {
                ggml_compute_forward_xielu(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_UNARY_OP_EXPM1:
            {
                ggml_compute_forward_expm1(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_UNARY_OP_SOFTPLUS:
            {
                ggml_compute_forward_softplus(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

//ggml_compute_forward_glu

ਖਾਲੀ ggml_compute_forward_glu(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_glu_op op = ggml_get_glu_op(dst);

    ਚੋਣ (op) {
        ਹਾਲਤ GGML_GLU_OP_REGLU:
            {
                ggml_compute_forward_reglu(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_GLU_OP_GEGLU:
            {
                ggml_compute_forward_geglu(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_GLU_OP_SWIGLU:
            {
                ggml_compute_forward_swiglu(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_GLU_OP_SWIGLU_OAI:
            {
                ggml_compute_forward_swiglu_oai(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_GLU_OP_GEGLU_ERF:
            {
                ggml_compute_forward_geglu_erf(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_GLU_OP_GEGLU_ਤੇਜ਼:
            {
                ggml_compute_forward_geglu_ਤੇਜ਼(params, dst);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_GLU_OP_SWIGLU_CLAMP:
            {
                ggml_compute_forward_swiglu_clamp(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_get_rel_pos

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_get_rel_pos_f16(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {
    GGML_ਅਣਵਰਤਿਆ(params);

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L292-L322

    GGML_TENSOR_UNARY_OP_LOCALS

    ਅਚਲ int64_t w = ne1;

    ggml_fp16_t * src0_ਅੰਕੜੇ = (ggml_fp16_t *) src0->ਅੰਕੜੇ;
    ggml_fp16_t * dst_ਅੰਕੜੇ  = (ggml_fp16_t *) dst->ਅੰਕੜੇ;

    ਲਈ (int64_t i2 = 0; i2 < ne2; ++i2) {
        ਲਈ (int64_t i1 = 0; i1 < ne1; ++i1) {
            ਅਚਲ int64_t pos = (w - i1 - 1) + i2;
            ਲਈ (int64_t i0 = 0; i0 < ne0; ++i0) {
                dst_ਅੰਕੜੇ[i2*ne1*ne0 + i1*ne0 + i0] = src0_ਅੰਕੜੇ[pos*ne00 + i0];
            }
        }
    }
}

ਖਾਲੀ ggml_compute_forward_get_rel_pos(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F16:
        ਹਾਲਤ GGML_TYPE_BF16:
            {
                ggml_compute_forward_get_rel_pos_f16(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_add_rel_pos

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_ਜੋੜੋ_rel_pos_f32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];
    ਅਚਲ ggml_tensor * src2 = dst->src[2];

    ਅਚਲ ਤਾਰਕਿਕ inplace = (ਤਾਰਕਿਕ) ((int32_t *) dst->op_params)[0];
    ਜੇ (!inplace) {
        ਜੇ (params->ith == 0) {
            ਸਮ੍ਰਿਤੀ_ਨਕਲ((ਅੱਖਰ *) dst->ਅੰਕੜੇ, (ਅੱਖਰ *) src0->ਅੰਕੜੇ, ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }
    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L357-L359

    ਦਸ਼ਮਲਵ * src1_ਅੰਕੜੇ = (ਦਸ਼ਮਲਵ *) src1->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * src2_ਅੰਕੜੇ = (ਦਸ਼ਮਲਵ *) src2->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * dst_ਅੰਕੜੇ  = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;

    ਅਚਲ int64_t ne10 = src1->ne[0];
    ਅਚਲ int64_t ne11 = src1->ne[1];
    ਅਚਲ int64_t ne12 = src1->ne[2];
    ਅਚਲ int64_t ne13 = src1->ne[3];

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    // total patches in dst
    ਅਚਲ ਪੂਰਨ np = ne13;

    // patches per thread
    ਅਚਲ ਪੂਰਨ dp = (np + nth - 1)/nth;

    // patch range for this thread
    ਅਚਲ ਪੂਰਨ ip0 = dp*ith;
    ਅਚਲ ਪੂਰਨ ip1 = MIN(ip0 + dp, np);

    ਲਈ (int64_t i13 = ip0; i13 < ip1; ++i13) {
        ਲਈ (int64_t i12 = 0; i12 < ne12; ++i12) {
            ਲਈ (int64_t i11 = 0; i11 < ne11; ++i11) {
                ਅਚਲ int64_t jp1 = i13*ne12*ne11*ne10 + i12*ne11*ne10 + i11*ne10;
                ਲਈ (int64_t i10 = 0; i10 < ne10; ++i10) {
                    ਅਚਲ int64_t jp0  = jp1 + i10;
                    ਅਚਲ ਦਸ਼ਮਲਵ src1_e = src1_ਅੰਕੜੇ[jp0];
                    ਅਚਲ ਦਸ਼ਮਲਵ src2_e = src2_ਅੰਕੜੇ[jp0];

                    ਅਚਲ int64_t jdh = jp0 * ne10;
                    ਅਚਲ int64_t jdw = jdh - (ne10 - 1) * i10;

                    ਲਈ (int64_t j = 0; j < ne10; ++j) {
                        dst_ਅੰਕੜੇ[jdh + j     ] += src2_e;
                        dst_ਅੰਕੜੇ[jdw + j*ne10] += src1_e;
                    }
                }
            }
        }
    }
}

ਖਾਲੀ ggml_compute_forward_ਜੋੜੋ_rel_pos(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_ਜੋੜੋ_rel_pos_f32(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_rwkv_wkv6

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_rwkv_wkv6_f32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {
    ਅਚਲ int64_t T = dst->src[1]->ne[2];
    ਅਚਲ int64_t C = dst->ne[0];
    ਅਚਲ int64_t HEADS = dst->src[1]->ne[1];
    ਅਚਲ int64_t n_seqs = dst->src[5]->ne[1];
    ਅਚਲ int64_t ਸਿਰ_ਮਾਪ = C / HEADS;

    ਦਸ਼ਮਲਵ * dst_ਅੰਕੜੇ = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * state = ((ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ) + C * T;

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ h_ਸ਼ੁਰੂ =  (HEADS * (ith    )) / nth;
    ਅਚਲ ਪੂਰਨ h_ਸਿਰਾ   = ((HEADS * (ith + 1)) / nth < HEADS) ?
                         (HEADS * (ith + 1)) / nth : HEADS;

    ਦਸ਼ਮਲਵ * k =          (ਦਸ਼ਮਲਵ *) dst->src[0]->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * v =          (ਦਸ਼ਮਲਵ *) dst->src[1]->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * r =          (ਦਸ਼ਮਲਵ *) dst->src[2]->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * ਸਮਾਂ_faaaa = (ਦਸ਼ਮਲਵ *) dst->src[3]->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * ਸਮਾਂ_decay = (ਦਸ਼ਮਲਵ *) dst->src[4]->ਅੰਕੜੇ;

    size_t t_stride = HEADS * ਸਿਰ_ਮਾਪ; // Same to C

    size_t h_stride = C / HEADS;
    GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
    size_t h_stride_2d = ਸਿਰ_ਮਾਪ * ਸਿਰ_ਮਾਪ;

    ਜੇ (ith == 0) {
        memset(dst_ਅੰਕੜੇ, 0, T * C * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    }
    ggml_barrier(params->threadpool);


    #ਜੇ defined(__AVX__) && !defined(__AVX512F__)
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32X GGML_F32x8
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32X_SET1 GGML_F32x8_SET1
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_X_ਲੱਦੋ GGML_F32x8_ਲੱਦੋ
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32X_STORE GGML_F32x8_STORE
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32X_MUL GGML_F32x8_MUL
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32X_FMA GGML_F32x8_FMA
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ WKV_ਸਦਿਸ਼_ਮਾਪ 8
    #ਜਾਂਜੇ defined(__AVX512F__)
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32X GGML_F32x16
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32X_SET1 GGML_F32x16_SET1
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_X_ਲੱਦੋ GGML_F32x16_ਲੱਦੋ
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32X_STORE GGML_F32x16_STORE
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32X_MUL GGML_F32x16_MUL
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32X_FMA GGML_F32x16_FMA
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ WKV_ਸਦਿਸ਼_ਮਾਪ 16
    #ਜਾਂਜੇ defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32X GGML_F32xt
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32X_SET1 GGML_F32xt_SET1
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_X_ਲੱਦੋ GGML_F32xt_ਲੱਦੋ
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32X_STORE GGML_F32xt_STORE
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32X_MUL GGML_F32xt_MUL
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32X_FMA GGML_F32xt_FMA
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ WKV_ਸਦਿਸ਼_ਮਾਪ 8
    #ਜਾਂਜੇ defined(__ARM_NEON) && defined(__aarch64__)
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32X GGML_F32x4
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32X_SET1 GGML_F32x4_SET1
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_X_ਲੱਦੋ GGML_F32x4_ਲੱਦੋ
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32X_STORE GGML_F32x4_STORE
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32X_MUL GGML_F32x4_MUL
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32X_FMA GGML_F32x4_FMA
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ WKV_ਸਦਿਸ਼_ਮਾਪ 4
    #ਜੇ_ਅੰਤ

    #ਜੇ_ਪਰਿਭਾਸ਼ਿਤ WKV_ਸਦਿਸ਼_ਮਾਪ
        ਪੂਰਨ wkv_ਸਦਿਸ਼_ਮਾਪ;
        #ਜੇ defined(__ARM_FEATURE_SVE)
            wkv_ਸਦਿਸ਼_ਮਾਪ = svcntw();
        #ਨਹੀਂਤਾਂ
            wkv_ਸਦਿਸ਼_ਮਾਪ = WKV_ਸਦਿਸ਼_ਮਾਪ;
        #ਜੇ_ਅੰਤ
        ਅਚਲ int64_t vec_ਗਿਣਤੀ = ਸਿਰ_ਮਾਪ / wkv_ਸਦਿਸ਼_ਮਾਪ;

        ਲਈ (int64_t t = 0; t < T; t++) {
            size_t t_ਵਿੱਥ = t * t_stride;
            size_t state_ਵਿੱਥ = ਸਿਰ_ਮਾਪ * C * (t / (T / n_seqs));
            ਦਸ਼ਮਲਵ * state_cur = state + state_ਵਿੱਥ;
            ਦਸ਼ਮਲਵ * state_prev = t % (T / n_seqs) ? state_cur : (ਦਸ਼ਮਲਵ*)dst->src[5]->ਅੰਕੜੇ + state_ਵਿੱਥ;

            ਲਈ (int64_t h = h_ਸ਼ੁਰੂ; h < h_ਸਿਰਾ; h++) {
                size_t h_ਵਿੱਥ = h * h_stride;
                size_t t_h_ਵਿੱਥ = t_ਵਿੱਥ + h_ਵਿੱਥ;
                size_t h_2d_ਵਿੱਥ = h * h_stride_2d;

                ਲਈ (int64_t i = 0; i < ਸਿਰ_ਮਾਪ; i++) {
                    size_t t_h_i_ਵਿੱਥ = t_h_ਵਿੱਥ + i;
                    size_t h_i_ਵਿੱਥ = h_ਵਿੱਥ + i;
                    size_t h_2d_i_ਵਿੱਥ = h_2d_ਵਿੱਥ + i * h_stride;

                    ਦਸ਼ਮਲਵ k_val = k[t_h_i_ਵਿੱਥ];
                    ਦਸ਼ਮਲਵ r_val = r[t_h_i_ਵਿੱਥ];
                    ਦਸ਼ਮਲਵ ਸਮਾਂ_faaaa_val = ਸਮਾਂ_faaaa[h_i_ਵਿੱਥ];
                    ਦਸ਼ਮਲਵ ਸਮਾਂ_decay_val = ਸਮਾਂ_decay[t_h_i_ਵਿੱਥ];

                    // Broadcast scalar values to vectors
                    GGML_F32X k_vec = GGML_F32X_SET1(k_val);
                    GGML_F32X r_vec = GGML_F32X_SET1(r_val);
                    GGML_F32X ਸਮਾਂ_faaaa_vec = GGML_F32X_SET1(ਸਮਾਂ_faaaa_val);
                    GGML_F32X ਸਮਾਂ_decay_vec = GGML_F32X_SET1(ਸਮਾਂ_decay_val);

                    ਲਈ (int64_t j = 0; j < vec_ਗਿਣਤੀ; j++) {
                        size_t ਅਧਾਰ_j = j * wkv_ਸਦਿਸ਼_ਮਾਪ;
                        size_t t_h_j_ਵਿੱਥ = t_h_ਵਿੱਥ + ਅਧਾਰ_j;
                        size_t h_2d_i_j_ਵਿੱਥ = h_2d_i_ਵਿੱਥ + ਅਧਾਰ_j;

                        // Load x elements at once
                        GGML_F32X v_vec = GGML_F32_X_ਲੱਦੋ(&v[t_h_j_ਵਿੱਥ]);
                        GGML_F32X prev_state_vec = GGML_F32_X_ਲੱਦੋ(&state_prev[h_2d_i_j_ਵਿੱਥ]);
                        GGML_F32X dst_vec = GGML_F32_X_ਲੱਦੋ(&dst_ਅੰਕੜੇ[t_h_j_ਵਿੱਥ]);

                        // Compute kv = v * k
                        GGML_F32X kv_vec = GGML_F32X_MUL(v_vec, k_vec);

                        // Compute temp = kv * time_faaaa + prev_state
                        GGML_F32X ਆਰਜ਼ੀ_vec = GGML_F32X_FMA(prev_state_vec, kv_vec, ਸਮਾਂ_faaaa_vec);

                        // Update dst: dst += temp * r
                        dst_vec = GGML_F32X_FMA(dst_vec, ਆਰਜ਼ੀ_vec, r_vec);
                        GGML_F32X_STORE(&dst_ਅੰਕੜੇ[t_h_j_ਵਿੱਥ], dst_vec);

                        // Update state: state = prev_state * time_decay + kv
                        GGML_F32X new_state_vec = GGML_F32X_FMA(kv_vec, prev_state_vec, ਸਮਾਂ_decay_vec);
                        GGML_F32X_STORE(&state_cur[h_2d_i_j_ਵਿੱਥ], new_state_vec);
                    }

                    // Handle remaining elements, this will not be used.
                    ਲਈ (int64_t j = vec_ਗਿਣਤੀ * wkv_ਸਦਿਸ਼_ਮਾਪ; j < ਸਿਰ_ਮਾਪ; j++) {
                        size_t t_h_j_ਵਿੱਥ = t_h_ਵਿੱਥ + j;
                        size_t h_2d_i_j_ਵਿੱਥ = h_2d_i_ਵਿੱਥ + j;
                        ਦਸ਼ਮਲਵ v_val = v[t_h_j_ਵਿੱਥ];
                        ਦਸ਼ਮਲਵ kv_val = v_val * k_val;
                        ਦਸ਼ਮਲਵ prev_state_val = state_prev[h_2d_i_j_ਵਿੱਥ];
                        ਦਸ਼ਮਲਵ ਆਰਜ਼ੀ_val = kv_val * ਸਮਾਂ_faaaa_val + prev_state_val;
                        dst_ਅੰਕੜੇ[t_h_j_ਵਿੱਥ] += ਆਰਜ਼ੀ_val * r_val;
                        state_cur[h_2d_i_j_ਵਿੱਥ] = prev_state_val * ਸਮਾਂ_decay_val + kv_val;
                    }
                }
            }
        }

    #ਨਹੀਂਤਾਂ
        // basically fused operations:
        // dst = r @ (time_faaaa * (k @ v) + state),
        // state = time_decay * state + (k @ v),
        // recursive through each token
        ਲਈ (int64_t t = 0; t < T; t++) {
            size_t t_ਵਿੱਥ = t * t_stride;
            size_t state_ਵਿੱਥ = ਸਿਰ_ਮਾਪ * C * (t / (T / n_seqs));
            ਦਸ਼ਮਲਵ * state_cur = state + state_ਵਿੱਥ;
            ਦਸ਼ਮਲਵ * state_prev = t % (T / n_seqs) ? state_cur : (ਦਸ਼ਮਲਵ*)dst->src[5]->ਅੰਕੜੇ + state_ਵਿੱਥ;

            ਲਈ (int64_t h = h_ਸ਼ੁਰੂ; h < h_ਸਿਰਾ; h++) {
                size_t h_ਵਿੱਥ = h * h_stride;
                size_t t_h_ਵਿੱਥ = t_ਵਿੱਥ + h_ਵਿੱਥ;
                size_t h_2d_ਵਿੱਥ = h * h_stride_2d;

                ਲਈ (int64_t i = 0; i < ਸਿਰ_ਮਾਪ; i++) {
                    size_t t_h_i_ਵਿੱਥ = t_h_ਵਿੱਥ + i;
                    size_t h_i_ਵਿੱਥ = h_ਵਿੱਥ + i;
                    size_t h_2d_i_ਵਿੱਥ = h_2d_ਵਿੱਥ + i * h_stride;

                    ਦਸ਼ਮਲਵ k_val = k[t_h_i_ਵਿੱਥ];
                    ਦਸ਼ਮਲਵ r_val = r[t_h_i_ਵਿੱਥ];
                    ਦਸ਼ਮਲਵ ਸਮਾਂ_faaaa_val = ਸਮਾਂ_faaaa[h_i_ਵਿੱਥ];
                    // RWKV v6: different time_decay for each token.
                    ਦਸ਼ਮਲਵ ਸਮਾਂ_decay_val = ਸਮਾਂ_decay[t_h_i_ਵਿੱਥ];

                    ਲਈ (int64_t j = 0; j < ਸਿਰ_ਮਾਪ; j++) {
                        size_t t_h_j_ਵਿੱਥ = t_h_ਵਿੱਥ + j;
                        size_t h_2d_i_j_ਵਿੱਥ = h_2d_i_ਵਿੱਥ + j;

                        ਦਸ਼ਮਲਵ v_val = v[t_h_j_ਵਿੱਥ];
                        ਦਸ਼ਮਲਵ kv_val = v_val * k_val;
                        ਦਸ਼ਮਲਵ prev_state_val = state_prev[h_2d_i_j_ਵਿੱਥ];
                        ਦਸ਼ਮਲਵ ਆਰਜ਼ੀ_val = kv_val * ਸਮਾਂ_faaaa_val + prev_state_val;
                        dst_ਅੰਕੜੇ[t_h_j_ਵਿੱਥ] += ਆਰਜ਼ੀ_val * r_val;
                        state_cur[h_2d_i_j_ਵਿੱਥ] = prev_state_val * ਸਮਾਂ_decay_val + kv_val;
                    }
                }
            }
        }
    #ਜੇ_ਅੰਤ
}


ਖਾਲੀ ggml_compute_forward_rwkv_wkv6(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_rwkv_wkv6_f32(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_gla

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_gla_f32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {
    ਅਚਲ int64_t T = dst->src[1]->ne[2];
    ਅਚਲ int64_t C = dst->ne[0];
    ਅਚਲ int64_t HEADS = dst->src[1]->ne[1];
    ਅਚਲ int64_t n_seqs = dst->src[4]->ne[1];
    ਅਚਲ int64_t ਸਿਰ_ਮਾਪ = C / HEADS;
    ਅਚਲ ਦਸ਼ਮਲਵ scale = ggml_get_op_params_f32(dst, 0);

    ਦਸ਼ਮਲਵ * dst_ਅੰਕੜੇ = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * state = ((ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ) + C * T;

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ h_ਸ਼ੁਰੂ =  (HEADS * (ith    )) / nth;
    ਅਚਲ ਪੂਰਨ h_ਸਿਰਾ   = ((HEADS * (ith + 1)) / nth < HEADS) ?
                         (HEADS * (ith + 1)) / nth : HEADS;

    ਦਸ਼ਮਲਵ * k = (ਦਸ਼ਮਲਵ *) dst->src[0]->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * v = (ਦਸ਼ਮਲਵ *) dst->src[1]->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * q = (ਦਸ਼ਮਲਵ *) dst->src[2]->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * g = (ਦਸ਼ਮਲਵ *) dst->src[3]->ਅੰਕੜੇ;

    size_t t_stride = HEADS * ਸਿਰ_ਮਾਪ; // Same to C

    size_t h_stride = C / HEADS;
    GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
    size_t h_stride_2d = ਸਿਰ_ਮਾਪ * ਸਿਰ_ਮਾਪ;

    ਜੇ (ith == 0) {
        memset(dst_ਅੰਕੜੇ, 0, T * C * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    }
    ggml_barrier(params->threadpool);


    #ਜੇ defined(__AVX__) && !defined(__AVX512F__)
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32X GGML_F32x8
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32X_SET1 GGML_F32x8_SET1
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_X_ਲੱਦੋ GGML_F32x8_ਲੱਦੋ
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32X_STORE GGML_F32x8_STORE
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32X_MUL GGML_F32x8_MUL
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32X_FMA GGML_F32x8_FMA
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GLA_ਸਦਿਸ਼_ਮਾਪ 8
    #ਜਾਂਜੇ defined(__AVX512F__)
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32X GGML_F32x16
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32X_SET1 GGML_F32x16_SET1
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_X_ਲੱਦੋ GGML_F32x16_ਲੱਦੋ
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32X_STORE GGML_F32x16_STORE
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32X_MUL GGML_F32x16_MUL
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32X_FMA GGML_F32x16_FMA
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GLA_ਸਦਿਸ਼_ਮਾਪ 16
    #ਜਾਂਜੇ defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32X GGML_F32xt
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32X_SET1 GGML_F32xt_SET1
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_X_ਲੱਦੋ GGML_F32xt_ਲੱਦੋ
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32X_STORE GGML_F32xt_STORE
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32X_MUL GGML_F32xt_MUL
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32X_FMA GGML_F32xt_FMA
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GLA_ਸਦਿਸ਼_ਮਾਪ 8
    #ਜਾਂਜੇ defined(__ARM_NEON) && defined(__aarch64__)
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32X GGML_F32x4
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32X_SET1 GGML_F32x4_SET1
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_X_ਲੱਦੋ GGML_F32x4_ਲੱਦੋ
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32X_STORE GGML_F32x4_STORE
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32X_MUL GGML_F32x4_MUL
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32X_FMA GGML_F32x4_FMA
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GLA_ਸਦਿਸ਼_ਮਾਪ 4
    #ਜੇ_ਅੰਤ

    #ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GLA_ਸਦਿਸ਼_ਮਾਪ
        ਪੂਰਨ gla_ਸਦਿਸ਼_ਮਾਪ;
        #ਜੇ defined(__ARM_FEATURE_SVE)
            gla_ਸਦਿਸ਼_ਮਾਪ = svcntw();
        #ਨਹੀਂਤਾਂ
            gla_ਸਦਿਸ਼_ਮਾਪ = GLA_ਸਦਿਸ਼_ਮਾਪ;
        #ਜੇ_ਅੰਤ
        ਅਚਲ int64_t vec_ਗਿਣਤੀ = ਸਿਰ_ਮਾਪ / gla_ਸਦਿਸ਼_ਮਾਪ;

        ਲਈ (int64_t t = 0; t < T; t++) {
            size_t t_ਵਿੱਥ = t * t_stride;
            size_t state_ਵਿੱਥ = ਸਿਰ_ਮਾਪ * C * (t / (T / n_seqs));
            ਦਸ਼ਮਲਵ * state_cur = state + state_ਵਿੱਥ;
            ਦਸ਼ਮਲਵ * state_prev = t % (T / n_seqs) ? state_cur : (ਦਸ਼ਮਲਵ*)dst->src[4]->ਅੰਕੜੇ + state_ਵਿੱਥ;

            ਲਈ (int64_t h = h_ਸ਼ੁਰੂ; h < h_ਸਿਰਾ; h++) {
                size_t h_ਵਿੱਥ = h * h_stride;
                size_t t_h_ਵਿੱਥ = t_ਵਿੱਥ + h_ਵਿੱਥ;
                size_t h_2d_ਵਿੱਥ = h * h_stride_2d;

                ਲਈ (int64_t i = 0; i < ਸਿਰ_ਮਾਪ; i++) {
                    size_t t_h_i_ਵਿੱਥ = t_h_ਵਿੱਥ + i;
                    size_t h_2d_i_ਵਿੱਥ = h_2d_ਵਿੱਥ + i * h_stride;

                    ਦਸ਼ਮਲਵ k_val = k[t_h_i_ਵਿੱਥ];
                    ਦਸ਼ਮਲਵ q_val = q[t_h_i_ਵਿੱਥ] * scale;
                    ਦਸ਼ਮਲਵ g_val = g[t_h_i_ਵਿੱਥ];

                    // Broadcast scalar values to vectors
                    GGML_F32X k_vec = GGML_F32X_SET1(k_val);
                    GGML_F32X q_vec = GGML_F32X_SET1(q_val);
                    GGML_F32X g_vec = GGML_F32X_SET1(g_val);

                    ਲਈ (int64_t j = 0; j < vec_ਗਿਣਤੀ; j++) {
                        size_t ਅਧਾਰ_j = j * gla_ਸਦਿਸ਼_ਮਾਪ;
                        size_t t_h_j_ਵਿੱਥ = t_h_ਵਿੱਥ + ਅਧਾਰ_j;
                        size_t h_2d_i_j_ਵਿੱਥ = h_2d_i_ਵਿੱਥ + ਅਧਾਰ_j;

                        // Load x elements at once
                        GGML_F32X v_vec = GGML_F32_X_ਲੱਦੋ(&v[t_h_j_ਵਿੱਥ]);
                        GGML_F32X prev_state_vec = GGML_F32_X_ਲੱਦੋ(&state_prev[h_2d_i_j_ਵਿੱਥ]);
                        GGML_F32X dst_vec = GGML_F32_X_ਲੱਦੋ(&dst_ਅੰਕੜੇ[t_h_j_ਵਿੱਥ]);

                        // Compute kv = v * k
                        GGML_F32X kv_vec = GGML_F32X_MUL(v_vec, k_vec);

                        // Compute temp = prev_state * g + kv
                        GGML_F32X ਆਰਜ਼ੀ_vec = GGML_F32X_FMA(kv_vec, prev_state_vec, g_vec);

                        // Update dst: dst += temp * q
                        dst_vec = GGML_F32X_FMA(dst_vec, ਆਰਜ਼ੀ_vec, q_vec);
                        GGML_F32X_STORE(&dst_ਅੰਕੜੇ[t_h_j_ਵਿੱਥ], dst_vec);

                        // Update state
                        GGML_F32X_STORE(&state_cur[h_2d_i_j_ਵਿੱਥ], ਆਰਜ਼ੀ_vec);
                    }

                    // Handle remaining elements, this will not be used.
                    ਲਈ (int64_t j = vec_ਗਿਣਤੀ * gla_ਸਦਿਸ਼_ਮਾਪ; j < ਸਿਰ_ਮਾਪ; j++) {
                        size_t t_h_j_ਵਿੱਥ = t_h_ਵਿੱਥ + j;
                        size_t h_2d_i_j_ਵਿੱਥ = h_2d_i_ਵਿੱਥ + j;
                        ਦਸ਼ਮਲਵ v_val = v[t_h_j_ਵਿੱਥ];
                        ਦਸ਼ਮਲਵ kv_val = v_val * k_val;
                        ਦਸ਼ਮਲਵ prev_state_val = state_prev[h_2d_i_j_ਵਿੱਥ];
                        ਦਸ਼ਮਲਵ ਆਰਜ਼ੀ_val = kv_val + prev_state_val * g_val;
                        dst_ਅੰਕੜੇ[t_h_j_ਵਿੱਥ] += ਆਰਜ਼ੀ_val * q_val;
                        state_cur[h_2d_i_j_ਵਿੱਥ] = ਆਰਜ਼ੀ_val;
                    }
                }
            }
        }

    #ਨਹੀਂਤਾਂ
        ਲਈ (int64_t t = 0; t < T; t++) {
            size_t t_ਵਿੱਥ = t * t_stride;
            size_t state_ਵਿੱਥ = ਸਿਰ_ਮਾਪ * C * (t / (T / n_seqs));
            ਦਸ਼ਮਲਵ * state_cur = state + state_ਵਿੱਥ;
            ਦਸ਼ਮਲਵ * state_prev = t % (T / n_seqs) ? state_cur : (ਦਸ਼ਮਲਵ*)dst->src[4]->ਅੰਕੜੇ + state_ਵਿੱਥ;

            ਲਈ (int64_t h = h_ਸ਼ੁਰੂ; h < h_ਸਿਰਾ; h++) {
                size_t h_ਵਿੱਥ = h * h_stride;
                size_t t_h_ਵਿੱਥ = t_ਵਿੱਥ + h_ਵਿੱਥ;
                size_t h_2d_ਵਿੱਥ = h * h_stride_2d;

                ਲਈ (int64_t i = 0; i < ਸਿਰ_ਮਾਪ; i++) {
                    size_t t_h_i_ਵਿੱਥ = t_h_ਵਿੱਥ + i;
                    size_t h_2d_i_ਵਿੱਥ = h_2d_ਵਿੱਥ + i * h_stride;

                    ਦਸ਼ਮਲਵ k_val = k[t_h_i_ਵਿੱਥ];
                    ਦਸ਼ਮਲਵ q_val = q[t_h_i_ਵਿੱਥ] * scale;
                    ਦਸ਼ਮਲਵ g_val = g[t_h_i_ਵਿੱਥ];

                    ਲਈ (int64_t j = 0; j < ਸਿਰ_ਮਾਪ; j++) {
                        size_t t_h_j_ਵਿੱਥ = t_h_ਵਿੱਥ + j;
                        size_t h_2d_i_j_ਵਿੱਥ = h_2d_i_ਵਿੱਥ + j;

                        ਦਸ਼ਮਲਵ v_val = v[t_h_j_ਵਿੱਥ];
                        ਦਸ਼ਮਲਵ kv_val = v_val * k_val;
                        ਦਸ਼ਮਲਵ prev_state_val = state_prev[h_2d_i_j_ਵਿੱਥ];
                        ਦਸ਼ਮਲਵ ਆਰਜ਼ੀ_val = prev_state_val * g_val + kv_val;
                        dst_ਅੰਕੜੇ[t_h_j_ਵਿੱਥ] += ਆਰਜ਼ੀ_val * q_val;
                        state_cur[h_2d_i_j_ਵਿੱਥ] = ਆਰਜ਼ੀ_val;
                    }
                }
            }
        }
    #ਜੇ_ਅੰਤ
}


ਖਾਲੀ ggml_compute_forward_gla(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_gla_f32(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_solve_tri_f32(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst) {
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0 = dst->src[0];  // A (lower triangular)
    ਅਚਲ ਬਣਤਰ ggml_tensor * src1 = dst->src[1];  // B (RHS)

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS;

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(ne00 == ne01); // A must be square
    GGML_ASSERT(ne0  == ne10); // solution cols == B cols
    GGML_ASSERT(ne1  == ne11); // solution rows == B rows

    GGML_ASSERT(ne02 == ne12 && ne12 == ne2);
    GGML_ASSERT(ne03 == ne13 && ne13 == ne3);

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ int64_t k = ne10;   // number of RHS columns
    ਅਚਲ int64_t n = ne11;   // A is n×n
    ਅਚਲ int64_t nr = ne02 * ne03 * k; // we're parallelizing on columns here, so seq x token x column will be the unit

    // chunks per thread
    ਅਚਲ int64_t dr = (nr + nth - 1)/nth;

    // chunk range for this thread
    ਅਚਲ int64_t ir0 = dr*ith;
    ਅਚਲ int64_t ir1 = MIN(ir0 + dr, nr);

    ਅਚਲ ਦਸ਼ਮਲਵ * A = (ਅਚਲ ਦਸ਼ਮਲਵ *) src0->ਅੰਕੜੇ;  // [n, n, B1, B2]
    ਅਚਲ ਦਸ਼ਮਲਵ * B = (ਅਚਲ ਦਸ਼ਮਲਵ *) src1->ਅੰਕੜੇ;  // [n, k, B1, B2]
          ਦਸ਼ਮਲਵ * X = (      ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;   // [n, k, B1, B2]

    ਲਈ (int64_t ir = ir0; ir < ir1; ++ir) {
        ਅਚਲ int64_t i03 = ir/(ne02*k);
        ਅਚਲ int64_t i02 = (ir - i03*ne02*k)/k;
        ਅਚਲ int64_t i01 = (ir - i03*ne02*k - i02*k);

        ਅਚਲ ਦਸ਼ਮਲਵ * A_batch = A + i02 * nb02 / ਆਕਾਰ(ਦਸ਼ਮਲਵ) + i03 * nb03 / ਆਕਾਰ(ਦਸ਼ਮਲਵ);
        ਅਚਲ ਦਸ਼ਮਲਵ * B_batch = B + i02 * nb12 / ਆਕਾਰ(ਦਸ਼ਮਲਵ) + i03 * nb13 / ਆਕਾਰ(ਦਸ਼ਮਲਵ);

        ਦਸ਼ਮਲਵ * X_batch = X + i02 * nb2 / ਆਕਾਰ(ਦਸ਼ਮਲਵ) + i03 * nb3 / ਆਕਾਰ(ਦਸ਼ਮਲਵ);

        ਲਈ (int64_t i00 = 0; i00 < n; ++i00) {
            ਦਸ਼ਮਲਵ ਜੋੜ = 0.0f;
            ਲਈ (int64_t t = 0; t < i00; ++t) {
                ਜੋੜ += A_batch[i00 * n + t] * X_batch[t * k + i01];
            }

            ਅਚਲ ਦਸ਼ਮਲਵ diag = A_batch[i00 * n + i00];
            ਪ੍ਰਤਿਜ੍ਞਾ(diag != 0.0f && "Zero diagonal in triangular matrix");

            X_batch[i00 * k + i01] = (B_batch[i00 * k + i01] - ਜੋੜ) / diag;
        }
    }
}

ਖਾਲੀ ggml_compute_forward_solve_tri(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    ਜੇ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        ggml_compute_forward_solve_tri_f32(params, dst);
    } ਨਹੀਂਤਾਂ {
        GGML_ABORT("fatal error");
    }
}

// ggml_compute_forward_gated_delta_net
ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_gated_delta_net_one_chunk(
    ਅਚਲ ggml_compute_params * params,
    ggml_tensor * dst,
    int64_t ir0,
    int64_t ir1) {

    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    ਅਚਲ int64_t S_v      = src_v->ne[0];
    ਅਚਲ int64_t H        = src_v->ne[1];
    ਅਚਲ int64_t n_ਟੋਕਨ = src_v->ne[2];
    ਅਚਲ int64_t n_seqs   = src_v->ne[3];

    GGML_ASSERT(ggml_is_contiguous_ਪੰਕਤੀਆਂ(src_q));
    GGML_ASSERT(ggml_is_contiguous_ਪੰਕਤੀਆਂ(src_k));
    GGML_ASSERT(ggml_is_contiguous_ਪੰਕਤੀਆਂ(src_v));
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

    ਅਚਲ ਤਾਰਕਿਕ kda = (neg0 == S_v);

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    ਅਚਲ int64_t K = ggml_get_op_params_i32(dst, 0);
    GGML_ASSERT(K >= 1);
    // per-seq stride in floats (seq s starts at state + s * seq_stride)
    ਅਚਲ int64_t state_seq_stride = src_state->nb[3] / ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    ਅਚਲ int64_t ਪ੍ਰਤੀ_thread = S_v + (K > 1 ? S_v * S_v : 0);
    ਅਚਲ ਪੂਰਨ ith = params->ith;

    ਦਸ਼ਮਲਵ * delta       = (ਦਸ਼ਮਲਵ *)params->wdata + ith * ਪ੍ਰਤੀ_thread + CACHE_ਸਤਰ_ਮਾਪ_F32;
    ਦਸ਼ਮਲਵ * state_work  = K > 1 ? (delta + S_v) : ਖਾਲੀ_ਸੂਚਕ;

    // output layout: [attn_scores | new_states]
    // attn_scores: S_v * H * n_tokens * n_seqs    floats
    // new_states:  S_v * S_v * H * n_seqs * K     floats  (K snapshot slots; last min(n_tokens, K))
    ਅਚਲ int64_t attn_score_elems    = S_v * H * n_ਟੋਕਨ * n_seqs;
    ਅਚਲ int64_t state_ਮਾਪ_ਪ੍ਰਤੀ_snap = S_v * S_v * H * n_seqs;
    ਦਸ਼ਮਲਵ * attn_ਬਾਹਰਲਾ_ਅਧਾਰ  = (ਦਸ਼ਮਲਵ *)dst->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * state_ਬਾਹਰਲਾ_ਅਧਾਰ = (ਦਸ਼ਮਲਵ *)dst->ਅੰਕੜੇ + attn_score_elems;

    // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
    // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.

    ਅਚਲ ਦਸ਼ਮਲਵ * state_ਵਿੱਚ_ਅਧਾਰ = (ਅਚਲ ਦਸ਼ਮਲਵ *)src_state->ਅੰਕੜੇ;

  //const int64_t rq1 = nev1 / neq1;
  //const int64_t rk1 = nev1 / nek1;
    ਅਚਲ int64_t rq3 = nev3 / neq3;
    ਅਚਲ int64_t rk3 = nev3 / nek3;

    ਅਚਲ ਦਸ਼ਮਲਵ scale = 1.0f / sqrtf((ਦਸ਼ਮਲਵ) S_v);

    ਲਈ (int64_t ir = ir0; ir < ir1; ++ir) {
        ਅਚਲ int64_t iv1 = ir % H; // head_index
        ਅਚਲ int64_t iv3 = ir / H; // sequence

        ਅਚਲ int64_t iq1 = iv1 % neq1;
        ਅਚਲ int64_t ik1 = iv1 % nek1;

        ਅਚਲ int64_t iq3 = iv3 / rq3;
        ਅਚਲ int64_t ik3 = iv3 / rk3;

        // For K=1, write directly to the single output slot to avoid an extra memcpy at the end.
        // For K>1, work in scratch and copy out per-token when the slot is in range.
        ਦਸ਼ਮਲਵ * s_ਬਾਹਰਲਾ = (K > 1)
            ? state_work
            : state_ਬਾਹਰਲਾ_ਅਧਾਰ + (iv3 * H + iv1) * S_v * S_v;

        // copy input state into the working buffer and operate in-place
        // state layout [S_v, S_v, H, n_seqs]: seq iv3 starts at iv3 * state_seq_stride.
        ਅਚਲ ਦਸ਼ਮਲਵ * s_ਵਿੱਚ = state_ਵਿੱਚ_ਅਧਾਰ + iv3 * state_seq_stride + iv1 * S_v * S_v;
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(s_ਬਾਹਰਲਾ, s_ਵਿੱਚ, S_v * S_v * ਆਕਾਰ(ਦਸ਼ਮਲਵ));

        // attn output pointer for first token of this (head, seq)
        ਦਸ਼ਮਲਵ * attn_ਅੰਕੜੇ = attn_ਬਾਹਰਲਾ_ਅਧਾਰ + (iv3 * n_ਟੋਕਨ * H + iv1) * S_v;

        ਲਈ (int64_t t = 0; t < n_ਟੋਕਨ; t++) {
            ਅਚਲ ਦਸ਼ਮਲਵ * q_d = (ਅਚਲ ਦਸ਼ਮਲਵ *)((ਅਚਲ ਅੱਖਰ *)src_q->ਅੰਕੜੇ + iq3 * nbq3 + t * nbq2 + iq1 * nbq1);
            ਅਚਲ ਦਸ਼ਮਲਵ * k_d = (ਅਚਲ ਦਸ਼ਮਲਵ *)((ਅਚਲ ਅੱਖਰ *)src_k->ਅੰਕੜੇ + ik3 * nbk3 + t * nbk2 + ik1 * nbk1);
            ਅਚਲ ਦਸ਼ਮਲਵ * v_d = (ਅਚਲ ਦਸ਼ਮਲਵ *)((ਅਚਲ ਅੱਖਰ *)src_v->ਅੰਕੜੇ + iv3 * nbv3 + t * nbv2 + iv1 * nbv1);

            ਅਚਲ ਦਸ਼ਮਲਵ beta_val = *(ਅਚਲ ਦਸ਼ਮਲਵ *)((ਅਚਲ ਅੱਖਰ *)src_beta->ਅੰਕੜੇ + iv3 * nbb3 + t * nbb2 + iv1 * nbb1);
            ਅਚਲ ਦਸ਼ਮਲਵ * g_d    =  (ਅਚਲ ਦਸ਼ਮਲਵ *)((ਅਚਲ ਅੱਖਰ *)src_g->ਅੰਕੜੇ    + iv3 * nbg3 + t * nbg2 + iv1 * nbg1);

            // state is stored transposed: s_out[j*S_v + i] = S[i][j]
            // so row j of s_out = column j of S (contiguous access)

            ਜੇ (kda) {
                // precompute exp(g) into delta scratch (reused below)
                ਲਈ (int64_t i = 0; i < S_v; ++i) {
                    delta[i] = expf(g_d[i]);
                }
                // S[i][:] *= exp(g[i]) => for each row j of M: M[j][i] *= exp(g[i])
                ਲਈ (int64_t j = 0; j < S_v; ++j) {
                    ggml_vec_mul_f32(S_v, &s_ਬਾਹਰਲਾ[j * S_v], &s_ਬਾਹਰਲਾ[j * S_v], delta);
                }
            } ਨਹੀਂਤਾਂ {
                ggml_vec_scale_f32(S_v * S_v, s_ਬਾਹਰਲਾ, expf(g_d[0]));
            }

            // delta[j] = sum_i S[i][j] * k[i] = dot(row j of M, k)
            ਲਈ (int64_t j = 0; j < S_v; ++j) {
                ਦਸ਼ਮਲਵ ਜੋੜ = 0.0f;
                ggml_vec_dot_f32(S_v, &ਜੋੜ, 0, &s_ਬਾਹਰਲਾ[j * S_v], 0, k_d, 0, 1);
                delta[j] = (v_d[j] - ਜੋੜ) * beta_val;
            }

            // outer product: S[i][j] += k[i] * delta[j] => M[j][i] += delta[j] * k[i]
            ਲਈ (int64_t j = 0; j < S_v; ++j) {
                ggml_vec_mad_f32(S_v, &s_ਬਾਹਰਲਾ[j * S_v], k_d, delta[j]);
            }

            // attn_out[j] = sum_i S[i][j] * q[i] = dot(row j of M, q)
            ਲਈ (int64_t j = 0; j < S_v; ++j) {
                ਦਸ਼ਮਲਵ ਜੋੜ = 0.0f;
                ggml_vec_dot_f32(S_v, &ਜੋੜ, 0, &s_ਬਾਹਰਲਾ[j * S_v], 0, q_d, 0, 1);
                attn_ਅੰਕੜੇ[j] = ਜੋੜ * scale;
            }

            attn_ਅੰਕੜੇ += S_v * H; // advance to next token

            ਜੇ (K > 1) {
                ਅਚਲ int64_t ਨਿਸ਼ਾਨਾ_ਥਾਂ = n_ਟੋਕਨ - 1 - t;
                ਜੇ (ਨਿਸ਼ਾਨਾ_ਥਾਂ >= 0 && ਨਿਸ਼ਾਨਾ_ਥਾਂ < K) {
                    ਦਸ਼ਮਲਵ * curr_state_o = state_ਬਾਹਰਲਾ_ਅਧਾਰ + ਨਿਸ਼ਾਨਾ_ਥਾਂ * state_ਮਾਪ_ਪ੍ਰਤੀ_snap +
                                     (iv3 * H + iv1) * S_v * S_v;
                    ਸਮ੍ਰਿਤੀ_ਨਕਲ(curr_state_o, s_ਬਾਹਰਲਾ, S_v * S_v * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                }
            }
        }
    }
}


ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_gated_delta_net_f32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ggml_tensor * V = dst->src[2];
    int64_t nr = V->ne[1] * V->ne[3];

    // disable for NUMA
    ਅਚਲ ਤਾਰਕਿਕ disable_chunking = ggml_is_numa();

    ਪੂਰਨ nth = params->nth;
    ਪੂਰਨ ith = params->ith;

    // 4x chunks per thread
    ਪੂਰਨ nth_scaled = nth * 4;
    int64_t chunk_ਮਾਪ = (nr + nth_scaled - 1) / nth_scaled;
    int64_t nchunk     = (nr + chunk_ਮਾਪ - 1) / chunk_ਮਾਪ;

    ਜੇ (nth == 1 || nchunk < nth || disable_chunking) {
      nchunk = nth;
    }

    ਜੇ (ith == 0) {
      ggml_threadpool_chunk_ਸੈੱਟ(params->threadpool, nth);
    }

    ggml_barrier(params->threadpool);

    ਅਚਲ int64_t dr = (nr + nchunk - 1) / nchunk;

    ਪੂਰਨ current_chunk = ith;

    ਜਦੋਂ (current_chunk < nchunk) {
        ਅਚਲ int64_t ir0 = dr * current_chunk;
        ਅਚਲ int64_t ir1 = MIN(ir0 + dr, nr);

        ggml_compute_forward_gated_delta_net_one_chunk(params, dst, ir0, ir1);
        current_chunk = ggml_threadpool_chunk_ਜੋੜੋ(params->threadpool, 1);
    }
}

ਖਾਲੀ ggml_compute_forward_gated_delta_net(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_gated_delta_net_f32(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}


// ggml_compute_forward_dsv4_hc_comb

ਸਥਿਰ ਖਾਲੀ ggml_dsv4_hc_comb_norm_ਖਾਨੇ_ਗਿਣਤੀ(ਦਸ਼ਮਲਵ * comb, ਦਸ਼ਮਲਵ eps) {
    constexpr int64_t hc = 4;

    ਲਈ (int64_t idst = 0; idst < hc; ++idst) {
        ਦਸ਼ਮਲਵ ਜੋੜ = eps;
        ਲਈ (int64_t isrc = 0; isrc < hc; ++isrc) {
            ਜੋੜ += comb[idst + hc*isrc];
        }

        ਅਚਲ ਦਸ਼ਮਲਵ inv_ਜੋੜ = 1.0f / ਜੋੜ;
        ਲਈ (int64_t isrc = 0; isrc < hc; ++isrc) {
            comb[idst + hc*isrc] *= inv_ਜੋੜ;
        }
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_dsv4_hc_comb_norm_ਪੰਕਤੀਆਂ(ਦਸ਼ਮਲਵ * comb, ਦਸ਼ਮਲਵ eps) {
    constexpr int64_t hc = 4;

    ਲਈ (int64_t isrc = 0; isrc < hc; ++isrc) {
        ਦਸ਼ਮਲਵ ਜੋੜ = eps;
        ਲਈ (int64_t idst = 0; idst < hc; ++idst) {
            ਜੋੜ += comb[idst + hc*isrc];
        }

        ਅਚਲ ਦਸ਼ਮਲਵ inv_ਜੋੜ = 1.0f / ਜੋੜ;
        ਲਈ (int64_t idst = 0; idst < hc; ++idst) {
            comb[idst + hc*isrc] *= inv_ਜੋੜ;
        }
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_dsv4_hc_comb_f32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * mixes = dst->src[0];
    ਅਚਲ ggml_tensor * scale = dst->src[1];
    ਅਚਲ ggml_tensor * ਅਧਾਰ  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(ਅਧਾਰ->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    constexpr int64_t hc = 4;
    constexpr int64_t comb_ਵਿੱਥ = 2*hc;
    constexpr int64_t hc_mix_dim = (2 + hc)*hc;

    ਅਚਲ int64_t n_ਟੋਕਨ = mixes->ne[1];

    GGML_ASSERT(mixes->ne[0] == hc_mix_dim);
    GGML_ASSERT(dst->ne[0] == hc);
    GGML_ASSERT(dst->ne[1] == hc);
    GGML_ASSERT(dst->ne[2] == n_ਟੋਕਨ);
    GGML_ASSERT(scale->ne[0] >= 3);
    GGML_ASSERT(ਅਧਾਰ->ne[0] == hc_mix_dim);

    GGML_TENSOR_LOCALS(size_t, nbm, mixes, nb);
    GGML_TENSOR_LOCALS(size_t, nbs, scale, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, ਅਧਾਰ,  nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,   nb);

    ਅਚਲ ਦਸ਼ਮਲਵ eps = ggml_get_op_params_f32(dst, 0);
    ਅਚਲ int32_t n_iter = ggml_get_op_params_i32(dst, 1);
    GGML_ASSERT(n_iter > 0);

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ int64_t dr  = (n_ਟੋਕਨ + nth - 1) / nth;
    ਅਚਲ int64_t it0 = dr * ith;
    ਅਚਲ int64_t it1 = MIN(it0 + dr, n_ਟੋਕਨ);

    ਅਚਲ ਦਸ਼ਮਲਵ scale_comb = *(ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) scale->ਅੰਕੜੇ + 2*nbs0);

    ਲਈ (int64_t it = it0; it < it1; ++it) {
        ਦਸ਼ਮਲਵ comb[hc*hc];

        ਲਈ (int64_t isrc = 0; isrc < hc; ++isrc) {
            ਦਸ਼ਮਲਵ max = -INFINITY;
            ਲਈ (int64_t idst = 0; idst < hc; ++idst) {
                ਅਚਲ int64_t idx = idst + hc*isrc;
                ਅਚਲ ਦਸ਼ਮਲਵ xv = *(ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) mixes->ਅੰਕੜੇ + (comb_ਵਿੱਥ + idx)*nbm0 + it*nbm1);
                ਅਚਲ ਦਸ਼ਮਲਵ bv = *(ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) ਅਧਾਰ->ਅੰਕੜੇ  + (comb_ਵਿੱਥ + idx)*nbb0);
                ਅਚਲ ਦਸ਼ਮਲਵ v = xv * scale_comb + bv;
                comb[idx] = v;
                max = MAX(max, v);
            }

            ਦਸ਼ਮਲਵ ਜੋੜ = 0.0f;
            ਲਈ (int64_t idst = 0; idst < hc; ++idst) {
                ਅਚਲ int64_t idx = idst + hc*isrc;
                ਅਚਲ ਦਸ਼ਮਲਵ v = expf(comb[idx] - max);
                comb[idx] = v;
                ਜੋੜ += v;
            }

            ਅਚਲ ਦਸ਼ਮਲਵ inv_ਜੋੜ = 1.0f / ਜੋੜ;
            ਲਈ (int64_t idst = 0; idst < hc; ++idst) {
                ਅਚਲ int64_t idx = idst + hc*isrc;
                comb[idx] = comb[idx] * inv_ਜੋੜ + eps;
            }
        }

        ggml_dsv4_hc_comb_norm_ਖਾਨੇ_ਗਿਣਤੀ(comb, eps);
        ਲਈ (int32_t i = 1; i < n_iter; ++i) {
            ggml_dsv4_hc_comb_norm_ਪੰਕਤੀਆਂ(comb, eps);
            ggml_dsv4_hc_comb_norm_ਖਾਨੇ_ਗਿਣਤੀ(comb, eps);
        }

        ਲਈ (int64_t isrc = 0; isrc < hc; ++isrc) {
            ਲਈ (int64_t idst = 0; idst < hc; ++idst) {
                ਅਚਲ int64_t idx = idst + hc*isrc;
                *(ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + idst*nbd0 + isrc*nbd1 + it*nbd2) = comb[idx];
            }
        }
    }
}

ਖਾਲੀ ggml_compute_forward_dsv4_hc_comb(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_dsv4_hc_comb_f32(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_dsv4_hc_pre

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_dsv4_hc_pre_f32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * x       = dst->src[0];
    ਅਚਲ ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ਅਚਲ int64_t n_embd   = x->ne[0];
    ਅਚਲ int64_t hc       = x->ne[1];
    ਅਚਲ int64_t n_ਟੋਕਨ = x->ne[2];

    ਅਚਲ ਦਸ਼ਮਲਵ scale = ggml_get_op_params_f32(dst, 0);
    ਅਚਲ ਤਾਰਕਿਕ  gated = ggml_get_op_params_i32(dst, 1) != 0;

    GGML_ASSERT(dst->ne[0] == n_embd);
    GGML_ASSERT(dst->ne[1] == n_ਟੋਕਨ);
    ਜੇ (gated) {
        GGML_ASSERT(weights->ne[0] == n_embd);
        GGML_ASSERT(weights->ne[1] == hc);
        GGML_ASSERT(weights->ne[2] == n_ਟੋਕਨ);
    } ਨਹੀਂਤਾਂ {
        GGML_ASSERT(weights->ne[0] == hc);
        GGML_ASSERT(weights->ne[1] == n_ਟੋਕਨ);
    }

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ int64_t nr  = n_embd * n_ਟੋਕਨ;
    ਅਚਲ int64_t dr  = (nr + nth - 1) / nth;
    ਅਚਲ int64_t ir0 = dr * ith;
    ਅਚਲ int64_t ir1 = MIN(ir0 + dr, nr);

    ਲਈ (int64_t ir = ir0; ir < ir1; ++ir) {
        ਅਚਲ int64_t i0 = ir % n_embd;
        ਅਚਲ int64_t it = ir / n_embd;

        ਦਸ਼ਮਲਵ ਜੋੜ = 0.0f;
        ਲਈ (int64_t ih = 0; ih < hc; ++ih) {
            ਅਚਲ ਦਸ਼ਮਲਵ xv = *(ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) x->ਅੰਕੜੇ + i0*nbx0 + ih*nbx1 + it*nbx2);
            ਦਸ਼ਮਲਵ wv;
            ਜੇ (gated) {
                ਅਚਲ ਦਸ਼ਮਲਵ gv = *(ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) weights->ਅੰਕੜੇ + i0*nbw0 + ih*nbw1 + it*nbw2);
                wv = 1.0f / (1.0f + expf(-gv));
            } ਨਹੀਂਤਾਂ {
                wv = *(ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) weights->ਅੰਕੜੇ + ih*nbw0 + it*nbw1);
            }
            ਜੋੜ += xv * wv;
        }

        *(ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i0*nbd0 + it*nbd1) = scale * ਜੋੜ;
    }
}

ਖਾਲੀ ggml_compute_forward_dsv4_hc_pre(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_dsv4_hc_pre_f32(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_dsv4_hc_post

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_dsv4_hc_post_f32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * x        = dst->src[0];
    ਅਚਲ ggml_tensor * residual = dst->src[1];
    ਅਚਲ ggml_tensor * post     = dst->src[2];
    ਅਚਲ ggml_tensor * comb     = dst->src[3];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ਅਚਲ int64_t n_embd   = x->ne[0];
    ਅਚਲ int64_t n_ਟੋਕਨ = x->ne[1];
    ਅਚਲ int64_t hc       = residual->ne[1];

    GGML_ASSERT(dst->ne[0] == n_embd);
    GGML_ASSERT(dst->ne[1] == hc);
    GGML_ASSERT(dst->ne[2] == n_ਟੋਕਨ);
    GGML_ASSERT(residual->ne[0] == n_embd);
    GGML_ASSERT(residual->ne[2] == n_ਟੋਕਨ);
    GGML_ASSERT(post->ne[0] == hc);
    GGML_ASSERT(post->ne[1] == n_ਟੋਕਨ);

    // comb == NULL: identity mixing, each stream keeps its own residual
    size_t nbc0 = 0;
    size_t nbc1 = 0;
    size_t nbc2 = 0;
    ਜੇ (comb) {
        GGML_ASSERT(comb->type == GGML_TYPE_F32);
        GGML_ASSERT(comb->ne[0] == hc);
        GGML_ASSERT(comb->ne[1] == hc);
        GGML_ASSERT(comb->ne[2] == n_ਟੋਕਨ);
        nbc0 = comb->nb[0];
        nbc1 = comb->nb[1];
        nbc2 = comb->nb[2];
    }

    GGML_TENSOR_LOCALS(size_t, nbx, x,        nb);
    GGML_TENSOR_LOCALS(size_t, nbr, residual, nb);
    GGML_TENSOR_LOCALS(size_t, nbp, post,     nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,      nb);

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ int64_t nr  = n_embd * hc * n_ਟੋਕਨ;
    ਅਚਲ int64_t dr  = (nr + nth - 1) / nth;
    ਅਚਲ int64_t ir0 = dr * ith;
    ਅਚਲ int64_t ir1 = MIN(ir0 + dr, nr);

    ਲਈ (int64_t ir = ir0; ir < ir1; ++ir) {
        ਅਚਲ int64_t i0     = ir % n_embd;
        ਅਚਲ int64_t idst   = (ir / n_embd) % hc;
        ਅਚਲ int64_t it     = ir / (n_embd * hc);

        ਅਚਲ ਦਸ਼ਮਲਵ xv = *(ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) x->ਅੰਕੜੇ    + i0*nbx0 + it*nbx1);
        ਅਚਲ ਦਸ਼ਮਲਵ pv = *(ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) post->ਅੰਕੜੇ + idst*nbp0 + it*nbp1);

        ਦਸ਼ਮਲਵ ਜੋੜ = xv * pv;
        ਜੇ (comb) {
            ਲਈ (int64_t isrc = 0; isrc < hc; ++isrc) {
                ਅਚਲ ਦਸ਼ਮਲਵ rv = *(ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) residual->ਅੰਕੜੇ + i0*nbr0 + isrc*nbr1 + it*nbr2);
                ਅਚਲ ਦਸ਼ਮਲਵ cv = *(ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) comb->ਅੰਕੜੇ     + idst*nbc0 + isrc*nbc1 + it*nbc2);
                ਜੋੜ += rv * cv;
            }
        } ਨਹੀਂਤਾਂ {
            ਜੋੜ += *(ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) residual->ਅੰਕੜੇ + i0*nbr0 + idst*nbr1 + it*nbr2);
        }

        *(ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i0*nbd0 + idst*nbd1 + it*nbd2) = ਜੋੜ;
    }
}

ਖਾਲੀ ggml_compute_forward_dsv4_hc_post(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_dsv4_hc_post_f32(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_top_k_mask

ਖਾਲੀ ggml_compute_forward_ਸਿਖਰ_k_ਨਕਾਬ(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * ਨਕਾਬ   = dst->src[0];
    ਅਚਲ ggml_tensor * ਸਿਖਰ_k  = dst->src[1];
    ਅਚਲ ggml_tensor * score  = dst->src[2];
    ਅਚਲ ggml_tensor * ਖੰਡਾਂ = dst->src[3];
    ਅਚਲ ggml_tensor * extra  = dst->src[4];

    GGML_ASSERT(dst->type == ਨਕਾਬ->type);
    GGML_ASSERT(dst->nb[0] == ggml_type_ਮਾਪ(dst->type));

    ਅਚਲ int64_t n_kv    = ਨਕਾਬ->ne[0];
    ਅਚਲ int64_t ne1     = dst->ne[1];
    ਅਚਲ int64_t ne2     = dst->ne[2];
    ਅਚਲ int64_t n_ਪੰਕਤੀਆਂ  = ggml_nrows(dst);
    ਅਚਲ int64_t n_ਸਿਖਰ   = ਸਿਖਰ_k->ne[0];
    ਅਚਲ int64_t n_blk   = ਖੰਡਾਂ ? ਖੰਡਾਂ->ne[0] : 1;
    ਅਚਲ int64_t n_pick  = ਖੰਡਾਂ ? ਖੰਡਾਂ->ne[1] : score ? score->ne[0] : n_kv;
    ਅਚਲ int64_t n_extra = extra ? extra->ne[0] : 0;

    ਅਚਲ size_t ts = ggml_type_ਮਾਪ(dst->type);

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਲਈ (int64_t ir = ith; ir < n_ਪੰਕਤੀਆਂ; ir += nth) {
        ਅਚਲ int64_t i1 = ir % ne1;
        ਅਚਲ int64_t i2 = (ir / ne1) % ne2;
        ਅਚਲ int64_t i3 = ir / (ne1*ne2);

        ਅਚਲ ਅੱਖਰ * ਨਕਾਬ_ਪੰਕਤੀ = (ਅਚਲ ਅੱਖਰ *) ਨਕਾਬ->ਅੰਕੜੇ
            + (i1 % ਨਕਾਬ->ne[1])*ਨਕਾਬ->nb[1] + (i2 % ਨਕਾਬ->ne[2])*ਨਕਾਬ->nb[2] + (i3 % ਨਕਾਬ->ne[3])*ਨਕਾਬ->nb[3];
        ਅੱਖਰ * dst_ਪੰਕਤੀ = (ਅੱਖਰ *) dst->ਅੰਕੜੇ + i1*dst->nb[1] + i2*dst->nb[2] + i3*dst->nb[3];

        ਜੇ (dst->type == GGML_TYPE_F16) {
            ਅਚਲ ggml_fp16_t ninf = GGML_CPU_FP32_ਮੰਜ਼ਿਲ_FP16(-INFINITY);
            ਲਈ (int64_t c = 0; c < n_kv; ++c) {
                ((ggml_fp16_t *) dst_ਪੰਕਤੀ)[c] = ninf;
            }
        } ਨਹੀਂਤਾਂ {
            ਲਈ (int64_t c = 0; c < n_kv; ++c) {
                ((ਦਸ਼ਮਲਵ *) dst_ਪੰਕਤੀ)[c] = -INFINITY;
            }
        }

        ਸਵੈਚਾਲਿਤ keep = [&](int32_t c) {
            ਜੇ (c >= 0 && c < n_kv) {
                ਸਮ੍ਰਿਤੀ_ਨਕਲ(dst_ਪੰਕਤੀ + c*ts, ਨਕਾਬ_ਪੰਕਤੀ + c*ts, ts);
            }
        };

        ਅਚਲ ਅੱਖਰ * ਸਿਖਰ_k_ਪੰਕਤੀ = (ਅਚਲ ਅੱਖਰ *) ਸਿਖਰ_k->ਅੰਕੜੇ + i1*ਸਿਖਰ_k->nb[1] + i2*ਸਿਖਰ_k->nb[2] + i3*ਸਿਖਰ_k->nb[3];

        ਲਈ (int64_t j = 0; j < n_ਸਿਖਰ; ++j) {
            ਅਚਲ int32_t p = *(ਅਚਲ int32_t *) (ਸਿਖਰ_k_ਪੰਕਤੀ + j*ਸਿਖਰ_k->nb[0]);
            ਜੇ (p < 0 || p >= n_pick) {
                ਜਾਰੀ;
            }
            ਜੇ (score) {
                ਅਚਲ ਦਸ਼ਮਲਵ s = *(ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) score->ਅੰਕੜੇ
                    + p*score->nb[0] + i1*score->nb[1] + i2*score->nb[2] + i3*score->nb[3]);
                ਜੇ (!(s > -INFINITY)) {
                    ਜਾਰੀ;
                }
            }
            ਜੇ (!ਖੰਡਾਂ) {
                keep(p);
                ਜਾਰੀ;
            }
            ਲਈ (int64_t k = 0; k < n_blk; ++k) {
                keep(*(ਅਚਲ int32_t *) ((ਅਚਲ ਅੱਖਰ *) ਖੰਡਾਂ->ਅੰਕੜੇ + k*ਖੰਡਾਂ->nb[0] + p*ਖੰਡਾਂ->nb[1]));
            }
        }

        ਜੇ (extra) {
            ਅਚਲ ਅੱਖਰ * extra_ਪੰਕਤੀ = (ਅਚਲ ਅੱਖਰ *) extra->ਅੰਕੜੇ + i1*extra->nb[1] + i2*extra->nb[2] + i3*extra->nb[3];
            ਲਈ (int64_t j = 0; j < n_extra; ++j) {
                keep(*(ਅਚਲ int32_t *) (extra_ਪੰਕਤੀ + j*extra->nb[0]));
            }
        }
    }
}

// ggml_compute_forward_rwkv_wkv7

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_rwkv_wkv7_f32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {
    ਅਚਲ int64_t T = dst->src[1]->ne[2];
    ਅਚਲ int64_t C = dst->ne[0];
    ਅਚਲ int64_t HEADS = dst->src[1]->ne[1];
    ਅਚਲ int64_t n_seqs = dst->src[6]->ne[1];
    ਅਚਲ int64_t ਸਿਰ_ਮਾਪ = C / HEADS;

    ਦਸ਼ਮਲਵ * dst_ਅੰਕੜੇ = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * state = ((ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ) + C * T;

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ h_ਸ਼ੁਰੂ =  (HEADS * (ith    )) / nth;
    ਅਚਲ ਪੂਰਨ h_ਸਿਰਾ   = ((HEADS * (ith + 1)) / nth < HEADS) ?
                         (HEADS * (ith + 1)) / nth : HEADS;

    ਦਸ਼ਮਲਵ * r = (ਦਸ਼ਮਲਵ *) dst->src[0]->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * w = (ਦਸ਼ਮਲਵ *) dst->src[1]->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * k = (ਦਸ਼ਮਲਵ *) dst->src[2]->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * v = (ਦਸ਼ਮਲਵ *) dst->src[3]->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * a = (ਦਸ਼ਮਲਵ *) dst->src[4]->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * b = (ਦਸ਼ਮਲਵ *) dst->src[5]->ਅੰਕੜੇ;

    int64_t t_stride = HEADS * ਸਿਰ_ਮਾਪ; // Same to C

    int64_t h_stride = C / HEADS;
    GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
    int64_t h_stride_2d = ਸਿਰ_ਮਾਪ * ਸਿਰ_ਮਾਪ;

    #ਜੇ defined(GGML_SIMD)
        #ਜੇ defined(__ARM_FEATURE_SVE) || defined(__riscv_v_intrinsic)
            // scalar Route to scalar implementation       //TODO: Write SVE code and RVV code
            ਲਈ (int64_t t = 0; t < T; t++) {
                int64_t t_ਵਿੱਥ = t * t_stride;
                int64_t state_ਵਿੱਥ = ਸਿਰ_ਮਾਪ * C * (t / (T / n_seqs));
                ਦਸ਼ਮਲਵ * state_cur = state + state_ਵਿੱਥ;
                ਦਸ਼ਮਲਵ * state_prev = t % (T / n_seqs) ? state_cur : (ਦਸ਼ਮਲਵ*)dst->src[6]->ਅੰਕੜੇ + state_ਵਿੱਥ;

                ਲਈ (int64_t h = h_ਸ਼ੁਰੂ; h < h_ਸਿਰਾ; h++) {
                    int64_t h_ਵਿੱਥ = h * h_stride;
                    int64_t t_h_ਵਿੱਥ = t_ਵਿੱਥ + h_ਵਿੱਥ;
                    int64_t h_2d_ਵਿੱਥ = h * h_stride_2d;

                    ਲਈ (int64_t i = 0; i < ਸਿਰ_ਮਾਪ; i++) {
                        int64_t t_h_i_ਵਿੱਥ = t_h_ਵਿੱਥ + i;
                        int64_t h_2d_i_ਵਿੱਥ = h_2d_ਵਿੱਥ + i * h_stride;

                        ਦਸ਼ਮਲਵ v_val = v[t_h_i_ਵਿੱਥ];

                        ਦਸ਼ਮਲਵ sa = 0, ਨਤੀਜਾ = 0;
                        ਲਈ (int64_t j = 0; j < ਸਿਰ_ਮਾਪ; j++) {
                            sa += a[t_h_ਵਿੱਥ + j] * state_prev[h_2d_i_ਵਿੱਥ + j];
                        }

                        ਲਈ (int64_t j = 0; j < ਸਿਰ_ਮਾਪ; j++) {
                            int64_t t_h_j_ਵਿੱਥ = t_h_ਵਿੱਥ + j;
                            int64_t h_2d_i_j_ਵਿੱਥ = h_2d_i_ਵਿੱਥ + j;

                            ਦਸ਼ਮਲਵ r_val = r[t_h_j_ਵਿੱਥ];
                            ਦਸ਼ਮਲਵ w_val = w[t_h_j_ਵਿੱਥ];
                            ਦਸ਼ਮਲਵ k_val = k[t_h_j_ਵਿੱਥ];
                            ਦਸ਼ਮਲਵ b_val = b[t_h_j_ਵਿੱਥ];
                            ਦਸ਼ਮਲਵ kv_val = v_val * k_val;
                            ਦਸ਼ਮਲਵ prev_state_val = state_prev[h_2d_i_j_ਵਿੱਥ];
                            state_cur[h_2d_i_j_ਵਿੱਥ] = prev_state_val * w_val + kv_val + sa * b_val;
                            ਨਤੀਜਾ += state_cur[h_2d_i_j_ਵਿੱਥ] * r_val;
                        }
                        dst_ਅੰਕੜੇ[t_h_i_ਵਿੱਥ] = ਨਤੀਜਾ;
                    }
                }
            }
        #ਨਹੀਂਤਾਂ
            ਲਈ (int64_t t = 0; t < T; t++) {
                int64_t t_ਵਿੱਥ = t * t_stride;
                int64_t state_ਵਿੱਥ = ਸਿਰ_ਮਾਪ * C * (t / (T / n_seqs));
                ਦਸ਼ਮਲਵ * state_cur = state + state_ਵਿੱਥ;
                ਦਸ਼ਮਲਵ * state_prev = t % (T / n_seqs) ? state_cur : (ਦਸ਼ਮਲਵ*)dst->src[6]->ਅੰਕੜੇ + state_ਵਿੱਥ;

                ਲਈ (int64_t h = h_ਸ਼ੁਰੂ; h < h_ਸਿਰਾ; h++) {
                    int64_t h_ਵਿੱਥ = h * h_stride;
                    int64_t t_h_ਵਿੱਥ = t_ਵਿੱਥ + h_ਵਿੱਥ;
                    int64_t h_2d_ਵਿੱਥ = h * h_stride_2d;

                    ਲਈ (int64_t ii = 0; ii < ਸਿਰ_ਮਾਪ; ii++) {
                        int64_t t_h_i_ਵਿੱਥ = t_h_ਵਿੱਥ + ii;
                        int64_t h_2d_i_ਵਿੱਥ = h_2d_ਵਿੱਥ + ii * h_stride;

                        GGML_F32_VEC v_vec = GGML_F32_VEC_SET1(v[t_h_i_ਵਿੱਥ]);

                        ਦਸ਼ਮਲਵ sa = 0;
                        {
                            GGML_F32_VEC ਜੋੜ[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };
                            GGML_F32_VEC ax[GGML_F32_ARR];
                            GGML_F32_VEC ay[GGML_F32_ARR];
                            ਲਈ (int64_t j = 0; j < ਸਿਰ_ਮਾਪ; j += GGML_F32_ਕਦਮ) {
                                ਲਈ (int64_t kk = 0; kk < GGML_F32_ARR; kk++) {
                                    ax[kk] = GGML_F32_VEC_ਲੱਦੋ(&a[t_h_ਵਿੱਥ + j + kk * GGML_F32_EPR]);
                                    ay[kk] = GGML_F32_VEC_ਲੱਦੋ(&state_prev[h_2d_i_ਵਿੱਥ + j + kk * GGML_F32_EPR]);
                                    ਜੋੜ[kk] = GGML_F32_VEC_FMA(ਜੋੜ[kk], ax[kk], ay[kk]);
                                }
                            }
                            GGML_F32_VEC_REDUCE(sa, ਜੋੜ);
                        }

                        GGML_F32_VEC sa_vec = GGML_F32_VEC_SET1(sa);

                        int64_t j = 0;
                        GGML_F32_VEC ਨਤੀਜਾ_vec[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };
                        ਲਈ (; j < ਸਿਰ_ਮਾਪ; j += GGML_F32_ਕਦਮ) {
                            ਲਈ (int64_t kk = 0; kk < GGML_F32_ARR; kk++) {
                                int64_t t_h_j_ਵਿੱਥ = t_h_ਵਿੱਥ + j + kk * GGML_F32_EPR;
                                int64_t h_2d_i_j_ਵਿੱਥ = h_2d_i_ਵਿੱਥ + j + kk * GGML_F32_EPR;

                                GGML_F32_VEC r_vec = GGML_F32_VEC_ਲੱਦੋ(&r[t_h_j_ਵਿੱਥ]);
                                GGML_F32_VEC w_vec = GGML_F32_VEC_ਲੱਦੋ(&w[t_h_j_ਵਿੱਥ]);
                                GGML_F32_VEC k_vec = GGML_F32_VEC_ਲੱਦੋ(&k[t_h_j_ਵਿੱਥ]);
                                GGML_F32_VEC b_vec = GGML_F32_VEC_ਲੱਦੋ(&b[t_h_j_ਵਿੱਥ]);

                                k_vec = GGML_F32_VEC_MUL(v_vec, k_vec);

                                GGML_F32_VEC state_vec = GGML_F32_VEC_ਲੱਦੋ(&state_prev[h_2d_i_j_ਵਿੱਥ]);
                                // kv + s * decay + sa * b
                                state_vec = GGML_F32_VEC_FMA(k_vec, state_vec, w_vec);
                                state_vec = GGML_F32_VEC_FMA(state_vec, sa_vec, b_vec);
                                GGML_F32_VEC_STORE(&state_cur[h_2d_i_j_ਵਿੱਥ], state_vec);

                                ਨਤੀਜਾ_vec[kk] = GGML_F32_VEC_FMA(ਨਤੀਜਾ_vec[kk], state_vec, r_vec);
                            }
                        }
                        GGML_F32_VEC_REDUCE(dst_ਅੰਕੜੇ[t_h_i_ਵਿੱਥ], ਨਤੀਜਾ_vec);

                        // There shouldn't be left-overs though.
                        ਲਈ (; j < ਸਿਰ_ਮਾਪ; j++) {
                            int64_t t_h_j_ਵਿੱਥ = t_h_ਵਿੱਥ + j;
                            int64_t h_2d_i_j_ਵਿੱਥ = h_2d_i_ਵਿੱਥ + j;

                            ਦਸ਼ਮਲਵ r_val = r[t_h_j_ਵਿੱਥ];
                            ਦਸ਼ਮਲਵ w_val = w[t_h_j_ਵਿੱਥ];
                            ਦਸ਼ਮਲਵ k_val = k[t_h_j_ਵਿੱਥ];
                            ਦਸ਼ਮਲਵ b_val = b[t_h_j_ਵਿੱਥ];
                            ਦਸ਼ਮਲਵ kv_val = v[t_h_i_ਵਿੱਥ] * k_val;

                            ਦਸ਼ਮਲਵ prev_state_val = state_prev[h_2d_i_j_ਵਿੱਥ];
                            state_cur[h_2d_i_j_ਵਿੱਥ] = prev_state_val * w_val + kv_val + sa * b_val;
                            dst_ਅੰਕੜੇ[t_h_i_ਵਿੱਥ] += state_cur[h_2d_i_j_ਵਿੱਥ] * r_val;
                        }
                    }
                }
            }
        #ਜੇ_ਅੰਤ
    #ਨਹੀਂਤਾਂ
        ਲਈ (int64_t t = 0; t < T; t++) {
            int64_t t_ਵਿੱਥ = t * t_stride;
            int64_t state_ਵਿੱਥ = ਸਿਰ_ਮਾਪ * C * (t / (T / n_seqs));
            ਦਸ਼ਮਲਵ * state_cur = state + state_ਵਿੱਥ;
            ਦਸ਼ਮਲਵ * state_prev = t % (T / n_seqs) ? state_cur : (ਦਸ਼ਮਲਵ*)dst->src[6]->ਅੰਕੜੇ + state_ਵਿੱਥ;

            ਲਈ (int64_t h = h_ਸ਼ੁਰੂ; h < h_ਸਿਰਾ; h++) {
                int64_t h_ਵਿੱਥ = h * h_stride;
                int64_t t_h_ਵਿੱਥ = t_ਵਿੱਥ + h_ਵਿੱਥ;
                int64_t h_2d_ਵਿੱਥ = h * h_stride_2d;

                ਲਈ (int64_t i = 0; i < ਸਿਰ_ਮਾਪ; i++) {
                    int64_t t_h_i_ਵਿੱਥ = t_h_ਵਿੱਥ + i;
                    int64_t h_2d_i_ਵਿੱਥ = h_2d_ਵਿੱਥ + i * h_stride;

                    ਦਸ਼ਮਲਵ v_val = v[t_h_i_ਵਿੱਥ];

                    ਦਸ਼ਮਲਵ sa = 0, ਨਤੀਜਾ = 0;
                    ਲਈ (int64_t j = 0; j < ਸਿਰ_ਮਾਪ; j++) {
                        sa += a[t_h_ਵਿੱਥ + j] * state_prev[h_2d_i_ਵਿੱਥ + j];
                    }

                    ਲਈ (int64_t j = 0; j < ਸਿਰ_ਮਾਪ; j++) {
                        int64_t t_h_j_ਵਿੱਥ = t_h_ਵਿੱਥ + j;
                        int64_t h_2d_i_j_ਵਿੱਥ = h_2d_i_ਵਿੱਥ + j;

                        ਦਸ਼ਮਲਵ r_val = r[t_h_j_ਵਿੱਥ];
                        ਦਸ਼ਮਲਵ w_val = w[t_h_j_ਵਿੱਥ];
                        ਦਸ਼ਮਲਵ k_val = k[t_h_j_ਵਿੱਥ];
                        ਦਸ਼ਮਲਵ b_val = b[t_h_j_ਵਿੱਥ];
                        ਦਸ਼ਮਲਵ kv_val = v_val * k_val;
                        ਦਸ਼ਮਲਵ prev_state_val = state_prev[h_2d_i_j_ਵਿੱਥ];
                        state_cur[h_2d_i_j_ਵਿੱਥ] = prev_state_val * w_val + kv_val + sa * b_val;
                        ਨਤੀਜਾ += state_cur[h_2d_i_j_ਵਿੱਥ] * r_val;
                    }
                    dst_ਅੰਕੜੇ[t_h_i_ਵਿੱਥ] = ਨਤੀਜਾ;
                }
            }
        }
    #ਜੇ_ਅੰਤ
}


ਖਾਲੀ ggml_compute_forward_rwkv_wkv7(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_rwkv_wkv7_f32(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_map_custom1

ਖਾਲੀ ggml_compute_forward_map_custom1(
        ਅਚਲ ggml_compute_params * params,
              ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * a = dst->src[0];

    ਬਣਤਰ ggml_map_custom1_op_params p;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&p, dst->op_params, ਆਕਾਰ(p));

    p.fun(dst, a, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_map_custom2

ਖਾਲੀ ggml_compute_forward_map_custom2(
        ਅਚਲ ggml_compute_params * params,
              ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * a = dst->src[0];
    ਅਚਲ ggml_tensor * b = dst->src[1];

    ਬਣਤਰ ggml_map_custom2_op_params p;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&p, dst->op_params, ਆਕਾਰ(p));

    p.fun(dst, a, b, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_map_custom3

ਖਾਲੀ ggml_compute_forward_map_custom3(
        ਅਚਲ ggml_compute_params * params,
              ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * a = dst->src[0];
    ਅਚਲ ggml_tensor * b = dst->src[1];
    ਅਚਲ ggml_tensor * c = dst->src[2];

    ਬਣਤਰ ggml_map_custom3_op_params p;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&p, dst->op_params, ਆਕਾਰ(p));

    p.fun(dst, a, b, c, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_custom

ਖਾਲੀ ggml_compute_forward_custom(
    ਅਚਲ ਬਣਤਰ ggml_compute_params * params,
          ਬਣਤਰ ggml_tensor * dst) {

    ਬਣਤਰ ggml_custom_op_params p;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&p, dst->op_params, ਆਕਾਰ(p));

    p.fun(dst, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_cross_entropy_loss

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_cross_entropy_loss_f32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->nb[0] == ggml_type_ਮਾਪ(src0->type));
    GGML_ASSERT(src1->nb[0] == ggml_type_ਮਾਪ(src1->type));
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // TODO: handle transposed/permuted matrices
    ਅਚਲ int64_t nc = src0->ne[0];
    ਅਚਲ int64_t nr = ggml_nrows(src0);

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਦਸ਼ਮਲਵ * sums =  (ਦਸ਼ਮਲਵ *) params->wdata;
    ਦਸ਼ਮਲਵ * st   = ((ਦਸ਼ਮਲਵ *) params->wdata) + nth + ith*nc;
    ਦਸ਼ਮਲਵ ਜੋੜ_thread = 0.0f;

    GGML_ASSERT(params->wsize >= ਆਕਾਰ(ਦਸ਼ਮਲਵ) * (nth + nth * nc));

    // rows per thread
    ਅਚਲ int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ int64_t ir0 = dr*ith;
    ਅਚਲ int64_t ir1 = MIN(ir0 + dr, nr);

    ਲਈ (int64_t i1 = ir0; i1 < ir1; ++i1) {
        ਅਚਲ ਦਸ਼ਮਲਵ * s0 = (ਅਚਲ ਦਸ਼ਮਲਵ *)((ਅਚਲ ਅੱਖਰ *) src0->ਅੰਕੜੇ + i1*src0->nb[1]);
        ਅਚਲ ਦਸ਼ਮਲਵ * s1 = (ਅਚਲ ਦਸ਼ਮਲਵ *)((ਅਚਲ ਅੱਖਰ *) src1->ਅੰਕੜੇ + i1*src1->nb[1]);

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NDEBUG
        ਲਈ (int64_t i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            ਪ੍ਰਤਿਜ੍ਞਾ(!isnan(s0[i]));
            ਪ੍ਰਤਿਜ੍ਞਾ(!isnan(s1[i]));
        }
#ਜੇ_ਅੰਤ // NDEBUG

        ਦਸ਼ਮਲਵ max = -INFINITY;
        ggml_vec_max_f32(nc, &max, s0);
        ਅਚਲ ggml_float ਜੋੜ_softmax = ggml_vec_log_soft_max_f32(nc, st, s0, max);
        ਪ੍ਰਤਿਜ੍ਞਾ(ਜੋੜ_softmax >= 0.0);

        ggml_vec_add1_f32(nc, st, st, -ਜੋੜ_softmax);
        ggml_vec_mul_f32(nc, st, st, s1);

        ਦਸ਼ਮਲਵ ਜੋੜ_st = 0.0f;
        ggml_vec_ਜੋੜ_f32(nc, &ਜੋੜ_st, st);
        ਜੋੜ_thread += ਜੋੜ_st;

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NDEBUG
        ਲਈ (int64_t i = 0; i < nc; ++i) {
            ਪ੍ਰਤਿਜ੍ਞਾ(!isnan(st[i]));
            ਪ੍ਰਤਿਜ੍ਞਾ(!isinf(st[i]));
        }
#ਜੇ_ਅੰਤ // NDEBUG
    }
    sums[ith] = ਜੋੜ_thread;
    ggml_barrier(params->threadpool);

    ਜੇ (ith == 0) {
        ਦਸ਼ਮਲਵ * dp = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;
        ggml_vec_ਜੋੜ_f32(nth, dp, sums);
        dp[0] *= -1.0f / (ਦਸ਼ਮਲਵ) nr;
    }
}

ਖਾਲੀ ggml_compute_forward_cross_entropy_loss(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_cross_entropy_loss_f32(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_cross_entropy_loss_back

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_cross_entropy_loss_ਪਿੱਠ_f32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * grad  = dst->src[0]; // gradient of forward pass output
    ਅਚਲ ggml_tensor * src0f = dst->src[1]; // src0 of forward pass
    ਅਚਲ ggml_tensor * src1f = dst->src[2]; // src1 of forward pass

    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src0f));
    GGML_ASSERT(ggml_is_contiguous(src1f));
    GGML_ASSERT(ggml_is_contiguous(grad));
    GGML_ASSERT(ggml_are_same_shape(src0f, src1f) && ggml_are_same_shape(src0f, dst));

    ਅਚਲ int64_t ith = params->ith;
    ਅਚਲ int64_t nth = params->nth;

    // TODO: handle transposed/permuted matrices
    ਅਚਲ int64_t nc = src0f->ne[0];
    ਅਚਲ int64_t nr = ggml_nrows(src0f);

    // rows per thread
    ਅਚਲ int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ int64_t ir0 = dr*ith;
    ਅਚਲ int64_t ir1 = MIN(ir0 + dr, nr);

    ਅਚਲ ਦਸ਼ਮਲਵ d_by_nr = ((ਅਚਲ ਦਸ਼ਮਲਵ *) grad->ਅੰਕੜੇ)[0] / (ਦਸ਼ਮਲਵ) nr;

    ਲਈ (int64_t i1 = ir0; i1 < ir1; i1++) {
        ਦਸ਼ਮਲਵ       * ds0 = (ਦਸ਼ਮਲਵ       *)((ਅੱਖਰ       *) dst->ਅੰਕੜੇ   + i1*dst->nb[1]);
        ਅਚਲ ਦਸ਼ਮਲਵ * s0  = (ਅਚਲ ਦਸ਼ਮਲਵ *)((ਅਚਲ ਅੱਖਰ *) src0f->ਅੰਕੜੇ + i1*src0f->nb[1]);
        ਅਚਲ ਦਸ਼ਮਲਵ * s1  = (ਅਚਲ ਦਸ਼ਮਲਵ *)((ਅਚਲ ਅੱਖਰ *) src1f->ਅੰਕੜੇ + i1*src1f->nb[1]);

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NDEBUG
        ਲਈ (int64_t i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            ਪ੍ਰਤਿਜ੍ਞਾ(!isnan(s0[i]));
            ਪ੍ਰਤਿਜ੍ਞਾ(!isnan(s1[i]));
        }
#ਜੇ_ਅੰਤ // NDEBUG

        // soft_max
        ਦਸ਼ਮਲਵ max = -INFINITY;
        ggml_vec_max_f32(nc, &max, s0);
        ਅਚਲ ggml_float ਜੋੜ = ggml_vec_soft_max_f32(nc, ds0, s0, max);
        ਪ੍ਰਤਿਜ੍ਞਾ(ਜੋੜ > 0.0);
        ggml_vec_scale_f32(nc, ds0, 1.0/ਜੋੜ);

        // grad(src0f) = (softmax(src0f) - src1f) * grad(cross_entropy_loss(src0f, src1f)) / nr
        ggml_vec_sub_f32(nc, ds0, ds0, s1);
        ggml_vec_scale_f32(nc, ds0, d_by_nr);

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NDEBUG
        ਲਈ (int64_t i = 0; i < nc; ++i) {
            ਪ੍ਰਤਿਜ੍ਞਾ(!isnan(ds0[i]));
            ਪ੍ਰਤਿਜ੍ਞਾ(!isinf(ds0[i]));
        }
#ਜੇ_ਅੰਤ // NDEBUG
    }
}

ਖਾਲੀ ggml_compute_forward_cross_entropy_loss_ਪਿੱਠ(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_cross_entropy_loss_ਪਿੱਠ_f32(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_opt_ਕਦਮ_adamw_f32(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0         = dst->src[0];
    ਅਚਲ ggml_tensor * src0_grad    = dst->src[1];
    ਅਚਲ ggml_tensor * src0_grad_m  = dst->src[2];
    ਅਚਲ ggml_tensor * src0_grad_v  = dst->src[3];
    ਅਚਲ ggml_tensor * adamw_params = dst->src[4];

    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad));
    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad_m));
    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad_v));
    GGML_ASSERT(ggml_nelements(adamw_params) == 7);

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS
    GGML_ASSERT(nb00 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    // rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr*ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਅਚਲ ਦਸ਼ਮਲਵ * adamw_params_ptr = ggml_get_ਅੰਕੜੇ_f32(adamw_params);

    ਅਚਲ ਦਸ਼ਮਲਵ alpha  = adamw_params_ptr[0];
    ਅਚਲ ਦਸ਼ਮਲਵ beta1  = adamw_params_ptr[1];
    ਅਚਲ ਦਸ਼ਮਲਵ beta2  = adamw_params_ptr[2];
    ਅਚਲ ਦਸ਼ਮਲਵ eps    = adamw_params_ptr[3];
    ਅਚਲ ਦਸ਼ਮਲਵ wd     = adamw_params_ptr[4];
    ਅਚਲ ਦਸ਼ਮਲਵ beta1h = adamw_params_ptr[5];
    ਅਚਲ ਦਸ਼ਮਲਵ beta2h = adamw_params_ptr[6];
    ਅਚਲ ਦਸ਼ਮਲਵ keep   = 1.f - alpha * wd;
    ਲਈ (ਪੂਰਨ ir = ir0; ir < ir1; ++ir) {
        ਅਚਲ int64_t i03 = ir/(ne02*ne01);
        ਅਚਲ int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        ਅਚਲ int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        ਅਚਲ size_t ਵਿੱਥ = i03*nb03 + i02*nb02 + i01*nb01;

        ਦਸ਼ਮਲਵ       * w = (ਦਸ਼ਮਲਵ       *) ((ਅੱਖਰ       *) src0->ਅੰਕੜੇ        + ਵਿੱਥ); //ਭਾਰ
        ਅਚਲ ਦਸ਼ਮਲਵ * g = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src0_grad->ਅੰਕੜੇ   + ਵਿੱਥ); // grad
        ਦਸ਼ਮਲਵ       * m = (ਦਸ਼ਮਲਵ       *) ((ਅੱਖਰ       *) src0_grad_m->ਅੰਕੜੇ + ਵਿੱਥ);
        ਦਸ਼ਮਲਵ       * v = (ਦਸ਼ਮਲਵ       *) ((ਅੱਖਰ       *) src0_grad_v->ਅੰਕੜੇ + ਵਿੱਥ);

        ਲਈ (ਪੂਰਨ i00 = 0; i00 < ne00; ++i00) {
            m[i00] = m[i00]*beta1 +        g[i00]*(1.0f - beta1);
            v[i00] = v[i00]*beta2 + g[i00]*g[i00]*(1.0f - beta2);

            ਅਚਲ ਦਸ਼ਮਲਵ mh =       m[i00]*beta1h;
            ਅਚਲ ਦਸ਼ਮਲਵ vh = sqrtf(v[i00]*beta2h) + eps;

            // The weight decay is applied independently of the Adam momenta m and v.
            // This is NOT equivalent to l2 regularization that adds w[i00]*w[i00] to the loss.
            // See: https://arxiv.org/pdf/1711.05101v3.pdf
            w[i00] = w[i00] * keep - alpha * mh / vh;
        }
    }
}

ਖਾਲੀ ggml_compute_forward_opt_ਕਦਮ_adamw(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_opt_ਕਦਮ_adamw_f32(params, dst);
            } ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_opt_ਕਦਮ_sgd_f32(ਅਚਲ ggml_compute_params * params, ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * src0       = dst->src[0];
    ਅਚਲ ggml_tensor * src0_grad  = dst->src[1];
    ਅਚਲ ggml_tensor * sgd_params = dst->src[2];

    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad));
    GGML_ASSERT(ggml_nelements(sgd_params) == 2);

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ ਪੂਰਨ nr = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS
    GGML_ASSERT(nb00 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    // rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1) / nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr * ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    // using adamw param subset we care about - alpha, wd - could have a separate struct
    ਅਚਲ ਦਸ਼ਮਲਵ * sgd_params_ptr   = ggml_get_ਅੰਕੜੇ_f32(sgd_params);
    ਅਚਲ ਦਸ਼ਮਲਵ   alpha            = sgd_params_ptr[0];
    ਅਚਲ ਦਸ਼ਮਲਵ   keep             = 1.f - alpha * sgd_params_ptr[1];

    ਲਈ (ਪੂਰਨ ir = ir0; ir < ir1; ++ir) {
        ਅਚਲ int64_t i03 = ir / (ne02 * ne01);
        ਅਚਲ int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
        ਅਚਲ int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

        ਅਚਲ size_t ਵਿੱਥ = i03 * nb03 + i02 * nb02 + i01 * nb01;

        ਦਸ਼ਮਲਵ *       w = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src0->ਅੰਕੜੇ + ਵਿੱਥ);                   //ਭਾਰ
        ਅਚਲ ਦਸ਼ਮਲਵ * g = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src0_grad->ਅੰਕੜੇ + ਵਿੱਥ);  // grad

        ਲਈ (ਪੂਰਨ i00 = 0; i00 < ne00; ++i00) {
            w[i00] = w[i00] * keep - alpha * g[i00];
        }
    }
}

ਖਾਲੀ ggml_compute_forward_opt_ਕਦਮ_sgd(ਅਚਲ ggml_compute_params * params, ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_opt_ਕਦਮ_sgd_f32(params, dst);
            }
            ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error - sgd is F32 only");
            }
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ ggml_fwht_ਲੱਦੋ(ਅਚਲ ਦਸ਼ਮਲਵ value) {
    ਵਾਪਸ value;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ ggml_fwht_ਲੱਦੋ(ਅਚਲ ggml_fp16_t value) {
    ਵਾਪਸ ggml_fp16_ਮੰਜ਼ਿਲ_fp32(value);
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ src_t>
ਸਥਿਰ ਖਾਲੀ ggml_compute_forward_fwht_impl(ਅਚਲ ggml_compute_params * params, ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == (std::is_same_v<src_t, ਦਸ਼ਮਲਵ> ? GGML_TYPE_F32 : GGML_TYPE_F16));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    ਅਚਲ int64_t n = ne10;
    GGML_ASSERT((n & (n - 1)) == 0); // must be power of 2

    ਅਚਲ int64_t nr = ne11 * ne12 * ne13;
    ਅਚਲ int64_t ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread = (nr + nth - 1) / nth;
    ਅਚਲ int64_t ਸ਼ੁਰੂ_ਪੰਕਤੀ = ith * ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread;
    ਅਚਲ int64_t ਸਿਰਾ_ਪੰਕਤੀ = MIN(ਸ਼ੁਰੂ_ਪੰਕਤੀ + ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread, nr);

    ਅਚਲ ਦਸ਼ਮਲਵ scale = 1.0f / sqrtf((ਦਸ਼ਮਲਵ)n);

#ਜੇ defined(GGML_SIMD)
    ਅਚਲ GGML_F32_VEC v_minus_one = GGML_F32_VEC_SET1(-1.0f);
#ਜੇ_ਅੰਤ

    ਲਈ (int64_t r = ਸ਼ੁਰੂ_ਪੰਕਤੀ; r < ਸਿਰਾ_ਪੰਕਤੀ; r++) {
        ਅਚਲ int64_t i13 = r / (ne11 * ne12);
        ਅਚਲ int64_t i12 = (r - i13 * ne11 * ne12) / ne11;
        ਅਚਲ int64_t i11 = r - i13 * ne11 * ne12 - i12 * ne11;

        ਅਚਲ src_t * src_ਪੰਕਤੀ = (ਅਚਲ src_t *) ((ਅਚਲ ਅੱਖਰ *) src1->ਅੰਕੜੇ + i11 * nb11 + i12 * nb12 + i13 * nb13);
        ਦਸ਼ਮਲਵ * dst_ਪੰਕਤੀ = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + i11 * nb1 + i12 * nb2 + i13 * nb3);

        ਲਈ (int64_t j = 0; j < n; j++) {
            dst_ਪੰਕਤੀ[j] = ggml_fwht_ਲੱਦੋ(src_ਪੰਕਤੀ[j]) * scale;
        }

        // Scalar passes
#ਜੇ defined(GGML_SIMD)
#ਜੇ defined(__ARM_FEATURE_SVE)
        ਅਚਲ ਪੂਰਨ ਕਦਮ = svcntw();
#ਨਹੀਂਤਾਂ
        ਅਚਲ ਪੂਰਨ ਕਦਮ = GGML_F32_EPR;
#ਜੇ_ਅੰਤ
#ਨਹੀਂਤਾਂ
        ਅਚਲ ਪੂਰਨ ਕਦਮ = n;
#ਜੇ_ਅੰਤ
        ਲਈ (int64_t len = 1; len < ਕਦਮ && len < n; len <<= 1) {
            ਲਈ (int64_t i = 0; i < n; i += 2 * len) {
                ਲਈ (int64_t j = 0; j < len; j++) {
                    ਦਸ਼ਮਲਵ u = dst_ਪੰਕਤੀ[i + j];
                    ਦਸ਼ਮਲਵ v = dst_ਪੰਕਤੀ[i + len + j];
                    dst_ਪੰਕਤੀ[i + j] = u + v;
                    dst_ਪੰਕਤੀ[i + len + j] = u - v;
                }
            }
        }

        // SIMD passes using GGML_F32_VEC_* macros for multi-architecture support
#ਜੇ defined(GGML_SIMD)
        ਲਈ (int64_t len = ਕਦਮ; len < n; len <<= 1) {
            ਲਈ (int64_t i = 0; i < n; i += 2 * len) {
                ਲਈ (int64_t j = 0; j < len; j += ਕਦਮ) {
                    GGML_F32_VEC u = GGML_F32_VEC_ਲੱਦੋ(dst_ਪੰਕਤੀ + i + j);
                    GGML_F32_VEC v = GGML_F32_VEC_ਲੱਦੋ(dst_ਪੰਕਤੀ + i + len + j);

                    GGML_F32_VEC_STORE(dst_ਪੰਕਤੀ + i + j,       GGML_F32_VEC_ਜੋੜੋ(u, v));
                    GGML_F32_VEC_STORE(dst_ਪੰਕਤੀ + i + len + j, GGML_F32_VEC_FMA(u, v, v_minus_one));
                }
            }
        }
#ਜੇ_ਅੰਤ
    }
}

ਖਾਲੀ ggml_compute_forward_fwht(ਅਚਲ ggml_compute_params * params, ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    ਚੋਣ (src1->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ggml_compute_forward_fwht_impl<ਦਸ਼ਮਲਵ>(params, dst);
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F16:
            {
                ggml_compute_forward_fwht_impl<ggml_fp16_t>(params, dst);
            }
            ਰੋਕੋ;
        ਮੂਲ:
            {
                GGML_ABORT("fatal error - fwht supports F32 and F16 input");
            }
    }
}

// ggml_compute_forward_lightning_indexer

ਖਾਲੀ ggml_compute_forward_lightning_indexer(
        ਅਚਲ ggml_compute_params * params,
        ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * q = dst->src[0];
    ਅਚਲ ggml_tensor * k = dst->src[1];
    ਅਚਲ ggml_tensor * w = dst->src[2]; // weights
    ਅਚਲ ggml_tensor * m = dst->src[3]; //ਨਕਾਬ

    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(   q->type == GGML_TYPE_F32);
    GGML_ASSERT(   w->type == GGML_TYPE_F32);
    GGML_ASSERT(   m->type == GGML_TYPE_F16);

    GGML_TENSOR_LOCALS(int64_t, neq,  q, ne)
    GGML_TENSOR_LOCALS(size_t,  nbq,  q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek,  k, ne)
    GGML_TENSOR_LOCALS(size_t,  nbk,  k, nb)
    GGML_TENSOR_LOCALS(int64_t, ਨਵਾਂ,  w, ne)
    GGML_TENSOR_LOCALS(size_t,  nbw,  w, nb)
    GGML_TENSOR_LOCALS(int64_t, nem,  m, ne)
    GGML_TENSOR_LOCALS(size_t,  nbm,  m, nb)
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb)

    GGML_ASSERT( nb0 == ggml_type_ਮਾਪ(dst->type));
    GGML_ASSERT(nbq0 == ggml_type_ਮਾਪ(  q->type));
    GGML_ASSERT(nbk0 == ggml_type_ਮਾਪ(  k->type));
    GGML_ASSERT(nbw0 == ggml_type_ਮਾਪ(  w->type));
    GGML_ASSERT(nbm0 == ggml_type_ਮਾਪ(  m->type));

    ਅਚਲ ਪੂਰਨ n_embd    = q->ne[0];
    ਅਚਲ ਪੂਰਨ n_ਸਿਰ    = q->ne[1];
    ਅਚਲ ਪੂਰਨ n_ਟੋਕਨ  = q->ne[2];
    ਅਚਲ ਪੂਰਨ n_ਧਾਰਾ  = q->ne[3];
    ਅਚਲ ਪੂਰਨ n_kv      = k->ne[2];

    ggml_ਮੰਜ਼ਿਲ_float_t ਅਚਲ k_ਮੰਜ਼ਿਲ_float = ggml_get_type_traits(k->type)->ਮੰਜ਼ਿਲ_float;
    GGML_ASSERT((k->type == GGML_TYPE_F32 || k_ਮੰਜ਼ਿਲ_float) && "lightning indexer: unsupported K-type");

    ਅਚਲ ਪੂਰਨ nr  = n_kv;
    ਅਚਲ ਪੂਰਨ ith = params->ith;
    ਅਚਲ ਪੂਰਨ nth = params->nth;

    // (temporary) buffer for K converted to float
    ਦਸ਼ਮਲਵ * k_ਪੰਕਤੀ_f32 = (ਦਸ਼ਮਲਵ *) params->wdata + ith*(1*n_embd + CACHE_ਸਤਰ_ਮਾਪ_F32);

    // rows per thread
    ਅਚਲ ਪੂਰਨ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ਅਚਲ ਪੂਰਨ ir0 = dr*ith;
    ਅਚਲ ਪੂਰਨ ir1 = MIN(ir0 + dr, nr);

    ਲਈ (ਪੂਰਨ s = 0; s < n_ਧਾਰਾ; ++s) {
        ਲਈ (ਪੂਰਨ t = 0; t < n_ਟੋਕਨ; ++t) {
            ਅਚਲ ਦਸ਼ਮਲਵ       *   w_ਪੰਕਤੀ =       (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *)   w->ਅੰਕੜੇ + t*nbw1 +        s*nbw3);
            ਅਚਲ ggml_fp16_t *   m_ਪੰਕਤੀ = (ggml_fp16_t *) ((ਅੱਖਰ *)   m->ਅੰਕੜੇ + t*nbm1 + (s%nem3)*nbm3);
            ਦਸ਼ਮਲਵ             * dst_ਪੰਕਤੀ =       (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + t*nb1  +        s*nb3 );
            ਲਈ (ਪੂਰਨ ik = ir0; ik < ir1; ++ik) {
                ਅੱਖਰ * k_ਪੰਕਤੀ = (ਅੱਖਰ *) k->ਅੰਕੜੇ + ik*nbk2 + s*nbk3;
                ਜੇ (k_ਮੰਜ਼ਿਲ_float) {
                    k_ਮੰਜ਼ਿਲ_float(k_ਪੰਕਤੀ, k_ਪੰਕਤੀ_f32, n_embd);
                } ਨਹੀਂਤਾਂ {
                    k_ਪੰਕਤੀ_f32 = (ਦਸ਼ਮਲਵ *) k_ਪੰਕਤੀ;
                }
                ਦਸ਼ਮਲਵ score = 0.0f;
                ਲਈ (ਪੂਰਨ h = 0; h < n_ਸਿਰ; ++h) {
                    // dot product of q and k for head h
                    ਦਸ਼ਮਲਵ qk = 0.0f;
                    ਅਚਲ ਦਸ਼ਮਲਵ * q_ਪੰਕਤੀ = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) q->ਅੰਕੜੇ + h*nbq1 + t*nbq2 + s*nbq3);
                    ggml_vec_dot_f32(n_embd, &qk, 0, q_ਪੰਕਤੀ, 0, k_ਪੰਕਤੀ_f32, 0, 1);
                    // ReLU and weights (prescaled)
                    score += MAX(qk, 0.0f) * w_ਪੰਕਤੀ[h];
                }
                // apply mask
                dst_ਪੰਕਤੀ[ik] = score + GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(m_ਪੰਕਤੀ[ik]);
            }
        }
    }
}
