#ಅನ್ತರ್ಭಾವ "ops.h"

#ಅನ್ತರ್ಭಾವ "ggml-cpu.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "binary-ops.h"
#ಅನ್ತರ್ಭಾವ "simd-gemm.h"
#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "unary-ops.h"
#ಅನ್ತರ್ಭಾವ "vec.h"

#ಅನ್ತರ್ಭಾವ <algorithm>
#ಅನ್ತರ್ಭಾವ <cfloat>
#ಅನ್ತರ್ಭಾವ <cmath>

// ggml_compute_forward_dup

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_dup_same_cont(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));
    GGML_ASSERT(src0->type == dst->type);

    ಸ್ಥಿರಾಂಕ size_t nb0 = ggml_type_ಗಾತ್ರ(src0->type);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith; // thread index
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth; // number of threads

    // parallelize by blocks
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nk = ggml_nelements(src0)/ggml_blck_ಗಾತ್ರ(src0->type);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nk + nth - 1) / nth;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k0 = dr * ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k1 = MIN(k0 + dr, nk);

    ಆದರೆ (k0 < k1) {
        ಸ್ಮರಣೆ_ನಕಲು(
            ((ಅಕ್ಷರ *)  dst->ದತ್ತಾಂಶ + k0*nb0),
            ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + k0*nb0),
            (k1 - k0) * nb0);
    }
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು src_t, ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_dup_flt(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(!ggml_is_quantized(src0->type) && !ggml_is_quantized(dst->type));

    GGML_TENSOR_UNARY_OP_LOCALS

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith; // thread index
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth; // number of threads

    // parallelize by rows
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr = ne01;
    // number of rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1) / nth;
    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr * ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    // case: type & row size equal
    ಆದರೆ (src0->type == dst->type &&
        ne00 == ne0 &&
        nb00 == ggml_type_ಗಾತ್ರ(src0->type) && nb0 == ggml_type_ಗಾತ್ರ(dst->type)) {
        // copy by rows
        ಸ್ಥಿರಾಂಕ size_t rs = ne00*nb00;
        ಗಾಗಿ (int64_t i03 = 0; i03 < ne03; i03++) {
            ಗಾಗಿ (int64_t i02 = 0; i02 < ne02; i02++) {
                ಗಾಗಿ (int64_t i01 = ir0; i01 < ir1; i01++) {
                    ಸ್ಮರಣೆ_ನಕಲು(
                        ((ಅಕ್ಷರ *)  dst->ದತ್ತಾಂಶ + i01*nb1  + i02*nb2  + i03*nb3),
                        ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i01*nb01 + i02*nb02 + i03*nb03),
                        rs);
                }
            }
        }
        ಹಿಂತಿರುಗಿ;
    }

    // case: dst tensor is contiguous
    ಆದರೆ (ggml_is_contiguous(dst)) {
        ಆದರೆ (nb00 == ಗಾತ್ರ(src_t)) {
            ಆದರೆ constexpr (std::is_same_v<dst_t, src_t>) {
                // same type
                size_t id = 0;
                ಸ್ಥಿರಾಂಕ size_t rs = ne00 * nb00;
                ಅಕ್ಷರ * dst_ptr = (ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ;

                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i03 = 0; i03 < ne03; i03++) {
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i02 = 0; i02 < ne02; i02++) {
                        id += rs * ir0;
                        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i01 = ir0; i01 < ir1; i01++) {
                            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * src0_ptr = (ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i01*nb01 + i02*nb02 + i03*nb03;
                            ಸ್ಮರಣೆ_ನಕಲು(dst_ptr + id, src0_ptr, rs);
                            id += rs;
                        }
                        id += rs * (ne01 - ir1);
                    }
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                // casting between non-quantized types
                size_t id = 0;
                dst_t * dst_ptr = (dst_t *) dst->ದತ್ತಾಂಶ;

                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i03 = 0; i03 < ne03; i03++) {
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i01 = ir0; i01 < ir1; i01++) {
                            ಸ್ಥಿರಾಂಕ src_t * src0_ptr = (src_t *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i01*nb01 + i02*nb02 + i03*nb03);
                            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i00 = 0; i00 < ne00; i00++) {
                                ದಶಮಾಂಶ tmp = type_conversion_ಕೋಷ್ಟಕ<src_t>::ಗಮ್ಯ_f32(src0_ptr[i00]);
                                dst_ptr[id] = type_conversion_ಕೋಷ್ಟಕ<dst_t>::from_f32(tmp);
                                id++;
                            }
                        }
                        id += ne00 * (ne01 - ir1);
                    }
                }
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            //printf("%s: this is not optimal - fix me\n", __func__);

            size_t id = 0;
            dst_t * dst_ptr = (dst_t *) dst->ದತ್ತಾಂಶ;

            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i03 = 0; i03 < ne03; i03++) {
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i02 = 0; i02 < ne02; i02++) {
                    id += ne00 * ir0;
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i01 = ir0; i01 < ir1; i01++) {
                        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i00 = 0; i00 < ne00; i00++) {
                            ಸ್ಥಿರಾಂಕ src_t * src0_ptr = (src_t *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);

                            ದಶಮಾಂಶ tmp = type_conversion_ಕೋಷ್ಟಕ<src_t>::ಗಮ್ಯ_f32(*src0_ptr);
                            dst_ptr[id] = type_conversion_ಕೋಷ್ಟಕ<dst_t>::from_f32(tmp);
                            id++;
                        }
                    }
                    id += ne00 * (ne01 - ir1);
                }
            }
        }
        ಹಿಂತಿರುಗಿ;
    }

    // dst counters
    int64_t i10 = 0;
    int64_t i11 = 0;
    int64_t i12 = 0;
    int64_t i13 = 0;

    ಆದರೆ constexpr (std::is_same_v<dst_t, src_t>) {
        ಗಾಗಿ (int64_t i03 = 0; i03 < ne03; i03++) {
            ಗಾಗಿ (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                ಆಗ (i10 >= ne0) {
                    i10 -= ne0;
                    ಆದರೆ (++i11 == ne1) {
                        i11 = 0;
                        ಆದರೆ (++i12 == ne2) {
                            i12 = 0;
                            ಆದರೆ (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                ಗಾಗಿ (int64_t i01 = ir0; i01 < ir1; i01++) {
                    ಗಾಗಿ (int64_t i00 = 0; i00 < ne00; i00++) {
                        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * src0_ptr = ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              ಅಕ್ಷರ * dst_ptr  = ((ಅಕ್ಷರ *)  dst->ದತ್ತಾಂಶ + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                        ಸ್ಮರಣೆ_ನಕಲು(dst_ptr, src0_ptr, ಗಾತ್ರ(dst_t));

                        ಆದರೆ (++i10 == ne00) {
                            i10 = 0;
                            ಆದರೆ (++i11 == ne01) {
                                i11 = 0;
                                ಆದರೆ (++i12 == ne02) {
                                    i12 = 0;
                                    ಆದರೆ (++i13 == ne03) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                ಆಗ (i10 >= ne0) {
                    i10 -= ne0;
                    ಆದರೆ (++i11 == ne1) {
                        i11 = 0;
                        ಆದರೆ (++i12 == ne2) {
                            i12 = 0;
                            ಆದರೆ (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
            }
        }

    } ಇಲ್ಲದಿದ್ದರೆ {
        ಗಾಗಿ (int64_t i03 = 0; i03 < ne03; i03++) {
            ಗಾಗಿ (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                ಆಗ (i10 >= ne0) {
                    i10 -= ne0;
                    ಆದರೆ (++i11 == ne1) {
                        i11 = 0;
                        ಆದರೆ (++i12 == ne2) {
                            i12 = 0;
                            ಆದರೆ (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                ಗಾಗಿ (int64_t i01 = ir0; i01 < ir1; i01++) {
                    ಗಾಗಿ (int64_t i00 = 0; i00 < ne00; i00++) {
                        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * src0_ptr = ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              ಅಕ್ಷರ * dst_ptr  = ((ಅಕ್ಷರ *)  dst->ದತ್ತಾಂಶ + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                        ದಶಮಾಂಶ tmp = type_conversion_ಕೋಷ್ಟಕ<src_t>::ಗಮ್ಯ_f32(*(ಸ್ಥಿರಾಂಕ src_t *) src0_ptr);
                        *(dst_t *) dst_ptr = type_conversion_ಕೋಷ್ಟಕ<dst_t>::from_f32(tmp);

                        ಆದರೆ (++i10 == ne0) {
                            i10 = 0;
                            ಆದರೆ (++i11 == ne1) {
                                i11 = 0;
                                ಆದರೆ (++i12 == ne2) {
                                    i12 = 0;
                                    ಆದರೆ (++i13 == ne3) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                ಆಗ (i10 >= ne0) {
                    i10 -= ne0;
                    ಆದರೆ (++i11 == ne1) {
                        i11 = 0;
                        ಆದರೆ (++i12 == ne2) {
                            i12 = 0;
                            ಆದರೆ (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
            }
        }
    }
}


ಮಾದರಿ<ಬಗೆ_ಹೆಸರು src_t>
ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_dup_ಗಮ್ಯ_q(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(!ggml_is_quantized(src0->type));

    GGML_TENSOR_UNARY_OP_LOCALS

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith; // thread index
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth; // number of threads

    // parallelize by rows
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr = ne01;
    // number of rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1) / nth;
    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr * ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ಆದರೆ (ggml_is_contiguous(dst) &&
            nb00 == ಗಾತ್ರ(src_t) &&
            ggml_get_type_traits_cpu(dst->type)->from_float) {
        // casting non-quantized types --> intermediate f32 --> quantized
        ggml_from_float_t ಸ್ಥಿರಾಂಕ quantize_ಅಡ್ಡಸಾಲು_q = ggml_get_type_traits_cpu(dst->type)->from_float;
        ದಶಮಾಂಶ * src0_f32 = (ದಶಮಾಂಶ *) params->wdata + (ne00 + CACHE_ಸಾಲು_ಗಾತ್ರ_F32) * ith;

        size_t id = 0;
        size_t rs = nb0 * (ne00 / ggml_blck_ಗಾತ್ರ(dst->type));
        ಅಕ್ಷರ * dst_ptr = (ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ;

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i03 = 0; i03 < ne03; i03++) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i02 = 0; i02 < ne02; i02++) {
                id += rs * ir0;
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i01 = ir0; i01 < ir1; i01++) {
                    ಸ್ಥಿರಾಂಕ src_t * src0_ptr = (src_t *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i01*nb01 + i02*nb02 + i03*nb03);

                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i00 = 0; i00 < ne00; i00++) {
                        src0_f32[i00] = type_conversion_ಕೋಷ್ಟಕ<src_t>::ಗಮ್ಯ_f32(src0_ptr[i00]);
                    }

                    quantize_ಅಡ್ಡಸಾಲು_q(src0_f32, dst_ptr + id, ne00);
                    id += rs;
                }
                id += rs * (ne01 - ir1);
            }
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        // printf("%s %s\n", ggml_type_name(src0->type), ggml_type_name(dst->type));
        GGML_ABORT("not implemented");
    }
}

// A simplified version of ggml_compute_forward_dup that doesn't do float upcasting, and just plain old memcpy.
ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_dup_ಬೈಟುಗಳು(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(src0->type == dst->type);

    GGML_TENSOR_UNARY_OP_LOCALS;

    ಆದರೆ (ggml_is_contiguous(src0) && ggml_is_contiguous(dst)) {
        ggml_compute_forward_dup_same_cont(params, dst);
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ size_t type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(src0->type);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith; // thread index
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth; // number of threads

    // parallelize by rows
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr = ne01;
    // number of rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1) / nth;
    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr * ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ಆದರೆ (src0->type == dst->type &&
        ggml_are_same_shape(src0, dst) &&
        nb00 == type_ಗಾತ್ರ && nb0 == type_ಗಾತ್ರ) {
        // copy by rows
        ಸ್ಥಿರಾಂಕ size_t rs = ggml_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(src0->type, ne00);
        ಗಾಗಿ (int64_t i03 = 0; i03 < ne03; i03++) {
            ಗಾಗಿ (int64_t i02 = 0; i02 < ne02; i02++) {
                ಗಾಗಿ (int64_t i01 = ir0; i01 < ir1; i01++) {
                    ಸ್ಮರಣೆ_ನಕಲು(
                        ((ಅಕ್ಷರ *)  dst->ದತ್ತಾಂಶ + i01*nb1  + i02*nb2  + i03*nb3),
                        ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i01*nb01 + i02*nb02 + i03*nb03),
                        rs);
                }
            }
        }
        ಹಿಂತಿರುಗಿ;
    }

    ಆದರೆ (ggml_is_contiguous(dst)) {
        size_t id = 0;
        ಅಕ್ಷರ * dst_ptr = (ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ;
        ಸ್ಥಿರಾಂಕ size_t rs = ne00 * type_ಗಾತ್ರ;

        ಆದರೆ (nb00 == type_ಗಾತ್ರ) {
            // src0 is contiguous on first dimension, copy by rows
            ಗಾಗಿ (int64_t i03 = 0; i03 < ne03; i03++) {
                ಗಾಗಿ (int64_t i02 = 0; i02 < ne02; i02++) {
                    id += rs * ir0;
                    ಗಾಗಿ (int64_t i01 = ir0; i01 < ir1; i01++) {
                        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * src0_ptr = (ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i01*nb01 + i02*nb02 + i03*nb03;
                        ಸ್ಮರಣೆ_ನಕಲು(dst_ptr + id, src0_ptr, rs);
                        id += rs;
                    }
                    id += rs * (ne01 - ir1);
                }
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            //printf("%s: this is not optimal - fix me\n", __func__);

            ಗಾಗಿ (int64_t i03 = 0; i03 < ne03; i03++) {
                ಗಾಗಿ (int64_t i02 = 0; i02 < ne02; i02++) {
                    id += rs * ir0;
                    ಗಾಗಿ (int64_t i01 = ir0; i01 < ir1; i01++) {
                        ಗಾಗಿ (int64_t i00 = 0; i00 < ne00; i00++) {
                            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * src0_ptr = (ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03;
                            ಸ್ಮರಣೆ_ನಕಲು(dst_ptr + id, src0_ptr, type_ಗಾತ್ರ);

                            id += type_ಗಾತ್ರ;
                        }
                    }
                    id += rs * (ne01 - ir1);
                }
            }
        }

        ಹಿಂತಿರುಗಿ;
    }

    // dst counters
    int64_t k10 = 0;
    int64_t i11 = 0;
    int64_t i12 = 0;
    int64_t i13 = 0;

    // number of blocks in a row
    ಸ್ಥಿರಾಂಕ int64_t nk00 = ne00 / ggml_blck_ಗಾತ್ರ(src0->type);
    ಸ್ಥಿರಾಂಕ int64_t nk0  = ne0  / ggml_blck_ಗಾತ್ರ(dst->type);

    ಗಾಗಿ (int64_t i03 = 0; i03 < ne03; i03++) {
        ಗಾಗಿ (int64_t i02 = 0; i02 < ne02; i02++) {
            k10 += nk00 * ir0;
            ಆಗ (k10 >= nk0) {
                k10 -= nk0;
                ಆದರೆ (++i11 == ne1) {
                    i11 = 0;
                    ಆದರೆ (++i12 == ne2) {
                        i12 = 0;
                        ಆದರೆ (++i13 == ne3) {
                            i13 = 0;
                        }
                    }
                }
            }
            ಗಾಗಿ (int64_t i01 = ir0; i01 < ir1; i01++) {
                ಗಾಗಿ (int64_t k00 = 0; k00 < nk00; k00++) {
                    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * src0_ptr = ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + k00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                          ಅಕ್ಷರ * dst_ptr  = ((ಅಕ್ಷರ *)  dst->ದತ್ತಾಂಶ + k10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                    ಸ್ಮರಣೆ_ನಕಲು(dst_ptr, src0_ptr, type_ಗಾತ್ರ);

                    ಆದರೆ (++k10 == nk0) {
                        k10 = 0;
                        ಆದರೆ (++i11 == ne1) {
                            i11 = 0;
                            ಆದರೆ (++i12 == ne2) {
                                i12 = 0;
                                ಆದರೆ (++i13 == ne3) {
                                    i13 = 0;
                                }
                            }
                        }
                    }
                }
            }
            k10 += nk00 * (ne01 - ir1);
            ಆಗ (k10 >= nk0) {
                k10 -= nk0;
                ಆದರೆ (++i11 == ne1) {
                    i11 = 0;
                    ಆದರೆ (++i12 == ne2) {
                        i12 = 0;
                        ಆದರೆ (++i13 == ne3) {
                            i13 = 0;
                        }
                    }
                }
            }
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_dup_from_q(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
              ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

    ಸ್ಥಿರಾಂಕ ggml_type type = src0->type;
    ggml_ಗಮ್ಯ_float_t ಸ್ಥಿರಾಂಕ dequantize_ಅಡ್ಡಸಾಲು_q = ggml_get_type_traits(type)->ಗಮ್ಯ_float;

    size_t qk = ggml_blck_ಗಾತ್ರ(type);
    ಸ್ಥಿರಾಂಕ int64_t nr = ggml_nelements(src1) / qk;

    // destination must be contiguous in the first dimension
    GGML_ASSERT(nb10 == ggml_type_ಗಾತ್ರ(dst->type));
    // must either have first dimension large enough to hold a row, or fully contiguous
    GGML_ASSERT((ne10 % qk) == 0 || ggml_is_contiguous(dst));

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ಗಾಗಿ (int64_t ir = ir0; ir < ir1; ++ir) {

        uint32_t i = ir * qk;

        ಸ್ಥಿರಾಂಕ int64_t i03 = i/(ne00 * ne01 * ne02);
        ಸ್ಥಿರಾಂಕ int64_t i02 = (i - i03*ne00*ne01*ne02 )/ (ne00*ne01);
        ಸ್ಥಿರಾಂಕ int64_t i01 = (i - i03*ne00*ne01*ne02  -  i02*ne01*ne00) / ne00;
        ಸ್ಥಿರಾಂಕ int64_t i00 = i - i03*ne00*ne01*ne02 - i02*ne01*ne00 - i01*ne00;
        ಸ್ಥಿರಾಂಕ int64_t x_ಸ್ಥಾನಾಂತರ = (i00/qk)*nb00 + i01*nb01 + i02*nb02 + i03 * nb03;

        ಸ್ಥಿರಾಂಕ int64_t i13 = i/(ne10 * ne11 * ne12);
        ಸ್ಥಿರಾಂಕ int64_t i12 = (i - i13*ne10*ne11*ne12) / (ne10*ne11);
        ಸ್ಥಿರಾಂಕ int64_t i11 = (i - i13*ne10*ne11*ne12 - i12*ne10*ne11) / ne10;
        ಸ್ಥಿರಾಂಕ int64_t i10 = i - i13*ne10*ne11*ne12 - i12*ne10*ne11 - i11*ne10;
        ಸ್ಥಿರಾಂಕ int64_t dst_ಸ್ಥಾನಾಂತರ = i10*nb10 + i11*nb11 + i12*nb12 + i13*nb13;

        dequantize_ಅಡ್ಡಸಾಲು_q(
                (ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + x_ಸ್ಥಾನಾಂತರ),
                     (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *)  dst->ದತ್ತಾಂಶ + dst_ಸ್ಥಾನಾಂತರ), qk);
    }
}

ಶೂನ್ಯ ggml_compute_forward_dup(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆದರೆ (src0->type == dst->type) {
        ggml_compute_forward_dup_ಬೈಟುಗಳು(params, dst);
        ಹಿಂತಿರುಗಿ;
    }

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            {
                /**/ ಆದರೆ (dst->type == GGML_TYPE_F16)  ggml_compute_forward_dup_flt<ggml_fp16_t, ggml_fp16_t>(params, dst);
                ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (dst->type == GGML_TYPE_BF16) ggml_compute_forward_dup_flt<ggml_fp16_t, ggml_bf16_t>(params, dst);
                ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (dst->type == GGML_TYPE_F32)  ggml_compute_forward_dup_flt<ggml_fp16_t, ದಶಮಾಂಶ      >(params, dst);
                ಇಲ್ಲದಿದ್ದರೆ ggml_compute_forward_dup_ಗಮ್ಯ_q<ggml_fp16_t>(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
            {
                /**/ ಆದರೆ (dst->type == GGML_TYPE_F16)  ggml_compute_forward_dup_flt<ggml_bf16_t, ggml_fp16_t>(params, dst);
                ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (dst->type == GGML_TYPE_BF16) ggml_compute_forward_dup_flt<ggml_bf16_t, ggml_bf16_t>(params, dst);
                ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (dst->type == GGML_TYPE_F32)  ggml_compute_forward_dup_flt<ggml_bf16_t, ದಶಮಾಂಶ      >(params, dst);
                ಇಲ್ಲದಿದ್ದರೆ ggml_compute_forward_dup_ಗಮ್ಯ_q<ggml_bf16_t>(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                /**/ ಆದರೆ (dst->type == GGML_TYPE_F16)  ggml_compute_forward_dup_flt<ದಶಮಾಂಶ, ggml_fp16_t>(params, dst);
                ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (dst->type == GGML_TYPE_BF16) ggml_compute_forward_dup_flt<ದಶಮಾಂಶ, ggml_bf16_t>(params, dst);
                ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (dst->type == GGML_TYPE_F32)  ggml_compute_forward_dup_flt<ದಶಮಾಂಶ, ದಶಮಾಂಶ      >(params, dst);
                ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (dst->type == GGML_TYPE_I32)  ggml_compute_forward_dup_flt<ದಶಮಾಂಶ, int32_t    >(params, dst);
                ಇಲ್ಲದಿದ್ದರೆ ggml_compute_forward_dup_ಗಮ್ಯ_q<ದಶಮಾಂಶ>(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_I32:
            {
                ಆದರೆ (dst->type == GGML_TYPE_F32) ggml_compute_forward_dup_flt<int32_t, ದಶಮಾಂಶ>(params, dst);
                ಇಲ್ಲದಿದ್ದರೆ GGML_ABORT("not implemented");
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                ಆದರೆ (ggml_is_quantized(src0->type) && dst->type == GGML_TYPE_F32) {
                    ggml_compute_forward_dup_from_q(params, dst);
                    ನಿಲ್ಲಿಸು;
                }
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_add

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_ಕೂಡಿಸು_q_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr  = ggml_nrows(src0);

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ggml_type type = src0->type;
    ಸ್ಥಿರಾಂಕ ggml_type dtype = dst->type;
    ggml_ಗಮ್ಯ_float_t ಸ್ಥಿರಾಂಕ dequantize_ಅಡ್ಡಸಾಲು_q = ggml_get_type_traits(type)->ಗಮ್ಯ_float;
    ggml_from_float_t ಸ್ಥಿರಾಂಕ quantize_ಅಡ್ಡಸಾಲು_q = ggml_get_type_traits_cpu(dtype)->from_float;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_ಗಾತ್ರ(type));
    GGML_ASSERT(nb10 == ಗಾತ್ರ(ದಶಮಾಂಶ));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ggml_is_quantized(src0->type));
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    // rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ದಶಮಾಂಶ * wdata = (ದಶಮಾಂಶ *) params->wdata + (ne00 + CACHE_ಸಾಲು_ಗಾತ್ರ_F32) * ith;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ir = ir0; ir < ir1; ++ir) {
        // src0 indices
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i03 = ir/(ne02*ne01);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i02 = (ir - i03*ne02*ne01)/ne01;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i01 = (ir - i03*ne02*ne01 - i02*ne01);

        // src1 and dst are same shape as src0 => same indices
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i13 = i03;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i12 = i02;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i11 = i01;

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i3 = i03;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i2 = i02;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i1 = i01;

        ಶೂನ್ಯ  * src0_ಅಡ್ಡಸಾಲು = (ಶೂನ್ಯ *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + (i01*nb01 + i02*nb02 + i03*nb03));
        ದಶಮಾಂಶ * src1_ಅಡ್ಡಸಾಲು = (ದಶಮಾಂಶ *)((ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ + (i11*nb11 + i12*nb12 + i13*nb13));
        ಶೂನ್ಯ  * dst_ಅಡ್ಡಸಾಲು  = (ಶೂನ್ಯ *) ((ಅಕ್ಷರ *)  dst->ದತ್ತಾಂಶ + ( i1*nb1  +  i2*nb2  +  i3*nb3));

        ಪ್ರತಿಜ್ಞಾ(ne00 % 32 == 0);

        // unquantize row from src0 to temp buffer
        dequantize_ಅಡ್ಡಸಾಲು_q(src0_ಅಡ್ಡಸಾಲು, wdata, ne00);
        // add src1
        ggml_vec_acc_f32(ne00, wdata, src1_ಅಡ್ಡಸಾಲು);
        // quantize row to dst
        ಆದರೆ (quantize_ಅಡ್ಡಸಾಲು_q != ಖಾಲಿ) {
            quantize_ಅಡ್ಡಸಾಲು_q(wdata, dst_ಅಡ್ಡಸಾಲು, ne00);
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಸ್ಮರಣೆ_ನಕಲು(dst_ಅಡ್ಡಸಾಲು, wdata, ne0*nb0);
        }
    }
}

ಶೂನ್ಯ ggml_compute_forward_ಕೂಡಿಸು(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
        ಸ್ಥಿತಿ GGML_TYPE_F16:
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
            {
                ggml_compute_forward_ಕೂಡಿಸು_non_quantized(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q1_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q2_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_1:
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
        ಸ್ಥಿತಿ GGML_TYPE_MXFP4:
        ಸ್ಥಿತಿ GGML_TYPE_NVFP4:
        ಸ್ಥಿತಿ GGML_TYPE_Q2_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q3_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
        ಸ್ಥಿತಿ GGML_TYPE_TQ1_0:
        ಸ್ಥಿತಿ GGML_TYPE_TQ2_0:
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XXS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ3_XXS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ1_S:
        ಸ್ಥಿತಿ GGML_TYPE_IQ1_M:
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_NL:
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_XS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ3_S:
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_ಕೂಡಿಸು_q_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_add_id

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_ಕೂಡಿಸು_id_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src2 = dst->src[2];

    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src2->type == GGML_TYPE_I32);

    GGML_ASSERT(src0->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ));
    GGML_ASSERT(src1->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr  = ggml_nrows(src0);

    GGML_TENSOR_TERNARY_OP_LOCALS

    GGML_ASSERT( nb0 == ಗಾತ್ರ(ದಶಮಾಂಶ));
    GGML_ASSERT(nb10 == ಗಾತ್ರ(ದಶಮಾಂಶ));

    // rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ir = ir0; ir < ir1; ++ir) {
        // src0 indices
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i3 = ir/(ne2*ne1);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i2 = (ir - i3*ne2*ne1)/ne1;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i1 = (ir - i3*ne2*ne1 - i2*ne1);

        // src1 indices
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i11 = *(int32_t *) ((ಅಕ್ಷರ *) src2->ದತ್ತಾಂಶ + i1*nb20 + i2*nb21);

        GGML_ASSERT(i11 >= 0 && i11 < ne11);

        ggml_vec_ಕೂಡಿಸು_f32(ne0,
                (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ  + i3*nb3  + i2*nb2  + i1*nb1 ),
                (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i3*nb03 + i2*nb02 + i1*nb01),
                (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ + i11*nb11));
    }
}

ಶೂನ್ಯ ggml_compute_forward_ಕೂಡಿಸು_id(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_ಕೂಡಿಸು_id_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("unsupported type for ggml_compute_forward_add_id: %s", ggml_type_name(src0->type));
            }
    }
}

// ggml_compute_forward_add1

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_add1_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == ಗಾತ್ರ(ದಶಮಾಂಶ));
    GGML_ASSERT(nb00 == ಗಾತ್ರ(ದಶಮಾಂಶ));

    // rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i3 = ir/(ne2*ne1);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i2 = (ir - i3*ne2*ne1)/ne1;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i1 = (ir - i3*ne2*ne1 - i2*ne1);

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_ACCELERATE
        GGML_ಬಳಸದ(ggml_vec_add1_f32);

        vDSP_vadd(
                (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i3*nb03 + i2*nb02 + i1*nb01), 1,
                (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ), 0,
                (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ  + i3*nb3  + i2*nb2  + i1*nb1 ), 1,
                ne0);
#ಇಲ್ಲದಿದ್ದರೆ
        ggml_vec_add1_f32(ne0,
                (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ  + i3*nb3  + i2*nb2  + i1*nb1 ),
                (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i3*nb03 + i2*nb02 + i1*nb01),
               *(ದಶಮಾಂಶ *) src1->ದತ್ತಾಂಶ);
#ಆದರೆ_ಅಂತ್ಯ
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_add1_f16_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ v = *(ದಶಮಾಂಶ *) src1->ದತ್ತಾಂಶ;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F16);

    GGML_ASSERT( nb0 == ಗಾತ್ರ(ggml_fp16_t));
    GGML_ASSERT(nb00 == ಗಾತ್ರ(ggml_fp16_t));

    // rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i3 = ir/(ne2*ne1);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i2 = (ir - i3*ne2*ne1)/ne1;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i3*nb03 + i2*nb02 + i1*nb01);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_CPU_FP32_ಗಮ್ಯ_FP16(GGML_CPU_FP16_ಗಮ್ಯ_FP32(src0_ptr[i]) + v);
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_add1_f16_f16(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ v = GGML_CPU_FP16_ಗಮ್ಯ_FP32(*(ggml_fp16_t *) src1->ದತ್ತಾಂಶ);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type  == GGML_TYPE_F16);

    GGML_ASSERT( nb0 == ಗಾತ್ರ(ggml_fp16_t));
    GGML_ASSERT(nb00 == ಗಾತ್ರ(ggml_fp16_t));

    // rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i3 = ir/(ne2*ne1);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i2 = (ir - i3*ne2*ne1)/ne1;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i3*nb03 + i2*nb02 + i1*nb01);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_CPU_FP32_ಗಮ್ಯ_FP16(GGML_CPU_FP16_ಗಮ್ಯ_FP32(src0_ptr[i]) + v);
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_add1_q_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ v = *(ದಶಮಾಂಶ *) src1->ದತ್ತಾಂಶ;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    ಸ್ಥಿರಾಂಕ ggml_type type = src0->type;
    ggml_ಗಮ್ಯ_float_t ಸ್ಥಿರಾಂಕ dequantize_ಅಡ್ಡಸಾಲು_q = ggml_get_type_traits(type)->ಗಮ್ಯ_float;
    ggml_from_float_t ಸ್ಥಿರಾಂಕ quantize_ಅಡ್ಡಸಾಲು_q = ggml_get_type_traits_cpu(type)->from_float;

    // we don't support permuted src0
    GGML_ASSERT(nb00 == ggml_type_ಗಾತ್ರ(type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ggml_is_quantized(src0->type));
    GGML_ASSERT(dst->type == src0->type);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    // rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ದಶಮಾಂಶ * wdata = (ದಶಮಾಂಶ *) params->wdata + (ne0 + CACHE_ಸಾಲು_ಗಾತ್ರ_F32) * ith;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i3 = ir/(ne2*ne1);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i2 = (ir - i3*ne2*ne1)/ne1;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ಶೂನ್ಯ  * src0_ಅಡ್ಡಸಾಲು = (ಶೂನ್ಯ *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + (i1*nb01 + i2*nb02 + i3*nb03));
        ಶೂನ್ಯ  * dst_ಅಡ್ಡಸಾಲು  = (ಶೂನ್ಯ *) ((ಅಕ್ಷರ *)  dst->ದತ್ತಾಂಶ + (i1*nb1  + i2*nb2  + i3*nb0 ));

        ಪ್ರತಿಜ್ಞಾ(ne0 % 32 == 0);

        // unquantize row from src0 to temp buffer
        dequantize_ಅಡ್ಡಸಾಲು_q(src0_ಅಡ್ಡಸಾಲು, wdata, ne0);
        // add src1
        ggml_vec_acc1_f32(ne0, wdata, v);
        // quantize row to dst
        quantize_ಅಡ್ಡಸಾಲು_q(wdata, dst_ಅಡ್ಡಸಾಲು, ne0);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_add1_bf16_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ v = *(ದಶಮಾಂಶ *) src1->ದತ್ತಾಂಶ;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_BF16);

    GGML_ASSERT( nb0 == ಗಾತ್ರ(ggml_bf16_t));
    GGML_ASSERT(nb00 == ಗಾತ್ರ(ggml_bf16_t));

    // rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i3 = ir/(ne2*ne1);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i2 = (ir - i3*ne2*ne1)/ne1;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_bf16_t * dst_ptr  = (ggml_bf16_t *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_bf16_t * src0_ptr = (ggml_bf16_t *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i3*nb03 + i2*nb02 + i1*nb01);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_FP32_ಗಮ್ಯ_BF16(GGML_BF16_ಗಮ್ಯ_FP32(src0_ptr[i]) + v);
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_add1_bf16_bf16(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ v = GGML_BF16_ಗಮ್ಯ_FP32(*(ggml_bf16_t *) src1->ದತ್ತಾಂಶ);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(src1->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->type  == GGML_TYPE_BF16);

    GGML_ASSERT( nb0 == ಗಾತ್ರ(ggml_bf16_t));
    GGML_ASSERT(nb00 == ಗಾತ್ರ(ggml_bf16_t));

    // rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i3 = ir/(ne2*ne1);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i2 = (ir - i3*ne2*ne1)/ne1;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_bf16_t * dst_ptr  = (ggml_bf16_t *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_bf16_t * src0_ptr = (ggml_bf16_t *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i3*nb03 + i2*nb02 + i1*nb01);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_FP32_ಗಮ್ಯ_BF16(GGML_BF16_ಗಮ್ಯ_FP32(src0_ptr[i]) + v);
        }
    }
}

ಶೂನ್ಯ ggml_compute_forward_add1(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_add1_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            {
                ಆದರೆ (src1->type == GGML_TYPE_F16) {
                    ggml_compute_forward_add1_f16_f16(params, dst);
                }
                ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src1->type == GGML_TYPE_F32) {
                    ggml_compute_forward_add1_f16_f32(params, dst);
                }
                ಇಲ್ಲದಿದ್ದರೆ {
                    GGML_ABORT("fatal error");
                }
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
            {
                ಆದರೆ (src1->type == GGML_TYPE_BF16) {
                    ggml_compute_forward_add1_bf16_bf16(params, dst);
                }
                ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src1->type == GGML_TYPE_F32) {
                    ggml_compute_forward_add1_bf16_f32(params, dst);
                }
                ಇಲ್ಲದಿದ್ದರೆ {
                    GGML_ABORT("fatal error");
                }
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q1_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q2_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_1:
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q8_1:
        ಸ್ಥಿತಿ GGML_TYPE_MXFP4:
        ಸ್ಥಿತಿ GGML_TYPE_NVFP4:
        ಸ್ಥಿತಿ GGML_TYPE_Q2_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q3_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
        ಸ್ಥಿತಿ GGML_TYPE_TQ1_0:
        ಸ್ಥಿತಿ GGML_TYPE_TQ2_0:
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XXS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ3_XXS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ1_S:
        ಸ್ಥಿತಿ GGML_TYPE_IQ1_M:
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_NL:
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_XS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ3_S:
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_add1_q_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_acc

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_acc_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));

    // view src0 and dst with these strides and data offset inbytes during acc
    // nb0 is implicitly element_size because src0 and dst are contiguous
    size_t nb1     = ((int32_t *) dst->op_params)[0];
    size_t nb2     = ((int32_t *) dst->op_params)[1];
    size_t nb3     = ((int32_t *) dst->op_params)[2];
    size_t ಸ್ಥಾನಾಂತರ  = ((int32_t *) dst->op_params)[3];
    ತಾರ್ಕಿಕ   inplace = (ತಾರ್ಕಿಕ) ((int32_t *) dst->op_params)[4];

    ಆದರೆ (!inplace) {
        ಆದರೆ (params->ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            ಸ್ಮರಣೆ_ನಕಲು(
                ((ಅಕ್ಷರ *)  dst->ದತ್ತಾಂಶ),
                ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr = ggml_nrows(src1);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nc = src1->ne[0];

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne)
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb)

    // src0 and dst as viewed during acc
    ಸ್ಥಿರಾಂಕ size_t nb0 = ggml_element_ಗಾತ್ರ(src0);

    ಸ್ಥಿರಾಂಕ size_t nb00 = nb0;
    ಸ್ಥಿರಾಂಕ size_t nb01 = nb1;
    ಸ್ಥಿರಾಂಕ size_t nb02 = nb2;
    ಸ್ಥಿರಾಂಕ size_t nb03 = nb3;

    GGML_ASSERT(ಸ್ಥಾನಾಂತರ + (ne10 == 0 ? 0 : ne10-1)*nb0  + (ne11 == 0 ? 0 : ne11-1)*nb1  + (ne12 == 0 ? 0 : ne12-1)*nb2  + (ne13 == 0 ? 0 : ne13-1)*nb3  < ggml_nbytes(dst));
    GGML_ASSERT(ಸ್ಥಾನಾಂತರ + (ne10 == 0 ? 0 : ne10-1)*nb00 + (ne11 == 0 ? 0 : ne11-1)*nb01 + (ne12 == 0 ? 0 : ne12-1)*nb02 + (ne13 == 0 ? 0 : ne13-1)*nb03 < ggml_nbytes(src0));

    GGML_ASSERT(nb10 == ಗಾತ್ರ(ದಶಮಾಂಶ));

    // rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are viewed with shape of src1 and offset
        // => same indices
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i3 = ir/(ne12*ne11);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i2 = (ir - i3*ne12*ne11)/ne11;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i1 = (ir - i3*ne12*ne11 - i2*ne11);

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_ACCELERATE
        vDSP_vadd(
                (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i3*nb03 + i2*nb02 + i1*nb01 + ಸ್ಥಾನಾಂತರ), 1,
                (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ + i3*nb13 + i2*nb12 + i1*nb11), 1,
                (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ  + i3*nb3  + i2*nb2  + i1*nb1  + ಸ್ಥಾನಾಂತರ), 1, nc);
#ಇಲ್ಲದಿದ್ದರೆ
        ggml_vec_ಕೂಡಿಸು_f32(nc,
                (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *)  dst->ದತ್ತಾಂಶ + i3*nb3  + i2*nb2  + i1*nb1  + ಸ್ಥಾನಾಂತರ),
                (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i3*nb03 + i2*nb02 + i1*nb01 + ಸ್ಥಾನಾಂತರ),
                (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ + i3*nb13 + i2*nb12 + i1*nb11));
#ಆದರೆ_ಅಂತ್ಯ
    }
}

ಶೂನ್ಯ ggml_compute_forward_acc(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_acc_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F16:
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
        ಸ್ಥಿತಿ GGML_TYPE_Q1_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q2_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_1:
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q8_1:
        ಸ್ಥಿತಿ GGML_TYPE_MXFP4:
        ಸ್ಥಿತಿ GGML_TYPE_NVFP4:
        ಸ್ಥಿತಿ GGML_TYPE_Q2_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q3_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
        ಸ್ಥಿತಿ GGML_TYPE_TQ1_0:
        ಸ್ಥಿತಿ GGML_TYPE_TQ2_0:
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XXS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ3_XXS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ1_S:
        ಸ್ಥಿತಿ GGML_TYPE_IQ1_M:
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_NL:
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_XS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ3_S:
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_S:
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_sum

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_ಮೊತ್ತ_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆದರೆ (params->ith != 0) {
        ಹಿಂತಿರುಗಿ;
    }

    ಪ್ರತಿಜ್ಞಾ(ggml_is_scalar(dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)

    ggml_float ಮೊತ್ತ = 0;

    ಆದರೆ (nb00 == ಗಾತ್ರ(ದಶಮಾಂಶ)) {
        ggml_float ಅಡ್ಡಸಾಲು_ಮೊತ್ತ = 0;
        ಗಾಗಿ (int64_t i03 = 0; i03 < ne03; i03++) {
            ಗಾಗಿ (int64_t i02 = 0; i02 < ne02; i02++) {
                ಗಾಗಿ (int64_t i01 = 0; i01 < ne01; i01++) {
                    ggml_vec_ಮೊತ್ತ_f32_ggf(ne00,
                            &ಅಡ್ಡಸಾಲು_ಮೊತ್ತ,
                            (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i01*nb01 + i02*nb02 + i03*nb03));
                    ಮೊತ್ತ += ಅಡ್ಡಸಾಲು_ಮೊತ್ತ;
                }
            }
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಗಾಗಿ (int64_t i03 = 0; i03 < ne03; i03++) {
            ಗಾಗಿ (int64_t i02 = 0; i02 < ne02; i02++) {
                ಗಾಗಿ (int64_t i01 = 0; i01 < ne01; i01++) {
                    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i01*nb01 + i02*nb02 + i03*nb03;
                    ಗಾಗಿ (int64_t i00 = 0; i00 < ne00; i00++) {
                        ಮೊತ್ತ += (ggml_float) *(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) (ಅಡ್ಡಸಾಲು + i00*nb00);
                    }
                }
            }
        }
    }

    ((ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ)[0] = ಮೊತ್ತ;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_ಮೊತ್ತ_f16(
    ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
          ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆದರೆ (params->ith != 0) {
        ಹಿಂತಿರುಗಿ;
    }

    ಪ್ರತಿಜ್ಞಾ(ggml_is_scalar(dst));

    ಪ್ರತಿಜ್ಞಾ(src0->nb[0] == ಗಾತ್ರ(ggml_fp16_t));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)

    ದಶಮಾಂಶ ಮೊತ್ತ = 0;
    ದಶಮಾಂಶ ಅಡ್ಡಸಾಲು_ಮೊತ್ತ = 0;

    ಗಾಗಿ (int64_t i03 = 0; i03 < ne03; i03++) {
        ಗಾಗಿ (int64_t i02 = 0; i02 < ne02; i02++) {
            ಗಾಗಿ (int64_t i01 = 0; i01 < ne01; i01++) {
                ggml_vec_ಮೊತ್ತ_f16_ggf(ne00,
                    &ಅಡ್ಡಸಾಲು_ಮೊತ್ತ,
                    (ggml_fp16_t *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i01 * nb01 + i02 * nb02 + i03 * nb03));
                ಮೊತ್ತ += ಅಡ್ಡಸಾಲು_ಮೊತ್ತ;
            }
        }
    }
    ((ggml_fp16_t *) dst->ದತ್ತಾಂಶ)[0] = GGML_CPU_FP32_ಗಮ್ಯ_FP16(ಮೊತ್ತ);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_ಮೊತ್ತ_bf16(
    ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
          ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆದರೆ (params->ith != 0) {
        ಹಿಂತಿರುಗಿ;
    }

    ಪ್ರತಿಜ್ಞಾ(ggml_is_scalar(dst));

    ಪ್ರತಿಜ್ಞಾ(src0->nb[0] == ಗಾತ್ರ(ggml_bf16_t));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)

    ದಶಮಾಂಶ ಮೊತ್ತ = 0;
    ದಶಮಾಂಶ ಅಡ್ಡಸಾಲು_ಮೊತ್ತ = 0;

    ಗಾಗಿ (int64_t i03 = 0; i03 < ne03; i03++) {
        ಗಾಗಿ (int64_t i02 = 0; i02 < ne02; i02++) {
            ಗಾಗಿ (int64_t i01 = 0; i01 < ne01; i01++) {
                ggml_vec_ಮೊತ್ತ_bf16_ggf(ne00,
                    &ಅಡ್ಡಸಾಲು_ಮೊತ್ತ,
                    (ggml_bf16_t *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i01 * nb01 + i02 * nb02 + i03 * nb03));
                ಮೊತ್ತ += ಅಡ್ಡಸಾಲು_ಮೊತ್ತ;
            }
        }
    }
    ((ggml_bf16_t *) dst->ದತ್ತಾಂಶ)[0] = GGML_FP32_ಗಮ್ಯ_BF16(ಮೊತ್ತ);
}

ಶೂನ್ಯ ggml_compute_forward_ಮೊತ್ತ(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_ಮೊತ್ತ_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            {
                ggml_compute_forward_ಮೊತ್ತ_f16(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
            {
                ggml_compute_forward_ಮೊತ್ತ_bf16(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_cumsum

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_cumsum_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ));
    GGML_ASSERT(dst->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne01);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ [ir0, ir1] = get_thread_range(params, src0);

    ಗಾಗಿ (int64_t ir = ir0; ir < ir1; ++ir) {
        ಸ್ಥಿರಾಂಕ int64_t i03 = ir/(ne02*ne01);
        ಸ್ಥಿರಾಂಕ int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        ಸ್ಥಿರಾಂಕ int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        ದಶಮಾಂಶ * src_ಅಡ್ಡಸಾಲು = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i01*nb01 + i02*nb02 + i03*nb03);
        ದಶಮಾಂಶ * dst_ಅಡ್ಡಸಾಲು = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ  + i01*nb1  + i02*nb2  + i03*nb3);

        ggml_vec_cumsum_f32(ne00, dst_ಅಡ್ಡಸಾಲು, src_ಅಡ್ಡಸಾಲು);
    }
}

ಶೂನ್ಯ ggml_compute_forward_cumsum(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_cumsum_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_sum_rows

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆದರೆ (params->ith != 0) {
        ಹಿಂತಿರುಗಿ;
    }

    GGML_ASSERT(src0->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ));
    GGML_ASSERT(dst->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(ne0 == 1);
    GGML_ASSERT(ne1 == ne01);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    ಗಾಗಿ (int64_t i3 = 0; i3 < ne03; i3++) {
        ಗಾಗಿ (int64_t i2 = 0; i2 < ne02; i2++) {
            ಗಾಗಿ (int64_t i1 = 0; i1 < ne01; i1++) {
                ದಶಮಾಂಶ * src_ಅಡ್ಡಸಾಲು = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i1*nb01 + i2*nb02 + i3*nb03);
                ದಶಮಾಂಶ * dst_ಅಡ್ಡಸಾಲು = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ  + i1*nb1  + i2*nb2  + i3*nb3);
                ದಶಮಾಂಶ ಅಡ್ಡಸಾಲು_ಮೊತ್ತ = 0;
                ggml_vec_ಮೊತ್ತ_f32(ne00, &ಅಡ್ಡಸಾಲು_ಮೊತ್ತ, src_ಅಡ್ಡಸಾಲು);
                dst_ಅಡ್ಡಸಾಲು[0] = ಅಡ್ಡಸಾಲು_ಮೊತ್ತ;
            }
        }
    }
}

ಶೂನ್ಯ ggml_compute_forward_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_mean

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_ಸರಾಸರಿ_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆದರೆ (params->ith != 0) {
        ಹಿಂತಿರುಗಿ;
    }

    ಪ್ರತಿಜ್ಞಾ(src0->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ));

    GGML_TENSOR_UNARY_OP_LOCALS

    ಪ್ರತಿಜ್ಞಾ(ne0 == 1);
    ಪ್ರತಿಜ್ಞಾ(ne1 == ne01);
    ಪ್ರತಿಜ್ಞಾ(ne2 == ne02);
    ಪ್ರತಿಜ್ಞಾ(ne3 == ne03);

    GGML_ಬಳಸದ(ne0);
    GGML_ಬಳಸದ(ne1);
    GGML_ಬಳಸದ(ne2);
    GGML_ಬಳಸದ(ne3);

    ಗಾಗಿ (int64_t i03 = 0; i03 < ne03; i03++) {
        ಗಾಗಿ (int64_t i02 = 0; i02 < ne02; i02++) {
            ಗಾಗಿ (int64_t i01 = 0; i01 < ne01; i01++) {
                ggml_vec_ಮೊತ್ತ_f32(ne00,
                        (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *)  dst->ದತ್ತಾಂಶ + i01*nb1  + i02*nb2  + i03*nb3),
                        (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i01*nb01 + i02*nb02 + i03*nb03));

                *(ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i01*nb1 + i02*nb2 + i03*nb3) /= (ದಶಮಾಂಶ) ne00;
            }
        }
    }
}

ಶೂನ್ಯ ggml_compute_forward_ಸರಾಸರಿ(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_ಸರಾಸರಿ_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_argmax

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_argmax_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆದರೆ (params->ith != 0) {
        ಹಿಂತಿರುಗಿ;
    }

    ಪ್ರತಿಜ್ಞಾ(src0->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಪ್ರತಿಜ್ಞಾ(dst->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಸ್ಥಿರಾಂಕ int64_t ne00 = src0->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t ne01 = src0->ne[1];

    ಸ್ಥಿರಾಂಕ size_t nb01 = src0->nb[1];
    ಸ್ಥಿರಾಂಕ size_t nb0 = dst->nb[0];

    ಗಾಗಿ (int64_t i1 = 0; i1 < ne01; i1++) {
        ದಶಮಾಂಶ * src = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i1*nb01);
        int32_t * dst_ = (int32_t *) ((ಅಕ್ಷರ *)  dst->ದತ್ತಾಂಶ + i1*nb0);
        ಪೂರ್ಣಾಂಕ v = 0;
        ggml_vec_argmax_f32(ne00, &v, src);
        dst_[0] = v;
    }
}

ಶೂನ್ಯ ggml_compute_forward_argmax(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_argmax_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_count_equal

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_ಎಣಿಕೆ_equal_i32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS;

    GGML_ASSERT(src0->type == GGML_TYPE_I32);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));
    GGML_ASSERT(dst->type == GGML_TYPE_I64);

    ಸ್ಥಿರಾಂಕ int64_t nr = ggml_nrows(src0);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    int64_t * sums = (int64_t *) params->wdata;
    int64_t ಮೊತ್ತ_thread = 0;

    // rows per thread
    ಸ್ಥಿರಾಂಕ int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ int64_t ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ int64_t ir1 = MIN(ir0 + dr, nr);

    ಗಾಗಿ (int64_t ir = ir0; ir < ir1; ++ir) {
        ಸ್ಥಿರಾಂಕ int64_t i03 =  ir                        / (ne02*ne01);
        ಸ್ಥಿರಾಂಕ int64_t i02 = (ir - i03*ne03)            /       ne01;
        ಸ್ಥಿರಾಂಕ int64_t i01 =  ir - i03*ne03 - i02*ne02;

        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * data0 = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i03*nb03 + i02*nb02 + i01*nb01;
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * data1 = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ + i03*nb13 + i02*nb12 + i01*nb11;

        ಗಾಗಿ (int64_t i00 = 0; i00 < ne00; ++i00) {
            ಸ್ಥಿರಾಂಕ int32_t val0 = *((ಸ್ಥಿರಾಂಕ int32_t *) (data0 + i00*nb00));
            ಸ್ಥಿರಾಂಕ int32_t val1 = *((ಸ್ಥಿರಾಂಕ int32_t *) (data1 + i00*nb10));

            ಮೊತ್ತ_thread += val0 == val1;
        }
    }
    ಆದರೆ (ith != 0) {
        sums[ith] = ಮೊತ್ತ_thread;
    }
    ggml_barrier(params->threadpool);

    ಆದರೆ (ith != 0) {
        ಹಿಂತಿರುಗಿ;
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ith_other = 1; ith_other < nth; ++ith_other) {
        ಮೊತ್ತ_thread += sums[ith_other];
    }
    *((int64_t *) dst->ದತ್ತಾಂಶ) = ಮೊತ್ತ_thread;
}

ಶೂನ್ಯ ggml_compute_forward_ಎಣಿಕೆ_equal(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_I32:
            {
                ggml_compute_forward_ಎಣಿಕೆ_equal_i32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_repeat

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_repeat_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆದರೆ (params->ith != 0) {
        ಹಿಂತಿರುಗಿ;
    }

    GGML_ASSERT(ggml_can_repeat(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    // guaranteed to be an integer due to the check in ggml_can_repeat
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr0 = (ಪೂರ್ಣಾಂಕ)(ne0/ne00);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr1 = (ಪೂರ್ಣಾಂಕ)(ne1/ne01);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr2 = (ಪೂರ್ಣಾಂಕ)(ne2/ne02);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr3 = (ಪೂರ್ಣಾಂಕ)(ne3/ne03);

    // TODO: support for transposed / permuted tensors
    GGML_ASSERT(nb0  == ಗಾತ್ರ(ದಶಮಾಂಶ));
    GGML_ASSERT(nb00 == ಗಾತ್ರ(ದಶಮಾಂಶ));

    // TODO: maybe this is not optimal?
    ಗಾಗಿ                         (ಪೂರ್ಣಾಂಕ i3 = 0; i3 < nr3;  i3++) {
        ಗಾಗಿ                     (ಪೂರ್ಣಾಂಕ k3 = 0; k3 < ne03; k3++) {
            ಗಾಗಿ                 (ಪೂರ್ಣಾಂಕ i2 = 0; i2 < nr2;  i2++) {
                ಗಾಗಿ             (ಪೂರ್ಣಾಂಕ k2 = 0; k2 < ne02; k2++) {
                    ಗಾಗಿ         (ಪೂರ್ಣಾಂಕ i1 = 0; i1 < nr1;  i1++) {
                        ಗಾಗಿ     (ಪೂರ್ಣಾಂಕ k1 = 0; k1 < ne01; k1++) {
                            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < nr0;  i0++) {
                                ggml_vec_cpy_f32(ne00,
                                        (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *)  dst->ದತ್ತಾಂಶ + (i3*ne03 + k3)*nb3  + (i2*ne02 + k2)*nb2  + (i1*ne01 + k1)*nb1  + (i0*ne00)*nb0),
                                        (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + (          k3)*nb03 + (          k2)*nb02 + (          k1)*nb01));
                            }
                        }
                    }
                }
            }
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_repeat_f16(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆದರೆ (params->ith != 0) {
        ಹಿಂತಿರುಗಿ;
    }

    GGML_ASSERT(ggml_can_repeat(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    // guaranteed to be an integer due to the check in ggml_can_repeat
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr0 = (ಪೂರ್ಣಾಂಕ)(ne0/ne00);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr1 = (ಪೂರ್ಣಾಂಕ)(ne1/ne01);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr2 = (ಪೂರ್ಣಾಂಕ)(ne2/ne02);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr3 = (ಪೂರ್ಣಾಂಕ)(ne3/ne03);

    // TODO: support for transposed / permuted tensors
    GGML_ASSERT(nb0  == ಗಾತ್ರ(ggml_fp16_t));
    GGML_ASSERT(nb00 == ಗಾತ್ರ(ggml_fp16_t));

    // TODO: maybe this is not optimal?
    ಗಾಗಿ                         (ಪೂರ್ಣಾಂಕ i3 = 0; i3 < nr3;  i3++) {
        ಗಾಗಿ                     (ಪೂರ್ಣಾಂಕ k3 = 0; k3 < ne03; k3++) {
            ಗಾಗಿ                 (ಪೂರ್ಣಾಂಕ i2 = 0; i2 < nr2;  i2++) {
                ಗಾಗಿ             (ಪೂರ್ಣಾಂಕ k2 = 0; k2 < ne02; k2++) {
                    ಗಾಗಿ         (ಪೂರ್ಣಾಂಕ i1 = 0; i1 < nr1;  i1++) {
                        ಗಾಗಿ     (ಪೂರ್ಣಾಂಕ k1 = 0; k1 < ne01; k1++) {
                            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < nr0;  i0++) {
                                ggml_fp16_t * y = (ggml_fp16_t *) ((ಅಕ್ಷರ *)  dst->ದತ್ತಾಂಶ + (i3*ne03 + k3)*nb3  + (i2*ne02 + k2)*nb2  + (i1*ne01 + k1)*nb1  + (i0*ne00)*nb0);
                                ggml_fp16_t * x = (ggml_fp16_t *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + (          k3)*nb03 + (          k2)*nb02 + (          k1)*nb01);
                                // ggml_vec_cpy_f16(ne00, y, x)
                                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ne00; ++i) {
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

ಶೂನ್ಯ ggml_compute_forward_repeat(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F16:
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
        ಸ್ಥಿತಿ GGML_TYPE_I16:
            {
                ggml_compute_forward_repeat_f16(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F32:
        ಸ್ಥಿತಿ GGML_TYPE_I32:
            {
                ggml_compute_forward_repeat_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        // TODO: templateify the implementation and support for I64
        //       ref https://github.com/ggml-org/llama.cpp/pull/14274#discussion_r2169492225
        //case GGML_TYPE_I64:
        //    {
        //        ggml_compute_forward_repeat_i64(params, dst);
        //    } break;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_repeat_back

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_repeat_ಹಿಂಭಾಗ_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆದರೆ (params->ith != 0) {
        ಹಿಂತಿರುಗಿ;
    }

    GGML_ASSERT(ggml_can_repeat(dst, src0));

    GGML_TENSOR_UNARY_OP_LOCALS

    // guaranteed to be an integer due to the check in ggml_can_repeat
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr0 = (ಪೂರ್ಣಾಂಕ)(ne00/ne0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr1 = (ಪೂರ್ಣಾಂಕ)(ne01/ne1);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr2 = (ಪೂರ್ಣಾಂಕ)(ne02/ne2);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr3 = (ಪೂರ್ಣಾಂಕ)(ne03/ne3);

    // TODO: support for transposed / permuted tensors
    GGML_ASSERT(nb0  == ಗಾತ್ರ(ದಶಮಾಂಶ));
    GGML_ASSERT(nb00 == ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಆದರೆ (ggml_is_contiguous(dst)) {
        ggml_vec_ಗಣ_f32(ne0*ne1*ne2*ne3, (ದಶಮಾಂಶ *)dst->ದತ್ತಾಂಶ, 0);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಗಾಗಿ         (ಪೂರ್ಣಾಂಕ k3 = 0; k3 < ne3; k3++) {
            ಗಾಗಿ     (ಪೂರ್ಣಾಂಕ k2 = 0; k2 < ne2; k2++) {
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k1 = 0; k1 < ne1; k1++) {
                    ggml_vec_ಗಣ_f32(ne0,
                        (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + k1*nb1 + k2*nb2 + k3*nb3),
                        0);
                }
            }
        }
    }

    // TODO: maybe this is not optimal?
    ಗಾಗಿ                         (ಪೂರ್ಣಾಂಕ i3 = 0; i3 < nr3; i3++) {
        ಗಾಗಿ                     (ಪೂರ್ಣಾಂಕ k3 = 0; k3 < ne3; k3++) {
            ಗಾಗಿ                 (ಪೂರ್ಣಾಂಕ i2 = 0; i2 < nr2; i2++) {
                ಗಾಗಿ             (ಪೂರ್ಣಾಂಕ k2 = 0; k2 < ne2; k2++) {
                    ಗಾಗಿ         (ಪೂರ್ಣಾಂಕ i1 = 0; i1 < nr1; i1++) {
                        ಗಾಗಿ     (ಪೂರ್ಣಾಂಕ k1 = 0; k1 < ne1; k1++) {
                            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < nr0; i0++) {
                                ggml_vec_acc_f32(ne0,
                                        (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *)  dst->ದತ್ತಾಂಶ + (         k3)*nb3  + (         k2)*nb2  + (         k1)*nb1),
                                        (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + (i3*ne3 + k3)*nb03 + (i2*ne2 + k2)*nb02 + (i1*ne1 + k1)*nb01 + (i0*ne0)*nb00));
                            }
                        }
                    }
                }
            }
        }
    }
}

ಶೂನ್ಯ ggml_compute_forward_repeat_ಹಿಂಭಾಗ(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_repeat_ಹಿಂಭಾಗ_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_concat
ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_concat_any(
    ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
    ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

    ಸ್ಥಿರಾಂಕ int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);
    GGML_ASSERT(ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(src0));
    GGML_ASSERT(ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(src1));

    int64_t o[4] = {0, 0, 0, 0};

    ಆದರೆ (dim == 0) {
        GGML_ASSERT(src0->ne[0] % ggml_blck_ಗಾತ್ರ(src0->type) == 0);
        GGML_ASSERT(src1->ne[0] % ggml_blck_ಗಾತ್ರ(src1->type) == 0);

        o[dim] = src0->ne[dim]/ggml_blck_ಗಾತ್ರ(src0->type);
    } ಇಲ್ಲದಿದ್ದರೆ {
        o[dim] = src0->ne[dim];
    }

    // Region 1: copy rows from src0
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i3 = 0; i3 < ne03; i3++) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i2 = ith; i2 < ne02; i2 += nth) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i1 = 0; i1 < ne01; i1++) {
                ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * x = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i1*nb01 + i2*nb02 + i3*nb03;
                      ಅಕ್ಷರ * y = (      ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ  + i1*nb1  + i2*nb2  + i3*nb3;
                ಸ್ಮರಣೆ_ನಕಲು(y, x, ggml_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(src0->type, ne00));
            }
        }
    }

    // Region 2: copy rows from src1, offset into dst by o[]
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i3 = 0; i3 < ne13; i3++) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i2 = ith; i2 < ne12; i2 += nth) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i1 = 0; i1 < ne11; i1++) {
                ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * x = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ + i1*nb11         + i2*nb12         + i3*nb13;
                      ಅಕ್ಷರ * y = (      ಅಕ್ಷರ *)  dst->ದತ್ತಾಂಶ + (i1 + o[1])*nb1 + (i2 + o[2])*nb2 + (i3 + o[3])*nb3 + o[0]*nb0;
                ಸ್ಮರಣೆ_ನಕಲು(y, x, ggml_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(src1->type, ne10));
            }
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_concat_i8(
    ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
    ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_type_ಗಾತ್ರ(src0->type) == ಗಾತ್ರ(int8_t));

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

    ಸ್ಥಿರಾಂಕ int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);

    int64_t o[4] = {0, 0, 0, 0};
    o[dim] = src0->ne[dim];

    ಸ್ಥಿರಾಂಕ int8_t * x;

    // TODO: smarter multi-theading
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i3 = 0; i3 < ne3; i3++) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i2 = ith; i2 < ne2; i2 += nth) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i1 = 0; i1 < ne1; i1++) {
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < ne0; i0++) {
                    ಆದರೆ (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                        x = (ಸ್ಥಿರಾಂಕ int8_t *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)src0->ದತ್ತಾಂಶ + (i0       )*nb00 + (i1       )*nb01 + (i2       )*nb02 + (i3       )*nb03);
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        x = (ಸ್ಥಿರಾಂಕ int8_t *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)src1->ದತ್ತಾಂಶ + (i0 - o[0])*nb10 + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13);
                    }

                    int8_t * y = (int8_t *)((ಅಕ್ಷರ *)dst->ದತ್ತಾಂಶ + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);

                    *y = *x;
                }
            }
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_concat_f16(
    ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
    ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_type_ಗಾತ್ರ(src0->type) == ಗಾತ್ರ(ggml_fp16_t));

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

    ಸ್ಥಿರಾಂಕ int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);

    int64_t o[4] = {0, 0, 0, 0};
    o[dim] = src0->ne[dim];

    ಸ್ಥಿರಾಂಕ ggml_fp16_t * x;

    // TODO: smarter multi-theading
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i3 = 0; i3 < ne3; i3++) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i2 = ith; i2 < ne2; i2 += nth) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i1 = 0; i1 < ne1; i1++) {
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < ne0; i0++) {
                    ಆದರೆ (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                        x = (ಸ್ಥಿರಾಂಕ ggml_fp16_t *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)src0->ದತ್ತಾಂಶ + (i0       )*nb00 + (i1       )*nb01 + (i2       )*nb02 + (i3       )*nb03);
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        x = (ಸ್ಥಿರಾಂಕ ggml_fp16_t *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)src1->ದತ್ತಾಂಶ + (i0 - o[0])*nb10 + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13);
                    }

                    ggml_fp16_t * y = (ggml_fp16_t *)((ಅಕ್ಷರ *)dst->ದತ್ತಾಂಶ + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);

                    *y = *x;
                }
            }
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_concat_f32(
    ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
    ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_type_ಗಾತ್ರ(src0->type) == ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

    ಸ್ಥಿರಾಂಕ int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);

    int64_t o[4] = {0, 0, 0, 0};
    o[dim] = src0->ne[dim];

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x;

    // TODO: smarter multi-theading
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i3 = 0; i3 < ne3; i3++) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i2 = ith; i2 < ne2; i2 += nth) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i1 = 0; i1 < ne1; i1++) {
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < ne0; i0++) {
                    ಆದರೆ (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                        x = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)src0->ದತ್ತಾಂಶ + (i0       )*nb00 + (i1       )*nb01 + (i2       )*nb02 + (i3       )*nb03);
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        x = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)src1->ದತ್ತಾಂಶ + (i0 - o[0])*nb10 + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13);
                    }

                    ದಶಮಾಂಶ * y = (ದಶಮಾಂಶ *)((ಅಕ್ಷರ *)dst->ದತ್ತಾಂಶ + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);

                    *y = *x;
                }
            }
        }
    }
}

ಶೂನ್ಯ ggml_compute_forward_concat(
    ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
    ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F16:
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
        ಸ್ಥಿತಿ GGML_TYPE_I16:
            {
                ggml_compute_forward_concat_f16(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_I8:
            {
                ggml_compute_forward_concat_i8(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F32:
        ಸ್ಥಿತಿ GGML_TYPE_I32:
            {
                ggml_compute_forward_concat_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                ggml_compute_forward_concat_any(params, dst);
            }
    }
}

// row loop shared by the BF16 unary ops
ಮಾದರಿ <ಶೂನ್ಯ (*vec_op)(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ, ggml_bf16_t *, ಸ್ಥಿರಾಂಕ ggml_bf16_t *)>
ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_unary_bf16_ಅಡ್ಡಸಾಲುಗಳು(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಪ್ರತಿಜ್ಞಾ(ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(src0));
    ಪ್ರತಿಜ್ಞಾ(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nc = src0->ne[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr = ggml_nrows(src0);

    // rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ir = ir0; ir < ir1; ++ir) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i3 = ir/(ne02*ne01);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i2 = (ir - i3*ne02*ne01)/ne01;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i1 = (ir - i3*ne02*ne01 - i2*ne01);

        vec_op(nc,
                (ggml_bf16_t *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_bf16_t *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i3*nb03 + i2*nb02 + i1*nb01));
    }
}

// ggml_compute_forward_gelu

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_gelu_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಪ್ರತಿಜ್ಞಾ(ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(src0));
    ಪ್ರತಿಜ್ಞಾ(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nc = src0->ne[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr = ggml_nrows(src0);

    // rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ir = ir0; ir < ir1; ++ir) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i3 = ir/(ne02*ne01);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i2 = (ir - i3*ne02*ne01)/ne01;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_f32(nc,
                (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ  + i3*nb3  + i2*nb2  + i1*nb1),
                (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i3*nb03 + i2*nb02 + i1*nb01));

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NDEBUG
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < nc; k++) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x = ((ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_ಬಳಸದ(x);
            ಪ್ರತಿಜ್ಞಾ(!isnan(x));
            ಪ್ರತಿಜ್ಞಾ(!isinf(x));
        }
#ಆದರೆ_ಅಂತ್ಯ // NDEBUG
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_gelu_f16(
    ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
    ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಪ್ರತಿಜ್ಞಾ(ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(src0));
    ಪ್ರತಿಜ್ಞಾ(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nc = src0->ne[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr = ggml_nrows(src0);

    // rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ir = ir0; ir < ir1; ++ir) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i3 = ir/(ne02*ne01);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i2 = (ir - i3*ne02*ne01)/ne01;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_f16(nc,
                (ggml_fp16_t *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i3*nb03 + i2*nb02 + i1*nb01));

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NDEBUG
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < nc; k++) {
            ಸ್ಥಿರಾಂಕ ggml_fp16_t x = ((ggml_fp16_t *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ v = GGML_CPU_FP16_ಗಮ್ಯ_FP32(x);
            GGML_ಬಳಸದ(v);
            ಪ್ರತಿಜ್ಞಾ(!isnan(v));
            ಪ್ರತಿಜ್ಞಾ(!isinf(v));
        }
#ಆದರೆ_ಅಂತ್ಯ // NDEBUG
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_gelu_bf16(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_ಅಡ್ಡಸಾಲುಗಳು<ggml_vec_gelu_bf16>(params, dst);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_gelu(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_gelu_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            {
                ggml_compute_forward_gelu_f16(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
            {
                ggml_compute_forward_gelu_bf16(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_fill

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_fill_f32(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ c = ggml_get_op_params_f32(dst, 0);

    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne);
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb);

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ [ir0, ir1] = get_thread_range(params, dst);

    ಗಾಗಿ (int64_t ir = ir0; ir < ir1; ++ir) {
        ಸ್ಥಿರಾಂಕ int64_t i03 = ir/(ne2*ne1);
        ಸ್ಥಿರಾಂಕ int64_t i02 = (ir - i03*ne2*ne1)/ne1;
        ಸ್ಥಿರಾಂಕ int64_t i01 = (ir - i03*ne2*ne1 - i02*ne1);

        ದಶಮಾಂಶ * dst_ptr  = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i03*nb3 + i02*nb2 + i01*nb1);

        ggml_vec_ಗಣ_f32(ne0, dst_ptr, c);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_fill_f16(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_fp16_t c = GGML_CPU_FP32_ಗಮ್ಯ_FP16(ggml_get_op_params_f32(dst, 0));

    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne);
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb);

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ [ir0, ir1] = get_thread_range(params, dst);

    ಗಾಗಿ (int64_t ir = ir0; ir < ir1; ++ir) {
        ಸ್ಥಿರಾಂಕ int64_t i03 = ir/(ne2*ne1);
        ಸ್ಥಿರಾಂಕ int64_t i02 = (ir - i03*ne2*ne1)/ne1;
        ಸ್ಥಿರಾಂಕ int64_t i01 = (ir - i03*ne2*ne1 - i02*ne1);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i03*nb3 + i02*nb2 + i01*nb1);

        ggml_vec_ಗಣ_f16(ne0, dst_ptr, c);
    }
}

ಶೂನ್ಯ ggml_compute_forward_fill(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_fill_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            {
                ggml_compute_forward_fill_f16(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("unsupported type for ggml_compute_forward_fill: %s", ggml_type_name(src0->type));
            }
    }
}

// ggml_compute_tri

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_tri_f32(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಸ್ಥಿರಾಂಕ ggml_tri_type ttype = (ggml_tri_type) ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_TENSOR_UNARY_OP_LOCALS

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ [ir0, ir1] = get_thread_range(params, src0);

    ತಾರ್ಕಿಕ (*bipred)(ಪೂರ್ಣಾಂಕ, ಪೂರ್ಣಾಂಕ);

    ಆಯ್ಕೆ (ttype) {
        ಸ್ಥಿತಿ GGML_TRI_TYPE_ಅಧೋ:      bipred = [](ಪೂರ್ಣಾಂಕ i, ಪೂರ್ಣಾಂಕ r) { ಹಿಂತಿರುಗಿ i <  r; }; ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TRI_TYPE_ಅಧೋ_DIAG: bipred = [](ಪೂರ್ಣಾಂಕ i, ಪೂರ್ಣಾಂಕ r) { ಹಿಂತಿರುಗಿ i <= r; }; ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TRI_TYPE_ಊರ್ಧ್ವ:      bipred = [](ಪೂರ್ಣಾಂಕ i, ಪೂರ್ಣಾಂಕ r) { ಹಿಂತಿರುಗಿ i >  r; }; ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TRI_TYPE_ಊರ್ಧ್ವ_DIAG: bipred = [](ಪೂರ್ಣಾಂಕ i, ಪೂರ್ಣಾಂಕ r) { ಹಿಂತಿರುಗಿ i >= r; }; ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: GGML_ABORT("invalid tri type");
    }

    ಗಾಗಿ (int64_t ir = ir0; ir < ir1; ++ir) {
        ಸ್ಥಿರಾಂಕ int64_t i03 = ir/(ne02*ne01);
        ಸ್ಥಿರಾಂಕ int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        ಸ್ಥಿರಾಂಕ int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src_ptr = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ  *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i03*nb03 + i02*nb02 + i01*nb01);
              ದಶಮಾಂಶ * dst_ptr = (      ದಶಮಾಂಶ  *) ((      ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ  + i03*nb3  + i02*nb2  + i01*nb1);

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < ne0; ++i0) {
            dst_ptr[i0] = bipred(i0, i01) ? src_ptr[i0] : 0.0f;
        }
    }
}

ಶೂನ್ಯ ggml_compute_forward_tri(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_tri_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_gelu_erf

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_gelu_erf_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಪ್ರತಿಜ್ಞಾ(ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(src0));
    ಪ್ರತಿಜ್ಞಾ(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nc = src0->ne[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr = ggml_nrows(src0);

    // rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ir = ir0; ir < ir1; ++ir) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i3 = ir/(ne02*ne01);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i2 = (ir - i3*ne02*ne01)/ne01;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_erf_f32(nc,
                (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ  + i3*nb3  + i2*nb2  + i1*nb1),
                (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i3*nb03 + i2*nb02 + i1*nb01));

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NDEBUG
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < nc; k++) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x = ((ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_ಬಳಸದ(x);
            ಪ್ರತಿಜ್ಞಾ(!isnan(x));
            ಪ್ರತಿಜ್ಞಾ(!isinf(x));
        }
#ಆದರೆ_ಅಂತ್ಯ // NDEBUG
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_gelu_erf_f16(
    ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
    ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಪ್ರತಿಜ್ಞಾ(ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(src0));
    ಪ್ರತಿಜ್ಞಾ(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nc = src0->ne[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr = ggml_nrows(src0);

    // rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ir = ir0; ir < ir1; ++ir) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i3 = ir/(ne02*ne01);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i2 = (ir - i3*ne02*ne01)/ne01;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_erf_f16(nc,
                (ggml_fp16_t *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i3*nb03 + i2*nb02 + i1*nb01));

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NDEBUG
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < nc; k++) {
            ಸ್ಥಿರಾಂಕ ggml_fp16_t x = ((ggml_fp16_t *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ v = GGML_CPU_FP16_ಗಮ್ಯ_FP32(x);
            GGML_ಬಳಸದ(v);
            ಪ್ರತಿಜ್ಞಾ(!isnan(v));
            ಪ್ರತಿಜ್ಞಾ(!isinf(v));
        }
#ಆದರೆ_ಅಂತ್ಯ // NDEBUG
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_gelu_erf_bf16(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_ಅಡ್ಡಸಾಲುಗಳು<ggml_vec_gelu_erf_bf16>(params, dst);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_gelu_erf(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_gelu_erf_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            {
                ggml_compute_forward_gelu_erf_f16(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
            {
                ggml_compute_forward_gelu_erf_bf16(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_gelu_quick

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_gelu_ಶೀಘ್ರ_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಪ್ರತಿಜ್ಞಾ(ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(src0));
    ಪ್ರತಿಜ್ಞಾ(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nc = src0->ne[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr = ggml_nrows(src0);

    // rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ir = ir0; ir < ir1; ++ir) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i3 = ir/(ne02*ne01);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i2 = (ir - i3*ne02*ne01)/ne01;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_ಶೀಘ್ರ_f32(nc,
                (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ  + i3*nb3  + i2*nb2  + i1*nb1),
                (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i3*nb03 + i2*nb02 + i1*nb01));

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NDEBUG
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < nc; k++) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x = ((ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_ಬಳಸದ(x);
            ಪ್ರತಿಜ್ಞಾ(!isnan(x));
            ಪ್ರತಿಜ್ಞಾ(!isinf(x));
        }
#ಆದರೆ_ಅಂತ್ಯ // NDEBUG
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_gelu_ಶೀಘ್ರ_f16(
    ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
    ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಪ್ರತಿಜ್ಞಾ(ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(src0));
    ಪ್ರತಿಜ್ಞಾ(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nc = src0->ne[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr = ggml_nrows(src0);

    // rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ir = ir0; ir < ir1; ++ir) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i3 = ir/(ne02*ne01);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i2 = (ir - i3*ne02*ne01)/ne01;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_ಶೀಘ್ರ_f16(nc,
                (ggml_fp16_t *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i3*nb03 + i2*nb02 + i1*nb01));

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NDEBUG
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < nc; k++) {
            ಸ್ಥಿರಾಂಕ ggml_fp16_t x = ((ggml_fp16_t *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ v = GGML_CPU_FP16_ಗಮ್ಯ_FP32(x);
            GGML_ಬಳಸದ(v);
            ಪ್ರತಿಜ್ಞಾ(!isnan(v));
            ಪ್ರತಿಜ್ಞಾ(!isinf(v));
        }
#ಆದರೆ_ಅಂತ್ಯ // NDEBUG
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_gelu_ಶೀಘ್ರ_bf16(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_ಅಡ್ಡಸಾಲುಗಳು<ggml_vec_gelu_ಶೀಘ್ರ_bf16>(params, dst);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_gelu_ಶೀಘ್ರ(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_gelu_ಶೀಘ್ರ_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            {
                ggml_compute_forward_gelu_ಶೀಘ್ರ_f16(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
            {
                ggml_compute_forward_gelu_ಶೀಘ್ರ_bf16(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_silu

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_silu_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಪ್ರತಿಜ್ಞಾ(ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(src0));
    ಪ್ರತಿಜ್ಞಾ(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nc = src0->ne[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr = ggml_nrows(src0);

    // rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ir = ir0; ir < ir1; ++ir) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i3 = ir/(ne02*ne01);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i2 = (ir - i3*ne02*ne01)/ne01;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_silu_f32(nc,
                (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ  + i3*nb3  + i2*nb2  + i1*nb1),
                (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i3*nb03 + i2*nb02 + i1*nb01));

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NDEBUG
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < nc; k++) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x = ((ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_ಬಳಸದ(x);
            ಪ್ರತಿಜ್ಞಾ(!isnan(x));
            ಪ್ರತಿಜ್ಞಾ(!isinf(x));
        }
#ಆದರೆ_ಅಂತ್ಯ // NDEBUG
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_silu_f16(
    ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
    ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಪ್ರತಿಜ್ಞಾ(ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(src0));
    ಪ್ರತಿಜ್ಞಾ(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nc = src0->ne[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr = ggml_nrows(src0);

    // rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ir = ir0; ir < ir1; ++ir) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i3 = ir/(ne02*ne01);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i2 = (ir - i3*ne02*ne01)/ne01;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_silu_f16(nc,
                (ggml_fp16_t *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i3*nb03 + i2*nb02 + i1*nb01));

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NDEBUG
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < nc; k++) {
            ಸ್ಥಿರಾಂಕ ggml_fp16_t x = ((ggml_fp16_t *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ v = GGML_CPU_FP16_ಗಮ್ಯ_FP32(x);
            GGML_ಬಳಸದ(v);
            ಪ್ರತಿಜ್ಞಾ(!isnan(v));
            ಪ್ರತಿಜ್ಞಾ(!isinf(v));
        }
#ಆದರೆ_ಅಂತ್ಯ // NDEBUG
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_silu_bf16(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_ಅಡ್ಡಸಾಲುಗಳು<ggml_vec_silu_bf16>(params, dst);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_silu(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_silu_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            {
                ggml_compute_forward_silu_f16(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
            {
                ggml_compute_forward_silu_bf16(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}
// ggml_compute_forward_leaky_relu

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_leaky_relu_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆದರೆ (params->ith != 0) {
        ಹಿಂತಿರುಗಿ;
    }

    ಪ್ರತಿಜ್ಞಾ(ggml_is_contiguous_1(src0));
    ಪ್ರತಿಜ್ಞಾ(ggml_is_contiguous_1(dst));
    ಪ್ರತಿಜ್ಞಾ(ggml_are_same_shape(src0, dst));

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n  = ggml_nrows(src0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nc = src0->ne[0];

    ದಶಮಾಂಶ negative_ಇಳಿಜಾರು;
    ಸ್ಮರಣೆ_ನಕಲು(&negative_ಇಳಿಜಾರು, dst->op_params, ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಪ್ರತಿಜ್ಞಾ(dst->nb[0]  == ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಪ್ರತಿಜ್ಞಾ(src0->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n; i++) {
        ggml_vec_leaky_relu_f32(nc,
                (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ  + i*( dst->nb[1])),
                (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i*(src0->nb[1])), negative_ಇಳಿಜಾರು);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_leaky_relu_f16(
    ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
    ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆದರೆ (params->ith != 0) {
        ಹಿಂತಿರುಗಿ;
    }

    ಪ್ರತಿಜ್ಞಾ(ggml_is_contiguous_1(src0));
    ಪ್ರತಿಜ್ಞಾ(ggml_is_contiguous_1(dst));
    ಪ್ರತಿಜ್ಞಾ(ggml_are_same_shape(src0, dst));

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n  = ggml_nrows(src0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nc = src0->ne[0];

    ದಶಮಾಂಶ negative_ಇಳಿಜಾರು;
    ಸ್ಮರಣೆ_ನಕಲು(&negative_ಇಳಿಜಾರು, dst->op_params, ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಪ್ರತಿಜ್ಞಾ(dst->nb[0]  == ಗಾತ್ರ(ggml_fp16_t));
    ಪ್ರತಿಜ್ಞಾ(src0->nb[0] == ಗಾತ್ರ(ggml_fp16_t));

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n; i++) {
        ggml_vec_leaky_relu_f16(nc,
                (ggml_fp16_t *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ  + i*( dst->nb[1])),
                (ggml_fp16_t *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i*(src0->nb[1])), negative_ಇಳಿಜಾರು);
    }
}

ಶೂನ್ಯ ggml_compute_forward_leaky_relu(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_leaky_relu_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            {
                ggml_compute_forward_leaky_relu_f16(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_silu_back

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_silu_ಹಿಂಭಾಗ_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * grad = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    ಪ್ರತಿಜ್ಞಾ(ggml_is_contiguous_1(grad));
    ಪ್ರತಿಜ್ಞಾ(ggml_is_contiguous_1(src1));
    ಪ್ರತಿಜ್ಞಾ(ggml_is_contiguous_1(dst));
    ಪ್ರತಿಜ್ಞಾ(ggml_are_same_shape(src1, dst));
    ಪ್ರತಿಜ್ಞಾ(ggml_are_same_shape(src1, grad));

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nc = src1->ne[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr = ggml_nrows(src1);

    // rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i1 = ir0; i1 < ir1; i1++) {
        ggml_vec_silu_backward_f32(nc,
                (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ  + i1*( dst->nb[1])),
                (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ + i1*(src1->nb[1])),
                (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) grad->ದತ್ತಾಂಶ + i1*(grad->nb[1])));

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NDEBUG
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < nc; k++) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x = ((ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i1*( dst->nb[1])))[k];
            GGML_ಬಳಸದ(x);
            ಪ್ರತಿಜ್ಞಾ(!isnan(x));
            ಪ್ರತಿಜ್ಞಾ(!isinf(x));
        }
#ಆದರೆ_ಅಂತ್ಯ // NDEBUG
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_silu_ಹಿಂಭಾಗ_f16(
    ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
    ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * grad = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    ಪ್ರತಿಜ್ಞಾ(ggml_is_contiguous_1(grad));
    ಪ್ರತಿಜ್ಞಾ(ggml_is_contiguous_1(src1));
    ಪ್ರತಿಜ್ಞಾ(ggml_is_contiguous_1(dst));
    ಪ್ರತಿಜ್ಞಾ(ggml_are_same_shape(src1, dst));
    ಪ್ರತಿಜ್ಞಾ(ggml_are_same_shape(src1, grad));

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nc = src1->ne[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr = ggml_nrows(src1);

    // rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i1 = ir0; i1 < ir1; i1++) {
        ggml_vec_silu_backward_f16(nc,
                (ggml_fp16_t *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ  + i1*( dst->nb[1])),
                (ggml_fp16_t *) ((ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ + i1*(src1->nb[1])),
                (ggml_fp16_t *) ((ಅಕ್ಷರ *) grad->ದತ್ತಾಂಶ + i1*(grad->nb[1])));

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NDEBUG
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < nc; k++) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x = ((ggml_fp16_t *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i1*( dst->nb[1])))[k];
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ v = GGML_CPU_FP16_ಗಮ್ಯ_FP32(x);
            GGML_ಬಳಸದ(v);
            ಪ್ರತಿಜ್ಞಾ(!isnan(v));
            ಪ್ರತಿಜ್ಞಾ(!isinf(v));
        }
#ಆದರೆ_ಅಂತ್ಯ // NDEBUG
    }
}

ಶೂನ್ಯ ggml_compute_forward_silu_ಹಿಂಭಾಗ(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_silu_ಹಿಂಭಾಗ_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            {
                ggml_compute_forward_silu_ಹಿಂಭಾಗ_f16(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_reglu

// row loop shared by the BF16 GLU ops
ಮಾದರಿ <ಶೂನ್ಯ (*vec_op)(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ, ggml_bf16_t *, ಸ್ಥಿರಾಂಕ ggml_bf16_t *, ಸ್ಥಿರಾಂಕ ggml_bf16_t *)>
ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_glu_bf16_ಅಡ್ಡಸಾಲುಗಳು(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];
    ಅಕ್ಷರ * src0_d = (ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ;
    ಅಕ್ಷರ * src1_d = (ಅಕ್ಷರ *) (src1 ? src1->ದತ್ತಾಂಶ : src0->ದತ್ತಾಂಶ);
    ಸ್ಥಿರಾಂಕ size_t src0_o = src0->nb[1];
    ಸ್ಥಿರಾಂಕ size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ಆದರೆ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ಸ್ಥಿರಾಂಕ int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i1 = ir0; i1 < ir1; i1++) {
        ಸ್ಥಿರಾಂಕ ggml_bf16_t * src0_p = (ಸ್ಥಿರಾಂಕ ggml_bf16_t *) (src0_d + i1*src0_o);
        ಸ್ಥಿರಾಂಕ ggml_bf16_t * src1_p = (ಸ್ಥಿರಾಂಕ ggml_bf16_t *) (src1_d + i1*src1_o);
        ggml_bf16_t       * dst_p  = (ggml_bf16_t *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i1*(dst->nb[1]));

        ಆದರೆ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        vec_op(nc, dst_p, src0_p, src1_p);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_reglu_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];
    ಅಕ್ಷರ * src0_d = (ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ;
    ಅಕ್ಷರ * src1_d = (ಅಕ್ಷರ *) (src1 ? src1->ದತ್ತಾಂಶ : src0->ದತ್ತಾಂಶ);
    ಸ್ಥಿರಾಂಕ size_t src0_o = src0->nb[1];
    ಸ್ಥಿರಾಂಕ size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ಆದರೆ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ಸ್ಥಿರಾಂಕ int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i1 = ir0; i1 < ir1; i1++) {
        ದಶಮಾಂಶ * src0_p = (ದಶಮಾಂಶ *) (src0_d + i1*src0_o);
        ದಶಮಾಂಶ * src1_p = (ದಶಮಾಂಶ *) (src1_d + i1*src1_o);

        ಆದರೆ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_reglu_f32(nc, (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i1*(dst->nb[1])), src0_p, src1_p);

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NDEBUG
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < nc; k++) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x = ((ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i1*( dst->nb[1])))[k];
            GGML_ಬಳಸದ(x);
            ಪ್ರತಿಜ್ಞಾ(!isnan(x));
            ಪ್ರತಿಜ್ಞಾ(!isinf(x));
        }
#ಆದರೆ_ಅಂತ್ಯ // NDEBUG
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_reglu_f16(
    ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
    ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];
    ಅಕ್ಷರ * src0_d = (ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ;
    ಅಕ್ಷರ * src1_d = (ಅಕ್ಷರ *) (src1 ? src1->ದತ್ತಾಂಶ : src0->ದತ್ತಾಂಶ);
    ಸ್ಥಿರಾಂಕ size_t src0_o = src0->nb[1];
    ಸ್ಥಿರಾಂಕ size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ಆದರೆ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ಸ್ಥಿರಾಂಕ int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        ಆದರೆ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_reglu_f16(nc, (ggml_fp16_t *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i1*(dst->nb[1])), src0_p, src1_p);

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NDEBUG
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < nc; k++) {
            ಸ್ಥಿರಾಂಕ ggml_fp16_t x = ((ggml_fp16_t *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i1*( dst->nb[1])))[k];
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ v = GGML_FP16_ಗಮ್ಯ_FP32(x);
            GGML_ಬಳಸದ(v);
            ಪ್ರತಿಜ್ಞಾ(!isnan(v));
            ಪ್ರತಿಜ್ಞಾ(!isinf(v));
        }
#ಆದರೆ_ಅಂತ್ಯ // NDEBUG
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_reglu_bf16(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_ಅಡ್ಡಸಾಲುಗಳು<ggml_vec_reglu_bf16>(params, dst);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_reglu(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_reglu_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            {
                ggml_compute_forward_reglu_f16(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
            {
                ggml_compute_forward_reglu_bf16(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_geglu

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_geglu_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];
    ಅಕ್ಷರ * src0_d = (ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ;
    ಅಕ್ಷರ * src1_d = (ಅಕ್ಷರ *) (src1 ? src1->ದತ್ತಾಂಶ : src0->ದತ್ತಾಂಶ);
    ಸ್ಥಿರಾಂಕ size_t src0_o = src0->nb[1];
    ಸ್ಥಿರಾಂಕ size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ಆದರೆ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ಸ್ಥಿರಾಂಕ int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i1 = ir0; i1 < ir1; i1++) {
        ದಶಮಾಂಶ * src0_p = (ದಶಮಾಂಶ *) (src0_d + i1*src0_o);
        ದಶಮಾಂಶ * src1_p = (ದಶಮಾಂಶ *) (src1_d + i1*src1_o);

        ಆದರೆ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_f32(nc, (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i1*(dst->nb[1])), src0_p, src1_p);

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NDEBUG
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < nc; k++) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x = ((ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i1*( dst->nb[1])))[k];
            GGML_ಬಳಸದ(x);
            ಪ್ರತಿಜ್ಞಾ(!isnan(x));
            ಪ್ರತಿಜ್ಞಾ(!isinf(x));
        }
#ಆದರೆ_ಅಂತ್ಯ // NDEBUG
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_geglu_f16(
    ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
    ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];
    ಅಕ್ಷರ * src0_d = (ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ;
    ಅಕ್ಷರ * src1_d = (ಅಕ್ಷರ *) (src1 ? src1->ದತ್ತಾಂಶ : src0->ದತ್ತಾಂಶ);
    ಸ್ಥಿರಾಂಕ size_t src0_o = src0->nb[1];
    ಸ್ಥಿರಾಂಕ size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ಆದರೆ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ಸ್ಥಿರಾಂಕ int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        ಆದರೆ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_f16(nc, (ggml_fp16_t *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i1*(dst->nb[1])), src0_p, src1_p);

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NDEBUG
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < nc; k++) {
            ಸ್ಥಿರಾಂಕ ggml_fp16_t x = ((ggml_fp16_t *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i1*( dst->nb[1])))[k];
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ v = GGML_FP16_ಗಮ್ಯ_FP32(x);
            GGML_ಬಳಸದ(v);
            ಪ್ರತಿಜ್ಞಾ(!isnan(v));
            ಪ್ರತಿಜ್ಞಾ(!isinf(v));
        }
#ಆದರೆ_ಅಂತ್ಯ // NDEBUG
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_geglu_bf16(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_ಅಡ್ಡಸಾಲುಗಳು<ggml_vec_geglu_bf16>(params, dst);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_geglu(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_geglu_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            {
                ggml_compute_forward_geglu_f16(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
            {
                ggml_compute_forward_geglu_bf16(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_swiglu

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_swiglu_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];
    ಅಕ್ಷರ * src0_d = (ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ;
    ಅಕ್ಷರ * src1_d = (ಅಕ್ಷರ *) (src1 ? src1->ದತ್ತಾಂಶ : src0->ದತ್ತಾಂಶ);
    ಸ್ಥಿರಾಂಕ size_t src0_o = src0->nb[1];
    ಸ್ಥಿರಾಂಕ size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ಆದರೆ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ಸ್ಥಿರಾಂಕ int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i1 = ir0; i1 < ir1; i1++) {
        ದಶಮಾಂಶ * src0_p = (ದಶಮಾಂಶ *) (src0_d + i1*src0_o);
        ದಶಮಾಂಶ * src1_p = (ದಶಮಾಂಶ *) (src1_d + i1*src1_o);

        ಆದರೆ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_swiglu_f32(nc, (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i1*(dst->nb[1])), src0_p, src1_p);

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NDEBUG
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < nc; k++) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x = ((ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i1*( dst->nb[1])))[k];
            GGML_ಬಳಸದ(x);
            ಪ್ರತಿಜ್ಞಾ(!isnan(x));
            ಪ್ರತಿಜ್ಞಾ(!isinf(x));
        }
#ಆದರೆ_ಅಂತ್ಯ // NDEBUG
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_swiglu_f16(
    ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
    ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];
    ಅಕ್ಷರ * src0_d = (ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ;
    ಅಕ್ಷರ * src1_d = (ಅಕ್ಷರ *) (src1 ? src1->ದತ್ತಾಂಶ : src0->ದತ್ತಾಂಶ);
    ಸ್ಥಿರಾಂಕ size_t src0_o = src0->nb[1];
    ಸ್ಥಿರಾಂಕ size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ಆದರೆ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ಸ್ಥಿರಾಂಕ int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        ಆದರೆ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_swiglu_f16(nc, (ggml_fp16_t *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i1*(dst->nb[1])), src0_p, src1_p);

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NDEBUG
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < nc; k++) {
            ಸ್ಥಿರಾಂಕ ggml_fp16_t x = ((ggml_fp16_t *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i1*( dst->nb[1])))[k];
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ v = GGML_FP16_ಗಮ್ಯ_FP32(x);
            GGML_ಬಳಸದ(v);
            ಪ್ರತಿಜ್ಞಾ(!isnan(v));
            ಪ್ರತಿಜ್ಞಾ(!isinf(v));
        }
#ಆದರೆ_ಅಂತ್ಯ // NDEBUG
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_swiglu_bf16(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_ಅಡ್ಡಸಾಲುಗಳು<ggml_vec_swiglu_bf16>(params, dst);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_swiglu(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_swiglu_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            {
                ggml_compute_forward_swiglu_f16(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
            {
                ggml_compute_forward_swiglu_bf16(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_swiglu_oai

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_swiglu_oai_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];
    ಅಕ್ಷರ * src0_d = (ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ;
    ಅಕ್ಷರ * src1_d = (ಅಕ್ಷರ *) (src1 ? src1->ದತ್ತಾಂಶ : src0->ದತ್ತಾಂಶ);
    ಸ್ಥಿರಾಂಕ size_t src0_o = src0->nb[1];
    ಸ್ಥಿರಾಂಕ size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ಆದರೆ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ಸ್ಥಿರಾಂಕ int32_t swapped = ggml_get_op_params_i32(dst, 1);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ alpha = ggml_get_op_params_f32(dst, 2);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಎಲ್ಲೆ = ggml_get_op_params_f32(dst, 3);

    // rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i1 = ir0; i1 < ir1; i1++) {
        ದಶಮಾಂಶ * src0_p = (ದಶಮಾಂಶ *) (src0_d + i1*src0_o);
        ದಶಮಾಂಶ * src1_p = (ದಶಮಾಂಶ *) (src1_d + i1*src1_o);
        ದಶಮಾಂಶ * dst_p  = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i1*(dst->nb[1]));

        ಆದರೆ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < nc; k++) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x = std::min(src0_p[k], ಎಲ್ಲೆ);
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ y = std::clamp(src1_p[k], -ಎಲ್ಲೆ, ಎಲ್ಲೆ);
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಹೊರಹೋಗು_glu = x / (1.f + expf(alpha * (-x)));
            dst_p[k] = ಹೊರಹೋಗು_glu * (y + 1.f);
        }

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NDEBUG
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < nc; k++) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x = dst_p[k];
            GGML_ಬಳಸದ(x);
            ಪ್ರತಿಜ್ಞಾ(!isnan(x));
            ಪ್ರತಿಜ್ಞಾ(!isinf(x));
        }
#ಆದರೆ_ಅಂತ್ಯ // NDEBUG
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_swiglu_oai_f16(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];
    ಅಕ್ಷರ * src0_d = (ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ;
    ಅಕ್ಷರ * src1_d = (ಅಕ್ಷರ *) (src1 ? src1->ದತ್ತಾಂಶ : src0->ದತ್ತಾಂಶ);
    ಸ್ಥಿರಾಂಕ size_t src0_o = src0->nb[1];
    ಸ್ಥಿರಾಂಕ size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ಆದರೆ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ಸ್ಥಿರಾಂಕ int32_t swapped = ggml_get_op_params_i32(dst, 1);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ alpha = ggml_get_op_params_f32(dst, 2);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಎಲ್ಲೆ = ggml_get_op_params_f32(dst, 3);

    // rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);
        ggml_fp16_t * dst_p  = (ggml_fp16_t *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i1*(dst->nb[1]));

        ಆದರೆ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < nc; k++) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x = std::min(GGML_CPU_FP16_ಗಮ್ಯ_FP32(src0_p[k]), ಎಲ್ಲೆ);
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ y = std::clamp(GGML_CPU_FP16_ಗಮ್ಯ_FP32(src1_p[k]), -ಎಲ್ಲೆ, ಎಲ್ಲೆ);
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಹೊರಹೋಗು_glu = x / (1.f + expf(alpha * (-x)));
            dst_p[k] = GGML_CPU_FP32_ಗಮ್ಯ_FP16(ಹೊರಹೋಗು_glu * (y + 1.f));
        }

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NDEBUG
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < nc; k++) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x = GGML_CPU_FP16_ಗಮ್ಯ_FP32(dst_p[k]);
            GGML_ಬಳಸದ(x);
            ಪ್ರತಿಜ್ಞಾ(!isnan(x));
            ಪ್ರತಿಜ್ಞಾ(!isinf(x));
        }
#ಆದರೆ_ಅಂತ್ಯ // NDEBUG
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_swiglu_oai(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_swiglu_oai_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            {
                ggml_compute_forward_swiglu_oai_f16(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_swiglu_clamp

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_swiglu_clamp_f32(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0   = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1   = dst->src[1];
    ಅಕ್ಷರ *              src0_d = (ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ;
    ಅಕ್ಷರ *              src1_d = (ಅಕ್ಷರ *) (src1 ? src1->ದತ್ತಾಂಶ : src0->ದತ್ತಾಂಶ);
    ಸ್ಥಿರಾಂಕ size_t        src0_o = src0->nb[1];
    ಸ್ಥಿರಾಂಕ size_t        src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ಆದರೆ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ಸ್ಥಿರಾಂಕ int32_t swapped = ggml_get_op_params_i32(dst, 1);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   ಎಲ್ಲೆ   = ggml_get_op_params_f32(dst, 3);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr  = (nr + nth - 1) / nth;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr * ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i1 = ir0; i1 < ir1; i1++) {
        ದಶಮಾಂಶ * src0_p = (ದಶಮಾಂಶ *) (src0_d + i1 * src0_o);
        ದಶಮಾಂಶ * src1_p = (ದಶಮಾಂಶ *) (src1_d + i1 * src1_o);
        ದಶಮಾಂಶ * dst_p  = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i1 * (dst->nb[1]));

        ಆದರೆ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < nc; k++) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ gate = std::min(src0_p[k], ಎಲ್ಲೆ);
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಮೇಲೆ   = std::clamp(src1_p[k], -ಎಲ್ಲೆ, ಎಲ್ಲೆ);
            dst_p[k]         = gate / (1.f + expf(-gate)) * ಮೇಲೆ;
        }

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NDEBUG
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < nc; k++) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x = dst_p[k];
            GGML_ಬಳಸದ(x);
            ಪ್ರತಿಜ್ಞಾ(!isnan(x));
            ಪ್ರತಿಜ್ಞಾ(!isinf(x));
        }
#ಆದರೆ_ಅಂತ್ಯ  // NDEBUG
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_swiglu_clamp_f16(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0   = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1   = dst->src[1];
    ಅಕ್ಷರ *              src0_d = (ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ;
    ಅಕ್ಷರ *              src1_d = (ಅಕ್ಷರ *) (src1 ? src1->ದತ್ತಾಂಶ : src0->ದತ್ತಾಂಶ);
    ಸ್ಥಿರಾಂಕ size_t        src0_o = src0->nb[1];
    ಸ್ಥಿರಾಂಕ size_t        src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ಆದರೆ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ಸ್ಥಿರಾಂಕ int32_t swapped = ggml_get_op_params_i32(dst, 1);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   ಎಲ್ಲೆ   = ggml_get_op_params_f32(dst, 3);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr  = (nr + nth - 1) / nth;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr * ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1 * src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1 * src1_o);
        ggml_fp16_t * dst_p  = (ggml_fp16_t *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i1 * (dst->nb[1]));

        ಆದರೆ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < nc; k++) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ gate = std::min(GGML_FP16_ಗಮ್ಯ_FP32(src0_p[k]), ಎಲ್ಲೆ);
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಮೇಲೆ   = std::clamp(GGML_FP16_ಗಮ್ಯ_FP32(src1_p[k]), -ಎಲ್ಲೆ, ಎಲ್ಲೆ);
            dst_p[k]         = GGML_FP32_ಗಮ್ಯ_FP16(gate / (1.f + expf(-gate)) * ಮೇಲೆ);
        }

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NDEBUG
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < nc; k++) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x = GGML_FP16_ಗಮ್ಯ_FP32(dst_p[k]);
            GGML_ಬಳಸದ(x);
            ಪ್ರತಿಜ್ಞಾ(!isnan(x));
            ಪ್ರತಿಜ್ಞಾ(!isinf(x));
        }
#ಆದರೆ_ಅಂತ್ಯ  // NDEBUG
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_swiglu_clamp(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ggml_tensor * dst) {
    ಆಯ್ಕೆ (dst->src[0]->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            ggml_compute_forward_swiglu_clamp_f32(params, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            ggml_compute_forward_swiglu_clamp_f16(params, dst);
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            GGML_ABORT("fatal error");
    }
}

// ggml_compute_forward_geglu_erf

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_geglu_erf_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];
    ಅಕ್ಷರ * src0_d = (ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ;
    ಅಕ್ಷರ * src1_d = (ಅಕ್ಷರ *) (src1 ? src1->ದತ್ತಾಂಶ : src0->ದತ್ತಾಂಶ);
    ಸ್ಥಿರಾಂಕ size_t src0_o = src0->nb[1];
    ಸ್ಥಿರಾಂಕ size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ಆದರೆ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ಸ್ಥಿರಾಂಕ int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i1 = ir0; i1 < ir1; i1++) {
        ದಶಮಾಂಶ * src0_p = (ದಶಮಾಂಶ *) (src0_d + i1*src0_o);
        ದಶಮಾಂಶ * src1_p = (ದಶಮಾಂಶ *) (src1_d + i1*src1_o);

        ಆದರೆ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_erf_f32(nc, (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i1*(dst->nb[1])), src0_p, src1_p);

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NDEBUG
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < nc; k++) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x = ((ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i1*( dst->nb[1])))[k];
            GGML_ಬಳಸದ(x);
            ಪ್ರತಿಜ್ಞಾ(!isnan(x));
            ಪ್ರತಿಜ್ಞಾ(!isinf(x));
        }
#ಆದರೆ_ಅಂತ್ಯ // NDEBUG
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_geglu_erf_f16(
    ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
    ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];
    ಅಕ್ಷರ * src0_d = (ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ;
    ಅಕ್ಷರ * src1_d = (ಅಕ್ಷರ *) (src1 ? src1->ದತ್ತಾಂಶ : src0->ದತ್ತಾಂಶ);
    ಸ್ಥಿರಾಂಕ size_t src0_o = src0->nb[1];
    ಸ್ಥಿರಾಂಕ size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ಆದರೆ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ಸ್ಥಿರಾಂಕ int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        ಆದರೆ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_erf_f16(nc, (ggml_fp16_t *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i1*(dst->nb[1])), src0_p, src1_p);

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NDEBUG
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < nc; k++) {
            ಸ್ಥಿರಾಂಕ ggml_fp16_t x = ((ggml_fp16_t *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i1*( dst->nb[1])))[k];
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ v = GGML_FP16_ಗಮ್ಯ_FP32(x);
            GGML_ಬಳಸದ(v);
            ಪ್ರತಿಜ್ಞಾ(!isnan(v));
            ಪ್ರತಿಜ್ಞಾ(!isinf(v));
        }
#ಆದರೆ_ಅಂತ್ಯ // NDEBUG
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_geglu_erf_bf16(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_ಅಡ್ಡಸಾಲುಗಳು<ggml_vec_geglu_erf_bf16>(params, dst);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_geglu_erf(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_geglu_erf_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            {
                ggml_compute_forward_geglu_erf_f16(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
            {
                ggml_compute_forward_geglu_erf_bf16(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_geglu_quick

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_geglu_ಶೀಘ್ರ_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];
    ಅಕ್ಷರ * src0_d = (ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ;
    ಅಕ್ಷರ * src1_d = (ಅಕ್ಷರ *) (src1 ? src1->ದತ್ತಾಂಶ : src0->ದತ್ತಾಂಶ);
    ಸ್ಥಿರಾಂಕ size_t src0_o = src0->nb[1];
    ಸ್ಥಿರಾಂಕ size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ಆದರೆ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ಸ್ಥಿರಾಂಕ int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i1 = ir0; i1 < ir1; i1++) {
        ದಶಮಾಂಶ * src0_p = (ದಶಮಾಂಶ *) (src0_d + i1*src0_o);
        ದಶಮಾಂಶ * src1_p = (ದಶಮಾಂಶ *) (src1_d + i1*src1_o);

        ಆದರೆ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_ಶೀಘ್ರ_f32(nc, (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i1*(dst->nb[1])), src0_p, src1_p);

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NDEBUG
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < nc; k++) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x = ((ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i1*( dst->nb[1])))[k];
            GGML_ಬಳಸದ(x);
            ಪ್ರತಿಜ್ಞಾ(!isnan(x));
            ಪ್ರತಿಜ್ಞಾ(!isinf(x));
        }
#ಆದರೆ_ಅಂತ್ಯ // NDEBUG
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_geglu_ಶೀಘ್ರ_f16(
    ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
    ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];
    ಅಕ್ಷರ * src0_d = (ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ;
    ಅಕ್ಷರ * src1_d = (ಅಕ್ಷರ *) (src1 ? src1->ದತ್ತಾಂಶ : src0->ದತ್ತಾಂಶ);
    ಸ್ಥಿರಾಂಕ size_t src0_o = src0->nb[1];
    ಸ್ಥಿರಾಂಕ size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ಆದರೆ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ಸ್ಥಿರಾಂಕ int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        ಆದರೆ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_ಶೀಘ್ರ_f16(nc, (ggml_fp16_t *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i1*(dst->nb[1])), src0_p, src1_p);

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NDEBUG
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < nc; k++) {
            ಸ್ಥಿರಾಂಕ ggml_fp16_t x = ((ggml_fp16_t *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i1*( dst->nb[1])))[k];
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ v = GGML_FP16_ಗಮ್ಯ_FP32(x);
            GGML_ಬಳಸದ(v);
            ಪ್ರತಿಜ್ಞಾ(!isnan(v));
            ಪ್ರತಿಜ್ಞಾ(!isinf(v));
        }
#ಆದರೆ_ಅಂತ್ಯ // NDEBUG
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_geglu_ಶೀಘ್ರ_bf16(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_ಅಡ್ಡಸಾಲುಗಳು<ggml_vec_geglu_ಶೀಘ್ರ_bf16>(params, dst);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_geglu_ಶೀಘ್ರ(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_geglu_ಶೀಘ್ರ_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            {
                ggml_compute_forward_geglu_ಶೀಘ್ರ_f16(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
            {
                ggml_compute_forward_geglu_ಶೀಘ್ರ_bf16(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_norm

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_norm_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    ದಶಮಾಂಶ eps;
    ಸ್ಮರಣೆ_ನಕಲು(&eps, dst->op_params, ಗಾತ್ರ(ದಶಮಾಂಶ));

    GGML_ASSERT(eps >= 0.0f);

    ಗಾಗಿ (int64_t i03 = 0; i03 < ne03; i03++) {
        ಗಾಗಿ (int64_t i02 = 0; i02 < ne02; i02++) {
            ಗಾಗಿ (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * x = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i01*nb01 + i02*nb02 + i03*nb03;
                ಅಕ್ಷರ * y = (ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i01*nb1 + i02*nb2 + i03*nb3;

                ಆದರೆ (nb00 == ಗಾತ್ರ(ದಶಮಾಂಶ) && nb0 == ಗಾತ್ರ(ದಶಮಾಂಶ)) {
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * xf = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) x;

                    ದಶಮಾಂಶ ಮೊತ್ತ = 0.0;
                    ggml_vec_ಮೊತ್ತ_f32(ne00, &ಮೊತ್ತ, xf);
                    ದಶಮಾಂಶ ಸರಾಸರಿ_2 = ಮೊತ್ತ/ne00;

                    ದಶಮಾಂಶ * yf = (ದಶಮಾಂಶ *) y;
                    ದಶಮಾಂಶ variance = 0;

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_ACCELERATE
                    ಸರಾಸರಿ_2 = -ಸರಾಸರಿ_2;
                    vDSP_vsadd(xf, 1, &ಸರಾಸರಿ_2, yf, 1, ne00);
                    vDSP_measqv(yf, 1, &variance, ne00);
#ಇಲ್ಲದಿದ್ದರೆ
                    variance = ggml_vec_cvar_f32(ne00, yf, xf, ಸರಾಸರಿ_2);
#ಆದರೆ_ಅಂತ್ಯ //GGML_USE_ACCELERATE

                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale = 1.0f/sqrtf(variance + eps);
                    ggml_vec_scale_f32(ne00, yf, scale);
                } ಇಲ್ಲದಿದ್ದರೆ {
                    ದಶಮಾಂಶ ಮೊತ್ತ = 0.0;
                    ಗಾಗಿ (int64_t i00 = 0; i00 < ne00; i00++) {
                        ಮೊತ್ತ += *(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) (x + i00*nb00);
                    }
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಸರಾಸರಿ_2 = ಮೊತ್ತ/ne00;

                    ದಶಮಾಂಶ variance = 0.0f;
                    ಗಾಗಿ (int64_t i00 = 0; i00 < ne00; i00++) {
                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ v = *(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) (x + i00*nb00) - ಸರಾಸರಿ_2;
                        *(ದಶಮಾಂಶ *) (y + i00*nb0) = v;
                        variance += v * v;
                    }
                    variance /= ne00;

                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale = 1.0f/sqrtf(variance + eps);
                    ಗಾಗಿ (int64_t i00 = 0; i00 < ne00; i00++) {
                        *(ದಶಮಾಂಶ *) (y + i00*nb0) *= scale;
                    }
                }
            }
        }
    }
}

ಶೂನ್ಯ ggml_compute_forward_norm(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_norm_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_group_rms_norm

// fusion kinds that can be combined with the rms_norm computation in a single pass.
// extend this enum when adding new fused variants (e.g. FUSE_ADD, FUSE_MUL_ADD, ...).
ಎಣಿಕೆ ggml_rms_norm_fuse_op {
    GGML_RMS_NORM_FUSE_OP_NONE,
    GGML_RMS_NORM_FUSE_OP_MUL,
};

ಮಾದರಿ <ggml_rms_norm_fuse_op FUSE_OP>
ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_rms_norm_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst_rms_norm,
        ggml_tensor * dst_fused = ಶೂನ್ಯಸೂಚಕ) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst_rms_norm->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = ಶೂನ್ಯಸೂಚಕ;
    ggml_tensor       * dst  = dst_rms_norm;

    ಆದರೆ constexpr (FUSE_OP == GGML_RMS_NORM_FUSE_OP_MUL) {
        src1 = (dst_fused->src[0] == dst_rms_norm) ? dst_fused->src[1] : dst_fused->src[0];
        dst  = dst_fused;
    }

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    GGML_ASSERT(src0->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

    ದಶಮಾಂಶ eps;
    ಸ್ಮರಣೆ_ನಕಲು(&eps, dst_rms_norm->op_params, ಗಾತ್ರ(ದಶಮಾಂಶ));
    GGML_ASSERT(eps >= 0.0f);

    // TODO: optimize
    ಗಾಗಿ (int64_t i03 = 0; i03 < ne03; i03++) {
        ಗಾಗಿ (int64_t i02 = 0; i02 < ne02; i02++) {
            ಗಾಗಿ (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i01*nb01 + i02*nb02 + i03*nb03);

                ggml_float ಮೊತ್ತ = 0.0;
                // worth switching to explicit SIMD?
                ಗಾಗಿ (int64_t i00 = 0; i00 < ne00; i00++) {
                    ಮೊತ್ತ += (ggml_float)(x[i00] * x[i00]);
                }

                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಸರಾಸರಿ_2  = ಮೊತ್ತ/ne00;
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale = 1.0f/sqrtf(ಸರಾಸರಿ_2 + eps);

                // if you hit this, likely you got an inf somewhere earlier
                ಪ್ರತಿಜ್ಞಾ(scale > 0.0f);

                ದಶಮಾಂಶ * y = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i01*nb1 + i02*nb2 + i03*nb3);

                ಆದರೆ constexpr (FUSE_OP == GGML_RMS_NORM_FUSE_OP_MUL) {
                    ಸ್ಥಿರಾಂಕ int64_t i11 = i01 % ne11;
                    ಸ್ಥಿರಾಂಕ int64_t i12 = i02 % ne12;
                    ಸ್ಥಿರಾಂಕ int64_t i13 = i03 % ne13;
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * w = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ + i11*nb11 + i12*nb12 + i13*nb13);

                    ಗಾಗಿ (int64_t i00 = 0; i00 < ne00; i00++) {
                        y[i00] = x[i00] * scale * w[i00];
                    }
                } ಇಲ್ಲದಿದ್ದರೆ {
                    ಸ್ಮರಣೆ_ನಕಲು(y, x, ne00 * ಗಾತ್ರ(ದಶಮಾಂಶ));
                    ggml_vec_scale_f32(ne00, y, scale);
                }
            }
        }
    }
}

ಶೂನ್ಯ ggml_compute_forward_rms_norm(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_rms_norm_f32<GGML_RMS_NORM_FUSE_OP_NONE>(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// Fused RMS_NORM + MUL: computes dst = rms_norm(src0) * src1 in a single pass.
// This avoids materializing the intermediate rms_norm result in memory.
ಶೂನ್ಯ ggml_compute_forward_rms_norm_mul_fused(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst_rms_norm,
        ggml_tensor * dst_mul) {

    GGML_ASSERT(dst_mul != ಶೂನ್ಯಸೂಚಕ);
    GGML_ASSERT(dst_mul->src[0] == dst_rms_norm || dst_mul->src[1] == dst_rms_norm);

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst_rms_norm->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_rms_norm_f32<GGML_RMS_NORM_FUSE_OP_MUL>(params, dst_rms_norm, dst_mul);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_rms_norm_ಹಿಂಭಾಗ_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0]; // gradients from forward pass output
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1]; // src1 from forward pass

    GGML_ASSERT(ggml_are_same_shape(src0, dst) && ggml_are_same_shape(src0, src1));

    GGML_ASSERT(src0->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ));
    GGML_ASSERT(src1->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

    ದಶಮಾಂಶ eps;
    ಸ್ಮರಣೆ_ನಕಲು(&eps, dst->op_params, ಗಾತ್ರ(ದಶಮಾಂಶ));

    // TODO: optimize
    ಗಾಗಿ (int64_t i03 = 0; i03 < ne03; i03++) {
        ಗಾಗಿ (int64_t i02 = 0; i02 < ne02; i02++) {
            ಗಾಗಿ (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                // src1 is same shape as src0 => same indices
                ಸ್ಥಿರಾಂಕ int64_t i11 = i01;
                ಸ್ಥಿರಾಂಕ int64_t i12 = i02;
                ಸ್ಥಿರಾಂಕ int64_t i13 = i03;

                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * dz = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i01*nb01 + i02*nb02 + i03*nb03);
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x  = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ + i11*nb11 + i12*nb12 + i13*nb13);

                ggml_float ಮೊತ್ತ_xx  = 0.0;
                ggml_float ಮೊತ್ತ_xdz = 0.0;

                ಗಾಗಿ (int64_t i00 = 0; i00 < ne00; i00++) {
                    ಮೊತ್ತ_xx  += (ggml_float)(x[i00] * x[i00]);
                    ಮೊತ್ತ_xdz += (ggml_float)(x[i00] * dz[i00]);
                }

                //const float mean     = (float)(sum_xx)/ne00;
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಸರಾಸರಿ_eps = (ದಶಮಾಂಶ)(ಮೊತ್ತ_xx)/ne00 + eps;
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಮೊತ್ತ_eps  = (ದಶಮಾಂಶ)(ಮೊತ್ತ_xx) + eps*ne00;
                //const float mean_xdz = (float)(sum_xdz)/ne00;
                // we could cache rms from forward pass to improve performance.
                // to do this implement ggml_rms and compose ggml_rms_norm using ggml_rms.
                //const float rms      = sqrtf(mean_eps);
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ rrms     = 1.0f / sqrtf(ಸರಾಸರಿ_eps);
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
                ದಶಮಾಂಶ * dx = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i01*nb1 + i02*nb2 + i03*nb3);

                // dx[i00] = (dz + x*(-sum_xdz/sum_eps)) * rrms
                // note: https://github.com/ggml-org/ggml/issues/1491
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale_x = (ದಶಮಾಂಶ) (-ಮೊತ್ತ_xdz) / ಮೊತ್ತ_eps;
                ಗಾಗಿ (int64_t i00 = 0; i00 < ne00; i00++) {
                    dx[i00] = (dz[i00] + x[i00] * scale_x) * rrms;
                }
            }
        }
    }
}

ಶೂನ್ಯ ggml_compute_forward_rms_norm_ಹಿಂಭಾಗ(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_rms_norm_ಹಿಂಭಾಗ_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_group_norm

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_ಗುಂಪು_norm_f32(
    ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
    ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    GGML_ASSERT(src0->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    // TODO: optimize

    ದಶಮಾಂಶ eps;
    ಸ್ಮರಣೆ_ನಕಲು(&eps, dst->op_params + 1, ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಪೂರ್ಣಾಂಕ n_channels = src0->ne[2];
    ಪೂರ್ಣಾಂಕ n_groups = dst->op_params[0];
    ಪೂರ್ಣಾಂಕ n_channels_ಪ್ರತಿ_ಗುಂಪು = (n_channels + n_groups - 1) / n_groups;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = ith; i < n_groups; i += nth) {
        ಪೂರ್ಣಾಂಕ start = i * n_channels_ಪ್ರತಿ_ಗುಂಪು;
        ಪೂರ್ಣಾಂಕ end = start + n_channels_ಪ್ರತಿ_ಗುಂಪು;
        ಆದರೆ (end > n_channels) {
            end = n_channels;
        }
        ಪೂರ್ಣಾಂಕ ಹೆಜ್ಜೆ = end - start;

        ಗಾಗಿ (int64_t i03 = 0; i03 < ne03; i03++) {
            ggml_float ಮೊತ್ತ = 0.0;
            ಗಾಗಿ (int64_t i02 = start; i02 < end; i02++) {
                ಗಾಗಿ (int64_t i01 = 0; i01 < ne01; i01++) {
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x = (ದಶಮಾಂಶ *)((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i01 * nb01 + i02 * nb02 + i03 * nb03);

                    ggml_float sumr = 0.0;
                    ಗಾಗಿ (int64_t i00 = 0; i00 < ne00; i00++) {
                        sumr += (ggml_float)x[i00];
                    }
                    ಮೊತ್ತ += sumr;
                }
            }
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಸರಾಸರಿ_2 = ಮೊತ್ತ / (ne00 * ne01 * ಹೆಜ್ಜೆ);

            ggml_float sum2 = 0.0;
            ಗಾಗಿ (int64_t i02 = start; i02 < end; i02++) {
                ಗಾಗಿ (int64_t i01 = 0; i01 < ne01; i01++) {
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x = (ದಶಮಾಂಶ *)((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i01 * nb01 + i02 * nb02 + i03 * nb03);

                    ದಶಮಾಂಶ * y = (ದಶಮಾಂಶ *)((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i01 * nb1 + i02 * nb2 + i03 * nb3);

                    ggml_float sumr = 0.0;
                    ಗಾಗಿ (int64_t i00 = 0; i00 < ne00; i00++) {
                        ದಶಮಾಂಶ v = x[i00] - ಸರಾಸರಿ_2;
                        y[i00] = v;
                        sumr += (ggml_float)(v * v);
                    }
                    sum2 += sumr;
                }
            }
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ variance = sum2 / (ne00 * ne01 * ಹೆಜ್ಜೆ);
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale = 1.0f / sqrtf(variance + eps);

            ಗಾಗಿ (int64_t i02 = start; i02 < end; i02++) {
                ಗಾಗಿ (int64_t i01 = 0; i01 < ne01; i01++) {
                    ದಶಮಾಂಶ * y = (ದಶಮಾಂಶ *)((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i01 * nb1 + i02 * nb2 + i03 * nb3);
                    ggml_vec_scale_f32(ne00, y, scale);
                }
            }
        }
    }
}

ಶೂನ್ಯ ggml_compute_forward_ಗುಂಪು_norm(
    ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
    ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_ಗುಂಪು_norm_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_l2_norm

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_l2_norm_f32(
    ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
    ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    ದಶಮಾಂಶ eps;
    ಸ್ಮರಣೆ_ನಕಲು(&eps, dst->op_params, ಗಾತ್ರ(ದಶಮಾಂಶ));

    GGML_ASSERT(eps >= 0.0f);

    // TODO: optimize
    ಗಾಗಿ (int64_t i03 = 0; i03 < ne03; i03++) {
        ಗಾಗಿ (int64_t i02 = 0; i02 < ne02; i02++) {
            ಗಾಗಿ (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * x = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i01*nb01 + i02*nb02 + i03*nb03;

                ggml_float ಮೊತ್ತ = 0.0;
                ಗಾಗಿ (int64_t i00 = 0; i00 < ne00; i00++) {
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ xi = *(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) (x + i00*nb00);
                    ಮೊತ್ತ += (ggml_float)(xi * xi);
                }

                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale = 1.0f/fmaxf(sqrtf(ಮೊತ್ತ), eps);

                ಅಕ್ಷರ * y = (ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i01*nb1 + i02*nb2 + i03*nb3;

                ಆದರೆ (nb00 == ಗಾತ್ರ(ದಶಮಾಂಶ) && nb0 == ಗಾತ್ರ(ದಶಮಾಂಶ)) {
                    ಸ್ಮರಣೆ_ನಕಲು(y, x, ne00 * ಗಾತ್ರ(ದಶಮಾಂಶ));
                    ggml_vec_scale_f32(ne00, (ದಶಮಾಂಶ *) y, scale);
                } ಇಲ್ಲದಿದ್ದರೆ {
                    ಗಾಗಿ (int64_t i00 = 0; i00 < ne00; i00++) {
                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ xi = *(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) (x + i00*nb00);
                        *(ದಶಮಾಂಶ *) (y + i00*nb0) = xi * scale;
                    }
                }
            }
        }
    }
}

ಶೂನ್ಯ ggml_compute_forward_l2_norm(
    ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
    ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_l2_norm_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_out_prod

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_ಹೊರಹೋಗು_prod_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
              ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    GGML_ASSERT(ne2 % ne02 == 0);
    GGML_ASSERT(ne3 % ne03 == 0);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ಗಾತ್ರ(ದಶಮಾಂಶ));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == ಗಾತ್ರ(ದಶಮಾಂಶ));
    // GGML_ASSERT(nb0 <= nb1);
    // GGML_ASSERT(nb1 <= nb2);
    // GGML_ASSERT(nb2 <= nb3);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    ಆದರೆ (ith == 0) {
        ggml_vec_ಗಣ_f32(ne0*ne1*ne2*ne3, (ದಶಮಾಂಶ *)dst->ದತ್ತಾಂಶ, 0);
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
    ಸ್ಥಿರಾಂಕ int64_t nr = ne1*ne2*ne3;

    // rows per thread
    ಸ್ಥಿರಾಂಕ int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ int64_t ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ int64_t ir1 = MIN(ir0 + dr, nr);

    // block-tiling attempt
    ಸ್ಥಿರಾಂಕ int64_t blck_0 = MAX(GGML_VEC_MAD_UNROLL, 32);
    ಸ್ಥಿರಾಂಕ int64_t blck_1 = 16;

    // dps == dst per src0, used for group query attention
    ಸ್ಥಿರಾಂಕ int64_t dps2 = ne2 / ne02;
    ಸ್ಥಿರಾಂಕ int64_t dps3 = ne3 / ne03;

    ಗಾಗಿ (int64_t bir = ir0; bir < ir1; bir += blck_1) {
        ಸ್ಥಿರಾಂಕ int64_t bir1 = MIN(bir + blck_1, ir1);
        ಗಾಗಿ (int64_t bi01 = 0; bi01 < ne01; bi01 += blck_0) {
            ಸ್ಥಿರಾಂಕ int64_t bne01 = MIN(bi01 + blck_0, ne01);
            ಗಾಗಿ (int64_t ir = bir; ir < bir1; ++ir) {
                // dst indices
                ಸ್ಥಿರಾಂಕ int64_t i3 = ir/(ne2*ne1);
                ಸ್ಥಿರಾಂಕ int64_t i2 = (ir - i3*ne2*ne1)/ne1;
                ಸ್ಥಿರಾಂಕ int64_t i1 = (ir - i3*ne2*ne1 - i2*ne1);

                ಸ್ಥಿರಾಂಕ int64_t i02 = i2 / dps2;
                ಸ್ಥಿರಾಂಕ int64_t i03 = i3 / dps3;

                //const int64_t i10 = i1;
                ಸ್ಥಿರಾಂಕ int64_t i12 = i2;
                ಸ್ಥಿರಾಂಕ int64_t i13 = i3;

#ಆದರೆ GGML_VEC_MAD_UNROLL > 2
                ಸ್ಥಿರಾಂಕ int64_t bne01_unroll = bne01 - (bne01 % GGML_VEC_MAD_UNROLL);
                ಗಾಗಿ (int64_t i01 = bi01; i01 < bne01_unroll; i01 += GGML_VEC_MAD_UNROLL) {
                    ಸ್ಥಿರಾಂಕ int64_t i11 = i01;

                    ದಶಮಾಂಶ * s0 = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + (          i01*nb01 + i02*nb02 + i03*nb03));
                    ದಶಮಾಂಶ * s1 = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
                    ದಶಮಾಂಶ * d  = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *)  dst->ದತ್ತಾಂಶ + (          i1*nb1   + i2*nb2   + i3*nb3));

                    ggml_vec_mad_f32_unroll(ne0, nb01, nb11, d, s0, s1);
                }
                ಗಾಗಿ (int64_t i01 = bne01_unroll; i01 < bne01; ++i01) {
                    ಸ್ಥಿರಾಂಕ int64_t i11 = i01;

                    ದಶಮಾಂಶ * s0 = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + (          i01*nb01 + i02*nb02 + i03*nb03));
                    ದಶಮಾಂಶ * s1 = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
                    ದಶಮಾಂಶ * d  = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *)  dst->ದತ್ತಾಂಶ + (          i1*nb1   + i2*nb2   + i3*nb3));

                    ggml_vec_mad_f32(ne0, d, s0, *s1);
                }
#ಇಲ್ಲದಿದ್ದರೆ
                ಗಾಗಿ (int64_t i01 = bi01; i01 < bne01; ++i01) {
                    ಸ್ಥಿರಾಂಕ int64_t i11 = i01;

                    ದಶಮಾಂಶ * s0 = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + (          i01*nb01 + i02*nb02 + i03*nb03));
                    ದಶಮಾಂಶ * s1 = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
                    ದಶಮಾಂಶ * d  = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *)  dst->ದತ್ತಾಂಶ + (          i1*nb1 + i2*nb2 + i3*nb3));

                    ggml_vec_mad_f32(ne0, d, s0, *s1);
                }
#ಆದರೆ_ಅಂತ್ಯ
            }
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_ಹೊರಹೋಗು_prod_q_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
              ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ggml_type type = src0->type;
    ggml_ಗಮ್ಯ_float_t ಸ್ಥಿರಾಂಕ dequantize_ಅಡ್ಡಸಾಲು_q = ggml_get_type_traits(type)->ಗಮ್ಯ_float;

    GGML_ASSERT(ne02 == ne12);
    GGML_ASSERT(ne03 == ne13);
    GGML_ASSERT(ne2  == ne12);
    GGML_ASSERT(ne3  == ne13);

    // we don't support permuted src0 dim0
    GGML_ASSERT(nb00 == ggml_type_ಗಾತ್ರ(type));

    // dst dim0 cannot be transposed or permuted
    GGML_ASSERT(nb0 == ಗಾತ್ರ(ದಶಮಾಂಶ));
    // GGML_ASSERT(nb0 <= nb1);
    // GGML_ASSERT(nb1 <= nb2);
    // GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    ಆದರೆ (ith == 0) {
        ggml_vec_ಗಣ_f32(ne0*ne1*ne2*ne3, (ದಶಮಾಂಶ *)dst->ದತ್ತಾಂಶ, 0);
    }
    ggml_barrier(params->threadpool);

    // parallelize by last three dimensions

    // total rows in dst
    ಸ್ಥಿರಾಂಕ int64_t nr = ne1*ne2*ne3;

    // rows per thread
    ಸ್ಥಿರಾಂಕ int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ int64_t ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ int64_t ir1 = MIN(ir0 + dr, nr);

    // dst[:,:,:,:] = 0
    // for i2,i3:
    //   for i1:
    //     for i01:
    //       for i0:
    //         dst[i0,i1,i2,i3] += src0[i0,i01,i2,i3] * src1[i1,i01,i2,i3]

    ದಶಮಾಂಶ * wdata = (ದಶಮಾಂಶ *) params->wdata + (ne0 + CACHE_ಸಾಲು_ಗಾತ್ರ_F32) * ith;

    ಗಾಗಿ (int64_t ir = ir0; ir < ir1; ++ir) {
        // dst indices
        ಸ್ಥಿರಾಂಕ int64_t i3 = ir/(ne2*ne1);
        ಸ್ಥಿರಾಂಕ int64_t i2 = (ir - i3*ne2*ne1)/ne1;
        ಸ್ಥಿರಾಂಕ int64_t i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ಸ್ಥಿರಾಂಕ int64_t i02 = i2;
        ಸ್ಥಿರಾಂಕ int64_t i03 = i3;

        //const int64_t i10 = i1;
        ಸ್ಥಿರಾಂಕ int64_t i12 = i2;
        ಸ್ಥಿರಾಂಕ int64_t i13 = i3;

        ಗಾಗಿ (int64_t i01 = 0; i01 < ne01; ++i01) {
            ಸ್ಥಿರಾಂಕ int64_t i11 = i01;

            ದಶಮಾಂಶ * s0 = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + (          i01*nb01 + i02*nb02 + i03*nb03));
            ದಶಮಾಂಶ * s1 = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
            ದಶಮಾಂಶ * d  = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *)  dst->ದತ್ತಾಂಶ + (          i1*nb1 + i2*nb2 + i3*nb3));

            dequantize_ಅಡ್ಡಸಾಲು_q(s0, wdata, ne0);
            ggml_vec_mad_f32(ne0, d, wdata, *s1);
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_ಹೊರಹೋಗು_prod_f16_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
              ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ne02 == ne12);
    GGML_ASSERT(ne03 == ne13);
    GGML_ASSERT(ne2  == ne12);
    GGML_ASSERT(ne3  == ne13);

    GGML_ASSERT(nb00 == ಗಾತ್ರ(ggml_fp16_t));
    GGML_ASSERT(nb0  == ಗಾತ್ರ(ದಶಮಾಂಶ));

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    ಆದರೆ (ith == 0) {
        ggml_vec_ಗಣ_f32(ne0*ne1*ne2*ne3, (ದಶಮಾಂಶ *)dst->ದತ್ತಾಂಶ, 0);
    }
    ggml_barrier(params->threadpool);

    ಸ್ಥಿರಾಂಕ int64_t nr = ne1*ne2*ne3;
    ಸ್ಥಿರಾಂಕ int64_t dr = (nr + nth - 1)/nth;
    ಸ್ಥಿರಾಂಕ int64_t ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ int64_t ir1 = MIN(ir0 + dr, nr);

    ದಶಮಾಂಶ * wdata = (ದಶಮಾಂಶ *) params->wdata + (ne0 + CACHE_ಸಾಲು_ಗಾತ್ರ_F32) * ith;

    ಗಾಗಿ (int64_t ir = ir0; ir < ir1; ++ir) {
        ಸ್ಥಿರಾಂಕ int64_t i3 = ir/(ne2*ne1);
        ಸ್ಥಿರಾಂಕ int64_t i2 = (ir - i3*ne2*ne1)/ne1;
        ಸ್ಥಿರಾಂಕ int64_t i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ಸ್ಥಿರಾಂಕ int64_t i02 = i2;
        ಸ್ಥಿರಾಂಕ int64_t i03 = i3;

        ಸ್ಥಿರಾಂಕ int64_t i12 = i2;
        ಸ್ಥಿರಾಂಕ int64_t i13 = i3;

        ದಶಮಾಂಶ * d = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + (i1*nb1 + i2*nb2 + i3*nb3));

        ಗಾಗಿ (int64_t i01 = 0; i01 < ne01; ++i01) {
            ಸ್ಥಿರಾಂಕ int64_t i11 = i01;
            ggml_fp16_t * s0 = (ggml_fp16_t *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + (i01*nb01 + i02*nb02 + i03*nb03));
            ದಶಮಾಂಶ * s1 = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
            ggml_fp16_ಗಮ್ಯ_fp32_ಅಡ್ಡಸಾಲು(s0, wdata, ne0);
            ggml_vec_mad_f32(ne0, d, wdata, *s1);
        }
    }
}

ಶೂನ್ಯ ggml_compute_forward_ಹೊರಹೋಗು_prod(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_Q1_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q2_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_1:
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
        ಸ್ಥಿತಿ GGML_TYPE_MXFP4:
        ಸ್ಥಿತಿ GGML_TYPE_NVFP4:
        ಸ್ಥಿತಿ GGML_TYPE_Q2_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q3_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
        ಸ್ಥಿತಿ GGML_TYPE_TQ1_0:
        ಸ್ಥಿತಿ GGML_TYPE_TQ2_0:
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XXS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ3_XXS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ1_S:
        ಸ್ಥಿತಿ GGML_TYPE_IQ1_M:
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_NL:
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_XS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ3_S:
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_ಹೊರಹೋಗು_prod_q_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            {
                ggml_compute_forward_ಹೊರಹೋಗು_prod_f16_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_ಹೊರಹೋಗು_prod_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_scale

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_scale_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    ದಶಮಾಂಶ s; // scale factor
    ದಶಮಾಂಶ b; // bias

    ಸ್ಮರಣೆ_ನಕಲು(&s, (ದಶಮಾಂಶ *) dst->op_params + 0, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&b, (ದಶಮಾಂಶ *) dst->op_params + 1, ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nc = src0->ne[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr = ggml_nrows(src0);

    // rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ಸ್ಥಿರಾಂಕ size_t nb01 = src0->nb[1];

    ಸ್ಥಿರಾಂಕ size_t nb1 = dst->nb[1];

    ಆದರೆ (b == 0.0f) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i1 = ir0; i1 < ir1; i1++) {
            ಆದರೆ (dst->ದತ್ತಾಂಶ != src0->ದತ್ತಾಂಶ) {
                // src0 is same shape as dst => same indices
                // TODO: add x parameter to ggml_vec_scale_f32 and remove this memcpy
                ಸ್ಮರಣೆ_ನಕಲು((ಅಕ್ಷರ *)dst->ದತ್ತಾಂಶ + i1*nb1, (ಅಕ್ಷರ *)src0->ದತ್ತಾಂಶ + i1*nb01, nc * ಗಾತ್ರ(ದಶಮಾಂಶ));
            }
            ggml_vec_scale_f32(nc, (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i1*nb1), s);
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i1 = ir0; i1 < ir1; i1++) {
            ggml_vec_mad1_f32(nc,
                (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ  + i1*nb1),
                (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i1*nb1),
                s, b);
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_scale_bf16(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    ದಶಮಾಂಶ s; // scale factor
    ದಶಮಾಂಶ b; // bias

    ಸ್ಮರಣೆ_ನಕಲು(&s, (ದಶಮಾಂಶ *) dst->op_params + 0, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&b, (ದಶಮಾಂಶ *) dst->op_params + 1, ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nc = src0->ne[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr = ggml_nrows(src0);

    // rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i1 = ir0; i1 < ir1; i1++) {
        ಸ್ಥಿರಾಂಕ ggml_bf16_t * x = (ಸ್ಥಿರಾಂಕ ggml_bf16_t *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i1*src0->nb[1]);
        ggml_bf16_t       * y = (ggml_bf16_t *)       ((ಅಕ್ಷರ *)       dst->ದತ್ತಾಂಶ  + i1*dst->nb[1]);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < nc; k++) {
            y[k] = GGML_FP32_ಗಮ್ಯ_BF16(s*GGML_BF16_ಗಮ್ಯ_FP32(x[k]) + b);
        }
    }
}

ಶೂನ್ಯ ggml_compute_forward_scale(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_scale_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
            {
                ggml_compute_forward_scale_bf16(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_set

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_ಗಣ_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));

    // view src0 and dst with these strides and data offset inbytes during set
    // nb0 is implicitly element_size because src0 and dst are contiguous
    size_t nb1     = ((int32_t *) dst->op_params)[0];
    size_t nb2     = ((int32_t *) dst->op_params)[1];
    size_t nb3     = ((int32_t *) dst->op_params)[2];
    size_t ಸ್ಥಾನಾಂತರ  = ((int32_t *) dst->op_params)[3];
    ತಾರ್ಕಿಕ   inplace = (ತಾರ್ಕಿಕ) ((int32_t *) dst->op_params)[4];

    ಆದರೆ (!inplace) {
        ಆದರೆ (params->ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            ಸ್ಮರಣೆ_ನಕಲು(
                ((ಅಕ್ಷರ *)  dst->ದತ್ತಾಂಶ),
                ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr = ggml_nrows(src1);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nc = src1->ne[0];

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne)
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb)

    // src0 and dst as viewed during set
    ಸ್ಥಿರಾಂಕ size_t nb0 = ggml_element_ಗಾತ್ರ(src0);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ im0 = (ne10 == 0 ? 0 : ne10-1);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ im1 = (ne11 == 0 ? 0 : ne11-1);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ im2 = (ne12 == 0 ? 0 : ne12-1);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ im3 = (ne13 == 0 ? 0 : ne13-1);

    GGML_ASSERT(ಸ್ಥಾನಾಂತರ + im0*nb0  + im1*nb1  + im2*nb2  + im3*nb3  <= ggml_nbytes(dst));

    GGML_ASSERT(nb10 == ಗಾತ್ರ(ದಶಮಾಂಶ));

    // rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are viewed with shape of src1 and offset
        // => same indices
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i3 = ir/(ne12*ne11);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i2 = (ir - i3*ne12*ne11)/ne11;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i1 = (ir - i3*ne12*ne11 - i2*ne11);

        ggml_vec_cpy_f32(nc,
                (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *)  dst->ದತ್ತಾಂಶ + i3*nb3  + i2*nb2  + i1*nb1  + ಸ್ಥಾನಾಂತರ),
                (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ + i3*nb13 + i2*nb12 + i1*nb11));
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_ಗಣ_i32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));

    // view src0 and dst with these strides and data offset inbytes during set
    // nb0 is implicitly element_size because src0 and dst are contiguous
    size_t nb1     = ((int32_t *) dst->op_params)[0];
    size_t nb2     = ((int32_t *) dst->op_params)[1];
    size_t nb3     = ((int32_t *) dst->op_params)[2];
    size_t ಸ್ಥಾನಾಂತರ  = ((int32_t *) dst->op_params)[3];
    ತಾರ್ಕಿಕ   inplace = (ತಾರ್ಕಿಕ) ((int32_t *) dst->op_params)[4];

    ಆದರೆ (!inplace) {
        ಆದರೆ (params->ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            ಸ್ಮರಣೆ_ನಕಲು(
                ((ಅಕ್ಷರ *)  dst->ದತ್ತಾಂಶ),
                ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr = ggml_nrows(src1);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nc = src1->ne[0];

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne)
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb)

    // src0 and dst as viewed during set
    ಸ್ಥಿರಾಂಕ size_t nb0 = ggml_element_ಗಾತ್ರ(src0);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ im0 = (ne10 == 0 ? 0 : ne10-1);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ im1 = (ne11 == 0 ? 0 : ne11-1);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ im2 = (ne12 == 0 ? 0 : ne12-1);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ im3 = (ne13 == 0 ? 0 : ne13-1);

    GGML_ASSERT(ಸ್ಥಾನಾಂತರ + im0*nb0  + im1*nb1  + im2*nb2  + im3*nb3  <= ggml_nbytes(dst));

    GGML_ASSERT(nb10 == ಗಾತ್ರ(int32_t));

    // rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are viewed with shape of src1 and offset
        // => same indices
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i3 = ir/(ne12*ne11);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i2 = (ir - i3*ne12*ne11)/ne11;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i1 = (ir - i3*ne12*ne11 - i2*ne11);

        ggml_vec_cpy_i32(nc,
                (int32_t *) ((ಅಕ್ಷರ *)  dst->ದತ್ತಾಂಶ + i3*nb3  + i2*nb2  + i1*nb1  + ಸ್ಥಾನಾಂತರ),
                (int32_t *) ((ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ + i3*nb13 + i2*nb12 + i1*nb11));
    }
}

ಶೂನ್ಯ ggml_compute_forward_ಗಣ(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_ಗಣ_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_I32:
            {
                ggml_compute_forward_ಗಣ_i32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F16:
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
        ಸ್ಥಿತಿ GGML_TYPE_Q1_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q2_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_1:
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q8_1:
        ಸ್ಥಿತಿ GGML_TYPE_MXFP4:
        ಸ್ಥಿತಿ GGML_TYPE_NVFP4:
        ಸ್ಥಿತಿ GGML_TYPE_Q2_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q3_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
        ಸ್ಥಿತಿ GGML_TYPE_TQ1_0:
        ಸ್ಥಿತಿ GGML_TYPE_TQ2_0:
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XXS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ3_XXS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ1_S:
        ಸ್ಥಿತಿ GGML_TYPE_IQ1_M:
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_NL:
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_XS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ3_S:
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_S:
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_cpy

ಶೂನ್ಯ ggml_compute_forward_cpy(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_dup(params, dst);
}

// ggml_compute_forward_cont

ಶೂನ್ಯ ggml_compute_forward_cont(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_dup(params, dst);
}

// ggml_compute_forward_get_rows

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_get_ಅಡ್ಡಸಾಲುಗಳು_q(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
              ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

    ಸ್ಥಿರಾಂಕ int64_t nc = ne00;
    ಸ್ಥಿರಾಂಕ int64_t nr = ggml_nelements(src1);

    ಸ್ಥಿರಾಂಕ ggml_type type = src0->type;
    ggml_ಗಮ್ಯ_float_t ಸ್ಥಿರಾಂಕ dequantize_ಅಡ್ಡಸಾಲು_q = ggml_get_type_traits(type)->ಗಮ್ಯ_float;

    ಪ್ರತಿಜ್ಞಾ(ne0  == nc);
    ಪ್ರತಿಜ್ಞಾ(ne02 == ne11);
    ಪ್ರತಿಜ್ಞಾ(nb00 == ggml_type_ಗಾತ್ರ(type));
    ಪ್ರತಿಜ್ಞಾ(ggml_nrows(dst) == nr);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    // rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ಗಾಗಿ (int64_t i = ir0; i < ir1; ++i) {
        ಸ್ಥಿರಾಂಕ int64_t i12 = i/(ne11*ne10);
        ಸ್ಥಿರಾಂಕ int64_t i11 = (i - i12*ne11*ne10)/ne10;
        ಸ್ಥಿರಾಂಕ int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        ಸ್ಥಿರಾಂಕ int64_t i01 = *(int32_t *) ((ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        dequantize_ಅಡ್ಡಸಾಲು_q(
                (ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i01*nb01 + i11*nb02 + i12*nb03),
                     (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *)  dst->ದತ್ತಾಂಶ + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_get_ಅಡ್ಡಸಾಲುಗಳು_f16(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
              ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

    ಸ್ಥಿರಾಂಕ int64_t nc = ne00;
    ಸ್ಥಿರಾಂಕ int64_t nr = ggml_nelements(src1);

    ಪ್ರತಿಜ್ಞಾ(ne0  == nc);
    ಪ್ರತಿಜ್ಞಾ(ne02 == ne11);
    ಪ್ರತಿಜ್ಞಾ(nb00 == ಗಾತ್ರ(ggml_fp16_t));
    ಪ್ರತಿಜ್ಞಾ(ggml_nrows(dst) == nr);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    // rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ಗಾಗಿ (int64_t i = ir0; i < ir1; ++i) {
        ಸ್ಥಿರಾಂಕ int64_t i12 = i/(ne11*ne10);
        ಸ್ಥಿರಾಂಕ int64_t i11 = (i - i12*ne11*ne10)/ne10;
        ಸ್ಥಿರಾಂಕ int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        ಸ್ಥಿರಾಂಕ int64_t i01 = *(int32_t *) ((ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        ggml_cpu_fp16_ಗಮ್ಯ_fp32(
            (ಸ್ಥಿರಾಂಕ ggml_fp16_t*) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i01*nb01 + i11*nb02 + i12*nb03),
                       (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *)  dst->ದತ್ತಾಂಶ + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_get_ಅಡ್ಡಸಾಲುಗಳು_bf16(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
              ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

    ಸ್ಥಿರಾಂಕ int64_t nc = ne00;
    ಸ್ಥಿರಾಂಕ int64_t nr = ggml_nelements(src1);

    ಪ್ರತಿಜ್ಞಾ(ne0  == nc);
    ಪ್ರತಿಜ್ಞಾ(ne02 == ne11);
    ಪ್ರತಿಜ್ಞಾ(nb00 == ಗಾತ್ರ(ggml_bf16_t));
    ಪ್ರತಿಜ್ಞಾ(ggml_nrows(dst) == nr);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    // rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ಗಾಗಿ (int64_t i = ir0; i < ir1; ++i) {
        ಸ್ಥಿರಾಂಕ int64_t i12 = i/(ne11*ne10);
        ಸ್ಥಿರಾಂಕ int64_t i11 = (i - i12*ne11*ne10)/ne10;
        ಸ್ಥಿರಾಂಕ int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        ಸ್ಥಿರಾಂಕ int64_t i01 = *(int32_t *) ((ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        ggml_cpu_bf16_ಗಮ್ಯ_fp32(
            (ಸ್ಥಿರಾಂಕ ggml_bf16_t *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i01*nb01 + i11*nb02 + i12*nb03),
                        (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *)  dst->ದತ್ತಾಂಶ + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_get_ಅಡ್ಡಸಾಲುಗಳು_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
              ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

    ಸ್ಥಿರಾಂಕ int64_t nc = ne00;
    ಸ್ಥಿರಾಂಕ int64_t nr = ggml_nelements(src1);

    ಪ್ರತಿಜ್ಞಾ(ne0  == nc);
    ಪ್ರತಿಜ್ಞಾ(ne02 == ne11);
    ಪ್ರತಿಜ್ಞಾ(nb00 == ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಪ್ರತಿಜ್ಞಾ(ggml_nrows(dst) == nr);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    // rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ಗಾಗಿ (int64_t i = ir0; i < ir1; ++i) {
        ಸ್ಥಿರಾಂಕ int64_t i12 = i/(ne11*ne10);
        ಸ್ಥಿರಾಂಕ int64_t i11 = (i - i12*ne11*ne10)/ne10;
        ಸ್ಥಿರಾಂಕ int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        ಸ್ಥಿರಾಂಕ int64_t i01 = *(int32_t *) ((ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        ggml_vec_cpy_f32(nc,
                (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *)  dst->ದತ್ತಾಂಶ + i10*nb1  + i11*nb2  + i12*nb3),
                (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i01*nb01 + i11*nb02 + i12*nb03));
    }
}

ಶೂನ್ಯ ggml_compute_forward_get_ಅಡ್ಡಸಾಲುಗಳು(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_Q1_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q2_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_1:
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q8_1:
        ಸ್ಥಿತಿ GGML_TYPE_MXFP4:
        ಸ್ಥಿತಿ GGML_TYPE_NVFP4:
        ಸ್ಥಿತಿ GGML_TYPE_Q2_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q3_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
        ಸ್ಥಿತಿ GGML_TYPE_TQ1_0:
        ಸ್ಥಿತಿ GGML_TYPE_TQ2_0:
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XXS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ3_XXS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ1_S:
        ಸ್ಥಿತಿ GGML_TYPE_IQ1_M:
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_NL:
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_XS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ3_S:
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_get_ಅಡ್ಡಸಾಲುಗಳು_q(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            {
                ggml_compute_forward_get_ಅಡ್ಡಸಾಲುಗಳು_f16(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
            {
                ggml_compute_forward_get_ಅಡ್ಡಸಾಲುಗಳು_bf16(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F32:
        ಸ್ಥಿತಿ GGML_TYPE_I32:
            {
                ggml_compute_forward_get_ಅಡ್ಡಸಾಲುಗಳು_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
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

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು src_t, ಬಗೆ_ಹೆಸರು idx_t>
ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_impl(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
              ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

    ಸ್ಥಿರಾಂಕ int64_t nc = ne00;
    ಸ್ಥಿರಾಂಕ int64_t nr = ne01;

    ಪ್ರತಿಜ್ಞಾ(ne0  == nc);
    ಪ್ರತಿಜ್ಞಾ(ne2  == ne02);
    ಪ್ರತಿಜ್ಞಾ(ne3  == ne03);
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    ಪ್ರತಿಜ್ಞಾ(ne02 % ne11 == 0);
    ಪ್ರತಿಜ್ಞಾ(ne03 % ne12 == 0);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    // rows per thread
    ಸ್ಥಿರಾಂಕ int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ int64_t ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ int64_t ir1 = std::min(ir0 + dr, nr);

    ಸ್ಥಿರಾಂಕ size_t rs = ggml_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(src0->type, nc);

    ggml_from_float_t ಸ್ಥಿರಾಂಕ from_float = ggml_get_type_traits_cpu(dst->type)->from_float;

    ಗಾಗಿ (int64_t i03 = 0; i03 < ne03; ++i03) {
        ಗಾಗಿ (int64_t i02 = 0; i02 < ne02; ++i02) {
            ಗಾಗಿ (int64_t i = ir0; i < ir1; ++i) {
                ಸ್ಥಿರಾಂಕ int64_t i12 = i03%ne12;
                ಸ್ಥಿರಾಂಕ int64_t i11 = i02%ne11;
                ಸ್ಥಿರಾಂಕ int64_t i10 = i;

                ಸ್ಥಿರಾಂಕ int64_t i1 = *(idx_t *) ((ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ + i10*nb10 + i11*nb11 + i12*nb12);

                GGML_ASSERT(i1 >= 0 && i1 < ne1);

                ಆದರೆ constexpr (std::is_same_v<src_t, ದಶಮಾಂಶ>) {
                    from_float(
                            (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ +  i*nb01 + i02*nb02 + i03*nb03),
                                            ((ಅಕ್ಷರ *)  dst->ದತ್ತಾಂಶ + i1*nb1  + i02*nb2  + i03*nb3), nc);
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<src_t, ggml_fp16_t>) {
                    ಆದರೆ (dst->type == GGML_TYPE_F16) {
                        ಸ್ಮರಣೆ_ನಕಲು(
                                            ((ಅಕ್ಷರ *)  dst->ದತ್ತಾಂಶ + i1*nb1  + i02*nb2  + i03*nb3),
                                            ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ +  i*nb01 + i02*nb02 + i03*nb03),
                                            rs);
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        ದಶಮಾಂಶ * wdata = (ದಶಮಾಂಶ *) params->wdata + (nc + CACHE_ಸಾಲು_ಗಾತ್ರ_F32) * ith;
                        ggml_fp16_ಗಮ್ಯ_fp32_ಅಡ್ಡಸಾಲು(
                                (ಸ್ಥಿರಾಂಕ ggml_fp16_t *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i*nb01 + i02*nb02 + i03*nb03),
                                wdata, nc);
                        from_float(wdata,
                                ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i1*nb1 + i02*nb2 + i03*nb3), nc);
                    }
                } ಇಲ್ಲದಿದ್ದರೆ {
                    GGML_ABORT("src0->type = %d (%s) not supported", src0->type, ggml_type_name(src0->type));
                }
            }
        }
    }
}

ಶೂನ್ಯ ggml_compute_forward_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ಆದರೆ (src1->type == GGML_TYPE_I64) {
                    ggml_compute_forward_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_impl<ದಶಮಾಂಶ, int64_t>(params, dst);
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src1->type == GGML_TYPE_I32) {
                    ggml_compute_forward_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_impl<ದಶಮಾಂಶ, int32_t>(params, dst);
                } ಇಲ್ಲದಿದ್ದರೆ {
                    GGML_ABORT("src1->type = %d (%s) not supported", src1->type, ggml_type_name(src1->type));
                }
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            {
                ಆದರೆ (src1->type == GGML_TYPE_I64) {
                    ggml_compute_forward_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_impl<ggml_fp16_t, int64_t>(params, dst);
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src1->type == GGML_TYPE_I32) {
                    ggml_compute_forward_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_impl<ggml_fp16_t, int32_t>(params, dst);
                } ಇಲ್ಲದಿದ್ದರೆ {
                    GGML_ABORT("src1->type = %d (%s) not supported", src1->type, ggml_type_name(src1->type));
                }
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("src0->type = %d (%s) not supported", src0->type, ggml_type_name(src0->type));
            }
    }
}

// ggml_compute_forward_get_rows_back

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_get_ಅಡ್ಡಸಾಲುಗಳು_ಹಿಂಭಾಗ_f32_f16(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
              ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    ಆದರೆ (params->ith != 0) {
        ಹಿಂತಿರುಗಿ;
    }

    GGML_ASSERT(ggml_is_contiguous(dst));

    // ggml_compute_forward_dup_same_cont(params, opt0, dst);

    memset(dst->ದತ್ತಾಂಶ, 0, ggml_nbytes(dst));

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nc = src0->ne[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr = ggml_nelements(src1);

    GGML_ASSERT( dst->ne[0] == nc);
    GGML_ASSERT(src0->nb[0] == ಗಾತ್ರ(ggml_fp16_t));

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nr; ++i) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ r = ((int32_t *) src1->ದತ್ತಾಂಶ)[i];

        GGML_ASSERT(r >= 0 && r < dst->ne[1]);

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < nc; ++j) {
            ggml_fp16_t v = ((ggml_fp16_t *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i*src0->nb[1]))[j];
            ((ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + r*dst->nb[1]))[j] += GGML_CPU_FP16_ಗಮ್ಯ_FP32(v);
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_get_ಅಡ್ಡಸಾಲುಗಳು_ಹಿಂಭಾಗ_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
              ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    ಆದರೆ (params->ith != 0) {
        ಹಿಂತಿರುಗಿ;
    }

    GGML_ASSERT(ggml_is_contiguous(dst));

    // ggml_compute_forward_dup_same_cont(params, opt0, dst);

    memset(dst->ದತ್ತಾಂಶ, 0, ggml_nbytes(dst));

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nc = src0->ne[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr = ggml_nelements(src1);

    GGML_ASSERT( dst->ne[0] == nc);
    GGML_ASSERT(src0->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nr; ++i) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ r = ((int32_t *) src1->ದತ್ತಾಂಶ)[i];

        GGML_ASSERT(r >= 0 && r < dst->ne[1]);

        ggml_vec_ಕೂಡಿಸು_f32(nc,
                (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *)  dst->ದತ್ತಾಂಶ + r*dst->nb[1]),
                (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *)  dst->ದತ್ತಾಂಶ + r*dst->nb[1]),
                (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i*src0->nb[1]));
    }
}

ಶೂನ್ಯ ggml_compute_forward_get_ಅಡ್ಡಸಾಲುಗಳು_ಹಿಂಭಾಗ(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            {
                ggml_compute_forward_get_ಅಡ್ಡಸಾಲುಗಳು_ಹಿಂಭಾಗ_f32_f16(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_get_ಅಡ್ಡಸಾಲುಗಳು_ಹಿಂಭಾಗ_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
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

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_diag_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆದರೆ (params->ith != 0) {
        ಹಿಂತಿರುಗಿ;
    }

    // TODO: handle transposed/permuted matrices

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(ne00 == ne0);
    GGML_ASSERT(ne00 == ne1);
    GGML_ASSERT(ne01 == 1);
    GGML_ASSERT(ne02 == ne2);
    GGML_ASSERT(ne03 == ne3);

    GGML_ASSERT(nb00 == ಗಾತ್ರ(ದಶಮಾಂಶ));
    GGML_ASSERT(nb0  == ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i3 = 0; i3 < ne3; i3++) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i2 = 0; i2 < ne2; i2++) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i1 = 0; i1 < ne1; i1++) {
                ದಶಮಾಂಶ * d = (ದಶಮಾಂಶ *)((ಅಕ್ಷರ *)  dst->ದತ್ತಾಂಶ + i3*nb3  + i2*nb2 + i1*nb1);
                ದಶಮಾಂಶ * s = (ದಶಮಾಂಶ *)((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i3*nb03 + i2*nb02);
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < i1; i0++) {
                    d[i0] = 0;
                }
                d[i1] = s[i1];
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = i1+1; i0 < ne0; i0++) {
                    d[i0] = 0;
                }
            }
        }
    }
}

ಶೂನ್ಯ ggml_compute_forward_diag(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_diag_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_diag_mask_inf

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_diag_ಮುಖವಾಡ_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst,
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ value) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ  n_past  = ((int32_t *) dst->op_params)[0];
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ inplace = src0->ದತ್ತಾಂಶ == dst->ದತ್ತಾಂಶ;

    GGML_ASSERT(n_past >= 0);

    ಆದರೆ (!inplace) {
        ಆದರೆ (ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
            GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));
            ಸ್ಮರಣೆ_ನಕಲು(
                ((ಅಕ್ಷರ *)  dst->ದತ್ತಾಂಶ),
                ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    // TODO: handle transposed/permuted matrices

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n  = ggml_nrows(src0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nc = src0->ne[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr = src0->ne[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nz = n/nr;

    GGML_ASSERT( dst->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ));
    GGML_ASSERT(src0->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < nz; k++) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = ith; j < nr; j += nth) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = n_past; i < nc; i++) {
                ಆದರೆ (i > n_past + j) {
                    *(ದಶಮಾಂಶ *)((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + k*dst->nb[2] + j*dst->nb[1] + i*dst->nb[0]) = value;
                }
            }
        }
    }
}

ಶೂನ್ಯ ggml_compute_forward_diag_ಮುಖವಾಡ_inf(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_diag_ಮುಖವಾಡ_f32(params, dst, -INFINITY);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ಶೂನ್ಯ ggml_compute_forward_diag_ಮುಖವಾಡ_zero(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_diag_ಮುಖವಾಡ_f32(params, dst, 0);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_soft_max

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_soft_max_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
              ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src2 = dst->src[2];

    ಪ್ರತಿಜ್ಞಾ(ggml_is_contiguous(dst));
    ಪ್ರತಿಜ್ಞಾ(ggml_are_same_shape(src0, dst));

    ದಶಮಾಂಶ scale    = 1.0f;
    ದಶಮಾಂಶ max_bias = 0.0f;

    ಸ್ಮರಣೆ_ನಕಲು(&scale,    (ದಶಮಾಂಶ *) dst->op_params + 0, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&max_bias, (ದಶಮಾಂಶ *) dst->op_params + 1, ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    ಸ್ಥಿರಾಂಕ int64_t nb11 = src1 ? src1->nb[1] : 1;
    ಸ್ಥಿರಾಂಕ int64_t nb12 = src1 ? src1->nb[2] : 1;
    ಸ್ಥಿರಾಂಕ int64_t nb13 = src1 ? src1->nb[3] : 1;

    ಸ್ಥಿರಾಂಕ int64_t ne12 = src1 ? src1->ne[2] : 1;
    ಸ್ಥಿರಾಂಕ int64_t ne13 = src1 ? src1->ne[3] : 1;

    // TODO: is this supposed to be ceil instead of floor?
    //       https://huggingface.co/mosaicml/mpt-7b/blob/main/attention.py#L370
    ಸ್ಥಿರಾಂಕ uint32_t n_ತಲೆ      = ne02;
    ಸ್ಥಿರಾಂಕ uint32_t n_ತಲೆ_log2 = 1u << (uint32_t) floor(log2(n_ತಲೆ));

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ m0 = powf(2.0f, -(max_bias       ) / n_ತಲೆ_log2);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ m1 = powf(2.0f, -(max_bias / 2.0f) / n_ತಲೆ_log2);

    ದಶಮಾಂಶ * wp = (ದಶಮಾಂಶ *) params->wdata + (ne00 + CACHE_ಸಾಲು_ಗಾತ್ರ_F32) * ith;

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ use_f16 = (src1 && src1->type == GGML_TYPE_F16);

    // sinks
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * sk = src2 ? (ದಶಮಾಂಶ *)((ಅಕ್ಷರ *) src2->ದತ್ತಾಂಶ) : ಶೂನ್ಯಸೂಚಕ;

    ಗಾಗಿ (int64_t i03 = 0; i03 < ne03; i03++) {
        ಗಾಗಿ (int64_t i02 = 0; i02 < ne02; i02++) {
            ಗಾಗಿ (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                ಸ್ಥಿರಾಂಕ int64_t i11 = i01;
                ಸ್ಥಿರಾಂಕ int64_t i12 = i02%ne12;
                ಸ್ಥಿರಾಂಕ int64_t i13 = i03%ne13;

                // ALiBi
                ಸ್ಥಿರಾಂಕ uint32_t h = i02; //ತಲೆ
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಇಳಿಜಾರು = (max_bias > 0.0f) ? h < n_ತಲೆ_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_ತಲೆ_log2) + 1) : 1.0f;

                ದಶಮಾಂಶ * sp = (ದಶಮಾಂಶ *)((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i01*nb01 + i02*nb02 + i03*nb03);
                ದಶಮಾಂಶ * dp = (ದಶಮಾಂಶ *)((ಅಕ್ಷರ *)  dst->ದತ್ತಾಂಶ + i01*nb1  + i02*nb2  + i03*nb3);

                // broadcast the mask across rows
                ggml_fp16_t * mp_f16 = src1 ? (ggml_fp16_t *)((ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ + i11*nb11 + i12*nb12 + i13*nb13) : ಖಾಲಿ;
                ದಶಮಾಂಶ       * mp_f32 = src1 ? (ದಶಮಾಂಶ       *)((ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ + i11*nb11 + i12*nb12 + i13*nb13) : ಖಾಲಿ;

                ggml_vec_cpy_f32  (ne00, wp, sp);
                ggml_vec_scale_f32(ne00, wp, scale);
                ಆದರೆ (mp_f32) {
                    ಆದರೆ (use_f16) {
                        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ne00; ++i) {
                            wp[i] += ಇಳಿಜಾರು*GGML_CPU_FP16_ಗಮ್ಯ_FP32(mp_f16[i]);
                        }
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ne00; ++i) {
                            wp[i] += ಇಳಿಜಾರು*mp_f32[i];
                        }
                    }
                }

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NDEBUG
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ne00; ++i) {
                    //printf("p[%d] = %f\n", i, p[i]);
                    ಪ್ರತಿಜ್ಞಾ(!isnan(wp[i]));
                }
#ಆದರೆ_ಅಂತ್ಯ // NDEBUG

                ದಶಮಾಂಶ max = -INFINITY;
                ggml_vec_max_f32(ne00, &max, wp);

                // if we have sinks, make a correction as if they were included in the softmax
                ಆದರೆ (sk) {
                    max = MAX(max, sk[i02]);
                }

                ggml_float ಮೊತ್ತ = ggml_vec_soft_max_f32(ne00, dp, wp, max);
                ಪ್ರತಿಜ್ಞಾ(ಮೊತ್ತ > 0.0);

                ಆದರೆ (sk) {
                    ಮೊತ್ತ += (ggml_float) expf(sk[i02] - max);
                }

                ಮೊತ್ತ = 1.0/ಮೊತ್ತ;
                ggml_vec_scale_f32(ne00, dp, ಮೊತ್ತ);

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NDEBUG
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ne00; ++i) {
                    ಪ್ರತಿಜ್ಞಾ(!isnan(dp[i]));
                    ಪ್ರತಿಜ್ಞಾ(!isinf(dp[i]));
                }
#ಆದರೆ_ಅಂತ್ಯ // NDEBUG
            }
        }
    }
}

ಶೂನ್ಯ ggml_compute_forward_soft_max(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
              ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_soft_max_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}


// ggml_compute_forward_soft_max_ext_back

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_soft_max_ext_ಹಿಂಭಾಗ_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_are_same_shape(src1, dst));

    ದಶಮಾಂಶ scale    = 1.0f;
    ದಶಮಾಂಶ max_bias = 0.0f;

    ಸ್ಮರಣೆ_ನಕಲು(&scale,    (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) dst->op_params + 0, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&max_bias, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) dst->op_params + 1, ಗಾತ್ರ(ದಶಮಾಂಶ));

    GGML_ASSERT(max_bias == 0.0f);

    // TODO: handle transposed/permuted matrices

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nc = src0->ne[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr = ggml_nrows(src0);

    // rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i1 = ir0; i1 < ir1; i1++) {
        ದಶಮಾಂಶ *dy = (ದಶಮಾಂಶ *)((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i1*src0->nb[1]);
        ದಶಮಾಂಶ *y  = (ದಶಮಾಂಶ *)((ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ + i1*src1->nb[1]);
        ದಶಮಾಂಶ *dx = (ದಶಮಾಂಶ *)((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ  + i1*dst->nb[1]);

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NDEBUG
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            ಪ್ರತಿಜ್ಞಾ(!isnan(dy[i]));
            ಪ್ರತಿಜ್ಞಾ(!isnan(y[i]));
        }
#ಆದರೆ_ಅಂತ್ಯ // NDEBUG
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
        ದಶಮಾಂಶ dot_y_dy = 0;
        ggml_vec_dot_f32(nc, &dot_y_dy, 0, y, 0, dy, 0, 1);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nc; i++) {
            dx[i] = scale * (dy[i] - dot_y_dy) * y[i];
        }

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NDEBUG
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nc; ++i) {
            ಪ್ರತಿಜ್ಞಾ(!isnan(dx[i]));
            ಪ್ರತಿಜ್ಞಾ(!isinf(dx[i]));
        }
#ಆದರೆ_ಅಂತ್ಯ // NDEBUG
    }
}

ಶೂನ್ಯ ggml_compute_forward_soft_max_ext_ಹಿಂಭಾಗ(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_soft_max_ext_ಹಿಂಭಾಗ_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_clamp

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_clamp_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ದಶಮಾಂಶ min;
    ದಶಮಾಂಶ max;
    ಸ್ಮರಣೆ_ನಕಲು(&min, (ದಶಮಾಂಶ *) dst->op_params + 0, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&max, (ದಶಮಾಂಶ *) dst->op_params + 1, ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n  = ggml_nrows(src0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nc = src0->ne[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == ಗಾತ್ರ(ದಶಮಾಂಶ));
    GGML_ASSERT(nb00 == ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = ith; j < n; j += nth) {
        ಸ್ಥಿರಾಂಕ int64_t i1 = j % ne01;
        ಸ್ಥಿರಾಂಕ int64_t i2 = (j / ne01) % ne02;
        ಸ್ಥಿರಾಂಕ int64_t i3 = j / (ne01*ne02);

        ದಶಮಾಂಶ * dst_ptr  = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *)  dst->ದತ್ತಾಂಶ + i1*nb1  + i2*nb2  + i3*nb3);
        ದಶಮಾಂಶ * src0_ptr = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i1*nb01 + i2*nb02 + i3*nb03);

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nc; i++) {
            dst_ptr[i] = MAX(MIN(src0_ptr[i], max), min);
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_clamp_f16(
    ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
    ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ದಶಮಾಂಶ min;
    ದಶಮಾಂಶ max;
    ಸ್ಮರಣೆ_ನಕಲು(&min, (ದಶಮಾಂಶ *) dst->op_params + 0, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&max, (ದಶಮಾಂಶ *) dst->op_params + 1, ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n  = ggml_nrows(src0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nc = src0->ne[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == ಗಾತ್ರ(ggml_fp16_t));
    GGML_ASSERT(nb00 == ಗಾತ್ರ(ggml_fp16_t));

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = ith; j < n; j += nth) {
        ಸ್ಥಿರಾಂಕ int64_t i1 = j % ne01;
        ಸ್ಥಿರಾಂಕ int64_t i2 = (j / ne01) % ne02;
        ಸ್ಥಿರಾಂಕ int64_t i3 = j / (ne01*ne02);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((ಅಕ್ಷರ *)  dst->ದತ್ತಾಂಶ + i1*nb1  + i2*nb2  + i3*nb3);
        ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i1*nb01 + i2*nb02 + i3*nb03);

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nc; i++) {
            ದಶಮಾಂಶ v = GGML_CPU_FP16_ಗಮ್ಯ_FP32(src0_ptr[i]);
            dst_ptr[i] = GGML_CPU_FP32_ಗಮ್ಯ_FP16(MAX(MIN(v, max), min));
        }
    }
}

ಶೂನ್ಯ ggml_compute_forward_clamp(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_clamp_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            {
                ggml_compute_forward_clamp_f16(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
        ಸ್ಥಿತಿ GGML_TYPE_Q1_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q2_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_1:
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q8_1:
        ಸ್ಥಿತಿ GGML_TYPE_MXFP4:
        ಸ್ಥಿತಿ GGML_TYPE_NVFP4:
        ಸ್ಥಿತಿ GGML_TYPE_Q2_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q3_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
        ಸ್ಥಿತಿ GGML_TYPE_TQ1_0:
        ಸ್ಥಿತಿ GGML_TYPE_TQ2_0:
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XXS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ3_XXS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ1_S:
        ಸ್ಥಿತಿ GGML_TYPE_IQ1_M:
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_NL:
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_XS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ3_S:
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_S:
        ಸ್ಥಿತಿ GGML_TYPE_Q8_K:
        ಸ್ಥಿತಿ GGML_TYPE_I8:
        ಸ್ಥಿತಿ GGML_TYPE_I16:
        ಸ್ಥಿತಿ GGML_TYPE_I32:
        ಸ್ಥಿತಿ GGML_TYPE_I64:
        ಸ್ಥಿತಿ GGML_TYPE_F64:
        ಸ್ಥಿತಿ GGML_TYPE_ಎಣಿಕೆ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_rope

ಸ್ಥಿರ ದಶಮಾಂಶ rope_yarn_ramp(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಕೆಳಗಿನ, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಮೇಲಿನ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i0) {
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ y = (i0 / 2 - ಕೆಳಗಿನ) / MAX(0.001f, ಮೇಲಿನ - ಕೆಳಗಿನ);
    ಹಿಂತಿರುಗಿ 1 - MIN(1, MAX(0, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
ಸ್ಥಿರ ಶೂನ್ಯ rope_yarn(
    ದಶಮಾಂಶ theta_extrap, ದಶಮಾಂಶ freq_scale, ದಶಮಾಂಶ corr_dims[2], int64_t i0, ದಶಮಾಂಶ ext_ಗುಣಕ, ದಶಮಾಂಶ mscale,
    ದಶಮಾಂಶ * cos_theta, ದಶಮಾಂಶ * sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    ದಶಮಾಂಶ theta_interp = freq_scale * theta_extrap;
    ದಶಮಾಂಶ theta = theta_interp;
    ಆದರೆ (ext_ಗುಣಕ != 0.0f) {
        ದಶಮಾಂಶ ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_ಗುಣಕ;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_rope_cache_init(
     ದಶಮಾಂಶ theta_ತಳ, ದಶಮಾಂಶ freq_scale, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * freq_factors, ದಶಮಾಂಶ corr_dims[2], int64_t ne0, ದಶಮಾಂಶ ext_ಗುಣಕ, ದಶಮಾಂಶ mscale,
     ದಶಮಾಂಶ * cache, ದಶಮಾಂಶ sin_sign, ದಶಮಾಂಶ theta_scale) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    ದಶಮಾಂಶ theta = theta_ತಳ;
    ಗಾಗಿ (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ff = freq_factors ? freq_factors[i0/2] : 1.0f;
        rope_yarn(
            theta/ff, freq_scale, corr_dims, i0, ext_ಗುಣಕ, mscale, &cache[i0 + 0], &cache[i0 + 1]
        );
        cache[i0 + 1] *= sin_sign;

        theta *= theta_scale;
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_mrope_cache_init(
     ದಶಮಾಂಶ theta_ತಳ_t, ದಶಮಾಂಶ theta_ತಳ_h, ದಶಮಾಂಶ theta_ತಳ_w, ದಶಮಾಂಶ theta_ತಳ_e, ಪೂರ್ಣಾಂಕ sections[4], ತಾರ್ಕಿಕ is_imrope, ತಾರ್ಕಿಕ indep_sects,
     ದಶಮಾಂಶ freq_scale, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * freq_factors, ದಶಮಾಂಶ corr_dims[2], int64_t ne0, ದಶಮಾಂಶ ext_ಗುಣಕ, ದಶಮಾಂಶ mscale,
     ದಶಮಾಂಶ * cache, ದಶಮಾಂಶ sin_sign, ದಶಮಾಂಶ theta_scale) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    ದಶಮಾಂಶ theta_t = theta_ತಳ_t;
    ದಶಮಾಂಶ theta_h = theta_ತಳ_h;
    ದಶಮಾಂಶ theta_w = theta_ತಳ_w;
    ದಶಮಾಂಶ theta_e = theta_ತಳ_e;  // extra position id for vision encoder
    ಪೂರ್ಣಾಂಕ sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    ಪೂರ್ಣಾಂಕ sec_w = sections[1] + sections[0];
    ಪೂರ್ಣಾಂಕ sec_e = sections[2] + sec_w;
    GGML_ASSERT(sect_dims <= ne0);

    ಗಾಗಿ (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ff = freq_factors ? freq_factors[i0/2] : 1.0f;

        ಪೂರ್ಣಾಂಕ sector = (i0 / 2) % sect_dims;
        ಆದರೆ (indep_sects) {
            // compute theta independently for each dim sections
            // (i.e. reset corresponding theta when `i0` go from one section to another)
            ಆದರೆ (sector == 0) {
                theta_t = theta_ತಳ_t;
            }
            ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (sector == sections[0]) {
                theta_h = theta_ತಳ_h;;
            }
            ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (sector == sec_w) {
                theta_w = theta_ತಳ_w;
            }
            ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (sector == sec_e) {
                theta_e = theta_ತಳ_e;
            }
        }

        ದಶಮಾಂಶ theta = theta_t;
        ಆದರೆ (is_imrope) { // qwen3vl apply interleaved mrope
            ಆದರೆ (sector % 3 == 1 && sector < 3 * sections[1]) {
                theta = theta_h;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (sector % 3 == 2 && sector < 3 * sections[2]) {
                theta = theta_w;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (sector % 3 == 0 && sector < 3 * sections[0]) {
                theta = theta_t;
            } ಇಲ್ಲದಿದ್ದರೆ {
                theta = theta_e;
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಆದರೆ (sector >= sections[0] && sector < sec_w) {
                theta = theta_h;
            }
            ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (sector >= sec_w && sector < sec_w + sections[2]) {
                theta = theta_w;
            }
            ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (sector >= sec_w + sections[2]) {
                theta = theta_e;
            }
        }

        rope_yarn(
            theta/ff, freq_scale, corr_dims, i0, ext_ಗುಣಕ, mscale, &cache[i0 + 0], &cache[i0 + 1]
        );
        cache[i0 + 1] *= sin_sign;

        theta_t *= theta_scale;
        theta_w *= theta_scale;
        theta_h *= theta_scale;
        theta_e *= theta_scale;
    }
}


ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ ಶೂನ್ಯ rotate_ಜೋಡಿಗಳು(ಸ್ಥಿರಾಂಕ int64_t n, ಸ್ಥಿರಾಂಕ int64_t n_ಸ್ಥಾನಾಂತರ, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * cache, ಸ್ಥಿರಾಂಕ T * src_ದತ್ತಾಂಶ, T * dst_ದತ್ತಾಂಶ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ scale = 2) {
  ಗಾಗಿ (int64_t i0 = 0; i0 < n; i0 += 2) {
    ಸ್ಥಿರಾಂಕ int64_t ic = i0/scale; // hack for GGML_ROPE_TYPE_NORMAL, where we need ic = i0; for all other cases, ic = i0/2

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ cos_theta = cache[i0 + 0];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ sin_theta = cache[i0 + 1];

    ಸ್ಥಿರಾಂಕ T * ಸ್ಥಿರಾಂಕ src = src_ದತ್ತಾಂಶ + ic;
    T * dst             = dst_ದತ್ತಾಂಶ + ic;

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x0 = type_conversion_ಕೋಷ್ಟಕ<T>::ಗಮ್ಯ_f32(src[0]);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x1 = type_conversion_ಕೋಷ್ಟಕ<T>::ಗಮ್ಯ_f32(src[n_ಸ್ಥಾನಾಂತರ]);

    dst[0]        = type_conversion_ಕೋಷ್ಟಕ<T>::from_f32(x0*cos_theta - x1*sin_theta);
    dst[n_ಸ್ಥಾನಾಂತರ] = type_conversion_ಕೋಷ್ಟಕ<T>::from_f32(x0*sin_theta + x1*cos_theta);
  }
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T> //float or ggml_fp16_t
ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_rope_flt(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst,
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ forward) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src2 = dst->src[2];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);

    ದಶಮಾಂಶ freq_ತಳ, freq_scale, ext_ಗುಣಕ, attn_ಗುಣಕ, beta_fast, beta_slow;
    ಪೂರ್ಣಾಂಕ sections[4];

    //const int n_past     = ((int32_t *) dst->op_params)[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_dims     = ((int32_t *) dst->op_params)[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mode       = ((int32_t *) dst->op_params)[2];
    //const int n_ctx      = ((int32_t *) dst->op_params)[3];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_ctx_orig = ((int32_t *) dst->op_params)[4];

    ಸ್ಮರಣೆ_ನಕಲು(&freq_ತಳ,   (int32_t *) dst->op_params +  5, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&freq_scale,  (int32_t *) dst->op_params +  6, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&ext_ಗುಣಕ,  (int32_t *) dst->op_params +  7, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&attn_ಗುಣಕ, (int32_t *) dst->op_params +  8, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&beta_fast,   (int32_t *) dst->op_params +  9, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&beta_slow,   (int32_t *) dst->op_params + 10, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&sections,    (int32_t *) dst->op_params + 11, ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ)*4);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_offs = ((int32_t *) dst->op_params)[15];

    GGML_TENSOR_UNARY_OP_LOCALS

    //printf("ne0: %d, ne1: %d, ne2: %d, ne3: %d\n", ne0, ne1, ne2, ne3);
    //printf("n_past = %d, ne2 = %d\n", n_past, ne2);

    GGML_ASSERT(nb0 == nb00);
    GGML_ASSERT(nb0 == ಗಾತ್ರ(T));

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr = ggml_nrows(dst);

    GGML_ASSERT(n_dims <= ne0);
    GGML_ASSERT(n_dims % 2 == 0);

    GGML_ASSERT(n_offs >= 0);
    GGML_ASSERT(n_offs % 2 == 0);
    GGML_ASSERT(n_offs + n_dims <= ne0);

    // rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    // row index used to determine which thread to use
    ಪೂರ್ಣಾಂಕ ir = 0;

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ theta_scale = powf(freq_ತಳ, -2.0f/n_dims);

    ದಶಮಾಂಶ corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_ತಳ, beta_fast, beta_slow, corr_dims);

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_imrope = mode == GGML_ROPE_TYPE_IMROPE; // qwen3vl apply interleaved mrope
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ mrope_ಬಳಸಿದ = mode & GGML_ROPE_TYPE_MROPE;  // ggml_rope_multi, note: also true for vision (24 & 8 == true) and for imrope
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_vision = mode == GGML_ROPE_TYPE_VISION;

    ಆದರೆ (mrope_ಬಳಸಿದ) {
        GGML_ASSERT(sections[0] > 0 || sections[1] > 0 || sections[2] > 0);
    }

    ಆದರೆ (is_vision) {
        GGML_ASSERT(n_dims == ne0/2);
        GGML_ASSERT(n_offs == 0);
    }

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * freq_factors = ಖಾಲಿ;
    ಆದರೆ (src2 != ಖಾಲಿ) {
        GGML_ASSERT(src2->type == GGML_TYPE_F32);
        GGML_ASSERT(src2->ne[0] >= n_dims / 2);
        freq_factors = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src2->ದತ್ತಾಂಶ;
    }

    // backward process uses inverse rotation by cos and sin.
    // cos and sin build a rotation matrix, where the inverse is the transpose.
    // this essentially just switches the sign of sin.
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ sin_sign = forward ? 1.0f : -1.0f;

    ಸ್ಥಿರಾಂಕ int32_t * pos = (ಸ್ಥಿರಾಂಕ int32_t *) src1->ದತ್ತಾಂಶ;

    int64_t ಕೊನೆಯ_i2 = -1;

    ಗಾಗಿ (int64_t i3 = 0; i3 < ne3; i3++) { // batch
        ಗಾಗಿ (int64_t i2 = 0; i2 < ne2; i2++) { // seq-len
            ಗಾಗಿ (int64_t i1 = 0; i1 < ne1; i1++) { // attn-heads
                ಆದರೆ (ir++ < ir0) ಮುಂದುವರಿ; // skip rows mapped to other threads
                ಆದರೆ (ir   > ir1) ನಿಲ್ಲಿಸು;

                ದಶಮಾಂಶ * cache = (ದಶಮಾಂಶ *) params->wdata + (ne0 + CACHE_ಸಾಲು_ಗಾತ್ರ_F32)*ith;
                ಆದರೆ (ಕೊನೆಯ_i2 != i2) {
                    ಆದರೆ (!mrope_ಬಳಸಿದ) {
                        ಸ್ಥಿರಾಂಕ int64_t p = pos[i2];
                        ggml_rope_cache_init(p, freq_scale, freq_factors, corr_dims, ne0, ext_ಗುಣಕ, attn_ಗುಣಕ, cache, sin_sign, theta_scale);
                    }
                    ಇಲ್ಲದಿದ್ದರೆ {
                        ಸ್ಥಿರಾಂಕ int64_t p_t = pos[i2];
                        ಸ್ಥಿರಾಂಕ int64_t p_h = pos[i2 + ne2];
                        ಸ್ಥಿರಾಂಕ int64_t p_w = pos[i2 + ne2 * 2];
                        ಸ್ಥಿರಾಂಕ int64_t p_e = pos[i2 + ne2 * 3];
                        ggml_mrope_cache_init(
                            p_t, p_h, p_w, p_e, sections, is_imrope, is_vision,
                            freq_scale, freq_factors, corr_dims, ne0, ext_ಗುಣಕ, attn_ಗುಣಕ, cache, sin_sign, theta_scale);
                    }

                    ಕೊನೆಯ_i2 = i2;
                }

                T * src = (T *)((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i3*nb03 + i2*nb02 + i1*nb01);
                T * dst_ದತ್ತಾಂಶ  = (T *)((ಅಕ್ಷರ *)  dst->ದತ್ತಾಂಶ + i3*nb3  + i2*nb2  + i1*nb1);

                ಆಯ್ಕೆ (mode) {
                    ಸ್ಥಿತಿ GGML_ROPE_TYPE_ಸಾಮಾನ್ಯ:
                        rotate_ಜೋಡಿಗಳು<T>(n_dims, 1, cache, src + n_offs, dst_ದತ್ತಾಂಶ + n_offs, 1);
                        ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ GGML_ROPE_TYPE_NEOX:
                    ಸ್ಥಿತಿ GGML_ROPE_TYPE_MROPE:
                    ಸ್ಥಿತಿ GGML_ROPE_TYPE_IMROPE:
                        rotate_ಜೋಡಿಗಳು<T>(n_dims, n_dims/2, cache, src + n_offs, dst_ದತ್ತಾಂಶ + n_offs);
                        ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ GGML_ROPE_TYPE_VISION:
                        rotate_ಜೋಡಿಗಳು<T>(ne0, n_dims, cache, src, dst_ದತ್ತಾಂಶ);
                        ನಿಲ್ಲಿಸು;
                    ಪೂರ್ವನಿಯೋಜಿತ:
                        GGML_ABORT("rope type not supported");
                }

                ಆದರೆ (!is_vision) {
                    // fill the remain channels with data from src tensor
                    ಗಾಗಿ (int64_t i0 = 0; i0 < ne0; i0 += 2) {
                        ಆದರೆ (i0 == n_offs) {
                            i0 += n_dims - 2; // skip the rotated channels
                            ಮುಂದುವರಿ;
                        }
                        ಸ್ಥಿರಾಂಕ T * ಸ್ಥಿರಾಂಕ src = (T *)((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                        T * dst_ದತ್ತಾಂಶ  = (T *)((ಅಕ್ಷರ *)  dst->ದತ್ತಾಂಶ + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

                        dst_ದತ್ತಾಂಶ[0] = src[0];
                        dst_ದತ್ತಾಂಶ[1] = src[1];
                    }
                }
            } //attn-heads
        }
    }
}

ಶೂನ್ಯ ggml_compute_forward_rope(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            {
                ggml_compute_forward_rope_flt<ggml_fp16_t>(params, dst, ನಿಜ);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_rope_flt<ದಶಮಾಂಶ>(params, dst, ನಿಜ);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_rope_back

ಶೂನ್ಯ ggml_compute_forward_rope_ಹಿಂಭಾಗ(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            {
                ggml_compute_forward_rope_flt<ggml_fp16_t>(params, dst, ಸುಳ್ಳು);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_rope_flt<ದಶಮಾಂಶ>(params, dst, ಸುಳ್ಳು);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_conv_transpose_1d

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_conv_transpose_1d_f16_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
              ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nk = ne00*ne01*ne02;

    GGML_ASSERT(nb00 == ಗಾತ್ರ(ggml_fp16_t));
    GGML_ASSERT(nb10 == ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಆದರೆ (ith == 0) {
        memset(params->wdata, 0, params->wsize);

        // permute kernel data (src0) from (K x Cout x Cin) to (Cin x K x Cout)
        {
            ggml_fp16_t * ಸ್ಥಿರಾಂಕ wdata = (ggml_fp16_t *) params->wdata + 0;

            ಗಾಗಿ (int64_t i02 = 0; i02 < ne02; i02++) {
                ಗಾಗಿ (int64_t i01 = 0; i01 < ne01; i01++) {
                    ಸ್ಥಿರಾಂಕ ggml_fp16_t * ಸ್ಥಿರಾಂಕ src = (ggml_fp16_t *)((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i02*nb02 + i01*nb01);
                    ggml_fp16_t * dst_ದತ್ತಾಂಶ = wdata + i01*ne00*ne02;
                    ಗಾಗಿ (int64_t i00 = 0; i00 < ne00; i00++) {
                        dst_ದತ್ತಾಂಶ[i00*ne02 + i02] = src[i00];
                    }
                }
            }
        }

        // permute source data (src1) from (L x Cin) to (Cin x L)
        {
            ggml_fp16_t * ಸ್ಥಿರಾಂಕ wdata = (ggml_fp16_t *) params->wdata + nk;
            ggml_fp16_t * dst_ದತ್ತಾಂಶ = wdata;

            ಗಾಗಿ (int64_t i11 = 0; i11 < ne11; i11++) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸ್ಥಿರಾಂಕ src = (ದಶಮಾಂಶ *)((ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ + i11*nb11);
                ಗಾಗಿ (int64_t i10 = 0; i10 < ne10; i10++) {
                    dst_ದತ್ತಾಂಶ[i10*ne11 + i11] = GGML_CPU_FP32_ಗಮ್ಯ_FP16(src[i10]);
                }
            }
        }

        // need to zero dst since we are accumulating into it
        memset(dst->ದತ್ತಾಂಶ, 0, ggml_nbytes(dst));
    }
    ggml_barrier(params->threadpool);

    ಸ್ಥಿರಾಂಕ int32_t s0 = ((ಸ್ಥಿರಾಂಕ int32_t*)(dst->op_params))[0];

    // total rows in dst
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr = ne1;

    // rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ggml_fp16_t * ಸ್ಥಿರಾಂಕ wdata     = (ggml_fp16_t *) params->wdata + 0;
    ggml_fp16_t * ಸ್ಥಿರಾಂಕ wdata_src = wdata + nk;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i1 = ir0; i1 < ir1; i1++) {
        ದಶಮಾಂಶ * dst_ದತ್ತಾಂಶ = (ದಶಮಾಂಶ *)((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i1*nb1);
        ggml_fp16_t * wdata_kernel = wdata + i1*ne02*ne00;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i10 = 0; i10 < ne10; i10++) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i1n = i10*ne11;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i00 = 0; i00 < ne00; i00++) {
                ದಶಮಾಂಶ v = 0;
                ggml_vec_dot_f16(ne02, &v, 0,
                        (ggml_fp16_t *)    wdata_src + i1n, 0,
                        (ggml_fp16_t *) wdata_kernel + i00*ne02, 0, 1);
                dst_ದತ್ತಾಂಶ[i10*s0 + i00] += v;
            }
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_conv_transpose_1d_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
              ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nk = ne00*ne01*ne02;

    GGML_ASSERT(nb00 == ಗಾತ್ರ(ದಶಮಾಂಶ));
    GGML_ASSERT(nb10 == ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಆದರೆ (ith == 0) {
        memset(params->wdata, 0, params->wsize);

        // prepare kernel data (src0) from (K x Cout x Cin) to (Cin x K x Cout)
        {
            ದಶಮಾಂಶ * ಸ್ಥಿರಾಂಕ wdata = (ದಶಮಾಂಶ *) params->wdata + 0;

            ಗಾಗಿ (int64_t i02 = 0; i02 < ne02; i02++) {
                ಗಾಗಿ (int64_t i01 = 0; i01 < ne01; i01++) {
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸ್ಥಿರಾಂಕ src = (ದಶಮಾಂಶ *)((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i02*nb02 + i01*nb01);
                    ದಶಮಾಂಶ * dst_ದತ್ತಾಂಶ = wdata + i01*ne00*ne02;
                    ಗಾಗಿ (int64_t i00 = 0; i00 < ne00; i00++) {
                        dst_ದತ್ತಾಂಶ[i00*ne02 + i02] = src[i00];
                    }
                }
            }
        }

        // prepare source data (src1)
        {
            ದಶಮಾಂಶ * ಸ್ಥಿರಾಂಕ wdata = (ದಶಮಾಂಶ *) params->wdata + nk;
            ದಶಮಾಂಶ * dst_ದತ್ತಾಂಶ = wdata;

            ಗಾಗಿ (int64_t i11 = 0; i11 < ne11; i11++) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸ್ಥಿರಾಂಕ src = (ದಶಮಾಂಶ *)((ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ + i11*nb11);
                ಗಾಗಿ (int64_t i10 = 0; i10 < ne10; i10++) {
                    dst_ದತ್ತಾಂಶ[i10*ne11 + i11] = src[i10];
                }
            }
        }

        // need to zero dst since we are accumulating into it
        memset(dst->ದತ್ತಾಂಶ, 0, ggml_nbytes(dst));
    }
    ggml_barrier(params->threadpool);

    ಸ್ಥಿರಾಂಕ int32_t s0 = ((ಸ್ಥಿರಾಂಕ int32_t*)(dst->op_params))[0];

    // total rows in dst
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr = ne1;

    // rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ದಶಮಾಂಶ * ಸ್ಥಿರಾಂಕ wdata     = (ದಶಮಾಂಶ *) params->wdata + 0;
    ದಶಮಾಂಶ * ಸ್ಥಿರಾಂಕ wdata_src = wdata + nk;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i1 = ir0; i1 < ir1; i1++) {
        ದಶಮಾಂಶ * dst_ದತ್ತಾಂಶ = (ದಶಮಾಂಶ *)((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i1*nb1);
        ದಶಮಾಂಶ * wdata_kernel = wdata + i1*ne02*ne00;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i10 = 0; i10 < ne10; i10++) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i1n = i10*ne11;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i00 = 0; i00 < ne00; i00++) {
                ದಶಮಾಂಶ v = 0;
                ggml_vec_dot_f32(ne02, &v, 0,
                        wdata_src + i1n, 0,
                        wdata_kernel + i00*ne02, 0, 1);
                dst_ದತ್ತಾಂಶ[i10*s0 + i00] += v;
            }
        }
    }
}

ಶೂನ್ಯ ggml_compute_forward_conv_transpose_1d(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
              ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            {
                ggml_compute_forward_conv_transpose_1d_f16_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_conv_transpose_1d_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_im2col_f32
// src0: kernel [OC, IC, KH, KW]
// src1: image [N, IC, IH, IW]
// dst:  result [N, OH, OW, IC*KH*KW]
ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_im2col_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
              ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS;

    ಸ್ಥಿರಾಂಕ int32_t s0 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[0];
    ಸ್ಥಿರಾಂಕ int32_t s1 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[1];
    ಸ್ಥಿರಾಂಕ int32_t p0 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[2];
    ಸ್ಥಿರಾಂಕ int32_t p1 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[3];
    ಸ್ಥಿರಾಂಕ int32_t d0 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[4];
    ಸ್ಥಿರಾಂಕ int32_t d1 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[5];
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_2D = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[6] == 1;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ int64_t N  = is_2D ? ne13 : ne12;
    ಸ್ಥಿರಾಂಕ int64_t IC = is_2D ? ne12 : ne11;
    ಸ್ಥಿರಾಂಕ int64_t IH = is_2D ? ne11 : 1;
    ಸ್ಥಿರಾಂಕ int64_t IW = ne10;

    ಸ್ಥಿರಾಂಕ int64_t KH = is_2D ? ne01 : 1;
    ಸ್ಥಿರಾಂಕ int64_t KW = ne00;

    ಸ್ಥಿರಾಂಕ int64_t OH = is_2D ? ne2 : 1;
    ಸ್ಥಿರಾಂಕ int64_t OW = ne1;

    ಪೂರ್ಣಾಂಕ ofs0 = is_2D ? nb13 : nb12;
    ಪೂರ್ಣಾಂಕ ofs1 = is_2D ? nb12 : nb11;

    GGML_ASSERT(nb10 == ಗಾತ್ರ(ದಶಮಾಂಶ));

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    {
        ದಶಮಾಂಶ * ಸ್ಥಿರಾಂಕ wdata = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;

        ಗಾಗಿ (int64_t ಒಳ = 0; ಒಳ < N; ಒಳ++) {
            ಗಾಗಿ (int64_t ioh = 0; ioh < OH; ioh++) { // 1
                ಗಾಗಿ (int64_t iow = 0; iow < OW; iow++) {
                    ಗಾಗಿ (int64_t iic = ith; iic < IC; iic += nth) {

                        // micro kernel
                        ದಶಮಾಂಶ * dst_ದತ್ತಾಂಶ = wdata + (ಒಳ*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸ್ಥಿರಾಂಕ src_ದತ್ತಾಂಶ = (ದಶಮಾಂಶ *)((ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ + ಒಳ*ofs0 + iic*ofs1); // [IH, IW]

                        ಗಾಗಿ (int64_t ikh = 0; ikh < KH; ikh++) {  // 1
                            ಗಾಗಿ (int64_t ikw = 0; ikw < KW; ikw++) {
                                ಸ್ಥಿರಾಂಕ int64_t iiw = iow*s0 + ikw*d0 - p0;
                                ಸ್ಥಿರಾಂಕ int64_t iih = ioh*s1 + ikh*d1 - p1;

                                ಆದರೆ (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                    dst_ದತ್ತಾಂಶ[iic*(KH*KW) + ikh*KW + ikw] = 0;
                                } ಇಲ್ಲದಿದ್ದರೆ {
                                    dst_ದತ್ತಾಂಶ[iic*(KH*KW) + ikh*KW + ikw] = (src_ದತ್ತಾಂಶ[iih*IW + iiw]);
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
ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_im2col_f16(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
              ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16);

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS;

    ಸ್ಥಿರಾಂಕ int32_t s0 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[0];
    ಸ್ಥಿರಾಂಕ int32_t s1 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[1];
    ಸ್ಥಿರಾಂಕ int32_t p0 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[2];
    ಸ್ಥಿರಾಂಕ int32_t p1 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[3];
    ಸ್ಥಿರಾಂಕ int32_t d0 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[4];
    ಸ್ಥಿರಾಂಕ int32_t d1 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[5];
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_2D = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[6] == 1;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ int64_t N  = is_2D ? ne13 : ne12;
    ಸ್ಥಿರಾಂಕ int64_t IC = is_2D ? ne12 : ne11;
    ಸ್ಥಿರಾಂಕ int64_t IH = is_2D ? ne11 : 1;
    ಸ್ಥಿರಾಂಕ int64_t IW = ne10;

    ಸ್ಥಿರಾಂಕ int64_t KH = is_2D ? ne01 : 1;
    ಸ್ಥಿರಾಂಕ int64_t KW = ne00;

    ಸ್ಥಿರಾಂಕ int64_t OH = is_2D ? ne2 : 1;
    ಸ್ಥಿರಾಂಕ int64_t OW = ne1;

    ಪೂರ್ಣಾಂಕ ofs0 = is_2D ? nb13 : nb12;
    ಪೂರ್ಣಾಂಕ ofs1 = is_2D ? nb12 : nb11;

    GGML_ASSERT(nb10 == ggml_type_ಗಾತ್ರ(src1->type));

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    {
        ggml_fp16_t * ಸ್ಥಿರಾಂಕ wdata = (ggml_fp16_t *) dst->ದತ್ತಾಂಶ;

        ಗಾಗಿ (int64_t ಒಳ = 0; ಒಳ < N; ಒಳ++) {
            ಗಾಗಿ (int64_t ioh = 0; ioh < OH; ioh++) { // 1
                ಗಾಗಿ (int64_t iow = 0; iow < OW; iow++) {
                    ಗಾಗಿ (int64_t iic = ith; iic < IC; iic += nth) {

                        // micro kernel
                        ggml_fp16_t * dst_ದತ್ತಾಂಶ = wdata + (ಒಳ*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸ್ಥಿರಾಂಕ src_ದತ್ತಾಂಶ_f32 = src1->type == GGML_TYPE_F32
                            ? (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ + ಒಳ*ofs0 + iic*ofs1)
                            : ಶೂನ್ಯಸೂಚಕ; // [IH, IW]
                        ಸ್ಥಿರಾಂಕ ggml_fp16_t * ಸ್ಥಿರಾಂಕ src_ದತ್ತಾಂಶ_f16 = src1->type == GGML_TYPE_F16
                            ? (ಸ್ಥಿರಾಂಕ ggml_fp16_t *)((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ + ಒಳ*ofs0 + iic*ofs1)
                            : ಶೂನ್ಯಸೂಚಕ; // [IH, IW]

                        ಗಾಗಿ (int64_t ikh = 0; ikh < KH; ikh++) {  // 1
                            ಗಾಗಿ (int64_t ikw = 0; ikw < KW; ikw++) {
                                ಸ್ಥಿರಾಂಕ int64_t iiw = iow*s0 + ikw*d0 - p0;
                                ಸ್ಥಿರಾಂಕ int64_t iih = ioh*s1 + ikh*d1 - p1;

                                ಆದರೆ (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                    dst_ದತ್ತಾಂಶ[iic*(KH*KW) + ikh*KW + ikw] = 0;
                                } ಇಲ್ಲದಿದ್ದರೆ {
                                    ಆದರೆ (src_ದತ್ತಾಂಶ_f32 != ಶೂನ್ಯಸೂಚಕ) {
                                        dst_ದತ್ತಾಂಶ[iic*(KH*KW) + ikh*KW + ikw] = GGML_CPU_FP32_ಗಮ್ಯ_FP16(src_ದತ್ತಾಂಶ_f32[iih*IW + iiw]);
                                    } ಇಲ್ಲದಿದ್ದರೆ {
                                        dst_ದತ್ತಾಂಶ[iic*(KH*KW) + ikh*KW + ikw] = src_ದತ್ತಾಂಶ_f16[iih*IW + iiw];
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

ಶೂನ್ಯ ggml_compute_forward_im2col(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
              ggml_tensor * dst) {
    ಆಯ್ಕೆ (dst->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            {
                ggml_compute_forward_im2col_f16(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_im2col_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_im2col_back_f32

ಶೂನ್ಯ ggml_compute_forward_im2col_ಹಿಂಭಾಗ_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
              ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0]; // gradients of forward pass output
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1]; // convolution kernel

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS;

    ಸ್ಥಿರಾಂಕ int32_t s0 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[0];
    ಸ್ಥಿರಾಂಕ int32_t s1 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[1];
    ಸ್ಥಿರಾಂಕ int32_t p0 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[2];
    ಸ್ಥಿರಾಂಕ int32_t p1 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[3];
    ಸ್ಥಿರಾಂಕ int32_t d0 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[4];
    ಸ್ಥಿರಾಂಕ int32_t d1 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[5];
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_2D = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[6] == 1;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ int64_t N  = is_2D ? ne3 : ne2;
    ಸ್ಥಿರಾಂಕ int64_t IC = is_2D ? ne2 : ne1;
    ಸ್ಥಿರಾಂಕ int64_t IH = is_2D ? ne1 : 1;
    ಸ್ಥಿರಾಂಕ int64_t IW = ne0;

    ಸ್ಥಿರಾಂಕ int64_t KH = is_2D ? ne11 : 1;
    ಸ್ಥಿರಾಂಕ int64_t KW = ne10;

    ಸ್ಥಿರಾಂಕ int64_t OH = is_2D ? ne02 : 1;
    ಸ್ಥಿರಾಂಕ int64_t OW = ne01;

    ಪೂರ್ಣಾಂಕ ofs0 = is_2D ? nb3 : nb2;
    ಪೂರ್ಣಾಂಕ ofs1 = is_2D ? nb2 : nb1;

    GGML_ASSERT(nb0  == ಗಾತ್ರ(ದಶಮಾಂಶ));

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    {
        ದಶಮಾಂಶ * ಸ್ಥಿರಾಂಕ wdata = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;

        ಗಾಗಿ (int64_t ಒಳ = 0; ಒಳ < N; ಒಳ++) {
            ಗಾಗಿ (int64_t iic = ith; iic < IC; iic += nth) {
                ಗಾಗಿ (int64_t iih = 0; iih < IH; iih++) {
                    ಗಾಗಿ (int64_t iiw = 0; iiw < IW; iiw++) {

                        // micro kernel
                        ದಶಮಾಂಶ grad = 0.0f;
                        ಗಾಗಿ (int64_t ikh = 0; ikh < KH; ikh++) {
                            ಗಾಗಿ (int64_t ikw = 0; ikw < KW; ikw++) {
                                // For s0 > 1 some values were skipped over in the forward pass.
                                // These values have tmpw % s0 != 0 and need to be skipped in the backwards pass as well.
                                ಸ್ಥಿರಾಂಕ int64_t tmpw = (iiw + p0 - ikw*d0);
                                ಆದರೆ (tmpw % s0 != 0) {
                                    ಮುಂದುವರಿ;
                                }
                                ಸ್ಥಿರಾಂಕ int64_t iow = tmpw / s0;

                                // Equivalent logic as above except for s1.
                                int64_t ioh;
                                ಆದರೆ (is_2D) {
                                    ಸ್ಥಿರಾಂಕ int64_t tmph = iih + p1 - ikh*d1;

                                    ಆದರೆ (tmph % s1 != 0) {
                                        ಮುಂದುವರಿ;
                                    }

                                    ioh = tmph / s1;
                                } ಇಲ್ಲದಿದ್ದರೆ {
                                    ioh = 0;
                                }

                                ಆದರೆ (iow < 0 || iow >= OW || ioh < 0 || ioh >= OH) {
                                    ಮುಂದುವರಿ;
                                }

                                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸ್ಥಿರಾಂಕ grad_ಒಳ = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src0->ದತ್ತಾಂಶ
                                    + (ಒಳ*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
                                grad += grad_ಒಳ[iic*(KH*KW) + ikh*KW + ikw];
                            }
                        }
                        ದಶಮಾಂಶ * dst_ದತ್ತಾಂಶ = (ದಶಮಾಂಶ *)((ಅಕ್ಷರ *) wdata + (ಒಳ*ofs0 + iic*ofs1)); // [IH, IW]
                        dst_ದತ್ತಾಂಶ[iih*IW + iiw] = grad;
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
ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_im2col_3d_f16(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
              ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16);

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS;

    ಸ್ಥಿರಾಂಕ int32_t s0 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[0];
    ಸ್ಥಿರಾಂಕ int32_t s1 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[1];
    ಸ್ಥಿರಾಂಕ int32_t s2 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[2];
    ಸ್ಥಿರಾಂಕ int32_t p0 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[3];
    ಸ್ಥಿರಾಂಕ int32_t p1 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[4];
    ಸ್ಥಿರಾಂಕ int32_t p2 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[5];
    ಸ್ಥಿರಾಂಕ int32_t d0 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[6];
    ಸ್ಥಿರಾಂಕ int32_t d1 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[7];
    ಸ್ಥಿರಾಂಕ int32_t d2 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[8];
    ಸ್ಥಿರಾಂಕ int32_t IC = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[9];


    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ int64_t N  = ne13 / IC;
    ಸ್ಥಿರಾಂಕ int64_t ID = ne12;
    ಸ್ಥಿರಾಂಕ int64_t IH = ne11;
    ಸ್ಥಿರಾಂಕ int64_t IW = ne10;

    ಸ್ಥಿರಾಂಕ int64_t OC = ne03 / IC;
    GGML_ಬಳಸದ(OC);
    ಸ್ಥಿರಾಂಕ int64_t KD = ne02;
    ಸ್ಥಿರಾಂಕ int64_t KH = ne01;
    ಸ್ಥಿರಾಂಕ int64_t KW = ne00;

    ಸ್ಥಿರಾಂಕ int64_t OD = ne3 / N;
    ಸ್ಥಿರಾಂಕ int64_t OH = ne2;
    ಸ್ಥಿರಾಂಕ int64_t OW = ne1;
    ಸ್ಥಿರಾಂಕ int64_t OH_OW = OH*OW;
    ಸ್ಥಿರಾಂಕ int64_t KD_KH_KW = KD*KH*KW;
    ಸ್ಥಿರಾಂಕ int64_t KH_KW = KH*KW;
    ಸ್ಥಿರಾಂಕ int64_t IC_KD_KH_KW = IC*KD*KH*KW;

    GGML_ASSERT(nb10 == ಗಾತ್ರ(ದಶಮಾಂಶ));

    // im2col: [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
    {
        ggml_fp16_t * ಸ್ಥಿರಾಂಕ wdata = (ggml_fp16_t *) dst->ದತ್ತಾಂಶ;

        ಗಾಗಿ (int64_t ಒಳ = 0; ಒಳ < N; ಒಳ++) {
            ಗಾಗಿ (int64_t iod = 0; iod < OD; iod++) {
                ಗಾಗಿ (int64_t ioh = 0; ioh < OH; ioh++) {
                    ಗಾಗಿ (int64_t iow = 0; iow < OW; iow++) {
                        ಗಾಗಿ (int64_t iic = ith; iic < IC; iic += nth) {

                            // micro kernel
                            ggml_fp16_t * dst_ದತ್ತಾಂಶ = wdata + (ಒಳ*OD*OH_OW + iod*OH_OW + ioh*OW + iow)*IC_KD_KH_KW; // [IC, KD, KH, KW]
                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸ್ಥಿರಾಂಕ src_ದತ್ತಾಂಶ = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)src1->ದತ್ತಾಂಶ + (ಒಳ*IC + iic)*nb13); // [ID, IH, IW]

                            ಗಾಗಿ (int64_t ikd = 0; ikd < KD; ikd++) {
                                ಗಾಗಿ (int64_t ikh = 0; ikh < KH; ikh++) {
                                    ಗಾಗಿ (int64_t ikw = 0; ikw < KW; ikw++) {
                                        ಸ್ಥಿರಾಂಕ int64_t iiw = iow*s0 + ikw*d0 - p0;
                                        ಸ್ಥಿರಾಂಕ int64_t iih = ioh*s1 + ikh*d1 - p1;
                                        ಸ್ಥಿರಾಂಕ int64_t iid = iod*s2 + ikd*d2 - p2;

                                        ಆದರೆ (iid < 0 || iid >= ID || iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                            dst_ದತ್ತಾಂಶ[iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw] = 0;
                                        } ಇಲ್ಲದಿದ್ದರೆ {
                                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸ್ಥಿರಾಂಕ s = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)src_ದತ್ತಾಂಶ + iid*nb12 + iih*nb11 + iiw*nb10); // [ID, IH, IW]
                                            dst_ದತ್ತಾಂಶ[iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw] = GGML_CPU_FP32_ಗಮ್ಯ_FP16(*s);
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
ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_im2col_3d_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
              ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS;

    ಸ್ಥಿರಾಂಕ int32_t s0 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[0];
    ಸ್ಥಿರಾಂಕ int32_t s1 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[1];
    ಸ್ಥಿರಾಂಕ int32_t s2 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[2];
    ಸ್ಥಿರಾಂಕ int32_t p0 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[3];
    ಸ್ಥಿರಾಂಕ int32_t p1 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[4];
    ಸ್ಥಿರಾಂಕ int32_t p2 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[5];
    ಸ್ಥಿರಾಂಕ int32_t d0 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[6];
    ಸ್ಥಿರಾಂಕ int32_t d1 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[7];
    ಸ್ಥಿರಾಂಕ int32_t d2 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[8];
    ಸ್ಥಿರಾಂಕ int32_t IC = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[9];


    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ int64_t N  = ne13 / IC;
    ಸ್ಥಿರಾಂಕ int64_t ID = ne12;
    ಸ್ಥಿರಾಂಕ int64_t IH = ne11;
    ಸ್ಥಿರಾಂಕ int64_t IW = ne10;

    ಸ್ಥಿರಾಂಕ int64_t OC = ne03 / IC;
    GGML_ಬಳಸದ(OC);
    ಸ್ಥಿರಾಂಕ int64_t KD = ne02;
    ಸ್ಥಿರಾಂಕ int64_t KH = ne01;
    ಸ್ಥಿರಾಂಕ int64_t KW = ne00;

    ಸ್ಥಿರಾಂಕ int64_t OD = ne3 / N;
    ಸ್ಥಿರಾಂಕ int64_t OH = ne2;
    ಸ್ಥಿರಾಂಕ int64_t OW = ne1;

    ಸ್ಥಿರಾಂಕ int64_t OH_OW = OH*OW;
    ಸ್ಥಿರಾಂಕ int64_t KD_KH_KW = KD*KH*KW;
    ಸ್ಥಿರಾಂಕ int64_t KH_KW = KH*KW;
    ಸ್ಥಿರಾಂಕ int64_t IC_KD_KH_KW = IC*KD*KH*KW;

    GGML_ASSERT(nb10 == ಗಾತ್ರ(ದಶಮಾಂಶ));

    // im2col: [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
    {
        ದಶಮಾಂಶ * ಸ್ಥಿರಾಂಕ wdata = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;

        ಗಾಗಿ (int64_t ಒಳ = 0; ಒಳ < N; ಒಳ++) {
            ಗಾಗಿ (int64_t iod = 0; iod < OD; iod++) {
                ಗಾಗಿ (int64_t ioh = 0; ioh < OH; ioh++) {
                    ಗಾಗಿ (int64_t iow = 0; iow < OW; iow++) {
                        ಗಾಗಿ (int64_t iic = ith; iic < IC; iic += nth) {

                            // micro kernel
                            ದಶಮಾಂಶ * dst_ದತ್ತಾಂಶ = wdata + (ಒಳ*OD*OH_OW + iod*OH_OW + ioh*OW + iow)*IC_KD_KH_KW; // [IC, KD, KH, KW]
                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸ್ಥಿರಾಂಕ src_ದತ್ತಾಂಶ = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)src1->ದತ್ತಾಂಶ + (ಒಳ*IC + iic)*nb13); // [ID, IH, IW]

                            ಗಾಗಿ (int64_t ikd = 0; ikd < KD; ikd++) {
                                ಗಾಗಿ (int64_t ikh = 0; ikh < KH; ikh++) {
                                    ಗಾಗಿ (int64_t ikw = 0; ikw < KW; ikw++) {
                                        ಸ್ಥಿರಾಂಕ int64_t iiw = iow*s0 + ikw*d0 - p0;
                                        ಸ್ಥಿರಾಂಕ int64_t iih = ioh*s1 + ikh*d1 - p1;
                                        ಸ್ಥಿರಾಂಕ int64_t iid = iod*s2 + ikd*d2 - p2;

                                        ಆದರೆ (iid < 0 || iid >= ID || iih < 0 || iih >= IH || iiw < 0 || iiw >= IW || iid < 0 || iid >= ID) {
                                            dst_ದತ್ತಾಂಶ[iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw] = 0;
                                        } ಇಲ್ಲದಿದ್ದರೆ {
                                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸ್ಥಿರಾಂಕ s = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)src_ದತ್ತಾಂಶ + iid*nb12 + iih*nb11 + iiw*nb10); // [ID, IH, IW]
                                            dst_ದತ್ತಾಂಶ[iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw] = *s;
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


ಶೂನ್ಯ ggml_compute_forward_im2col_3d(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
              ggml_tensor * dst) {
    ಆಯ್ಕೆ (dst->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            {
                ggml_compute_forward_im2col_3d_f16(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_im2col_3d_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_call_mul_mat(ggml_type type, ಸ್ಥಿರಾಂಕ ggml_compute_params * params, int64_t m, int64_t n, int64_t k,
                              ಶೂನ್ಯ * a, ಶೂನ್ಯ * b, ದಶಮಾಂಶ * c) {
    ಸ್ಥಿರಾಂಕ ggml_type_traits * traits = ggml_get_type_traits(type);
    ರಚನೆ ggml_tensor src1 = {};
    src1.type  = type;
    src1.ne[0] = k;
    src1.ne[1] = m;
    src1.ne[2] = 1;
    src1.ne[3] = 1;
    src1.nb[0] = traits->type_ಗಾತ್ರ;
    src1.nb[1] = k * traits->type_ಗಾತ್ರ;
    src1.nb[2] = src1.nb[1];
    src1.nb[3] = src1.nb[2];
    src1.ದತ್ತಾಂಶ  = a;

    ರಚನೆ ggml_tensor src0 = {};
    src0.type  = type;
    src0.ne[0] = k;
    src0.ne[1] = n;
    src0.ne[2] = 1;
    src0.ne[3] = 1;
    src0.nb[0] = traits->type_ಗಾತ್ರ;
    src0.nb[1] = k * traits->type_ಗಾತ್ರ;
    src0.nb[2] = src0.nb[1];
    src0.nb[3] = src0.nb[2];
    src0.ದತ್ತಾಂಶ  = b;

    ರಚನೆ ggml_tensor dst = {};
    dst.ne[0] = n;
    dst.ne[1] = m;
    dst.ne[2] = 1;
    dst.ne[3] = 1;
    dst.nb[0] = ಗಾತ್ರ(ದಶಮಾಂಶ);
    dst.nb[1] = n * ಗಾತ್ರ(ದಶಮಾಂಶ);
    dst.nb[2] = dst.nb[1];
    dst.nb[3] = dst.nb[2];
    dst.ದತ್ತಾಂಶ  = c;
    dst.src[0] = &src0;
    dst.src[1] = &src1;

    ggml_compute_forward_mul_mat(params, &dst);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ int64_t ggml_wrap_around(int64_t coord, int64_t size) {
    ಹಿಂತಿರುಗಿ (coord  + size) % size; // adding size avoids negative number weirdness
}

// ggml_compute_forward_col2im_1d
//
// Scatter-add columns [K*OC, T_in] -> signal [T_out, OC]
// where T_out = (T_in - 1)*s + K - 2*p.  Gather approach: each output reads ceil(K/s) inputs.
// Parallelized over the time axis so the split stays balanced whatever OC is.
// Supports F32, F16, BF16 input/output (same type), F32 accumulator.

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು elem_t>
ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_col2im_1d_impl(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src = dst->src[0];  // [K*OC, T_in]

    GGML_ASSERT(ggml_is_contiguous(src));
    GGML_ASSERT(ggml_is_contiguous(dst));

    ಸ್ಥಿರಾಂಕ int32_t s0 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[0];
    ಸ್ಥಿರಾಂಕ int32_t OC = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[1];
    ಸ್ಥಿರಾಂಕ int32_t p0 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[2];

    ಸ್ಥಿರಾಂಕ int64_t K_OC = src->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t T_ಒಳ = src->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t K    = K_OC / OC;
    ಸ್ಥಿರಾಂಕ int64_t T_ಹೊರಹೋಗು = dst->ne[0];

    ಸ್ಥಿರಾಂಕ elem_t * ಲಂಬಸಾಲು_ದತ್ತಾಂಶ = (ಸ್ಥಿರಾಂಕ elem_t *) src->ದತ್ತಾಂಶ;
    elem_t       * dst_ದತ್ತಾಂಶ = (elem_t *) dst->ದತ್ತಾಂಶ;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    // Parallelize over the time axis: the split stays balanced whatever OC is,
    // down to OC = 1 for mono audio, and threads read disjoint column bands
    ಸ್ಥಿರಾಂಕ int64_t dr = (T_ಹೊರಹೋಗು + nth - 1) / nth;
    ಸ್ಥಿರಾಂಕ int64_t it0 = dr * ith;
    ಸ್ಥಿರಾಂಕ int64_t it1 = it0 + dr < T_ಹೊರಹೋಗು ? it0 + dr : T_ಹೊರಹೋಗು;

    ಗಾಗಿ (int64_t oc = 0; oc < OC; oc++) {
        ಗಾಗಿ (int64_t t_ಹೊರಹೋಗು = it0; t_ಹೊರಹೋಗು < it1; t_ಹೊರಹೋಗು++) {
            ಸ್ಥಿರಾಂಕ int64_t t_abs = t_ಹೊರಹೋಗು + p0;  // absolute position in uncropped signal
            // Gather: find all (t_in, k) where t_in * s + k == t_abs, 0 <= k < K
            int64_t t_ಒಳ_min = (t_abs - K + 1 + s0 - 1) / s0;  // ceil((t_abs-K+1)/s)
            ಆದರೆ (t_ಒಳ_min < 0) t_ಒಳ_min = 0;
            int64_t t_ಒಳ_max = t_abs / s0;
            ಆದರೆ (t_ಒಳ_max >= T_ಒಳ) t_ಒಳ_max = T_ಒಳ - 1;

            ದಶಮಾಂಶ ಮೊತ್ತ = 0.0f;
            ಗಾಗಿ (int64_t t_ಒಳ = t_ಒಳ_min; t_ಒಳ <= t_ಒಳ_max; t_ಒಳ++) {
                int64_t k = t_abs - t_ಒಳ * s0;
                ಆದರೆ (k >= 0 && k < K) {
                    // col layout: [K*OC, T_in], element (oc*K+k, t_in)
                    ಮೊತ್ತ += type_conversion_ಕೋಷ್ಟಕ<elem_t>::ಗಮ್ಯ_f32(ಲಂಬಸಾಲು_ದತ್ತಾಂಶ[(oc * K + k) + t_ಒಳ * K_OC]);
                }
            }
            // dst layout: [T_out, OC], element (t_out, oc)
            dst_ದತ್ತಾಂಶ[t_ಹೊರಹೋಗು + oc * T_ಹೊರಹೋಗು] = type_conversion_ಕೋಷ್ಟಕ<elem_t>::from_f32(ಮೊತ್ತ);
        }
    }
}

ಶೂನ್ಯ ggml_compute_forward_col2im_1d(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {
    ಆಯ್ಕೆ (dst->src[0]->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:  ggml_compute_forward_col2im_1d_impl<ದಶಮಾಂಶ>      (params, dst); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F16:  ggml_compute_forward_col2im_1d_impl<ggml_fp16_t>(params, dst); ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_BF16: ggml_compute_forward_col2im_1d_impl<ggml_bf16_t>(params, dst); ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: GGML_ABORT("col2im_1d: unsupported type %d", dst->src[0]->type);
    }
}

// ggml_compute_forward_conv_2d


ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_conv_2d_impl(ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
                                              ಸ್ಥಿರಾಂಕ ggml_tensor *         kernel,  // [KW, KH, IC, OC]
                                              ಸ್ಥಿರಾಂಕ ggml_tensor *         src,     // [W, H, C, N]
                                              ggml_tensor *               dst,     // [OW, OH, OC, N]
                                              ggml_type                   kernel_type) {

    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(kernel_type == GGML_TYPE_F16 || kernel_type == GGML_TYPE_F32);
    GGML_ASSERT(kernel->type == kernel_type);

    ಸ್ಥಿರಾಂಕ ggml_type_traits * traits = ggml_get_type_traits(kernel_type);

    ಸ್ಥಿರಾಂಕ int32_t stride_x   = dst->op_params[0];
    ಸ್ಥಿರಾಂಕ int32_t stride_y   = dst->op_params[1];
    ಸ್ಥಿರಾಂಕ int32_t pad_x      = dst->op_params[2];
    ಸ್ಥಿರಾಂಕ int32_t pad_y      = dst->op_params[3];
    ಸ್ಥಿರಾಂಕ int32_t dilation_x = dst->op_params[4];
    ಸ್ಥಿರಾಂಕ int32_t dilation_y = dst->op_params[5];

    ಸ್ಥಿರಾಂಕ int64_t c_ಒಳ  = src->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t c_ಹೊರಹೋಗು = kernel->ne[3];
    GGML_ASSERT(c_ಒಳ == kernel->ne[2]);

    ಸ್ಥಿರಾಂಕ int64_t src_w = src->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t src_h = src->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t knl_w = kernel->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t knl_h = kernel->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t dst_w = dst->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t dst_h = dst->ne[1];

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src_ದತ್ತಾಂಶ = (ದಶಮಾಂಶ *) src->ದತ್ತಾಂಶ;
    ಶೂನ್ಯ  * knl_ದತ್ತಾಂಶ       = kernel->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * dst_ದತ್ತಾಂಶ       = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;

    ಸ್ಥಿರಾಂಕ int64_t knl_n           = knl_w * knl_h * c_ಒಳ;
    ಸ್ಥಿರಾಂಕ int64_t patch_ಒಟ್ಟು     = dst->ne[3] * dst_w * dst_h;

    ಸ್ಥಿರಾಂಕ int64_t space_ಪ್ರತಿ_patch   = knl_n * traits->type_ಗಾತ್ರ + c_ಹೊರಹೋಗು * ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ int64_t batch_ಗಾತ್ರ        = params->wsize / space_ಪ್ರತಿ_patch;
    ಸ್ಥಿರಾಂಕ int64_t patches_ಪ್ರತಿ_batch = batch_ಗಾತ್ರ > 8 ? (batch_ಗಾತ್ರ / 8) * 8 : batch_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ int64_t batch_n           = (patch_ಒಟ್ಟು + patches_ಪ್ರತಿ_batch - 1) / patches_ಪ್ರತಿ_batch;

    GGML_ASSERT(patches_ಪ್ರತಿ_batch > 0 && batch_ಗಾತ್ರ >= 1);

    ಶೂನ್ಯ * tmp = params->wdata;

    ಗಾಗಿ (int64_t batch_i = 0; batch_i < batch_n; ++batch_i) {

        ಸ್ಥಿರಾಂಕ int64_t patch_ಆರಂಭ_batch = batch_i * patches_ಪ್ರತಿ_batch;
        ಸ್ಥಿರಾಂಕ int64_t patch_ಕೊನೆ_batch   = std::min(patch_ಆರಂಭ_batch + patches_ಪ್ರತಿ_batch,
                                              patch_ಒಟ್ಟು);
        ಸ್ಥಿರಾಂಕ int64_t patch_n           = patch_ಕೊನೆ_batch - patch_ಆರಂಭ_batch;

        ಸ್ಥಿರಾಂಕ int64_t patch_ಪ್ರತಿ_thread  = (patch_n + params->nth - 1) / params->nth;
        ಸ್ಥಿರಾಂಕ int64_t patch_ಆರಂಭ       = patch_ಆರಂಭ_batch + params->ith * patch_ಪ್ರತಿ_thread;
        ಸ್ಥಿರಾಂಕ int64_t patch_ಕೊನೆ         = std::min(patch_ಆರಂಭ + patch_ಪ್ರತಿ_thread, patch_ಕೊನೆ_batch);

        //im2col for a patch
        ಗಾಗಿ (int64_t p = patch_ಆರಂಭ; p < patch_ಕೊನೆ; ++p) {
            ಸ್ಥಿರಾಂಕ int64_t  batch_n     =  p / (dst_w * dst_h);
            ಸ್ಥಿರಾಂಕ int64_t  src_x       = (p / dst_w) % dst_h;
            ಸ್ಥಿರಾಂಕ int64_t  src_y       =  p % dst_w;

            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src_ತಳ = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)src_ದತ್ತಾಂಶ + batch_n * src->nb[3]);
            ಅಕ್ಷರ *        dst_ಅಡ್ಡಸಾಲು  = (ಅಕ್ಷರ *) tmp + (p % patches_ಪ್ರತಿ_batch) * knl_n * traits->type_ಗಾತ್ರ;

            ಗಾಗಿ (int64_t ic = 0; ic < c_ಒಳ; ++ic) {
                ಗಾಗಿ (int64_t ky = 0; ky < knl_h; ++ky) {
                    ಗಾಗಿ (int64_t kx = 0; kx < knl_w; ++kx) {
                        ಸ್ಥಿರಾಂಕ int64_t sy = src_x * stride_y + ky * dilation_y - pad_y;
                        ಸ್ಥಿರಾಂಕ int64_t sx = src_y * stride_x + kx * dilation_x - pad_x;

                        int64_t dst_idx = ic * (knl_h * knl_w) + ky * knl_w + kx;

                        ದಶಮಾಂಶ src_val;
                        ಆದರೆ (sy < 0 || sy >= src_h || sx < 0 || sx >= src_w) {
                            src_val = 0.0f;
                        } ಇಲ್ಲದಿದ್ದರೆ {
                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src_ptr = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)src_ತಳ + sx * src->nb[0] + sy * src->nb[1] + ic * src->nb[2]);
                            src_val               = *src_ptr;
                        }

                        ಅಕ್ಷರ * element_ptr = dst_ಅಡ್ಡಸಾಲು + dst_idx * traits->type_ಗಾತ್ರ;
                        ಆದರೆ (kernel_type == GGML_TYPE_F32) {
                            *(ದಶಮಾಂಶ *) element_ptr = src_val;
                        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (kernel_type == GGML_TYPE_F16) {
                            *(ggml_fp16_t *) element_ptr = GGML_CPU_FP32_ಗಮ್ಯ_FP16(src_val);
                        }
                    }
                }
            }
        }   // patches handled by this thread

        ggml_barrier(params->threadpool);

        ದಶಮಾಂಶ * gemm_output = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) tmp + patches_ಪ್ರತಿ_batch * knl_n * traits->type_ಗಾತ್ರ);

        GGML_ASSERT(gemm_output + patch_n * c_ಹೊರಹೋಗು <= (ದಶಮಾಂಶ*)tmp + params->wsize);

        // GEMM: patches[patch_n, knl_n] × kernel[knl_n, c_out] = output[patch_n, c_out]
        ggml_call_mul_mat(kernel_type, params, patch_n, c_ಹೊರಹೋಗು, knl_n, tmp, knl_ದತ್ತಾಂಶ, gemm_output);

        ggml_barrier(params->threadpool);


        //permute back [OC, N, OH, OW] to [N, OC, OH, OW]
        ಸ್ಥಿರಾಂಕ int64_t ಕ್ರಮಬದಲಾವಣೆ_ಪ್ರತಿ_thread = (patch_n + params->nth - 1) / params->nth;
        ಸ್ಥಿರಾಂಕ int64_t ಕ್ರಮಬದಲಾವಣೆ_ಆರಂಭ = params->ith * ಕ್ರಮಬದಲಾವಣೆ_ಪ್ರತಿ_thread;
        ಸ್ಥಿರಾಂಕ int64_t ಕ್ರಮಬದಲಾವಣೆ_ಕೊನೆ = std::min(ಕ್ರಮಬದಲಾವಣೆ_ಆರಂಭ + ಕ್ರಮಬದಲಾವಣೆ_ಪ್ರತಿ_thread, patch_n);

        ಗಾಗಿ (int64_t i = ಕ್ರಮಬದಲಾವಣೆ_ಆರಂಭ; i < ಕ್ರಮಬದಲಾವಣೆ_ಕೊನೆ; ++i) {
            ಸ್ಥಿರಾಂಕ int64_t p       = patch_ಆರಂಭ_batch + i;
            ಸ್ಥಿರಾಂಕ int64_t batch_n = p / (dst_w * dst_h);
            ಸ್ಥಿರಾಂಕ int64_t dst_y   = (p / dst_w) % dst_h;
            ಸ್ಥಿರಾಂಕ int64_t dst_x   = p % dst_w;

            ಗಾಗಿ (int64_t oc = 0; oc < c_ಹೊರಹೋಗು; ++oc) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ value = gemm_output[i * c_ಹೊರಹೋಗು + oc];
                ದಶಮಾಂಶ * dst_ptr = (ದಶಮಾಂಶ *)((ಅಕ್ಷರ *)dst_ದತ್ತಾಂಶ + dst_x * dst->nb[0] + dst_y * dst->nb[1] + oc * dst->nb[2] + batch_n * dst->nb[3]);
                *dst_ptr = value;
            }
        }
    }
}

ಶೂನ್ಯ ggml_compute_forward_conv_2d(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    ggml_compute_forward_conv_2d_impl(params, src0, src1, dst, src0->type);
}

// ggml_compute_forward_conv_3d

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_conv_3d_impl(ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
                                              ಸ್ಥಿರಾಂಕ ggml_tensor *         kernel,
                                              ಸ್ಥಿರಾಂಕ ggml_tensor *         src,
                                              ggml_tensor *               dst,
                                              ggml_type                   kernel_type) {

    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(kernel_type == GGML_TYPE_F16 || kernel_type == GGML_TYPE_F32);
    GGML_ASSERT(kernel->type == kernel_type);

    ಸ್ಥಿರಾಂಕ ggml_type_traits * traits = ggml_get_type_traits(kernel_type);

    ಸ್ಥಿರಾಂಕ int32_t s0 = dst->op_params[0];
    ಸ್ಥಿರಾಂಕ int32_t s1 = dst->op_params[1];
    ಸ್ಥಿರಾಂಕ int32_t s2 = dst->op_params[2];
    ಸ್ಥಿರಾಂಕ int32_t p0 = dst->op_params[3];
    ಸ್ಥಿರಾಂಕ int32_t p1 = dst->op_params[4];
    ಸ್ಥಿರಾಂಕ int32_t p2 = dst->op_params[5];
    ಸ್ಥಿರಾಂಕ int32_t d0 = dst->op_params[6];
    ಸ್ಥಿರಾಂಕ int32_t d1 = dst->op_params[7];
    ಸ್ಥಿರಾಂಕ int32_t d2 = dst->op_params[8];
    ಸ್ಥಿರಾಂಕ int32_t c  = dst->op_params[9];
    ಸ್ಥಿರಾಂಕ int32_t n  = dst->op_params[10];
    ಸ್ಥಿರಾಂಕ int32_t oc = dst->op_params[11];

    ಸ್ಥಿರಾಂಕ int64_t src_w = src->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t src_h = src->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t src_d = src->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t knl_w = kernel->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t knl_h = kernel->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t knl_d = kernel->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t dst_w = dst->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t dst_h = dst->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t dst_d = dst->ne[2];

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src_ದತ್ತಾಂಶ = (ದಶಮಾಂಶ *) src->ದತ್ತಾಂಶ;
    ಶೂನ್ಯ  * knl_ದತ್ತಾಂಶ       = kernel->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * dst_ದತ್ತಾಂಶ       = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;

    ಸ್ಥಿರಾಂಕ int64_t knl_n_ಪ್ರತಿ_channel = knl_w * knl_h * knl_d;
    ಸ್ಥಿರಾಂಕ int64_t knl_n_ಒಟ್ಟು       = knl_n_ಪ್ರತಿ_channel * c;
    ಸ್ಥಿರಾಂಕ int64_t patch_ಒಟ್ಟು       = n * dst_w * dst_h * dst_d;

    ಸ್ಥಿರಾಂಕ int64_t space_ಪ್ರತಿ_patch   = knl_n_ಒಟ್ಟು * traits->type_ಗಾತ್ರ + oc * ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ int64_t batch_ಗಾತ್ರ        = params->wsize / space_ಪ್ರತಿ_patch;
    ಸ್ಥಿರಾಂಕ int64_t patches_ಪ್ರತಿ_batch = batch_ಗಾತ್ರ > 8 ? (batch_ಗಾತ್ರ / 8) * 8 : batch_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ int64_t batch_n           = (patch_ಒಟ್ಟು + patches_ಪ್ರತಿ_batch - 1) / patches_ಪ್ರತಿ_batch;

    GGML_ASSERT(patches_ಪ್ರತಿ_batch > 0 && batch_ಗಾತ್ರ >= 1);

    ಶೂನ್ಯ * tmp = params->wdata;

    ಗಾಗಿ (int64_t batch_i = 0; batch_i < batch_n; ++batch_i) {
        ಸ್ಥಿರಾಂಕ int64_t patch_ಆರಂಭ_batch = batch_i * patches_ಪ್ರತಿ_batch;
        ಸ್ಥಿರಾಂಕ int64_t patch_ಕೊನೆ_batch   = std::min(patch_ಆರಂಭ_batch + patches_ಪ್ರತಿ_batch, patch_ಒಟ್ಟು);
        ಸ್ಥಿರಾಂಕ int64_t patch_n_ಒಳ_batch  = patch_ಕೊನೆ_batch - patch_ಆರಂಭ_batch;

        ಸ್ಥಿರಾಂಕ int64_t patch_ಪ್ರತಿ_thread  = (patch_n_ಒಳ_batch + params->nth - 1) / params->nth;
        ಸ್ಥಿರಾಂಕ int64_t patch_ಆರಂಭ       = patch_ಆರಂಭ_batch + params->ith * patch_ಪ್ರತಿ_thread;
        ಸ್ಥಿರಾಂಕ int64_t patch_ಕೊನೆ         = std::min(patch_ಆರಂಭ + patch_ಪ್ರತಿ_thread, patch_ಕೊನೆ_batch);

        ಗಾಗಿ (int64_t p = patch_ಆರಂಭ; p < patch_ಕೊನೆ; ++p) {
            ಸ್ಥಿರಾಂಕ int64_t p_ಒಳ_batch = p % (dst_w * dst_h * dst_d);
            ಸ್ಥಿರಾಂಕ int64_t p_ಒಳ_ಆಳ = p_ಒಳ_batch % (dst_w * dst_h);
            ಸ್ಥಿರಾಂಕ int64_t batch_idx  = p / (dst_w * dst_h * dst_d);
            ಸ್ಥಿರಾಂಕ int64_t dst_z      = p_ಒಳ_batch / (dst_w * dst_h);
            ಸ್ಥಿರಾಂಕ int64_t dst_y      = p_ಒಳ_ಆಳ / dst_w;
            ಸ್ಥಿರಾಂಕ int64_t dst_x      = p_ಒಳ_ಆಳ % dst_w;

            ಅಕ್ಷರ * dst_ಅಡ್ಡಸಾಲು = (ಅಕ್ಷರ *) tmp + (p % patches_ಪ್ರತಿ_batch) * knl_n_ಒಟ್ಟು * traits->type_ಗಾತ್ರ;

            ಗಾಗಿ (int64_t ic = 0; ic < c; ++ic) {
                ಗಾಗಿ (int64_t kz = 0; kz < knl_d; ++kz) {
                    ಗಾಗಿ (int64_t ky = 0; ky < knl_h; ++ky) {
                        ಗಾಗಿ (int64_t kx = 0; kx < knl_w; ++kx) {
                            ಸ್ಥಿರಾಂಕ int64_t sz = dst_z * s2 + kz * d2 - p2;
                            ಸ್ಥಿರಾಂಕ int64_t sy = dst_y * s1 + ky * d1 - p1;
                            ಸ್ಥಿರಾಂಕ int64_t sx = dst_x * s0 + kx * d0 - p0;

                            int64_t dst_idx = ic * knl_n_ಪ್ರತಿ_channel + kz * (knl_h * knl_w) + ky * knl_w + kx;

                            ದಶಮಾಂಶ src_val;
                            ಆದರೆ (sz < 0 || sz >= src_d || sy < 0 || sy >= src_h || sx < 0 || sx >= src_w) {
                                src_val = 0.0f;
                            } ಇಲ್ಲದಿದ್ದರೆ {
                                ಸ್ಥಿರಾಂಕ int64_t cn_idx = batch_idx * c + ic;
                                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src_ptr = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)src_ದತ್ತಾಂಶ + sx*src->nb[0] + sy*src->nb[1] + sz*src->nb[2] + cn_idx*src->nb[3]);
                                src_val = *src_ptr;
                            }

                            ಅಕ್ಷರ * element_ptr = dst_ಅಡ್ಡಸಾಲು + dst_idx * traits->type_ಗಾತ್ರ;
                            ಆದರೆ (kernel_type == GGML_TYPE_F32) {
                                *(ದಶಮಾಂಶ *)element_ptr = src_val;
                            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (kernel_type == GGML_TYPE_F16) {
                                *(ggml_fp16_t *)element_ptr = GGML_CPU_FP32_ಗಮ್ಯ_FP16(src_val);
                            }
                        }
                    }
                }
            }
        }

        ggml_barrier(params->threadpool);

        ದಶಮಾಂಶ * gemm_output = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) tmp + patches_ಪ್ರತಿ_batch * knl_n_ಒಟ್ಟು * traits->type_ಗಾತ್ರ);
        ggml_call_mul_mat(kernel_type, params, patch_n_ಒಳ_batch, oc, knl_n_ಒಟ್ಟು, tmp, knl_ದತ್ತಾಂಶ, gemm_output);

        ggml_barrier(params->threadpool);

        ಸ್ಥಿರಾಂಕ int64_t ಕ್ರಮಬದಲಾವಣೆ_ಪ್ರತಿ_thread = (patch_n_ಒಳ_batch + params->nth - 1) / params->nth;
        ಸ್ಥಿರಾಂಕ int64_t ಕ್ರಮಬದಲಾವಣೆ_ಆರಂಭ = params->ith * ಕ್ರಮಬದಲಾವಣೆ_ಪ್ರತಿ_thread;
        ಸ್ಥಿರಾಂಕ int64_t ಕ್ರಮಬದಲಾವಣೆ_ಕೊನೆ = std::min(ಕ್ರಮಬದಲಾವಣೆ_ಆರಂಭ + ಕ್ರಮಬದಲಾವಣೆ_ಪ್ರತಿ_thread, patch_n_ಒಳ_batch);

        ಗಾಗಿ (int64_t i = ಕ್ರಮಬದಲಾವಣೆ_ಆರಂಭ; i < ಕ್ರಮಬದಲಾವಣೆ_ಕೊನೆ; ++i) {
            ಸ್ಥಿರಾಂಕ int64_t p = patch_ಆರಂಭ_batch + i;
            ಸ್ಥಿರಾಂಕ int64_t p_ಒಳ_batch = p % (dst_w * dst_h * dst_d);
            ಸ್ಥಿರಾಂಕ int64_t p_ಒಳ_ಆಳ = p_ಒಳ_batch % (dst_w * dst_h);
            ಸ್ಥಿರಾಂಕ int64_t batch_idx  = p / (dst_w * dst_h * dst_d);
            ಸ್ಥಿರಾಂಕ int64_t dst_z      = p_ಒಳ_batch / (dst_w * dst_h);
            ಸ್ಥಿರಾಂಕ int64_t dst_y      = p_ಒಳ_ಆಳ / dst_w;
            ಸ್ಥಿರಾಂಕ int64_t dst_x      = p_ಒಳ_ಆಳ % dst_w;

            ಗಾಗಿ (int64_t ioc = 0; ioc < oc; ++ioc) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ value = gemm_output[i * oc + ioc];
                ಸ್ಥಿರಾಂಕ int64_t ocn_idx = batch_idx * oc + ioc;
                ದಶಮಾಂಶ * dst_ptr = (ದಶಮಾಂಶ *)((ಅಕ್ಷರ *)dst_ದತ್ತಾಂಶ + dst_x*dst->nb[0] + dst_y*dst->nb[1] + dst_z*dst->nb[2] + ocn_idx*dst->nb[3]);
                *dst_ptr = value;
            }
        }
    }
}

ಶೂನ್ಯ ggml_compute_forward_conv_3d(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];
    ggml_compute_forward_conv_3d_impl(params, src0, src1, dst, src0->type);
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು kernel_t>
ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_conv_transpose_2d_impl(
    ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
          ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nk = ne00*ne01*ne02*ne03;

    GGML_ASSERT(nb00 == ggml_type_ಗಾತ್ರ(src0->type));
    GGML_ASSERT(nb10 == ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಆದರೆ (ith == 0) {
        memset(params->wdata, 0, params->wsize);

        // permute kernel data (src0) from (Kw x Kh x Cout x Cin) to (Cin x Kw x Kh x Cout)
        {
            kernel_t * ಸ್ಥಿರಾಂಕ wdata = (kernel_t *) params->wdata + 0;

            ಗಾಗಿ (int64_t i03 = 0; i03 < ne03; i03++) {
                ಗಾಗಿ (int64_t i02 = 0; i02 < ne02; i02++) {
                    ಸ್ಥಿರಾಂಕ kernel_t * ಸ್ಥಿರಾಂಕ src = (kernel_t *)((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i03*nb03 + i02*nb02);
                    kernel_t * dst_ದತ್ತಾಂಶ = wdata + i02*ne01*ne00*ne03;
                    ಗಾಗಿ (int64_t i01 = 0; i01 < ne01; i01++) {
                        ಗಾಗಿ (int64_t i00 = 0; i00 < ne00; i00++) {
                            dst_ದತ್ತಾಂಶ[i01*ne00*ne03 + i00*ne03 + i03] = src[i01 * ne00 + i00];
                        }
                    }
                }
            }
        }

        // permute source data (src1) from (Sw x Sh x Cin) to (Cin x Sw x Sh), for all batches
        {
            kernel_t * ಸ್ಥಿರಾಂಕ wdata = (kernel_t *) params->wdata + nk;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i13 = 0; i13 < ne13; i13++) {
                kernel_t * ಸ್ಥಿರಾಂಕ wdata_b = wdata + i13*ne10*ne11*ne12;
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i12 = 0; i12 < ne12; i12++) {
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i11 = 0; i11 < ne11; i11++) {
                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸ್ಥಿರಾಂಕ src = (ದಶಮಾಂಶ *)((ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ + i13*nb13 + i12*nb12 + i11*nb11);
                        kernel_t * dst_ದತ್ತಾಂಶ = wdata_b + i11*ne10*ne12;
                        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i10 = 0; i10 < ne10; i10++) {
                            ಆದರೆ constexpr (std::is_same_v<kernel_t, ggml_fp16_t>) {
                                dst_ದತ್ತಾಂಶ[i10*ne12 + i12] = GGML_CPU_FP32_ಗಮ್ಯ_FP16(src[i10]);
                            } ಇಲ್ಲದಿದ್ದರೆ {
                                dst_ದತ್ತಾಂಶ[i10*ne12 + i12] = src[i10];
                            }
                        }
                    }
                }
            }
        }

        memset(dst->ದತ್ತಾಂಶ, 0, ggml_nbytes(dst));
    }
    ggml_barrier(params->threadpool);

    ಸ್ಥಿರಾಂಕ int32_t stride = ggml_get_op_params_i32(dst, 0);

    // total patches in dst
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ np = ne2;

    // patches per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dp = (np + nth - 1)/nth;

    // patch range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ip0 = dp*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ip1 = MIN(ip0 + dp, np);

    kernel_t * ಸ್ಥಿರಾಂಕ wdata = (kernel_t *) params->wdata + 0;
    kernel_t * ಸ್ಥಿರಾಂಕ wdata_src = wdata + nk;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i2 = ip0; i2 < ip1; i2++) { // Cout
        kernel_t * wdata_kernel = wdata + i2*ne01*ne00*ne03;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i3 = 0; i3 < ne3; i3++) { // batch
            ದಶಮಾಂಶ * dst_ದತ್ತಾಂಶ = (ದಶಮಾಂಶ *)((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i3*nb3 + i2*nb2);
            kernel_t * wdata_src_b = wdata_src + i3*ne10*ne11*ne12;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i11 = 0; i11 < ne11; i11++) {
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i10 = 0; i10 < ne10; i10++) {
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i1n = i11*ne10*ne12 + i10*ne12;
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i01 = 0; i01 < ne01; i01++) {
                        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i00 = 0; i00 < ne00; i00++) {
                            ದಶಮಾಂಶ v = 0;
                            ಆದರೆ constexpr (std::is_same_v<kernel_t, ggml_fp16_t>) {
                                ggml_vec_dot_f16(ne03, &v, 0,
                                        wdata_src_b + i1n, 0,
                                        wdata_kernel + i01*ne00*ne03 + i00*ne03, 0, 1);
                            } ಇಲ್ಲದಿದ್ದರೆ {
                                ggml_vec_dot_f32(ne03, &v, 0,
                                        wdata_src_b + i1n, 0,
                                        wdata_kernel + i01*ne00*ne03 + i00*ne03, 0, 1);
                            }
                            dst_ದತ್ತಾಂಶ[(i11*stride + i01)*ne0 + i10*stride + i00] += v;
                        }
                    }
                }
            }
        }
    }
}

ಶೂನ್ಯ ggml_compute_forward_conv_transpose_2d(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
              ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            {
                ggml_compute_forward_conv_transpose_2d_impl<ggml_fp16_t>(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_conv_transpose_2d_impl<ದಶಮಾಂಶ>(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_conv_2d_dw

ರಚನೆ ggml_conv_2d_dw_params {
    int64_t channels;
    int64_t batch;
    int64_t src_w;
    int64_t src_h;
    int64_t dst_w;
    int64_t dst_h;
    int64_t knl_w;
    int64_t knl_h;
    ಪೂರ್ಣಾಂಕ stride_x;
    ಪೂರ್ಣಾಂಕ stride_y;
    ಪೂರ್ಣಾಂಕ pad_x;
    ಪೂರ್ಣಾಂಕ pad_y;
    ಪೂರ್ಣಾಂಕ dilation_x;
    ಪೂರ್ಣಾಂಕ dilation_y;
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ ggml_conv_2d_dw_knl_f32(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ದತ್ತಾಂಶ, int64_t i, ggml_type type) {
    ಆದರೆ (type == GGML_TYPE_F16) {
        ಹಿಂತಿರುಗಿ GGML_FP16_ಗಮ್ಯ_FP32(((ಸ್ಥಿರಾಂಕ ggml_fp16_t *)ದತ್ತಾಂಶ)[i]);
    }
    ಹಿಂತಿರುಗಿ ((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)ದತ್ತಾಂಶ)[i];
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_conv_2d_dw_cwhn(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ಸ್ಥಿರಾಂಕ ggml_tensor * src,
        ಸ್ಥಿರಾಂಕ ggml_tensor * kernel,
        ggml_tensor * dst,
        ಸ್ಥಿರಾಂಕ ggml_conv_2d_dw_params & p) {

    ಸ್ಥಿರಾಂಕ int64_t c = p.channels;
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * knl_ದತ್ತಾಂಶ = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)kernel->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ggml_type knl_type = kernel->type;

    ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲುಗಳು_ಒಟ್ಟು = p.dst_h * p.batch;
    ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread = (ಅಡ್ಡಸಾಲುಗಳು_ಒಟ್ಟು + params->nth - 1) / params->nth;
    ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_ಆರಂಭ = params->ith * ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread;
    ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_ಕೊನೆ = MIN(ಅಡ್ಡಸಾಲು_ಆರಂಭ + ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread, ಅಡ್ಡಸಾಲುಗಳು_ಒಟ್ಟು);

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SIMD
    int64_t c_pkg_ಕೊನೆ = 0;
    int64_t pkg_ಗಾತ್ರ = GGML_F32_EPR;
    ಆದರೆ (knl_type == GGML_TYPE_F32) {
    #ಆದರೆ defined(__ARM_FEATURE_SVE)
        pkg_ಗಾತ್ರ = svcntw();
    #ಇಲ್ಲದಿದ್ದರೆ
        pkg_ಗಾತ್ರ = GGML_F32_EPR;
    #ಆದರೆ_ಅಂತ್ಯ
        c_pkg_ಕೊನೆ = (c / pkg_ಗಾತ್ರ) * pkg_ಗಾತ್ರ;
    }
#ಇಲ್ಲದಿದ್ದರೆ
    ಸ್ಥಿರಾಂಕ int64_t c_pkg_ಕೊನೆ = 0;
#ಆದರೆ_ಅಂತ್ಯ

    ಗಾಗಿ (int64_t ಅಡ್ಡಸಾಲು = ಅಡ್ಡಸಾಲು_ಆರಂಭ; ಅಡ್ಡಸಾಲು < ಅಡ್ಡಸಾಲು_ಕೊನೆ; ++ಅಡ್ಡಸಾಲು) {
        ಸ್ಥಿರಾಂಕ int64_t dst_y = ಅಡ್ಡಸಾಲು % p.dst_h;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src_ದತ್ತಾಂಶ = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)src->ದತ್ತಾಂಶ + (ಅಡ್ಡಸಾಲು / p.dst_h) * p.src_w * p.src_h * c;
        ಗಾಗಿ (int64_t dst_x = 0; dst_x < p.dst_w; ++dst_x) {
            ದಶಮಾಂಶ * dst_ದತ್ತಾಂಶ = (ದಶಮಾಂಶ *)dst->ದತ್ತಾಂಶ + (ಅಡ್ಡಸಾಲು * p.dst_w + dst_x) * c;
            ಸ್ಥಿರಾಂಕ int64_t src_y_ತಳ = dst_y * p.stride_y - p.pad_y;
            ಸ್ಥಿರಾಂಕ int64_t src_x_ತಳ = dst_x * p.stride_x - p.pad_x;

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SIMD
            ಗಾಗಿ (int64_t c_i = 0; c_i < c_pkg_ಕೊನೆ; c_i += pkg_ಗಾತ್ರ) {
                GGML_F32_VEC ಮೊತ್ತ = GGML_F32_VEC_ZERO;
                ಗಾಗಿ (int64_t knl_y = 0; knl_y < p.knl_h; ++knl_y) {
                    ಸ್ಥಿರಾಂಕ int64_t src_y = src_y_ತಳ + knl_y * p.dilation_y;
                    ಆದರೆ (src_y < 0 || src_y >= p.src_h) {
                        ಮುಂದುವರಿ;
                    }
                    ಗಾಗಿ (int64_t knl_x = 0; knl_x < p.knl_w; ++knl_x) {
                        ಸ್ಥಿರಾಂಕ int64_t src_x = src_x_ತಳ + knl_x * p.dilation_x;
                        ಆದರೆ (src_x < 0 || src_x >= p.src_w) {
                            ಮುಂದುವರಿ;
                        }
                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * kp = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)knl_ದತ್ತಾಂಶ + (knl_y * p.knl_w + knl_x) * c + c_i;
                        GGML_F32_VEC k = GGML_F32_VEC_ತುಂಬು(kp);
                        GGML_F32_VEC s = GGML_F32_VEC_ತುಂಬು(src_ದತ್ತಾಂಶ + (src_y * p.src_w + src_x) * c + c_i);
                        ಮೊತ್ತ = GGML_F32_VEC_FMA(ಮೊತ್ತ, k, s);
                    }
                }
                GGML_F32_VEC_STORE(dst_ದತ್ತಾಂಶ + c_i, ಮೊತ್ತ);
            }
#ಆದರೆ_ಅಂತ್ಯ
            ಗಾಗಿ (int64_t c_i = c_pkg_ಕೊನೆ; c_i < c; ++c_i) {
                ದಶಮಾಂಶ ಮೊತ್ತ = 0.0f;
                ಗಾಗಿ (int64_t knl_y = 0; knl_y < p.knl_h; ++knl_y) {
                    ಸ್ಥಿರಾಂಕ int64_t src_y = src_y_ತಳ + knl_y * p.dilation_y;
                    ಆದರೆ (src_y < 0 || src_y >= p.src_h) {
                        ಮುಂದುವರಿ;
                    }
                    ಗಾಗಿ (int64_t knl_x = 0; knl_x < p.knl_w; ++knl_x) {
                        ಸ್ಥಿರಾಂಕ int64_t src_x = src_x_ತಳ + knl_x * p.dilation_x;
                        ಆದರೆ (src_x < 0 || src_x >= p.src_w) {
                            ಮುಂದುವರಿ;
                        }
                        ಮೊತ್ತ += ggml_conv_2d_dw_knl_f32(knl_ದತ್ತಾಂಶ, (knl_y * p.knl_w + knl_x) * c + c_i, knl_type)
                             * src_ದತ್ತಾಂಶ[(src_y * p.src_w + src_x) * c + c_i];
                    }
                }
                dst_ದತ್ತಾಂಶ[c_i] = ಮೊತ್ತ;
            }
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_conv_2d_dw_whcn(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ಸ್ಥಿರಾಂಕ ggml_tensor * src,
        ಸ್ಥಿರಾಂಕ ggml_tensor * kernel,
        ggml_tensor * dst,
        ಸ್ಥಿರಾಂಕ ggml_conv_2d_dw_params & p) {

    ಸ್ಥಿರಾಂಕ int64_t n = p.channels * p.batch;
    ಸ್ಥಿರಾಂಕ int64_t ಪ್ರತಿ_thread = (n + params->nth - 1) / params->nth;
    ಸ್ಥಿರಾಂಕ int64_t start = params->ith * ಪ್ರತಿ_thread;
    ಸ್ಥಿರಾಂಕ int64_t end = MIN(start + ಪ್ರತಿ_thread, n);
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * knl_ತಳ = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)kernel->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ggml_type knl_type = kernel->type;

    ಗಾಗಿ (int64_t i = start; i < end; ++i) {
        ಸ್ಥಿರಾಂಕ int64_t knl_ಸ್ಥಾನಾಂತರ = (i % p.channels) * p.knl_w * p.knl_h;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src_ದತ್ತಾಂಶ = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)src->ದತ್ತಾಂಶ + i * p.src_w * p.src_h;
        ದಶಮಾಂಶ * dst_ದತ್ತಾಂಶ = (ದಶಮಾಂಶ *)dst->ದತ್ತಾಂಶ + i * p.dst_w * p.dst_h;

        ಗಾಗಿ (int64_t dst_y = 0; dst_y < p.dst_h; ++dst_y) {
            ಗಾಗಿ (int64_t dst_x = 0; dst_x < p.dst_w; ++dst_x) {

                ದಶಮಾಂಶ ಮೊತ್ತ = 0.0f;
                ಗಾಗಿ (int64_t knl_y = 0; knl_y < p.knl_h; ++knl_y) {
                    ಸ್ಥಿರಾಂಕ int64_t src_y = dst_y * p.stride_y + knl_y * p.dilation_y - p.pad_y;
                    ಆದರೆ (src_y < 0 || src_y >= p.src_h) {
                        ಮುಂದುವರಿ;
                    }
                    ಗಾಗಿ (int64_t knl_x = 0; knl_x < p.knl_w; ++knl_x) {
                        ಸ್ಥಿರಾಂಕ int64_t src_x = dst_x * p.stride_x + knl_x * p.dilation_x - p.pad_x;
                        ಆದರೆ (src_x < 0 || src_x >= p.src_w) {
                            ಮುಂದುವರಿ;
                        }
                        ಮೊತ್ತ += ggml_conv_2d_dw_knl_f32(knl_ತಳ, knl_ಸ್ಥಾನಾಂತರ + knl_y * p.knl_w + knl_x, knl_type)
                             * src_ದತ್ತಾಂಶ[src_y * p.src_w + src_x];
                    }
                }
                dst_ದತ್ತಾಂಶ[dst_y * p.dst_w + dst_x] = ಮೊತ್ತ;
            }
        }
    }
}

ಶೂನ್ಯ ggml_compute_forward_conv_2d_dw(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * kernel = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src = dst->src[1];
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

    ಆದರೆ (ggml_is_contiguous(src)) {
        ggml_compute_forward_conv_2d_dw_whcn(params, src, kernel, dst, p);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ggml_is_contiguous_channels(src)) {
        GGML_ASSERT(kernel->nb[0] >= kernel->nb[2] && kernel->nb[1] >= kernel->nb[0]);
        ggml_compute_forward_conv_2d_dw_cwhn(params, src, kernel, dst, p);
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_ABORT("non-contiguous memory layout not supported");
    }
}

// ggml_compute_forward_pool_1d_ksp
ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_pool_1d_ksp(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ಸ್ಥಿರಾಂಕ ggml_op_pool op,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ p,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src = dst->src[0];

    ಪ್ರತಿಜ್ಞಾ(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16);

    ಆದರೆ (params->ith != 0) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ int64_t IW = src->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t OW = dst->ne[0];

    ಸ್ಥಿರಾಂಕ int64_t nr = ggml_nrows(src);

    ಗಾಗಿ (int64_t ir = 0; ir < nr; ++ir) {
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * srow_ಬೈಟುಗಳು =            (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src->ದತ್ತಾಂಶ + ir * src->nb[1];
        ದಶಮಾಂಶ      * drow       = (ದಶಮಾಂಶ *) ((      ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + ir * dst->nb[1]);

        ಗಾಗಿ (int64_t ow = 0; ow < OW; ++ow) {
            ದಶಮಾಂಶ res = 0;
            ಆಯ್ಕೆ (op) {
                ಸ್ಥಿತಿ GGML_OP_POOL_AVG: res = 0.0f;     ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_OP_POOL_MAX: res = -FLT_MAX; ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_OP_POOL_ಎಣಿಕೆ: GGML_ABORT("fatal error");
            }

            ಪೂರ್ಣಾಂಕ count = 0;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ತಳ = (ಪೂರ್ಣಾಂಕ) ow * s - p;

            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ki = 0; ki < k; ++ki) {
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ j = ತಳ + ki;
                ಆದರೆ (j < 0 || j >= (ಪೂರ್ಣಾಂಕ) IW) {
                    ಮುಂದುವರಿ;
                }

                ದಶಮಾಂಶ v;
                ಆದರೆ (src->type == GGML_TYPE_F32) {
                    v = ((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) srow_ಬೈಟುಗಳು)[j];
                } ಇಲ್ಲದಿದ್ದರೆ {
                    v = GGML_CPU_FP16_ಗಮ್ಯ_FP32(((ಸ್ಥಿರಾಂಕ ggml_fp16_t *) srow_ಬೈಟುಗಳು)[j]);
                }

                ಆಯ್ಕೆ (op) {
                    ಸ್ಥಿತಿ GGML_OP_POOL_AVG: res += v;                ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ GGML_OP_POOL_MAX: res =  std::max(v, res); ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ GGML_OP_POOL_ಎಣಿಕೆ: GGML_ABORT("fatal error");
                }

                ++count;
            }

            ಆಯ್ಕೆ (op) {
                ಸ್ಥಿತಿ GGML_OP_POOL_AVG: res = (count > 0) ? (res / count) : 0.0f; ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_OP_POOL_MAX:                                           ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_OP_POOL_ಎಣಿಕೆ: GGML_ABORT("fatal error");
            }

            drow[ow] = res;
        }
    }
}

// ggml_compute_forward_pool_1d

ಶೂನ್ಯ ggml_compute_forward_pool_1d(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
              ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ int32_t * opts = (ಸ್ಥಿರಾಂಕ int32_t *)dst->op_params;
    ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k0 = opts[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s0 = opts[2];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ p0 = opts[3];

    ggml_compute_forward_pool_1d_ksp(params, op, k0, s0, p0, dst);
}

// ggml_compute_forward_pool_2d

ಶೂನ್ಯ ggml_compute_forward_pool_2d(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src = dst->src[0];

    ಪ್ರತಿಜ್ಞಾ(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16);

    ಆದರೆ (params->ith != 0) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ int32_t * opts = (ಸ್ಥಿರಾಂಕ int32_t *)dst->op_params;

    ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k0 = opts[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k1 = opts[2];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s0 = opts[3];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s1 = opts[4];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ p0 = opts[5];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ p1 = opts[6];
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * cdata = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ*)src->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಸ್ಥಿರಾಂಕ ದತ್ತಾಂಶ_ಕೊನೆ = cdata + ggml_nbytes(src);

    ಸ್ಥಿರಾಂಕ int64_t px = dst->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t py = dst->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t pa = px * py;

    ದಶಮಾಂಶ * dplane = (ದಶಮಾಂಶ *)dst->ದತ್ತಾಂಶ;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ka = k0 * k1;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ offset0 = -p0;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ offset1 = -p1;

    ಆಗ (cdata < ದತ್ತಾಂಶ_ಕೊನೆ) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ oy = 0; oy < py; ++oy) {
            ದಶಮಾಂಶ * ಸ್ಥಿರಾಂಕ drow = dplane + oy * px;
            ದಶಮಾಂಶ * ಸ್ಥಿರಾಂಕ ಹೊರಹೋಗು  = drow;

            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ox = 0; ox < px; ++ox) {
                ದಶಮಾಂಶ res = 0;
                ಆಯ್ಕೆ (op) {
                    ಸ್ಥಿತಿ GGML_OP_POOL_AVG: res = 0;        ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ GGML_OP_POOL_MAX: res = -FLT_MAX; ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ GGML_OP_POOL_ಎಣಿಕೆ: GGML_ABORT("fatal error");
                }

                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ix = offset0 + ox * s0;
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iy = offset1 + oy * s1;

                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ky = 0; ky < k1; ++ky) {
                    ಆದರೆ (iy + ky < 0 || iy + ky >= src->ne[1]) {
                        ಮುಂದುವರಿ;
                    }

                    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * srow = (ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *)(cdata + src->nb[1] * (iy + ky));
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ kx = 0; kx < k0; ++kx) {
                        ಪೂರ್ಣಾಂಕ j = ix + kx;
                        ಆದರೆ (j < 0 || j >= src->ne[0]) {
                            ಮುಂದುವರಿ;
                        }

                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ srow_j = (src->type == GGML_TYPE_F32) ? ((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ*)srow)[j] : GGML_CPU_FP16_ಗಮ್ಯ_FP32(((ಸ್ಥಿರಾಂಕ ggml_fp16_t*)srow)[j]);
                        ಆಯ್ಕೆ (op) {
                            ಸ್ಥಿತಿ GGML_OP_POOL_AVG: res += srow_j;                ನಿಲ್ಲಿಸು;
                            ಸ್ಥಿತಿ GGML_OP_POOL_MAX: res =  std::max(srow_j, res); ನಿಲ್ಲಿಸು;
                            ಸ್ಥಿತಿ GGML_OP_POOL_ಎಣಿಕೆ:               GGML_ABORT("fatal error");
                        }
                    }
                }
                ಆಯ್ಕೆ (op) {
                    ಸ್ಥಿತಿ GGML_OP_POOL_AVG:           res /= ka; ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ GGML_OP_POOL_MAX:                      ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ GGML_OP_POOL_ಎಣಿಕೆ: GGML_ABORT("fatal error");
                }

                ಹೊರಹೋಗು[ox] = res;
            }
        }

        cdata  += src->nb[2];
        dplane += pa;
    }
}

// ggml_compute_forward_pool_2d_back

ಶೂನ್ಯ ggml_compute_forward_pool_2d_ಹಿಂಭಾಗ(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src  = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * dstf = dst->src[1]; // forward tensor of dst

    ಪ್ರತಿಜ್ಞಾ(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);

    ಆದರೆ (params->ith != 0) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ int32_t * opts = (ಸ್ಥಿರಾಂಕ int32_t *)dst->op_params;
    ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k0 = opts[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k1 = opts[2];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s0 = opts[3];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s1 = opts[4];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ p0 = opts[5];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ p1 = opts[6];

    ಅಕ್ಷರ       * cdata  = (ಅಕ್ಷರ       *) dst->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * cdataf = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) dstf->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಸ್ಥಿರಾಂಕ ದತ್ತಾಂಶ_ಕೊನೆ = cdata + ggml_nbytes(dst);

    GGML_ASSERT(params->ith == 0);
    memset(cdata, 0, ggml_nbytes(dst));

    ಸ್ಥಿರಾಂಕ int64_t px = src->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t py = src->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t pa = px * py;

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * splane = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src->ದತ್ತಾಂಶ;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ka = k0 * k1;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ offset0 = -p0;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ offset1 = -p1;

    ಆಗ (cdata < ದತ್ತಾಂಶ_ಕೊನೆ) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ oy = 0; oy < py; ++oy) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸ್ಥಿರಾಂಕ srow = splane + oy * px;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ox = 0; ox < px; ++ox) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ grad0 = srow[ox];

                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ix = offset0 + ox * s0;
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iy = offset1 + oy * s1;

                ಆದರೆ (op == GGML_OP_POOL_MAX) {
                    ದಶಮಾಂಶ maxval = -FLT_MAX;
                    ಪೂರ್ಣಾಂಕ kxmax = -1;
                    ಪೂರ್ಣಾಂಕ kymax = -1;

                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ky = 0; ky < k1; ++ky) {
                        ಆದರೆ (iy + ky < 0 || iy + ky >= dst->ne[1]) {
                            ಮುಂದುವರಿ;
                        }
                        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * drowf = (ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *)(cdataf + dst->nb[1] * (iy + ky));
                        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ kx = 0; kx < k0; ++kx) {
                            ಪೂರ್ಣಾಂಕ j = ix + kx;
                            ಆದರೆ (j < 0 || j >= dst->ne[0]) {
                                ಮುಂದುವರಿ;
                            }

                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ val = dst->type == GGML_TYPE_F32 ?
                                ((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) drowf)[j] : GGML_CPU_FP16_ಗಮ್ಯ_FP32(((ಸ್ಥಿರಾಂಕ ggml_fp16_t *) drowf)[j]);
                            ಆದರೆ (val <= maxval) {
                                ಮುಂದುವರಿ;
                            }

                            maxval = val;
                            kxmax = kx;
                            kymax = ky;
                        }
                    }

                    ಆದರೆ (kxmax == -1 || kymax == -1) {
                        ಮುಂದುವರಿ;
                    }

                    ಶೂನ್ಯ * drow = (ಶೂನ್ಯ *)(cdata + dst->nb[1] * (iy + kymax));
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ j = ix + kxmax;
                    ಆದರೆ (dst->type == GGML_TYPE_F32) {
                        ((ದಶಮಾಂಶ *) drow)[j] += grad0;
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        ((ggml_fp16_t *) drow)[j] = GGML_CPU_FP32_ಗಮ್ಯ_FP16(grad0 + GGML_CPU_FP16_ಗಮ್ಯ_FP32(((ಸ್ಥಿರಾಂಕ ggml_fp16_t *) drow)[j]));
                    }
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (op == GGML_OP_POOL_AVG) {
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ grad = grad0 / ka;

                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ky = 0; ky < k1; ++ky) {
                        ಆದರೆ (iy + ky < 0 || iy + ky >= dst->ne[1]) {
                            ಮುಂದುವರಿ;
                        }
                        ಶೂನ್ಯ * drow = (ಶೂನ್ಯ *)(cdata + dst->nb[1] * (iy + ky));
                        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ kx = 0; kx < k0; ++kx) {
                            ಪೂರ್ಣಾಂಕ j = ix + kx;
                            ಆದರೆ (j < 0 || j >= dst->ne[0]) {
                                ಮುಂದುವರಿ;
                            }

                            ಆದರೆ (dst->type == GGML_TYPE_F32) {
                                ((ದಶಮಾಂಶ *) drow)[j] += grad;
                            } ಇಲ್ಲದಿದ್ದರೆ {
                                ((ggml_fp16_t *) drow)[j] += GGML_CPU_FP32_ಗಮ್ಯ_FP16(grad);
                            }
                        }
                    }
                } ಇಲ್ಲದಿದ್ದರೆ {
                    GGML_ASSERT(ಸುಳ್ಳು);
                }
            }
        }

        cdata  += dst->nb[2];
        cdataf += dst->nb[2];
        splane += pa;
    }
}

// ggml_compute_forward_upscale

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_upscale_f32(
    ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
    ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    ದಶಮಾಂಶ sf0 = (ದಶಮಾಂಶ)ne0/src0->ne[0];
    ದಶಮಾಂಶ sf1 = (ದಶಮಾಂಶ)ne1/src0->ne[1];
    ದಶಮಾಂಶ sf2 = (ದಶಮಾಂಶ)ne2/src0->ne[2];
    ದಶಮಾಂಶ sf3 = (ದಶಮಾಂಶ)ne3/src0->ne[3];
    ದಶಮಾಂಶ pixel_ಸ್ಥಾನಾಂತರ = 0.5f;

    ಸ್ಥಿರಾಂಕ int32_t ಬಹುಲಕ_flags = ggml_get_op_params_i32(dst, 0);
    ಸ್ಥಿರಾಂಕ ggml_scale_ಬಹುಲಕ mode = (ggml_scale_ಬಹುಲಕ) (ಬಹುಲಕ_flags & 0xFF);

    ಆದರೆ (ಬಹುಲಕ_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        pixel_ಸ್ಥಾನಾಂತರ = 0.0f;
        sf0 = ne0 > 1 && ne00 > 1 ? (ದಶಮಾಂಶ)(ne0 - 1) / (ne00 - 1) : sf0;
        sf1 = ne1 > 1 && ne01 > 1 ? (ದಶಮಾಂಶ)(ne1 - 1) / (ne01 - 1) : sf1;
    }

    ಆದರೆ (mode == GGML_SCALE_ಬಹುಲಕ_ಹತ್ತಿರದ) {
        ಗಾಗಿ (int64_t i3 = 0; i3 < ne3; i3++) {
            ಸ್ಥಿರಾಂಕ int64_t i03 = i3 / sf3;
            ಗಾಗಿ (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                ಸ್ಥಿರಾಂಕ int64_t i02 = i2 / sf2;
                ಗಾಗಿ (int64_t i1 = 0; i1 < ne1; i1++) {
                    ಸ್ಥಿರಾಂಕ int64_t i01 = i1 / sf1;
                    ಗಾಗಿ (int64_t i0 = 0; i0 < ne0; i0++) {
                        ಸ್ಥಿರಾಂಕ int64_t i00 = i0 / sf0;

                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x = (ದಶಮಾಂಶ *)((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              ದಶಮಾಂಶ * y = (ದಶಮಾಂಶ *)((ಅಕ್ಷರ *)  dst->ದತ್ತಾಂಶ +  i0*nb0  +  i1*nb1  +  i2*nb2  +  i3*nb3);

                        *y = *x;
                    }
                }
            }
        }
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (mode == GGML_SCALE_ಬಹುಲಕ_BILINEAR && (ಬಹುಲಕ_flags & GGML_SCALE_FLAG_ANTIALIAS)) {
        // Similar to F.interpolate(..., mode="bilinear", align_corners=False, antialias=True)
        // https://github.com/pytorch/pytorch/blob/8871ff29b743948d1225389d5b7068f37b22750b/aten/src/ATen/native/cpu/UpSampleKernel.cpp
        ಸ್ವಯಂಚಾಲಿತ triangle_filter = [](ದಶಮಾಂಶ x) -> ದಶಮಾಂಶ {
            ಹಿಂತಿರುಗಿ std::max(1.0f - fabsf(x), 0.0f);
        };

        // support and invscale, minimum 1 pixel for bilinear
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ support1  = std::max(1.0f, 1.0f / sf1);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ invscale1 = 1.0f / support1;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ support0  = std::max(1.0f, 1.0f / sf0);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ invscale0 = 1.0f / support0;

        ಗಾಗಿ (int64_t i3 = 0; i3 < ne3; i3++) {
            ಸ್ಥಿರಾಂಕ int64_t i03 = i3 / sf3;
            ಗಾಗಿ (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                ಸ್ಥಿರಾಂಕ int64_t i02 = i2 / sf2;
                ಗಾಗಿ (int64_t i1 = 0; i1 < ne1; i1++) {
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ y = ((ದಶಮಾಂಶ) i1 + pixel_ಸ್ಥಾನಾಂತರ) / sf1;
                    ಗಾಗಿ (int64_t i0 = 0; i0 < ne0; i0++) {
                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x = ((ದಶಮಾಂಶ) i0 + pixel_ಸ್ಥಾನಾಂತರ) / sf0;

                        // the range of source pixels that contribute
                        ಸ್ಥಿರಾಂಕ int64_t x_min = std::max<int64_t>(x - support0 + pixel_ಸ್ಥಾನಾಂತರ, 0);
                        ಸ್ಥಿರಾಂಕ int64_t x_max = std::min<int64_t>(x + support0 + pixel_ಸ್ಥಾನಾಂತರ, ne00);
                        ಸ್ಥಿರಾಂಕ int64_t y_min = std::max<int64_t>(y - support1 + pixel_ಸ್ಥಾನಾಂತರ, 0);
                        ಸ್ಥಿರಾಂಕ int64_t y_max = std::min<int64_t>(y + support1 + pixel_ಸ್ಥಾನಾಂತರ, ne01);

                        // bilinear filter with antialiasing
                        ದಶಮಾಂಶ val = 0.0f;
                        ದಶಮಾಂಶ ಒಟ್ಟು_ತೂಕ = 0.0f;

                        ಗಾಗಿ (int64_t sy = y_min; sy < y_max; sy++) {
                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ತೂಕ_y = triangle_filter((sy - y + pixel_ಸ್ಥಾನಾಂತರ) * invscale1);

                            ಗಾಗಿ (int64_t sx = x_min; sx < x_max; sx++) {
                                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ತೂಕ_x = triangle_filter((sx - x + pixel_ಸ್ಥಾನಾಂತರ) * invscale0);
                                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ತೂಕ = ತೂಕ_x * ತೂಕ_y;

                                ಆದರೆ (ತೂಕ <= 0.0f) {
                                    ಮುಂದುವರಿ;
                                }

                                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ pixel = *(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)src0->ದತ್ತಾಂಶ + sx*nb00 + sy*nb01 + i02*nb02 + i03*nb03);
                                val += pixel * ತೂಕ;
                                ಒಟ್ಟು_ತೂಕ += ತೂಕ;
                            }
                        }

                        ಆದರೆ (ಒಟ್ಟು_ತೂಕ > 0.0f) {
                            val /= ಒಟ್ಟು_ತೂಕ;
                        }

                        ದಶಮಾಂಶ * dst_ptr = (ದಶಮಾಂಶ *)((ಅಕ್ಷರ *)dst->ದತ್ತಾಂಶ + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);
                        *dst_ptr = val;
                    }
                }
            }
        }
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (mode == GGML_SCALE_ಬಹುಲಕ_BILINEAR) {
        ಗಾಗಿ (int64_t i3 = 0; i3 < ne3; i3++) {
            ಸ್ಥಿರಾಂಕ int64_t i03 = i3 / sf3;
            ಗಾಗಿ (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                ಸ್ಥಿರಾಂಕ int64_t i02 = i2 / sf2;
                ಗಾಗಿ (int64_t i1 = 0; i1 < ne1; i1++) {
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ y = ((ದಶಮಾಂಶ)i1 + pixel_ಸ್ಥಾನಾಂತರ) / sf1 - pixel_ಸ್ಥಾನಾಂತರ;
                    int64_t y0 = (int64_t)floorf(y);
                    int64_t y1 = y0 + 1;

                    y0 = std::max(int64_t(0), std::min(y0, ne01 - 1));
                    y1 = std::max(int64_t(0), std::min(y1, ne01 - 1));

                    ದಶಮಾಂಶ dy = y - (ದಶಮಾಂಶ)y0;
                    dy = std::max(0.0f, std::min(dy, 1.0f));

                    ಗಾಗಿ (int64_t i0 = 0; i0 < ne0; i0++) {
                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x = ((ದಶಮಾಂಶ)i0 + pixel_ಸ್ಥಾನಾಂತರ) / sf0 - pixel_ಸ್ಥಾನಾಂತರ;
                        int64_t x0 = (int64_t)floorf(x);
                        int64_t x1 = x0 + 1;

                        x0 = std::max(int64_t(0), std::min(x0, ne00 - 1));
                        x1 = std::max(int64_t(0), std::min(x1, ne00 - 1));

                        ದಶಮಾಂಶ dx = x - (ದಶಮಾಂಶ)x0;
                        dx = std::max(0.0f, std::min(dx, 1.0f));

                        // fetch the four surrounding pixel values and interpolate
                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ a = *(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)src0->ದತ್ತಾಂಶ + x0*nb00 + y0*nb01 + i02*nb02 + i03*nb03);
                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ b = *(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)src0->ದತ್ತಾಂಶ + x1*nb00 + y0*nb01 + i02*nb02 + i03*nb03);
                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ c = *(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)src0->ದತ್ತಾಂಶ + x0*nb00 + y1*nb01 + i02*nb02 + i03*nb03);
                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = *(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)src0->ದತ್ತಾಂಶ + x1*nb00 + y1*nb01 + i02*nb02 + i03*nb03);

                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ val = a*(1 - dx)*(1 - dy) + b*dx*(1 - dy) + c*(1 - dx)*dy + d*dx*dy;

                        ದಶಮಾಂಶ * y_dst = (ದಶಮಾಂಶ *)((ಅಕ್ಷರ *)dst->ದತ್ತಾಂಶ + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);
                        *y_dst = val;
                    }
                }
            }
        }
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (mode == GGML_SCALE_ಬಹುಲಕ_BICUBIC) {
        // https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ a = -0.75f; // use alpha = -0.75 (same as PyTorch)
        ಸ್ವಯಂಚಾಲಿತ weight1 = [a](ದಶಮಾಂಶ x) { ಹಿಂತಿರುಗಿ ((a + 2) * x - (a + 3)) * x * x + 1; };
        ಸ್ವಯಂಚಾಲಿತ weight2 = [a](ದಶಮಾಂಶ x) { ಹಿಂತಿರುಗಿ ((a * x - 5 * a) * x + 8 * a) * x - 4 * a; };
        ಸ್ವಯಂಚಾಲಿತ bicubic = [=](ದಶಮಾಂಶ p0, ದಶಮಾಂಶ p1, ದಶಮಾಂಶ p2, ದಶಮಾಂಶ p3, ದಶಮಾಂಶ x) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ w0 = weight2(x + 1);
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ w1 = weight1(x + 0);
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ w2 = weight1(1 - x);
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ w3 = weight2(2 - x);
            ಹಿಂತಿರುಗಿ p0*w0 + p1*w1 + p2*w2 + p3*w3;
        };

        ಗಾಗಿ (int64_t i3 = 0; i3 < ne3; i3++) {
            ಸ್ಥಿರಾಂಕ int64_t i03 = i3 / sf3;
            ಗಾಗಿ (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                ಸ್ಥಿರಾಂಕ int64_t i02 = i2 / sf2;
                ಗಾಗಿ (int64_t i1 = 0; i1 < ne1; i1++) {
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ y = ((ದಶಮಾಂಶ)i1 + pixel_ಸ್ಥಾನಾಂತರ) / sf1 - pixel_ಸ್ಥಾನಾಂತರ;
                    ಸ್ಥಿರಾಂಕ int64_t y0 = (int64_t)floorf(y);
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dy = y - (ದಶಮಾಂಶ)y0;

                    ಗಾಗಿ (int64_t i0 = 0; i0 < ne0; i0++) {
                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x = ((ದಶಮಾಂಶ)i0 + pixel_ಸ್ಥಾನಾಂತರ) / sf0 - pixel_ಸ್ಥಾನಾಂತರ;
                        ಸ್ಥಿರಾಂಕ int64_t x0 = (int64_t)floorf(x);
                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dx = x - (ದಶಮಾಂಶ)x0;

                        ಸ್ವಯಂಚಾಲಿತ p = [=](int64_t x_off, int64_t y_off) -> ದಶಮಾಂಶ {
                            int64_t i00 = std::max(int64_t(0), std::min(x0 + x_off, ne00 - 1));
                            int64_t i01 = std::max(int64_t(0), std::min(y0 + y_off, ne01 - 1));
                            ಹಿಂತಿರುಗಿ *(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)src0->ದತ್ತಾಂಶ + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                        };

                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ val = bicubic(
                            bicubic(p(-1,-1), p(0,-1), p(1,-1), p(2,-1), dx),
                            bicubic(p(-1, 0), p(0, 0), p(1, 0), p(2, 0), dx),
                            bicubic(p(-1, 1), p(0, 1), p(1, 1), p(2, 1), dx),
                            bicubic(p(-1, 2), p(0, 2), p(1, 2), p(2, 2), dx), dy);

                        ದಶಮಾಂಶ * y_dst = (ದಶಮಾಂಶ *)((ಅಕ್ಷರ *)dst->ದತ್ತಾಂಶ + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);
                        *y_dst = val;
                    }
                }
            }
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_ABORT("unsupported upscale mode");
    }
}

ಶೂನ್ಯ ggml_compute_forward_upscale(
    ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
    ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_upscale_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}


// ggml_compute_forward_pad

ಮಾದರಿ<ತಾರ್ಕಿಕ circular_t>
ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_pad_f32(
    ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
          ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಪ್ರತಿಜ್ಞಾ(dst->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    ದಶಮಾಂಶ * dst_ptr = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ int32_t lp0 = ggml_get_op_params_i32(dst, 0);
    ಸ್ಥಿರಾಂಕ int32_t rp0 = ggml_get_op_params_i32(dst, 1);
    ಸ್ಥಿರಾಂಕ int32_t lp1 = ggml_get_op_params_i32(dst, 2);
    ಸ್ಥಿರಾಂಕ int32_t rp1 = ggml_get_op_params_i32(dst, 3);
    ಸ್ಥಿರಾಂಕ int32_t lp2 = ggml_get_op_params_i32(dst, 4);
    ಸ್ಥಿರಾಂಕ int32_t rp2 = ggml_get_op_params_i32(dst, 5);
    ಸ್ಥಿರಾಂಕ int32_t lp3 = ggml_get_op_params_i32(dst, 6);
    ಸ್ಥಿರಾಂಕ int32_t rp3 = ggml_get_op_params_i32(dst, 7);

    ಸ್ಥಿರಾಂಕ int64_t nrows = ne1 * ne2 * ne3;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ src_ಅಡ್ಡಸಾಲು_contig = src0->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ);

    ಗಾಗಿ (int64_t ಅಡ್ಡಸಾಲು = ith; ಅಡ್ಡಸಾಲು < nrows; ಅಡ್ಡಸಾಲು += nth) {
        ಸ್ಥಿರಾಂಕ int64_t i1 = ಅಡ್ಡಸಾಲು % ne1;
        ಸ್ಥಿರಾಂಕ int64_t i2 = (ಅಡ್ಡಸಾಲು / ne1) % ne2;
        ಸ್ಥಿರಾಂಕ int64_t i3 = ಅಡ್ಡಸಾಲು / (ne1 * ne2);

        ದಶಮಾಂಶ * dst_ಅಡ್ಡಸಾಲು = dst_ptr + ಅಡ್ಡಸಾಲು * ne0;

        ಆದರೆ constexpr (circular_t) {
            // circular padding wraps around on a torus
            ಸ್ಥಿರಾಂಕ int64_t src_i1 = ggml_wrap_around(i1 - lp1, ne01);
            ಸ್ಥಿರಾಂಕ int64_t src_i2 = ggml_wrap_around(i2 - lp2, ne02);
            ಸ್ಥಿರಾಂಕ int64_t src_i3 = ggml_wrap_around(i3 - lp3, ne03);

            ಗಾಗಿ (int64_t i0 = 0; i0 < ne0; ++i0) {
                ಸ್ಥಿರಾಂಕ int64_t src_i0 = ggml_wrap_around(i0 - lp0, ne00);
                dst_ಅಡ್ಡಸಾಲು[i0] = *(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + src_i3*nb03 + src_i2*nb02 + src_i1*nb01 + src_i0*nb00);
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ interior = (i1 >= lp1 && i1 < ne1 - rp1)
                               && (i2 >= lp2 && i2 < ne2 - rp2)
                               && (i3 >= lp3 && i3 < ne3 - rp3);
            ಆದರೆ (!interior) {
                memset(dst_ಅಡ್ಡಸಾಲು, 0, ne0 * ಗಾತ್ರ(ದಶಮಾಂಶ));
                ಮುಂದುವರಿ;
            }

            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * src_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + (i3 - lp3)*nb03 + (i2 - lp2)*nb02 + (i1 - lp1)*nb01;

            ಆದರೆ (src_ಅಡ್ಡಸಾಲು_contig && lp0 >= 0 && rp0 >= 0) {
                memset(dst_ಅಡ್ಡಸಾಲು, 0, lp0 * ಗಾತ್ರ(ದಶಮಾಂಶ));
                ಸ್ಮರಣೆ_ನಕಲು(dst_ಅಡ್ಡಸಾಲು + lp0, src_ಅಡ್ಡಸಾಲು, ne00 * ಗಾತ್ರ(ದಶಮಾಂಶ));
                memset(dst_ಅಡ್ಡಸಾಲು + lp0 + ne00, 0, rp0 * ಗಾತ್ರ(ದಶಮಾಂಶ));
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಗಾಗಿ (int64_t i0 = 0; i0 < ne0; ++i0) {
                    ಆದರೆ (i0 >= lp0 && i0 < ne0 - rp0) {
                        dst_ಅಡ್ಡಸಾಲು[i0] = *(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)(src_ಅಡ್ಡಸಾಲು + (i0 - lp0)*nb00);
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        dst_ಅಡ್ಡಸಾಲು[i0] = 0.0f;
                    }
                }
            }
        }
    }
}


ಶೂನ್ಯ ggml_compute_forward_pad(
    ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
    ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ circular = (ತಾರ್ಕಿಕ) ggml_get_op_params_i32(dst, 8);
    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ಆದರೆ (circular) {
                    ggml_compute_forward_pad_f32<ನಿಜ>(params, dst);
                } ಇಲ್ಲದಿದ್ದರೆ {
                    ggml_compute_forward_pad_f32<ಸುಳ್ಳು>(params, dst);
                }
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_pad_reflect_1d

ಶೂನ್ಯ ggml_compute_forward_pad_reflect_1d(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
              ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ int32_t * opts = (ಸ್ಥಿರಾಂಕ int32_t *) dst->op_params;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ p0 = opts[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ p1 = opts[1];

    GGML_TENSOR_UNARY_OP_LOCALS

    ಗಾಗಿ (int64_t i3 = 0; i3 < ne3; i3++) {
        ಗಾಗಿ (int64_t i2 = 0; i2 < ne2; i2++) {
            ಗಾಗಿ (int64_t i1 = ith; i1 < ne1; i1 += nth) {
                ದಶಮಾಂಶ * ಎಡ  = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i3*nb3 + i2*nb2 + i1*nb1 +         p0*nb0);
                ದಶಮಾಂಶ * ಬಲ = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i3*nb3 + i2*nb2 + i1*nb1 + (ne0-p1-1)*nb0);

                ggml_vec_cpy_f32(ne00, ಎಡ, (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i3*nb03 + i2*nb02 + i1*nb01));

                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 1; i0 <= p0; i0++) { ಎಡ[-i0] = ಎಡ[i0];   }
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 1; i0 <= p1; i0++) { ಬಲ[i0] = ಬಲ[-i0]; }
            }
        }
    }
}

// ggml_compute_forward_roll

ಸ್ಥಿರ int64_t ggml_wrap_ಸೂಚ್ಯಂಕ(int64_t i, int64_t ne) {
    ಆದರೆ (i < 0) {
        ಹಿಂತಿರುಗಿ i + ne;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (i >= ne) {
        ಹಿಂತಿರುಗಿ i - ne;
    }
    ಹಿಂತಿರುಗಿ i;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_roll_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src_ದತ್ತಾಂಶ = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src0->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * dst_ದತ್ತಾಂಶ = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;

    GGML_TENSOR_UNARY_OP_LOCALS

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s0 = ggml_get_op_params_i32(dst, 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s1 = ggml_get_op_params_i32(dst, 1);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s2 = ggml_get_op_params_i32(dst, 2);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s3 = ggml_get_op_params_i32(dst, 3);

    ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು = ne1 * ne2 * ne3;
    ಸ್ಥಿರಾಂಕ int64_t ಪ್ರತಿ_thread = (ಒಟ್ಟು + params->nth) / params->nth;
    ಸ್ಥಿರಾಂಕ int64_t start = params->ith * ಪ್ರತಿ_thread;
    ಸ್ಥಿರಾಂಕ int64_t end   = std::min(start + ಪ್ರತಿ_thread, ಒಟ್ಟು);

    ಗಾಗಿ (int64_t i = start; i < end; ++i) {
        ಸ್ಥಿರಾಂಕ int64_t i1 = i % ne1;
        ಸ್ಥಿರಾಂಕ int64_t i2 = (i / ne1) % ne2;
        ಸ್ಥಿರಾಂಕ int64_t i3 = i / (ne2 * ne1);
        ದಶಮಾಂಶ * dst_ಅಡ್ಡಸಾಲು = dst_ದತ್ತಾಂಶ + (i3*nb3 + i2*nb2 + i1*nb1) / ಗಾತ್ರ(ದಶಮಾಂಶ);

        ಸ್ಥಿರಾಂಕ int64_t i01 = ggml_wrap_ಸೂಚ್ಯಂಕ(i1 - s1, ne01);
        ಸ್ಥಿರಾಂಕ int64_t i02 = ggml_wrap_ಸೂಚ್ಯಂಕ(i2 - s2, ne02);
        ಸ್ಥಿರಾಂಕ int64_t i03 = ggml_wrap_ಸೂಚ್ಯಂಕ(i3 - s3, ne03);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src_ಅಡ್ಡಸಾಲು = src_ದತ್ತಾಂಶ + (i03*nb03 + i02*nb02 + i01*nb01) / ಗಾತ್ರ(ದಶಮಾಂಶ);

        ಸ್ಥಿರಾಂಕ int64_t s = ggml_wrap_ಸೂಚ್ಯಂಕ(-s0, ne00);
        ಸ್ಥಿರಾಂಕ int64_t n = ne00 - s;
        ggml_vec_cpy_f32(n, dst_ಅಡ್ಡಸಾಲು,     src_ಅಡ್ಡಸಾಲು + s);
        ggml_vec_cpy_f32(s, dst_ಅಡ್ಡಸಾಲು + n, src_ಅಡ್ಡಸಾಲು);
    }
}

ಶೂನ್ಯ ggml_compute_forward_roll(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_roll_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_arange

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_arange_f32(
    ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
    ggml_tensor * dst) {

    GGML_ASSERT(dst->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ start = ggml_get_op_params_f32(dst, 0);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ stop  = ggml_get_op_params_f32(dst, 1);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಹೆಜ್ಜೆ  = ggml_get_op_params_f32(dst, 2);

    ಸ್ಥಿರಾಂಕ int64_t ಹೆಜ್ಜೆಗಳು = (int64_t) ceilf((stop - start) / ಹೆಜ್ಜೆ);

    GGML_ASSERT(ggml_nelements(dst) == ಹೆಜ್ಜೆಗಳು);

    ಗಾಗಿ (int64_t i = ith; i < ಹೆಜ್ಜೆಗಳು; i+= nth) {
        ದಶಮಾಂಶ value = start + ಹೆಜ್ಜೆ * i;
        ((ದಶಮಾಂಶ *)dst->ದತ್ತಾಂಶ)[i] = value;
    }
}

ಶೂನ್ಯ ggml_compute_forward_arange(
    ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
    ggml_tensor * dst) {
    ಆಯ್ಕೆ (dst->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_arange_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_timestep_embedding_f32(
    ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
    ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dim = ggml_get_op_params_i32(dst, 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ max_period = ggml_get_op_params_i32(dst, 1);

    ಪೂರ್ಣಾಂಕ half = dim / 2;

    ಗಾಗಿ (int64_t i = 0; i < ne00; i++) {
        ದಶಮಾಂಶ * embed_ದತ್ತಾಂಶ = (ದಶಮಾಂಶ *)((ಅಕ್ಷರ *)  dst->ದತ್ತಾಂಶ +  i*nb1);
        ಗಾಗಿ (int64_t j = ith; j < half; j += nth) {
            ದಶಮಾಂಶ timestep = ((ದಶಮಾಂಶ *)src0->ದತ್ತಾಂಶ)[i];
            ದಶಮಾಂಶ freq = (ದಶಮಾಂಶ)expf(-logf(max_period) * j / half);
            ದಶಮಾಂಶ arg = timestep * freq;
            embed_ದತ್ತಾಂಶ[j] = cosf(arg);
            embed_ದತ್ತಾಂಶ[j + half] = sinf(arg);
        }
        ಆದರೆ (dim % 2 != 0 && ith == 0) {
            embed_ದತ್ತಾಂಶ[2 * half] = 0.f;
        }
    }
}

ಶೂನ್ಯ ggml_compute_forward_timestep_embedding(
    ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
    ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_timestep_embedding_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_argsort

ಮಾದರಿ<ಎಣಿಕೆ ggml_ವಿಂಗಡಿಸು_ಕ್ರಮ ಕ್ರಮ>
ರಚನೆ cmp_argsort {
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ದತ್ತಾಂಶ;
    ತಾರ್ಕಿಕ ಕ್ರಿಯಾಕಾರಕ()(int32_t a, int32_t b) ಸ್ಥಿರಾಂಕ {
        ಆದರೆ constexpr (ಕ್ರಮ == GGML_ವಿಂಗಡಿಸು_ಕ್ರಮ_ASC) {
            ಹಿಂತಿರುಗಿ ದತ್ತಾಂಶ[a] < ದತ್ತಾಂಶ[b];
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಹಿಂತಿರುಗಿ ದತ್ತಾಂಶ[a] > ದತ್ತಾಂಶ[b];
        }
    }
};

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_argsort_f32(
    ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
    ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(nb0 == ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ int64_t nr = ggml_nrows(src0);

    ggml_ವಿಂಗಡಿಸು_ಕ್ರಮ ಕ್ರಮ = (ggml_ವಿಂಗಡಿಸು_ಕ್ರಮ) ggml_get_op_params_i32(dst, 0);

    ಗಾಗಿ (int64_t i = ith; i < nr; i += nth) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src_ದತ್ತಾಂಶ = (ದಶಮಾಂಶ *)((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i*nb01);

        int32_t * dst_ದತ್ತಾಂಶ = (int32_t *)((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i*nb1);

        ಗಾಗಿ (int64_t j = 0; j < ne0; j++) {
            dst_ದತ್ತಾಂಶ[j] = j;
        }

        ಆಯ್ಕೆ (ಕ್ರಮ) {
            ಸ್ಥಿತಿ GGML_ವಿಂಗಡಿಸು_ಕ್ರಮ_ASC:
                std::sort(dst_ದತ್ತಾಂಶ, dst_ದತ್ತಾಂಶ + ne0, cmp_argsort<GGML_ವಿಂಗಡಿಸು_ಕ್ರಮ_ASC>{src_ದತ್ತಾಂಶ});
                ನಿಲ್ಲಿಸು;

            ಸ್ಥಿತಿ GGML_ವಿಂಗಡಿಸು_ಕ್ರಮ_DESC:
                std::sort(dst_ದತ್ತಾಂಶ, dst_ದತ್ತಾಂಶ + ne0, cmp_argsort<GGML_ವಿಂಗಡಿಸು_ಕ್ರಮ_DESC>{src_ದತ್ತಾಂಶ});
                ನಿಲ್ಲಿಸು;

            ಪೂರ್ವನಿಯೋಜಿತ:
                GGML_ABORT("invalid sort order");
        }
    }
}

ಶೂನ್ಯ ggml_compute_forward_argsort(
    ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
    ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_argsort_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_top_k

ರಚನೆ cmp_ತುದಿ_k {
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ದತ್ತಾಂಶ;
    ತಾರ್ಕಿಕ ಕ್ರಿಯಾಕಾರಕ()(int32_t a, int32_t b) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ ದತ್ತಾಂಶ[a] > ದತ್ತಾಂಶ[b];
    }
};

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_ತುದಿ_k_f32(
    ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
    ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(nb0 == ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ int64_t nr = ggml_nrows(src0);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ತುದಿ_k = ne0;

    int32_t * tmp = (int32_t *) params->wdata + (ne00 + CACHE_ಸಾಲು_ಗಾತ್ರ_F32) * ith;

    ಗಾಗಿ (int64_t i = ith; i < nr; i += nth) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src_ದತ್ತಾಂಶ = (ದಶಮಾಂಶ *)((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i*nb01);

        ಗಾಗಿ (int64_t j = 0; j < ne00; j++) {
            tmp[j] = j;
        }

        std::partial_sort(tmp, tmp + ತುದಿ_k, tmp + ne00, cmp_ತುದಿ_k{src_ದತ್ತಾಂಶ});

        int32_t * dst_ದತ್ತಾಂಶ = (int32_t *)((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i*nb1);

        std::copy(tmp, tmp + ತುದಿ_k, dst_ದತ್ತಾಂಶ);

        // emphasize that the order is not important
        ಆದರೆ (ತುದಿ_k > 1) {
            std::swap(dst_ದತ್ತಾಂಶ[0], dst_ದತ್ತಾಂಶ[1]);
        }
    }
}

ಶೂನ್ಯ ggml_compute_forward_ತುದಿ_k(
    ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
    ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_ತುದಿ_k_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_flash_attn_ext_f16_one_chunk(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst,
        ಪೂರ್ಣಾಂಕ ir0, ಪೂರ್ಣಾಂಕ ir1,
        int64_t ic_ಆರಂಭ, int64_t ic_ಕೊನೆ,
        ದಶಮಾಂಶ * partials, int64_t partial_stride) {

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ write_partials = (partials != ಶೂನ್ಯಸೂಚಕ);
    ಸ್ಥಿರಾಂಕ ggml_tensor * q     = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * k     = dst->src[1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * v     = dst->src[2];
    ಸ್ಥಿರಾಂಕ ggml_tensor * ಮುಖವಾಡ  = dst->src[3];
    ಸ್ಥಿರಾಂಕ ggml_tensor * sinks = dst->src[4];

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    ಸ್ಥಿರಾಂಕ int64_t DK = nek0;
    ಸ್ಥಿರಾಂಕ int64_t DV = nev0;
    ಸ್ಥಿರಾಂಕ int64_t N  = neq1;

    GGML_ASSERT(ne0 == DV);
    GGML_ASSERT(ne2 == N);

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_ಗಾತ್ರ(q->type));
    GGML_ASSERT(nbk0 == ggml_type_ಗಾತ್ರ(k->type));
    GGML_ASSERT(nbv0 == ggml_type_ಗಾತ್ರ(v->type));

    GGML_ASSERT(neq0 == DK);
    GGML_ASSERT(nek0 == DK);
    GGML_ASSERT(nev0 == DV);

    GGML_ASSERT(neq1 == N);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == ಗಾತ್ರ(ದಶಮಾಂಶ));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    ಸ್ಥಿರಾಂಕ int64_t rk2 = neq2/nek2;
    ಸ್ಥಿರಾಂಕ int64_t rk3 = neq3/nek3;

    ಸ್ಥಿರಾಂಕ int64_t rv2 = neq2/nev2;
    ಸ್ಥಿರಾಂಕ int64_t rv3 = neq3/nev3;

    // parallelize by q rows using ggml_vec_dot_f32

    ದಶಮಾಂಶ scale         = 1.0f;
    ದಶಮಾಂಶ max_bias      = 0.0f;
    ದಶಮಾಂಶ logit_softcap = 0.0f;

    ಸ್ಮರಣೆ_ನಕಲು(&scale,         (ದಶಮಾಂಶ *) dst->op_params + 0, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&max_bias,      (ದಶಮಾಂಶ *) dst->op_params + 1, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&logit_softcap, (ದಶಮಾಂಶ *) dst->op_params + 2, ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಆದರೆ (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    ಸ್ಥಿರಾಂಕ uint32_t n_ತಲೆ      = neq2;
    ಸ್ಥಿರಾಂಕ uint32_t n_ತಲೆ_log2 = 1u << (uint32_t) floor(log2(n_ತಲೆ));

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ m0 = powf(2.0f, -(max_bias       ) / n_ತಲೆ_log2);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ m1 = powf(2.0f, -(max_bias / 2.0f) / n_ತಲೆ_log2);

    ggml_type         ಸ್ಥಿರಾಂಕ k_vec_dot_type = ggml_get_type_traits_cpu(k->type)->vec_dot_type;
    ggml_from_float_t ಸ್ಥಿರಾಂಕ q_ಗಮ್ಯ_vec_dot   = ggml_get_type_traits_cpu(k_vec_dot_type)->from_float;
    ggml_vec_dot_t    ಸ್ಥಿರಾಂಕ kq_vec_dot     = ggml_get_type_traits_cpu(k->type)->vec_dot;
    ggml_ಗಮ್ಯ_float_t   ಸ್ಥಿರಾಂಕ v_ಗಮ್ಯ_float     = ggml_get_type_traits(v->type)->ಗಮ್ಯ_float;

    GGML_ASSERT((                            q_ಗಮ್ಯ_vec_dot) && "fattn: unsupported K-type");
    GGML_ASSERT((v->type == GGML_TYPE_F32 || v_ಗಮ್ಯ_float  ) && "fattn: unsupported V-type");

    ಪೂರ್ಣಾಂಕ ith = params->ith;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ir = ir0; ir < ir1; ++ir) {
        // q indices
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iq3 = ir/(neq2*neq1);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iq2 = (ir - iq3*neq2*neq1)/neq1;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iq1 = (ir - iq3*neq2*neq1 - iq2*neq1);

        ಸ್ಥಿರಾಂಕ uint32_t h = iq2; // head index
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಇಳಿಜಾರು = (max_bias > 0.0f) ? h < n_ತಲೆ_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_ತಲೆ_log2) + 1) : 1.0f;

        ದಶಮಾಂಶ S = 0.0f;      //ಮೊತ್ತ
        ದಶಮಾಂಶ M = -INFINITY; // maximum KQ value

        ದಶಮಾಂಶ       * VKQ32 = (ದಶಮಾಂಶ       *) params->wdata + ith*(1*DK + 2*DV + CACHE_ಸಾಲು_ಗಾತ್ರ_F32); // FP32 VKQ accumulator
        ದಶಮಾಂಶ       * V32   =                 (VKQ32 + 1*DV); // (temporary) FP32 V buffer
        ggml_fp16_t * VKQ16 = (ggml_fp16_t *) (VKQ32 + 1*DV); // (temporary) FP16 VKQ accumulator
        ggml_fp16_t * Q_q   = (ggml_fp16_t *) (VKQ32 + 2*DV); // (temporary) buffer for Q converted to quantized/FP16

        ಆದರೆ (v->type == GGML_TYPE_F16) {
            memset(VKQ16, 0, DV*ಗಾತ್ರ(ggml_fp16_t));
        } ಇಲ್ಲದಿದ್ದರೆ {
            memset(VKQ32, 0, DV*ಗಾತ್ರ(ದಶಮಾಂಶ));
        }

        ಸ್ಥಿರಾಂಕ ggml_fp16_t * mp = ಮುಖವಾಡ ? (ggml_fp16_t *)((ಅಕ್ಷರ *) ಮುಖವಾಡ->ದತ್ತಾಂಶ + iq1*ಮುಖವಾಡ->nb[1] + (iq2%ಮುಖವಾಡ->ne[2])*ಮುಖವಾಡ->nb[2] + (iq3%ಮುಖವಾಡ->ne[3])*ಮುಖವಾಡ->nb[3]) : ಖಾಲಿ;

        // k indices
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ik3 = iq3 / rk3;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ik2 = iq2 / rk2;

        // v indices
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iv3 = iq3 / rv3;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iv2 = iq2 / rv2;

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * pq = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) q->ದತ್ತಾಂಶ + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3));
        q_ಗಮ್ಯ_vec_dot(pq, Q_q, DK);

        // online softmax / attention
        // loop over n_kv and n_head_kv
        // ref: https://arxiv.org/pdf/2112.05682.pdf

        ಗಾಗಿ (int64_t ic = ic_ಆರಂಭ; ic < ic_ಕೊನೆ; ++ic) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ mv = mp ? ಇಳಿಜಾರು*GGML_CPU_FP16_ಗಮ್ಯ_FP32(mp[ic]) : 0.0f;
            ಆದರೆ (mv == -INFINITY) {
                ಮುಂದುವರಿ;
            }

            ದಶಮಾಂಶ s; // KQ value

            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * k_ದತ್ತಾಂಶ = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) k->ದತ್ತಾಂಶ + ( ic*nbk1 + ik2*nbk2 + ik3*nbk3);
            kq_vec_dot(DK, &s, 0, k_ದತ್ತಾಂಶ, 0, Q_q, 0, 1);

            s = s*scale; // scale KQ value

            ಆದರೆ (logit_softcap != 0.0f) {
                s = logit_softcap*tanhf(s);
            }

            s += mv; // apply mask

            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ Mold = M;

            ದಶಮಾಂಶ ms = 1.0f; // upon new higher max val, scale VKQ and KQ sum with this value
            ದಶಮಾಂಶ vs = 1.0f; // post-softmax KQ value, expf(s - M)

            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * v_ದತ್ತಾಂಶ = ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) v->ದತ್ತಾಂಶ + (ic*nbv1 + iv2*nbv2 + iv3*nbv3));

            ಆದರೆ (v->type == GGML_TYPE_F16) {
                ಆದರೆ (s > M) {
                    // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
                    M = s;
                    ms = expf(Mold - M);

                    // V = V*expf(Mold - M)
                    ggml_vec_scale_f16(DV, VKQ16, ms);
                } ಇಲ್ಲದಿದ್ದರೆ {
                    // no new maximum, ms == 1.0f, vs != 1.0f
                    vs = expf(s - M);
                }

                // V += v*expf(s - M)
                ggml_vec_mad_f16(DV, VKQ16, (ಸ್ಥಿರಾಂಕ ggml_fp16_t *) v_ದತ್ತಾಂಶ, vs);
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಆದರೆ (s > M) {
                    // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
                    M = s;
                    ms = expf(Mold - M);

                    // V = V*expf(Mold - M)
                    ggml_vec_scale_f32(DV, VKQ32, ms);
                } ಇಲ್ಲದಿದ್ದರೆ {
                    // no new maximum, ms == 1.0f, vs != 1.0f
                    vs = expf(s - M);
                }

                // V += v*expf(s - M)
                ಆದರೆ (v_ಗಮ್ಯ_float) {
                    v_ಗಮ್ಯ_float(v_ದತ್ತಾಂಶ, V32, DV);
                    ggml_vec_mad_f32(DV, VKQ32, V32, vs);
                } ಇಲ್ಲದಿದ್ದರೆ {
                    // V is F32
                    ggml_vec_mad_f32(DV, VKQ32, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) v_ದತ್ತಾಂಶ, vs);
                }
            }

            S = S*ms + vs; // scale and increment sum with partial sum
        }

        ಆದರೆ (v->type == GGML_TYPE_F16) {
            ಗಾಗಿ (int64_t d = 0; d < DV; ++d) {
                VKQ32[d] = GGML_CPU_FP16_ಗಮ್ಯ_FP32(VKQ16[d]);
            }
        }

        // sinks - apply only on the first kv-chunk
        ಆದರೆ (sinks && ic_ಆರಂಭ == 0) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ s = ((ದಶಮಾಂಶ *)((ಅಕ್ಷರ *) sinks->ದತ್ತಾಂಶ))[h];

            ದಶಮಾಂಶ ms = 1.0f;
            ದಶಮಾಂಶ vs = 1.0f;

            ಆದರೆ (s > M) {
                ms = expf(M - s);
                M = s;
                ggml_vec_scale_f32(DV, VKQ32, ms);
            } ಇಲ್ಲದಿದ್ದರೆ {
                vs = expf(s - M);
            }

            S = S*ms + vs;
        }

        ಆದರೆ (write_partials) {
            // Write M, S, VKQ to partials for later reduction
            // partials layout: [M, S, VKQ[DV]] per query head
            ದಶಮಾಂಶ * partial = partials + ir * partial_stride;
            partial[0] = M;
            partial[1] = S;
            ಸ್ಮರಣೆ_ನಕಲು(partial + 2, VKQ32, DV * ಗಾತ್ರ(ದಶಮಾಂಶ));
        } ಇಲ್ಲದಿದ್ದರೆ {
            // V /= S
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ S_inv = S == 0.0f ? 0.0f : 1.0f/S;
            ggml_vec_scale_f32(DV, VKQ32, S_inv);

            // dst indices
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i1 = iq1;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i2 = iq2;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i3 = iq3;

            // permute(0, 2, 1, 3)
            ಸ್ಮರಣೆ_ನಕಲು((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + (i3*ne2*ne1 + i2 + i1*ne1)*nb1, VKQ32, nb1);
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_flash_attn_ext_tiled(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst,
        ಪೂರ್ಣಾಂಕ ir0, ಪೂರ್ಣಾಂಕ ir1) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * q     = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * k     = dst->src[1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * v     = dst->src[2];
    ಸ್ಥಿರಾಂಕ ggml_tensor * ಮುಖವಾಡ  = dst->src[3];
    ಸ್ಥಿರಾಂಕ ggml_tensor * sinks = dst->src[4];

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    ಸ್ಥಿರಾಂಕ int64_t DK = nek0;
    ಸ್ಥಿರಾಂಕ int64_t DV = nev0;
    ಸ್ಥಿರಾಂಕ int64_t N  = neq1;

    GGML_ASSERT(ne0 == DV);
    GGML_ASSERT(ne2 == N);

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_ಗಾತ್ರ(q->type));
    GGML_ASSERT(nbk0 == ggml_type_ಗಾತ್ರ(k->type));
    GGML_ASSERT(nbv0 == ggml_type_ಗಾತ್ರ(v->type));

    GGML_ASSERT(neq0 == DK);
    GGML_ASSERT(nek0 == DK);
    GGML_ASSERT(nev0 == DV);

    GGML_ASSERT(neq1 == N);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == ಗಾತ್ರ(ದಶಮಾಂಶ));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(k->type == v->type);
    ಸ್ಥಿರಾಂಕ ggml_type kv_type = k->type;


    // broadcast factors
    ಸ್ಥಿರಾಂಕ int64_t rk2 = neq2/nek2;
    ಸ್ಥಿರಾಂಕ int64_t rk3 = neq3/nek3;

    ಸ್ಥಿರಾಂಕ int64_t rv2 = neq2/nev2;
    ಸ್ಥಿರಾಂಕ int64_t rv3 = neq3/nev3;

    ದಶಮಾಂಶ scale         = 1.0f;
    ದಶಮಾಂಶ max_bias      = 0.0f;
    ದಶಮಾಂಶ logit_softcap = 0.0f;

    ಸ್ಮರಣೆ_ನಕಲು(&scale,         (ದಶಮಾಂಶ *) dst->op_params + 0, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&max_bias,      (ದಶಮಾಂಶ *) dst->op_params + 1, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&logit_softcap, (ದಶಮಾಂಶ *) dst->op_params + 2, ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಆದರೆ (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    ಸ್ಥಿರಾಂಕ uint32_t n_ತಲೆ      = neq2;
    ಸ್ಥಿರಾಂಕ uint32_t n_ತಲೆ_log2 = 1u << (uint32_t) floor(log2(n_ತಲೆ));

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ m0 = powf(2.0f, -(max_bias       ) / n_ತಲೆ_log2);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ m1 = powf(2.0f, -(max_bias / 2.0f) / n_ತಲೆ_log2);

    ಪೂರ್ಣಾಂಕ ith = params->ith;

    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ Q_TILE_SZ  = ggml_fa_tile_config::Q;
    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ KV_TILE_SZ = ggml_fa_tile_config::KV;

    ಪೂರ್ಣಾಂಕ ir = ir0;
    ಆಗ (ir < ir1) {
        // q indices for the start of this tile
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iq3 = ir/(neq2*neq1);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iq2 = (ir - iq3*neq2*neq1)/neq1;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iq1 = (ir - iq3*neq2*neq1 - iq2*neq1);

        // Number of valid rows in this tile:
        // - limited by tile size (Q_TILE_SZ)
        // - limited by chunk boundary (ir1 - ir)
        // - limited by head boundary (neq1 - iq1) to avoid crossing into next head
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ tile_ಅಡ್ಡಸಾಲುಗಳು = MIN(Q_TILE_SZ, MIN((ಪೂರ್ಣಾಂಕ)(ir1 - ir), (ಪೂರ್ಣಾಂಕ)(neq1 - iq1)));
        GGML_ASSERT(tile_ಅಡ್ಡಸಾಲುಗಳು > 0);

        ಸ್ಥಿರಾಂಕ uint32_t h = iq2; // head index
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಇಳಿಜಾರು = (max_bias > 0.0f) ? h < n_ತಲೆ_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_ತಲೆ_log2) + 1) : 1.0f;

        ದಶಮಾಂಶ S[Q_TILE_SZ];
        ದಶಮಾಂಶ M[Q_TILE_SZ];

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0 ; i < Q_TILE_SZ; ++i) {
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
        ದಶಮಾಂಶ * ತಳ  = (ದಶಮಾಂಶ *) params->wdata + ith*(Q_TILE_SZ*DK + 2*Q_TILE_SZ*KV_TILE_SZ + Q_TILE_SZ*DV + KV_TILE_SZ*DV + KV_TILE_SZ*DK + CACHE_ಸಾಲು_ಗಾತ್ರ_F32);

        ಶೂನ್ಯ  * Q_q    = ತಳ;
        ದಶಮಾಂಶ * KQ     = (ದಶಮಾಂಶ *)((ಅಕ್ಷರ *)ತಳ + Q_TILE_SZ * DK * ಗಾತ್ರ(ದಶಮಾಂಶ));
        ದಶಮಾಂಶ * mask32 = KQ + Q_TILE_SZ * KV_TILE_SZ;
        ದಶಮಾಂಶ * VKQ32  = mask32 + Q_TILE_SZ * KV_TILE_SZ;
        ದಶಮಾಂಶ * V32    = VKQ32 + Q_TILE_SZ * DV;
        ದಶಮಾಂಶ * K_f32  = V32 + KV_TILE_SZ * DV;

        memset(VKQ32, 0, Q_TILE_SZ * DV * ಗಾತ್ರ(ದಶಮಾಂಶ));
        memset(mask32, 0, Q_TILE_SZ * KV_TILE_SZ * ಗಾತ್ರ(ದಶಮಾಂಶ));

        // k indices
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ik3 = iq3 / rk3;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ik2 = iq2 / rk2;

        // v indices
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iv3 = iq3 / rv3;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iv2 = iq2 / rv2;

        {
            ದಶಮಾಂಶ * Q_f32 = (ದಶಮಾಂಶ *)Q_q;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ tq = 0; tq < tile_ಅಡ್ಡಸಾಲುಗಳು; tq++) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * pq = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) q->ದತ್ತಾಂಶ + ((iq1 + tq)*nbq1 + iq2*nbq2 + iq3*nbq3));
                ಸ್ಮರಣೆ_ನಕಲು(Q_f32 + tq * DK, pq, DK * ಗಾತ್ರ(ದಶಮಾಂಶ));
            }
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ tq = tile_ಅಡ್ಡಸಾಲುಗಳು; tq < Q_TILE_SZ; tq++) {
                memset(Q_f32 + tq * DK, 0, DK * ಗಾತ್ರ(ದಶಮಾಂಶ));
            }
        }

        memset(K_f32, 0, DK * KV_TILE_SZ * ಗಾತ್ರ(ದಶಮಾಂಶ));
        memset(V32,   0, KV_TILE_SZ * DV * ಗಾತ್ರ(ದಶಮಾಂಶ));

        ಗಾಗಿ (int64_t ic = 0; ic < nek1; ic += KV_TILE_SZ) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kv_tile = (ಪೂರ್ಣಾಂಕ)std::min((int64_t)KV_TILE_SZ, nek1 - ic);

            // skip the tile entirely if all the masks are -inf
            ಆದರೆ (ಮುಖವಾಡ) {
                ತಾರ್ಕಿಕ can_skip = ನಿಜ;
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ tq = 0; tq < tile_ಅಡ್ಡಸಾಲುಗಳು; tq++) {
                    ಸ್ಥಿರಾಂಕ ggml_fp16_t * mp_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ ggml_fp16_t *)((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) ಮುಖವಾಡ->ದತ್ತಾಂಶ + (iq1 + tq)*ಮುಖವಾಡ->nb[1] + (iq2%ಮುಖವಾಡ->ne[2])*ಮುಖವಾಡ->nb[2] + (iq3%ಮುಖವಾಡ->ne[3])*ಮುಖವಾಡ->nb[3]);
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ tk = 0; tk < kv_tile; tk++) {
                        mask32[tq * KV_TILE_SZ + tk] = ಇಳಿಜಾರು * GGML_CPU_FP16_ಗಮ್ಯ_FP32(mp_ಅಡ್ಡಸಾಲು[ic + tk]);
                        ಆದರೆ (mask32[tq * KV_TILE_SZ + tk] != -INFINITY) {
                            can_skip = ಸುಳ್ಳು;
                        }
                    }
                    // Pad remaining mask entries with -inf
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ tk = kv_tile; tk < KV_TILE_SZ; tk++) {
                        mask32[tq * KV_TILE_SZ + tk] = -INFINITY;
                    }
                }

                ಆದರೆ (can_skip) {
                    ಮುಂದುವರಿ;
                }
            }

            // Pack K tile transposed: K_f32[dk][kv] so KV_TILE is contiguous (SIMD dim)
            // Zero-pad the last tile so the GEMM always operates on KV_TILE_SZ columns
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ tk = 0; tk < kv_tile; tk++) {
                ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * k_ದತ್ತಾಂಶ = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)k->ದತ್ತಾಂಶ + (ic + tk)*nbk1 + ik2*nbk2 + ik3*nbk3;
                ಆದರೆ (kv_type == GGML_TYPE_F16) {
                    ಸ್ಥಿರಾಂಕ ggml_fp16_t * k_f16 = (ಸ್ಥಿರಾಂಕ ggml_fp16_t *)k_ದತ್ತಾಂಶ;
                    ಗಾಗಿ (int64_t dk = 0; dk < DK; dk++) {
                        K_f32[dk * KV_TILE_SZ + tk] = GGML_CPU_FP16_ಗಮ್ಯ_FP32(k_f16[dk]);
                    }
                } ಇಲ್ಲದಿದ್ದರೆ {
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * k_f32_src = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)k_ದತ್ತಾಂಶ;
                    ಗಾಗಿ (int64_t dk = 0; dk < DK; dk++) {
                        K_f32[dk * KV_TILE_SZ + tk] = k_f32_src[dk];
                    }
                }
            }
            memset(KQ, 0, Q_TILE_SZ * KV_TILE_SZ * ಗಾತ್ರ(ದಶಮಾಂಶ));
            simd_gemm(KQ, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)Q_q, K_f32, Q_TILE_SZ, DK, KV_TILE_SZ);
            ggml_vec_scale_f32(Q_TILE_SZ * KV_TILE_SZ, KQ, scale);

            ಆದರೆ (logit_softcap != 0.0f) {
                ggml_vec_tanh_f32(Q_TILE_SZ * KV_TILE_SZ, KQ, KQ);
                ggml_vec_scale_f32(Q_TILE_SZ * KV_TILE_SZ, KQ, logit_softcap);
            }

            // Set padded KQ entries to -inf so softmax gives them zero weight
            ಆದರೆ (kv_tile < KV_TILE_SZ) {
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ tq = 0; tq < Q_TILE_SZ; tq++) {
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ tk = kv_tile; tk < KV_TILE_SZ; tk++) {
                        KQ[tq * KV_TILE_SZ + tk] = -INFINITY;
                    }
                }
            }

            ಆದರೆ (ಮುಖವಾಡ) {
                ggml_vec_ಕೂಡಿಸು_f32(tile_ಅಡ್ಡಸಾಲುಗಳು * KV_TILE_SZ, KQ, KQ, mask32);
            }

            ತಾರ್ಕಿಕ skip[Q_TILE_SZ] = {};

            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ tq = 0; tq < Q_TILE_SZ; tq++) {
                ದಶಮಾಂಶ * kq_ಅಡ್ಡಸಾಲು = KQ + tq * KV_TILE_SZ;

                ದಶಮಾಂಶ tile_max;
                ggml_vec_max_f32(KV_TILE_SZ, &tile_max, kq_ಅಡ್ಡಸಾಲು);

                ಆದರೆ (tile_max == -INFINITY) {
                    skip[tq] = ನಿಜ;
                    ಮುಂದುವರಿ;
                }

                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ Mold = M[tq];
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ Mnew = fmaxf(Mold, tile_max);

                ಆದರೆ (Mnew > Mold) {
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ms = expf(Mold - Mnew);
                    ggml_vec_scale_f32(DV, VKQ32 + tq * DV, ms);
                    S[tq] *= ms;
                }
                M[tq] = Mnew;


                S[tq] += ggml_vec_soft_max_f32(KV_TILE_SZ, kq_ಅಡ್ಡಸಾಲು, kq_ಅಡ್ಡಸಾಲು, Mnew);
            }

            // V accumulation: VKQ32 += softmax(KQ) * V
            // Pack V tile to contiguous F32, zero-padded
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ tk = 0; tk < kv_tile; tk++) {
                ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * v_ದತ್ತಾಂಶ = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)v->ದತ್ತಾಂಶ + (ic + tk)*nbv1 + iv2*nbv2 + iv3*nbv3;
                ಆದರೆ (kv_type == GGML_TYPE_F16) {
                    ggml_cpu_fp16_ಗಮ್ಯ_fp32((ಸ್ಥಿರಾಂಕ ggml_fp16_t *)v_ದತ್ತಾಂಶ, V32 + tk * DV, DV);
                } ಇಲ್ಲದಿದ್ದರೆ {
                    ಸ್ಮರಣೆ_ನಕಲು(V32 + tk * DV, v_ದತ್ತಾಂಶ, DV * ಗಾತ್ರ(ದಶಮಾಂಶ));
                }
            }
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ tq = 0; tq < Q_TILE_SZ; tq++) {
                ಆದರೆ (skip[tq]) {
                    memset(KQ + tq * KV_TILE_SZ, 0, KV_TILE_SZ * ಗಾತ್ರ(ದಶಮಾಂಶ));
                }
            }
            simd_gemm(VKQ32, KQ, V32, Q_TILE_SZ, KV_TILE_SZ, DV);
        }

        // sinks (apply only to valid rows in the tile)
        ಆದರೆ (sinks) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ s = ((ದಶಮಾಂಶ *)((ಅಕ್ಷರ *) sinks->ದತ್ತಾಂಶ))[h];

            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ tq = 0; tq < tile_ಅಡ್ಡಸಾಲುಗಳು; tq++) {
                ದಶಮಾಂಶ ms = 1.0f;
                ದಶಮಾಂಶ vs = 1.0f;

                ಆದರೆ (s > M[tq]) {
                    ms = expf(M[tq] - s);
                    ggml_vec_scale_f32(DV, VKQ32 + tq * DV, ms);
                } ಇಲ್ಲದಿದ್ದರೆ {
                    vs = expf(s - M[tq]);
                }

                S[tq] = S[tq] * ms + vs;
            }
        }

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ tq = 0; tq < tile_ಅಡ್ಡಸಾಲುಗಳು; tq++) {
            // V /= S
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ S_inv = S[tq] == 0.0f ? 0.0f : 1.0f / S[tq];
            ggml_vec_scale_f32(DV, VKQ32 + tq * DV, S_inv);

            // dst indices
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i1 = iq1 + tq;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i2 = iq2;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i3 = iq3;

            // permute(0, 2, 1, 3)
            ಸ್ಮರಣೆ_ನಕಲು((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + (i3*ne2*ne1 + i2 + i1*ne1)*nb1, VKQ32 + tq * DV, nb1);
        }

        ir += tile_ಅಡ್ಡಸಾಲುಗಳು;
    }
}

// Reduction function: combines partial results across KV chunks
// Partials layout in wdata: [n_q_heads][n_chunks][2 + DV]
ಸ್ಥಿರ ಶೂನ್ಯ ggml_flash_attn_ext_reduce_partials(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst,
        ಸ್ಥಿರಾಂಕ int64_t n_chunks,
        ಸ್ಥಿರಾಂಕ int64_t chunk_ಗಾತ್ರ) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * q = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * k = dst->src[1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * v = dst->src[2];

    ಸ್ಥಿರಾಂಕ int64_t DK        = k->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t DV        = v->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t nek1      = k->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t n_q_heads = q->ne[2];

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ int64_t wdata_ಪ್ರತಿ_thread = DK + 2*DV + CACHE_ಸಾಲು_ಗಾತ್ರ_F32;
    ದಶಮಾಂಶ *       thread_wdata     = (ದಶಮಾಂಶ *) params->wdata + ith * wdata_ಪ್ರತಿ_thread;

    ಸ್ಥಿರಾಂಕ int64_t partials_ಸ್ಥಾನಾಂತರ  = nth * (DK + 2*DV + CACHE_ಸಾಲು_ಗಾತ್ರ_F32);
    ಸ್ಥಿರಾಂಕ int64_t partial_ಗಾತ್ರ     = 2 + DV;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * partials_ತಳ    = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) params->wdata + partials_ಸ್ಥಾನಾಂತರ;

    // Output layout
    ಸ್ಥಿರಾಂಕ int64_t ne1 = dst->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t ne2 = dst->ne[2];
    ಸ್ಥಿರಾಂಕ size_t  nb1 = dst->nb[1];

    // Each thread reduces a subset of query heads
    ಗಾಗಿ (int64_t q_ತಲೆ = ith; q_ತಲೆ < n_q_heads; q_ತಲೆ += nth) {
        ದಶಮಾಂಶ   M_final   = -INFINITY;
        ದಶಮಾಂಶ   S_final   = 0.0f;
        ದಶಮಾಂಶ * VKQ_final = thread_wdata;
        memset(VKQ_final, 0, DV * ಗಾತ್ರ(ದಶಮಾಂಶ));

        // Combine partials from all chunks
        ಗಾಗಿ (int64_t chunk_idx = 0; chunk_idx < n_chunks; ++chunk_idx) {
            ಸ್ಥಿರಾಂಕ int64_t ic_ಆರಂಭ = chunk_idx * chunk_ಗಾತ್ರ;
            ಆದರೆ (ic_ಆರಂಭ >= nek1) ಮುಂದುವರಿ;

            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * partial   = partials_ತಳ + (q_ತಲೆ * n_chunks + chunk_idx) * partial_ಗಾತ್ರ;
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   M_chunk   = partial[0];
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   S_chunk   = partial[1];
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * VKQ_chunk = partial + 2;

            ಆದರೆ (S_chunk == 0.0f) ಮುಂದುವರಿ;

            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ M_new     = fmaxf(M_final, M_chunk);
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale_old = expf(M_final - M_new);
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale_new = expf(M_chunk - M_new);

            ಗಾಗಿ (int64_t d = 0; d < DV; ++d) {
                VKQ_final[d] = VKQ_final[d] * scale_old + VKQ_chunk[d] * scale_new;
            }
            S_final = S_final * scale_old + S_chunk * scale_new;
            M_final = M_new;
        }

        // Normalize and write to output
        ಆದರೆ (S_final != 0.0f) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ S_inv = 1.0f / S_final;
            ggml_vec_scale_f32(DV, VKQ_final, S_inv);
        }
        // iq1=0, iq3=0 for decode
        ಸ್ಮರಣೆ_ನಕಲು((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + (0*ne2*ne1 + q_ತಲೆ + 0*ne1)*nb1, VKQ_final, nb1);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_flash_attn_ext_f16(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * q     = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * k     = dst->src[1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * v     = dst->src[2];

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    ಸ್ಥಿರಾಂಕ int64_t DK = nek0;
    ಸ್ಥಿರಾಂಕ int64_t DV = nev0;
    ಸ್ಥಿರಾಂಕ int64_t N  = neq1;


    GGML_ASSERT(ne0 == DV);
    GGML_ASSERT(ne2 == N);

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_ಗಾತ್ರ(q->type));
    GGML_ASSERT(nbk0 == ggml_type_ಗಾತ್ರ(k->type));
    GGML_ASSERT(nbv0 == ggml_type_ಗಾತ್ರ(v->type));

    GGML_ASSERT(neq0 == DK);
    GGML_ASSERT(nek0 == DK);
    GGML_ASSERT(nev0 == DV);

    GGML_ASSERT(neq1 == N);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == ಗಾತ್ರ(ದಶಮಾಂಶ));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    // When use_ref is set, force the vec-only reference implementation (no tiling, no KV-chunking)
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ use_ref = params->use_ref;

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ kv_is_f32_or_f16 = (k->type == GGML_TYPE_F32 || k->type == GGML_TYPE_F16);
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ use_ವಿಭಜಿಸು_kv_ಮಾರ್ಗ = !use_ref && (neq1 == 1 && neq3 == 1) && kv_is_f32_or_f16 && (k->type == v->type) && q->type == GGML_TYPE_F32 && nek1 >= 512;

    ಆದರೆ (use_ವಿಭಜಿಸು_kv_ಮಾರ್ಗ) {
        ಸ್ಥಿರಾಂಕ int64_t chunk_ಗಾತ್ರ = (nek1 + nth - 1) / nth;

        // Partials buffer layout: [q_head][kv_chunk][M, S, VKQ]
        ಸ್ಥಿರಾಂಕ int64_t partial_ಗಾತ್ರ  = 2 + DV;
        ದಶಮಾಂಶ *       partials_ತಳ = (ದಶಮಾಂಶ *) params->wdata + nth * (DK + 2*DV + CACHE_ಸಾಲು_ಗಾತ್ರ_F32);

        ಸ್ಥಿರಾಂಕ int64_t ic_ಆರಂಭ = ith * chunk_ಗಾತ್ರ;
        ಸ್ಥಿರಾಂಕ int64_t ic_ಕೊನೆ   = std::min(ic_ಆರಂಭ + chunk_ಗಾತ್ರ, nek1);

        ಸ್ಥಿರಾಂಕ int64_t partial_stride = nth * partial_ಗಾತ್ರ;
        ದಶಮಾಂಶ *       chunk_partials = partials_ತಳ + ith * partial_ಗಾತ್ರ;

        ಆದರೆ (ic_ಆರಂಭ < nek1) {
            ಗಾಗಿ (int64_t q_ತಲೆ = 0; q_ತಲೆ < neq2; q_ತಲೆ++) {
                ggml_compute_forward_flash_attn_ext_f16_one_chunk(
                    params, dst, q_ತಲೆ, q_ತಲೆ + 1, ic_ಆರಂಭ, ic_ಕೊನೆ,
                    chunk_partials, partial_stride);
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಗಾಗಿ (int64_t q_ತಲೆ = 0; q_ತಲೆ < neq2; q_ತಲೆ++) {
                ದಶಮಾಂಶ * q_partials = chunk_partials + q_ತಲೆ * partial_stride;
                q_partials[0] = -INFINITY;  // M
                q_partials[1] = 0.0f;       // S
            }
        }

        ggml_barrier(params->threadpool);
        ggml_flash_attn_ext_reduce_partials(params, dst, nth, chunk_ಗಾತ್ರ);
    } ಇಲ್ಲದಿದ್ದರೆ {

        // total rows in q
        ಸ್ಥಿರಾಂಕ int64_t nr = neq1*neq2*neq3;

        // disable for NUMA
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ disable_chunking = ggml_is_numa();

        // 4x chunks per thread
        ಪೂರ್ಣಾಂಕ nth_scaled = nth * 4;
        int64_t chunk_ಗಾತ್ರ = (nr + nth_scaled - 1) / nth_scaled;
        int64_t nchunk     = (nr + chunk_ಗಾತ್ರ - 1) / chunk_ಗಾತ್ರ;

        ಆದರೆ (nth == 1 || nchunk < nth || disable_chunking) {
            nchunk = nth;
        }

        ಆದರೆ (ith == 0) {
            ggml_threadpool_chunk_ಗಣ(params->threadpool, nth);
        }

        ggml_barrier(params->threadpool);

        ಸ್ಥಿರಾಂಕ int64_t dr = (nr + nchunk - 1) / nchunk;

        ಸ್ಥಿರ constexpr int64_t Q_TILE_SZ  = ggml_fa_tile_config::Q;
        ತಾರ್ಕಿಕ use_tiled = !use_ref &&
                               (q->type == GGML_TYPE_F32 &&
                                kv_is_f32_or_f16 &&
                                k->type == v->type &&
                                neq1 >= Q_TILE_SZ);
#ಆದರೆ defined(GGML_SIMD) && !defined(__x86_64__) && !defined(_M_X64)
#ಆದರೆ defined(__ARM_FEATURE_SVE)
        ಸ್ಥಿರಾಂಕ int64_t f32_epr = svcntw();
#ಇಲ್ಲದಿದ್ದರೆ
        ಸ್ಥಿರಾಂಕ int64_t f32_epr = GGML_F32_EPR;
#ಆದರೆ_ಅಂತ್ಯ
        use_tiled &= (DV % f32_epr == 0);
#ಆದರೆ_ಅಂತ್ಯ
        ಪೂರ್ಣಾಂಕ current_chunk = ith;

        ಆಗ (current_chunk < nchunk) {
            ಸ್ಥಿರಾಂಕ int64_t ir0 = dr * current_chunk;
            ಸ್ಥಿರಾಂಕ int64_t ir1 = MIN(ir0 + dr, nr);

            ಆದರೆ (use_tiled) {
                ggml_compute_forward_flash_attn_ext_tiled(params, dst, ir0, ir1);
            } ಇಲ್ಲದಿದ್ದರೆ {
                ggml_compute_forward_flash_attn_ext_f16_one_chunk(params, dst, ir0, ir1, 0, nek1, ಶೂನ್ಯಸೂಚಕ, 0);
            }

            current_chunk = ggml_threadpool_chunk_ಕೂಡಿಸು(params->threadpool, 1);
        }
    }
}

ಶೂನ್ಯ ggml_compute_forward_flash_attn_ext(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {
    ಆಯ್ಕೆ (dst->op_params[3]) {
        ಸ್ಥಿತಿ GGML_PREC_DEFAULT:
        ಸ್ಥಿತಿ GGML_PREC_F32:
            {
                // uses F32 accumulators
                ggml_compute_forward_flash_attn_ext_f16(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_flash_attn_back

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_flash_attn_ಹಿಂಭಾಗ_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ masked,
              ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * q = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * k = dst->src[1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * v = dst->src[2];
    ಸ್ಥಿರಾಂಕ ggml_tensor * d = dst->src[3];

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

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ int64_t D = neq0;
    ಸ್ಥಿರಾಂಕ int64_t N = neq1;
    ಸ್ಥಿರಾಂಕ int64_t P = nek1 - N;
    ಸ್ಥಿರಾಂಕ int64_t M = P + N;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ Mup  = ggml_ಮೇಲೆ(M, GGML_SOFT_MAX_UNROLL);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mxDM = MAX(D, Mup);

    // GGML_ASSERT(ne0 == D);
    // GGML_ASSERT(ne1 == N);
    GGML_ASSERT(P >= 0);

    GGML_ASSERT(nbq0 == ಗಾತ್ರ(ದಶಮಾಂಶ));
    GGML_ASSERT(nbk0 == ಗಾತ್ರ(ದಶಮಾಂಶ));
    GGML_ASSERT(nbv0 == ಗಾತ್ರ(ದಶಮಾಂಶ));

    GGML_ASSERT(neq0 == D);
    GGML_ASSERT(nek0 == D);
    GGML_ASSERT(nev1 == D);
    GGML_ASSERT(ned0 == D);

    GGML_ASSERT(neq1 == N);
    GGML_ASSERT(nek1 == N + P);
    GGML_ASSERT(nev1 == D);
    GGML_ASSERT(ned1 == N);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == ಗಾತ್ರ(ದಶಮಾಂಶ));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    ಆದರೆ (ith == 0) {
        memset(dst->ದತ್ತಾಂಶ, 0, nb0*ne0*ne1*ne2*ne3);
    }
    ggml_barrier(params->threadpool);

    ಸ್ಥಿರಾಂಕ int64_t elem_q = ggml_nelements(q);
    ಸ್ಥಿರಾಂಕ int64_t elem_k = ggml_nelements(k);

    ggml_type result_type = dst->type;
    GGML_ASSERT(ggml_blck_ಗಾತ್ರ(result_type) == 1);
    ಸ್ಥಿರಾಂಕ size_t tsize = ggml_type_ಗಾತ್ರ(result_type);

    ಸ್ಥಿರಾಂಕ size_t offs_q = 0;
    ಸ್ಥಿರಾಂಕ size_t offs_k = offs_q + GGML_PAD(elem_q * tsize, GGML_MEM_ALIGN);
    ಸ್ಥಿರಾಂಕ size_t offs_v = offs_k + GGML_PAD(elem_k * tsize, GGML_MEM_ALIGN);

    ಶೂನ್ಯ * grad_q = (ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ;
    ಶೂನ್ಯ * grad_k = (ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + offs_k;
    ಶೂನ್ಯ * grad_v = (ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + offs_v;

    ಸ್ಥಿರಾಂಕ size_t nbgq1 = nb0*neq0;
    ಸ್ಥಿರಾಂಕ size_t nbgq2 = nb0*neq0*neq1;
    ಸ್ಥಿರಾಂಕ size_t nbgq3 = nb0*neq0*neq1*neq2;

    ಸ್ಥಿರಾಂಕ size_t nbgk1 = nb0*nek0;
    ಸ್ಥಿರಾಂಕ size_t nbgk2 = nb0*nek0*nek1;
    ಸ್ಥಿರಾಂಕ size_t nbgk3 = nb0*nek0*nek1*neq2;

    ಸ್ಥಿರಾಂಕ size_t nbgv1 = nb0*nev0;
    ಸ್ಥಿರಾಂಕ size_t nbgv2 = nb0*nev0*nev1;
    ಸ್ಥಿರಾಂಕ size_t nbgv3 = nb0*nev0*nev1*neq2;

    // parallelize by k rows using ggml_vec_dot_f32

    // total rows in k
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr = nek2*nek3;

    // rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale = 1.0f/sqrtf(D);

    //printf("P=%d N=%d D=%d ir0=%d ir1=%d scale = %f\n", P, N, D, ir0, ir1, scale);

    // how often k2 (and v2) is repeated in q2
    ಪೂರ್ಣಾಂಕ nrep = neq2/nek2;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ir = ir0; ir < ir1; ++ir) {
        // q indices
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ik3 = ir/(nek2);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ik2 = ir - ik3*nek2;

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iq3 = ik3;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ id3 = ik3;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iv3 = ik3;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iv2 = ik2;

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ irep = 0; irep < nrep; ++irep) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iq2 = ik2 + irep*nek2;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ id2 = iq2;

            // (ik2 + irep*nek2) % nek2 == ik2
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ iq1 = 0; iq1 < neq1; ++iq1) {
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ id1 = iq1;

                // not sure about CACHE_LINE_SIZE_F32..
                // - maybe it must not be multiplied by 2 and excluded from .. in SM 1*(..) offset?
                ದಶಮಾಂಶ * S  = (ದಶಮಾಂಶ *) params->wdata + ith*2*(mxDM + CACHE_ಸಾಲು_ಗಾತ್ರ_F32) + 0*(mxDM+CACHE_ಸಾಲು_ಗಾತ್ರ_F32);
                ದಶಮಾಂಶ * SM = (ದಶಮಾಂಶ *) params->wdata + ith*2*(mxDM + CACHE_ಸಾಲು_ಗಾತ್ರ_F32) + 1*(mxDM+CACHE_ಸಾಲು_ಗಾತ್ರ_F32);

                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = M; i < Mup; ++i) {
                    S[i] = -INFINITY;
                }

                ಸ್ಥಿರಾಂಕ int64_t masked_begin = masked ? (P + iq1 + 1) : M;
                ಗಾಗಿ (int64_t ic = 0; ic < masked_begin; ++ic) {
                    // k indices
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ik1 = ic;

                    // S indices
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i1 = ik1;

                    ggml_vec_dot_f32(neq0,
                            S + i1, 0,
                            (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) k->ದತ್ತಾಂಶ + (ik1*nbk1 + ik2*nbk2 + ik3*nbk3)), 0,
                            (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) q->ದತ್ತಾಂಶ + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3)), 0, 1);
                }

                // scale
                ggml_vec_scale_f32(masked_begin, S, scale);

                ಗಾಗಿ (int64_t i = masked_begin; i < M; i++) {
                    S[i] = -INFINITY;
                }

                // softmax
                // exclude known -INF S[..] values from max and loop
                // dont forget to set their SM values to zero
                {
                    ದಶಮಾಂಶ max = -INFINITY;
                    ggml_vec_max_f32(masked_begin, &max, S);

                    ggml_float ಮೊತ್ತ = 0.0;
                    {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SOFT_MAX_ACCELERATE
                        max = -max;
                        vDSP_vsadd(SM, 1, &max, SM, 1, Mup);
                        vvexpf(SM, SM, &Mup);
                        ggml_vec_ಮೊತ್ತ_f32(Mup, &ಮೊತ್ತ, SM);
#ಇಲ್ಲದಿದ್ದರೆ
                        ಮೊತ್ತ = ggml_vec_soft_max_f32(Mup, SM, S, max);
#ಆದರೆ_ಅಂತ್ಯ
                    }

                    ಪ್ರತಿಜ್ಞಾ(ಮೊತ್ತ > 0.0);

                    ಮೊತ್ತ = 1.0/ಮೊತ್ತ;
                    ggml_vec_scale_f32(masked_begin, SM, ಮೊತ್ತ);

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
                ggml_vec_ಗಣ_f32(masked_begin, S, 0);
                ಗಾಗಿ (int64_t ic = 0; ic < D; ++ic) {
                    ggml_vec_mad_f32(masked_begin,
                            S,
                             (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) v->ದತ್ತಾಂಶ + (          ic*nbv1  + iv2*nbv2 + iv3*nbv3)),
                            *(ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) d->ದತ್ತಾಂಶ + (ic*nbd0 + id1*nbd1 + id2*nbd2 + id3*nbd3)));
                }

                // S = SM * (S - dot(SM, S))
                ದಶಮಾಂಶ dot_SM_gradSM = 0;
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
                ಗಾಗಿ (int64_t ic = 0; ic < masked_begin; ++ic) {
                    ggml_vec_mad_f32(D,
                            (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) grad_q  + (iq1*nbgq1 + iq2*nbgq2  + iq3*nbgq3)),
                            (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) k->ದತ್ತಾಂಶ + (ic*nbk1   + ik2*nbk2   + ik3*nbk3)),
                            S[ic]);
                }

                // grad[k][:D,:M,iq2,iq3] += S.T @ qcur
                // for ic:
                //  grad[k][:D,ic,iq2,iq3] += S.T[0,ic] * qcur[:D,0]
                //  grad[k][:D,ic,iq2,iq3] += S[ic]     * qcur[:D,0]
                // exclude known zero S[..] values from loop
                ಗಾಗಿ (int64_t ic = 0; ic < masked_begin; ++ic) {
                    ggml_vec_mad_f32(D,
                            (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) grad_k  + (ic*nbgk1  + ik2*nbgk2  + ik3*nbgk3)),
                            (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) q->ದತ್ತಾಂಶ + (iq1*nbq1  + iq2*nbq2   + iq3*nbq3)),
                            S[ic]);
                }

                // grad[v][:M,:D,iv2,iv3] += d[:D,id1,id2,id3].T       @ SM
                // for ic:
                //  grad[v][:M,ic,iv2,iv3] += d[:D,id1,id2,id3].T[0,ic] * SM[:M]
                //  grad[v][:M,ic,iv2,iv3] += d[ic,id1,id2,id3]         * SM[:M]
                // exclude known zero SM[..] values from mad
                ಗಾಗಿ (int64_t ic = 0; ic < D; ++ic) {
                    ggml_vec_mad_f32(masked_begin,
                            (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) grad_v   + (          ic*nbgv1 + iv2*nbgv2 + iv3*nbgv3)),
                            SM,
                            *(ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) d->ದತ್ತಾಂಶ + (ic*nbd0 + id1*nbd1 + id2*nbd2  + id3*nbd3)));
                }
            }
        }
    }
}

ಶೂನ್ಯ ggml_compute_forward_flash_attn_ಹಿಂಭಾಗ(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ masked,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * q = dst->src[0];

    ಆಯ್ಕೆ (q->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_flash_attn_ಹಿಂಭಾಗ_f32(params, masked, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_ssm_conv

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_ssm_conv_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0]; // conv_x
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1]; // conv1d.weight

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nc  = src1->ne[0]; // d_conv
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncs = src0->ne[0]; // d_conv - 1 + n_t
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr  = src0->ne[1]; // d_inner
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_t =  dst->ne[1]; // tokens per sequence
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_s =  dst->ne[2]; // number of sequences in the batch

    GGML_ASSERT( dst->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ));
    GGML_ASSERT(src1->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ));
    GGML_ASSERT(src0->nb[1] == src0->ne[0]*ಗಾತ್ರ(ದಶಮಾಂಶ));

    // rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir  = ir1 - ir0;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i3 = 0; i3 < n_s; ++i3) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i2 = 0; i2 < n_t; ++i2) {
            // {d_conv - 1 + n_t, d_inner, n_seqs}
            // sliding window
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * s = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + ir0*(src0->nb[1]) + i2*(src0->nb[0]) + i3*(src0->nb[2])); // {d_conv, d_inner, n_s}
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * c = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ + ir0*(src1->nb[1])); // {d_conv, d_inner}
            ದಶಮಾಂಶ * x = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + ir0*(dst->nb[0]) + i2*(dst->nb[1]) + i3*(dst->nb[2])); // {d_inner, n_t, n_s}

            // TODO: transpose the output for smaller strides for big batches?
            // d_inner
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i1 = 0; i1 < ir; ++i1) {
                // rowwise dot product
                // NOTE: not using ggml_vec_dot_f32, because its sum is in double precision
                ದಶಮಾಂಶ sumf = 0.0f;

                // d_conv
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < nc; ++i0) {
                    sumf += s[i0 + i1*ncs] * c[i0 + i1*nc];
                }
                x[i1] = sumf;
            }
        }
    }
}

ಶೂನ್ಯ ggml_compute_forward_ssm_conv(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {
    ಆಯ್ಕೆ (dst->src[0]->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_ssm_conv_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_ssm_scan

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_ssm_ಪರಿಶೀಲನೆ_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0]; // s  {d_state, dim, n_head, n_seqs+}
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1]; // x  {dim, n_head, n_seq_tokens, n_seqs}
    ಸ್ಥಿರಾಂಕ ggml_tensor * src2 = dst->src[2]; // dt {n_head, n_seq_tokens, n_seqs}
    ಸ್ಥಿರಾಂಕ ggml_tensor * src3 = dst->src[3]; // A  {d_state, n_head} or {1, n_head}
    ಸ್ಥಿರಾಂಕ ggml_tensor * src4 = dst->src[4]; // B  {d_state, n_group, n_seq_tokens, n_seqs}
    ಸ್ಥಿರಾಂಕ ggml_tensor * src5 = dst->src[5]; // C  {d_state, n_group, n_seq_tokens, n_seqs}
    ಸ್ಥಿರಾಂಕ ggml_tensor * src6 = dst->src[6]; // ids {n_seqs}

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ int64_t nc = src0->ne[0]; // d_state
    ಸ್ಥಿರಾಂಕ int64_t nr = src0->ne[1]; // dim
    ಸ್ಥಿರಾಂಕ int64_t nh = src1->ne[1]; // n_head
    ಸ್ಥಿರಾಂಕ int64_t ng = src4->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t nt = src1->ne[2]; // number of tokens per sequence
    ಸ್ಥಿರಾಂಕ int64_t ns = src1->ne[3]; // number of sequences in the batch
    ಸ್ಥಿರಾಂಕ int64_t K  = ggml_get_op_params_i32(dst, 0);

    // can't use ggml_nbytes because src1 is not necessarily contiguous
    ಸ್ಥಿರಾಂಕ int64_t s_off = ggml_nelements(src1) * ggml_element_ಗಾತ್ರ(src1);

    GGML_ASSERT(K >= 1);
    GGML_ASSERT(ggml_nelements(src1) + K*nc*nr*nh*ns == ggml_nelements(dst));
    GGML_ASSERT(src0->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ));
    GGML_ASSERT(src1->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ));
    GGML_ASSERT(src2->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ));
    GGML_ASSERT(src3->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ));
    GGML_ASSERT(src4->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ));
    GGML_ASSERT(src5->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ));
    GGML_ASSERT(src6->nb[0] == ಗಾತ್ರ(int32_t));
    GGML_ASSERT(nh % ng == 0);
    GGML_ASSERT(src3->ne[0] == 1 || K == 1);

    // heads per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dh = (nh + nth - 1)/nth;

    // head range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ih0 = dh*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ih1 = MIN(ih0 + dh, nh);

    ಸ್ಥಿರಾಂಕ int32_t * ids = (ಸ್ಥಿರಾಂಕ int32_t *) src6->ದತ್ತಾಂಶ;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i3 = 0; i3 < ns; ++i3) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * s0 = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + ids[i3]*(src0->nb[3])); // {d_state, dim, nh, ns}
              ದಶಮಾಂಶ * s  = (      ದಶಮಾಂಶ *) ((      ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ  + i3*(src0->nb[3]) + s_off); // {d_state, dim, nh, ns}

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i2 = 0; i2 < nt; ++i2) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x  = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ + i2*(src1->nb[2]) + i3*(src1->nb[3])); // {dim, nh, nt, ns}
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * dt = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src2->ದತ್ತಾಂಶ + i2*(src2->nb[1]) + i3*(src2->nb[2])); // {nh, nt, ns}
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * A  = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src3->ದತ್ತಾಂಶ); // {d_state, nh} or {1, nh}
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * B  = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src4->ದತ್ತಾಂಶ + i2*(src4->nb[2]) + i3*(src4->nb[3])); // {d_state, ng, nt, ns}
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * C  = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src5->ದತ್ತಾಂಶ + i2*(src5->nb[2]) + i3*(src5->nb[3])); // {d_state, ng, nt, ns}
                  ದಶಮಾಂಶ * y  = (      ದಶಮಾಂಶ *) ((      ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i2*(nh*nr*ಗಾತ್ರ(ದಶಮಾಂಶ)) + i3*(nt*nh*nr*ಗಾತ್ರ(ದಶಮಾಂಶ))); // {dim, nh, nt, ns}

            ಆದರೆ (src3->ne[0] == 1) {
                // Mamba-2 has a scalar decay factor per head; dA can be outside the state-wise loop

                // n_head
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ h = ih0; h < ih1; ++h) {
                    // ref: https://github.com/state-spaces/mamba/blob/62db608da60f6fc790b8ed9f4b3225e95ca15fde/mamba_ssm/ops/triton/softplus.py#L16
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dt_soft_plus = ggml_compute_softplus_f32(dt[h]);
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dA = expf(dt_soft_plus * A[h]);
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ g = h / (nh / ng); // repeat_interleave

                    // dim
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i1 = 0; i1 < nr; ++i1) {
                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ii = i1 + h*nr;
                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x_dt = x[ii] * dt_soft_plus;
                        ದಶಮಾಂಶ sumf = 0.0f;
#ಆದರೆ defined(GGML_SIMD)
    #ಆದರೆ defined(__ARM_FEATURE_SVE)
                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ggml_f32_epr = svcntw();
                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ggml_f32_ಹೆಜ್ಜೆ = 1 * ggml_f32_epr;

                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ np = (nc & ~(ggml_f32_ಹೆಜ್ಜೆ - 1));

                        GGML_F32_VEC ಮೊತ್ತ = GGML_F32_VEC_ZERO;

                        GGML_F32_VEC adA = GGML_F32_VEC_SET1(dA);
                        GGML_F32_VEC axdt = GGML_F32_VEC_SET1(x_dt);

                        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < np; i += ggml_f32_ಹೆಜ್ಜೆ) {
                            // TODO: maybe unroll more?
                            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 1; j++) {
                                GGML_F32_VEC t0 = GGML_F32_VEC_ತುಂಬು(s0 + i + j*ggml_f32_epr + ii*nc);
                                GGML_F32_VEC t1 = GGML_F32_VEC_ತುಂಬು(B + i + j*ggml_f32_epr + g*nc);
                                GGML_F32_VEC t2 = GGML_F32_VEC_ತುಂಬು(C + i + j*ggml_f32_epr + g*nc);

                                t0 = GGML_F32_VEC_MUL(t0, adA);
                                t1 = GGML_F32_VEC_MUL(t1, axdt);

                                t0 = GGML_F32_VEC_ಕೂಡಿಸು(t0, t1);

                                ಮೊತ್ತ = GGML_F32_VEC_FMA(ಮೊತ್ತ, t0, t2);

                                GGML_F32_VEC_STORE(s + i + j*ggml_f32_epr + ii*nc, t0);
                            }
                        }

                        sumf = GGML_F32xt_REDUCE_ONE(ಮೊತ್ತ);
    #ಅಥವಾಆದರೆ defined(__riscv_v_intrinsic)
                        // todo: RVV implementation
                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ np = 0;
    #ಇಲ್ಲದಿದ್ದರೆ
                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ np = (nc & ~(GGML_F32_ಹೆಜ್ಜೆ - 1));

                        GGML_F32_VEC ಮೊತ್ತ[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };

                        GGML_F32_VEC adA = GGML_F32_VEC_SET1(dA);
                        GGML_F32_VEC axdt = GGML_F32_VEC_SET1(x_dt);

                        GGML_F32_VEC ax[GGML_F32_ARR];
                        GGML_F32_VEC ay[GGML_F32_ARR];
                        GGML_F32_VEC az[GGML_F32_ARR];

                        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < np; i += GGML_F32_ಹೆಜ್ಜೆ) {
                            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < GGML_F32_ARR; j++) {
                                ax[j] = GGML_F32_VEC_ತುಂಬು(s0 + i + j*GGML_F32_EPR + ii*nc);
                                ay[j] = GGML_F32_VEC_ತುಂಬು(B + i + j*GGML_F32_EPR + g*nc);
                                az[j] = GGML_F32_VEC_ತುಂಬು(C + i + j*GGML_F32_EPR + g*nc);

                                ax[j] = GGML_F32_VEC_MUL(ax[j], adA);
                                ay[j] = GGML_F32_VEC_MUL(ay[j], axdt);

                                ax[j] = GGML_F32_VEC_ಕೂಡಿಸು(ax[j], ay[j]);

                                ಮೊತ್ತ[j] = GGML_F32_VEC_FMA(ಮೊತ್ತ[j], ax[j], az[j]);

                                GGML_F32_VEC_STORE(s + i + j*GGML_F32_EPR + ii*nc, ax[j]);
                            }
                        }

                        // reduce sum0..sum3 to sum0
                        GGML_F32_VEC_REDUCE(sumf, ಮೊತ್ತ);
    #ಆದರೆ_ಅಂತ್ಯ
#ಇಲ್ಲದಿದ್ದರೆ
                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ np = 0;
#ಆದರೆ_ಅಂತ್ಯ
                        // d_state
                        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = np; i0 < nc; ++i0) {
                            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i = i0 + ii*nc;
                            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ig = i0 + g*nc;
                            // state = prev_state * dA + dB * x
                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ state = (s0[i] * dA) + (B[ig] * x_dt);
                            // y = rowwise_dotprod(state, C)
                            sumf += state * C[ig];
                            s[i] = state;
                        }
                        y[ii] = sumf;
                    }
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                // Mamba-1 has an element-wise decay factor for the states

                // n_head
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ h = ih0; h < ih1; ++h) {
                    // ref: https://github.com/state-spaces/mamba/blob/62db608da60f6fc790b8ed9f4b3225e95ca15fde/mamba_ssm/ops/triton/softplus.py#L16
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dt_soft_plus = ggml_compute_softplus_f32(dt[h]);
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ g = h / (nh / ng); // repeat_interleave

                    // dim
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i1 = 0; i1 < nr; ++i1) {
                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ii = i1 + h*nr;
                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ x_dt = x[ii] * dt_soft_plus;
#ಆದರೆ defined(__ARM_FEATURE_SVE)
                        svfloat32_t vx_dt = GGML_F32_VEC_SET1(x_dt);
                        svfloat32_t vdt_soft_plus = GGML_F32_VEC_SET1(dt_soft_plus);
                        svfloat32_t r1_ಸದಿಶ = GGML_F32_VEC_ZERO;

                        // d_state
                        // TODO: what happens when (d_state % svcntw()) != 0?
                        ಗಾಗಿ (int64_t k = 0; k < nc; k += svcntw()) {
                            svfloat32_t vA = GGML_F32_VEC_ತುಂಬು(&A[h*nc + k]);
                            svfloat32_t vB = GGML_F32_VEC_ತುಂಬು(&B[k + g*nc]);
                            svfloat32_t vC = GGML_F32_VEC_ತುಂಬು(&C[k + g*nc]);
                            svfloat32_t vs0 = GGML_F32_VEC_ತುಂಬು(&s0[ii*nc + k]);

                            svfloat32_t t1 = GGML_F32_VEC_MUL(vdt_soft_plus, vA);
                            t1 = exp_ps_sve(svptrue_b32(), t1);
                            svfloat32_t t2 = GGML_F32_VEC_MUL(vx_dt, vB);

                            vs0 = GGML_F32_VEC_FMA(t2, vs0, t1);
                            r1_ಸದಿಶ = GGML_F32_VEC_ಕೂಡಿಸು(GGML_F32_VEC_MUL(vs0, vC), r1_ಸದಿಶ);

                            GGML_F32_VEC_STORE(&s[ii*nc + k], vs0);
                        }
                        y[ii] = GGML_F32xt_REDUCE_ONE(r1_ಸದಿಶ);
#ಇಲ್ಲದಿದ್ದರೆ
                        ದಶಮಾಂಶ sumf = 0.0f;
                        // NOTE: can't really use GGML_SIMD here because d_state is usually 16
                        //       and also because expf is used within the loop.
                        // d_state
                        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < nc; ++i0) {
                            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i = i0 + ii*nc;
                            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ig = i0 + g*nc;
                            // state = prev_state * dA + dB * x
                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ state = (s0[i] * expf(dt_soft_plus * A[i0 + h*nc])) + (B[ig] * x_dt);
                            // y = rowwise_dotprod(state, C)
                            sumf += state * C[ig];
                            s[i] = state;
                        }
                        y[ii] = sumf;
#ಆದರೆ_ಅಂತ್ಯ
                    }
                }
            }
            ಸ್ಥಿರಾಂಕ int64_t ಸ್ಥಾನ = nt - 1 - i2;
            ಆದರೆ (K > 1 && ಸ್ಥಾನ > 0 && ಸ್ಥಾನ < K) {
                ದಶಮಾಂಶ * s_snapshot = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + s_off + (ಸ್ಥಾನ*ns + i3)*(src0->nb[3]));
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ h = ih0; h < ih1; ++h) {
                    ಸ್ಮರಣೆ_ನಕಲು((ಅಕ್ಷರ *) s_snapshot + h*src0->nb[2], (ಅಕ್ಷರ *) s + h*src0->nb[2], src0->nb[2]);
                }
            }
            // use the output as the source when it's not the first token-wise iteration
            s0 = s;
        }
    }
}

ಶೂನ್ಯ ggml_compute_forward_ssm_ಪರಿಶೀಲನೆ(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {
    ಆಯ್ಕೆ (dst->src[0]->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_ssm_ಪರಿಶೀಲನೆ_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_win_part

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_win_part_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {
    GGML_ಬಳಸದ(params);

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)

    ಸ್ಥಿರಾಂಕ int32_t nep0 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[0];
    ಸ್ಥಿರಾಂಕ int32_t nep1 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[1];
    ಸ್ಥಿರಾಂಕ int32_t w    = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[2];

    ಪ್ರತಿಜ್ಞಾ(ne00 == ne0);
    ಪ್ರತಿಜ್ಞಾ(ne3  == nep0*nep1);

    // TODO: optimize / multi-thread
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ py = 0; py < nep1; ++py) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ px = 0; px < nep0; ++px) {
            ಸ್ಥಿರಾಂಕ int64_t i3 = py*nep0 + px;
            ಗಾಗಿ (int64_t i2 = 0; i2 < ne2; ++i2) {
                ಗಾಗಿ (int64_t i1 = 0; i1 < ne1; ++i1) {
                    ಗಾಗಿ (int64_t i0 = 0; i0 < ne0; ++i0) {
                        ಸ್ಥಿರಾಂಕ int64_t i02 = py*w + i2;
                        ಸ್ಥಿರಾಂಕ int64_t i01 = px*w + i1;
                        ಸ್ಥಿರಾಂಕ int64_t i00 = i0;

                        ಸ್ಥಿರಾಂಕ int64_t i = i3*ne2*ne1*ne0 + i2*ne1*ne0    + i1*ne0   + i0;
                        ಸ್ಥಿರಾಂಕ int64_t j =                  i02*ne01*ne00 + i01*ne00 + i00;

                        ಆದರೆ (py*w + i2 >= ne02 || px*w + i1 >= ne01) {
                            ((ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ)[i] = 0.0f;
                        } ಇಲ್ಲದಿದ್ದರೆ {
                            ((ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ)[i] = ((ದಶಮಾಂಶ *) src0->ದತ್ತಾಂಶ)[j];
                        }
                    }
                }
            }
        }
    }
}

ಶೂನ್ಯ ggml_compute_forward_win_part(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_win_part_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_win_unpart

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_win_unpart_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {
    GGML_ಬಳಸದ(params);

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)

    ಸ್ಥಿರಾಂಕ int32_t w = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[0];

    // padding
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ px = (w - ne1%w)%w;
    //const int py = (w - ne2%w)%w;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ npx = (px + ne1)/w;
    //const int npy = (py + ne2)/w;

    ಪ್ರತಿಜ್ಞಾ(ne0 == ne00);

    // TODO: optimize / multi-thread
    ಗಾಗಿ (int64_t i2 = 0; i2 < ne2; ++i2) {
        ಗಾಗಿ (int64_t i1 = 0; i1 < ne1; ++i1) {
            ಗಾಗಿ (int64_t i0 = 0; i0 < ne0; ++i0) {
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ip2 = i2/w;
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ip1 = i1/w;

                ಸ್ಥಿರಾಂಕ int64_t i02 = i2%w;
                ಸ್ಥಿರಾಂಕ int64_t i01 = i1%w;
                ಸ್ಥಿರಾಂಕ int64_t i00 = i0;

                ಸ್ಥಿರಾಂಕ int64_t i = (ip2*npx + ip1)*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00 + i00;
                ಸ್ಥಿರಾಂಕ int64_t j =                                  i2*ne1*ne0    + i1*ne0   + i0;

                ((ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ)[j] = ((ದಶಮಾಂಶ *) src0->ದತ್ತಾಂಶ)[i];
            }
        }
    }
}

ಶೂನ್ಯ ggml_compute_forward_win_unpart(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_win_unpart_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

//ggml_compute_forward_unary

ಶೂನ್ಯ ggml_compute_forward_unary(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_unary_op op = ggml_get_unary_op(dst);

    ಆಯ್ಕೆ (op) {
        ಸ್ಥಿತಿ GGML_UNARY_OP_ABS:
            {
                ggml_compute_forward_abs(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_UNARY_OP_SGN:
            {
                ggml_compute_forward_sgn(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_UNARY_OP_NEG:
            {
                ggml_compute_forward_neg(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_UNARY_OP_ಹೆಜ್ಜೆ:
            {
                ggml_compute_forward_ಹೆಜ್ಜೆ(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_UNARY_OP_TANH:
            {
                ggml_compute_forward_tanh(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_UNARY_OP_ELU:
            {
                ggml_compute_forward_elu(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_UNARY_OP_RELU:
            {
                ggml_compute_forward_relu(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_UNARY_OP_SIGMOID:
            {
                ggml_compute_forward_sigmoid(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_UNARY_OP_GELU:
            {
                ggml_compute_forward_gelu(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_UNARY_OP_GELU_ERF:
            {
                ggml_compute_forward_gelu_erf(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_UNARY_OP_GELU_ಶೀಘ್ರ:
            {
                ggml_compute_forward_gelu_ಶೀಘ್ರ(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_UNARY_OP_SILU:
            {
                ggml_compute_forward_silu(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_UNARY_OP_HARDSWISH:
            {
                ggml_compute_forward_hardswish(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_UNARY_OP_HARDSIGMOID:
            {
                ggml_compute_forward_hardsigmoid(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_UNARY_OP_EXP:
            {
                ggml_compute_forward_exp(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_UNARY_OP_FLOOR:
            {
                ggml_compute_forward_floor(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_UNARY_OP_CEIL:
            {
                ggml_compute_forward_ceil(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_UNARY_OP_ಸುತ್ತು:
            {
                ggml_compute_forward_ಸುತ್ತು(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_UNARY_OP_TRUNC:
            {
                ggml_compute_forward_trunc(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_UNARY_OP_XIELU:
            {
                ggml_compute_forward_xielu(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_UNARY_OP_EXPM1:
            {
                ggml_compute_forward_expm1(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_UNARY_OP_SOFTPLUS:
            {
                ggml_compute_forward_softplus(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

//ggml_compute_forward_glu

ಶೂನ್ಯ ggml_compute_forward_glu(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_glu_op op = ggml_get_glu_op(dst);

    ಆಯ್ಕೆ (op) {
        ಸ್ಥಿತಿ GGML_GLU_OP_REGLU:
            {
                ggml_compute_forward_reglu(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_GLU_OP_GEGLU:
            {
                ggml_compute_forward_geglu(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_GLU_OP_SWIGLU:
            {
                ggml_compute_forward_swiglu(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_GLU_OP_SWIGLU_OAI:
            {
                ggml_compute_forward_swiglu_oai(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_GLU_OP_GEGLU_ERF:
            {
                ggml_compute_forward_geglu_erf(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_GLU_OP_GEGLU_ಶೀಘ್ರ:
            {
                ggml_compute_forward_geglu_ಶೀಘ್ರ(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_GLU_OP_SWIGLU_CLAMP:
            {
                ggml_compute_forward_swiglu_clamp(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_get_rel_pos

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_get_rel_pos_f16(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {
    GGML_ಬಳಸದ(params);

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L292-L322

    GGML_TENSOR_UNARY_OP_LOCALS

    ಸ್ಥಿರಾಂಕ int64_t w = ne1;

    ggml_fp16_t * src0_ದತ್ತಾಂಶ = (ggml_fp16_t *) src0->ದತ್ತಾಂಶ;
    ggml_fp16_t * dst_ದತ್ತಾಂಶ  = (ggml_fp16_t *) dst->ದತ್ತಾಂಶ;

    ಗಾಗಿ (int64_t i2 = 0; i2 < ne2; ++i2) {
        ಗಾಗಿ (int64_t i1 = 0; i1 < ne1; ++i1) {
            ಸ್ಥಿರಾಂಕ int64_t pos = (w - i1 - 1) + i2;
            ಗಾಗಿ (int64_t i0 = 0; i0 < ne0; ++i0) {
                dst_ದತ್ತಾಂಶ[i2*ne1*ne0 + i1*ne0 + i0] = src0_ದತ್ತಾಂಶ[pos*ne00 + i0];
            }
        }
    }
}

ಶೂನ್ಯ ggml_compute_forward_get_rel_pos(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F16:
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
            {
                ggml_compute_forward_get_rel_pos_f16(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_add_rel_pos

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_ಕೂಡಿಸು_rel_pos_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src2 = dst->src[2];

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ inplace = (ತಾರ್ಕಿಕ) ((int32_t *) dst->op_params)[0];
    ಆದರೆ (!inplace) {
        ಆದರೆ (params->ith == 0) {
            ಸ್ಮರಣೆ_ನಕಲು((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ, (ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ, ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }
    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L357-L359

    ದಶಮಾಂಶ * src1_ದತ್ತಾಂಶ = (ದಶಮಾಂಶ *) src1->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * src2_ದತ್ತಾಂಶ = (ದಶಮಾಂಶ *) src2->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * dst_ದತ್ತಾಂಶ  = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;

    ಸ್ಥಿರಾಂಕ int64_t ne10 = src1->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t ne11 = src1->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t ne12 = src1->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t ne13 = src1->ne[3];

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    // total patches in dst
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ np = ne13;

    // patches per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dp = (np + nth - 1)/nth;

    // patch range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ip0 = dp*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ip1 = MIN(ip0 + dp, np);

    ಗಾಗಿ (int64_t i13 = ip0; i13 < ip1; ++i13) {
        ಗಾಗಿ (int64_t i12 = 0; i12 < ne12; ++i12) {
            ಗಾಗಿ (int64_t i11 = 0; i11 < ne11; ++i11) {
                ಸ್ಥಿರಾಂಕ int64_t jp1 = i13*ne12*ne11*ne10 + i12*ne11*ne10 + i11*ne10;
                ಗಾಗಿ (int64_t i10 = 0; i10 < ne10; ++i10) {
                    ಸ್ಥಿರಾಂಕ int64_t jp0  = jp1 + i10;
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ src1_e = src1_ದತ್ತಾಂಶ[jp0];
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ src2_e = src2_ದತ್ತಾಂಶ[jp0];

                    ಸ್ಥಿರಾಂಕ int64_t jdh = jp0 * ne10;
                    ಸ್ಥಿರಾಂಕ int64_t jdw = jdh - (ne10 - 1) * i10;

                    ಗಾಗಿ (int64_t j = 0; j < ne10; ++j) {
                        dst_ದತ್ತಾಂಶ[jdh + j     ] += src2_e;
                        dst_ದತ್ತಾಂಶ[jdw + j*ne10] += src1_e;
                    }
                }
            }
        }
    }
}

ಶೂನ್ಯ ggml_compute_forward_ಕೂಡಿಸು_rel_pos(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_ಕೂಡಿಸು_rel_pos_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_rwkv_wkv6

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_rwkv_wkv6_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ int64_t T = dst->src[1]->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t C = dst->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t HEADS = dst->src[1]->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t n_seqs = dst->src[5]->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t ತಲೆ_ಗಾತ್ರ = C / HEADS;

    ದಶಮಾಂಶ * dst_ದತ್ತಾಂಶ = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * state = ((ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ) + C * T;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ h_ಆರಂಭ =  (HEADS * (ith    )) / nth;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ h_ಕೊನೆ   = ((HEADS * (ith + 1)) / nth < HEADS) ?
                         (HEADS * (ith + 1)) / nth : HEADS;

    ದಶಮಾಂಶ * k =          (ದಶಮಾಂಶ *) dst->src[0]->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * v =          (ದಶಮಾಂಶ *) dst->src[1]->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * r =          (ದಶಮಾಂಶ *) dst->src[2]->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * ಸಮಯ_faaaa = (ದಶಮಾಂಶ *) dst->src[3]->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * ಸಮಯ_decay = (ದಶಮಾಂಶ *) dst->src[4]->ದತ್ತಾಂಶ;

    size_t t_stride = HEADS * ತಲೆ_ಗಾತ್ರ; // Same to C

    size_t h_stride = C / HEADS;
    GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
    size_t h_stride_2d = ತಲೆ_ಗಾತ್ರ * ತಲೆ_ಗಾತ್ರ;

    ಆದರೆ (ith == 0) {
        memset(dst_ದತ್ತಾಂಶ, 0, T * C * ಗಾತ್ರ(ದಶಮಾಂಶ));
    }
    ggml_barrier(params->threadpool);


    #ಆದರೆ defined(__AVX__) && !defined(__AVX512F__)
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32X GGML_F32x8
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32X_SET1 GGML_F32x8_SET1
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_X_ತುಂಬು GGML_F32x8_ತುಂಬು
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32X_STORE GGML_F32x8_STORE
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32X_MUL GGML_F32x8_MUL
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32X_FMA GGML_F32x8_FMA
        #ವ್ಯಾಖ್ಯಾನಿಸು WKV_ಸದಿಶ_ಗಾತ್ರ 8
    #ಅಥವಾಆದರೆ defined(__AVX512F__)
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32X GGML_F32x16
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32X_SET1 GGML_F32x16_SET1
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_X_ತುಂಬು GGML_F32x16_ತುಂಬು
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32X_STORE GGML_F32x16_STORE
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32X_MUL GGML_F32x16_MUL
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32X_FMA GGML_F32x16_FMA
        #ವ್ಯಾಖ್ಯಾನಿಸು WKV_ಸದಿಶ_ಗಾತ್ರ 16
    #ಅಥವಾಆದರೆ defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32X GGML_F32xt
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32X_SET1 GGML_F32xt_SET1
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_X_ತುಂಬು GGML_F32xt_ತುಂಬು
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32X_STORE GGML_F32xt_STORE
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32X_MUL GGML_F32xt_MUL
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32X_FMA GGML_F32xt_FMA
        #ವ್ಯಾಖ್ಯಾನಿಸು WKV_ಸದಿಶ_ಗಾತ್ರ 8
    #ಅಥವಾಆದರೆ defined(__ARM_NEON) && defined(__aarch64__)
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32X GGML_F32x4
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32X_SET1 GGML_F32x4_SET1
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_X_ತುಂಬು GGML_F32x4_ತುಂಬು
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32X_STORE GGML_F32x4_STORE
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32X_MUL GGML_F32x4_MUL
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32X_FMA GGML_F32x4_FMA
        #ವ್ಯಾಖ್ಯಾನಿಸು WKV_ಸದಿಶ_ಗಾತ್ರ 4
    #ಆದರೆ_ಅಂತ್ಯ

    #ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ WKV_ಸದಿಶ_ಗಾತ್ರ
        ಪೂರ್ಣಾಂಕ wkv_ಸದಿಶ_ಗಾತ್ರ;
        #ಆದರೆ defined(__ARM_FEATURE_SVE)
            wkv_ಸದಿಶ_ಗಾತ್ರ = svcntw();
        #ಇಲ್ಲದಿದ್ದರೆ
            wkv_ಸದಿಶ_ಗಾತ್ರ = WKV_ಸದಿಶ_ಗಾತ್ರ;
        #ಆದರೆ_ಅಂತ್ಯ
        ಸ್ಥಿರಾಂಕ int64_t vec_ಎಣಿಕೆ = ತಲೆ_ಗಾತ್ರ / wkv_ಸದಿಶ_ಗಾತ್ರ;

        ಗಾಗಿ (int64_t t = 0; t < T; t++) {
            size_t t_ಸ್ಥಾನಾಂತರ = t * t_stride;
            size_t state_ಸ್ಥಾನಾಂತರ = ತಲೆ_ಗಾತ್ರ * C * (t / (T / n_seqs));
            ದಶಮಾಂಶ * state_cur = state + state_ಸ್ಥಾನಾಂತರ;
            ದಶಮಾಂಶ * state_prev = t % (T / n_seqs) ? state_cur : (ದಶಮಾಂಶ*)dst->src[5]->ದತ್ತಾಂಶ + state_ಸ್ಥಾನಾಂತರ;

            ಗಾಗಿ (int64_t h = h_ಆರಂಭ; h < h_ಕೊನೆ; h++) {
                size_t h_ಸ್ಥಾನಾಂತರ = h * h_stride;
                size_t t_h_ಸ್ಥಾನಾಂತರ = t_ಸ್ಥಾನಾಂತರ + h_ಸ್ಥಾನಾಂತರ;
                size_t h_2d_ಸ್ಥಾನಾಂತರ = h * h_stride_2d;

                ಗಾಗಿ (int64_t i = 0; i < ತಲೆ_ಗಾತ್ರ; i++) {
                    size_t t_h_i_ಸ್ಥಾನಾಂತರ = t_h_ಸ್ಥಾನಾಂತರ + i;
                    size_t h_i_ಸ್ಥಾನಾಂತರ = h_ಸ್ಥಾನಾಂತರ + i;
                    size_t h_2d_i_ಸ್ಥಾನಾಂತರ = h_2d_ಸ್ಥಾನಾಂತರ + i * h_stride;

                    ದಶಮಾಂಶ k_val = k[t_h_i_ಸ್ಥಾನಾಂತರ];
                    ದಶಮಾಂಶ r_val = r[t_h_i_ಸ್ಥಾನಾಂತರ];
                    ದಶಮಾಂಶ ಸಮಯ_faaaa_val = ಸಮಯ_faaaa[h_i_ಸ್ಥಾನಾಂತರ];
                    ದಶಮಾಂಶ ಸಮಯ_decay_val = ಸಮಯ_decay[t_h_i_ಸ್ಥಾನಾಂತರ];

                    // Broadcast scalar values to vectors
                    GGML_F32X k_vec = GGML_F32X_SET1(k_val);
                    GGML_F32X r_vec = GGML_F32X_SET1(r_val);
                    GGML_F32X ಸಮಯ_faaaa_vec = GGML_F32X_SET1(ಸಮಯ_faaaa_val);
                    GGML_F32X ಸಮಯ_decay_vec = GGML_F32X_SET1(ಸಮಯ_decay_val);

                    ಗಾಗಿ (int64_t j = 0; j < vec_ಎಣಿಕೆ; j++) {
                        size_t ತಳ_j = j * wkv_ಸದಿಶ_ಗಾತ್ರ;
                        size_t t_h_j_ಸ್ಥಾನಾಂತರ = t_h_ಸ್ಥಾನಾಂತರ + ತಳ_j;
                        size_t h_2d_i_j_ಸ್ಥಾನಾಂತರ = h_2d_i_ಸ್ಥಾನಾಂತರ + ತಳ_j;

                        // Load x elements at once
                        GGML_F32X v_vec = GGML_F32_X_ತುಂಬು(&v[t_h_j_ಸ್ಥಾನಾಂತರ]);
                        GGML_F32X prev_state_vec = GGML_F32_X_ತುಂಬು(&state_prev[h_2d_i_j_ಸ್ಥಾನಾಂತರ]);
                        GGML_F32X dst_vec = GGML_F32_X_ತುಂಬು(&dst_ದತ್ತಾಂಶ[t_h_j_ಸ್ಥಾನಾಂತರ]);

                        // Compute kv = v * k
                        GGML_F32X kv_vec = GGML_F32X_MUL(v_vec, k_vec);

                        // Compute temp = kv * time_faaaa + prev_state
                        GGML_F32X ತಾತ್ಕಾಲಿಕ_vec = GGML_F32X_FMA(prev_state_vec, kv_vec, ಸಮಯ_faaaa_vec);

                        // Update dst: dst += temp * r
                        dst_vec = GGML_F32X_FMA(dst_vec, ತಾತ್ಕಾಲಿಕ_vec, r_vec);
                        GGML_F32X_STORE(&dst_ದತ್ತಾಂಶ[t_h_j_ಸ್ಥಾನಾಂತರ], dst_vec);

                        // Update state: state = prev_state * time_decay + kv
                        GGML_F32X new_state_vec = GGML_F32X_FMA(kv_vec, prev_state_vec, ಸಮಯ_decay_vec);
                        GGML_F32X_STORE(&state_cur[h_2d_i_j_ಸ್ಥಾನಾಂತರ], new_state_vec);
                    }

                    // Handle remaining elements, this will not be used.
                    ಗಾಗಿ (int64_t j = vec_ಎಣಿಕೆ * wkv_ಸದಿಶ_ಗಾತ್ರ; j < ತಲೆ_ಗಾತ್ರ; j++) {
                        size_t t_h_j_ಸ್ಥಾನಾಂತರ = t_h_ಸ್ಥಾನಾಂತರ + j;
                        size_t h_2d_i_j_ಸ್ಥಾನಾಂತರ = h_2d_i_ಸ್ಥಾನಾಂತರ + j;
                        ದಶಮಾಂಶ v_val = v[t_h_j_ಸ್ಥಾನಾಂತರ];
                        ದಶಮಾಂಶ kv_val = v_val * k_val;
                        ದಶಮಾಂಶ prev_state_val = state_prev[h_2d_i_j_ಸ್ಥಾನಾಂತರ];
                        ದಶಮಾಂಶ ತಾತ್ಕಾಲಿಕ_val = kv_val * ಸಮಯ_faaaa_val + prev_state_val;
                        dst_ದತ್ತಾಂಶ[t_h_j_ಸ್ಥಾನಾಂತರ] += ತಾತ್ಕಾಲಿಕ_val * r_val;
                        state_cur[h_2d_i_j_ಸ್ಥಾನಾಂತರ] = prev_state_val * ಸಮಯ_decay_val + kv_val;
                    }
                }
            }
        }

    #ಇಲ್ಲದಿದ್ದರೆ
        // basically fused operations:
        // dst = r @ (time_faaaa * (k @ v) + state),
        // state = time_decay * state + (k @ v),
        // recursive through each token
        ಗಾಗಿ (int64_t t = 0; t < T; t++) {
            size_t t_ಸ್ಥಾನಾಂತರ = t * t_stride;
            size_t state_ಸ್ಥಾನಾಂತರ = ತಲೆ_ಗಾತ್ರ * C * (t / (T / n_seqs));
            ದಶಮಾಂಶ * state_cur = state + state_ಸ್ಥಾನಾಂತರ;
            ದಶಮಾಂಶ * state_prev = t % (T / n_seqs) ? state_cur : (ದಶಮಾಂಶ*)dst->src[5]->ದತ್ತಾಂಶ + state_ಸ್ಥಾನಾಂತರ;

            ಗಾಗಿ (int64_t h = h_ಆರಂಭ; h < h_ಕೊನೆ; h++) {
                size_t h_ಸ್ಥಾನಾಂತರ = h * h_stride;
                size_t t_h_ಸ್ಥಾನಾಂತರ = t_ಸ್ಥಾನಾಂತರ + h_ಸ್ಥಾನಾಂತರ;
                size_t h_2d_ಸ್ಥಾನಾಂತರ = h * h_stride_2d;

                ಗಾಗಿ (int64_t i = 0; i < ತಲೆ_ಗಾತ್ರ; i++) {
                    size_t t_h_i_ಸ್ಥಾನಾಂತರ = t_h_ಸ್ಥಾನಾಂತರ + i;
                    size_t h_i_ಸ್ಥಾನಾಂತರ = h_ಸ್ಥಾನಾಂತರ + i;
                    size_t h_2d_i_ಸ್ಥಾನಾಂತರ = h_2d_ಸ್ಥಾನಾಂತರ + i * h_stride;

                    ದಶಮಾಂಶ k_val = k[t_h_i_ಸ್ಥಾನಾಂತರ];
                    ದಶಮಾಂಶ r_val = r[t_h_i_ಸ್ಥಾನಾಂತರ];
                    ದಶಮಾಂಶ ಸಮಯ_faaaa_val = ಸಮಯ_faaaa[h_i_ಸ್ಥಾನಾಂತರ];
                    // RWKV v6: different time_decay for each token.
                    ದಶಮಾಂಶ ಸಮಯ_decay_val = ಸಮಯ_decay[t_h_i_ಸ್ಥಾನಾಂತರ];

                    ಗಾಗಿ (int64_t j = 0; j < ತಲೆ_ಗಾತ್ರ; j++) {
                        size_t t_h_j_ಸ್ಥಾನಾಂತರ = t_h_ಸ್ಥಾನಾಂತರ + j;
                        size_t h_2d_i_j_ಸ್ಥಾನಾಂತರ = h_2d_i_ಸ್ಥಾನಾಂತರ + j;

                        ದಶಮಾಂಶ v_val = v[t_h_j_ಸ್ಥಾನಾಂತರ];
                        ದಶಮಾಂಶ kv_val = v_val * k_val;
                        ದಶಮಾಂಶ prev_state_val = state_prev[h_2d_i_j_ಸ್ಥಾನಾಂತರ];
                        ದಶಮಾಂಶ ತಾತ್ಕಾಲಿಕ_val = kv_val * ಸಮಯ_faaaa_val + prev_state_val;
                        dst_ದತ್ತಾಂಶ[t_h_j_ಸ್ಥಾನಾಂತರ] += ತಾತ್ಕಾಲಿಕ_val * r_val;
                        state_cur[h_2d_i_j_ಸ್ಥಾನಾಂತರ] = prev_state_val * ಸಮಯ_decay_val + kv_val;
                    }
                }
            }
        }
    #ಆದರೆ_ಅಂತ್ಯ
}


ಶೂನ್ಯ ggml_compute_forward_rwkv_wkv6(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_rwkv_wkv6_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_gla

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_gla_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ int64_t T = dst->src[1]->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t C = dst->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t HEADS = dst->src[1]->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t n_seqs = dst->src[4]->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t ತಲೆ_ಗಾತ್ರ = C / HEADS;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale = ggml_get_op_params_f32(dst, 0);

    ದಶಮಾಂಶ * dst_ದತ್ತಾಂಶ = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * state = ((ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ) + C * T;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ h_ಆರಂಭ =  (HEADS * (ith    )) / nth;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ h_ಕೊನೆ   = ((HEADS * (ith + 1)) / nth < HEADS) ?
                         (HEADS * (ith + 1)) / nth : HEADS;

    ದಶಮಾಂಶ * k = (ದಶಮಾಂಶ *) dst->src[0]->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * v = (ದಶಮಾಂಶ *) dst->src[1]->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * q = (ದಶಮಾಂಶ *) dst->src[2]->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * g = (ದಶಮಾಂಶ *) dst->src[3]->ದತ್ತಾಂಶ;

    size_t t_stride = HEADS * ತಲೆ_ಗಾತ್ರ; // Same to C

    size_t h_stride = C / HEADS;
    GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
    size_t h_stride_2d = ತಲೆ_ಗಾತ್ರ * ತಲೆ_ಗಾತ್ರ;

    ಆದರೆ (ith == 0) {
        memset(dst_ದತ್ತಾಂಶ, 0, T * C * ಗಾತ್ರ(ದಶಮಾಂಶ));
    }
    ggml_barrier(params->threadpool);


    #ಆದರೆ defined(__AVX__) && !defined(__AVX512F__)
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32X GGML_F32x8
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32X_SET1 GGML_F32x8_SET1
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_X_ತುಂಬು GGML_F32x8_ತುಂಬು
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32X_STORE GGML_F32x8_STORE
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32X_MUL GGML_F32x8_MUL
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32X_FMA GGML_F32x8_FMA
        #ವ್ಯಾಖ್ಯಾನಿಸು GLA_ಸದಿಶ_ಗಾತ್ರ 8
    #ಅಥವಾಆದರೆ defined(__AVX512F__)
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32X GGML_F32x16
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32X_SET1 GGML_F32x16_SET1
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_X_ತುಂಬು GGML_F32x16_ತುಂಬು
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32X_STORE GGML_F32x16_STORE
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32X_MUL GGML_F32x16_MUL
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32X_FMA GGML_F32x16_FMA
        #ವ್ಯಾಖ್ಯಾನಿಸು GLA_ಸದಿಶ_ಗಾತ್ರ 16
    #ಅಥವಾಆದರೆ defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32X GGML_F32xt
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32X_SET1 GGML_F32xt_SET1
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_X_ತುಂಬು GGML_F32xt_ತುಂಬು
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32X_STORE GGML_F32xt_STORE
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32X_MUL GGML_F32xt_MUL
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32X_FMA GGML_F32xt_FMA
        #ವ್ಯಾಖ್ಯಾನಿಸು GLA_ಸದಿಶ_ಗಾತ್ರ 8
    #ಅಥವಾಆದರೆ defined(__ARM_NEON) && defined(__aarch64__)
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32X GGML_F32x4
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32X_SET1 GGML_F32x4_SET1
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_X_ತುಂಬು GGML_F32x4_ತುಂಬು
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32X_STORE GGML_F32x4_STORE
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32X_MUL GGML_F32x4_MUL
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32X_FMA GGML_F32x4_FMA
        #ವ್ಯಾಖ್ಯಾನಿಸು GLA_ಸದಿಶ_ಗಾತ್ರ 4
    #ಆದರೆ_ಅಂತ್ಯ

    #ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GLA_ಸದಿಶ_ಗಾತ್ರ
        ಪೂರ್ಣಾಂಕ gla_ಸದಿಶ_ಗಾತ್ರ;
        #ಆದರೆ defined(__ARM_FEATURE_SVE)
            gla_ಸದಿಶ_ಗಾತ್ರ = svcntw();
        #ಇಲ್ಲದಿದ್ದರೆ
            gla_ಸದಿಶ_ಗಾತ್ರ = GLA_ಸದಿಶ_ಗಾತ್ರ;
        #ಆದರೆ_ಅಂತ್ಯ
        ಸ್ಥಿರಾಂಕ int64_t vec_ಎಣಿಕೆ = ತಲೆ_ಗಾತ್ರ / gla_ಸದಿಶ_ಗಾತ್ರ;

        ಗಾಗಿ (int64_t t = 0; t < T; t++) {
            size_t t_ಸ್ಥಾನಾಂತರ = t * t_stride;
            size_t state_ಸ್ಥಾನಾಂತರ = ತಲೆ_ಗಾತ್ರ * C * (t / (T / n_seqs));
            ದಶಮಾಂಶ * state_cur = state + state_ಸ್ಥಾನಾಂತರ;
            ದಶಮಾಂಶ * state_prev = t % (T / n_seqs) ? state_cur : (ದಶಮಾಂಶ*)dst->src[4]->ದತ್ತಾಂಶ + state_ಸ್ಥಾನಾಂತರ;

            ಗಾಗಿ (int64_t h = h_ಆರಂಭ; h < h_ಕೊನೆ; h++) {
                size_t h_ಸ್ಥಾನಾಂತರ = h * h_stride;
                size_t t_h_ಸ್ಥಾನಾಂತರ = t_ಸ್ಥಾನಾಂತರ + h_ಸ್ಥಾನಾಂತರ;
                size_t h_2d_ಸ್ಥಾನಾಂತರ = h * h_stride_2d;

                ಗಾಗಿ (int64_t i = 0; i < ತಲೆ_ಗಾತ್ರ; i++) {
                    size_t t_h_i_ಸ್ಥಾನಾಂತರ = t_h_ಸ್ಥಾನಾಂತರ + i;
                    size_t h_2d_i_ಸ್ಥಾನಾಂತರ = h_2d_ಸ್ಥಾನಾಂತರ + i * h_stride;

                    ದಶಮಾಂಶ k_val = k[t_h_i_ಸ್ಥಾನಾಂತರ];
                    ದಶಮಾಂಶ q_val = q[t_h_i_ಸ್ಥಾನಾಂತರ] * scale;
                    ದಶಮಾಂಶ g_val = g[t_h_i_ಸ್ಥಾನಾಂತರ];

                    // Broadcast scalar values to vectors
                    GGML_F32X k_vec = GGML_F32X_SET1(k_val);
                    GGML_F32X q_vec = GGML_F32X_SET1(q_val);
                    GGML_F32X g_vec = GGML_F32X_SET1(g_val);

                    ಗಾಗಿ (int64_t j = 0; j < vec_ಎಣಿಕೆ; j++) {
                        size_t ತಳ_j = j * gla_ಸದಿಶ_ಗಾತ್ರ;
                        size_t t_h_j_ಸ್ಥಾನಾಂತರ = t_h_ಸ್ಥಾನಾಂತರ + ತಳ_j;
                        size_t h_2d_i_j_ಸ್ಥಾನಾಂತರ = h_2d_i_ಸ್ಥಾನಾಂತರ + ತಳ_j;

                        // Load x elements at once
                        GGML_F32X v_vec = GGML_F32_X_ತುಂಬು(&v[t_h_j_ಸ್ಥಾನಾಂತರ]);
                        GGML_F32X prev_state_vec = GGML_F32_X_ತುಂಬು(&state_prev[h_2d_i_j_ಸ್ಥಾನಾಂತರ]);
                        GGML_F32X dst_vec = GGML_F32_X_ತುಂಬು(&dst_ದತ್ತಾಂಶ[t_h_j_ಸ್ಥಾನಾಂತರ]);

                        // Compute kv = v * k
                        GGML_F32X kv_vec = GGML_F32X_MUL(v_vec, k_vec);

                        // Compute temp = prev_state * g + kv
                        GGML_F32X ತಾತ್ಕಾಲಿಕ_vec = GGML_F32X_FMA(kv_vec, prev_state_vec, g_vec);

                        // Update dst: dst += temp * q
                        dst_vec = GGML_F32X_FMA(dst_vec, ತಾತ್ಕಾಲಿಕ_vec, q_vec);
                        GGML_F32X_STORE(&dst_ದತ್ತಾಂಶ[t_h_j_ಸ್ಥಾನಾಂತರ], dst_vec);

                        // Update state
                        GGML_F32X_STORE(&state_cur[h_2d_i_j_ಸ್ಥಾನಾಂತರ], ತಾತ್ಕಾಲಿಕ_vec);
                    }

                    // Handle remaining elements, this will not be used.
                    ಗಾಗಿ (int64_t j = vec_ಎಣಿಕೆ * gla_ಸದಿಶ_ಗಾತ್ರ; j < ತಲೆ_ಗಾತ್ರ; j++) {
                        size_t t_h_j_ಸ್ಥಾನಾಂತರ = t_h_ಸ್ಥಾನಾಂತರ + j;
                        size_t h_2d_i_j_ಸ್ಥಾನಾಂತರ = h_2d_i_ಸ್ಥಾನಾಂತರ + j;
                        ದಶಮಾಂಶ v_val = v[t_h_j_ಸ್ಥಾನಾಂತರ];
                        ದಶಮಾಂಶ kv_val = v_val * k_val;
                        ದಶಮಾಂಶ prev_state_val = state_prev[h_2d_i_j_ಸ್ಥಾನಾಂತರ];
                        ದಶಮಾಂಶ ತಾತ್ಕಾಲಿಕ_val = kv_val + prev_state_val * g_val;
                        dst_ದತ್ತಾಂಶ[t_h_j_ಸ್ಥಾನಾಂತರ] += ತಾತ್ಕಾಲಿಕ_val * q_val;
                        state_cur[h_2d_i_j_ಸ್ಥಾನಾಂತರ] = ತಾತ್ಕಾಲಿಕ_val;
                    }
                }
            }
        }

    #ಇಲ್ಲದಿದ್ದರೆ
        ಗಾಗಿ (int64_t t = 0; t < T; t++) {
            size_t t_ಸ್ಥಾನಾಂತರ = t * t_stride;
            size_t state_ಸ್ಥಾನಾಂತರ = ತಲೆ_ಗಾತ್ರ * C * (t / (T / n_seqs));
            ದಶಮಾಂಶ * state_cur = state + state_ಸ್ಥಾನಾಂತರ;
            ದಶಮಾಂಶ * state_prev = t % (T / n_seqs) ? state_cur : (ದಶಮಾಂಶ*)dst->src[4]->ದತ್ತಾಂಶ + state_ಸ್ಥಾನಾಂತರ;

            ಗಾಗಿ (int64_t h = h_ಆರಂಭ; h < h_ಕೊನೆ; h++) {
                size_t h_ಸ್ಥಾನಾಂತರ = h * h_stride;
                size_t t_h_ಸ್ಥಾನಾಂತರ = t_ಸ್ಥಾನಾಂತರ + h_ಸ್ಥಾನಾಂತರ;
                size_t h_2d_ಸ್ಥಾನಾಂತರ = h * h_stride_2d;

                ಗಾಗಿ (int64_t i = 0; i < ತಲೆ_ಗಾತ್ರ; i++) {
                    size_t t_h_i_ಸ್ಥಾನಾಂತರ = t_h_ಸ್ಥಾನಾಂತರ + i;
                    size_t h_2d_i_ಸ್ಥಾನಾಂತರ = h_2d_ಸ್ಥಾನಾಂತರ + i * h_stride;

                    ದಶಮಾಂಶ k_val = k[t_h_i_ಸ್ಥಾನಾಂತರ];
                    ದಶಮಾಂಶ q_val = q[t_h_i_ಸ್ಥಾನಾಂತರ] * scale;
                    ದಶಮಾಂಶ g_val = g[t_h_i_ಸ್ಥಾನಾಂತರ];

                    ಗಾಗಿ (int64_t j = 0; j < ತಲೆ_ಗಾತ್ರ; j++) {
                        size_t t_h_j_ಸ್ಥಾನಾಂತರ = t_h_ಸ್ಥಾನಾಂತರ + j;
                        size_t h_2d_i_j_ಸ್ಥಾನಾಂತರ = h_2d_i_ಸ್ಥಾನಾಂತರ + j;

                        ದಶಮಾಂಶ v_val = v[t_h_j_ಸ್ಥಾನಾಂತರ];
                        ದಶಮಾಂಶ kv_val = v_val * k_val;
                        ದಶಮಾಂಶ prev_state_val = state_prev[h_2d_i_j_ಸ್ಥಾನಾಂತರ];
                        ದಶಮಾಂಶ ತಾತ್ಕಾಲಿಕ_val = prev_state_val * g_val + kv_val;
                        dst_ದತ್ತಾಂಶ[t_h_j_ಸ್ಥಾನಾಂತರ] += ತಾತ್ಕಾಲಿಕ_val * q_val;
                        state_cur[h_2d_i_j_ಸ್ಥಾನಾಂತರ] = ತಾತ್ಕಾಲಿಕ_val;
                    }
                }
            }
        }
    #ಆದರೆ_ಅಂತ್ಯ
}


ಶೂನ್ಯ ggml_compute_forward_gla(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_gla_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_solve_tri_f32(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0 = dst->src[0];  // A (lower triangular)
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src1 = dst->src[1];  // B (RHS)

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS;

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(ne00 == ne01); // A must be square
    GGML_ASSERT(ne0  == ne10); // solution cols == B cols
    GGML_ASSERT(ne1  == ne11); // solution rows == B rows

    GGML_ASSERT(ne02 == ne12 && ne12 == ne2);
    GGML_ASSERT(ne03 == ne13 && ne13 == ne3);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ int64_t k = ne10;   // number of RHS columns
    ಸ್ಥಿರಾಂಕ int64_t n = ne11;   // A is n×n
    ಸ್ಥಿರಾಂಕ int64_t nr = ne02 * ne03 * k; // we're parallelizing on columns here, so seq x token x column will be the unit

    // chunks per thread
    ಸ್ಥಿರಾಂಕ int64_t dr = (nr + nth - 1)/nth;

    // chunk range for this thread
    ಸ್ಥಿರಾಂಕ int64_t ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ int64_t ir1 = MIN(ir0 + dr, nr);

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * A = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src0->ದತ್ತಾಂಶ;  // [n, n, B1, B2]
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * B = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src1->ದತ್ತಾಂಶ;  // [n, k, B1, B2]
          ದಶಮಾಂಶ * X = (      ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;   // [n, k, B1, B2]

    ಗಾಗಿ (int64_t ir = ir0; ir < ir1; ++ir) {
        ಸ್ಥಿರಾಂಕ int64_t i03 = ir/(ne02*k);
        ಸ್ಥಿರಾಂಕ int64_t i02 = (ir - i03*ne02*k)/k;
        ಸ್ಥಿರಾಂಕ int64_t i01 = (ir - i03*ne02*k - i02*k);

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * A_batch = A + i02 * nb02 / ಗಾತ್ರ(ದಶಮಾಂಶ) + i03 * nb03 / ಗಾತ್ರ(ದಶಮಾಂಶ);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * B_batch = B + i02 * nb12 / ಗಾತ್ರ(ದಶಮಾಂಶ) + i03 * nb13 / ಗಾತ್ರ(ದಶಮಾಂಶ);

        ದಶಮಾಂಶ * X_batch = X + i02 * nb2 / ಗಾತ್ರ(ದಶಮಾಂಶ) + i03 * nb3 / ಗಾತ್ರ(ದಶಮಾಂಶ);

        ಗಾಗಿ (int64_t i00 = 0; i00 < n; ++i00) {
            ದಶಮಾಂಶ ಮೊತ್ತ = 0.0f;
            ಗಾಗಿ (int64_t t = 0; t < i00; ++t) {
                ಮೊತ್ತ += A_batch[i00 * n + t] * X_batch[t * k + i01];
            }

            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ diag = A_batch[i00 * n + i00];
            ಪ್ರತಿಜ್ಞಾ(diag != 0.0f && "Zero diagonal in triangular matrix");

            X_batch[i00 * k + i01] = (B_batch[i00 * k + i01] - ಮೊತ್ತ) / diag;
        }
    }
}

ಶೂನ್ಯ ggml_compute_forward_solve_tri(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    ಆದರೆ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        ggml_compute_forward_solve_tri_f32(params, dst);
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_ABORT("fatal error");
    }
}

// ggml_compute_forward_gated_delta_net
ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_gated_delta_net_one_chunk(
    ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
    ggml_tensor * dst,
    int64_t ir0,
    int64_t ir1) {

    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    ಸ್ಥಿರಾಂಕ int64_t S_v      = src_v->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t H        = src_v->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t n_ಟೋಕನುಗಳು = src_v->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t n_seqs   = src_v->ne[3];

    GGML_ASSERT(ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(src_q));
    GGML_ASSERT(ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(src_k));
    GGML_ASSERT(ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(src_v));
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

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ kda = (neg0 == S_v);

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    ಸ್ಥಿರಾಂಕ int64_t K = ggml_get_op_params_i32(dst, 0);
    GGML_ASSERT(K >= 1);
    // per-seq stride in floats (seq s starts at state + s * seq_stride)
    ಸ್ಥಿರಾಂಕ int64_t state_seq_stride = src_state->nb[3] / ಗಾತ್ರ(ದಶಮಾಂಶ);

    ಸ್ಥಿರಾಂಕ int64_t ಪ್ರತಿ_thread = S_v + (K > 1 ? S_v * S_v : 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;

    ದಶಮಾಂಶ * delta       = (ದಶಮಾಂಶ *)params->wdata + ith * ಪ್ರತಿ_thread + CACHE_ಸಾಲು_ಗಾತ್ರ_F32;
    ದಶಮಾಂಶ * state_work  = K > 1 ? (delta + S_v) : ಶೂನ್ಯಸೂಚಕ;

    // output layout: [attn_scores | new_states]
    // attn_scores: S_v * H * n_tokens * n_seqs    floats
    // new_states:  S_v * S_v * H * n_seqs * K     floats  (K snapshot slots; last min(n_tokens, K))
    ಸ್ಥಿರಾಂಕ int64_t attn_score_elems    = S_v * H * n_ಟೋಕನುಗಳು * n_seqs;
    ಸ್ಥಿರಾಂಕ int64_t state_ಗಾತ್ರ_ಪ್ರತಿ_snap = S_v * S_v * H * n_seqs;
    ದಶಮಾಂಶ * attn_ಹೊರಹೋಗು_ತಳ  = (ದಶಮಾಂಶ *)dst->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * state_ಹೊರಹೋಗು_ತಳ = (ದಶಮಾಂಶ *)dst->ದತ್ತಾಂಶ + attn_score_elems;

    // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
    // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * state_ಒಳ_ತಳ = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)src_state->ದತ್ತಾಂಶ;

  //const int64_t rq1 = nev1 / neq1;
  //const int64_t rk1 = nev1 / nek1;
    ಸ್ಥಿರಾಂಕ int64_t rq3 = nev3 / neq3;
    ಸ್ಥಿರಾಂಕ int64_t rk3 = nev3 / nek3;

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale = 1.0f / sqrtf((ದಶಮಾಂಶ) S_v);

    ಗಾಗಿ (int64_t ir = ir0; ir < ir1; ++ir) {
        ಸ್ಥಿರಾಂಕ int64_t iv1 = ir % H; // head_index
        ಸ್ಥಿರಾಂಕ int64_t iv3 = ir / H; // sequence

        ಸ್ಥಿರಾಂಕ int64_t iq1 = iv1 % neq1;
        ಸ್ಥಿರಾಂಕ int64_t ik1 = iv1 % nek1;

        ಸ್ಥಿರಾಂಕ int64_t iq3 = iv3 / rq3;
        ಸ್ಥಿರಾಂಕ int64_t ik3 = iv3 / rk3;

        // For K=1, write directly to the single output slot to avoid an extra memcpy at the end.
        // For K>1, work in scratch and copy out per-token when the slot is in range.
        ದಶಮಾಂಶ * s_ಹೊರಹೋಗು = (K > 1)
            ? state_work
            : state_ಹೊರಹೋಗು_ತಳ + (iv3 * H + iv1) * S_v * S_v;

        // copy input state into the working buffer and operate in-place
        // state layout [S_v, S_v, H, n_seqs]: seq iv3 starts at iv3 * state_seq_stride.
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * s_ಒಳ = state_ಒಳ_ತಳ + iv3 * state_seq_stride + iv1 * S_v * S_v;
        ಸ್ಮರಣೆ_ನಕಲು(s_ಹೊರಹೋಗು, s_ಒಳ, S_v * S_v * ಗಾತ್ರ(ದಶಮಾಂಶ));

        // attn output pointer for first token of this (head, seq)
        ದಶಮಾಂಶ * attn_ದತ್ತಾಂಶ = attn_ಹೊರಹೋಗು_ತಳ + (iv3 * n_ಟೋಕನುಗಳು * H + iv1) * S_v;

        ಗಾಗಿ (int64_t t = 0; t < n_ಟೋಕನುಗಳು; t++) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * q_d = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)src_q->ದತ್ತಾಂಶ + iq3 * nbq3 + t * nbq2 + iq1 * nbq1);
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * k_d = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)src_k->ದತ್ತಾಂಶ + ik3 * nbk3 + t * nbk2 + ik1 * nbk1);
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * v_d = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)src_v->ದತ್ತಾಂಶ + iv3 * nbv3 + t * nbv2 + iv1 * nbv1);

            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ beta_val = *(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)src_beta->ದತ್ತಾಂಶ + iv3 * nbb3 + t * nbb2 + iv1 * nbb1);
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * g_d    =  (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)src_g->ದತ್ತಾಂಶ    + iv3 * nbg3 + t * nbg2 + iv1 * nbg1);

            // state is stored transposed: s_out[j*S_v + i] = S[i][j]
            // so row j of s_out = column j of S (contiguous access)

            ಆದರೆ (kda) {
                // precompute exp(g) into delta scratch (reused below)
                ಗಾಗಿ (int64_t i = 0; i < S_v; ++i) {
                    delta[i] = expf(g_d[i]);
                }
                // S[i][:] *= exp(g[i]) => for each row j of M: M[j][i] *= exp(g[i])
                ಗಾಗಿ (int64_t j = 0; j < S_v; ++j) {
                    ggml_vec_mul_f32(S_v, &s_ಹೊರಹೋಗು[j * S_v], &s_ಹೊರಹೋಗು[j * S_v], delta);
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                ggml_vec_scale_f32(S_v * S_v, s_ಹೊರಹೋಗು, expf(g_d[0]));
            }

            // delta[j] = sum_i S[i][j] * k[i] = dot(row j of M, k)
            ಗಾಗಿ (int64_t j = 0; j < S_v; ++j) {
                ದಶಮಾಂಶ ಮೊತ್ತ = 0.0f;
                ggml_vec_dot_f32(S_v, &ಮೊತ್ತ, 0, &s_ಹೊರಹೋಗು[j * S_v], 0, k_d, 0, 1);
                delta[j] = (v_d[j] - ಮೊತ್ತ) * beta_val;
            }

            // outer product: S[i][j] += k[i] * delta[j] => M[j][i] += delta[j] * k[i]
            ಗಾಗಿ (int64_t j = 0; j < S_v; ++j) {
                ggml_vec_mad_f32(S_v, &s_ಹೊರಹೋಗು[j * S_v], k_d, delta[j]);
            }

            // attn_out[j] = sum_i S[i][j] * q[i] = dot(row j of M, q)
            ಗಾಗಿ (int64_t j = 0; j < S_v; ++j) {
                ದಶಮಾಂಶ ಮೊತ್ತ = 0.0f;
                ggml_vec_dot_f32(S_v, &ಮೊತ್ತ, 0, &s_ಹೊರಹೋಗು[j * S_v], 0, q_d, 0, 1);
                attn_ದತ್ತಾಂಶ[j] = ಮೊತ್ತ * scale;
            }

            attn_ದತ್ತಾಂಶ += S_v * H; // advance to next token

            ಆದರೆ (K > 1) {
                ಸ್ಥಿರಾಂಕ int64_t ಗುರಿ_ಸ್ಥಾನ = n_ಟೋಕನುಗಳು - 1 - t;
                ಆದರೆ (ಗುರಿ_ಸ್ಥಾನ >= 0 && ಗುರಿ_ಸ್ಥಾನ < K) {
                    ದಶಮಾಂಶ * curr_state_o = state_ಹೊರಹೋಗು_ತಳ + ಗುರಿ_ಸ್ಥಾನ * state_ಗಾತ್ರ_ಪ್ರತಿ_snap +
                                     (iv3 * H + iv1) * S_v * S_v;
                    ಸ್ಮರಣೆ_ನಕಲು(curr_state_o, s_ಹೊರಹೋಗು, S_v * S_v * ಗಾತ್ರ(ದಶಮಾಂಶ));
                }
            }
        }
    }
}


ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_gated_delta_net_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ggml_tensor * V = dst->src[2];
    int64_t nr = V->ne[1] * V->ne[3];

    // disable for NUMA
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ disable_chunking = ggml_is_numa();

    ಪೂರ್ಣಾಂಕ nth = params->nth;
    ಪೂರ್ಣಾಂಕ ith = params->ith;

    // 4x chunks per thread
    ಪೂರ್ಣಾಂಕ nth_scaled = nth * 4;
    int64_t chunk_ಗಾತ್ರ = (nr + nth_scaled - 1) / nth_scaled;
    int64_t nchunk     = (nr + chunk_ಗಾತ್ರ - 1) / chunk_ಗಾತ್ರ;

    ಆದರೆ (nth == 1 || nchunk < nth || disable_chunking) {
      nchunk = nth;
    }

    ಆದರೆ (ith == 0) {
      ggml_threadpool_chunk_ಗಣ(params->threadpool, nth);
    }

    ggml_barrier(params->threadpool);

    ಸ್ಥಿರಾಂಕ int64_t dr = (nr + nchunk - 1) / nchunk;

    ಪೂರ್ಣಾಂಕ current_chunk = ith;

    ಆಗ (current_chunk < nchunk) {
        ಸ್ಥಿರಾಂಕ int64_t ir0 = dr * current_chunk;
        ಸ್ಥಿರಾಂಕ int64_t ir1 = MIN(ir0 + dr, nr);

        ggml_compute_forward_gated_delta_net_one_chunk(params, dst, ir0, ir1);
        current_chunk = ggml_threadpool_chunk_ಕೂಡಿಸು(params->threadpool, 1);
    }
}

ಶೂನ್ಯ ggml_compute_forward_gated_delta_net(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_gated_delta_net_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}


// ggml_compute_forward_dsv4_hc_comb

ಸ್ಥಿರ ಶೂನ್ಯ ggml_dsv4_hc_comb_norm_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ(ದಶಮಾಂಶ * comb, ದಶಮಾಂಶ eps) {
    constexpr int64_t hc = 4;

    ಗಾಗಿ (int64_t idst = 0; idst < hc; ++idst) {
        ದಶಮಾಂಶ ಮೊತ್ತ = eps;
        ಗಾಗಿ (int64_t isrc = 0; isrc < hc; ++isrc) {
            ಮೊತ್ತ += comb[idst + hc*isrc];
        }

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ inv_ಮೊತ್ತ = 1.0f / ಮೊತ್ತ;
        ಗಾಗಿ (int64_t isrc = 0; isrc < hc; ++isrc) {
            comb[idst + hc*isrc] *= inv_ಮೊತ್ತ;
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_dsv4_hc_comb_norm_ಅಡ್ಡಸಾಲುಗಳು(ದಶಮಾಂಶ * comb, ದಶಮಾಂಶ eps) {
    constexpr int64_t hc = 4;

    ಗಾಗಿ (int64_t isrc = 0; isrc < hc; ++isrc) {
        ದಶಮಾಂಶ ಮೊತ್ತ = eps;
        ಗಾಗಿ (int64_t idst = 0; idst < hc; ++idst) {
            ಮೊತ್ತ += comb[idst + hc*isrc];
        }

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ inv_ಮೊತ್ತ = 1.0f / ಮೊತ್ತ;
        ಗಾಗಿ (int64_t idst = 0; idst < hc; ++idst) {
            comb[idst + hc*isrc] *= inv_ಮೊತ್ತ;
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_dsv4_hc_comb_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * mixes = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * scale = dst->src[1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * ತಳ  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(ತಳ->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    constexpr int64_t hc = 4;
    constexpr int64_t comb_ಸ್ಥಾನಾಂತರ = 2*hc;
    constexpr int64_t hc_mix_dim = (2 + hc)*hc;

    ಸ್ಥಿರಾಂಕ int64_t n_ಟೋಕನುಗಳು = mixes->ne[1];

    GGML_ASSERT(mixes->ne[0] == hc_mix_dim);
    GGML_ASSERT(dst->ne[0] == hc);
    GGML_ASSERT(dst->ne[1] == hc);
    GGML_ASSERT(dst->ne[2] == n_ಟೋಕನುಗಳು);
    GGML_ASSERT(scale->ne[0] >= 3);
    GGML_ASSERT(ತಳ->ne[0] == hc_mix_dim);

    GGML_TENSOR_LOCALS(size_t, nbm, mixes, nb);
    GGML_TENSOR_LOCALS(size_t, nbs, scale, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, ತಳ,  nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,   nb);

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ eps = ggml_get_op_params_f32(dst, 0);
    ಸ್ಥಿರಾಂಕ int32_t n_iter = ggml_get_op_params_i32(dst, 1);
    GGML_ASSERT(n_iter > 0);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ int64_t dr  = (n_ಟೋಕನುಗಳು + nth - 1) / nth;
    ಸ್ಥಿರಾಂಕ int64_t it0 = dr * ith;
    ಸ್ಥಿರಾಂಕ int64_t it1 = MIN(it0 + dr, n_ಟೋಕನುಗಳು);

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale_comb = *(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) scale->ದತ್ತಾಂಶ + 2*nbs0);

    ಗಾಗಿ (int64_t it = it0; it < it1; ++it) {
        ದಶಮಾಂಶ comb[hc*hc];

        ಗಾಗಿ (int64_t isrc = 0; isrc < hc; ++isrc) {
            ದಶಮಾಂಶ max = -INFINITY;
            ಗಾಗಿ (int64_t idst = 0; idst < hc; ++idst) {
                ಸ್ಥಿರಾಂಕ int64_t idx = idst + hc*isrc;
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ xv = *(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) mixes->ದತ್ತಾಂಶ + (comb_ಸ್ಥಾನಾಂತರ + idx)*nbm0 + it*nbm1);
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ bv = *(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) ತಳ->ದತ್ತಾಂಶ  + (comb_ಸ್ಥಾನಾಂತರ + idx)*nbb0);
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ v = xv * scale_comb + bv;
                comb[idx] = v;
                max = MAX(max, v);
            }

            ದಶಮಾಂಶ ಮೊತ್ತ = 0.0f;
            ಗಾಗಿ (int64_t idst = 0; idst < hc; ++idst) {
                ಸ್ಥಿರಾಂಕ int64_t idx = idst + hc*isrc;
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ v = expf(comb[idx] - max);
                comb[idx] = v;
                ಮೊತ್ತ += v;
            }

            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ inv_ಮೊತ್ತ = 1.0f / ಮೊತ್ತ;
            ಗಾಗಿ (int64_t idst = 0; idst < hc; ++idst) {
                ಸ್ಥಿರಾಂಕ int64_t idx = idst + hc*isrc;
                comb[idx] = comb[idx] * inv_ಮೊತ್ತ + eps;
            }
        }

        ggml_dsv4_hc_comb_norm_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ(comb, eps);
        ಗಾಗಿ (int32_t i = 1; i < n_iter; ++i) {
            ggml_dsv4_hc_comb_norm_ಅಡ್ಡಸಾಲುಗಳು(comb, eps);
            ggml_dsv4_hc_comb_norm_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ(comb, eps);
        }

        ಗಾಗಿ (int64_t isrc = 0; isrc < hc; ++isrc) {
            ಗಾಗಿ (int64_t idst = 0; idst < hc; ++idst) {
                ಸ್ಥಿರಾಂಕ int64_t idx = idst + hc*isrc;
                *(ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + idst*nbd0 + isrc*nbd1 + it*nbd2) = comb[idx];
            }
        }
    }
}

ಶೂನ್ಯ ggml_compute_forward_dsv4_hc_comb(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_dsv4_hc_comb_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_dsv4_hc_pre

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_dsv4_hc_pre_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * x       = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ಸ್ಥಿರಾಂಕ int64_t n_embd   = x->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t hc       = x->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t n_ಟೋಕನುಗಳು = x->ne[2];

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale = ggml_get_op_params_f32(dst, 0);
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ  gated = ggml_get_op_params_i32(dst, 1) != 0;

    GGML_ASSERT(dst->ne[0] == n_embd);
    GGML_ASSERT(dst->ne[1] == n_ಟೋಕನುಗಳು);
    ಆದರೆ (gated) {
        GGML_ASSERT(weights->ne[0] == n_embd);
        GGML_ASSERT(weights->ne[1] == hc);
        GGML_ASSERT(weights->ne[2] == n_ಟೋಕನುಗಳು);
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_ASSERT(weights->ne[0] == hc);
        GGML_ASSERT(weights->ne[1] == n_ಟೋಕನುಗಳು);
    }

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ int64_t nr  = n_embd * n_ಟೋಕನುಗಳು;
    ಸ್ಥಿರಾಂಕ int64_t dr  = (nr + nth - 1) / nth;
    ಸ್ಥಿರಾಂಕ int64_t ir0 = dr * ith;
    ಸ್ಥಿರಾಂಕ int64_t ir1 = MIN(ir0 + dr, nr);

    ಗಾಗಿ (int64_t ir = ir0; ir < ir1; ++ir) {
        ಸ್ಥಿರಾಂಕ int64_t i0 = ir % n_embd;
        ಸ್ಥಿರಾಂಕ int64_t it = ir / n_embd;

        ದಶಮಾಂಶ ಮೊತ್ತ = 0.0f;
        ಗಾಗಿ (int64_t ih = 0; ih < hc; ++ih) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ xv = *(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) x->ದತ್ತಾಂಶ + i0*nbx0 + ih*nbx1 + it*nbx2);
            ದಶಮಾಂಶ wv;
            ಆದರೆ (gated) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ gv = *(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) weights->ದತ್ತಾಂಶ + i0*nbw0 + ih*nbw1 + it*nbw2);
                wv = 1.0f / (1.0f + expf(-gv));
            } ಇಲ್ಲದಿದ್ದರೆ {
                wv = *(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) weights->ದತ್ತಾಂಶ + ih*nbw0 + it*nbw1);
            }
            ಮೊತ್ತ += xv * wv;
        }

        *(ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i0*nbd0 + it*nbd1) = scale * ಮೊತ್ತ;
    }
}

ಶೂನ್ಯ ggml_compute_forward_dsv4_hc_pre(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_dsv4_hc_pre_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_dsv4_hc_post

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_dsv4_hc_post_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * x        = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * residual = dst->src[1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * post     = dst->src[2];
    ಸ್ಥಿರಾಂಕ ggml_tensor * comb     = dst->src[3];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ಸ್ಥಿರಾಂಕ int64_t n_embd   = x->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t n_ಟೋಕನುಗಳು = x->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t hc       = residual->ne[1];

    GGML_ASSERT(dst->ne[0] == n_embd);
    GGML_ASSERT(dst->ne[1] == hc);
    GGML_ASSERT(dst->ne[2] == n_ಟೋಕನುಗಳು);
    GGML_ASSERT(residual->ne[0] == n_embd);
    GGML_ASSERT(residual->ne[2] == n_ಟೋಕನುಗಳು);
    GGML_ASSERT(post->ne[0] == hc);
    GGML_ASSERT(post->ne[1] == n_ಟೋಕನುಗಳು);

    // comb == NULL: identity mixing, each stream keeps its own residual
    size_t nbc0 = 0;
    size_t nbc1 = 0;
    size_t nbc2 = 0;
    ಆದರೆ (comb) {
        GGML_ASSERT(comb->type == GGML_TYPE_F32);
        GGML_ASSERT(comb->ne[0] == hc);
        GGML_ASSERT(comb->ne[1] == hc);
        GGML_ASSERT(comb->ne[2] == n_ಟೋಕನುಗಳು);
        nbc0 = comb->nb[0];
        nbc1 = comb->nb[1];
        nbc2 = comb->nb[2];
    }

    GGML_TENSOR_LOCALS(size_t, nbx, x,        nb);
    GGML_TENSOR_LOCALS(size_t, nbr, residual, nb);
    GGML_TENSOR_LOCALS(size_t, nbp, post,     nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,      nb);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ int64_t nr  = n_embd * hc * n_ಟೋಕನುಗಳು;
    ಸ್ಥಿರಾಂಕ int64_t dr  = (nr + nth - 1) / nth;
    ಸ್ಥಿರಾಂಕ int64_t ir0 = dr * ith;
    ಸ್ಥಿರಾಂಕ int64_t ir1 = MIN(ir0 + dr, nr);

    ಗಾಗಿ (int64_t ir = ir0; ir < ir1; ++ir) {
        ಸ್ಥಿರಾಂಕ int64_t i0     = ir % n_embd;
        ಸ್ಥಿರಾಂಕ int64_t idst   = (ir / n_embd) % hc;
        ಸ್ಥಿರಾಂಕ int64_t it     = ir / (n_embd * hc);

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ xv = *(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) x->ದತ್ತಾಂಶ    + i0*nbx0 + it*nbx1);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ pv = *(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) post->ದತ್ತಾಂಶ + idst*nbp0 + it*nbp1);

        ದಶಮಾಂಶ ಮೊತ್ತ = xv * pv;
        ಆದರೆ (comb) {
            ಗಾಗಿ (int64_t isrc = 0; isrc < hc; ++isrc) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ rv = *(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) residual->ದತ್ತಾಂಶ + i0*nbr0 + isrc*nbr1 + it*nbr2);
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ cv = *(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) comb->ದತ್ತಾಂಶ     + idst*nbc0 + isrc*nbc1 + it*nbc2);
                ಮೊತ್ತ += rv * cv;
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಮೊತ್ತ += *(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) residual->ದತ್ತಾಂಶ + i0*nbr0 + idst*nbr1 + it*nbr2);
        }

        *(ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i0*nbd0 + idst*nbd1 + it*nbd2) = ಮೊತ್ತ;
    }
}

ಶೂನ್ಯ ggml_compute_forward_dsv4_hc_post(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_dsv4_hc_post_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_top_k_mask

ಶೂನ್ಯ ggml_compute_forward_ತುದಿ_k_ಮುಖವಾಡ(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * ಮುಖವಾಡ   = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * ತುದಿ_k  = dst->src[1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * score  = dst->src[2];
    ಸ್ಥಿರಾಂಕ ggml_tensor * ಖಂಡಗಳು = dst->src[3];
    ಸ್ಥಿರಾಂಕ ggml_tensor * extra  = dst->src[4];

    GGML_ASSERT(dst->type == ಮುಖವಾಡ->type);
    GGML_ASSERT(dst->nb[0] == ggml_type_ಗಾತ್ರ(dst->type));

    ಸ್ಥಿರಾಂಕ int64_t n_kv    = ಮುಖವಾಡ->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t ne1     = dst->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t ne2     = dst->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t n_ಅಡ್ಡಸಾಲುಗಳು  = ggml_nrows(dst);
    ಸ್ಥಿರಾಂಕ int64_t n_ತುದಿ   = ತುದಿ_k->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t n_blk   = ಖಂಡಗಳು ? ಖಂಡಗಳು->ne[0] : 1;
    ಸ್ಥಿರಾಂಕ int64_t n_pick  = ಖಂಡಗಳು ? ಖಂಡಗಳು->ne[1] : score ? score->ne[0] : n_kv;
    ಸ್ಥಿರಾಂಕ int64_t n_extra = extra ? extra->ne[0] : 0;

    ಸ್ಥಿರಾಂಕ size_t ts = ggml_type_ಗಾತ್ರ(dst->type);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಗಾಗಿ (int64_t ir = ith; ir < n_ಅಡ್ಡಸಾಲುಗಳು; ir += nth) {
        ಸ್ಥಿರಾಂಕ int64_t i1 = ir % ne1;
        ಸ್ಥಿರಾಂಕ int64_t i2 = (ir / ne1) % ne2;
        ಸ್ಥಿರಾಂಕ int64_t i3 = ir / (ne1*ne2);

        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಮುಖವಾಡ_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) ಮುಖವಾಡ->ದತ್ತಾಂಶ
            + (i1 % ಮುಖವಾಡ->ne[1])*ಮುಖವಾಡ->nb[1] + (i2 % ಮುಖವಾಡ->ne[2])*ಮುಖವಾಡ->nb[2] + (i3 % ಮುಖವಾಡ->ne[3])*ಮುಖವಾಡ->nb[3];
        ಅಕ್ಷರ * dst_ಅಡ್ಡಸಾಲು = (ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i1*dst->nb[1] + i2*dst->nb[2] + i3*dst->nb[3];

        ಆದರೆ (dst->type == GGML_TYPE_F16) {
            ಸ್ಥಿರಾಂಕ ggml_fp16_t ninf = GGML_CPU_FP32_ಗಮ್ಯ_FP16(-INFINITY);
            ಗಾಗಿ (int64_t c = 0; c < n_kv; ++c) {
                ((ggml_fp16_t *) dst_ಅಡ್ಡಸಾಲು)[c] = ninf;
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಗಾಗಿ (int64_t c = 0; c < n_kv; ++c) {
                ((ದಶಮಾಂಶ *) dst_ಅಡ್ಡಸಾಲು)[c] = -INFINITY;
            }
        }

        ಸ್ವಯಂಚಾಲಿತ keep = [&](int32_t c) {
            ಆದರೆ (c >= 0 && c < n_kv) {
                ಸ್ಮರಣೆ_ನಕಲು(dst_ಅಡ್ಡಸಾಲು + c*ts, ಮುಖವಾಡ_ಅಡ್ಡಸಾಲು + c*ts, ts);
            }
        };

        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ತುದಿ_k_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) ತುದಿ_k->ದತ್ತಾಂಶ + i1*ತುದಿ_k->nb[1] + i2*ತುದಿ_k->nb[2] + i3*ತುದಿ_k->nb[3];

        ಗಾಗಿ (int64_t j = 0; j < n_ತುದಿ; ++j) {
            ಸ್ಥಿರಾಂಕ int32_t p = *(ಸ್ಥಿರಾಂಕ int32_t *) (ತುದಿ_k_ಅಡ್ಡಸಾಲು + j*ತುದಿ_k->nb[0]);
            ಆದರೆ (p < 0 || p >= n_pick) {
                ಮುಂದುವರಿ;
            }
            ಆದರೆ (score) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ s = *(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) score->ದತ್ತಾಂಶ
                    + p*score->nb[0] + i1*score->nb[1] + i2*score->nb[2] + i3*score->nb[3]);
                ಆದರೆ (!(s > -INFINITY)) {
                    ಮುಂದುವರಿ;
                }
            }
            ಆದರೆ (!ಖಂಡಗಳು) {
                keep(p);
                ಮುಂದುವರಿ;
            }
            ಗಾಗಿ (int64_t k = 0; k < n_blk; ++k) {
                keep(*(ಸ್ಥಿರಾಂಕ int32_t *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) ಖಂಡಗಳು->ದತ್ತಾಂಶ + k*ಖಂಡಗಳು->nb[0] + p*ಖಂಡಗಳು->nb[1]));
            }
        }

        ಆದರೆ (extra) {
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * extra_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) extra->ದತ್ತಾಂಶ + i1*extra->nb[1] + i2*extra->nb[2] + i3*extra->nb[3];
            ಗಾಗಿ (int64_t j = 0; j < n_extra; ++j) {
                keep(*(ಸ್ಥಿರಾಂಕ int32_t *) (extra_ಅಡ್ಡಸಾಲು + j*extra->nb[0]));
            }
        }
    }
}

// ggml_compute_forward_rwkv_wkv7

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_rwkv_wkv7_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ int64_t T = dst->src[1]->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t C = dst->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t HEADS = dst->src[1]->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t n_seqs = dst->src[6]->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t ತಲೆ_ಗಾತ್ರ = C / HEADS;

    ದಶಮಾಂಶ * dst_ದತ್ತಾಂಶ = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * state = ((ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ) + C * T;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ h_ಆರಂಭ =  (HEADS * (ith    )) / nth;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ h_ಕೊನೆ   = ((HEADS * (ith + 1)) / nth < HEADS) ?
                         (HEADS * (ith + 1)) / nth : HEADS;

    ದಶಮಾಂಶ * r = (ದಶಮಾಂಶ *) dst->src[0]->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * w = (ದಶಮಾಂಶ *) dst->src[1]->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * k = (ದಶಮಾಂಶ *) dst->src[2]->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * v = (ದಶಮಾಂಶ *) dst->src[3]->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * a = (ದಶಮಾಂಶ *) dst->src[4]->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * b = (ದಶಮಾಂಶ *) dst->src[5]->ದತ್ತಾಂಶ;

    int64_t t_stride = HEADS * ತಲೆ_ಗಾತ್ರ; // Same to C

    int64_t h_stride = C / HEADS;
    GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
    int64_t h_stride_2d = ತಲೆ_ಗಾತ್ರ * ತಲೆ_ಗಾತ್ರ;

    #ಆದರೆ defined(GGML_SIMD)
        #ಆದರೆ defined(__ARM_FEATURE_SVE) || defined(__riscv_v_intrinsic)
            // scalar Route to scalar implementation       //TODO: Write SVE code and RVV code
            ಗಾಗಿ (int64_t t = 0; t < T; t++) {
                int64_t t_ಸ್ಥಾನಾಂತರ = t * t_stride;
                int64_t state_ಸ್ಥಾನಾಂತರ = ತಲೆ_ಗಾತ್ರ * C * (t / (T / n_seqs));
                ದಶಮಾಂಶ * state_cur = state + state_ಸ್ಥಾನಾಂತರ;
                ದಶಮಾಂಶ * state_prev = t % (T / n_seqs) ? state_cur : (ದಶಮಾಂಶ*)dst->src[6]->ದತ್ತಾಂಶ + state_ಸ್ಥಾನಾಂತರ;

                ಗಾಗಿ (int64_t h = h_ಆರಂಭ; h < h_ಕೊನೆ; h++) {
                    int64_t h_ಸ್ಥಾನಾಂತರ = h * h_stride;
                    int64_t t_h_ಸ್ಥಾನಾಂತರ = t_ಸ್ಥಾನಾಂತರ + h_ಸ್ಥಾನಾಂತರ;
                    int64_t h_2d_ಸ್ಥಾನಾಂತರ = h * h_stride_2d;

                    ಗಾಗಿ (int64_t i = 0; i < ತಲೆ_ಗಾತ್ರ; i++) {
                        int64_t t_h_i_ಸ್ಥಾನಾಂತರ = t_h_ಸ್ಥಾನಾಂತರ + i;
                        int64_t h_2d_i_ಸ್ಥಾನಾಂತರ = h_2d_ಸ್ಥಾನಾಂತರ + i * h_stride;

                        ದಶಮಾಂಶ v_val = v[t_h_i_ಸ್ಥಾನಾಂತರ];

                        ದಶಮಾಂಶ sa = 0, ಫಲಿತಾಂಶ = 0;
                        ಗಾಗಿ (int64_t j = 0; j < ತಲೆ_ಗಾತ್ರ; j++) {
                            sa += a[t_h_ಸ್ಥಾನಾಂತರ + j] * state_prev[h_2d_i_ಸ್ಥಾನಾಂತರ + j];
                        }

                        ಗಾಗಿ (int64_t j = 0; j < ತಲೆ_ಗಾತ್ರ; j++) {
                            int64_t t_h_j_ಸ್ಥಾನಾಂತರ = t_h_ಸ್ಥಾನಾಂತರ + j;
                            int64_t h_2d_i_j_ಸ್ಥಾನಾಂತರ = h_2d_i_ಸ್ಥಾನಾಂತರ + j;

                            ದಶಮಾಂಶ r_val = r[t_h_j_ಸ್ಥಾನಾಂತರ];
                            ದಶಮಾಂಶ w_val = w[t_h_j_ಸ್ಥಾನಾಂತರ];
                            ದಶಮಾಂಶ k_val = k[t_h_j_ಸ್ಥಾನಾಂತರ];
                            ದಶಮಾಂಶ b_val = b[t_h_j_ಸ್ಥಾನಾಂತರ];
                            ದಶಮಾಂಶ kv_val = v_val * k_val;
                            ದಶಮಾಂಶ prev_state_val = state_prev[h_2d_i_j_ಸ್ಥಾನಾಂತರ];
                            state_cur[h_2d_i_j_ಸ್ಥಾನಾಂತರ] = prev_state_val * w_val + kv_val + sa * b_val;
                            ಫಲಿತಾಂಶ += state_cur[h_2d_i_j_ಸ್ಥಾನಾಂತರ] * r_val;
                        }
                        dst_ದತ್ತಾಂಶ[t_h_i_ಸ್ಥಾನಾಂತರ] = ಫಲಿತಾಂಶ;
                    }
                }
            }
        #ಇಲ್ಲದಿದ್ದರೆ
            ಗಾಗಿ (int64_t t = 0; t < T; t++) {
                int64_t t_ಸ್ಥಾನಾಂತರ = t * t_stride;
                int64_t state_ಸ್ಥಾನಾಂತರ = ತಲೆ_ಗಾತ್ರ * C * (t / (T / n_seqs));
                ದಶಮಾಂಶ * state_cur = state + state_ಸ್ಥಾನಾಂತರ;
                ದಶಮಾಂಶ * state_prev = t % (T / n_seqs) ? state_cur : (ದಶಮಾಂಶ*)dst->src[6]->ದತ್ತಾಂಶ + state_ಸ್ಥಾನಾಂತರ;

                ಗಾಗಿ (int64_t h = h_ಆರಂಭ; h < h_ಕೊನೆ; h++) {
                    int64_t h_ಸ್ಥಾನಾಂತರ = h * h_stride;
                    int64_t t_h_ಸ್ಥಾನಾಂತರ = t_ಸ್ಥಾನಾಂತರ + h_ಸ್ಥಾನಾಂತರ;
                    int64_t h_2d_ಸ್ಥಾನಾಂತರ = h * h_stride_2d;

                    ಗಾಗಿ (int64_t ii = 0; ii < ತಲೆ_ಗಾತ್ರ; ii++) {
                        int64_t t_h_i_ಸ್ಥಾನಾಂತರ = t_h_ಸ್ಥಾನಾಂತರ + ii;
                        int64_t h_2d_i_ಸ್ಥಾನಾಂತರ = h_2d_ಸ್ಥಾನಾಂತರ + ii * h_stride;

                        GGML_F32_VEC v_vec = GGML_F32_VEC_SET1(v[t_h_i_ಸ್ಥಾನಾಂತರ]);

                        ದಶಮಾಂಶ sa = 0;
                        {
                            GGML_F32_VEC ಮೊತ್ತ[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };
                            GGML_F32_VEC ax[GGML_F32_ARR];
                            GGML_F32_VEC ay[GGML_F32_ARR];
                            ಗಾಗಿ (int64_t j = 0; j < ತಲೆ_ಗಾತ್ರ; j += GGML_F32_ಹೆಜ್ಜೆ) {
                                ಗಾಗಿ (int64_t kk = 0; kk < GGML_F32_ARR; kk++) {
                                    ax[kk] = GGML_F32_VEC_ತುಂಬು(&a[t_h_ಸ್ಥಾನಾಂತರ + j + kk * GGML_F32_EPR]);
                                    ay[kk] = GGML_F32_VEC_ತುಂಬು(&state_prev[h_2d_i_ಸ್ಥಾನಾಂತರ + j + kk * GGML_F32_EPR]);
                                    ಮೊತ್ತ[kk] = GGML_F32_VEC_FMA(ಮೊತ್ತ[kk], ax[kk], ay[kk]);
                                }
                            }
                            GGML_F32_VEC_REDUCE(sa, ಮೊತ್ತ);
                        }

                        GGML_F32_VEC sa_vec = GGML_F32_VEC_SET1(sa);

                        int64_t j = 0;
                        GGML_F32_VEC ಫಲಿತಾಂಶ_vec[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };
                        ಗಾಗಿ (; j < ತಲೆ_ಗಾತ್ರ; j += GGML_F32_ಹೆಜ್ಜೆ) {
                            ಗಾಗಿ (int64_t kk = 0; kk < GGML_F32_ARR; kk++) {
                                int64_t t_h_j_ಸ್ಥಾನಾಂತರ = t_h_ಸ್ಥಾನಾಂತರ + j + kk * GGML_F32_EPR;
                                int64_t h_2d_i_j_ಸ್ಥಾನಾಂತರ = h_2d_i_ಸ್ಥಾನಾಂತರ + j + kk * GGML_F32_EPR;

                                GGML_F32_VEC r_vec = GGML_F32_VEC_ತುಂಬು(&r[t_h_j_ಸ್ಥಾನಾಂತರ]);
                                GGML_F32_VEC w_vec = GGML_F32_VEC_ತುಂಬು(&w[t_h_j_ಸ್ಥಾನಾಂತರ]);
                                GGML_F32_VEC k_vec = GGML_F32_VEC_ತುಂಬು(&k[t_h_j_ಸ್ಥಾನಾಂತರ]);
                                GGML_F32_VEC b_vec = GGML_F32_VEC_ತುಂಬು(&b[t_h_j_ಸ್ಥಾನಾಂತರ]);

                                k_vec = GGML_F32_VEC_MUL(v_vec, k_vec);

                                GGML_F32_VEC state_vec = GGML_F32_VEC_ತುಂಬು(&state_prev[h_2d_i_j_ಸ್ಥಾನಾಂತರ]);
                                // kv + s * decay + sa * b
                                state_vec = GGML_F32_VEC_FMA(k_vec, state_vec, w_vec);
                                state_vec = GGML_F32_VEC_FMA(state_vec, sa_vec, b_vec);
                                GGML_F32_VEC_STORE(&state_cur[h_2d_i_j_ಸ್ಥಾನಾಂತರ], state_vec);

                                ಫಲಿತಾಂಶ_vec[kk] = GGML_F32_VEC_FMA(ಫಲಿತಾಂಶ_vec[kk], state_vec, r_vec);
                            }
                        }
                        GGML_F32_VEC_REDUCE(dst_ದತ್ತಾಂಶ[t_h_i_ಸ್ಥಾನಾಂತರ], ಫಲಿತಾಂಶ_vec);

                        // There shouldn't be left-overs though.
                        ಗಾಗಿ (; j < ತಲೆ_ಗಾತ್ರ; j++) {
                            int64_t t_h_j_ಸ್ಥಾನಾಂತರ = t_h_ಸ್ಥಾನಾಂತರ + j;
                            int64_t h_2d_i_j_ಸ್ಥಾನಾಂತರ = h_2d_i_ಸ್ಥಾನಾಂತರ + j;

                            ದಶಮಾಂಶ r_val = r[t_h_j_ಸ್ಥಾನಾಂತರ];
                            ದಶಮಾಂಶ w_val = w[t_h_j_ಸ್ಥಾನಾಂತರ];
                            ದಶಮಾಂಶ k_val = k[t_h_j_ಸ್ಥಾನಾಂತರ];
                            ದಶಮಾಂಶ b_val = b[t_h_j_ಸ್ಥಾನಾಂತರ];
                            ದಶಮಾಂಶ kv_val = v[t_h_i_ಸ್ಥಾನಾಂತರ] * k_val;

                            ದಶಮಾಂಶ prev_state_val = state_prev[h_2d_i_j_ಸ್ಥಾನಾಂತರ];
                            state_cur[h_2d_i_j_ಸ್ಥಾನಾಂತರ] = prev_state_val * w_val + kv_val + sa * b_val;
                            dst_ದತ್ತಾಂಶ[t_h_i_ಸ್ಥಾನಾಂತರ] += state_cur[h_2d_i_j_ಸ್ಥಾನಾಂತರ] * r_val;
                        }
                    }
                }
            }
        #ಆದರೆ_ಅಂತ್ಯ
    #ಇಲ್ಲದಿದ್ದರೆ
        ಗಾಗಿ (int64_t t = 0; t < T; t++) {
            int64_t t_ಸ್ಥಾನಾಂತರ = t * t_stride;
            int64_t state_ಸ್ಥಾನಾಂತರ = ತಲೆ_ಗಾತ್ರ * C * (t / (T / n_seqs));
            ದಶಮಾಂಶ * state_cur = state + state_ಸ್ಥಾನಾಂತರ;
            ದಶಮಾಂಶ * state_prev = t % (T / n_seqs) ? state_cur : (ದಶಮಾಂಶ*)dst->src[6]->ದತ್ತಾಂಶ + state_ಸ್ಥಾನಾಂತರ;

            ಗಾಗಿ (int64_t h = h_ಆರಂಭ; h < h_ಕೊನೆ; h++) {
                int64_t h_ಸ್ಥಾನಾಂತರ = h * h_stride;
                int64_t t_h_ಸ್ಥಾನಾಂತರ = t_ಸ್ಥಾನಾಂತರ + h_ಸ್ಥಾನಾಂತರ;
                int64_t h_2d_ಸ್ಥಾನಾಂತರ = h * h_stride_2d;

                ಗಾಗಿ (int64_t i = 0; i < ತಲೆ_ಗಾತ್ರ; i++) {
                    int64_t t_h_i_ಸ್ಥಾನಾಂತರ = t_h_ಸ್ಥಾನಾಂತರ + i;
                    int64_t h_2d_i_ಸ್ಥಾನಾಂತರ = h_2d_ಸ್ಥಾನಾಂತರ + i * h_stride;

                    ದಶಮಾಂಶ v_val = v[t_h_i_ಸ್ಥಾನಾಂತರ];

                    ದಶಮಾಂಶ sa = 0, ಫಲಿತಾಂಶ = 0;
                    ಗಾಗಿ (int64_t j = 0; j < ತಲೆ_ಗಾತ್ರ; j++) {
                        sa += a[t_h_ಸ್ಥಾನಾಂತರ + j] * state_prev[h_2d_i_ಸ್ಥಾನಾಂತರ + j];
                    }

                    ಗಾಗಿ (int64_t j = 0; j < ತಲೆ_ಗಾತ್ರ; j++) {
                        int64_t t_h_j_ಸ್ಥಾನಾಂತರ = t_h_ಸ್ಥಾನಾಂತರ + j;
                        int64_t h_2d_i_j_ಸ್ಥಾನಾಂತರ = h_2d_i_ಸ್ಥಾನಾಂತರ + j;

                        ದಶಮಾಂಶ r_val = r[t_h_j_ಸ್ಥಾನಾಂತರ];
                        ದಶಮಾಂಶ w_val = w[t_h_j_ಸ್ಥಾನಾಂತರ];
                        ದಶಮಾಂಶ k_val = k[t_h_j_ಸ್ಥಾನಾಂತರ];
                        ದಶಮಾಂಶ b_val = b[t_h_j_ಸ್ಥಾನಾಂತರ];
                        ದಶಮಾಂಶ kv_val = v_val * k_val;
                        ದಶಮಾಂಶ prev_state_val = state_prev[h_2d_i_j_ಸ್ಥಾನಾಂತರ];
                        state_cur[h_2d_i_j_ಸ್ಥಾನಾಂತರ] = prev_state_val * w_val + kv_val + sa * b_val;
                        ಫಲಿತಾಂಶ += state_cur[h_2d_i_j_ಸ್ಥಾನಾಂತರ] * r_val;
                    }
                    dst_ದತ್ತಾಂಶ[t_h_i_ಸ್ಥಾನಾಂತರ] = ಫಲಿತಾಂಶ;
                }
            }
        }
    #ಆದರೆ_ಅಂತ್ಯ
}


ಶೂನ್ಯ ggml_compute_forward_rwkv_wkv7(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_rwkv_wkv7_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_map_custom1

ಶೂನ್ಯ ggml_compute_forward_map_custom1(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
              ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * a = dst->src[0];

    ರಚನೆ ggml_map_custom1_op_params p;
    ಸ್ಮರಣೆ_ನಕಲು(&p, dst->op_params, ಗಾತ್ರ(p));

    p.fun(dst, a, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_map_custom2

ಶೂನ್ಯ ggml_compute_forward_map_custom2(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
              ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * a = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * b = dst->src[1];

    ರಚನೆ ggml_map_custom2_op_params p;
    ಸ್ಮರಣೆ_ನಕಲು(&p, dst->op_params, ಗಾತ್ರ(p));

    p.fun(dst, a, b, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_map_custom3

ಶೂನ್ಯ ggml_compute_forward_map_custom3(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
              ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * a = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * b = dst->src[1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * c = dst->src[2];

    ರಚನೆ ggml_map_custom3_op_params p;
    ಸ್ಮರಣೆ_ನಕಲು(&p, dst->op_params, ಗಾತ್ರ(p));

    p.fun(dst, a, b, c, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_custom

ಶೂನ್ಯ ggml_compute_forward_custom(
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params,
          ರಚನೆ ggml_tensor * dst) {

    ರಚನೆ ggml_custom_op_params p;
    ಸ್ಮರಣೆ_ನಕಲು(&p, dst->op_params, ಗಾತ್ರ(p));

    p.fun(dst, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_cross_entropy_loss

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_cross_entropy_loss_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->nb[0] == ggml_type_ಗಾತ್ರ(src0->type));
    GGML_ASSERT(src1->nb[0] == ggml_type_ಗಾತ್ರ(src1->type));
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // TODO: handle transposed/permuted matrices
    ಸ್ಥಿರಾಂಕ int64_t nc = src0->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t nr = ggml_nrows(src0);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ದಶಮಾಂಶ * sums =  (ದಶಮಾಂಶ *) params->wdata;
    ದಶಮಾಂಶ * st   = ((ದಶಮಾಂಶ *) params->wdata) + nth + ith*nc;
    ದಶಮಾಂಶ ಮೊತ್ತ_thread = 0.0f;

    GGML_ASSERT(params->wsize >= ಗಾತ್ರ(ದಶಮಾಂಶ) * (nth + nth * nc));

    // rows per thread
    ಸ್ಥಿರಾಂಕ int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ int64_t ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ int64_t ir1 = MIN(ir0 + dr, nr);

    ಗಾಗಿ (int64_t i1 = ir0; i1 < ir1; ++i1) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * s0 = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i1*src0->nb[1]);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * s1 = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ + i1*src1->nb[1]);

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NDEBUG
        ಗಾಗಿ (int64_t i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            ಪ್ರತಿಜ್ಞಾ(!isnan(s0[i]));
            ಪ್ರತಿಜ್ಞಾ(!isnan(s1[i]));
        }
#ಆದರೆ_ಅಂತ್ಯ // NDEBUG

        ದಶಮಾಂಶ max = -INFINITY;
        ggml_vec_max_f32(nc, &max, s0);
        ಸ್ಥಿರಾಂಕ ggml_float ಮೊತ್ತ_softmax = ggml_vec_log_soft_max_f32(nc, st, s0, max);
        ಪ್ರತಿಜ್ಞಾ(ಮೊತ್ತ_softmax >= 0.0);

        ggml_vec_add1_f32(nc, st, st, -ಮೊತ್ತ_softmax);
        ggml_vec_mul_f32(nc, st, st, s1);

        ದಶಮಾಂಶ ಮೊತ್ತ_st = 0.0f;
        ggml_vec_ಮೊತ್ತ_f32(nc, &ಮೊತ್ತ_st, st);
        ಮೊತ್ತ_thread += ಮೊತ್ತ_st;

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NDEBUG
        ಗಾಗಿ (int64_t i = 0; i < nc; ++i) {
            ಪ್ರತಿಜ್ಞಾ(!isnan(st[i]));
            ಪ್ರತಿಜ್ಞಾ(!isinf(st[i]));
        }
#ಆದರೆ_ಅಂತ್ಯ // NDEBUG
    }
    sums[ith] = ಮೊತ್ತ_thread;
    ggml_barrier(params->threadpool);

    ಆದರೆ (ith == 0) {
        ದಶಮಾಂಶ * dp = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;
        ggml_vec_ಮೊತ್ತ_f32(nth, dp, sums);
        dp[0] *= -1.0f / (ದಶಮಾಂಶ) nr;
    }
}

ಶೂನ್ಯ ggml_compute_forward_cross_entropy_loss(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_cross_entropy_loss_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_cross_entropy_loss_back

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_cross_entropy_loss_ಹಿಂಭಾಗ_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * grad  = dst->src[0]; // gradient of forward pass output
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0f = dst->src[1]; // src0 of forward pass
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1f = dst->src[2]; // src1 of forward pass

    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src0f));
    GGML_ASSERT(ggml_is_contiguous(src1f));
    GGML_ASSERT(ggml_is_contiguous(grad));
    GGML_ASSERT(ggml_are_same_shape(src0f, src1f) && ggml_are_same_shape(src0f, dst));

    ಸ್ಥಿರಾಂಕ int64_t ith = params->ith;
    ಸ್ಥಿರಾಂಕ int64_t nth = params->nth;

    // TODO: handle transposed/permuted matrices
    ಸ್ಥಿರಾಂಕ int64_t nc = src0f->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t nr = ggml_nrows(src0f);

    // rows per thread
    ಸ್ಥಿರಾಂಕ int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ int64_t ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ int64_t ir1 = MIN(ir0 + dr, nr);

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d_by_nr = ((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) grad->ದತ್ತಾಂಶ)[0] / (ದಶಮಾಂಶ) nr;

    ಗಾಗಿ (int64_t i1 = ir0; i1 < ir1; i1++) {
        ದಶಮಾಂಶ       * ds0 = (ದಶಮಾಂಶ       *)((ಅಕ್ಷರ       *) dst->ದತ್ತಾಂಶ   + i1*dst->nb[1]);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * s0  = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0f->ದತ್ತಾಂಶ + i1*src0f->nb[1]);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * s1  = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src1f->ದತ್ತಾಂಶ + i1*src1f->nb[1]);

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NDEBUG
        ಗಾಗಿ (int64_t i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            ಪ್ರತಿಜ್ಞಾ(!isnan(s0[i]));
            ಪ್ರತಿಜ್ಞಾ(!isnan(s1[i]));
        }
#ಆದರೆ_ಅಂತ್ಯ // NDEBUG

        // soft_max
        ದಶಮಾಂಶ max = -INFINITY;
        ggml_vec_max_f32(nc, &max, s0);
        ಸ್ಥಿರಾಂಕ ggml_float ಮೊತ್ತ = ggml_vec_soft_max_f32(nc, ds0, s0, max);
        ಪ್ರತಿಜ್ಞಾ(ಮೊತ್ತ > 0.0);
        ggml_vec_scale_f32(nc, ds0, 1.0/ಮೊತ್ತ);

        // grad(src0f) = (softmax(src0f) - src1f) * grad(cross_entropy_loss(src0f, src1f)) / nr
        ggml_vec_sub_f32(nc, ds0, ds0, s1);
        ggml_vec_scale_f32(nc, ds0, d_by_nr);

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NDEBUG
        ಗಾಗಿ (int64_t i = 0; i < nc; ++i) {
            ಪ್ರತಿಜ್ಞಾ(!isnan(ds0[i]));
            ಪ್ರತಿಜ್ಞಾ(!isinf(ds0[i]));
        }
#ಆದರೆ_ಅಂತ್ಯ // NDEBUG
    }
}

ಶೂನ್ಯ ggml_compute_forward_cross_entropy_loss_ಹಿಂಭಾಗ(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_cross_entropy_loss_ಹಿಂಭಾಗ_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_opt_ಹೆಜ್ಜೆ_adamw_f32(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0         = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0_grad    = dst->src[1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0_grad_m  = dst->src[2];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0_grad_v  = dst->src[3];
    ಸ್ಥಿರಾಂಕ ggml_tensor * adamw_params = dst->src[4];

    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad));
    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad_m));
    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad_v));
    GGML_ASSERT(ggml_nelements(adamw_params) == 7);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS
    GGML_ASSERT(nb00 == ಗಾತ್ರ(ದಶಮಾಂಶ));

    // rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * adamw_params_ptr = ggml_get_ದತ್ತಾಂಶ_f32(adamw_params);

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ alpha  = adamw_params_ptr[0];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ beta1  = adamw_params_ptr[1];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ beta2  = adamw_params_ptr[2];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ eps    = adamw_params_ptr[3];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ wd     = adamw_params_ptr[4];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ beta1h = adamw_params_ptr[5];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ beta2h = adamw_params_ptr[6];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ keep   = 1.f - alpha * wd;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ir = ir0; ir < ir1; ++ir) {
        ಸ್ಥಿರಾಂಕ int64_t i03 = ir/(ne02*ne01);
        ಸ್ಥಿರಾಂಕ int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        ಸ್ಥಿರಾಂಕ int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        ಸ್ಥಿರಾಂಕ size_t ಸ್ಥಾನಾಂತರ = i03*nb03 + i02*nb02 + i01*nb01;

        ದಶಮಾಂಶ       * w = (ದಶಮಾಂಶ       *) ((ಅಕ್ಷರ       *) src0->ದತ್ತಾಂಶ        + ಸ್ಥಾನಾಂತರ); //ತೂಕ
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * g = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0_grad->ದತ್ತಾಂಶ   + ಸ್ಥಾನಾಂತರ); // grad
        ದಶಮಾಂಶ       * m = (ದಶಮಾಂಶ       *) ((ಅಕ್ಷರ       *) src0_grad_m->ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ);
        ದಶಮಾಂಶ       * v = (ದಶಮಾಂಶ       *) ((ಅಕ್ಷರ       *) src0_grad_v->ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ);

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i00 = 0; i00 < ne00; ++i00) {
            m[i00] = m[i00]*beta1 +        g[i00]*(1.0f - beta1);
            v[i00] = v[i00]*beta2 + g[i00]*g[i00]*(1.0f - beta2);

            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ mh =       m[i00]*beta1h;
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ vh = sqrtf(v[i00]*beta2h) + eps;

            // The weight decay is applied independently of the Adam momenta m and v.
            // This is NOT equivalent to l2 regularization that adds w[i00]*w[i00] to the loss.
            // See: https://arxiv.org/pdf/1711.05101v3.pdf
            w[i00] = w[i00] * keep - alpha * mh / vh;
        }
    }
}

ಶೂನ್ಯ ggml_compute_forward_opt_ಹೆಜ್ಜೆ_adamw(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_opt_ಹೆಜ್ಜೆ_adamw_f32(params, dst);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_opt_ಹೆಜ್ಜೆ_sgd_f32(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0       = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0_grad  = dst->src[1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * sgd_params = dst->src[2];

    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad));
    GGML_ASSERT(ggml_nelements(sgd_params) == 2);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS
    GGML_ASSERT(nb00 == ಗಾತ್ರ(ದಶಮಾಂಶ));

    // rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1) / nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr * ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    // using adamw param subset we care about - alpha, wd - could have a separate struct
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * sgd_params_ptr   = ggml_get_ದತ್ತಾಂಶ_f32(sgd_params);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   alpha            = sgd_params_ptr[0];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   keep             = 1.f - alpha * sgd_params_ptr[1];

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ir = ir0; ir < ir1; ++ir) {
        ಸ್ಥಿರಾಂಕ int64_t i03 = ir / (ne02 * ne01);
        ಸ್ಥಿರಾಂಕ int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
        ಸ್ಥಿರಾಂಕ int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

        ಸ್ಥಿರಾಂಕ size_t ಸ್ಥಾನಾಂತರ = i03 * nb03 + i02 * nb02 + i01 * nb01;

        ದಶಮಾಂಶ *       w = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ);                   //ತೂಕ
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * g = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0_grad->ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ);  // grad

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i00 = 0; i00 < ne00; ++i00) {
            w[i00] = w[i00] * keep - alpha * g[i00];
        }
    }
}

ಶೂನ್ಯ ggml_compute_forward_opt_ಹೆಜ್ಜೆ_sgd(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_opt_ಹೆಜ್ಜೆ_sgd_f32(params, dst);
            }
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error - sgd is F32 only");
            }
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ ggml_fwht_ತುಂಬು(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ value) {
    ಹಿಂತಿರುಗಿ value;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ ggml_fwht_ತುಂಬು(ಸ್ಥಿರಾಂಕ ggml_fp16_t value) {
    ಹಿಂತಿರುಗಿ ggml_fp16_ಗಮ್ಯ_fp32(value);
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು src_t>
ಸ್ಥಿರ ಶೂನ್ಯ ggml_compute_forward_fwht_impl(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == (std::is_same_v<src_t, ದಶಮಾಂಶ> ? GGML_TYPE_F32 : GGML_TYPE_F16));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    ಸ್ಥಿರಾಂಕ int64_t n = ne10;
    GGML_ASSERT((n & (n - 1)) == 0); // must be power of 2

    ಸ್ಥಿರಾಂಕ int64_t nr = ne11 * ne12 * ne13;
    ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread = (nr + nth - 1) / nth;
    ಸ್ಥಿರಾಂಕ int64_t ಆರಂಭ_ಅಡ್ಡಸಾಲು = ith * ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread;
    ಸ್ಥಿರಾಂಕ int64_t ಕೊನೆ_ಅಡ್ಡಸಾಲು = MIN(ಆರಂಭ_ಅಡ್ಡಸಾಲು + ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread, nr);

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale = 1.0f / sqrtf((ದಶಮಾಂಶ)n);

#ಆದರೆ defined(GGML_SIMD)
    ಸ್ಥಿರಾಂಕ GGML_F32_VEC v_minus_one = GGML_F32_VEC_SET1(-1.0f);
#ಆದರೆ_ಅಂತ್ಯ

    ಗಾಗಿ (int64_t r = ಆರಂಭ_ಅಡ್ಡಸಾಲು; r < ಕೊನೆ_ಅಡ್ಡಸಾಲು; r++) {
        ಸ್ಥಿರಾಂಕ int64_t i13 = r / (ne11 * ne12);
        ಸ್ಥಿರಾಂಕ int64_t i12 = (r - i13 * ne11 * ne12) / ne11;
        ಸ್ಥಿರಾಂಕ int64_t i11 = r - i13 * ne11 * ne12 - i12 * ne11;

        ಸ್ಥಿರಾಂಕ src_t * src_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ src_t *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ + i11 * nb11 + i12 * nb12 + i13 * nb13);
        ದಶಮಾಂಶ * dst_ಅಡ್ಡಸಾಲು = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i11 * nb1 + i12 * nb2 + i13 * nb3);

        ಗಾಗಿ (int64_t j = 0; j < n; j++) {
            dst_ಅಡ್ಡಸಾಲು[j] = ggml_fwht_ತುಂಬು(src_ಅಡ್ಡಸಾಲು[j]) * scale;
        }

        // Scalar passes
#ಆದರೆ defined(GGML_SIMD)
#ಆದರೆ defined(__ARM_FEATURE_SVE)
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಹೆಜ್ಜೆ = svcntw();
#ಇಲ್ಲದಿದ್ದರೆ
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಹೆಜ್ಜೆ = GGML_F32_EPR;
#ಆದರೆ_ಅಂತ್ಯ
#ಇಲ್ಲದಿದ್ದರೆ
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಹೆಜ್ಜೆ = n;
#ಆದರೆ_ಅಂತ್ಯ
        ಗಾಗಿ (int64_t len = 1; len < ಹೆಜ್ಜೆ && len < n; len <<= 1) {
            ಗಾಗಿ (int64_t i = 0; i < n; i += 2 * len) {
                ಗಾಗಿ (int64_t j = 0; j < len; j++) {
                    ದಶಮಾಂಶ u = dst_ಅಡ್ಡಸಾಲು[i + j];
                    ದಶಮಾಂಶ v = dst_ಅಡ್ಡಸಾಲು[i + len + j];
                    dst_ಅಡ್ಡಸಾಲು[i + j] = u + v;
                    dst_ಅಡ್ಡಸಾಲು[i + len + j] = u - v;
                }
            }
        }

        // SIMD passes using GGML_F32_VEC_* macros for multi-architecture support
#ಆದರೆ defined(GGML_SIMD)
        ಗಾಗಿ (int64_t len = ಹೆಜ್ಜೆ; len < n; len <<= 1) {
            ಗಾಗಿ (int64_t i = 0; i < n; i += 2 * len) {
                ಗಾಗಿ (int64_t j = 0; j < len; j += ಹೆಜ್ಜೆ) {
                    GGML_F32_VEC u = GGML_F32_VEC_ತುಂಬು(dst_ಅಡ್ಡಸಾಲು + i + j);
                    GGML_F32_VEC v = GGML_F32_VEC_ತುಂಬು(dst_ಅಡ್ಡಸಾಲು + i + len + j);

                    GGML_F32_VEC_STORE(dst_ಅಡ್ಡಸಾಲು + i + j,       GGML_F32_VEC_ಕೂಡಿಸು(u, v));
                    GGML_F32_VEC_STORE(dst_ಅಡ್ಡಸಾಲು + i + len + j, GGML_F32_VEC_FMA(u, v, v_minus_one));
                }
            }
        }
#ಆದರೆ_ಅಂತ್ಯ
    }
}

ಶೂನ್ಯ ggml_compute_forward_fwht(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    ಆಯ್ಕೆ (src1->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ggml_compute_forward_fwht_impl<ದಶಮಾಂಶ>(params, dst);
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            {
                ggml_compute_forward_fwht_impl<ggml_fp16_t>(params, dst);
            }
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            {
                GGML_ABORT("fatal error - fwht supports F32 and F16 input");
            }
    }
}

// ggml_compute_forward_lightning_indexer

ಶೂನ್ಯ ggml_compute_forward_lightning_indexer(
        ಸ್ಥಿರಾಂಕ ggml_compute_params * params,
        ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * q = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * k = dst->src[1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * w = dst->src[2]; // weights
    ಸ್ಥಿರಾಂಕ ggml_tensor * m = dst->src[3]; //ಮುಖವಾಡ

    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(   q->type == GGML_TYPE_F32);
    GGML_ASSERT(   w->type == GGML_TYPE_F32);
    GGML_ASSERT(   m->type == GGML_TYPE_F16);

    GGML_TENSOR_LOCALS(int64_t, neq,  q, ne)
    GGML_TENSOR_LOCALS(size_t,  nbq,  q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek,  k, ne)
    GGML_TENSOR_LOCALS(size_t,  nbk,  k, nb)
    GGML_TENSOR_LOCALS(int64_t, ಹೊಸ,  w, ne)
    GGML_TENSOR_LOCALS(size_t,  nbw,  w, nb)
    GGML_TENSOR_LOCALS(int64_t, nem,  m, ne)
    GGML_TENSOR_LOCALS(size_t,  nbm,  m, nb)
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb)

    GGML_ASSERT( nb0 == ggml_type_ಗಾತ್ರ(dst->type));
    GGML_ASSERT(nbq0 == ggml_type_ಗಾತ್ರ(  q->type));
    GGML_ASSERT(nbk0 == ggml_type_ಗಾತ್ರ(  k->type));
    GGML_ASSERT(nbw0 == ggml_type_ಗಾತ್ರ(  w->type));
    GGML_ASSERT(nbm0 == ggml_type_ಗಾತ್ರ(  m->type));

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_embd    = q->ne[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_ತಲೆ    = q->ne[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_ಟೋಕನುಗಳು  = q->ne[2];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_ಹರಿವು  = q->ne[3];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_kv      = k->ne[2];

    ggml_ಗಮ್ಯ_float_t ಸ್ಥಿರಾಂಕ k_ಗಮ್ಯ_float = ggml_get_type_traits(k->type)->ಗಮ್ಯ_float;
    GGML_ASSERT((k->type == GGML_TYPE_F32 || k_ಗಮ್ಯ_float) && "lightning indexer: unsupported K-type");

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nr  = n_kv;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith = params->ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth = params->nth;

    // (temporary) buffer for K converted to float
    ದಶಮಾಂಶ * k_ಅಡ್ಡಸಾಲು_f32 = (ದಶಮಾಂಶ *) params->wdata + ith*(1*n_embd + CACHE_ಸಾಲು_ಗಾತ್ರ_F32);

    // rows per thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dr = (nr + nth - 1)/nth;

    // row range for this thread
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir0 = dr*ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir1 = MIN(ir0 + dr, nr);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ s = 0; s < n_ಹರಿವು; ++s) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = 0; t < n_ಟೋಕನುಗಳು; ++t) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ       *   w_ಅಡ್ಡಸಾಲು =       (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *)   w->ದತ್ತಾಂಶ + t*nbw1 +        s*nbw3);
            ಸ್ಥಿರಾಂಕ ggml_fp16_t *   m_ಅಡ್ಡಸಾಲು = (ggml_fp16_t *) ((ಅಕ್ಷರ *)   m->ದತ್ತಾಂಶ + t*nbm1 + (s%nem3)*nbm3);
            ದಶಮಾಂಶ             * dst_ಅಡ್ಡಸಾಲು =       (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + t*nb1  +        s*nb3 );
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ik = ir0; ik < ir1; ++ik) {
                ಅಕ್ಷರ * k_ಅಡ್ಡಸಾಲು = (ಅಕ್ಷರ *) k->ದತ್ತಾಂಶ + ik*nbk2 + s*nbk3;
                ಆದರೆ (k_ಗಮ್ಯ_float) {
                    k_ಗಮ್ಯ_float(k_ಅಡ್ಡಸಾಲು, k_ಅಡ್ಡಸಾಲು_f32, n_embd);
                } ಇಲ್ಲದಿದ್ದರೆ {
                    k_ಅಡ್ಡಸಾಲು_f32 = (ದಶಮಾಂಶ *) k_ಅಡ್ಡಸಾಲು;
                }
                ದಶಮಾಂಶ score = 0.0f;
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ h = 0; h < n_ತಲೆ; ++h) {
                    // dot product of q and k for head h
                    ದಶಮಾಂಶ qk = 0.0f;
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * q_ಅಡ್ಡಸಾಲು = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) q->ದತ್ತಾಂಶ + h*nbq1 + t*nbq2 + s*nbq3);
                    ggml_vec_dot_f32(n_embd, &qk, 0, q_ಅಡ್ಡಸಾಲು, 0, k_ಅಡ್ಡಸಾಲು_f32, 0, 1);
                    // ReLU and weights (prescaled)
                    score += MAX(qk, 0.0f) * w_ಅಡ್ಡಸಾಲು[h];
                }
                // apply mask
                dst_ಅಡ್ಡಸಾಲು[ik] = score + GGML_CPU_FP16_ಗಮ್ಯ_FP32(m_ಅಡ್ಡಸಾಲು[ik]);
            }
        }
    }
}
