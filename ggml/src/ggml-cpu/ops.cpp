#ضمن "ops.h"

#ضمن "ggml-cpu.h"
#ضمن "ggml-impl.h"
#ضمن "binary-ops.h"
#ضمن "simd-gemm.h"
#ضمن "ggml.h"
#ضمن "unary-ops.h"
#ضمن "vec.h"

#ضمن <algorithm>
#ضمن <cfloat>
#ضمن <cmath>

// ggml_compute_forward_dup

ساكن فراغ ggml_compute_forward_dup_same_cont(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));
    GGML_ASSERT(src0->type == dst->type);

    ثابت size_t nb0 = ggml_type_سائيز(src0->type);

    ثابت صحيح ith = params->ith; // thread index
    ثابت صحيح nth = params->nth; // number of threads

    // parallelize by blocks
    ثابت صحيح nk = ggml_nelements(src0)/ggml_blck_سائيز(src0->type);
    ثابت صحيح dr = (nk + nth - 1) / nth;
    ثابت صحيح k0 = dr * ith;
    ثابت صحيح k1 = MIN(k0 + dr, nk);

    إذا (k0 < k1) {
        يادگيري_نقل(
            ((حرف *)  dst->معلومات + k0*nb0),
            ((حرف *) src0->معلومات + k0*nb0),
            (k1 - k0) * nb0);
    }
}

سانچو<قسم_نالو src_t, قسم_نالو dst_t>
ساكن فراغ ggml_compute_forward_dup_flt(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(!ggml_is_quantized(src0->type) && !ggml_is_quantized(dst->type));

    GGML_TENSOR_UNARY_OP_LOCALS

    ثابت صحيح ith = params->ith; // thread index
    ثابت صحيح nth = params->nth; // number of threads

    // parallelize by rows
    ثابت صحيح nr = ne01;
    // number of rows per thread
    ثابت صحيح dr = (nr + nth - 1) / nth;
    // row range for this thread
    ثابت صحيح ir0 = dr * ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    // case: type & row size equal
    إذا (src0->type == dst->type &&
        ne00 == ne0 &&
        nb00 == ggml_type_سائيز(src0->type) && nb0 == ggml_type_سائيز(dst->type)) {
        // copy by rows
        ثابت size_t rs = ne00*nb00;
        لكل (int64_t i03 = 0; i03 < ne03; i03++) {
            لكل (int64_t i02 = 0; i02 < ne02; i02++) {
                لكل (int64_t i01 = ir0; i01 < ir1; i01++) {
                    يادگيري_نقل(
                        ((حرف *)  dst->معلومات + i01*nb1  + i02*nb2  + i03*nb3),
                        ((حرف *) src0->معلومات + i01*nb01 + i02*nb02 + i03*nb03),
                        rs);
                }
            }
        }
        أرجع;
    }

    // case: dst tensor is contiguous
    إذا (ggml_is_contiguous(dst)) {
        إذا (nb00 == حجم(src_t)) {
            إذا constexpr (std::is_same_v<dst_t, src_t>) {
                // same type
                size_t id = 0;
                ثابت size_t rs = ne00 * nb00;
                حرف * dst_ptr = (حرف *) dst->معلومات;

                لكل (صحيح i03 = 0; i03 < ne03; i03++) {
                    لكل (صحيح i02 = 0; i02 < ne02; i02++) {
                        id += rs * ir0;
                        لكل (صحيح i01 = ir0; i01 < ir1; i01++) {
                            ثابت حرف * src0_ptr = (حرف *) src0->معلومات + i01*nb01 + i02*nb02 + i03*nb03;
                            يادگيري_نقل(dst_ptr + id, src0_ptr, rs);
                            id += rs;
                        }
                        id += rs * (ne01 - ir1);
                    }
                }
            } وإلا {
                // casting between non-quantized types
                size_t id = 0;
                dst_t * dst_ptr = (dst_t *) dst->معلومات;

                لكل (صحيح i03 = 0; i03 < ne03; i03++) {
                    لكل (صحيح i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        لكل (صحيح i01 = ir0; i01 < ir1; i01++) {
                            ثابت src_t * src0_ptr = (src_t *) ((حرف *) src0->معلومات + i01*nb01 + i02*nb02 + i03*nb03);
                            لكل (صحيح i00 = 0; i00 < ne00; i00++) {
                                عائم tmp = type_conversion_جدول<src_t>::منزل_f32(src0_ptr[i00]);
                                dst_ptr[id] = type_conversion_جدول<dst_t>::from_f32(tmp);
                                id++;
                            }
                        }
                        id += ne00 * (ne01 - ir1);
                    }
                }
            }
        } وإلا {
            //printf("%s: this is not optimal - fix me\n", __func__);

            size_t id = 0;
            dst_t * dst_ptr = (dst_t *) dst->معلومات;

            لكل (صحيح i03 = 0; i03 < ne03; i03++) {
                لكل (صحيح i02 = 0; i02 < ne02; i02++) {
                    id += ne00 * ir0;
                    لكل (صحيح i01 = ir0; i01 < ir1; i01++) {
                        لكل (صحيح i00 = 0; i00 < ne00; i00++) {
                            ثابت src_t * src0_ptr = (src_t *) ((حرف *) src0->معلومات + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);

                            عائم tmp = type_conversion_جدول<src_t>::منزل_f32(*src0_ptr);
                            dst_ptr[id] = type_conversion_جدول<dst_t>::from_f32(tmp);
                            id++;
                        }
                    }
                    id += ne00 * (ne01 - ir1);
                }
            }
        }
        أرجع;
    }

    // dst counters
    int64_t i10 = 0;
    int64_t i11 = 0;
    int64_t i12 = 0;
    int64_t i13 = 0;

    إذا constexpr (std::is_same_v<dst_t, src_t>) {
        لكل (int64_t i03 = 0; i03 < ne03; i03++) {
            لكل (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                طالما (i10 >= ne0) {
                    i10 -= ne0;
                    إذا (++i11 == ne1) {
                        i11 = 0;
                        إذا (++i12 == ne2) {
                            i12 = 0;
                            إذا (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                لكل (int64_t i01 = ir0; i01 < ir1; i01++) {
                    لكل (int64_t i00 = 0; i00 < ne00; i00++) {
                        ثابت حرف * src0_ptr = ((حرف *) src0->معلومات + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              حرف * dst_ptr  = ((حرف *)  dst->معلومات + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                        يادگيري_نقل(dst_ptr, src0_ptr, حجم(dst_t));

                        إذا (++i10 == ne00) {
                            i10 = 0;
                            إذا (++i11 == ne01) {
                                i11 = 0;
                                إذا (++i12 == ne02) {
                                    i12 = 0;
                                    إذا (++i13 == ne03) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                طالما (i10 >= ne0) {
                    i10 -= ne0;
                    إذا (++i11 == ne1) {
                        i11 = 0;
                        إذا (++i12 == ne2) {
                            i12 = 0;
                            إذا (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
            }
        }

    } وإلا {
        لكل (int64_t i03 = 0; i03 < ne03; i03++) {
            لكل (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                طالما (i10 >= ne0) {
                    i10 -= ne0;
                    إذا (++i11 == ne1) {
                        i11 = 0;
                        إذا (++i12 == ne2) {
                            i12 = 0;
                            إذا (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                لكل (int64_t i01 = ir0; i01 < ir1; i01++) {
                    لكل (int64_t i00 = 0; i00 < ne00; i00++) {
                        ثابت حرف * src0_ptr = ((حرف *) src0->معلومات + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              حرف * dst_ptr  = ((حرف *)  dst->معلومات + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                        عائم tmp = type_conversion_جدول<src_t>::منزل_f32(*(ثابت src_t *) src0_ptr);
                        *(dst_t *) dst_ptr = type_conversion_جدول<dst_t>::from_f32(tmp);

                        إذا (++i10 == ne0) {
                            i10 = 0;
                            إذا (++i11 == ne1) {
                                i11 = 0;
                                إذا (++i12 == ne2) {
                                    i12 = 0;
                                    إذا (++i13 == ne3) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                طالما (i10 >= ne0) {
                    i10 -= ne0;
                    إذا (++i11 == ne1) {
                        i11 = 0;
                        إذا (++i12 == ne2) {
                            i12 = 0;
                            إذا (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
            }
        }
    }
}


سانچو<قسم_نالو src_t>
ساكن فراغ ggml_compute_forward_dup_منزل_q(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(!ggml_is_quantized(src0->type));

    GGML_TENSOR_UNARY_OP_LOCALS

    ثابت صحيح ith = params->ith; // thread index
    ثابت صحيح nth = params->nth; // number of threads

    // parallelize by rows
    ثابت صحيح nr = ne01;
    // number of rows per thread
    ثابت صحيح dr = (nr + nth - 1) / nth;
    // row range for this thread
    ثابت صحيح ir0 = dr * ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    إذا (ggml_is_contiguous(dst) &&
            nb00 == حجم(src_t) &&
            ggml_get_type_traits_cpu(dst->type)->from_float) {
        // casting non-quantized types --> intermediate f32 --> quantized
        ggml_from_float_t ثابت quantize_صف_q = ggml_get_type_traits_cpu(dst->type)->from_float;
        عائم * src0_f32 = (عائم *) params->wdata + (ne00 + CACHE_سٽ_سائيز_F32) * ith;

        size_t id = 0;
        size_t rs = nb0 * (ne00 / ggml_blck_سائيز(dst->type));
        حرف * dst_ptr = (حرف *) dst->معلومات;

        لكل (صحيح i03 = 0; i03 < ne03; i03++) {
            لكل (صحيح i02 = 0; i02 < ne02; i02++) {
                id += rs * ir0;
                لكل (صحيح i01 = ir0; i01 < ir1; i01++) {
                    ثابت src_t * src0_ptr = (src_t *) ((حرف *) src0->معلومات + i01*nb01 + i02*nb02 + i03*nb03);

                    لكل (صحيح i00 = 0; i00 < ne00; i00++) {
                        src0_f32[i00] = type_conversion_جدول<src_t>::منزل_f32(src0_ptr[i00]);
                    }

                    quantize_صف_q(src0_f32, dst_ptr + id, ne00);
                    id += rs;
                }
                id += rs * (ne01 - ir1);
            }
        }
    } وإلا {
        // printf("%s %s\n", ggml_type_name(src0->type), ggml_type_name(dst->type));
        GGML_ABORT("not implemented");
    }
}

// A simplified version of ggml_compute_forward_dup that doesn't do float upcasting, and just plain old memcpy.
ساكن فراغ ggml_compute_forward_dup_بائيٽون(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(src0->type == dst->type);

    GGML_TENSOR_UNARY_OP_LOCALS;

    إذا (ggml_is_contiguous(src0) && ggml_is_contiguous(dst)) {
        ggml_compute_forward_dup_same_cont(params, dst);
        أرجع;
    }

    ثابت size_t type_سائيز = ggml_type_سائيز(src0->type);

    ثابت صحيح ith = params->ith; // thread index
    ثابت صحيح nth = params->nth; // number of threads

    // parallelize by rows
    ثابت صحيح nr = ne01;
    // number of rows per thread
    ثابت صحيح dr = (nr + nth - 1) / nth;
    // row range for this thread
    ثابت صحيح ir0 = dr * ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    إذا (src0->type == dst->type &&
        ggml_are_same_shape(src0, dst) &&
        nb00 == type_سائيز && nb0 == type_سائيز) {
        // copy by rows
        ثابت size_t rs = ggml_صف_سائيز(src0->type, ne00);
        لكل (int64_t i03 = 0; i03 < ne03; i03++) {
            لكل (int64_t i02 = 0; i02 < ne02; i02++) {
                لكل (int64_t i01 = ir0; i01 < ir1; i01++) {
                    يادگيري_نقل(
                        ((حرف *)  dst->معلومات + i01*nb1  + i02*nb2  + i03*nb3),
                        ((حرف *) src0->معلومات + i01*nb01 + i02*nb02 + i03*nb03),
                        rs);
                }
            }
        }
        أرجع;
    }

    إذا (ggml_is_contiguous(dst)) {
        size_t id = 0;
        حرف * dst_ptr = (حرف *) dst->معلومات;
        ثابت size_t rs = ne00 * type_سائيز;

        إذا (nb00 == type_سائيز) {
            // src0 is contiguous on first dimension, copy by rows
            لكل (int64_t i03 = 0; i03 < ne03; i03++) {
                لكل (int64_t i02 = 0; i02 < ne02; i02++) {
                    id += rs * ir0;
                    لكل (int64_t i01 = ir0; i01 < ir1; i01++) {
                        ثابت حرف * src0_ptr = (حرف *) src0->معلومات + i01*nb01 + i02*nb02 + i03*nb03;
                        يادگيري_نقل(dst_ptr + id, src0_ptr, rs);
                        id += rs;
                    }
                    id += rs * (ne01 - ir1);
                }
            }
        } وإلا {
            //printf("%s: this is not optimal - fix me\n", __func__);

            لكل (int64_t i03 = 0; i03 < ne03; i03++) {
                لكل (int64_t i02 = 0; i02 < ne02; i02++) {
                    id += rs * ir0;
                    لكل (int64_t i01 = ir0; i01 < ir1; i01++) {
                        لكل (int64_t i00 = 0; i00 < ne00; i00++) {
                            ثابت حرف * src0_ptr = (حرف *) src0->معلومات + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03;
                            يادگيري_نقل(dst_ptr + id, src0_ptr, type_سائيز);

                            id += type_سائيز;
                        }
                    }
                    id += rs * (ne01 - ir1);
                }
            }
        }

        أرجع;
    }

    // dst counters
    int64_t k10 = 0;
    int64_t i11 = 0;
    int64_t i12 = 0;
    int64_t i13 = 0;

    // number of blocks in a row
    ثابت int64_t nk00 = ne00 / ggml_blck_سائيز(src0->type);
    ثابت int64_t nk0  = ne0  / ggml_blck_سائيز(dst->type);

    لكل (int64_t i03 = 0; i03 < ne03; i03++) {
        لكل (int64_t i02 = 0; i02 < ne02; i02++) {
            k10 += nk00 * ir0;
            طالما (k10 >= nk0) {
                k10 -= nk0;
                إذا (++i11 == ne1) {
                    i11 = 0;
                    إذا (++i12 == ne2) {
                        i12 = 0;
                        إذا (++i13 == ne3) {
                            i13 = 0;
                        }
                    }
                }
            }
            لكل (int64_t i01 = ir0; i01 < ir1; i01++) {
                لكل (int64_t k00 = 0; k00 < nk00; k00++) {
                    ثابت حرف * src0_ptr = ((حرف *) src0->معلومات + k00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                          حرف * dst_ptr  = ((حرف *)  dst->معلومات + k10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                    يادگيري_نقل(dst_ptr, src0_ptr, type_سائيز);

                    إذا (++k10 == nk0) {
                        k10 = 0;
                        إذا (++i11 == ne1) {
                            i11 = 0;
                            إذا (++i12 == ne2) {
                                i12 = 0;
                                إذا (++i13 == ne3) {
                                    i13 = 0;
                                }
                            }
                        }
                    }
                }
            }
            k10 += nk00 * (ne01 - ir1);
            طالما (k10 >= nk0) {
                k10 -= nk0;
                إذا (++i11 == ne1) {
                    i11 = 0;
                    إذا (++i12 == ne2) {
                        i12 = 0;
                        إذا (++i13 == ne3) {
                            i13 = 0;
                        }
                    }
                }
            }
        }
    }
}

ساكن فراغ ggml_compute_forward_dup_from_q(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ٻٽو_OP_LOCALS

    ثابت ggml_type type = src0->type;
    ggml_منزل_float_t ثابت dequantize_صف_q = ggml_get_type_traits(type)->منزل_float;

    size_t qk = ggml_blck_سائيز(type);
    ثابت int64_t nr = ggml_nelements(src1) / qk;

    // destination must be contiguous in the first dimension
    GGML_ASSERT(nb10 == ggml_type_سائيز(dst->type));
    // must either have first dimension large enough to hold a row, or fully contiguous
    GGML_ASSERT((ne10 % qk) == 0 || ggml_is_contiguous(dst));

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحيح ir0 = dr*ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    لكل (int64_t ir = ir0; ir < ir1; ++ir) {

        uint32_t i = ir * qk;

        ثابت int64_t i03 = i/(ne00 * ne01 * ne02);
        ثابت int64_t i02 = (i - i03*ne00*ne01*ne02 )/ (ne00*ne01);
        ثابت int64_t i01 = (i - i03*ne00*ne01*ne02  -  i02*ne01*ne00) / ne00;
        ثابت int64_t i00 = i - i03*ne00*ne01*ne02 - i02*ne01*ne00 - i01*ne00;
        ثابت int64_t x_وٿي = (i00/qk)*nb00 + i01*nb01 + i02*nb02 + i03 * nb03;

        ثابت int64_t i13 = i/(ne10 * ne11 * ne12);
        ثابت int64_t i12 = (i - i13*ne10*ne11*ne12) / (ne10*ne11);
        ثابت int64_t i11 = (i - i13*ne10*ne11*ne12 - i12*ne10*ne11) / ne10;
        ثابت int64_t i10 = i - i13*ne10*ne11*ne12 - i12*ne10*ne11 - i11*ne10;
        ثابت int64_t dst_وٿي = i10*nb10 + i11*nb11 + i12*nb12 + i13*nb13;

        dequantize_صف_q(
                (ثابت فراغ *) ((حرف *) src0->معلومات + x_وٿي),
                     (عائم *) ((حرف *)  dst->معلومات + dst_وٿي), qk);
    }
}

فراغ ggml_compute_forward_dup(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    إذا (src0->type == dst->type) {
        ggml_compute_forward_dup_بائيٽون(params, dst);
        أرجع;
    }

    بدل (src0->type) {
        حالة GGML_TYPE_F16:
            {
                /**/ إذا (dst->type == GGML_TYPE_F16)  ggml_compute_forward_dup_flt<ggml_fp16_t, ggml_fp16_t>(params, dst);
                وإلا إذا (dst->type == GGML_TYPE_BF16) ggml_compute_forward_dup_flt<ggml_fp16_t, ggml_bf16_t>(params, dst);
                وإلا إذا (dst->type == GGML_TYPE_F32)  ggml_compute_forward_dup_flt<ggml_fp16_t, عائم      >(params, dst);
                وإلا ggml_compute_forward_dup_منزل_q<ggml_fp16_t>(params, dst);
            } اكسر;
        حالة GGML_TYPE_BF16:
            {
                /**/ إذا (dst->type == GGML_TYPE_F16)  ggml_compute_forward_dup_flt<ggml_bf16_t, ggml_fp16_t>(params, dst);
                وإلا إذا (dst->type == GGML_TYPE_BF16) ggml_compute_forward_dup_flt<ggml_bf16_t, ggml_bf16_t>(params, dst);
                وإلا إذا (dst->type == GGML_TYPE_F32)  ggml_compute_forward_dup_flt<ggml_bf16_t, عائم      >(params, dst);
                وإلا ggml_compute_forward_dup_منزل_q<ggml_bf16_t>(params, dst);
            } اكسر;
        حالة GGML_TYPE_F32:
            {
                /**/ إذا (dst->type == GGML_TYPE_F16)  ggml_compute_forward_dup_flt<عائم, ggml_fp16_t>(params, dst);
                وإلا إذا (dst->type == GGML_TYPE_BF16) ggml_compute_forward_dup_flt<عائم, ggml_bf16_t>(params, dst);
                وإلا إذا (dst->type == GGML_TYPE_F32)  ggml_compute_forward_dup_flt<عائم, عائم      >(params, dst);
                وإلا إذا (dst->type == GGML_TYPE_I32)  ggml_compute_forward_dup_flt<عائم, int32_t    >(params, dst);
                وإلا ggml_compute_forward_dup_منزل_q<عائم>(params, dst);
            } اكسر;
        حالة GGML_TYPE_I32:
            {
                إذا (dst->type == GGML_TYPE_F32) ggml_compute_forward_dup_flt<int32_t, عائم>(params, dst);
                وإلا GGML_ABORT("not implemented");
            } اكسر;
        افتراضي:
            {
                إذا (ggml_is_quantized(src0->type) && dst->type == GGML_TYPE_F32) {
                    ggml_compute_forward_dup_from_q(params, dst);
                    اكسر;
                }
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_add

ساكن فراغ ggml_compute_forward_جوڙيو_q_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

    ثابت صحيح nr  = ggml_nrows(src0);

    GGML_TENSOR_ٻٽو_OP_LOCALS

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت ggml_type type = src0->type;
    ثابت ggml_type dtype = dst->type;
    ggml_منزل_float_t ثابت dequantize_صف_q = ggml_get_type_traits(type)->منزل_float;
    ggml_from_float_t ثابت quantize_صف_q = ggml_get_type_traits_cpu(dtype)->from_float;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_سائيز(type));
    GGML_ASSERT(nb10 == حجم(عائم));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ggml_is_quantized(src0->type));
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    // rows per thread
    ثابت صحيح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحيح ir0 = dr*ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    عائم * wdata = (عائم *) params->wdata + (ne00 + CACHE_سٽ_سائيز_F32) * ith;

    لكل (صحيح ir = ir0; ir < ir1; ++ir) {
        // src0 indices
        ثابت صحيح i03 = ir/(ne02*ne01);
        ثابت صحيح i02 = (ir - i03*ne02*ne01)/ne01;
        ثابت صحيح i01 = (ir - i03*ne02*ne01 - i02*ne01);

        // src1 and dst are same shape as src0 => same indices
        ثابت صحيح i13 = i03;
        ثابت صحيح i12 = i02;
        ثابت صحيح i11 = i01;

        ثابت صحيح i3 = i03;
        ثابت صحيح i2 = i02;
        ثابت صحيح i1 = i01;

        فراغ  * src0_صف = (فراغ *) ((حرف *) src0->معلومات + (i01*nb01 + i02*nb02 + i03*nb03));
        عائم * src1_صف = (عائم *)((حرف *) src1->معلومات + (i11*nb11 + i12*nb12 + i13*nb13));
        فراغ  * dst_صف  = (فراغ *) ((حرف *)  dst->معلومات + ( i1*nb1  +  i2*nb2  +  i3*nb3));

        أكد(ne00 % 32 == 0);

        // unquantize row from src0 to temp buffer
        dequantize_صف_q(src0_صف, wdata, ne00);
        // add src1
        ggml_vec_acc_f32(ne00, wdata, src1_صف);
        // quantize row to dst
        إذا (quantize_صف_q != عدم) {
            quantize_صف_q(wdata, dst_صف, ne00);
        } وإلا {
            يادگيري_نقل(dst_صف, wdata, ne0*nb0);
        }
    }
}

فراغ ggml_compute_forward_جوڙيو(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
        حالة GGML_TYPE_F16:
        حالة GGML_TYPE_BF16:
            {
                ggml_compute_forward_جوڙيو_non_quantized(params, dst);
            } اكسر;
        حالة GGML_TYPE_Q1_0:
        حالة GGML_TYPE_Q2_0:
        حالة GGML_TYPE_Q4_0:
        حالة GGML_TYPE_Q4_1:
        حالة GGML_TYPE_Q5_0:
        حالة GGML_TYPE_Q5_1:
        حالة GGML_TYPE_Q8_0:
        حالة GGML_TYPE_MXFP4:
        حالة GGML_TYPE_NVFP4:
        حالة GGML_TYPE_Q2_K:
        حالة GGML_TYPE_Q3_K:
        حالة GGML_TYPE_Q4_K:
        حالة GGML_TYPE_Q5_K:
        حالة GGML_TYPE_Q6_K:
        حالة GGML_TYPE_TQ1_0:
        حالة GGML_TYPE_TQ2_0:
        حالة GGML_TYPE_IQ2_XXS:
        حالة GGML_TYPE_IQ2_XS:
        حالة GGML_TYPE_IQ3_XXS:
        حالة GGML_TYPE_IQ1_S:
        حالة GGML_TYPE_IQ1_M:
        حالة GGML_TYPE_IQ4_NL:
        حالة GGML_TYPE_IQ4_XS:
        حالة GGML_TYPE_IQ3_S:
        حالة GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_جوڙيو_q_f32(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_add_id

ساكن فراغ ggml_compute_forward_جوڙيو_id_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    ثابت ggml_tensor * src2 = dst->src[2];

    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src2->type == GGML_TYPE_I32);

    GGML_ASSERT(src0->nb[0] == حجم(عائم));
    GGML_ASSERT(src1->nb[0] == حجم(عائم));

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح nr  = ggml_nrows(src0);

    GGML_TENSOR_TERNARY_OP_LOCALS

    GGML_ASSERT( nb0 == حجم(عائم));
    GGML_ASSERT(nb10 == حجم(عائم));

    // rows per thread
    ثابت صحيح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحيح ir0 = dr*ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    لكل (صحيح ir = ir0; ir < ir1; ++ir) {
        // src0 indices
        ثابت صحيح i3 = ir/(ne2*ne1);
        ثابت صحيح i2 = (ir - i3*ne2*ne1)/ne1;
        ثابت صحيح i1 = (ir - i3*ne2*ne1 - i2*ne1);

        // src1 indices
        ثابت صحيح i11 = *(int32_t *) ((حرف *) src2->معلومات + i1*nb20 + i2*nb21);

        GGML_ASSERT(i11 >= 0 && i11 < ne11);

        ggml_vec_جوڙيو_f32(ne0,
                (عائم *) ((حرف *) dst->معلومات  + i3*nb3  + i2*nb2  + i1*nb1 ),
                (عائم *) ((حرف *) src0->معلومات + i3*nb03 + i2*nb02 + i1*nb01),
                (عائم *) ((حرف *) src1->معلومات + i11*nb11));
    }
}

فراغ ggml_compute_forward_جوڙيو_id(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_جوڙيو_id_f32(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("unsupported type for ggml_compute_forward_add_id: %s", ggml_type_name(src0->type));
            }
    }
}

// ggml_compute_forward_add1

ساكن فراغ ggml_compute_forward_add1_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == حجم(عائم));
    GGML_ASSERT(nb00 == حجم(عائم));

    // rows per thread
    ثابت صحيح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحيح ir0 = dr*ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    لكل (صحيح ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        ثابت صحيح i3 = ir/(ne2*ne1);
        ثابت صحيح i2 = (ir - i3*ne2*ne1)/ne1;
        ثابت صحيح i1 = (ir - i3*ne2*ne1 - i2*ne1);

#جيڪڏهن_وصف_ٿيل GGML_USE_ACCELERATE
        GGML_اڻ_استعمال_ٿيل(ggml_vec_add1_f32);

        vDSP_vadd(
                (عائم *) ((حرف *) src0->معلومات + i3*nb03 + i2*nb02 + i1*nb01), 1,
                (عائم *) ((حرف *) src1->معلومات), 0,
                (عائم *) ((حرف *) dst->معلومات  + i3*nb3  + i2*nb2  + i1*nb1 ), 1,
                ne0);
#وإلا
        ggml_vec_add1_f32(ne0,
                (عائم *) ((حرف *) dst->معلومات  + i3*nb3  + i2*nb2  + i1*nb1 ),
                (عائم *) ((حرف *) src0->معلومات + i3*nb03 + i2*nb02 + i1*nb01),
               *(عائم *) src1->معلومات);
#جيڪڏهن_پورو
    }
}

ساكن فراغ ggml_compute_forward_add1_f16_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    ثابت عائم v = *(عائم *) src1->معلومات;

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F16);

    GGML_ASSERT( nb0 == حجم(ggml_fp16_t));
    GGML_ASSERT(nb00 == حجم(ggml_fp16_t));

    // rows per thread
    ثابت صحيح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحيح ir0 = dr*ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    لكل (صحيح ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        ثابت صحيح i3 = ir/(ne2*ne1);
        ثابت صحيح i2 = (ir - i3*ne2*ne1)/ne1;
        ثابت صحيح i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((حرف *) dst->معلومات  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((حرف *) src0->معلومات + i3*nb03 + i2*nb02 + i1*nb01);
        لكل (صحيح i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_CPU_FP32_منزل_FP16(GGML_CPU_FP16_منزل_FP32(src0_ptr[i]) + v);
        }
    }
}

ساكن فراغ ggml_compute_forward_add1_f16_f16(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    ثابت عائم v = GGML_CPU_FP16_منزل_FP32(*(ggml_fp16_t *) src1->معلومات);

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type  == GGML_TYPE_F16);

    GGML_ASSERT( nb0 == حجم(ggml_fp16_t));
    GGML_ASSERT(nb00 == حجم(ggml_fp16_t));

    // rows per thread
    ثابت صحيح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحيح ir0 = dr*ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    لكل (صحيح ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        ثابت صحيح i3 = ir/(ne2*ne1);
        ثابت صحيح i2 = (ir - i3*ne2*ne1)/ne1;
        ثابت صحيح i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((حرف *) dst->معلومات  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((حرف *) src0->معلومات + i3*nb03 + i2*nb02 + i1*nb01);
        لكل (صحيح i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_CPU_FP32_منزل_FP16(GGML_CPU_FP16_منزل_FP32(src0_ptr[i]) + v);
        }
    }
}

ساكن فراغ ggml_compute_forward_add1_q_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    ثابت عائم v = *(عائم *) src1->معلومات;

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    ثابت ggml_type type = src0->type;
    ggml_منزل_float_t ثابت dequantize_صف_q = ggml_get_type_traits(type)->منزل_float;
    ggml_from_float_t ثابت quantize_صف_q = ggml_get_type_traits_cpu(type)->from_float;

    // we don't support permuted src0
    GGML_ASSERT(nb00 == ggml_type_سائيز(type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ggml_is_quantized(src0->type));
    GGML_ASSERT(dst->type == src0->type);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    // rows per thread
    ثابت صحيح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحيح ir0 = dr*ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    عائم * wdata = (عائم *) params->wdata + (ne0 + CACHE_سٽ_سائيز_F32) * ith;

    لكل (صحيح ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        ثابت صحيح i3 = ir/(ne2*ne1);
        ثابت صحيح i2 = (ir - i3*ne2*ne1)/ne1;
        ثابت صحيح i1 = (ir - i3*ne2*ne1 - i2*ne1);

        فراغ  * src0_صف = (فراغ *) ((حرف *) src0->معلومات + (i1*nb01 + i2*nb02 + i3*nb03));
        فراغ  * dst_صف  = (فراغ *) ((حرف *)  dst->معلومات + (i1*nb1  + i2*nb2  + i3*nb0 ));

        أكد(ne0 % 32 == 0);

        // unquantize row from src0 to temp buffer
        dequantize_صف_q(src0_صف, wdata, ne0);
        // add src1
        ggml_vec_acc1_f32(ne0, wdata, v);
        // quantize row to dst
        quantize_صف_q(wdata, dst_صف, ne0);
    }
}

ساكن فراغ ggml_compute_forward_add1_bf16_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    ثابت عائم v = *(عائم *) src1->معلومات;

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_BF16);

    GGML_ASSERT( nb0 == حجم(ggml_bf16_t));
    GGML_ASSERT(nb00 == حجم(ggml_bf16_t));

    // rows per thread
    ثابت صحيح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحيح ir0 = dr*ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    لكل (صحيح ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        ثابت صحيح i3 = ir/(ne2*ne1);
        ثابت صحيح i2 = (ir - i3*ne2*ne1)/ne1;
        ثابت صحيح i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_bf16_t * dst_ptr  = (ggml_bf16_t *) ((حرف *) dst->معلومات  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_bf16_t * src0_ptr = (ggml_bf16_t *) ((حرف *) src0->معلومات + i3*nb03 + i2*nb02 + i1*nb01);
        لكل (صحيح i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_FP32_منزل_BF16(GGML_BF16_منزل_FP32(src0_ptr[i]) + v);
        }
    }
}

ساكن فراغ ggml_compute_forward_add1_bf16_bf16(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    ثابت عائم v = GGML_BF16_منزل_FP32(*(ggml_bf16_t *) src1->معلومات);

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(src1->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->type  == GGML_TYPE_BF16);

    GGML_ASSERT( nb0 == حجم(ggml_bf16_t));
    GGML_ASSERT(nb00 == حجم(ggml_bf16_t));

    // rows per thread
    ثابت صحيح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحيح ir0 = dr*ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    لكل (صحيح ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        ثابت صحيح i3 = ir/(ne2*ne1);
        ثابت صحيح i2 = (ir - i3*ne2*ne1)/ne1;
        ثابت صحيح i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_bf16_t * dst_ptr  = (ggml_bf16_t *) ((حرف *) dst->معلومات  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_bf16_t * src0_ptr = (ggml_bf16_t *) ((حرف *) src0->معلومات + i3*nb03 + i2*nb02 + i1*nb01);
        لكل (صحيح i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_FP32_منزل_BF16(GGML_BF16_منزل_FP32(src0_ptr[i]) + v);
        }
    }
}

فراغ ggml_compute_forward_add1(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_add1_f32(params, dst);
            } اكسر;
        حالة GGML_TYPE_F16:
            {
                إذا (src1->type == GGML_TYPE_F16) {
                    ggml_compute_forward_add1_f16_f16(params, dst);
                }
                وإلا إذا (src1->type == GGML_TYPE_F32) {
                    ggml_compute_forward_add1_f16_f32(params, dst);
                }
                وإلا {
                    GGML_ABORT("fatal error");
                }
            } اكسر;
        حالة GGML_TYPE_BF16:
            {
                إذا (src1->type == GGML_TYPE_BF16) {
                    ggml_compute_forward_add1_bf16_bf16(params, dst);
                }
                وإلا إذا (src1->type == GGML_TYPE_F32) {
                    ggml_compute_forward_add1_bf16_f32(params, dst);
                }
                وإلا {
                    GGML_ABORT("fatal error");
                }
            } اكسر;
        حالة GGML_TYPE_Q1_0:
        حالة GGML_TYPE_Q2_0:
        حالة GGML_TYPE_Q4_0:
        حالة GGML_TYPE_Q4_1:
        حالة GGML_TYPE_Q5_0:
        حالة GGML_TYPE_Q5_1:
        حالة GGML_TYPE_Q8_0:
        حالة GGML_TYPE_Q8_1:
        حالة GGML_TYPE_MXFP4:
        حالة GGML_TYPE_NVFP4:
        حالة GGML_TYPE_Q2_K:
        حالة GGML_TYPE_Q3_K:
        حالة GGML_TYPE_Q4_K:
        حالة GGML_TYPE_Q5_K:
        حالة GGML_TYPE_Q6_K:
        حالة GGML_TYPE_TQ1_0:
        حالة GGML_TYPE_TQ2_0:
        حالة GGML_TYPE_IQ2_XXS:
        حالة GGML_TYPE_IQ2_XS:
        حالة GGML_TYPE_IQ3_XXS:
        حالة GGML_TYPE_IQ1_S:
        حالة GGML_TYPE_IQ1_M:
        حالة GGML_TYPE_IQ4_NL:
        حالة GGML_TYPE_IQ4_XS:
        حالة GGML_TYPE_IQ3_S:
        حالة GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_add1_q_f32(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_acc

ساكن فراغ ggml_compute_forward_acc_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));

    // view src0 and dst with these strides and data offset inbytes during acc
    // nb0 is implicitly element_size because src0 and dst are contiguous
    size_t nb1     = ((int32_t *) dst->op_params)[0];
    size_t nb2     = ((int32_t *) dst->op_params)[1];
    size_t nb3     = ((int32_t *) dst->op_params)[2];
    size_t وٿي  = ((int32_t *) dst->op_params)[3];
    منطقي   inplace = (منطقي) ((int32_t *) dst->op_params)[4];

    إذا (!inplace) {
        إذا (params->ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            يادگيري_نقل(
                ((حرف *)  dst->معلومات),
                ((حرف *) src0->معلومات),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح nr = ggml_nrows(src1);
    ثابت صحيح nc = src1->ne[0];

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne)
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb)

    // src0 and dst as viewed during acc
    ثابت size_t nb0 = ggml_element_سائيز(src0);

    ثابت size_t nb00 = nb0;
    ثابت size_t nb01 = nb1;
    ثابت size_t nb02 = nb2;
    ثابت size_t nb03 = nb3;

    GGML_ASSERT(وٿي + (ne10 == 0 ? 0 : ne10-1)*nb0  + (ne11 == 0 ? 0 : ne11-1)*nb1  + (ne12 == 0 ? 0 : ne12-1)*nb2  + (ne13 == 0 ? 0 : ne13-1)*nb3  < ggml_nbytes(dst));
    GGML_ASSERT(وٿي + (ne10 == 0 ? 0 : ne10-1)*nb00 + (ne11 == 0 ? 0 : ne11-1)*nb01 + (ne12 == 0 ? 0 : ne12-1)*nb02 + (ne13 == 0 ? 0 : ne13-1)*nb03 < ggml_nbytes(src0));

    GGML_ASSERT(nb10 == حجم(عائم));

    // rows per thread
    ثابت صحيح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحيح ir0 = dr*ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    لكل (صحيح ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are viewed with shape of src1 and offset
        // => same indices
        ثابت صحيح i3 = ir/(ne12*ne11);
        ثابت صحيح i2 = (ir - i3*ne12*ne11)/ne11;
        ثابت صحيح i1 = (ir - i3*ne12*ne11 - i2*ne11);

#جيڪڏهن_وصف_ٿيل GGML_USE_ACCELERATE
        vDSP_vadd(
                (عائم *) ((حرف *) src0->معلومات + i3*nb03 + i2*nb02 + i1*nb01 + وٿي), 1,
                (عائم *) ((حرف *) src1->معلومات + i3*nb13 + i2*nb12 + i1*nb11), 1,
                (عائم *) ((حرف *) dst->معلومات  + i3*nb3  + i2*nb2  + i1*nb1  + وٿي), 1, nc);
#وإلا
        ggml_vec_جوڙيو_f32(nc,
                (عائم *) ((حرف *)  dst->معلومات + i3*nb3  + i2*nb2  + i1*nb1  + وٿي),
                (عائم *) ((حرف *) src0->معلومات + i3*nb03 + i2*nb02 + i1*nb01 + وٿي),
                (عائم *) ((حرف *) src1->معلومات + i3*nb13 + i2*nb12 + i1*nb11));
#جيڪڏهن_پورو
    }
}

فراغ ggml_compute_forward_acc(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_acc_f32(params, dst);
            } اكسر;
        حالة GGML_TYPE_F16:
        حالة GGML_TYPE_BF16:
        حالة GGML_TYPE_Q1_0:
        حالة GGML_TYPE_Q2_0:
        حالة GGML_TYPE_Q4_0:
        حالة GGML_TYPE_Q4_1:
        حالة GGML_TYPE_Q5_0:
        حالة GGML_TYPE_Q5_1:
        حالة GGML_TYPE_Q8_0:
        حالة GGML_TYPE_Q8_1:
        حالة GGML_TYPE_MXFP4:
        حالة GGML_TYPE_NVFP4:
        حالة GGML_TYPE_Q2_K:
        حالة GGML_TYPE_Q3_K:
        حالة GGML_TYPE_Q4_K:
        حالة GGML_TYPE_Q5_K:
        حالة GGML_TYPE_Q6_K:
        حالة GGML_TYPE_TQ1_0:
        حالة GGML_TYPE_TQ2_0:
        حالة GGML_TYPE_IQ2_XXS:
        حالة GGML_TYPE_IQ2_XS:
        حالة GGML_TYPE_IQ3_XXS:
        حالة GGML_TYPE_IQ1_S:
        حالة GGML_TYPE_IQ1_M:
        حالة GGML_TYPE_IQ4_NL:
        حالة GGML_TYPE_IQ4_XS:
        حالة GGML_TYPE_IQ3_S:
        حالة GGML_TYPE_IQ2_S:
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_sum

ساكن فراغ ggml_compute_forward_جوڙ_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    إذا (params->ith != 0) {
        أرجع;
    }

    أكد(ggml_is_scalar(dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)

    ggml_float جوڙ = 0;

    إذا (nb00 == حجم(عائم)) {
        ggml_float صف_جوڙ = 0;
        لكل (int64_t i03 = 0; i03 < ne03; i03++) {
            لكل (int64_t i02 = 0; i02 < ne02; i02++) {
                لكل (int64_t i01 = 0; i01 < ne01; i01++) {
                    ggml_vec_جوڙ_f32_ggf(ne00,
                            &صف_جوڙ,
                            (عائم *) ((حرف *) src0->معلومات + i01*nb01 + i02*nb02 + i03*nb03));
                    جوڙ += صف_جوڙ;
                }
            }
        }
    } وإلا {
        لكل (int64_t i03 = 0; i03 < ne03; i03++) {
            لكل (int64_t i02 = 0; i02 < ne02; i02++) {
                لكل (int64_t i01 = 0; i01 < ne01; i01++) {
                    ثابت حرف * صف = (ثابت حرف *) src0->معلومات + i01*nb01 + i02*nb02 + i03*nb03;
                    لكل (int64_t i00 = 0; i00 < ne00; i00++) {
                        جوڙ += (ggml_float) *(ثابت عائم *) (صف + i00*nb00);
                    }
                }
            }
        }
    }

    ((عائم *) dst->معلومات)[0] = جوڙ;
}

ساكن فراغ ggml_compute_forward_جوڙ_f16(
    ثابت ggml_compute_params * params,
          ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    إذا (params->ith != 0) {
        أرجع;
    }

    أكد(ggml_is_scalar(dst));

    أكد(src0->nb[0] == حجم(ggml_fp16_t));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)

    عائم جوڙ = 0;
    عائم صف_جوڙ = 0;

    لكل (int64_t i03 = 0; i03 < ne03; i03++) {
        لكل (int64_t i02 = 0; i02 < ne02; i02++) {
            لكل (int64_t i01 = 0; i01 < ne01; i01++) {
                ggml_vec_جوڙ_f16_ggf(ne00,
                    &صف_جوڙ,
                    (ggml_fp16_t *) ((حرف *) src0->معلومات + i01 * nb01 + i02 * nb02 + i03 * nb03));
                جوڙ += صف_جوڙ;
            }
        }
    }
    ((ggml_fp16_t *) dst->معلومات)[0] = GGML_CPU_FP32_منزل_FP16(جوڙ);
}

ساكن فراغ ggml_compute_forward_جوڙ_bf16(
    ثابت ggml_compute_params * params,
          ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    إذا (params->ith != 0) {
        أرجع;
    }

    أكد(ggml_is_scalar(dst));

    أكد(src0->nb[0] == حجم(ggml_bf16_t));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)

    عائم جوڙ = 0;
    عائم صف_جوڙ = 0;

    لكل (int64_t i03 = 0; i03 < ne03; i03++) {
        لكل (int64_t i02 = 0; i02 < ne02; i02++) {
            لكل (int64_t i01 = 0; i01 < ne01; i01++) {
                ggml_vec_جوڙ_bf16_ggf(ne00,
                    &صف_جوڙ,
                    (ggml_bf16_t *) ((حرف *) src0->معلومات + i01 * nb01 + i02 * nb02 + i03 * nb03));
                جوڙ += صف_جوڙ;
            }
        }
    }
    ((ggml_bf16_t *) dst->معلومات)[0] = GGML_FP32_منزل_BF16(جوڙ);
}

فراغ ggml_compute_forward_جوڙ(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_جوڙ_f32(params, dst);
            } اكسر;
        حالة GGML_TYPE_F16:
            {
                ggml_compute_forward_جوڙ_f16(params, dst);
            } اكسر;
        حالة GGML_TYPE_BF16:
            {
                ggml_compute_forward_جوڙ_bf16(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_cumsum

ساكن فراغ ggml_compute_forward_cumsum_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->nb[0] == حجم(عائم));
    GGML_ASSERT(dst->nb[0] == حجم(عائم));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne01);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    ثابت خودڪار [ir0, ir1] = get_thread_range(params, src0);

    لكل (int64_t ir = ir0; ir < ir1; ++ir) {
        ثابت int64_t i03 = ir/(ne02*ne01);
        ثابت int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        ثابت int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        عائم * src_صف = (عائم *) ((حرف *) src0->معلومات + i01*nb01 + i02*nb02 + i03*nb03);
        عائم * dst_صف = (عائم *) ((حرف *) dst->معلومات  + i01*nb1  + i02*nb2  + i03*nb3);

        ggml_vec_cumsum_f32(ne00, dst_صف, src_صف);
    }
}

فراغ ggml_compute_forward_cumsum(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_cumsum_f32(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_sum_rows

ساكن فراغ ggml_compute_forward_جوڙ_صفون_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    إذا (params->ith != 0) {
        أرجع;
    }

    GGML_ASSERT(src0->nb[0] == حجم(عائم));
    GGML_ASSERT(dst->nb[0] == حجم(عائم));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(ne0 == 1);
    GGML_ASSERT(ne1 == ne01);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    لكل (int64_t i3 = 0; i3 < ne03; i3++) {
        لكل (int64_t i2 = 0; i2 < ne02; i2++) {
            لكل (int64_t i1 = 0; i1 < ne01; i1++) {
                عائم * src_صف = (عائم *) ((حرف *) src0->معلومات + i1*nb01 + i2*nb02 + i3*nb03);
                عائم * dst_صف = (عائم *) ((حرف *) dst->معلومات  + i1*nb1  + i2*nb2  + i3*nb3);
                عائم صف_جوڙ = 0;
                ggml_vec_جوڙ_f32(ne00, &صف_جوڙ, src_صف);
                dst_صف[0] = صف_جوڙ;
            }
        }
    }
}

فراغ ggml_compute_forward_جوڙ_صفون(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_جوڙ_صفون_f32(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_mean

ساكن فراغ ggml_compute_forward_سراسري_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    إذا (params->ith != 0) {
        أرجع;
    }

    أكد(src0->nb[0] == حجم(عائم));

    GGML_TENSOR_UNARY_OP_LOCALS

    أكد(ne0 == 1);
    أكد(ne1 == ne01);
    أكد(ne2 == ne02);
    أكد(ne3 == ne03);

    GGML_اڻ_استعمال_ٿيل(ne0);
    GGML_اڻ_استعمال_ٿيل(ne1);
    GGML_اڻ_استعمال_ٿيل(ne2);
    GGML_اڻ_استعمال_ٿيل(ne3);

    لكل (int64_t i03 = 0; i03 < ne03; i03++) {
        لكل (int64_t i02 = 0; i02 < ne02; i02++) {
            لكل (int64_t i01 = 0; i01 < ne01; i01++) {
                ggml_vec_جوڙ_f32(ne00,
                        (عائم *) ((حرف *)  dst->معلومات + i01*nb1  + i02*nb2  + i03*nb3),
                        (عائم *) ((حرف *) src0->معلومات + i01*nb01 + i02*nb02 + i03*nb03));

                *(عائم *) ((حرف *) dst->معلومات + i01*nb1 + i02*nb2 + i03*nb3) /= (عائم) ne00;
            }
        }
    }
}

فراغ ggml_compute_forward_سراسري(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_سراسري_f32(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_argmax

ساكن فراغ ggml_compute_forward_argmax_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    إذا (params->ith != 0) {
        أرجع;
    }

    أكد(src0->nb[0] == حجم(عائم));
    أكد(dst->nb[0] == حجم(عائم));

    ثابت int64_t ne00 = src0->ne[0];
    ثابت int64_t ne01 = src0->ne[1];

    ثابت size_t nb01 = src0->nb[1];
    ثابت size_t nb0 = dst->nb[0];

    لكل (int64_t i1 = 0; i1 < ne01; i1++) {
        عائم * src = (عائم *) ((حرف *) src0->معلومات + i1*nb01);
        int32_t * dst_ = (int32_t *) ((حرف *)  dst->معلومات + i1*nb0);
        صحيح v = 0;
        ggml_vec_argmax_f32(ne00, &v, src);
        dst_[0] = v;
    }
}

فراغ ggml_compute_forward_argmax(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_argmax_f32(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_count_equal

ساكن فراغ ggml_compute_forward_ڳڻپ_equal_i32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ٻٽو_OP_LOCALS;

    GGML_ASSERT(src0->type == GGML_TYPE_I32);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));
    GGML_ASSERT(dst->type == GGML_TYPE_I64);

    ثابت int64_t nr = ggml_nrows(src0);

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    int64_t * sums = (int64_t *) params->wdata;
    int64_t جوڙ_thread = 0;

    // rows per thread
    ثابت int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت int64_t ir0 = dr*ith;
    ثابت int64_t ir1 = MIN(ir0 + dr, nr);

    لكل (int64_t ir = ir0; ir < ir1; ++ir) {
        ثابت int64_t i03 =  ir                        / (ne02*ne01);
        ثابت int64_t i02 = (ir - i03*ne03)            /       ne01;
        ثابت int64_t i01 =  ir - i03*ne03 - i02*ne02;

        ثابت حرف * data0 = (ثابت حرف *) src0->معلومات + i03*nb03 + i02*nb02 + i01*nb01;
        ثابت حرف * data1 = (ثابت حرف *) src1->معلومات + i03*nb13 + i02*nb12 + i01*nb11;

        لكل (int64_t i00 = 0; i00 < ne00; ++i00) {
            ثابت int32_t val0 = *((ثابت int32_t *) (data0 + i00*nb00));
            ثابت int32_t val1 = *((ثابت int32_t *) (data1 + i00*nb10));

            جوڙ_thread += val0 == val1;
        }
    }
    إذا (ith != 0) {
        sums[ith] = جوڙ_thread;
    }
    ggml_barrier(params->threadpool);

    إذا (ith != 0) {
        أرجع;
    }

    لكل (صحيح ith_other = 1; ith_other < nth; ++ith_other) {
        جوڙ_thread += sums[ith_other];
    }
    *((int64_t *) dst->معلومات) = جوڙ_thread;
}

فراغ ggml_compute_forward_ڳڻپ_equal(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_I32:
            {
                ggml_compute_forward_ڳڻپ_equal_i32(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_repeat

ساكن فراغ ggml_compute_forward_repeat_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    إذا (params->ith != 0) {
        أرجع;
    }

    GGML_ASSERT(ggml_can_repeat(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    // guaranteed to be an integer due to the check in ggml_can_repeat
    ثابت صحيح nr0 = (صحيح)(ne0/ne00);
    ثابت صحيح nr1 = (صحيح)(ne1/ne01);
    ثابت صحيح nr2 = (صحيح)(ne2/ne02);
    ثابت صحيح nr3 = (صحيح)(ne3/ne03);

    // TODO: support for transposed / permuted tensors
    GGML_ASSERT(nb0  == حجم(عائم));
    GGML_ASSERT(nb00 == حجم(عائم));

    // TODO: maybe this is not optimal?
    لكل                         (صحيح i3 = 0; i3 < nr3;  i3++) {
        لكل                     (صحيح k3 = 0; k3 < ne03; k3++) {
            لكل                 (صحيح i2 = 0; i2 < nr2;  i2++) {
                لكل             (صحيح k2 = 0; k2 < ne02; k2++) {
                    لكل         (صحيح i1 = 0; i1 < nr1;  i1++) {
                        لكل     (صحيح k1 = 0; k1 < ne01; k1++) {
                            لكل (صحيح i0 = 0; i0 < nr0;  i0++) {
                                ggml_vec_cpy_f32(ne00,
                                        (عائم *) ((حرف *)  dst->معلومات + (i3*ne03 + k3)*nb3  + (i2*ne02 + k2)*nb2  + (i1*ne01 + k1)*nb1  + (i0*ne00)*nb0),
                                        (عائم *) ((حرف *) src0->معلومات + (          k3)*nb03 + (          k2)*nb02 + (          k1)*nb01));
                            }
                        }
                    }
                }
            }
        }
    }
}

ساكن فراغ ggml_compute_forward_repeat_f16(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    إذا (params->ith != 0) {
        أرجع;
    }

    GGML_ASSERT(ggml_can_repeat(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    // guaranteed to be an integer due to the check in ggml_can_repeat
    ثابت صحيح nr0 = (صحيح)(ne0/ne00);
    ثابت صحيح nr1 = (صحيح)(ne1/ne01);
    ثابت صحيح nr2 = (صحيح)(ne2/ne02);
    ثابت صحيح nr3 = (صحيح)(ne3/ne03);

    // TODO: support for transposed / permuted tensors
    GGML_ASSERT(nb0  == حجم(ggml_fp16_t));
    GGML_ASSERT(nb00 == حجم(ggml_fp16_t));

    // TODO: maybe this is not optimal?
    لكل                         (صحيح i3 = 0; i3 < nr3;  i3++) {
        لكل                     (صحيح k3 = 0; k3 < ne03; k3++) {
            لكل                 (صحيح i2 = 0; i2 < nr2;  i2++) {
                لكل             (صحيح k2 = 0; k2 < ne02; k2++) {
                    لكل         (صحيح i1 = 0; i1 < nr1;  i1++) {
                        لكل     (صحيح k1 = 0; k1 < ne01; k1++) {
                            لكل (صحيح i0 = 0; i0 < nr0;  i0++) {
                                ggml_fp16_t * y = (ggml_fp16_t *) ((حرف *)  dst->معلومات + (i3*ne03 + k3)*nb3  + (i2*ne02 + k2)*nb2  + (i1*ne01 + k1)*nb1  + (i0*ne00)*nb0);
                                ggml_fp16_t * x = (ggml_fp16_t *) ((حرف *) src0->معلومات + (          k3)*nb03 + (          k2)*nb02 + (          k1)*nb01);
                                // ggml_vec_cpy_f16(ne00, y, x)
                                لكل (صحيح i = 0; i < ne00; ++i) {
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

فراغ ggml_compute_forward_repeat(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F16:
        حالة GGML_TYPE_BF16:
        حالة GGML_TYPE_I16:
            {
                ggml_compute_forward_repeat_f16(params, dst);
            } اكسر;
        حالة GGML_TYPE_F32:
        حالة GGML_TYPE_I32:
            {
                ggml_compute_forward_repeat_f32(params, dst);
            } اكسر;
        // TODO: templateify the implementation and support for I64
        //       ref https://github.com/ggml-org/llama.cpp/pull/14274#discussion_r2169492225
        //case GGML_TYPE_I64:
        //    {
        //        ggml_compute_forward_repeat_i64(params, dst);
        //    } break;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_repeat_back

ساكن فراغ ggml_compute_forward_repeat_پٺ_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    إذا (params->ith != 0) {
        أرجع;
    }

    GGML_ASSERT(ggml_can_repeat(dst, src0));

    GGML_TENSOR_UNARY_OP_LOCALS

    // guaranteed to be an integer due to the check in ggml_can_repeat
    ثابت صحيح nr0 = (صحيح)(ne00/ne0);
    ثابت صحيح nr1 = (صحيح)(ne01/ne1);
    ثابت صحيح nr2 = (صحيح)(ne02/ne2);
    ثابت صحيح nr3 = (صحيح)(ne03/ne3);

    // TODO: support for transposed / permuted tensors
    GGML_ASSERT(nb0  == حجم(عائم));
    GGML_ASSERT(nb00 == حجم(عائم));

    إذا (ggml_is_contiguous(dst)) {
        ggml_vec_سيٽ_f32(ne0*ne1*ne2*ne3, (عائم *)dst->معلومات, 0);
    } وإلا {
        لكل         (صحيح k3 = 0; k3 < ne3; k3++) {
            لكل     (صحيح k2 = 0; k2 < ne2; k2++) {
                لكل (صحيح k1 = 0; k1 < ne1; k1++) {
                    ggml_vec_سيٽ_f32(ne0,
                        (عائم *) ((حرف *) dst->معلومات + k1*nb1 + k2*nb2 + k3*nb3),
                        0);
                }
            }
        }
    }

    // TODO: maybe this is not optimal?
    لكل                         (صحيح i3 = 0; i3 < nr3; i3++) {
        لكل                     (صحيح k3 = 0; k3 < ne3; k3++) {
            لكل                 (صحيح i2 = 0; i2 < nr2; i2++) {
                لكل             (صحيح k2 = 0; k2 < ne2; k2++) {
                    لكل         (صحيح i1 = 0; i1 < nr1; i1++) {
                        لكل     (صحيح k1 = 0; k1 < ne1; k1++) {
                            لكل (صحيح i0 = 0; i0 < nr0; i0++) {
                                ggml_vec_acc_f32(ne0,
                                        (عائم *) ((حرف *)  dst->معلومات + (         k3)*nb3  + (         k2)*nb2  + (         k1)*nb1),
                                        (عائم *) ((حرف *) src0->معلومات + (i3*ne3 + k3)*nb03 + (i2*ne2 + k2)*nb02 + (i1*ne1 + k1)*nb01 + (i0*ne0)*nb00));
                            }
                        }
                    }
                }
            }
        }
    }
}

فراغ ggml_compute_forward_repeat_پٺ(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_repeat_پٺ_f32(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_concat
ساكن فراغ ggml_compute_forward_concat_any(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    GGML_TENSOR_ٻٽو_OP_LOCALS

    ثابت int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);
    GGML_ASSERT(ggml_is_contiguous_صفون(src0));
    GGML_ASSERT(ggml_is_contiguous_صفون(src1));

    int64_t o[4] = {0, 0, 0, 0};

    إذا (dim == 0) {
        GGML_ASSERT(src0->ne[0] % ggml_blck_سائيز(src0->type) == 0);
        GGML_ASSERT(src1->ne[0] % ggml_blck_سائيز(src1->type) == 0);

        o[dim] = src0->ne[dim]/ggml_blck_سائيز(src0->type);
    } وإلا {
        o[dim] = src0->ne[dim];
    }

    // Region 1: copy rows from src0
    لكل (صحيح i3 = 0; i3 < ne03; i3++) {
        لكل (صحيح i2 = ith; i2 < ne02; i2 += nth) {
            لكل (صحيح i1 = 0; i1 < ne01; i1++) {
                ثابت حرف * x = (ثابت حرف *) src0->معلومات + i1*nb01 + i2*nb02 + i3*nb03;
                      حرف * y = (      حرف *) dst->معلومات  + i1*nb1  + i2*nb2  + i3*nb3;
                يادگيري_نقل(y, x, ggml_صف_سائيز(src0->type, ne00));
            }
        }
    }

    // Region 2: copy rows from src1, offset into dst by o[]
    لكل (صحيح i3 = 0; i3 < ne13; i3++) {
        لكل (صحيح i2 = ith; i2 < ne12; i2 += nth) {
            لكل (صحيح i1 = 0; i1 < ne11; i1++) {
                ثابت حرف * x = (ثابت حرف *) src1->معلومات + i1*nb11         + i2*nb12         + i3*nb13;
                      حرف * y = (      حرف *)  dst->معلومات + (i1 + o[1])*nb1 + (i2 + o[2])*nb2 + (i3 + o[3])*nb3 + o[0]*nb0;
                يادگيري_نقل(y, x, ggml_صف_سائيز(src1->type, ne10));
            }
        }
    }
}

ساكن فراغ ggml_compute_forward_concat_i8(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_type_سائيز(src0->type) == حجم(int8_t));

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    GGML_TENSOR_ٻٽو_OP_LOCALS

    ثابت int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);

    int64_t o[4] = {0, 0, 0, 0};
    o[dim] = src0->ne[dim];

    ثابت int8_t * x;

    // TODO: smarter multi-theading
    لكل (صحيح i3 = 0; i3 < ne3; i3++) {
        لكل (صحيح i2 = ith; i2 < ne2; i2 += nth) {
            لكل (صحيح i1 = 0; i1 < ne1; i1++) {
                لكل (صحيح i0 = 0; i0 < ne0; i0++) {
                    إذا (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                        x = (ثابت int8_t *) ((ثابت حرف *)src0->معلومات + (i0       )*nb00 + (i1       )*nb01 + (i2       )*nb02 + (i3       )*nb03);
                    } وإلا {
                        x = (ثابت int8_t *) ((ثابت حرف *)src1->معلومات + (i0 - o[0])*nb10 + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13);
                    }

                    int8_t * y = (int8_t *)((حرف *)dst->معلومات + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);

                    *y = *x;
                }
            }
        }
    }
}

ساكن فراغ ggml_compute_forward_concat_f16(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_type_سائيز(src0->type) == حجم(ggml_fp16_t));

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    GGML_TENSOR_ٻٽو_OP_LOCALS

    ثابت int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);

    int64_t o[4] = {0, 0, 0, 0};
    o[dim] = src0->ne[dim];

    ثابت ggml_fp16_t * x;

    // TODO: smarter multi-theading
    لكل (صحيح i3 = 0; i3 < ne3; i3++) {
        لكل (صحيح i2 = ith; i2 < ne2; i2 += nth) {
            لكل (صحيح i1 = 0; i1 < ne1; i1++) {
                لكل (صحيح i0 = 0; i0 < ne0; i0++) {
                    إذا (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                        x = (ثابت ggml_fp16_t *) ((ثابت حرف *)src0->معلومات + (i0       )*nb00 + (i1       )*nb01 + (i2       )*nb02 + (i3       )*nb03);
                    } وإلا {
                        x = (ثابت ggml_fp16_t *) ((ثابت حرف *)src1->معلومات + (i0 - o[0])*nb10 + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13);
                    }

                    ggml_fp16_t * y = (ggml_fp16_t *)((حرف *)dst->معلومات + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);

                    *y = *x;
                }
            }
        }
    }
}

ساكن فراغ ggml_compute_forward_concat_f32(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_type_سائيز(src0->type) == حجم(عائم));

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    GGML_TENSOR_ٻٽو_OP_LOCALS

    ثابت int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);

    int64_t o[4] = {0, 0, 0, 0};
    o[dim] = src0->ne[dim];

    ثابت عائم * x;

    // TODO: smarter multi-theading
    لكل (صحيح i3 = 0; i3 < ne3; i3++) {
        لكل (صحيح i2 = ith; i2 < ne2; i2 += nth) {
            لكل (صحيح i1 = 0; i1 < ne1; i1++) {
                لكل (صحيح i0 = 0; i0 < ne0; i0++) {
                    إذا (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                        x = (ثابت عائم *) ((ثابت حرف *)src0->معلومات + (i0       )*nb00 + (i1       )*nb01 + (i2       )*nb02 + (i3       )*nb03);
                    } وإلا {
                        x = (ثابت عائم *) ((ثابت حرف *)src1->معلومات + (i0 - o[0])*nb10 + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13);
                    }

                    عائم * y = (عائم *)((حرف *)dst->معلومات + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);

                    *y = *x;
                }
            }
        }
    }
}

فراغ ggml_compute_forward_concat(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F16:
        حالة GGML_TYPE_BF16:
        حالة GGML_TYPE_I16:
            {
                ggml_compute_forward_concat_f16(params, dst);
            } اكسر;
        حالة GGML_TYPE_I8:
            {
                ggml_compute_forward_concat_i8(params, dst);
            } اكسر;
        حالة GGML_TYPE_F32:
        حالة GGML_TYPE_I32:
            {
                ggml_compute_forward_concat_f32(params, dst);
            } اكسر;
        افتراضي:
            {
                ggml_compute_forward_concat_any(params, dst);
            }
    }
}

// row loop shared by the BF16 unary ops
سانچو <فراغ (*vec_op)(ثابت صحيح, ggml_bf16_t *, ثابت ggml_bf16_t *)>
ساكن فراغ ggml_compute_forward_unary_bf16_صفون(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    أكد(ggml_is_contiguous_صفون(src0));
    أكد(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح nc = src0->ne[0];
    ثابت صحيح nr = ggml_nrows(src0);

    // rows per thread
    ثابت صحيح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحيح ir0 = dr*ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    لكل (صحيح ir = ir0; ir < ir1; ++ir) {
        ثابت صحيح i3 = ir/(ne02*ne01);
        ثابت صحيح i2 = (ir - i3*ne02*ne01)/ne01;
        ثابت صحيح i1 = (ir - i3*ne02*ne01 - i2*ne01);

        vec_op(nc,
                (ggml_bf16_t *) ((حرف *) dst->معلومات  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_bf16_t *) ((حرف *) src0->معلومات + i3*nb03 + i2*nb02 + i1*nb01));
    }
}

// ggml_compute_forward_gelu

ساكن فراغ ggml_compute_forward_gelu_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    أكد(ggml_is_contiguous_صفون(src0));
    أكد(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح nc = src0->ne[0];
    ثابت صحيح nr = ggml_nrows(src0);

    // rows per thread
    ثابت صحيح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحيح ir0 = dr*ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    لكل (صحيح ir = ir0; ir < ir1; ++ir) {
        ثابت صحيح i3 = ir/(ne02*ne01);
        ثابت صحيح i2 = (ir - i3*ne02*ne01)/ne01;
        ثابت صحيح i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_f32(nc,
                (عائم *) ((حرف *) dst->معلومات  + i3*nb3  + i2*nb2  + i1*nb1),
                (عائم *) ((حرف *) src0->معلومات + i3*nb03 + i2*nb02 + i1*nb01));

#جيڪڏهن_وصف_نه_ٿيل NDEBUG
        لكل (صحيح k = 0; k < nc; k++) {
            ثابت عائم x = ((عائم *) ((حرف *) dst->معلومات + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_اڻ_استعمال_ٿيل(x);
            أكد(!isnan(x));
            أكد(!isinf(x));
        }
#جيڪڏهن_پورو // NDEBUG
    }
}

ساكن فراغ ggml_compute_forward_gelu_f16(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    أكد(ggml_is_contiguous_صفون(src0));
    أكد(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح nc = src0->ne[0];
    ثابت صحيح nr = ggml_nrows(src0);

    // rows per thread
    ثابت صحيح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحيح ir0 = dr*ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    لكل (صحيح ir = ir0; ir < ir1; ++ir) {
        ثابت صحيح i3 = ir/(ne02*ne01);
        ثابت صحيح i2 = (ir - i3*ne02*ne01)/ne01;
        ثابت صحيح i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_f16(nc,
                (ggml_fp16_t *) ((حرف *) dst->معلومات  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((حرف *) src0->معلومات + i3*nb03 + i2*nb02 + i1*nb01));

#جيڪڏهن_وصف_نه_ٿيل NDEBUG
        لكل (صحيح k = 0; k < nc; k++) {
            ثابت ggml_fp16_t x = ((ggml_fp16_t *) ((حرف *) dst->معلومات + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            ثابت عائم v = GGML_CPU_FP16_منزل_FP32(x);
            GGML_اڻ_استعمال_ٿيل(v);
            أكد(!isnan(v));
            أكد(!isinf(v));
        }
#جيڪڏهن_پورو // NDEBUG
    }
}

ساكن فراغ ggml_compute_forward_gelu_bf16(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_صفون<ggml_vec_gelu_bf16>(params, dst);
}

ساكن فراغ ggml_compute_forward_gelu(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_gelu_f32(params, dst);
            } اكسر;
        حالة GGML_TYPE_F16:
            {
                ggml_compute_forward_gelu_f16(params, dst);
            } اكسر;
        حالة GGML_TYPE_BF16:
            {
                ggml_compute_forward_gelu_bf16(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_fill

ساكن فراغ ggml_compute_forward_fill_f32(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    ثابت عائم c = ggml_get_op_params_f32(dst, 0);

    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne);
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb);

    ثابت خودڪار [ir0, ir1] = get_thread_range(params, dst);

    لكل (int64_t ir = ir0; ir < ir1; ++ir) {
        ثابت int64_t i03 = ir/(ne2*ne1);
        ثابت int64_t i02 = (ir - i03*ne2*ne1)/ne1;
        ثابت int64_t i01 = (ir - i03*ne2*ne1 - i02*ne1);

        عائم * dst_ptr  = (عائم *) ((حرف *) dst->معلومات + i03*nb3 + i02*nb2 + i01*nb1);

        ggml_vec_سيٽ_f32(ne0, dst_ptr, c);
    }
}

ساكن فراغ ggml_compute_forward_fill_f16(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    ثابت ggml_fp16_t c = GGML_CPU_FP32_منزل_FP16(ggml_get_op_params_f32(dst, 0));

    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne);
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb);

    ثابت خودڪار [ir0, ir1] = get_thread_range(params, dst);

    لكل (int64_t ir = ir0; ir < ir1; ++ir) {
        ثابت int64_t i03 = ir/(ne2*ne1);
        ثابت int64_t i02 = (ir - i03*ne2*ne1)/ne1;
        ثابت int64_t i01 = (ir - i03*ne2*ne1 - i02*ne1);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((حرف *) dst->معلومات + i03*nb3 + i02*nb2 + i01*nb1);

        ggml_vec_سيٽ_f16(ne0, dst_ptr, c);
    }
}

فراغ ggml_compute_forward_fill(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_fill_f32(params, dst);
            } اكسر;
        حالة GGML_TYPE_F16:
            {
                ggml_compute_forward_fill_f16(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("unsupported type for ggml_compute_forward_fill: %s", ggml_type_name(src0->type));
            }
    }
}

// ggml_compute_tri

ساكن فراغ ggml_compute_forward_tri_f32(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];

    ثابت ggml_tri_type ttype = (ggml_tri_type) ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_TENSOR_UNARY_OP_LOCALS

    ثابت خودڪار [ir0, ir1] = get_thread_range(params, src0);

    منطقي (*bipred)(صحيح, صحيح);

    بدل (ttype) {
        حالة GGML_TRI_TYPE_هيٺاهون:      bipred = [](صحيح i, صحيح r) { أرجع i <  r; }; اكسر;
        حالة GGML_TRI_TYPE_هيٺاهون_DIAG: bipred = [](صحيح i, صحيح r) { أرجع i <= r; }; اكسر;
        حالة GGML_TRI_TYPE_مٿانهون:      bipred = [](صحيح i, صحيح r) { أرجع i >  r; }; اكسر;
        حالة GGML_TRI_TYPE_مٿانهون_DIAG: bipred = [](صحيح i, صحيح r) { أرجع i >= r; }; اكسر;
        افتراضي: GGML_ABORT("invalid tri type");
    }

    لكل (int64_t ir = ir0; ir < ir1; ++ir) {
        ثابت int64_t i03 = ir/(ne02*ne01);
        ثابت int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        ثابت int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        ثابت عائم * src_ptr = (ثابت عائم  *) ((ثابت حرف *) src0->معلومات + i03*nb03 + i02*nb02 + i01*nb01);
              عائم * dst_ptr = (      عائم  *) ((      حرف *) dst->معلومات  + i03*nb3  + i02*nb2  + i01*nb1);

        لكل (صحيح i0 = 0; i0 < ne0; ++i0) {
            dst_ptr[i0] = bipred(i0, i01) ? src_ptr[i0] : 0.0f;
        }
    }
}

فراغ ggml_compute_forward_tri(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_tri_f32(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_gelu_erf

ساكن فراغ ggml_compute_forward_gelu_erf_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    أكد(ggml_is_contiguous_صفون(src0));
    أكد(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح nc = src0->ne[0];
    ثابت صحيح nr = ggml_nrows(src0);

    // rows per thread
    ثابت صحيح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحيح ir0 = dr*ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    لكل (صحيح ir = ir0; ir < ir1; ++ir) {
        ثابت صحيح i3 = ir/(ne02*ne01);
        ثابت صحيح i2 = (ir - i3*ne02*ne01)/ne01;
        ثابت صحيح i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_erf_f32(nc,
                (عائم *) ((حرف *) dst->معلومات  + i3*nb3  + i2*nb2  + i1*nb1),
                (عائم *) ((حرف *) src0->معلومات + i3*nb03 + i2*nb02 + i1*nb01));

#جيڪڏهن_وصف_نه_ٿيل NDEBUG
        لكل (صحيح k = 0; k < nc; k++) {
            ثابت عائم x = ((عائم *) ((حرف *) dst->معلومات + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_اڻ_استعمال_ٿيل(x);
            أكد(!isnan(x));
            أكد(!isinf(x));
        }
#جيڪڏهن_پورو // NDEBUG
    }
}

ساكن فراغ ggml_compute_forward_gelu_erf_f16(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    أكد(ggml_is_contiguous_صفون(src0));
    أكد(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح nc = src0->ne[0];
    ثابت صحيح nr = ggml_nrows(src0);

    // rows per thread
    ثابت صحيح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحيح ir0 = dr*ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    لكل (صحيح ir = ir0; ir < ir1; ++ir) {
        ثابت صحيح i3 = ir/(ne02*ne01);
        ثابت صحيح i2 = (ir - i3*ne02*ne01)/ne01;
        ثابت صحيح i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_erf_f16(nc,
                (ggml_fp16_t *) ((حرف *) dst->معلومات  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((حرف *) src0->معلومات + i3*nb03 + i2*nb02 + i1*nb01));

#جيڪڏهن_وصف_نه_ٿيل NDEBUG
        لكل (صحيح k = 0; k < nc; k++) {
            ثابت ggml_fp16_t x = ((ggml_fp16_t *) ((حرف *) dst->معلومات + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            ثابت عائم v = GGML_CPU_FP16_منزل_FP32(x);
            GGML_اڻ_استعمال_ٿيل(v);
            أكد(!isnan(v));
            أكد(!isinf(v));
        }
#جيڪڏهن_پورو // NDEBUG
    }
}

ساكن فراغ ggml_compute_forward_gelu_erf_bf16(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_صفون<ggml_vec_gelu_erf_bf16>(params, dst);
}

ساكن فراغ ggml_compute_forward_gelu_erf(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_gelu_erf_f32(params, dst);
            } اكسر;
        حالة GGML_TYPE_F16:
            {
                ggml_compute_forward_gelu_erf_f16(params, dst);
            } اكسر;
        حالة GGML_TYPE_BF16:
            {
                ggml_compute_forward_gelu_erf_bf16(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_gelu_quick

ساكن فراغ ggml_compute_forward_gelu_تيز_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    أكد(ggml_is_contiguous_صفون(src0));
    أكد(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح nc = src0->ne[0];
    ثابت صحيح nr = ggml_nrows(src0);

    // rows per thread
    ثابت صحيح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحيح ir0 = dr*ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    لكل (صحيح ir = ir0; ir < ir1; ++ir) {
        ثابت صحيح i3 = ir/(ne02*ne01);
        ثابت صحيح i2 = (ir - i3*ne02*ne01)/ne01;
        ثابت صحيح i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_تيز_f32(nc,
                (عائم *) ((حرف *) dst->معلومات  + i3*nb3  + i2*nb2  + i1*nb1),
                (عائم *) ((حرف *) src0->معلومات + i3*nb03 + i2*nb02 + i1*nb01));

#جيڪڏهن_وصف_نه_ٿيل NDEBUG
        لكل (صحيح k = 0; k < nc; k++) {
            ثابت عائم x = ((عائم *) ((حرف *) dst->معلومات + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_اڻ_استعمال_ٿيل(x);
            أكد(!isnan(x));
            أكد(!isinf(x));
        }
#جيڪڏهن_پورو // NDEBUG
    }
}

ساكن فراغ ggml_compute_forward_gelu_تيز_f16(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    أكد(ggml_is_contiguous_صفون(src0));
    أكد(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح nc = src0->ne[0];
    ثابت صحيح nr = ggml_nrows(src0);

    // rows per thread
    ثابت صحيح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحيح ir0 = dr*ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    لكل (صحيح ir = ir0; ir < ir1; ++ir) {
        ثابت صحيح i3 = ir/(ne02*ne01);
        ثابت صحيح i2 = (ir - i3*ne02*ne01)/ne01;
        ثابت صحيح i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_تيز_f16(nc,
                (ggml_fp16_t *) ((حرف *) dst->معلومات  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((حرف *) src0->معلومات + i3*nb03 + i2*nb02 + i1*nb01));

#جيڪڏهن_وصف_نه_ٿيل NDEBUG
        لكل (صحيح k = 0; k < nc; k++) {
            ثابت ggml_fp16_t x = ((ggml_fp16_t *) ((حرف *) dst->معلومات + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            ثابت عائم v = GGML_CPU_FP16_منزل_FP32(x);
            GGML_اڻ_استعمال_ٿيل(v);
            أكد(!isnan(v));
            أكد(!isinf(v));
        }
#جيڪڏهن_پورو // NDEBUG
    }
}

ساكن فراغ ggml_compute_forward_gelu_تيز_bf16(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_صفون<ggml_vec_gelu_تيز_bf16>(params, dst);
}

ساكن فراغ ggml_compute_forward_gelu_تيز(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_gelu_تيز_f32(params, dst);
            } اكسر;
        حالة GGML_TYPE_F16:
            {
                ggml_compute_forward_gelu_تيز_f16(params, dst);
            } اكسر;
        حالة GGML_TYPE_BF16:
            {
                ggml_compute_forward_gelu_تيز_bf16(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_silu

ساكن فراغ ggml_compute_forward_silu_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    أكد(ggml_is_contiguous_صفون(src0));
    أكد(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح nc = src0->ne[0];
    ثابت صحيح nr = ggml_nrows(src0);

    // rows per thread
    ثابت صحيح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحيح ir0 = dr*ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    لكل (صحيح ir = ir0; ir < ir1; ++ir) {
        ثابت صحيح i3 = ir/(ne02*ne01);
        ثابت صحيح i2 = (ir - i3*ne02*ne01)/ne01;
        ثابت صحيح i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_silu_f32(nc,
                (عائم *) ((حرف *) dst->معلومات  + i3*nb3  + i2*nb2  + i1*nb1),
                (عائم *) ((حرف *) src0->معلومات + i3*nb03 + i2*nb02 + i1*nb01));

#جيڪڏهن_وصف_نه_ٿيل NDEBUG
        لكل (صحيح k = 0; k < nc; k++) {
            ثابت عائم x = ((عائم *) ((حرف *) dst->معلومات + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_اڻ_استعمال_ٿيل(x);
            أكد(!isnan(x));
            أكد(!isinf(x));
        }
#جيڪڏهن_پورو // NDEBUG
    }
}

ساكن فراغ ggml_compute_forward_silu_f16(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    أكد(ggml_is_contiguous_صفون(src0));
    أكد(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح nc = src0->ne[0];
    ثابت صحيح nr = ggml_nrows(src0);

    // rows per thread
    ثابت صحيح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحيح ir0 = dr*ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    لكل (صحيح ir = ir0; ir < ir1; ++ir) {
        ثابت صحيح i3 = ir/(ne02*ne01);
        ثابت صحيح i2 = (ir - i3*ne02*ne01)/ne01;
        ثابت صحيح i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_silu_f16(nc,
                (ggml_fp16_t *) ((حرف *) dst->معلومات  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((حرف *) src0->معلومات + i3*nb03 + i2*nb02 + i1*nb01));

#جيڪڏهن_وصف_نه_ٿيل NDEBUG
        لكل (صحيح k = 0; k < nc; k++) {
            ثابت ggml_fp16_t x = ((ggml_fp16_t *) ((حرف *) dst->معلومات + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            ثابت عائم v = GGML_CPU_FP16_منزل_FP32(x);
            GGML_اڻ_استعمال_ٿيل(v);
            أكد(!isnan(v));
            أكد(!isinf(v));
        }
#جيڪڏهن_پورو // NDEBUG
    }
}

ساكن فراغ ggml_compute_forward_silu_bf16(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_صفون<ggml_vec_silu_bf16>(params, dst);
}

ساكن فراغ ggml_compute_forward_silu(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_silu_f32(params, dst);
            } اكسر;
        حالة GGML_TYPE_F16:
            {
                ggml_compute_forward_silu_f16(params, dst);
            } اكسر;
        حالة GGML_TYPE_BF16:
            {
                ggml_compute_forward_silu_bf16(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}
// ggml_compute_forward_leaky_relu

ساكن فراغ ggml_compute_forward_leaky_relu_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    إذا (params->ith != 0) {
        أرجع;
    }

    أكد(ggml_is_contiguous_1(src0));
    أكد(ggml_is_contiguous_1(dst));
    أكد(ggml_are_same_shape(src0, dst));

    ثابت صحيح n  = ggml_nrows(src0);
    ثابت صحيح nc = src0->ne[0];

    عائم negative_لاهي;
    يادگيري_نقل(&negative_لاهي, dst->op_params, حجم(عائم));

    أكد(dst->nb[0]  == حجم(عائم));
    أكد(src0->nb[0] == حجم(عائم));

    لكل (صحيح i = 0; i < n; i++) {
        ggml_vec_leaky_relu_f32(nc,
                (عائم *) ((حرف *) dst->معلومات  + i*( dst->nb[1])),
                (عائم *) ((حرف *) src0->معلومات + i*(src0->nb[1])), negative_لاهي);
    }
}

ساكن فراغ ggml_compute_forward_leaky_relu_f16(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    إذا (params->ith != 0) {
        أرجع;
    }

    أكد(ggml_is_contiguous_1(src0));
    أكد(ggml_is_contiguous_1(dst));
    أكد(ggml_are_same_shape(src0, dst));

    ثابت صحيح n  = ggml_nrows(src0);
    ثابت صحيح nc = src0->ne[0];

    عائم negative_لاهي;
    يادگيري_نقل(&negative_لاهي, dst->op_params, حجم(عائم));

    أكد(dst->nb[0]  == حجم(ggml_fp16_t));
    أكد(src0->nb[0] == حجم(ggml_fp16_t));

    لكل (صحيح i = 0; i < n; i++) {
        ggml_vec_leaky_relu_f16(nc,
                (ggml_fp16_t *) ((حرف *) dst->معلومات  + i*( dst->nb[1])),
                (ggml_fp16_t *) ((حرف *) src0->معلومات + i*(src0->nb[1])), negative_لاهي);
    }
}

فراغ ggml_compute_forward_leaky_relu(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_leaky_relu_f32(params, dst);
            } اكسر;
        حالة GGML_TYPE_F16:
            {
                ggml_compute_forward_leaky_relu_f16(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_silu_back

ساكن فراغ ggml_compute_forward_silu_پٺ_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * grad = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    أكد(ggml_is_contiguous_1(grad));
    أكد(ggml_is_contiguous_1(src1));
    أكد(ggml_is_contiguous_1(dst));
    أكد(ggml_are_same_shape(src1, dst));
    أكد(ggml_are_same_shape(src1, grad));

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح nc = src1->ne[0];
    ثابت صحيح nr = ggml_nrows(src1);

    // rows per thread
    ثابت صحيح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحيح ir0 = dr*ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    لكل (صحيح i1 = ir0; i1 < ir1; i1++) {
        ggml_vec_silu_backward_f32(nc,
                (عائم *) ((حرف *) dst->معلومات  + i1*( dst->nb[1])),
                (عائم *) ((حرف *) src1->معلومات + i1*(src1->nb[1])),
                (عائم *) ((حرف *) grad->معلومات + i1*(grad->nb[1])));

#جيڪڏهن_وصف_نه_ٿيل NDEBUG
        لكل (صحيح k = 0; k < nc; k++) {
            ثابت عائم x = ((عائم *) ((حرف *) dst->معلومات + i1*( dst->nb[1])))[k];
            GGML_اڻ_استعمال_ٿيل(x);
            أكد(!isnan(x));
            أكد(!isinf(x));
        }
#جيڪڏهن_پورو // NDEBUG
    }
}

ساكن فراغ ggml_compute_forward_silu_پٺ_f16(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * grad = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    أكد(ggml_is_contiguous_1(grad));
    أكد(ggml_is_contiguous_1(src1));
    أكد(ggml_is_contiguous_1(dst));
    أكد(ggml_are_same_shape(src1, dst));
    أكد(ggml_are_same_shape(src1, grad));

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح nc = src1->ne[0];
    ثابت صحيح nr = ggml_nrows(src1);

    // rows per thread
    ثابت صحيح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحيح ir0 = dr*ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    لكل (صحيح i1 = ir0; i1 < ir1; i1++) {
        ggml_vec_silu_backward_f16(nc,
                (ggml_fp16_t *) ((حرف *) dst->معلومات  + i1*( dst->nb[1])),
                (ggml_fp16_t *) ((حرف *) src1->معلومات + i1*(src1->nb[1])),
                (ggml_fp16_t *) ((حرف *) grad->معلومات + i1*(grad->nb[1])));

#جيڪڏهن_وصف_نه_ٿيل NDEBUG
        لكل (صحيح k = 0; k < nc; k++) {
            ثابت عائم x = ((ggml_fp16_t *) ((حرف *) dst->معلومات + i1*( dst->nb[1])))[k];
            ثابت عائم v = GGML_CPU_FP16_منزل_FP32(x);
            GGML_اڻ_استعمال_ٿيل(v);
            أكد(!isnan(v));
            أكد(!isinf(v));
        }
#جيڪڏهن_پورو // NDEBUG
    }
}

فراغ ggml_compute_forward_silu_پٺ(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_silu_پٺ_f32(params, dst);
            } اكسر;
        حالة GGML_TYPE_F16:
            {
                ggml_compute_forward_silu_پٺ_f16(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_reglu

// row loop shared by the BF16 GLU ops
سانچو <فراغ (*vec_op)(ثابت صحيح, ggml_bf16_t *, ثابت ggml_bf16_t *, ثابت ggml_bf16_t *)>
ساكن فراغ ggml_compute_forward_glu_bf16_صفون(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    حرف * src0_d = (حرف *) src0->معلومات;
    حرف * src1_d = (حرف *) (src1 ? src1->معلومات : src0->معلومات);
    ثابت size_t src0_o = src0->nb[1];
    ثابت size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    إذا (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ثابت صحيح nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ثابت int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ثابت صحيح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحيح ir0 = dr*ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    لكل (صحيح i1 = ir0; i1 < ir1; i1++) {
        ثابت ggml_bf16_t * src0_p = (ثابت ggml_bf16_t *) (src0_d + i1*src0_o);
        ثابت ggml_bf16_t * src1_p = (ثابت ggml_bf16_t *) (src1_d + i1*src1_o);
        ggml_bf16_t       * dst_p  = (ggml_bf16_t *) ((حرف *) dst->معلومات + i1*(dst->nb[1]));

        إذا (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        vec_op(nc, dst_p, src0_p, src1_p);
    }
}

ساكن فراغ ggml_compute_forward_reglu_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    حرف * src0_d = (حرف *) src0->معلومات;
    حرف * src1_d = (حرف *) (src1 ? src1->معلومات : src0->معلومات);
    ثابت size_t src0_o = src0->nb[1];
    ثابت size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    إذا (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ثابت صحيح nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ثابت int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ثابت صحيح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحيح ir0 = dr*ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    لكل (صحيح i1 = ir0; i1 < ir1; i1++) {
        عائم * src0_p = (عائم *) (src0_d + i1*src0_o);
        عائم * src1_p = (عائم *) (src1_d + i1*src1_o);

        إذا (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_reglu_f32(nc, (عائم *) ((حرف *) dst->معلومات + i1*(dst->nb[1])), src0_p, src1_p);

#جيڪڏهن_وصف_نه_ٿيل NDEBUG
        لكل (صحيح k = 0; k < nc; k++) {
            ثابت عائم x = ((عائم *) ((حرف *) dst->معلومات + i1*( dst->nb[1])))[k];
            GGML_اڻ_استعمال_ٿيل(x);
            أكد(!isnan(x));
            أكد(!isinf(x));
        }
#جيڪڏهن_پورو // NDEBUG
    }
}

ساكن فراغ ggml_compute_forward_reglu_f16(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    حرف * src0_d = (حرف *) src0->معلومات;
    حرف * src1_d = (حرف *) (src1 ? src1->معلومات : src0->معلومات);
    ثابت size_t src0_o = src0->nb[1];
    ثابت size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    إذا (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ثابت صحيح nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ثابت int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ثابت صحيح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحيح ir0 = dr*ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    لكل (صحيح i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        إذا (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_reglu_f16(nc, (ggml_fp16_t *) ((حرف *) dst->معلومات + i1*(dst->nb[1])), src0_p, src1_p);

#جيڪڏهن_وصف_نه_ٿيل NDEBUG
        لكل (صحيح k = 0; k < nc; k++) {
            ثابت ggml_fp16_t x = ((ggml_fp16_t *) ((حرف *) dst->معلومات + i1*( dst->nb[1])))[k];
            ثابت عائم v = GGML_FP16_منزل_FP32(x);
            GGML_اڻ_استعمال_ٿيل(v);
            أكد(!isnan(v));
            أكد(!isinf(v));
        }
#جيڪڏهن_پورو // NDEBUG
    }
}

ساكن فراغ ggml_compute_forward_reglu_bf16(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_صفون<ggml_vec_reglu_bf16>(params, dst);
}

ساكن فراغ ggml_compute_forward_reglu(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_reglu_f32(params, dst);
            } اكسر;
        حالة GGML_TYPE_F16:
            {
                ggml_compute_forward_reglu_f16(params, dst);
            } اكسر;
        حالة GGML_TYPE_BF16:
            {
                ggml_compute_forward_reglu_bf16(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_geglu

ساكن فراغ ggml_compute_forward_geglu_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    حرف * src0_d = (حرف *) src0->معلومات;
    حرف * src1_d = (حرف *) (src1 ? src1->معلومات : src0->معلومات);
    ثابت size_t src0_o = src0->nb[1];
    ثابت size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    إذا (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ثابت صحيح nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ثابت int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ثابت صحيح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحيح ir0 = dr*ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    لكل (صحيح i1 = ir0; i1 < ir1; i1++) {
        عائم * src0_p = (عائم *) (src0_d + i1*src0_o);
        عائم * src1_p = (عائم *) (src1_d + i1*src1_o);

        إذا (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_f32(nc, (عائم *) ((حرف *) dst->معلومات + i1*(dst->nb[1])), src0_p, src1_p);

#جيڪڏهن_وصف_نه_ٿيل NDEBUG
        لكل (صحيح k = 0; k < nc; k++) {
            ثابت عائم x = ((عائم *) ((حرف *) dst->معلومات + i1*( dst->nb[1])))[k];
            GGML_اڻ_استعمال_ٿيل(x);
            أكد(!isnan(x));
            أكد(!isinf(x));
        }
#جيڪڏهن_پورو // NDEBUG
    }
}

ساكن فراغ ggml_compute_forward_geglu_f16(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    حرف * src0_d = (حرف *) src0->معلومات;
    حرف * src1_d = (حرف *) (src1 ? src1->معلومات : src0->معلومات);
    ثابت size_t src0_o = src0->nb[1];
    ثابت size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    إذا (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ثابت صحيح nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ثابت int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ثابت صحيح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحيح ir0 = dr*ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    لكل (صحيح i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        إذا (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_f16(nc, (ggml_fp16_t *) ((حرف *) dst->معلومات + i1*(dst->nb[1])), src0_p, src1_p);

#جيڪڏهن_وصف_نه_ٿيل NDEBUG
        لكل (صحيح k = 0; k < nc; k++) {
            ثابت ggml_fp16_t x = ((ggml_fp16_t *) ((حرف *) dst->معلومات + i1*( dst->nb[1])))[k];
            ثابت عائم v = GGML_FP16_منزل_FP32(x);
            GGML_اڻ_استعمال_ٿيل(v);
            أكد(!isnan(v));
            أكد(!isinf(v));
        }
#جيڪڏهن_پورو // NDEBUG
    }
}

ساكن فراغ ggml_compute_forward_geglu_bf16(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_صفون<ggml_vec_geglu_bf16>(params, dst);
}

ساكن فراغ ggml_compute_forward_geglu(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_geglu_f32(params, dst);
            } اكسر;
        حالة GGML_TYPE_F16:
            {
                ggml_compute_forward_geglu_f16(params, dst);
            } اكسر;
        حالة GGML_TYPE_BF16:
            {
                ggml_compute_forward_geglu_bf16(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_swiglu

ساكن فراغ ggml_compute_forward_swiglu_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    حرف * src0_d = (حرف *) src0->معلومات;
    حرف * src1_d = (حرف *) (src1 ? src1->معلومات : src0->معلومات);
    ثابت size_t src0_o = src0->nb[1];
    ثابت size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    إذا (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ثابت صحيح nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ثابت int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ثابت صحيح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحيح ir0 = dr*ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    لكل (صحيح i1 = ir0; i1 < ir1; i1++) {
        عائم * src0_p = (عائم *) (src0_d + i1*src0_o);
        عائم * src1_p = (عائم *) (src1_d + i1*src1_o);

        إذا (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_swiglu_f32(nc, (عائم *) ((حرف *) dst->معلومات + i1*(dst->nb[1])), src0_p, src1_p);

#جيڪڏهن_وصف_نه_ٿيل NDEBUG
        لكل (صحيح k = 0; k < nc; k++) {
            ثابت عائم x = ((عائم *) ((حرف *) dst->معلومات + i1*( dst->nb[1])))[k];
            GGML_اڻ_استعمال_ٿيل(x);
            أكد(!isnan(x));
            أكد(!isinf(x));
        }
#جيڪڏهن_پورو // NDEBUG
    }
}

ساكن فراغ ggml_compute_forward_swiglu_f16(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    حرف * src0_d = (حرف *) src0->معلومات;
    حرف * src1_d = (حرف *) (src1 ? src1->معلومات : src0->معلومات);
    ثابت size_t src0_o = src0->nb[1];
    ثابت size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    إذا (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ثابت صحيح nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ثابت int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ثابت صحيح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحيح ir0 = dr*ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    لكل (صحيح i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        إذا (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_swiglu_f16(nc, (ggml_fp16_t *) ((حرف *) dst->معلومات + i1*(dst->nb[1])), src0_p, src1_p);

#جيڪڏهن_وصف_نه_ٿيل NDEBUG
        لكل (صحيح k = 0; k < nc; k++) {
            ثابت ggml_fp16_t x = ((ggml_fp16_t *) ((حرف *) dst->معلومات + i1*( dst->nb[1])))[k];
            ثابت عائم v = GGML_FP16_منزل_FP32(x);
            GGML_اڻ_استعمال_ٿيل(v);
            أكد(!isnan(v));
            أكد(!isinf(v));
        }
#جيڪڏهن_پورو // NDEBUG
    }
}

ساكن فراغ ggml_compute_forward_swiglu_bf16(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_صفون<ggml_vec_swiglu_bf16>(params, dst);
}

ساكن فراغ ggml_compute_forward_swiglu(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_swiglu_f32(params, dst);
            } اكسر;
        حالة GGML_TYPE_F16:
            {
                ggml_compute_forward_swiglu_f16(params, dst);
            } اكسر;
        حالة GGML_TYPE_BF16:
            {
                ggml_compute_forward_swiglu_bf16(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_swiglu_oai

ساكن فراغ ggml_compute_forward_swiglu_oai_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    حرف * src0_d = (حرف *) src0->معلومات;
    حرف * src1_d = (حرف *) (src1 ? src1->معلومات : src0->معلومات);
    ثابت size_t src0_o = src0->nb[1];
    ثابت size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    إذا (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ثابت صحيح nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ثابت int32_t swapped = ggml_get_op_params_i32(dst, 1);
    ثابت عائم alpha = ggml_get_op_params_f32(dst, 2);
    ثابت عائم حد_بندي = ggml_get_op_params_f32(dst, 3);

    // rows per thread
    ثابت صحيح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحيح ir0 = dr*ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    لكل (صحيح i1 = ir0; i1 < ir1; i1++) {
        عائم * src0_p = (عائم *) (src0_d + i1*src0_o);
        عائم * src1_p = (عائم *) (src1_d + i1*src1_o);
        عائم * dst_p  = (عائم *) ((حرف *) dst->معلومات + i1*(dst->nb[1]));

        إذا (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        لكل (صحيح k = 0; k < nc; k++) {
            ثابت عائم x = std::min(src0_p[k], حد_بندي);
            ثابت عائم y = std::clamp(src1_p[k], -حد_بندي, حد_بندي);
            ثابت عائم ٻاهريون_glu = x / (1.f + expf(alpha * (-x)));
            dst_p[k] = ٻاهريون_glu * (y + 1.f);
        }

#جيڪڏهن_وصف_نه_ٿيل NDEBUG
        لكل (صحيح k = 0; k < nc; k++) {
            ثابت عائم x = dst_p[k];
            GGML_اڻ_استعمال_ٿيل(x);
            أكد(!isnan(x));
            أكد(!isinf(x));
        }
#جيڪڏهن_پورو // NDEBUG
    }
}

ساكن فراغ ggml_compute_forward_swiglu_oai_f16(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    حرف * src0_d = (حرف *) src0->معلومات;
    حرف * src1_d = (حرف *) (src1 ? src1->معلومات : src0->معلومات);
    ثابت size_t src0_o = src0->nb[1];
    ثابت size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    إذا (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ثابت صحيح nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ثابت int32_t swapped = ggml_get_op_params_i32(dst, 1);
    ثابت عائم alpha = ggml_get_op_params_f32(dst, 2);
    ثابت عائم حد_بندي = ggml_get_op_params_f32(dst, 3);

    // rows per thread
    ثابت صحيح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحيح ir0 = dr*ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    لكل (صحيح i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);
        ggml_fp16_t * dst_p  = (ggml_fp16_t *) ((حرف *) dst->معلومات + i1*(dst->nb[1]));

        إذا (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        لكل (صحيح k = 0; k < nc; k++) {
            ثابت عائم x = std::min(GGML_CPU_FP16_منزل_FP32(src0_p[k]), حد_بندي);
            ثابت عائم y = std::clamp(GGML_CPU_FP16_منزل_FP32(src1_p[k]), -حد_بندي, حد_بندي);
            ثابت عائم ٻاهريون_glu = x / (1.f + expf(alpha * (-x)));
            dst_p[k] = GGML_CPU_FP32_منزل_FP16(ٻاهريون_glu * (y + 1.f));
        }

#جيڪڏهن_وصف_نه_ٿيل NDEBUG
        لكل (صحيح k = 0; k < nc; k++) {
            ثابت عائم x = GGML_CPU_FP16_منزل_FP32(dst_p[k]);
            GGML_اڻ_استعمال_ٿيل(x);
            أكد(!isnan(x));
            أكد(!isinf(x));
        }
#جيڪڏهن_پورو // NDEBUG
    }
}

ساكن فراغ ggml_compute_forward_swiglu_oai(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_swiglu_oai_f32(params, dst);
            } اكسر;
        حالة GGML_TYPE_F16:
            {
                ggml_compute_forward_swiglu_oai_f16(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_swiglu_clamp

ساكن فراغ ggml_compute_forward_swiglu_clamp_f32(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    ثابت ggml_tensor * src0   = dst->src[0];
    ثابت ggml_tensor * src1   = dst->src[1];
    حرف *              src0_d = (حرف *) src0->معلومات;
    حرف *              src1_d = (حرف *) (src1 ? src1->معلومات : src0->معلومات);
    ثابت size_t        src0_o = src0->nb[1];
    ثابت size_t        src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    إذا (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ثابت صحيح nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ثابت int32_t swapped = ggml_get_op_params_i32(dst, 1);
    ثابت عائم   حد_بندي   = ggml_get_op_params_f32(dst, 3);

    ثابت صحيح dr  = (nr + nth - 1) / nth;
    ثابت صحيح ir0 = dr * ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    لكل (صحيح i1 = ir0; i1 < ir1; i1++) {
        عائم * src0_p = (عائم *) (src0_d + i1 * src0_o);
        عائم * src1_p = (عائم *) (src1_d + i1 * src1_o);
        عائم * dst_p  = (عائم *) ((حرف *) dst->معلومات + i1 * (dst->nb[1]));

        إذا (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        لكل (صحيح k = 0; k < nc; k++) {
            ثابت عائم gate = std::min(src0_p[k], حد_بندي);
            ثابت عائم مٿي   = std::clamp(src1_p[k], -حد_بندي, حد_بندي);
            dst_p[k]         = gate / (1.f + expf(-gate)) * مٿي;
        }

#جيڪڏهن_وصف_نه_ٿيل NDEBUG
        لكل (صحيح k = 0; k < nc; k++) {
            ثابت عائم x = dst_p[k];
            GGML_اڻ_استعمال_ٿيل(x);
            أكد(!isnan(x));
            أكد(!isinf(x));
        }
#جيڪڏهن_پورو  // NDEBUG
    }
}

ساكن فراغ ggml_compute_forward_swiglu_clamp_f16(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    ثابت ggml_tensor * src0   = dst->src[0];
    ثابت ggml_tensor * src1   = dst->src[1];
    حرف *              src0_d = (حرف *) src0->معلومات;
    حرف *              src1_d = (حرف *) (src1 ? src1->معلومات : src0->معلومات);
    ثابت size_t        src0_o = src0->nb[1];
    ثابت size_t        src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    إذا (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ثابت صحيح nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ثابت int32_t swapped = ggml_get_op_params_i32(dst, 1);
    ثابت عائم   حد_بندي   = ggml_get_op_params_f32(dst, 3);

    ثابت صحيح dr  = (nr + nth - 1) / nth;
    ثابت صحيح ir0 = dr * ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    لكل (صحيح i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1 * src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1 * src1_o);
        ggml_fp16_t * dst_p  = (ggml_fp16_t *) ((حرف *) dst->معلومات + i1 * (dst->nb[1]));

        إذا (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        لكل (صحيح k = 0; k < nc; k++) {
            ثابت عائم gate = std::min(GGML_FP16_منزل_FP32(src0_p[k]), حد_بندي);
            ثابت عائم مٿي   = std::clamp(GGML_FP16_منزل_FP32(src1_p[k]), -حد_بندي, حد_بندي);
            dst_p[k]         = GGML_FP32_منزل_FP16(gate / (1.f + expf(-gate)) * مٿي);
        }

#جيڪڏهن_وصف_نه_ٿيل NDEBUG
        لكل (صحيح k = 0; k < nc; k++) {
            ثابت عائم x = GGML_FP16_منزل_FP32(dst_p[k]);
            GGML_اڻ_استعمال_ٿيل(x);
            أكد(!isnan(x));
            أكد(!isinf(x));
        }
#جيڪڏهن_پورو  // NDEBUG
    }
}

ساكن فراغ ggml_compute_forward_swiglu_clamp(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    بدل (dst->src[0]->type) {
        حالة GGML_TYPE_F32:
            ggml_compute_forward_swiglu_clamp_f32(params, dst);
            اكسر;
        حالة GGML_TYPE_F16:
            ggml_compute_forward_swiglu_clamp_f16(params, dst);
            اكسر;
        افتراضي:
            GGML_ABORT("fatal error");
    }
}

// ggml_compute_forward_geglu_erf

ساكن فراغ ggml_compute_forward_geglu_erf_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    حرف * src0_d = (حرف *) src0->معلومات;
    حرف * src1_d = (حرف *) (src1 ? src1->معلومات : src0->معلومات);
    ثابت size_t src0_o = src0->nb[1];
    ثابت size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    إذا (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ثابت صحيح nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ثابت int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ثابت صحيح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحيح ir0 = dr*ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    لكل (صحيح i1 = ir0; i1 < ir1; i1++) {
        عائم * src0_p = (عائم *) (src0_d + i1*src0_o);
        عائم * src1_p = (عائم *) (src1_d + i1*src1_o);

        إذا (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_erf_f32(nc, (عائم *) ((حرف *) dst->معلومات + i1*(dst->nb[1])), src0_p, src1_p);

#جيڪڏهن_وصف_نه_ٿيل NDEBUG
        لكل (صحيح k = 0; k < nc; k++) {
            ثابت عائم x = ((عائم *) ((حرف *) dst->معلومات + i1*( dst->nb[1])))[k];
            GGML_اڻ_استعمال_ٿيل(x);
            أكد(!isnan(x));
            أكد(!isinf(x));
        }
#جيڪڏهن_پورو // NDEBUG
    }
}

ساكن فراغ ggml_compute_forward_geglu_erf_f16(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    حرف * src0_d = (حرف *) src0->معلومات;
    حرف * src1_d = (حرف *) (src1 ? src1->معلومات : src0->معلومات);
    ثابت size_t src0_o = src0->nb[1];
    ثابت size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    إذا (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ثابت صحيح nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ثابت int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ثابت صحيح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحيح ir0 = dr*ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    لكل (صحيح i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        إذا (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_erf_f16(nc, (ggml_fp16_t *) ((حرف *) dst->معلومات + i1*(dst->nb[1])), src0_p, src1_p);

#جيڪڏهن_وصف_نه_ٿيل NDEBUG
        لكل (صحيح k = 0; k < nc; k++) {
            ثابت ggml_fp16_t x = ((ggml_fp16_t *) ((حرف *) dst->معلومات + i1*( dst->nb[1])))[k];
            ثابت عائم v = GGML_FP16_منزل_FP32(x);
            GGML_اڻ_استعمال_ٿيل(v);
            أكد(!isnan(v));
            أكد(!isinf(v));
        }
#جيڪڏهن_پورو // NDEBUG
    }
}

ساكن فراغ ggml_compute_forward_geglu_erf_bf16(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_صفون<ggml_vec_geglu_erf_bf16>(params, dst);
}

ساكن فراغ ggml_compute_forward_geglu_erf(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_geglu_erf_f32(params, dst);
            } اكسر;
        حالة GGML_TYPE_F16:
            {
                ggml_compute_forward_geglu_erf_f16(params, dst);
            } اكسر;
        حالة GGML_TYPE_BF16:
            {
                ggml_compute_forward_geglu_erf_bf16(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_geglu_quick

ساكن فراغ ggml_compute_forward_geglu_تيز_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    حرف * src0_d = (حرف *) src0->معلومات;
    حرف * src1_d = (حرف *) (src1 ? src1->معلومات : src0->معلومات);
    ثابت size_t src0_o = src0->nb[1];
    ثابت size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    إذا (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ثابت صحيح nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ثابت int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ثابت صحيح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحيح ir0 = dr*ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    لكل (صحيح i1 = ir0; i1 < ir1; i1++) {
        عائم * src0_p = (عائم *) (src0_d + i1*src0_o);
        عائم * src1_p = (عائم *) (src1_d + i1*src1_o);

        إذا (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_تيز_f32(nc, (عائم *) ((حرف *) dst->معلومات + i1*(dst->nb[1])), src0_p, src1_p);

#جيڪڏهن_وصف_نه_ٿيل NDEBUG
        لكل (صحيح k = 0; k < nc; k++) {
            ثابت عائم x = ((عائم *) ((حرف *) dst->معلومات + i1*( dst->nb[1])))[k];
            GGML_اڻ_استعمال_ٿيل(x);
            أكد(!isnan(x));
            أكد(!isinf(x));
        }
#جيڪڏهن_پورو // NDEBUG
    }
}

ساكن فراغ ggml_compute_forward_geglu_تيز_f16(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    حرف * src0_d = (حرف *) src0->معلومات;
    حرف * src1_d = (حرف *) (src1 ? src1->معلومات : src0->معلومات);
    ثابت size_t src0_o = src0->nb[1];
    ثابت size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    إذا (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ثابت صحيح nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ثابت int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ثابت صحيح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحيح ir0 = dr*ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    لكل (صحيح i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        إذا (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_تيز_f16(nc, (ggml_fp16_t *) ((حرف *) dst->معلومات + i1*(dst->nb[1])), src0_p, src1_p);

#جيڪڏهن_وصف_نه_ٿيل NDEBUG
        لكل (صحيح k = 0; k < nc; k++) {
            ثابت ggml_fp16_t x = ((ggml_fp16_t *) ((حرف *) dst->معلومات + i1*( dst->nb[1])))[k];
            ثابت عائم v = GGML_FP16_منزل_FP32(x);
            GGML_اڻ_استعمال_ٿيل(v);
            أكد(!isnan(v));
            أكد(!isinf(v));
        }
#جيڪڏهن_پورو // NDEBUG
    }
}

ساكن فراغ ggml_compute_forward_geglu_تيز_bf16(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_صفون<ggml_vec_geglu_تيز_bf16>(params, dst);
}

ساكن فراغ ggml_compute_forward_geglu_تيز(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_geglu_تيز_f32(params, dst);
            } اكسر;
        حالة GGML_TYPE_F16:
            {
                ggml_compute_forward_geglu_تيز_f16(params, dst);
            } اكسر;
        حالة GGML_TYPE_BF16:
            {
                ggml_compute_forward_geglu_تيز_bf16(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_norm

ساكن فراغ ggml_compute_forward_norm_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    عائم eps;
    يادگيري_نقل(&eps, dst->op_params, حجم(عائم));

    GGML_ASSERT(eps >= 0.0f);

    لكل (int64_t i03 = 0; i03 < ne03; i03++) {
        لكل (int64_t i02 = 0; i02 < ne02; i02++) {
            لكل (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                ثابت حرف * x = (ثابت حرف *) src0->معلومات + i01*nb01 + i02*nb02 + i03*nb03;
                حرف * y = (حرف *) dst->معلومات + i01*nb1 + i02*nb2 + i03*nb3;

                إذا (nb00 == حجم(عائم) && nb0 == حجم(عائم)) {
                    ثابت عائم * xf = (ثابت عائم *) x;

                    عائم جوڙ = 0.0;
                    ggml_vec_جوڙ_f32(ne00, &جوڙ, xf);
                    عائم سراسري_2 = جوڙ/ne00;

                    عائم * yf = (عائم *) y;
                    عائم variance = 0;

#جيڪڏهن_وصف_ٿيل GGML_USE_ACCELERATE
                    سراسري_2 = -سراسري_2;
                    vDSP_vsadd(xf, 1, &سراسري_2, yf, 1, ne00);
                    vDSP_measqv(yf, 1, &variance, ne00);
#وإلا
                    variance = ggml_vec_cvar_f32(ne00, yf, xf, سراسري_2);
#جيڪڏهن_پورو //GGML_USE_ACCELERATE

                    ثابت عائم scale = 1.0f/sqrtf(variance + eps);
                    ggml_vec_scale_f32(ne00, yf, scale);
                } وإلا {
                    عائم جوڙ = 0.0;
                    لكل (int64_t i00 = 0; i00 < ne00; i00++) {
                        جوڙ += *(ثابت عائم *) (x + i00*nb00);
                    }
                    ثابت عائم سراسري_2 = جوڙ/ne00;

                    عائم variance = 0.0f;
                    لكل (int64_t i00 = 0; i00 < ne00; i00++) {
                        ثابت عائم v = *(ثابت عائم *) (x + i00*nb00) - سراسري_2;
                        *(عائم *) (y + i00*nb0) = v;
                        variance += v * v;
                    }
                    variance /= ne00;

                    ثابت عائم scale = 1.0f/sqrtf(variance + eps);
                    لكل (int64_t i00 = 0; i00 < ne00; i00++) {
                        *(عائم *) (y + i00*nb0) *= scale;
                    }
                }
            }
        }
    }
}

فراغ ggml_compute_forward_norm(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_norm_f32(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_group_rms_norm

// fusion kinds that can be combined with the rms_norm computation in a single pass.
// extend this enum when adding new fused variants (e.g. FUSE_ADD, FUSE_MUL_ADD, ...).
تعداد ggml_rms_norm_fuse_op {
    GGML_RMS_NORM_FUSE_OP_NONE,
    GGML_RMS_NORM_FUSE_OP_MUL,
};

سانچو <ggml_rms_norm_fuse_op FUSE_OP>
ساكن فراغ ggml_compute_forward_rms_norm_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst_rms_norm,
        ggml_tensor * dst_fused = خالي_اشارو) {

    ثابت ggml_tensor * src0 = dst_rms_norm->src[0];
    ثابت ggml_tensor * src1 = خالي_اشارو;
    ggml_tensor       * dst  = dst_rms_norm;

    إذا constexpr (FUSE_OP == GGML_RMS_NORM_FUSE_OP_MUL) {
        src1 = (dst_fused->src[0] == dst_rms_norm) ? dst_fused->src[1] : dst_fused->src[0];
        dst  = dst_fused;
    }

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    GGML_ASSERT(src0->nb[0] == حجم(عائم));

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    GGML_TENSOR_ٻٽو_OP_LOCALS

    عائم eps;
    يادگيري_نقل(&eps, dst_rms_norm->op_params, حجم(عائم));
    GGML_ASSERT(eps >= 0.0f);

    // TODO: optimize
    لكل (int64_t i03 = 0; i03 < ne03; i03++) {
        لكل (int64_t i02 = 0; i02 < ne02; i02++) {
            لكل (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                ثابت عائم * x = (عائم *) ((حرف *) src0->معلومات + i01*nb01 + i02*nb02 + i03*nb03);

                ggml_float جوڙ = 0.0;
                // worth switching to explicit SIMD?
                لكل (int64_t i00 = 0; i00 < ne00; i00++) {
                    جوڙ += (ggml_float)(x[i00] * x[i00]);
                }

                ثابت عائم سراسري_2  = جوڙ/ne00;
                ثابت عائم scale = 1.0f/sqrtf(سراسري_2 + eps);

                // if you hit this, likely you got an inf somewhere earlier
                أكد(scale > 0.0f);

                عائم * y = (عائم *) ((حرف *) dst->معلومات + i01*nb1 + i02*nb2 + i03*nb3);

                إذا constexpr (FUSE_OP == GGML_RMS_NORM_FUSE_OP_MUL) {
                    ثابت int64_t i11 = i01 % ne11;
                    ثابت int64_t i12 = i02 % ne12;
                    ثابت int64_t i13 = i03 % ne13;
                    ثابت عائم * w = (عائم *) ((حرف *) src1->معلومات + i11*nb11 + i12*nb12 + i13*nb13);

                    لكل (int64_t i00 = 0; i00 < ne00; i00++) {
                        y[i00] = x[i00] * scale * w[i00];
                    }
                } وإلا {
                    يادگيري_نقل(y, x, ne00 * حجم(عائم));
                    ggml_vec_scale_f32(ne00, y, scale);
                }
            }
        }
    }
}

فراغ ggml_compute_forward_rms_norm(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_rms_norm_f32<GGML_RMS_NORM_FUSE_OP_NONE>(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// Fused RMS_NORM + MUL: computes dst = rms_norm(src0) * src1 in a single pass.
// This avoids materializing the intermediate rms_norm result in memory.
فراغ ggml_compute_forward_rms_norm_mul_fused(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst_rms_norm,
        ggml_tensor * dst_mul) {

    GGML_ASSERT(dst_mul != خالي_اشارو);
    GGML_ASSERT(dst_mul->src[0] == dst_rms_norm || dst_mul->src[1] == dst_rms_norm);

    ثابت ggml_tensor * src0 = dst_rms_norm->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_rms_norm_f32<GGML_RMS_NORM_FUSE_OP_MUL>(params, dst_rms_norm, dst_mul);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ساكن فراغ ggml_compute_forward_rms_norm_پٺ_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0]; // gradients from forward pass output
    ثابت ggml_tensor * src1 = dst->src[1]; // src1 from forward pass

    GGML_ASSERT(ggml_are_same_shape(src0, dst) && ggml_are_same_shape(src0, src1));

    GGML_ASSERT(src0->nb[0] == حجم(عائم));
    GGML_ASSERT(src1->nb[0] == حجم(عائم));

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    GGML_TENSOR_ٻٽو_OP_LOCALS

    عائم eps;
    يادگيري_نقل(&eps, dst->op_params, حجم(عائم));

    // TODO: optimize
    لكل (int64_t i03 = 0; i03 < ne03; i03++) {
        لكل (int64_t i02 = 0; i02 < ne02; i02++) {
            لكل (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                // src1 is same shape as src0 => same indices
                ثابت int64_t i11 = i01;
                ثابت int64_t i12 = i02;
                ثابت int64_t i13 = i03;

                ثابت عائم * dz = (عائم *) ((حرف *) src0->معلومات + i01*nb01 + i02*nb02 + i03*nb03);
                ثابت عائم * x  = (عائم *) ((حرف *) src1->معلومات + i11*nb11 + i12*nb12 + i13*nb13);

                ggml_float جوڙ_xx  = 0.0;
                ggml_float جوڙ_xdz = 0.0;

                لكل (int64_t i00 = 0; i00 < ne00; i00++) {
                    جوڙ_xx  += (ggml_float)(x[i00] * x[i00]);
                    جوڙ_xdz += (ggml_float)(x[i00] * dz[i00]);
                }

                //const float mean     = (float)(sum_xx)/ne00;
                ثابت عائم سراسري_eps = (عائم)(جوڙ_xx)/ne00 + eps;
                ثابت عائم جوڙ_eps  = (عائم)(جوڙ_xx) + eps*ne00;
                //const float mean_xdz = (float)(sum_xdz)/ne00;
                // we could cache rms from forward pass to improve performance.
                // to do this implement ggml_rms and compose ggml_rms_norm using ggml_rms.
                //const float rms      = sqrtf(mean_eps);
                ثابت عائم rrms     = 1.0f / sqrtf(سراسري_eps);
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
                عائم * dx = (عائم *) ((حرف *) dst->معلومات + i01*nb1 + i02*nb2 + i03*nb3);

                // dx[i00] = (dz + x*(-sum_xdz/sum_eps)) * rrms
                // note: https://github.com/ggml-org/ggml/issues/1491
                ثابت عائم scale_x = (عائم) (-جوڙ_xdz) / جوڙ_eps;
                لكل (int64_t i00 = 0; i00 < ne00; i00++) {
                    dx[i00] = (dz[i00] + x[i00] * scale_x) * rrms;
                }
            }
        }
    }
}

فراغ ggml_compute_forward_rms_norm_پٺ(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_rms_norm_پٺ_f32(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_group_norm

ساكن فراغ ggml_compute_forward_گروهه_norm_f32(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    GGML_ASSERT(src0->nb[0] == حجم(عائم));

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    // TODO: optimize

    عائم eps;
    يادگيري_نقل(&eps, dst->op_params + 1, حجم(عائم));

    صحيح n_channels = src0->ne[2];
    صحيح n_groups = dst->op_params[0];
    صحيح n_channels_في_گروهه = (n_channels + n_groups - 1) / n_groups;
    لكل (صحيح i = ith; i < n_groups; i += nth) {
        صحيح start = i * n_channels_في_گروهه;
        صحيح end = start + n_channels_في_گروهه;
        إذا (end > n_channels) {
            end = n_channels;
        }
        صحيح قدم = end - start;

        لكل (int64_t i03 = 0; i03 < ne03; i03++) {
            ggml_float جوڙ = 0.0;
            لكل (int64_t i02 = start; i02 < end; i02++) {
                لكل (int64_t i01 = 0; i01 < ne01; i01++) {
                    ثابت عائم * x = (عائم *)((حرف *) src0->معلومات + i01 * nb01 + i02 * nb02 + i03 * nb03);

                    ggml_float sumr = 0.0;
                    لكل (int64_t i00 = 0; i00 < ne00; i00++) {
                        sumr += (ggml_float)x[i00];
                    }
                    جوڙ += sumr;
                }
            }
            ثابت عائم سراسري_2 = جوڙ / (ne00 * ne01 * قدم);

            ggml_float sum2 = 0.0;
            لكل (int64_t i02 = start; i02 < end; i02++) {
                لكل (int64_t i01 = 0; i01 < ne01; i01++) {
                    ثابت عائم * x = (عائم *)((حرف *) src0->معلومات + i01 * nb01 + i02 * nb02 + i03 * nb03);

                    عائم * y = (عائم *)((حرف *) dst->معلومات + i01 * nb1 + i02 * nb2 + i03 * nb3);

                    ggml_float sumr = 0.0;
                    لكل (int64_t i00 = 0; i00 < ne00; i00++) {
                        عائم v = x[i00] - سراسري_2;
                        y[i00] = v;
                        sumr += (ggml_float)(v * v);
                    }
                    sum2 += sumr;
                }
            }
            ثابت عائم variance = sum2 / (ne00 * ne01 * قدم);
            ثابت عائم scale = 1.0f / sqrtf(variance + eps);

            لكل (int64_t i02 = start; i02 < end; i02++) {
                لكل (int64_t i01 = 0; i01 < ne01; i01++) {
                    عائم * y = (عائم *)((حرف *) dst->معلومات + i01 * nb1 + i02 * nb2 + i03 * nb3);
                    ggml_vec_scale_f32(ne00, y, scale);
                }
            }
        }
    }
}

فراغ ggml_compute_forward_گروهه_norm(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_گروهه_norm_f32(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_l2_norm

ساكن فراغ ggml_compute_forward_l2_norm_f32(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    عائم eps;
    يادگيري_نقل(&eps, dst->op_params, حجم(عائم));

    GGML_ASSERT(eps >= 0.0f);

    // TODO: optimize
    لكل (int64_t i03 = 0; i03 < ne03; i03++) {
        لكل (int64_t i02 = 0; i02 < ne02; i02++) {
            لكل (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                ثابت حرف * x = (ثابت حرف *) src0->معلومات + i01*nb01 + i02*nb02 + i03*nb03;

                ggml_float جوڙ = 0.0;
                لكل (int64_t i00 = 0; i00 < ne00; i00++) {
                    ثابت عائم xi = *(ثابت عائم *) (x + i00*nb00);
                    جوڙ += (ggml_float)(xi * xi);
                }

                ثابت عائم scale = 1.0f/fmaxf(sqrtf(جوڙ), eps);

                حرف * y = (حرف *) dst->معلومات + i01*nb1 + i02*nb2 + i03*nb3;

                إذا (nb00 == حجم(عائم) && nb0 == حجم(عائم)) {
                    يادگيري_نقل(y, x, ne00 * حجم(عائم));
                    ggml_vec_scale_f32(ne00, (عائم *) y, scale);
                } وإلا {
                    لكل (int64_t i00 = 0; i00 < ne00; i00++) {
                        ثابت عائم xi = *(ثابت عائم *) (x + i00*nb00);
                        *(عائم *) (y + i00*nb0) = xi * scale;
                    }
                }
            }
        }
    }
}

فراغ ggml_compute_forward_l2_norm(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_l2_norm_f32(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_out_prod

ساكن فراغ ggml_compute_forward_ٻاهريون_prod_f32(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ٻٽو_OP_LOCALS

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    GGML_ASSERT(ne2 % ne02 == 0);
    GGML_ASSERT(ne3 % ne03 == 0);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == حجم(عائم));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == حجم(عائم));
    // GGML_ASSERT(nb0 <= nb1);
    // GGML_ASSERT(nb1 <= nb2);
    // GGML_ASSERT(nb2 <= nb3);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    إذا (ith == 0) {
        ggml_vec_سيٽ_f32(ne0*ne1*ne2*ne3, (عائم *)dst->معلومات, 0);
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
    ثابت int64_t nr = ne1*ne2*ne3;

    // rows per thread
    ثابت int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت int64_t ir0 = dr*ith;
    ثابت int64_t ir1 = MIN(ir0 + dr, nr);

    // block-tiling attempt
    ثابت int64_t blck_0 = MAX(GGML_VEC_MAD_UNROLL, 32);
    ثابت int64_t blck_1 = 16;

    // dps == dst per src0, used for group query attention
    ثابت int64_t dps2 = ne2 / ne02;
    ثابت int64_t dps3 = ne3 / ne03;

    لكل (int64_t bir = ir0; bir < ir1; bir += blck_1) {
        ثابت int64_t bir1 = MIN(bir + blck_1, ir1);
        لكل (int64_t bi01 = 0; bi01 < ne01; bi01 += blck_0) {
            ثابت int64_t bne01 = MIN(bi01 + blck_0, ne01);
            لكل (int64_t ir = bir; ir < bir1; ++ir) {
                // dst indices
                ثابت int64_t i3 = ir/(ne2*ne1);
                ثابت int64_t i2 = (ir - i3*ne2*ne1)/ne1;
                ثابت int64_t i1 = (ir - i3*ne2*ne1 - i2*ne1);

                ثابت int64_t i02 = i2 / dps2;
                ثابت int64_t i03 = i3 / dps3;

                //const int64_t i10 = i1;
                ثابت int64_t i12 = i2;
                ثابت int64_t i13 = i3;

#إذا GGML_VEC_MAD_UNROLL > 2
                ثابت int64_t bne01_unroll = bne01 - (bne01 % GGML_VEC_MAD_UNROLL);
                لكل (int64_t i01 = bi01; i01 < bne01_unroll; i01 += GGML_VEC_MAD_UNROLL) {
                    ثابت int64_t i11 = i01;

                    عائم * s0 = (عائم *) ((حرف *) src0->معلومات + (          i01*nb01 + i02*nb02 + i03*nb03));
                    عائم * s1 = (عائم *) ((حرف *) src1->معلومات + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
                    عائم * d  = (عائم *) ((حرف *)  dst->معلومات + (          i1*nb1   + i2*nb2   + i3*nb3));

                    ggml_vec_mad_f32_unroll(ne0, nb01, nb11, d, s0, s1);
                }
                لكل (int64_t i01 = bne01_unroll; i01 < bne01; ++i01) {
                    ثابت int64_t i11 = i01;

                    عائم * s0 = (عائم *) ((حرف *) src0->معلومات + (          i01*nb01 + i02*nb02 + i03*nb03));
                    عائم * s1 = (عائم *) ((حرف *) src1->معلومات + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
                    عائم * d  = (عائم *) ((حرف *)  dst->معلومات + (          i1*nb1   + i2*nb2   + i3*nb3));

                    ggml_vec_mad_f32(ne0, d, s0, *s1);
                }
#وإلا
                لكل (int64_t i01 = bi01; i01 < bne01; ++i01) {
                    ثابت int64_t i11 = i01;

                    عائم * s0 = (عائم *) ((حرف *) src0->معلومات + (          i01*nb01 + i02*nb02 + i03*nb03));
                    عائم * s1 = (عائم *) ((حرف *) src1->معلومات + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
                    عائم * d  = (عائم *) ((حرف *)  dst->معلومات + (          i1*nb1 + i2*nb2 + i3*nb3));

                    ggml_vec_mad_f32(ne0, d, s0, *s1);
                }
#جيڪڏهن_پورو
            }
        }
    }
}

ساكن فراغ ggml_compute_forward_ٻاهريون_prod_q_f32(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ٻٽو_OP_LOCALS;

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت ggml_type type = src0->type;
    ggml_منزل_float_t ثابت dequantize_صف_q = ggml_get_type_traits(type)->منزل_float;

    GGML_ASSERT(ne02 == ne12);
    GGML_ASSERT(ne03 == ne13);
    GGML_ASSERT(ne2  == ne12);
    GGML_ASSERT(ne3  == ne13);

    // we don't support permuted src0 dim0
    GGML_ASSERT(nb00 == ggml_type_سائيز(type));

    // dst dim0 cannot be transposed or permuted
    GGML_ASSERT(nb0 == حجم(عائم));
    // GGML_ASSERT(nb0 <= nb1);
    // GGML_ASSERT(nb1 <= nb2);
    // GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    إذا (ith == 0) {
        ggml_vec_سيٽ_f32(ne0*ne1*ne2*ne3, (عائم *)dst->معلومات, 0);
    }
    ggml_barrier(params->threadpool);

    // parallelize by last three dimensions

    // total rows in dst
    ثابت int64_t nr = ne1*ne2*ne3;

    // rows per thread
    ثابت int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت int64_t ir0 = dr*ith;
    ثابت int64_t ir1 = MIN(ir0 + dr, nr);

    // dst[:,:,:,:] = 0
    // for i2,i3:
    //   for i1:
    //     for i01:
    //       for i0:
    //         dst[i0,i1,i2,i3] += src0[i0,i01,i2,i3] * src1[i1,i01,i2,i3]

    عائم * wdata = (عائم *) params->wdata + (ne0 + CACHE_سٽ_سائيز_F32) * ith;

    لكل (int64_t ir = ir0; ir < ir1; ++ir) {
        // dst indices
        ثابت int64_t i3 = ir/(ne2*ne1);
        ثابت int64_t i2 = (ir - i3*ne2*ne1)/ne1;
        ثابت int64_t i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ثابت int64_t i02 = i2;
        ثابت int64_t i03 = i3;

        //const int64_t i10 = i1;
        ثابت int64_t i12 = i2;
        ثابت int64_t i13 = i3;

        لكل (int64_t i01 = 0; i01 < ne01; ++i01) {
            ثابت int64_t i11 = i01;

            عائم * s0 = (عائم *) ((حرف *) src0->معلومات + (          i01*nb01 + i02*nb02 + i03*nb03));
            عائم * s1 = (عائم *) ((حرف *) src1->معلومات + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
            عائم * d  = (عائم *) ((حرف *)  dst->معلومات + (          i1*nb1 + i2*nb2 + i3*nb3));

            dequantize_صف_q(s0, wdata, ne0);
            ggml_vec_mad_f32(ne0, d, wdata, *s1);
        }
    }
}

ساكن فراغ ggml_compute_forward_ٻاهريون_prod_f16_f32(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ٻٽو_OP_LOCALS;

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ne02 == ne12);
    GGML_ASSERT(ne03 == ne13);
    GGML_ASSERT(ne2  == ne12);
    GGML_ASSERT(ne3  == ne13);

    GGML_ASSERT(nb00 == حجم(ggml_fp16_t));
    GGML_ASSERT(nb0  == حجم(عائم));

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    إذا (ith == 0) {
        ggml_vec_سيٽ_f32(ne0*ne1*ne2*ne3, (عائم *)dst->معلومات, 0);
    }
    ggml_barrier(params->threadpool);

    ثابت int64_t nr = ne1*ne2*ne3;
    ثابت int64_t dr = (nr + nth - 1)/nth;
    ثابت int64_t ir0 = dr*ith;
    ثابت int64_t ir1 = MIN(ir0 + dr, nr);

    عائم * wdata = (عائم *) params->wdata + (ne0 + CACHE_سٽ_سائيز_F32) * ith;

    لكل (int64_t ir = ir0; ir < ir1; ++ir) {
        ثابت int64_t i3 = ir/(ne2*ne1);
        ثابت int64_t i2 = (ir - i3*ne2*ne1)/ne1;
        ثابت int64_t i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ثابت int64_t i02 = i2;
        ثابت int64_t i03 = i3;

        ثابت int64_t i12 = i2;
        ثابت int64_t i13 = i3;

        عائم * d = (عائم *) ((حرف *) dst->معلومات + (i1*nb1 + i2*nb2 + i3*nb3));

        لكل (int64_t i01 = 0; i01 < ne01; ++i01) {
            ثابت int64_t i11 = i01;
            ggml_fp16_t * s0 = (ggml_fp16_t *) ((حرف *) src0->معلومات + (i01*nb01 + i02*nb02 + i03*nb03));
            عائم * s1 = (عائم *) ((حرف *) src1->معلومات + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
            ggml_fp16_منزل_fp32_صف(s0, wdata, ne0);
            ggml_vec_mad_f32(ne0, d, wdata, *s1);
        }
    }
}

فراغ ggml_compute_forward_ٻاهريون_prod(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_Q1_0:
        حالة GGML_TYPE_Q2_0:
        حالة GGML_TYPE_Q4_0:
        حالة GGML_TYPE_Q4_1:
        حالة GGML_TYPE_Q5_0:
        حالة GGML_TYPE_Q5_1:
        حالة GGML_TYPE_Q8_0:
        حالة GGML_TYPE_MXFP4:
        حالة GGML_TYPE_NVFP4:
        حالة GGML_TYPE_Q2_K:
        حالة GGML_TYPE_Q3_K:
        حالة GGML_TYPE_Q4_K:
        حالة GGML_TYPE_Q5_K:
        حالة GGML_TYPE_Q6_K:
        حالة GGML_TYPE_TQ1_0:
        حالة GGML_TYPE_TQ2_0:
        حالة GGML_TYPE_IQ2_XXS:
        حالة GGML_TYPE_IQ2_XS:
        حالة GGML_TYPE_IQ3_XXS:
        حالة GGML_TYPE_IQ1_S:
        حالة GGML_TYPE_IQ1_M:
        حالة GGML_TYPE_IQ4_NL:
        حالة GGML_TYPE_IQ4_XS:
        حالة GGML_TYPE_IQ3_S:
        حالة GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_ٻاهريون_prod_q_f32(params, dst);
            } اكسر;
        حالة GGML_TYPE_F16:
            {
                ggml_compute_forward_ٻاهريون_prod_f16_f32(params, dst);
            } اكسر;
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_ٻاهريون_prod_f32(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_scale

ساكن فراغ ggml_compute_forward_scale_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    عائم s; // scale factor
    عائم b; // bias

    يادگيري_نقل(&s, (عائم *) dst->op_params + 0, حجم(عائم));
    يادگيري_نقل(&b, (عائم *) dst->op_params + 1, حجم(عائم));

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح nc = src0->ne[0];
    ثابت صحيح nr = ggml_nrows(src0);

    // rows per thread
    ثابت صحيح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحيح ir0 = dr*ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    ثابت size_t nb01 = src0->nb[1];

    ثابت size_t nb1 = dst->nb[1];

    إذا (b == 0.0f) {
        لكل (صحيح i1 = ir0; i1 < ir1; i1++) {
            إذا (dst->معلومات != src0->معلومات) {
                // src0 is same shape as dst => same indices
                // TODO: add x parameter to ggml_vec_scale_f32 and remove this memcpy
                يادگيري_نقل((حرف *)dst->معلومات + i1*nb1, (حرف *)src0->معلومات + i1*nb01, nc * حجم(عائم));
            }
            ggml_vec_scale_f32(nc, (عائم *) ((حرف *) dst->معلومات + i1*nb1), s);
        }
    } وإلا {
        لكل (صحيح i1 = ir0; i1 < ir1; i1++) {
            ggml_vec_mad1_f32(nc,
                (عائم *) ((حرف *) dst->معلومات  + i1*nb1),
                (عائم *) ((حرف *) src0->معلومات + i1*nb1),
                s, b);
        }
    }
}

ساكن فراغ ggml_compute_forward_scale_bf16(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    عائم s; // scale factor
    عائم b; // bias

    يادگيري_نقل(&s, (عائم *) dst->op_params + 0, حجم(عائم));
    يادگيري_نقل(&b, (عائم *) dst->op_params + 1, حجم(عائم));

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح nc = src0->ne[0];
    ثابت صحيح nr = ggml_nrows(src0);

    // rows per thread
    ثابت صحيح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحيح ir0 = dr*ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    لكل (صحيح i1 = ir0; i1 < ir1; i1++) {
        ثابت ggml_bf16_t * x = (ثابت ggml_bf16_t *) ((ثابت حرف *) src0->معلومات + i1*src0->nb[1]);
        ggml_bf16_t       * y = (ggml_bf16_t *)       ((حرف *)       dst->معلومات  + i1*dst->nb[1]);
        لكل (صحيح k = 0; k < nc; k++) {
            y[k] = GGML_FP32_منزل_BF16(s*GGML_BF16_منزل_FP32(x[k]) + b);
        }
    }
}

فراغ ggml_compute_forward_scale(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_scale_f32(params, dst);
            } اكسر;
        حالة GGML_TYPE_BF16:
            {
                ggml_compute_forward_scale_bf16(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_set

ساكن فراغ ggml_compute_forward_سيٽ_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));

    // view src0 and dst with these strides and data offset inbytes during set
    // nb0 is implicitly element_size because src0 and dst are contiguous
    size_t nb1     = ((int32_t *) dst->op_params)[0];
    size_t nb2     = ((int32_t *) dst->op_params)[1];
    size_t nb3     = ((int32_t *) dst->op_params)[2];
    size_t وٿي  = ((int32_t *) dst->op_params)[3];
    منطقي   inplace = (منطقي) ((int32_t *) dst->op_params)[4];

    إذا (!inplace) {
        إذا (params->ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            يادگيري_نقل(
                ((حرف *)  dst->معلومات),
                ((حرف *) src0->معلومات),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح nr = ggml_nrows(src1);
    ثابت صحيح nc = src1->ne[0];

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne)
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb)

    // src0 and dst as viewed during set
    ثابت size_t nb0 = ggml_element_سائيز(src0);

    ثابت صحيح im0 = (ne10 == 0 ? 0 : ne10-1);
    ثابت صحيح im1 = (ne11 == 0 ? 0 : ne11-1);
    ثابت صحيح im2 = (ne12 == 0 ? 0 : ne12-1);
    ثابت صحيح im3 = (ne13 == 0 ? 0 : ne13-1);

    GGML_ASSERT(وٿي + im0*nb0  + im1*nb1  + im2*nb2  + im3*nb3  <= ggml_nbytes(dst));

    GGML_ASSERT(nb10 == حجم(عائم));

    // rows per thread
    ثابت صحيح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحيح ir0 = dr*ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    لكل (صحيح ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are viewed with shape of src1 and offset
        // => same indices
        ثابت صحيح i3 = ir/(ne12*ne11);
        ثابت صحيح i2 = (ir - i3*ne12*ne11)/ne11;
        ثابت صحيح i1 = (ir - i3*ne12*ne11 - i2*ne11);

        ggml_vec_cpy_f32(nc,
                (عائم *) ((حرف *)  dst->معلومات + i3*nb3  + i2*nb2  + i1*nb1  + وٿي),
                (عائم *) ((حرف *) src1->معلومات + i3*nb13 + i2*nb12 + i1*nb11));
    }
}

ساكن فراغ ggml_compute_forward_سيٽ_i32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));

    // view src0 and dst with these strides and data offset inbytes during set
    // nb0 is implicitly element_size because src0 and dst are contiguous
    size_t nb1     = ((int32_t *) dst->op_params)[0];
    size_t nb2     = ((int32_t *) dst->op_params)[1];
    size_t nb3     = ((int32_t *) dst->op_params)[2];
    size_t وٿي  = ((int32_t *) dst->op_params)[3];
    منطقي   inplace = (منطقي) ((int32_t *) dst->op_params)[4];

    إذا (!inplace) {
        إذا (params->ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            يادگيري_نقل(
                ((حرف *)  dst->معلومات),
                ((حرف *) src0->معلومات),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح nr = ggml_nrows(src1);
    ثابت صحيح nc = src1->ne[0];

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne)
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb)

    // src0 and dst as viewed during set
    ثابت size_t nb0 = ggml_element_سائيز(src0);

    ثابت صحيح im0 = (ne10 == 0 ? 0 : ne10-1);
    ثابت صحيح im1 = (ne11 == 0 ? 0 : ne11-1);
    ثابت صحيح im2 = (ne12 == 0 ? 0 : ne12-1);
    ثابت صحيح im3 = (ne13 == 0 ? 0 : ne13-1);

    GGML_ASSERT(وٿي + im0*nb0  + im1*nb1  + im2*nb2  + im3*nb3  <= ggml_nbytes(dst));

    GGML_ASSERT(nb10 == حجم(int32_t));

    // rows per thread
    ثابت صحيح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحيح ir0 = dr*ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    لكل (صحيح ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are viewed with shape of src1 and offset
        // => same indices
        ثابت صحيح i3 = ir/(ne12*ne11);
        ثابت صحيح i2 = (ir - i3*ne12*ne11)/ne11;
        ثابت صحيح i1 = (ir - i3*ne12*ne11 - i2*ne11);

        ggml_vec_cpy_i32(nc,
                (int32_t *) ((حرف *)  dst->معلومات + i3*nb3  + i2*nb2  + i1*nb1  + وٿي),
                (int32_t *) ((حرف *) src1->معلومات + i3*nb13 + i2*nb12 + i1*nb11));
    }
}

فراغ ggml_compute_forward_سيٽ(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_سيٽ_f32(params, dst);
            } اكسر;
        حالة GGML_TYPE_I32:
            {
                ggml_compute_forward_سيٽ_i32(params, dst);
            } اكسر;
        حالة GGML_TYPE_F16:
        حالة GGML_TYPE_BF16:
        حالة GGML_TYPE_Q1_0:
        حالة GGML_TYPE_Q2_0:
        حالة GGML_TYPE_Q4_0:
        حالة GGML_TYPE_Q4_1:
        حالة GGML_TYPE_Q5_0:
        حالة GGML_TYPE_Q5_1:
        حالة GGML_TYPE_Q8_0:
        حالة GGML_TYPE_Q8_1:
        حالة GGML_TYPE_MXFP4:
        حالة GGML_TYPE_NVFP4:
        حالة GGML_TYPE_Q2_K:
        حالة GGML_TYPE_Q3_K:
        حالة GGML_TYPE_Q4_K:
        حالة GGML_TYPE_Q5_K:
        حالة GGML_TYPE_Q6_K:
        حالة GGML_TYPE_TQ1_0:
        حالة GGML_TYPE_TQ2_0:
        حالة GGML_TYPE_IQ2_XXS:
        حالة GGML_TYPE_IQ2_XS:
        حالة GGML_TYPE_IQ3_XXS:
        حالة GGML_TYPE_IQ1_S:
        حالة GGML_TYPE_IQ1_M:
        حالة GGML_TYPE_IQ4_NL:
        حالة GGML_TYPE_IQ4_XS:
        حالة GGML_TYPE_IQ3_S:
        حالة GGML_TYPE_IQ2_S:
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_cpy

فراغ ggml_compute_forward_cpy(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_dup(params, dst);
}

// ggml_compute_forward_cont

فراغ ggml_compute_forward_cont(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_dup(params, dst);
}

// ggml_compute_forward_get_rows

ساكن فراغ ggml_compute_forward_get_صفون_q(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ٻٽو_OP_LOCALS

    ثابت int64_t nc = ne00;
    ثابت int64_t nr = ggml_nelements(src1);

    ثابت ggml_type type = src0->type;
    ggml_منزل_float_t ثابت dequantize_صف_q = ggml_get_type_traits(type)->منزل_float;

    أكد(ne0  == nc);
    أكد(ne02 == ne11);
    أكد(nb00 == ggml_type_سائيز(type));
    أكد(ggml_nrows(dst) == nr);

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    // rows per thread
    ثابت صحيح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحيح ir0 = dr*ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    لكل (int64_t i = ir0; i < ir1; ++i) {
        ثابت int64_t i12 = i/(ne11*ne10);
        ثابت int64_t i11 = (i - i12*ne11*ne10)/ne10;
        ثابت int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        ثابت int64_t i01 = *(int32_t *) ((حرف *) src1->معلومات + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        dequantize_صف_q(
                (ثابت فراغ *) ((حرف *) src0->معلومات + i01*nb01 + i11*nb02 + i12*nb03),
                     (عائم *) ((حرف *)  dst->معلومات + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

ساكن فراغ ggml_compute_forward_get_صفون_f16(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ٻٽو_OP_LOCALS

    ثابت int64_t nc = ne00;
    ثابت int64_t nr = ggml_nelements(src1);

    أكد(ne0  == nc);
    أكد(ne02 == ne11);
    أكد(nb00 == حجم(ggml_fp16_t));
    أكد(ggml_nrows(dst) == nr);

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    // rows per thread
    ثابت صحيح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحيح ir0 = dr*ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    لكل (int64_t i = ir0; i < ir1; ++i) {
        ثابت int64_t i12 = i/(ne11*ne10);
        ثابت int64_t i11 = (i - i12*ne11*ne10)/ne10;
        ثابت int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        ثابت int64_t i01 = *(int32_t *) ((حرف *) src1->معلومات + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        ggml_cpu_fp16_منزل_fp32(
            (ثابت ggml_fp16_t*) ((حرف *) src0->معلومات + i01*nb01 + i11*nb02 + i12*nb03),
                       (عائم *) ((حرف *)  dst->معلومات + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

ساكن فراغ ggml_compute_forward_get_صفون_bf16(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ٻٽو_OP_LOCALS

    ثابت int64_t nc = ne00;
    ثابت int64_t nr = ggml_nelements(src1);

    أكد(ne0  == nc);
    أكد(ne02 == ne11);
    أكد(nb00 == حجم(ggml_bf16_t));
    أكد(ggml_nrows(dst) == nr);

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    // rows per thread
    ثابت صحيح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحيح ir0 = dr*ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    لكل (int64_t i = ir0; i < ir1; ++i) {
        ثابت int64_t i12 = i/(ne11*ne10);
        ثابت int64_t i11 = (i - i12*ne11*ne10)/ne10;
        ثابت int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        ثابت int64_t i01 = *(int32_t *) ((حرف *) src1->معلومات + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        ggml_cpu_bf16_منزل_fp32(
            (ثابت ggml_bf16_t *) ((حرف *) src0->معلومات + i01*nb01 + i11*nb02 + i12*nb03),
                        (عائم *) ((حرف *)  dst->معلومات + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

ساكن فراغ ggml_compute_forward_get_صفون_f32(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ٻٽو_OP_LOCALS

    ثابت int64_t nc = ne00;
    ثابت int64_t nr = ggml_nelements(src1);

    أكد(ne0  == nc);
    أكد(ne02 == ne11);
    أكد(nb00 == حجم(عائم));
    أكد(ggml_nrows(dst) == nr);

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    // rows per thread
    ثابت صحيح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحيح ir0 = dr*ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    لكل (int64_t i = ir0; i < ir1; ++i) {
        ثابت int64_t i12 = i/(ne11*ne10);
        ثابت int64_t i11 = (i - i12*ne11*ne10)/ne10;
        ثابت int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        ثابت int64_t i01 = *(int32_t *) ((حرف *) src1->معلومات + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        ggml_vec_cpy_f32(nc,
                (عائم *) ((حرف *)  dst->معلومات + i10*nb1  + i11*nb2  + i12*nb3),
                (عائم *) ((حرف *) src0->معلومات + i01*nb01 + i11*nb02 + i12*nb03));
    }
}

فراغ ggml_compute_forward_get_صفون(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_Q1_0:
        حالة GGML_TYPE_Q2_0:
        حالة GGML_TYPE_Q4_0:
        حالة GGML_TYPE_Q4_1:
        حالة GGML_TYPE_Q5_0:
        حالة GGML_TYPE_Q5_1:
        حالة GGML_TYPE_Q8_0:
        حالة GGML_TYPE_Q8_1:
        حالة GGML_TYPE_MXFP4:
        حالة GGML_TYPE_NVFP4:
        حالة GGML_TYPE_Q2_K:
        حالة GGML_TYPE_Q3_K:
        حالة GGML_TYPE_Q4_K:
        حالة GGML_TYPE_Q5_K:
        حالة GGML_TYPE_Q6_K:
        حالة GGML_TYPE_TQ1_0:
        حالة GGML_TYPE_TQ2_0:
        حالة GGML_TYPE_IQ2_XXS:
        حالة GGML_TYPE_IQ2_XS:
        حالة GGML_TYPE_IQ3_XXS:
        حالة GGML_TYPE_IQ1_S:
        حالة GGML_TYPE_IQ1_M:
        حالة GGML_TYPE_IQ4_NL:
        حالة GGML_TYPE_IQ4_XS:
        حالة GGML_TYPE_IQ3_S:
        حالة GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_get_صفون_q(params, dst);
            } اكسر;
        حالة GGML_TYPE_F16:
            {
                ggml_compute_forward_get_صفون_f16(params, dst);
            } اكسر;
        حالة GGML_TYPE_BF16:
            {
                ggml_compute_forward_get_صفون_bf16(params, dst);
            } اكسر;
        حالة GGML_TYPE_F32:
        حالة GGML_TYPE_I32:
            {
                ggml_compute_forward_get_صفون_f32(params, dst);
            } اكسر;
        افتراضي:
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

سانچو<قسم_نالو src_t, قسم_نالو idx_t>
ساكن فراغ ggml_compute_forward_سيٽ_صفون_impl(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ٻٽو_OP_LOCALS

    ثابت int64_t nc = ne00;
    ثابت int64_t nr = ne01;

    أكد(ne0  == nc);
    أكد(ne2  == ne02);
    أكد(ne3  == ne03);
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    أكد(ne02 % ne11 == 0);
    أكد(ne03 % ne12 == 0);

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    // rows per thread
    ثابت int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت int64_t ir0 = dr*ith;
    ثابت int64_t ir1 = std::min(ir0 + dr, nr);

    ثابت size_t rs = ggml_صف_سائيز(src0->type, nc);

    ggml_from_float_t ثابت from_float = ggml_get_type_traits_cpu(dst->type)->from_float;

    لكل (int64_t i03 = 0; i03 < ne03; ++i03) {
        لكل (int64_t i02 = 0; i02 < ne02; ++i02) {
            لكل (int64_t i = ir0; i < ir1; ++i) {
                ثابت int64_t i12 = i03%ne12;
                ثابت int64_t i11 = i02%ne11;
                ثابت int64_t i10 = i;

                ثابت int64_t i1 = *(idx_t *) ((حرف *) src1->معلومات + i10*nb10 + i11*nb11 + i12*nb12);

                GGML_ASSERT(i1 >= 0 && i1 < ne1);

                إذا constexpr (std::is_same_v<src_t, عائم>) {
                    from_float(
                            (ثابت عائم *) ((حرف *) src0->معلومات +  i*nb01 + i02*nb02 + i03*nb03),
                                            ((حرف *)  dst->معلومات + i1*nb1  + i02*nb2  + i03*nb3), nc);
                } وإلا إذا constexpr (std::is_same_v<src_t, ggml_fp16_t>) {
                    إذا (dst->type == GGML_TYPE_F16) {
                        يادگيري_نقل(
                                            ((حرف *)  dst->معلومات + i1*nb1  + i02*nb2  + i03*nb3),
                                            ((حرف *) src0->معلومات +  i*nb01 + i02*nb02 + i03*nb03),
                                            rs);
                    } وإلا {
                        عائم * wdata = (عائم *) params->wdata + (nc + CACHE_سٽ_سائيز_F32) * ith;
                        ggml_fp16_منزل_fp32_صف(
                                (ثابت ggml_fp16_t *) ((حرف *) src0->معلومات + i*nb01 + i02*nb02 + i03*nb03),
                                wdata, nc);
                        from_float(wdata,
                                ((حرف *) dst->معلومات + i1*nb1 + i02*nb2 + i03*nb3), nc);
                    }
                } وإلا {
                    GGML_ABORT("src0->type = %d (%s) not supported", src0->type, ggml_type_name(src0->type));
                }
            }
        }
    }
}

فراغ ggml_compute_forward_سيٽ_صفون(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                إذا (src1->type == GGML_TYPE_I64) {
                    ggml_compute_forward_سيٽ_صفون_impl<عائم, int64_t>(params, dst);
                } وإلا إذا (src1->type == GGML_TYPE_I32) {
                    ggml_compute_forward_سيٽ_صفون_impl<عائم, int32_t>(params, dst);
                } وإلا {
                    GGML_ABORT("src1->type = %d (%s) not supported", src1->type, ggml_type_name(src1->type));
                }
            } اكسر;
        حالة GGML_TYPE_F16:
            {
                إذا (src1->type == GGML_TYPE_I64) {
                    ggml_compute_forward_سيٽ_صفون_impl<ggml_fp16_t, int64_t>(params, dst);
                } وإلا إذا (src1->type == GGML_TYPE_I32) {
                    ggml_compute_forward_سيٽ_صفون_impl<ggml_fp16_t, int32_t>(params, dst);
                } وإلا {
                    GGML_ABORT("src1->type = %d (%s) not supported", src1->type, ggml_type_name(src1->type));
                }
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("src0->type = %d (%s) not supported", src0->type, ggml_type_name(src0->type));
            }
    }
}

// ggml_compute_forward_get_rows_back

ساكن فراغ ggml_compute_forward_get_صفون_پٺ_f32_f16(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    إذا (params->ith != 0) {
        أرجع;
    }

    GGML_ASSERT(ggml_is_contiguous(dst));

    // ggml_compute_forward_dup_same_cont(params, opt0, dst);

    memset(dst->معلومات, 0, ggml_nbytes(dst));

    ثابت صحيح nc = src0->ne[0];
    ثابت صحيح nr = ggml_nelements(src1);

    GGML_ASSERT( dst->ne[0] == nc);
    GGML_ASSERT(src0->nb[0] == حجم(ggml_fp16_t));

    لكل (صحيح i = 0; i < nr; ++i) {
        ثابت صحيح r = ((int32_t *) src1->معلومات)[i];

        GGML_ASSERT(r >= 0 && r < dst->ne[1]);

        لكل (صحيح j = 0; j < nc; ++j) {
            ggml_fp16_t v = ((ggml_fp16_t *) ((حرف *) src0->معلومات + i*src0->nb[1]))[j];
            ((عائم *) ((حرف *) dst->معلومات + r*dst->nb[1]))[j] += GGML_CPU_FP16_منزل_FP32(v);
        }
    }
}

ساكن فراغ ggml_compute_forward_get_صفون_پٺ_f32(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    إذا (params->ith != 0) {
        أرجع;
    }

    GGML_ASSERT(ggml_is_contiguous(dst));

    // ggml_compute_forward_dup_same_cont(params, opt0, dst);

    memset(dst->معلومات, 0, ggml_nbytes(dst));

    ثابت صحيح nc = src0->ne[0];
    ثابت صحيح nr = ggml_nelements(src1);

    GGML_ASSERT( dst->ne[0] == nc);
    GGML_ASSERT(src0->nb[0] == حجم(عائم));

    لكل (صحيح i = 0; i < nr; ++i) {
        ثابت صحيح r = ((int32_t *) src1->معلومات)[i];

        GGML_ASSERT(r >= 0 && r < dst->ne[1]);

        ggml_vec_جوڙيو_f32(nc,
                (عائم *) ((حرف *)  dst->معلومات + r*dst->nb[1]),
                (عائم *) ((حرف *)  dst->معلومات + r*dst->nb[1]),
                (عائم *) ((حرف *) src0->معلومات + i*src0->nb[1]));
    }
}

فراغ ggml_compute_forward_get_صفون_پٺ(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F16:
            {
                ggml_compute_forward_get_صفون_پٺ_f32_f16(params, dst);
            } اكسر;
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_get_صفون_پٺ_f32(params, dst);
            } اكسر;
        افتراضي:
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

ساكن فراغ ggml_compute_forward_diag_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    إذا (params->ith != 0) {
        أرجع;
    }

    // TODO: handle transposed/permuted matrices

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(ne00 == ne0);
    GGML_ASSERT(ne00 == ne1);
    GGML_ASSERT(ne01 == 1);
    GGML_ASSERT(ne02 == ne2);
    GGML_ASSERT(ne03 == ne3);

    GGML_ASSERT(nb00 == حجم(عائم));
    GGML_ASSERT(nb0  == حجم(عائم));

    لكل (صحيح i3 = 0; i3 < ne3; i3++) {
        لكل (صحيح i2 = 0; i2 < ne2; i2++) {
            لكل (صحيح i1 = 0; i1 < ne1; i1++) {
                عائم * d = (عائم *)((حرف *)  dst->معلومات + i3*nb3  + i2*nb2 + i1*nb1);
                عائم * s = (عائم *)((حرف *) src0->معلومات + i3*nb03 + i2*nb02);
                لكل (صحيح i0 = 0; i0 < i1; i0++) {
                    d[i0] = 0;
                }
                d[i1] = s[i1];
                لكل (صحيح i0 = i1+1; i0 < ne0; i0++) {
                    d[i0] = 0;
                }
            }
        }
    }
}

فراغ ggml_compute_forward_diag(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_diag_f32(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_diag_mask_inf

ساكن فراغ ggml_compute_forward_diag_نقاب_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst,
        ثابت عائم value) {

    ثابت ggml_tensor * src0 = dst->src[0];

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح  n_past  = ((int32_t *) dst->op_params)[0];
    ثابت منطقي inplace = src0->معلومات == dst->معلومات;

    GGML_ASSERT(n_past >= 0);

    إذا (!inplace) {
        إذا (ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
            GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));
            يادگيري_نقل(
                ((حرف *)  dst->معلومات),
                ((حرف *) src0->معلومات),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    // TODO: handle transposed/permuted matrices

    ثابت صحيح n  = ggml_nrows(src0);
    ثابت صحيح nc = src0->ne[0];
    ثابت صحيح nr = src0->ne[1];
    ثابت صحيح nz = n/nr;

    GGML_ASSERT( dst->nb[0] == حجم(عائم));
    GGML_ASSERT(src0->nb[0] == حجم(عائم));

    لكل (صحيح k = 0; k < nz; k++) {
        لكل (صحيح j = ith; j < nr; j += nth) {
            لكل (صحيح i = n_past; i < nc; i++) {
                إذا (i > n_past + j) {
                    *(عائم *)((حرف *) dst->معلومات + k*dst->nb[2] + j*dst->nb[1] + i*dst->nb[0]) = value;
                }
            }
        }
    }
}

فراغ ggml_compute_forward_diag_نقاب_inf(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_diag_نقاب_f32(params, dst, -INFINITY);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

فراغ ggml_compute_forward_diag_نقاب_zero(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_diag_نقاب_f32(params, dst, 0);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_soft_max

ساكن فراغ ggml_compute_forward_soft_max_f32(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    ثابت ggml_tensor * src2 = dst->src[2];

    أكد(ggml_is_contiguous(dst));
    أكد(ggml_are_same_shape(src0, dst));

    عائم scale    = 1.0f;
    عائم max_bias = 0.0f;

    يادگيري_نقل(&scale,    (عائم *) dst->op_params + 0, حجم(عائم));
    يادگيري_نقل(&max_bias, (عائم *) dst->op_params + 1, حجم(عائم));

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    ثابت int64_t nb11 = src1 ? src1->nb[1] : 1;
    ثابت int64_t nb12 = src1 ? src1->nb[2] : 1;
    ثابت int64_t nb13 = src1 ? src1->nb[3] : 1;

    ثابت int64_t ne12 = src1 ? src1->ne[2] : 1;
    ثابت int64_t ne13 = src1 ? src1->ne[3] : 1;

    // TODO: is this supposed to be ceil instead of floor?
    //       https://huggingface.co/mosaicml/mpt-7b/blob/main/attention.py#L370
    ثابت uint32_t n_سر      = ne02;
    ثابت uint32_t n_سر_log2 = 1u << (uint32_t) floor(log2(n_سر));

    ثابت عائم m0 = powf(2.0f, -(max_bias       ) / n_سر_log2);
    ثابت عائم m1 = powf(2.0f, -(max_bias / 2.0f) / n_سر_log2);

    عائم * wp = (عائم *) params->wdata + (ne00 + CACHE_سٽ_سائيز_F32) * ith;

    ثابت منطقي use_f16 = (src1 && src1->type == GGML_TYPE_F16);

    // sinks
    ثابت عائم * sk = src2 ? (عائم *)((حرف *) src2->معلومات) : خالي_اشارو;

    لكل (int64_t i03 = 0; i03 < ne03; i03++) {
        لكل (int64_t i02 = 0; i02 < ne02; i02++) {
            لكل (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                ثابت int64_t i11 = i01;
                ثابت int64_t i12 = i02%ne12;
                ثابت int64_t i13 = i03%ne13;

                // ALiBi
                ثابت uint32_t h = i02; //سر
                ثابت عائم لاهي = (max_bias > 0.0f) ? h < n_سر_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_سر_log2) + 1) : 1.0f;

                عائم * sp = (عائم *)((حرف *) src0->معلومات + i01*nb01 + i02*nb02 + i03*nb03);
                عائم * dp = (عائم *)((حرف *)  dst->معلومات + i01*nb1  + i02*nb2  + i03*nb3);

                // broadcast the mask across rows
                ggml_fp16_t * mp_f16 = src1 ? (ggml_fp16_t *)((حرف *) src1->معلومات + i11*nb11 + i12*nb12 + i13*nb13) : عدم;
                عائم       * mp_f32 = src1 ? (عائم       *)((حرف *) src1->معلومات + i11*nb11 + i12*nb12 + i13*nb13) : عدم;

                ggml_vec_cpy_f32  (ne00, wp, sp);
                ggml_vec_scale_f32(ne00, wp, scale);
                إذا (mp_f32) {
                    إذا (use_f16) {
                        لكل (صحيح i = 0; i < ne00; ++i) {
                            wp[i] += لاهي*GGML_CPU_FP16_منزل_FP32(mp_f16[i]);
                        }
                    } وإلا {
                        لكل (صحيح i = 0; i < ne00; ++i) {
                            wp[i] += لاهي*mp_f32[i];
                        }
                    }
                }

#جيڪڏهن_وصف_نه_ٿيل NDEBUG
                لكل (صحيح i = 0; i < ne00; ++i) {
                    //printf("p[%d] = %f\n", i, p[i]);
                    أكد(!isnan(wp[i]));
                }
#جيڪڏهن_پورو // NDEBUG

                عائم max = -INFINITY;
                ggml_vec_max_f32(ne00, &max, wp);

                // if we have sinks, make a correction as if they were included in the softmax
                إذا (sk) {
                    max = MAX(max, sk[i02]);
                }

                ggml_float جوڙ = ggml_vec_soft_max_f32(ne00, dp, wp, max);
                أكد(جوڙ > 0.0);

                إذا (sk) {
                    جوڙ += (ggml_float) expf(sk[i02] - max);
                }

                جوڙ = 1.0/جوڙ;
                ggml_vec_scale_f32(ne00, dp, جوڙ);

#جيڪڏهن_وصف_نه_ٿيل NDEBUG
                لكل (صحيح i = 0; i < ne00; ++i) {
                    أكد(!isnan(dp[i]));
                    أكد(!isinf(dp[i]));
                }
#جيڪڏهن_پورو // NDEBUG
            }
        }
    }
}

فراغ ggml_compute_forward_soft_max(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_soft_max_f32(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}


// ggml_compute_forward_soft_max_ext_back

ساكن فراغ ggml_compute_forward_soft_max_ext_پٺ_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_are_same_shape(src1, dst));

    عائم scale    = 1.0f;
    عائم max_bias = 0.0f;

    يادگيري_نقل(&scale,    (ثابت عائم *) dst->op_params + 0, حجم(عائم));
    يادگيري_نقل(&max_bias, (ثابت عائم *) dst->op_params + 1, حجم(عائم));

    GGML_ASSERT(max_bias == 0.0f);

    // TODO: handle transposed/permuted matrices

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح nc = src0->ne[0];
    ثابت صحيح nr = ggml_nrows(src0);

    // rows per thread
    ثابت صحيح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحيح ir0 = dr*ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    لكل (صحيح i1 = ir0; i1 < ir1; i1++) {
        عائم *dy = (عائم *)((حرف *) src0->معلومات + i1*src0->nb[1]);
        عائم *y  = (عائم *)((حرف *) src1->معلومات + i1*src1->nb[1]);
        عائم *dx = (عائم *)((حرف *) dst->معلومات  + i1*dst->nb[1]);

#جيڪڏهن_وصف_نه_ٿيل NDEBUG
        لكل (صحيح i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            أكد(!isnan(dy[i]));
            أكد(!isnan(y[i]));
        }
#جيڪڏهن_پورو // NDEBUG
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
        عائم dot_y_dy = 0;
        ggml_vec_dot_f32(nc, &dot_y_dy, 0, y, 0, dy, 0, 1);
        لكل (صحيح i = 0; i < nc; i++) {
            dx[i] = scale * (dy[i] - dot_y_dy) * y[i];
        }

#جيڪڏهن_وصف_نه_ٿيل NDEBUG
        لكل (صحيح i = 0; i < nc; ++i) {
            أكد(!isnan(dx[i]));
            أكد(!isinf(dx[i]));
        }
#جيڪڏهن_پورو // NDEBUG
    }
}

فراغ ggml_compute_forward_soft_max_ext_پٺ(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_soft_max_ext_پٺ_f32(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_clamp

ساكن فراغ ggml_compute_forward_clamp_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    عائم min;
    عائم max;
    يادگيري_نقل(&min, (عائم *) dst->op_params + 0, حجم(عائم));
    يادگيري_نقل(&max, (عائم *) dst->op_params + 1, حجم(عائم));

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح n  = ggml_nrows(src0);
    ثابت صحيح nc = src0->ne[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == حجم(عائم));
    GGML_ASSERT(nb00 == حجم(عائم));

    لكل (صحيح j = ith; j < n; j += nth) {
        ثابت int64_t i1 = j % ne01;
        ثابت int64_t i2 = (j / ne01) % ne02;
        ثابت int64_t i3 = j / (ne01*ne02);

        عائم * dst_ptr  = (عائم *) ((حرف *)  dst->معلومات + i1*nb1  + i2*nb2  + i3*nb3);
        عائم * src0_ptr = (عائم *) ((حرف *) src0->معلومات + i1*nb01 + i2*nb02 + i3*nb03);

        لكل (صحيح i = 0; i < nc; i++) {
            dst_ptr[i] = MAX(MIN(src0_ptr[i], max), min);
        }
    }
}

ساكن فراغ ggml_compute_forward_clamp_f16(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    عائم min;
    عائم max;
    يادگيري_نقل(&min, (عائم *) dst->op_params + 0, حجم(عائم));
    يادگيري_نقل(&max, (عائم *) dst->op_params + 1, حجم(عائم));

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح n  = ggml_nrows(src0);
    ثابت صحيح nc = src0->ne[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == حجم(ggml_fp16_t));
    GGML_ASSERT(nb00 == حجم(ggml_fp16_t));

    لكل (صحيح j = ith; j < n; j += nth) {
        ثابت int64_t i1 = j % ne01;
        ثابت int64_t i2 = (j / ne01) % ne02;
        ثابت int64_t i3 = j / (ne01*ne02);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((حرف *)  dst->معلومات + i1*nb1  + i2*nb2  + i3*nb3);
        ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((حرف *) src0->معلومات + i1*nb01 + i2*nb02 + i3*nb03);

        لكل (صحيح i = 0; i < nc; i++) {
            عائم v = GGML_CPU_FP16_منزل_FP32(src0_ptr[i]);
            dst_ptr[i] = GGML_CPU_FP32_منزل_FP16(MAX(MIN(v, max), min));
        }
    }
}

فراغ ggml_compute_forward_clamp(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_clamp_f32(params, dst);
            } اكسر;
        حالة GGML_TYPE_F16:
            {
                ggml_compute_forward_clamp_f16(params, dst);
            } اكسر;
        حالة GGML_TYPE_BF16:
        حالة GGML_TYPE_Q1_0:
        حالة GGML_TYPE_Q2_0:
        حالة GGML_TYPE_Q4_0:
        حالة GGML_TYPE_Q4_1:
        حالة GGML_TYPE_Q5_0:
        حالة GGML_TYPE_Q5_1:
        حالة GGML_TYPE_Q8_0:
        حالة GGML_TYPE_Q8_1:
        حالة GGML_TYPE_MXFP4:
        حالة GGML_TYPE_NVFP4:
        حالة GGML_TYPE_Q2_K:
        حالة GGML_TYPE_Q3_K:
        حالة GGML_TYPE_Q4_K:
        حالة GGML_TYPE_Q5_K:
        حالة GGML_TYPE_Q6_K:
        حالة GGML_TYPE_TQ1_0:
        حالة GGML_TYPE_TQ2_0:
        حالة GGML_TYPE_IQ2_XXS:
        حالة GGML_TYPE_IQ2_XS:
        حالة GGML_TYPE_IQ3_XXS:
        حالة GGML_TYPE_IQ1_S:
        حالة GGML_TYPE_IQ1_M:
        حالة GGML_TYPE_IQ4_NL:
        حالة GGML_TYPE_IQ4_XS:
        حالة GGML_TYPE_IQ3_S:
        حالة GGML_TYPE_IQ2_S:
        حالة GGML_TYPE_Q8_K:
        حالة GGML_TYPE_I8:
        حالة GGML_TYPE_I16:
        حالة GGML_TYPE_I32:
        حالة GGML_TYPE_I64:
        حالة GGML_TYPE_F64:
        حالة GGML_TYPE_ڳڻپ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_rope

ساكن عائم rope_yarn_ramp(ثابت عائم هيٺيون, ثابت عائم مٿاهون, ثابت صحيح i0) {
    ثابت عائم y = (i0 / 2 - هيٺيون) / MAX(0.001f, مٿاهون - هيٺيون);
    أرجع 1 - MIN(1, MAX(0, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
ساكن فراغ rope_yarn(
    عائم theta_extrap, عائم freq_scale, عائم corr_dims[2], int64_t i0, عائم ext_ضرب_ڪندڙ, عائم mscale,
    عائم * cos_theta, عائم * sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    عائم theta_interp = freq_scale * theta_extrap;
    عائم theta = theta_interp;
    إذا (ext_ضرب_ڪندڙ != 0.0f) {
        عائم ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_ضرب_ڪندڙ;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
}

ساكن فراغ ggml_rope_cache_init(
     عائم theta_بنياد, عائم freq_scale, ثابت عائم * freq_factors, عائم corr_dims[2], int64_t ne0, عائم ext_ضرب_ڪندڙ, عائم mscale,
     عائم * cache, عائم sin_sign, عائم theta_scale) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    عائم theta = theta_بنياد;
    لكل (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        ثابت عائم ff = freq_factors ? freq_factors[i0/2] : 1.0f;
        rope_yarn(
            theta/ff, freq_scale, corr_dims, i0, ext_ضرب_ڪندڙ, mscale, &cache[i0 + 0], &cache[i0 + 1]
        );
        cache[i0 + 1] *= sin_sign;

        theta *= theta_scale;
    }
}

ساكن فراغ ggml_mrope_cache_init(
     عائم theta_بنياد_t, عائم theta_بنياد_h, عائم theta_بنياد_w, عائم theta_بنياد_e, صحيح sections[4], منطقي is_imrope, منطقي indep_sects,
     عائم freq_scale, ثابت عائم * freq_factors, عائم corr_dims[2], int64_t ne0, عائم ext_ضرب_ڪندڙ, عائم mscale,
     عائم * cache, عائم sin_sign, عائم theta_scale) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    عائم theta_t = theta_بنياد_t;
    عائم theta_h = theta_بنياد_h;
    عائم theta_w = theta_بنياد_w;
    عائم theta_e = theta_بنياد_e;  // extra position id for vision encoder
    صحيح sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    صحيح sec_w = sections[1] + sections[0];
    صحيح sec_e = sections[2] + sec_w;
    GGML_ASSERT(sect_dims <= ne0);

    لكل (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        ثابت عائم ff = freq_factors ? freq_factors[i0/2] : 1.0f;

        صحيح sector = (i0 / 2) % sect_dims;
        إذا (indep_sects) {
            // compute theta independently for each dim sections
            // (i.e. reset corresponding theta when `i0` go from one section to another)
            إذا (sector == 0) {
                theta_t = theta_بنياد_t;
            }
            وإلا إذا (sector == sections[0]) {
                theta_h = theta_بنياد_h;;
            }
            وإلا إذا (sector == sec_w) {
                theta_w = theta_بنياد_w;
            }
            وإلا إذا (sector == sec_e) {
                theta_e = theta_بنياد_e;
            }
        }

        عائم theta = theta_t;
        إذا (is_imrope) { // qwen3vl apply interleaved mrope
            إذا (sector % 3 == 1 && sector < 3 * sections[1]) {
                theta = theta_h;
            } وإلا إذا (sector % 3 == 2 && sector < 3 * sections[2]) {
                theta = theta_w;
            } وإلا إذا (sector % 3 == 0 && sector < 3 * sections[0]) {
                theta = theta_t;
            } وإلا {
                theta = theta_e;
            }
        } وإلا {
            إذا (sector >= sections[0] && sector < sec_w) {
                theta = theta_h;
            }
            وإلا إذا (sector >= sec_w && sector < sec_w + sections[2]) {
                theta = theta_w;
            }
            وإلا إذا (sector >= sec_w + sections[2]) {
                theta = theta_e;
            }
        }

        rope_yarn(
            theta/ff, freq_scale, corr_dims, i0, ext_ضرب_ڪندڙ, mscale, &cache[i0 + 0], &cache[i0 + 1]
        );
        cache[i0 + 1] *= sin_sign;

        theta_t *= theta_scale;
        theta_w *= theta_scale;
        theta_h *= theta_scale;
        theta_e *= theta_scale;
    }
}


سانچو<قسم_نالو T>
ساكن فراغ rotate_جوڙا(ثابت int64_t n, ثابت int64_t n_وٿي, ثابت عائم * cache, ثابت T * src_معلومات, T * dst_معلومات, ثابت صحيح scale = 2) {
  لكل (int64_t i0 = 0; i0 < n; i0 += 2) {
    ثابت int64_t ic = i0/scale; // hack for GGML_ROPE_TYPE_NORMAL, where we need ic = i0; for all other cases, ic = i0/2

    ثابت عائم cos_theta = cache[i0 + 0];
    ثابت عائم sin_theta = cache[i0 + 1];

    ثابت T * ثابت src = src_معلومات + ic;
    T * dst             = dst_معلومات + ic;

    ثابت عائم x0 = type_conversion_جدول<T>::منزل_f32(src[0]);
    ثابت عائم x1 = type_conversion_جدول<T>::منزل_f32(src[n_وٿي]);

    dst[0]        = type_conversion_جدول<T>::from_f32(x0*cos_theta - x1*sin_theta);
    dst[n_وٿي] = type_conversion_جدول<T>::from_f32(x0*sin_theta + x1*cos_theta);
  }
}

سانچو<قسم_نالو T> //float or ggml_fp16_t
ساكن فراغ ggml_compute_forward_rope_flt(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst,
        ثابت منطقي forward) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    ثابت ggml_tensor * src2 = dst->src[2];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);

    عائم freq_بنياد, freq_scale, ext_ضرب_ڪندڙ, attn_ضرب_ڪندڙ, beta_fast, beta_slow;
    صحيح sections[4];

    //const int n_past     = ((int32_t *) dst->op_params)[0];
    ثابت صحيح n_dims     = ((int32_t *) dst->op_params)[1];
    ثابت صحيح mode       = ((int32_t *) dst->op_params)[2];
    //const int n_ctx      = ((int32_t *) dst->op_params)[3];
    ثابت صحيح n_ctx_orig = ((int32_t *) dst->op_params)[4];

    يادگيري_نقل(&freq_بنياد,   (int32_t *) dst->op_params +  5, حجم(عائم));
    يادگيري_نقل(&freq_scale,  (int32_t *) dst->op_params +  6, حجم(عائم));
    يادگيري_نقل(&ext_ضرب_ڪندڙ,  (int32_t *) dst->op_params +  7, حجم(عائم));
    يادگيري_نقل(&attn_ضرب_ڪندڙ, (int32_t *) dst->op_params +  8, حجم(عائم));
    يادگيري_نقل(&beta_fast,   (int32_t *) dst->op_params +  9, حجم(عائم));
    يادگيري_نقل(&beta_slow,   (int32_t *) dst->op_params + 10, حجم(عائم));
    يادگيري_نقل(&sections,    (int32_t *) dst->op_params + 11, حجم(صحيح)*4);

    ثابت صحيح n_offs = ((int32_t *) dst->op_params)[15];

    GGML_TENSOR_UNARY_OP_LOCALS

    //printf("ne0: %d, ne1: %d, ne2: %d, ne3: %d\n", ne0, ne1, ne2, ne3);
    //printf("n_past = %d, ne2 = %d\n", n_past, ne2);

    GGML_ASSERT(nb0 == nb00);
    GGML_ASSERT(nb0 == حجم(T));

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح nr = ggml_nrows(dst);

    GGML_ASSERT(n_dims <= ne0);
    GGML_ASSERT(n_dims % 2 == 0);

    GGML_ASSERT(n_offs >= 0);
    GGML_ASSERT(n_offs % 2 == 0);
    GGML_ASSERT(n_offs + n_dims <= ne0);

    // rows per thread
    ثابت صحيح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحيح ir0 = dr*ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    // row index used to determine which thread to use
    صحيح ir = 0;

    ثابت عائم theta_scale = powf(freq_بنياد, -2.0f/n_dims);

    عائم corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_بنياد, beta_fast, beta_slow, corr_dims);

    ثابت منطقي is_imrope = mode == GGML_ROPE_TYPE_IMROPE; // qwen3vl apply interleaved mrope
    ثابت منطقي mrope_استعمال_ٿيل = mode & GGML_ROPE_TYPE_MROPE;  // ggml_rope_multi, note: also true for vision (24 & 8 == true) and for imrope
    ثابت منطقي is_vision = mode == GGML_ROPE_TYPE_VISION;

    إذا (mrope_استعمال_ٿيل) {
        GGML_ASSERT(sections[0] > 0 || sections[1] > 0 || sections[2] > 0);
    }

    إذا (is_vision) {
        GGML_ASSERT(n_dims == ne0/2);
        GGML_ASSERT(n_offs == 0);
    }

    ثابت عائم * freq_factors = عدم;
    إذا (src2 != عدم) {
        GGML_ASSERT(src2->type == GGML_TYPE_F32);
        GGML_ASSERT(src2->ne[0] >= n_dims / 2);
        freq_factors = (ثابت عائم *) src2->معلومات;
    }

    // backward process uses inverse rotation by cos and sin.
    // cos and sin build a rotation matrix, where the inverse is the transpose.
    // this essentially just switches the sign of sin.
    ثابت عائم sin_sign = forward ? 1.0f : -1.0f;

    ثابت int32_t * pos = (ثابت int32_t *) src1->معلومات;

    int64_t آخري_i2 = -1;

    لكل (int64_t i3 = 0; i3 < ne3; i3++) { // batch
        لكل (int64_t i2 = 0; i2 < ne2; i2++) { // seq-len
            لكل (int64_t i1 = 0; i1 < ne1; i1++) { // attn-heads
                إذا (ir++ < ir0) استمر; // skip rows mapped to other threads
                إذا (ir   > ir1) اكسر;

                عائم * cache = (عائم *) params->wdata + (ne0 + CACHE_سٽ_سائيز_F32)*ith;
                إذا (آخري_i2 != i2) {
                    إذا (!mrope_استعمال_ٿيل) {
                        ثابت int64_t p = pos[i2];
                        ggml_rope_cache_init(p, freq_scale, freq_factors, corr_dims, ne0, ext_ضرب_ڪندڙ, attn_ضرب_ڪندڙ, cache, sin_sign, theta_scale);
                    }
                    وإلا {
                        ثابت int64_t p_t = pos[i2];
                        ثابت int64_t p_h = pos[i2 + ne2];
                        ثابت int64_t p_w = pos[i2 + ne2 * 2];
                        ثابت int64_t p_e = pos[i2 + ne2 * 3];
                        ggml_mrope_cache_init(
                            p_t, p_h, p_w, p_e, sections, is_imrope, is_vision,
                            freq_scale, freq_factors, corr_dims, ne0, ext_ضرب_ڪندڙ, attn_ضرب_ڪندڙ, cache, sin_sign, theta_scale);
                    }

                    آخري_i2 = i2;
                }

                T * src = (T *)((حرف *) src0->معلومات + i3*nb03 + i2*nb02 + i1*nb01);
                T * dst_معلومات  = (T *)((حرف *)  dst->معلومات + i3*nb3  + i2*nb2  + i1*nb1);

                بدل (mode) {
                    حالة GGML_ROPE_TYPE_عام:
                        rotate_جوڙا<T>(n_dims, 1, cache, src + n_offs, dst_معلومات + n_offs, 1);
                        اكسر;
                    حالة GGML_ROPE_TYPE_NEOX:
                    حالة GGML_ROPE_TYPE_MROPE:
                    حالة GGML_ROPE_TYPE_IMROPE:
                        rotate_جوڙا<T>(n_dims, n_dims/2, cache, src + n_offs, dst_معلومات + n_offs);
                        اكسر;
                    حالة GGML_ROPE_TYPE_VISION:
                        rotate_جوڙا<T>(ne0, n_dims, cache, src, dst_معلومات);
                        اكسر;
                    افتراضي:
                        GGML_ABORT("rope type not supported");
                }

                إذا (!is_vision) {
                    // fill the remain channels with data from src tensor
                    لكل (int64_t i0 = 0; i0 < ne0; i0 += 2) {
                        إذا (i0 == n_offs) {
                            i0 += n_dims - 2; // skip the rotated channels
                            استمر;
                        }
                        ثابت T * ثابت src = (T *)((حرف *) src0->معلومات + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                        T * dst_معلومات  = (T *)((حرف *)  dst->معلومات + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

                        dst_معلومات[0] = src[0];
                        dst_معلومات[1] = src[1];
                    }
                }
            } //attn-heads
        }
    }
}

فراغ ggml_compute_forward_rope(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F16:
            {
                ggml_compute_forward_rope_flt<ggml_fp16_t>(params, dst, سچ);
            } اكسر;
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_rope_flt<عائم>(params, dst, سچ);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_rope_back

فراغ ggml_compute_forward_rope_پٺ(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F16:
            {
                ggml_compute_forward_rope_flt<ggml_fp16_t>(params, dst, خطأ);
            } اكسر;
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_rope_flt<عائم>(params, dst, خطأ);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_conv_transpose_1d

ساكن فراغ ggml_compute_forward_conv_transpose_1d_f16_f32(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ٻٽو_OP_LOCALS

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح nk = ne00*ne01*ne02;

    GGML_ASSERT(nb00 == حجم(ggml_fp16_t));
    GGML_ASSERT(nb10 == حجم(عائم));

    إذا (ith == 0) {
        memset(params->wdata, 0, params->wsize);

        // permute kernel data (src0) from (K x Cout x Cin) to (Cin x K x Cout)
        {
            ggml_fp16_t * ثابت wdata = (ggml_fp16_t *) params->wdata + 0;

            لكل (int64_t i02 = 0; i02 < ne02; i02++) {
                لكل (int64_t i01 = 0; i01 < ne01; i01++) {
                    ثابت ggml_fp16_t * ثابت src = (ggml_fp16_t *)((حرف *) src0->معلومات + i02*nb02 + i01*nb01);
                    ggml_fp16_t * dst_معلومات = wdata + i01*ne00*ne02;
                    لكل (int64_t i00 = 0; i00 < ne00; i00++) {
                        dst_معلومات[i00*ne02 + i02] = src[i00];
                    }
                }
            }
        }

        // permute source data (src1) from (L x Cin) to (Cin x L)
        {
            ggml_fp16_t * ثابت wdata = (ggml_fp16_t *) params->wdata + nk;
            ggml_fp16_t * dst_معلومات = wdata;

            لكل (int64_t i11 = 0; i11 < ne11; i11++) {
                ثابت عائم * ثابت src = (عائم *)((حرف *) src1->معلومات + i11*nb11);
                لكل (int64_t i10 = 0; i10 < ne10; i10++) {
                    dst_معلومات[i10*ne11 + i11] = GGML_CPU_FP32_منزل_FP16(src[i10]);
                }
            }
        }

        // need to zero dst since we are accumulating into it
        memset(dst->معلومات, 0, ggml_nbytes(dst));
    }
    ggml_barrier(params->threadpool);

    ثابت int32_t s0 = ((ثابت int32_t*)(dst->op_params))[0];

    // total rows in dst
    ثابت صحيح nr = ne1;

    // rows per thread
    ثابت صحيح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحيح ir0 = dr*ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    ggml_fp16_t * ثابت wdata     = (ggml_fp16_t *) params->wdata + 0;
    ggml_fp16_t * ثابت wdata_src = wdata + nk;

    لكل (صحيح i1 = ir0; i1 < ir1; i1++) {
        عائم * dst_معلومات = (عائم *)((حرف *) dst->معلومات + i1*nb1);
        ggml_fp16_t * wdata_kernel = wdata + i1*ne02*ne00;
        لكل (صحيح i10 = 0; i10 < ne10; i10++) {
            ثابت صحيح i1n = i10*ne11;
            لكل (صحيح i00 = 0; i00 < ne00; i00++) {
                عائم v = 0;
                ggml_vec_dot_f16(ne02, &v, 0,
                        (ggml_fp16_t *)    wdata_src + i1n, 0,
                        (ggml_fp16_t *) wdata_kernel + i00*ne02, 0, 1);
                dst_معلومات[i10*s0 + i00] += v;
            }
        }
    }
}

ساكن فراغ ggml_compute_forward_conv_transpose_1d_f32(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ٻٽو_OP_LOCALS

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح nk = ne00*ne01*ne02;

    GGML_ASSERT(nb00 == حجم(عائم));
    GGML_ASSERT(nb10 == حجم(عائم));

    إذا (ith == 0) {
        memset(params->wdata, 0, params->wsize);

        // prepare kernel data (src0) from (K x Cout x Cin) to (Cin x K x Cout)
        {
            عائم * ثابت wdata = (عائم *) params->wdata + 0;

            لكل (int64_t i02 = 0; i02 < ne02; i02++) {
                لكل (int64_t i01 = 0; i01 < ne01; i01++) {
                    ثابت عائم * ثابت src = (عائم *)((حرف *) src0->معلومات + i02*nb02 + i01*nb01);
                    عائم * dst_معلومات = wdata + i01*ne00*ne02;
                    لكل (int64_t i00 = 0; i00 < ne00; i00++) {
                        dst_معلومات[i00*ne02 + i02] = src[i00];
                    }
                }
            }
        }

        // prepare source data (src1)
        {
            عائم * ثابت wdata = (عائم *) params->wdata + nk;
            عائم * dst_معلومات = wdata;

            لكل (int64_t i11 = 0; i11 < ne11; i11++) {
                ثابت عائم * ثابت src = (عائم *)((حرف *) src1->معلومات + i11*nb11);
                لكل (int64_t i10 = 0; i10 < ne10; i10++) {
                    dst_معلومات[i10*ne11 + i11] = src[i10];
                }
            }
        }

        // need to zero dst since we are accumulating into it
        memset(dst->معلومات, 0, ggml_nbytes(dst));
    }
    ggml_barrier(params->threadpool);

    ثابت int32_t s0 = ((ثابت int32_t*)(dst->op_params))[0];

    // total rows in dst
    ثابت صحيح nr = ne1;

    // rows per thread
    ثابت صحيح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحيح ir0 = dr*ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    عائم * ثابت wdata     = (عائم *) params->wdata + 0;
    عائم * ثابت wdata_src = wdata + nk;

    لكل (صحيح i1 = ir0; i1 < ir1; i1++) {
        عائم * dst_معلومات = (عائم *)((حرف *) dst->معلومات + i1*nb1);
        عائم * wdata_kernel = wdata + i1*ne02*ne00;
        لكل (صحيح i10 = 0; i10 < ne10; i10++) {
            ثابت صحيح i1n = i10*ne11;
            لكل (صحيح i00 = 0; i00 < ne00; i00++) {
                عائم v = 0;
                ggml_vec_dot_f32(ne02, &v, 0,
                        wdata_src + i1n, 0,
                        wdata_kernel + i00*ne02, 0, 1);
                dst_معلومات[i10*s0 + i00] += v;
            }
        }
    }
}

فراغ ggml_compute_forward_conv_transpose_1d(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F16:
            {
                ggml_compute_forward_conv_transpose_1d_f16_f32(params, dst);
            } اكسر;
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_conv_transpose_1d_f32(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_im2col_f32
// src0: kernel [OC, IC, KH, KW]
// src1: image [N, IC, IH, IW]
// dst:  result [N, OH, OW, IC*KH*KW]
ساكن فراغ ggml_compute_forward_im2col_f32(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ٻٽو_OP_LOCALS;

    ثابت int32_t s0 = ((ثابت int32_t *)(dst->op_params))[0];
    ثابت int32_t s1 = ((ثابت int32_t *)(dst->op_params))[1];
    ثابت int32_t p0 = ((ثابت int32_t *)(dst->op_params))[2];
    ثابت int32_t p1 = ((ثابت int32_t *)(dst->op_params))[3];
    ثابت int32_t d0 = ((ثابت int32_t *)(dst->op_params))[4];
    ثابت int32_t d1 = ((ثابت int32_t *)(dst->op_params))[5];
    ثابت منطقي is_2D = ((ثابت int32_t *)(dst->op_params))[6] == 1;

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت int64_t N  = is_2D ? ne13 : ne12;
    ثابت int64_t IC = is_2D ? ne12 : ne11;
    ثابت int64_t IH = is_2D ? ne11 : 1;
    ثابت int64_t IW = ne10;

    ثابت int64_t KH = is_2D ? ne01 : 1;
    ثابت int64_t KW = ne00;

    ثابت int64_t OH = is_2D ? ne2 : 1;
    ثابت int64_t OW = ne1;

    صحيح ofs0 = is_2D ? nb13 : nb12;
    صحيح ofs1 = is_2D ? nb12 : nb11;

    GGML_ASSERT(nb10 == حجم(عائم));

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    {
        عائم * ثابت wdata = (عائم *) dst->معلومات;

        لكل (int64_t اندريون = 0; اندريون < N; اندريون++) {
            لكل (int64_t ioh = 0; ioh < OH; ioh++) { // 1
                لكل (int64_t iow = 0; iow < OW; iow++) {
                    لكل (int64_t iic = ith; iic < IC; iic += nth) {

                        // micro kernel
                        عائم * dst_معلومات = wdata + (اندريون*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
                        ثابت عائم * ثابت src_معلومات = (عائم *)((حرف *) src1->معلومات + اندريون*ofs0 + iic*ofs1); // [IH, IW]

                        لكل (int64_t ikh = 0; ikh < KH; ikh++) {  // 1
                            لكل (int64_t ikw = 0; ikw < KW; ikw++) {
                                ثابت int64_t iiw = iow*s0 + ikw*d0 - p0;
                                ثابت int64_t iih = ioh*s1 + ikh*d1 - p1;

                                إذا (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                    dst_معلومات[iic*(KH*KW) + ikh*KW + ikw] = 0;
                                } وإلا {
                                    dst_معلومات[iic*(KH*KW) + ikh*KW + ikw] = (src_معلومات[iih*IW + iiw]);
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
ساكن فراغ ggml_compute_forward_im2col_f16(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16);

    GGML_TENSOR_ٻٽو_OP_LOCALS;

    ثابت int32_t s0 = ((ثابت int32_t *)(dst->op_params))[0];
    ثابت int32_t s1 = ((ثابت int32_t *)(dst->op_params))[1];
    ثابت int32_t p0 = ((ثابت int32_t *)(dst->op_params))[2];
    ثابت int32_t p1 = ((ثابت int32_t *)(dst->op_params))[3];
    ثابت int32_t d0 = ((ثابت int32_t *)(dst->op_params))[4];
    ثابت int32_t d1 = ((ثابت int32_t *)(dst->op_params))[5];
    ثابت منطقي is_2D = ((ثابت int32_t *)(dst->op_params))[6] == 1;

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت int64_t N  = is_2D ? ne13 : ne12;
    ثابت int64_t IC = is_2D ? ne12 : ne11;
    ثابت int64_t IH = is_2D ? ne11 : 1;
    ثابت int64_t IW = ne10;

    ثابت int64_t KH = is_2D ? ne01 : 1;
    ثابت int64_t KW = ne00;

    ثابت int64_t OH = is_2D ? ne2 : 1;
    ثابت int64_t OW = ne1;

    صحيح ofs0 = is_2D ? nb13 : nb12;
    صحيح ofs1 = is_2D ? nb12 : nb11;

    GGML_ASSERT(nb10 == ggml_type_سائيز(src1->type));

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    {
        ggml_fp16_t * ثابت wdata = (ggml_fp16_t *) dst->معلومات;

        لكل (int64_t اندريون = 0; اندريون < N; اندريون++) {
            لكل (int64_t ioh = 0; ioh < OH; ioh++) { // 1
                لكل (int64_t iow = 0; iow < OW; iow++) {
                    لكل (int64_t iic = ith; iic < IC; iic += nth) {

                        // micro kernel
                        ggml_fp16_t * dst_معلومات = wdata + (اندريون*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
                        ثابت عائم * ثابت src_معلومات_f32 = src1->type == GGML_TYPE_F32
                            ? (ثابت عائم *)((ثابت حرف *) src1->معلومات + اندريون*ofs0 + iic*ofs1)
                            : خالي_اشارو; // [IH, IW]
                        ثابت ggml_fp16_t * ثابت src_معلومات_f16 = src1->type == GGML_TYPE_F16
                            ? (ثابت ggml_fp16_t *)((ثابت حرف *) src1->معلومات + اندريون*ofs0 + iic*ofs1)
                            : خالي_اشارو; // [IH, IW]

                        لكل (int64_t ikh = 0; ikh < KH; ikh++) {  // 1
                            لكل (int64_t ikw = 0; ikw < KW; ikw++) {
                                ثابت int64_t iiw = iow*s0 + ikw*d0 - p0;
                                ثابت int64_t iih = ioh*s1 + ikh*d1 - p1;

                                إذا (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                    dst_معلومات[iic*(KH*KW) + ikh*KW + ikw] = 0;
                                } وإلا {
                                    إذا (src_معلومات_f32 != خالي_اشارو) {
                                        dst_معلومات[iic*(KH*KW) + ikh*KW + ikw] = GGML_CPU_FP32_منزل_FP16(src_معلومات_f32[iih*IW + iiw]);
                                    } وإلا {
                                        dst_معلومات[iic*(KH*KW) + ikh*KW + ikw] = src_معلومات_f16[iih*IW + iiw];
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

فراغ ggml_compute_forward_im2col(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {
    بدل (dst->type) {
        حالة GGML_TYPE_F16:
            {
                ggml_compute_forward_im2col_f16(params, dst);
            } اكسر;
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_im2col_f32(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_im2col_back_f32

فراغ ggml_compute_forward_im2col_پٺ_f32(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0]; // gradients of forward pass output
    ثابت ggml_tensor * src1 = dst->src[1]; // convolution kernel

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ٻٽو_OP_LOCALS;

    ثابت int32_t s0 = ((ثابت int32_t *)(dst->op_params))[0];
    ثابت int32_t s1 = ((ثابت int32_t *)(dst->op_params))[1];
    ثابت int32_t p0 = ((ثابت int32_t *)(dst->op_params))[2];
    ثابت int32_t p1 = ((ثابت int32_t *)(dst->op_params))[3];
    ثابت int32_t d0 = ((ثابت int32_t *)(dst->op_params))[4];
    ثابت int32_t d1 = ((ثابت int32_t *)(dst->op_params))[5];
    ثابت منطقي is_2D = ((ثابت int32_t *)(dst->op_params))[6] == 1;

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت int64_t N  = is_2D ? ne3 : ne2;
    ثابت int64_t IC = is_2D ? ne2 : ne1;
    ثابت int64_t IH = is_2D ? ne1 : 1;
    ثابت int64_t IW = ne0;

    ثابت int64_t KH = is_2D ? ne11 : 1;
    ثابت int64_t KW = ne10;

    ثابت int64_t OH = is_2D ? ne02 : 1;
    ثابت int64_t OW = ne01;

    صحيح ofs0 = is_2D ? nb3 : nb2;
    صحيح ofs1 = is_2D ? nb2 : nb1;

    GGML_ASSERT(nb0  == حجم(عائم));

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    {
        عائم * ثابت wdata = (عائم *) dst->معلومات;

        لكل (int64_t اندريون = 0; اندريون < N; اندريون++) {
            لكل (int64_t iic = ith; iic < IC; iic += nth) {
                لكل (int64_t iih = 0; iih < IH; iih++) {
                    لكل (int64_t iiw = 0; iiw < IW; iiw++) {

                        // micro kernel
                        عائم grad = 0.0f;
                        لكل (int64_t ikh = 0; ikh < KH; ikh++) {
                            لكل (int64_t ikw = 0; ikw < KW; ikw++) {
                                // For s0 > 1 some values were skipped over in the forward pass.
                                // These values have tmpw % s0 != 0 and need to be skipped in the backwards pass as well.
                                ثابت int64_t tmpw = (iiw + p0 - ikw*d0);
                                إذا (tmpw % s0 != 0) {
                                    استمر;
                                }
                                ثابت int64_t iow = tmpw / s0;

                                // Equivalent logic as above except for s1.
                                int64_t ioh;
                                إذا (is_2D) {
                                    ثابت int64_t tmph = iih + p1 - ikh*d1;

                                    إذا (tmph % s1 != 0) {
                                        استمر;
                                    }

                                    ioh = tmph / s1;
                                } وإلا {
                                    ioh = 0;
                                }

                                إذا (iow < 0 || iow >= OW || ioh < 0 || ioh >= OH) {
                                    استمر;
                                }

                                ثابت عائم * ثابت grad_اندريون = (ثابت عائم *) src0->معلومات
                                    + (اندريون*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
                                grad += grad_اندريون[iic*(KH*KW) + ikh*KW + ikw];
                            }
                        }
                        عائم * dst_معلومات = (عائم *)((حرف *) wdata + (اندريون*ofs0 + iic*ofs1)); // [IH, IW]
                        dst_معلومات[iih*IW + iiw] = grad;
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
ساكن فراغ ggml_compute_forward_im2col_3d_f16(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16);

    GGML_TENSOR_ٻٽو_OP_LOCALS;

    ثابت int32_t s0 = ((ثابت int32_t *)(dst->op_params))[0];
    ثابت int32_t s1 = ((ثابت int32_t *)(dst->op_params))[1];
    ثابت int32_t s2 = ((ثابت int32_t *)(dst->op_params))[2];
    ثابت int32_t p0 = ((ثابت int32_t *)(dst->op_params))[3];
    ثابت int32_t p1 = ((ثابت int32_t *)(dst->op_params))[4];
    ثابت int32_t p2 = ((ثابت int32_t *)(dst->op_params))[5];
    ثابت int32_t d0 = ((ثابت int32_t *)(dst->op_params))[6];
    ثابت int32_t d1 = ((ثابت int32_t *)(dst->op_params))[7];
    ثابت int32_t d2 = ((ثابت int32_t *)(dst->op_params))[8];
    ثابت int32_t IC = ((ثابت int32_t *)(dst->op_params))[9];


    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت int64_t N  = ne13 / IC;
    ثابت int64_t ID = ne12;
    ثابت int64_t IH = ne11;
    ثابت int64_t IW = ne10;

    ثابت int64_t OC = ne03 / IC;
    GGML_اڻ_استعمال_ٿيل(OC);
    ثابت int64_t KD = ne02;
    ثابت int64_t KH = ne01;
    ثابت int64_t KW = ne00;

    ثابت int64_t OD = ne3 / N;
    ثابت int64_t OH = ne2;
    ثابت int64_t OW = ne1;
    ثابت int64_t OH_OW = OH*OW;
    ثابت int64_t KD_KH_KW = KD*KH*KW;
    ثابت int64_t KH_KW = KH*KW;
    ثابت int64_t IC_KD_KH_KW = IC*KD*KH*KW;

    GGML_ASSERT(nb10 == حجم(عائم));

    // im2col: [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
    {
        ggml_fp16_t * ثابت wdata = (ggml_fp16_t *) dst->معلومات;

        لكل (int64_t اندريون = 0; اندريون < N; اندريون++) {
            لكل (int64_t iod = 0; iod < OD; iod++) {
                لكل (int64_t ioh = 0; ioh < OH; ioh++) {
                    لكل (int64_t iow = 0; iow < OW; iow++) {
                        لكل (int64_t iic = ith; iic < IC; iic += nth) {

                            // micro kernel
                            ggml_fp16_t * dst_معلومات = wdata + (اندريون*OD*OH_OW + iod*OH_OW + ioh*OW + iow)*IC_KD_KH_KW; // [IC, KD, KH, KW]
                            ثابت عائم * ثابت src_معلومات = (ثابت عائم *) ((ثابت حرف *)src1->معلومات + (اندريون*IC + iic)*nb13); // [ID, IH, IW]

                            لكل (int64_t ikd = 0; ikd < KD; ikd++) {
                                لكل (int64_t ikh = 0; ikh < KH; ikh++) {
                                    لكل (int64_t ikw = 0; ikw < KW; ikw++) {
                                        ثابت int64_t iiw = iow*s0 + ikw*d0 - p0;
                                        ثابت int64_t iih = ioh*s1 + ikh*d1 - p1;
                                        ثابت int64_t iid = iod*s2 + ikd*d2 - p2;

                                        إذا (iid < 0 || iid >= ID || iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                            dst_معلومات[iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw] = 0;
                                        } وإلا {
                                            ثابت عائم * ثابت s = (ثابت عائم *) ((ثابت حرف *)src_معلومات + iid*nb12 + iih*nb11 + iiw*nb10); // [ID, IH, IW]
                                            dst_معلومات[iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw] = GGML_CPU_FP32_منزل_FP16(*s);
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
ساكن فراغ ggml_compute_forward_im2col_3d_f32(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ٻٽو_OP_LOCALS;

    ثابت int32_t s0 = ((ثابت int32_t *)(dst->op_params))[0];
    ثابت int32_t s1 = ((ثابت int32_t *)(dst->op_params))[1];
    ثابت int32_t s2 = ((ثابت int32_t *)(dst->op_params))[2];
    ثابت int32_t p0 = ((ثابت int32_t *)(dst->op_params))[3];
    ثابت int32_t p1 = ((ثابت int32_t *)(dst->op_params))[4];
    ثابت int32_t p2 = ((ثابت int32_t *)(dst->op_params))[5];
    ثابت int32_t d0 = ((ثابت int32_t *)(dst->op_params))[6];
    ثابت int32_t d1 = ((ثابت int32_t *)(dst->op_params))[7];
    ثابت int32_t d2 = ((ثابت int32_t *)(dst->op_params))[8];
    ثابت int32_t IC = ((ثابت int32_t *)(dst->op_params))[9];


    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت int64_t N  = ne13 / IC;
    ثابت int64_t ID = ne12;
    ثابت int64_t IH = ne11;
    ثابت int64_t IW = ne10;

    ثابت int64_t OC = ne03 / IC;
    GGML_اڻ_استعمال_ٿيل(OC);
    ثابت int64_t KD = ne02;
    ثابت int64_t KH = ne01;
    ثابت int64_t KW = ne00;

    ثابت int64_t OD = ne3 / N;
    ثابت int64_t OH = ne2;
    ثابت int64_t OW = ne1;

    ثابت int64_t OH_OW = OH*OW;
    ثابت int64_t KD_KH_KW = KD*KH*KW;
    ثابت int64_t KH_KW = KH*KW;
    ثابت int64_t IC_KD_KH_KW = IC*KD*KH*KW;

    GGML_ASSERT(nb10 == حجم(عائم));

    // im2col: [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
    {
        عائم * ثابت wdata = (عائم *) dst->معلومات;

        لكل (int64_t اندريون = 0; اندريون < N; اندريون++) {
            لكل (int64_t iod = 0; iod < OD; iod++) {
                لكل (int64_t ioh = 0; ioh < OH; ioh++) {
                    لكل (int64_t iow = 0; iow < OW; iow++) {
                        لكل (int64_t iic = ith; iic < IC; iic += nth) {

                            // micro kernel
                            عائم * dst_معلومات = wdata + (اندريون*OD*OH_OW + iod*OH_OW + ioh*OW + iow)*IC_KD_KH_KW; // [IC, KD, KH, KW]
                            ثابت عائم * ثابت src_معلومات = (ثابت عائم *) ((ثابت حرف *)src1->معلومات + (اندريون*IC + iic)*nb13); // [ID, IH, IW]

                            لكل (int64_t ikd = 0; ikd < KD; ikd++) {
                                لكل (int64_t ikh = 0; ikh < KH; ikh++) {
                                    لكل (int64_t ikw = 0; ikw < KW; ikw++) {
                                        ثابت int64_t iiw = iow*s0 + ikw*d0 - p0;
                                        ثابت int64_t iih = ioh*s1 + ikh*d1 - p1;
                                        ثابت int64_t iid = iod*s2 + ikd*d2 - p2;

                                        إذا (iid < 0 || iid >= ID || iih < 0 || iih >= IH || iiw < 0 || iiw >= IW || iid < 0 || iid >= ID) {
                                            dst_معلومات[iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw] = 0;
                                        } وإلا {
                                            ثابت عائم * ثابت s = (ثابت عائم *) ((ثابت حرف *)src_معلومات + iid*nb12 + iih*nb11 + iiw*nb10); // [ID, IH, IW]
                                            dst_معلومات[iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw] = *s;
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


فراغ ggml_compute_forward_im2col_3d(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {
    بدل (dst->type) {
        حالة GGML_TYPE_F16:
            {
                ggml_compute_forward_im2col_3d_f16(params, dst);
            } اكسر;
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_im2col_3d_f32(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ساكن فراغ ggml_call_mul_mat(ggml_type type, ثابت ggml_compute_params * params, int64_t m, int64_t n, int64_t k,
                              فراغ * a, فراغ * b, عائم * c) {
    ثابت ggml_type_traits * traits = ggml_get_type_traits(type);
    بنية ggml_tensor src1 = {};
    src1.type  = type;
    src1.ne[0] = k;
    src1.ne[1] = m;
    src1.ne[2] = 1;
    src1.ne[3] = 1;
    src1.nb[0] = traits->type_سائيز;
    src1.nb[1] = k * traits->type_سائيز;
    src1.nb[2] = src1.nb[1];
    src1.nb[3] = src1.nb[2];
    src1.معلومات  = a;

    بنية ggml_tensor src0 = {};
    src0.type  = type;
    src0.ne[0] = k;
    src0.ne[1] = n;
    src0.ne[2] = 1;
    src0.ne[3] = 1;
    src0.nb[0] = traits->type_سائيز;
    src0.nb[1] = k * traits->type_سائيز;
    src0.nb[2] = src0.nb[1];
    src0.nb[3] = src0.nb[2];
    src0.معلومات  = b;

    بنية ggml_tensor dst = {};
    dst.ne[0] = n;
    dst.ne[1] = m;
    dst.ne[2] = 1;
    dst.ne[3] = 1;
    dst.nb[0] = حجم(عائم);
    dst.nb[1] = n * حجم(عائم);
    dst.nb[2] = dst.nb[1];
    dst.nb[3] = dst.nb[2];
    dst.معلومات  = c;
    dst.src[0] = &src0;
    dst.src[1] = &src1;

    ggml_compute_forward_mul_mat(params, &dst);
}

ساكن مضمن int64_t ggml_wrap_around(int64_t coord, int64_t size) {
    أرجع (coord  + size) % size; // adding size avoids negative number weirdness
}

// ggml_compute_forward_col2im_1d
//
// Scatter-add columns [K*OC, T_in] -> signal [T_out, OC]
// where T_out = (T_in - 1)*s + K - 2*p.  Gather approach: each output reads ceil(K/s) inputs.
// Parallelized over the time axis so the split stays balanced whatever OC is.
// Supports F32, F16, BF16 input/output (same type), F32 accumulator.

سانچو <قسم_نالو elem_t>
ساكن فراغ ggml_compute_forward_col2im_1d_impl(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src = dst->src[0];  // [K*OC, T_in]

    GGML_ASSERT(ggml_is_contiguous(src));
    GGML_ASSERT(ggml_is_contiguous(dst));

    ثابت int32_t s0 = ((ثابت int32_t *)(dst->op_params))[0];
    ثابت int32_t OC = ((ثابت int32_t *)(dst->op_params))[1];
    ثابت int32_t p0 = ((ثابت int32_t *)(dst->op_params))[2];

    ثابت int64_t K_OC = src->ne[0];
    ثابت int64_t T_اندريون = src->ne[1];
    ثابت int64_t K    = K_OC / OC;
    ثابت int64_t T_ٻاهريون = dst->ne[0];

    ثابت elem_t * کانو_معلومات = (ثابت elem_t *) src->معلومات;
    elem_t       * dst_معلومات = (elem_t *) dst->معلومات;

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    // Parallelize over the time axis: the split stays balanced whatever OC is,
    // down to OC = 1 for mono audio, and threads read disjoint column bands
    ثابت int64_t dr = (T_ٻاهريون + nth - 1) / nth;
    ثابت int64_t it0 = dr * ith;
    ثابت int64_t it1 = it0 + dr < T_ٻاهريون ? it0 + dr : T_ٻاهريون;

    لكل (int64_t oc = 0; oc < OC; oc++) {
        لكل (int64_t t_ٻاهريون = it0; t_ٻاهريون < it1; t_ٻاهريون++) {
            ثابت int64_t t_abs = t_ٻاهريون + p0;  // absolute position in uncropped signal
            // Gather: find all (t_in, k) where t_in * s + k == t_abs, 0 <= k < K
            int64_t t_اندريون_min = (t_abs - K + 1 + s0 - 1) / s0;  // ceil((t_abs-K+1)/s)
            إذا (t_اندريون_min < 0) t_اندريون_min = 0;
            int64_t t_اندريون_max = t_abs / s0;
            إذا (t_اندريون_max >= T_اندريون) t_اندريون_max = T_اندريون - 1;

            عائم جوڙ = 0.0f;
            لكل (int64_t t_اندريون = t_اندريون_min; t_اندريون <= t_اندريون_max; t_اندريون++) {
                int64_t k = t_abs - t_اندريون * s0;
                إذا (k >= 0 && k < K) {
                    // col layout: [K*OC, T_in], element (oc*K+k, t_in)
                    جوڙ += type_conversion_جدول<elem_t>::منزل_f32(کانو_معلومات[(oc * K + k) + t_اندريون * K_OC]);
                }
            }
            // dst layout: [T_out, OC], element (t_out, oc)
            dst_معلومات[t_ٻاهريون + oc * T_ٻاهريون] = type_conversion_جدول<elem_t>::from_f32(جوڙ);
        }
    }
}

فراغ ggml_compute_forward_col2im_1d(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    بدل (dst->src[0]->type) {
        حالة GGML_TYPE_F32:  ggml_compute_forward_col2im_1d_impl<عائم>      (params, dst); اكسر;
        حالة GGML_TYPE_F16:  ggml_compute_forward_col2im_1d_impl<ggml_fp16_t>(params, dst); اكسر;
        حالة GGML_TYPE_BF16: ggml_compute_forward_col2im_1d_impl<ggml_bf16_t>(params, dst); اكسر;
        افتراضي: GGML_ABORT("col2im_1d: unsupported type %d", dst->src[0]->type);
    }
}

// ggml_compute_forward_conv_2d


ساكن فراغ ggml_compute_forward_conv_2d_impl(ثابت ggml_compute_params * params,
                                              ثابت ggml_tensor *         kernel,  // [KW, KH, IC, OC]
                                              ثابت ggml_tensor *         src,     // [W, H, C, N]
                                              ggml_tensor *               dst,     // [OW, OH, OC, N]
                                              ggml_type                   kernel_type) {

    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(kernel_type == GGML_TYPE_F16 || kernel_type == GGML_TYPE_F32);
    GGML_ASSERT(kernel->type == kernel_type);

    ثابت ggml_type_traits * traits = ggml_get_type_traits(kernel_type);

    ثابت int32_t stride_x   = dst->op_params[0];
    ثابت int32_t stride_y   = dst->op_params[1];
    ثابت int32_t pad_x      = dst->op_params[2];
    ثابت int32_t pad_y      = dst->op_params[3];
    ثابت int32_t dilation_x = dst->op_params[4];
    ثابت int32_t dilation_y = dst->op_params[5];

    ثابت int64_t c_اندريون  = src->ne[2];
    ثابت int64_t c_ٻاهريون = kernel->ne[3];
    GGML_ASSERT(c_اندريون == kernel->ne[2]);

    ثابت int64_t src_w = src->ne[0];
    ثابت int64_t src_h = src->ne[1];
    ثابت int64_t knl_w = kernel->ne[0];
    ثابت int64_t knl_h = kernel->ne[1];
    ثابت int64_t dst_w = dst->ne[0];
    ثابت int64_t dst_h = dst->ne[1];

    ثابت عائم * src_معلومات = (عائم *) src->معلومات;
    فراغ  * knl_معلومات       = kernel->معلومات;
    عائم * dst_معلومات       = (عائم *) dst->معلومات;

    ثابت int64_t knl_n           = knl_w * knl_h * c_اندريون;
    ثابت int64_t patch_ڪل     = dst->ne[3] * dst_w * dst_h;

    ثابت int64_t space_في_patch   = knl_n * traits->type_سائيز + c_ٻاهريون * حجم(عائم);
    ثابت int64_t batch_سائيز        = params->wsize / space_في_patch;
    ثابت int64_t patches_في_batch = batch_سائيز > 8 ? (batch_سائيز / 8) * 8 : batch_سائيز;
    ثابت int64_t batch_n           = (patch_ڪل + patches_في_batch - 1) / patches_في_batch;

    GGML_ASSERT(patches_في_batch > 0 && batch_سائيز >= 1);

    فراغ * tmp = params->wdata;

    لكل (int64_t batch_i = 0; batch_i < batch_n; ++batch_i) {

        ثابت int64_t patch_شروعات_batch = batch_i * patches_في_batch;
        ثابت int64_t patch_پڇاڙي_batch   = std::min(patch_شروعات_batch + patches_في_batch,
                                              patch_ڪل);
        ثابت int64_t patch_n           = patch_پڇاڙي_batch - patch_شروعات_batch;

        ثابت int64_t patch_في_thread  = (patch_n + params->nth - 1) / params->nth;
        ثابت int64_t patch_شروعات       = patch_شروعات_batch + params->ith * patch_في_thread;
        ثابت int64_t patch_پڇاڙي         = std::min(patch_شروعات + patch_في_thread, patch_پڇاڙي_batch);

        //im2col for a patch
        لكل (int64_t p = patch_شروعات; p < patch_پڇاڙي; ++p) {
            ثابت int64_t  batch_n     =  p / (dst_w * dst_h);
            ثابت int64_t  src_x       = (p / dst_w) % dst_h;
            ثابت int64_t  src_y       =  p % dst_w;

            ثابت عائم * src_بنياد = (ثابت عائم *)((ثابت حرف *)src_معلومات + batch_n * src->nb[3]);
            حرف *        dst_صف  = (حرف *) tmp + (p % patches_في_batch) * knl_n * traits->type_سائيز;

            لكل (int64_t ic = 0; ic < c_اندريون; ++ic) {
                لكل (int64_t ky = 0; ky < knl_h; ++ky) {
                    لكل (int64_t kx = 0; kx < knl_w; ++kx) {
                        ثابت int64_t sy = src_x * stride_y + ky * dilation_y - pad_y;
                        ثابت int64_t sx = src_y * stride_x + kx * dilation_x - pad_x;

                        int64_t dst_idx = ic * (knl_h * knl_w) + ky * knl_w + kx;

                        عائم src_val;
                        إذا (sy < 0 || sy >= src_h || sx < 0 || sx >= src_w) {
                            src_val = 0.0f;
                        } وإلا {
                            ثابت عائم * src_ptr = (ثابت عائم *)((ثابت حرف *)src_بنياد + sx * src->nb[0] + sy * src->nb[1] + ic * src->nb[2]);
                            src_val               = *src_ptr;
                        }

                        حرف * element_ptr = dst_صف + dst_idx * traits->type_سائيز;
                        إذا (kernel_type == GGML_TYPE_F32) {
                            *(عائم *) element_ptr = src_val;
                        } وإلا إذا (kernel_type == GGML_TYPE_F16) {
                            *(ggml_fp16_t *) element_ptr = GGML_CPU_FP32_منزل_FP16(src_val);
                        }
                    }
                }
            }
        }   // patches handled by this thread

        ggml_barrier(params->threadpool);

        عائم * gemm_output = (عائم *) ((حرف *) tmp + patches_في_batch * knl_n * traits->type_سائيز);

        GGML_ASSERT(gemm_output + patch_n * c_ٻاهريون <= (عائم*)tmp + params->wsize);

        // GEMM: patches[patch_n, knl_n] × kernel[knl_n, c_out] = output[patch_n, c_out]
        ggml_call_mul_mat(kernel_type, params, patch_n, c_ٻاهريون, knl_n, tmp, knl_معلومات, gemm_output);

        ggml_barrier(params->threadpool);


        //permute back [OC, N, OH, OW] to [N, OC, OH, OW]
        ثابت int64_t ترتيب_بدلايو_في_thread = (patch_n + params->nth - 1) / params->nth;
        ثابت int64_t ترتيب_بدلايو_شروعات = params->ith * ترتيب_بدلايو_في_thread;
        ثابت int64_t ترتيب_بدلايو_پڇاڙي = std::min(ترتيب_بدلايو_شروعات + ترتيب_بدلايو_في_thread, patch_n);

        لكل (int64_t i = ترتيب_بدلايو_شروعات; i < ترتيب_بدلايو_پڇاڙي; ++i) {
            ثابت int64_t p       = patch_شروعات_batch + i;
            ثابت int64_t batch_n = p / (dst_w * dst_h);
            ثابت int64_t dst_y   = (p / dst_w) % dst_h;
            ثابت int64_t dst_x   = p % dst_w;

            لكل (int64_t oc = 0; oc < c_ٻاهريون; ++oc) {
                ثابت عائم value = gemm_output[i * c_ٻاهريون + oc];
                عائم * dst_ptr = (عائم *)((حرف *)dst_معلومات + dst_x * dst->nb[0] + dst_y * dst->nb[1] + oc * dst->nb[2] + batch_n * dst->nb[3]);
                *dst_ptr = value;
            }
        }
    }
}

فراغ ggml_compute_forward_conv_2d(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    ggml_compute_forward_conv_2d_impl(params, src0, src1, dst, src0->type);
}

// ggml_compute_forward_conv_3d

ساكن فراغ ggml_compute_forward_conv_3d_impl(ثابت ggml_compute_params * params,
                                              ثابت ggml_tensor *         kernel,
                                              ثابت ggml_tensor *         src,
                                              ggml_tensor *               dst,
                                              ggml_type                   kernel_type) {

    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(kernel_type == GGML_TYPE_F16 || kernel_type == GGML_TYPE_F32);
    GGML_ASSERT(kernel->type == kernel_type);

    ثابت ggml_type_traits * traits = ggml_get_type_traits(kernel_type);

    ثابت int32_t s0 = dst->op_params[0];
    ثابت int32_t s1 = dst->op_params[1];
    ثابت int32_t s2 = dst->op_params[2];
    ثابت int32_t p0 = dst->op_params[3];
    ثابت int32_t p1 = dst->op_params[4];
    ثابت int32_t p2 = dst->op_params[5];
    ثابت int32_t d0 = dst->op_params[6];
    ثابت int32_t d1 = dst->op_params[7];
    ثابت int32_t d2 = dst->op_params[8];
    ثابت int32_t c  = dst->op_params[9];
    ثابت int32_t n  = dst->op_params[10];
    ثابت int32_t oc = dst->op_params[11];

    ثابت int64_t src_w = src->ne[0];
    ثابت int64_t src_h = src->ne[1];
    ثابت int64_t src_d = src->ne[2];
    ثابت int64_t knl_w = kernel->ne[0];
    ثابت int64_t knl_h = kernel->ne[1];
    ثابت int64_t knl_d = kernel->ne[2];
    ثابت int64_t dst_w = dst->ne[0];
    ثابت int64_t dst_h = dst->ne[1];
    ثابت int64_t dst_d = dst->ne[2];

    ثابت عائم * src_معلومات = (عائم *) src->معلومات;
    فراغ  * knl_معلومات       = kernel->معلومات;
    عائم * dst_معلومات       = (عائم *) dst->معلومات;

    ثابت int64_t knl_n_في_channel = knl_w * knl_h * knl_d;
    ثابت int64_t knl_n_ڪل       = knl_n_في_channel * c;
    ثابت int64_t patch_ڪل       = n * dst_w * dst_h * dst_d;

    ثابت int64_t space_في_patch   = knl_n_ڪل * traits->type_سائيز + oc * حجم(عائم);
    ثابت int64_t batch_سائيز        = params->wsize / space_في_patch;
    ثابت int64_t patches_في_batch = batch_سائيز > 8 ? (batch_سائيز / 8) * 8 : batch_سائيز;
    ثابت int64_t batch_n           = (patch_ڪل + patches_في_batch - 1) / patches_في_batch;

    GGML_ASSERT(patches_في_batch > 0 && batch_سائيز >= 1);

    فراغ * tmp = params->wdata;

    لكل (int64_t batch_i = 0; batch_i < batch_n; ++batch_i) {
        ثابت int64_t patch_شروعات_batch = batch_i * patches_في_batch;
        ثابت int64_t patch_پڇاڙي_batch   = std::min(patch_شروعات_batch + patches_في_batch, patch_ڪل);
        ثابت int64_t patch_n_اندريون_batch  = patch_پڇاڙي_batch - patch_شروعات_batch;

        ثابت int64_t patch_في_thread  = (patch_n_اندريون_batch + params->nth - 1) / params->nth;
        ثابت int64_t patch_شروعات       = patch_شروعات_batch + params->ith * patch_في_thread;
        ثابت int64_t patch_پڇاڙي         = std::min(patch_شروعات + patch_في_thread, patch_پڇاڙي_batch);

        لكل (int64_t p = patch_شروعات; p < patch_پڇاڙي; ++p) {
            ثابت int64_t p_اندريون_batch = p % (dst_w * dst_h * dst_d);
            ثابت int64_t p_اندريون_اونهائي = p_اندريون_batch % (dst_w * dst_h);
            ثابت int64_t batch_idx  = p / (dst_w * dst_h * dst_d);
            ثابت int64_t dst_z      = p_اندريون_batch / (dst_w * dst_h);
            ثابت int64_t dst_y      = p_اندريون_اونهائي / dst_w;
            ثابت int64_t dst_x      = p_اندريون_اونهائي % dst_w;

            حرف * dst_صف = (حرف *) tmp + (p % patches_في_batch) * knl_n_ڪل * traits->type_سائيز;

            لكل (int64_t ic = 0; ic < c; ++ic) {
                لكل (int64_t kz = 0; kz < knl_d; ++kz) {
                    لكل (int64_t ky = 0; ky < knl_h; ++ky) {
                        لكل (int64_t kx = 0; kx < knl_w; ++kx) {
                            ثابت int64_t sz = dst_z * s2 + kz * d2 - p2;
                            ثابت int64_t sy = dst_y * s1 + ky * d1 - p1;
                            ثابت int64_t sx = dst_x * s0 + kx * d0 - p0;

                            int64_t dst_idx = ic * knl_n_في_channel + kz * (knl_h * knl_w) + ky * knl_w + kx;

                            عائم src_val;
                            إذا (sz < 0 || sz >= src_d || sy < 0 || sy >= src_h || sx < 0 || sx >= src_w) {
                                src_val = 0.0f;
                            } وإلا {
                                ثابت int64_t cn_idx = batch_idx * c + ic;
                                ثابت عائم * src_ptr = (ثابت عائم *)((ثابت حرف *)src_معلومات + sx*src->nb[0] + sy*src->nb[1] + sz*src->nb[2] + cn_idx*src->nb[3]);
                                src_val = *src_ptr;
                            }

                            حرف * element_ptr = dst_صف + dst_idx * traits->type_سائيز;
                            إذا (kernel_type == GGML_TYPE_F32) {
                                *(عائم *)element_ptr = src_val;
                            } وإلا إذا (kernel_type == GGML_TYPE_F16) {
                                *(ggml_fp16_t *)element_ptr = GGML_CPU_FP32_منزل_FP16(src_val);
                            }
                        }
                    }
                }
            }
        }

        ggml_barrier(params->threadpool);

        عائم * gemm_output = (عائم *) ((حرف *) tmp + patches_في_batch * knl_n_ڪل * traits->type_سائيز);
        ggml_call_mul_mat(kernel_type, params, patch_n_اندريون_batch, oc, knl_n_ڪل, tmp, knl_معلومات, gemm_output);

        ggml_barrier(params->threadpool);

        ثابت int64_t ترتيب_بدلايو_في_thread = (patch_n_اندريون_batch + params->nth - 1) / params->nth;
        ثابت int64_t ترتيب_بدلايو_شروعات = params->ith * ترتيب_بدلايو_في_thread;
        ثابت int64_t ترتيب_بدلايو_پڇاڙي = std::min(ترتيب_بدلايو_شروعات + ترتيب_بدلايو_في_thread, patch_n_اندريون_batch);

        لكل (int64_t i = ترتيب_بدلايو_شروعات; i < ترتيب_بدلايو_پڇاڙي; ++i) {
            ثابت int64_t p = patch_شروعات_batch + i;
            ثابت int64_t p_اندريون_batch = p % (dst_w * dst_h * dst_d);
            ثابت int64_t p_اندريون_اونهائي = p_اندريون_batch % (dst_w * dst_h);
            ثابت int64_t batch_idx  = p / (dst_w * dst_h * dst_d);
            ثابت int64_t dst_z      = p_اندريون_batch / (dst_w * dst_h);
            ثابت int64_t dst_y      = p_اندريون_اونهائي / dst_w;
            ثابت int64_t dst_x      = p_اندريون_اونهائي % dst_w;

            لكل (int64_t ioc = 0; ioc < oc; ++ioc) {
                ثابت عائم value = gemm_output[i * oc + ioc];
                ثابت int64_t ocn_idx = batch_idx * oc + ioc;
                عائم * dst_ptr = (عائم *)((حرف *)dst_معلومات + dst_x*dst->nb[0] + dst_y*dst->nb[1] + dst_z*dst->nb[2] + ocn_idx*dst->nb[3]);
                *dst_ptr = value;
            }
        }
    }
}

فراغ ggml_compute_forward_conv_3d(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    ggml_compute_forward_conv_3d_impl(params, src0, src1, dst, src0->type);
}

سانچو <قسم_نالو kernel_t>
ساكن فراغ ggml_compute_forward_conv_transpose_2d_impl(
    ثابت ggml_compute_params * params,
          ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ٻٽو_OP_LOCALS

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح nk = ne00*ne01*ne02*ne03;

    GGML_ASSERT(nb00 == ggml_type_سائيز(src0->type));
    GGML_ASSERT(nb10 == حجم(عائم));

    إذا (ith == 0) {
        memset(params->wdata, 0, params->wsize);

        // permute kernel data (src0) from (Kw x Kh x Cout x Cin) to (Cin x Kw x Kh x Cout)
        {
            kernel_t * ثابت wdata = (kernel_t *) params->wdata + 0;

            لكل (int64_t i03 = 0; i03 < ne03; i03++) {
                لكل (int64_t i02 = 0; i02 < ne02; i02++) {
                    ثابت kernel_t * ثابت src = (kernel_t *)((حرف *) src0->معلومات + i03*nb03 + i02*nb02);
                    kernel_t * dst_معلومات = wdata + i02*ne01*ne00*ne03;
                    لكل (int64_t i01 = 0; i01 < ne01; i01++) {
                        لكل (int64_t i00 = 0; i00 < ne00; i00++) {
                            dst_معلومات[i01*ne00*ne03 + i00*ne03 + i03] = src[i01 * ne00 + i00];
                        }
                    }
                }
            }
        }

        // permute source data (src1) from (Sw x Sh x Cin) to (Cin x Sw x Sh), for all batches
        {
            kernel_t * ثابت wdata = (kernel_t *) params->wdata + nk;
            لكل (صحيح i13 = 0; i13 < ne13; i13++) {
                kernel_t * ثابت wdata_b = wdata + i13*ne10*ne11*ne12;
                لكل (صحيح i12 = 0; i12 < ne12; i12++) {
                    لكل (صحيح i11 = 0; i11 < ne11; i11++) {
                        ثابت عائم * ثابت src = (عائم *)((حرف *) src1->معلومات + i13*nb13 + i12*nb12 + i11*nb11);
                        kernel_t * dst_معلومات = wdata_b + i11*ne10*ne12;
                        لكل (صحيح i10 = 0; i10 < ne10; i10++) {
                            إذا constexpr (std::is_same_v<kernel_t, ggml_fp16_t>) {
                                dst_معلومات[i10*ne12 + i12] = GGML_CPU_FP32_منزل_FP16(src[i10]);
                            } وإلا {
                                dst_معلومات[i10*ne12 + i12] = src[i10];
                            }
                        }
                    }
                }
            }
        }

        memset(dst->معلومات, 0, ggml_nbytes(dst));
    }
    ggml_barrier(params->threadpool);

    ثابت int32_t stride = ggml_get_op_params_i32(dst, 0);

    // total patches in dst
    ثابت صحيح np = ne2;

    // patches per thread
    ثابت صحيح dp = (np + nth - 1)/nth;

    // patch range for this thread
    ثابت صحيح ip0 = dp*ith;
    ثابت صحيح ip1 = MIN(ip0 + dp, np);

    kernel_t * ثابت wdata = (kernel_t *) params->wdata + 0;
    kernel_t * ثابت wdata_src = wdata + nk;

    لكل (صحيح i2 = ip0; i2 < ip1; i2++) { // Cout
        kernel_t * wdata_kernel = wdata + i2*ne01*ne00*ne03;
        لكل (صحيح i3 = 0; i3 < ne3; i3++) { // batch
            عائم * dst_معلومات = (عائم *)((حرف *) dst->معلومات + i3*nb3 + i2*nb2);
            kernel_t * wdata_src_b = wdata_src + i3*ne10*ne11*ne12;
            لكل (صحيح i11 = 0; i11 < ne11; i11++) {
                لكل (صحيح i10 = 0; i10 < ne10; i10++) {
                    ثابت صحيح i1n = i11*ne10*ne12 + i10*ne12;
                    لكل (صحيح i01 = 0; i01 < ne01; i01++) {
                        لكل (صحيح i00 = 0; i00 < ne00; i00++) {
                            عائم v = 0;
                            إذا constexpr (std::is_same_v<kernel_t, ggml_fp16_t>) {
                                ggml_vec_dot_f16(ne03, &v, 0,
                                        wdata_src_b + i1n, 0,
                                        wdata_kernel + i01*ne00*ne03 + i00*ne03, 0, 1);
                            } وإلا {
                                ggml_vec_dot_f32(ne03, &v, 0,
                                        wdata_src_b + i1n, 0,
                                        wdata_kernel + i01*ne00*ne03 + i00*ne03, 0, 1);
                            }
                            dst_معلومات[(i11*stride + i01)*ne0 + i10*stride + i00] += v;
                        }
                    }
                }
            }
        }
    }
}

فراغ ggml_compute_forward_conv_transpose_2d(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F16:
            {
                ggml_compute_forward_conv_transpose_2d_impl<ggml_fp16_t>(params, dst);
            } اكسر;
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_conv_transpose_2d_impl<عائم>(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_conv_2d_dw

بنية ggml_conv_2d_dw_params {
    int64_t channels;
    int64_t batch;
    int64_t src_w;
    int64_t src_h;
    int64_t dst_w;
    int64_t dst_h;
    int64_t knl_w;
    int64_t knl_h;
    صحيح stride_x;
    صحيح stride_y;
    صحيح pad_x;
    صحيح pad_y;
    صحيح dilation_x;
    صحيح dilation_y;
};

ساكن مضمن عائم ggml_conv_2d_dw_knl_f32(ثابت حرف * معلومات, int64_t i, ggml_type type) {
    إذا (type == GGML_TYPE_F16) {
        أرجع GGML_FP16_منزل_FP32(((ثابت ggml_fp16_t *)معلومات)[i]);
    }
    أرجع ((ثابت عائم *)معلومات)[i];
}

ساكن فراغ ggml_compute_forward_conv_2d_dw_cwhn(
        ثابت ggml_compute_params * params,
        ثابت ggml_tensor * src,
        ثابت ggml_tensor * kernel,
        ggml_tensor * dst,
        ثابت ggml_conv_2d_dw_params & p) {

    ثابت int64_t c = p.channels;
    ثابت حرف * knl_معلومات = (ثابت حرف *)kernel->معلومات;
    ثابت ggml_type knl_type = kernel->type;

    ثابت int64_t صفون_ڪل = p.dst_h * p.batch;
    ثابت int64_t صفون_في_thread = (صفون_ڪل + params->nth - 1) / params->nth;
    ثابت int64_t صف_شروعات = params->ith * صفون_في_thread;
    ثابت int64_t صف_پڇاڙي = MIN(صف_شروعات + صفون_في_thread, صفون_ڪل);

#جيڪڏهن_وصف_ٿيل GGML_SIMD
    int64_t c_pkg_پڇاڙي = 0;
    int64_t pkg_سائيز = GGML_F32_EPR;
    إذا (knl_type == GGML_TYPE_F32) {
    #إذا defined(__ARM_FEATURE_SVE)
        pkg_سائيز = svcntw();
    #وإلا
        pkg_سائيز = GGML_F32_EPR;
    #جيڪڏهن_پورو
        c_pkg_پڇاڙي = (c / pkg_سائيز) * pkg_سائيز;
    }
#وإلا
    ثابت int64_t c_pkg_پڇاڙي = 0;
#جيڪڏهن_پورو

    لكل (int64_t صف = صف_شروعات; صف < صف_پڇاڙي; ++صف) {
        ثابت int64_t dst_y = صف % p.dst_h;
        ثابت عائم * src_معلومات = (ثابت عائم *)src->معلومات + (صف / p.dst_h) * p.src_w * p.src_h * c;
        لكل (int64_t dst_x = 0; dst_x < p.dst_w; ++dst_x) {
            عائم * dst_معلومات = (عائم *)dst->معلومات + (صف * p.dst_w + dst_x) * c;
            ثابت int64_t src_y_بنياد = dst_y * p.stride_y - p.pad_y;
            ثابت int64_t src_x_بنياد = dst_x * p.stride_x - p.pad_x;

#جيڪڏهن_وصف_ٿيل GGML_SIMD
            لكل (int64_t c_i = 0; c_i < c_pkg_پڇاڙي; c_i += pkg_سائيز) {
                GGML_F32_VEC جوڙ = GGML_F32_VEC_ZERO;
                لكل (int64_t knl_y = 0; knl_y < p.knl_h; ++knl_y) {
                    ثابت int64_t src_y = src_y_بنياد + knl_y * p.dilation_y;
                    إذا (src_y < 0 || src_y >= p.src_h) {
                        استمر;
                    }
                    لكل (int64_t knl_x = 0; knl_x < p.knl_w; ++knl_x) {
                        ثابت int64_t src_x = src_x_بنياد + knl_x * p.dilation_x;
                        إذا (src_x < 0 || src_x >= p.src_w) {
                            استمر;
                        }
                        ثابت عائم * kp = (ثابت عائم *)knl_معلومات + (knl_y * p.knl_w + knl_x) * c + c_i;
                        GGML_F32_VEC k = GGML_F32_VEC_ڀريو(kp);
                        GGML_F32_VEC s = GGML_F32_VEC_ڀريو(src_معلومات + (src_y * p.src_w + src_x) * c + c_i);
                        جوڙ = GGML_F32_VEC_FMA(جوڙ, k, s);
                    }
                }
                GGML_F32_VEC_STORE(dst_معلومات + c_i, جوڙ);
            }
#جيڪڏهن_پورو
            لكل (int64_t c_i = c_pkg_پڇاڙي; c_i < c; ++c_i) {
                عائم جوڙ = 0.0f;
                لكل (int64_t knl_y = 0; knl_y < p.knl_h; ++knl_y) {
                    ثابت int64_t src_y = src_y_بنياد + knl_y * p.dilation_y;
                    إذا (src_y < 0 || src_y >= p.src_h) {
                        استمر;
                    }
                    لكل (int64_t knl_x = 0; knl_x < p.knl_w; ++knl_x) {
                        ثابت int64_t src_x = src_x_بنياد + knl_x * p.dilation_x;
                        إذا (src_x < 0 || src_x >= p.src_w) {
                            استمر;
                        }
                        جوڙ += ggml_conv_2d_dw_knl_f32(knl_معلومات, (knl_y * p.knl_w + knl_x) * c + c_i, knl_type)
                             * src_معلومات[(src_y * p.src_w + src_x) * c + c_i];
                    }
                }
                dst_معلومات[c_i] = جوڙ;
            }
        }
    }
}

ساكن فراغ ggml_compute_forward_conv_2d_dw_whcn(
        ثابت ggml_compute_params * params,
        ثابت ggml_tensor * src,
        ثابت ggml_tensor * kernel,
        ggml_tensor * dst,
        ثابت ggml_conv_2d_dw_params & p) {

    ثابت int64_t n = p.channels * p.batch;
    ثابت int64_t في_thread = (n + params->nth - 1) / params->nth;
    ثابت int64_t start = params->ith * في_thread;
    ثابت int64_t end = MIN(start + في_thread, n);
    ثابت حرف * knl_بنياد = (ثابت حرف *)kernel->معلومات;
    ثابت ggml_type knl_type = kernel->type;

    لكل (int64_t i = start; i < end; ++i) {
        ثابت int64_t knl_وٿي = (i % p.channels) * p.knl_w * p.knl_h;
        ثابت عائم * src_معلومات = (ثابت عائم *)src->معلومات + i * p.src_w * p.src_h;
        عائم * dst_معلومات = (عائم *)dst->معلومات + i * p.dst_w * p.dst_h;

        لكل (int64_t dst_y = 0; dst_y < p.dst_h; ++dst_y) {
            لكل (int64_t dst_x = 0; dst_x < p.dst_w; ++dst_x) {

                عائم جوڙ = 0.0f;
                لكل (int64_t knl_y = 0; knl_y < p.knl_h; ++knl_y) {
                    ثابت int64_t src_y = dst_y * p.stride_y + knl_y * p.dilation_y - p.pad_y;
                    إذا (src_y < 0 || src_y >= p.src_h) {
                        استمر;
                    }
                    لكل (int64_t knl_x = 0; knl_x < p.knl_w; ++knl_x) {
                        ثابت int64_t src_x = dst_x * p.stride_x + knl_x * p.dilation_x - p.pad_x;
                        إذا (src_x < 0 || src_x >= p.src_w) {
                            استمر;
                        }
                        جوڙ += ggml_conv_2d_dw_knl_f32(knl_بنياد, knl_وٿي + knl_y * p.knl_w + knl_x, knl_type)
                             * src_معلومات[src_y * p.src_w + src_x];
                    }
                }
                dst_معلومات[dst_y * p.dst_w + dst_x] = جوڙ;
            }
        }
    }
}

فراغ ggml_compute_forward_conv_2d_dw(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * kernel = dst->src[0];
    ثابت ggml_tensor * src = dst->src[1];
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

    إذا (ggml_is_contiguous(src)) {
        ggml_compute_forward_conv_2d_dw_whcn(params, src, kernel, dst, p);
    } وإلا إذا (ggml_is_contiguous_channels(src)) {
        GGML_ASSERT(kernel->nb[0] >= kernel->nb[2] && kernel->nb[1] >= kernel->nb[0]);
        ggml_compute_forward_conv_2d_dw_cwhn(params, src, kernel, dst, p);
    } وإلا {
        GGML_ABORT("non-contiguous memory layout not supported");
    }
}

// ggml_compute_forward_pool_1d_ksp
ساكن فراغ ggml_compute_forward_pool_1d_ksp(
        ثابت ggml_compute_params * params,
        ثابت ggml_op_pool op,
        ثابت صحيح k,
        ثابت صحيح s,
        ثابت صحيح p,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src = dst->src[0];

    أكد(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16);

    إذا (params->ith != 0) {
        أرجع;
    }

    ثابت int64_t IW = src->ne[0];
    ثابت int64_t OW = dst->ne[0];

    ثابت int64_t nr = ggml_nrows(src);

    لكل (int64_t ir = 0; ir < nr; ++ir) {
        ثابت حرف * srow_بائيٽون =            (ثابت حرف *) src->معلومات + ir * src->nb[1];
        عائم      * drow       = (عائم *) ((      حرف *) dst->معلومات + ir * dst->nb[1]);

        لكل (int64_t ow = 0; ow < OW; ++ow) {
            عائم res = 0;
            بدل (op) {
                حالة GGML_OP_POOL_AVG: res = 0.0f;     اكسر;
                حالة GGML_OP_POOL_MAX: res = -FLT_MAX; اكسر;
                حالة GGML_OP_POOL_ڳڻپ: GGML_ABORT("fatal error");
            }

            صحيح count = 0;
            ثابت صحيح بنياد = (صحيح) ow * s - p;

            لكل (صحيح ki = 0; ki < k; ++ki) {
                ثابت صحيح j = بنياد + ki;
                إذا (j < 0 || j >= (صحيح) IW) {
                    استمر;
                }

                عائم v;
                إذا (src->type == GGML_TYPE_F32) {
                    v = ((ثابت عائم *) srow_بائيٽون)[j];
                } وإلا {
                    v = GGML_CPU_FP16_منزل_FP32(((ثابت ggml_fp16_t *) srow_بائيٽون)[j]);
                }

                بدل (op) {
                    حالة GGML_OP_POOL_AVG: res += v;                اكسر;
                    حالة GGML_OP_POOL_MAX: res =  std::max(v, res); اكسر;
                    حالة GGML_OP_POOL_ڳڻپ: GGML_ABORT("fatal error");
                }

                ++count;
            }

            بدل (op) {
                حالة GGML_OP_POOL_AVG: res = (count > 0) ? (res / count) : 0.0f; اكسر;
                حالة GGML_OP_POOL_MAX:                                           اكسر;
                حالة GGML_OP_POOL_ڳڻپ: GGML_ABORT("fatal error");
            }

            drow[ow] = res;
        }
    }
}

// ggml_compute_forward_pool_1d

فراغ ggml_compute_forward_pool_1d(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت int32_t * opts = (ثابت int32_t *)dst->op_params;
    ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    ثابت صحيح k0 = opts[1];
    ثابت صحيح s0 = opts[2];
    ثابت صحيح p0 = opts[3];

    ggml_compute_forward_pool_1d_ksp(params, op, k0, s0, p0, dst);
}

// ggml_compute_forward_pool_2d

فراغ ggml_compute_forward_pool_2d(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src = dst->src[0];

    أكد(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16);

    إذا (params->ith != 0) {
        أرجع;
    }

    ثابت int32_t * opts = (ثابت int32_t *)dst->op_params;

    ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    ثابت صحيح k0 = opts[1];
    ثابت صحيح k1 = opts[2];
    ثابت صحيح s0 = opts[3];
    ثابت صحيح s1 = opts[4];
    ثابت صحيح p0 = opts[5];
    ثابت صحيح p1 = opts[6];
    ثابت حرف * cdata = (ثابت حرف*)src->معلومات;
    ثابت حرف * ثابت معلومات_پڇاڙي = cdata + ggml_nbytes(src);

    ثابت int64_t px = dst->ne[0];
    ثابت int64_t py = dst->ne[1];
    ثابت int64_t pa = px * py;

    عائم * dplane = (عائم *)dst->معلومات;

    ثابت صحيح ka = k0 * k1;
    ثابت صحيح offset0 = -p0;
    ثابت صحيح offset1 = -p1;

    طالما (cdata < معلومات_پڇاڙي) {
        لكل (صحيح oy = 0; oy < py; ++oy) {
            عائم * ثابت drow = dplane + oy * px;
            عائم * ثابت ٻاهريون  = drow;

            لكل (صحيح ox = 0; ox < px; ++ox) {
                عائم res = 0;
                بدل (op) {
                    حالة GGML_OP_POOL_AVG: res = 0;        اكسر;
                    حالة GGML_OP_POOL_MAX: res = -FLT_MAX; اكسر;
                    حالة GGML_OP_POOL_ڳڻپ: GGML_ABORT("fatal error");
                }

                ثابت صحيح ix = offset0 + ox * s0;
                ثابت صحيح iy = offset1 + oy * s1;

                لكل (صحيح ky = 0; ky < k1; ++ky) {
                    إذا (iy + ky < 0 || iy + ky >= src->ne[1]) {
                        استمر;
                    }

                    ثابت فراغ * srow = (ثابت فراغ *)(cdata + src->nb[1] * (iy + ky));
                    لكل (صحيح kx = 0; kx < k0; ++kx) {
                        صحيح j = ix + kx;
                        إذا (j < 0 || j >= src->ne[0]) {
                            استمر;
                        }

                        ثابت عائم srow_j = (src->type == GGML_TYPE_F32) ? ((ثابت عائم*)srow)[j] : GGML_CPU_FP16_منزل_FP32(((ثابت ggml_fp16_t*)srow)[j]);
                        بدل (op) {
                            حالة GGML_OP_POOL_AVG: res += srow_j;                اكسر;
                            حالة GGML_OP_POOL_MAX: res =  std::max(srow_j, res); اكسر;
                            حالة GGML_OP_POOL_ڳڻپ:               GGML_ABORT("fatal error");
                        }
                    }
                }
                بدل (op) {
                    حالة GGML_OP_POOL_AVG:           res /= ka; اكسر;
                    حالة GGML_OP_POOL_MAX:                      اكسر;
                    حالة GGML_OP_POOL_ڳڻپ: GGML_ABORT("fatal error");
                }

                ٻاهريون[ox] = res;
            }
        }

        cdata  += src->nb[2];
        dplane += pa;
    }
}

// ggml_compute_forward_pool_2d_back

فراغ ggml_compute_forward_pool_2d_پٺ(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src  = dst->src[0];
    ثابت ggml_tensor * dstf = dst->src[1]; // forward tensor of dst

    أكد(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);

    إذا (params->ith != 0) {
        أرجع;
    }

    ثابت int32_t * opts = (ثابت int32_t *)dst->op_params;
    ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    ثابت صحيح k0 = opts[1];
    ثابت صحيح k1 = opts[2];
    ثابت صحيح s0 = opts[3];
    ثابت صحيح s1 = opts[4];
    ثابت صحيح p0 = opts[5];
    ثابت صحيح p1 = opts[6];

    حرف       * cdata  = (حرف       *) dst->معلومات;
    ثابت حرف * cdataf = (ثابت حرف *) dstf->معلومات;
    ثابت حرف * ثابت معلومات_پڇاڙي = cdata + ggml_nbytes(dst);

    GGML_ASSERT(params->ith == 0);
    memset(cdata, 0, ggml_nbytes(dst));

    ثابت int64_t px = src->ne[0];
    ثابت int64_t py = src->ne[1];
    ثابت int64_t pa = px * py;

    ثابت عائم * splane = (ثابت عائم *) src->معلومات;

    ثابت صحيح ka = k0 * k1;
    ثابت صحيح offset0 = -p0;
    ثابت صحيح offset1 = -p1;

    طالما (cdata < معلومات_پڇاڙي) {
        لكل (صحيح oy = 0; oy < py; ++oy) {
            ثابت عائم * ثابت srow = splane + oy * px;
            لكل (صحيح ox = 0; ox < px; ++ox) {
                ثابت عائم grad0 = srow[ox];

                ثابت صحيح ix = offset0 + ox * s0;
                ثابت صحيح iy = offset1 + oy * s1;

                إذا (op == GGML_OP_POOL_MAX) {
                    عائم maxval = -FLT_MAX;
                    صحيح kxmax = -1;
                    صحيح kymax = -1;

                    لكل (صحيح ky = 0; ky < k1; ++ky) {
                        إذا (iy + ky < 0 || iy + ky >= dst->ne[1]) {
                            استمر;
                        }
                        ثابت فراغ * drowf = (ثابت فراغ *)(cdataf + dst->nb[1] * (iy + ky));
                        لكل (صحيح kx = 0; kx < k0; ++kx) {
                            صحيح j = ix + kx;
                            إذا (j < 0 || j >= dst->ne[0]) {
                                استمر;
                            }

                            ثابت عائم val = dst->type == GGML_TYPE_F32 ?
                                ((ثابت عائم *) drowf)[j] : GGML_CPU_FP16_منزل_FP32(((ثابت ggml_fp16_t *) drowf)[j]);
                            إذا (val <= maxval) {
                                استمر;
                            }

                            maxval = val;
                            kxmax = kx;
                            kymax = ky;
                        }
                    }

                    إذا (kxmax == -1 || kymax == -1) {
                        استمر;
                    }

                    فراغ * drow = (فراغ *)(cdata + dst->nb[1] * (iy + kymax));
                    ثابت صحيح j = ix + kxmax;
                    إذا (dst->type == GGML_TYPE_F32) {
                        ((عائم *) drow)[j] += grad0;
                    } وإلا {
                        ((ggml_fp16_t *) drow)[j] = GGML_CPU_FP32_منزل_FP16(grad0 + GGML_CPU_FP16_منزل_FP32(((ثابت ggml_fp16_t *) drow)[j]));
                    }
                } وإلا إذا (op == GGML_OP_POOL_AVG) {
                    ثابت عائم grad = grad0 / ka;

                    لكل (صحيح ky = 0; ky < k1; ++ky) {
                        إذا (iy + ky < 0 || iy + ky >= dst->ne[1]) {
                            استمر;
                        }
                        فراغ * drow = (فراغ *)(cdata + dst->nb[1] * (iy + ky));
                        لكل (صحيح kx = 0; kx < k0; ++kx) {
                            صحيح j = ix + kx;
                            إذا (j < 0 || j >= dst->ne[0]) {
                                استمر;
                            }

                            إذا (dst->type == GGML_TYPE_F32) {
                                ((عائم *) drow)[j] += grad;
                            } وإلا {
                                ((ggml_fp16_t *) drow)[j] += GGML_CPU_FP32_منزل_FP16(grad);
                            }
                        }
                    }
                } وإلا {
                    GGML_ASSERT(خطأ);
                }
            }
        }

        cdata  += dst->nb[2];
        cdataf += dst->nb[2];
        splane += pa;
    }
}

// ggml_compute_forward_upscale

ساكن فراغ ggml_compute_forward_upscale_f32(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    عائم sf0 = (عائم)ne0/src0->ne[0];
    عائم sf1 = (عائم)ne1/src0->ne[1];
    عائم sf2 = (عائم)ne2/src0->ne[2];
    عائم sf3 = (عائم)ne3/src0->ne[3];
    عائم pixel_وٿي = 0.5f;

    ثابت int32_t گهڻائي_flags = ggml_get_op_params_i32(dst, 0);
    ثابت ggml_scale_گهڻائي mode = (ggml_scale_گهڻائي) (گهڻائي_flags & 0xFF);

    إذا (گهڻائي_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        pixel_وٿي = 0.0f;
        sf0 = ne0 > 1 && ne00 > 1 ? (عائم)(ne0 - 1) / (ne00 - 1) : sf0;
        sf1 = ne1 > 1 && ne01 > 1 ? (عائم)(ne1 - 1) / (ne01 - 1) : sf1;
    }

    إذا (mode == GGML_SCALE_گهڻائي_سڀ_کان_ويجهو) {
        لكل (int64_t i3 = 0; i3 < ne3; i3++) {
            ثابت int64_t i03 = i3 / sf3;
            لكل (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                ثابت int64_t i02 = i2 / sf2;
                لكل (int64_t i1 = 0; i1 < ne1; i1++) {
                    ثابت int64_t i01 = i1 / sf1;
                    لكل (int64_t i0 = 0; i0 < ne0; i0++) {
                        ثابت int64_t i00 = i0 / sf0;

                        ثابت عائم * x = (عائم *)((حرف *) src0->معلومات + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              عائم * y = (عائم *)((حرف *)  dst->معلومات +  i0*nb0  +  i1*nb1  +  i2*nb2  +  i3*nb3);

                        *y = *x;
                    }
                }
            }
        }
    } وإلا إذا (mode == GGML_SCALE_گهڻائي_BILINEAR && (گهڻائي_flags & GGML_SCALE_FLAG_ANTIALIAS)) {
        // Similar to F.interpolate(..., mode="bilinear", align_corners=False, antialias=True)
        // https://github.com/pytorch/pytorch/blob/8871ff29b743948d1225389d5b7068f37b22750b/aten/src/ATen/native/cpu/UpSampleKernel.cpp
        خودڪار triangle_filter = [](عائم x) -> عائم {
            أرجع std::max(1.0f - fabsf(x), 0.0f);
        };

        // support and invscale, minimum 1 pixel for bilinear
        ثابت عائم support1  = std::max(1.0f, 1.0f / sf1);
        ثابت عائم invscale1 = 1.0f / support1;
        ثابت عائم support0  = std::max(1.0f, 1.0f / sf0);
        ثابت عائم invscale0 = 1.0f / support0;

        لكل (int64_t i3 = 0; i3 < ne3; i3++) {
            ثابت int64_t i03 = i3 / sf3;
            لكل (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                ثابت int64_t i02 = i2 / sf2;
                لكل (int64_t i1 = 0; i1 < ne1; i1++) {
                    ثابت عائم y = ((عائم) i1 + pixel_وٿي) / sf1;
                    لكل (int64_t i0 = 0; i0 < ne0; i0++) {
                        ثابت عائم x = ((عائم) i0 + pixel_وٿي) / sf0;

                        // the range of source pixels that contribute
                        ثابت int64_t x_min = std::max<int64_t>(x - support0 + pixel_وٿي, 0);
                        ثابت int64_t x_max = std::min<int64_t>(x + support0 + pixel_وٿي, ne00);
                        ثابت int64_t y_min = std::max<int64_t>(y - support1 + pixel_وٿي, 0);
                        ثابت int64_t y_max = std::min<int64_t>(y + support1 + pixel_وٿي, ne01);

                        // bilinear filter with antialiasing
                        عائم val = 0.0f;
                        عائم ڪل_وزن = 0.0f;

                        لكل (int64_t sy = y_min; sy < y_max; sy++) {
                            ثابت عائم وزن_y = triangle_filter((sy - y + pixel_وٿي) * invscale1);

                            لكل (int64_t sx = x_min; sx < x_max; sx++) {
                                ثابت عائم وزن_x = triangle_filter((sx - x + pixel_وٿي) * invscale0);
                                ثابت عائم وزن = وزن_x * وزن_y;

                                إذا (وزن <= 0.0f) {
                                    استمر;
                                }

                                ثابت عائم pixel = *(ثابت عائم *)((ثابت حرف *)src0->معلومات + sx*nb00 + sy*nb01 + i02*nb02 + i03*nb03);
                                val += pixel * وزن;
                                ڪل_وزن += وزن;
                            }
                        }

                        إذا (ڪل_وزن > 0.0f) {
                            val /= ڪل_وزن;
                        }

                        عائم * dst_ptr = (عائم *)((حرف *)dst->معلومات + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);
                        *dst_ptr = val;
                    }
                }
            }
        }
    } وإلا إذا (mode == GGML_SCALE_گهڻائي_BILINEAR) {
        لكل (int64_t i3 = 0; i3 < ne3; i3++) {
            ثابت int64_t i03 = i3 / sf3;
            لكل (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                ثابت int64_t i02 = i2 / sf2;
                لكل (int64_t i1 = 0; i1 < ne1; i1++) {
                    ثابت عائم y = ((عائم)i1 + pixel_وٿي) / sf1 - pixel_وٿي;
                    int64_t y0 = (int64_t)floorf(y);
                    int64_t y1 = y0 + 1;

                    y0 = std::max(int64_t(0), std::min(y0, ne01 - 1));
                    y1 = std::max(int64_t(0), std::min(y1, ne01 - 1));

                    عائم dy = y - (عائم)y0;
                    dy = std::max(0.0f, std::min(dy, 1.0f));

                    لكل (int64_t i0 = 0; i0 < ne0; i0++) {
                        ثابت عائم x = ((عائم)i0 + pixel_وٿي) / sf0 - pixel_وٿي;
                        int64_t x0 = (int64_t)floorf(x);
                        int64_t x1 = x0 + 1;

                        x0 = std::max(int64_t(0), std::min(x0, ne00 - 1));
                        x1 = std::max(int64_t(0), std::min(x1, ne00 - 1));

                        عائم dx = x - (عائم)x0;
                        dx = std::max(0.0f, std::min(dx, 1.0f));

                        // fetch the four surrounding pixel values and interpolate
                        ثابت عائم a = *(ثابت عائم *)((ثابت حرف *)src0->معلومات + x0*nb00 + y0*nb01 + i02*nb02 + i03*nb03);
                        ثابت عائم b = *(ثابت عائم *)((ثابت حرف *)src0->معلومات + x1*nb00 + y0*nb01 + i02*nb02 + i03*nb03);
                        ثابت عائم c = *(ثابت عائم *)((ثابت حرف *)src0->معلومات + x0*nb00 + y1*nb01 + i02*nb02 + i03*nb03);
                        ثابت عائم d = *(ثابت عائم *)((ثابت حرف *)src0->معلومات + x1*nb00 + y1*nb01 + i02*nb02 + i03*nb03);

                        ثابت عائم val = a*(1 - dx)*(1 - dy) + b*dx*(1 - dy) + c*(1 - dx)*dy + d*dx*dy;

                        عائم * y_dst = (عائم *)((حرف *)dst->معلومات + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);
                        *y_dst = val;
                    }
                }
            }
        }
    } وإلا إذا (mode == GGML_SCALE_گهڻائي_BICUBIC) {
        // https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
        ثابت عائم a = -0.75f; // use alpha = -0.75 (same as PyTorch)
        خودڪار weight1 = [a](عائم x) { أرجع ((a + 2) * x - (a + 3)) * x * x + 1; };
        خودڪار weight2 = [a](عائم x) { أرجع ((a * x - 5 * a) * x + 8 * a) * x - 4 * a; };
        خودڪار bicubic = [=](عائم p0, عائم p1, عائم p2, عائم p3, عائم x) {
            ثابت عائم w0 = weight2(x + 1);
            ثابت عائم w1 = weight1(x + 0);
            ثابت عائم w2 = weight1(1 - x);
            ثابت عائم w3 = weight2(2 - x);
            أرجع p0*w0 + p1*w1 + p2*w2 + p3*w3;
        };

        لكل (int64_t i3 = 0; i3 < ne3; i3++) {
            ثابت int64_t i03 = i3 / sf3;
            لكل (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                ثابت int64_t i02 = i2 / sf2;
                لكل (int64_t i1 = 0; i1 < ne1; i1++) {
                    ثابت عائم y = ((عائم)i1 + pixel_وٿي) / sf1 - pixel_وٿي;
                    ثابت int64_t y0 = (int64_t)floorf(y);
                    ثابت عائم dy = y - (عائم)y0;

                    لكل (int64_t i0 = 0; i0 < ne0; i0++) {
                        ثابت عائم x = ((عائم)i0 + pixel_وٿي) / sf0 - pixel_وٿي;
                        ثابت int64_t x0 = (int64_t)floorf(x);
                        ثابت عائم dx = x - (عائم)x0;

                        خودڪار p = [=](int64_t x_off, int64_t y_off) -> عائم {
                            int64_t i00 = std::max(int64_t(0), std::min(x0 + x_off, ne00 - 1));
                            int64_t i01 = std::max(int64_t(0), std::min(y0 + y_off, ne01 - 1));
                            أرجع *(ثابت عائم *)((ثابت حرف *)src0->معلومات + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                        };

                        ثابت عائم val = bicubic(
                            bicubic(p(-1,-1), p(0,-1), p(1,-1), p(2,-1), dx),
                            bicubic(p(-1, 0), p(0, 0), p(1, 0), p(2, 0), dx),
                            bicubic(p(-1, 1), p(0, 1), p(1, 1), p(2, 1), dx),
                            bicubic(p(-1, 2), p(0, 2), p(1, 2), p(2, 2), dx), dy);

                        عائم * y_dst = (عائم *)((حرف *)dst->معلومات + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);
                        *y_dst = val;
                    }
                }
            }
        }
    } وإلا {
        GGML_ABORT("unsupported upscale mode");
    }
}

فراغ ggml_compute_forward_upscale(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_upscale_f32(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}


// ggml_compute_forward_pad

سانچو<منطقي circular_t>
ساكن فراغ ggml_compute_forward_pad_f32(
    ثابت ggml_compute_params * params,
          ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    أكد(dst->nb[0] == حجم(عائم));

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    عائم * dst_ptr = (عائم *) dst->معلومات;
    ثابت int32_t lp0 = ggml_get_op_params_i32(dst, 0);
    ثابت int32_t rp0 = ggml_get_op_params_i32(dst, 1);
    ثابت int32_t lp1 = ggml_get_op_params_i32(dst, 2);
    ثابت int32_t rp1 = ggml_get_op_params_i32(dst, 3);
    ثابت int32_t lp2 = ggml_get_op_params_i32(dst, 4);
    ثابت int32_t rp2 = ggml_get_op_params_i32(dst, 5);
    ثابت int32_t lp3 = ggml_get_op_params_i32(dst, 6);
    ثابت int32_t rp3 = ggml_get_op_params_i32(dst, 7);

    ثابت int64_t nrows = ne1 * ne2 * ne3;
    ثابت منطقي src_صف_contig = src0->nb[0] == حجم(عائم);

    لكل (int64_t صف = ith; صف < nrows; صف += nth) {
        ثابت int64_t i1 = صف % ne1;
        ثابت int64_t i2 = (صف / ne1) % ne2;
        ثابت int64_t i3 = صف / (ne1 * ne2);

        عائم * dst_صف = dst_ptr + صف * ne0;

        إذا constexpr (circular_t) {
            // circular padding wraps around on a torus
            ثابت int64_t src_i1 = ggml_wrap_around(i1 - lp1, ne01);
            ثابت int64_t src_i2 = ggml_wrap_around(i2 - lp2, ne02);
            ثابت int64_t src_i3 = ggml_wrap_around(i3 - lp3, ne03);

            لكل (int64_t i0 = 0; i0 < ne0; ++i0) {
                ثابت int64_t src_i0 = ggml_wrap_around(i0 - lp0, ne00);
                dst_صف[i0] = *(ثابت عائم *)((ثابت حرف *) src0->معلومات + src_i3*nb03 + src_i2*nb02 + src_i1*nb01 + src_i0*nb00);
            }
        } وإلا {
            ثابت منطقي interior = (i1 >= lp1 && i1 < ne1 - rp1)
                               && (i2 >= lp2 && i2 < ne2 - rp2)
                               && (i3 >= lp3 && i3 < ne3 - rp3);
            إذا (!interior) {
                memset(dst_صف, 0, ne0 * حجم(عائم));
                استمر;
            }

            ثابت حرف * src_صف = (ثابت حرف *) src0->معلومات + (i3 - lp3)*nb03 + (i2 - lp2)*nb02 + (i1 - lp1)*nb01;

            إذا (src_صف_contig && lp0 >= 0 && rp0 >= 0) {
                memset(dst_صف, 0, lp0 * حجم(عائم));
                يادگيري_نقل(dst_صف + lp0, src_صف, ne00 * حجم(عائم));
                memset(dst_صف + lp0 + ne00, 0, rp0 * حجم(عائم));
            } وإلا {
                لكل (int64_t i0 = 0; i0 < ne0; ++i0) {
                    إذا (i0 >= lp0 && i0 < ne0 - rp0) {
                        dst_صف[i0] = *(ثابت عائم *)(src_صف + (i0 - lp0)*nb00);
                    } وإلا {
                        dst_صف[i0] = 0.0f;
                    }
                }
            }
        }
    }
}


فراغ ggml_compute_forward_pad(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت منطقي circular = (منطقي) ggml_get_op_params_i32(dst, 8);
    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                إذا (circular) {
                    ggml_compute_forward_pad_f32<سچ>(params, dst);
                } وإلا {
                    ggml_compute_forward_pad_f32<خطأ>(params, dst);
                }
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_pad_reflect_1d

فراغ ggml_compute_forward_pad_reflect_1d(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت int32_t * opts = (ثابت int32_t *) dst->op_params;
    ثابت صحيح p0 = opts[0];
    ثابت صحيح p1 = opts[1];

    GGML_TENSOR_UNARY_OP_LOCALS

    لكل (int64_t i3 = 0; i3 < ne3; i3++) {
        لكل (int64_t i2 = 0; i2 < ne2; i2++) {
            لكل (int64_t i1 = ith; i1 < ne1; i1 += nth) {
                عائم * کاٻي  = (عائم *) ((حرف *) dst->معلومات + i3*nb3 + i2*nb2 + i1*nb1 +         p0*nb0);
                عائم * ساڄي = (عائم *) ((حرف *) dst->معلومات + i3*nb3 + i2*nb2 + i1*nb1 + (ne0-p1-1)*nb0);

                ggml_vec_cpy_f32(ne00, کاٻي, (عائم *) ((حرف *) src0->معلومات + i3*nb03 + i2*nb02 + i1*nb01));

                لكل (صحيح i0 = 1; i0 <= p0; i0++) { کاٻي[-i0] = کاٻي[i0];   }
                لكل (صحيح i0 = 1; i0 <= p1; i0++) { ساڄي[i0] = ساڄي[-i0]; }
            }
        }
    }
}

// ggml_compute_forward_roll

ساكن int64_t ggml_wrap_اشاريو(int64_t i, int64_t ne) {
    إذا (i < 0) {
        أرجع i + ne;
    } وإلا إذا (i >= ne) {
        أرجع i - ne;
    }
    أرجع i;
}

ساكن فراغ ggml_compute_forward_roll_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت عائم * src_معلومات = (ثابت عائم *) src0->معلومات;
    عائم * dst_معلومات = (عائم *) dst->معلومات;

    GGML_TENSOR_UNARY_OP_LOCALS

    ثابت صحيح s0 = ggml_get_op_params_i32(dst, 0);
    ثابت صحيح s1 = ggml_get_op_params_i32(dst, 1);
    ثابت صحيح s2 = ggml_get_op_params_i32(dst, 2);
    ثابت صحيح s3 = ggml_get_op_params_i32(dst, 3);

    ثابت int64_t ڪل = ne1 * ne2 * ne3;
    ثابت int64_t في_thread = (ڪل + params->nth) / params->nth;
    ثابت int64_t start = params->ith * في_thread;
    ثابت int64_t end   = std::min(start + في_thread, ڪل);

    لكل (int64_t i = start; i < end; ++i) {
        ثابت int64_t i1 = i % ne1;
        ثابت int64_t i2 = (i / ne1) % ne2;
        ثابت int64_t i3 = i / (ne2 * ne1);
        عائم * dst_صف = dst_معلومات + (i3*nb3 + i2*nb2 + i1*nb1) / حجم(عائم);

        ثابت int64_t i01 = ggml_wrap_اشاريو(i1 - s1, ne01);
        ثابت int64_t i02 = ggml_wrap_اشاريو(i2 - s2, ne02);
        ثابت int64_t i03 = ggml_wrap_اشاريو(i3 - s3, ne03);
        ثابت عائم * src_صف = src_معلومات + (i03*nb03 + i02*nb02 + i01*nb01) / حجم(عائم);

        ثابت int64_t s = ggml_wrap_اشاريو(-s0, ne00);
        ثابت int64_t n = ne00 - s;
        ggml_vec_cpy_f32(n, dst_صف,     src_صف + s);
        ggml_vec_cpy_f32(s, dst_صف + n, src_صف);
    }
}

فراغ ggml_compute_forward_roll(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_roll_f32(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_arange

ساكن فراغ ggml_compute_forward_arange_f32(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    GGML_ASSERT(dst->nb[0] == حجم(عائم));

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت عائم start = ggml_get_op_params_f32(dst, 0);
    ثابت عائم stop  = ggml_get_op_params_f32(dst, 1);
    ثابت عائم قدم  = ggml_get_op_params_f32(dst, 2);

    ثابت int64_t قدمن = (int64_t) ceilf((stop - start) / قدم);

    GGML_ASSERT(ggml_nelements(dst) == قدمن);

    لكل (int64_t i = ith; i < قدمن; i+= nth) {
        عائم value = start + قدم * i;
        ((عائم *)dst->معلومات)[i] = value;
    }
}

فراغ ggml_compute_forward_arange(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {
    بدل (dst->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_arange_f32(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ساكن فراغ ggml_compute_forward_timestep_embedding_f32(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->nb[0] == حجم(عائم));

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    ثابت صحيح dim = ggml_get_op_params_i32(dst, 0);
    ثابت صحيح max_period = ggml_get_op_params_i32(dst, 1);

    صحيح half = dim / 2;

    لكل (int64_t i = 0; i < ne00; i++) {
        عائم * embed_معلومات = (عائم *)((حرف *)  dst->معلومات +  i*nb1);
        لكل (int64_t j = ith; j < half; j += nth) {
            عائم timestep = ((عائم *)src0->معلومات)[i];
            عائم freq = (عائم)expf(-logf(max_period) * j / half);
            عائم arg = timestep * freq;
            embed_معلومات[j] = cosf(arg);
            embed_معلومات[j + half] = sinf(arg);
        }
        إذا (dim % 2 != 0 && ith == 0) {
            embed_معلومات[2 * half] = 0.f;
        }
    }
}

فراغ ggml_compute_forward_timestep_embedding(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_timestep_embedding_f32(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_argsort

سانچو<تعداد ggml_ترتيب_ڏيو_ترتيب ترتيب>
بنية cmp_argsort {
    ثابت عائم * معلومات;
    منطقي عامل()(int32_t a, int32_t b) ثابت {
        إذا constexpr (ترتيب == GGML_ترتيب_ڏيو_ترتيب_ASC) {
            أرجع معلومات[a] < معلومات[b];
        } وإلا {
            أرجع معلومات[a] > معلومات[b];
        }
    }
};

ساكن فراغ ggml_compute_forward_argsort_f32(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(nb0 == حجم(عائم));

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت int64_t nr = ggml_nrows(src0);

    ggml_ترتيب_ڏيو_ترتيب ترتيب = (ggml_ترتيب_ڏيو_ترتيب) ggml_get_op_params_i32(dst, 0);

    لكل (int64_t i = ith; i < nr; i += nth) {
        ثابت عائم * src_معلومات = (عائم *)((حرف *) src0->معلومات + i*nb01);

        int32_t * dst_معلومات = (int32_t *)((حرف *) dst->معلومات + i*nb1);

        لكل (int64_t j = 0; j < ne0; j++) {
            dst_معلومات[j] = j;
        }

        بدل (ترتيب) {
            حالة GGML_ترتيب_ڏيو_ترتيب_ASC:
                std::sort(dst_معلومات, dst_معلومات + ne0, cmp_argsort<GGML_ترتيب_ڏيو_ترتيب_ASC>{src_معلومات});
                اكسر;

            حالة GGML_ترتيب_ڏيو_ترتيب_DESC:
                std::sort(dst_معلومات, dst_معلومات + ne0, cmp_argsort<GGML_ترتيب_ڏيو_ترتيب_DESC>{src_معلومات});
                اكسر;

            افتراضي:
                GGML_ABORT("invalid sort order");
        }
    }
}

فراغ ggml_compute_forward_argsort(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_argsort_f32(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_top_k

بنية cmp_چوٽي_k {
    ثابت عائم * معلومات;
    منطقي عامل()(int32_t a, int32_t b) ثابت {
        أرجع معلومات[a] > معلومات[b];
    }
};

ساكن فراغ ggml_compute_forward_چوٽي_k_f32(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(nb0 == حجم(عائم));

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت int64_t nr = ggml_nrows(src0);

    ثابت صحيح چوٽي_k = ne0;

    int32_t * tmp = (int32_t *) params->wdata + (ne00 + CACHE_سٽ_سائيز_F32) * ith;

    لكل (int64_t i = ith; i < nr; i += nth) {
        ثابت عائم * src_معلومات = (عائم *)((حرف *) src0->معلومات + i*nb01);

        لكل (int64_t j = 0; j < ne00; j++) {
            tmp[j] = j;
        }

        std::partial_sort(tmp, tmp + چوٽي_k, tmp + ne00, cmp_چوٽي_k{src_معلومات});

        int32_t * dst_معلومات = (int32_t *)((حرف *) dst->معلومات + i*nb1);

        std::copy(tmp, tmp + چوٽي_k, dst_معلومات);

        // emphasize that the order is not important
        إذا (چوٽي_k > 1) {
            std::swap(dst_معلومات[0], dst_معلومات[1]);
        }
    }
}

فراغ ggml_compute_forward_چوٽي_k(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_چوٽي_k_f32(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ساكن فراغ ggml_compute_forward_flash_attn_ext_f16_one_chunk(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst,
        صحيح ir0, صحيح ir1,
        int64_t ic_شروعات, int64_t ic_پڇاڙي,
        عائم * partials, int64_t partial_stride) {

    ثابت منطقي write_partials = (partials != خالي_اشارو);
    ثابت ggml_tensor * q     = dst->src[0];
    ثابت ggml_tensor * k     = dst->src[1];
    ثابت ggml_tensor * v     = dst->src[2];
    ثابت ggml_tensor * نقاب  = dst->src[3];
    ثابت ggml_tensor * sinks = dst->src[4];

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    ثابت int64_t DK = nek0;
    ثابت int64_t DV = nev0;
    ثابت int64_t N  = neq1;

    GGML_ASSERT(ne0 == DV);
    GGML_ASSERT(ne2 == N);

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_سائيز(q->type));
    GGML_ASSERT(nbk0 == ggml_type_سائيز(k->type));
    GGML_ASSERT(nbv0 == ggml_type_سائيز(v->type));

    GGML_ASSERT(neq0 == DK);
    GGML_ASSERT(nek0 == DK);
    GGML_ASSERT(nev0 == DV);

    GGML_ASSERT(neq1 == N);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == حجم(عائم));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    ثابت int64_t rk2 = neq2/nek2;
    ثابت int64_t rk3 = neq3/nek3;

    ثابت int64_t rv2 = neq2/nev2;
    ثابت int64_t rv3 = neq3/nev3;

    // parallelize by q rows using ggml_vec_dot_f32

    عائم scale         = 1.0f;
    عائم max_bias      = 0.0f;
    عائم logit_softcap = 0.0f;

    يادگيري_نقل(&scale,         (عائم *) dst->op_params + 0, حجم(عائم));
    يادگيري_نقل(&max_bias,      (عائم *) dst->op_params + 1, حجم(عائم));
    يادگيري_نقل(&logit_softcap, (عائم *) dst->op_params + 2, حجم(عائم));

    إذا (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    ثابت uint32_t n_سر      = neq2;
    ثابت uint32_t n_سر_log2 = 1u << (uint32_t) floor(log2(n_سر));

    ثابت عائم m0 = powf(2.0f, -(max_bias       ) / n_سر_log2);
    ثابت عائم m1 = powf(2.0f, -(max_bias / 2.0f) / n_سر_log2);

    ggml_type         ثابت k_vec_dot_type = ggml_get_type_traits_cpu(k->type)->vec_dot_type;
    ggml_from_float_t ثابت q_منزل_vec_dot   = ggml_get_type_traits_cpu(k_vec_dot_type)->from_float;
    ggml_vec_dot_t    ثابت kq_vec_dot     = ggml_get_type_traits_cpu(k->type)->vec_dot;
    ggml_منزل_float_t   ثابت v_منزل_float     = ggml_get_type_traits(v->type)->منزل_float;

    GGML_ASSERT((                            q_منزل_vec_dot) && "fattn: unsupported K-type");
    GGML_ASSERT((v->type == GGML_TYPE_F32 || v_منزل_float  ) && "fattn: unsupported V-type");

    صحيح ith = params->ith;

    لكل (صحيح ir = ir0; ir < ir1; ++ir) {
        // q indices
        ثابت صحيح iq3 = ir/(neq2*neq1);
        ثابت صحيح iq2 = (ir - iq3*neq2*neq1)/neq1;
        ثابت صحيح iq1 = (ir - iq3*neq2*neq1 - iq2*neq1);

        ثابت uint32_t h = iq2; // head index
        ثابت عائم لاهي = (max_bias > 0.0f) ? h < n_سر_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_سر_log2) + 1) : 1.0f;

        عائم S = 0.0f;      //جوڙ
        عائم M = -INFINITY; // maximum KQ value

        عائم       * VKQ32 = (عائم       *) params->wdata + ith*(1*DK + 2*DV + CACHE_سٽ_سائيز_F32); // FP32 VKQ accumulator
        عائم       * V32   =                 (VKQ32 + 1*DV); // (temporary) FP32 V buffer
        ggml_fp16_t * VKQ16 = (ggml_fp16_t *) (VKQ32 + 1*DV); // (temporary) FP16 VKQ accumulator
        ggml_fp16_t * Q_q   = (ggml_fp16_t *) (VKQ32 + 2*DV); // (temporary) buffer for Q converted to quantized/FP16

        إذا (v->type == GGML_TYPE_F16) {
            memset(VKQ16, 0, DV*حجم(ggml_fp16_t));
        } وإلا {
            memset(VKQ32, 0, DV*حجم(عائم));
        }

        ثابت ggml_fp16_t * mp = نقاب ? (ggml_fp16_t *)((حرف *) نقاب->معلومات + iq1*نقاب->nb[1] + (iq2%نقاب->ne[2])*نقاب->nb[2] + (iq3%نقاب->ne[3])*نقاب->nb[3]) : عدم;

        // k indices
        ثابت صحيح ik3 = iq3 / rk3;
        ثابت صحيح ik2 = iq2 / rk2;

        // v indices
        ثابت صحيح iv3 = iq3 / rv3;
        ثابت صحيح iv2 = iq2 / rv2;

        ثابت عائم * pq = (ثابت عائم *) ((حرف *) q->معلومات + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3));
        q_منزل_vec_dot(pq, Q_q, DK);

        // online softmax / attention
        // loop over n_kv and n_head_kv
        // ref: https://arxiv.org/pdf/2112.05682.pdf

        لكل (int64_t ic = ic_شروعات; ic < ic_پڇاڙي; ++ic) {
            ثابت عائم mv = mp ? لاهي*GGML_CPU_FP16_منزل_FP32(mp[ic]) : 0.0f;
            إذا (mv == -INFINITY) {
                استمر;
            }

            عائم s; // KQ value

            ثابت حرف * k_معلومات = (ثابت حرف *) k->معلومات + ( ic*nbk1 + ik2*nbk2 + ik3*nbk3);
            kq_vec_dot(DK, &s, 0, k_معلومات, 0, Q_q, 0, 1);

            s = s*scale; // scale KQ value

            إذا (logit_softcap != 0.0f) {
                s = logit_softcap*tanhf(s);
            }

            s += mv; // apply mask

            ثابت عائم Mold = M;

            عائم ms = 1.0f; // upon new higher max val, scale VKQ and KQ sum with this value
            عائم vs = 1.0f; // post-softmax KQ value, expf(s - M)

            ثابت حرف * v_معلومات = ((ثابت حرف *) v->معلومات + (ic*nbv1 + iv2*nbv2 + iv3*nbv3));

            إذا (v->type == GGML_TYPE_F16) {
                إذا (s > M) {
                    // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
                    M = s;
                    ms = expf(Mold - M);

                    // V = V*expf(Mold - M)
                    ggml_vec_scale_f16(DV, VKQ16, ms);
                } وإلا {
                    // no new maximum, ms == 1.0f, vs != 1.0f
                    vs = expf(s - M);
                }

                // V += v*expf(s - M)
                ggml_vec_mad_f16(DV, VKQ16, (ثابت ggml_fp16_t *) v_معلومات, vs);
            } وإلا {
                إذا (s > M) {
                    // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
                    M = s;
                    ms = expf(Mold - M);

                    // V = V*expf(Mold - M)
                    ggml_vec_scale_f32(DV, VKQ32, ms);
                } وإلا {
                    // no new maximum, ms == 1.0f, vs != 1.0f
                    vs = expf(s - M);
                }

                // V += v*expf(s - M)
                إذا (v_منزل_float) {
                    v_منزل_float(v_معلومات, V32, DV);
                    ggml_vec_mad_f32(DV, VKQ32, V32, vs);
                } وإلا {
                    // V is F32
                    ggml_vec_mad_f32(DV, VKQ32, (ثابت عائم *) v_معلومات, vs);
                }
            }

            S = S*ms + vs; // scale and increment sum with partial sum
        }

        إذا (v->type == GGML_TYPE_F16) {
            لكل (int64_t d = 0; d < DV; ++d) {
                VKQ32[d] = GGML_CPU_FP16_منزل_FP32(VKQ16[d]);
            }
        }

        // sinks - apply only on the first kv-chunk
        إذا (sinks && ic_شروعات == 0) {
            ثابت عائم s = ((عائم *)((حرف *) sinks->معلومات))[h];

            عائم ms = 1.0f;
            عائم vs = 1.0f;

            إذا (s > M) {
                ms = expf(M - s);
                M = s;
                ggml_vec_scale_f32(DV, VKQ32, ms);
            } وإلا {
                vs = expf(s - M);
            }

            S = S*ms + vs;
        }

        إذا (write_partials) {
            // Write M, S, VKQ to partials for later reduction
            // partials layout: [M, S, VKQ[DV]] per query head
            عائم * partial = partials + ir * partial_stride;
            partial[0] = M;
            partial[1] = S;
            يادگيري_نقل(partial + 2, VKQ32, DV * حجم(عائم));
        } وإلا {
            // V /= S
            ثابت عائم S_inv = S == 0.0f ? 0.0f : 1.0f/S;
            ggml_vec_scale_f32(DV, VKQ32, S_inv);

            // dst indices
            ثابت صحيح i1 = iq1;
            ثابت صحيح i2 = iq2;
            ثابت صحيح i3 = iq3;

            // permute(0, 2, 1, 3)
            يادگيري_نقل((حرف *) dst->معلومات + (i3*ne2*ne1 + i2 + i1*ne1)*nb1, VKQ32, nb1);
        }
    }
}

ساكن فراغ ggml_compute_forward_flash_attn_ext_tiled(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst,
        صحيح ir0, صحيح ir1) {
    ثابت ggml_tensor * q     = dst->src[0];
    ثابت ggml_tensor * k     = dst->src[1];
    ثابت ggml_tensor * v     = dst->src[2];
    ثابت ggml_tensor * نقاب  = dst->src[3];
    ثابت ggml_tensor * sinks = dst->src[4];

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    ثابت int64_t DK = nek0;
    ثابت int64_t DV = nev0;
    ثابت int64_t N  = neq1;

    GGML_ASSERT(ne0 == DV);
    GGML_ASSERT(ne2 == N);

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_سائيز(q->type));
    GGML_ASSERT(nbk0 == ggml_type_سائيز(k->type));
    GGML_ASSERT(nbv0 == ggml_type_سائيز(v->type));

    GGML_ASSERT(neq0 == DK);
    GGML_ASSERT(nek0 == DK);
    GGML_ASSERT(nev0 == DV);

    GGML_ASSERT(neq1 == N);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == حجم(عائم));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(k->type == v->type);
    ثابت ggml_type kv_type = k->type;


    // broadcast factors
    ثابت int64_t rk2 = neq2/nek2;
    ثابت int64_t rk3 = neq3/nek3;

    ثابت int64_t rv2 = neq2/nev2;
    ثابت int64_t rv3 = neq3/nev3;

    عائم scale         = 1.0f;
    عائم max_bias      = 0.0f;
    عائم logit_softcap = 0.0f;

    يادگيري_نقل(&scale,         (عائم *) dst->op_params + 0, حجم(عائم));
    يادگيري_نقل(&max_bias,      (عائم *) dst->op_params + 1, حجم(عائم));
    يادگيري_نقل(&logit_softcap, (عائم *) dst->op_params + 2, حجم(عائم));

    إذا (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    ثابت uint32_t n_سر      = neq2;
    ثابت uint32_t n_سر_log2 = 1u << (uint32_t) floor(log2(n_سر));

    ثابت عائم m0 = powf(2.0f, -(max_bias       ) / n_سر_log2);
    ثابت عائم m1 = powf(2.0f, -(max_bias / 2.0f) / n_سر_log2);

    صحيح ith = params->ith;

    ساكن constexpr صحيح Q_TILE_SZ  = ggml_fa_tile_config::Q;
    ساكن constexpr صحيح KV_TILE_SZ = ggml_fa_tile_config::KV;

    صحيح ir = ir0;
    طالما (ir < ir1) {
        // q indices for the start of this tile
        ثابت صحيح iq3 = ir/(neq2*neq1);
        ثابت صحيح iq2 = (ir - iq3*neq2*neq1)/neq1;
        ثابت صحيح iq1 = (ir - iq3*neq2*neq1 - iq2*neq1);

        // Number of valid rows in this tile:
        // - limited by tile size (Q_TILE_SZ)
        // - limited by chunk boundary (ir1 - ir)
        // - limited by head boundary (neq1 - iq1) to avoid crossing into next head
        ثابت صحيح tile_صفون = MIN(Q_TILE_SZ, MIN((صحيح)(ir1 - ir), (صحيح)(neq1 - iq1)));
        GGML_ASSERT(tile_صفون > 0);

        ثابت uint32_t h = iq2; // head index
        ثابت عائم لاهي = (max_bias > 0.0f) ? h < n_سر_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_سر_log2) + 1) : 1.0f;

        عائم S[Q_TILE_SZ];
        عائم M[Q_TILE_SZ];

        لكل (صحيح i = 0 ; i < Q_TILE_SZ; ++i) {
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
        عائم * بنياد  = (عائم *) params->wdata + ith*(Q_TILE_SZ*DK + 2*Q_TILE_SZ*KV_TILE_SZ + Q_TILE_SZ*DV + KV_TILE_SZ*DV + KV_TILE_SZ*DK + CACHE_سٽ_سائيز_F32);

        فراغ  * Q_q    = بنياد;
        عائم * KQ     = (عائم *)((حرف *)بنياد + Q_TILE_SZ * DK * حجم(عائم));
        عائم * mask32 = KQ + Q_TILE_SZ * KV_TILE_SZ;
        عائم * VKQ32  = mask32 + Q_TILE_SZ * KV_TILE_SZ;
        عائم * V32    = VKQ32 + Q_TILE_SZ * DV;
        عائم * K_f32  = V32 + KV_TILE_SZ * DV;

        memset(VKQ32, 0, Q_TILE_SZ * DV * حجم(عائم));
        memset(mask32, 0, Q_TILE_SZ * KV_TILE_SZ * حجم(عائم));

        // k indices
        ثابت صحيح ik3 = iq3 / rk3;
        ثابت صحيح ik2 = iq2 / rk2;

        // v indices
        ثابت صحيح iv3 = iq3 / rv3;
        ثابت صحيح iv2 = iq2 / rv2;

        {
            عائم * Q_f32 = (عائم *)Q_q;
            لكل (صحيح tq = 0; tq < tile_صفون; tq++) {
                ثابت عائم * pq = (ثابت عائم *) ((حرف *) q->معلومات + ((iq1 + tq)*nbq1 + iq2*nbq2 + iq3*nbq3));
                يادگيري_نقل(Q_f32 + tq * DK, pq, DK * حجم(عائم));
            }
            لكل (صحيح tq = tile_صفون; tq < Q_TILE_SZ; tq++) {
                memset(Q_f32 + tq * DK, 0, DK * حجم(عائم));
            }
        }

        memset(K_f32, 0, DK * KV_TILE_SZ * حجم(عائم));
        memset(V32,   0, KV_TILE_SZ * DV * حجم(عائم));

        لكل (int64_t ic = 0; ic < nek1; ic += KV_TILE_SZ) {
            ثابت صحيح kv_tile = (صحيح)std::min((int64_t)KV_TILE_SZ, nek1 - ic);

            // skip the tile entirely if all the masks are -inf
            إذا (نقاب) {
                منطقي can_skip = سچ;
                لكل (صحيح tq = 0; tq < tile_صفون; tq++) {
                    ثابت ggml_fp16_t * mp_صف = (ثابت ggml_fp16_t *)((ثابت حرف *) نقاب->معلومات + (iq1 + tq)*نقاب->nb[1] + (iq2%نقاب->ne[2])*نقاب->nb[2] + (iq3%نقاب->ne[3])*نقاب->nb[3]);
                    لكل (صحيح tk = 0; tk < kv_tile; tk++) {
                        mask32[tq * KV_TILE_SZ + tk] = لاهي * GGML_CPU_FP16_منزل_FP32(mp_صف[ic + tk]);
                        إذا (mask32[tq * KV_TILE_SZ + tk] != -INFINITY) {
                            can_skip = خطأ;
                        }
                    }
                    // Pad remaining mask entries with -inf
                    لكل (صحيح tk = kv_tile; tk < KV_TILE_SZ; tk++) {
                        mask32[tq * KV_TILE_SZ + tk] = -INFINITY;
                    }
                }

                إذا (can_skip) {
                    استمر;
                }
            }

            // Pack K tile transposed: K_f32[dk][kv] so KV_TILE is contiguous (SIMD dim)
            // Zero-pad the last tile so the GEMM always operates on KV_TILE_SZ columns
            لكل (صحيح tk = 0; tk < kv_tile; tk++) {
                ثابت حرف * k_معلومات = (ثابت حرف *)k->معلومات + (ic + tk)*nbk1 + ik2*nbk2 + ik3*nbk3;
                إذا (kv_type == GGML_TYPE_F16) {
                    ثابت ggml_fp16_t * k_f16 = (ثابت ggml_fp16_t *)k_معلومات;
                    لكل (int64_t dk = 0; dk < DK; dk++) {
                        K_f32[dk * KV_TILE_SZ + tk] = GGML_CPU_FP16_منزل_FP32(k_f16[dk]);
                    }
                } وإلا {
                    ثابت عائم * k_f32_src = (ثابت عائم *)k_معلومات;
                    لكل (int64_t dk = 0; dk < DK; dk++) {
                        K_f32[dk * KV_TILE_SZ + tk] = k_f32_src[dk];
                    }
                }
            }
            memset(KQ, 0, Q_TILE_SZ * KV_TILE_SZ * حجم(عائم));
            simd_gemm(KQ, (ثابت عائم *)Q_q, K_f32, Q_TILE_SZ, DK, KV_TILE_SZ);
            ggml_vec_scale_f32(Q_TILE_SZ * KV_TILE_SZ, KQ, scale);

            إذا (logit_softcap != 0.0f) {
                ggml_vec_tanh_f32(Q_TILE_SZ * KV_TILE_SZ, KQ, KQ);
                ggml_vec_scale_f32(Q_TILE_SZ * KV_TILE_SZ, KQ, logit_softcap);
            }

            // Set padded KQ entries to -inf so softmax gives them zero weight
            إذا (kv_tile < KV_TILE_SZ) {
                لكل (صحيح tq = 0; tq < Q_TILE_SZ; tq++) {
                    لكل (صحيح tk = kv_tile; tk < KV_TILE_SZ; tk++) {
                        KQ[tq * KV_TILE_SZ + tk] = -INFINITY;
                    }
                }
            }

            إذا (نقاب) {
                ggml_vec_جوڙيو_f32(tile_صفون * KV_TILE_SZ, KQ, KQ, mask32);
            }

            منطقي skip[Q_TILE_SZ] = {};

            لكل (صحيح tq = 0; tq < Q_TILE_SZ; tq++) {
                عائم * kq_صف = KQ + tq * KV_TILE_SZ;

                عائم tile_max;
                ggml_vec_max_f32(KV_TILE_SZ, &tile_max, kq_صف);

                إذا (tile_max == -INFINITY) {
                    skip[tq] = سچ;
                    استمر;
                }

                ثابت عائم Mold = M[tq];
                ثابت عائم Mnew = fmaxf(Mold, tile_max);

                إذا (Mnew > Mold) {
                    ثابت عائم ms = expf(Mold - Mnew);
                    ggml_vec_scale_f32(DV, VKQ32 + tq * DV, ms);
                    S[tq] *= ms;
                }
                M[tq] = Mnew;


                S[tq] += ggml_vec_soft_max_f32(KV_TILE_SZ, kq_صف, kq_صف, Mnew);
            }

            // V accumulation: VKQ32 += softmax(KQ) * V
            // Pack V tile to contiguous F32, zero-padded
            لكل (صحيح tk = 0; tk < kv_tile; tk++) {
                ثابت حرف * v_معلومات = (ثابت حرف *)v->معلومات + (ic + tk)*nbv1 + iv2*nbv2 + iv3*nbv3;
                إذا (kv_type == GGML_TYPE_F16) {
                    ggml_cpu_fp16_منزل_fp32((ثابت ggml_fp16_t *)v_معلومات, V32 + tk * DV, DV);
                } وإلا {
                    يادگيري_نقل(V32 + tk * DV, v_معلومات, DV * حجم(عائم));
                }
            }
            لكل (صحيح tq = 0; tq < Q_TILE_SZ; tq++) {
                إذا (skip[tq]) {
                    memset(KQ + tq * KV_TILE_SZ, 0, KV_TILE_SZ * حجم(عائم));
                }
            }
            simd_gemm(VKQ32, KQ, V32, Q_TILE_SZ, KV_TILE_SZ, DV);
        }

        // sinks (apply only to valid rows in the tile)
        إذا (sinks) {
            ثابت عائم s = ((عائم *)((حرف *) sinks->معلومات))[h];

            لكل (صحيح tq = 0; tq < tile_صفون; tq++) {
                عائم ms = 1.0f;
                عائم vs = 1.0f;

                إذا (s > M[tq]) {
                    ms = expf(M[tq] - s);
                    ggml_vec_scale_f32(DV, VKQ32 + tq * DV, ms);
                } وإلا {
                    vs = expf(s - M[tq]);
                }

                S[tq] = S[tq] * ms + vs;
            }
        }

        لكل (صحيح tq = 0; tq < tile_صفون; tq++) {
            // V /= S
            ثابت عائم S_inv = S[tq] == 0.0f ? 0.0f : 1.0f / S[tq];
            ggml_vec_scale_f32(DV, VKQ32 + tq * DV, S_inv);

            // dst indices
            ثابت صحيح i1 = iq1 + tq;
            ثابت صحيح i2 = iq2;
            ثابت صحيح i3 = iq3;

            // permute(0, 2, 1, 3)
            يادگيري_نقل((حرف *) dst->معلومات + (i3*ne2*ne1 + i2 + i1*ne1)*nb1, VKQ32 + tq * DV, nb1);
        }

        ir += tile_صفون;
    }
}

// Reduction function: combines partial results across KV chunks
// Partials layout in wdata: [n_q_heads][n_chunks][2 + DV]
ساكن فراغ ggml_flash_attn_ext_reduce_partials(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst,
        ثابت int64_t n_chunks,
        ثابت int64_t chunk_سائيز) {

    ثابت ggml_tensor * q = dst->src[0];
    ثابت ggml_tensor * k = dst->src[1];
    ثابت ggml_tensor * v = dst->src[2];

    ثابت int64_t DK        = k->ne[0];
    ثابت int64_t DV        = v->ne[0];
    ثابت int64_t nek1      = k->ne[1];
    ثابت int64_t n_q_heads = q->ne[2];

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت int64_t wdata_في_thread = DK + 2*DV + CACHE_سٽ_سائيز_F32;
    عائم *       thread_wdata     = (عائم *) params->wdata + ith * wdata_في_thread;

    ثابت int64_t partials_وٿي  = nth * (DK + 2*DV + CACHE_سٽ_سائيز_F32);
    ثابت int64_t partial_سائيز     = 2 + DV;
    ثابت عائم * partials_بنياد    = (ثابت عائم *) params->wdata + partials_وٿي;

    // Output layout
    ثابت int64_t ne1 = dst->ne[1];
    ثابت int64_t ne2 = dst->ne[2];
    ثابت size_t  nb1 = dst->nb[1];

    // Each thread reduces a subset of query heads
    لكل (int64_t q_سر = ith; q_سر < n_q_heads; q_سر += nth) {
        عائم   M_final   = -INFINITY;
        عائم   S_final   = 0.0f;
        عائم * VKQ_final = thread_wdata;
        memset(VKQ_final, 0, DV * حجم(عائم));

        // Combine partials from all chunks
        لكل (int64_t chunk_idx = 0; chunk_idx < n_chunks; ++chunk_idx) {
            ثابت int64_t ic_شروعات = chunk_idx * chunk_سائيز;
            إذا (ic_شروعات >= nek1) استمر;

            ثابت عائم * partial   = partials_بنياد + (q_سر * n_chunks + chunk_idx) * partial_سائيز;
            ثابت عائم   M_chunk   = partial[0];
            ثابت عائم   S_chunk   = partial[1];
            ثابت عائم * VKQ_chunk = partial + 2;

            إذا (S_chunk == 0.0f) استمر;

            ثابت عائم M_new     = fmaxf(M_final, M_chunk);
            ثابت عائم scale_old = expf(M_final - M_new);
            ثابت عائم scale_new = expf(M_chunk - M_new);

            لكل (int64_t d = 0; d < DV; ++d) {
                VKQ_final[d] = VKQ_final[d] * scale_old + VKQ_chunk[d] * scale_new;
            }
            S_final = S_final * scale_old + S_chunk * scale_new;
            M_final = M_new;
        }

        // Normalize and write to output
        إذا (S_final != 0.0f) {
            ثابت عائم S_inv = 1.0f / S_final;
            ggml_vec_scale_f32(DV, VKQ_final, S_inv);
        }
        // iq1=0, iq3=0 for decode
        يادگيري_نقل((حرف *) dst->معلومات + (0*ne2*ne1 + q_سر + 0*ne1)*nb1, VKQ_final, nb1);
    }
}

ساكن فراغ ggml_compute_forward_flash_attn_ext_f16(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * q     = dst->src[0];
    ثابت ggml_tensor * k     = dst->src[1];
    ثابت ggml_tensor * v     = dst->src[2];

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    ثابت int64_t DK = nek0;
    ثابت int64_t DV = nev0;
    ثابت int64_t N  = neq1;


    GGML_ASSERT(ne0 == DV);
    GGML_ASSERT(ne2 == N);

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_سائيز(q->type));
    GGML_ASSERT(nbk0 == ggml_type_سائيز(k->type));
    GGML_ASSERT(nbv0 == ggml_type_سائيز(v->type));

    GGML_ASSERT(neq0 == DK);
    GGML_ASSERT(nek0 == DK);
    GGML_ASSERT(nev0 == DV);

    GGML_ASSERT(neq1 == N);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == حجم(عائم));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    // When use_ref is set, force the vec-only reference implementation (no tiling, no KV-chunking)
    ثابت منطقي use_ref = params->use_ref;

    ثابت منطقي kv_is_f32_or_f16 = (k->type == GGML_TYPE_F32 || k->type == GGML_TYPE_F16);
    ثابت منطقي use_ورهايو_kv_رستو = !use_ref && (neq1 == 1 && neq3 == 1) && kv_is_f32_or_f16 && (k->type == v->type) && q->type == GGML_TYPE_F32 && nek1 >= 512;

    إذا (use_ورهايو_kv_رستو) {
        ثابت int64_t chunk_سائيز = (nek1 + nth - 1) / nth;

        // Partials buffer layout: [q_head][kv_chunk][M, S, VKQ]
        ثابت int64_t partial_سائيز  = 2 + DV;
        عائم *       partials_بنياد = (عائم *) params->wdata + nth * (DK + 2*DV + CACHE_سٽ_سائيز_F32);

        ثابت int64_t ic_شروعات = ith * chunk_سائيز;
        ثابت int64_t ic_پڇاڙي   = std::min(ic_شروعات + chunk_سائيز, nek1);

        ثابت int64_t partial_stride = nth * partial_سائيز;
        عائم *       chunk_partials = partials_بنياد + ith * partial_سائيز;

        إذا (ic_شروعات < nek1) {
            لكل (int64_t q_سر = 0; q_سر < neq2; q_سر++) {
                ggml_compute_forward_flash_attn_ext_f16_one_chunk(
                    params, dst, q_سر, q_سر + 1, ic_شروعات, ic_پڇاڙي,
                    chunk_partials, partial_stride);
            }
        } وإلا {
            لكل (int64_t q_سر = 0; q_سر < neq2; q_سر++) {
                عائم * q_partials = chunk_partials + q_سر * partial_stride;
                q_partials[0] = -INFINITY;  // M
                q_partials[1] = 0.0f;       // S
            }
        }

        ggml_barrier(params->threadpool);
        ggml_flash_attn_ext_reduce_partials(params, dst, nth, chunk_سائيز);
    } وإلا {

        // total rows in q
        ثابت int64_t nr = neq1*neq2*neq3;

        // disable for NUMA
        ثابت منطقي disable_chunking = ggml_is_numa();

        // 4x chunks per thread
        صحيح nth_scaled = nth * 4;
        int64_t chunk_سائيز = (nr + nth_scaled - 1) / nth_scaled;
        int64_t nchunk     = (nr + chunk_سائيز - 1) / chunk_سائيز;

        إذا (nth == 1 || nchunk < nth || disable_chunking) {
            nchunk = nth;
        }

        إذا (ith == 0) {
            ggml_threadpool_chunk_سيٽ(params->threadpool, nth);
        }

        ggml_barrier(params->threadpool);

        ثابت int64_t dr = (nr + nchunk - 1) / nchunk;

        ساكن constexpr int64_t Q_TILE_SZ  = ggml_fa_tile_config::Q;
        منطقي use_tiled = !use_ref &&
                               (q->type == GGML_TYPE_F32 &&
                                kv_is_f32_or_f16 &&
                                k->type == v->type &&
                                neq1 >= Q_TILE_SZ);
#إذا defined(GGML_SIMD) && !defined(__x86_64__) && !defined(_M_X64)
#إذا defined(__ARM_FEATURE_SVE)
        ثابت int64_t f32_epr = svcntw();
#وإلا
        ثابت int64_t f32_epr = GGML_F32_EPR;
#جيڪڏهن_پورو
        use_tiled &= (DV % f32_epr == 0);
#جيڪڏهن_پورو
        صحيح current_chunk = ith;

        طالما (current_chunk < nchunk) {
            ثابت int64_t ir0 = dr * current_chunk;
            ثابت int64_t ir1 = MIN(ir0 + dr, nr);

            إذا (use_tiled) {
                ggml_compute_forward_flash_attn_ext_tiled(params, dst, ir0, ir1);
            } وإلا {
                ggml_compute_forward_flash_attn_ext_f16_one_chunk(params, dst, ir0, ir1, 0, nek1, خالي_اشارو, 0);
            }

            current_chunk = ggml_threadpool_chunk_جوڙيو(params->threadpool, 1);
        }
    }
}

فراغ ggml_compute_forward_flash_attn_ext(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    بدل (dst->op_params[3]) {
        حالة GGML_PREC_DEFAULT:
        حالة GGML_PREC_F32:
            {
                // uses F32 accumulators
                ggml_compute_forward_flash_attn_ext_f16(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_flash_attn_back

ساكن فراغ ggml_compute_forward_flash_attn_پٺ_f32(
        ثابت ggml_compute_params * params,
        ثابت منطقي masked,
              ggml_tensor * dst) {

    ثابت ggml_tensor * q = dst->src[0];
    ثابت ggml_tensor * k = dst->src[1];
    ثابت ggml_tensor * v = dst->src[2];
    ثابت ggml_tensor * d = dst->src[3];

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

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت int64_t D = neq0;
    ثابت int64_t N = neq1;
    ثابت int64_t P = nek1 - N;
    ثابت int64_t M = P + N;

    ثابت صحيح Mup  = ggml_مٿي(M, GGML_SOFT_MAX_UNROLL);
    ثابت صحيح mxDM = MAX(D, Mup);

    // GGML_ASSERT(ne0 == D);
    // GGML_ASSERT(ne1 == N);
    GGML_ASSERT(P >= 0);

    GGML_ASSERT(nbq0 == حجم(عائم));
    GGML_ASSERT(nbk0 == حجم(عائم));
    GGML_ASSERT(nbv0 == حجم(عائم));

    GGML_ASSERT(neq0 == D);
    GGML_ASSERT(nek0 == D);
    GGML_ASSERT(nev1 == D);
    GGML_ASSERT(ned0 == D);

    GGML_ASSERT(neq1 == N);
    GGML_ASSERT(nek1 == N + P);
    GGML_ASSERT(nev1 == D);
    GGML_ASSERT(ned1 == N);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == حجم(عائم));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    إذا (ith == 0) {
        memset(dst->معلومات, 0, nb0*ne0*ne1*ne2*ne3);
    }
    ggml_barrier(params->threadpool);

    ثابت int64_t elem_q = ggml_nelements(q);
    ثابت int64_t elem_k = ggml_nelements(k);

    ggml_type result_type = dst->type;
    GGML_ASSERT(ggml_blck_سائيز(result_type) == 1);
    ثابت size_t tsize = ggml_type_سائيز(result_type);

    ثابت size_t offs_q = 0;
    ثابت size_t offs_k = offs_q + GGML_PAD(elem_q * tsize, GGML_MEM_ALIGN);
    ثابت size_t offs_v = offs_k + GGML_PAD(elem_k * tsize, GGML_MEM_ALIGN);

    فراغ * grad_q = (حرف *) dst->معلومات;
    فراغ * grad_k = (حرف *) dst->معلومات + offs_k;
    فراغ * grad_v = (حرف *) dst->معلومات + offs_v;

    ثابت size_t nbgq1 = nb0*neq0;
    ثابت size_t nbgq2 = nb0*neq0*neq1;
    ثابت size_t nbgq3 = nb0*neq0*neq1*neq2;

    ثابت size_t nbgk1 = nb0*nek0;
    ثابت size_t nbgk2 = nb0*nek0*nek1;
    ثابت size_t nbgk3 = nb0*nek0*nek1*neq2;

    ثابت size_t nbgv1 = nb0*nev0;
    ثابت size_t nbgv2 = nb0*nev0*nev1;
    ثابت size_t nbgv3 = nb0*nev0*nev1*neq2;

    // parallelize by k rows using ggml_vec_dot_f32

    // total rows in k
    ثابت صحيح nr = nek2*nek3;

    // rows per thread
    ثابت صحيح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحيح ir0 = dr*ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    ثابت عائم scale = 1.0f/sqrtf(D);

    //printf("P=%d N=%d D=%d ir0=%d ir1=%d scale = %f\n", P, N, D, ir0, ir1, scale);

    // how often k2 (and v2) is repeated in q2
    صحيح nrep = neq2/nek2;

    لكل (صحيح ir = ir0; ir < ir1; ++ir) {
        // q indices
        ثابت صحيح ik3 = ir/(nek2);
        ثابت صحيح ik2 = ir - ik3*nek2;

        ثابت صحيح iq3 = ik3;
        ثابت صحيح id3 = ik3;
        ثابت صحيح iv3 = ik3;
        ثابت صحيح iv2 = ik2;

        لكل (صحيح irep = 0; irep < nrep; ++irep) {
            ثابت صحيح iq2 = ik2 + irep*nek2;
            ثابت صحيح id2 = iq2;

            // (ik2 + irep*nek2) % nek2 == ik2
            لكل (صحيح iq1 = 0; iq1 < neq1; ++iq1) {
                ثابت صحيح id1 = iq1;

                // not sure about CACHE_LINE_SIZE_F32..
                // - maybe it must not be multiplied by 2 and excluded from .. in SM 1*(..) offset?
                عائم * S  = (عائم *) params->wdata + ith*2*(mxDM + CACHE_سٽ_سائيز_F32) + 0*(mxDM+CACHE_سٽ_سائيز_F32);
                عائم * SM = (عائم *) params->wdata + ith*2*(mxDM + CACHE_سٽ_سائيز_F32) + 1*(mxDM+CACHE_سٽ_سائيز_F32);

                لكل (صحيح i = M; i < Mup; ++i) {
                    S[i] = -INFINITY;
                }

                ثابت int64_t masked_begin = masked ? (P + iq1 + 1) : M;
                لكل (int64_t ic = 0; ic < masked_begin; ++ic) {
                    // k indices
                    ثابت صحيح ik1 = ic;

                    // S indices
                    ثابت صحيح i1 = ik1;

                    ggml_vec_dot_f32(neq0,
                            S + i1, 0,
                            (عائم *) ((حرف *) k->معلومات + (ik1*nbk1 + ik2*nbk2 + ik3*nbk3)), 0,
                            (عائم *) ((حرف *) q->معلومات + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3)), 0, 1);
                }

                // scale
                ggml_vec_scale_f32(masked_begin, S, scale);

                لكل (int64_t i = masked_begin; i < M; i++) {
                    S[i] = -INFINITY;
                }

                // softmax
                // exclude known -INF S[..] values from max and loop
                // dont forget to set their SM values to zero
                {
                    عائم max = -INFINITY;
                    ggml_vec_max_f32(masked_begin, &max, S);

                    ggml_float جوڙ = 0.0;
                    {
#جيڪڏهن_وصف_ٿيل GGML_SOFT_MAX_ACCELERATE
                        max = -max;
                        vDSP_vsadd(SM, 1, &max, SM, 1, Mup);
                        vvexpf(SM, SM, &Mup);
                        ggml_vec_جوڙ_f32(Mup, &جوڙ, SM);
#وإلا
                        جوڙ = ggml_vec_soft_max_f32(Mup, SM, S, max);
#جيڪڏهن_پورو
                    }

                    أكد(جوڙ > 0.0);

                    جوڙ = 1.0/جوڙ;
                    ggml_vec_scale_f32(masked_begin, SM, جوڙ);

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
                ggml_vec_سيٽ_f32(masked_begin, S, 0);
                لكل (int64_t ic = 0; ic < D; ++ic) {
                    ggml_vec_mad_f32(masked_begin,
                            S,
                             (عائم *) ((حرف *) v->معلومات + (          ic*nbv1  + iv2*nbv2 + iv3*nbv3)),
                            *(عائم *) ((حرف *) d->معلومات + (ic*nbd0 + id1*nbd1 + id2*nbd2 + id3*nbd3)));
                }

                // S = SM * (S - dot(SM, S))
                عائم dot_SM_gradSM = 0;
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
                لكل (int64_t ic = 0; ic < masked_begin; ++ic) {
                    ggml_vec_mad_f32(D,
                            (عائم *) ((حرف *) grad_q  + (iq1*nbgq1 + iq2*nbgq2  + iq3*nbgq3)),
                            (عائم *) ((حرف *) k->معلومات + (ic*nbk1   + ik2*nbk2   + ik3*nbk3)),
                            S[ic]);
                }

                // grad[k][:D,:M,iq2,iq3] += S.T @ qcur
                // for ic:
                //  grad[k][:D,ic,iq2,iq3] += S.T[0,ic] * qcur[:D,0]
                //  grad[k][:D,ic,iq2,iq3] += S[ic]     * qcur[:D,0]
                // exclude known zero S[..] values from loop
                لكل (int64_t ic = 0; ic < masked_begin; ++ic) {
                    ggml_vec_mad_f32(D,
                            (عائم *) ((حرف *) grad_k  + (ic*nbgk1  + ik2*nbgk2  + ik3*nbgk3)),
                            (عائم *) ((حرف *) q->معلومات + (iq1*nbq1  + iq2*nbq2   + iq3*nbq3)),
                            S[ic]);
                }

                // grad[v][:M,:D,iv2,iv3] += d[:D,id1,id2,id3].T       @ SM
                // for ic:
                //  grad[v][:M,ic,iv2,iv3] += d[:D,id1,id2,id3].T[0,ic] * SM[:M]
                //  grad[v][:M,ic,iv2,iv3] += d[ic,id1,id2,id3]         * SM[:M]
                // exclude known zero SM[..] values from mad
                لكل (int64_t ic = 0; ic < D; ++ic) {
                    ggml_vec_mad_f32(masked_begin,
                            (عائم *) ((حرف *) grad_v   + (          ic*nbgv1 + iv2*nbgv2 + iv3*nbgv3)),
                            SM,
                            *(عائم *) ((حرف *) d->معلومات + (ic*nbd0 + id1*nbd1 + id2*nbd2  + id3*nbd3)));
                }
            }
        }
    }
}

فراغ ggml_compute_forward_flash_attn_پٺ(
        ثابت ggml_compute_params * params,
        ثابت منطقي masked,
        ggml_tensor * dst) {

    ثابت ggml_tensor * q = dst->src[0];

    بدل (q->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_flash_attn_پٺ_f32(params, masked, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_ssm_conv

ساكن فراغ ggml_compute_forward_ssm_conv_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0]; // conv_x
    ثابت ggml_tensor * src1 = dst->src[1]; // conv1d.weight

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح nc  = src1->ne[0]; // d_conv
    ثابت صحيح ncs = src0->ne[0]; // d_conv - 1 + n_t
    ثابت صحيح nr  = src0->ne[1]; // d_inner
    ثابت صحيح n_t =  dst->ne[1]; // tokens per sequence
    ثابت صحيح n_s =  dst->ne[2]; // number of sequences in the batch

    GGML_ASSERT( dst->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == حجم(عائم));
    GGML_ASSERT(src1->nb[0] == حجم(عائم));
    GGML_ASSERT(src0->nb[1] == src0->ne[0]*حجم(عائم));

    // rows per thread
    ثابت صحيح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحيح ir0 = dr*ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);
    ثابت صحيح ir  = ir1 - ir0;

    لكل (صحيح i3 = 0; i3 < n_s; ++i3) {
        لكل (صحيح i2 = 0; i2 < n_t; ++i2) {
            // {d_conv - 1 + n_t, d_inner, n_seqs}
            // sliding window
            ثابت عائم * s = (ثابت عائم *) ((ثابت حرف *) src0->معلومات + ir0*(src0->nb[1]) + i2*(src0->nb[0]) + i3*(src0->nb[2])); // {d_conv, d_inner, n_s}
            ثابت عائم * c = (ثابت عائم *) ((ثابت حرف *) src1->معلومات + ir0*(src1->nb[1])); // {d_conv, d_inner}
            عائم * x = (عائم *) ((حرف *) dst->معلومات + ir0*(dst->nb[0]) + i2*(dst->nb[1]) + i3*(dst->nb[2])); // {d_inner, n_t, n_s}

            // TODO: transpose the output for smaller strides for big batches?
            // d_inner
            لكل (صحيح i1 = 0; i1 < ir; ++i1) {
                // rowwise dot product
                // NOTE: not using ggml_vec_dot_f32, because its sum is in double precision
                عائم sumf = 0.0f;

                // d_conv
                لكل (صحيح i0 = 0; i0 < nc; ++i0) {
                    sumf += s[i0 + i1*ncs] * c[i0 + i1*nc];
                }
                x[i1] = sumf;
            }
        }
    }
}

فراغ ggml_compute_forward_ssm_conv(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    بدل (dst->src[0]->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_ssm_conv_f32(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_ssm_scan

ساكن فراغ ggml_compute_forward_ssm_جاچ_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0]; // s  {d_state, dim, n_head, n_seqs+}
    ثابت ggml_tensor * src1 = dst->src[1]; // x  {dim, n_head, n_seq_tokens, n_seqs}
    ثابت ggml_tensor * src2 = dst->src[2]; // dt {n_head, n_seq_tokens, n_seqs}
    ثابت ggml_tensor * src3 = dst->src[3]; // A  {d_state, n_head} or {1, n_head}
    ثابت ggml_tensor * src4 = dst->src[4]; // B  {d_state, n_group, n_seq_tokens, n_seqs}
    ثابت ggml_tensor * src5 = dst->src[5]; // C  {d_state, n_group, n_seq_tokens, n_seqs}
    ثابت ggml_tensor * src6 = dst->src[6]; // ids {n_seqs}

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت int64_t nc = src0->ne[0]; // d_state
    ثابت int64_t nr = src0->ne[1]; // dim
    ثابت int64_t nh = src1->ne[1]; // n_head
    ثابت int64_t ng = src4->ne[1];
    ثابت int64_t nt = src1->ne[2]; // number of tokens per sequence
    ثابت int64_t ns = src1->ne[3]; // number of sequences in the batch
    ثابت int64_t K  = ggml_get_op_params_i32(dst, 0);

    // can't use ggml_nbytes because src1 is not necessarily contiguous
    ثابت int64_t s_off = ggml_nelements(src1) * ggml_element_سائيز(src1);

    GGML_ASSERT(K >= 1);
    GGML_ASSERT(ggml_nelements(src1) + K*nc*nr*nh*ns == ggml_nelements(dst));
    GGML_ASSERT(src0->nb[0] == حجم(عائم));
    GGML_ASSERT(src1->nb[0] == حجم(عائم));
    GGML_ASSERT(src2->nb[0] == حجم(عائم));
    GGML_ASSERT(src3->nb[0] == حجم(عائم));
    GGML_ASSERT(src4->nb[0] == حجم(عائم));
    GGML_ASSERT(src5->nb[0] == حجم(عائم));
    GGML_ASSERT(src6->nb[0] == حجم(int32_t));
    GGML_ASSERT(nh % ng == 0);
    GGML_ASSERT(src3->ne[0] == 1 || K == 1);

    // heads per thread
    ثابت صحيح dh = (nh + nth - 1)/nth;

    // head range for this thread
    ثابت صحيح ih0 = dh*ith;
    ثابت صحيح ih1 = MIN(ih0 + dh, nh);

    ثابت int32_t * ids = (ثابت int32_t *) src6->معلومات;

    لكل (صحيح i3 = 0; i3 < ns; ++i3) {
        ثابت عائم * s0 = (ثابت عائم *) ((ثابت حرف *) src0->معلومات + ids[i3]*(src0->nb[3])); // {d_state, dim, nh, ns}
              عائم * s  = (      عائم *) ((      حرف *) dst->معلومات  + i3*(src0->nb[3]) + s_off); // {d_state, dim, nh, ns}

        لكل (صحيح i2 = 0; i2 < nt; ++i2) {
            ثابت عائم * x  = (ثابت عائم *) ((ثابت حرف *) src1->معلومات + i2*(src1->nb[2]) + i3*(src1->nb[3])); // {dim, nh, nt, ns}
            ثابت عائم * dt = (ثابت عائم *) ((ثابت حرف *) src2->معلومات + i2*(src2->nb[1]) + i3*(src2->nb[2])); // {nh, nt, ns}
            ثابت عائم * A  = (ثابت عائم *) ((ثابت حرف *) src3->معلومات); // {d_state, nh} or {1, nh}
            ثابت عائم * B  = (ثابت عائم *) ((ثابت حرف *) src4->معلومات + i2*(src4->nb[2]) + i3*(src4->nb[3])); // {d_state, ng, nt, ns}
            ثابت عائم * C  = (ثابت عائم *) ((ثابت حرف *) src5->معلومات + i2*(src5->nb[2]) + i3*(src5->nb[3])); // {d_state, ng, nt, ns}
                  عائم * y  = (      عائم *) ((      حرف *) dst->معلومات + i2*(nh*nr*حجم(عائم)) + i3*(nt*nh*nr*حجم(عائم))); // {dim, nh, nt, ns}

            إذا (src3->ne[0] == 1) {
                // Mamba-2 has a scalar decay factor per head; dA can be outside the state-wise loop

                // n_head
                لكل (صحيح h = ih0; h < ih1; ++h) {
                    // ref: https://github.com/state-spaces/mamba/blob/62db608da60f6fc790b8ed9f4b3225e95ca15fde/mamba_ssm/ops/triton/softplus.py#L16
                    ثابت عائم dt_soft_plus = ggml_compute_softplus_f32(dt[h]);
                    ثابت عائم dA = expf(dt_soft_plus * A[h]);
                    ثابت صحيح g = h / (nh / ng); // repeat_interleave

                    // dim
                    لكل (صحيح i1 = 0; i1 < nr; ++i1) {
                        ثابت صحيح ii = i1 + h*nr;
                        ثابت عائم x_dt = x[ii] * dt_soft_plus;
                        عائم sumf = 0.0f;
#إذا defined(GGML_SIMD)
    #إذا defined(__ARM_FEATURE_SVE)
                        ثابت صحيح ggml_f32_epr = svcntw();
                        ثابت صحيح ggml_f32_قدم = 1 * ggml_f32_epr;

                        ثابت صحيح np = (nc & ~(ggml_f32_قدم - 1));

                        GGML_F32_VEC جوڙ = GGML_F32_VEC_ZERO;

                        GGML_F32_VEC adA = GGML_F32_VEC_SET1(dA);
                        GGML_F32_VEC axdt = GGML_F32_VEC_SET1(x_dt);

                        لكل (صحيح i = 0; i < np; i += ggml_f32_قدم) {
                            // TODO: maybe unroll more?
                            لكل (صحيح j = 0; j < 1; j++) {
                                GGML_F32_VEC t0 = GGML_F32_VEC_ڀريو(s0 + i + j*ggml_f32_epr + ii*nc);
                                GGML_F32_VEC t1 = GGML_F32_VEC_ڀريو(B + i + j*ggml_f32_epr + g*nc);
                                GGML_F32_VEC t2 = GGML_F32_VEC_ڀريو(C + i + j*ggml_f32_epr + g*nc);

                                t0 = GGML_F32_VEC_MUL(t0, adA);
                                t1 = GGML_F32_VEC_MUL(t1, axdt);

                                t0 = GGML_F32_VEC_جوڙيو(t0, t1);

                                جوڙ = GGML_F32_VEC_FMA(جوڙ, t0, t2);

                                GGML_F32_VEC_STORE(s + i + j*ggml_f32_epr + ii*nc, t0);
                            }
                        }

                        sumf = GGML_F32xt_REDUCE_ONE(جوڙ);
    #وإلاإذا defined(__riscv_v_intrinsic)
                        // todo: RVV implementation
                        ثابت صحيح np = 0;
    #وإلا
                        ثابت صحيح np = (nc & ~(GGML_F32_قدم - 1));

                        GGML_F32_VEC جوڙ[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };

                        GGML_F32_VEC adA = GGML_F32_VEC_SET1(dA);
                        GGML_F32_VEC axdt = GGML_F32_VEC_SET1(x_dt);

                        GGML_F32_VEC ax[GGML_F32_ARR];
                        GGML_F32_VEC ay[GGML_F32_ARR];
                        GGML_F32_VEC az[GGML_F32_ARR];

                        لكل (صحيح i = 0; i < np; i += GGML_F32_قدم) {
                            لكل (صحيح j = 0; j < GGML_F32_ARR; j++) {
                                ax[j] = GGML_F32_VEC_ڀريو(s0 + i + j*GGML_F32_EPR + ii*nc);
                                ay[j] = GGML_F32_VEC_ڀريو(B + i + j*GGML_F32_EPR + g*nc);
                                az[j] = GGML_F32_VEC_ڀريو(C + i + j*GGML_F32_EPR + g*nc);

                                ax[j] = GGML_F32_VEC_MUL(ax[j], adA);
                                ay[j] = GGML_F32_VEC_MUL(ay[j], axdt);

                                ax[j] = GGML_F32_VEC_جوڙيو(ax[j], ay[j]);

                                جوڙ[j] = GGML_F32_VEC_FMA(جوڙ[j], ax[j], az[j]);

                                GGML_F32_VEC_STORE(s + i + j*GGML_F32_EPR + ii*nc, ax[j]);
                            }
                        }

                        // reduce sum0..sum3 to sum0
                        GGML_F32_VEC_REDUCE(sumf, جوڙ);
    #جيڪڏهن_پورو
#وإلا
                        ثابت صحيح np = 0;
#جيڪڏهن_پورو
                        // d_state
                        لكل (صحيح i0 = np; i0 < nc; ++i0) {
                            ثابت صحيح i = i0 + ii*nc;
                            ثابت صحيح ig = i0 + g*nc;
                            // state = prev_state * dA + dB * x
                            ثابت عائم state = (s0[i] * dA) + (B[ig] * x_dt);
                            // y = rowwise_dotprod(state, C)
                            sumf += state * C[ig];
                            s[i] = state;
                        }
                        y[ii] = sumf;
                    }
                }
            } وإلا {
                // Mamba-1 has an element-wise decay factor for the states

                // n_head
                لكل (صحيح h = ih0; h < ih1; ++h) {
                    // ref: https://github.com/state-spaces/mamba/blob/62db608da60f6fc790b8ed9f4b3225e95ca15fde/mamba_ssm/ops/triton/softplus.py#L16
                    ثابت عائم dt_soft_plus = ggml_compute_softplus_f32(dt[h]);
                    ثابت صحيح g = h / (nh / ng); // repeat_interleave

                    // dim
                    لكل (صحيح i1 = 0; i1 < nr; ++i1) {
                        ثابت صحيح ii = i1 + h*nr;
                        ثابت عائم x_dt = x[ii] * dt_soft_plus;
#إذا defined(__ARM_FEATURE_SVE)
                        svfloat32_t vx_dt = GGML_F32_VEC_SET1(x_dt);
                        svfloat32_t vdt_soft_plus = GGML_F32_VEC_SET1(dt_soft_plus);
                        svfloat32_t r1_ويڪٽر = GGML_F32_VEC_ZERO;

                        // d_state
                        // TODO: what happens when (d_state % svcntw()) != 0?
                        لكل (int64_t k = 0; k < nc; k += svcntw()) {
                            svfloat32_t vA = GGML_F32_VEC_ڀريو(&A[h*nc + k]);
                            svfloat32_t vB = GGML_F32_VEC_ڀريو(&B[k + g*nc]);
                            svfloat32_t vC = GGML_F32_VEC_ڀريو(&C[k + g*nc]);
                            svfloat32_t vs0 = GGML_F32_VEC_ڀريو(&s0[ii*nc + k]);

                            svfloat32_t t1 = GGML_F32_VEC_MUL(vdt_soft_plus, vA);
                            t1 = exp_ps_sve(svptrue_b32(), t1);
                            svfloat32_t t2 = GGML_F32_VEC_MUL(vx_dt, vB);

                            vs0 = GGML_F32_VEC_FMA(t2, vs0, t1);
                            r1_ويڪٽر = GGML_F32_VEC_جوڙيو(GGML_F32_VEC_MUL(vs0, vC), r1_ويڪٽر);

                            GGML_F32_VEC_STORE(&s[ii*nc + k], vs0);
                        }
                        y[ii] = GGML_F32xt_REDUCE_ONE(r1_ويڪٽر);
#وإلا
                        عائم sumf = 0.0f;
                        // NOTE: can't really use GGML_SIMD here because d_state is usually 16
                        //       and also because expf is used within the loop.
                        // d_state
                        لكل (صحيح i0 = 0; i0 < nc; ++i0) {
                            ثابت صحيح i = i0 + ii*nc;
                            ثابت صحيح ig = i0 + g*nc;
                            // state = prev_state * dA + dB * x
                            ثابت عائم state = (s0[i] * expf(dt_soft_plus * A[i0 + h*nc])) + (B[ig] * x_dt);
                            // y = rowwise_dotprod(state, C)
                            sumf += state * C[ig];
                            s[i] = state;
                        }
                        y[ii] = sumf;
#جيڪڏهن_پورو
                    }
                }
            }
            ثابت int64_t جڳهه = nt - 1 - i2;
            إذا (K > 1 && جڳهه > 0 && جڳهه < K) {
                عائم * s_snapshot = (عائم *) ((حرف *) dst->معلومات + s_off + (جڳهه*ns + i3)*(src0->nb[3]));
                لكل (صحيح h = ih0; h < ih1; ++h) {
                    يادگيري_نقل((حرف *) s_snapshot + h*src0->nb[2], (حرف *) s + h*src0->nb[2], src0->nb[2]);
                }
            }
            // use the output as the source when it's not the first token-wise iteration
            s0 = s;
        }
    }
}

فراغ ggml_compute_forward_ssm_جاچ(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    بدل (dst->src[0]->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_ssm_جاچ_f32(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_win_part

ساكن فراغ ggml_compute_forward_win_part_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    GGML_اڻ_استعمال_ٿيل(params);

    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)

    ثابت int32_t nep0 = ((ثابت int32_t *)(dst->op_params))[0];
    ثابت int32_t nep1 = ((ثابت int32_t *)(dst->op_params))[1];
    ثابت int32_t w    = ((ثابت int32_t *)(dst->op_params))[2];

    أكد(ne00 == ne0);
    أكد(ne3  == nep0*nep1);

    // TODO: optimize / multi-thread
    لكل (صحيح py = 0; py < nep1; ++py) {
        لكل (صحيح px = 0; px < nep0; ++px) {
            ثابت int64_t i3 = py*nep0 + px;
            لكل (int64_t i2 = 0; i2 < ne2; ++i2) {
                لكل (int64_t i1 = 0; i1 < ne1; ++i1) {
                    لكل (int64_t i0 = 0; i0 < ne0; ++i0) {
                        ثابت int64_t i02 = py*w + i2;
                        ثابت int64_t i01 = px*w + i1;
                        ثابت int64_t i00 = i0;

                        ثابت int64_t i = i3*ne2*ne1*ne0 + i2*ne1*ne0    + i1*ne0   + i0;
                        ثابت int64_t j =                  i02*ne01*ne00 + i01*ne00 + i00;

                        إذا (py*w + i2 >= ne02 || px*w + i1 >= ne01) {
                            ((عائم *) dst->معلومات)[i] = 0.0f;
                        } وإلا {
                            ((عائم *) dst->معلومات)[i] = ((عائم *) src0->معلومات)[j];
                        }
                    }
                }
            }
        }
    }
}

فراغ ggml_compute_forward_win_part(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_win_part_f32(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_win_unpart

ساكن فراغ ggml_compute_forward_win_unpart_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    GGML_اڻ_استعمال_ٿيل(params);

    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)

    ثابت int32_t w = ((ثابت int32_t *)(dst->op_params))[0];

    // padding
    ثابت صحيح px = (w - ne1%w)%w;
    //const int py = (w - ne2%w)%w;

    ثابت صحيح npx = (px + ne1)/w;
    //const int npy = (py + ne2)/w;

    أكد(ne0 == ne00);

    // TODO: optimize / multi-thread
    لكل (int64_t i2 = 0; i2 < ne2; ++i2) {
        لكل (int64_t i1 = 0; i1 < ne1; ++i1) {
            لكل (int64_t i0 = 0; i0 < ne0; ++i0) {
                ثابت صحيح ip2 = i2/w;
                ثابت صحيح ip1 = i1/w;

                ثابت int64_t i02 = i2%w;
                ثابت int64_t i01 = i1%w;
                ثابت int64_t i00 = i0;

                ثابت int64_t i = (ip2*npx + ip1)*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00 + i00;
                ثابت int64_t j =                                  i2*ne1*ne0    + i1*ne0   + i0;

                ((عائم *) dst->معلومات)[j] = ((عائم *) src0->معلومات)[i];
            }
        }
    }
}

فراغ ggml_compute_forward_win_unpart(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_win_unpart_f32(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

//ggml_compute_forward_unary

فراغ ggml_compute_forward_unary(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_unary_op op = ggml_get_unary_op(dst);

    بدل (op) {
        حالة GGML_UNARY_OP_ABS:
            {
                ggml_compute_forward_abs(params, dst);
            } اكسر;
        حالة GGML_UNARY_OP_SGN:
            {
                ggml_compute_forward_sgn(params, dst);
            } اكسر;
        حالة GGML_UNARY_OP_NEG:
            {
                ggml_compute_forward_neg(params, dst);
            } اكسر;
        حالة GGML_UNARY_OP_قدم:
            {
                ggml_compute_forward_قدم(params, dst);
            } اكسر;
        حالة GGML_UNARY_OP_TANH:
            {
                ggml_compute_forward_tanh(params, dst);
            } اكسر;
        حالة GGML_UNARY_OP_ELU:
            {
                ggml_compute_forward_elu(params, dst);
            } اكسر;
        حالة GGML_UNARY_OP_RELU:
            {
                ggml_compute_forward_relu(params, dst);
            } اكسر;
        حالة GGML_UNARY_OP_SIGMOID:
            {
                ggml_compute_forward_sigmoid(params, dst);
            } اكسر;
        حالة GGML_UNARY_OP_GELU:
            {
                ggml_compute_forward_gelu(params, dst);
            } اكسر;
        حالة GGML_UNARY_OP_GELU_ERF:
            {
                ggml_compute_forward_gelu_erf(params, dst);
            } اكسر;
        حالة GGML_UNARY_OP_GELU_تيز:
            {
                ggml_compute_forward_gelu_تيز(params, dst);
            } اكسر;
        حالة GGML_UNARY_OP_SILU:
            {
                ggml_compute_forward_silu(params, dst);
            } اكسر;
        حالة GGML_UNARY_OP_HARDSWISH:
            {
                ggml_compute_forward_hardswish(params, dst);
            } اكسر;
        حالة GGML_UNARY_OP_HARDSIGMOID:
            {
                ggml_compute_forward_hardsigmoid(params, dst);
            } اكسر;
        حالة GGML_UNARY_OP_EXP:
            {
                ggml_compute_forward_exp(params, dst);
            } اكسر;
        حالة GGML_UNARY_OP_FLOOR:
            {
                ggml_compute_forward_floor(params, dst);
            } اكسر;
        حالة GGML_UNARY_OP_CEIL:
            {
                ggml_compute_forward_ceil(params, dst);
            } اكسر;
        حالة GGML_UNARY_OP_ڦيرو:
            {
                ggml_compute_forward_ڦيرو(params, dst);
            } اكسر;
        حالة GGML_UNARY_OP_TRUNC:
            {
                ggml_compute_forward_trunc(params, dst);
            } اكسر;
        حالة GGML_UNARY_OP_XIELU:
            {
                ggml_compute_forward_xielu(params, dst);
            } اكسر;
        حالة GGML_UNARY_OP_EXPM1:
            {
                ggml_compute_forward_expm1(params, dst);
            } اكسر;
        حالة GGML_UNARY_OP_SOFTPLUS:
            {
                ggml_compute_forward_softplus(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

//ggml_compute_forward_glu

فراغ ggml_compute_forward_glu(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_glu_op op = ggml_get_glu_op(dst);

    بدل (op) {
        حالة GGML_GLU_OP_REGLU:
            {
                ggml_compute_forward_reglu(params, dst);
            } اكسر;
        حالة GGML_GLU_OP_GEGLU:
            {
                ggml_compute_forward_geglu(params, dst);
            } اكسر;
        حالة GGML_GLU_OP_SWIGLU:
            {
                ggml_compute_forward_swiglu(params, dst);
            } اكسر;
        حالة GGML_GLU_OP_SWIGLU_OAI:
            {
                ggml_compute_forward_swiglu_oai(params, dst);
            } اكسر;
        حالة GGML_GLU_OP_GEGLU_ERF:
            {
                ggml_compute_forward_geglu_erf(params, dst);
            } اكسر;
        حالة GGML_GLU_OP_GEGLU_تيز:
            {
                ggml_compute_forward_geglu_تيز(params, dst);
            } اكسر;
        حالة GGML_GLU_OP_SWIGLU_CLAMP:
            {
                ggml_compute_forward_swiglu_clamp(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_get_rel_pos

ساكن فراغ ggml_compute_forward_get_rel_pos_f16(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    GGML_اڻ_استعمال_ٿيل(params);

    ثابت ggml_tensor * src0 = dst->src[0];

    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L292-L322

    GGML_TENSOR_UNARY_OP_LOCALS

    ثابت int64_t w = ne1;

    ggml_fp16_t * src0_معلومات = (ggml_fp16_t *) src0->معلومات;
    ggml_fp16_t * dst_معلومات  = (ggml_fp16_t *) dst->معلومات;

    لكل (int64_t i2 = 0; i2 < ne2; ++i2) {
        لكل (int64_t i1 = 0; i1 < ne1; ++i1) {
            ثابت int64_t pos = (w - i1 - 1) + i2;
            لكل (int64_t i0 = 0; i0 < ne0; ++i0) {
                dst_معلومات[i2*ne1*ne0 + i1*ne0 + i0] = src0_معلومات[pos*ne00 + i0];
            }
        }
    }
}

فراغ ggml_compute_forward_get_rel_pos(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F16:
        حالة GGML_TYPE_BF16:
            {
                ggml_compute_forward_get_rel_pos_f16(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_add_rel_pos

ساكن فراغ ggml_compute_forward_جوڙيو_rel_pos_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    ثابت ggml_tensor * src2 = dst->src[2];

    ثابت منطقي inplace = (منطقي) ((int32_t *) dst->op_params)[0];
    إذا (!inplace) {
        إذا (params->ith == 0) {
            يادگيري_نقل((حرف *) dst->معلومات, (حرف *) src0->معلومات, ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }
    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L357-L359

    عائم * src1_معلومات = (عائم *) src1->معلومات;
    عائم * src2_معلومات = (عائم *) src2->معلومات;
    عائم * dst_معلومات  = (عائم *) dst->معلومات;

    ثابت int64_t ne10 = src1->ne[0];
    ثابت int64_t ne11 = src1->ne[1];
    ثابت int64_t ne12 = src1->ne[2];
    ثابت int64_t ne13 = src1->ne[3];

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    // total patches in dst
    ثابت صحيح np = ne13;

    // patches per thread
    ثابت صحيح dp = (np + nth - 1)/nth;

    // patch range for this thread
    ثابت صحيح ip0 = dp*ith;
    ثابت صحيح ip1 = MIN(ip0 + dp, np);

    لكل (int64_t i13 = ip0; i13 < ip1; ++i13) {
        لكل (int64_t i12 = 0; i12 < ne12; ++i12) {
            لكل (int64_t i11 = 0; i11 < ne11; ++i11) {
                ثابت int64_t jp1 = i13*ne12*ne11*ne10 + i12*ne11*ne10 + i11*ne10;
                لكل (int64_t i10 = 0; i10 < ne10; ++i10) {
                    ثابت int64_t jp0  = jp1 + i10;
                    ثابت عائم src1_e = src1_معلومات[jp0];
                    ثابت عائم src2_e = src2_معلومات[jp0];

                    ثابت int64_t jdh = jp0 * ne10;
                    ثابت int64_t jdw = jdh - (ne10 - 1) * i10;

                    لكل (int64_t j = 0; j < ne10; ++j) {
                        dst_معلومات[jdh + j     ] += src2_e;
                        dst_معلومات[jdw + j*ne10] += src1_e;
                    }
                }
            }
        }
    }
}

فراغ ggml_compute_forward_جوڙيو_rel_pos(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_جوڙيو_rel_pos_f32(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_rwkv_wkv6

ساكن فراغ ggml_compute_forward_rwkv_wkv6_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    ثابت int64_t T = dst->src[1]->ne[2];
    ثابت int64_t C = dst->ne[0];
    ثابت int64_t HEADS = dst->src[1]->ne[1];
    ثابت int64_t n_seqs = dst->src[5]->ne[1];
    ثابت int64_t سر_سائيز = C / HEADS;

    عائم * dst_معلومات = (عائم *) dst->معلومات;
    عائم * state = ((عائم *) dst->معلومات) + C * T;

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح h_شروعات =  (HEADS * (ith    )) / nth;
    ثابت صحيح h_پڇاڙي   = ((HEADS * (ith + 1)) / nth < HEADS) ?
                         (HEADS * (ith + 1)) / nth : HEADS;

    عائم * k =          (عائم *) dst->src[0]->معلومات;
    عائم * v =          (عائم *) dst->src[1]->معلومات;
    عائم * r =          (عائم *) dst->src[2]->معلومات;
    عائم * وقت_faaaa = (عائم *) dst->src[3]->معلومات;
    عائم * وقت_decay = (عائم *) dst->src[4]->معلومات;

    size_t t_stride = HEADS * سر_سائيز; // Same to C

    size_t h_stride = C / HEADS;
    GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
    size_t h_stride_2d = سر_سائيز * سر_سائيز;

    إذا (ith == 0) {
        memset(dst_معلومات, 0, T * C * حجم(عائم));
    }
    ggml_barrier(params->threadpool);


    #إذا defined(__AVX__) && !defined(__AVX512F__)
        #وصف_ڪريو GGML_F32X GGML_F32x8
        #وصف_ڪريو GGML_F32X_SET1 GGML_F32x8_SET1
        #وصف_ڪريو GGML_F32_X_ڀريو GGML_F32x8_ڀريو
        #وصف_ڪريو GGML_F32X_STORE GGML_F32x8_STORE
        #وصف_ڪريو GGML_F32X_MUL GGML_F32x8_MUL
        #وصف_ڪريو GGML_F32X_FMA GGML_F32x8_FMA
        #وصف_ڪريو WKV_ويڪٽر_سائيز 8
    #وإلاإذا defined(__AVX512F__)
        #وصف_ڪريو GGML_F32X GGML_F32x16
        #وصف_ڪريو GGML_F32X_SET1 GGML_F32x16_SET1
        #وصف_ڪريو GGML_F32_X_ڀريو GGML_F32x16_ڀريو
        #وصف_ڪريو GGML_F32X_STORE GGML_F32x16_STORE
        #وصف_ڪريو GGML_F32X_MUL GGML_F32x16_MUL
        #وصف_ڪريو GGML_F32X_FMA GGML_F32x16_FMA
        #وصف_ڪريو WKV_ويڪٽر_سائيز 16
    #وإلاإذا defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
        #وصف_ڪريو GGML_F32X GGML_F32xt
        #وصف_ڪريو GGML_F32X_SET1 GGML_F32xt_SET1
        #وصف_ڪريو GGML_F32_X_ڀريو GGML_F32xt_ڀريو
        #وصف_ڪريو GGML_F32X_STORE GGML_F32xt_STORE
        #وصف_ڪريو GGML_F32X_MUL GGML_F32xt_MUL
        #وصف_ڪريو GGML_F32X_FMA GGML_F32xt_FMA
        #وصف_ڪريو WKV_ويڪٽر_سائيز 8
    #وإلاإذا defined(__ARM_NEON) && defined(__aarch64__)
        #وصف_ڪريو GGML_F32X GGML_F32x4
        #وصف_ڪريو GGML_F32X_SET1 GGML_F32x4_SET1
        #وصف_ڪريو GGML_F32_X_ڀريو GGML_F32x4_ڀريو
        #وصف_ڪريو GGML_F32X_STORE GGML_F32x4_STORE
        #وصف_ڪريو GGML_F32X_MUL GGML_F32x4_MUL
        #وصف_ڪريو GGML_F32X_FMA GGML_F32x4_FMA
        #وصف_ڪريو WKV_ويڪٽر_سائيز 4
    #جيڪڏهن_پورو

    #جيڪڏهن_وصف_ٿيل WKV_ويڪٽر_سائيز
        صحيح wkv_ويڪٽر_سائيز;
        #إذا defined(__ARM_FEATURE_SVE)
            wkv_ويڪٽر_سائيز = svcntw();
        #وإلا
            wkv_ويڪٽر_سائيز = WKV_ويڪٽر_سائيز;
        #جيڪڏهن_پورو
        ثابت int64_t vec_ڳڻپ = سر_سائيز / wkv_ويڪٽر_سائيز;

        لكل (int64_t t = 0; t < T; t++) {
            size_t t_وٿي = t * t_stride;
            size_t state_وٿي = سر_سائيز * C * (t / (T / n_seqs));
            عائم * state_cur = state + state_وٿي;
            عائم * state_prev = t % (T / n_seqs) ? state_cur : (عائم*)dst->src[5]->معلومات + state_وٿي;

            لكل (int64_t h = h_شروعات; h < h_پڇاڙي; h++) {
                size_t h_وٿي = h * h_stride;
                size_t t_h_وٿي = t_وٿي + h_وٿي;
                size_t h_2d_وٿي = h * h_stride_2d;

                لكل (int64_t i = 0; i < سر_سائيز; i++) {
                    size_t t_h_i_وٿي = t_h_وٿي + i;
                    size_t h_i_وٿي = h_وٿي + i;
                    size_t h_2d_i_وٿي = h_2d_وٿي + i * h_stride;

                    عائم k_val = k[t_h_i_وٿي];
                    عائم r_val = r[t_h_i_وٿي];
                    عائم وقت_faaaa_val = وقت_faaaa[h_i_وٿي];
                    عائم وقت_decay_val = وقت_decay[t_h_i_وٿي];

                    // Broadcast scalar values to vectors
                    GGML_F32X k_vec = GGML_F32X_SET1(k_val);
                    GGML_F32X r_vec = GGML_F32X_SET1(r_val);
                    GGML_F32X وقت_faaaa_vec = GGML_F32X_SET1(وقت_faaaa_val);
                    GGML_F32X وقت_decay_vec = GGML_F32X_SET1(وقت_decay_val);

                    لكل (int64_t j = 0; j < vec_ڳڻپ; j++) {
                        size_t بنياد_j = j * wkv_ويڪٽر_سائيز;
                        size_t t_h_j_وٿي = t_h_وٿي + بنياد_j;
                        size_t h_2d_i_j_وٿي = h_2d_i_وٿي + بنياد_j;

                        // Load x elements at once
                        GGML_F32X v_vec = GGML_F32_X_ڀريو(&v[t_h_j_وٿي]);
                        GGML_F32X prev_state_vec = GGML_F32_X_ڀريو(&state_prev[h_2d_i_j_وٿي]);
                        GGML_F32X dst_vec = GGML_F32_X_ڀريو(&dst_معلومات[t_h_j_وٿي]);

                        // Compute kv = v * k
                        GGML_F32X kv_vec = GGML_F32X_MUL(v_vec, k_vec);

                        // Compute temp = kv * time_faaaa + prev_state
                        GGML_F32X عارضي_vec = GGML_F32X_FMA(prev_state_vec, kv_vec, وقت_faaaa_vec);

                        // Update dst: dst += temp * r
                        dst_vec = GGML_F32X_FMA(dst_vec, عارضي_vec, r_vec);
                        GGML_F32X_STORE(&dst_معلومات[t_h_j_وٿي], dst_vec);

                        // Update state: state = prev_state * time_decay + kv
                        GGML_F32X new_state_vec = GGML_F32X_FMA(kv_vec, prev_state_vec, وقت_decay_vec);
                        GGML_F32X_STORE(&state_cur[h_2d_i_j_وٿي], new_state_vec);
                    }

                    // Handle remaining elements, this will not be used.
                    لكل (int64_t j = vec_ڳڻپ * wkv_ويڪٽر_سائيز; j < سر_سائيز; j++) {
                        size_t t_h_j_وٿي = t_h_وٿي + j;
                        size_t h_2d_i_j_وٿي = h_2d_i_وٿي + j;
                        عائم v_val = v[t_h_j_وٿي];
                        عائم kv_val = v_val * k_val;
                        عائم prev_state_val = state_prev[h_2d_i_j_وٿي];
                        عائم عارضي_val = kv_val * وقت_faaaa_val + prev_state_val;
                        dst_معلومات[t_h_j_وٿي] += عارضي_val * r_val;
                        state_cur[h_2d_i_j_وٿي] = prev_state_val * وقت_decay_val + kv_val;
                    }
                }
            }
        }

    #وإلا
        // basically fused operations:
        // dst = r @ (time_faaaa * (k @ v) + state),
        // state = time_decay * state + (k @ v),
        // recursive through each token
        لكل (int64_t t = 0; t < T; t++) {
            size_t t_وٿي = t * t_stride;
            size_t state_وٿي = سر_سائيز * C * (t / (T / n_seqs));
            عائم * state_cur = state + state_وٿي;
            عائم * state_prev = t % (T / n_seqs) ? state_cur : (عائم*)dst->src[5]->معلومات + state_وٿي;

            لكل (int64_t h = h_شروعات; h < h_پڇاڙي; h++) {
                size_t h_وٿي = h * h_stride;
                size_t t_h_وٿي = t_وٿي + h_وٿي;
                size_t h_2d_وٿي = h * h_stride_2d;

                لكل (int64_t i = 0; i < سر_سائيز; i++) {
                    size_t t_h_i_وٿي = t_h_وٿي + i;
                    size_t h_i_وٿي = h_وٿي + i;
                    size_t h_2d_i_وٿي = h_2d_وٿي + i * h_stride;

                    عائم k_val = k[t_h_i_وٿي];
                    عائم r_val = r[t_h_i_وٿي];
                    عائم وقت_faaaa_val = وقت_faaaa[h_i_وٿي];
                    // RWKV v6: different time_decay for each token.
                    عائم وقت_decay_val = وقت_decay[t_h_i_وٿي];

                    لكل (int64_t j = 0; j < سر_سائيز; j++) {
                        size_t t_h_j_وٿي = t_h_وٿي + j;
                        size_t h_2d_i_j_وٿي = h_2d_i_وٿي + j;

                        عائم v_val = v[t_h_j_وٿي];
                        عائم kv_val = v_val * k_val;
                        عائم prev_state_val = state_prev[h_2d_i_j_وٿي];
                        عائم عارضي_val = kv_val * وقت_faaaa_val + prev_state_val;
                        dst_معلومات[t_h_j_وٿي] += عارضي_val * r_val;
                        state_cur[h_2d_i_j_وٿي] = prev_state_val * وقت_decay_val + kv_val;
                    }
                }
            }
        }
    #جيڪڏهن_پورو
}


فراغ ggml_compute_forward_rwkv_wkv6(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_rwkv_wkv6_f32(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_gla

ساكن فراغ ggml_compute_forward_gla_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    ثابت int64_t T = dst->src[1]->ne[2];
    ثابت int64_t C = dst->ne[0];
    ثابت int64_t HEADS = dst->src[1]->ne[1];
    ثابت int64_t n_seqs = dst->src[4]->ne[1];
    ثابت int64_t سر_سائيز = C / HEADS;
    ثابت عائم scale = ggml_get_op_params_f32(dst, 0);

    عائم * dst_معلومات = (عائم *) dst->معلومات;
    عائم * state = ((عائم *) dst->معلومات) + C * T;

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح h_شروعات =  (HEADS * (ith    )) / nth;
    ثابت صحيح h_پڇاڙي   = ((HEADS * (ith + 1)) / nth < HEADS) ?
                         (HEADS * (ith + 1)) / nth : HEADS;

    عائم * k = (عائم *) dst->src[0]->معلومات;
    عائم * v = (عائم *) dst->src[1]->معلومات;
    عائم * q = (عائم *) dst->src[2]->معلومات;
    عائم * g = (عائم *) dst->src[3]->معلومات;

    size_t t_stride = HEADS * سر_سائيز; // Same to C

    size_t h_stride = C / HEADS;
    GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
    size_t h_stride_2d = سر_سائيز * سر_سائيز;

    إذا (ith == 0) {
        memset(dst_معلومات, 0, T * C * حجم(عائم));
    }
    ggml_barrier(params->threadpool);


    #إذا defined(__AVX__) && !defined(__AVX512F__)
        #وصف_ڪريو GGML_F32X GGML_F32x8
        #وصف_ڪريو GGML_F32X_SET1 GGML_F32x8_SET1
        #وصف_ڪريو GGML_F32_X_ڀريو GGML_F32x8_ڀريو
        #وصف_ڪريو GGML_F32X_STORE GGML_F32x8_STORE
        #وصف_ڪريو GGML_F32X_MUL GGML_F32x8_MUL
        #وصف_ڪريو GGML_F32X_FMA GGML_F32x8_FMA
        #وصف_ڪريو GLA_ويڪٽر_سائيز 8
    #وإلاإذا defined(__AVX512F__)
        #وصف_ڪريو GGML_F32X GGML_F32x16
        #وصف_ڪريو GGML_F32X_SET1 GGML_F32x16_SET1
        #وصف_ڪريو GGML_F32_X_ڀريو GGML_F32x16_ڀريو
        #وصف_ڪريو GGML_F32X_STORE GGML_F32x16_STORE
        #وصف_ڪريو GGML_F32X_MUL GGML_F32x16_MUL
        #وصف_ڪريو GGML_F32X_FMA GGML_F32x16_FMA
        #وصف_ڪريو GLA_ويڪٽر_سائيز 16
    #وإلاإذا defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
        #وصف_ڪريو GGML_F32X GGML_F32xt
        #وصف_ڪريو GGML_F32X_SET1 GGML_F32xt_SET1
        #وصف_ڪريو GGML_F32_X_ڀريو GGML_F32xt_ڀريو
        #وصف_ڪريو GGML_F32X_STORE GGML_F32xt_STORE
        #وصف_ڪريو GGML_F32X_MUL GGML_F32xt_MUL
        #وصف_ڪريو GGML_F32X_FMA GGML_F32xt_FMA
        #وصف_ڪريو GLA_ويڪٽر_سائيز 8
    #وإلاإذا defined(__ARM_NEON) && defined(__aarch64__)
        #وصف_ڪريو GGML_F32X GGML_F32x4
        #وصف_ڪريو GGML_F32X_SET1 GGML_F32x4_SET1
        #وصف_ڪريو GGML_F32_X_ڀريو GGML_F32x4_ڀريو
        #وصف_ڪريو GGML_F32X_STORE GGML_F32x4_STORE
        #وصف_ڪريو GGML_F32X_MUL GGML_F32x4_MUL
        #وصف_ڪريو GGML_F32X_FMA GGML_F32x4_FMA
        #وصف_ڪريو GLA_ويڪٽر_سائيز 4
    #جيڪڏهن_پورو

    #جيڪڏهن_وصف_ٿيل GLA_ويڪٽر_سائيز
        صحيح gla_ويڪٽر_سائيز;
        #إذا defined(__ARM_FEATURE_SVE)
            gla_ويڪٽر_سائيز = svcntw();
        #وإلا
            gla_ويڪٽر_سائيز = GLA_ويڪٽر_سائيز;
        #جيڪڏهن_پورو
        ثابت int64_t vec_ڳڻپ = سر_سائيز / gla_ويڪٽر_سائيز;

        لكل (int64_t t = 0; t < T; t++) {
            size_t t_وٿي = t * t_stride;
            size_t state_وٿي = سر_سائيز * C * (t / (T / n_seqs));
            عائم * state_cur = state + state_وٿي;
            عائم * state_prev = t % (T / n_seqs) ? state_cur : (عائم*)dst->src[4]->معلومات + state_وٿي;

            لكل (int64_t h = h_شروعات; h < h_پڇاڙي; h++) {
                size_t h_وٿي = h * h_stride;
                size_t t_h_وٿي = t_وٿي + h_وٿي;
                size_t h_2d_وٿي = h * h_stride_2d;

                لكل (int64_t i = 0; i < سر_سائيز; i++) {
                    size_t t_h_i_وٿي = t_h_وٿي + i;
                    size_t h_2d_i_وٿي = h_2d_وٿي + i * h_stride;

                    عائم k_val = k[t_h_i_وٿي];
                    عائم q_val = q[t_h_i_وٿي] * scale;
                    عائم g_val = g[t_h_i_وٿي];

                    // Broadcast scalar values to vectors
                    GGML_F32X k_vec = GGML_F32X_SET1(k_val);
                    GGML_F32X q_vec = GGML_F32X_SET1(q_val);
                    GGML_F32X g_vec = GGML_F32X_SET1(g_val);

                    لكل (int64_t j = 0; j < vec_ڳڻپ; j++) {
                        size_t بنياد_j = j * gla_ويڪٽر_سائيز;
                        size_t t_h_j_وٿي = t_h_وٿي + بنياد_j;
                        size_t h_2d_i_j_وٿي = h_2d_i_وٿي + بنياد_j;

                        // Load x elements at once
                        GGML_F32X v_vec = GGML_F32_X_ڀريو(&v[t_h_j_وٿي]);
                        GGML_F32X prev_state_vec = GGML_F32_X_ڀريو(&state_prev[h_2d_i_j_وٿي]);
                        GGML_F32X dst_vec = GGML_F32_X_ڀريو(&dst_معلومات[t_h_j_وٿي]);

                        // Compute kv = v * k
                        GGML_F32X kv_vec = GGML_F32X_MUL(v_vec, k_vec);

                        // Compute temp = prev_state * g + kv
                        GGML_F32X عارضي_vec = GGML_F32X_FMA(kv_vec, prev_state_vec, g_vec);

                        // Update dst: dst += temp * q
                        dst_vec = GGML_F32X_FMA(dst_vec, عارضي_vec, q_vec);
                        GGML_F32X_STORE(&dst_معلومات[t_h_j_وٿي], dst_vec);

                        // Update state
                        GGML_F32X_STORE(&state_cur[h_2d_i_j_وٿي], عارضي_vec);
                    }

                    // Handle remaining elements, this will not be used.
                    لكل (int64_t j = vec_ڳڻپ * gla_ويڪٽر_سائيز; j < سر_سائيز; j++) {
                        size_t t_h_j_وٿي = t_h_وٿي + j;
                        size_t h_2d_i_j_وٿي = h_2d_i_وٿي + j;
                        عائم v_val = v[t_h_j_وٿي];
                        عائم kv_val = v_val * k_val;
                        عائم prev_state_val = state_prev[h_2d_i_j_وٿي];
                        عائم عارضي_val = kv_val + prev_state_val * g_val;
                        dst_معلومات[t_h_j_وٿي] += عارضي_val * q_val;
                        state_cur[h_2d_i_j_وٿي] = عارضي_val;
                    }
                }
            }
        }

    #وإلا
        لكل (int64_t t = 0; t < T; t++) {
            size_t t_وٿي = t * t_stride;
            size_t state_وٿي = سر_سائيز * C * (t / (T / n_seqs));
            عائم * state_cur = state + state_وٿي;
            عائم * state_prev = t % (T / n_seqs) ? state_cur : (عائم*)dst->src[4]->معلومات + state_وٿي;

            لكل (int64_t h = h_شروعات; h < h_پڇاڙي; h++) {
                size_t h_وٿي = h * h_stride;
                size_t t_h_وٿي = t_وٿي + h_وٿي;
                size_t h_2d_وٿي = h * h_stride_2d;

                لكل (int64_t i = 0; i < سر_سائيز; i++) {
                    size_t t_h_i_وٿي = t_h_وٿي + i;
                    size_t h_2d_i_وٿي = h_2d_وٿي + i * h_stride;

                    عائم k_val = k[t_h_i_وٿي];
                    عائم q_val = q[t_h_i_وٿي] * scale;
                    عائم g_val = g[t_h_i_وٿي];

                    لكل (int64_t j = 0; j < سر_سائيز; j++) {
                        size_t t_h_j_وٿي = t_h_وٿي + j;
                        size_t h_2d_i_j_وٿي = h_2d_i_وٿي + j;

                        عائم v_val = v[t_h_j_وٿي];
                        عائم kv_val = v_val * k_val;
                        عائم prev_state_val = state_prev[h_2d_i_j_وٿي];
                        عائم عارضي_val = prev_state_val * g_val + kv_val;
                        dst_معلومات[t_h_j_وٿي] += عارضي_val * q_val;
                        state_cur[h_2d_i_j_وٿي] = عارضي_val;
                    }
                }
            }
        }
    #جيڪڏهن_پورو
}


فراغ ggml_compute_forward_gla(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_gla_f32(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ساكن فراغ ggml_compute_forward_solve_tri_f32(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst) {
    ثابت بنية ggml_tensor * src0 = dst->src[0];  // A (lower triangular)
    ثابت بنية ggml_tensor * src1 = dst->src[1];  // B (RHS)

    GGML_TENSOR_ٻٽو_OP_LOCALS;

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(ne00 == ne01); // A must be square
    GGML_ASSERT(ne0  == ne10); // solution cols == B cols
    GGML_ASSERT(ne1  == ne11); // solution rows == B rows

    GGML_ASSERT(ne02 == ne12 && ne12 == ne2);
    GGML_ASSERT(ne03 == ne13 && ne13 == ne3);

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت int64_t k = ne10;   // number of RHS columns
    ثابت int64_t n = ne11;   // A is n×n
    ثابت int64_t nr = ne02 * ne03 * k; // we're parallelizing on columns here, so seq x token x column will be the unit

    // chunks per thread
    ثابت int64_t dr = (nr + nth - 1)/nth;

    // chunk range for this thread
    ثابت int64_t ir0 = dr*ith;
    ثابت int64_t ir1 = MIN(ir0 + dr, nr);

    ثابت عائم * A = (ثابت عائم *) src0->معلومات;  // [n, n, B1, B2]
    ثابت عائم * B = (ثابت عائم *) src1->معلومات;  // [n, k, B1, B2]
          عائم * X = (      عائم *) dst->معلومات;   // [n, k, B1, B2]

    لكل (int64_t ir = ir0; ir < ir1; ++ir) {
        ثابت int64_t i03 = ir/(ne02*k);
        ثابت int64_t i02 = (ir - i03*ne02*k)/k;
        ثابت int64_t i01 = (ir - i03*ne02*k - i02*k);

        ثابت عائم * A_batch = A + i02 * nb02 / حجم(عائم) + i03 * nb03 / حجم(عائم);
        ثابت عائم * B_batch = B + i02 * nb12 / حجم(عائم) + i03 * nb13 / حجم(عائم);

        عائم * X_batch = X + i02 * nb2 / حجم(عائم) + i03 * nb3 / حجم(عائم);

        لكل (int64_t i00 = 0; i00 < n; ++i00) {
            عائم جوڙ = 0.0f;
            لكل (int64_t t = 0; t < i00; ++t) {
                جوڙ += A_batch[i00 * n + t] * X_batch[t * k + i01];
            }

            ثابت عائم diag = A_batch[i00 * n + i00];
            أكد(diag != 0.0f && "Zero diagonal in triangular matrix");

            X_batch[i00 * k + i01] = (B_batch[i00 * k + i01] - جوڙ) / diag;
        }
    }
}

فراغ ggml_compute_forward_solve_tri(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    إذا (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        ggml_compute_forward_solve_tri_f32(params, dst);
    } وإلا {
        GGML_ABORT("fatal error");
    }
}

// ggml_compute_forward_gated_delta_net
ساكن فراغ ggml_compute_forward_gated_delta_net_one_chunk(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst,
    int64_t ir0,
    int64_t ir1) {

    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    ثابت int64_t S_v      = src_v->ne[0];
    ثابت int64_t H        = src_v->ne[1];
    ثابت int64_t n_ٽوڪن = src_v->ne[2];
    ثابت int64_t n_seqs   = src_v->ne[3];

    GGML_ASSERT(ggml_is_contiguous_صفون(src_q));
    GGML_ASSERT(ggml_is_contiguous_صفون(src_k));
    GGML_ASSERT(ggml_is_contiguous_صفون(src_v));
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

    ثابت منطقي kda = (neg0 == S_v);

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    ثابت int64_t K = ggml_get_op_params_i32(dst, 0);
    GGML_ASSERT(K >= 1);
    // per-seq stride in floats (seq s starts at state + s * seq_stride)
    ثابت int64_t state_seq_stride = src_state->nb[3] / حجم(عائم);

    ثابت int64_t في_thread = S_v + (K > 1 ? S_v * S_v : 0);
    ثابت صحيح ith = params->ith;

    عائم * delta       = (عائم *)params->wdata + ith * في_thread + CACHE_سٽ_سائيز_F32;
    عائم * state_work  = K > 1 ? (delta + S_v) : خالي_اشارو;

    // output layout: [attn_scores | new_states]
    // attn_scores: S_v * H * n_tokens * n_seqs    floats
    // new_states:  S_v * S_v * H * n_seqs * K     floats  (K snapshot slots; last min(n_tokens, K))
    ثابت int64_t attn_score_elems    = S_v * H * n_ٽوڪن * n_seqs;
    ثابت int64_t state_سائيز_في_snap = S_v * S_v * H * n_seqs;
    عائم * attn_ٻاهريون_بنياد  = (عائم *)dst->معلومات;
    عائم * state_ٻاهريون_بنياد = (عائم *)dst->معلومات + attn_score_elems;

    // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
    // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.

    ثابت عائم * state_اندريون_بنياد = (ثابت عائم *)src_state->معلومات;

  //const int64_t rq1 = nev1 / neq1;
  //const int64_t rk1 = nev1 / nek1;
    ثابت int64_t rq3 = nev3 / neq3;
    ثابت int64_t rk3 = nev3 / nek3;

    ثابت عائم scale = 1.0f / sqrtf((عائم) S_v);

    لكل (int64_t ir = ir0; ir < ir1; ++ir) {
        ثابت int64_t iv1 = ir % H; // head_index
        ثابت int64_t iv3 = ir / H; // sequence

        ثابت int64_t iq1 = iv1 % neq1;
        ثابت int64_t ik1 = iv1 % nek1;

        ثابت int64_t iq3 = iv3 / rq3;
        ثابت int64_t ik3 = iv3 / rk3;

        // For K=1, write directly to the single output slot to avoid an extra memcpy at the end.
        // For K>1, work in scratch and copy out per-token when the slot is in range.
        عائم * s_ٻاهريون = (K > 1)
            ? state_work
            : state_ٻاهريون_بنياد + (iv3 * H + iv1) * S_v * S_v;

        // copy input state into the working buffer and operate in-place
        // state layout [S_v, S_v, H, n_seqs]: seq iv3 starts at iv3 * state_seq_stride.
        ثابت عائم * s_اندريون = state_اندريون_بنياد + iv3 * state_seq_stride + iv1 * S_v * S_v;
        يادگيري_نقل(s_ٻاهريون, s_اندريون, S_v * S_v * حجم(عائم));

        // attn output pointer for first token of this (head, seq)
        عائم * attn_معلومات = attn_ٻاهريون_بنياد + (iv3 * n_ٽوڪن * H + iv1) * S_v;

        لكل (int64_t t = 0; t < n_ٽوڪن; t++) {
            ثابت عائم * q_d = (ثابت عائم *)((ثابت حرف *)src_q->معلومات + iq3 * nbq3 + t * nbq2 + iq1 * nbq1);
            ثابت عائم * k_d = (ثابت عائم *)((ثابت حرف *)src_k->معلومات + ik3 * nbk3 + t * nbk2 + ik1 * nbk1);
            ثابت عائم * v_d = (ثابت عائم *)((ثابت حرف *)src_v->معلومات + iv3 * nbv3 + t * nbv2 + iv1 * nbv1);

            ثابت عائم beta_val = *(ثابت عائم *)((ثابت حرف *)src_beta->معلومات + iv3 * nbb3 + t * nbb2 + iv1 * nbb1);
            ثابت عائم * g_d    =  (ثابت عائم *)((ثابت حرف *)src_g->معلومات    + iv3 * nbg3 + t * nbg2 + iv1 * nbg1);

            // state is stored transposed: s_out[j*S_v + i] = S[i][j]
            // so row j of s_out = column j of S (contiguous access)

            إذا (kda) {
                // precompute exp(g) into delta scratch (reused below)
                لكل (int64_t i = 0; i < S_v; ++i) {
                    delta[i] = expf(g_d[i]);
                }
                // S[i][:] *= exp(g[i]) => for each row j of M: M[j][i] *= exp(g[i])
                لكل (int64_t j = 0; j < S_v; ++j) {
                    ggml_vec_mul_f32(S_v, &s_ٻاهريون[j * S_v], &s_ٻاهريون[j * S_v], delta);
                }
            } وإلا {
                ggml_vec_scale_f32(S_v * S_v, s_ٻاهريون, expf(g_d[0]));
            }

            // delta[j] = sum_i S[i][j] * k[i] = dot(row j of M, k)
            لكل (int64_t j = 0; j < S_v; ++j) {
                عائم جوڙ = 0.0f;
                ggml_vec_dot_f32(S_v, &جوڙ, 0, &s_ٻاهريون[j * S_v], 0, k_d, 0, 1);
                delta[j] = (v_d[j] - جوڙ) * beta_val;
            }

            // outer product: S[i][j] += k[i] * delta[j] => M[j][i] += delta[j] * k[i]
            لكل (int64_t j = 0; j < S_v; ++j) {
                ggml_vec_mad_f32(S_v, &s_ٻاهريون[j * S_v], k_d, delta[j]);
            }

            // attn_out[j] = sum_i S[i][j] * q[i] = dot(row j of M, q)
            لكل (int64_t j = 0; j < S_v; ++j) {
                عائم جوڙ = 0.0f;
                ggml_vec_dot_f32(S_v, &جوڙ, 0, &s_ٻاهريون[j * S_v], 0, q_d, 0, 1);
                attn_معلومات[j] = جوڙ * scale;
            }

            attn_معلومات += S_v * H; // advance to next token

            إذا (K > 1) {
                ثابت int64_t هدف_جڳهه = n_ٽوڪن - 1 - t;
                إذا (هدف_جڳهه >= 0 && هدف_جڳهه < K) {
                    عائم * curr_state_o = state_ٻاهريون_بنياد + هدف_جڳهه * state_سائيز_في_snap +
                                     (iv3 * H + iv1) * S_v * S_v;
                    يادگيري_نقل(curr_state_o, s_ٻاهريون, S_v * S_v * حجم(عائم));
                }
            }
        }
    }
}


ساكن فراغ ggml_compute_forward_gated_delta_net_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ggml_tensor * V = dst->src[2];
    int64_t nr = V->ne[1] * V->ne[3];

    // disable for NUMA
    ثابت منطقي disable_chunking = ggml_is_numa();

    صحيح nth = params->nth;
    صحيح ith = params->ith;

    // 4x chunks per thread
    صحيح nth_scaled = nth * 4;
    int64_t chunk_سائيز = (nr + nth_scaled - 1) / nth_scaled;
    int64_t nchunk     = (nr + chunk_سائيز - 1) / chunk_سائيز;

    إذا (nth == 1 || nchunk < nth || disable_chunking) {
      nchunk = nth;
    }

    إذا (ith == 0) {
      ggml_threadpool_chunk_سيٽ(params->threadpool, nth);
    }

    ggml_barrier(params->threadpool);

    ثابت int64_t dr = (nr + nchunk - 1) / nchunk;

    صحيح current_chunk = ith;

    طالما (current_chunk < nchunk) {
        ثابت int64_t ir0 = dr * current_chunk;
        ثابت int64_t ir1 = MIN(ir0 + dr, nr);

        ggml_compute_forward_gated_delta_net_one_chunk(params, dst, ir0, ir1);
        current_chunk = ggml_threadpool_chunk_جوڙيو(params->threadpool, 1);
    }
}

فراغ ggml_compute_forward_gated_delta_net(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_gated_delta_net_f32(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}


// ggml_compute_forward_dsv4_hc_comb

ساكن فراغ ggml_dsv4_hc_comb_norm_کانن_ڳڻپ(عائم * comb, عائم eps) {
    constexpr int64_t hc = 4;

    لكل (int64_t idst = 0; idst < hc; ++idst) {
        عائم جوڙ = eps;
        لكل (int64_t isrc = 0; isrc < hc; ++isrc) {
            جوڙ += comb[idst + hc*isrc];
        }

        ثابت عائم inv_جوڙ = 1.0f / جوڙ;
        لكل (int64_t isrc = 0; isrc < hc; ++isrc) {
            comb[idst + hc*isrc] *= inv_جوڙ;
        }
    }
}

ساكن فراغ ggml_dsv4_hc_comb_norm_صفون(عائم * comb, عائم eps) {
    constexpr int64_t hc = 4;

    لكل (int64_t isrc = 0; isrc < hc; ++isrc) {
        عائم جوڙ = eps;
        لكل (int64_t idst = 0; idst < hc; ++idst) {
            جوڙ += comb[idst + hc*isrc];
        }

        ثابت عائم inv_جوڙ = 1.0f / جوڙ;
        لكل (int64_t idst = 0; idst < hc; ++idst) {
            comb[idst + hc*isrc] *= inv_جوڙ;
        }
    }
}

ساكن فراغ ggml_compute_forward_dsv4_hc_comb_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    ثابت ggml_tensor * mixes = dst->src[0];
    ثابت ggml_tensor * scale = dst->src[1];
    ثابت ggml_tensor * بنياد  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(بنياد->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    constexpr int64_t hc = 4;
    constexpr int64_t comb_وٿي = 2*hc;
    constexpr int64_t hc_mix_dim = (2 + hc)*hc;

    ثابت int64_t n_ٽوڪن = mixes->ne[1];

    GGML_ASSERT(mixes->ne[0] == hc_mix_dim);
    GGML_ASSERT(dst->ne[0] == hc);
    GGML_ASSERT(dst->ne[1] == hc);
    GGML_ASSERT(dst->ne[2] == n_ٽوڪن);
    GGML_ASSERT(scale->ne[0] >= 3);
    GGML_ASSERT(بنياد->ne[0] == hc_mix_dim);

    GGML_TENSOR_LOCALS(size_t, nbm, mixes, nb);
    GGML_TENSOR_LOCALS(size_t, nbs, scale, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, بنياد,  nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,   nb);

    ثابت عائم eps = ggml_get_op_params_f32(dst, 0);
    ثابت int32_t n_iter = ggml_get_op_params_i32(dst, 1);
    GGML_ASSERT(n_iter > 0);

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت int64_t dr  = (n_ٽوڪن + nth - 1) / nth;
    ثابت int64_t it0 = dr * ith;
    ثابت int64_t it1 = MIN(it0 + dr, n_ٽوڪن);

    ثابت عائم scale_comb = *(ثابت عائم *) ((ثابت حرف *) scale->معلومات + 2*nbs0);

    لكل (int64_t it = it0; it < it1; ++it) {
        عائم comb[hc*hc];

        لكل (int64_t isrc = 0; isrc < hc; ++isrc) {
            عائم max = -INFINITY;
            لكل (int64_t idst = 0; idst < hc; ++idst) {
                ثابت int64_t idx = idst + hc*isrc;
                ثابت عائم xv = *(ثابت عائم *) ((ثابت حرف *) mixes->معلومات + (comb_وٿي + idx)*nbm0 + it*nbm1);
                ثابت عائم bv = *(ثابت عائم *) ((ثابت حرف *) بنياد->معلومات  + (comb_وٿي + idx)*nbb0);
                ثابت عائم v = xv * scale_comb + bv;
                comb[idx] = v;
                max = MAX(max, v);
            }

            عائم جوڙ = 0.0f;
            لكل (int64_t idst = 0; idst < hc; ++idst) {
                ثابت int64_t idx = idst + hc*isrc;
                ثابت عائم v = expf(comb[idx] - max);
                comb[idx] = v;
                جوڙ += v;
            }

            ثابت عائم inv_جوڙ = 1.0f / جوڙ;
            لكل (int64_t idst = 0; idst < hc; ++idst) {
                ثابت int64_t idx = idst + hc*isrc;
                comb[idx] = comb[idx] * inv_جوڙ + eps;
            }
        }

        ggml_dsv4_hc_comb_norm_کانن_ڳڻپ(comb, eps);
        لكل (int32_t i = 1; i < n_iter; ++i) {
            ggml_dsv4_hc_comb_norm_صفون(comb, eps);
            ggml_dsv4_hc_comb_norm_کانن_ڳڻپ(comb, eps);
        }

        لكل (int64_t isrc = 0; isrc < hc; ++isrc) {
            لكل (int64_t idst = 0; idst < hc; ++idst) {
                ثابت int64_t idx = idst + hc*isrc;
                *(عائم *) ((حرف *) dst->معلومات + idst*nbd0 + isrc*nbd1 + it*nbd2) = comb[idx];
            }
        }
    }
}

فراغ ggml_compute_forward_dsv4_hc_comb(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_dsv4_hc_comb_f32(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_dsv4_hc_pre

ساكن فراغ ggml_compute_forward_dsv4_hc_pre_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    ثابت ggml_tensor * x       = dst->src[0];
    ثابت ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ثابت int64_t n_embd   = x->ne[0];
    ثابت int64_t hc       = x->ne[1];
    ثابت int64_t n_ٽوڪن = x->ne[2];

    ثابت عائم scale = ggml_get_op_params_f32(dst, 0);
    ثابت منطقي  gated = ggml_get_op_params_i32(dst, 1) != 0;

    GGML_ASSERT(dst->ne[0] == n_embd);
    GGML_ASSERT(dst->ne[1] == n_ٽوڪن);
    إذا (gated) {
        GGML_ASSERT(weights->ne[0] == n_embd);
        GGML_ASSERT(weights->ne[1] == hc);
        GGML_ASSERT(weights->ne[2] == n_ٽوڪن);
    } وإلا {
        GGML_ASSERT(weights->ne[0] == hc);
        GGML_ASSERT(weights->ne[1] == n_ٽوڪن);
    }

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت int64_t nr  = n_embd * n_ٽوڪن;
    ثابت int64_t dr  = (nr + nth - 1) / nth;
    ثابت int64_t ir0 = dr * ith;
    ثابت int64_t ir1 = MIN(ir0 + dr, nr);

    لكل (int64_t ir = ir0; ir < ir1; ++ir) {
        ثابت int64_t i0 = ir % n_embd;
        ثابت int64_t it = ir / n_embd;

        عائم جوڙ = 0.0f;
        لكل (int64_t ih = 0; ih < hc; ++ih) {
            ثابت عائم xv = *(ثابت عائم *) ((ثابت حرف *) x->معلومات + i0*nbx0 + ih*nbx1 + it*nbx2);
            عائم wv;
            إذا (gated) {
                ثابت عائم gv = *(ثابت عائم *) ((ثابت حرف *) weights->معلومات + i0*nbw0 + ih*nbw1 + it*nbw2);
                wv = 1.0f / (1.0f + expf(-gv));
            } وإلا {
                wv = *(ثابت عائم *) ((ثابت حرف *) weights->معلومات + ih*nbw0 + it*nbw1);
            }
            جوڙ += xv * wv;
        }

        *(عائم *) ((حرف *) dst->معلومات + i0*nbd0 + it*nbd1) = scale * جوڙ;
    }
}

فراغ ggml_compute_forward_dsv4_hc_pre(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_dsv4_hc_pre_f32(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_dsv4_hc_post

ساكن فراغ ggml_compute_forward_dsv4_hc_post_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    ثابت ggml_tensor * x        = dst->src[0];
    ثابت ggml_tensor * residual = dst->src[1];
    ثابت ggml_tensor * post     = dst->src[2];
    ثابت ggml_tensor * comb     = dst->src[3];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ثابت int64_t n_embd   = x->ne[0];
    ثابت int64_t n_ٽوڪن = x->ne[1];
    ثابت int64_t hc       = residual->ne[1];

    GGML_ASSERT(dst->ne[0] == n_embd);
    GGML_ASSERT(dst->ne[1] == hc);
    GGML_ASSERT(dst->ne[2] == n_ٽوڪن);
    GGML_ASSERT(residual->ne[0] == n_embd);
    GGML_ASSERT(residual->ne[2] == n_ٽوڪن);
    GGML_ASSERT(post->ne[0] == hc);
    GGML_ASSERT(post->ne[1] == n_ٽوڪن);

    // comb == NULL: identity mixing, each stream keeps its own residual
    size_t nbc0 = 0;
    size_t nbc1 = 0;
    size_t nbc2 = 0;
    إذا (comb) {
        GGML_ASSERT(comb->type == GGML_TYPE_F32);
        GGML_ASSERT(comb->ne[0] == hc);
        GGML_ASSERT(comb->ne[1] == hc);
        GGML_ASSERT(comb->ne[2] == n_ٽوڪن);
        nbc0 = comb->nb[0];
        nbc1 = comb->nb[1];
        nbc2 = comb->nb[2];
    }

    GGML_TENSOR_LOCALS(size_t, nbx, x,        nb);
    GGML_TENSOR_LOCALS(size_t, nbr, residual, nb);
    GGML_TENSOR_LOCALS(size_t, nbp, post,     nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,      nb);

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت int64_t nr  = n_embd * hc * n_ٽوڪن;
    ثابت int64_t dr  = (nr + nth - 1) / nth;
    ثابت int64_t ir0 = dr * ith;
    ثابت int64_t ir1 = MIN(ir0 + dr, nr);

    لكل (int64_t ir = ir0; ir < ir1; ++ir) {
        ثابت int64_t i0     = ir % n_embd;
        ثابت int64_t idst   = (ir / n_embd) % hc;
        ثابت int64_t it     = ir / (n_embd * hc);

        ثابت عائم xv = *(ثابت عائم *) ((ثابت حرف *) x->معلومات    + i0*nbx0 + it*nbx1);
        ثابت عائم pv = *(ثابت عائم *) ((ثابت حرف *) post->معلومات + idst*nbp0 + it*nbp1);

        عائم جوڙ = xv * pv;
        إذا (comb) {
            لكل (int64_t isrc = 0; isrc < hc; ++isrc) {
                ثابت عائم rv = *(ثابت عائم *) ((ثابت حرف *) residual->معلومات + i0*nbr0 + isrc*nbr1 + it*nbr2);
                ثابت عائم cv = *(ثابت عائم *) ((ثابت حرف *) comb->معلومات     + idst*nbc0 + isrc*nbc1 + it*nbc2);
                جوڙ += rv * cv;
            }
        } وإلا {
            جوڙ += *(ثابت عائم *) ((ثابت حرف *) residual->معلومات + i0*nbr0 + idst*nbr1 + it*nbr2);
        }

        *(عائم *) ((حرف *) dst->معلومات + i0*nbd0 + idst*nbd1 + it*nbd2) = جوڙ;
    }
}

فراغ ggml_compute_forward_dsv4_hc_post(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_dsv4_hc_post_f32(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_top_k_mask

فراغ ggml_compute_forward_چوٽي_k_نقاب(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    ثابت ggml_tensor * نقاب   = dst->src[0];
    ثابت ggml_tensor * چوٽي_k  = dst->src[1];
    ثابت ggml_tensor * score  = dst->src[2];
    ثابت ggml_tensor * بلاڪون = dst->src[3];
    ثابت ggml_tensor * extra  = dst->src[4];

    GGML_ASSERT(dst->type == نقاب->type);
    GGML_ASSERT(dst->nb[0] == ggml_type_سائيز(dst->type));

    ثابت int64_t n_kv    = نقاب->ne[0];
    ثابت int64_t ne1     = dst->ne[1];
    ثابت int64_t ne2     = dst->ne[2];
    ثابت int64_t n_صفون  = ggml_nrows(dst);
    ثابت int64_t n_چوٽي   = چوٽي_k->ne[0];
    ثابت int64_t n_blk   = بلاڪون ? بلاڪون->ne[0] : 1;
    ثابت int64_t n_pick  = بلاڪون ? بلاڪون->ne[1] : score ? score->ne[0] : n_kv;
    ثابت int64_t n_extra = extra ? extra->ne[0] : 0;

    ثابت size_t ts = ggml_type_سائيز(dst->type);

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    لكل (int64_t ir = ith; ir < n_صفون; ir += nth) {
        ثابت int64_t i1 = ir % ne1;
        ثابت int64_t i2 = (ir / ne1) % ne2;
        ثابت int64_t i3 = ir / (ne1*ne2);

        ثابت حرف * نقاب_صف = (ثابت حرف *) نقاب->معلومات
            + (i1 % نقاب->ne[1])*نقاب->nb[1] + (i2 % نقاب->ne[2])*نقاب->nb[2] + (i3 % نقاب->ne[3])*نقاب->nb[3];
        حرف * dst_صف = (حرف *) dst->معلومات + i1*dst->nb[1] + i2*dst->nb[2] + i3*dst->nb[3];

        إذا (dst->type == GGML_TYPE_F16) {
            ثابت ggml_fp16_t ninf = GGML_CPU_FP32_منزل_FP16(-INFINITY);
            لكل (int64_t c = 0; c < n_kv; ++c) {
                ((ggml_fp16_t *) dst_صف)[c] = ninf;
            }
        } وإلا {
            لكل (int64_t c = 0; c < n_kv; ++c) {
                ((عائم *) dst_صف)[c] = -INFINITY;
            }
        }

        خودڪار keep = [&](int32_t c) {
            إذا (c >= 0 && c < n_kv) {
                يادگيري_نقل(dst_صف + c*ts, نقاب_صف + c*ts, ts);
            }
        };

        ثابت حرف * چوٽي_k_صف = (ثابت حرف *) چوٽي_k->معلومات + i1*چوٽي_k->nb[1] + i2*چوٽي_k->nb[2] + i3*چوٽي_k->nb[3];

        لكل (int64_t j = 0; j < n_چوٽي; ++j) {
            ثابت int32_t p = *(ثابت int32_t *) (چوٽي_k_صف + j*چوٽي_k->nb[0]);
            إذا (p < 0 || p >= n_pick) {
                استمر;
            }
            إذا (score) {
                ثابت عائم s = *(ثابت عائم *) ((ثابت حرف *) score->معلومات
                    + p*score->nb[0] + i1*score->nb[1] + i2*score->nb[2] + i3*score->nb[3]);
                إذا (!(s > -INFINITY)) {
                    استمر;
                }
            }
            إذا (!بلاڪون) {
                keep(p);
                استمر;
            }
            لكل (int64_t k = 0; k < n_blk; ++k) {
                keep(*(ثابت int32_t *) ((ثابت حرف *) بلاڪون->معلومات + k*بلاڪون->nb[0] + p*بلاڪون->nb[1]));
            }
        }

        إذا (extra) {
            ثابت حرف * extra_صف = (ثابت حرف *) extra->معلومات + i1*extra->nb[1] + i2*extra->nb[2] + i3*extra->nb[3];
            لكل (int64_t j = 0; j < n_extra; ++j) {
                keep(*(ثابت int32_t *) (extra_صف + j*extra->nb[0]));
            }
        }
    }
}

// ggml_compute_forward_rwkv_wkv7

ساكن فراغ ggml_compute_forward_rwkv_wkv7_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    ثابت int64_t T = dst->src[1]->ne[2];
    ثابت int64_t C = dst->ne[0];
    ثابت int64_t HEADS = dst->src[1]->ne[1];
    ثابت int64_t n_seqs = dst->src[6]->ne[1];
    ثابت int64_t سر_سائيز = C / HEADS;

    عائم * dst_معلومات = (عائم *) dst->معلومات;
    عائم * state = ((عائم *) dst->معلومات) + C * T;

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح h_شروعات =  (HEADS * (ith    )) / nth;
    ثابت صحيح h_پڇاڙي   = ((HEADS * (ith + 1)) / nth < HEADS) ?
                         (HEADS * (ith + 1)) / nth : HEADS;

    عائم * r = (عائم *) dst->src[0]->معلومات;
    عائم * w = (عائم *) dst->src[1]->معلومات;
    عائم * k = (عائم *) dst->src[2]->معلومات;
    عائم * v = (عائم *) dst->src[3]->معلومات;
    عائم * a = (عائم *) dst->src[4]->معلومات;
    عائم * b = (عائم *) dst->src[5]->معلومات;

    int64_t t_stride = HEADS * سر_سائيز; // Same to C

    int64_t h_stride = C / HEADS;
    GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
    int64_t h_stride_2d = سر_سائيز * سر_سائيز;

    #إذا defined(GGML_SIMD)
        #إذا defined(__ARM_FEATURE_SVE) || defined(__riscv_v_intrinsic)
            // scalar Route to scalar implementation       //TODO: Write SVE code and RVV code
            لكل (int64_t t = 0; t < T; t++) {
                int64_t t_وٿي = t * t_stride;
                int64_t state_وٿي = سر_سائيز * C * (t / (T / n_seqs));
                عائم * state_cur = state + state_وٿي;
                عائم * state_prev = t % (T / n_seqs) ? state_cur : (عائم*)dst->src[6]->معلومات + state_وٿي;

                لكل (int64_t h = h_شروعات; h < h_پڇاڙي; h++) {
                    int64_t h_وٿي = h * h_stride;
                    int64_t t_h_وٿي = t_وٿي + h_وٿي;
                    int64_t h_2d_وٿي = h * h_stride_2d;

                    لكل (int64_t i = 0; i < سر_سائيز; i++) {
                        int64_t t_h_i_وٿي = t_h_وٿي + i;
                        int64_t h_2d_i_وٿي = h_2d_وٿي + i * h_stride;

                        عائم v_val = v[t_h_i_وٿي];

                        عائم sa = 0, نتيجو = 0;
                        لكل (int64_t j = 0; j < سر_سائيز; j++) {
                            sa += a[t_h_وٿي + j] * state_prev[h_2d_i_وٿي + j];
                        }

                        لكل (int64_t j = 0; j < سر_سائيز; j++) {
                            int64_t t_h_j_وٿي = t_h_وٿي + j;
                            int64_t h_2d_i_j_وٿي = h_2d_i_وٿي + j;

                            عائم r_val = r[t_h_j_وٿي];
                            عائم w_val = w[t_h_j_وٿي];
                            عائم k_val = k[t_h_j_وٿي];
                            عائم b_val = b[t_h_j_وٿي];
                            عائم kv_val = v_val * k_val;
                            عائم prev_state_val = state_prev[h_2d_i_j_وٿي];
                            state_cur[h_2d_i_j_وٿي] = prev_state_val * w_val + kv_val + sa * b_val;
                            نتيجو += state_cur[h_2d_i_j_وٿي] * r_val;
                        }
                        dst_معلومات[t_h_i_وٿي] = نتيجو;
                    }
                }
            }
        #وإلا
            لكل (int64_t t = 0; t < T; t++) {
                int64_t t_وٿي = t * t_stride;
                int64_t state_وٿي = سر_سائيز * C * (t / (T / n_seqs));
                عائم * state_cur = state + state_وٿي;
                عائم * state_prev = t % (T / n_seqs) ? state_cur : (عائم*)dst->src[6]->معلومات + state_وٿي;

                لكل (int64_t h = h_شروعات; h < h_پڇاڙي; h++) {
                    int64_t h_وٿي = h * h_stride;
                    int64_t t_h_وٿي = t_وٿي + h_وٿي;
                    int64_t h_2d_وٿي = h * h_stride_2d;

                    لكل (int64_t ii = 0; ii < سر_سائيز; ii++) {
                        int64_t t_h_i_وٿي = t_h_وٿي + ii;
                        int64_t h_2d_i_وٿي = h_2d_وٿي + ii * h_stride;

                        GGML_F32_VEC v_vec = GGML_F32_VEC_SET1(v[t_h_i_وٿي]);

                        عائم sa = 0;
                        {
                            GGML_F32_VEC جوڙ[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };
                            GGML_F32_VEC ax[GGML_F32_ARR];
                            GGML_F32_VEC ay[GGML_F32_ARR];
                            لكل (int64_t j = 0; j < سر_سائيز; j += GGML_F32_قدم) {
                                لكل (int64_t kk = 0; kk < GGML_F32_ARR; kk++) {
                                    ax[kk] = GGML_F32_VEC_ڀريو(&a[t_h_وٿي + j + kk * GGML_F32_EPR]);
                                    ay[kk] = GGML_F32_VEC_ڀريو(&state_prev[h_2d_i_وٿي + j + kk * GGML_F32_EPR]);
                                    جوڙ[kk] = GGML_F32_VEC_FMA(جوڙ[kk], ax[kk], ay[kk]);
                                }
                            }
                            GGML_F32_VEC_REDUCE(sa, جوڙ);
                        }

                        GGML_F32_VEC sa_vec = GGML_F32_VEC_SET1(sa);

                        int64_t j = 0;
                        GGML_F32_VEC نتيجو_vec[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };
                        لكل (; j < سر_سائيز; j += GGML_F32_قدم) {
                            لكل (int64_t kk = 0; kk < GGML_F32_ARR; kk++) {
                                int64_t t_h_j_وٿي = t_h_وٿي + j + kk * GGML_F32_EPR;
                                int64_t h_2d_i_j_وٿي = h_2d_i_وٿي + j + kk * GGML_F32_EPR;

                                GGML_F32_VEC r_vec = GGML_F32_VEC_ڀريو(&r[t_h_j_وٿي]);
                                GGML_F32_VEC w_vec = GGML_F32_VEC_ڀريو(&w[t_h_j_وٿي]);
                                GGML_F32_VEC k_vec = GGML_F32_VEC_ڀريو(&k[t_h_j_وٿي]);
                                GGML_F32_VEC b_vec = GGML_F32_VEC_ڀريو(&b[t_h_j_وٿي]);

                                k_vec = GGML_F32_VEC_MUL(v_vec, k_vec);

                                GGML_F32_VEC state_vec = GGML_F32_VEC_ڀريو(&state_prev[h_2d_i_j_وٿي]);
                                // kv + s * decay + sa * b
                                state_vec = GGML_F32_VEC_FMA(k_vec, state_vec, w_vec);
                                state_vec = GGML_F32_VEC_FMA(state_vec, sa_vec, b_vec);
                                GGML_F32_VEC_STORE(&state_cur[h_2d_i_j_وٿي], state_vec);

                                نتيجو_vec[kk] = GGML_F32_VEC_FMA(نتيجو_vec[kk], state_vec, r_vec);
                            }
                        }
                        GGML_F32_VEC_REDUCE(dst_معلومات[t_h_i_وٿي], نتيجو_vec);

                        // There shouldn't be left-overs though.
                        لكل (; j < سر_سائيز; j++) {
                            int64_t t_h_j_وٿي = t_h_وٿي + j;
                            int64_t h_2d_i_j_وٿي = h_2d_i_وٿي + j;

                            عائم r_val = r[t_h_j_وٿي];
                            عائم w_val = w[t_h_j_وٿي];
                            عائم k_val = k[t_h_j_وٿي];
                            عائم b_val = b[t_h_j_وٿي];
                            عائم kv_val = v[t_h_i_وٿي] * k_val;

                            عائم prev_state_val = state_prev[h_2d_i_j_وٿي];
                            state_cur[h_2d_i_j_وٿي] = prev_state_val * w_val + kv_val + sa * b_val;
                            dst_معلومات[t_h_i_وٿي] += state_cur[h_2d_i_j_وٿي] * r_val;
                        }
                    }
                }
            }
        #جيڪڏهن_پورو
    #وإلا
        لكل (int64_t t = 0; t < T; t++) {
            int64_t t_وٿي = t * t_stride;
            int64_t state_وٿي = سر_سائيز * C * (t / (T / n_seqs));
            عائم * state_cur = state + state_وٿي;
            عائم * state_prev = t % (T / n_seqs) ? state_cur : (عائم*)dst->src[6]->معلومات + state_وٿي;

            لكل (int64_t h = h_شروعات; h < h_پڇاڙي; h++) {
                int64_t h_وٿي = h * h_stride;
                int64_t t_h_وٿي = t_وٿي + h_وٿي;
                int64_t h_2d_وٿي = h * h_stride_2d;

                لكل (int64_t i = 0; i < سر_سائيز; i++) {
                    int64_t t_h_i_وٿي = t_h_وٿي + i;
                    int64_t h_2d_i_وٿي = h_2d_وٿي + i * h_stride;

                    عائم v_val = v[t_h_i_وٿي];

                    عائم sa = 0, نتيجو = 0;
                    لكل (int64_t j = 0; j < سر_سائيز; j++) {
                        sa += a[t_h_وٿي + j] * state_prev[h_2d_i_وٿي + j];
                    }

                    لكل (int64_t j = 0; j < سر_سائيز; j++) {
                        int64_t t_h_j_وٿي = t_h_وٿي + j;
                        int64_t h_2d_i_j_وٿي = h_2d_i_وٿي + j;

                        عائم r_val = r[t_h_j_وٿي];
                        عائم w_val = w[t_h_j_وٿي];
                        عائم k_val = k[t_h_j_وٿي];
                        عائم b_val = b[t_h_j_وٿي];
                        عائم kv_val = v_val * k_val;
                        عائم prev_state_val = state_prev[h_2d_i_j_وٿي];
                        state_cur[h_2d_i_j_وٿي] = prev_state_val * w_val + kv_val + sa * b_val;
                        نتيجو += state_cur[h_2d_i_j_وٿي] * r_val;
                    }
                    dst_معلومات[t_h_i_وٿي] = نتيجو;
                }
            }
        }
    #جيڪڏهن_پورو
}


فراغ ggml_compute_forward_rwkv_wkv7(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_rwkv_wkv7_f32(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_map_custom1

فراغ ggml_compute_forward_map_custom1(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت ggml_tensor * a = dst->src[0];

    بنية ggml_map_custom1_op_params p;
    يادگيري_نقل(&p, dst->op_params, حجم(p));

    p.fun(dst, a, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_map_custom2

فراغ ggml_compute_forward_map_custom2(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت ggml_tensor * a = dst->src[0];
    ثابت ggml_tensor * b = dst->src[1];

    بنية ggml_map_custom2_op_params p;
    يادگيري_نقل(&p, dst->op_params, حجم(p));

    p.fun(dst, a, b, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_map_custom3

فراغ ggml_compute_forward_map_custom3(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت ggml_tensor * a = dst->src[0];
    ثابت ggml_tensor * b = dst->src[1];
    ثابت ggml_tensor * c = dst->src[2];

    بنية ggml_map_custom3_op_params p;
    يادگيري_نقل(&p, dst->op_params, حجم(p));

    p.fun(dst, a, b, c, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_custom

فراغ ggml_compute_forward_custom(
    ثابت بنية ggml_compute_params * params,
          بنية ggml_tensor * dst) {

    بنية ggml_custom_op_params p;
    يادگيري_نقل(&p, dst->op_params, حجم(p));

    p.fun(dst, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_cross_entropy_loss

ساكن فراغ ggml_compute_forward_cross_entropy_loss_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->nb[0] == ggml_type_سائيز(src0->type));
    GGML_ASSERT(src1->nb[0] == ggml_type_سائيز(src1->type));
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // TODO: handle transposed/permuted matrices
    ثابت int64_t nc = src0->ne[0];
    ثابت int64_t nr = ggml_nrows(src0);

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    عائم * sums =  (عائم *) params->wdata;
    عائم * st   = ((عائم *) params->wdata) + nth + ith*nc;
    عائم جوڙ_thread = 0.0f;

    GGML_ASSERT(params->wsize >= حجم(عائم) * (nth + nth * nc));

    // rows per thread
    ثابت int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت int64_t ir0 = dr*ith;
    ثابت int64_t ir1 = MIN(ir0 + dr, nr);

    لكل (int64_t i1 = ir0; i1 < ir1; ++i1) {
        ثابت عائم * s0 = (ثابت عائم *)((ثابت حرف *) src0->معلومات + i1*src0->nb[1]);
        ثابت عائم * s1 = (ثابت عائم *)((ثابت حرف *) src1->معلومات + i1*src1->nb[1]);

#جيڪڏهن_وصف_نه_ٿيل NDEBUG
        لكل (int64_t i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            أكد(!isnan(s0[i]));
            أكد(!isnan(s1[i]));
        }
#جيڪڏهن_پورو // NDEBUG

        عائم max = -INFINITY;
        ggml_vec_max_f32(nc, &max, s0);
        ثابت ggml_float جوڙ_softmax = ggml_vec_log_soft_max_f32(nc, st, s0, max);
        أكد(جوڙ_softmax >= 0.0);

        ggml_vec_add1_f32(nc, st, st, -جوڙ_softmax);
        ggml_vec_mul_f32(nc, st, st, s1);

        عائم جوڙ_st = 0.0f;
        ggml_vec_جوڙ_f32(nc, &جوڙ_st, st);
        جوڙ_thread += جوڙ_st;

#جيڪڏهن_وصف_نه_ٿيل NDEBUG
        لكل (int64_t i = 0; i < nc; ++i) {
            أكد(!isnan(st[i]));
            أكد(!isinf(st[i]));
        }
#جيڪڏهن_پورو // NDEBUG
    }
    sums[ith] = جوڙ_thread;
    ggml_barrier(params->threadpool);

    إذا (ith == 0) {
        عائم * dp = (عائم *) dst->معلومات;
        ggml_vec_جوڙ_f32(nth, dp, sums);
        dp[0] *= -1.0f / (عائم) nr;
    }
}

فراغ ggml_compute_forward_cross_entropy_loss(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_cross_entropy_loss_f32(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_cross_entropy_loss_back

ساكن فراغ ggml_compute_forward_cross_entropy_loss_پٺ_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * grad  = dst->src[0]; // gradient of forward pass output
    ثابت ggml_tensor * src0f = dst->src[1]; // src0 of forward pass
    ثابت ggml_tensor * src1f = dst->src[2]; // src1 of forward pass

    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src0f));
    GGML_ASSERT(ggml_is_contiguous(src1f));
    GGML_ASSERT(ggml_is_contiguous(grad));
    GGML_ASSERT(ggml_are_same_shape(src0f, src1f) && ggml_are_same_shape(src0f, dst));

    ثابت int64_t ith = params->ith;
    ثابت int64_t nth = params->nth;

    // TODO: handle transposed/permuted matrices
    ثابت int64_t nc = src0f->ne[0];
    ثابت int64_t nr = ggml_nrows(src0f);

    // rows per thread
    ثابت int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت int64_t ir0 = dr*ith;
    ثابت int64_t ir1 = MIN(ir0 + dr, nr);

    ثابت عائم d_by_nr = ((ثابت عائم *) grad->معلومات)[0] / (عائم) nr;

    لكل (int64_t i1 = ir0; i1 < ir1; i1++) {
        عائم       * ds0 = (عائم       *)((حرف       *) dst->معلومات   + i1*dst->nb[1]);
        ثابت عائم * s0  = (ثابت عائم *)((ثابت حرف *) src0f->معلومات + i1*src0f->nb[1]);
        ثابت عائم * s1  = (ثابت عائم *)((ثابت حرف *) src1f->معلومات + i1*src1f->nb[1]);

#جيڪڏهن_وصف_نه_ٿيل NDEBUG
        لكل (int64_t i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            أكد(!isnan(s0[i]));
            أكد(!isnan(s1[i]));
        }
#جيڪڏهن_پورو // NDEBUG

        // soft_max
        عائم max = -INFINITY;
        ggml_vec_max_f32(nc, &max, s0);
        ثابت ggml_float جوڙ = ggml_vec_soft_max_f32(nc, ds0, s0, max);
        أكد(جوڙ > 0.0);
        ggml_vec_scale_f32(nc, ds0, 1.0/جوڙ);

        // grad(src0f) = (softmax(src0f) - src1f) * grad(cross_entropy_loss(src0f, src1f)) / nr
        ggml_vec_sub_f32(nc, ds0, ds0, s1);
        ggml_vec_scale_f32(nc, ds0, d_by_nr);

#جيڪڏهن_وصف_نه_ٿيل NDEBUG
        لكل (int64_t i = 0; i < nc; ++i) {
            أكد(!isnan(ds0[i]));
            أكد(!isinf(ds0[i]));
        }
#جيڪڏهن_پورو // NDEBUG
    }
}

فراغ ggml_compute_forward_cross_entropy_loss_پٺ(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_cross_entropy_loss_پٺ_f32(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ساكن فراغ ggml_compute_forward_opt_قدم_adamw_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0         = dst->src[0];
    ثابت ggml_tensor * src0_grad    = dst->src[1];
    ثابت ggml_tensor * src0_grad_m  = dst->src[2];
    ثابت ggml_tensor * src0_grad_v  = dst->src[3];
    ثابت ggml_tensor * adamw_params = dst->src[4];

    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad));
    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad_m));
    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad_v));
    GGML_ASSERT(ggml_nelements(adamw_params) == 7);

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS
    GGML_ASSERT(nb00 == حجم(عائم));

    // rows per thread
    ثابت صحيح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحيح ir0 = dr*ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    ثابت عائم * adamw_params_ptr = ggml_get_معلومات_f32(adamw_params);

    ثابت عائم alpha  = adamw_params_ptr[0];
    ثابت عائم beta1  = adamw_params_ptr[1];
    ثابت عائم beta2  = adamw_params_ptr[2];
    ثابت عائم eps    = adamw_params_ptr[3];
    ثابت عائم wd     = adamw_params_ptr[4];
    ثابت عائم beta1h = adamw_params_ptr[5];
    ثابت عائم beta2h = adamw_params_ptr[6];
    ثابت عائم keep   = 1.f - alpha * wd;
    لكل (صحيح ir = ir0; ir < ir1; ++ir) {
        ثابت int64_t i03 = ir/(ne02*ne01);
        ثابت int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        ثابت int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        ثابت size_t وٿي = i03*nb03 + i02*nb02 + i01*nb01;

        عائم       * w = (عائم       *) ((حرف       *) src0->معلومات        + وٿي); //وزن
        ثابت عائم * g = (ثابت عائم *) ((ثابت حرف *) src0_grad->معلومات   + وٿي); // grad
        عائم       * m = (عائم       *) ((حرف       *) src0_grad_m->معلومات + وٿي);
        عائم       * v = (عائم       *) ((حرف       *) src0_grad_v->معلومات + وٿي);

        لكل (صحيح i00 = 0; i00 < ne00; ++i00) {
            m[i00] = m[i00]*beta1 +        g[i00]*(1.0f - beta1);
            v[i00] = v[i00]*beta2 + g[i00]*g[i00]*(1.0f - beta2);

            ثابت عائم mh =       m[i00]*beta1h;
            ثابت عائم vh = sqrtf(v[i00]*beta2h) + eps;

            // The weight decay is applied independently of the Adam momenta m and v.
            // This is NOT equivalent to l2 regularization that adds w[i00]*w[i00] to the loss.
            // See: https://arxiv.org/pdf/1711.05101v3.pdf
            w[i00] = w[i00] * keep - alpha * mh / vh;
        }
    }
}

فراغ ggml_compute_forward_opt_قدم_adamw(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_opt_قدم_adamw_f32(params, dst);
            } اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ساكن فراغ ggml_compute_forward_opt_قدم_sgd_f32(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    ثابت ggml_tensor * src0       = dst->src[0];
    ثابت ggml_tensor * src0_grad  = dst->src[1];
    ثابت ggml_tensor * sgd_params = dst->src[2];

    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad));
    GGML_ASSERT(ggml_nelements(sgd_params) == 2);

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت صحيح nr = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS
    GGML_ASSERT(nb00 == حجم(عائم));

    // rows per thread
    ثابت صحيح dr = (nr + nth - 1) / nth;

    // row range for this thread
    ثابت صحيح ir0 = dr * ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    // using adamw param subset we care about - alpha, wd - could have a separate struct
    ثابت عائم * sgd_params_ptr   = ggml_get_معلومات_f32(sgd_params);
    ثابت عائم   alpha            = sgd_params_ptr[0];
    ثابت عائم   keep             = 1.f - alpha * sgd_params_ptr[1];

    لكل (صحيح ir = ir0; ir < ir1; ++ir) {
        ثابت int64_t i03 = ir / (ne02 * ne01);
        ثابت int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
        ثابت int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

        ثابت size_t وٿي = i03 * nb03 + i02 * nb02 + i01 * nb01;

        عائم *       w = (عائم *) ((حرف *) src0->معلومات + وٿي);                   //وزن
        ثابت عائم * g = (ثابت عائم *) ((ثابت حرف *) src0_grad->معلومات + وٿي);  // grad

        لكل (صحيح i00 = 0; i00 < ne00; ++i00) {
            w[i00] = w[i00] * keep - alpha * g[i00];
        }
    }
}

فراغ ggml_compute_forward_opt_قدم_sgd(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];

    بدل (src0->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_opt_قدم_sgd_f32(params, dst);
            }
            اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error - sgd is F32 only");
            }
    }
}

ساكن مضمن عائم ggml_fwht_ڀريو(ثابت عائم value) {
    أرجع value;
}

ساكن مضمن عائم ggml_fwht_ڀريو(ثابت ggml_fp16_t value) {
    أرجع ggml_fp16_منزل_fp32(value);
}

سانچو<قسم_نالو src_t>
ساكن فراغ ggml_compute_forward_fwht_impl(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == (std::is_same_v<src_t, عائم> ? GGML_TYPE_F32 : GGML_TYPE_F16));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ٻٽو_OP_LOCALS

    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    ثابت int64_t n = ne10;
    GGML_ASSERT((n & (n - 1)) == 0); // must be power of 2

    ثابت int64_t nr = ne11 * ne12 * ne13;
    ثابت int64_t صفون_في_thread = (nr + nth - 1) / nth;
    ثابت int64_t شروعات_صف = ith * صفون_في_thread;
    ثابت int64_t پڇاڙي_صف = MIN(شروعات_صف + صفون_في_thread, nr);

    ثابت عائم scale = 1.0f / sqrtf((عائم)n);

#إذا defined(GGML_SIMD)
    ثابت GGML_F32_VEC v_minus_one = GGML_F32_VEC_SET1(-1.0f);
#جيڪڏهن_پورو

    لكل (int64_t r = شروعات_صف; r < پڇاڙي_صف; r++) {
        ثابت int64_t i13 = r / (ne11 * ne12);
        ثابت int64_t i12 = (r - i13 * ne11 * ne12) / ne11;
        ثابت int64_t i11 = r - i13 * ne11 * ne12 - i12 * ne11;

        ثابت src_t * src_صف = (ثابت src_t *) ((ثابت حرف *) src1->معلومات + i11 * nb11 + i12 * nb12 + i13 * nb13);
        عائم * dst_صف = (عائم *) ((حرف *) dst->معلومات + i11 * nb1 + i12 * nb2 + i13 * nb3);

        لكل (int64_t j = 0; j < n; j++) {
            dst_صف[j] = ggml_fwht_ڀريو(src_صف[j]) * scale;
        }

        // Scalar passes
#إذا defined(GGML_SIMD)
#إذا defined(__ARM_FEATURE_SVE)
        ثابت صحيح قدم = svcntw();
#وإلا
        ثابت صحيح قدم = GGML_F32_EPR;
#جيڪڏهن_پورو
#وإلا
        ثابت صحيح قدم = n;
#جيڪڏهن_پورو
        لكل (int64_t len = 1; len < قدم && len < n; len <<= 1) {
            لكل (int64_t i = 0; i < n; i += 2 * len) {
                لكل (int64_t j = 0; j < len; j++) {
                    عائم u = dst_صف[i + j];
                    عائم v = dst_صف[i + len + j];
                    dst_صف[i + j] = u + v;
                    dst_صف[i + len + j] = u - v;
                }
            }
        }

        // SIMD passes using GGML_F32_VEC_* macros for multi-architecture support
#إذا defined(GGML_SIMD)
        لكل (int64_t len = قدم; len < n; len <<= 1) {
            لكل (int64_t i = 0; i < n; i += 2 * len) {
                لكل (int64_t j = 0; j < len; j += قدم) {
                    GGML_F32_VEC u = GGML_F32_VEC_ڀريو(dst_صف + i + j);
                    GGML_F32_VEC v = GGML_F32_VEC_ڀريو(dst_صف + i + len + j);

                    GGML_F32_VEC_STORE(dst_صف + i + j,       GGML_F32_VEC_جوڙيو(u, v));
                    GGML_F32_VEC_STORE(dst_صف + i + len + j, GGML_F32_VEC_FMA(u, v, v_minus_one));
                }
            }
        }
#جيڪڏهن_پورو
    }
}

فراغ ggml_compute_forward_fwht(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    ثابت ggml_tensor * src1 = dst->src[1];

    بدل (src1->type) {
        حالة GGML_TYPE_F32:
            {
                ggml_compute_forward_fwht_impl<عائم>(params, dst);
            }
            اكسر;
        حالة GGML_TYPE_F16:
            {
                ggml_compute_forward_fwht_impl<ggml_fp16_t>(params, dst);
            }
            اكسر;
        افتراضي:
            {
                GGML_ABORT("fatal error - fwht supports F32 and F16 input");
            }
    }
}

// ggml_compute_forward_lightning_indexer

فراغ ggml_compute_forward_lightning_indexer(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * q = dst->src[0];
    ثابت ggml_tensor * k = dst->src[1];
    ثابت ggml_tensor * w = dst->src[2]; // weights
    ثابت ggml_tensor * m = dst->src[3]; //نقاب

    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(   q->type == GGML_TYPE_F32);
    GGML_ASSERT(   w->type == GGML_TYPE_F32);
    GGML_ASSERT(   m->type == GGML_TYPE_F16);

    GGML_TENSOR_LOCALS(int64_t, neq,  q, ne)
    GGML_TENSOR_LOCALS(size_t,  nbq,  q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek,  k, ne)
    GGML_TENSOR_LOCALS(size_t,  nbk,  k, nb)
    GGML_TENSOR_LOCALS(int64_t, جديد,  w, ne)
    GGML_TENSOR_LOCALS(size_t,  nbw,  w, nb)
    GGML_TENSOR_LOCALS(int64_t, nem,  m, ne)
    GGML_TENSOR_LOCALS(size_t,  nbm,  m, nb)
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb)

    GGML_ASSERT( nb0 == ggml_type_سائيز(dst->type));
    GGML_ASSERT(nbq0 == ggml_type_سائيز(  q->type));
    GGML_ASSERT(nbk0 == ggml_type_سائيز(  k->type));
    GGML_ASSERT(nbw0 == ggml_type_سائيز(  w->type));
    GGML_ASSERT(nbm0 == ggml_type_سائيز(  m->type));

    ثابت صحيح n_embd    = q->ne[0];
    ثابت صحيح n_سر    = q->ne[1];
    ثابت صحيح n_ٽوڪن  = q->ne[2];
    ثابت صحيح n_وهڪرو  = q->ne[3];
    ثابت صحيح n_kv      = k->ne[2];

    ggml_منزل_float_t ثابت k_منزل_float = ggml_get_type_traits(k->type)->منزل_float;
    GGML_ASSERT((k->type == GGML_TYPE_F32 || k_منزل_float) && "lightning indexer: unsupported K-type");

    ثابت صحيح nr  = n_kv;
    ثابت صحيح ith = params->ith;
    ثابت صحيح nth = params->nth;

    // (temporary) buffer for K converted to float
    عائم * k_صف_f32 = (عائم *) params->wdata + ith*(1*n_embd + CACHE_سٽ_سائيز_F32);

    // rows per thread
    ثابت صحيح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحيح ir0 = dr*ith;
    ثابت صحيح ir1 = MIN(ir0 + dr, nr);

    لكل (صحيح s = 0; s < n_وهڪرو; ++s) {
        لكل (صحيح t = 0; t < n_ٽوڪن; ++t) {
            ثابت عائم       *   w_صف =       (عائم *) ((حرف *)   w->معلومات + t*nbw1 +        s*nbw3);
            ثابت ggml_fp16_t *   m_صف = (ggml_fp16_t *) ((حرف *)   m->معلومات + t*nbm1 + (s%nem3)*nbm3);
            عائم             * dst_صف =       (عائم *) ((حرف *) dst->معلومات + t*nb1  +        s*nb3 );
            لكل (صحيح ik = ir0; ik < ir1; ++ik) {
                حرف * k_صف = (حرف *) k->معلومات + ik*nbk2 + s*nbk3;
                إذا (k_منزل_float) {
                    k_منزل_float(k_صف, k_صف_f32, n_embd);
                } وإلا {
                    k_صف_f32 = (عائم *) k_صف;
                }
                عائم score = 0.0f;
                لكل (صحيح h = 0; h < n_سر; ++h) {
                    // dot product of q and k for head h
                    عائم qk = 0.0f;
                    ثابت عائم * q_صف = (عائم *) ((حرف *) q->معلومات + h*nbq1 + t*nbq2 + s*nbq3);
                    ggml_vec_dot_f32(n_embd, &qk, 0, q_صف, 0, k_صف_f32, 0, 1);
                    // ReLU and weights (prescaled)
                    score += MAX(qk, 0.0f) * w_صف[h];
                }
                // apply mask
                dst_صف[ik] = score + GGML_CPU_FP16_منزل_FP32(m_صف[ik]);
            }
        }
    }
}
