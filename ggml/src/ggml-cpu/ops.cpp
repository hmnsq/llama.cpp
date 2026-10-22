#شامل "ops.h"

#شامل "ggml-cpu.h"
#شامل "ggml-impl.h"
#شامل "binary-ops.h"
#شامل "simd-gemm.h"
#شامل "ggml.h"
#شامل "unary-ops.h"
#شامل "vec.h"

#شامل <algorithm>
#شامل <cfloat>
#شامل <cmath>

// ggml_compute_forward_dup

ساکن خالی ggml_compute_forward_dup_same_cont(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));
    GGML_ASSERT(src0->type == dst->type);

    ثابت size_t nb0 = ggml_type_سائز(src0->type);

    ثابت صحیح ith = params->ith; // thread index
    ثابت صحیح nth = params->nth; // number of threads

    // parallelize by blocks
    ثابت صحیح nk = ggml_nelements(src0)/ggml_blck_سائز(src0->type);
    ثابت صحیح dr = (nk + nth - 1) / nth;
    ثابت صحیح k0 = dr * ith;
    ثابت صحیح k1 = MIN(k0 + dr, nk);

    اگر (k0 < k1) {
        حافظہ_نقل(
            ((حرف *)  dst->ڈیٹا + k0*nb0),
            ((حرف *) src0->ڈیٹا + k0*nb0),
            (k1 - k0) * nb0);
    }
}

سانچہ<قسم_نام src_t, قسم_نام dst_t>
ساکن خالی ggml_compute_forward_dup_flt(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(!ggml_is_quantized(src0->type) && !ggml_is_quantized(dst->type));

    GGML_TENSOR_UNARY_OP_LOCALS

    ثابت صحیح ith = params->ith; // thread index
    ثابت صحیح nth = params->nth; // number of threads

    // parallelize by rows
    ثابت صحیح nr = ne01;
    // number of rows per thread
    ثابت صحیح dr = (nr + nth - 1) / nth;
    // row range for this thread
    ثابت صحیح ir0 = dr * ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    // case: type & row size equal
    اگر (src0->type == dst->type &&
        ne00 == ne0 &&
        nb00 == ggml_type_سائز(src0->type) && nb0 == ggml_type_سائز(dst->type)) {
        // copy by rows
        ثابت size_t rs = ne00*nb00;
        برائے (int64_t i03 = 0; i03 < ne03; i03++) {
            برائے (int64_t i02 = 0; i02 < ne02; i02++) {
                برائے (int64_t i01 = ir0; i01 < ir1; i01++) {
                    حافظہ_نقل(
                        ((حرف *)  dst->ڈیٹا + i01*nb1  + i02*nb2  + i03*nb3),
                        ((حرف *) src0->ڈیٹا + i01*nb01 + i02*nb02 + i03*nb03),
                        rs);
                }
            }
        }
        واپس;
    }

    // case: dst tensor is contiguous
    اگر (ggml_is_contiguous(dst)) {
        اگر (nb00 == حجم(src_t)) {
            اگر constexpr (std::is_same_v<dst_t, src_t>) {
                // same type
                size_t id = 0;
                ثابت size_t rs = ne00 * nb00;
                حرف * dst_ptr = (حرف *) dst->ڈیٹا;

                برائے (صحیح i03 = 0; i03 < ne03; i03++) {
                    برائے (صحیح i02 = 0; i02 < ne02; i02++) {
                        id += rs * ir0;
                        برائے (صحیح i01 = ir0; i01 < ir1; i01++) {
                            ثابت حرف * src0_ptr = (حرف *) src0->ڈیٹا + i01*nb01 + i02*nb02 + i03*nb03;
                            حافظہ_نقل(dst_ptr + id, src0_ptr, rs);
                            id += rs;
                        }
                        id += rs * (ne01 - ir1);
                    }
                }
            } ورنہ {
                // casting between non-quantized types
                size_t id = 0;
                dst_t * dst_ptr = (dst_t *) dst->ڈیٹا;

                برائے (صحیح i03 = 0; i03 < ne03; i03++) {
                    برائے (صحیح i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        برائے (صحیح i01 = ir0; i01 < ir1; i01++) {
                            ثابت src_t * src0_ptr = (src_t *) ((حرف *) src0->ڈیٹا + i01*nb01 + i02*nb02 + i03*nb03);
                            برائے (صحیح i00 = 0; i00 < ne00; i00++) {
                                اعشاری tmp = type_conversion_جدول<src_t>::مقصد_f32(src0_ptr[i00]);
                                dst_ptr[id] = type_conversion_جدول<dst_t>::from_f32(tmp);
                                id++;
                            }
                        }
                        id += ne00 * (ne01 - ir1);
                    }
                }
            }
        } ورنہ {
            //printf("%s: this is not optimal - fix me\n", __func__);

            size_t id = 0;
            dst_t * dst_ptr = (dst_t *) dst->ڈیٹا;

            برائے (صحیح i03 = 0; i03 < ne03; i03++) {
                برائے (صحیح i02 = 0; i02 < ne02; i02++) {
                    id += ne00 * ir0;
                    برائے (صحیح i01 = ir0; i01 < ir1; i01++) {
                        برائے (صحیح i00 = 0; i00 < ne00; i00++) {
                            ثابت src_t * src0_ptr = (src_t *) ((حرف *) src0->ڈیٹا + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);

                            اعشاری tmp = type_conversion_جدول<src_t>::مقصد_f32(*src0_ptr);
                            dst_ptr[id] = type_conversion_جدول<dst_t>::from_f32(tmp);
                            id++;
                        }
                    }
                    id += ne00 * (ne01 - ir1);
                }
            }
        }
        واپس;
    }

    // dst counters
    int64_t i10 = 0;
    int64_t i11 = 0;
    int64_t i12 = 0;
    int64_t i13 = 0;

    اگر constexpr (std::is_same_v<dst_t, src_t>) {
        برائے (int64_t i03 = 0; i03 < ne03; i03++) {
            برائے (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                جب (i10 >= ne0) {
                    i10 -= ne0;
                    اگر (++i11 == ne1) {
                        i11 = 0;
                        اگر (++i12 == ne2) {
                            i12 = 0;
                            اگر (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                برائے (int64_t i01 = ir0; i01 < ir1; i01++) {
                    برائے (int64_t i00 = 0; i00 < ne00; i00++) {
                        ثابت حرف * src0_ptr = ((حرف *) src0->ڈیٹا + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              حرف * dst_ptr  = ((حرف *)  dst->ڈیٹا + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                        حافظہ_نقل(dst_ptr, src0_ptr, حجم(dst_t));

                        اگر (++i10 == ne00) {
                            i10 = 0;
                            اگر (++i11 == ne01) {
                                i11 = 0;
                                اگر (++i12 == ne02) {
                                    i12 = 0;
                                    اگر (++i13 == ne03) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                جب (i10 >= ne0) {
                    i10 -= ne0;
                    اگر (++i11 == ne1) {
                        i11 = 0;
                        اگر (++i12 == ne2) {
                            i12 = 0;
                            اگر (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
            }
        }

    } ورنہ {
        برائے (int64_t i03 = 0; i03 < ne03; i03++) {
            برائے (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                جب (i10 >= ne0) {
                    i10 -= ne0;
                    اگر (++i11 == ne1) {
                        i11 = 0;
                        اگر (++i12 == ne2) {
                            i12 = 0;
                            اگر (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                برائے (int64_t i01 = ir0; i01 < ir1; i01++) {
                    برائے (int64_t i00 = 0; i00 < ne00; i00++) {
                        ثابت حرف * src0_ptr = ((حرف *) src0->ڈیٹا + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              حرف * dst_ptr  = ((حرف *)  dst->ڈیٹا + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                        اعشاری tmp = type_conversion_جدول<src_t>::مقصد_f32(*(ثابت src_t *) src0_ptr);
                        *(dst_t *) dst_ptr = type_conversion_جدول<dst_t>::from_f32(tmp);

                        اگر (++i10 == ne0) {
                            i10 = 0;
                            اگر (++i11 == ne1) {
                                i11 = 0;
                                اگر (++i12 == ne2) {
                                    i12 = 0;
                                    اگر (++i13 == ne3) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                جب (i10 >= ne0) {
                    i10 -= ne0;
                    اگر (++i11 == ne1) {
                        i11 = 0;
                        اگر (++i12 == ne2) {
                            i12 = 0;
                            اگر (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
            }
        }
    }
}


سانچہ<قسم_نام src_t>
ساکن خالی ggml_compute_forward_dup_مقصد_q(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(!ggml_is_quantized(src0->type));

    GGML_TENSOR_UNARY_OP_LOCALS

    ثابت صحیح ith = params->ith; // thread index
    ثابت صحیح nth = params->nth; // number of threads

    // parallelize by rows
    ثابت صحیح nr = ne01;
    // number of rows per thread
    ثابت صحیح dr = (nr + nth - 1) / nth;
    // row range for this thread
    ثابت صحیح ir0 = dr * ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    اگر (ggml_is_contiguous(dst) &&
            nb00 == حجم(src_t) &&
            ggml_get_type_traits_cpu(dst->type)->from_float) {
        // casting non-quantized types --> intermediate f32 --> quantized
        ggml_from_float_t ثابت quantize_صف_q = ggml_get_type_traits_cpu(dst->type)->from_float;
        اعشاری * src0_f32 = (اعشاری *) params->wdata + (ne00 + CACHE_سطر_سائز_F32) * ith;

        size_t id = 0;
        size_t rs = nb0 * (ne00 / ggml_blck_سائز(dst->type));
        حرف * dst_ptr = (حرف *) dst->ڈیٹا;

        برائے (صحیح i03 = 0; i03 < ne03; i03++) {
            برائے (صحیح i02 = 0; i02 < ne02; i02++) {
                id += rs * ir0;
                برائے (صحیح i01 = ir0; i01 < ir1; i01++) {
                    ثابت src_t * src0_ptr = (src_t *) ((حرف *) src0->ڈیٹا + i01*nb01 + i02*nb02 + i03*nb03);

                    برائے (صحیح i00 = 0; i00 < ne00; i00++) {
                        src0_f32[i00] = type_conversion_جدول<src_t>::مقصد_f32(src0_ptr[i00]);
                    }

                    quantize_صف_q(src0_f32, dst_ptr + id, ne00);
                    id += rs;
                }
                id += rs * (ne01 - ir1);
            }
        }
    } ورنہ {
        // printf("%s %s\n", ggml_type_name(src0->type), ggml_type_name(dst->type));
        GGML_ABORT("not implemented");
    }
}

// A simplified version of ggml_compute_forward_dup that doesn't do float upcasting, and just plain old memcpy.
ساکن خالی ggml_compute_forward_dup_بائٹیں(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(src0->type == dst->type);

    GGML_TENSOR_UNARY_OP_LOCALS;

    اگر (ggml_is_contiguous(src0) && ggml_is_contiguous(dst)) {
        ggml_compute_forward_dup_same_cont(params, dst);
        واپس;
    }

    ثابت size_t type_سائز = ggml_type_سائز(src0->type);

    ثابت صحیح ith = params->ith; // thread index
    ثابت صحیح nth = params->nth; // number of threads

    // parallelize by rows
    ثابت صحیح nr = ne01;
    // number of rows per thread
    ثابت صحیح dr = (nr + nth - 1) / nth;
    // row range for this thread
    ثابت صحیح ir0 = dr * ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    اگر (src0->type == dst->type &&
        ggml_are_same_shape(src0, dst) &&
        nb00 == type_سائز && nb0 == type_سائز) {
        // copy by rows
        ثابت size_t rs = ggml_صف_سائز(src0->type, ne00);
        برائے (int64_t i03 = 0; i03 < ne03; i03++) {
            برائے (int64_t i02 = 0; i02 < ne02; i02++) {
                برائے (int64_t i01 = ir0; i01 < ir1; i01++) {
                    حافظہ_نقل(
                        ((حرف *)  dst->ڈیٹا + i01*nb1  + i02*nb2  + i03*nb3),
                        ((حرف *) src0->ڈیٹا + i01*nb01 + i02*nb02 + i03*nb03),
                        rs);
                }
            }
        }
        واپس;
    }

    اگر (ggml_is_contiguous(dst)) {
        size_t id = 0;
        حرف * dst_ptr = (حرف *) dst->ڈیٹا;
        ثابت size_t rs = ne00 * type_سائز;

        اگر (nb00 == type_سائز) {
            // src0 is contiguous on first dimension, copy by rows
            برائے (int64_t i03 = 0; i03 < ne03; i03++) {
                برائے (int64_t i02 = 0; i02 < ne02; i02++) {
                    id += rs * ir0;
                    برائے (int64_t i01 = ir0; i01 < ir1; i01++) {
                        ثابت حرف * src0_ptr = (حرف *) src0->ڈیٹا + i01*nb01 + i02*nb02 + i03*nb03;
                        حافظہ_نقل(dst_ptr + id, src0_ptr, rs);
                        id += rs;
                    }
                    id += rs * (ne01 - ir1);
                }
            }
        } ورنہ {
            //printf("%s: this is not optimal - fix me\n", __func__);

            برائے (int64_t i03 = 0; i03 < ne03; i03++) {
                برائے (int64_t i02 = 0; i02 < ne02; i02++) {
                    id += rs * ir0;
                    برائے (int64_t i01 = ir0; i01 < ir1; i01++) {
                        برائے (int64_t i00 = 0; i00 < ne00; i00++) {
                            ثابت حرف * src0_ptr = (حرف *) src0->ڈیٹا + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03;
                            حافظہ_نقل(dst_ptr + id, src0_ptr, type_سائز);

                            id += type_سائز;
                        }
                    }
                    id += rs * (ne01 - ir1);
                }
            }
        }

        واپس;
    }

    // dst counters
    int64_t k10 = 0;
    int64_t i11 = 0;
    int64_t i12 = 0;
    int64_t i13 = 0;

    // number of blocks in a row
    ثابت int64_t nk00 = ne00 / ggml_blck_سائز(src0->type);
    ثابت int64_t nk0  = ne0  / ggml_blck_سائز(dst->type);

    برائے (int64_t i03 = 0; i03 < ne03; i03++) {
        برائے (int64_t i02 = 0; i02 < ne02; i02++) {
            k10 += nk00 * ir0;
            جب (k10 >= nk0) {
                k10 -= nk0;
                اگر (++i11 == ne1) {
                    i11 = 0;
                    اگر (++i12 == ne2) {
                        i12 = 0;
                        اگر (++i13 == ne3) {
                            i13 = 0;
                        }
                    }
                }
            }
            برائے (int64_t i01 = ir0; i01 < ir1; i01++) {
                برائے (int64_t k00 = 0; k00 < nk00; k00++) {
                    ثابت حرف * src0_ptr = ((حرف *) src0->ڈیٹا + k00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                          حرف * dst_ptr  = ((حرف *)  dst->ڈیٹا + k10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                    حافظہ_نقل(dst_ptr, src0_ptr, type_سائز);

                    اگر (++k10 == nk0) {
                        k10 = 0;
                        اگر (++i11 == ne1) {
                            i11 = 0;
                            اگر (++i12 == ne2) {
                                i12 = 0;
                                اگر (++i13 == ne3) {
                                    i13 = 0;
                                }
                            }
                        }
                    }
                }
            }
            k10 += nk00 * (ne01 - ir1);
            جب (k10 >= nk0) {
                k10 -= nk0;
                اگر (++i11 == ne1) {
                    i11 = 0;
                    اگر (++i12 == ne2) {
                        i12 = 0;
                        اگر (++i13 == ne3) {
                            i13 = 0;
                        }
                    }
                }
            }
        }
    }
}

ساکن خالی ggml_compute_forward_dup_from_q(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ثنائی_OP_LOCALS

    ثابت ggml_type type = src0->type;
    ggml_مقصد_float_t ثابت dequantize_صف_q = ggml_get_type_traits(type)->مقصد_float;

    size_t qk = ggml_blck_سائز(type);
    ثابت int64_t nr = ggml_nelements(src1) / qk;

    // destination must be contiguous in the first dimension
    GGML_ASSERT(nb10 == ggml_type_سائز(dst->type));
    // must either have first dimension large enough to hold a row, or fully contiguous
    GGML_ASSERT((ne10 % qk) == 0 || ggml_is_contiguous(dst));

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحیح ir0 = dr*ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    برائے (int64_t ir = ir0; ir < ir1; ++ir) {

        uint32_t i = ir * qk;

        ثابت int64_t i03 = i/(ne00 * ne01 * ne02);
        ثابت int64_t i02 = (i - i03*ne00*ne01*ne02 )/ (ne00*ne01);
        ثابت int64_t i01 = (i - i03*ne00*ne01*ne02  -  i02*ne01*ne00) / ne00;
        ثابت int64_t i00 = i - i03*ne00*ne01*ne02 - i02*ne01*ne00 - i01*ne00;
        ثابت int64_t x_انحراف = (i00/qk)*nb00 + i01*nb01 + i02*nb02 + i03 * nb03;

        ثابت int64_t i13 = i/(ne10 * ne11 * ne12);
        ثابت int64_t i12 = (i - i13*ne10*ne11*ne12) / (ne10*ne11);
        ثابت int64_t i11 = (i - i13*ne10*ne11*ne12 - i12*ne10*ne11) / ne10;
        ثابت int64_t i10 = i - i13*ne10*ne11*ne12 - i12*ne10*ne11 - i11*ne10;
        ثابت int64_t dst_انحراف = i10*nb10 + i11*nb11 + i12*nb12 + i13*nb13;

        dequantize_صف_q(
                (ثابت خالی *) ((حرف *) src0->ڈیٹا + x_انحراف),
                     (اعشاری *) ((حرف *)  dst->ڈیٹا + dst_انحراف), qk);
    }
}

خالی ggml_compute_forward_dup(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    اگر (src0->type == dst->type) {
        ggml_compute_forward_dup_بائٹیں(params, dst);
        واپس;
    }

    انتخاب (src0->type) {
        حالت GGML_TYPE_F16:
            {
                /**/ اگر (dst->type == GGML_TYPE_F16)  ggml_compute_forward_dup_flt<ggml_fp16_t, ggml_fp16_t>(params, dst);
                ورنہ اگر (dst->type == GGML_TYPE_BF16) ggml_compute_forward_dup_flt<ggml_fp16_t, ggml_bf16_t>(params, dst);
                ورنہ اگر (dst->type == GGML_TYPE_F32)  ggml_compute_forward_dup_flt<ggml_fp16_t, اعشاری      >(params, dst);
                ورنہ ggml_compute_forward_dup_مقصد_q<ggml_fp16_t>(params, dst);
            } توڑو;
        حالت GGML_TYPE_BF16:
            {
                /**/ اگر (dst->type == GGML_TYPE_F16)  ggml_compute_forward_dup_flt<ggml_bf16_t, ggml_fp16_t>(params, dst);
                ورنہ اگر (dst->type == GGML_TYPE_BF16) ggml_compute_forward_dup_flt<ggml_bf16_t, ggml_bf16_t>(params, dst);
                ورنہ اگر (dst->type == GGML_TYPE_F32)  ggml_compute_forward_dup_flt<ggml_bf16_t, اعشاری      >(params, dst);
                ورنہ ggml_compute_forward_dup_مقصد_q<ggml_bf16_t>(params, dst);
            } توڑو;
        حالت GGML_TYPE_F32:
            {
                /**/ اگر (dst->type == GGML_TYPE_F16)  ggml_compute_forward_dup_flt<اعشاری, ggml_fp16_t>(params, dst);
                ورنہ اگر (dst->type == GGML_TYPE_BF16) ggml_compute_forward_dup_flt<اعشاری, ggml_bf16_t>(params, dst);
                ورنہ اگر (dst->type == GGML_TYPE_F32)  ggml_compute_forward_dup_flt<اعشاری, اعشاری      >(params, dst);
                ورنہ اگر (dst->type == GGML_TYPE_I32)  ggml_compute_forward_dup_flt<اعشاری, int32_t    >(params, dst);
                ورنہ ggml_compute_forward_dup_مقصد_q<اعشاری>(params, dst);
            } توڑو;
        حالت GGML_TYPE_I32:
            {
                اگر (dst->type == GGML_TYPE_F32) ggml_compute_forward_dup_flt<int32_t, اعشاری>(params, dst);
                ورنہ GGML_ABORT("not implemented");
            } توڑو;
        طےشدہ:
            {
                اگر (ggml_is_quantized(src0->type) && dst->type == GGML_TYPE_F32) {
                    ggml_compute_forward_dup_from_q(params, dst);
                    توڑو;
                }
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_add

ساکن خالی ggml_compute_forward_جوڑو_q_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

    ثابت صحیح nr  = ggml_nrows(src0);

    GGML_TENSOR_ثنائی_OP_LOCALS

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت ggml_type type = src0->type;
    ثابت ggml_type dtype = dst->type;
    ggml_مقصد_float_t ثابت dequantize_صف_q = ggml_get_type_traits(type)->مقصد_float;
    ggml_from_float_t ثابت quantize_صف_q = ggml_get_type_traits_cpu(dtype)->from_float;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_سائز(type));
    GGML_ASSERT(nb10 == حجم(اعشاری));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ggml_is_quantized(src0->type));
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    // rows per thread
    ثابت صحیح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحیح ir0 = dr*ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    اعشاری * wdata = (اعشاری *) params->wdata + (ne00 + CACHE_سطر_سائز_F32) * ith;

    برائے (صحیح ir = ir0; ir < ir1; ++ir) {
        // src0 indices
        ثابت صحیح i03 = ir/(ne02*ne01);
        ثابت صحیح i02 = (ir - i03*ne02*ne01)/ne01;
        ثابت صحیح i01 = (ir - i03*ne02*ne01 - i02*ne01);

        // src1 and dst are same shape as src0 => same indices
        ثابت صحیح i13 = i03;
        ثابت صحیح i12 = i02;
        ثابت صحیح i11 = i01;

        ثابت صحیح i3 = i03;
        ثابت صحیح i2 = i02;
        ثابت صحیح i1 = i01;

        خالی  * src0_صف = (خالی *) ((حرف *) src0->ڈیٹا + (i01*nb01 + i02*nb02 + i03*nb03));
        اعشاری * src1_صف = (اعشاری *)((حرف *) src1->ڈیٹا + (i11*nb11 + i12*nb12 + i13*nb13));
        خالی  * dst_صف  = (خالی *) ((حرف *)  dst->ڈیٹا + ( i1*nb1  +  i2*nb2  +  i3*nb3));

        أكد(ne00 % 32 == 0);

        // unquantize row from src0 to temp buffer
        dequantize_صف_q(src0_صف, wdata, ne00);
        // add src1
        ggml_vec_acc_f32(ne00, wdata, src1_صف);
        // quantize row to dst
        اگر (quantize_صف_q != تہی) {
            quantize_صف_q(wdata, dst_صف, ne00);
        } ورنہ {
            حافظہ_نقل(dst_صف, wdata, ne0*nb0);
        }
    }
}

خالی ggml_compute_forward_جوڑو(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
        حالت GGML_TYPE_F16:
        حالت GGML_TYPE_BF16:
            {
                ggml_compute_forward_جوڑو_non_quantized(params, dst);
            } توڑو;
        حالت GGML_TYPE_Q1_0:
        حالت GGML_TYPE_Q2_0:
        حالت GGML_TYPE_Q4_0:
        حالت GGML_TYPE_Q4_1:
        حالت GGML_TYPE_Q5_0:
        حالت GGML_TYPE_Q5_1:
        حالت GGML_TYPE_Q8_0:
        حالت GGML_TYPE_MXFP4:
        حالت GGML_TYPE_NVFP4:
        حالت GGML_TYPE_Q2_K:
        حالت GGML_TYPE_Q3_K:
        حالت GGML_TYPE_Q4_K:
        حالت GGML_TYPE_Q5_K:
        حالت GGML_TYPE_Q6_K:
        حالت GGML_TYPE_TQ1_0:
        حالت GGML_TYPE_TQ2_0:
        حالت GGML_TYPE_IQ2_XXS:
        حالت GGML_TYPE_IQ2_XS:
        حالت GGML_TYPE_IQ3_XXS:
        حالت GGML_TYPE_IQ1_S:
        حالت GGML_TYPE_IQ1_M:
        حالت GGML_TYPE_IQ4_NL:
        حالت GGML_TYPE_IQ4_XS:
        حالت GGML_TYPE_IQ3_S:
        حالت GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_جوڑو_q_f32(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_add_id

ساکن خالی ggml_compute_forward_جوڑو_id_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    ثابت ggml_tensor * src2 = dst->src[2];

    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src2->type == GGML_TYPE_I32);

    GGML_ASSERT(src0->nb[0] == حجم(اعشاری));
    GGML_ASSERT(src1->nb[0] == حجم(اعشاری));

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح nr  = ggml_nrows(src0);

    GGML_TENSOR_TERNARY_OP_LOCALS

    GGML_ASSERT( nb0 == حجم(اعشاری));
    GGML_ASSERT(nb10 == حجم(اعشاری));

    // rows per thread
    ثابت صحیح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحیح ir0 = dr*ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    برائے (صحیح ir = ir0; ir < ir1; ++ir) {
        // src0 indices
        ثابت صحیح i3 = ir/(ne2*ne1);
        ثابت صحیح i2 = (ir - i3*ne2*ne1)/ne1;
        ثابت صحیح i1 = (ir - i3*ne2*ne1 - i2*ne1);

        // src1 indices
        ثابت صحیح i11 = *(int32_t *) ((حرف *) src2->ڈیٹا + i1*nb20 + i2*nb21);

        GGML_ASSERT(i11 >= 0 && i11 < ne11);

        ggml_vec_جوڑو_f32(ne0,
                (اعشاری *) ((حرف *) dst->ڈیٹا  + i3*nb3  + i2*nb2  + i1*nb1 ),
                (اعشاری *) ((حرف *) src0->ڈیٹا + i3*nb03 + i2*nb02 + i1*nb01),
                (اعشاری *) ((حرف *) src1->ڈیٹا + i11*nb11));
    }
}

خالی ggml_compute_forward_جوڑو_id(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_جوڑو_id_f32(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("unsupported type for ggml_compute_forward_add_id: %s", ggml_type_name(src0->type));
            }
    }
}

// ggml_compute_forward_add1

ساکن خالی ggml_compute_forward_add1_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == حجم(اعشاری));
    GGML_ASSERT(nb00 == حجم(اعشاری));

    // rows per thread
    ثابت صحیح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحیح ir0 = dr*ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    برائے (صحیح ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        ثابت صحیح i3 = ir/(ne2*ne1);
        ثابت صحیح i2 = (ir - i3*ne2*ne1)/ne1;
        ثابت صحیح i1 = (ir - i3*ne2*ne1 - i2*ne1);

#اگر_تعریف_شدہ GGML_USE_ACCELERATE
        GGML_غیر_مستعمل(ggml_vec_add1_f32);

        vDSP_vadd(
                (اعشاری *) ((حرف *) src0->ڈیٹا + i3*nb03 + i2*nb02 + i1*nb01), 1,
                (اعشاری *) ((حرف *) src1->ڈیٹا), 0,
                (اعشاری *) ((حرف *) dst->ڈیٹا  + i3*nb3  + i2*nb2  + i1*nb1 ), 1,
                ne0);
#ورنہ
        ggml_vec_add1_f32(ne0,
                (اعشاری *) ((حرف *) dst->ڈیٹا  + i3*nb3  + i2*nb2  + i1*nb1 ),
                (اعشاری *) ((حرف *) src0->ڈیٹا + i3*nb03 + i2*nb02 + i1*nb01),
               *(اعشاری *) src1->ڈیٹا);
#اگر_ختم
    }
}

ساکن خالی ggml_compute_forward_add1_f16_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    ثابت اعشاری v = *(اعشاری *) src1->ڈیٹا;

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F16);

    GGML_ASSERT( nb0 == حجم(ggml_fp16_t));
    GGML_ASSERT(nb00 == حجم(ggml_fp16_t));

    // rows per thread
    ثابت صحیح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحیح ir0 = dr*ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    برائے (صحیح ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        ثابت صحیح i3 = ir/(ne2*ne1);
        ثابت صحیح i2 = (ir - i3*ne2*ne1)/ne1;
        ثابت صحیح i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((حرف *) dst->ڈیٹا  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((حرف *) src0->ڈیٹا + i3*nb03 + i2*nb02 + i1*nb01);
        برائے (صحیح i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_CPU_FP32_مقصد_FP16(GGML_CPU_FP16_مقصد_FP32(src0_ptr[i]) + v);
        }
    }
}

ساکن خالی ggml_compute_forward_add1_f16_f16(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    ثابت اعشاری v = GGML_CPU_FP16_مقصد_FP32(*(ggml_fp16_t *) src1->ڈیٹا);

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type  == GGML_TYPE_F16);

    GGML_ASSERT( nb0 == حجم(ggml_fp16_t));
    GGML_ASSERT(nb00 == حجم(ggml_fp16_t));

    // rows per thread
    ثابت صحیح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحیح ir0 = dr*ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    برائے (صحیح ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        ثابت صحیح i3 = ir/(ne2*ne1);
        ثابت صحیح i2 = (ir - i3*ne2*ne1)/ne1;
        ثابت صحیح i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((حرف *) dst->ڈیٹا  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((حرف *) src0->ڈیٹا + i3*nb03 + i2*nb02 + i1*nb01);
        برائے (صحیح i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_CPU_FP32_مقصد_FP16(GGML_CPU_FP16_مقصد_FP32(src0_ptr[i]) + v);
        }
    }
}

ساکن خالی ggml_compute_forward_add1_q_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    ثابت اعشاری v = *(اعشاری *) src1->ڈیٹا;

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    ثابت ggml_type type = src0->type;
    ggml_مقصد_float_t ثابت dequantize_صف_q = ggml_get_type_traits(type)->مقصد_float;
    ggml_from_float_t ثابت quantize_صف_q = ggml_get_type_traits_cpu(type)->from_float;

    // we don't support permuted src0
    GGML_ASSERT(nb00 == ggml_type_سائز(type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ggml_is_quantized(src0->type));
    GGML_ASSERT(dst->type == src0->type);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    // rows per thread
    ثابت صحیح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحیح ir0 = dr*ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    اعشاری * wdata = (اعشاری *) params->wdata + (ne0 + CACHE_سطر_سائز_F32) * ith;

    برائے (صحیح ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        ثابت صحیح i3 = ir/(ne2*ne1);
        ثابت صحیح i2 = (ir - i3*ne2*ne1)/ne1;
        ثابت صحیح i1 = (ir - i3*ne2*ne1 - i2*ne1);

        خالی  * src0_صف = (خالی *) ((حرف *) src0->ڈیٹا + (i1*nb01 + i2*nb02 + i3*nb03));
        خالی  * dst_صف  = (خالی *) ((حرف *)  dst->ڈیٹا + (i1*nb1  + i2*nb2  + i3*nb0 ));

        أكد(ne0 % 32 == 0);

        // unquantize row from src0 to temp buffer
        dequantize_صف_q(src0_صف, wdata, ne0);
        // add src1
        ggml_vec_acc1_f32(ne0, wdata, v);
        // quantize row to dst
        quantize_صف_q(wdata, dst_صف, ne0);
    }
}

ساکن خالی ggml_compute_forward_add1_bf16_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    ثابت اعشاری v = *(اعشاری *) src1->ڈیٹا;

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_BF16);

    GGML_ASSERT( nb0 == حجم(ggml_bf16_t));
    GGML_ASSERT(nb00 == حجم(ggml_bf16_t));

    // rows per thread
    ثابت صحیح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحیح ir0 = dr*ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    برائے (صحیح ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        ثابت صحیح i3 = ir/(ne2*ne1);
        ثابت صحیح i2 = (ir - i3*ne2*ne1)/ne1;
        ثابت صحیح i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_bf16_t * dst_ptr  = (ggml_bf16_t *) ((حرف *) dst->ڈیٹا  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_bf16_t * src0_ptr = (ggml_bf16_t *) ((حرف *) src0->ڈیٹا + i3*nb03 + i2*nb02 + i1*nb01);
        برائے (صحیح i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_FP32_مقصد_BF16(GGML_BF16_مقصد_FP32(src0_ptr[i]) + v);
        }
    }
}

ساکن خالی ggml_compute_forward_add1_bf16_bf16(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    ثابت اعشاری v = GGML_BF16_مقصد_FP32(*(ggml_bf16_t *) src1->ڈیٹا);

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(src1->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->type  == GGML_TYPE_BF16);

    GGML_ASSERT( nb0 == حجم(ggml_bf16_t));
    GGML_ASSERT(nb00 == حجم(ggml_bf16_t));

    // rows per thread
    ثابت صحیح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحیح ir0 = dr*ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    برائے (صحیح ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        ثابت صحیح i3 = ir/(ne2*ne1);
        ثابت صحیح i2 = (ir - i3*ne2*ne1)/ne1;
        ثابت صحیح i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_bf16_t * dst_ptr  = (ggml_bf16_t *) ((حرف *) dst->ڈیٹا  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_bf16_t * src0_ptr = (ggml_bf16_t *) ((حرف *) src0->ڈیٹا + i3*nb03 + i2*nb02 + i1*nb01);
        برائے (صحیح i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_FP32_مقصد_BF16(GGML_BF16_مقصد_FP32(src0_ptr[i]) + v);
        }
    }
}

خالی ggml_compute_forward_add1(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_add1_f32(params, dst);
            } توڑو;
        حالت GGML_TYPE_F16:
            {
                اگر (src1->type == GGML_TYPE_F16) {
                    ggml_compute_forward_add1_f16_f16(params, dst);
                }
                ورنہ اگر (src1->type == GGML_TYPE_F32) {
                    ggml_compute_forward_add1_f16_f32(params, dst);
                }
                ورنہ {
                    GGML_ABORT("fatal error");
                }
            } توڑو;
        حالت GGML_TYPE_BF16:
            {
                اگر (src1->type == GGML_TYPE_BF16) {
                    ggml_compute_forward_add1_bf16_bf16(params, dst);
                }
                ورنہ اگر (src1->type == GGML_TYPE_F32) {
                    ggml_compute_forward_add1_bf16_f32(params, dst);
                }
                ورنہ {
                    GGML_ABORT("fatal error");
                }
            } توڑو;
        حالت GGML_TYPE_Q1_0:
        حالت GGML_TYPE_Q2_0:
        حالت GGML_TYPE_Q4_0:
        حالت GGML_TYPE_Q4_1:
        حالت GGML_TYPE_Q5_0:
        حالت GGML_TYPE_Q5_1:
        حالت GGML_TYPE_Q8_0:
        حالت GGML_TYPE_Q8_1:
        حالت GGML_TYPE_MXFP4:
        حالت GGML_TYPE_NVFP4:
        حالت GGML_TYPE_Q2_K:
        حالت GGML_TYPE_Q3_K:
        حالت GGML_TYPE_Q4_K:
        حالت GGML_TYPE_Q5_K:
        حالت GGML_TYPE_Q6_K:
        حالت GGML_TYPE_TQ1_0:
        حالت GGML_TYPE_TQ2_0:
        حالت GGML_TYPE_IQ2_XXS:
        حالت GGML_TYPE_IQ2_XS:
        حالت GGML_TYPE_IQ3_XXS:
        حالت GGML_TYPE_IQ1_S:
        حالت GGML_TYPE_IQ1_M:
        حالت GGML_TYPE_IQ4_NL:
        حالت GGML_TYPE_IQ4_XS:
        حالت GGML_TYPE_IQ3_S:
        حالت GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_add1_q_f32(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_acc

ساکن خالی ggml_compute_forward_acc_f32(
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
    size_t انحراف  = ((int32_t *) dst->op_params)[3];
    منطقی   inplace = (منطقی) ((int32_t *) dst->op_params)[4];

    اگر (!inplace) {
        اگر (params->ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            حافظہ_نقل(
                ((حرف *)  dst->ڈیٹا),
                ((حرف *) src0->ڈیٹا),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح nr = ggml_nrows(src1);
    ثابت صحیح nc = src1->ne[0];

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne)
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb)

    // src0 and dst as viewed during acc
    ثابت size_t nb0 = ggml_element_سائز(src0);

    ثابت size_t nb00 = nb0;
    ثابت size_t nb01 = nb1;
    ثابت size_t nb02 = nb2;
    ثابت size_t nb03 = nb3;

    GGML_ASSERT(انحراف + (ne10 == 0 ? 0 : ne10-1)*nb0  + (ne11 == 0 ? 0 : ne11-1)*nb1  + (ne12 == 0 ? 0 : ne12-1)*nb2  + (ne13 == 0 ? 0 : ne13-1)*nb3  < ggml_nbytes(dst));
    GGML_ASSERT(انحراف + (ne10 == 0 ? 0 : ne10-1)*nb00 + (ne11 == 0 ? 0 : ne11-1)*nb01 + (ne12 == 0 ? 0 : ne12-1)*nb02 + (ne13 == 0 ? 0 : ne13-1)*nb03 < ggml_nbytes(src0));

    GGML_ASSERT(nb10 == حجم(اعشاری));

    // rows per thread
    ثابت صحیح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحیح ir0 = dr*ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    برائے (صحیح ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are viewed with shape of src1 and offset
        // => same indices
        ثابت صحیح i3 = ir/(ne12*ne11);
        ثابت صحیح i2 = (ir - i3*ne12*ne11)/ne11;
        ثابت صحیح i1 = (ir - i3*ne12*ne11 - i2*ne11);

#اگر_تعریف_شدہ GGML_USE_ACCELERATE
        vDSP_vadd(
                (اعشاری *) ((حرف *) src0->ڈیٹا + i3*nb03 + i2*nb02 + i1*nb01 + انحراف), 1,
                (اعشاری *) ((حرف *) src1->ڈیٹا + i3*nb13 + i2*nb12 + i1*nb11), 1,
                (اعشاری *) ((حرف *) dst->ڈیٹا  + i3*nb3  + i2*nb2  + i1*nb1  + انحراف), 1, nc);
#ورنہ
        ggml_vec_جوڑو_f32(nc,
                (اعشاری *) ((حرف *)  dst->ڈیٹا + i3*nb3  + i2*nb2  + i1*nb1  + انحراف),
                (اعشاری *) ((حرف *) src0->ڈیٹا + i3*nb03 + i2*nb02 + i1*nb01 + انحراف),
                (اعشاری *) ((حرف *) src1->ڈیٹا + i3*nb13 + i2*nb12 + i1*nb11));
#اگر_ختم
    }
}

خالی ggml_compute_forward_acc(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_acc_f32(params, dst);
            } توڑو;
        حالت GGML_TYPE_F16:
        حالت GGML_TYPE_BF16:
        حالت GGML_TYPE_Q1_0:
        حالت GGML_TYPE_Q2_0:
        حالت GGML_TYPE_Q4_0:
        حالت GGML_TYPE_Q4_1:
        حالت GGML_TYPE_Q5_0:
        حالت GGML_TYPE_Q5_1:
        حالت GGML_TYPE_Q8_0:
        حالت GGML_TYPE_Q8_1:
        حالت GGML_TYPE_MXFP4:
        حالت GGML_TYPE_NVFP4:
        حالت GGML_TYPE_Q2_K:
        حالت GGML_TYPE_Q3_K:
        حالت GGML_TYPE_Q4_K:
        حالت GGML_TYPE_Q5_K:
        حالت GGML_TYPE_Q6_K:
        حالت GGML_TYPE_TQ1_0:
        حالت GGML_TYPE_TQ2_0:
        حالت GGML_TYPE_IQ2_XXS:
        حالت GGML_TYPE_IQ2_XS:
        حالت GGML_TYPE_IQ3_XXS:
        حالت GGML_TYPE_IQ1_S:
        حالت GGML_TYPE_IQ1_M:
        حالت GGML_TYPE_IQ4_NL:
        حالت GGML_TYPE_IQ4_XS:
        حالت GGML_TYPE_IQ3_S:
        حالت GGML_TYPE_IQ2_S:
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_sum

ساکن خالی ggml_compute_forward_مجموع_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    اگر (params->ith != 0) {
        واپس;
    }

    أكد(ggml_is_scalar(dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)

    ggml_float مجموع = 0;

    اگر (nb00 == حجم(اعشاری)) {
        ggml_float صف_مجموع = 0;
        برائے (int64_t i03 = 0; i03 < ne03; i03++) {
            برائے (int64_t i02 = 0; i02 < ne02; i02++) {
                برائے (int64_t i01 = 0; i01 < ne01; i01++) {
                    ggml_vec_مجموع_f32_ggf(ne00,
                            &صف_مجموع,
                            (اعشاری *) ((حرف *) src0->ڈیٹا + i01*nb01 + i02*nb02 + i03*nb03));
                    مجموع += صف_مجموع;
                }
            }
        }
    } ورنہ {
        برائے (int64_t i03 = 0; i03 < ne03; i03++) {
            برائے (int64_t i02 = 0; i02 < ne02; i02++) {
                برائے (int64_t i01 = 0; i01 < ne01; i01++) {
                    ثابت حرف * صف = (ثابت حرف *) src0->ڈیٹا + i01*nb01 + i02*nb02 + i03*nb03;
                    برائے (int64_t i00 = 0; i00 < ne00; i00++) {
                        مجموع += (ggml_float) *(ثابت اعشاری *) (صف + i00*nb00);
                    }
                }
            }
        }
    }

    ((اعشاری *) dst->ڈیٹا)[0] = مجموع;
}

ساکن خالی ggml_compute_forward_مجموع_f16(
    ثابت ggml_compute_params * params,
          ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    اگر (params->ith != 0) {
        واپس;
    }

    أكد(ggml_is_scalar(dst));

    أكد(src0->nb[0] == حجم(ggml_fp16_t));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)

    اعشاری مجموع = 0;
    اعشاری صف_مجموع = 0;

    برائے (int64_t i03 = 0; i03 < ne03; i03++) {
        برائے (int64_t i02 = 0; i02 < ne02; i02++) {
            برائے (int64_t i01 = 0; i01 < ne01; i01++) {
                ggml_vec_مجموع_f16_ggf(ne00,
                    &صف_مجموع,
                    (ggml_fp16_t *) ((حرف *) src0->ڈیٹا + i01 * nb01 + i02 * nb02 + i03 * nb03));
                مجموع += صف_مجموع;
            }
        }
    }
    ((ggml_fp16_t *) dst->ڈیٹا)[0] = GGML_CPU_FP32_مقصد_FP16(مجموع);
}

ساکن خالی ggml_compute_forward_مجموع_bf16(
    ثابت ggml_compute_params * params,
          ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    اگر (params->ith != 0) {
        واپس;
    }

    أكد(ggml_is_scalar(dst));

    أكد(src0->nb[0] == حجم(ggml_bf16_t));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)

    اعشاری مجموع = 0;
    اعشاری صف_مجموع = 0;

    برائے (int64_t i03 = 0; i03 < ne03; i03++) {
        برائے (int64_t i02 = 0; i02 < ne02; i02++) {
            برائے (int64_t i01 = 0; i01 < ne01; i01++) {
                ggml_vec_مجموع_bf16_ggf(ne00,
                    &صف_مجموع,
                    (ggml_bf16_t *) ((حرف *) src0->ڈیٹا + i01 * nb01 + i02 * nb02 + i03 * nb03));
                مجموع += صف_مجموع;
            }
        }
    }
    ((ggml_bf16_t *) dst->ڈیٹا)[0] = GGML_FP32_مقصد_BF16(مجموع);
}

خالی ggml_compute_forward_مجموع(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_مجموع_f32(params, dst);
            } توڑو;
        حالت GGML_TYPE_F16:
            {
                ggml_compute_forward_مجموع_f16(params, dst);
            } توڑو;
        حالت GGML_TYPE_BF16:
            {
                ggml_compute_forward_مجموع_bf16(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_cumsum

ساکن خالی ggml_compute_forward_cumsum_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->nb[0] == حجم(اعشاری));
    GGML_ASSERT(dst->nb[0] == حجم(اعشاری));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne01);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    ثابت خودکار [ir0, ir1] = get_thread_range(params, src0);

    برائے (int64_t ir = ir0; ir < ir1; ++ir) {
        ثابت int64_t i03 = ir/(ne02*ne01);
        ثابت int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        ثابت int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        اعشاری * src_صف = (اعشاری *) ((حرف *) src0->ڈیٹا + i01*nb01 + i02*nb02 + i03*nb03);
        اعشاری * dst_صف = (اعشاری *) ((حرف *) dst->ڈیٹا  + i01*nb1  + i02*nb2  + i03*nb3);

        ggml_vec_cumsum_f32(ne00, dst_صف, src_صف);
    }
}

خالی ggml_compute_forward_cumsum(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_cumsum_f32(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_sum_rows

ساکن خالی ggml_compute_forward_مجموع_صفیں_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    اگر (params->ith != 0) {
        واپس;
    }

    GGML_ASSERT(src0->nb[0] == حجم(اعشاری));
    GGML_ASSERT(dst->nb[0] == حجم(اعشاری));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(ne0 == 1);
    GGML_ASSERT(ne1 == ne01);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    برائے (int64_t i3 = 0; i3 < ne03; i3++) {
        برائے (int64_t i2 = 0; i2 < ne02; i2++) {
            برائے (int64_t i1 = 0; i1 < ne01; i1++) {
                اعشاری * src_صف = (اعشاری *) ((حرف *) src0->ڈیٹا + i1*nb01 + i2*nb02 + i3*nb03);
                اعشاری * dst_صف = (اعشاری *) ((حرف *) dst->ڈیٹا  + i1*nb1  + i2*nb2  + i3*nb3);
                اعشاری صف_مجموع = 0;
                ggml_vec_مجموع_f32(ne00, &صف_مجموع, src_صف);
                dst_صف[0] = صف_مجموع;
            }
        }
    }
}

خالی ggml_compute_forward_مجموع_صفیں(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_مجموع_صفیں_f32(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_mean

ساکن خالی ggml_compute_forward_اوسط_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    اگر (params->ith != 0) {
        واپس;
    }

    أكد(src0->nb[0] == حجم(اعشاری));

    GGML_TENSOR_UNARY_OP_LOCALS

    أكد(ne0 == 1);
    أكد(ne1 == ne01);
    أكد(ne2 == ne02);
    أكد(ne3 == ne03);

    GGML_غیر_مستعمل(ne0);
    GGML_غیر_مستعمل(ne1);
    GGML_غیر_مستعمل(ne2);
    GGML_غیر_مستعمل(ne3);

    برائے (int64_t i03 = 0; i03 < ne03; i03++) {
        برائے (int64_t i02 = 0; i02 < ne02; i02++) {
            برائے (int64_t i01 = 0; i01 < ne01; i01++) {
                ggml_vec_مجموع_f32(ne00,
                        (اعشاری *) ((حرف *)  dst->ڈیٹا + i01*nb1  + i02*nb2  + i03*nb3),
                        (اعشاری *) ((حرف *) src0->ڈیٹا + i01*nb01 + i02*nb02 + i03*nb03));

                *(اعشاری *) ((حرف *) dst->ڈیٹا + i01*nb1 + i02*nb2 + i03*nb3) /= (اعشاری) ne00;
            }
        }
    }
}

خالی ggml_compute_forward_اوسط(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_اوسط_f32(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_argmax

ساکن خالی ggml_compute_forward_argmax_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    اگر (params->ith != 0) {
        واپس;
    }

    أكد(src0->nb[0] == حجم(اعشاری));
    أكد(dst->nb[0] == حجم(اعشاری));

    ثابت int64_t ne00 = src0->ne[0];
    ثابت int64_t ne01 = src0->ne[1];

    ثابت size_t nb01 = src0->nb[1];
    ثابت size_t nb0 = dst->nb[0];

    برائے (int64_t i1 = 0; i1 < ne01; i1++) {
        اعشاری * src = (اعشاری *) ((حرف *) src0->ڈیٹا + i1*nb01);
        int32_t * dst_ = (int32_t *) ((حرف *)  dst->ڈیٹا + i1*nb0);
        صحیح v = 0;
        ggml_vec_argmax_f32(ne00, &v, src);
        dst_[0] = v;
    }
}

خالی ggml_compute_forward_argmax(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_argmax_f32(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_count_equal

ساکن خالی ggml_compute_forward_گنتی_equal_i32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ثنائی_OP_LOCALS;

    GGML_ASSERT(src0->type == GGML_TYPE_I32);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));
    GGML_ASSERT(dst->type == GGML_TYPE_I64);

    ثابت int64_t nr = ggml_nrows(src0);

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    int64_t * sums = (int64_t *) params->wdata;
    int64_t مجموع_thread = 0;

    // rows per thread
    ثابت int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت int64_t ir0 = dr*ith;
    ثابت int64_t ir1 = MIN(ir0 + dr, nr);

    برائے (int64_t ir = ir0; ir < ir1; ++ir) {
        ثابت int64_t i03 =  ir                        / (ne02*ne01);
        ثابت int64_t i02 = (ir - i03*ne03)            /       ne01;
        ثابت int64_t i01 =  ir - i03*ne03 - i02*ne02;

        ثابت حرف * data0 = (ثابت حرف *) src0->ڈیٹا + i03*nb03 + i02*nb02 + i01*nb01;
        ثابت حرف * data1 = (ثابت حرف *) src1->ڈیٹا + i03*nb13 + i02*nb12 + i01*nb11;

        برائے (int64_t i00 = 0; i00 < ne00; ++i00) {
            ثابت int32_t val0 = *((ثابت int32_t *) (data0 + i00*nb00));
            ثابت int32_t val1 = *((ثابت int32_t *) (data1 + i00*nb10));

            مجموع_thread += val0 == val1;
        }
    }
    اگر (ith != 0) {
        sums[ith] = مجموع_thread;
    }
    ggml_barrier(params->threadpool);

    اگر (ith != 0) {
        واپس;
    }

    برائے (صحیح ith_other = 1; ith_other < nth; ++ith_other) {
        مجموع_thread += sums[ith_other];
    }
    *((int64_t *) dst->ڈیٹا) = مجموع_thread;
}

خالی ggml_compute_forward_گنتی_equal(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_I32:
            {
                ggml_compute_forward_گنتی_equal_i32(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_repeat

ساکن خالی ggml_compute_forward_repeat_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    اگر (params->ith != 0) {
        واپس;
    }

    GGML_ASSERT(ggml_can_repeat(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    // guaranteed to be an integer due to the check in ggml_can_repeat
    ثابت صحیح nr0 = (صحیح)(ne0/ne00);
    ثابت صحیح nr1 = (صحیح)(ne1/ne01);
    ثابت صحیح nr2 = (صحیح)(ne2/ne02);
    ثابت صحیح nr3 = (صحیح)(ne3/ne03);

    // TODO: support for transposed / permuted tensors
    GGML_ASSERT(nb0  == حجم(اعشاری));
    GGML_ASSERT(nb00 == حجم(اعشاری));

    // TODO: maybe this is not optimal?
    برائے                         (صحیح i3 = 0; i3 < nr3;  i3++) {
        برائے                     (صحیح k3 = 0; k3 < ne03; k3++) {
            برائے                 (صحیح i2 = 0; i2 < nr2;  i2++) {
                برائے             (صحیح k2 = 0; k2 < ne02; k2++) {
                    برائے         (صحیح i1 = 0; i1 < nr1;  i1++) {
                        برائے     (صحیح k1 = 0; k1 < ne01; k1++) {
                            برائے (صحیح i0 = 0; i0 < nr0;  i0++) {
                                ggml_vec_cpy_f32(ne00,
                                        (اعشاری *) ((حرف *)  dst->ڈیٹا + (i3*ne03 + k3)*nb3  + (i2*ne02 + k2)*nb2  + (i1*ne01 + k1)*nb1  + (i0*ne00)*nb0),
                                        (اعشاری *) ((حرف *) src0->ڈیٹا + (          k3)*nb03 + (          k2)*nb02 + (          k1)*nb01));
                            }
                        }
                    }
                }
            }
        }
    }
}

ساکن خالی ggml_compute_forward_repeat_f16(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    اگر (params->ith != 0) {
        واپس;
    }

    GGML_ASSERT(ggml_can_repeat(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    // guaranteed to be an integer due to the check in ggml_can_repeat
    ثابت صحیح nr0 = (صحیح)(ne0/ne00);
    ثابت صحیح nr1 = (صحیح)(ne1/ne01);
    ثابت صحیح nr2 = (صحیح)(ne2/ne02);
    ثابت صحیح nr3 = (صحیح)(ne3/ne03);

    // TODO: support for transposed / permuted tensors
    GGML_ASSERT(nb0  == حجم(ggml_fp16_t));
    GGML_ASSERT(nb00 == حجم(ggml_fp16_t));

    // TODO: maybe this is not optimal?
    برائے                         (صحیح i3 = 0; i3 < nr3;  i3++) {
        برائے                     (صحیح k3 = 0; k3 < ne03; k3++) {
            برائے                 (صحیح i2 = 0; i2 < nr2;  i2++) {
                برائے             (صحیح k2 = 0; k2 < ne02; k2++) {
                    برائے         (صحیح i1 = 0; i1 < nr1;  i1++) {
                        برائے     (صحیح k1 = 0; k1 < ne01; k1++) {
                            برائے (صحیح i0 = 0; i0 < nr0;  i0++) {
                                ggml_fp16_t * y = (ggml_fp16_t *) ((حرف *)  dst->ڈیٹا + (i3*ne03 + k3)*nb3  + (i2*ne02 + k2)*nb2  + (i1*ne01 + k1)*nb1  + (i0*ne00)*nb0);
                                ggml_fp16_t * x = (ggml_fp16_t *) ((حرف *) src0->ڈیٹا + (          k3)*nb03 + (          k2)*nb02 + (          k1)*nb01);
                                // ggml_vec_cpy_f16(ne00, y, x)
                                برائے (صحیح i = 0; i < ne00; ++i) {
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

خالی ggml_compute_forward_repeat(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F16:
        حالت GGML_TYPE_BF16:
        حالت GGML_TYPE_I16:
            {
                ggml_compute_forward_repeat_f16(params, dst);
            } توڑو;
        حالت GGML_TYPE_F32:
        حالت GGML_TYPE_I32:
            {
                ggml_compute_forward_repeat_f32(params, dst);
            } توڑو;
        // TODO: templateify the implementation and support for I64
        //       ref https://github.com/ggml-org/llama.cpp/pull/14274#discussion_r2169492225
        //case GGML_TYPE_I64:
        //    {
        //        ggml_compute_forward_repeat_i64(params, dst);
        //    } break;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_repeat_back

ساکن خالی ggml_compute_forward_repeat_عقب_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    اگر (params->ith != 0) {
        واپس;
    }

    GGML_ASSERT(ggml_can_repeat(dst, src0));

    GGML_TENSOR_UNARY_OP_LOCALS

    // guaranteed to be an integer due to the check in ggml_can_repeat
    ثابت صحیح nr0 = (صحیح)(ne00/ne0);
    ثابت صحیح nr1 = (صحیح)(ne01/ne1);
    ثابت صحیح nr2 = (صحیح)(ne02/ne2);
    ثابت صحیح nr3 = (صحیح)(ne03/ne3);

    // TODO: support for transposed / permuted tensors
    GGML_ASSERT(nb0  == حجم(اعشاری));
    GGML_ASSERT(nb00 == حجم(اعشاری));

    اگر (ggml_is_contiguous(dst)) {
        ggml_vec_سیٹ_f32(ne0*ne1*ne2*ne3, (اعشاری *)dst->ڈیٹا, 0);
    } ورنہ {
        برائے         (صحیح k3 = 0; k3 < ne3; k3++) {
            برائے     (صحیح k2 = 0; k2 < ne2; k2++) {
                برائے (صحیح k1 = 0; k1 < ne1; k1++) {
                    ggml_vec_سیٹ_f32(ne0,
                        (اعشاری *) ((حرف *) dst->ڈیٹا + k1*nb1 + k2*nb2 + k3*nb3),
                        0);
                }
            }
        }
    }

    // TODO: maybe this is not optimal?
    برائے                         (صحیح i3 = 0; i3 < nr3; i3++) {
        برائے                     (صحیح k3 = 0; k3 < ne3; k3++) {
            برائے                 (صحیح i2 = 0; i2 < nr2; i2++) {
                برائے             (صحیح k2 = 0; k2 < ne2; k2++) {
                    برائے         (صحیح i1 = 0; i1 < nr1; i1++) {
                        برائے     (صحیح k1 = 0; k1 < ne1; k1++) {
                            برائے (صحیح i0 = 0; i0 < nr0; i0++) {
                                ggml_vec_acc_f32(ne0,
                                        (اعشاری *) ((حرف *)  dst->ڈیٹا + (         k3)*nb3  + (         k2)*nb2  + (         k1)*nb1),
                                        (اعشاری *) ((حرف *) src0->ڈیٹا + (i3*ne3 + k3)*nb03 + (i2*ne2 + k2)*nb02 + (i1*ne1 + k1)*nb01 + (i0*ne0)*nb00));
                            }
                        }
                    }
                }
            }
        }
    }
}

خالی ggml_compute_forward_repeat_عقب(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_repeat_عقب_f32(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_concat
ساکن خالی ggml_compute_forward_concat_any(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    GGML_TENSOR_ثنائی_OP_LOCALS

    ثابت int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);
    GGML_ASSERT(ggml_is_contiguous_صفیں(src0));
    GGML_ASSERT(ggml_is_contiguous_صفیں(src1));

    int64_t o[4] = {0, 0, 0, 0};

    اگر (dim == 0) {
        GGML_ASSERT(src0->ne[0] % ggml_blck_سائز(src0->type) == 0);
        GGML_ASSERT(src1->ne[0] % ggml_blck_سائز(src1->type) == 0);

        o[dim] = src0->ne[dim]/ggml_blck_سائز(src0->type);
    } ورنہ {
        o[dim] = src0->ne[dim];
    }

    // Region 1: copy rows from src0
    برائے (صحیح i3 = 0; i3 < ne03; i3++) {
        برائے (صحیح i2 = ith; i2 < ne02; i2 += nth) {
            برائے (صحیح i1 = 0; i1 < ne01; i1++) {
                ثابت حرف * x = (ثابت حرف *) src0->ڈیٹا + i1*nb01 + i2*nb02 + i3*nb03;
                      حرف * y = (      حرف *) dst->ڈیٹا  + i1*nb1  + i2*nb2  + i3*nb3;
                حافظہ_نقل(y, x, ggml_صف_سائز(src0->type, ne00));
            }
        }
    }

    // Region 2: copy rows from src1, offset into dst by o[]
    برائے (صحیح i3 = 0; i3 < ne13; i3++) {
        برائے (صحیح i2 = ith; i2 < ne12; i2 += nth) {
            برائے (صحیح i1 = 0; i1 < ne11; i1++) {
                ثابت حرف * x = (ثابت حرف *) src1->ڈیٹا + i1*nb11         + i2*nb12         + i3*nb13;
                      حرف * y = (      حرف *)  dst->ڈیٹا + (i1 + o[1])*nb1 + (i2 + o[2])*nb2 + (i3 + o[3])*nb3 + o[0]*nb0;
                حافظہ_نقل(y, x, ggml_صف_سائز(src1->type, ne10));
            }
        }
    }
}

ساکن خالی ggml_compute_forward_concat_i8(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_type_سائز(src0->type) == حجم(int8_t));

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    GGML_TENSOR_ثنائی_OP_LOCALS

    ثابت int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);

    int64_t o[4] = {0, 0, 0, 0};
    o[dim] = src0->ne[dim];

    ثابت int8_t * x;

    // TODO: smarter multi-theading
    برائے (صحیح i3 = 0; i3 < ne3; i3++) {
        برائے (صحیح i2 = ith; i2 < ne2; i2 += nth) {
            برائے (صحیح i1 = 0; i1 < ne1; i1++) {
                برائے (صحیح i0 = 0; i0 < ne0; i0++) {
                    اگر (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                        x = (ثابت int8_t *) ((ثابت حرف *)src0->ڈیٹا + (i0       )*nb00 + (i1       )*nb01 + (i2       )*nb02 + (i3       )*nb03);
                    } ورنہ {
                        x = (ثابت int8_t *) ((ثابت حرف *)src1->ڈیٹا + (i0 - o[0])*nb10 + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13);
                    }

                    int8_t * y = (int8_t *)((حرف *)dst->ڈیٹا + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);

                    *y = *x;
                }
            }
        }
    }
}

ساکن خالی ggml_compute_forward_concat_f16(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_type_سائز(src0->type) == حجم(ggml_fp16_t));

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    GGML_TENSOR_ثنائی_OP_LOCALS

    ثابت int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);

    int64_t o[4] = {0, 0, 0, 0};
    o[dim] = src0->ne[dim];

    ثابت ggml_fp16_t * x;

    // TODO: smarter multi-theading
    برائے (صحیح i3 = 0; i3 < ne3; i3++) {
        برائے (صحیح i2 = ith; i2 < ne2; i2 += nth) {
            برائے (صحیح i1 = 0; i1 < ne1; i1++) {
                برائے (صحیح i0 = 0; i0 < ne0; i0++) {
                    اگر (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                        x = (ثابت ggml_fp16_t *) ((ثابت حرف *)src0->ڈیٹا + (i0       )*nb00 + (i1       )*nb01 + (i2       )*nb02 + (i3       )*nb03);
                    } ورنہ {
                        x = (ثابت ggml_fp16_t *) ((ثابت حرف *)src1->ڈیٹا + (i0 - o[0])*nb10 + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13);
                    }

                    ggml_fp16_t * y = (ggml_fp16_t *)((حرف *)dst->ڈیٹا + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);

                    *y = *x;
                }
            }
        }
    }
}

ساکن خالی ggml_compute_forward_concat_f32(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_type_سائز(src0->type) == حجم(اعشاری));

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    GGML_TENSOR_ثنائی_OP_LOCALS

    ثابت int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);

    int64_t o[4] = {0, 0, 0, 0};
    o[dim] = src0->ne[dim];

    ثابت اعشاری * x;

    // TODO: smarter multi-theading
    برائے (صحیح i3 = 0; i3 < ne3; i3++) {
        برائے (صحیح i2 = ith; i2 < ne2; i2 += nth) {
            برائے (صحیح i1 = 0; i1 < ne1; i1++) {
                برائے (صحیح i0 = 0; i0 < ne0; i0++) {
                    اگر (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                        x = (ثابت اعشاری *) ((ثابت حرف *)src0->ڈیٹا + (i0       )*nb00 + (i1       )*nb01 + (i2       )*nb02 + (i3       )*nb03);
                    } ورنہ {
                        x = (ثابت اعشاری *) ((ثابت حرف *)src1->ڈیٹا + (i0 - o[0])*nb10 + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13);
                    }

                    اعشاری * y = (اعشاری *)((حرف *)dst->ڈیٹا + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);

                    *y = *x;
                }
            }
        }
    }
}

خالی ggml_compute_forward_concat(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F16:
        حالت GGML_TYPE_BF16:
        حالت GGML_TYPE_I16:
            {
                ggml_compute_forward_concat_f16(params, dst);
            } توڑو;
        حالت GGML_TYPE_I8:
            {
                ggml_compute_forward_concat_i8(params, dst);
            } توڑو;
        حالت GGML_TYPE_F32:
        حالت GGML_TYPE_I32:
            {
                ggml_compute_forward_concat_f32(params, dst);
            } توڑو;
        طےشدہ:
            {
                ggml_compute_forward_concat_any(params, dst);
            }
    }
}

// row loop shared by the BF16 unary ops
سانچہ <خالی (*vec_op)(ثابت صحیح, ggml_bf16_t *, ثابت ggml_bf16_t *)>
ساکن خالی ggml_compute_forward_unary_bf16_صفیں(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    أكد(ggml_is_contiguous_صفیں(src0));
    أكد(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح nc = src0->ne[0];
    ثابت صحیح nr = ggml_nrows(src0);

    // rows per thread
    ثابت صحیح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحیح ir0 = dr*ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    برائے (صحیح ir = ir0; ir < ir1; ++ir) {
        ثابت صحیح i3 = ir/(ne02*ne01);
        ثابت صحیح i2 = (ir - i3*ne02*ne01)/ne01;
        ثابت صحیح i1 = (ir - i3*ne02*ne01 - i2*ne01);

        vec_op(nc,
                (ggml_bf16_t *) ((حرف *) dst->ڈیٹا  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_bf16_t *) ((حرف *) src0->ڈیٹا + i3*nb03 + i2*nb02 + i1*nb01));
    }
}

// ggml_compute_forward_gelu

ساکن خالی ggml_compute_forward_gelu_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    أكد(ggml_is_contiguous_صفیں(src0));
    أكد(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح nc = src0->ne[0];
    ثابت صحیح nr = ggml_nrows(src0);

    // rows per thread
    ثابت صحیح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحیح ir0 = dr*ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    برائے (صحیح ir = ir0; ir < ir1; ++ir) {
        ثابت صحیح i3 = ir/(ne02*ne01);
        ثابت صحیح i2 = (ir - i3*ne02*ne01)/ne01;
        ثابت صحیح i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_f32(nc,
                (اعشاری *) ((حرف *) dst->ڈیٹا  + i3*nb3  + i2*nb2  + i1*nb1),
                (اعشاری *) ((حرف *) src0->ڈیٹا + i3*nb03 + i2*nb02 + i1*nb01));

#اگر_غیر_تعریف_شدہ NDEBUG
        برائے (صحیح k = 0; k < nc; k++) {
            ثابت اعشاری x = ((اعشاری *) ((حرف *) dst->ڈیٹا + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_غیر_مستعمل(x);
            أكد(!isnan(x));
            أكد(!isinf(x));
        }
#اگر_ختم // NDEBUG
    }
}

ساکن خالی ggml_compute_forward_gelu_f16(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    أكد(ggml_is_contiguous_صفیں(src0));
    أكد(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح nc = src0->ne[0];
    ثابت صحیح nr = ggml_nrows(src0);

    // rows per thread
    ثابت صحیح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحیح ir0 = dr*ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    برائے (صحیح ir = ir0; ir < ir1; ++ir) {
        ثابت صحیح i3 = ir/(ne02*ne01);
        ثابت صحیح i2 = (ir - i3*ne02*ne01)/ne01;
        ثابت صحیح i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_f16(nc,
                (ggml_fp16_t *) ((حرف *) dst->ڈیٹا  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((حرف *) src0->ڈیٹا + i3*nb03 + i2*nb02 + i1*nb01));

#اگر_غیر_تعریف_شدہ NDEBUG
        برائے (صحیح k = 0; k < nc; k++) {
            ثابت ggml_fp16_t x = ((ggml_fp16_t *) ((حرف *) dst->ڈیٹا + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            ثابت اعشاری v = GGML_CPU_FP16_مقصد_FP32(x);
            GGML_غیر_مستعمل(v);
            أكد(!isnan(v));
            أكد(!isinf(v));
        }
#اگر_ختم // NDEBUG
    }
}

ساکن خالی ggml_compute_forward_gelu_bf16(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_صفیں<ggml_vec_gelu_bf16>(params, dst);
}

ساکن خالی ggml_compute_forward_gelu(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_gelu_f32(params, dst);
            } توڑو;
        حالت GGML_TYPE_F16:
            {
                ggml_compute_forward_gelu_f16(params, dst);
            } توڑو;
        حالت GGML_TYPE_BF16:
            {
                ggml_compute_forward_gelu_bf16(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_fill

ساکن خالی ggml_compute_forward_fill_f32(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    ثابت اعشاری c = ggml_get_op_params_f32(dst, 0);

    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne);
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb);

    ثابت خودکار [ir0, ir1] = get_thread_range(params, dst);

    برائے (int64_t ir = ir0; ir < ir1; ++ir) {
        ثابت int64_t i03 = ir/(ne2*ne1);
        ثابت int64_t i02 = (ir - i03*ne2*ne1)/ne1;
        ثابت int64_t i01 = (ir - i03*ne2*ne1 - i02*ne1);

        اعشاری * dst_ptr  = (اعشاری *) ((حرف *) dst->ڈیٹا + i03*nb3 + i02*nb2 + i01*nb1);

        ggml_vec_سیٹ_f32(ne0, dst_ptr, c);
    }
}

ساکن خالی ggml_compute_forward_fill_f16(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    ثابت ggml_fp16_t c = GGML_CPU_FP32_مقصد_FP16(ggml_get_op_params_f32(dst, 0));

    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne);
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb);

    ثابت خودکار [ir0, ir1] = get_thread_range(params, dst);

    برائے (int64_t ir = ir0; ir < ir1; ++ir) {
        ثابت int64_t i03 = ir/(ne2*ne1);
        ثابت int64_t i02 = (ir - i03*ne2*ne1)/ne1;
        ثابت int64_t i01 = (ir - i03*ne2*ne1 - i02*ne1);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((حرف *) dst->ڈیٹا + i03*nb3 + i02*nb2 + i01*nb1);

        ggml_vec_سیٹ_f16(ne0, dst_ptr, c);
    }
}

خالی ggml_compute_forward_fill(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_fill_f32(params, dst);
            } توڑو;
        حالت GGML_TYPE_F16:
            {
                ggml_compute_forward_fill_f16(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("unsupported type for ggml_compute_forward_fill: %s", ggml_type_name(src0->type));
            }
    }
}

// ggml_compute_tri

ساکن خالی ggml_compute_forward_tri_f32(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];

    ثابت ggml_tri_type ttype = (ggml_tri_type) ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_TENSOR_UNARY_OP_LOCALS

    ثابت خودکار [ir0, ir1] = get_thread_range(params, src0);

    منطقی (*bipred)(صحیح, صحیح);

    انتخاب (ttype) {
        حالت GGML_TRI_TYPE_زیریں:      bipred = [](صحیح i, صحیح r) { واپس i <  r; }; توڑو;
        حالت GGML_TRI_TYPE_زیریں_DIAG: bipred = [](صحیح i, صحیح r) { واپس i <= r; }; توڑو;
        حالت GGML_TRI_TYPE_بالائی:      bipred = [](صحیح i, صحیح r) { واپس i >  r; }; توڑو;
        حالت GGML_TRI_TYPE_بالائی_DIAG: bipred = [](صحیح i, صحیح r) { واپس i >= r; }; توڑو;
        طےشدہ: GGML_ABORT("invalid tri type");
    }

    برائے (int64_t ir = ir0; ir < ir1; ++ir) {
        ثابت int64_t i03 = ir/(ne02*ne01);
        ثابت int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        ثابت int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        ثابت اعشاری * src_ptr = (ثابت اعشاری  *) ((ثابت حرف *) src0->ڈیٹا + i03*nb03 + i02*nb02 + i01*nb01);
              اعشاری * dst_ptr = (      اعشاری  *) ((      حرف *) dst->ڈیٹا  + i03*nb3  + i02*nb2  + i01*nb1);

        برائے (صحیح i0 = 0; i0 < ne0; ++i0) {
            dst_ptr[i0] = bipred(i0, i01) ? src_ptr[i0] : 0.0f;
        }
    }
}

خالی ggml_compute_forward_tri(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_tri_f32(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_gelu_erf

ساکن خالی ggml_compute_forward_gelu_erf_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    أكد(ggml_is_contiguous_صفیں(src0));
    أكد(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح nc = src0->ne[0];
    ثابت صحیح nr = ggml_nrows(src0);

    // rows per thread
    ثابت صحیح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحیح ir0 = dr*ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    برائے (صحیح ir = ir0; ir < ir1; ++ir) {
        ثابت صحیح i3 = ir/(ne02*ne01);
        ثابت صحیح i2 = (ir - i3*ne02*ne01)/ne01;
        ثابت صحیح i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_erf_f32(nc,
                (اعشاری *) ((حرف *) dst->ڈیٹا  + i3*nb3  + i2*nb2  + i1*nb1),
                (اعشاری *) ((حرف *) src0->ڈیٹا + i3*nb03 + i2*nb02 + i1*nb01));

#اگر_غیر_تعریف_شدہ NDEBUG
        برائے (صحیح k = 0; k < nc; k++) {
            ثابت اعشاری x = ((اعشاری *) ((حرف *) dst->ڈیٹا + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_غیر_مستعمل(x);
            أكد(!isnan(x));
            أكد(!isinf(x));
        }
#اگر_ختم // NDEBUG
    }
}

ساکن خالی ggml_compute_forward_gelu_erf_f16(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    أكد(ggml_is_contiguous_صفیں(src0));
    أكد(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح nc = src0->ne[0];
    ثابت صحیح nr = ggml_nrows(src0);

    // rows per thread
    ثابت صحیح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحیح ir0 = dr*ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    برائے (صحیح ir = ir0; ir < ir1; ++ir) {
        ثابت صحیح i3 = ir/(ne02*ne01);
        ثابت صحیح i2 = (ir - i3*ne02*ne01)/ne01;
        ثابت صحیح i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_erf_f16(nc,
                (ggml_fp16_t *) ((حرف *) dst->ڈیٹا  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((حرف *) src0->ڈیٹا + i3*nb03 + i2*nb02 + i1*nb01));

#اگر_غیر_تعریف_شدہ NDEBUG
        برائے (صحیح k = 0; k < nc; k++) {
            ثابت ggml_fp16_t x = ((ggml_fp16_t *) ((حرف *) dst->ڈیٹا + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            ثابت اعشاری v = GGML_CPU_FP16_مقصد_FP32(x);
            GGML_غیر_مستعمل(v);
            أكد(!isnan(v));
            أكد(!isinf(v));
        }
#اگر_ختم // NDEBUG
    }
}

ساکن خالی ggml_compute_forward_gelu_erf_bf16(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_صفیں<ggml_vec_gelu_erf_bf16>(params, dst);
}

ساکن خالی ggml_compute_forward_gelu_erf(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_gelu_erf_f32(params, dst);
            } توڑو;
        حالت GGML_TYPE_F16:
            {
                ggml_compute_forward_gelu_erf_f16(params, dst);
            } توڑو;
        حالت GGML_TYPE_BF16:
            {
                ggml_compute_forward_gelu_erf_bf16(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_gelu_quick

ساکن خالی ggml_compute_forward_gelu_تیز_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    أكد(ggml_is_contiguous_صفیں(src0));
    أكد(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح nc = src0->ne[0];
    ثابت صحیح nr = ggml_nrows(src0);

    // rows per thread
    ثابت صحیح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحیح ir0 = dr*ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    برائے (صحیح ir = ir0; ir < ir1; ++ir) {
        ثابت صحیح i3 = ir/(ne02*ne01);
        ثابت صحیح i2 = (ir - i3*ne02*ne01)/ne01;
        ثابت صحیح i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_تیز_f32(nc,
                (اعشاری *) ((حرف *) dst->ڈیٹا  + i3*nb3  + i2*nb2  + i1*nb1),
                (اعشاری *) ((حرف *) src0->ڈیٹا + i3*nb03 + i2*nb02 + i1*nb01));

#اگر_غیر_تعریف_شدہ NDEBUG
        برائے (صحیح k = 0; k < nc; k++) {
            ثابت اعشاری x = ((اعشاری *) ((حرف *) dst->ڈیٹا + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_غیر_مستعمل(x);
            أكد(!isnan(x));
            أكد(!isinf(x));
        }
#اگر_ختم // NDEBUG
    }
}

ساکن خالی ggml_compute_forward_gelu_تیز_f16(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    أكد(ggml_is_contiguous_صفیں(src0));
    أكد(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح nc = src0->ne[0];
    ثابت صحیح nr = ggml_nrows(src0);

    // rows per thread
    ثابت صحیح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحیح ir0 = dr*ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    برائے (صحیح ir = ir0; ir < ir1; ++ir) {
        ثابت صحیح i3 = ir/(ne02*ne01);
        ثابت صحیح i2 = (ir - i3*ne02*ne01)/ne01;
        ثابت صحیح i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_تیز_f16(nc,
                (ggml_fp16_t *) ((حرف *) dst->ڈیٹا  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((حرف *) src0->ڈیٹا + i3*nb03 + i2*nb02 + i1*nb01));

#اگر_غیر_تعریف_شدہ NDEBUG
        برائے (صحیح k = 0; k < nc; k++) {
            ثابت ggml_fp16_t x = ((ggml_fp16_t *) ((حرف *) dst->ڈیٹا + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            ثابت اعشاری v = GGML_CPU_FP16_مقصد_FP32(x);
            GGML_غیر_مستعمل(v);
            أكد(!isnan(v));
            أكد(!isinf(v));
        }
#اگر_ختم // NDEBUG
    }
}

ساکن خالی ggml_compute_forward_gelu_تیز_bf16(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_صفیں<ggml_vec_gelu_تیز_bf16>(params, dst);
}

ساکن خالی ggml_compute_forward_gelu_تیز(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_gelu_تیز_f32(params, dst);
            } توڑو;
        حالت GGML_TYPE_F16:
            {
                ggml_compute_forward_gelu_تیز_f16(params, dst);
            } توڑو;
        حالت GGML_TYPE_BF16:
            {
                ggml_compute_forward_gelu_تیز_bf16(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_silu

ساکن خالی ggml_compute_forward_silu_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    أكد(ggml_is_contiguous_صفیں(src0));
    أكد(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح nc = src0->ne[0];
    ثابت صحیح nr = ggml_nrows(src0);

    // rows per thread
    ثابت صحیح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحیح ir0 = dr*ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    برائے (صحیح ir = ir0; ir < ir1; ++ir) {
        ثابت صحیح i3 = ir/(ne02*ne01);
        ثابت صحیح i2 = (ir - i3*ne02*ne01)/ne01;
        ثابت صحیح i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_silu_f32(nc,
                (اعشاری *) ((حرف *) dst->ڈیٹا  + i3*nb3  + i2*nb2  + i1*nb1),
                (اعشاری *) ((حرف *) src0->ڈیٹا + i3*nb03 + i2*nb02 + i1*nb01));

#اگر_غیر_تعریف_شدہ NDEBUG
        برائے (صحیح k = 0; k < nc; k++) {
            ثابت اعشاری x = ((اعشاری *) ((حرف *) dst->ڈیٹا + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_غیر_مستعمل(x);
            أكد(!isnan(x));
            أكد(!isinf(x));
        }
#اگر_ختم // NDEBUG
    }
}

ساکن خالی ggml_compute_forward_silu_f16(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    أكد(ggml_is_contiguous_صفیں(src0));
    أكد(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح nc = src0->ne[0];
    ثابت صحیح nr = ggml_nrows(src0);

    // rows per thread
    ثابت صحیح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحیح ir0 = dr*ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    برائے (صحیح ir = ir0; ir < ir1; ++ir) {
        ثابت صحیح i3 = ir/(ne02*ne01);
        ثابت صحیح i2 = (ir - i3*ne02*ne01)/ne01;
        ثابت صحیح i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_silu_f16(nc,
                (ggml_fp16_t *) ((حرف *) dst->ڈیٹا  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((حرف *) src0->ڈیٹا + i3*nb03 + i2*nb02 + i1*nb01));

#اگر_غیر_تعریف_شدہ NDEBUG
        برائے (صحیح k = 0; k < nc; k++) {
            ثابت ggml_fp16_t x = ((ggml_fp16_t *) ((حرف *) dst->ڈیٹا + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            ثابت اعشاری v = GGML_CPU_FP16_مقصد_FP32(x);
            GGML_غیر_مستعمل(v);
            أكد(!isnan(v));
            أكد(!isinf(v));
        }
#اگر_ختم // NDEBUG
    }
}

ساکن خالی ggml_compute_forward_silu_bf16(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_صفیں<ggml_vec_silu_bf16>(params, dst);
}

ساکن خالی ggml_compute_forward_silu(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_silu_f32(params, dst);
            } توڑو;
        حالت GGML_TYPE_F16:
            {
                ggml_compute_forward_silu_f16(params, dst);
            } توڑو;
        حالت GGML_TYPE_BF16:
            {
                ggml_compute_forward_silu_bf16(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}
// ggml_compute_forward_leaky_relu

ساکن خالی ggml_compute_forward_leaky_relu_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    اگر (params->ith != 0) {
        واپس;
    }

    أكد(ggml_is_contiguous_1(src0));
    أكد(ggml_is_contiguous_1(dst));
    أكد(ggml_are_same_shape(src0, dst));

    ثابت صحیح n  = ggml_nrows(src0);
    ثابت صحیح nc = src0->ne[0];

    اعشاری negative_ڈھلان;
    حافظہ_نقل(&negative_ڈھلان, dst->op_params, حجم(اعشاری));

    أكد(dst->nb[0]  == حجم(اعشاری));
    أكد(src0->nb[0] == حجم(اعشاری));

    برائے (صحیح i = 0; i < n; i++) {
        ggml_vec_leaky_relu_f32(nc,
                (اعشاری *) ((حرف *) dst->ڈیٹا  + i*( dst->nb[1])),
                (اعشاری *) ((حرف *) src0->ڈیٹا + i*(src0->nb[1])), negative_ڈھلان);
    }
}

ساکن خالی ggml_compute_forward_leaky_relu_f16(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    اگر (params->ith != 0) {
        واپس;
    }

    أكد(ggml_is_contiguous_1(src0));
    أكد(ggml_is_contiguous_1(dst));
    أكد(ggml_are_same_shape(src0, dst));

    ثابت صحیح n  = ggml_nrows(src0);
    ثابت صحیح nc = src0->ne[0];

    اعشاری negative_ڈھلان;
    حافظہ_نقل(&negative_ڈھلان, dst->op_params, حجم(اعشاری));

    أكد(dst->nb[0]  == حجم(ggml_fp16_t));
    أكد(src0->nb[0] == حجم(ggml_fp16_t));

    برائے (صحیح i = 0; i < n; i++) {
        ggml_vec_leaky_relu_f16(nc,
                (ggml_fp16_t *) ((حرف *) dst->ڈیٹا  + i*( dst->nb[1])),
                (ggml_fp16_t *) ((حرف *) src0->ڈیٹا + i*(src0->nb[1])), negative_ڈھلان);
    }
}

خالی ggml_compute_forward_leaky_relu(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_leaky_relu_f32(params, dst);
            } توڑو;
        حالت GGML_TYPE_F16:
            {
                ggml_compute_forward_leaky_relu_f16(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_silu_back

ساکن خالی ggml_compute_forward_silu_عقب_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * grad = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    أكد(ggml_is_contiguous_1(grad));
    أكد(ggml_is_contiguous_1(src1));
    أكد(ggml_is_contiguous_1(dst));
    أكد(ggml_are_same_shape(src1, dst));
    أكد(ggml_are_same_shape(src1, grad));

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح nc = src1->ne[0];
    ثابت صحیح nr = ggml_nrows(src1);

    // rows per thread
    ثابت صحیح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحیح ir0 = dr*ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    برائے (صحیح i1 = ir0; i1 < ir1; i1++) {
        ggml_vec_silu_backward_f32(nc,
                (اعشاری *) ((حرف *) dst->ڈیٹا  + i1*( dst->nb[1])),
                (اعشاری *) ((حرف *) src1->ڈیٹا + i1*(src1->nb[1])),
                (اعشاری *) ((حرف *) grad->ڈیٹا + i1*(grad->nb[1])));

#اگر_غیر_تعریف_شدہ NDEBUG
        برائے (صحیح k = 0; k < nc; k++) {
            ثابت اعشاری x = ((اعشاری *) ((حرف *) dst->ڈیٹا + i1*( dst->nb[1])))[k];
            GGML_غیر_مستعمل(x);
            أكد(!isnan(x));
            أكد(!isinf(x));
        }
#اگر_ختم // NDEBUG
    }
}

ساکن خالی ggml_compute_forward_silu_عقب_f16(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * grad = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    أكد(ggml_is_contiguous_1(grad));
    أكد(ggml_is_contiguous_1(src1));
    أكد(ggml_is_contiguous_1(dst));
    أكد(ggml_are_same_shape(src1, dst));
    أكد(ggml_are_same_shape(src1, grad));

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح nc = src1->ne[0];
    ثابت صحیح nr = ggml_nrows(src1);

    // rows per thread
    ثابت صحیح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحیح ir0 = dr*ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    برائے (صحیح i1 = ir0; i1 < ir1; i1++) {
        ggml_vec_silu_backward_f16(nc,
                (ggml_fp16_t *) ((حرف *) dst->ڈیٹا  + i1*( dst->nb[1])),
                (ggml_fp16_t *) ((حرف *) src1->ڈیٹا + i1*(src1->nb[1])),
                (ggml_fp16_t *) ((حرف *) grad->ڈیٹا + i1*(grad->nb[1])));

#اگر_غیر_تعریف_شدہ NDEBUG
        برائے (صحیح k = 0; k < nc; k++) {
            ثابت اعشاری x = ((ggml_fp16_t *) ((حرف *) dst->ڈیٹا + i1*( dst->nb[1])))[k];
            ثابت اعشاری v = GGML_CPU_FP16_مقصد_FP32(x);
            GGML_غیر_مستعمل(v);
            أكد(!isnan(v));
            أكد(!isinf(v));
        }
#اگر_ختم // NDEBUG
    }
}

خالی ggml_compute_forward_silu_عقب(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_silu_عقب_f32(params, dst);
            } توڑو;
        حالت GGML_TYPE_F16:
            {
                ggml_compute_forward_silu_عقب_f16(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_reglu

// row loop shared by the BF16 GLU ops
سانچہ <خالی (*vec_op)(ثابت صحیح, ggml_bf16_t *, ثابت ggml_bf16_t *, ثابت ggml_bf16_t *)>
ساکن خالی ggml_compute_forward_glu_bf16_صفیں(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    حرف * src0_d = (حرف *) src0->ڈیٹا;
    حرف * src1_d = (حرف *) (src1 ? src1->ڈیٹا : src0->ڈیٹا);
    ثابت size_t src0_o = src0->nb[1];
    ثابت size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    اگر (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ثابت صحیح nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ثابت int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ثابت صحیح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحیح ir0 = dr*ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    برائے (صحیح i1 = ir0; i1 < ir1; i1++) {
        ثابت ggml_bf16_t * src0_p = (ثابت ggml_bf16_t *) (src0_d + i1*src0_o);
        ثابت ggml_bf16_t * src1_p = (ثابت ggml_bf16_t *) (src1_d + i1*src1_o);
        ggml_bf16_t       * dst_p  = (ggml_bf16_t *) ((حرف *) dst->ڈیٹا + i1*(dst->nb[1]));

        اگر (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        vec_op(nc, dst_p, src0_p, src1_p);
    }
}

ساکن خالی ggml_compute_forward_reglu_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    حرف * src0_d = (حرف *) src0->ڈیٹا;
    حرف * src1_d = (حرف *) (src1 ? src1->ڈیٹا : src0->ڈیٹا);
    ثابت size_t src0_o = src0->nb[1];
    ثابت size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    اگر (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ثابت صحیح nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ثابت int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ثابت صحیح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحیح ir0 = dr*ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    برائے (صحیح i1 = ir0; i1 < ir1; i1++) {
        اعشاری * src0_p = (اعشاری *) (src0_d + i1*src0_o);
        اعشاری * src1_p = (اعشاری *) (src1_d + i1*src1_o);

        اگر (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_reglu_f32(nc, (اعشاری *) ((حرف *) dst->ڈیٹا + i1*(dst->nb[1])), src0_p, src1_p);

#اگر_غیر_تعریف_شدہ NDEBUG
        برائے (صحیح k = 0; k < nc; k++) {
            ثابت اعشاری x = ((اعشاری *) ((حرف *) dst->ڈیٹا + i1*( dst->nb[1])))[k];
            GGML_غیر_مستعمل(x);
            أكد(!isnan(x));
            أكد(!isinf(x));
        }
#اگر_ختم // NDEBUG
    }
}

ساکن خالی ggml_compute_forward_reglu_f16(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    حرف * src0_d = (حرف *) src0->ڈیٹا;
    حرف * src1_d = (حرف *) (src1 ? src1->ڈیٹا : src0->ڈیٹا);
    ثابت size_t src0_o = src0->nb[1];
    ثابت size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    اگر (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ثابت صحیح nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ثابت int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ثابت صحیح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحیح ir0 = dr*ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    برائے (صحیح i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        اگر (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_reglu_f16(nc, (ggml_fp16_t *) ((حرف *) dst->ڈیٹا + i1*(dst->nb[1])), src0_p, src1_p);

#اگر_غیر_تعریف_شدہ NDEBUG
        برائے (صحیح k = 0; k < nc; k++) {
            ثابت ggml_fp16_t x = ((ggml_fp16_t *) ((حرف *) dst->ڈیٹا + i1*( dst->nb[1])))[k];
            ثابت اعشاری v = GGML_FP16_مقصد_FP32(x);
            GGML_غیر_مستعمل(v);
            أكد(!isnan(v));
            أكد(!isinf(v));
        }
#اگر_ختم // NDEBUG
    }
}

ساکن خالی ggml_compute_forward_reglu_bf16(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_صفیں<ggml_vec_reglu_bf16>(params, dst);
}

ساکن خالی ggml_compute_forward_reglu(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_reglu_f32(params, dst);
            } توڑو;
        حالت GGML_TYPE_F16:
            {
                ggml_compute_forward_reglu_f16(params, dst);
            } توڑو;
        حالت GGML_TYPE_BF16:
            {
                ggml_compute_forward_reglu_bf16(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_geglu

ساکن خالی ggml_compute_forward_geglu_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    حرف * src0_d = (حرف *) src0->ڈیٹا;
    حرف * src1_d = (حرف *) (src1 ? src1->ڈیٹا : src0->ڈیٹا);
    ثابت size_t src0_o = src0->nb[1];
    ثابت size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    اگر (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ثابت صحیح nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ثابت int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ثابت صحیح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحیح ir0 = dr*ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    برائے (صحیح i1 = ir0; i1 < ir1; i1++) {
        اعشاری * src0_p = (اعشاری *) (src0_d + i1*src0_o);
        اعشاری * src1_p = (اعشاری *) (src1_d + i1*src1_o);

        اگر (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_f32(nc, (اعشاری *) ((حرف *) dst->ڈیٹا + i1*(dst->nb[1])), src0_p, src1_p);

#اگر_غیر_تعریف_شدہ NDEBUG
        برائے (صحیح k = 0; k < nc; k++) {
            ثابت اعشاری x = ((اعشاری *) ((حرف *) dst->ڈیٹا + i1*( dst->nb[1])))[k];
            GGML_غیر_مستعمل(x);
            أكد(!isnan(x));
            أكد(!isinf(x));
        }
#اگر_ختم // NDEBUG
    }
}

ساکن خالی ggml_compute_forward_geglu_f16(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    حرف * src0_d = (حرف *) src0->ڈیٹا;
    حرف * src1_d = (حرف *) (src1 ? src1->ڈیٹا : src0->ڈیٹا);
    ثابت size_t src0_o = src0->nb[1];
    ثابت size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    اگر (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ثابت صحیح nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ثابت int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ثابت صحیح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحیح ir0 = dr*ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    برائے (صحیح i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        اگر (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_f16(nc, (ggml_fp16_t *) ((حرف *) dst->ڈیٹا + i1*(dst->nb[1])), src0_p, src1_p);

#اگر_غیر_تعریف_شدہ NDEBUG
        برائے (صحیح k = 0; k < nc; k++) {
            ثابت ggml_fp16_t x = ((ggml_fp16_t *) ((حرف *) dst->ڈیٹا + i1*( dst->nb[1])))[k];
            ثابت اعشاری v = GGML_FP16_مقصد_FP32(x);
            GGML_غیر_مستعمل(v);
            أكد(!isnan(v));
            أكد(!isinf(v));
        }
#اگر_ختم // NDEBUG
    }
}

ساکن خالی ggml_compute_forward_geglu_bf16(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_صفیں<ggml_vec_geglu_bf16>(params, dst);
}

ساکن خالی ggml_compute_forward_geglu(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_geglu_f32(params, dst);
            } توڑو;
        حالت GGML_TYPE_F16:
            {
                ggml_compute_forward_geglu_f16(params, dst);
            } توڑو;
        حالت GGML_TYPE_BF16:
            {
                ggml_compute_forward_geglu_bf16(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_swiglu

ساکن خالی ggml_compute_forward_swiglu_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    حرف * src0_d = (حرف *) src0->ڈیٹا;
    حرف * src1_d = (حرف *) (src1 ? src1->ڈیٹا : src0->ڈیٹا);
    ثابت size_t src0_o = src0->nb[1];
    ثابت size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    اگر (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ثابت صحیح nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ثابت int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ثابت صحیح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحیح ir0 = dr*ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    برائے (صحیح i1 = ir0; i1 < ir1; i1++) {
        اعشاری * src0_p = (اعشاری *) (src0_d + i1*src0_o);
        اعشاری * src1_p = (اعشاری *) (src1_d + i1*src1_o);

        اگر (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_swiglu_f32(nc, (اعشاری *) ((حرف *) dst->ڈیٹا + i1*(dst->nb[1])), src0_p, src1_p);

#اگر_غیر_تعریف_شدہ NDEBUG
        برائے (صحیح k = 0; k < nc; k++) {
            ثابت اعشاری x = ((اعشاری *) ((حرف *) dst->ڈیٹا + i1*( dst->nb[1])))[k];
            GGML_غیر_مستعمل(x);
            أكد(!isnan(x));
            أكد(!isinf(x));
        }
#اگر_ختم // NDEBUG
    }
}

ساکن خالی ggml_compute_forward_swiglu_f16(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    حرف * src0_d = (حرف *) src0->ڈیٹا;
    حرف * src1_d = (حرف *) (src1 ? src1->ڈیٹا : src0->ڈیٹا);
    ثابت size_t src0_o = src0->nb[1];
    ثابت size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    اگر (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ثابت صحیح nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ثابت int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ثابت صحیح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحیح ir0 = dr*ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    برائے (صحیح i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        اگر (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_swiglu_f16(nc, (ggml_fp16_t *) ((حرف *) dst->ڈیٹا + i1*(dst->nb[1])), src0_p, src1_p);

#اگر_غیر_تعریف_شدہ NDEBUG
        برائے (صحیح k = 0; k < nc; k++) {
            ثابت ggml_fp16_t x = ((ggml_fp16_t *) ((حرف *) dst->ڈیٹا + i1*( dst->nb[1])))[k];
            ثابت اعشاری v = GGML_FP16_مقصد_FP32(x);
            GGML_غیر_مستعمل(v);
            أكد(!isnan(v));
            أكد(!isinf(v));
        }
#اگر_ختم // NDEBUG
    }
}

ساکن خالی ggml_compute_forward_swiglu_bf16(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_صفیں<ggml_vec_swiglu_bf16>(params, dst);
}

ساکن خالی ggml_compute_forward_swiglu(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_swiglu_f32(params, dst);
            } توڑو;
        حالت GGML_TYPE_F16:
            {
                ggml_compute_forward_swiglu_f16(params, dst);
            } توڑو;
        حالت GGML_TYPE_BF16:
            {
                ggml_compute_forward_swiglu_bf16(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_swiglu_oai

ساکن خالی ggml_compute_forward_swiglu_oai_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    حرف * src0_d = (حرف *) src0->ڈیٹا;
    حرف * src1_d = (حرف *) (src1 ? src1->ڈیٹا : src0->ڈیٹا);
    ثابت size_t src0_o = src0->nb[1];
    ثابت size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    اگر (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ثابت صحیح nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ثابت int32_t swapped = ggml_get_op_params_i32(dst, 1);
    ثابت اعشاری alpha = ggml_get_op_params_f32(dst, 2);
    ثابت اعشاری حد_بندی = ggml_get_op_params_f32(dst, 3);

    // rows per thread
    ثابت صحیح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحیح ir0 = dr*ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    برائے (صحیح i1 = ir0; i1 < ir1; i1++) {
        اعشاری * src0_p = (اعشاری *) (src0_d + i1*src0_o);
        اعشاری * src1_p = (اعشاری *) (src1_d + i1*src1_o);
        اعشاری * dst_p  = (اعشاری *) ((حرف *) dst->ڈیٹا + i1*(dst->nb[1]));

        اگر (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        برائے (صحیح k = 0; k < nc; k++) {
            ثابت اعشاری x = std::min(src0_p[k], حد_بندی);
            ثابت اعشاری y = std::clamp(src1_p[k], -حد_بندی, حد_بندی);
            ثابت اعشاری برآمد_glu = x / (1.f + expf(alpha * (-x)));
            dst_p[k] = برآمد_glu * (y + 1.f);
        }

#اگر_غیر_تعریف_شدہ NDEBUG
        برائے (صحیح k = 0; k < nc; k++) {
            ثابت اعشاری x = dst_p[k];
            GGML_غیر_مستعمل(x);
            أكد(!isnan(x));
            أكد(!isinf(x));
        }
#اگر_ختم // NDEBUG
    }
}

ساکن خالی ggml_compute_forward_swiglu_oai_f16(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    حرف * src0_d = (حرف *) src0->ڈیٹا;
    حرف * src1_d = (حرف *) (src1 ? src1->ڈیٹا : src0->ڈیٹا);
    ثابت size_t src0_o = src0->nb[1];
    ثابت size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    اگر (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ثابت صحیح nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ثابت int32_t swapped = ggml_get_op_params_i32(dst, 1);
    ثابت اعشاری alpha = ggml_get_op_params_f32(dst, 2);
    ثابت اعشاری حد_بندی = ggml_get_op_params_f32(dst, 3);

    // rows per thread
    ثابت صحیح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحیح ir0 = dr*ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    برائے (صحیح i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);
        ggml_fp16_t * dst_p  = (ggml_fp16_t *) ((حرف *) dst->ڈیٹا + i1*(dst->nb[1]));

        اگر (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        برائے (صحیح k = 0; k < nc; k++) {
            ثابت اعشاری x = std::min(GGML_CPU_FP16_مقصد_FP32(src0_p[k]), حد_بندی);
            ثابت اعشاری y = std::clamp(GGML_CPU_FP16_مقصد_FP32(src1_p[k]), -حد_بندی, حد_بندی);
            ثابت اعشاری برآمد_glu = x / (1.f + expf(alpha * (-x)));
            dst_p[k] = GGML_CPU_FP32_مقصد_FP16(برآمد_glu * (y + 1.f));
        }

#اگر_غیر_تعریف_شدہ NDEBUG
        برائے (صحیح k = 0; k < nc; k++) {
            ثابت اعشاری x = GGML_CPU_FP16_مقصد_FP32(dst_p[k]);
            GGML_غیر_مستعمل(x);
            أكد(!isnan(x));
            أكد(!isinf(x));
        }
#اگر_ختم // NDEBUG
    }
}

ساکن خالی ggml_compute_forward_swiglu_oai(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_swiglu_oai_f32(params, dst);
            } توڑو;
        حالت GGML_TYPE_F16:
            {
                ggml_compute_forward_swiglu_oai_f16(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_swiglu_clamp

ساکن خالی ggml_compute_forward_swiglu_clamp_f32(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    ثابت ggml_tensor * src0   = dst->src[0];
    ثابت ggml_tensor * src1   = dst->src[1];
    حرف *              src0_d = (حرف *) src0->ڈیٹا;
    حرف *              src1_d = (حرف *) (src1 ? src1->ڈیٹا : src0->ڈیٹا);
    ثابت size_t        src0_o = src0->nb[1];
    ثابت size_t        src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    اگر (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ثابت صحیح nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ثابت int32_t swapped = ggml_get_op_params_i32(dst, 1);
    ثابت اعشاری   حد_بندی   = ggml_get_op_params_f32(dst, 3);

    ثابت صحیح dr  = (nr + nth - 1) / nth;
    ثابت صحیح ir0 = dr * ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    برائے (صحیح i1 = ir0; i1 < ir1; i1++) {
        اعشاری * src0_p = (اعشاری *) (src0_d + i1 * src0_o);
        اعشاری * src1_p = (اعشاری *) (src1_d + i1 * src1_o);
        اعشاری * dst_p  = (اعشاری *) ((حرف *) dst->ڈیٹا + i1 * (dst->nb[1]));

        اگر (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        برائے (صحیح k = 0; k < nc; k++) {
            ثابت اعشاری gate = std::min(src0_p[k], حد_بندی);
            ثابت اعشاری اوپر   = std::clamp(src1_p[k], -حد_بندی, حد_بندی);
            dst_p[k]         = gate / (1.f + expf(-gate)) * اوپر;
        }

#اگر_غیر_تعریف_شدہ NDEBUG
        برائے (صحیح k = 0; k < nc; k++) {
            ثابت اعشاری x = dst_p[k];
            GGML_غیر_مستعمل(x);
            أكد(!isnan(x));
            أكد(!isinf(x));
        }
#اگر_ختم  // NDEBUG
    }
}

ساکن خالی ggml_compute_forward_swiglu_clamp_f16(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    ثابت ggml_tensor * src0   = dst->src[0];
    ثابت ggml_tensor * src1   = dst->src[1];
    حرف *              src0_d = (حرف *) src0->ڈیٹا;
    حرف *              src1_d = (حرف *) (src1 ? src1->ڈیٹا : src0->ڈیٹا);
    ثابت size_t        src0_o = src0->nb[1];
    ثابت size_t        src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    اگر (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ثابت صحیح nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ثابت int32_t swapped = ggml_get_op_params_i32(dst, 1);
    ثابت اعشاری   حد_بندی   = ggml_get_op_params_f32(dst, 3);

    ثابت صحیح dr  = (nr + nth - 1) / nth;
    ثابت صحیح ir0 = dr * ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    برائے (صحیح i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1 * src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1 * src1_o);
        ggml_fp16_t * dst_p  = (ggml_fp16_t *) ((حرف *) dst->ڈیٹا + i1 * (dst->nb[1]));

        اگر (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        برائے (صحیح k = 0; k < nc; k++) {
            ثابت اعشاری gate = std::min(GGML_FP16_مقصد_FP32(src0_p[k]), حد_بندی);
            ثابت اعشاری اوپر   = std::clamp(GGML_FP16_مقصد_FP32(src1_p[k]), -حد_بندی, حد_بندی);
            dst_p[k]         = GGML_FP32_مقصد_FP16(gate / (1.f + expf(-gate)) * اوپر);
        }

#اگر_غیر_تعریف_شدہ NDEBUG
        برائے (صحیح k = 0; k < nc; k++) {
            ثابت اعشاری x = GGML_FP16_مقصد_FP32(dst_p[k]);
            GGML_غیر_مستعمل(x);
            أكد(!isnan(x));
            أكد(!isinf(x));
        }
#اگر_ختم  // NDEBUG
    }
}

ساکن خالی ggml_compute_forward_swiglu_clamp(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    انتخاب (dst->src[0]->type) {
        حالت GGML_TYPE_F32:
            ggml_compute_forward_swiglu_clamp_f32(params, dst);
            توڑو;
        حالت GGML_TYPE_F16:
            ggml_compute_forward_swiglu_clamp_f16(params, dst);
            توڑو;
        طےشدہ:
            GGML_ABORT("fatal error");
    }
}

// ggml_compute_forward_geglu_erf

ساکن خالی ggml_compute_forward_geglu_erf_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    حرف * src0_d = (حرف *) src0->ڈیٹا;
    حرف * src1_d = (حرف *) (src1 ? src1->ڈیٹا : src0->ڈیٹا);
    ثابت size_t src0_o = src0->nb[1];
    ثابت size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    اگر (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ثابت صحیح nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ثابت int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ثابت صحیح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحیح ir0 = dr*ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    برائے (صحیح i1 = ir0; i1 < ir1; i1++) {
        اعشاری * src0_p = (اعشاری *) (src0_d + i1*src0_o);
        اعشاری * src1_p = (اعشاری *) (src1_d + i1*src1_o);

        اگر (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_erf_f32(nc, (اعشاری *) ((حرف *) dst->ڈیٹا + i1*(dst->nb[1])), src0_p, src1_p);

#اگر_غیر_تعریف_شدہ NDEBUG
        برائے (صحیح k = 0; k < nc; k++) {
            ثابت اعشاری x = ((اعشاری *) ((حرف *) dst->ڈیٹا + i1*( dst->nb[1])))[k];
            GGML_غیر_مستعمل(x);
            أكد(!isnan(x));
            أكد(!isinf(x));
        }
#اگر_ختم // NDEBUG
    }
}

ساکن خالی ggml_compute_forward_geglu_erf_f16(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    حرف * src0_d = (حرف *) src0->ڈیٹا;
    حرف * src1_d = (حرف *) (src1 ? src1->ڈیٹا : src0->ڈیٹا);
    ثابت size_t src0_o = src0->nb[1];
    ثابت size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    اگر (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ثابت صحیح nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ثابت int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ثابت صحیح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحیح ir0 = dr*ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    برائے (صحیح i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        اگر (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_erf_f16(nc, (ggml_fp16_t *) ((حرف *) dst->ڈیٹا + i1*(dst->nb[1])), src0_p, src1_p);

#اگر_غیر_تعریف_شدہ NDEBUG
        برائے (صحیح k = 0; k < nc; k++) {
            ثابت ggml_fp16_t x = ((ggml_fp16_t *) ((حرف *) dst->ڈیٹا + i1*( dst->nb[1])))[k];
            ثابت اعشاری v = GGML_FP16_مقصد_FP32(x);
            GGML_غیر_مستعمل(v);
            أكد(!isnan(v));
            أكد(!isinf(v));
        }
#اگر_ختم // NDEBUG
    }
}

ساکن خالی ggml_compute_forward_geglu_erf_bf16(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_صفیں<ggml_vec_geglu_erf_bf16>(params, dst);
}

ساکن خالی ggml_compute_forward_geglu_erf(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_geglu_erf_f32(params, dst);
            } توڑو;
        حالت GGML_TYPE_F16:
            {
                ggml_compute_forward_geglu_erf_f16(params, dst);
            } توڑو;
        حالت GGML_TYPE_BF16:
            {
                ggml_compute_forward_geglu_erf_bf16(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_geglu_quick

ساکن خالی ggml_compute_forward_geglu_تیز_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    حرف * src0_d = (حرف *) src0->ڈیٹا;
    حرف * src1_d = (حرف *) (src1 ? src1->ڈیٹا : src0->ڈیٹا);
    ثابت size_t src0_o = src0->nb[1];
    ثابت size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    اگر (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ثابت صحیح nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ثابت int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ثابت صحیح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحیح ir0 = dr*ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    برائے (صحیح i1 = ir0; i1 < ir1; i1++) {
        اعشاری * src0_p = (اعشاری *) (src0_d + i1*src0_o);
        اعشاری * src1_p = (اعشاری *) (src1_d + i1*src1_o);

        اگر (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_تیز_f32(nc, (اعشاری *) ((حرف *) dst->ڈیٹا + i1*(dst->nb[1])), src0_p, src1_p);

#اگر_غیر_تعریف_شدہ NDEBUG
        برائے (صحیح k = 0; k < nc; k++) {
            ثابت اعشاری x = ((اعشاری *) ((حرف *) dst->ڈیٹا + i1*( dst->nb[1])))[k];
            GGML_غیر_مستعمل(x);
            أكد(!isnan(x));
            أكد(!isinf(x));
        }
#اگر_ختم // NDEBUG
    }
}

ساکن خالی ggml_compute_forward_geglu_تیز_f16(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    حرف * src0_d = (حرف *) src0->ڈیٹا;
    حرف * src1_d = (حرف *) (src1 ? src1->ڈیٹا : src0->ڈیٹا);
    ثابت size_t src0_o = src0->nb[1];
    ثابت size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    اگر (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ثابت صحیح nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ثابت int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ثابت صحیح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحیح ir0 = dr*ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    برائے (صحیح i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        اگر (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_تیز_f16(nc, (ggml_fp16_t *) ((حرف *) dst->ڈیٹا + i1*(dst->nb[1])), src0_p, src1_p);

#اگر_غیر_تعریف_شدہ NDEBUG
        برائے (صحیح k = 0; k < nc; k++) {
            ثابت ggml_fp16_t x = ((ggml_fp16_t *) ((حرف *) dst->ڈیٹا + i1*( dst->nb[1])))[k];
            ثابت اعشاری v = GGML_FP16_مقصد_FP32(x);
            GGML_غیر_مستعمل(v);
            أكد(!isnan(v));
            أكد(!isinf(v));
        }
#اگر_ختم // NDEBUG
    }
}

ساکن خالی ggml_compute_forward_geglu_تیز_bf16(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_صفیں<ggml_vec_geglu_تیز_bf16>(params, dst);
}

ساکن خالی ggml_compute_forward_geglu_تیز(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_geglu_تیز_f32(params, dst);
            } توڑو;
        حالت GGML_TYPE_F16:
            {
                ggml_compute_forward_geglu_تیز_f16(params, dst);
            } توڑو;
        حالت GGML_TYPE_BF16:
            {
                ggml_compute_forward_geglu_تیز_bf16(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_norm

ساکن خالی ggml_compute_forward_norm_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    اعشاری eps;
    حافظہ_نقل(&eps, dst->op_params, حجم(اعشاری));

    GGML_ASSERT(eps >= 0.0f);

    برائے (int64_t i03 = 0; i03 < ne03; i03++) {
        برائے (int64_t i02 = 0; i02 < ne02; i02++) {
            برائے (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                ثابت حرف * x = (ثابت حرف *) src0->ڈیٹا + i01*nb01 + i02*nb02 + i03*nb03;
                حرف * y = (حرف *) dst->ڈیٹا + i01*nb1 + i02*nb2 + i03*nb3;

                اگر (nb00 == حجم(اعشاری) && nb0 == حجم(اعشاری)) {
                    ثابت اعشاری * xf = (ثابت اعشاری *) x;

                    اعشاری مجموع = 0.0;
                    ggml_vec_مجموع_f32(ne00, &مجموع, xf);
                    اعشاری اوسط_2 = مجموع/ne00;

                    اعشاری * yf = (اعشاری *) y;
                    اعشاری variance = 0;

#اگر_تعریف_شدہ GGML_USE_ACCELERATE
                    اوسط_2 = -اوسط_2;
                    vDSP_vsadd(xf, 1, &اوسط_2, yf, 1, ne00);
                    vDSP_measqv(yf, 1, &variance, ne00);
#ورنہ
                    variance = ggml_vec_cvar_f32(ne00, yf, xf, اوسط_2);
#اگر_ختم //GGML_USE_ACCELERATE

                    ثابت اعشاری scale = 1.0f/sqrtf(variance + eps);
                    ggml_vec_scale_f32(ne00, yf, scale);
                } ورنہ {
                    اعشاری مجموع = 0.0;
                    برائے (int64_t i00 = 0; i00 < ne00; i00++) {
                        مجموع += *(ثابت اعشاری *) (x + i00*nb00);
                    }
                    ثابت اعشاری اوسط_2 = مجموع/ne00;

                    اعشاری variance = 0.0f;
                    برائے (int64_t i00 = 0; i00 < ne00; i00++) {
                        ثابت اعشاری v = *(ثابت اعشاری *) (x + i00*nb00) - اوسط_2;
                        *(اعشاری *) (y + i00*nb0) = v;
                        variance += v * v;
                    }
                    variance /= ne00;

                    ثابت اعشاری scale = 1.0f/sqrtf(variance + eps);
                    برائے (int64_t i00 = 0; i00 < ne00; i00++) {
                        *(اعشاری *) (y + i00*nb0) *= scale;
                    }
                }
            }
        }
    }
}

خالی ggml_compute_forward_norm(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_norm_f32(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_group_rms_norm

// fusion kinds that can be combined with the rms_norm computation in a single pass.
// extend this enum when adding new fused variants (e.g. FUSE_ADD, FUSE_MUL_ADD, ...).
شمار ggml_rms_norm_fuse_op {
    GGML_RMS_NORM_FUSE_OP_NONE,
    GGML_RMS_NORM_FUSE_OP_MUL,
};

سانچہ <ggml_rms_norm_fuse_op FUSE_OP>
ساکن خالی ggml_compute_forward_rms_norm_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst_rms_norm,
        ggml_tensor * dst_fused = خالی_اشاریہ) {

    ثابت ggml_tensor * src0 = dst_rms_norm->src[0];
    ثابت ggml_tensor * src1 = خالی_اشاریہ;
    ggml_tensor       * dst  = dst_rms_norm;

    اگر constexpr (FUSE_OP == GGML_RMS_NORM_FUSE_OP_MUL) {
        src1 = (dst_fused->src[0] == dst_rms_norm) ? dst_fused->src[1] : dst_fused->src[0];
        dst  = dst_fused;
    }

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    GGML_ASSERT(src0->nb[0] == حجم(اعشاری));

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    GGML_TENSOR_ثنائی_OP_LOCALS

    اعشاری eps;
    حافظہ_نقل(&eps, dst_rms_norm->op_params, حجم(اعشاری));
    GGML_ASSERT(eps >= 0.0f);

    // TODO: optimize
    برائے (int64_t i03 = 0; i03 < ne03; i03++) {
        برائے (int64_t i02 = 0; i02 < ne02; i02++) {
            برائے (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                ثابت اعشاری * x = (اعشاری *) ((حرف *) src0->ڈیٹا + i01*nb01 + i02*nb02 + i03*nb03);

                ggml_float مجموع = 0.0;
                // worth switching to explicit SIMD?
                برائے (int64_t i00 = 0; i00 < ne00; i00++) {
                    مجموع += (ggml_float)(x[i00] * x[i00]);
                }

                ثابت اعشاری اوسط_2  = مجموع/ne00;
                ثابت اعشاری scale = 1.0f/sqrtf(اوسط_2 + eps);

                // if you hit this, likely you got an inf somewhere earlier
                أكد(scale > 0.0f);

                اعشاری * y = (اعشاری *) ((حرف *) dst->ڈیٹا + i01*nb1 + i02*nb2 + i03*nb3);

                اگر constexpr (FUSE_OP == GGML_RMS_NORM_FUSE_OP_MUL) {
                    ثابت int64_t i11 = i01 % ne11;
                    ثابت int64_t i12 = i02 % ne12;
                    ثابت int64_t i13 = i03 % ne13;
                    ثابت اعشاری * w = (اعشاری *) ((حرف *) src1->ڈیٹا + i11*nb11 + i12*nb12 + i13*nb13);

                    برائے (int64_t i00 = 0; i00 < ne00; i00++) {
                        y[i00] = x[i00] * scale * w[i00];
                    }
                } ورنہ {
                    حافظہ_نقل(y, x, ne00 * حجم(اعشاری));
                    ggml_vec_scale_f32(ne00, y, scale);
                }
            }
        }
    }
}

خالی ggml_compute_forward_rms_norm(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_rms_norm_f32<GGML_RMS_NORM_FUSE_OP_NONE>(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// Fused RMS_NORM + MUL: computes dst = rms_norm(src0) * src1 in a single pass.
// This avoids materializing the intermediate rms_norm result in memory.
خالی ggml_compute_forward_rms_norm_mul_fused(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst_rms_norm,
        ggml_tensor * dst_mul) {

    GGML_ASSERT(dst_mul != خالی_اشاریہ);
    GGML_ASSERT(dst_mul->src[0] == dst_rms_norm || dst_mul->src[1] == dst_rms_norm);

    ثابت ggml_tensor * src0 = dst_rms_norm->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_rms_norm_f32<GGML_RMS_NORM_FUSE_OP_MUL>(params, dst_rms_norm, dst_mul);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ساکن خالی ggml_compute_forward_rms_norm_عقب_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0]; // gradients from forward pass output
    ثابت ggml_tensor * src1 = dst->src[1]; // src1 from forward pass

    GGML_ASSERT(ggml_are_same_shape(src0, dst) && ggml_are_same_shape(src0, src1));

    GGML_ASSERT(src0->nb[0] == حجم(اعشاری));
    GGML_ASSERT(src1->nb[0] == حجم(اعشاری));

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    GGML_TENSOR_ثنائی_OP_LOCALS

    اعشاری eps;
    حافظہ_نقل(&eps, dst->op_params, حجم(اعشاری));

    // TODO: optimize
    برائے (int64_t i03 = 0; i03 < ne03; i03++) {
        برائے (int64_t i02 = 0; i02 < ne02; i02++) {
            برائے (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                // src1 is same shape as src0 => same indices
                ثابت int64_t i11 = i01;
                ثابت int64_t i12 = i02;
                ثابت int64_t i13 = i03;

                ثابت اعشاری * dz = (اعشاری *) ((حرف *) src0->ڈیٹا + i01*nb01 + i02*nb02 + i03*nb03);
                ثابت اعشاری * x  = (اعشاری *) ((حرف *) src1->ڈیٹا + i11*nb11 + i12*nb12 + i13*nb13);

                ggml_float مجموع_xx  = 0.0;
                ggml_float مجموع_xdz = 0.0;

                برائے (int64_t i00 = 0; i00 < ne00; i00++) {
                    مجموع_xx  += (ggml_float)(x[i00] * x[i00]);
                    مجموع_xdz += (ggml_float)(x[i00] * dz[i00]);
                }

                //const float mean     = (float)(sum_xx)/ne00;
                ثابت اعشاری اوسط_eps = (اعشاری)(مجموع_xx)/ne00 + eps;
                ثابت اعشاری مجموع_eps  = (اعشاری)(مجموع_xx) + eps*ne00;
                //const float mean_xdz = (float)(sum_xdz)/ne00;
                // we could cache rms from forward pass to improve performance.
                // to do this implement ggml_rms and compose ggml_rms_norm using ggml_rms.
                //const float rms      = sqrtf(mean_eps);
                ثابت اعشاری rrms     = 1.0f / sqrtf(اوسط_eps);
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
                اعشاری * dx = (اعشاری *) ((حرف *) dst->ڈیٹا + i01*nb1 + i02*nb2 + i03*nb3);

                // dx[i00] = (dz + x*(-sum_xdz/sum_eps)) * rrms
                // note: https://github.com/ggml-org/ggml/issues/1491
                ثابت اعشاری scale_x = (اعشاری) (-مجموع_xdz) / مجموع_eps;
                برائے (int64_t i00 = 0; i00 < ne00; i00++) {
                    dx[i00] = (dz[i00] + x[i00] * scale_x) * rrms;
                }
            }
        }
    }
}

خالی ggml_compute_forward_rms_norm_عقب(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_rms_norm_عقب_f32(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_group_norm

ساکن خالی ggml_compute_forward_گروہ_norm_f32(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    GGML_ASSERT(src0->nb[0] == حجم(اعشاری));

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    // TODO: optimize

    اعشاری eps;
    حافظہ_نقل(&eps, dst->op_params + 1, حجم(اعشاری));

    صحیح n_channels = src0->ne[2];
    صحیح n_groups = dst->op_params[0];
    صحیح n_channels_فی_گروہ = (n_channels + n_groups - 1) / n_groups;
    برائے (صحیح i = ith; i < n_groups; i += nth) {
        صحیح start = i * n_channels_فی_گروہ;
        صحیح end = start + n_channels_فی_گروہ;
        اگر (end > n_channels) {
            end = n_channels;
        }
        صحیح قدم = end - start;

        برائے (int64_t i03 = 0; i03 < ne03; i03++) {
            ggml_float مجموع = 0.0;
            برائے (int64_t i02 = start; i02 < end; i02++) {
                برائے (int64_t i01 = 0; i01 < ne01; i01++) {
                    ثابت اعشاری * x = (اعشاری *)((حرف *) src0->ڈیٹا + i01 * nb01 + i02 * nb02 + i03 * nb03);

                    ggml_float sumr = 0.0;
                    برائے (int64_t i00 = 0; i00 < ne00; i00++) {
                        sumr += (ggml_float)x[i00];
                    }
                    مجموع += sumr;
                }
            }
            ثابت اعشاری اوسط_2 = مجموع / (ne00 * ne01 * قدم);

            ggml_float sum2 = 0.0;
            برائے (int64_t i02 = start; i02 < end; i02++) {
                برائے (int64_t i01 = 0; i01 < ne01; i01++) {
                    ثابت اعشاری * x = (اعشاری *)((حرف *) src0->ڈیٹا + i01 * nb01 + i02 * nb02 + i03 * nb03);

                    اعشاری * y = (اعشاری *)((حرف *) dst->ڈیٹا + i01 * nb1 + i02 * nb2 + i03 * nb3);

                    ggml_float sumr = 0.0;
                    برائے (int64_t i00 = 0; i00 < ne00; i00++) {
                        اعشاری v = x[i00] - اوسط_2;
                        y[i00] = v;
                        sumr += (ggml_float)(v * v);
                    }
                    sum2 += sumr;
                }
            }
            ثابت اعشاری variance = sum2 / (ne00 * ne01 * قدم);
            ثابت اعشاری scale = 1.0f / sqrtf(variance + eps);

            برائے (int64_t i02 = start; i02 < end; i02++) {
                برائے (int64_t i01 = 0; i01 < ne01; i01++) {
                    اعشاری * y = (اعشاری *)((حرف *) dst->ڈیٹا + i01 * nb1 + i02 * nb2 + i03 * nb3);
                    ggml_vec_scale_f32(ne00, y, scale);
                }
            }
        }
    }
}

خالی ggml_compute_forward_گروہ_norm(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_گروہ_norm_f32(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_l2_norm

ساکن خالی ggml_compute_forward_l2_norm_f32(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    اعشاری eps;
    حافظہ_نقل(&eps, dst->op_params, حجم(اعشاری));

    GGML_ASSERT(eps >= 0.0f);

    // TODO: optimize
    برائے (int64_t i03 = 0; i03 < ne03; i03++) {
        برائے (int64_t i02 = 0; i02 < ne02; i02++) {
            برائے (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                ثابت حرف * x = (ثابت حرف *) src0->ڈیٹا + i01*nb01 + i02*nb02 + i03*nb03;

                ggml_float مجموع = 0.0;
                برائے (int64_t i00 = 0; i00 < ne00; i00++) {
                    ثابت اعشاری xi = *(ثابت اعشاری *) (x + i00*nb00);
                    مجموع += (ggml_float)(xi * xi);
                }

                ثابت اعشاری scale = 1.0f/fmaxf(sqrtf(مجموع), eps);

                حرف * y = (حرف *) dst->ڈیٹا + i01*nb1 + i02*nb2 + i03*nb3;

                اگر (nb00 == حجم(اعشاری) && nb0 == حجم(اعشاری)) {
                    حافظہ_نقل(y, x, ne00 * حجم(اعشاری));
                    ggml_vec_scale_f32(ne00, (اعشاری *) y, scale);
                } ورنہ {
                    برائے (int64_t i00 = 0; i00 < ne00; i00++) {
                        ثابت اعشاری xi = *(ثابت اعشاری *) (x + i00*nb00);
                        *(اعشاری *) (y + i00*nb0) = xi * scale;
                    }
                }
            }
        }
    }
}

خالی ggml_compute_forward_l2_norm(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_l2_norm_f32(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_out_prod

ساکن خالی ggml_compute_forward_برآمد_prod_f32(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ثنائی_OP_LOCALS

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    GGML_ASSERT(ne2 % ne02 == 0);
    GGML_ASSERT(ne3 % ne03 == 0);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == حجم(اعشاری));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == حجم(اعشاری));
    // GGML_ASSERT(nb0 <= nb1);
    // GGML_ASSERT(nb1 <= nb2);
    // GGML_ASSERT(nb2 <= nb3);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    اگر (ith == 0) {
        ggml_vec_سیٹ_f32(ne0*ne1*ne2*ne3, (اعشاری *)dst->ڈیٹا, 0);
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

    برائے (int64_t bir = ir0; bir < ir1; bir += blck_1) {
        ثابت int64_t bir1 = MIN(bir + blck_1, ir1);
        برائے (int64_t bi01 = 0; bi01 < ne01; bi01 += blck_0) {
            ثابت int64_t bne01 = MIN(bi01 + blck_0, ne01);
            برائے (int64_t ir = bir; ir < bir1; ++ir) {
                // dst indices
                ثابت int64_t i3 = ir/(ne2*ne1);
                ثابت int64_t i2 = (ir - i3*ne2*ne1)/ne1;
                ثابت int64_t i1 = (ir - i3*ne2*ne1 - i2*ne1);

                ثابت int64_t i02 = i2 / dps2;
                ثابت int64_t i03 = i3 / dps3;

                //const int64_t i10 = i1;
                ثابت int64_t i12 = i2;
                ثابت int64_t i13 = i3;

#اگر GGML_VEC_MAD_UNROLL > 2
                ثابت int64_t bne01_unroll = bne01 - (bne01 % GGML_VEC_MAD_UNROLL);
                برائے (int64_t i01 = bi01; i01 < bne01_unroll; i01 += GGML_VEC_MAD_UNROLL) {
                    ثابت int64_t i11 = i01;

                    اعشاری * s0 = (اعشاری *) ((حرف *) src0->ڈیٹا + (          i01*nb01 + i02*nb02 + i03*nb03));
                    اعشاری * s1 = (اعشاری *) ((حرف *) src1->ڈیٹا + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
                    اعشاری * d  = (اعشاری *) ((حرف *)  dst->ڈیٹا + (          i1*nb1   + i2*nb2   + i3*nb3));

                    ggml_vec_mad_f32_unroll(ne0, nb01, nb11, d, s0, s1);
                }
                برائے (int64_t i01 = bne01_unroll; i01 < bne01; ++i01) {
                    ثابت int64_t i11 = i01;

                    اعشاری * s0 = (اعشاری *) ((حرف *) src0->ڈیٹا + (          i01*nb01 + i02*nb02 + i03*nb03));
                    اعشاری * s1 = (اعشاری *) ((حرف *) src1->ڈیٹا + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
                    اعشاری * d  = (اعشاری *) ((حرف *)  dst->ڈیٹا + (          i1*nb1   + i2*nb2   + i3*nb3));

                    ggml_vec_mad_f32(ne0, d, s0, *s1);
                }
#ورنہ
                برائے (int64_t i01 = bi01; i01 < bne01; ++i01) {
                    ثابت int64_t i11 = i01;

                    اعشاری * s0 = (اعشاری *) ((حرف *) src0->ڈیٹا + (          i01*nb01 + i02*nb02 + i03*nb03));
                    اعشاری * s1 = (اعشاری *) ((حرف *) src1->ڈیٹا + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
                    اعشاری * d  = (اعشاری *) ((حرف *)  dst->ڈیٹا + (          i1*nb1 + i2*nb2 + i3*nb3));

                    ggml_vec_mad_f32(ne0, d, s0, *s1);
                }
#اگر_ختم
            }
        }
    }
}

ساکن خالی ggml_compute_forward_برآمد_prod_q_f32(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ثنائی_OP_LOCALS;

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت ggml_type type = src0->type;
    ggml_مقصد_float_t ثابت dequantize_صف_q = ggml_get_type_traits(type)->مقصد_float;

    GGML_ASSERT(ne02 == ne12);
    GGML_ASSERT(ne03 == ne13);
    GGML_ASSERT(ne2  == ne12);
    GGML_ASSERT(ne3  == ne13);

    // we don't support permuted src0 dim0
    GGML_ASSERT(nb00 == ggml_type_سائز(type));

    // dst dim0 cannot be transposed or permuted
    GGML_ASSERT(nb0 == حجم(اعشاری));
    // GGML_ASSERT(nb0 <= nb1);
    // GGML_ASSERT(nb1 <= nb2);
    // GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    اگر (ith == 0) {
        ggml_vec_سیٹ_f32(ne0*ne1*ne2*ne3, (اعشاری *)dst->ڈیٹا, 0);
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

    اعشاری * wdata = (اعشاری *) params->wdata + (ne0 + CACHE_سطر_سائز_F32) * ith;

    برائے (int64_t ir = ir0; ir < ir1; ++ir) {
        // dst indices
        ثابت int64_t i3 = ir/(ne2*ne1);
        ثابت int64_t i2 = (ir - i3*ne2*ne1)/ne1;
        ثابت int64_t i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ثابت int64_t i02 = i2;
        ثابت int64_t i03 = i3;

        //const int64_t i10 = i1;
        ثابت int64_t i12 = i2;
        ثابت int64_t i13 = i3;

        برائے (int64_t i01 = 0; i01 < ne01; ++i01) {
            ثابت int64_t i11 = i01;

            اعشاری * s0 = (اعشاری *) ((حرف *) src0->ڈیٹا + (          i01*nb01 + i02*nb02 + i03*nb03));
            اعشاری * s1 = (اعشاری *) ((حرف *) src1->ڈیٹا + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
            اعشاری * d  = (اعشاری *) ((حرف *)  dst->ڈیٹا + (          i1*nb1 + i2*nb2 + i3*nb3));

            dequantize_صف_q(s0, wdata, ne0);
            ggml_vec_mad_f32(ne0, d, wdata, *s1);
        }
    }
}

ساکن خالی ggml_compute_forward_برآمد_prod_f16_f32(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ثنائی_OP_LOCALS;

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ne02 == ne12);
    GGML_ASSERT(ne03 == ne13);
    GGML_ASSERT(ne2  == ne12);
    GGML_ASSERT(ne3  == ne13);

    GGML_ASSERT(nb00 == حجم(ggml_fp16_t));
    GGML_ASSERT(nb0  == حجم(اعشاری));

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    اگر (ith == 0) {
        ggml_vec_سیٹ_f32(ne0*ne1*ne2*ne3, (اعشاری *)dst->ڈیٹا, 0);
    }
    ggml_barrier(params->threadpool);

    ثابت int64_t nr = ne1*ne2*ne3;
    ثابت int64_t dr = (nr + nth - 1)/nth;
    ثابت int64_t ir0 = dr*ith;
    ثابت int64_t ir1 = MIN(ir0 + dr, nr);

    اعشاری * wdata = (اعشاری *) params->wdata + (ne0 + CACHE_سطر_سائز_F32) * ith;

    برائے (int64_t ir = ir0; ir < ir1; ++ir) {
        ثابت int64_t i3 = ir/(ne2*ne1);
        ثابت int64_t i2 = (ir - i3*ne2*ne1)/ne1;
        ثابت int64_t i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ثابت int64_t i02 = i2;
        ثابت int64_t i03 = i3;

        ثابت int64_t i12 = i2;
        ثابت int64_t i13 = i3;

        اعشاری * d = (اعشاری *) ((حرف *) dst->ڈیٹا + (i1*nb1 + i2*nb2 + i3*nb3));

        برائے (int64_t i01 = 0; i01 < ne01; ++i01) {
            ثابت int64_t i11 = i01;
            ggml_fp16_t * s0 = (ggml_fp16_t *) ((حرف *) src0->ڈیٹا + (i01*nb01 + i02*nb02 + i03*nb03));
            اعشاری * s1 = (اعشاری *) ((حرف *) src1->ڈیٹا + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
            ggml_fp16_مقصد_fp32_صف(s0, wdata, ne0);
            ggml_vec_mad_f32(ne0, d, wdata, *s1);
        }
    }
}

خالی ggml_compute_forward_برآمد_prod(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_Q1_0:
        حالت GGML_TYPE_Q2_0:
        حالت GGML_TYPE_Q4_0:
        حالت GGML_TYPE_Q4_1:
        حالت GGML_TYPE_Q5_0:
        حالت GGML_TYPE_Q5_1:
        حالت GGML_TYPE_Q8_0:
        حالت GGML_TYPE_MXFP4:
        حالت GGML_TYPE_NVFP4:
        حالت GGML_TYPE_Q2_K:
        حالت GGML_TYPE_Q3_K:
        حالت GGML_TYPE_Q4_K:
        حالت GGML_TYPE_Q5_K:
        حالت GGML_TYPE_Q6_K:
        حالت GGML_TYPE_TQ1_0:
        حالت GGML_TYPE_TQ2_0:
        حالت GGML_TYPE_IQ2_XXS:
        حالت GGML_TYPE_IQ2_XS:
        حالت GGML_TYPE_IQ3_XXS:
        حالت GGML_TYPE_IQ1_S:
        حالت GGML_TYPE_IQ1_M:
        حالت GGML_TYPE_IQ4_NL:
        حالت GGML_TYPE_IQ4_XS:
        حالت GGML_TYPE_IQ3_S:
        حالت GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_برآمد_prod_q_f32(params, dst);
            } توڑو;
        حالت GGML_TYPE_F16:
            {
                ggml_compute_forward_برآمد_prod_f16_f32(params, dst);
            } توڑو;
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_برآمد_prod_f32(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_scale

ساکن خالی ggml_compute_forward_scale_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    اعشاری s; // scale factor
    اعشاری b; // bias

    حافظہ_نقل(&s, (اعشاری *) dst->op_params + 0, حجم(اعشاری));
    حافظہ_نقل(&b, (اعشاری *) dst->op_params + 1, حجم(اعشاری));

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح nc = src0->ne[0];
    ثابت صحیح nr = ggml_nrows(src0);

    // rows per thread
    ثابت صحیح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحیح ir0 = dr*ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    ثابت size_t nb01 = src0->nb[1];

    ثابت size_t nb1 = dst->nb[1];

    اگر (b == 0.0f) {
        برائے (صحیح i1 = ir0; i1 < ir1; i1++) {
            اگر (dst->ڈیٹا != src0->ڈیٹا) {
                // src0 is same shape as dst => same indices
                // TODO: add x parameter to ggml_vec_scale_f32 and remove this memcpy
                حافظہ_نقل((حرف *)dst->ڈیٹا + i1*nb1, (حرف *)src0->ڈیٹا + i1*nb01, nc * حجم(اعشاری));
            }
            ggml_vec_scale_f32(nc, (اعشاری *) ((حرف *) dst->ڈیٹا + i1*nb1), s);
        }
    } ورنہ {
        برائے (صحیح i1 = ir0; i1 < ir1; i1++) {
            ggml_vec_mad1_f32(nc,
                (اعشاری *) ((حرف *) dst->ڈیٹا  + i1*nb1),
                (اعشاری *) ((حرف *) src0->ڈیٹا + i1*nb1),
                s, b);
        }
    }
}

ساکن خالی ggml_compute_forward_scale_bf16(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    اعشاری s; // scale factor
    اعشاری b; // bias

    حافظہ_نقل(&s, (اعشاری *) dst->op_params + 0, حجم(اعشاری));
    حافظہ_نقل(&b, (اعشاری *) dst->op_params + 1, حجم(اعشاری));

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح nc = src0->ne[0];
    ثابت صحیح nr = ggml_nrows(src0);

    // rows per thread
    ثابت صحیح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحیح ir0 = dr*ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    برائے (صحیح i1 = ir0; i1 < ir1; i1++) {
        ثابت ggml_bf16_t * x = (ثابت ggml_bf16_t *) ((ثابت حرف *) src0->ڈیٹا + i1*src0->nb[1]);
        ggml_bf16_t       * y = (ggml_bf16_t *)       ((حرف *)       dst->ڈیٹا  + i1*dst->nb[1]);
        برائے (صحیح k = 0; k < nc; k++) {
            y[k] = GGML_FP32_مقصد_BF16(s*GGML_BF16_مقصد_FP32(x[k]) + b);
        }
    }
}

خالی ggml_compute_forward_scale(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_scale_f32(params, dst);
            } توڑو;
        حالت GGML_TYPE_BF16:
            {
                ggml_compute_forward_scale_bf16(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_set

ساکن خالی ggml_compute_forward_سیٹ_f32(
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
    size_t انحراف  = ((int32_t *) dst->op_params)[3];
    منطقی   inplace = (منطقی) ((int32_t *) dst->op_params)[4];

    اگر (!inplace) {
        اگر (params->ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            حافظہ_نقل(
                ((حرف *)  dst->ڈیٹا),
                ((حرف *) src0->ڈیٹا),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح nr = ggml_nrows(src1);
    ثابت صحیح nc = src1->ne[0];

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne)
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb)

    // src0 and dst as viewed during set
    ثابت size_t nb0 = ggml_element_سائز(src0);

    ثابت صحیح im0 = (ne10 == 0 ? 0 : ne10-1);
    ثابت صحیح im1 = (ne11 == 0 ? 0 : ne11-1);
    ثابت صحیح im2 = (ne12 == 0 ? 0 : ne12-1);
    ثابت صحیح im3 = (ne13 == 0 ? 0 : ne13-1);

    GGML_ASSERT(انحراف + im0*nb0  + im1*nb1  + im2*nb2  + im3*nb3  <= ggml_nbytes(dst));

    GGML_ASSERT(nb10 == حجم(اعشاری));

    // rows per thread
    ثابت صحیح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحیح ir0 = dr*ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    برائے (صحیح ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are viewed with shape of src1 and offset
        // => same indices
        ثابت صحیح i3 = ir/(ne12*ne11);
        ثابت صحیح i2 = (ir - i3*ne12*ne11)/ne11;
        ثابت صحیح i1 = (ir - i3*ne12*ne11 - i2*ne11);

        ggml_vec_cpy_f32(nc,
                (اعشاری *) ((حرف *)  dst->ڈیٹا + i3*nb3  + i2*nb2  + i1*nb1  + انحراف),
                (اعشاری *) ((حرف *) src1->ڈیٹا + i3*nb13 + i2*nb12 + i1*nb11));
    }
}

ساکن خالی ggml_compute_forward_سیٹ_i32(
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
    size_t انحراف  = ((int32_t *) dst->op_params)[3];
    منطقی   inplace = (منطقی) ((int32_t *) dst->op_params)[4];

    اگر (!inplace) {
        اگر (params->ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            حافظہ_نقل(
                ((حرف *)  dst->ڈیٹا),
                ((حرف *) src0->ڈیٹا),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح nr = ggml_nrows(src1);
    ثابت صحیح nc = src1->ne[0];

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne)
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb)

    // src0 and dst as viewed during set
    ثابت size_t nb0 = ggml_element_سائز(src0);

    ثابت صحیح im0 = (ne10 == 0 ? 0 : ne10-1);
    ثابت صحیح im1 = (ne11 == 0 ? 0 : ne11-1);
    ثابت صحیح im2 = (ne12 == 0 ? 0 : ne12-1);
    ثابت صحیح im3 = (ne13 == 0 ? 0 : ne13-1);

    GGML_ASSERT(انحراف + im0*nb0  + im1*nb1  + im2*nb2  + im3*nb3  <= ggml_nbytes(dst));

    GGML_ASSERT(nb10 == حجم(int32_t));

    // rows per thread
    ثابت صحیح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحیح ir0 = dr*ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    برائے (صحیح ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are viewed with shape of src1 and offset
        // => same indices
        ثابت صحیح i3 = ir/(ne12*ne11);
        ثابت صحیح i2 = (ir - i3*ne12*ne11)/ne11;
        ثابت صحیح i1 = (ir - i3*ne12*ne11 - i2*ne11);

        ggml_vec_cpy_i32(nc,
                (int32_t *) ((حرف *)  dst->ڈیٹا + i3*nb3  + i2*nb2  + i1*nb1  + انحراف),
                (int32_t *) ((حرف *) src1->ڈیٹا + i3*nb13 + i2*nb12 + i1*nb11));
    }
}

خالی ggml_compute_forward_سیٹ(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_سیٹ_f32(params, dst);
            } توڑو;
        حالت GGML_TYPE_I32:
            {
                ggml_compute_forward_سیٹ_i32(params, dst);
            } توڑو;
        حالت GGML_TYPE_F16:
        حالت GGML_TYPE_BF16:
        حالت GGML_TYPE_Q1_0:
        حالت GGML_TYPE_Q2_0:
        حالت GGML_TYPE_Q4_0:
        حالت GGML_TYPE_Q4_1:
        حالت GGML_TYPE_Q5_0:
        حالت GGML_TYPE_Q5_1:
        حالت GGML_TYPE_Q8_0:
        حالت GGML_TYPE_Q8_1:
        حالت GGML_TYPE_MXFP4:
        حالت GGML_TYPE_NVFP4:
        حالت GGML_TYPE_Q2_K:
        حالت GGML_TYPE_Q3_K:
        حالت GGML_TYPE_Q4_K:
        حالت GGML_TYPE_Q5_K:
        حالت GGML_TYPE_Q6_K:
        حالت GGML_TYPE_TQ1_0:
        حالت GGML_TYPE_TQ2_0:
        حالت GGML_TYPE_IQ2_XXS:
        حالت GGML_TYPE_IQ2_XS:
        حالت GGML_TYPE_IQ3_XXS:
        حالت GGML_TYPE_IQ1_S:
        حالت GGML_TYPE_IQ1_M:
        حالت GGML_TYPE_IQ4_NL:
        حالت GGML_TYPE_IQ4_XS:
        حالت GGML_TYPE_IQ3_S:
        حالت GGML_TYPE_IQ2_S:
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_cpy

خالی ggml_compute_forward_cpy(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_dup(params, dst);
}

// ggml_compute_forward_cont

خالی ggml_compute_forward_cont(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_dup(params, dst);
}

// ggml_compute_forward_get_rows

ساکن خالی ggml_compute_forward_get_صفیں_q(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ثنائی_OP_LOCALS

    ثابت int64_t nc = ne00;
    ثابت int64_t nr = ggml_nelements(src1);

    ثابت ggml_type type = src0->type;
    ggml_مقصد_float_t ثابت dequantize_صف_q = ggml_get_type_traits(type)->مقصد_float;

    أكد(ne0  == nc);
    أكد(ne02 == ne11);
    أكد(nb00 == ggml_type_سائز(type));
    أكد(ggml_nrows(dst) == nr);

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    // rows per thread
    ثابت صحیح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحیح ir0 = dr*ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    برائے (int64_t i = ir0; i < ir1; ++i) {
        ثابت int64_t i12 = i/(ne11*ne10);
        ثابت int64_t i11 = (i - i12*ne11*ne10)/ne10;
        ثابت int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        ثابت int64_t i01 = *(int32_t *) ((حرف *) src1->ڈیٹا + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        dequantize_صف_q(
                (ثابت خالی *) ((حرف *) src0->ڈیٹا + i01*nb01 + i11*nb02 + i12*nb03),
                     (اعشاری *) ((حرف *)  dst->ڈیٹا + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

ساکن خالی ggml_compute_forward_get_صفیں_f16(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ثنائی_OP_LOCALS

    ثابت int64_t nc = ne00;
    ثابت int64_t nr = ggml_nelements(src1);

    أكد(ne0  == nc);
    أكد(ne02 == ne11);
    أكد(nb00 == حجم(ggml_fp16_t));
    أكد(ggml_nrows(dst) == nr);

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    // rows per thread
    ثابت صحیح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحیح ir0 = dr*ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    برائے (int64_t i = ir0; i < ir1; ++i) {
        ثابت int64_t i12 = i/(ne11*ne10);
        ثابت int64_t i11 = (i - i12*ne11*ne10)/ne10;
        ثابت int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        ثابت int64_t i01 = *(int32_t *) ((حرف *) src1->ڈیٹا + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        ggml_cpu_fp16_مقصد_fp32(
            (ثابت ggml_fp16_t*) ((حرف *) src0->ڈیٹا + i01*nb01 + i11*nb02 + i12*nb03),
                       (اعشاری *) ((حرف *)  dst->ڈیٹا + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

ساکن خالی ggml_compute_forward_get_صفیں_bf16(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ثنائی_OP_LOCALS

    ثابت int64_t nc = ne00;
    ثابت int64_t nr = ggml_nelements(src1);

    أكد(ne0  == nc);
    أكد(ne02 == ne11);
    أكد(nb00 == حجم(ggml_bf16_t));
    أكد(ggml_nrows(dst) == nr);

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    // rows per thread
    ثابت صحیح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحیح ir0 = dr*ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    برائے (int64_t i = ir0; i < ir1; ++i) {
        ثابت int64_t i12 = i/(ne11*ne10);
        ثابت int64_t i11 = (i - i12*ne11*ne10)/ne10;
        ثابت int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        ثابت int64_t i01 = *(int32_t *) ((حرف *) src1->ڈیٹا + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        ggml_cpu_bf16_مقصد_fp32(
            (ثابت ggml_bf16_t *) ((حرف *) src0->ڈیٹا + i01*nb01 + i11*nb02 + i12*nb03),
                        (اعشاری *) ((حرف *)  dst->ڈیٹا + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

ساکن خالی ggml_compute_forward_get_صفیں_f32(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ثنائی_OP_LOCALS

    ثابت int64_t nc = ne00;
    ثابت int64_t nr = ggml_nelements(src1);

    أكد(ne0  == nc);
    أكد(ne02 == ne11);
    أكد(nb00 == حجم(اعشاری));
    أكد(ggml_nrows(dst) == nr);

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    // rows per thread
    ثابت صحیح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحیح ir0 = dr*ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    برائے (int64_t i = ir0; i < ir1; ++i) {
        ثابت int64_t i12 = i/(ne11*ne10);
        ثابت int64_t i11 = (i - i12*ne11*ne10)/ne10;
        ثابت int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        ثابت int64_t i01 = *(int32_t *) ((حرف *) src1->ڈیٹا + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        ggml_vec_cpy_f32(nc,
                (اعشاری *) ((حرف *)  dst->ڈیٹا + i10*nb1  + i11*nb2  + i12*nb3),
                (اعشاری *) ((حرف *) src0->ڈیٹا + i01*nb01 + i11*nb02 + i12*nb03));
    }
}

خالی ggml_compute_forward_get_صفیں(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_Q1_0:
        حالت GGML_TYPE_Q2_0:
        حالت GGML_TYPE_Q4_0:
        حالت GGML_TYPE_Q4_1:
        حالت GGML_TYPE_Q5_0:
        حالت GGML_TYPE_Q5_1:
        حالت GGML_TYPE_Q8_0:
        حالت GGML_TYPE_Q8_1:
        حالت GGML_TYPE_MXFP4:
        حالت GGML_TYPE_NVFP4:
        حالت GGML_TYPE_Q2_K:
        حالت GGML_TYPE_Q3_K:
        حالت GGML_TYPE_Q4_K:
        حالت GGML_TYPE_Q5_K:
        حالت GGML_TYPE_Q6_K:
        حالت GGML_TYPE_TQ1_0:
        حالت GGML_TYPE_TQ2_0:
        حالت GGML_TYPE_IQ2_XXS:
        حالت GGML_TYPE_IQ2_XS:
        حالت GGML_TYPE_IQ3_XXS:
        حالت GGML_TYPE_IQ1_S:
        حالت GGML_TYPE_IQ1_M:
        حالت GGML_TYPE_IQ4_NL:
        حالت GGML_TYPE_IQ4_XS:
        حالت GGML_TYPE_IQ3_S:
        حالت GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_get_صفیں_q(params, dst);
            } توڑو;
        حالت GGML_TYPE_F16:
            {
                ggml_compute_forward_get_صفیں_f16(params, dst);
            } توڑو;
        حالت GGML_TYPE_BF16:
            {
                ggml_compute_forward_get_صفیں_bf16(params, dst);
            } توڑو;
        حالت GGML_TYPE_F32:
        حالت GGML_TYPE_I32:
            {
                ggml_compute_forward_get_صفیں_f32(params, dst);
            } توڑو;
        طےشدہ:
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

سانچہ<قسم_نام src_t, قسم_نام idx_t>
ساکن خالی ggml_compute_forward_سیٹ_صفیں_impl(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ثنائی_OP_LOCALS

    ثابت int64_t nc = ne00;
    ثابت int64_t nr = ne01;

    أكد(ne0  == nc);
    أكد(ne2  == ne02);
    أكد(ne3  == ne03);
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    أكد(ne02 % ne11 == 0);
    أكد(ne03 % ne12 == 0);

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    // rows per thread
    ثابت int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت int64_t ir0 = dr*ith;
    ثابت int64_t ir1 = std::min(ir0 + dr, nr);

    ثابت size_t rs = ggml_صف_سائز(src0->type, nc);

    ggml_from_float_t ثابت from_float = ggml_get_type_traits_cpu(dst->type)->from_float;

    برائے (int64_t i03 = 0; i03 < ne03; ++i03) {
        برائے (int64_t i02 = 0; i02 < ne02; ++i02) {
            برائے (int64_t i = ir0; i < ir1; ++i) {
                ثابت int64_t i12 = i03%ne12;
                ثابت int64_t i11 = i02%ne11;
                ثابت int64_t i10 = i;

                ثابت int64_t i1 = *(idx_t *) ((حرف *) src1->ڈیٹا + i10*nb10 + i11*nb11 + i12*nb12);

                GGML_ASSERT(i1 >= 0 && i1 < ne1);

                اگر constexpr (std::is_same_v<src_t, اعشاری>) {
                    from_float(
                            (ثابت اعشاری *) ((حرف *) src0->ڈیٹا +  i*nb01 + i02*nb02 + i03*nb03),
                                            ((حرف *)  dst->ڈیٹا + i1*nb1  + i02*nb2  + i03*nb3), nc);
                } ورنہ اگر constexpr (std::is_same_v<src_t, ggml_fp16_t>) {
                    اگر (dst->type == GGML_TYPE_F16) {
                        حافظہ_نقل(
                                            ((حرف *)  dst->ڈیٹا + i1*nb1  + i02*nb2  + i03*nb3),
                                            ((حرف *) src0->ڈیٹا +  i*nb01 + i02*nb02 + i03*nb03),
                                            rs);
                    } ورنہ {
                        اعشاری * wdata = (اعشاری *) params->wdata + (nc + CACHE_سطر_سائز_F32) * ith;
                        ggml_fp16_مقصد_fp32_صف(
                                (ثابت ggml_fp16_t *) ((حرف *) src0->ڈیٹا + i*nb01 + i02*nb02 + i03*nb03),
                                wdata, nc);
                        from_float(wdata,
                                ((حرف *) dst->ڈیٹا + i1*nb1 + i02*nb2 + i03*nb3), nc);
                    }
                } ورنہ {
                    GGML_ABORT("src0->type = %d (%s) not supported", src0->type, ggml_type_name(src0->type));
                }
            }
        }
    }
}

خالی ggml_compute_forward_سیٹ_صفیں(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                اگر (src1->type == GGML_TYPE_I64) {
                    ggml_compute_forward_سیٹ_صفیں_impl<اعشاری, int64_t>(params, dst);
                } ورنہ اگر (src1->type == GGML_TYPE_I32) {
                    ggml_compute_forward_سیٹ_صفیں_impl<اعشاری, int32_t>(params, dst);
                } ورنہ {
                    GGML_ABORT("src1->type = %d (%s) not supported", src1->type, ggml_type_name(src1->type));
                }
            } توڑو;
        حالت GGML_TYPE_F16:
            {
                اگر (src1->type == GGML_TYPE_I64) {
                    ggml_compute_forward_سیٹ_صفیں_impl<ggml_fp16_t, int64_t>(params, dst);
                } ورنہ اگر (src1->type == GGML_TYPE_I32) {
                    ggml_compute_forward_سیٹ_صفیں_impl<ggml_fp16_t, int32_t>(params, dst);
                } ورنہ {
                    GGML_ABORT("src1->type = %d (%s) not supported", src1->type, ggml_type_name(src1->type));
                }
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("src0->type = %d (%s) not supported", src0->type, ggml_type_name(src0->type));
            }
    }
}

// ggml_compute_forward_get_rows_back

ساکن خالی ggml_compute_forward_get_صفیں_عقب_f32_f16(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    اگر (params->ith != 0) {
        واپس;
    }

    GGML_ASSERT(ggml_is_contiguous(dst));

    // ggml_compute_forward_dup_same_cont(params, opt0, dst);

    memset(dst->ڈیٹا, 0, ggml_nbytes(dst));

    ثابت صحیح nc = src0->ne[0];
    ثابت صحیح nr = ggml_nelements(src1);

    GGML_ASSERT( dst->ne[0] == nc);
    GGML_ASSERT(src0->nb[0] == حجم(ggml_fp16_t));

    برائے (صحیح i = 0; i < nr; ++i) {
        ثابت صحیح r = ((int32_t *) src1->ڈیٹا)[i];

        GGML_ASSERT(r >= 0 && r < dst->ne[1]);

        برائے (صحیح j = 0; j < nc; ++j) {
            ggml_fp16_t v = ((ggml_fp16_t *) ((حرف *) src0->ڈیٹا + i*src0->nb[1]))[j];
            ((اعشاری *) ((حرف *) dst->ڈیٹا + r*dst->nb[1]))[j] += GGML_CPU_FP16_مقصد_FP32(v);
        }
    }
}

ساکن خالی ggml_compute_forward_get_صفیں_عقب_f32(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    اگر (params->ith != 0) {
        واپس;
    }

    GGML_ASSERT(ggml_is_contiguous(dst));

    // ggml_compute_forward_dup_same_cont(params, opt0, dst);

    memset(dst->ڈیٹا, 0, ggml_nbytes(dst));

    ثابت صحیح nc = src0->ne[0];
    ثابت صحیح nr = ggml_nelements(src1);

    GGML_ASSERT( dst->ne[0] == nc);
    GGML_ASSERT(src0->nb[0] == حجم(اعشاری));

    برائے (صحیح i = 0; i < nr; ++i) {
        ثابت صحیح r = ((int32_t *) src1->ڈیٹا)[i];

        GGML_ASSERT(r >= 0 && r < dst->ne[1]);

        ggml_vec_جوڑو_f32(nc,
                (اعشاری *) ((حرف *)  dst->ڈیٹا + r*dst->nb[1]),
                (اعشاری *) ((حرف *)  dst->ڈیٹا + r*dst->nb[1]),
                (اعشاری *) ((حرف *) src0->ڈیٹا + i*src0->nb[1]));
    }
}

خالی ggml_compute_forward_get_صفیں_عقب(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F16:
            {
                ggml_compute_forward_get_صفیں_عقب_f32_f16(params, dst);
            } توڑو;
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_get_صفیں_عقب_f32(params, dst);
            } توڑو;
        طےشدہ:
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

ساکن خالی ggml_compute_forward_diag_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    اگر (params->ith != 0) {
        واپس;
    }

    // TODO: handle transposed/permuted matrices

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(ne00 == ne0);
    GGML_ASSERT(ne00 == ne1);
    GGML_ASSERT(ne01 == 1);
    GGML_ASSERT(ne02 == ne2);
    GGML_ASSERT(ne03 == ne3);

    GGML_ASSERT(nb00 == حجم(اعشاری));
    GGML_ASSERT(nb0  == حجم(اعشاری));

    برائے (صحیح i3 = 0; i3 < ne3; i3++) {
        برائے (صحیح i2 = 0; i2 < ne2; i2++) {
            برائے (صحیح i1 = 0; i1 < ne1; i1++) {
                اعشاری * d = (اعشاری *)((حرف *)  dst->ڈیٹا + i3*nb3  + i2*nb2 + i1*nb1);
                اعشاری * s = (اعشاری *)((حرف *) src0->ڈیٹا + i3*nb03 + i2*nb02);
                برائے (صحیح i0 = 0; i0 < i1; i0++) {
                    d[i0] = 0;
                }
                d[i1] = s[i1];
                برائے (صحیح i0 = i1+1; i0 < ne0; i0++) {
                    d[i0] = 0;
                }
            }
        }
    }
}

خالی ggml_compute_forward_diag(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_diag_f32(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_diag_mask_inf

ساکن خالی ggml_compute_forward_diag_نقاب_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst,
        ثابت اعشاری value) {

    ثابت ggml_tensor * src0 = dst->src[0];

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح  n_past  = ((int32_t *) dst->op_params)[0];
    ثابت منطقی inplace = src0->ڈیٹا == dst->ڈیٹا;

    GGML_ASSERT(n_past >= 0);

    اگر (!inplace) {
        اگر (ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
            GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));
            حافظہ_نقل(
                ((حرف *)  dst->ڈیٹا),
                ((حرف *) src0->ڈیٹا),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    // TODO: handle transposed/permuted matrices

    ثابت صحیح n  = ggml_nrows(src0);
    ثابت صحیح nc = src0->ne[0];
    ثابت صحیح nr = src0->ne[1];
    ثابت صحیح nz = n/nr;

    GGML_ASSERT( dst->nb[0] == حجم(اعشاری));
    GGML_ASSERT(src0->nb[0] == حجم(اعشاری));

    برائے (صحیح k = 0; k < nz; k++) {
        برائے (صحیح j = ith; j < nr; j += nth) {
            برائے (صحیح i = n_past; i < nc; i++) {
                اگر (i > n_past + j) {
                    *(اعشاری *)((حرف *) dst->ڈیٹا + k*dst->nb[2] + j*dst->nb[1] + i*dst->nb[0]) = value;
                }
            }
        }
    }
}

خالی ggml_compute_forward_diag_نقاب_inf(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_diag_نقاب_f32(params, dst, -INFINITY);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

خالی ggml_compute_forward_diag_نقاب_zero(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_diag_نقاب_f32(params, dst, 0);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_soft_max

ساکن خالی ggml_compute_forward_soft_max_f32(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    ثابت ggml_tensor * src2 = dst->src[2];

    أكد(ggml_is_contiguous(dst));
    أكد(ggml_are_same_shape(src0, dst));

    اعشاری scale    = 1.0f;
    اعشاری max_bias = 0.0f;

    حافظہ_نقل(&scale,    (اعشاری *) dst->op_params + 0, حجم(اعشاری));
    حافظہ_نقل(&max_bias, (اعشاری *) dst->op_params + 1, حجم(اعشاری));

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

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

    ثابت اعشاری m0 = powf(2.0f, -(max_bias       ) / n_سر_log2);
    ثابت اعشاری m1 = powf(2.0f, -(max_bias / 2.0f) / n_سر_log2);

    اعشاری * wp = (اعشاری *) params->wdata + (ne00 + CACHE_سطر_سائز_F32) * ith;

    ثابت منطقی use_f16 = (src1 && src1->type == GGML_TYPE_F16);

    // sinks
    ثابت اعشاری * sk = src2 ? (اعشاری *)((حرف *) src2->ڈیٹا) : خالی_اشاریہ;

    برائے (int64_t i03 = 0; i03 < ne03; i03++) {
        برائے (int64_t i02 = 0; i02 < ne02; i02++) {
            برائے (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                ثابت int64_t i11 = i01;
                ثابت int64_t i12 = i02%ne12;
                ثابت int64_t i13 = i03%ne13;

                // ALiBi
                ثابت uint32_t h = i02; //سر
                ثابت اعشاری ڈھلان = (max_bias > 0.0f) ? h < n_سر_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_سر_log2) + 1) : 1.0f;

                اعشاری * sp = (اعشاری *)((حرف *) src0->ڈیٹا + i01*nb01 + i02*nb02 + i03*nb03);
                اعشاری * dp = (اعشاری *)((حرف *)  dst->ڈیٹا + i01*nb1  + i02*nb2  + i03*nb3);

                // broadcast the mask across rows
                ggml_fp16_t * mp_f16 = src1 ? (ggml_fp16_t *)((حرف *) src1->ڈیٹا + i11*nb11 + i12*nb12 + i13*nb13) : تہی;
                اعشاری       * mp_f32 = src1 ? (اعشاری       *)((حرف *) src1->ڈیٹا + i11*nb11 + i12*nb12 + i13*nb13) : تہی;

                ggml_vec_cpy_f32  (ne00, wp, sp);
                ggml_vec_scale_f32(ne00, wp, scale);
                اگر (mp_f32) {
                    اگر (use_f16) {
                        برائے (صحیح i = 0; i < ne00; ++i) {
                            wp[i] += ڈھلان*GGML_CPU_FP16_مقصد_FP32(mp_f16[i]);
                        }
                    } ورنہ {
                        برائے (صحیح i = 0; i < ne00; ++i) {
                            wp[i] += ڈھلان*mp_f32[i];
                        }
                    }
                }

#اگر_غیر_تعریف_شدہ NDEBUG
                برائے (صحیح i = 0; i < ne00; ++i) {
                    //printf("p[%d] = %f\n", i, p[i]);
                    أكد(!isnan(wp[i]));
                }
#اگر_ختم // NDEBUG

                اعشاری max = -INFINITY;
                ggml_vec_max_f32(ne00, &max, wp);

                // if we have sinks, make a correction as if they were included in the softmax
                اگر (sk) {
                    max = MAX(max, sk[i02]);
                }

                ggml_float مجموع = ggml_vec_soft_max_f32(ne00, dp, wp, max);
                أكد(مجموع > 0.0);

                اگر (sk) {
                    مجموع += (ggml_float) expf(sk[i02] - max);
                }

                مجموع = 1.0/مجموع;
                ggml_vec_scale_f32(ne00, dp, مجموع);

#اگر_غیر_تعریف_شدہ NDEBUG
                برائے (صحیح i = 0; i < ne00; ++i) {
                    أكد(!isnan(dp[i]));
                    أكد(!isinf(dp[i]));
                }
#اگر_ختم // NDEBUG
            }
        }
    }
}

خالی ggml_compute_forward_soft_max(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_soft_max_f32(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}


// ggml_compute_forward_soft_max_ext_back

ساکن خالی ggml_compute_forward_soft_max_ext_عقب_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_are_same_shape(src1, dst));

    اعشاری scale    = 1.0f;
    اعشاری max_bias = 0.0f;

    حافظہ_نقل(&scale,    (ثابت اعشاری *) dst->op_params + 0, حجم(اعشاری));
    حافظہ_نقل(&max_bias, (ثابت اعشاری *) dst->op_params + 1, حجم(اعشاری));

    GGML_ASSERT(max_bias == 0.0f);

    // TODO: handle transposed/permuted matrices

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح nc = src0->ne[0];
    ثابت صحیح nr = ggml_nrows(src0);

    // rows per thread
    ثابت صحیح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحیح ir0 = dr*ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    برائے (صحیح i1 = ir0; i1 < ir1; i1++) {
        اعشاری *dy = (اعشاری *)((حرف *) src0->ڈیٹا + i1*src0->nb[1]);
        اعشاری *y  = (اعشاری *)((حرف *) src1->ڈیٹا + i1*src1->nb[1]);
        اعشاری *dx = (اعشاری *)((حرف *) dst->ڈیٹا  + i1*dst->nb[1]);

#اگر_غیر_تعریف_شدہ NDEBUG
        برائے (صحیح i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            أكد(!isnan(dy[i]));
            أكد(!isnan(y[i]));
        }
#اگر_ختم // NDEBUG
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
        اعشاری dot_y_dy = 0;
        ggml_vec_dot_f32(nc, &dot_y_dy, 0, y, 0, dy, 0, 1);
        برائے (صحیح i = 0; i < nc; i++) {
            dx[i] = scale * (dy[i] - dot_y_dy) * y[i];
        }

#اگر_غیر_تعریف_شدہ NDEBUG
        برائے (صحیح i = 0; i < nc; ++i) {
            أكد(!isnan(dx[i]));
            أكد(!isinf(dx[i]));
        }
#اگر_ختم // NDEBUG
    }
}

خالی ggml_compute_forward_soft_max_ext_عقب(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_soft_max_ext_عقب_f32(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_clamp

ساکن خالی ggml_compute_forward_clamp_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    اعشاری min;
    اعشاری max;
    حافظہ_نقل(&min, (اعشاری *) dst->op_params + 0, حجم(اعشاری));
    حافظہ_نقل(&max, (اعشاری *) dst->op_params + 1, حجم(اعشاری));

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح n  = ggml_nrows(src0);
    ثابت صحیح nc = src0->ne[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == حجم(اعشاری));
    GGML_ASSERT(nb00 == حجم(اعشاری));

    برائے (صحیح j = ith; j < n; j += nth) {
        ثابت int64_t i1 = j % ne01;
        ثابت int64_t i2 = (j / ne01) % ne02;
        ثابت int64_t i3 = j / (ne01*ne02);

        اعشاری * dst_ptr  = (اعشاری *) ((حرف *)  dst->ڈیٹا + i1*nb1  + i2*nb2  + i3*nb3);
        اعشاری * src0_ptr = (اعشاری *) ((حرف *) src0->ڈیٹا + i1*nb01 + i2*nb02 + i3*nb03);

        برائے (صحیح i = 0; i < nc; i++) {
            dst_ptr[i] = MAX(MIN(src0_ptr[i], max), min);
        }
    }
}

ساکن خالی ggml_compute_forward_clamp_f16(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    اعشاری min;
    اعشاری max;
    حافظہ_نقل(&min, (اعشاری *) dst->op_params + 0, حجم(اعشاری));
    حافظہ_نقل(&max, (اعشاری *) dst->op_params + 1, حجم(اعشاری));

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح n  = ggml_nrows(src0);
    ثابت صحیح nc = src0->ne[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == حجم(ggml_fp16_t));
    GGML_ASSERT(nb00 == حجم(ggml_fp16_t));

    برائے (صحیح j = ith; j < n; j += nth) {
        ثابت int64_t i1 = j % ne01;
        ثابت int64_t i2 = (j / ne01) % ne02;
        ثابت int64_t i3 = j / (ne01*ne02);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((حرف *)  dst->ڈیٹا + i1*nb1  + i2*nb2  + i3*nb3);
        ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((حرف *) src0->ڈیٹا + i1*nb01 + i2*nb02 + i3*nb03);

        برائے (صحیح i = 0; i < nc; i++) {
            اعشاری v = GGML_CPU_FP16_مقصد_FP32(src0_ptr[i]);
            dst_ptr[i] = GGML_CPU_FP32_مقصد_FP16(MAX(MIN(v, max), min));
        }
    }
}

خالی ggml_compute_forward_clamp(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_clamp_f32(params, dst);
            } توڑو;
        حالت GGML_TYPE_F16:
            {
                ggml_compute_forward_clamp_f16(params, dst);
            } توڑو;
        حالت GGML_TYPE_BF16:
        حالت GGML_TYPE_Q1_0:
        حالت GGML_TYPE_Q2_0:
        حالت GGML_TYPE_Q4_0:
        حالت GGML_TYPE_Q4_1:
        حالت GGML_TYPE_Q5_0:
        حالت GGML_TYPE_Q5_1:
        حالت GGML_TYPE_Q8_0:
        حالت GGML_TYPE_Q8_1:
        حالت GGML_TYPE_MXFP4:
        حالت GGML_TYPE_NVFP4:
        حالت GGML_TYPE_Q2_K:
        حالت GGML_TYPE_Q3_K:
        حالت GGML_TYPE_Q4_K:
        حالت GGML_TYPE_Q5_K:
        حالت GGML_TYPE_Q6_K:
        حالت GGML_TYPE_TQ1_0:
        حالت GGML_TYPE_TQ2_0:
        حالت GGML_TYPE_IQ2_XXS:
        حالت GGML_TYPE_IQ2_XS:
        حالت GGML_TYPE_IQ3_XXS:
        حالت GGML_TYPE_IQ1_S:
        حالت GGML_TYPE_IQ1_M:
        حالت GGML_TYPE_IQ4_NL:
        حالت GGML_TYPE_IQ4_XS:
        حالت GGML_TYPE_IQ3_S:
        حالت GGML_TYPE_IQ2_S:
        حالت GGML_TYPE_Q8_K:
        حالت GGML_TYPE_I8:
        حالت GGML_TYPE_I16:
        حالت GGML_TYPE_I32:
        حالت GGML_TYPE_I64:
        حالت GGML_TYPE_F64:
        حالت GGML_TYPE_گنتی:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_rope

ساکن اعشاری rope_yarn_ramp(ثابت اعشاری کم, ثابت اعشاری بلند, ثابت صحیح i0) {
    ثابت اعشاری y = (i0 / 2 - کم) / MAX(0.001f, بلند - کم);
    واپس 1 - MIN(1, MAX(0, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
ساکن خالی rope_yarn(
    اعشاری theta_extrap, اعشاری freq_scale, اعشاری corr_dims[2], int64_t i0, اعشاری ext_ضربی_عدد, اعشاری mscale,
    اعشاری * cos_theta, اعشاری * sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    اعشاری theta_interp = freq_scale * theta_extrap;
    اعشاری theta = theta_interp;
    اگر (ext_ضربی_عدد != 0.0f) {
        اعشاری ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_ضربی_عدد;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
}

ساکن خالی ggml_rope_cache_init(
     اعشاری theta_بنیاد, اعشاری freq_scale, ثابت اعشاری * freq_factors, اعشاری corr_dims[2], int64_t ne0, اعشاری ext_ضربی_عدد, اعشاری mscale,
     اعشاری * cache, اعشاری sin_sign, اعشاری theta_scale) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    اعشاری theta = theta_بنیاد;
    برائے (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        ثابت اعشاری ff = freq_factors ? freq_factors[i0/2] : 1.0f;
        rope_yarn(
            theta/ff, freq_scale, corr_dims, i0, ext_ضربی_عدد, mscale, &cache[i0 + 0], &cache[i0 + 1]
        );
        cache[i0 + 1] *= sin_sign;

        theta *= theta_scale;
    }
}

ساکن خالی ggml_mrope_cache_init(
     اعشاری theta_بنیاد_t, اعشاری theta_بنیاد_h, اعشاری theta_بنیاد_w, اعشاری theta_بنیاد_e, صحیح sections[4], منطقی is_imrope, منطقی indep_sects,
     اعشاری freq_scale, ثابت اعشاری * freq_factors, اعشاری corr_dims[2], int64_t ne0, اعشاری ext_ضربی_عدد, اعشاری mscale,
     اعشاری * cache, اعشاری sin_sign, اعشاری theta_scale) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    اعشاری theta_t = theta_بنیاد_t;
    اعشاری theta_h = theta_بنیاد_h;
    اعشاری theta_w = theta_بنیاد_w;
    اعشاری theta_e = theta_بنیاد_e;  // extra position id for vision encoder
    صحیح sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    صحیح sec_w = sections[1] + sections[0];
    صحیح sec_e = sections[2] + sec_w;
    GGML_ASSERT(sect_dims <= ne0);

    برائے (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        ثابت اعشاری ff = freq_factors ? freq_factors[i0/2] : 1.0f;

        صحیح sector = (i0 / 2) % sect_dims;
        اگر (indep_sects) {
            // compute theta independently for each dim sections
            // (i.e. reset corresponding theta when `i0` go from one section to another)
            اگر (sector == 0) {
                theta_t = theta_بنیاد_t;
            }
            ورنہ اگر (sector == sections[0]) {
                theta_h = theta_بنیاد_h;;
            }
            ورنہ اگر (sector == sec_w) {
                theta_w = theta_بنیاد_w;
            }
            ورنہ اگر (sector == sec_e) {
                theta_e = theta_بنیاد_e;
            }
        }

        اعشاری theta = theta_t;
        اگر (is_imrope) { // qwen3vl apply interleaved mrope
            اگر (sector % 3 == 1 && sector < 3 * sections[1]) {
                theta = theta_h;
            } ورنہ اگر (sector % 3 == 2 && sector < 3 * sections[2]) {
                theta = theta_w;
            } ورنہ اگر (sector % 3 == 0 && sector < 3 * sections[0]) {
                theta = theta_t;
            } ورنہ {
                theta = theta_e;
            }
        } ورنہ {
            اگر (sector >= sections[0] && sector < sec_w) {
                theta = theta_h;
            }
            ورنہ اگر (sector >= sec_w && sector < sec_w + sections[2]) {
                theta = theta_w;
            }
            ورنہ اگر (sector >= sec_w + sections[2]) {
                theta = theta_e;
            }
        }

        rope_yarn(
            theta/ff, freq_scale, corr_dims, i0, ext_ضربی_عدد, mscale, &cache[i0 + 0], &cache[i0 + 1]
        );
        cache[i0 + 1] *= sin_sign;

        theta_t *= theta_scale;
        theta_w *= theta_scale;
        theta_h *= theta_scale;
        theta_e *= theta_scale;
    }
}


سانچہ<قسم_نام T>
ساکن خالی rotate_جوڑے(ثابت int64_t n, ثابت int64_t n_انحراف, ثابت اعشاری * cache, ثابت T * src_ڈیٹا, T * dst_ڈیٹا, ثابت صحیح scale = 2) {
  برائے (int64_t i0 = 0; i0 < n; i0 += 2) {
    ثابت int64_t ic = i0/scale; // hack for GGML_ROPE_TYPE_NORMAL, where we need ic = i0; for all other cases, ic = i0/2

    ثابت اعشاری cos_theta = cache[i0 + 0];
    ثابت اعشاری sin_theta = cache[i0 + 1];

    ثابت T * ثابت src = src_ڈیٹا + ic;
    T * dst             = dst_ڈیٹا + ic;

    ثابت اعشاری x0 = type_conversion_جدول<T>::مقصد_f32(src[0]);
    ثابت اعشاری x1 = type_conversion_جدول<T>::مقصد_f32(src[n_انحراف]);

    dst[0]        = type_conversion_جدول<T>::from_f32(x0*cos_theta - x1*sin_theta);
    dst[n_انحراف] = type_conversion_جدول<T>::from_f32(x0*sin_theta + x1*cos_theta);
  }
}

سانچہ<قسم_نام T> //float or ggml_fp16_t
ساکن خالی ggml_compute_forward_rope_flt(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst,
        ثابت منطقی forward) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    ثابت ggml_tensor * src2 = dst->src[2];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);

    اعشاری freq_بنیاد, freq_scale, ext_ضربی_عدد, attn_ضربی_عدد, beta_fast, beta_slow;
    صحیح sections[4];

    //const int n_past     = ((int32_t *) dst->op_params)[0];
    ثابت صحیح n_dims     = ((int32_t *) dst->op_params)[1];
    ثابت صحیح mode       = ((int32_t *) dst->op_params)[2];
    //const int n_ctx      = ((int32_t *) dst->op_params)[3];
    ثابت صحیح n_ctx_orig = ((int32_t *) dst->op_params)[4];

    حافظہ_نقل(&freq_بنیاد,   (int32_t *) dst->op_params +  5, حجم(اعشاری));
    حافظہ_نقل(&freq_scale,  (int32_t *) dst->op_params +  6, حجم(اعشاری));
    حافظہ_نقل(&ext_ضربی_عدد,  (int32_t *) dst->op_params +  7, حجم(اعشاری));
    حافظہ_نقل(&attn_ضربی_عدد, (int32_t *) dst->op_params +  8, حجم(اعشاری));
    حافظہ_نقل(&beta_fast,   (int32_t *) dst->op_params +  9, حجم(اعشاری));
    حافظہ_نقل(&beta_slow,   (int32_t *) dst->op_params + 10, حجم(اعشاری));
    حافظہ_نقل(&sections,    (int32_t *) dst->op_params + 11, حجم(صحیح)*4);

    ثابت صحیح n_offs = ((int32_t *) dst->op_params)[15];

    GGML_TENSOR_UNARY_OP_LOCALS

    //printf("ne0: %d, ne1: %d, ne2: %d, ne3: %d\n", ne0, ne1, ne2, ne3);
    //printf("n_past = %d, ne2 = %d\n", n_past, ne2);

    GGML_ASSERT(nb0 == nb00);
    GGML_ASSERT(nb0 == حجم(T));

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح nr = ggml_nrows(dst);

    GGML_ASSERT(n_dims <= ne0);
    GGML_ASSERT(n_dims % 2 == 0);

    GGML_ASSERT(n_offs >= 0);
    GGML_ASSERT(n_offs % 2 == 0);
    GGML_ASSERT(n_offs + n_dims <= ne0);

    // rows per thread
    ثابت صحیح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحیح ir0 = dr*ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    // row index used to determine which thread to use
    صحیح ir = 0;

    ثابت اعشاری theta_scale = powf(freq_بنیاد, -2.0f/n_dims);

    اعشاری corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_بنیاد, beta_fast, beta_slow, corr_dims);

    ثابت منطقی is_imrope = mode == GGML_ROPE_TYPE_IMROPE; // qwen3vl apply interleaved mrope
    ثابت منطقی mrope_مستعمل = mode & GGML_ROPE_TYPE_MROPE;  // ggml_rope_multi, note: also true for vision (24 & 8 == true) and for imrope
    ثابت منطقی is_vision = mode == GGML_ROPE_TYPE_VISION;

    اگر (mrope_مستعمل) {
        GGML_ASSERT(sections[0] > 0 || sections[1] > 0 || sections[2] > 0);
    }

    اگر (is_vision) {
        GGML_ASSERT(n_dims == ne0/2);
        GGML_ASSERT(n_offs == 0);
    }

    ثابت اعشاری * freq_factors = تہی;
    اگر (src2 != تہی) {
        GGML_ASSERT(src2->type == GGML_TYPE_F32);
        GGML_ASSERT(src2->ne[0] >= n_dims / 2);
        freq_factors = (ثابت اعشاری *) src2->ڈیٹا;
    }

    // backward process uses inverse rotation by cos and sin.
    // cos and sin build a rotation matrix, where the inverse is the transpose.
    // this essentially just switches the sign of sin.
    ثابت اعشاری sin_sign = forward ? 1.0f : -1.0f;

    ثابت int32_t * pos = (ثابت int32_t *) src1->ڈیٹا;

    int64_t آخری_i2 = -1;

    برائے (int64_t i3 = 0; i3 < ne3; i3++) { // batch
        برائے (int64_t i2 = 0; i2 < ne2; i2++) { // seq-len
            برائے (int64_t i1 = 0; i1 < ne1; i1++) { // attn-heads
                اگر (ir++ < ir0) جاری; // skip rows mapped to other threads
                اگر (ir   > ir1) توڑو;

                اعشاری * cache = (اعشاری *) params->wdata + (ne0 + CACHE_سطر_سائز_F32)*ith;
                اگر (آخری_i2 != i2) {
                    اگر (!mrope_مستعمل) {
                        ثابت int64_t p = pos[i2];
                        ggml_rope_cache_init(p, freq_scale, freq_factors, corr_dims, ne0, ext_ضربی_عدد, attn_ضربی_عدد, cache, sin_sign, theta_scale);
                    }
                    ورنہ {
                        ثابت int64_t p_t = pos[i2];
                        ثابت int64_t p_h = pos[i2 + ne2];
                        ثابت int64_t p_w = pos[i2 + ne2 * 2];
                        ثابت int64_t p_e = pos[i2 + ne2 * 3];
                        ggml_mrope_cache_init(
                            p_t, p_h, p_w, p_e, sections, is_imrope, is_vision,
                            freq_scale, freq_factors, corr_dims, ne0, ext_ضربی_عدد, attn_ضربی_عدد, cache, sin_sign, theta_scale);
                    }

                    آخری_i2 = i2;
                }

                T * src = (T *)((حرف *) src0->ڈیٹا + i3*nb03 + i2*nb02 + i1*nb01);
                T * dst_ڈیٹا  = (T *)((حرف *)  dst->ڈیٹا + i3*nb3  + i2*nb2  + i1*nb1);

                انتخاب (mode) {
                    حالت GGML_ROPE_TYPE_معمول:
                        rotate_جوڑے<T>(n_dims, 1, cache, src + n_offs, dst_ڈیٹا + n_offs, 1);
                        توڑو;
                    حالت GGML_ROPE_TYPE_NEOX:
                    حالت GGML_ROPE_TYPE_MROPE:
                    حالت GGML_ROPE_TYPE_IMROPE:
                        rotate_جوڑے<T>(n_dims, n_dims/2, cache, src + n_offs, dst_ڈیٹا + n_offs);
                        توڑو;
                    حالت GGML_ROPE_TYPE_VISION:
                        rotate_جوڑے<T>(ne0, n_dims, cache, src, dst_ڈیٹا);
                        توڑو;
                    طےشدہ:
                        GGML_ABORT("rope type not supported");
                }

                اگر (!is_vision) {
                    // fill the remain channels with data from src tensor
                    برائے (int64_t i0 = 0; i0 < ne0; i0 += 2) {
                        اگر (i0 == n_offs) {
                            i0 += n_dims - 2; // skip the rotated channels
                            جاری;
                        }
                        ثابت T * ثابت src = (T *)((حرف *) src0->ڈیٹا + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                        T * dst_ڈیٹا  = (T *)((حرف *)  dst->ڈیٹا + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

                        dst_ڈیٹا[0] = src[0];
                        dst_ڈیٹا[1] = src[1];
                    }
                }
            } //attn-heads
        }
    }
}

خالی ggml_compute_forward_rope(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F16:
            {
                ggml_compute_forward_rope_flt<ggml_fp16_t>(params, dst, سچ);
            } توڑو;
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_rope_flt<اعشاری>(params, dst, سچ);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_rope_back

خالی ggml_compute_forward_rope_عقب(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F16:
            {
                ggml_compute_forward_rope_flt<ggml_fp16_t>(params, dst, جھوٹ);
            } توڑو;
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_rope_flt<اعشاری>(params, dst, جھوٹ);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_conv_transpose_1d

ساکن خالی ggml_compute_forward_conv_transpose_1d_f16_f32(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ثنائی_OP_LOCALS

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح nk = ne00*ne01*ne02;

    GGML_ASSERT(nb00 == حجم(ggml_fp16_t));
    GGML_ASSERT(nb10 == حجم(اعشاری));

    اگر (ith == 0) {
        memset(params->wdata, 0, params->wsize);

        // permute kernel data (src0) from (K x Cout x Cin) to (Cin x K x Cout)
        {
            ggml_fp16_t * ثابت wdata = (ggml_fp16_t *) params->wdata + 0;

            برائے (int64_t i02 = 0; i02 < ne02; i02++) {
                برائے (int64_t i01 = 0; i01 < ne01; i01++) {
                    ثابت ggml_fp16_t * ثابت src = (ggml_fp16_t *)((حرف *) src0->ڈیٹا + i02*nb02 + i01*nb01);
                    ggml_fp16_t * dst_ڈیٹا = wdata + i01*ne00*ne02;
                    برائے (int64_t i00 = 0; i00 < ne00; i00++) {
                        dst_ڈیٹا[i00*ne02 + i02] = src[i00];
                    }
                }
            }
        }

        // permute source data (src1) from (L x Cin) to (Cin x L)
        {
            ggml_fp16_t * ثابت wdata = (ggml_fp16_t *) params->wdata + nk;
            ggml_fp16_t * dst_ڈیٹا = wdata;

            برائے (int64_t i11 = 0; i11 < ne11; i11++) {
                ثابت اعشاری * ثابت src = (اعشاری *)((حرف *) src1->ڈیٹا + i11*nb11);
                برائے (int64_t i10 = 0; i10 < ne10; i10++) {
                    dst_ڈیٹا[i10*ne11 + i11] = GGML_CPU_FP32_مقصد_FP16(src[i10]);
                }
            }
        }

        // need to zero dst since we are accumulating into it
        memset(dst->ڈیٹا, 0, ggml_nbytes(dst));
    }
    ggml_barrier(params->threadpool);

    ثابت int32_t s0 = ((ثابت int32_t*)(dst->op_params))[0];

    // total rows in dst
    ثابت صحیح nr = ne1;

    // rows per thread
    ثابت صحیح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحیح ir0 = dr*ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    ggml_fp16_t * ثابت wdata     = (ggml_fp16_t *) params->wdata + 0;
    ggml_fp16_t * ثابت wdata_src = wdata + nk;

    برائے (صحیح i1 = ir0; i1 < ir1; i1++) {
        اعشاری * dst_ڈیٹا = (اعشاری *)((حرف *) dst->ڈیٹا + i1*nb1);
        ggml_fp16_t * wdata_kernel = wdata + i1*ne02*ne00;
        برائے (صحیح i10 = 0; i10 < ne10; i10++) {
            ثابت صحیح i1n = i10*ne11;
            برائے (صحیح i00 = 0; i00 < ne00; i00++) {
                اعشاری v = 0;
                ggml_vec_dot_f16(ne02, &v, 0,
                        (ggml_fp16_t *)    wdata_src + i1n, 0,
                        (ggml_fp16_t *) wdata_kernel + i00*ne02, 0, 1);
                dst_ڈیٹا[i10*s0 + i00] += v;
            }
        }
    }
}

ساکن خالی ggml_compute_forward_conv_transpose_1d_f32(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ثنائی_OP_LOCALS

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح nk = ne00*ne01*ne02;

    GGML_ASSERT(nb00 == حجم(اعشاری));
    GGML_ASSERT(nb10 == حجم(اعشاری));

    اگر (ith == 0) {
        memset(params->wdata, 0, params->wsize);

        // prepare kernel data (src0) from (K x Cout x Cin) to (Cin x K x Cout)
        {
            اعشاری * ثابت wdata = (اعشاری *) params->wdata + 0;

            برائے (int64_t i02 = 0; i02 < ne02; i02++) {
                برائے (int64_t i01 = 0; i01 < ne01; i01++) {
                    ثابت اعشاری * ثابت src = (اعشاری *)((حرف *) src0->ڈیٹا + i02*nb02 + i01*nb01);
                    اعشاری * dst_ڈیٹا = wdata + i01*ne00*ne02;
                    برائے (int64_t i00 = 0; i00 < ne00; i00++) {
                        dst_ڈیٹا[i00*ne02 + i02] = src[i00];
                    }
                }
            }
        }

        // prepare source data (src1)
        {
            اعشاری * ثابت wdata = (اعشاری *) params->wdata + nk;
            اعشاری * dst_ڈیٹا = wdata;

            برائے (int64_t i11 = 0; i11 < ne11; i11++) {
                ثابت اعشاری * ثابت src = (اعشاری *)((حرف *) src1->ڈیٹا + i11*nb11);
                برائے (int64_t i10 = 0; i10 < ne10; i10++) {
                    dst_ڈیٹا[i10*ne11 + i11] = src[i10];
                }
            }
        }

        // need to zero dst since we are accumulating into it
        memset(dst->ڈیٹا, 0, ggml_nbytes(dst));
    }
    ggml_barrier(params->threadpool);

    ثابت int32_t s0 = ((ثابت int32_t*)(dst->op_params))[0];

    // total rows in dst
    ثابت صحیح nr = ne1;

    // rows per thread
    ثابت صحیح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحیح ir0 = dr*ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    اعشاری * ثابت wdata     = (اعشاری *) params->wdata + 0;
    اعشاری * ثابت wdata_src = wdata + nk;

    برائے (صحیح i1 = ir0; i1 < ir1; i1++) {
        اعشاری * dst_ڈیٹا = (اعشاری *)((حرف *) dst->ڈیٹا + i1*nb1);
        اعشاری * wdata_kernel = wdata + i1*ne02*ne00;
        برائے (صحیح i10 = 0; i10 < ne10; i10++) {
            ثابت صحیح i1n = i10*ne11;
            برائے (صحیح i00 = 0; i00 < ne00; i00++) {
                اعشاری v = 0;
                ggml_vec_dot_f32(ne02, &v, 0,
                        wdata_src + i1n, 0,
                        wdata_kernel + i00*ne02, 0, 1);
                dst_ڈیٹا[i10*s0 + i00] += v;
            }
        }
    }
}

خالی ggml_compute_forward_conv_transpose_1d(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F16:
            {
                ggml_compute_forward_conv_transpose_1d_f16_f32(params, dst);
            } توڑو;
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_conv_transpose_1d_f32(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_im2col_f32
// src0: kernel [OC, IC, KH, KW]
// src1: image [N, IC, IH, IW]
// dst:  result [N, OH, OW, IC*KH*KW]
ساکن خالی ggml_compute_forward_im2col_f32(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ثنائی_OP_LOCALS;

    ثابت int32_t s0 = ((ثابت int32_t *)(dst->op_params))[0];
    ثابت int32_t s1 = ((ثابت int32_t *)(dst->op_params))[1];
    ثابت int32_t p0 = ((ثابت int32_t *)(dst->op_params))[2];
    ثابت int32_t p1 = ((ثابت int32_t *)(dst->op_params))[3];
    ثابت int32_t d0 = ((ثابت int32_t *)(dst->op_params))[4];
    ثابت int32_t d1 = ((ثابت int32_t *)(dst->op_params))[5];
    ثابت منطقی is_2D = ((ثابت int32_t *)(dst->op_params))[6] == 1;

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت int64_t N  = is_2D ? ne13 : ne12;
    ثابت int64_t IC = is_2D ? ne12 : ne11;
    ثابت int64_t IH = is_2D ? ne11 : 1;
    ثابت int64_t IW = ne10;

    ثابت int64_t KH = is_2D ? ne01 : 1;
    ثابت int64_t KW = ne00;

    ثابت int64_t OH = is_2D ? ne2 : 1;
    ثابت int64_t OW = ne1;

    صحیح ofs0 = is_2D ? nb13 : nb12;
    صحیح ofs1 = is_2D ? nb12 : nb11;

    GGML_ASSERT(nb10 == حجم(اعشاری));

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    {
        اعشاری * ثابت wdata = (اعشاری *) dst->ڈیٹا;

        برائے (int64_t اندر = 0; اندر < N; اندر++) {
            برائے (int64_t ioh = 0; ioh < OH; ioh++) { // 1
                برائے (int64_t iow = 0; iow < OW; iow++) {
                    برائے (int64_t iic = ith; iic < IC; iic += nth) {

                        // micro kernel
                        اعشاری * dst_ڈیٹا = wdata + (اندر*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
                        ثابت اعشاری * ثابت src_ڈیٹا = (اعشاری *)((حرف *) src1->ڈیٹا + اندر*ofs0 + iic*ofs1); // [IH, IW]

                        برائے (int64_t ikh = 0; ikh < KH; ikh++) {  // 1
                            برائے (int64_t ikw = 0; ikw < KW; ikw++) {
                                ثابت int64_t iiw = iow*s0 + ikw*d0 - p0;
                                ثابت int64_t iih = ioh*s1 + ikh*d1 - p1;

                                اگر (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                    dst_ڈیٹا[iic*(KH*KW) + ikh*KW + ikw] = 0;
                                } ورنہ {
                                    dst_ڈیٹا[iic*(KH*KW) + ikh*KW + ikw] = (src_ڈیٹا[iih*IW + iiw]);
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
ساکن خالی ggml_compute_forward_im2col_f16(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16);

    GGML_TENSOR_ثنائی_OP_LOCALS;

    ثابت int32_t s0 = ((ثابت int32_t *)(dst->op_params))[0];
    ثابت int32_t s1 = ((ثابت int32_t *)(dst->op_params))[1];
    ثابت int32_t p0 = ((ثابت int32_t *)(dst->op_params))[2];
    ثابت int32_t p1 = ((ثابت int32_t *)(dst->op_params))[3];
    ثابت int32_t d0 = ((ثابت int32_t *)(dst->op_params))[4];
    ثابت int32_t d1 = ((ثابت int32_t *)(dst->op_params))[5];
    ثابت منطقی is_2D = ((ثابت int32_t *)(dst->op_params))[6] == 1;

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت int64_t N  = is_2D ? ne13 : ne12;
    ثابت int64_t IC = is_2D ? ne12 : ne11;
    ثابت int64_t IH = is_2D ? ne11 : 1;
    ثابت int64_t IW = ne10;

    ثابت int64_t KH = is_2D ? ne01 : 1;
    ثابت int64_t KW = ne00;

    ثابت int64_t OH = is_2D ? ne2 : 1;
    ثابت int64_t OW = ne1;

    صحیح ofs0 = is_2D ? nb13 : nb12;
    صحیح ofs1 = is_2D ? nb12 : nb11;

    GGML_ASSERT(nb10 == ggml_type_سائز(src1->type));

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    {
        ggml_fp16_t * ثابت wdata = (ggml_fp16_t *) dst->ڈیٹا;

        برائے (int64_t اندر = 0; اندر < N; اندر++) {
            برائے (int64_t ioh = 0; ioh < OH; ioh++) { // 1
                برائے (int64_t iow = 0; iow < OW; iow++) {
                    برائے (int64_t iic = ith; iic < IC; iic += nth) {

                        // micro kernel
                        ggml_fp16_t * dst_ڈیٹا = wdata + (اندر*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
                        ثابت اعشاری * ثابت src_ڈیٹا_f32 = src1->type == GGML_TYPE_F32
                            ? (ثابت اعشاری *)((ثابت حرف *) src1->ڈیٹا + اندر*ofs0 + iic*ofs1)
                            : خالی_اشاریہ; // [IH, IW]
                        ثابت ggml_fp16_t * ثابت src_ڈیٹا_f16 = src1->type == GGML_TYPE_F16
                            ? (ثابت ggml_fp16_t *)((ثابت حرف *) src1->ڈیٹا + اندر*ofs0 + iic*ofs1)
                            : خالی_اشاریہ; // [IH, IW]

                        برائے (int64_t ikh = 0; ikh < KH; ikh++) {  // 1
                            برائے (int64_t ikw = 0; ikw < KW; ikw++) {
                                ثابت int64_t iiw = iow*s0 + ikw*d0 - p0;
                                ثابت int64_t iih = ioh*s1 + ikh*d1 - p1;

                                اگر (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                    dst_ڈیٹا[iic*(KH*KW) + ikh*KW + ikw] = 0;
                                } ورنہ {
                                    اگر (src_ڈیٹا_f32 != خالی_اشاریہ) {
                                        dst_ڈیٹا[iic*(KH*KW) + ikh*KW + ikw] = GGML_CPU_FP32_مقصد_FP16(src_ڈیٹا_f32[iih*IW + iiw]);
                                    } ورنہ {
                                        dst_ڈیٹا[iic*(KH*KW) + ikh*KW + ikw] = src_ڈیٹا_f16[iih*IW + iiw];
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

خالی ggml_compute_forward_im2col(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {
    انتخاب (dst->type) {
        حالت GGML_TYPE_F16:
            {
                ggml_compute_forward_im2col_f16(params, dst);
            } توڑو;
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_im2col_f32(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_im2col_back_f32

خالی ggml_compute_forward_im2col_عقب_f32(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0]; // gradients of forward pass output
    ثابت ggml_tensor * src1 = dst->src[1]; // convolution kernel

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ثنائی_OP_LOCALS;

    ثابت int32_t s0 = ((ثابت int32_t *)(dst->op_params))[0];
    ثابت int32_t s1 = ((ثابت int32_t *)(dst->op_params))[1];
    ثابت int32_t p0 = ((ثابت int32_t *)(dst->op_params))[2];
    ثابت int32_t p1 = ((ثابت int32_t *)(dst->op_params))[3];
    ثابت int32_t d0 = ((ثابت int32_t *)(dst->op_params))[4];
    ثابت int32_t d1 = ((ثابت int32_t *)(dst->op_params))[5];
    ثابت منطقی is_2D = ((ثابت int32_t *)(dst->op_params))[6] == 1;

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت int64_t N  = is_2D ? ne3 : ne2;
    ثابت int64_t IC = is_2D ? ne2 : ne1;
    ثابت int64_t IH = is_2D ? ne1 : 1;
    ثابت int64_t IW = ne0;

    ثابت int64_t KH = is_2D ? ne11 : 1;
    ثابت int64_t KW = ne10;

    ثابت int64_t OH = is_2D ? ne02 : 1;
    ثابت int64_t OW = ne01;

    صحیح ofs0 = is_2D ? nb3 : nb2;
    صحیح ofs1 = is_2D ? nb2 : nb1;

    GGML_ASSERT(nb0  == حجم(اعشاری));

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    {
        اعشاری * ثابت wdata = (اعشاری *) dst->ڈیٹا;

        برائے (int64_t اندر = 0; اندر < N; اندر++) {
            برائے (int64_t iic = ith; iic < IC; iic += nth) {
                برائے (int64_t iih = 0; iih < IH; iih++) {
                    برائے (int64_t iiw = 0; iiw < IW; iiw++) {

                        // micro kernel
                        اعشاری grad = 0.0f;
                        برائے (int64_t ikh = 0; ikh < KH; ikh++) {
                            برائے (int64_t ikw = 0; ikw < KW; ikw++) {
                                // For s0 > 1 some values were skipped over in the forward pass.
                                // These values have tmpw % s0 != 0 and need to be skipped in the backwards pass as well.
                                ثابت int64_t tmpw = (iiw + p0 - ikw*d0);
                                اگر (tmpw % s0 != 0) {
                                    جاری;
                                }
                                ثابت int64_t iow = tmpw / s0;

                                // Equivalent logic as above except for s1.
                                int64_t ioh;
                                اگر (is_2D) {
                                    ثابت int64_t tmph = iih + p1 - ikh*d1;

                                    اگر (tmph % s1 != 0) {
                                        جاری;
                                    }

                                    ioh = tmph / s1;
                                } ورنہ {
                                    ioh = 0;
                                }

                                اگر (iow < 0 || iow >= OW || ioh < 0 || ioh >= OH) {
                                    جاری;
                                }

                                ثابت اعشاری * ثابت grad_اندر = (ثابت اعشاری *) src0->ڈیٹا
                                    + (اندر*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
                                grad += grad_اندر[iic*(KH*KW) + ikh*KW + ikw];
                            }
                        }
                        اعشاری * dst_ڈیٹا = (اعشاری *)((حرف *) wdata + (اندر*ofs0 + iic*ofs1)); // [IH, IW]
                        dst_ڈیٹا[iih*IW + iiw] = grad;
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
ساکن خالی ggml_compute_forward_im2col_3d_f16(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16);

    GGML_TENSOR_ثنائی_OP_LOCALS;

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


    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت int64_t N  = ne13 / IC;
    ثابت int64_t ID = ne12;
    ثابت int64_t IH = ne11;
    ثابت int64_t IW = ne10;

    ثابت int64_t OC = ne03 / IC;
    GGML_غیر_مستعمل(OC);
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

    GGML_ASSERT(nb10 == حجم(اعشاری));

    // im2col: [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
    {
        ggml_fp16_t * ثابت wdata = (ggml_fp16_t *) dst->ڈیٹا;

        برائے (int64_t اندر = 0; اندر < N; اندر++) {
            برائے (int64_t iod = 0; iod < OD; iod++) {
                برائے (int64_t ioh = 0; ioh < OH; ioh++) {
                    برائے (int64_t iow = 0; iow < OW; iow++) {
                        برائے (int64_t iic = ith; iic < IC; iic += nth) {

                            // micro kernel
                            ggml_fp16_t * dst_ڈیٹا = wdata + (اندر*OD*OH_OW + iod*OH_OW + ioh*OW + iow)*IC_KD_KH_KW; // [IC, KD, KH, KW]
                            ثابت اعشاری * ثابت src_ڈیٹا = (ثابت اعشاری *) ((ثابت حرف *)src1->ڈیٹا + (اندر*IC + iic)*nb13); // [ID, IH, IW]

                            برائے (int64_t ikd = 0; ikd < KD; ikd++) {
                                برائے (int64_t ikh = 0; ikh < KH; ikh++) {
                                    برائے (int64_t ikw = 0; ikw < KW; ikw++) {
                                        ثابت int64_t iiw = iow*s0 + ikw*d0 - p0;
                                        ثابت int64_t iih = ioh*s1 + ikh*d1 - p1;
                                        ثابت int64_t iid = iod*s2 + ikd*d2 - p2;

                                        اگر (iid < 0 || iid >= ID || iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                            dst_ڈیٹا[iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw] = 0;
                                        } ورنہ {
                                            ثابت اعشاری * ثابت s = (ثابت اعشاری *) ((ثابت حرف *)src_ڈیٹا + iid*nb12 + iih*nb11 + iiw*nb10); // [ID, IH, IW]
                                            dst_ڈیٹا[iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw] = GGML_CPU_FP32_مقصد_FP16(*s);
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
ساکن خالی ggml_compute_forward_im2col_3d_f32(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ثنائی_OP_LOCALS;

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


    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت int64_t N  = ne13 / IC;
    ثابت int64_t ID = ne12;
    ثابت int64_t IH = ne11;
    ثابت int64_t IW = ne10;

    ثابت int64_t OC = ne03 / IC;
    GGML_غیر_مستعمل(OC);
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

    GGML_ASSERT(nb10 == حجم(اعشاری));

    // im2col: [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
    {
        اعشاری * ثابت wdata = (اعشاری *) dst->ڈیٹا;

        برائے (int64_t اندر = 0; اندر < N; اندر++) {
            برائے (int64_t iod = 0; iod < OD; iod++) {
                برائے (int64_t ioh = 0; ioh < OH; ioh++) {
                    برائے (int64_t iow = 0; iow < OW; iow++) {
                        برائے (int64_t iic = ith; iic < IC; iic += nth) {

                            // micro kernel
                            اعشاری * dst_ڈیٹا = wdata + (اندر*OD*OH_OW + iod*OH_OW + ioh*OW + iow)*IC_KD_KH_KW; // [IC, KD, KH, KW]
                            ثابت اعشاری * ثابت src_ڈیٹا = (ثابت اعشاری *) ((ثابت حرف *)src1->ڈیٹا + (اندر*IC + iic)*nb13); // [ID, IH, IW]

                            برائے (int64_t ikd = 0; ikd < KD; ikd++) {
                                برائے (int64_t ikh = 0; ikh < KH; ikh++) {
                                    برائے (int64_t ikw = 0; ikw < KW; ikw++) {
                                        ثابت int64_t iiw = iow*s0 + ikw*d0 - p0;
                                        ثابت int64_t iih = ioh*s1 + ikh*d1 - p1;
                                        ثابت int64_t iid = iod*s2 + ikd*d2 - p2;

                                        اگر (iid < 0 || iid >= ID || iih < 0 || iih >= IH || iiw < 0 || iiw >= IW || iid < 0 || iid >= ID) {
                                            dst_ڈیٹا[iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw] = 0;
                                        } ورنہ {
                                            ثابت اعشاری * ثابت s = (ثابت اعشاری *) ((ثابت حرف *)src_ڈیٹا + iid*nb12 + iih*nb11 + iiw*nb10); // [ID, IH, IW]
                                            dst_ڈیٹا[iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw] = *s;
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


خالی ggml_compute_forward_im2col_3d(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {
    انتخاب (dst->type) {
        حالت GGML_TYPE_F16:
            {
                ggml_compute_forward_im2col_3d_f16(params, dst);
            } توڑو;
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_im2col_3d_f32(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ساکن خالی ggml_call_mul_mat(ggml_type type, ثابت ggml_compute_params * params, int64_t m, int64_t n, int64_t k,
                              خالی * a, خالی * b, اعشاری * c) {
    ثابت ggml_type_traits * traits = ggml_get_type_traits(type);
    ساخت ggml_tensor src1 = {};
    src1.type  = type;
    src1.ne[0] = k;
    src1.ne[1] = m;
    src1.ne[2] = 1;
    src1.ne[3] = 1;
    src1.nb[0] = traits->type_سائز;
    src1.nb[1] = k * traits->type_سائز;
    src1.nb[2] = src1.nb[1];
    src1.nb[3] = src1.nb[2];
    src1.ڈیٹا  = a;

    ساخت ggml_tensor src0 = {};
    src0.type  = type;
    src0.ne[0] = k;
    src0.ne[1] = n;
    src0.ne[2] = 1;
    src0.ne[3] = 1;
    src0.nb[0] = traits->type_سائز;
    src0.nb[1] = k * traits->type_سائز;
    src0.nb[2] = src0.nb[1];
    src0.nb[3] = src0.nb[2];
    src0.ڈیٹا  = b;

    ساخت ggml_tensor dst = {};
    dst.ne[0] = n;
    dst.ne[1] = m;
    dst.ne[2] = 1;
    dst.ne[3] = 1;
    dst.nb[0] = حجم(اعشاری);
    dst.nb[1] = n * حجم(اعشاری);
    dst.nb[2] = dst.nb[1];
    dst.nb[3] = dst.nb[2];
    dst.ڈیٹا  = c;
    dst.src[0] = &src0;
    dst.src[1] = &src1;

    ggml_compute_forward_mul_mat(params, &dst);
}

ساکن اندرونی int64_t ggml_wrap_around(int64_t coord, int64_t size) {
    واپس (coord  + size) % size; // adding size avoids negative number weirdness
}

// ggml_compute_forward_col2im_1d
//
// Scatter-add columns [K*OC, T_in] -> signal [T_out, OC]
// where T_out = (T_in - 1)*s + K - 2*p.  Gather approach: each output reads ceil(K/s) inputs.
// Parallelized over the time axis so the split stays balanced whatever OC is.
// Supports F32, F16, BF16 input/output (same type), F32 accumulator.

سانچہ <قسم_نام elem_t>
ساکن خالی ggml_compute_forward_col2im_1d_impl(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src = dst->src[0];  // [K*OC, T_in]

    GGML_ASSERT(ggml_is_contiguous(src));
    GGML_ASSERT(ggml_is_contiguous(dst));

    ثابت int32_t s0 = ((ثابت int32_t *)(dst->op_params))[0];
    ثابت int32_t OC = ((ثابت int32_t *)(dst->op_params))[1];
    ثابت int32_t p0 = ((ثابت int32_t *)(dst->op_params))[2];

    ثابت int64_t K_OC = src->ne[0];
    ثابت int64_t T_اندر = src->ne[1];
    ثابت int64_t K    = K_OC / OC;
    ثابت int64_t T_برآمد = dst->ne[0];

    ثابت elem_t * ستون_ڈیٹا = (ثابت elem_t *) src->ڈیٹا;
    elem_t       * dst_ڈیٹا = (elem_t *) dst->ڈیٹا;

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    // Parallelize over the time axis: the split stays balanced whatever OC is,
    // down to OC = 1 for mono audio, and threads read disjoint column bands
    ثابت int64_t dr = (T_برآمد + nth - 1) / nth;
    ثابت int64_t it0 = dr * ith;
    ثابت int64_t it1 = it0 + dr < T_برآمد ? it0 + dr : T_برآمد;

    برائے (int64_t oc = 0; oc < OC; oc++) {
        برائے (int64_t t_برآمد = it0; t_برآمد < it1; t_برآمد++) {
            ثابت int64_t t_abs = t_برآمد + p0;  // absolute position in uncropped signal
            // Gather: find all (t_in, k) where t_in * s + k == t_abs, 0 <= k < K
            int64_t t_اندر_min = (t_abs - K + 1 + s0 - 1) / s0;  // ceil((t_abs-K+1)/s)
            اگر (t_اندر_min < 0) t_اندر_min = 0;
            int64_t t_اندر_max = t_abs / s0;
            اگر (t_اندر_max >= T_اندر) t_اندر_max = T_اندر - 1;

            اعشاری مجموع = 0.0f;
            برائے (int64_t t_اندر = t_اندر_min; t_اندر <= t_اندر_max; t_اندر++) {
                int64_t k = t_abs - t_اندر * s0;
                اگر (k >= 0 && k < K) {
                    // col layout: [K*OC, T_in], element (oc*K+k, t_in)
                    مجموع += type_conversion_جدول<elem_t>::مقصد_f32(ستون_ڈیٹا[(oc * K + k) + t_اندر * K_OC]);
                }
            }
            // dst layout: [T_out, OC], element (t_out, oc)
            dst_ڈیٹا[t_برآمد + oc * T_برآمد] = type_conversion_جدول<elem_t>::from_f32(مجموع);
        }
    }
}

خالی ggml_compute_forward_col2im_1d(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    انتخاب (dst->src[0]->type) {
        حالت GGML_TYPE_F32:  ggml_compute_forward_col2im_1d_impl<اعشاری>      (params, dst); توڑو;
        حالت GGML_TYPE_F16:  ggml_compute_forward_col2im_1d_impl<ggml_fp16_t>(params, dst); توڑو;
        حالت GGML_TYPE_BF16: ggml_compute_forward_col2im_1d_impl<ggml_bf16_t>(params, dst); توڑو;
        طےشدہ: GGML_ABORT("col2im_1d: unsupported type %d", dst->src[0]->type);
    }
}

// ggml_compute_forward_conv_2d


ساکن خالی ggml_compute_forward_conv_2d_impl(ثابت ggml_compute_params * params,
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

    ثابت int64_t c_اندر  = src->ne[2];
    ثابت int64_t c_برآمد = kernel->ne[3];
    GGML_ASSERT(c_اندر == kernel->ne[2]);

    ثابت int64_t src_w = src->ne[0];
    ثابت int64_t src_h = src->ne[1];
    ثابت int64_t knl_w = kernel->ne[0];
    ثابت int64_t knl_h = kernel->ne[1];
    ثابت int64_t dst_w = dst->ne[0];
    ثابت int64_t dst_h = dst->ne[1];

    ثابت اعشاری * src_ڈیٹا = (اعشاری *) src->ڈیٹا;
    خالی  * knl_ڈیٹا       = kernel->ڈیٹا;
    اعشاری * dst_ڈیٹا       = (اعشاری *) dst->ڈیٹا;

    ثابت int64_t knl_n           = knl_w * knl_h * c_اندر;
    ثابت int64_t patch_کل     = dst->ne[3] * dst_w * dst_h;

    ثابت int64_t space_فی_patch   = knl_n * traits->type_سائز + c_برآمد * حجم(اعشاری);
    ثابت int64_t batch_سائز        = params->wsize / space_فی_patch;
    ثابت int64_t patches_فی_batch = batch_سائز > 8 ? (batch_سائز / 8) * 8 : batch_سائز;
    ثابت int64_t batch_n           = (patch_کل + patches_فی_batch - 1) / patches_فی_batch;

    GGML_ASSERT(patches_فی_batch > 0 && batch_سائز >= 1);

    خالی * tmp = params->wdata;

    برائے (int64_t batch_i = 0; batch_i < batch_n; ++batch_i) {

        ثابت int64_t patch_آغاز_batch = batch_i * patches_فی_batch;
        ثابت int64_t patch_انتہا_batch   = std::min(patch_آغاز_batch + patches_فی_batch,
                                              patch_کل);
        ثابت int64_t patch_n           = patch_انتہا_batch - patch_آغاز_batch;

        ثابت int64_t patch_فی_thread  = (patch_n + params->nth - 1) / params->nth;
        ثابت int64_t patch_آغاز       = patch_آغاز_batch + params->ith * patch_فی_thread;
        ثابت int64_t patch_انتہا         = std::min(patch_آغاز + patch_فی_thread, patch_انتہا_batch);

        //im2col for a patch
        برائے (int64_t p = patch_آغاز; p < patch_انتہا; ++p) {
            ثابت int64_t  batch_n     =  p / (dst_w * dst_h);
            ثابت int64_t  src_x       = (p / dst_w) % dst_h;
            ثابت int64_t  src_y       =  p % dst_w;

            ثابت اعشاری * src_بنیاد = (ثابت اعشاری *)((ثابت حرف *)src_ڈیٹا + batch_n * src->nb[3]);
            حرف *        dst_صف  = (حرف *) tmp + (p % patches_فی_batch) * knl_n * traits->type_سائز;

            برائے (int64_t ic = 0; ic < c_اندر; ++ic) {
                برائے (int64_t ky = 0; ky < knl_h; ++ky) {
                    برائے (int64_t kx = 0; kx < knl_w; ++kx) {
                        ثابت int64_t sy = src_x * stride_y + ky * dilation_y - pad_y;
                        ثابت int64_t sx = src_y * stride_x + kx * dilation_x - pad_x;

                        int64_t dst_idx = ic * (knl_h * knl_w) + ky * knl_w + kx;

                        اعشاری src_val;
                        اگر (sy < 0 || sy >= src_h || sx < 0 || sx >= src_w) {
                            src_val = 0.0f;
                        } ورنہ {
                            ثابت اعشاری * src_ptr = (ثابت اعشاری *)((ثابت حرف *)src_بنیاد + sx * src->nb[0] + sy * src->nb[1] + ic * src->nb[2]);
                            src_val               = *src_ptr;
                        }

                        حرف * element_ptr = dst_صف + dst_idx * traits->type_سائز;
                        اگر (kernel_type == GGML_TYPE_F32) {
                            *(اعشاری *) element_ptr = src_val;
                        } ورنہ اگر (kernel_type == GGML_TYPE_F16) {
                            *(ggml_fp16_t *) element_ptr = GGML_CPU_FP32_مقصد_FP16(src_val);
                        }
                    }
                }
            }
        }   // patches handled by this thread

        ggml_barrier(params->threadpool);

        اعشاری * gemm_output = (اعشاری *) ((حرف *) tmp + patches_فی_batch * knl_n * traits->type_سائز);

        GGML_ASSERT(gemm_output + patch_n * c_برآمد <= (اعشاری*)tmp + params->wsize);

        // GEMM: patches[patch_n, knl_n] × kernel[knl_n, c_out] = output[patch_n, c_out]
        ggml_call_mul_mat(kernel_type, params, patch_n, c_برآمد, knl_n, tmp, knl_ڈیٹا, gemm_output);

        ggml_barrier(params->threadpool);


        //permute back [OC, N, OH, OW] to [N, OC, OH, OW]
        ثابت int64_t ترتیب_بدلو_فی_thread = (patch_n + params->nth - 1) / params->nth;
        ثابت int64_t ترتیب_بدلو_آغاز = params->ith * ترتیب_بدلو_فی_thread;
        ثابت int64_t ترتیب_بدلو_انتہا = std::min(ترتیب_بدلو_آغاز + ترتیب_بدلو_فی_thread, patch_n);

        برائے (int64_t i = ترتیب_بدلو_آغاز; i < ترتیب_بدلو_انتہا; ++i) {
            ثابت int64_t p       = patch_آغاز_batch + i;
            ثابت int64_t batch_n = p / (dst_w * dst_h);
            ثابت int64_t dst_y   = (p / dst_w) % dst_h;
            ثابت int64_t dst_x   = p % dst_w;

            برائے (int64_t oc = 0; oc < c_برآمد; ++oc) {
                ثابت اعشاری value = gemm_output[i * c_برآمد + oc];
                اعشاری * dst_ptr = (اعشاری *)((حرف *)dst_ڈیٹا + dst_x * dst->nb[0] + dst_y * dst->nb[1] + oc * dst->nb[2] + batch_n * dst->nb[3]);
                *dst_ptr = value;
            }
        }
    }
}

خالی ggml_compute_forward_conv_2d(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    ggml_compute_forward_conv_2d_impl(params, src0, src1, dst, src0->type);
}

// ggml_compute_forward_conv_3d

ساکن خالی ggml_compute_forward_conv_3d_impl(ثابت ggml_compute_params * params,
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

    ثابت اعشاری * src_ڈیٹا = (اعشاری *) src->ڈیٹا;
    خالی  * knl_ڈیٹا       = kernel->ڈیٹا;
    اعشاری * dst_ڈیٹا       = (اعشاری *) dst->ڈیٹا;

    ثابت int64_t knl_n_فی_channel = knl_w * knl_h * knl_d;
    ثابت int64_t knl_n_کل       = knl_n_فی_channel * c;
    ثابت int64_t patch_کل       = n * dst_w * dst_h * dst_d;

    ثابت int64_t space_فی_patch   = knl_n_کل * traits->type_سائز + oc * حجم(اعشاری);
    ثابت int64_t batch_سائز        = params->wsize / space_فی_patch;
    ثابت int64_t patches_فی_batch = batch_سائز > 8 ? (batch_سائز / 8) * 8 : batch_سائز;
    ثابت int64_t batch_n           = (patch_کل + patches_فی_batch - 1) / patches_فی_batch;

    GGML_ASSERT(patches_فی_batch > 0 && batch_سائز >= 1);

    خالی * tmp = params->wdata;

    برائے (int64_t batch_i = 0; batch_i < batch_n; ++batch_i) {
        ثابت int64_t patch_آغاز_batch = batch_i * patches_فی_batch;
        ثابت int64_t patch_انتہا_batch   = std::min(patch_آغاز_batch + patches_فی_batch, patch_کل);
        ثابت int64_t patch_n_اندر_batch  = patch_انتہا_batch - patch_آغاز_batch;

        ثابت int64_t patch_فی_thread  = (patch_n_اندر_batch + params->nth - 1) / params->nth;
        ثابت int64_t patch_آغاز       = patch_آغاز_batch + params->ith * patch_فی_thread;
        ثابت int64_t patch_انتہا         = std::min(patch_آغاز + patch_فی_thread, patch_انتہا_batch);

        برائے (int64_t p = patch_آغاز; p < patch_انتہا; ++p) {
            ثابت int64_t p_اندر_batch = p % (dst_w * dst_h * dst_d);
            ثابت int64_t p_اندر_گہرائی = p_اندر_batch % (dst_w * dst_h);
            ثابت int64_t batch_idx  = p / (dst_w * dst_h * dst_d);
            ثابت int64_t dst_z      = p_اندر_batch / (dst_w * dst_h);
            ثابت int64_t dst_y      = p_اندر_گہرائی / dst_w;
            ثابت int64_t dst_x      = p_اندر_گہرائی % dst_w;

            حرف * dst_صف = (حرف *) tmp + (p % patches_فی_batch) * knl_n_کل * traits->type_سائز;

            برائے (int64_t ic = 0; ic < c; ++ic) {
                برائے (int64_t kz = 0; kz < knl_d; ++kz) {
                    برائے (int64_t ky = 0; ky < knl_h; ++ky) {
                        برائے (int64_t kx = 0; kx < knl_w; ++kx) {
                            ثابت int64_t sz = dst_z * s2 + kz * d2 - p2;
                            ثابت int64_t sy = dst_y * s1 + ky * d1 - p1;
                            ثابت int64_t sx = dst_x * s0 + kx * d0 - p0;

                            int64_t dst_idx = ic * knl_n_فی_channel + kz * (knl_h * knl_w) + ky * knl_w + kx;

                            اعشاری src_val;
                            اگر (sz < 0 || sz >= src_d || sy < 0 || sy >= src_h || sx < 0 || sx >= src_w) {
                                src_val = 0.0f;
                            } ورنہ {
                                ثابت int64_t cn_idx = batch_idx * c + ic;
                                ثابت اعشاری * src_ptr = (ثابت اعشاری *)((ثابت حرف *)src_ڈیٹا + sx*src->nb[0] + sy*src->nb[1] + sz*src->nb[2] + cn_idx*src->nb[3]);
                                src_val = *src_ptr;
                            }

                            حرف * element_ptr = dst_صف + dst_idx * traits->type_سائز;
                            اگر (kernel_type == GGML_TYPE_F32) {
                                *(اعشاری *)element_ptr = src_val;
                            } ورنہ اگر (kernel_type == GGML_TYPE_F16) {
                                *(ggml_fp16_t *)element_ptr = GGML_CPU_FP32_مقصد_FP16(src_val);
                            }
                        }
                    }
                }
            }
        }

        ggml_barrier(params->threadpool);

        اعشاری * gemm_output = (اعشاری *) ((حرف *) tmp + patches_فی_batch * knl_n_کل * traits->type_سائز);
        ggml_call_mul_mat(kernel_type, params, patch_n_اندر_batch, oc, knl_n_کل, tmp, knl_ڈیٹا, gemm_output);

        ggml_barrier(params->threadpool);

        ثابت int64_t ترتیب_بدلو_فی_thread = (patch_n_اندر_batch + params->nth - 1) / params->nth;
        ثابت int64_t ترتیب_بدلو_آغاز = params->ith * ترتیب_بدلو_فی_thread;
        ثابت int64_t ترتیب_بدلو_انتہا = std::min(ترتیب_بدلو_آغاز + ترتیب_بدلو_فی_thread, patch_n_اندر_batch);

        برائے (int64_t i = ترتیب_بدلو_آغاز; i < ترتیب_بدلو_انتہا; ++i) {
            ثابت int64_t p = patch_آغاز_batch + i;
            ثابت int64_t p_اندر_batch = p % (dst_w * dst_h * dst_d);
            ثابت int64_t p_اندر_گہرائی = p_اندر_batch % (dst_w * dst_h);
            ثابت int64_t batch_idx  = p / (dst_w * dst_h * dst_d);
            ثابت int64_t dst_z      = p_اندر_batch / (dst_w * dst_h);
            ثابت int64_t dst_y      = p_اندر_گہرائی / dst_w;
            ثابت int64_t dst_x      = p_اندر_گہرائی % dst_w;

            برائے (int64_t ioc = 0; ioc < oc; ++ioc) {
                ثابت اعشاری value = gemm_output[i * oc + ioc];
                ثابت int64_t ocn_idx = batch_idx * oc + ioc;
                اعشاری * dst_ptr = (اعشاری *)((حرف *)dst_ڈیٹا + dst_x*dst->nb[0] + dst_y*dst->nb[1] + dst_z*dst->nb[2] + ocn_idx*dst->nb[3]);
                *dst_ptr = value;
            }
        }
    }
}

خالی ggml_compute_forward_conv_3d(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    ggml_compute_forward_conv_3d_impl(params, src0, src1, dst, src0->type);
}

سانچہ <قسم_نام kernel_t>
ساکن خالی ggml_compute_forward_conv_transpose_2d_impl(
    ثابت ggml_compute_params * params,
          ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ثنائی_OP_LOCALS

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح nk = ne00*ne01*ne02*ne03;

    GGML_ASSERT(nb00 == ggml_type_سائز(src0->type));
    GGML_ASSERT(nb10 == حجم(اعشاری));

    اگر (ith == 0) {
        memset(params->wdata, 0, params->wsize);

        // permute kernel data (src0) from (Kw x Kh x Cout x Cin) to (Cin x Kw x Kh x Cout)
        {
            kernel_t * ثابت wdata = (kernel_t *) params->wdata + 0;

            برائے (int64_t i03 = 0; i03 < ne03; i03++) {
                برائے (int64_t i02 = 0; i02 < ne02; i02++) {
                    ثابت kernel_t * ثابت src = (kernel_t *)((حرف *) src0->ڈیٹا + i03*nb03 + i02*nb02);
                    kernel_t * dst_ڈیٹا = wdata + i02*ne01*ne00*ne03;
                    برائے (int64_t i01 = 0; i01 < ne01; i01++) {
                        برائے (int64_t i00 = 0; i00 < ne00; i00++) {
                            dst_ڈیٹا[i01*ne00*ne03 + i00*ne03 + i03] = src[i01 * ne00 + i00];
                        }
                    }
                }
            }
        }

        // permute source data (src1) from (Sw x Sh x Cin) to (Cin x Sw x Sh), for all batches
        {
            kernel_t * ثابت wdata = (kernel_t *) params->wdata + nk;
            برائے (صحیح i13 = 0; i13 < ne13; i13++) {
                kernel_t * ثابت wdata_b = wdata + i13*ne10*ne11*ne12;
                برائے (صحیح i12 = 0; i12 < ne12; i12++) {
                    برائے (صحیح i11 = 0; i11 < ne11; i11++) {
                        ثابت اعشاری * ثابت src = (اعشاری *)((حرف *) src1->ڈیٹا + i13*nb13 + i12*nb12 + i11*nb11);
                        kernel_t * dst_ڈیٹا = wdata_b + i11*ne10*ne12;
                        برائے (صحیح i10 = 0; i10 < ne10; i10++) {
                            اگر constexpr (std::is_same_v<kernel_t, ggml_fp16_t>) {
                                dst_ڈیٹا[i10*ne12 + i12] = GGML_CPU_FP32_مقصد_FP16(src[i10]);
                            } ورنہ {
                                dst_ڈیٹا[i10*ne12 + i12] = src[i10];
                            }
                        }
                    }
                }
            }
        }

        memset(dst->ڈیٹا, 0, ggml_nbytes(dst));
    }
    ggml_barrier(params->threadpool);

    ثابت int32_t stride = ggml_get_op_params_i32(dst, 0);

    // total patches in dst
    ثابت صحیح np = ne2;

    // patches per thread
    ثابت صحیح dp = (np + nth - 1)/nth;

    // patch range for this thread
    ثابت صحیح ip0 = dp*ith;
    ثابت صحیح ip1 = MIN(ip0 + dp, np);

    kernel_t * ثابت wdata = (kernel_t *) params->wdata + 0;
    kernel_t * ثابت wdata_src = wdata + nk;

    برائے (صحیح i2 = ip0; i2 < ip1; i2++) { // Cout
        kernel_t * wdata_kernel = wdata + i2*ne01*ne00*ne03;
        برائے (صحیح i3 = 0; i3 < ne3; i3++) { // batch
            اعشاری * dst_ڈیٹا = (اعشاری *)((حرف *) dst->ڈیٹا + i3*nb3 + i2*nb2);
            kernel_t * wdata_src_b = wdata_src + i3*ne10*ne11*ne12;
            برائے (صحیح i11 = 0; i11 < ne11; i11++) {
                برائے (صحیح i10 = 0; i10 < ne10; i10++) {
                    ثابت صحیح i1n = i11*ne10*ne12 + i10*ne12;
                    برائے (صحیح i01 = 0; i01 < ne01; i01++) {
                        برائے (صحیح i00 = 0; i00 < ne00; i00++) {
                            اعشاری v = 0;
                            اگر constexpr (std::is_same_v<kernel_t, ggml_fp16_t>) {
                                ggml_vec_dot_f16(ne03, &v, 0,
                                        wdata_src_b + i1n, 0,
                                        wdata_kernel + i01*ne00*ne03 + i00*ne03, 0, 1);
                            } ورنہ {
                                ggml_vec_dot_f32(ne03, &v, 0,
                                        wdata_src_b + i1n, 0,
                                        wdata_kernel + i01*ne00*ne03 + i00*ne03, 0, 1);
                            }
                            dst_ڈیٹا[(i11*stride + i01)*ne0 + i10*stride + i00] += v;
                        }
                    }
                }
            }
        }
    }
}

خالی ggml_compute_forward_conv_transpose_2d(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F16:
            {
                ggml_compute_forward_conv_transpose_2d_impl<ggml_fp16_t>(params, dst);
            } توڑو;
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_conv_transpose_2d_impl<اعشاری>(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_conv_2d_dw

ساخت ggml_conv_2d_dw_params {
    int64_t channels;
    int64_t batch;
    int64_t src_w;
    int64_t src_h;
    int64_t dst_w;
    int64_t dst_h;
    int64_t knl_w;
    int64_t knl_h;
    صحیح stride_x;
    صحیح stride_y;
    صحیح pad_x;
    صحیح pad_y;
    صحیح dilation_x;
    صحیح dilation_y;
};

ساکن اندرونی اعشاری ggml_conv_2d_dw_knl_f32(ثابت حرف * ڈیٹا, int64_t i, ggml_type type) {
    اگر (type == GGML_TYPE_F16) {
        واپس GGML_FP16_مقصد_FP32(((ثابت ggml_fp16_t *)ڈیٹا)[i]);
    }
    واپس ((ثابت اعشاری *)ڈیٹا)[i];
}

ساکن خالی ggml_compute_forward_conv_2d_dw_cwhn(
        ثابت ggml_compute_params * params,
        ثابت ggml_tensor * src,
        ثابت ggml_tensor * kernel,
        ggml_tensor * dst,
        ثابت ggml_conv_2d_dw_params & p) {

    ثابت int64_t c = p.channels;
    ثابت حرف * knl_ڈیٹا = (ثابت حرف *)kernel->ڈیٹا;
    ثابت ggml_type knl_type = kernel->type;

    ثابت int64_t صفیں_کل = p.dst_h * p.batch;
    ثابت int64_t صفیں_فی_thread = (صفیں_کل + params->nth - 1) / params->nth;
    ثابت int64_t صف_آغاز = params->ith * صفیں_فی_thread;
    ثابت int64_t صف_انتہا = MIN(صف_آغاز + صفیں_فی_thread, صفیں_کل);

#اگر_تعریف_شدہ GGML_SIMD
    int64_t c_pkg_انتہا = 0;
    int64_t pkg_سائز = GGML_F32_EPR;
    اگر (knl_type == GGML_TYPE_F32) {
    #اگر defined(__ARM_FEATURE_SVE)
        pkg_سائز = svcntw();
    #ورنہ
        pkg_سائز = GGML_F32_EPR;
    #اگر_ختم
        c_pkg_انتہا = (c / pkg_سائز) * pkg_سائز;
    }
#ورنہ
    ثابت int64_t c_pkg_انتہا = 0;
#اگر_ختم

    برائے (int64_t صف = صف_آغاز; صف < صف_انتہا; ++صف) {
        ثابت int64_t dst_y = صف % p.dst_h;
        ثابت اعشاری * src_ڈیٹا = (ثابت اعشاری *)src->ڈیٹا + (صف / p.dst_h) * p.src_w * p.src_h * c;
        برائے (int64_t dst_x = 0; dst_x < p.dst_w; ++dst_x) {
            اعشاری * dst_ڈیٹا = (اعشاری *)dst->ڈیٹا + (صف * p.dst_w + dst_x) * c;
            ثابت int64_t src_y_بنیاد = dst_y * p.stride_y - p.pad_y;
            ثابت int64_t src_x_بنیاد = dst_x * p.stride_x - p.pad_x;

#اگر_تعریف_شدہ GGML_SIMD
            برائے (int64_t c_i = 0; c_i < c_pkg_انتہا; c_i += pkg_سائز) {
                GGML_F32_VEC مجموع = GGML_F32_VEC_ZERO;
                برائے (int64_t knl_y = 0; knl_y < p.knl_h; ++knl_y) {
                    ثابت int64_t src_y = src_y_بنیاد + knl_y * p.dilation_y;
                    اگر (src_y < 0 || src_y >= p.src_h) {
                        جاری;
                    }
                    برائے (int64_t knl_x = 0; knl_x < p.knl_w; ++knl_x) {
                        ثابت int64_t src_x = src_x_بنیاد + knl_x * p.dilation_x;
                        اگر (src_x < 0 || src_x >= p.src_w) {
                            جاری;
                        }
                        ثابت اعشاری * kp = (ثابت اعشاری *)knl_ڈیٹا + (knl_y * p.knl_w + knl_x) * c + c_i;
                        GGML_F32_VEC k = GGML_F32_VEC_بھرو(kp);
                        GGML_F32_VEC s = GGML_F32_VEC_بھرو(src_ڈیٹا + (src_y * p.src_w + src_x) * c + c_i);
                        مجموع = GGML_F32_VEC_FMA(مجموع, k, s);
                    }
                }
                GGML_F32_VEC_STORE(dst_ڈیٹا + c_i, مجموع);
            }
#اگر_ختم
            برائے (int64_t c_i = c_pkg_انتہا; c_i < c; ++c_i) {
                اعشاری مجموع = 0.0f;
                برائے (int64_t knl_y = 0; knl_y < p.knl_h; ++knl_y) {
                    ثابت int64_t src_y = src_y_بنیاد + knl_y * p.dilation_y;
                    اگر (src_y < 0 || src_y >= p.src_h) {
                        جاری;
                    }
                    برائے (int64_t knl_x = 0; knl_x < p.knl_w; ++knl_x) {
                        ثابت int64_t src_x = src_x_بنیاد + knl_x * p.dilation_x;
                        اگر (src_x < 0 || src_x >= p.src_w) {
                            جاری;
                        }
                        مجموع += ggml_conv_2d_dw_knl_f32(knl_ڈیٹا, (knl_y * p.knl_w + knl_x) * c + c_i, knl_type)
                             * src_ڈیٹا[(src_y * p.src_w + src_x) * c + c_i];
                    }
                }
                dst_ڈیٹا[c_i] = مجموع;
            }
        }
    }
}

ساکن خالی ggml_compute_forward_conv_2d_dw_whcn(
        ثابت ggml_compute_params * params,
        ثابت ggml_tensor * src,
        ثابت ggml_tensor * kernel,
        ggml_tensor * dst,
        ثابت ggml_conv_2d_dw_params & p) {

    ثابت int64_t n = p.channels * p.batch;
    ثابت int64_t فی_thread = (n + params->nth - 1) / params->nth;
    ثابت int64_t start = params->ith * فی_thread;
    ثابت int64_t end = MIN(start + فی_thread, n);
    ثابت حرف * knl_بنیاد = (ثابت حرف *)kernel->ڈیٹا;
    ثابت ggml_type knl_type = kernel->type;

    برائے (int64_t i = start; i < end; ++i) {
        ثابت int64_t knl_انحراف = (i % p.channels) * p.knl_w * p.knl_h;
        ثابت اعشاری * src_ڈیٹا = (ثابت اعشاری *)src->ڈیٹا + i * p.src_w * p.src_h;
        اعشاری * dst_ڈیٹا = (اعشاری *)dst->ڈیٹا + i * p.dst_w * p.dst_h;

        برائے (int64_t dst_y = 0; dst_y < p.dst_h; ++dst_y) {
            برائے (int64_t dst_x = 0; dst_x < p.dst_w; ++dst_x) {

                اعشاری مجموع = 0.0f;
                برائے (int64_t knl_y = 0; knl_y < p.knl_h; ++knl_y) {
                    ثابت int64_t src_y = dst_y * p.stride_y + knl_y * p.dilation_y - p.pad_y;
                    اگر (src_y < 0 || src_y >= p.src_h) {
                        جاری;
                    }
                    برائے (int64_t knl_x = 0; knl_x < p.knl_w; ++knl_x) {
                        ثابت int64_t src_x = dst_x * p.stride_x + knl_x * p.dilation_x - p.pad_x;
                        اگر (src_x < 0 || src_x >= p.src_w) {
                            جاری;
                        }
                        مجموع += ggml_conv_2d_dw_knl_f32(knl_بنیاد, knl_انحراف + knl_y * p.knl_w + knl_x, knl_type)
                             * src_ڈیٹا[src_y * p.src_w + src_x];
                    }
                }
                dst_ڈیٹا[dst_y * p.dst_w + dst_x] = مجموع;
            }
        }
    }
}

خالی ggml_compute_forward_conv_2d_dw(
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

    اگر (ggml_is_contiguous(src)) {
        ggml_compute_forward_conv_2d_dw_whcn(params, src, kernel, dst, p);
    } ورنہ اگر (ggml_is_contiguous_channels(src)) {
        GGML_ASSERT(kernel->nb[0] >= kernel->nb[2] && kernel->nb[1] >= kernel->nb[0]);
        ggml_compute_forward_conv_2d_dw_cwhn(params, src, kernel, dst, p);
    } ورنہ {
        GGML_ABORT("non-contiguous memory layout not supported");
    }
}

// ggml_compute_forward_pool_1d_ksp
ساکن خالی ggml_compute_forward_pool_1d_ksp(
        ثابت ggml_compute_params * params,
        ثابت ggml_op_pool op,
        ثابت صحیح k,
        ثابت صحیح s,
        ثابت صحیح p,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src = dst->src[0];

    أكد(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16);

    اگر (params->ith != 0) {
        واپس;
    }

    ثابت int64_t IW = src->ne[0];
    ثابت int64_t OW = dst->ne[0];

    ثابت int64_t nr = ggml_nrows(src);

    برائے (int64_t ir = 0; ir < nr; ++ir) {
        ثابت حرف * srow_بائٹیں =            (ثابت حرف *) src->ڈیٹا + ir * src->nb[1];
        اعشاری      * drow       = (اعشاری *) ((      حرف *) dst->ڈیٹا + ir * dst->nb[1]);

        برائے (int64_t ow = 0; ow < OW; ++ow) {
            اعشاری res = 0;
            انتخاب (op) {
                حالت GGML_OP_POOL_AVG: res = 0.0f;     توڑو;
                حالت GGML_OP_POOL_MAX: res = -FLT_MAX; توڑو;
                حالت GGML_OP_POOL_گنتی: GGML_ABORT("fatal error");
            }

            صحیح count = 0;
            ثابت صحیح بنیاد = (صحیح) ow * s - p;

            برائے (صحیح ki = 0; ki < k; ++ki) {
                ثابت صحیح j = بنیاد + ki;
                اگر (j < 0 || j >= (صحیح) IW) {
                    جاری;
                }

                اعشاری v;
                اگر (src->type == GGML_TYPE_F32) {
                    v = ((ثابت اعشاری *) srow_بائٹیں)[j];
                } ورنہ {
                    v = GGML_CPU_FP16_مقصد_FP32(((ثابت ggml_fp16_t *) srow_بائٹیں)[j]);
                }

                انتخاب (op) {
                    حالت GGML_OP_POOL_AVG: res += v;                توڑو;
                    حالت GGML_OP_POOL_MAX: res =  std::max(v, res); توڑو;
                    حالت GGML_OP_POOL_گنتی: GGML_ABORT("fatal error");
                }

                ++count;
            }

            انتخاب (op) {
                حالت GGML_OP_POOL_AVG: res = (count > 0) ? (res / count) : 0.0f; توڑو;
                حالت GGML_OP_POOL_MAX:                                           توڑو;
                حالت GGML_OP_POOL_گنتی: GGML_ABORT("fatal error");
            }

            drow[ow] = res;
        }
    }
}

// ggml_compute_forward_pool_1d

خالی ggml_compute_forward_pool_1d(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت int32_t * opts = (ثابت int32_t *)dst->op_params;
    ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    ثابت صحیح k0 = opts[1];
    ثابت صحیح s0 = opts[2];
    ثابت صحیح p0 = opts[3];

    ggml_compute_forward_pool_1d_ksp(params, op, k0, s0, p0, dst);
}

// ggml_compute_forward_pool_2d

خالی ggml_compute_forward_pool_2d(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src = dst->src[0];

    أكد(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16);

    اگر (params->ith != 0) {
        واپس;
    }

    ثابت int32_t * opts = (ثابت int32_t *)dst->op_params;

    ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    ثابت صحیح k0 = opts[1];
    ثابت صحیح k1 = opts[2];
    ثابت صحیح s0 = opts[3];
    ثابت صحیح s1 = opts[4];
    ثابت صحیح p0 = opts[5];
    ثابت صحیح p1 = opts[6];
    ثابت حرف * cdata = (ثابت حرف*)src->ڈیٹا;
    ثابت حرف * ثابت ڈیٹا_انتہا = cdata + ggml_nbytes(src);

    ثابت int64_t px = dst->ne[0];
    ثابت int64_t py = dst->ne[1];
    ثابت int64_t pa = px * py;

    اعشاری * dplane = (اعشاری *)dst->ڈیٹا;

    ثابت صحیح ka = k0 * k1;
    ثابت صحیح offset0 = -p0;
    ثابت صحیح offset1 = -p1;

    جب (cdata < ڈیٹا_انتہا) {
        برائے (صحیح oy = 0; oy < py; ++oy) {
            اعشاری * ثابت drow = dplane + oy * px;
            اعشاری * ثابت برآمد  = drow;

            برائے (صحیح ox = 0; ox < px; ++ox) {
                اعشاری res = 0;
                انتخاب (op) {
                    حالت GGML_OP_POOL_AVG: res = 0;        توڑو;
                    حالت GGML_OP_POOL_MAX: res = -FLT_MAX; توڑو;
                    حالت GGML_OP_POOL_گنتی: GGML_ABORT("fatal error");
                }

                ثابت صحیح ix = offset0 + ox * s0;
                ثابت صحیح iy = offset1 + oy * s1;

                برائے (صحیح ky = 0; ky < k1; ++ky) {
                    اگر (iy + ky < 0 || iy + ky >= src->ne[1]) {
                        جاری;
                    }

                    ثابت خالی * srow = (ثابت خالی *)(cdata + src->nb[1] * (iy + ky));
                    برائے (صحیح kx = 0; kx < k0; ++kx) {
                        صحیح j = ix + kx;
                        اگر (j < 0 || j >= src->ne[0]) {
                            جاری;
                        }

                        ثابت اعشاری srow_j = (src->type == GGML_TYPE_F32) ? ((ثابت اعشاری*)srow)[j] : GGML_CPU_FP16_مقصد_FP32(((ثابت ggml_fp16_t*)srow)[j]);
                        انتخاب (op) {
                            حالت GGML_OP_POOL_AVG: res += srow_j;                توڑو;
                            حالت GGML_OP_POOL_MAX: res =  std::max(srow_j, res); توڑو;
                            حالت GGML_OP_POOL_گنتی:               GGML_ABORT("fatal error");
                        }
                    }
                }
                انتخاب (op) {
                    حالت GGML_OP_POOL_AVG:           res /= ka; توڑو;
                    حالت GGML_OP_POOL_MAX:                      توڑو;
                    حالت GGML_OP_POOL_گنتی: GGML_ABORT("fatal error");
                }

                برآمد[ox] = res;
            }
        }

        cdata  += src->nb[2];
        dplane += pa;
    }
}

// ggml_compute_forward_pool_2d_back

خالی ggml_compute_forward_pool_2d_عقب(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src  = dst->src[0];
    ثابت ggml_tensor * dstf = dst->src[1]; // forward tensor of dst

    أكد(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);

    اگر (params->ith != 0) {
        واپس;
    }

    ثابت int32_t * opts = (ثابت int32_t *)dst->op_params;
    ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    ثابت صحیح k0 = opts[1];
    ثابت صحیح k1 = opts[2];
    ثابت صحیح s0 = opts[3];
    ثابت صحیح s1 = opts[4];
    ثابت صحیح p0 = opts[5];
    ثابت صحیح p1 = opts[6];

    حرف       * cdata  = (حرف       *) dst->ڈیٹا;
    ثابت حرف * cdataf = (ثابت حرف *) dstf->ڈیٹا;
    ثابت حرف * ثابت ڈیٹا_انتہا = cdata + ggml_nbytes(dst);

    GGML_ASSERT(params->ith == 0);
    memset(cdata, 0, ggml_nbytes(dst));

    ثابت int64_t px = src->ne[0];
    ثابت int64_t py = src->ne[1];
    ثابت int64_t pa = px * py;

    ثابت اعشاری * splane = (ثابت اعشاری *) src->ڈیٹا;

    ثابت صحیح ka = k0 * k1;
    ثابت صحیح offset0 = -p0;
    ثابت صحیح offset1 = -p1;

    جب (cdata < ڈیٹا_انتہا) {
        برائے (صحیح oy = 0; oy < py; ++oy) {
            ثابت اعشاری * ثابت srow = splane + oy * px;
            برائے (صحیح ox = 0; ox < px; ++ox) {
                ثابت اعشاری grad0 = srow[ox];

                ثابت صحیح ix = offset0 + ox * s0;
                ثابت صحیح iy = offset1 + oy * s1;

                اگر (op == GGML_OP_POOL_MAX) {
                    اعشاری maxval = -FLT_MAX;
                    صحیح kxmax = -1;
                    صحیح kymax = -1;

                    برائے (صحیح ky = 0; ky < k1; ++ky) {
                        اگر (iy + ky < 0 || iy + ky >= dst->ne[1]) {
                            جاری;
                        }
                        ثابت خالی * drowf = (ثابت خالی *)(cdataf + dst->nb[1] * (iy + ky));
                        برائے (صحیح kx = 0; kx < k0; ++kx) {
                            صحیح j = ix + kx;
                            اگر (j < 0 || j >= dst->ne[0]) {
                                جاری;
                            }

                            ثابت اعشاری val = dst->type == GGML_TYPE_F32 ?
                                ((ثابت اعشاری *) drowf)[j] : GGML_CPU_FP16_مقصد_FP32(((ثابت ggml_fp16_t *) drowf)[j]);
                            اگر (val <= maxval) {
                                جاری;
                            }

                            maxval = val;
                            kxmax = kx;
                            kymax = ky;
                        }
                    }

                    اگر (kxmax == -1 || kymax == -1) {
                        جاری;
                    }

                    خالی * drow = (خالی *)(cdata + dst->nb[1] * (iy + kymax));
                    ثابت صحیح j = ix + kxmax;
                    اگر (dst->type == GGML_TYPE_F32) {
                        ((اعشاری *) drow)[j] += grad0;
                    } ورنہ {
                        ((ggml_fp16_t *) drow)[j] = GGML_CPU_FP32_مقصد_FP16(grad0 + GGML_CPU_FP16_مقصد_FP32(((ثابت ggml_fp16_t *) drow)[j]));
                    }
                } ورنہ اگر (op == GGML_OP_POOL_AVG) {
                    ثابت اعشاری grad = grad0 / ka;

                    برائے (صحیح ky = 0; ky < k1; ++ky) {
                        اگر (iy + ky < 0 || iy + ky >= dst->ne[1]) {
                            جاری;
                        }
                        خالی * drow = (خالی *)(cdata + dst->nb[1] * (iy + ky));
                        برائے (صحیح kx = 0; kx < k0; ++kx) {
                            صحیح j = ix + kx;
                            اگر (j < 0 || j >= dst->ne[0]) {
                                جاری;
                            }

                            اگر (dst->type == GGML_TYPE_F32) {
                                ((اعشاری *) drow)[j] += grad;
                            } ورنہ {
                                ((ggml_fp16_t *) drow)[j] += GGML_CPU_FP32_مقصد_FP16(grad);
                            }
                        }
                    }
                } ورنہ {
                    GGML_ASSERT(جھوٹ);
                }
            }
        }

        cdata  += dst->nb[2];
        cdataf += dst->nb[2];
        splane += pa;
    }
}

// ggml_compute_forward_upscale

ساکن خالی ggml_compute_forward_upscale_f32(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    اعشاری sf0 = (اعشاری)ne0/src0->ne[0];
    اعشاری sf1 = (اعشاری)ne1/src0->ne[1];
    اعشاری sf2 = (اعشاری)ne2/src0->ne[2];
    اعشاری sf3 = (اعشاری)ne3/src0->ne[3];
    اعشاری pixel_انحراف = 0.5f;

    ثابت int32_t منوال_flags = ggml_get_op_params_i32(dst, 0);
    ثابت ggml_scale_منوال mode = (ggml_scale_منوال) (منوال_flags & 0xFF);

    اگر (منوال_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        pixel_انحراف = 0.0f;
        sf0 = ne0 > 1 && ne00 > 1 ? (اعشاری)(ne0 - 1) / (ne00 - 1) : sf0;
        sf1 = ne1 > 1 && ne01 > 1 ? (اعشاری)(ne1 - 1) / (ne01 - 1) : sf1;
    }

    اگر (mode == GGML_SCALE_منوال_قریب_ترین) {
        برائے (int64_t i3 = 0; i3 < ne3; i3++) {
            ثابت int64_t i03 = i3 / sf3;
            برائے (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                ثابت int64_t i02 = i2 / sf2;
                برائے (int64_t i1 = 0; i1 < ne1; i1++) {
                    ثابت int64_t i01 = i1 / sf1;
                    برائے (int64_t i0 = 0; i0 < ne0; i0++) {
                        ثابت int64_t i00 = i0 / sf0;

                        ثابت اعشاری * x = (اعشاری *)((حرف *) src0->ڈیٹا + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              اعشاری * y = (اعشاری *)((حرف *)  dst->ڈیٹا +  i0*nb0  +  i1*nb1  +  i2*nb2  +  i3*nb3);

                        *y = *x;
                    }
                }
            }
        }
    } ورنہ اگر (mode == GGML_SCALE_منوال_BILINEAR && (منوال_flags & GGML_SCALE_FLAG_ANTIALIAS)) {
        // Similar to F.interpolate(..., mode="bilinear", align_corners=False, antialias=True)
        // https://github.com/pytorch/pytorch/blob/8871ff29b743948d1225389d5b7068f37b22750b/aten/src/ATen/native/cpu/UpSampleKernel.cpp
        خودکار triangle_filter = [](اعشاری x) -> اعشاری {
            واپس std::max(1.0f - fabsf(x), 0.0f);
        };

        // support and invscale, minimum 1 pixel for bilinear
        ثابت اعشاری support1  = std::max(1.0f, 1.0f / sf1);
        ثابت اعشاری invscale1 = 1.0f / support1;
        ثابت اعشاری support0  = std::max(1.0f, 1.0f / sf0);
        ثابت اعشاری invscale0 = 1.0f / support0;

        برائے (int64_t i3 = 0; i3 < ne3; i3++) {
            ثابت int64_t i03 = i3 / sf3;
            برائے (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                ثابت int64_t i02 = i2 / sf2;
                برائے (int64_t i1 = 0; i1 < ne1; i1++) {
                    ثابت اعشاری y = ((اعشاری) i1 + pixel_انحراف) / sf1;
                    برائے (int64_t i0 = 0; i0 < ne0; i0++) {
                        ثابت اعشاری x = ((اعشاری) i0 + pixel_انحراف) / sf0;

                        // the range of source pixels that contribute
                        ثابت int64_t x_min = std::max<int64_t>(x - support0 + pixel_انحراف, 0);
                        ثابت int64_t x_max = std::min<int64_t>(x + support0 + pixel_انحراف, ne00);
                        ثابت int64_t y_min = std::max<int64_t>(y - support1 + pixel_انحراف, 0);
                        ثابت int64_t y_max = std::min<int64_t>(y + support1 + pixel_انحراف, ne01);

                        // bilinear filter with antialiasing
                        اعشاری val = 0.0f;
                        اعشاری کل_وزن = 0.0f;

                        برائے (int64_t sy = y_min; sy < y_max; sy++) {
                            ثابت اعشاری وزن_y = triangle_filter((sy - y + pixel_انحراف) * invscale1);

                            برائے (int64_t sx = x_min; sx < x_max; sx++) {
                                ثابت اعشاری وزن_x = triangle_filter((sx - x + pixel_انحراف) * invscale0);
                                ثابت اعشاری وزن = وزن_x * وزن_y;

                                اگر (وزن <= 0.0f) {
                                    جاری;
                                }

                                ثابت اعشاری pixel = *(ثابت اعشاری *)((ثابت حرف *)src0->ڈیٹا + sx*nb00 + sy*nb01 + i02*nb02 + i03*nb03);
                                val += pixel * وزن;
                                کل_وزن += وزن;
                            }
                        }

                        اگر (کل_وزن > 0.0f) {
                            val /= کل_وزن;
                        }

                        اعشاری * dst_ptr = (اعشاری *)((حرف *)dst->ڈیٹا + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);
                        *dst_ptr = val;
                    }
                }
            }
        }
    } ورنہ اگر (mode == GGML_SCALE_منوال_BILINEAR) {
        برائے (int64_t i3 = 0; i3 < ne3; i3++) {
            ثابت int64_t i03 = i3 / sf3;
            برائے (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                ثابت int64_t i02 = i2 / sf2;
                برائے (int64_t i1 = 0; i1 < ne1; i1++) {
                    ثابت اعشاری y = ((اعشاری)i1 + pixel_انحراف) / sf1 - pixel_انحراف;
                    int64_t y0 = (int64_t)floorf(y);
                    int64_t y1 = y0 + 1;

                    y0 = std::max(int64_t(0), std::min(y0, ne01 - 1));
                    y1 = std::max(int64_t(0), std::min(y1, ne01 - 1));

                    اعشاری dy = y - (اعشاری)y0;
                    dy = std::max(0.0f, std::min(dy, 1.0f));

                    برائے (int64_t i0 = 0; i0 < ne0; i0++) {
                        ثابت اعشاری x = ((اعشاری)i0 + pixel_انحراف) / sf0 - pixel_انحراف;
                        int64_t x0 = (int64_t)floorf(x);
                        int64_t x1 = x0 + 1;

                        x0 = std::max(int64_t(0), std::min(x0, ne00 - 1));
                        x1 = std::max(int64_t(0), std::min(x1, ne00 - 1));

                        اعشاری dx = x - (اعشاری)x0;
                        dx = std::max(0.0f, std::min(dx, 1.0f));

                        // fetch the four surrounding pixel values and interpolate
                        ثابت اعشاری a = *(ثابت اعشاری *)((ثابت حرف *)src0->ڈیٹا + x0*nb00 + y0*nb01 + i02*nb02 + i03*nb03);
                        ثابت اعشاری b = *(ثابت اعشاری *)((ثابت حرف *)src0->ڈیٹا + x1*nb00 + y0*nb01 + i02*nb02 + i03*nb03);
                        ثابت اعشاری c = *(ثابت اعشاری *)((ثابت حرف *)src0->ڈیٹا + x0*nb00 + y1*nb01 + i02*nb02 + i03*nb03);
                        ثابت اعشاری d = *(ثابت اعشاری *)((ثابت حرف *)src0->ڈیٹا + x1*nb00 + y1*nb01 + i02*nb02 + i03*nb03);

                        ثابت اعشاری val = a*(1 - dx)*(1 - dy) + b*dx*(1 - dy) + c*(1 - dx)*dy + d*dx*dy;

                        اعشاری * y_dst = (اعشاری *)((حرف *)dst->ڈیٹا + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);
                        *y_dst = val;
                    }
                }
            }
        }
    } ورنہ اگر (mode == GGML_SCALE_منوال_BICUBIC) {
        // https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
        ثابت اعشاری a = -0.75f; // use alpha = -0.75 (same as PyTorch)
        خودکار weight1 = [a](اعشاری x) { واپس ((a + 2) * x - (a + 3)) * x * x + 1; };
        خودکار weight2 = [a](اعشاری x) { واپس ((a * x - 5 * a) * x + 8 * a) * x - 4 * a; };
        خودکار bicubic = [=](اعشاری p0, اعشاری p1, اعشاری p2, اعشاری p3, اعشاری x) {
            ثابت اعشاری w0 = weight2(x + 1);
            ثابت اعشاری w1 = weight1(x + 0);
            ثابت اعشاری w2 = weight1(1 - x);
            ثابت اعشاری w3 = weight2(2 - x);
            واپس p0*w0 + p1*w1 + p2*w2 + p3*w3;
        };

        برائے (int64_t i3 = 0; i3 < ne3; i3++) {
            ثابت int64_t i03 = i3 / sf3;
            برائے (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                ثابت int64_t i02 = i2 / sf2;
                برائے (int64_t i1 = 0; i1 < ne1; i1++) {
                    ثابت اعشاری y = ((اعشاری)i1 + pixel_انحراف) / sf1 - pixel_انحراف;
                    ثابت int64_t y0 = (int64_t)floorf(y);
                    ثابت اعشاری dy = y - (اعشاری)y0;

                    برائے (int64_t i0 = 0; i0 < ne0; i0++) {
                        ثابت اعشاری x = ((اعشاری)i0 + pixel_انحراف) / sf0 - pixel_انحراف;
                        ثابت int64_t x0 = (int64_t)floorf(x);
                        ثابت اعشاری dx = x - (اعشاری)x0;

                        خودکار p = [=](int64_t x_off, int64_t y_off) -> اعشاری {
                            int64_t i00 = std::max(int64_t(0), std::min(x0 + x_off, ne00 - 1));
                            int64_t i01 = std::max(int64_t(0), std::min(y0 + y_off, ne01 - 1));
                            واپس *(ثابت اعشاری *)((ثابت حرف *)src0->ڈیٹا + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                        };

                        ثابت اعشاری val = bicubic(
                            bicubic(p(-1,-1), p(0,-1), p(1,-1), p(2,-1), dx),
                            bicubic(p(-1, 0), p(0, 0), p(1, 0), p(2, 0), dx),
                            bicubic(p(-1, 1), p(0, 1), p(1, 1), p(2, 1), dx),
                            bicubic(p(-1, 2), p(0, 2), p(1, 2), p(2, 2), dx), dy);

                        اعشاری * y_dst = (اعشاری *)((حرف *)dst->ڈیٹا + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);
                        *y_dst = val;
                    }
                }
            }
        }
    } ورنہ {
        GGML_ABORT("unsupported upscale mode");
    }
}

خالی ggml_compute_forward_upscale(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_upscale_f32(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}


// ggml_compute_forward_pad

سانچہ<منطقی circular_t>
ساکن خالی ggml_compute_forward_pad_f32(
    ثابت ggml_compute_params * params,
          ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    أكد(dst->nb[0] == حجم(اعشاری));

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    اعشاری * dst_ptr = (اعشاری *) dst->ڈیٹا;
    ثابت int32_t lp0 = ggml_get_op_params_i32(dst, 0);
    ثابت int32_t rp0 = ggml_get_op_params_i32(dst, 1);
    ثابت int32_t lp1 = ggml_get_op_params_i32(dst, 2);
    ثابت int32_t rp1 = ggml_get_op_params_i32(dst, 3);
    ثابت int32_t lp2 = ggml_get_op_params_i32(dst, 4);
    ثابت int32_t rp2 = ggml_get_op_params_i32(dst, 5);
    ثابت int32_t lp3 = ggml_get_op_params_i32(dst, 6);
    ثابت int32_t rp3 = ggml_get_op_params_i32(dst, 7);

    ثابت int64_t nrows = ne1 * ne2 * ne3;
    ثابت منطقی src_صف_contig = src0->nb[0] == حجم(اعشاری);

    برائے (int64_t صف = ith; صف < nrows; صف += nth) {
        ثابت int64_t i1 = صف % ne1;
        ثابت int64_t i2 = (صف / ne1) % ne2;
        ثابت int64_t i3 = صف / (ne1 * ne2);

        اعشاری * dst_صف = dst_ptr + صف * ne0;

        اگر constexpr (circular_t) {
            // circular padding wraps around on a torus
            ثابت int64_t src_i1 = ggml_wrap_around(i1 - lp1, ne01);
            ثابت int64_t src_i2 = ggml_wrap_around(i2 - lp2, ne02);
            ثابت int64_t src_i3 = ggml_wrap_around(i3 - lp3, ne03);

            برائے (int64_t i0 = 0; i0 < ne0; ++i0) {
                ثابت int64_t src_i0 = ggml_wrap_around(i0 - lp0, ne00);
                dst_صف[i0] = *(ثابت اعشاری *)((ثابت حرف *) src0->ڈیٹا + src_i3*nb03 + src_i2*nb02 + src_i1*nb01 + src_i0*nb00);
            }
        } ورنہ {
            ثابت منطقی interior = (i1 >= lp1 && i1 < ne1 - rp1)
                               && (i2 >= lp2 && i2 < ne2 - rp2)
                               && (i3 >= lp3 && i3 < ne3 - rp3);
            اگر (!interior) {
                memset(dst_صف, 0, ne0 * حجم(اعشاری));
                جاری;
            }

            ثابت حرف * src_صف = (ثابت حرف *) src0->ڈیٹا + (i3 - lp3)*nb03 + (i2 - lp2)*nb02 + (i1 - lp1)*nb01;

            اگر (src_صف_contig && lp0 >= 0 && rp0 >= 0) {
                memset(dst_صف, 0, lp0 * حجم(اعشاری));
                حافظہ_نقل(dst_صف + lp0, src_صف, ne00 * حجم(اعشاری));
                memset(dst_صف + lp0 + ne00, 0, rp0 * حجم(اعشاری));
            } ورنہ {
                برائے (int64_t i0 = 0; i0 < ne0; ++i0) {
                    اگر (i0 >= lp0 && i0 < ne0 - rp0) {
                        dst_صف[i0] = *(ثابت اعشاری *)(src_صف + (i0 - lp0)*nb00);
                    } ورنہ {
                        dst_صف[i0] = 0.0f;
                    }
                }
            }
        }
    }
}


خالی ggml_compute_forward_pad(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت منطقی circular = (منطقی) ggml_get_op_params_i32(dst, 8);
    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                اگر (circular) {
                    ggml_compute_forward_pad_f32<سچ>(params, dst);
                } ورنہ {
                    ggml_compute_forward_pad_f32<جھوٹ>(params, dst);
                }
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_pad_reflect_1d

خالی ggml_compute_forward_pad_reflect_1d(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت int32_t * opts = (ثابت int32_t *) dst->op_params;
    ثابت صحیح p0 = opts[0];
    ثابت صحیح p1 = opts[1];

    GGML_TENSOR_UNARY_OP_LOCALS

    برائے (int64_t i3 = 0; i3 < ne3; i3++) {
        برائے (int64_t i2 = 0; i2 < ne2; i2++) {
            برائے (int64_t i1 = ith; i1 < ne1; i1 += nth) {
                اعشاری * بائیں  = (اعشاری *) ((حرف *) dst->ڈیٹا + i3*nb3 + i2*nb2 + i1*nb1 +         p0*nb0);
                اعشاری * دائیں = (اعشاری *) ((حرف *) dst->ڈیٹا + i3*nb3 + i2*nb2 + i1*nb1 + (ne0-p1-1)*nb0);

                ggml_vec_cpy_f32(ne00, بائیں, (اعشاری *) ((حرف *) src0->ڈیٹا + i3*nb03 + i2*nb02 + i1*nb01));

                برائے (صحیح i0 = 1; i0 <= p0; i0++) { بائیں[-i0] = بائیں[i0];   }
                برائے (صحیح i0 = 1; i0 <= p1; i0++) { دائیں[i0] = دائیں[-i0]; }
            }
        }
    }
}

// ggml_compute_forward_roll

ساکن int64_t ggml_wrap_اشاریہ(int64_t i, int64_t ne) {
    اگر (i < 0) {
        واپس i + ne;
    } ورنہ اگر (i >= ne) {
        واپس i - ne;
    }
    واپس i;
}

ساکن خالی ggml_compute_forward_roll_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت اعشاری * src_ڈیٹا = (ثابت اعشاری *) src0->ڈیٹا;
    اعشاری * dst_ڈیٹا = (اعشاری *) dst->ڈیٹا;

    GGML_TENSOR_UNARY_OP_LOCALS

    ثابت صحیح s0 = ggml_get_op_params_i32(dst, 0);
    ثابت صحیح s1 = ggml_get_op_params_i32(dst, 1);
    ثابت صحیح s2 = ggml_get_op_params_i32(dst, 2);
    ثابت صحیح s3 = ggml_get_op_params_i32(dst, 3);

    ثابت int64_t کل = ne1 * ne2 * ne3;
    ثابت int64_t فی_thread = (کل + params->nth) / params->nth;
    ثابت int64_t start = params->ith * فی_thread;
    ثابت int64_t end   = std::min(start + فی_thread, کل);

    برائے (int64_t i = start; i < end; ++i) {
        ثابت int64_t i1 = i % ne1;
        ثابت int64_t i2 = (i / ne1) % ne2;
        ثابت int64_t i3 = i / (ne2 * ne1);
        اعشاری * dst_صف = dst_ڈیٹا + (i3*nb3 + i2*nb2 + i1*nb1) / حجم(اعشاری);

        ثابت int64_t i01 = ggml_wrap_اشاریہ(i1 - s1, ne01);
        ثابت int64_t i02 = ggml_wrap_اشاریہ(i2 - s2, ne02);
        ثابت int64_t i03 = ggml_wrap_اشاریہ(i3 - s3, ne03);
        ثابت اعشاری * src_صف = src_ڈیٹا + (i03*nb03 + i02*nb02 + i01*nb01) / حجم(اعشاری);

        ثابت int64_t s = ggml_wrap_اشاریہ(-s0, ne00);
        ثابت int64_t n = ne00 - s;
        ggml_vec_cpy_f32(n, dst_صف,     src_صف + s);
        ggml_vec_cpy_f32(s, dst_صف + n, src_صف);
    }
}

خالی ggml_compute_forward_roll(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_roll_f32(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_arange

ساکن خالی ggml_compute_forward_arange_f32(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    GGML_ASSERT(dst->nb[0] == حجم(اعشاری));

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت اعشاری start = ggml_get_op_params_f32(dst, 0);
    ثابت اعشاری stop  = ggml_get_op_params_f32(dst, 1);
    ثابت اعشاری قدم  = ggml_get_op_params_f32(dst, 2);

    ثابت int64_t اقدام = (int64_t) ceilf((stop - start) / قدم);

    GGML_ASSERT(ggml_nelements(dst) == اقدام);

    برائے (int64_t i = ith; i < اقدام; i+= nth) {
        اعشاری value = start + قدم * i;
        ((اعشاری *)dst->ڈیٹا)[i] = value;
    }
}

خالی ggml_compute_forward_arange(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {
    انتخاب (dst->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_arange_f32(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ساکن خالی ggml_compute_forward_timestep_embedding_f32(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->nb[0] == حجم(اعشاری));

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    ثابت صحیح dim = ggml_get_op_params_i32(dst, 0);
    ثابت صحیح max_period = ggml_get_op_params_i32(dst, 1);

    صحیح half = dim / 2;

    برائے (int64_t i = 0; i < ne00; i++) {
        اعشاری * embed_ڈیٹا = (اعشاری *)((حرف *)  dst->ڈیٹا +  i*nb1);
        برائے (int64_t j = ith; j < half; j += nth) {
            اعشاری timestep = ((اعشاری *)src0->ڈیٹا)[i];
            اعشاری freq = (اعشاری)expf(-logf(max_period) * j / half);
            اعشاری arg = timestep * freq;
            embed_ڈیٹا[j] = cosf(arg);
            embed_ڈیٹا[j + half] = sinf(arg);
        }
        اگر (dim % 2 != 0 && ith == 0) {
            embed_ڈیٹا[2 * half] = 0.f;
        }
    }
}

خالی ggml_compute_forward_timestep_embedding(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_timestep_embedding_f32(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_argsort

سانچہ<شمار ggml_ترتیب_دو_ترتیب ترتیب>
ساخت cmp_argsort {
    ثابت اعشاری * ڈیٹا;
    منطقی عامل()(int32_t a, int32_t b) ثابت {
        اگر constexpr (ترتیب == GGML_ترتیب_دو_ترتیب_ASC) {
            واپس ڈیٹا[a] < ڈیٹا[b];
        } ورنہ {
            واپس ڈیٹا[a] > ڈیٹا[b];
        }
    }
};

ساکن خالی ggml_compute_forward_argsort_f32(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(nb0 == حجم(اعشاری));

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت int64_t nr = ggml_nrows(src0);

    ggml_ترتیب_دو_ترتیب ترتیب = (ggml_ترتیب_دو_ترتیب) ggml_get_op_params_i32(dst, 0);

    برائے (int64_t i = ith; i < nr; i += nth) {
        ثابت اعشاری * src_ڈیٹا = (اعشاری *)((حرف *) src0->ڈیٹا + i*nb01);

        int32_t * dst_ڈیٹا = (int32_t *)((حرف *) dst->ڈیٹا + i*nb1);

        برائے (int64_t j = 0; j < ne0; j++) {
            dst_ڈیٹا[j] = j;
        }

        انتخاب (ترتیب) {
            حالت GGML_ترتیب_دو_ترتیب_ASC:
                std::sort(dst_ڈیٹا, dst_ڈیٹا + ne0, cmp_argsort<GGML_ترتیب_دو_ترتیب_ASC>{src_ڈیٹا});
                توڑو;

            حالت GGML_ترتیب_دو_ترتیب_DESC:
                std::sort(dst_ڈیٹا, dst_ڈیٹا + ne0, cmp_argsort<GGML_ترتیب_دو_ترتیب_DESC>{src_ڈیٹا});
                توڑو;

            طےشدہ:
                GGML_ABORT("invalid sort order");
        }
    }
}

خالی ggml_compute_forward_argsort(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_argsort_f32(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_top_k

ساخت cmp_چوٹی_k {
    ثابت اعشاری * ڈیٹا;
    منطقی عامل()(int32_t a, int32_t b) ثابت {
        واپس ڈیٹا[a] > ڈیٹا[b];
    }
};

ساکن خالی ggml_compute_forward_چوٹی_k_f32(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(nb0 == حجم(اعشاری));

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت int64_t nr = ggml_nrows(src0);

    ثابت صحیح چوٹی_k = ne0;

    int32_t * tmp = (int32_t *) params->wdata + (ne00 + CACHE_سطر_سائز_F32) * ith;

    برائے (int64_t i = ith; i < nr; i += nth) {
        ثابت اعشاری * src_ڈیٹا = (اعشاری *)((حرف *) src0->ڈیٹا + i*nb01);

        برائے (int64_t j = 0; j < ne00; j++) {
            tmp[j] = j;
        }

        std::partial_sort(tmp, tmp + چوٹی_k, tmp + ne00, cmp_چوٹی_k{src_ڈیٹا});

        int32_t * dst_ڈیٹا = (int32_t *)((حرف *) dst->ڈیٹا + i*nb1);

        std::copy(tmp, tmp + چوٹی_k, dst_ڈیٹا);

        // emphasize that the order is not important
        اگر (چوٹی_k > 1) {
            std::swap(dst_ڈیٹا[0], dst_ڈیٹا[1]);
        }
    }
}

خالی ggml_compute_forward_چوٹی_k(
    ثابت ggml_compute_params * params,
    ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_چوٹی_k_f32(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ساکن خالی ggml_compute_forward_flash_attn_ext_f16_one_chunk(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst,
        صحیح ir0, صحیح ir1,
        int64_t ic_آغاز, int64_t ic_انتہا,
        اعشاری * partials, int64_t partial_stride) {

    ثابت منطقی write_partials = (partials != خالی_اشاریہ);
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
    GGML_ASSERT(nbq0 == ggml_type_سائز(q->type));
    GGML_ASSERT(nbk0 == ggml_type_سائز(k->type));
    GGML_ASSERT(nbv0 == ggml_type_سائز(v->type));

    GGML_ASSERT(neq0 == DK);
    GGML_ASSERT(nek0 == DK);
    GGML_ASSERT(nev0 == DV);

    GGML_ASSERT(neq1 == N);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == حجم(اعشاری));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    ثابت int64_t rk2 = neq2/nek2;
    ثابت int64_t rk3 = neq3/nek3;

    ثابت int64_t rv2 = neq2/nev2;
    ثابت int64_t rv3 = neq3/nev3;

    // parallelize by q rows using ggml_vec_dot_f32

    اعشاری scale         = 1.0f;
    اعشاری max_bias      = 0.0f;
    اعشاری logit_softcap = 0.0f;

    حافظہ_نقل(&scale,         (اعشاری *) dst->op_params + 0, حجم(اعشاری));
    حافظہ_نقل(&max_bias,      (اعشاری *) dst->op_params + 1, حجم(اعشاری));
    حافظہ_نقل(&logit_softcap, (اعشاری *) dst->op_params + 2, حجم(اعشاری));

    اگر (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    ثابت uint32_t n_سر      = neq2;
    ثابت uint32_t n_سر_log2 = 1u << (uint32_t) floor(log2(n_سر));

    ثابت اعشاری m0 = powf(2.0f, -(max_bias       ) / n_سر_log2);
    ثابت اعشاری m1 = powf(2.0f, -(max_bias / 2.0f) / n_سر_log2);

    ggml_type         ثابت k_vec_dot_type = ggml_get_type_traits_cpu(k->type)->vec_dot_type;
    ggml_from_float_t ثابت q_مقصد_vec_dot   = ggml_get_type_traits_cpu(k_vec_dot_type)->from_float;
    ggml_vec_dot_t    ثابت kq_vec_dot     = ggml_get_type_traits_cpu(k->type)->vec_dot;
    ggml_مقصد_float_t   ثابت v_مقصد_float     = ggml_get_type_traits(v->type)->مقصد_float;

    GGML_ASSERT((                            q_مقصد_vec_dot) && "fattn: unsupported K-type");
    GGML_ASSERT((v->type == GGML_TYPE_F32 || v_مقصد_float  ) && "fattn: unsupported V-type");

    صحیح ith = params->ith;

    برائے (صحیح ir = ir0; ir < ir1; ++ir) {
        // q indices
        ثابت صحیح iq3 = ir/(neq2*neq1);
        ثابت صحیح iq2 = (ir - iq3*neq2*neq1)/neq1;
        ثابت صحیح iq1 = (ir - iq3*neq2*neq1 - iq2*neq1);

        ثابت uint32_t h = iq2; // head index
        ثابت اعشاری ڈھلان = (max_bias > 0.0f) ? h < n_سر_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_سر_log2) + 1) : 1.0f;

        اعشاری S = 0.0f;      //مجموع
        اعشاری M = -INFINITY; // maximum KQ value

        اعشاری       * VKQ32 = (اعشاری       *) params->wdata + ith*(1*DK + 2*DV + CACHE_سطر_سائز_F32); // FP32 VKQ accumulator
        اعشاری       * V32   =                 (VKQ32 + 1*DV); // (temporary) FP32 V buffer
        ggml_fp16_t * VKQ16 = (ggml_fp16_t *) (VKQ32 + 1*DV); // (temporary) FP16 VKQ accumulator
        ggml_fp16_t * Q_q   = (ggml_fp16_t *) (VKQ32 + 2*DV); // (temporary) buffer for Q converted to quantized/FP16

        اگر (v->type == GGML_TYPE_F16) {
            memset(VKQ16, 0, DV*حجم(ggml_fp16_t));
        } ورنہ {
            memset(VKQ32, 0, DV*حجم(اعشاری));
        }

        ثابت ggml_fp16_t * mp = نقاب ? (ggml_fp16_t *)((حرف *) نقاب->ڈیٹا + iq1*نقاب->nb[1] + (iq2%نقاب->ne[2])*نقاب->nb[2] + (iq3%نقاب->ne[3])*نقاب->nb[3]) : تہی;

        // k indices
        ثابت صحیح ik3 = iq3 / rk3;
        ثابت صحیح ik2 = iq2 / rk2;

        // v indices
        ثابت صحیح iv3 = iq3 / rv3;
        ثابت صحیح iv2 = iq2 / rv2;

        ثابت اعشاری * pq = (ثابت اعشاری *) ((حرف *) q->ڈیٹا + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3));
        q_مقصد_vec_dot(pq, Q_q, DK);

        // online softmax / attention
        // loop over n_kv and n_head_kv
        // ref: https://arxiv.org/pdf/2112.05682.pdf

        برائے (int64_t ic = ic_آغاز; ic < ic_انتہا; ++ic) {
            ثابت اعشاری mv = mp ? ڈھلان*GGML_CPU_FP16_مقصد_FP32(mp[ic]) : 0.0f;
            اگر (mv == -INFINITY) {
                جاری;
            }

            اعشاری s; // KQ value

            ثابت حرف * k_ڈیٹا = (ثابت حرف *) k->ڈیٹا + ( ic*nbk1 + ik2*nbk2 + ik3*nbk3);
            kq_vec_dot(DK, &s, 0, k_ڈیٹا, 0, Q_q, 0, 1);

            s = s*scale; // scale KQ value

            اگر (logit_softcap != 0.0f) {
                s = logit_softcap*tanhf(s);
            }

            s += mv; // apply mask

            ثابت اعشاری Mold = M;

            اعشاری ms = 1.0f; // upon new higher max val, scale VKQ and KQ sum with this value
            اعشاری vs = 1.0f; // post-softmax KQ value, expf(s - M)

            ثابت حرف * v_ڈیٹا = ((ثابت حرف *) v->ڈیٹا + (ic*nbv1 + iv2*nbv2 + iv3*nbv3));

            اگر (v->type == GGML_TYPE_F16) {
                اگر (s > M) {
                    // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
                    M = s;
                    ms = expf(Mold - M);

                    // V = V*expf(Mold - M)
                    ggml_vec_scale_f16(DV, VKQ16, ms);
                } ورنہ {
                    // no new maximum, ms == 1.0f, vs != 1.0f
                    vs = expf(s - M);
                }

                // V += v*expf(s - M)
                ggml_vec_mad_f16(DV, VKQ16, (ثابت ggml_fp16_t *) v_ڈیٹا, vs);
            } ورنہ {
                اگر (s > M) {
                    // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
                    M = s;
                    ms = expf(Mold - M);

                    // V = V*expf(Mold - M)
                    ggml_vec_scale_f32(DV, VKQ32, ms);
                } ورنہ {
                    // no new maximum, ms == 1.0f, vs != 1.0f
                    vs = expf(s - M);
                }

                // V += v*expf(s - M)
                اگر (v_مقصد_float) {
                    v_مقصد_float(v_ڈیٹا, V32, DV);
                    ggml_vec_mad_f32(DV, VKQ32, V32, vs);
                } ورنہ {
                    // V is F32
                    ggml_vec_mad_f32(DV, VKQ32, (ثابت اعشاری *) v_ڈیٹا, vs);
                }
            }

            S = S*ms + vs; // scale and increment sum with partial sum
        }

        اگر (v->type == GGML_TYPE_F16) {
            برائے (int64_t d = 0; d < DV; ++d) {
                VKQ32[d] = GGML_CPU_FP16_مقصد_FP32(VKQ16[d]);
            }
        }

        // sinks - apply only on the first kv-chunk
        اگر (sinks && ic_آغاز == 0) {
            ثابت اعشاری s = ((اعشاری *)((حرف *) sinks->ڈیٹا))[h];

            اعشاری ms = 1.0f;
            اعشاری vs = 1.0f;

            اگر (s > M) {
                ms = expf(M - s);
                M = s;
                ggml_vec_scale_f32(DV, VKQ32, ms);
            } ورنہ {
                vs = expf(s - M);
            }

            S = S*ms + vs;
        }

        اگر (write_partials) {
            // Write M, S, VKQ to partials for later reduction
            // partials layout: [M, S, VKQ[DV]] per query head
            اعشاری * partial = partials + ir * partial_stride;
            partial[0] = M;
            partial[1] = S;
            حافظہ_نقل(partial + 2, VKQ32, DV * حجم(اعشاری));
        } ورنہ {
            // V /= S
            ثابت اعشاری S_inv = S == 0.0f ? 0.0f : 1.0f/S;
            ggml_vec_scale_f32(DV, VKQ32, S_inv);

            // dst indices
            ثابت صحیح i1 = iq1;
            ثابت صحیح i2 = iq2;
            ثابت صحیح i3 = iq3;

            // permute(0, 2, 1, 3)
            حافظہ_نقل((حرف *) dst->ڈیٹا + (i3*ne2*ne1 + i2 + i1*ne1)*nb1, VKQ32, nb1);
        }
    }
}

ساکن خالی ggml_compute_forward_flash_attn_ext_tiled(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst,
        صحیح ir0, صحیح ir1) {
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
    GGML_ASSERT(nbq0 == ggml_type_سائز(q->type));
    GGML_ASSERT(nbk0 == ggml_type_سائز(k->type));
    GGML_ASSERT(nbv0 == ggml_type_سائز(v->type));

    GGML_ASSERT(neq0 == DK);
    GGML_ASSERT(nek0 == DK);
    GGML_ASSERT(nev0 == DV);

    GGML_ASSERT(neq1 == N);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == حجم(اعشاری));
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

    اعشاری scale         = 1.0f;
    اعشاری max_bias      = 0.0f;
    اعشاری logit_softcap = 0.0f;

    حافظہ_نقل(&scale,         (اعشاری *) dst->op_params + 0, حجم(اعشاری));
    حافظہ_نقل(&max_bias,      (اعشاری *) dst->op_params + 1, حجم(اعشاری));
    حافظہ_نقل(&logit_softcap, (اعشاری *) dst->op_params + 2, حجم(اعشاری));

    اگر (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    ثابت uint32_t n_سر      = neq2;
    ثابت uint32_t n_سر_log2 = 1u << (uint32_t) floor(log2(n_سر));

    ثابت اعشاری m0 = powf(2.0f, -(max_bias       ) / n_سر_log2);
    ثابت اعشاری m1 = powf(2.0f, -(max_bias / 2.0f) / n_سر_log2);

    صحیح ith = params->ith;

    ساکن constexpr صحیح Q_TILE_SZ  = ggml_fa_tile_config::Q;
    ساکن constexpr صحیح KV_TILE_SZ = ggml_fa_tile_config::KV;

    صحیح ir = ir0;
    جب (ir < ir1) {
        // q indices for the start of this tile
        ثابت صحیح iq3 = ir/(neq2*neq1);
        ثابت صحیح iq2 = (ir - iq3*neq2*neq1)/neq1;
        ثابت صحیح iq1 = (ir - iq3*neq2*neq1 - iq2*neq1);

        // Number of valid rows in this tile:
        // - limited by tile size (Q_TILE_SZ)
        // - limited by chunk boundary (ir1 - ir)
        // - limited by head boundary (neq1 - iq1) to avoid crossing into next head
        ثابت صحیح tile_صفیں = MIN(Q_TILE_SZ, MIN((صحیح)(ir1 - ir), (صحیح)(neq1 - iq1)));
        GGML_ASSERT(tile_صفیں > 0);

        ثابت uint32_t h = iq2; // head index
        ثابت اعشاری ڈھلان = (max_bias > 0.0f) ? h < n_سر_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_سر_log2) + 1) : 1.0f;

        اعشاری S[Q_TILE_SZ];
        اعشاری M[Q_TILE_SZ];

        برائے (صحیح i = 0 ; i < Q_TILE_SZ; ++i) {
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
        اعشاری * بنیاد  = (اعشاری *) params->wdata + ith*(Q_TILE_SZ*DK + 2*Q_TILE_SZ*KV_TILE_SZ + Q_TILE_SZ*DV + KV_TILE_SZ*DV + KV_TILE_SZ*DK + CACHE_سطر_سائز_F32);

        خالی  * Q_q    = بنیاد;
        اعشاری * KQ     = (اعشاری *)((حرف *)بنیاد + Q_TILE_SZ * DK * حجم(اعشاری));
        اعشاری * mask32 = KQ + Q_TILE_SZ * KV_TILE_SZ;
        اعشاری * VKQ32  = mask32 + Q_TILE_SZ * KV_TILE_SZ;
        اعشاری * V32    = VKQ32 + Q_TILE_SZ * DV;
        اعشاری * K_f32  = V32 + KV_TILE_SZ * DV;

        memset(VKQ32, 0, Q_TILE_SZ * DV * حجم(اعشاری));
        memset(mask32, 0, Q_TILE_SZ * KV_TILE_SZ * حجم(اعشاری));

        // k indices
        ثابت صحیح ik3 = iq3 / rk3;
        ثابت صحیح ik2 = iq2 / rk2;

        // v indices
        ثابت صحیح iv3 = iq3 / rv3;
        ثابت صحیح iv2 = iq2 / rv2;

        {
            اعشاری * Q_f32 = (اعشاری *)Q_q;
            برائے (صحیح tq = 0; tq < tile_صفیں; tq++) {
                ثابت اعشاری * pq = (ثابت اعشاری *) ((حرف *) q->ڈیٹا + ((iq1 + tq)*nbq1 + iq2*nbq2 + iq3*nbq3));
                حافظہ_نقل(Q_f32 + tq * DK, pq, DK * حجم(اعشاری));
            }
            برائے (صحیح tq = tile_صفیں; tq < Q_TILE_SZ; tq++) {
                memset(Q_f32 + tq * DK, 0, DK * حجم(اعشاری));
            }
        }

        memset(K_f32, 0, DK * KV_TILE_SZ * حجم(اعشاری));
        memset(V32,   0, KV_TILE_SZ * DV * حجم(اعشاری));

        برائے (int64_t ic = 0; ic < nek1; ic += KV_TILE_SZ) {
            ثابت صحیح kv_tile = (صحیح)std::min((int64_t)KV_TILE_SZ, nek1 - ic);

            // skip the tile entirely if all the masks are -inf
            اگر (نقاب) {
                منطقی can_skip = سچ;
                برائے (صحیح tq = 0; tq < tile_صفیں; tq++) {
                    ثابت ggml_fp16_t * mp_صف = (ثابت ggml_fp16_t *)((ثابت حرف *) نقاب->ڈیٹا + (iq1 + tq)*نقاب->nb[1] + (iq2%نقاب->ne[2])*نقاب->nb[2] + (iq3%نقاب->ne[3])*نقاب->nb[3]);
                    برائے (صحیح tk = 0; tk < kv_tile; tk++) {
                        mask32[tq * KV_TILE_SZ + tk] = ڈھلان * GGML_CPU_FP16_مقصد_FP32(mp_صف[ic + tk]);
                        اگر (mask32[tq * KV_TILE_SZ + tk] != -INFINITY) {
                            can_skip = جھوٹ;
                        }
                    }
                    // Pad remaining mask entries with -inf
                    برائے (صحیح tk = kv_tile; tk < KV_TILE_SZ; tk++) {
                        mask32[tq * KV_TILE_SZ + tk] = -INFINITY;
                    }
                }

                اگر (can_skip) {
                    جاری;
                }
            }

            // Pack K tile transposed: K_f32[dk][kv] so KV_TILE is contiguous (SIMD dim)
            // Zero-pad the last tile so the GEMM always operates on KV_TILE_SZ columns
            برائے (صحیح tk = 0; tk < kv_tile; tk++) {
                ثابت حرف * k_ڈیٹا = (ثابت حرف *)k->ڈیٹا + (ic + tk)*nbk1 + ik2*nbk2 + ik3*nbk3;
                اگر (kv_type == GGML_TYPE_F16) {
                    ثابت ggml_fp16_t * k_f16 = (ثابت ggml_fp16_t *)k_ڈیٹا;
                    برائے (int64_t dk = 0; dk < DK; dk++) {
                        K_f32[dk * KV_TILE_SZ + tk] = GGML_CPU_FP16_مقصد_FP32(k_f16[dk]);
                    }
                } ورنہ {
                    ثابت اعشاری * k_f32_src = (ثابت اعشاری *)k_ڈیٹا;
                    برائے (int64_t dk = 0; dk < DK; dk++) {
                        K_f32[dk * KV_TILE_SZ + tk] = k_f32_src[dk];
                    }
                }
            }
            memset(KQ, 0, Q_TILE_SZ * KV_TILE_SZ * حجم(اعشاری));
            simd_gemm(KQ, (ثابت اعشاری *)Q_q, K_f32, Q_TILE_SZ, DK, KV_TILE_SZ);
            ggml_vec_scale_f32(Q_TILE_SZ * KV_TILE_SZ, KQ, scale);

            اگر (logit_softcap != 0.0f) {
                ggml_vec_tanh_f32(Q_TILE_SZ * KV_TILE_SZ, KQ, KQ);
                ggml_vec_scale_f32(Q_TILE_SZ * KV_TILE_SZ, KQ, logit_softcap);
            }

            // Set padded KQ entries to -inf so softmax gives them zero weight
            اگر (kv_tile < KV_TILE_SZ) {
                برائے (صحیح tq = 0; tq < Q_TILE_SZ; tq++) {
                    برائے (صحیح tk = kv_tile; tk < KV_TILE_SZ; tk++) {
                        KQ[tq * KV_TILE_SZ + tk] = -INFINITY;
                    }
                }
            }

            اگر (نقاب) {
                ggml_vec_جوڑو_f32(tile_صفیں * KV_TILE_SZ, KQ, KQ, mask32);
            }

            منطقی skip[Q_TILE_SZ] = {};

            برائے (صحیح tq = 0; tq < Q_TILE_SZ; tq++) {
                اعشاری * kq_صف = KQ + tq * KV_TILE_SZ;

                اعشاری tile_max;
                ggml_vec_max_f32(KV_TILE_SZ, &tile_max, kq_صف);

                اگر (tile_max == -INFINITY) {
                    skip[tq] = سچ;
                    جاری;
                }

                ثابت اعشاری Mold = M[tq];
                ثابت اعشاری Mnew = fmaxf(Mold, tile_max);

                اگر (Mnew > Mold) {
                    ثابت اعشاری ms = expf(Mold - Mnew);
                    ggml_vec_scale_f32(DV, VKQ32 + tq * DV, ms);
                    S[tq] *= ms;
                }
                M[tq] = Mnew;


                S[tq] += ggml_vec_soft_max_f32(KV_TILE_SZ, kq_صف, kq_صف, Mnew);
            }

            // V accumulation: VKQ32 += softmax(KQ) * V
            // Pack V tile to contiguous F32, zero-padded
            برائے (صحیح tk = 0; tk < kv_tile; tk++) {
                ثابت حرف * v_ڈیٹا = (ثابت حرف *)v->ڈیٹا + (ic + tk)*nbv1 + iv2*nbv2 + iv3*nbv3;
                اگر (kv_type == GGML_TYPE_F16) {
                    ggml_cpu_fp16_مقصد_fp32((ثابت ggml_fp16_t *)v_ڈیٹا, V32 + tk * DV, DV);
                } ورنہ {
                    حافظہ_نقل(V32 + tk * DV, v_ڈیٹا, DV * حجم(اعشاری));
                }
            }
            برائے (صحیح tq = 0; tq < Q_TILE_SZ; tq++) {
                اگر (skip[tq]) {
                    memset(KQ + tq * KV_TILE_SZ, 0, KV_TILE_SZ * حجم(اعشاری));
                }
            }
            simd_gemm(VKQ32, KQ, V32, Q_TILE_SZ, KV_TILE_SZ, DV);
        }

        // sinks (apply only to valid rows in the tile)
        اگر (sinks) {
            ثابت اعشاری s = ((اعشاری *)((حرف *) sinks->ڈیٹا))[h];

            برائے (صحیح tq = 0; tq < tile_صفیں; tq++) {
                اعشاری ms = 1.0f;
                اعشاری vs = 1.0f;

                اگر (s > M[tq]) {
                    ms = expf(M[tq] - s);
                    ggml_vec_scale_f32(DV, VKQ32 + tq * DV, ms);
                } ورنہ {
                    vs = expf(s - M[tq]);
                }

                S[tq] = S[tq] * ms + vs;
            }
        }

        برائے (صحیح tq = 0; tq < tile_صفیں; tq++) {
            // V /= S
            ثابت اعشاری S_inv = S[tq] == 0.0f ? 0.0f : 1.0f / S[tq];
            ggml_vec_scale_f32(DV, VKQ32 + tq * DV, S_inv);

            // dst indices
            ثابت صحیح i1 = iq1 + tq;
            ثابت صحیح i2 = iq2;
            ثابت صحیح i3 = iq3;

            // permute(0, 2, 1, 3)
            حافظہ_نقل((حرف *) dst->ڈیٹا + (i3*ne2*ne1 + i2 + i1*ne1)*nb1, VKQ32 + tq * DV, nb1);
        }

        ir += tile_صفیں;
    }
}

// Reduction function: combines partial results across KV chunks
// Partials layout in wdata: [n_q_heads][n_chunks][2 + DV]
ساکن خالی ggml_flash_attn_ext_reduce_partials(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst,
        ثابت int64_t n_chunks,
        ثابت int64_t chunk_سائز) {

    ثابت ggml_tensor * q = dst->src[0];
    ثابت ggml_tensor * k = dst->src[1];
    ثابت ggml_tensor * v = dst->src[2];

    ثابت int64_t DK        = k->ne[0];
    ثابت int64_t DV        = v->ne[0];
    ثابت int64_t nek1      = k->ne[1];
    ثابت int64_t n_q_heads = q->ne[2];

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت int64_t wdata_فی_thread = DK + 2*DV + CACHE_سطر_سائز_F32;
    اعشاری *       thread_wdata     = (اعشاری *) params->wdata + ith * wdata_فی_thread;

    ثابت int64_t partials_انحراف  = nth * (DK + 2*DV + CACHE_سطر_سائز_F32);
    ثابت int64_t partial_سائز     = 2 + DV;
    ثابت اعشاری * partials_بنیاد    = (ثابت اعشاری *) params->wdata + partials_انحراف;

    // Output layout
    ثابت int64_t ne1 = dst->ne[1];
    ثابت int64_t ne2 = dst->ne[2];
    ثابت size_t  nb1 = dst->nb[1];

    // Each thread reduces a subset of query heads
    برائے (int64_t q_سر = ith; q_سر < n_q_heads; q_سر += nth) {
        اعشاری   M_final   = -INFINITY;
        اعشاری   S_final   = 0.0f;
        اعشاری * VKQ_final = thread_wdata;
        memset(VKQ_final, 0, DV * حجم(اعشاری));

        // Combine partials from all chunks
        برائے (int64_t chunk_idx = 0; chunk_idx < n_chunks; ++chunk_idx) {
            ثابت int64_t ic_آغاز = chunk_idx * chunk_سائز;
            اگر (ic_آغاز >= nek1) جاری;

            ثابت اعشاری * partial   = partials_بنیاد + (q_سر * n_chunks + chunk_idx) * partial_سائز;
            ثابت اعشاری   M_chunk   = partial[0];
            ثابت اعشاری   S_chunk   = partial[1];
            ثابت اعشاری * VKQ_chunk = partial + 2;

            اگر (S_chunk == 0.0f) جاری;

            ثابت اعشاری M_new     = fmaxf(M_final, M_chunk);
            ثابت اعشاری scale_old = expf(M_final - M_new);
            ثابت اعشاری scale_new = expf(M_chunk - M_new);

            برائے (int64_t d = 0; d < DV; ++d) {
                VKQ_final[d] = VKQ_final[d] * scale_old + VKQ_chunk[d] * scale_new;
            }
            S_final = S_final * scale_old + S_chunk * scale_new;
            M_final = M_new;
        }

        // Normalize and write to output
        اگر (S_final != 0.0f) {
            ثابت اعشاری S_inv = 1.0f / S_final;
            ggml_vec_scale_f32(DV, VKQ_final, S_inv);
        }
        // iq1=0, iq3=0 for decode
        حافظہ_نقل((حرف *) dst->ڈیٹا + (0*ne2*ne1 + q_سر + 0*ne1)*nb1, VKQ_final, nb1);
    }
}

ساکن خالی ggml_compute_forward_flash_attn_ext_f16(
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
    GGML_ASSERT(nbq0 == ggml_type_سائز(q->type));
    GGML_ASSERT(nbk0 == ggml_type_سائز(k->type));
    GGML_ASSERT(nbv0 == ggml_type_سائز(v->type));

    GGML_ASSERT(neq0 == DK);
    GGML_ASSERT(nek0 == DK);
    GGML_ASSERT(nev0 == DV);

    GGML_ASSERT(neq1 == N);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == حجم(اعشاری));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    // When use_ref is set, force the vec-only reference implementation (no tiling, no KV-chunking)
    ثابت منطقی use_ref = params->use_ref;

    ثابت منطقی kv_is_f32_or_f16 = (k->type == GGML_TYPE_F32 || k->type == GGML_TYPE_F16);
    ثابت منطقی use_بانٹو_kv_راستہ = !use_ref && (neq1 == 1 && neq3 == 1) && kv_is_f32_or_f16 && (k->type == v->type) && q->type == GGML_TYPE_F32 && nek1 >= 512;

    اگر (use_بانٹو_kv_راستہ) {
        ثابت int64_t chunk_سائز = (nek1 + nth - 1) / nth;

        // Partials buffer layout: [q_head][kv_chunk][M, S, VKQ]
        ثابت int64_t partial_سائز  = 2 + DV;
        اعشاری *       partials_بنیاد = (اعشاری *) params->wdata + nth * (DK + 2*DV + CACHE_سطر_سائز_F32);

        ثابت int64_t ic_آغاز = ith * chunk_سائز;
        ثابت int64_t ic_انتہا   = std::min(ic_آغاز + chunk_سائز, nek1);

        ثابت int64_t partial_stride = nth * partial_سائز;
        اعشاری *       chunk_partials = partials_بنیاد + ith * partial_سائز;

        اگر (ic_آغاز < nek1) {
            برائے (int64_t q_سر = 0; q_سر < neq2; q_سر++) {
                ggml_compute_forward_flash_attn_ext_f16_one_chunk(
                    params, dst, q_سر, q_سر + 1, ic_آغاز, ic_انتہا,
                    chunk_partials, partial_stride);
            }
        } ورنہ {
            برائے (int64_t q_سر = 0; q_سر < neq2; q_سر++) {
                اعشاری * q_partials = chunk_partials + q_سر * partial_stride;
                q_partials[0] = -INFINITY;  // M
                q_partials[1] = 0.0f;       // S
            }
        }

        ggml_barrier(params->threadpool);
        ggml_flash_attn_ext_reduce_partials(params, dst, nth, chunk_سائز);
    } ورنہ {

        // total rows in q
        ثابت int64_t nr = neq1*neq2*neq3;

        // disable for NUMA
        ثابت منطقی disable_chunking = ggml_is_numa();

        // 4x chunks per thread
        صحیح nth_scaled = nth * 4;
        int64_t chunk_سائز = (nr + nth_scaled - 1) / nth_scaled;
        int64_t nchunk     = (nr + chunk_سائز - 1) / chunk_سائز;

        اگر (nth == 1 || nchunk < nth || disable_chunking) {
            nchunk = nth;
        }

        اگر (ith == 0) {
            ggml_threadpool_chunk_سیٹ(params->threadpool, nth);
        }

        ggml_barrier(params->threadpool);

        ثابت int64_t dr = (nr + nchunk - 1) / nchunk;

        ساکن constexpr int64_t Q_TILE_SZ  = ggml_fa_tile_config::Q;
        منطقی use_tiled = !use_ref &&
                               (q->type == GGML_TYPE_F32 &&
                                kv_is_f32_or_f16 &&
                                k->type == v->type &&
                                neq1 >= Q_TILE_SZ);
#اگر defined(GGML_SIMD) && !defined(__x86_64__) && !defined(_M_X64)
#اگر defined(__ARM_FEATURE_SVE)
        ثابت int64_t f32_epr = svcntw();
#ورنہ
        ثابت int64_t f32_epr = GGML_F32_EPR;
#اگر_ختم
        use_tiled &= (DV % f32_epr == 0);
#اگر_ختم
        صحیح current_chunk = ith;

        جب (current_chunk < nchunk) {
            ثابت int64_t ir0 = dr * current_chunk;
            ثابت int64_t ir1 = MIN(ir0 + dr, nr);

            اگر (use_tiled) {
                ggml_compute_forward_flash_attn_ext_tiled(params, dst, ir0, ir1);
            } ورنہ {
                ggml_compute_forward_flash_attn_ext_f16_one_chunk(params, dst, ir0, ir1, 0, nek1, خالی_اشاریہ, 0);
            }

            current_chunk = ggml_threadpool_chunk_جوڑو(params->threadpool, 1);
        }
    }
}

خالی ggml_compute_forward_flash_attn_ext(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    انتخاب (dst->op_params[3]) {
        حالت GGML_PREC_DEFAULT:
        حالت GGML_PREC_F32:
            {
                // uses F32 accumulators
                ggml_compute_forward_flash_attn_ext_f16(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_flash_attn_back

ساکن خالی ggml_compute_forward_flash_attn_عقب_f32(
        ثابت ggml_compute_params * params,
        ثابت منطقی masked,
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

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت int64_t D = neq0;
    ثابت int64_t N = neq1;
    ثابت int64_t P = nek1 - N;
    ثابت int64_t M = P + N;

    ثابت صحیح Mup  = ggml_اوپر(M, GGML_SOFT_MAX_UNROLL);
    ثابت صحیح mxDM = MAX(D, Mup);

    // GGML_ASSERT(ne0 == D);
    // GGML_ASSERT(ne1 == N);
    GGML_ASSERT(P >= 0);

    GGML_ASSERT(nbq0 == حجم(اعشاری));
    GGML_ASSERT(nbk0 == حجم(اعشاری));
    GGML_ASSERT(nbv0 == حجم(اعشاری));

    GGML_ASSERT(neq0 == D);
    GGML_ASSERT(nek0 == D);
    GGML_ASSERT(nev1 == D);
    GGML_ASSERT(ned0 == D);

    GGML_ASSERT(neq1 == N);
    GGML_ASSERT(nek1 == N + P);
    GGML_ASSERT(nev1 == D);
    GGML_ASSERT(ned1 == N);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == حجم(اعشاری));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    اگر (ith == 0) {
        memset(dst->ڈیٹا, 0, nb0*ne0*ne1*ne2*ne3);
    }
    ggml_barrier(params->threadpool);

    ثابت int64_t elem_q = ggml_nelements(q);
    ثابت int64_t elem_k = ggml_nelements(k);

    ggml_type result_type = dst->type;
    GGML_ASSERT(ggml_blck_سائز(result_type) == 1);
    ثابت size_t tsize = ggml_type_سائز(result_type);

    ثابت size_t offs_q = 0;
    ثابت size_t offs_k = offs_q + GGML_PAD(elem_q * tsize, GGML_MEM_ALIGN);
    ثابت size_t offs_v = offs_k + GGML_PAD(elem_k * tsize, GGML_MEM_ALIGN);

    خالی * grad_q = (حرف *) dst->ڈیٹا;
    خالی * grad_k = (حرف *) dst->ڈیٹا + offs_k;
    خالی * grad_v = (حرف *) dst->ڈیٹا + offs_v;

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
    ثابت صحیح nr = nek2*nek3;

    // rows per thread
    ثابت صحیح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحیح ir0 = dr*ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    ثابت اعشاری scale = 1.0f/sqrtf(D);

    //printf("P=%d N=%d D=%d ir0=%d ir1=%d scale = %f\n", P, N, D, ir0, ir1, scale);

    // how often k2 (and v2) is repeated in q2
    صحیح nrep = neq2/nek2;

    برائے (صحیح ir = ir0; ir < ir1; ++ir) {
        // q indices
        ثابت صحیح ik3 = ir/(nek2);
        ثابت صحیح ik2 = ir - ik3*nek2;

        ثابت صحیح iq3 = ik3;
        ثابت صحیح id3 = ik3;
        ثابت صحیح iv3 = ik3;
        ثابت صحیح iv2 = ik2;

        برائے (صحیح irep = 0; irep < nrep; ++irep) {
            ثابت صحیح iq2 = ik2 + irep*nek2;
            ثابت صحیح id2 = iq2;

            // (ik2 + irep*nek2) % nek2 == ik2
            برائے (صحیح iq1 = 0; iq1 < neq1; ++iq1) {
                ثابت صحیح id1 = iq1;

                // not sure about CACHE_LINE_SIZE_F32..
                // - maybe it must not be multiplied by 2 and excluded from .. in SM 1*(..) offset?
                اعشاری * S  = (اعشاری *) params->wdata + ith*2*(mxDM + CACHE_سطر_سائز_F32) + 0*(mxDM+CACHE_سطر_سائز_F32);
                اعشاری * SM = (اعشاری *) params->wdata + ith*2*(mxDM + CACHE_سطر_سائز_F32) + 1*(mxDM+CACHE_سطر_سائز_F32);

                برائے (صحیح i = M; i < Mup; ++i) {
                    S[i] = -INFINITY;
                }

                ثابت int64_t masked_begin = masked ? (P + iq1 + 1) : M;
                برائے (int64_t ic = 0; ic < masked_begin; ++ic) {
                    // k indices
                    ثابت صحیح ik1 = ic;

                    // S indices
                    ثابت صحیح i1 = ik1;

                    ggml_vec_dot_f32(neq0,
                            S + i1, 0,
                            (اعشاری *) ((حرف *) k->ڈیٹا + (ik1*nbk1 + ik2*nbk2 + ik3*nbk3)), 0,
                            (اعشاری *) ((حرف *) q->ڈیٹا + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3)), 0, 1);
                }

                // scale
                ggml_vec_scale_f32(masked_begin, S, scale);

                برائے (int64_t i = masked_begin; i < M; i++) {
                    S[i] = -INFINITY;
                }

                // softmax
                // exclude known -INF S[..] values from max and loop
                // dont forget to set their SM values to zero
                {
                    اعشاری max = -INFINITY;
                    ggml_vec_max_f32(masked_begin, &max, S);

                    ggml_float مجموع = 0.0;
                    {
#اگر_تعریف_شدہ GGML_SOFT_MAX_ACCELERATE
                        max = -max;
                        vDSP_vsadd(SM, 1, &max, SM, 1, Mup);
                        vvexpf(SM, SM, &Mup);
                        ggml_vec_مجموع_f32(Mup, &مجموع, SM);
#ورنہ
                        مجموع = ggml_vec_soft_max_f32(Mup, SM, S, max);
#اگر_ختم
                    }

                    أكد(مجموع > 0.0);

                    مجموع = 1.0/مجموع;
                    ggml_vec_scale_f32(masked_begin, SM, مجموع);

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
                ggml_vec_سیٹ_f32(masked_begin, S, 0);
                برائے (int64_t ic = 0; ic < D; ++ic) {
                    ggml_vec_mad_f32(masked_begin,
                            S,
                             (اعشاری *) ((حرف *) v->ڈیٹا + (          ic*nbv1  + iv2*nbv2 + iv3*nbv3)),
                            *(اعشاری *) ((حرف *) d->ڈیٹا + (ic*nbd0 + id1*nbd1 + id2*nbd2 + id3*nbd3)));
                }

                // S = SM * (S - dot(SM, S))
                اعشاری dot_SM_gradSM = 0;
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
                برائے (int64_t ic = 0; ic < masked_begin; ++ic) {
                    ggml_vec_mad_f32(D,
                            (اعشاری *) ((حرف *) grad_q  + (iq1*nbgq1 + iq2*nbgq2  + iq3*nbgq3)),
                            (اعشاری *) ((حرف *) k->ڈیٹا + (ic*nbk1   + ik2*nbk2   + ik3*nbk3)),
                            S[ic]);
                }

                // grad[k][:D,:M,iq2,iq3] += S.T @ qcur
                // for ic:
                //  grad[k][:D,ic,iq2,iq3] += S.T[0,ic] * qcur[:D,0]
                //  grad[k][:D,ic,iq2,iq3] += S[ic]     * qcur[:D,0]
                // exclude known zero S[..] values from loop
                برائے (int64_t ic = 0; ic < masked_begin; ++ic) {
                    ggml_vec_mad_f32(D,
                            (اعشاری *) ((حرف *) grad_k  + (ic*nbgk1  + ik2*nbgk2  + ik3*nbgk3)),
                            (اعشاری *) ((حرف *) q->ڈیٹا + (iq1*nbq1  + iq2*nbq2   + iq3*nbq3)),
                            S[ic]);
                }

                // grad[v][:M,:D,iv2,iv3] += d[:D,id1,id2,id3].T       @ SM
                // for ic:
                //  grad[v][:M,ic,iv2,iv3] += d[:D,id1,id2,id3].T[0,ic] * SM[:M]
                //  grad[v][:M,ic,iv2,iv3] += d[ic,id1,id2,id3]         * SM[:M]
                // exclude known zero SM[..] values from mad
                برائے (int64_t ic = 0; ic < D; ++ic) {
                    ggml_vec_mad_f32(masked_begin,
                            (اعشاری *) ((حرف *) grad_v   + (          ic*nbgv1 + iv2*nbgv2 + iv3*nbgv3)),
                            SM,
                            *(اعشاری *) ((حرف *) d->ڈیٹا + (ic*nbd0 + id1*nbd1 + id2*nbd2  + id3*nbd3)));
                }
            }
        }
    }
}

خالی ggml_compute_forward_flash_attn_عقب(
        ثابت ggml_compute_params * params,
        ثابت منطقی masked,
        ggml_tensor * dst) {

    ثابت ggml_tensor * q = dst->src[0];

    انتخاب (q->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_flash_attn_عقب_f32(params, masked, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_ssm_conv

ساکن خالی ggml_compute_forward_ssm_conv_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0]; // conv_x
    ثابت ggml_tensor * src1 = dst->src[1]; // conv1d.weight

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح nc  = src1->ne[0]; // d_conv
    ثابت صحیح ncs = src0->ne[0]; // d_conv - 1 + n_t
    ثابت صحیح nr  = src0->ne[1]; // d_inner
    ثابت صحیح n_t =  dst->ne[1]; // tokens per sequence
    ثابت صحیح n_s =  dst->ne[2]; // number of sequences in the batch

    GGML_ASSERT( dst->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == حجم(اعشاری));
    GGML_ASSERT(src1->nb[0] == حجم(اعشاری));
    GGML_ASSERT(src0->nb[1] == src0->ne[0]*حجم(اعشاری));

    // rows per thread
    ثابت صحیح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحیح ir0 = dr*ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);
    ثابت صحیح ir  = ir1 - ir0;

    برائے (صحیح i3 = 0; i3 < n_s; ++i3) {
        برائے (صحیح i2 = 0; i2 < n_t; ++i2) {
            // {d_conv - 1 + n_t, d_inner, n_seqs}
            // sliding window
            ثابت اعشاری * s = (ثابت اعشاری *) ((ثابت حرف *) src0->ڈیٹا + ir0*(src0->nb[1]) + i2*(src0->nb[0]) + i3*(src0->nb[2])); // {d_conv, d_inner, n_s}
            ثابت اعشاری * c = (ثابت اعشاری *) ((ثابت حرف *) src1->ڈیٹا + ir0*(src1->nb[1])); // {d_conv, d_inner}
            اعشاری * x = (اعشاری *) ((حرف *) dst->ڈیٹا + ir0*(dst->nb[0]) + i2*(dst->nb[1]) + i3*(dst->nb[2])); // {d_inner, n_t, n_s}

            // TODO: transpose the output for smaller strides for big batches?
            // d_inner
            برائے (صحیح i1 = 0; i1 < ir; ++i1) {
                // rowwise dot product
                // NOTE: not using ggml_vec_dot_f32, because its sum is in double precision
                اعشاری sumf = 0.0f;

                // d_conv
                برائے (صحیح i0 = 0; i0 < nc; ++i0) {
                    sumf += s[i0 + i1*ncs] * c[i0 + i1*nc];
                }
                x[i1] = sumf;
            }
        }
    }
}

خالی ggml_compute_forward_ssm_conv(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    انتخاب (dst->src[0]->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_ssm_conv_f32(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_ssm_scan

ساکن خالی ggml_compute_forward_ssm_جائزہ_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0]; // s  {d_state, dim, n_head, n_seqs+}
    ثابت ggml_tensor * src1 = dst->src[1]; // x  {dim, n_head, n_seq_tokens, n_seqs}
    ثابت ggml_tensor * src2 = dst->src[2]; // dt {n_head, n_seq_tokens, n_seqs}
    ثابت ggml_tensor * src3 = dst->src[3]; // A  {d_state, n_head} or {1, n_head}
    ثابت ggml_tensor * src4 = dst->src[4]; // B  {d_state, n_group, n_seq_tokens, n_seqs}
    ثابت ggml_tensor * src5 = dst->src[5]; // C  {d_state, n_group, n_seq_tokens, n_seqs}
    ثابت ggml_tensor * src6 = dst->src[6]; // ids {n_seqs}

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت int64_t nc = src0->ne[0]; // d_state
    ثابت int64_t nr = src0->ne[1]; // dim
    ثابت int64_t nh = src1->ne[1]; // n_head
    ثابت int64_t ng = src4->ne[1];
    ثابت int64_t nt = src1->ne[2]; // number of tokens per sequence
    ثابت int64_t ns = src1->ne[3]; // number of sequences in the batch
    ثابت int64_t K  = ggml_get_op_params_i32(dst, 0);

    // can't use ggml_nbytes because src1 is not necessarily contiguous
    ثابت int64_t s_off = ggml_nelements(src1) * ggml_element_سائز(src1);

    GGML_ASSERT(K >= 1);
    GGML_ASSERT(ggml_nelements(src1) + K*nc*nr*nh*ns == ggml_nelements(dst));
    GGML_ASSERT(src0->nb[0] == حجم(اعشاری));
    GGML_ASSERT(src1->nb[0] == حجم(اعشاری));
    GGML_ASSERT(src2->nb[0] == حجم(اعشاری));
    GGML_ASSERT(src3->nb[0] == حجم(اعشاری));
    GGML_ASSERT(src4->nb[0] == حجم(اعشاری));
    GGML_ASSERT(src5->nb[0] == حجم(اعشاری));
    GGML_ASSERT(src6->nb[0] == حجم(int32_t));
    GGML_ASSERT(nh % ng == 0);
    GGML_ASSERT(src3->ne[0] == 1 || K == 1);

    // heads per thread
    ثابت صحیح dh = (nh + nth - 1)/nth;

    // head range for this thread
    ثابت صحیح ih0 = dh*ith;
    ثابت صحیح ih1 = MIN(ih0 + dh, nh);

    ثابت int32_t * ids = (ثابت int32_t *) src6->ڈیٹا;

    برائے (صحیح i3 = 0; i3 < ns; ++i3) {
        ثابت اعشاری * s0 = (ثابت اعشاری *) ((ثابت حرف *) src0->ڈیٹا + ids[i3]*(src0->nb[3])); // {d_state, dim, nh, ns}
              اعشاری * s  = (      اعشاری *) ((      حرف *) dst->ڈیٹا  + i3*(src0->nb[3]) + s_off); // {d_state, dim, nh, ns}

        برائے (صحیح i2 = 0; i2 < nt; ++i2) {
            ثابت اعشاری * x  = (ثابت اعشاری *) ((ثابت حرف *) src1->ڈیٹا + i2*(src1->nb[2]) + i3*(src1->nb[3])); // {dim, nh, nt, ns}
            ثابت اعشاری * dt = (ثابت اعشاری *) ((ثابت حرف *) src2->ڈیٹا + i2*(src2->nb[1]) + i3*(src2->nb[2])); // {nh, nt, ns}
            ثابت اعشاری * A  = (ثابت اعشاری *) ((ثابت حرف *) src3->ڈیٹا); // {d_state, nh} or {1, nh}
            ثابت اعشاری * B  = (ثابت اعشاری *) ((ثابت حرف *) src4->ڈیٹا + i2*(src4->nb[2]) + i3*(src4->nb[3])); // {d_state, ng, nt, ns}
            ثابت اعشاری * C  = (ثابت اعشاری *) ((ثابت حرف *) src5->ڈیٹا + i2*(src5->nb[2]) + i3*(src5->nb[3])); // {d_state, ng, nt, ns}
                  اعشاری * y  = (      اعشاری *) ((      حرف *) dst->ڈیٹا + i2*(nh*nr*حجم(اعشاری)) + i3*(nt*nh*nr*حجم(اعشاری))); // {dim, nh, nt, ns}

            اگر (src3->ne[0] == 1) {
                // Mamba-2 has a scalar decay factor per head; dA can be outside the state-wise loop

                // n_head
                برائے (صحیح h = ih0; h < ih1; ++h) {
                    // ref: https://github.com/state-spaces/mamba/blob/62db608da60f6fc790b8ed9f4b3225e95ca15fde/mamba_ssm/ops/triton/softplus.py#L16
                    ثابت اعشاری dt_soft_plus = ggml_compute_softplus_f32(dt[h]);
                    ثابت اعشاری dA = expf(dt_soft_plus * A[h]);
                    ثابت صحیح g = h / (nh / ng); // repeat_interleave

                    // dim
                    برائے (صحیح i1 = 0; i1 < nr; ++i1) {
                        ثابت صحیح ii = i1 + h*nr;
                        ثابت اعشاری x_dt = x[ii] * dt_soft_plus;
                        اعشاری sumf = 0.0f;
#اگر defined(GGML_SIMD)
    #اگر defined(__ARM_FEATURE_SVE)
                        ثابت صحیح ggml_f32_epr = svcntw();
                        ثابت صحیح ggml_f32_قدم = 1 * ggml_f32_epr;

                        ثابت صحیح np = (nc & ~(ggml_f32_قدم - 1));

                        GGML_F32_VEC مجموع = GGML_F32_VEC_ZERO;

                        GGML_F32_VEC adA = GGML_F32_VEC_SET1(dA);
                        GGML_F32_VEC axdt = GGML_F32_VEC_SET1(x_dt);

                        برائے (صحیح i = 0; i < np; i += ggml_f32_قدم) {
                            // TODO: maybe unroll more?
                            برائے (صحیح j = 0; j < 1; j++) {
                                GGML_F32_VEC t0 = GGML_F32_VEC_بھرو(s0 + i + j*ggml_f32_epr + ii*nc);
                                GGML_F32_VEC t1 = GGML_F32_VEC_بھرو(B + i + j*ggml_f32_epr + g*nc);
                                GGML_F32_VEC t2 = GGML_F32_VEC_بھرو(C + i + j*ggml_f32_epr + g*nc);

                                t0 = GGML_F32_VEC_MUL(t0, adA);
                                t1 = GGML_F32_VEC_MUL(t1, axdt);

                                t0 = GGML_F32_VEC_جوڑو(t0, t1);

                                مجموع = GGML_F32_VEC_FMA(مجموع, t0, t2);

                                GGML_F32_VEC_STORE(s + i + j*ggml_f32_epr + ii*nc, t0);
                            }
                        }

                        sumf = GGML_F32xt_REDUCE_ONE(مجموع);
    #وگرنہ_اگر defined(__riscv_v_intrinsic)
                        // todo: RVV implementation
                        ثابت صحیح np = 0;
    #ورنہ
                        ثابت صحیح np = (nc & ~(GGML_F32_قدم - 1));

                        GGML_F32_VEC مجموع[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };

                        GGML_F32_VEC adA = GGML_F32_VEC_SET1(dA);
                        GGML_F32_VEC axdt = GGML_F32_VEC_SET1(x_dt);

                        GGML_F32_VEC ax[GGML_F32_ARR];
                        GGML_F32_VEC ay[GGML_F32_ARR];
                        GGML_F32_VEC az[GGML_F32_ARR];

                        برائے (صحیح i = 0; i < np; i += GGML_F32_قدم) {
                            برائے (صحیح j = 0; j < GGML_F32_ARR; j++) {
                                ax[j] = GGML_F32_VEC_بھرو(s0 + i + j*GGML_F32_EPR + ii*nc);
                                ay[j] = GGML_F32_VEC_بھرو(B + i + j*GGML_F32_EPR + g*nc);
                                az[j] = GGML_F32_VEC_بھرو(C + i + j*GGML_F32_EPR + g*nc);

                                ax[j] = GGML_F32_VEC_MUL(ax[j], adA);
                                ay[j] = GGML_F32_VEC_MUL(ay[j], axdt);

                                ax[j] = GGML_F32_VEC_جوڑو(ax[j], ay[j]);

                                مجموع[j] = GGML_F32_VEC_FMA(مجموع[j], ax[j], az[j]);

                                GGML_F32_VEC_STORE(s + i + j*GGML_F32_EPR + ii*nc, ax[j]);
                            }
                        }

                        // reduce sum0..sum3 to sum0
                        GGML_F32_VEC_REDUCE(sumf, مجموع);
    #اگر_ختم
#ورنہ
                        ثابت صحیح np = 0;
#اگر_ختم
                        // d_state
                        برائے (صحیح i0 = np; i0 < nc; ++i0) {
                            ثابت صحیح i = i0 + ii*nc;
                            ثابت صحیح ig = i0 + g*nc;
                            // state = prev_state * dA + dB * x
                            ثابت اعشاری state = (s0[i] * dA) + (B[ig] * x_dt);
                            // y = rowwise_dotprod(state, C)
                            sumf += state * C[ig];
                            s[i] = state;
                        }
                        y[ii] = sumf;
                    }
                }
            } ورنہ {
                // Mamba-1 has an element-wise decay factor for the states

                // n_head
                برائے (صحیح h = ih0; h < ih1; ++h) {
                    // ref: https://github.com/state-spaces/mamba/blob/62db608da60f6fc790b8ed9f4b3225e95ca15fde/mamba_ssm/ops/triton/softplus.py#L16
                    ثابت اعشاری dt_soft_plus = ggml_compute_softplus_f32(dt[h]);
                    ثابت صحیح g = h / (nh / ng); // repeat_interleave

                    // dim
                    برائے (صحیح i1 = 0; i1 < nr; ++i1) {
                        ثابت صحیح ii = i1 + h*nr;
                        ثابت اعشاری x_dt = x[ii] * dt_soft_plus;
#اگر defined(__ARM_FEATURE_SVE)
                        svfloat32_t vx_dt = GGML_F32_VEC_SET1(x_dt);
                        svfloat32_t vdt_soft_plus = GGML_F32_VEC_SET1(dt_soft_plus);
                        svfloat32_t r1_سمتیہ = GGML_F32_VEC_ZERO;

                        // d_state
                        // TODO: what happens when (d_state % svcntw()) != 0?
                        برائے (int64_t k = 0; k < nc; k += svcntw()) {
                            svfloat32_t vA = GGML_F32_VEC_بھرو(&A[h*nc + k]);
                            svfloat32_t vB = GGML_F32_VEC_بھرو(&B[k + g*nc]);
                            svfloat32_t vC = GGML_F32_VEC_بھرو(&C[k + g*nc]);
                            svfloat32_t vs0 = GGML_F32_VEC_بھرو(&s0[ii*nc + k]);

                            svfloat32_t t1 = GGML_F32_VEC_MUL(vdt_soft_plus, vA);
                            t1 = exp_ps_sve(svptrue_b32(), t1);
                            svfloat32_t t2 = GGML_F32_VEC_MUL(vx_dt, vB);

                            vs0 = GGML_F32_VEC_FMA(t2, vs0, t1);
                            r1_سمتیہ = GGML_F32_VEC_جوڑو(GGML_F32_VEC_MUL(vs0, vC), r1_سمتیہ);

                            GGML_F32_VEC_STORE(&s[ii*nc + k], vs0);
                        }
                        y[ii] = GGML_F32xt_REDUCE_ONE(r1_سمتیہ);
#ورنہ
                        اعشاری sumf = 0.0f;
                        // NOTE: can't really use GGML_SIMD here because d_state is usually 16
                        //       and also because expf is used within the loop.
                        // d_state
                        برائے (صحیح i0 = 0; i0 < nc; ++i0) {
                            ثابت صحیح i = i0 + ii*nc;
                            ثابت صحیح ig = i0 + g*nc;
                            // state = prev_state * dA + dB * x
                            ثابت اعشاری state = (s0[i] * expf(dt_soft_plus * A[i0 + h*nc])) + (B[ig] * x_dt);
                            // y = rowwise_dotprod(state, C)
                            sumf += state * C[ig];
                            s[i] = state;
                        }
                        y[ii] = sumf;
#اگر_ختم
                    }
                }
            }
            ثابت int64_t خانہ = nt - 1 - i2;
            اگر (K > 1 && خانہ > 0 && خانہ < K) {
                اعشاری * s_snapshot = (اعشاری *) ((حرف *) dst->ڈیٹا + s_off + (خانہ*ns + i3)*(src0->nb[3]));
                برائے (صحیح h = ih0; h < ih1; ++h) {
                    حافظہ_نقل((حرف *) s_snapshot + h*src0->nb[2], (حرف *) s + h*src0->nb[2], src0->nb[2]);
                }
            }
            // use the output as the source when it's not the first token-wise iteration
            s0 = s;
        }
    }
}

خالی ggml_compute_forward_ssm_جائزہ(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    انتخاب (dst->src[0]->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_ssm_جائزہ_f32(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_win_part

ساکن خالی ggml_compute_forward_win_part_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    GGML_غیر_مستعمل(params);

    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)

    ثابت int32_t nep0 = ((ثابت int32_t *)(dst->op_params))[0];
    ثابت int32_t nep1 = ((ثابت int32_t *)(dst->op_params))[1];
    ثابت int32_t w    = ((ثابت int32_t *)(dst->op_params))[2];

    أكد(ne00 == ne0);
    أكد(ne3  == nep0*nep1);

    // TODO: optimize / multi-thread
    برائے (صحیح py = 0; py < nep1; ++py) {
        برائے (صحیح px = 0; px < nep0; ++px) {
            ثابت int64_t i3 = py*nep0 + px;
            برائے (int64_t i2 = 0; i2 < ne2; ++i2) {
                برائے (int64_t i1 = 0; i1 < ne1; ++i1) {
                    برائے (int64_t i0 = 0; i0 < ne0; ++i0) {
                        ثابت int64_t i02 = py*w + i2;
                        ثابت int64_t i01 = px*w + i1;
                        ثابت int64_t i00 = i0;

                        ثابت int64_t i = i3*ne2*ne1*ne0 + i2*ne1*ne0    + i1*ne0   + i0;
                        ثابت int64_t j =                  i02*ne01*ne00 + i01*ne00 + i00;

                        اگر (py*w + i2 >= ne02 || px*w + i1 >= ne01) {
                            ((اعشاری *) dst->ڈیٹا)[i] = 0.0f;
                        } ورنہ {
                            ((اعشاری *) dst->ڈیٹا)[i] = ((اعشاری *) src0->ڈیٹا)[j];
                        }
                    }
                }
            }
        }
    }
}

خالی ggml_compute_forward_win_part(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_win_part_f32(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_win_unpart

ساکن خالی ggml_compute_forward_win_unpart_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    GGML_غیر_مستعمل(params);

    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)

    ثابت int32_t w = ((ثابت int32_t *)(dst->op_params))[0];

    // padding
    ثابت صحیح px = (w - ne1%w)%w;
    //const int py = (w - ne2%w)%w;

    ثابت صحیح npx = (px + ne1)/w;
    //const int npy = (py + ne2)/w;

    أكد(ne0 == ne00);

    // TODO: optimize / multi-thread
    برائے (int64_t i2 = 0; i2 < ne2; ++i2) {
        برائے (int64_t i1 = 0; i1 < ne1; ++i1) {
            برائے (int64_t i0 = 0; i0 < ne0; ++i0) {
                ثابت صحیح ip2 = i2/w;
                ثابت صحیح ip1 = i1/w;

                ثابت int64_t i02 = i2%w;
                ثابت int64_t i01 = i1%w;
                ثابت int64_t i00 = i0;

                ثابت int64_t i = (ip2*npx + ip1)*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00 + i00;
                ثابت int64_t j =                                  i2*ne1*ne0    + i1*ne0   + i0;

                ((اعشاری *) dst->ڈیٹا)[j] = ((اعشاری *) src0->ڈیٹا)[i];
            }
        }
    }
}

خالی ggml_compute_forward_win_unpart(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_win_unpart_f32(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

//ggml_compute_forward_unary

خالی ggml_compute_forward_unary(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_unary_op op = ggml_get_unary_op(dst);

    انتخاب (op) {
        حالت GGML_UNARY_OP_ABS:
            {
                ggml_compute_forward_abs(params, dst);
            } توڑو;
        حالت GGML_UNARY_OP_SGN:
            {
                ggml_compute_forward_sgn(params, dst);
            } توڑو;
        حالت GGML_UNARY_OP_NEG:
            {
                ggml_compute_forward_neg(params, dst);
            } توڑو;
        حالت GGML_UNARY_OP_قدم:
            {
                ggml_compute_forward_قدم(params, dst);
            } توڑو;
        حالت GGML_UNARY_OP_TANH:
            {
                ggml_compute_forward_tanh(params, dst);
            } توڑو;
        حالت GGML_UNARY_OP_ELU:
            {
                ggml_compute_forward_elu(params, dst);
            } توڑو;
        حالت GGML_UNARY_OP_RELU:
            {
                ggml_compute_forward_relu(params, dst);
            } توڑو;
        حالت GGML_UNARY_OP_SIGMOID:
            {
                ggml_compute_forward_sigmoid(params, dst);
            } توڑو;
        حالت GGML_UNARY_OP_GELU:
            {
                ggml_compute_forward_gelu(params, dst);
            } توڑو;
        حالت GGML_UNARY_OP_GELU_ERF:
            {
                ggml_compute_forward_gelu_erf(params, dst);
            } توڑو;
        حالت GGML_UNARY_OP_GELU_تیز:
            {
                ggml_compute_forward_gelu_تیز(params, dst);
            } توڑو;
        حالت GGML_UNARY_OP_SILU:
            {
                ggml_compute_forward_silu(params, dst);
            } توڑو;
        حالت GGML_UNARY_OP_HARDSWISH:
            {
                ggml_compute_forward_hardswish(params, dst);
            } توڑو;
        حالت GGML_UNARY_OP_HARDSIGMOID:
            {
                ggml_compute_forward_hardsigmoid(params, dst);
            } توڑو;
        حالت GGML_UNARY_OP_EXP:
            {
                ggml_compute_forward_exp(params, dst);
            } توڑو;
        حالت GGML_UNARY_OP_FLOOR:
            {
                ggml_compute_forward_floor(params, dst);
            } توڑو;
        حالت GGML_UNARY_OP_CEIL:
            {
                ggml_compute_forward_ceil(params, dst);
            } توڑو;
        حالت GGML_UNARY_OP_باری:
            {
                ggml_compute_forward_باری(params, dst);
            } توڑو;
        حالت GGML_UNARY_OP_TRUNC:
            {
                ggml_compute_forward_trunc(params, dst);
            } توڑو;
        حالت GGML_UNARY_OP_XIELU:
            {
                ggml_compute_forward_xielu(params, dst);
            } توڑو;
        حالت GGML_UNARY_OP_EXPM1:
            {
                ggml_compute_forward_expm1(params, dst);
            } توڑو;
        حالت GGML_UNARY_OP_SOFTPLUS:
            {
                ggml_compute_forward_softplus(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

//ggml_compute_forward_glu

خالی ggml_compute_forward_glu(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_glu_op op = ggml_get_glu_op(dst);

    انتخاب (op) {
        حالت GGML_GLU_OP_REGLU:
            {
                ggml_compute_forward_reglu(params, dst);
            } توڑو;
        حالت GGML_GLU_OP_GEGLU:
            {
                ggml_compute_forward_geglu(params, dst);
            } توڑو;
        حالت GGML_GLU_OP_SWIGLU:
            {
                ggml_compute_forward_swiglu(params, dst);
            } توڑو;
        حالت GGML_GLU_OP_SWIGLU_OAI:
            {
                ggml_compute_forward_swiglu_oai(params, dst);
            } توڑو;
        حالت GGML_GLU_OP_GEGLU_ERF:
            {
                ggml_compute_forward_geglu_erf(params, dst);
            } توڑو;
        حالت GGML_GLU_OP_GEGLU_تیز:
            {
                ggml_compute_forward_geglu_تیز(params, dst);
            } توڑو;
        حالت GGML_GLU_OP_SWIGLU_CLAMP:
            {
                ggml_compute_forward_swiglu_clamp(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_get_rel_pos

ساکن خالی ggml_compute_forward_get_rel_pos_f16(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    GGML_غیر_مستعمل(params);

    ثابت ggml_tensor * src0 = dst->src[0];

    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L292-L322

    GGML_TENSOR_UNARY_OP_LOCALS

    ثابت int64_t w = ne1;

    ggml_fp16_t * src0_ڈیٹا = (ggml_fp16_t *) src0->ڈیٹا;
    ggml_fp16_t * dst_ڈیٹا  = (ggml_fp16_t *) dst->ڈیٹا;

    برائے (int64_t i2 = 0; i2 < ne2; ++i2) {
        برائے (int64_t i1 = 0; i1 < ne1; ++i1) {
            ثابت int64_t pos = (w - i1 - 1) + i2;
            برائے (int64_t i0 = 0; i0 < ne0; ++i0) {
                dst_ڈیٹا[i2*ne1*ne0 + i1*ne0 + i0] = src0_ڈیٹا[pos*ne00 + i0];
            }
        }
    }
}

خالی ggml_compute_forward_get_rel_pos(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F16:
        حالت GGML_TYPE_BF16:
            {
                ggml_compute_forward_get_rel_pos_f16(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_add_rel_pos

ساکن خالی ggml_compute_forward_جوڑو_rel_pos_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    ثابت ggml_tensor * src2 = dst->src[2];

    ثابت منطقی inplace = (منطقی) ((int32_t *) dst->op_params)[0];
    اگر (!inplace) {
        اگر (params->ith == 0) {
            حافظہ_نقل((حرف *) dst->ڈیٹا, (حرف *) src0->ڈیٹا, ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }
    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L357-L359

    اعشاری * src1_ڈیٹا = (اعشاری *) src1->ڈیٹا;
    اعشاری * src2_ڈیٹا = (اعشاری *) src2->ڈیٹا;
    اعشاری * dst_ڈیٹا  = (اعشاری *) dst->ڈیٹا;

    ثابت int64_t ne10 = src1->ne[0];
    ثابت int64_t ne11 = src1->ne[1];
    ثابت int64_t ne12 = src1->ne[2];
    ثابت int64_t ne13 = src1->ne[3];

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    // total patches in dst
    ثابت صحیح np = ne13;

    // patches per thread
    ثابت صحیح dp = (np + nth - 1)/nth;

    // patch range for this thread
    ثابت صحیح ip0 = dp*ith;
    ثابت صحیح ip1 = MIN(ip0 + dp, np);

    برائے (int64_t i13 = ip0; i13 < ip1; ++i13) {
        برائے (int64_t i12 = 0; i12 < ne12; ++i12) {
            برائے (int64_t i11 = 0; i11 < ne11; ++i11) {
                ثابت int64_t jp1 = i13*ne12*ne11*ne10 + i12*ne11*ne10 + i11*ne10;
                برائے (int64_t i10 = 0; i10 < ne10; ++i10) {
                    ثابت int64_t jp0  = jp1 + i10;
                    ثابت اعشاری src1_e = src1_ڈیٹا[jp0];
                    ثابت اعشاری src2_e = src2_ڈیٹا[jp0];

                    ثابت int64_t jdh = jp0 * ne10;
                    ثابت int64_t jdw = jdh - (ne10 - 1) * i10;

                    برائے (int64_t j = 0; j < ne10; ++j) {
                        dst_ڈیٹا[jdh + j     ] += src2_e;
                        dst_ڈیٹا[jdw + j*ne10] += src1_e;
                    }
                }
            }
        }
    }
}

خالی ggml_compute_forward_جوڑو_rel_pos(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_جوڑو_rel_pos_f32(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_rwkv_wkv6

ساکن خالی ggml_compute_forward_rwkv_wkv6_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    ثابت int64_t T = dst->src[1]->ne[2];
    ثابت int64_t C = dst->ne[0];
    ثابت int64_t HEADS = dst->src[1]->ne[1];
    ثابت int64_t n_seqs = dst->src[5]->ne[1];
    ثابت int64_t سر_سائز = C / HEADS;

    اعشاری * dst_ڈیٹا = (اعشاری *) dst->ڈیٹا;
    اعشاری * state = ((اعشاری *) dst->ڈیٹا) + C * T;

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح h_آغاز =  (HEADS * (ith    )) / nth;
    ثابت صحیح h_انتہا   = ((HEADS * (ith + 1)) / nth < HEADS) ?
                         (HEADS * (ith + 1)) / nth : HEADS;

    اعشاری * k =          (اعشاری *) dst->src[0]->ڈیٹا;
    اعشاری * v =          (اعشاری *) dst->src[1]->ڈیٹا;
    اعشاری * r =          (اعشاری *) dst->src[2]->ڈیٹا;
    اعشاری * وقت_faaaa = (اعشاری *) dst->src[3]->ڈیٹا;
    اعشاری * وقت_decay = (اعشاری *) dst->src[4]->ڈیٹا;

    size_t t_stride = HEADS * سر_سائز; // Same to C

    size_t h_stride = C / HEADS;
    GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
    size_t h_stride_2d = سر_سائز * سر_سائز;

    اگر (ith == 0) {
        memset(dst_ڈیٹا, 0, T * C * حجم(اعشاری));
    }
    ggml_barrier(params->threadpool);


    #اگر defined(__AVX__) && !defined(__AVX512F__)
        #تعریف_کرو GGML_F32X GGML_F32x8
        #تعریف_کرو GGML_F32X_SET1 GGML_F32x8_SET1
        #تعریف_کرو GGML_F32_X_بھرو GGML_F32x8_بھرو
        #تعریف_کرو GGML_F32X_STORE GGML_F32x8_STORE
        #تعریف_کرو GGML_F32X_MUL GGML_F32x8_MUL
        #تعریف_کرو GGML_F32X_FMA GGML_F32x8_FMA
        #تعریف_کرو WKV_سمتیہ_سائز 8
    #وگرنہ_اگر defined(__AVX512F__)
        #تعریف_کرو GGML_F32X GGML_F32x16
        #تعریف_کرو GGML_F32X_SET1 GGML_F32x16_SET1
        #تعریف_کرو GGML_F32_X_بھرو GGML_F32x16_بھرو
        #تعریف_کرو GGML_F32X_STORE GGML_F32x16_STORE
        #تعریف_کرو GGML_F32X_MUL GGML_F32x16_MUL
        #تعریف_کرو GGML_F32X_FMA GGML_F32x16_FMA
        #تعریف_کرو WKV_سمتیہ_سائز 16
    #وگرنہ_اگر defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
        #تعریف_کرو GGML_F32X GGML_F32xt
        #تعریف_کرو GGML_F32X_SET1 GGML_F32xt_SET1
        #تعریف_کرو GGML_F32_X_بھرو GGML_F32xt_بھرو
        #تعریف_کرو GGML_F32X_STORE GGML_F32xt_STORE
        #تعریف_کرو GGML_F32X_MUL GGML_F32xt_MUL
        #تعریف_کرو GGML_F32X_FMA GGML_F32xt_FMA
        #تعریف_کرو WKV_سمتیہ_سائز 8
    #وگرنہ_اگر defined(__ARM_NEON) && defined(__aarch64__)
        #تعریف_کرو GGML_F32X GGML_F32x4
        #تعریف_کرو GGML_F32X_SET1 GGML_F32x4_SET1
        #تعریف_کرو GGML_F32_X_بھرو GGML_F32x4_بھرو
        #تعریف_کرو GGML_F32X_STORE GGML_F32x4_STORE
        #تعریف_کرو GGML_F32X_MUL GGML_F32x4_MUL
        #تعریف_کرو GGML_F32X_FMA GGML_F32x4_FMA
        #تعریف_کرو WKV_سمتیہ_سائز 4
    #اگر_ختم

    #اگر_تعریف_شدہ WKV_سمتیہ_سائز
        صحیح wkv_سمتیہ_سائز;
        #اگر defined(__ARM_FEATURE_SVE)
            wkv_سمتیہ_سائز = svcntw();
        #ورنہ
            wkv_سمتیہ_سائز = WKV_سمتیہ_سائز;
        #اگر_ختم
        ثابت int64_t vec_گنتی = سر_سائز / wkv_سمتیہ_سائز;

        برائے (int64_t t = 0; t < T; t++) {
            size_t t_انحراف = t * t_stride;
            size_t state_انحراف = سر_سائز * C * (t / (T / n_seqs));
            اعشاری * state_cur = state + state_انحراف;
            اعشاری * state_پچھلا = t % (T / n_seqs) ? state_cur : (اعشاری*)dst->src[5]->ڈیٹا + state_انحراف;

            برائے (int64_t h = h_آغاز; h < h_انتہا; h++) {
                size_t h_انحراف = h * h_stride;
                size_t t_h_انحراف = t_انحراف + h_انحراف;
                size_t h_2d_انحراف = h * h_stride_2d;

                برائے (int64_t i = 0; i < سر_سائز; i++) {
                    size_t t_h_i_انحراف = t_h_انحراف + i;
                    size_t h_i_انحراف = h_انحراف + i;
                    size_t h_2d_i_انحراف = h_2d_انحراف + i * h_stride;

                    اعشاری k_val = k[t_h_i_انحراف];
                    اعشاری r_val = r[t_h_i_انحراف];
                    اعشاری وقت_faaaa_val = وقت_faaaa[h_i_انحراف];
                    اعشاری وقت_decay_val = وقت_decay[t_h_i_انحراف];

                    // Broadcast scalar values to vectors
                    GGML_F32X k_vec = GGML_F32X_SET1(k_val);
                    GGML_F32X r_vec = GGML_F32X_SET1(r_val);
                    GGML_F32X وقت_faaaa_vec = GGML_F32X_SET1(وقت_faaaa_val);
                    GGML_F32X وقت_decay_vec = GGML_F32X_SET1(وقت_decay_val);

                    برائے (int64_t j = 0; j < vec_گنتی; j++) {
                        size_t بنیاد_j = j * wkv_سمتیہ_سائز;
                        size_t t_h_j_انحراف = t_h_انحراف + بنیاد_j;
                        size_t h_2d_i_j_انحراف = h_2d_i_انحراف + بنیاد_j;

                        // Load x elements at once
                        GGML_F32X v_vec = GGML_F32_X_بھرو(&v[t_h_j_انحراف]);
                        GGML_F32X پچھلا_state_vec = GGML_F32_X_بھرو(&state_پچھلا[h_2d_i_j_انحراف]);
                        GGML_F32X dst_vec = GGML_F32_X_بھرو(&dst_ڈیٹا[t_h_j_انحراف]);

                        // Compute kv = v * k
                        GGML_F32X kv_vec = GGML_F32X_MUL(v_vec, k_vec);

                        // Compute temp = kv * time_faaaa + prev_state
                        GGML_F32X عارضی_vec = GGML_F32X_FMA(پچھلا_state_vec, kv_vec, وقت_faaaa_vec);

                        // Update dst: dst += temp * r
                        dst_vec = GGML_F32X_FMA(dst_vec, عارضی_vec, r_vec);
                        GGML_F32X_STORE(&dst_ڈیٹا[t_h_j_انحراف], dst_vec);

                        // Update state: state = prev_state * time_decay + kv
                        GGML_F32X new_state_vec = GGML_F32X_FMA(kv_vec, پچھلا_state_vec, وقت_decay_vec);
                        GGML_F32X_STORE(&state_cur[h_2d_i_j_انحراف], new_state_vec);
                    }

                    // Handle remaining elements, this will not be used.
                    برائے (int64_t j = vec_گنتی * wkv_سمتیہ_سائز; j < سر_سائز; j++) {
                        size_t t_h_j_انحراف = t_h_انحراف + j;
                        size_t h_2d_i_j_انحراف = h_2d_i_انحراف + j;
                        اعشاری v_val = v[t_h_j_انحراف];
                        اعشاری kv_val = v_val * k_val;
                        اعشاری پچھلا_state_val = state_پچھلا[h_2d_i_j_انحراف];
                        اعشاری عارضی_val = kv_val * وقت_faaaa_val + پچھلا_state_val;
                        dst_ڈیٹا[t_h_j_انحراف] += عارضی_val * r_val;
                        state_cur[h_2d_i_j_انحراف] = پچھلا_state_val * وقت_decay_val + kv_val;
                    }
                }
            }
        }

    #ورنہ
        // basically fused operations:
        // dst = r @ (time_faaaa * (k @ v) + state),
        // state = time_decay * state + (k @ v),
        // recursive through each token
        برائے (int64_t t = 0; t < T; t++) {
            size_t t_انحراف = t * t_stride;
            size_t state_انحراف = سر_سائز * C * (t / (T / n_seqs));
            اعشاری * state_cur = state + state_انحراف;
            اعشاری * state_پچھلا = t % (T / n_seqs) ? state_cur : (اعشاری*)dst->src[5]->ڈیٹا + state_انحراف;

            برائے (int64_t h = h_آغاز; h < h_انتہا; h++) {
                size_t h_انحراف = h * h_stride;
                size_t t_h_انحراف = t_انحراف + h_انحراف;
                size_t h_2d_انحراف = h * h_stride_2d;

                برائے (int64_t i = 0; i < سر_سائز; i++) {
                    size_t t_h_i_انحراف = t_h_انحراف + i;
                    size_t h_i_انحراف = h_انحراف + i;
                    size_t h_2d_i_انحراف = h_2d_انحراف + i * h_stride;

                    اعشاری k_val = k[t_h_i_انحراف];
                    اعشاری r_val = r[t_h_i_انحراف];
                    اعشاری وقت_faaaa_val = وقت_faaaa[h_i_انحراف];
                    // RWKV v6: different time_decay for each token.
                    اعشاری وقت_decay_val = وقت_decay[t_h_i_انحراف];

                    برائے (int64_t j = 0; j < سر_سائز; j++) {
                        size_t t_h_j_انحراف = t_h_انحراف + j;
                        size_t h_2d_i_j_انحراف = h_2d_i_انحراف + j;

                        اعشاری v_val = v[t_h_j_انحراف];
                        اعشاری kv_val = v_val * k_val;
                        اعشاری پچھلا_state_val = state_پچھلا[h_2d_i_j_انحراف];
                        اعشاری عارضی_val = kv_val * وقت_faaaa_val + پچھلا_state_val;
                        dst_ڈیٹا[t_h_j_انحراف] += عارضی_val * r_val;
                        state_cur[h_2d_i_j_انحراف] = پچھلا_state_val * وقت_decay_val + kv_val;
                    }
                }
            }
        }
    #اگر_ختم
}


خالی ggml_compute_forward_rwkv_wkv6(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_rwkv_wkv6_f32(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_gla

ساکن خالی ggml_compute_forward_gla_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    ثابت int64_t T = dst->src[1]->ne[2];
    ثابت int64_t C = dst->ne[0];
    ثابت int64_t HEADS = dst->src[1]->ne[1];
    ثابت int64_t n_seqs = dst->src[4]->ne[1];
    ثابت int64_t سر_سائز = C / HEADS;
    ثابت اعشاری scale = ggml_get_op_params_f32(dst, 0);

    اعشاری * dst_ڈیٹا = (اعشاری *) dst->ڈیٹا;
    اعشاری * state = ((اعشاری *) dst->ڈیٹا) + C * T;

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح h_آغاز =  (HEADS * (ith    )) / nth;
    ثابت صحیح h_انتہا   = ((HEADS * (ith + 1)) / nth < HEADS) ?
                         (HEADS * (ith + 1)) / nth : HEADS;

    اعشاری * k = (اعشاری *) dst->src[0]->ڈیٹا;
    اعشاری * v = (اعشاری *) dst->src[1]->ڈیٹا;
    اعشاری * q = (اعشاری *) dst->src[2]->ڈیٹا;
    اعشاری * g = (اعشاری *) dst->src[3]->ڈیٹا;

    size_t t_stride = HEADS * سر_سائز; // Same to C

    size_t h_stride = C / HEADS;
    GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
    size_t h_stride_2d = سر_سائز * سر_سائز;

    اگر (ith == 0) {
        memset(dst_ڈیٹا, 0, T * C * حجم(اعشاری));
    }
    ggml_barrier(params->threadpool);


    #اگر defined(__AVX__) && !defined(__AVX512F__)
        #تعریف_کرو GGML_F32X GGML_F32x8
        #تعریف_کرو GGML_F32X_SET1 GGML_F32x8_SET1
        #تعریف_کرو GGML_F32_X_بھرو GGML_F32x8_بھرو
        #تعریف_کرو GGML_F32X_STORE GGML_F32x8_STORE
        #تعریف_کرو GGML_F32X_MUL GGML_F32x8_MUL
        #تعریف_کرو GGML_F32X_FMA GGML_F32x8_FMA
        #تعریف_کرو GLA_سمتیہ_سائز 8
    #وگرنہ_اگر defined(__AVX512F__)
        #تعریف_کرو GGML_F32X GGML_F32x16
        #تعریف_کرو GGML_F32X_SET1 GGML_F32x16_SET1
        #تعریف_کرو GGML_F32_X_بھرو GGML_F32x16_بھرو
        #تعریف_کرو GGML_F32X_STORE GGML_F32x16_STORE
        #تعریف_کرو GGML_F32X_MUL GGML_F32x16_MUL
        #تعریف_کرو GGML_F32X_FMA GGML_F32x16_FMA
        #تعریف_کرو GLA_سمتیہ_سائز 16
    #وگرنہ_اگر defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
        #تعریف_کرو GGML_F32X GGML_F32xt
        #تعریف_کرو GGML_F32X_SET1 GGML_F32xt_SET1
        #تعریف_کرو GGML_F32_X_بھرو GGML_F32xt_بھرو
        #تعریف_کرو GGML_F32X_STORE GGML_F32xt_STORE
        #تعریف_کرو GGML_F32X_MUL GGML_F32xt_MUL
        #تعریف_کرو GGML_F32X_FMA GGML_F32xt_FMA
        #تعریف_کرو GLA_سمتیہ_سائز 8
    #وگرنہ_اگر defined(__ARM_NEON) && defined(__aarch64__)
        #تعریف_کرو GGML_F32X GGML_F32x4
        #تعریف_کرو GGML_F32X_SET1 GGML_F32x4_SET1
        #تعریف_کرو GGML_F32_X_بھرو GGML_F32x4_بھرو
        #تعریف_کرو GGML_F32X_STORE GGML_F32x4_STORE
        #تعریف_کرو GGML_F32X_MUL GGML_F32x4_MUL
        #تعریف_کرو GGML_F32X_FMA GGML_F32x4_FMA
        #تعریف_کرو GLA_سمتیہ_سائز 4
    #اگر_ختم

    #اگر_تعریف_شدہ GLA_سمتیہ_سائز
        صحیح gla_سمتیہ_سائز;
        #اگر defined(__ARM_FEATURE_SVE)
            gla_سمتیہ_سائز = svcntw();
        #ورنہ
            gla_سمتیہ_سائز = GLA_سمتیہ_سائز;
        #اگر_ختم
        ثابت int64_t vec_گنتی = سر_سائز / gla_سمتیہ_سائز;

        برائے (int64_t t = 0; t < T; t++) {
            size_t t_انحراف = t * t_stride;
            size_t state_انحراف = سر_سائز * C * (t / (T / n_seqs));
            اعشاری * state_cur = state + state_انحراف;
            اعشاری * state_پچھلا = t % (T / n_seqs) ? state_cur : (اعشاری*)dst->src[4]->ڈیٹا + state_انحراف;

            برائے (int64_t h = h_آغاز; h < h_انتہا; h++) {
                size_t h_انحراف = h * h_stride;
                size_t t_h_انحراف = t_انحراف + h_انحراف;
                size_t h_2d_انحراف = h * h_stride_2d;

                برائے (int64_t i = 0; i < سر_سائز; i++) {
                    size_t t_h_i_انحراف = t_h_انحراف + i;
                    size_t h_2d_i_انحراف = h_2d_انحراف + i * h_stride;

                    اعشاری k_val = k[t_h_i_انحراف];
                    اعشاری q_val = q[t_h_i_انحراف] * scale;
                    اعشاری g_val = g[t_h_i_انحراف];

                    // Broadcast scalar values to vectors
                    GGML_F32X k_vec = GGML_F32X_SET1(k_val);
                    GGML_F32X q_vec = GGML_F32X_SET1(q_val);
                    GGML_F32X g_vec = GGML_F32X_SET1(g_val);

                    برائے (int64_t j = 0; j < vec_گنتی; j++) {
                        size_t بنیاد_j = j * gla_سمتیہ_سائز;
                        size_t t_h_j_انحراف = t_h_انحراف + بنیاد_j;
                        size_t h_2d_i_j_انحراف = h_2d_i_انحراف + بنیاد_j;

                        // Load x elements at once
                        GGML_F32X v_vec = GGML_F32_X_بھرو(&v[t_h_j_انحراف]);
                        GGML_F32X پچھلا_state_vec = GGML_F32_X_بھرو(&state_پچھلا[h_2d_i_j_انحراف]);
                        GGML_F32X dst_vec = GGML_F32_X_بھرو(&dst_ڈیٹا[t_h_j_انحراف]);

                        // Compute kv = v * k
                        GGML_F32X kv_vec = GGML_F32X_MUL(v_vec, k_vec);

                        // Compute temp = prev_state * g + kv
                        GGML_F32X عارضی_vec = GGML_F32X_FMA(kv_vec, پچھلا_state_vec, g_vec);

                        // Update dst: dst += temp * q
                        dst_vec = GGML_F32X_FMA(dst_vec, عارضی_vec, q_vec);
                        GGML_F32X_STORE(&dst_ڈیٹا[t_h_j_انحراف], dst_vec);

                        // Update state
                        GGML_F32X_STORE(&state_cur[h_2d_i_j_انحراف], عارضی_vec);
                    }

                    // Handle remaining elements, this will not be used.
                    برائے (int64_t j = vec_گنتی * gla_سمتیہ_سائز; j < سر_سائز; j++) {
                        size_t t_h_j_انحراف = t_h_انحراف + j;
                        size_t h_2d_i_j_انحراف = h_2d_i_انحراف + j;
                        اعشاری v_val = v[t_h_j_انحراف];
                        اعشاری kv_val = v_val * k_val;
                        اعشاری پچھلا_state_val = state_پچھلا[h_2d_i_j_انحراف];
                        اعشاری عارضی_val = kv_val + پچھلا_state_val * g_val;
                        dst_ڈیٹا[t_h_j_انحراف] += عارضی_val * q_val;
                        state_cur[h_2d_i_j_انحراف] = عارضی_val;
                    }
                }
            }
        }

    #ورنہ
        برائے (int64_t t = 0; t < T; t++) {
            size_t t_انحراف = t * t_stride;
            size_t state_انحراف = سر_سائز * C * (t / (T / n_seqs));
            اعشاری * state_cur = state + state_انحراف;
            اعشاری * state_پچھلا = t % (T / n_seqs) ? state_cur : (اعشاری*)dst->src[4]->ڈیٹا + state_انحراف;

            برائے (int64_t h = h_آغاز; h < h_انتہا; h++) {
                size_t h_انحراف = h * h_stride;
                size_t t_h_انحراف = t_انحراف + h_انحراف;
                size_t h_2d_انحراف = h * h_stride_2d;

                برائے (int64_t i = 0; i < سر_سائز; i++) {
                    size_t t_h_i_انحراف = t_h_انحراف + i;
                    size_t h_2d_i_انحراف = h_2d_انحراف + i * h_stride;

                    اعشاری k_val = k[t_h_i_انحراف];
                    اعشاری q_val = q[t_h_i_انحراف] * scale;
                    اعشاری g_val = g[t_h_i_انحراف];

                    برائے (int64_t j = 0; j < سر_سائز; j++) {
                        size_t t_h_j_انحراف = t_h_انحراف + j;
                        size_t h_2d_i_j_انحراف = h_2d_i_انحراف + j;

                        اعشاری v_val = v[t_h_j_انحراف];
                        اعشاری kv_val = v_val * k_val;
                        اعشاری پچھلا_state_val = state_پچھلا[h_2d_i_j_انحراف];
                        اعشاری عارضی_val = پچھلا_state_val * g_val + kv_val;
                        dst_ڈیٹا[t_h_j_انحراف] += عارضی_val * q_val;
                        state_cur[h_2d_i_j_انحراف] = عارضی_val;
                    }
                }
            }
        }
    #اگر_ختم
}


خالی ggml_compute_forward_gla(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_gla_f32(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ساکن خالی ggml_compute_forward_solve_tri_f32(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst) {
    ثابت ساخت ggml_tensor * src0 = dst->src[0];  // A (lower triangular)
    ثابت ساخت ggml_tensor * src1 = dst->src[1];  // B (RHS)

    GGML_TENSOR_ثنائی_OP_LOCALS;

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(ne00 == ne01); // A must be square
    GGML_ASSERT(ne0  == ne10); // solution cols == B cols
    GGML_ASSERT(ne1  == ne11); // solution rows == B rows

    GGML_ASSERT(ne02 == ne12 && ne12 == ne2);
    GGML_ASSERT(ne03 == ne13 && ne13 == ne3);

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت int64_t k = ne10;   // number of RHS columns
    ثابت int64_t n = ne11;   // A is n×n
    ثابت int64_t nr = ne02 * ne03 * k; // we're parallelizing on columns here, so seq x token x column will be the unit

    // chunks per thread
    ثابت int64_t dr = (nr + nth - 1)/nth;

    // chunk range for this thread
    ثابت int64_t ir0 = dr*ith;
    ثابت int64_t ir1 = MIN(ir0 + dr, nr);

    ثابت اعشاری * A = (ثابت اعشاری *) src0->ڈیٹا;  // [n, n, B1, B2]
    ثابت اعشاری * B = (ثابت اعشاری *) src1->ڈیٹا;  // [n, k, B1, B2]
          اعشاری * X = (      اعشاری *) dst->ڈیٹا;   // [n, k, B1, B2]

    برائے (int64_t ir = ir0; ir < ir1; ++ir) {
        ثابت int64_t i03 = ir/(ne02*k);
        ثابت int64_t i02 = (ir - i03*ne02*k)/k;
        ثابت int64_t i01 = (ir - i03*ne02*k - i02*k);

        ثابت اعشاری * A_batch = A + i02 * nb02 / حجم(اعشاری) + i03 * nb03 / حجم(اعشاری);
        ثابت اعشاری * B_batch = B + i02 * nb12 / حجم(اعشاری) + i03 * nb13 / حجم(اعشاری);

        اعشاری * X_batch = X + i02 * nb2 / حجم(اعشاری) + i03 * nb3 / حجم(اعشاری);

        برائے (int64_t i00 = 0; i00 < n; ++i00) {
            اعشاری مجموع = 0.0f;
            برائے (int64_t t = 0; t < i00; ++t) {
                مجموع += A_batch[i00 * n + t] * X_batch[t * k + i01];
            }

            ثابت اعشاری diag = A_batch[i00 * n + i00];
            أكد(diag != 0.0f && "Zero diagonal in triangular matrix");

            X_batch[i00 * k + i01] = (B_batch[i00 * k + i01] - مجموع) / diag;
        }
    }
}

خالی ggml_compute_forward_solve_tri(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    اگر (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        ggml_compute_forward_solve_tri_f32(params, dst);
    } ورنہ {
        GGML_ABORT("fatal error");
    }
}

// ggml_compute_forward_gated_delta_net
ساکن خالی ggml_compute_forward_gated_delta_net_one_chunk(
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
    ثابت int64_t n_علامات = src_v->ne[2];
    ثابت int64_t n_seqs   = src_v->ne[3];

    GGML_ASSERT(ggml_is_contiguous_صفیں(src_q));
    GGML_ASSERT(ggml_is_contiguous_صفیں(src_k));
    GGML_ASSERT(ggml_is_contiguous_صفیں(src_v));
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

    ثابت منطقی kda = (neg0 == S_v);

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    ثابت int64_t K = ggml_get_op_params_i32(dst, 0);
    GGML_ASSERT(K >= 1);
    // per-seq stride in floats (seq s starts at state + s * seq_stride)
    ثابت int64_t state_seq_stride = src_state->nb[3] / حجم(اعشاری);

    ثابت int64_t فی_thread = S_v + (K > 1 ? S_v * S_v : 0);
    ثابت صحیح ith = params->ith;

    اعشاری * delta       = (اعشاری *)params->wdata + ith * فی_thread + CACHE_سطر_سائز_F32;
    اعشاری * state_work  = K > 1 ? (delta + S_v) : خالی_اشاریہ;

    // output layout: [attn_scores | new_states]
    // attn_scores: S_v * H * n_tokens * n_seqs    floats
    // new_states:  S_v * S_v * H * n_seqs * K     floats  (K snapshot slots; last min(n_tokens, K))
    ثابت int64_t attn_score_elems    = S_v * H * n_علامات * n_seqs;
    ثابت int64_t state_سائز_فی_snap = S_v * S_v * H * n_seqs;
    اعشاری * attn_برآمد_بنیاد  = (اعشاری *)dst->ڈیٹا;
    اعشاری * state_برآمد_بنیاد = (اعشاری *)dst->ڈیٹا + attn_score_elems;

    // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
    // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.

    ثابت اعشاری * state_اندر_بنیاد = (ثابت اعشاری *)src_state->ڈیٹا;

  //const int64_t rq1 = nev1 / neq1;
  //const int64_t rk1 = nev1 / nek1;
    ثابت int64_t rq3 = nev3 / neq3;
    ثابت int64_t rk3 = nev3 / nek3;

    ثابت اعشاری scale = 1.0f / sqrtf((اعشاری) S_v);

    برائے (int64_t ir = ir0; ir < ir1; ++ir) {
        ثابت int64_t iv1 = ir % H; // head_index
        ثابت int64_t iv3 = ir / H; // sequence

        ثابت int64_t iq1 = iv1 % neq1;
        ثابت int64_t ik1 = iv1 % nek1;

        ثابت int64_t iq3 = iv3 / rq3;
        ثابت int64_t ik3 = iv3 / rk3;

        // For K=1, write directly to the single output slot to avoid an extra memcpy at the end.
        // For K>1, work in scratch and copy out per-token when the slot is in range.
        اعشاری * s_برآمد = (K > 1)
            ? state_work
            : state_برآمد_بنیاد + (iv3 * H + iv1) * S_v * S_v;

        // copy input state into the working buffer and operate in-place
        // state layout [S_v, S_v, H, n_seqs]: seq iv3 starts at iv3 * state_seq_stride.
        ثابت اعشاری * s_اندر = state_اندر_بنیاد + iv3 * state_seq_stride + iv1 * S_v * S_v;
        حافظہ_نقل(s_برآمد, s_اندر, S_v * S_v * حجم(اعشاری));

        // attn output pointer for first token of this (head, seq)
        اعشاری * attn_ڈیٹا = attn_برآمد_بنیاد + (iv3 * n_علامات * H + iv1) * S_v;

        برائے (int64_t t = 0; t < n_علامات; t++) {
            ثابت اعشاری * q_d = (ثابت اعشاری *)((ثابت حرف *)src_q->ڈیٹا + iq3 * nbq3 + t * nbq2 + iq1 * nbq1);
            ثابت اعشاری * k_d = (ثابت اعشاری *)((ثابت حرف *)src_k->ڈیٹا + ik3 * nbk3 + t * nbk2 + ik1 * nbk1);
            ثابت اعشاری * v_d = (ثابت اعشاری *)((ثابت حرف *)src_v->ڈیٹا + iv3 * nbv3 + t * nbv2 + iv1 * nbv1);

            ثابت اعشاری beta_val = *(ثابت اعشاری *)((ثابت حرف *)src_beta->ڈیٹا + iv3 * nbb3 + t * nbb2 + iv1 * nbb1);
            ثابت اعشاری * g_d    =  (ثابت اعشاری *)((ثابت حرف *)src_g->ڈیٹا    + iv3 * nbg3 + t * nbg2 + iv1 * nbg1);

            // state is stored transposed: s_out[j*S_v + i] = S[i][j]
            // so row j of s_out = column j of S (contiguous access)

            اگر (kda) {
                // precompute exp(g) into delta scratch (reused below)
                برائے (int64_t i = 0; i < S_v; ++i) {
                    delta[i] = expf(g_d[i]);
                }
                // S[i][:] *= exp(g[i]) => for each row j of M: M[j][i] *= exp(g[i])
                برائے (int64_t j = 0; j < S_v; ++j) {
                    ggml_vec_mul_f32(S_v, &s_برآمد[j * S_v], &s_برآمد[j * S_v], delta);
                }
            } ورنہ {
                ggml_vec_scale_f32(S_v * S_v, s_برآمد, expf(g_d[0]));
            }

            // delta[j] = sum_i S[i][j] * k[i] = dot(row j of M, k)
            برائے (int64_t j = 0; j < S_v; ++j) {
                اعشاری مجموع = 0.0f;
                ggml_vec_dot_f32(S_v, &مجموع, 0, &s_برآمد[j * S_v], 0, k_d, 0, 1);
                delta[j] = (v_d[j] - مجموع) * beta_val;
            }

            // outer product: S[i][j] += k[i] * delta[j] => M[j][i] += delta[j] * k[i]
            برائے (int64_t j = 0; j < S_v; ++j) {
                ggml_vec_mad_f32(S_v, &s_برآمد[j * S_v], k_d, delta[j]);
            }

            // attn_out[j] = sum_i S[i][j] * q[i] = dot(row j of M, q)
            برائے (int64_t j = 0; j < S_v; ++j) {
                اعشاری مجموع = 0.0f;
                ggml_vec_dot_f32(S_v, &مجموع, 0, &s_برآمد[j * S_v], 0, q_d, 0, 1);
                attn_ڈیٹا[j] = مجموع * scale;
            }

            attn_ڈیٹا += S_v * H; // advance to next token

            اگر (K > 1) {
                ثابت int64_t ہدف_خانہ = n_علامات - 1 - t;
                اگر (ہدف_خانہ >= 0 && ہدف_خانہ < K) {
                    اعشاری * curr_state_o = state_برآمد_بنیاد + ہدف_خانہ * state_سائز_فی_snap +
                                     (iv3 * H + iv1) * S_v * S_v;
                    حافظہ_نقل(curr_state_o, s_برآمد, S_v * S_v * حجم(اعشاری));
                }
            }
        }
    }
}


ساکن خالی ggml_compute_forward_gated_delta_net_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ggml_tensor * V = dst->src[2];
    int64_t nr = V->ne[1] * V->ne[3];

    // disable for NUMA
    ثابت منطقی disable_chunking = ggml_is_numa();

    صحیح nth = params->nth;
    صحیح ith = params->ith;

    // 4x chunks per thread
    صحیح nth_scaled = nth * 4;
    int64_t chunk_سائز = (nr + nth_scaled - 1) / nth_scaled;
    int64_t nchunk     = (nr + chunk_سائز - 1) / chunk_سائز;

    اگر (nth == 1 || nchunk < nth || disable_chunking) {
      nchunk = nth;
    }

    اگر (ith == 0) {
      ggml_threadpool_chunk_سیٹ(params->threadpool, nth);
    }

    ggml_barrier(params->threadpool);

    ثابت int64_t dr = (nr + nchunk - 1) / nchunk;

    صحیح current_chunk = ith;

    جب (current_chunk < nchunk) {
        ثابت int64_t ir0 = dr * current_chunk;
        ثابت int64_t ir1 = MIN(ir0 + dr, nr);

        ggml_compute_forward_gated_delta_net_one_chunk(params, dst, ir0, ir1);
        current_chunk = ggml_threadpool_chunk_جوڑو(params->threadpool, 1);
    }
}

خالی ggml_compute_forward_gated_delta_net(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_gated_delta_net_f32(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}


// ggml_compute_forward_dsv4_hc_comb

ساکن خالی ggml_dsv4_hc_comb_norm_ستون_تعداد(اعشاری * comb, اعشاری eps) {
    constexpr int64_t hc = 4;

    برائے (int64_t idst = 0; idst < hc; ++idst) {
        اعشاری مجموع = eps;
        برائے (int64_t isrc = 0; isrc < hc; ++isrc) {
            مجموع += comb[idst + hc*isrc];
        }

        ثابت اعشاری inv_مجموع = 1.0f / مجموع;
        برائے (int64_t isrc = 0; isrc < hc; ++isrc) {
            comb[idst + hc*isrc] *= inv_مجموع;
        }
    }
}

ساکن خالی ggml_dsv4_hc_comb_norm_صفیں(اعشاری * comb, اعشاری eps) {
    constexpr int64_t hc = 4;

    برائے (int64_t isrc = 0; isrc < hc; ++isrc) {
        اعشاری مجموع = eps;
        برائے (int64_t idst = 0; idst < hc; ++idst) {
            مجموع += comb[idst + hc*isrc];
        }

        ثابت اعشاری inv_مجموع = 1.0f / مجموع;
        برائے (int64_t idst = 0; idst < hc; ++idst) {
            comb[idst + hc*isrc] *= inv_مجموع;
        }
    }
}

ساکن خالی ggml_compute_forward_dsv4_hc_comb_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    ثابت ggml_tensor * mixes = dst->src[0];
    ثابت ggml_tensor * scale = dst->src[1];
    ثابت ggml_tensor * بنیاد  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(بنیاد->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    constexpr int64_t hc = 4;
    constexpr int64_t comb_انحراف = 2*hc;
    constexpr int64_t hc_mix_dim = (2 + hc)*hc;

    ثابت int64_t n_علامات = mixes->ne[1];

    GGML_ASSERT(mixes->ne[0] == hc_mix_dim);
    GGML_ASSERT(dst->ne[0] == hc);
    GGML_ASSERT(dst->ne[1] == hc);
    GGML_ASSERT(dst->ne[2] == n_علامات);
    GGML_ASSERT(scale->ne[0] >= 3);
    GGML_ASSERT(بنیاد->ne[0] == hc_mix_dim);

    GGML_TENSOR_LOCALS(size_t, nbm, mixes, nb);
    GGML_TENSOR_LOCALS(size_t, nbs, scale, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, بنیاد,  nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,   nb);

    ثابت اعشاری eps = ggml_get_op_params_f32(dst, 0);
    ثابت int32_t n_iter = ggml_get_op_params_i32(dst, 1);
    GGML_ASSERT(n_iter > 0);

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت int64_t dr  = (n_علامات + nth - 1) / nth;
    ثابت int64_t it0 = dr * ith;
    ثابت int64_t it1 = MIN(it0 + dr, n_علامات);

    ثابت اعشاری scale_comb = *(ثابت اعشاری *) ((ثابت حرف *) scale->ڈیٹا + 2*nbs0);

    برائے (int64_t it = it0; it < it1; ++it) {
        اعشاری comb[hc*hc];

        برائے (int64_t isrc = 0; isrc < hc; ++isrc) {
            اعشاری max = -INFINITY;
            برائے (int64_t idst = 0; idst < hc; ++idst) {
                ثابت int64_t idx = idst + hc*isrc;
                ثابت اعشاری xv = *(ثابت اعشاری *) ((ثابت حرف *) mixes->ڈیٹا + (comb_انحراف + idx)*nbm0 + it*nbm1);
                ثابت اعشاری bv = *(ثابت اعشاری *) ((ثابت حرف *) بنیاد->ڈیٹا  + (comb_انحراف + idx)*nbb0);
                ثابت اعشاری v = xv * scale_comb + bv;
                comb[idx] = v;
                max = MAX(max, v);
            }

            اعشاری مجموع = 0.0f;
            برائے (int64_t idst = 0; idst < hc; ++idst) {
                ثابت int64_t idx = idst + hc*isrc;
                ثابت اعشاری v = expf(comb[idx] - max);
                comb[idx] = v;
                مجموع += v;
            }

            ثابت اعشاری inv_مجموع = 1.0f / مجموع;
            برائے (int64_t idst = 0; idst < hc; ++idst) {
                ثابت int64_t idx = idst + hc*isrc;
                comb[idx] = comb[idx] * inv_مجموع + eps;
            }
        }

        ggml_dsv4_hc_comb_norm_ستون_تعداد(comb, eps);
        برائے (int32_t i = 1; i < n_iter; ++i) {
            ggml_dsv4_hc_comb_norm_صفیں(comb, eps);
            ggml_dsv4_hc_comb_norm_ستون_تعداد(comb, eps);
        }

        برائے (int64_t isrc = 0; isrc < hc; ++isrc) {
            برائے (int64_t idst = 0; idst < hc; ++idst) {
                ثابت int64_t idx = idst + hc*isrc;
                *(اعشاری *) ((حرف *) dst->ڈیٹا + idst*nbd0 + isrc*nbd1 + it*nbd2) = comb[idx];
            }
        }
    }
}

خالی ggml_compute_forward_dsv4_hc_comb(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_dsv4_hc_comb_f32(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_dsv4_hc_pre

ساکن خالی ggml_compute_forward_dsv4_hc_pre_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    ثابت ggml_tensor * x       = dst->src[0];
    ثابت ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ثابت int64_t n_embd   = x->ne[0];
    ثابت int64_t hc       = x->ne[1];
    ثابت int64_t n_علامات = x->ne[2];

    ثابت اعشاری scale = ggml_get_op_params_f32(dst, 0);
    ثابت منطقی  gated = ggml_get_op_params_i32(dst, 1) != 0;

    GGML_ASSERT(dst->ne[0] == n_embd);
    GGML_ASSERT(dst->ne[1] == n_علامات);
    اگر (gated) {
        GGML_ASSERT(weights->ne[0] == n_embd);
        GGML_ASSERT(weights->ne[1] == hc);
        GGML_ASSERT(weights->ne[2] == n_علامات);
    } ورنہ {
        GGML_ASSERT(weights->ne[0] == hc);
        GGML_ASSERT(weights->ne[1] == n_علامات);
    }

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت int64_t nr  = n_embd * n_علامات;
    ثابت int64_t dr  = (nr + nth - 1) / nth;
    ثابت int64_t ir0 = dr * ith;
    ثابت int64_t ir1 = MIN(ir0 + dr, nr);

    برائے (int64_t ir = ir0; ir < ir1; ++ir) {
        ثابت int64_t i0 = ir % n_embd;
        ثابت int64_t it = ir / n_embd;

        اعشاری مجموع = 0.0f;
        برائے (int64_t ih = 0; ih < hc; ++ih) {
            ثابت اعشاری xv = *(ثابت اعشاری *) ((ثابت حرف *) x->ڈیٹا + i0*nbx0 + ih*nbx1 + it*nbx2);
            اعشاری wv;
            اگر (gated) {
                ثابت اعشاری gv = *(ثابت اعشاری *) ((ثابت حرف *) weights->ڈیٹا + i0*nbw0 + ih*nbw1 + it*nbw2);
                wv = 1.0f / (1.0f + expf(-gv));
            } ورنہ {
                wv = *(ثابت اعشاری *) ((ثابت حرف *) weights->ڈیٹا + ih*nbw0 + it*nbw1);
            }
            مجموع += xv * wv;
        }

        *(اعشاری *) ((حرف *) dst->ڈیٹا + i0*nbd0 + it*nbd1) = scale * مجموع;
    }
}

خالی ggml_compute_forward_dsv4_hc_pre(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_dsv4_hc_pre_f32(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_dsv4_hc_post

ساکن خالی ggml_compute_forward_dsv4_hc_post_f32(
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
    ثابت int64_t n_علامات = x->ne[1];
    ثابت int64_t hc       = residual->ne[1];

    GGML_ASSERT(dst->ne[0] == n_embd);
    GGML_ASSERT(dst->ne[1] == hc);
    GGML_ASSERT(dst->ne[2] == n_علامات);
    GGML_ASSERT(residual->ne[0] == n_embd);
    GGML_ASSERT(residual->ne[2] == n_علامات);
    GGML_ASSERT(post->ne[0] == hc);
    GGML_ASSERT(post->ne[1] == n_علامات);

    // comb == NULL: identity mixing, each stream keeps its own residual
    size_t nbc0 = 0;
    size_t nbc1 = 0;
    size_t nbc2 = 0;
    اگر (comb) {
        GGML_ASSERT(comb->type == GGML_TYPE_F32);
        GGML_ASSERT(comb->ne[0] == hc);
        GGML_ASSERT(comb->ne[1] == hc);
        GGML_ASSERT(comb->ne[2] == n_علامات);
        nbc0 = comb->nb[0];
        nbc1 = comb->nb[1];
        nbc2 = comb->nb[2];
    }

    GGML_TENSOR_LOCALS(size_t, nbx, x,        nb);
    GGML_TENSOR_LOCALS(size_t, nbr, residual, nb);
    GGML_TENSOR_LOCALS(size_t, nbp, post,     nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,      nb);

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت int64_t nr  = n_embd * hc * n_علامات;
    ثابت int64_t dr  = (nr + nth - 1) / nth;
    ثابت int64_t ir0 = dr * ith;
    ثابت int64_t ir1 = MIN(ir0 + dr, nr);

    برائے (int64_t ir = ir0; ir < ir1; ++ir) {
        ثابت int64_t i0     = ir % n_embd;
        ثابت int64_t idst   = (ir / n_embd) % hc;
        ثابت int64_t it     = ir / (n_embd * hc);

        ثابت اعشاری xv = *(ثابت اعشاری *) ((ثابت حرف *) x->ڈیٹا    + i0*nbx0 + it*nbx1);
        ثابت اعشاری pv = *(ثابت اعشاری *) ((ثابت حرف *) post->ڈیٹا + idst*nbp0 + it*nbp1);

        اعشاری مجموع = xv * pv;
        اگر (comb) {
            برائے (int64_t isrc = 0; isrc < hc; ++isrc) {
                ثابت اعشاری rv = *(ثابت اعشاری *) ((ثابت حرف *) residual->ڈیٹا + i0*nbr0 + isrc*nbr1 + it*nbr2);
                ثابت اعشاری cv = *(ثابت اعشاری *) ((ثابت حرف *) comb->ڈیٹا     + idst*nbc0 + isrc*nbc1 + it*nbc2);
                مجموع += rv * cv;
            }
        } ورنہ {
            مجموع += *(ثابت اعشاری *) ((ثابت حرف *) residual->ڈیٹا + i0*nbr0 + idst*nbr1 + it*nbr2);
        }

        *(اعشاری *) ((حرف *) dst->ڈیٹا + i0*nbd0 + idst*nbd1 + it*nbd2) = مجموع;
    }
}

خالی ggml_compute_forward_dsv4_hc_post(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_dsv4_hc_post_f32(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_top_k_mask

خالی ggml_compute_forward_چوٹی_k_نقاب(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    ثابت ggml_tensor * نقاب   = dst->src[0];
    ثابت ggml_tensor * چوٹی_k  = dst->src[1];
    ثابت ggml_tensor * score  = dst->src[2];
    ثابت ggml_tensor * بلاکیں = dst->src[3];
    ثابت ggml_tensor * extra  = dst->src[4];

    GGML_ASSERT(dst->type == نقاب->type);
    GGML_ASSERT(dst->nb[0] == ggml_type_سائز(dst->type));

    ثابت int64_t n_kv    = نقاب->ne[0];
    ثابت int64_t ne1     = dst->ne[1];
    ثابت int64_t ne2     = dst->ne[2];
    ثابت int64_t n_صفیں  = ggml_nrows(dst);
    ثابت int64_t n_چوٹی   = چوٹی_k->ne[0];
    ثابت int64_t n_blk   = بلاکیں ? بلاکیں->ne[0] : 1;
    ثابت int64_t n_pick  = بلاکیں ? بلاکیں->ne[1] : score ? score->ne[0] : n_kv;
    ثابت int64_t n_extra = extra ? extra->ne[0] : 0;

    ثابت size_t ts = ggml_type_سائز(dst->type);

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    برائے (int64_t ir = ith; ir < n_صفیں; ir += nth) {
        ثابت int64_t i1 = ir % ne1;
        ثابت int64_t i2 = (ir / ne1) % ne2;
        ثابت int64_t i3 = ir / (ne1*ne2);

        ثابت حرف * نقاب_صف = (ثابت حرف *) نقاب->ڈیٹا
            + (i1 % نقاب->ne[1])*نقاب->nb[1] + (i2 % نقاب->ne[2])*نقاب->nb[2] + (i3 % نقاب->ne[3])*نقاب->nb[3];
        حرف * dst_صف = (حرف *) dst->ڈیٹا + i1*dst->nb[1] + i2*dst->nb[2] + i3*dst->nb[3];

        اگر (dst->type == GGML_TYPE_F16) {
            ثابت ggml_fp16_t ninf = GGML_CPU_FP32_مقصد_FP16(-INFINITY);
            برائے (int64_t c = 0; c < n_kv; ++c) {
                ((ggml_fp16_t *) dst_صف)[c] = ninf;
            }
        } ورنہ {
            برائے (int64_t c = 0; c < n_kv; ++c) {
                ((اعشاری *) dst_صف)[c] = -INFINITY;
            }
        }

        خودکار keep = [&](int32_t c) {
            اگر (c >= 0 && c < n_kv) {
                حافظہ_نقل(dst_صف + c*ts, نقاب_صف + c*ts, ts);
            }
        };

        ثابت حرف * چوٹی_k_صف = (ثابت حرف *) چوٹی_k->ڈیٹا + i1*چوٹی_k->nb[1] + i2*چوٹی_k->nb[2] + i3*چوٹی_k->nb[3];

        برائے (int64_t j = 0; j < n_چوٹی; ++j) {
            ثابت int32_t p = *(ثابت int32_t *) (چوٹی_k_صف + j*چوٹی_k->nb[0]);
            اگر (p < 0 || p >= n_pick) {
                جاری;
            }
            اگر (score) {
                ثابت اعشاری s = *(ثابت اعشاری *) ((ثابت حرف *) score->ڈیٹا
                    + p*score->nb[0] + i1*score->nb[1] + i2*score->nb[2] + i3*score->nb[3]);
                اگر (!(s > -INFINITY)) {
                    جاری;
                }
            }
            اگر (!بلاکیں) {
                keep(p);
                جاری;
            }
            برائے (int64_t k = 0; k < n_blk; ++k) {
                keep(*(ثابت int32_t *) ((ثابت حرف *) بلاکیں->ڈیٹا + k*بلاکیں->nb[0] + p*بلاکیں->nb[1]));
            }
        }

        اگر (extra) {
            ثابت حرف * extra_صف = (ثابت حرف *) extra->ڈیٹا + i1*extra->nb[1] + i2*extra->nb[2] + i3*extra->nb[3];
            برائے (int64_t j = 0; j < n_extra; ++j) {
                keep(*(ثابت int32_t *) (extra_صف + j*extra->nb[0]));
            }
        }
    }
}

// ggml_compute_forward_rwkv_wkv7

ساکن خالی ggml_compute_forward_rwkv_wkv7_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {
    ثابت int64_t T = dst->src[1]->ne[2];
    ثابت int64_t C = dst->ne[0];
    ثابت int64_t HEADS = dst->src[1]->ne[1];
    ثابت int64_t n_seqs = dst->src[6]->ne[1];
    ثابت int64_t سر_سائز = C / HEADS;

    اعشاری * dst_ڈیٹا = (اعشاری *) dst->ڈیٹا;
    اعشاری * state = ((اعشاری *) dst->ڈیٹا) + C * T;

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح h_آغاز =  (HEADS * (ith    )) / nth;
    ثابت صحیح h_انتہا   = ((HEADS * (ith + 1)) / nth < HEADS) ?
                         (HEADS * (ith + 1)) / nth : HEADS;

    اعشاری * r = (اعشاری *) dst->src[0]->ڈیٹا;
    اعشاری * w = (اعشاری *) dst->src[1]->ڈیٹا;
    اعشاری * k = (اعشاری *) dst->src[2]->ڈیٹا;
    اعشاری * v = (اعشاری *) dst->src[3]->ڈیٹا;
    اعشاری * a = (اعشاری *) dst->src[4]->ڈیٹا;
    اعشاری * b = (اعشاری *) dst->src[5]->ڈیٹا;

    int64_t t_stride = HEADS * سر_سائز; // Same to C

    int64_t h_stride = C / HEADS;
    GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
    int64_t h_stride_2d = سر_سائز * سر_سائز;

    #اگر defined(GGML_SIMD)
        #اگر defined(__ARM_FEATURE_SVE) || defined(__riscv_v_intrinsic)
            // scalar Route to scalar implementation       //TODO: Write SVE code and RVV code
            برائے (int64_t t = 0; t < T; t++) {
                int64_t t_انحراف = t * t_stride;
                int64_t state_انحراف = سر_سائز * C * (t / (T / n_seqs));
                اعشاری * state_cur = state + state_انحراف;
                اعشاری * state_پچھلا = t % (T / n_seqs) ? state_cur : (اعشاری*)dst->src[6]->ڈیٹا + state_انحراف;

                برائے (int64_t h = h_آغاز; h < h_انتہا; h++) {
                    int64_t h_انحراف = h * h_stride;
                    int64_t t_h_انحراف = t_انحراف + h_انحراف;
                    int64_t h_2d_انحراف = h * h_stride_2d;

                    برائے (int64_t i = 0; i < سر_سائز; i++) {
                        int64_t t_h_i_انحراف = t_h_انحراف + i;
                        int64_t h_2d_i_انحراف = h_2d_انحراف + i * h_stride;

                        اعشاری v_val = v[t_h_i_انحراف];

                        اعشاری sa = 0, نتیجہ = 0;
                        برائے (int64_t j = 0; j < سر_سائز; j++) {
                            sa += a[t_h_انحراف + j] * state_پچھلا[h_2d_i_انحراف + j];
                        }

                        برائے (int64_t j = 0; j < سر_سائز; j++) {
                            int64_t t_h_j_انحراف = t_h_انحراف + j;
                            int64_t h_2d_i_j_انحراف = h_2d_i_انحراف + j;

                            اعشاری r_val = r[t_h_j_انحراف];
                            اعشاری w_val = w[t_h_j_انحراف];
                            اعشاری k_val = k[t_h_j_انحراف];
                            اعشاری b_val = b[t_h_j_انحراف];
                            اعشاری kv_val = v_val * k_val;
                            اعشاری پچھلا_state_val = state_پچھلا[h_2d_i_j_انحراف];
                            state_cur[h_2d_i_j_انحراف] = پچھلا_state_val * w_val + kv_val + sa * b_val;
                            نتیجہ += state_cur[h_2d_i_j_انحراف] * r_val;
                        }
                        dst_ڈیٹا[t_h_i_انحراف] = نتیجہ;
                    }
                }
            }
        #ورنہ
            برائے (int64_t t = 0; t < T; t++) {
                int64_t t_انحراف = t * t_stride;
                int64_t state_انحراف = سر_سائز * C * (t / (T / n_seqs));
                اعشاری * state_cur = state + state_انحراف;
                اعشاری * state_پچھلا = t % (T / n_seqs) ? state_cur : (اعشاری*)dst->src[6]->ڈیٹا + state_انحراف;

                برائے (int64_t h = h_آغاز; h < h_انتہا; h++) {
                    int64_t h_انحراف = h * h_stride;
                    int64_t t_h_انحراف = t_انحراف + h_انحراف;
                    int64_t h_2d_انحراف = h * h_stride_2d;

                    برائے (int64_t ii = 0; ii < سر_سائز; ii++) {
                        int64_t t_h_i_انحراف = t_h_انحراف + ii;
                        int64_t h_2d_i_انحراف = h_2d_انحراف + ii * h_stride;

                        GGML_F32_VEC v_vec = GGML_F32_VEC_SET1(v[t_h_i_انحراف]);

                        اعشاری sa = 0;
                        {
                            GGML_F32_VEC مجموع[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };
                            GGML_F32_VEC ax[GGML_F32_ARR];
                            GGML_F32_VEC ay[GGML_F32_ARR];
                            برائے (int64_t j = 0; j < سر_سائز; j += GGML_F32_قدم) {
                                برائے (int64_t kk = 0; kk < GGML_F32_ARR; kk++) {
                                    ax[kk] = GGML_F32_VEC_بھرو(&a[t_h_انحراف + j + kk * GGML_F32_EPR]);
                                    ay[kk] = GGML_F32_VEC_بھرو(&state_پچھلا[h_2d_i_انحراف + j + kk * GGML_F32_EPR]);
                                    مجموع[kk] = GGML_F32_VEC_FMA(مجموع[kk], ax[kk], ay[kk]);
                                }
                            }
                            GGML_F32_VEC_REDUCE(sa, مجموع);
                        }

                        GGML_F32_VEC sa_vec = GGML_F32_VEC_SET1(sa);

                        int64_t j = 0;
                        GGML_F32_VEC نتیجہ_vec[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };
                        برائے (; j < سر_سائز; j += GGML_F32_قدم) {
                            برائے (int64_t kk = 0; kk < GGML_F32_ARR; kk++) {
                                int64_t t_h_j_انحراف = t_h_انحراف + j + kk * GGML_F32_EPR;
                                int64_t h_2d_i_j_انحراف = h_2d_i_انحراف + j + kk * GGML_F32_EPR;

                                GGML_F32_VEC r_vec = GGML_F32_VEC_بھرو(&r[t_h_j_انحراف]);
                                GGML_F32_VEC w_vec = GGML_F32_VEC_بھرو(&w[t_h_j_انحراف]);
                                GGML_F32_VEC k_vec = GGML_F32_VEC_بھرو(&k[t_h_j_انحراف]);
                                GGML_F32_VEC b_vec = GGML_F32_VEC_بھرو(&b[t_h_j_انحراف]);

                                k_vec = GGML_F32_VEC_MUL(v_vec, k_vec);

                                GGML_F32_VEC state_vec = GGML_F32_VEC_بھرو(&state_پچھلا[h_2d_i_j_انحراف]);
                                // kv + s * decay + sa * b
                                state_vec = GGML_F32_VEC_FMA(k_vec, state_vec, w_vec);
                                state_vec = GGML_F32_VEC_FMA(state_vec, sa_vec, b_vec);
                                GGML_F32_VEC_STORE(&state_cur[h_2d_i_j_انحراف], state_vec);

                                نتیجہ_vec[kk] = GGML_F32_VEC_FMA(نتیجہ_vec[kk], state_vec, r_vec);
                            }
                        }
                        GGML_F32_VEC_REDUCE(dst_ڈیٹا[t_h_i_انحراف], نتیجہ_vec);

                        // There shouldn't be left-overs though.
                        برائے (; j < سر_سائز; j++) {
                            int64_t t_h_j_انحراف = t_h_انحراف + j;
                            int64_t h_2d_i_j_انحراف = h_2d_i_انحراف + j;

                            اعشاری r_val = r[t_h_j_انحراف];
                            اعشاری w_val = w[t_h_j_انحراف];
                            اعشاری k_val = k[t_h_j_انحراف];
                            اعشاری b_val = b[t_h_j_انحراف];
                            اعشاری kv_val = v[t_h_i_انحراف] * k_val;

                            اعشاری پچھلا_state_val = state_پچھلا[h_2d_i_j_انحراف];
                            state_cur[h_2d_i_j_انحراف] = پچھلا_state_val * w_val + kv_val + sa * b_val;
                            dst_ڈیٹا[t_h_i_انحراف] += state_cur[h_2d_i_j_انحراف] * r_val;
                        }
                    }
                }
            }
        #اگر_ختم
    #ورنہ
        برائے (int64_t t = 0; t < T; t++) {
            int64_t t_انحراف = t * t_stride;
            int64_t state_انحراف = سر_سائز * C * (t / (T / n_seqs));
            اعشاری * state_cur = state + state_انحراف;
            اعشاری * state_پچھلا = t % (T / n_seqs) ? state_cur : (اعشاری*)dst->src[6]->ڈیٹا + state_انحراف;

            برائے (int64_t h = h_آغاز; h < h_انتہا; h++) {
                int64_t h_انحراف = h * h_stride;
                int64_t t_h_انحراف = t_انحراف + h_انحراف;
                int64_t h_2d_انحراف = h * h_stride_2d;

                برائے (int64_t i = 0; i < سر_سائز; i++) {
                    int64_t t_h_i_انحراف = t_h_انحراف + i;
                    int64_t h_2d_i_انحراف = h_2d_انحراف + i * h_stride;

                    اعشاری v_val = v[t_h_i_انحراف];

                    اعشاری sa = 0, نتیجہ = 0;
                    برائے (int64_t j = 0; j < سر_سائز; j++) {
                        sa += a[t_h_انحراف + j] * state_پچھلا[h_2d_i_انحراف + j];
                    }

                    برائے (int64_t j = 0; j < سر_سائز; j++) {
                        int64_t t_h_j_انحراف = t_h_انحراف + j;
                        int64_t h_2d_i_j_انحراف = h_2d_i_انحراف + j;

                        اعشاری r_val = r[t_h_j_انحراف];
                        اعشاری w_val = w[t_h_j_انحراف];
                        اعشاری k_val = k[t_h_j_انحراف];
                        اعشاری b_val = b[t_h_j_انحراف];
                        اعشاری kv_val = v_val * k_val;
                        اعشاری پچھلا_state_val = state_پچھلا[h_2d_i_j_انحراف];
                        state_cur[h_2d_i_j_انحراف] = پچھلا_state_val * w_val + kv_val + sa * b_val;
                        نتیجہ += state_cur[h_2d_i_j_انحراف] * r_val;
                    }
                    dst_ڈیٹا[t_h_i_انحراف] = نتیجہ;
                }
            }
        }
    #اگر_ختم
}


خالی ggml_compute_forward_rwkv_wkv7(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_rwkv_wkv7_f32(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_map_custom1

خالی ggml_compute_forward_map_custom1(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت ggml_tensor * a = dst->src[0];

    ساخت ggml_map_custom1_op_params p;
    حافظہ_نقل(&p, dst->op_params, حجم(p));

    p.fun(dst, a, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_map_custom2

خالی ggml_compute_forward_map_custom2(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت ggml_tensor * a = dst->src[0];
    ثابت ggml_tensor * b = dst->src[1];

    ساخت ggml_map_custom2_op_params p;
    حافظہ_نقل(&p, dst->op_params, حجم(p));

    p.fun(dst, a, b, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_map_custom3

خالی ggml_compute_forward_map_custom3(
        ثابت ggml_compute_params * params,
              ggml_tensor * dst) {

    ثابت ggml_tensor * a = dst->src[0];
    ثابت ggml_tensor * b = dst->src[1];
    ثابت ggml_tensor * c = dst->src[2];

    ساخت ggml_map_custom3_op_params p;
    حافظہ_نقل(&p, dst->op_params, حجم(p));

    p.fun(dst, a, b, c, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_custom

خالی ggml_compute_forward_custom(
    ثابت ساخت ggml_compute_params * params,
          ساخت ggml_tensor * dst) {

    ساخت ggml_custom_op_params p;
    حافظہ_نقل(&p, dst->op_params, حجم(p));

    p.fun(dst, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_cross_entropy_loss

ساکن خالی ggml_compute_forward_cross_entropy_loss_f32(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->nb[0] == ggml_type_سائز(src0->type));
    GGML_ASSERT(src1->nb[0] == ggml_type_سائز(src1->type));
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // TODO: handle transposed/permuted matrices
    ثابت int64_t nc = src0->ne[0];
    ثابت int64_t nr = ggml_nrows(src0);

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    اعشاری * sums =  (اعشاری *) params->wdata;
    اعشاری * st   = ((اعشاری *) params->wdata) + nth + ith*nc;
    اعشاری مجموع_thread = 0.0f;

    GGML_ASSERT(params->wsize >= حجم(اعشاری) * (nth + nth * nc));

    // rows per thread
    ثابت int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت int64_t ir0 = dr*ith;
    ثابت int64_t ir1 = MIN(ir0 + dr, nr);

    برائے (int64_t i1 = ir0; i1 < ir1; ++i1) {
        ثابت اعشاری * s0 = (ثابت اعشاری *)((ثابت حرف *) src0->ڈیٹا + i1*src0->nb[1]);
        ثابت اعشاری * s1 = (ثابت اعشاری *)((ثابت حرف *) src1->ڈیٹا + i1*src1->nb[1]);

#اگر_غیر_تعریف_شدہ NDEBUG
        برائے (int64_t i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            أكد(!isnan(s0[i]));
            أكد(!isnan(s1[i]));
        }
#اگر_ختم // NDEBUG

        اعشاری max = -INFINITY;
        ggml_vec_max_f32(nc, &max, s0);
        ثابت ggml_float مجموع_softmax = ggml_vec_log_soft_max_f32(nc, st, s0, max);
        أكد(مجموع_softmax >= 0.0);

        ggml_vec_add1_f32(nc, st, st, -مجموع_softmax);
        ggml_vec_mul_f32(nc, st, st, s1);

        اعشاری مجموع_st = 0.0f;
        ggml_vec_مجموع_f32(nc, &مجموع_st, st);
        مجموع_thread += مجموع_st;

#اگر_غیر_تعریف_شدہ NDEBUG
        برائے (int64_t i = 0; i < nc; ++i) {
            أكد(!isnan(st[i]));
            أكد(!isinf(st[i]));
        }
#اگر_ختم // NDEBUG
    }
    sums[ith] = مجموع_thread;
    ggml_barrier(params->threadpool);

    اگر (ith == 0) {
        اعشاری * dp = (اعشاری *) dst->ڈیٹا;
        ggml_vec_مجموع_f32(nth, dp, sums);
        dp[0] *= -1.0f / (اعشاری) nr;
    }
}

خالی ggml_compute_forward_cross_entropy_loss(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_cross_entropy_loss_f32(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_cross_entropy_loss_back

ساکن خالی ggml_compute_forward_cross_entropy_loss_عقب_f32(
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

    ثابت اعشاری d_by_nr = ((ثابت اعشاری *) grad->ڈیٹا)[0] / (اعشاری) nr;

    برائے (int64_t i1 = ir0; i1 < ir1; i1++) {
        اعشاری       * ds0 = (اعشاری       *)((حرف       *) dst->ڈیٹا   + i1*dst->nb[1]);
        ثابت اعشاری * s0  = (ثابت اعشاری *)((ثابت حرف *) src0f->ڈیٹا + i1*src0f->nb[1]);
        ثابت اعشاری * s1  = (ثابت اعشاری *)((ثابت حرف *) src1f->ڈیٹا + i1*src1f->nb[1]);

#اگر_غیر_تعریف_شدہ NDEBUG
        برائے (int64_t i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            أكد(!isnan(s0[i]));
            أكد(!isnan(s1[i]));
        }
#اگر_ختم // NDEBUG

        // soft_max
        اعشاری max = -INFINITY;
        ggml_vec_max_f32(nc, &max, s0);
        ثابت ggml_float مجموع = ggml_vec_soft_max_f32(nc, ds0, s0, max);
        أكد(مجموع > 0.0);
        ggml_vec_scale_f32(nc, ds0, 1.0/مجموع);

        // grad(src0f) = (softmax(src0f) - src1f) * grad(cross_entropy_loss(src0f, src1f)) / nr
        ggml_vec_sub_f32(nc, ds0, ds0, s1);
        ggml_vec_scale_f32(nc, ds0, d_by_nr);

#اگر_غیر_تعریف_شدہ NDEBUG
        برائے (int64_t i = 0; i < nc; ++i) {
            أكد(!isnan(ds0[i]));
            أكد(!isinf(ds0[i]));
        }
#اگر_ختم // NDEBUG
    }
}

خالی ggml_compute_forward_cross_entropy_loss_عقب(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_cross_entropy_loss_عقب_f32(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ساکن خالی ggml_compute_forward_opt_قدم_adamw_f32(
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

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS
    GGML_ASSERT(nb00 == حجم(اعشاری));

    // rows per thread
    ثابت صحیح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحیح ir0 = dr*ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    ثابت اعشاری * adamw_params_ptr = ggml_get_ڈیٹا_f32(adamw_params);

    ثابت اعشاری alpha  = adamw_params_ptr[0];
    ثابت اعشاری beta1  = adamw_params_ptr[1];
    ثابت اعشاری beta2  = adamw_params_ptr[2];
    ثابت اعشاری eps    = adamw_params_ptr[3];
    ثابت اعشاری wd     = adamw_params_ptr[4];
    ثابت اعشاری beta1h = adamw_params_ptr[5];
    ثابت اعشاری beta2h = adamw_params_ptr[6];
    ثابت اعشاری keep   = 1.f - alpha * wd;
    برائے (صحیح ir = ir0; ir < ir1; ++ir) {
        ثابت int64_t i03 = ir/(ne02*ne01);
        ثابت int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        ثابت int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        ثابت size_t انحراف = i03*nb03 + i02*nb02 + i01*nb01;

        اعشاری       * w = (اعشاری       *) ((حرف       *) src0->ڈیٹا        + انحراف); //وزن
        ثابت اعشاری * g = (ثابت اعشاری *) ((ثابت حرف *) src0_grad->ڈیٹا   + انحراف); // grad
        اعشاری       * m = (اعشاری       *) ((حرف       *) src0_grad_m->ڈیٹا + انحراف);
        اعشاری       * v = (اعشاری       *) ((حرف       *) src0_grad_v->ڈیٹا + انحراف);

        برائے (صحیح i00 = 0; i00 < ne00; ++i00) {
            m[i00] = m[i00]*beta1 +        g[i00]*(1.0f - beta1);
            v[i00] = v[i00]*beta2 + g[i00]*g[i00]*(1.0f - beta2);

            ثابت اعشاری mh =       m[i00]*beta1h;
            ثابت اعشاری vh = sqrtf(v[i00]*beta2h) + eps;

            // The weight decay is applied independently of the Adam momenta m and v.
            // This is NOT equivalent to l2 regularization that adds w[i00]*w[i00] to the loss.
            // See: https://arxiv.org/pdf/1711.05101v3.pdf
            w[i00] = w[i00] * keep - alpha * mh / vh;
        }
    }
}

خالی ggml_compute_forward_opt_قدم_adamw(
        ثابت ggml_compute_params * params,
        ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_opt_قدم_adamw_f32(params, dst);
            } توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

ساکن خالی ggml_compute_forward_opt_قدم_sgd_f32(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    ثابت ggml_tensor * src0       = dst->src[0];
    ثابت ggml_tensor * src0_grad  = dst->src[1];
    ثابت ggml_tensor * sgd_params = dst->src[2];

    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad));
    GGML_ASSERT(ggml_nelements(sgd_params) == 2);

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت صحیح nr = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS
    GGML_ASSERT(nb00 == حجم(اعشاری));

    // rows per thread
    ثابت صحیح dr = (nr + nth - 1) / nth;

    // row range for this thread
    ثابت صحیح ir0 = dr * ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    // using adamw param subset we care about - alpha, wd - could have a separate struct
    ثابت اعشاری * sgd_params_ptr   = ggml_get_ڈیٹا_f32(sgd_params);
    ثابت اعشاری   alpha            = sgd_params_ptr[0];
    ثابت اعشاری   keep             = 1.f - alpha * sgd_params_ptr[1];

    برائے (صحیح ir = ir0; ir < ir1; ++ir) {
        ثابت int64_t i03 = ir / (ne02 * ne01);
        ثابت int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
        ثابت int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

        ثابت size_t انحراف = i03 * nb03 + i02 * nb02 + i01 * nb01;

        اعشاری *       w = (اعشاری *) ((حرف *) src0->ڈیٹا + انحراف);                   //وزن
        ثابت اعشاری * g = (ثابت اعشاری *) ((ثابت حرف *) src0_grad->ڈیٹا + انحراف);  // grad

        برائے (صحیح i00 = 0; i00 < ne00; ++i00) {
            w[i00] = w[i00] * keep - alpha * g[i00];
        }
    }
}

خالی ggml_compute_forward_opt_قدم_sgd(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];

    انتخاب (src0->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_opt_قدم_sgd_f32(params, dst);
            }
            توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error - sgd is F32 only");
            }
    }
}

ساکن اندرونی اعشاری ggml_fwht_بھرو(ثابت اعشاری value) {
    واپس value;
}

ساکن اندرونی اعشاری ggml_fwht_بھرو(ثابت ggml_fp16_t value) {
    واپس ggml_fp16_مقصد_fp32(value);
}

سانچہ<قسم_نام src_t>
ساکن خالی ggml_compute_forward_fwht_impl(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == (std::is_same_v<src_t, اعشاری> ? GGML_TYPE_F32 : GGML_TYPE_F16));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ثنائی_OP_LOCALS

    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    ثابت int64_t n = ne10;
    GGML_ASSERT((n & (n - 1)) == 0); // must be power of 2

    ثابت int64_t nr = ne11 * ne12 * ne13;
    ثابت int64_t صفیں_فی_thread = (nr + nth - 1) / nth;
    ثابت int64_t آغاز_صف = ith * صفیں_فی_thread;
    ثابت int64_t انتہا_صف = MIN(آغاز_صف + صفیں_فی_thread, nr);

    ثابت اعشاری scale = 1.0f / sqrtf((اعشاری)n);

#اگر defined(GGML_SIMD)
    ثابت GGML_F32_VEC v_minus_one = GGML_F32_VEC_SET1(-1.0f);
#اگر_ختم

    برائے (int64_t r = آغاز_صف; r < انتہا_صف; r++) {
        ثابت int64_t i13 = r / (ne11 * ne12);
        ثابت int64_t i12 = (r - i13 * ne11 * ne12) / ne11;
        ثابت int64_t i11 = r - i13 * ne11 * ne12 - i12 * ne11;

        ثابت src_t * src_صف = (ثابت src_t *) ((ثابت حرف *) src1->ڈیٹا + i11 * nb11 + i12 * nb12 + i13 * nb13);
        اعشاری * dst_صف = (اعشاری *) ((حرف *) dst->ڈیٹا + i11 * nb1 + i12 * nb2 + i13 * nb3);

        برائے (int64_t j = 0; j < n; j++) {
            dst_صف[j] = ggml_fwht_بھرو(src_صف[j]) * scale;
        }

        // Scalar passes
#اگر defined(GGML_SIMD)
#اگر defined(__ARM_FEATURE_SVE)
        ثابت صحیح قدم = svcntw();
#ورنہ
        ثابت صحیح قدم = GGML_F32_EPR;
#اگر_ختم
#ورنہ
        ثابت صحیح قدم = n;
#اگر_ختم
        برائے (int64_t len = 1; len < قدم && len < n; len <<= 1) {
            برائے (int64_t i = 0; i < n; i += 2 * len) {
                برائے (int64_t j = 0; j < len; j++) {
                    اعشاری u = dst_صف[i + j];
                    اعشاری v = dst_صف[i + len + j];
                    dst_صف[i + j] = u + v;
                    dst_صف[i + len + j] = u - v;
                }
            }
        }

        // SIMD passes using GGML_F32_VEC_* macros for multi-architecture support
#اگر defined(GGML_SIMD)
        برائے (int64_t len = قدم; len < n; len <<= 1) {
            برائے (int64_t i = 0; i < n; i += 2 * len) {
                برائے (int64_t j = 0; j < len; j += قدم) {
                    GGML_F32_VEC u = GGML_F32_VEC_بھرو(dst_صف + i + j);
                    GGML_F32_VEC v = GGML_F32_VEC_بھرو(dst_صف + i + len + j);

                    GGML_F32_VEC_STORE(dst_صف + i + j,       GGML_F32_VEC_جوڑو(u, v));
                    GGML_F32_VEC_STORE(dst_صف + i + len + j, GGML_F32_VEC_FMA(u, v, v_minus_one));
                }
            }
        }
#اگر_ختم
    }
}

خالی ggml_compute_forward_fwht(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    ثابت ggml_tensor * src1 = dst->src[1];

    انتخاب (src1->type) {
        حالت GGML_TYPE_F32:
            {
                ggml_compute_forward_fwht_impl<اعشاری>(params, dst);
            }
            توڑو;
        حالت GGML_TYPE_F16:
            {
                ggml_compute_forward_fwht_impl<ggml_fp16_t>(params, dst);
            }
            توڑو;
        طےشدہ:
            {
                GGML_ABORT("fatal error - fwht supports F32 and F16 input");
            }
    }
}

// ggml_compute_forward_lightning_indexer

خالی ggml_compute_forward_lightning_indexer(
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
    GGML_TENSOR_LOCALS(int64_t, نیا,  w, ne)
    GGML_TENSOR_LOCALS(size_t,  nbw,  w, nb)
    GGML_TENSOR_LOCALS(int64_t, nem,  m, ne)
    GGML_TENSOR_LOCALS(size_t,  nbm,  m, nb)
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb)

    GGML_ASSERT( nb0 == ggml_type_سائز(dst->type));
    GGML_ASSERT(nbq0 == ggml_type_سائز(  q->type));
    GGML_ASSERT(nbk0 == ggml_type_سائز(  k->type));
    GGML_ASSERT(nbw0 == ggml_type_سائز(  w->type));
    GGML_ASSERT(nbm0 == ggml_type_سائز(  m->type));

    ثابت صحیح n_embd    = q->ne[0];
    ثابت صحیح n_سر    = q->ne[1];
    ثابت صحیح n_علامات  = q->ne[2];
    ثابت صحیح n_دھارا  = q->ne[3];
    ثابت صحیح n_kv      = k->ne[2];

    ggml_مقصد_float_t ثابت k_مقصد_float = ggml_get_type_traits(k->type)->مقصد_float;
    GGML_ASSERT((k->type == GGML_TYPE_F32 || k_مقصد_float) && "lightning indexer: unsupported K-type");

    ثابت صحیح nr  = n_kv;
    ثابت صحیح ith = params->ith;
    ثابت صحیح nth = params->nth;

    // (temporary) buffer for K converted to float
    اعشاری * k_صف_f32 = (اعشاری *) params->wdata + ith*(1*n_embd + CACHE_سطر_سائز_F32);

    // rows per thread
    ثابت صحیح dr = (nr + nth - 1)/nth;

    // row range for this thread
    ثابت صحیح ir0 = dr*ith;
    ثابت صحیح ir1 = MIN(ir0 + dr, nr);

    برائے (صحیح s = 0; s < n_دھارا; ++s) {
        برائے (صحیح t = 0; t < n_علامات; ++t) {
            ثابت اعشاری       *   w_صف =       (اعشاری *) ((حرف *)   w->ڈیٹا + t*nbw1 +        s*nbw3);
            ثابت ggml_fp16_t *   m_صف = (ggml_fp16_t *) ((حرف *)   m->ڈیٹا + t*nbm1 + (s%nem3)*nbm3);
            اعشاری             * dst_صف =       (اعشاری *) ((حرف *) dst->ڈیٹا + t*nb1  +        s*nb3 );
            برائے (صحیح ik = ir0; ik < ir1; ++ik) {
                حرف * k_صف = (حرف *) k->ڈیٹا + ik*nbk2 + s*nbk3;
                اگر (k_مقصد_float) {
                    k_مقصد_float(k_صف, k_صف_f32, n_embd);
                } ورنہ {
                    k_صف_f32 = (اعشاری *) k_صف;
                }
                اعشاری score = 0.0f;
                برائے (صحیح h = 0; h < n_سر; ++h) {
                    // dot product of q and k for head h
                    اعشاری qk = 0.0f;
                    ثابت اعشاری * q_صف = (اعشاری *) ((حرف *) q->ڈیٹا + h*nbq1 + t*nbq2 + s*nbq3);
                    ggml_vec_dot_f32(n_embd, &qk, 0, q_صف, 0, k_صف_f32, 0, 1);
                    // ReLU and weights (prescaled)
                    score += MAX(qk, 0.0f) * w_صف[h];
                }
                // apply mask
                dst_صف[ik] = score + GGML_CPU_FP16_مقصد_FP32(m_صف[ik]);
            }
        }
    }
}
