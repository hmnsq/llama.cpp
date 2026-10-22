#ضمن "binary-ops.h"

#إذا defined(GGML_USE_ACCELERATE)
#ضمن <Accelerate/Accelerate.h>

استفاده vDSP_fn_t = فراغ (*)(ثابت عائم *, vDSP_Stride, ثابت عائم *, vDSP_Stride, عائم *, vDSP_Stride, v_DSP_درازی);
#پایان_اگر

ساكن مضمن عائم op_جمع_کن(عائم a, عائم b) {
    أرجع a + b;
}

ساكن مضمن عائم op_sub(عائم a, عائم b) {
    أرجع a - b;
}

ساكن مضمن عائم op_mul(عائم a, عائم b) {
    أرجع a * b;
}

ساكن مضمن عائم op_div(عائم a, عائم b) {
    أرجع a / b;
}

قالب <عائم (*op)(عائم, عائم), نام_نوع src0_t, نام_نوع src1_t, نام_نوع dst_t>
ساكن مضمن فراغ vec_دودویی_op_contiguous(ثابت int64_t n, dst_t * z, ثابت src0_t * x, ثابت src1_t * y) {
    constexpr خودکار src0_مقصد_f32 = type_conversion_جدول<src0_t>::مقصد_f32;
    constexpr خودکار src1_مقصد_f32 = type_conversion_جدول<src1_t>::مقصد_f32;
    constexpr خودکار f32_مقصد_dst  = type_conversion_جدول<dst_t >::from_f32;

    لكل (صحيح i = 0; i < n; i++) {
        z[i] = f32_مقصد_dst(op(src0_مقصد_f32(x[i]), src1_مقصد_f32(y[i])));
    }
}

قالب <عائم (*op)(عائم, عائم), نام_نوع src0_t, نام_نوع src1_t, نام_نوع dst_t>
ساكن مضمن فراغ vec_دودویی_op_non_contiguous(ثابت int64_t n, ثابت int64_t ne10, ثابت int64_t nb10, dst_t * z, ثابت src0_t * x, ثابت src1_t * y) {
    constexpr خودکار src0_مقصد_f32 = type_conversion_جدول<src0_t>::مقصد_f32;
    constexpr خودکار src1_مقصد_f32 = type_conversion_جدول<src1_t>::مقصد_f32;
    constexpr خودکار f32_مقصد_dst  = type_conversion_جدول<dst_t >::from_f32;

    لكل (صحيح i = 0; i < n; i++) {
        صحيح i10 = i % ne10;
        ثابت src1_t * y_ptr = (ثابت src1_t *)((ثابت حرف *)y + i10*nb10);
        z[i] = f32_مقصد_dst(op(src0_مقصد_f32(x[i]), src1_مقصد_f32(*y_ptr)));
    }
}

قالب <عائم (*op)(عائم, عائم), نام_نوع src0_t, نام_نوع src1_t, نام_نوع dst_t>
ساكن فراغ apply_دودویی_op(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_دودویی_OP_LOCALS

    GGML_ASSERT( nb0 == حجم(dst_t));
    GGML_ASSERT(nb00 == حجم(src0_t));

    ثابت خودکار [ir0, ir1] = get_thread_range(params, src0);
    ثابت منطقي is_src1_contiguous_ردیفها = ggml_is_contiguous_ردیفها(src1);

#اگر_تعریف_شده GGML_USE_ACCELERATE
    vDSP_fn_t vDSP_op = اشاره_تهی;
    // TODO - avoid the f32-only check using type 'trait' lookup tables and row-based src-to-float conversion functions
    إذا (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        إذا (op == op_جمع_کن) {
            vDSP_op = vDSP_vadd;
        } وإلا إذا (op == op_sub) {
            vDSP_op = vDSP_vsub;
        } وإلا إذا (op == op_mul) {
            vDSP_op = vDSP_vmul;
        } وإلا إذا (op == op_div) {
            vDSP_op = vDSP_vdiv;
        }
    }
#پایان_اگر

    لكل (int64_t ir = ir0; ir < ir1; ++ir) {
        ثابت int64_t i03 = ir/(ne02*ne01);
        ثابت int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        ثابت int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        ثابت int64_t i13 = i03 % ne13;
        ثابت int64_t i12 = i02 % ne12;
        ثابت int64_t i11 = i01 % ne11;

        dst_t        * dst_ptr  = (dst_t  *)       ((حرف *)       dst->داده  + i03*nb3  + i02*nb2  + i01*nb1 );
        ثابت src0_t * src0_ptr = (ثابت src0_t *) ((ثابت حرف *) src0->داده + i03*nb03 + i02*nb02 + i01*nb01);
        ثابت src1_t * src1_ptr = (ثابت src1_t *) ((ثابت حرف *) src1->داده + i13*nb13 + i12*nb12 + i11*nb11);

        إذا (is_src1_contiguous_ردیفها) {
            // src1 is broadcastable across src0 and dst in i1, i2, i3
            ثابت int64_t nr0 = ne00 / ne10;

            لكل (int64_t r = 0; r < nr0; ++r) {
#اگر_تعریف_شده GGML_USE_ACCELERATE
                إذا constexpr (std::is_same_v<src0_t, عائم> && std::is_same_v<src1_t, عائم> && std::is_same_v<dst_t, عائم>) {
                    إذا (vDSP_op != اشاره_تهی) {
                        vDSP_op(src1_ptr, 1, src0_ptr + r*ne10, 1, dst_ptr + r*ne10, 1, ne10);
                        استمر;
                    }
                }
#پایان_اگر
                vec_دودویی_op_contiguous<op>(ne10, dst_ptr + r*ne10, src0_ptr + r*ne10, src1_ptr);
            }
        } وإلا {
            vec_دودویی_op_non_contiguous<op>(ne0, ne10, nb10, dst_ptr, src0_ptr, src1_ptr);
        }
    }
}

// TODO: Use the 'traits' lookup table (for type conversion fns), instead of a mass of 'if' conditions with long templates
قالب <عائم (*op)(عائم, عائم)>
ساكن فراغ دودویی_op(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    /*  */ إذا (src0->type == GGML_TYPE_F32  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_دودویی_op<op, عائم, عائم, عائم>(params, dst);
    } وإلا إذا (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_دودویی_op<op, ggml_fp16_t, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } وإلا إذا (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_دودویی_op<op, ggml_bf16_t, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } وإلا إذا (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_BF16) {
        apply_دودویی_op<op, ggml_bf16_t, عائم, ggml_bf16_t>(params, dst);
    } وإلا إذا (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) {
        apply_دودویی_op<op, ggml_bf16_t, عائم, عائم>(params, dst);
    } وإلا إذا (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F16) {
        apply_دودویی_op<op, ggml_fp16_t, عائم, ggml_fp16_t>(params, dst);
    } وإلا إذا (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) {
        apply_دودویی_op<op, ggml_fp16_t, عائم, عائم>(params, dst);
    } وإلا {
        GGML_ABORT("%s: unsupported types: dst: %s, src0: %s, src1: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type), ggml_type_name(src1->type));
    }
}

فراغ ggml_compute_forward_جمع_کن_non_quantized(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    دودویی_op<op_جمع_کن>(params, dst);
}

فراغ ggml_compute_forward_sub(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    دودویی_op<op_sub>(params, dst);
}

فراغ ggml_compute_forward_mul(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    دودویی_op<op_mul>(params, dst);
}

فراغ ggml_compute_forward_div(ثابت ggml_compute_params * params, ggml_tensor * dst) {
    دودویی_op<op_div>(params, dst);
}
