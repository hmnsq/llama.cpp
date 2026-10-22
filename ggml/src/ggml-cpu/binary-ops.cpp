#अन्तर्भाव "binary-ops.h"

#यदि defined(GGML_USE_ACCELERATE)
#अन्तर्भाव <Accelerate/Accelerate.h>

उपयुज्य vDSP_fn_t = शून्य (*)(ध्रुव भिन्न *, vDSP_Stride, ध्रुव भिन्न *, vDSP_Stride, भिन्न *, vDSP_Stride, v_DSP_दीर्घता);
#यदि_अन्तः

स्थिर अन्तःस्थ भिन्न op_सङ्कलय(भिन्न a, भिन्न b) {
    प्रत्यागम a + b;
}

स्थिर अन्तःस्थ भिन्न op_sub(भिन्न a, भिन्न b) {
    प्रत्यागम a - b;
}

स्थिर अन्तःस्थ भिन्न op_mul(भिन्न a, भिन्न b) {
    प्रत्यागम a * b;
}

स्थिर अन्तःस्थ भिन्न op_div(भिन्न a, भिन्न b) {
    प्रत्यागम a / b;
}

प्रतिरूपम् <भिन्न (*op)(भिन्न, भिन्न), प्रकारनाम src0_t, प्रकारनाम src1_t, प्रकारनाम dst_t>
स्थिर अन्तःस्थ शून्य vec_द्विमान_op_contiguous(ध्रुव int64_t n, dst_t * z, ध्रुव src0_t * x, ध्रुव src1_t * y) {
    constexpr स्वचालित src0_गन्तव्यम्_f32 = type_conversion_सारणी<src0_t>::गन्तव्यम्_f32;
    constexpr स्वचालित src1_गन्तव्यम्_f32 = type_conversion_सारणी<src1_t>::गन्तव्यम्_f32;
    constexpr स्वचालित f32_गन्तव्यम्_dst  = type_conversion_सारणी<dst_t >::from_f32;

    प्रति (पूर्णाङ्क i = 0; i < n; i++) {
        z[i] = f32_गन्तव्यम्_dst(op(src0_गन्तव्यम्_f32(x[i]), src1_गन्तव्यम्_f32(y[i])));
    }
}

प्रतिरूपम् <भिन्न (*op)(भिन्न, भिन्न), प्रकारनाम src0_t, प्रकारनाम src1_t, प्रकारनाम dst_t>
स्थिर अन्तःस्थ शून्य vec_द्विमान_op_non_contiguous(ध्रुव int64_t n, ध्रुव int64_t ne10, ध्रुव int64_t nb10, dst_t * z, ध्रुव src0_t * x, ध्रुव src1_t * y) {
    constexpr स्वचालित src0_गन्तव्यम्_f32 = type_conversion_सारणी<src0_t>::गन्तव्यम्_f32;
    constexpr स्वचालित src1_गन्तव्यम्_f32 = type_conversion_सारणी<src1_t>::गन्तव्यम्_f32;
    constexpr स्वचालित f32_गन्तव्यम्_dst  = type_conversion_सारणी<dst_t >::from_f32;

    प्रति (पूर्णाङ्क i = 0; i < n; i++) {
        पूर्णाङ्क i10 = i % ne10;
        ध्रुव src1_t * y_ptr = (ध्रुव src1_t *)((ध्रुव वर्ण *)y + i10*nb10);
        z[i] = f32_गन्तव्यम्_dst(op(src0_गन्तव्यम्_f32(x[i]), src1_गन्तव्यम्_f32(*y_ptr)));
    }
}

प्रतिरूपम् <भिन्न (*op)(भिन्न, भिन्न), प्रकारनाम src0_t, प्रकारनाम src1_t, प्रकारनाम dst_t>
स्थिर शून्य apply_द्विमान_op(ध्रुव ggml_compute_params * params, ggml_tensor * dst) {
    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_द्विमान_OP_LOCALS

    GGML_ASSERT( nb0 == परिमाण(dst_t));
    GGML_ASSERT(nb00 == परिमाण(src0_t));

    ध्रुव स्वचालित [ir0, ir1] = get_thread_range(params, src0);
    ध्रुव तार्किक is_src1_contiguous_पङ्क्तिकाः = ggml_is_contiguous_पङ्क्तिकाः(src1);

#यदि_परिभाषितम् GGML_USE_ACCELERATE
    vDSP_fn_t vDSP_op = शून्यसूचकः;
    // TODO - avoid the f32-only check using type 'trait' lookup tables and row-based src-to-float conversion functions
    यदि (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        यदि (op == op_सङ्कलय) {
            vDSP_op = vDSP_vadd;
        } अन्यथा यदि (op == op_sub) {
            vDSP_op = vDSP_vsub;
        } अन्यथा यदि (op == op_mul) {
            vDSP_op = vDSP_vmul;
        } अन्यथा यदि (op == op_div) {
            vDSP_op = vDSP_vdiv;
        }
    }
#यदि_अन्तः

    प्रति (int64_t ir = ir0; ir < ir1; ++ir) {
        ध्रुव int64_t i03 = ir/(ne02*ne01);
        ध्रुव int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        ध्रुव int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        ध्रुव int64_t i13 = i03 % ne13;
        ध्रुव int64_t i12 = i02 % ne12;
        ध्रुव int64_t i11 = i01 % ne11;

        dst_t        * dst_ptr  = (dst_t  *)       ((वर्ण *)       dst->दत्त  + i03*nb3  + i02*nb2  + i01*nb1 );
        ध्रुव src0_t * src0_ptr = (ध्रुव src0_t *) ((ध्रुव वर्ण *) src0->दत्त + i03*nb03 + i02*nb02 + i01*nb01);
        ध्रुव src1_t * src1_ptr = (ध्रुव src1_t *) ((ध्रुव वर्ण *) src1->दत्त + i13*nb13 + i12*nb12 + i11*nb11);

        यदि (is_src1_contiguous_पङ्क्तिकाः) {
            // src1 is broadcastable across src0 and dst in i1, i2, i3
            ध्रुव int64_t nr0 = ne00 / ne10;

            प्रति (int64_t r = 0; r < nr0; ++r) {
#यदि_परिभाषितम् GGML_USE_ACCELERATE
                यदि constexpr (std::is_same_v<src0_t, भिन्न> && std::is_same_v<src1_t, भिन्न> && std::is_same_v<dst_t, भिन्न>) {
                    यदि (vDSP_op != शून्यसूचकः) {
                        vDSP_op(src1_ptr, 1, src0_ptr + r*ne10, 1, dst_ptr + r*ne10, 1, ne10);
                        अनुवर्त;
                    }
                }
#यदि_अन्तः
                vec_द्विमान_op_contiguous<op>(ne10, dst_ptr + r*ne10, src0_ptr + r*ne10, src1_ptr);
            }
        } अन्यथा {
            vec_द्विमान_op_non_contiguous<op>(ne0, ne10, nb10, dst_ptr, src0_ptr, src1_ptr);
        }
    }
}

// TODO: Use the 'traits' lookup table (for type conversion fns), instead of a mass of 'if' conditions with long templates
प्रतिरूपम् <भिन्न (*op)(भिन्न, भिन्न)>
स्थिर शून्य द्विमान_op(ध्रुव ggml_compute_params * params, ggml_tensor * dst) {
    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    /*  */ यदि (src0->type == GGML_TYPE_F32  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_द्विमान_op<op, भिन्न, भिन्न, भिन्न>(params, dst);
    } अन्यथा यदि (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_द्विमान_op<op, ggml_fp16_t, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } अन्यथा यदि (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_द्विमान_op<op, ggml_bf16_t, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } अन्यथा यदि (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_BF16) {
        apply_द्विमान_op<op, ggml_bf16_t, भिन्न, ggml_bf16_t>(params, dst);
    } अन्यथा यदि (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) {
        apply_द्विमान_op<op, ggml_bf16_t, भिन्न, भिन्न>(params, dst);
    } अन्यथा यदि (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F16) {
        apply_द्विमान_op<op, ggml_fp16_t, भिन्न, ggml_fp16_t>(params, dst);
    } अन्यथा यदि (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) {
        apply_द्विमान_op<op, ggml_fp16_t, भिन्न, भिन्न>(params, dst);
    } अन्यथा {
        GGML_ABORT("%s: unsupported types: dst: %s, src0: %s, src1: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type), ggml_type_name(src1->type));
    }
}

शून्य ggml_compute_forward_सङ्कलय_non_quantized(ध्रुव ggml_compute_params * params, ggml_tensor * dst) {
    द्विमान_op<op_सङ्कलय>(params, dst);
}

शून्य ggml_compute_forward_sub(ध्रुव ggml_compute_params * params, ggml_tensor * dst) {
    द्विमान_op<op_sub>(params, dst);
}

शून्य ggml_compute_forward_mul(ध्रुव ggml_compute_params * params, ggml_tensor * dst) {
    द्विमान_op<op_mul>(params, dst);
}

शून्य ggml_compute_forward_div(ध्रुव ggml_compute_params * params, ggml_tensor * dst) {
    द्विमान_op<op_div>(params, dst);
}
