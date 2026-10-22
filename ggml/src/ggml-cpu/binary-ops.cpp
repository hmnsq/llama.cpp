#समावेश "binary-ops.h"

#अगर defined(GGML_USE_ACCELERATE)
#समावेश <Accelerate/Accelerate.h>

using vDSP_fn_t = व्योम (*)(स्थिर भग्न *, vDSP_Stride, स्थिर भग्न *, vDSP_Stride, भग्न *, vDSP_Stride, v_DSP_लंबाई);
#पूर्ण_अगर

अटल अंतरभूत भग्न op_add(भग्न a, भग्न b) {
    वापस a + b;
}

अटल अंतरभूत भग्न op_sub(भग्न a, भग्न b) {
    वापस a - b;
}

अटल अंतरभूत भग्न op_mul(भग्न a, भग्न b) {
    वापस a * b;
}

अटल अंतरभूत भग्न op_div(भग्न a, भग्न b) {
    वापस a / b;
}

ढाँचा <भग्न (*op)(भग्न, भग्न), typename src0_t, typename src1_t, typename dst_t>
अटल अंतरभूत व्योम vec_द्विआधारी_op_contiguous(स्थिर int64_t n, dst_t * z, स्थिर src0_t * x, स्थिर src1_t * y) {
    constexpr स्वतः src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr स्वतः src1_to_f32 = type_conversion_table<src1_t>::to_f32;
    constexpr स्वतः f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    क्रम (पूर्णांक i = 0; i < n; i++) {
        z[i] = f32_to_dst(op(src0_to_f32(x[i]), src1_to_f32(y[i])));
    }
}

ढाँचा <भग्न (*op)(भग्न, भग्न), typename src0_t, typename src1_t, typename dst_t>
अटल अंतरभूत व्योम vec_द्विआधारी_op_non_contiguous(स्थिर int64_t n, स्थिर int64_t ne10, स्थिर int64_t nb10, dst_t * z, स्थिर src0_t * x, स्थिर src1_t * y) {
    constexpr स्वतः src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr स्वतः src1_to_f32 = type_conversion_table<src1_t>::to_f32;
    constexpr स्वतः f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    क्रम (पूर्णांक i = 0; i < n; i++) {
        पूर्णांक i10 = i % ne10;
        स्थिर src1_t * y_ptr = (स्थिर src1_t *)((स्थिर अक्षर *)y + i10*nb10);
        z[i] = f32_to_dst(op(src0_to_f32(x[i]), src1_to_f32(*y_ptr)));
    }
}

ढाँचा <भग्न (*op)(भग्न, भग्न), typename src0_t, typename src1_t, typename dst_t>
अटल व्योम apply_द्विआधारी_op(स्थिर ggml_compute_params * params, ggml_tensor * dst) {
    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_द्विआधारी_OP_LOCALS

    GGML_ASSERT( nb0 == माप(dst_t));
    GGML_ASSERT(nb00 == माप(src0_t));

    स्थिर स्वतः [ir0, ir1] = get_thread_range(params, src0);
    स्थिर bool is_src1_contiguous_rows = ggml_is_contiguous_rows(src1);

#अगर_घोषित GGML_USE_ACCELERATE
    vDSP_fn_t vDSP_op = nullptr;
    // TODO - avoid the f32-only check using type 'trait' lookup tables and row-based src-to-float conversion functions
    अगर (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        अगर (op == op_add) {
            vDSP_op = vDSP_vadd;
        } अन्यथा अगर (op == op_sub) {
            vDSP_op = vDSP_vsub;
        } अन्यथा अगर (op == op_mul) {
            vDSP_op = vDSP_vmul;
        } अन्यथा अगर (op == op_div) {
            vDSP_op = vDSP_vdiv;
        }
    }
#पूर्ण_अगर

    क्रम (int64_t ir = ir0; ir < ir1; ++ir) {
        स्थिर int64_t i03 = ir/(ne02*ne01);
        स्थिर int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        स्थिर int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        स्थिर int64_t i13 = i03 % ne13;
        स्थिर int64_t i12 = i02 % ne12;
        स्थिर int64_t i11 = i01 % ne11;

        dst_t        * dst_ptr  = (dst_t  *)       ((अक्षर *)       dst->तथ्य  + i03*nb3  + i02*nb2  + i01*nb1 );
        स्थिर src0_t * src0_ptr = (स्थिर src0_t *) ((स्थिर अक्षर *) src0->तथ्य + i03*nb03 + i02*nb02 + i01*nb01);
        स्थिर src1_t * src1_ptr = (स्थिर src1_t *) ((स्थिर अक्षर *) src1->तथ्य + i13*nb13 + i12*nb12 + i11*nb11);

        अगर (is_src1_contiguous_rows) {
            // src1 is broadcastable across src0 and dst in i1, i2, i3
            स्थिर int64_t nr0 = ne00 / ne10;

            क्रम (int64_t r = 0; r < nr0; ++r) {
#अगर_घोषित GGML_USE_ACCELERATE
                अगर constexpr (std::is_same_v<src0_t, भग्न> && std::is_same_v<src1_t, भग्न> && std::is_same_v<dst_t, भग्न>) {
                    अगर (vDSP_op != nullptr) {
                        vDSP_op(src1_ptr, 1, src0_ptr + r*ne10, 1, dst_ptr + r*ne10, 1, ne10);
                        जारी;
                    }
                }
#पूर्ण_अगर
                vec_द्विआधारी_op_contiguous<op>(ne10, dst_ptr + r*ne10, src0_ptr + r*ne10, src1_ptr);
            }
        } अन्यथा {
            vec_द्विआधारी_op_non_contiguous<op>(ne0, ne10, nb10, dst_ptr, src0_ptr, src1_ptr);
        }
    }
}

// TODO: Use the 'traits' lookup table (for type conversion fns), instead of a mass of 'if' conditions with long templates
ढाँचा <भग्न (*op)(भग्न, भग्न)>
अटल व्योम द्विआधारी_op(स्थिर ggml_compute_params * params, ggml_tensor * dst) {
    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];

    /*  */ अगर (src0->type == GGML_TYPE_F32  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_द्विआधारी_op<op, भग्न, भग्न, भग्न>(params, dst);
    } अन्यथा अगर (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_द्विआधारी_op<op, ggml_fp16_t, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } अन्यथा अगर (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_द्विआधारी_op<op, ggml_bf16_t, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } अन्यथा अगर (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_BF16) {
        apply_द्विआधारी_op<op, ggml_bf16_t, भग्न, ggml_bf16_t>(params, dst);
    } अन्यथा अगर (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) {
        apply_द्विआधारी_op<op, ggml_bf16_t, भग्न, भग्न>(params, dst);
    } अन्यथा अगर (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F16) {
        apply_द्विआधारी_op<op, ggml_fp16_t, भग्न, ggml_fp16_t>(params, dst);
    } अन्यथा अगर (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) {
        apply_द्विआधारी_op<op, ggml_fp16_t, भग्न, भग्न>(params, dst);
    } अन्यथा {
        GGML_ABORT("%s: unsupported types: dst: %s, src0: %s, src1: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type), ggml_type_name(src1->type));
    }
}

व्योम ggml_compute_forward_add_non_quantized(स्थिर ggml_compute_params * params, ggml_tensor * dst) {
    द्विआधारी_op<op_add>(params, dst);
}

व्योम ggml_compute_forward_sub(स्थिर ggml_compute_params * params, ggml_tensor * dst) {
    द्विआधारी_op<op_sub>(params, dst);
}

व्योम ggml_compute_forward_mul(स्थिर ggml_compute_params * params, ggml_tensor * dst) {
    द्विआधारी_op<op_mul>(params, dst);
}

व्योम ggml_compute_forward_div(स्थिर ggml_compute_params * params, ggml_tensor * dst) {
    द्विआधारी_op<op_div>(params, dst);
}
