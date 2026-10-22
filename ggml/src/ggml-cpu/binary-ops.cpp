#ଅନ୍ତର୍ଭାଵ "binary-ops.h"

#ଯଦି defined(GGML_USE_ACCELERATE)
#ଅନ୍ତର୍ଭାଵ <Accelerate/Accelerate.h>

using vDSP_fn_t = ଶୂନ୍ୟ (*)(ଅଚଳ ଦଶମିକ *, vDSP_Stride, ଅଚଳ ଦଶମିକ *, vDSP_Stride, ଦଶମିକ *, vDSP_Stride, vDSP_Length);
#endif

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ op_add(ଦଶମିକ a, ଦଶମିକ b) {
    ଫେରନ୍ତୁ a + b;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ op_sub(ଦଶମିକ a, ଦଶମିକ b) {
    ଫେରନ୍ତୁ a - b;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ op_mul(ଦଶମିକ a, ଦଶମିକ b) {
    ଫେରନ୍ତୁ a * b;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ op_div(ଦଶମିକ a, ଦଶମିକ b) {
    ଫେରନ୍ତୁ a / b;
}

template <ଦଶମିକ (*op)(ଦଶମିକ, ଦଶମିକ), typename src0_t, typename src1_t, typename dst_t>
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ vec_ଦ୍ୱିମିକ_op_contiguous(ଅଚଳ int64_t n, dst_t * z, ଅଚଳ src0_t * x, ଅଚଳ src1_t * y) {
    constexpr auto src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr auto src1_to_f32 = type_conversion_table<src1_t>::to_f32;
    constexpr auto f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; i++) {
        z[i] = f32_to_dst(op(src0_to_f32(x[i]), src1_to_f32(y[i])));
    }
}

template <ଦଶମିକ (*op)(ଦଶମିକ, ଦଶମିକ), typename src0_t, typename src1_t, typename dst_t>
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ vec_ଦ୍ୱିମିକ_op_non_contiguous(ଅଚଳ int64_t n, ଅଚଳ int64_t ne10, ଅଚଳ int64_t nb10, dst_t * z, ଅଚଳ src0_t * x, ଅଚଳ src1_t * y) {
    constexpr auto src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr auto src1_to_f32 = type_conversion_table<src1_t>::to_f32;
    constexpr auto f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; i++) {
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା i10 = i % ne10;
        ଅଚଳ src1_t * y_ptr = (ଅଚଳ src1_t *)((ଅଚଳ ଅକ୍ଷର *)y + i10*nb10);
        z[i] = f32_to_dst(op(src0_to_f32(x[i]), src1_to_f32(*y_ptr)));
    }
}

template <ଦଶମିକ (*op)(ଦଶମିକ, ଦଶମିକ), typename src0_t, typename src1_t, typename dst_t>
ସ୍ଥିର ଶୂନ୍ୟ apply_ଦ୍ୱିମିକ_op(ଅଚଳ ggml_compute_params * params, ggml_tensor * dst) {
    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_ଦ୍ୱିମିକ_OP_LOCALS

    GGML_ASSERT( nb0 == ଆକାର(dst_t));
    GGML_ASSERT(nb00 == ଆକାର(src0_t));

    ଅଚଳ auto [ir0, ir1] = get_thread_range(params, src0);
    ଅଚଳ ତାର୍କିକ is_src1_contiguous_rows = ggml_is_contiguous_rows(src1);

#ifdef GGML_USE_ACCELERATE
    vDSP_fn_t vDSP_op = nullptr;
    // TODO - avoid the f32-only check using type 'trait' lookup tables and row-based src-to-float conversion functions
    ଯଦି (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        ଯଦି (op == op_add) {
            vDSP_op = vDSP_vadd;
        } ଅନ୍ୟଥା ଯଦି (op == op_sub) {
            vDSP_op = vDSP_vsub;
        } ଅନ୍ୟଥା ଯଦି (op == op_mul) {
            vDSP_op = vDSP_vmul;
        } ଅନ୍ୟଥା ଯଦି (op == op_div) {
            vDSP_op = vDSP_vdiv;
        }
    }
#endif

    ପାଇଁ (int64_t ir = ir0; ir < ir1; ++ir) {
        ଅଚଳ int64_t i03 = ir/(ne02*ne01);
        ଅଚଳ int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        ଅଚଳ int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        ଅଚଳ int64_t i13 = i03 % ne13;
        ଅଚଳ int64_t i12 = i02 % ne12;
        ଅଚଳ int64_t i11 = i01 % ne11;

        dst_t        * dst_ptr  = (dst_t  *)       ((ଅକ୍ଷର *)       dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
        ଅଚଳ src0_t * src0_ptr = (ଅଚଳ src0_t *) ((ଅଚଳ ଅକ୍ଷର *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);
        ଅଚଳ src1_t * src1_ptr = (ଅଚଳ src1_t *) ((ଅଚଳ ଅକ୍ଷର *) src1->data + i13*nb13 + i12*nb12 + i11*nb11);

        ଯଦି (is_src1_contiguous_rows) {
            // src1 is broadcastable across src0 and dst in i1, i2, i3
            ଅଚଳ int64_t nr0 = ne00 / ne10;

            ପାଇଁ (int64_t r = 0; r < nr0; ++r) {
#ifdef GGML_USE_ACCELERATE
                ଯଦି constexpr (std::is_same_v<src0_t, ଦଶମିକ> && std::is_same_v<src1_t, ଦଶମିକ> && std::is_same_v<dst_t, ଦଶମିକ>) {
                    ଯଦି (vDSP_op != nullptr) {
                        vDSP_op(src1_ptr, 1, src0_ptr + r*ne10, 1, dst_ptr + r*ne10, 1, ne10);
                        ଜାରି;
                    }
                }
#endif
                vec_ଦ୍ୱିମିକ_op_contiguous<op>(ne10, dst_ptr + r*ne10, src0_ptr + r*ne10, src1_ptr);
            }
        } ଅନ୍ୟଥା {
            vec_ଦ୍ୱିମିକ_op_non_contiguous<op>(ne0, ne10, nb10, dst_ptr, src0_ptr, src1_ptr);
        }
    }
}

// TODO: Use the 'traits' lookup table (for type conversion fns), instead of a mass of 'if' conditions with long templates
template <ଦଶମିକ (*op)(ଦଶମିକ, ଦଶମିକ)>
ସ୍ଥିର ଶୂନ୍ୟ ଦ୍ୱିମିକ_op(ଅଚଳ ggml_compute_params * params, ggml_tensor * dst) {
    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];

    /*  */ ଯଦି (src0->type == GGML_TYPE_F32  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_ଦ୍ୱିମିକ_op<op, ଦଶମିକ, ଦଶମିକ, ଦଶମିକ>(params, dst);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_ଦ୍ୱିମିକ_op<op, ggml_fp16_t, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_ଦ୍ୱିମିକ_op<op, ggml_bf16_t, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_BF16) {
        apply_ଦ୍ୱିମିକ_op<op, ggml_bf16_t, ଦଶମିକ, ggml_bf16_t>(params, dst);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) {
        apply_ଦ୍ୱିମିକ_op<op, ggml_bf16_t, ଦଶମିକ, ଦଶମିକ>(params, dst);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F16) {
        apply_ଦ୍ୱିମିକ_op<op, ggml_fp16_t, ଦଶମିକ, ggml_fp16_t>(params, dst);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) {
        apply_ଦ୍ୱିମିକ_op<op, ggml_fp16_t, ଦଶମିକ, ଦଶମିକ>(params, dst);
    } ଅନ୍ୟଥା {
        GGML_ABORT("%s: unsupported types: dst: %s, src0: %s, src1: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type), ggml_type_name(src1->type));
    }
}

ଶୂନ୍ୟ ggml_compute_forward_add_non_quantized(ଅଚଳ ggml_compute_params * params, ggml_tensor * dst) {
    ଦ୍ୱିମିକ_op<op_add>(params, dst);
}

ଶୂନ୍ୟ ggml_compute_forward_sub(ଅଚଳ ggml_compute_params * params, ggml_tensor * dst) {
    ଦ୍ୱିମିକ_op<op_sub>(params, dst);
}

ଶୂନ୍ୟ ggml_compute_forward_mul(ଅଚଳ ggml_compute_params * params, ggml_tensor * dst) {
    ଦ୍ୱିମିକ_op<op_mul>(params, dst);
}

ଶୂନ୍ୟ ggml_compute_forward_div(ଅଚଳ ggml_compute_params * params, ggml_tensor * dst) {
    ଦ୍ୱିମିକ_op<op_div>(params, dst);
}
