#அந்தர்பாவ "binary-ops.h"

#எனில் defined(GGML_USE_ACCELERATE)
#அந்தர்பாவ <Accelerate/Accelerate.h>

using vDSP_fn_t = வெறுமை (*)(மாறிலி தசம *, vDSP_Stride, மாறிலி தசம *, vDSP_Stride, தசம *, vDSP_Stride, vDSP_Length);
#endif

நிலையான உள்வரி தசம op_add(தசம a, தசம b) {
    திரும்பு a + b;
}

நிலையான உள்வரி தசம op_sub(தசம a, தசம b) {
    திரும்பு a - b;
}

நிலையான உள்வரி தசம op_mul(தசம a, தசம b) {
    திரும்பு a * b;
}

நிலையான உள்வரி தசம op_div(தசம a, தசம b) {
    திரும்பு a / b;
}

template <தசம (*op)(தசம, தசம), typename src0_t, typename src1_t, typename dst_t>
நிலையான உள்வரி வெறுமை vec_இருமம்_op_contiguous(மாறிலி int64_t n, dst_t * z, மாறிலி src0_t * x, மாறிலி src1_t * y) {
    constexpr auto src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr auto src1_to_f32 = type_conversion_table<src1_t>::to_f32;
    constexpr auto f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    க்கு (முழு i = 0; i < n; i++) {
        z[i] = f32_to_dst(op(src0_to_f32(x[i]), src1_to_f32(y[i])));
    }
}

template <தசம (*op)(தசம, தசம), typename src0_t, typename src1_t, typename dst_t>
நிலையான உள்வரி வெறுமை vec_இருமம்_op_non_contiguous(மாறிலி int64_t n, மாறிலி int64_t ne10, மாறிலி int64_t nb10, dst_t * z, மாறிலி src0_t * x, மாறிலி src1_t * y) {
    constexpr auto src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr auto src1_to_f32 = type_conversion_table<src1_t>::to_f32;
    constexpr auto f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    க்கு (முழு i = 0; i < n; i++) {
        முழு i10 = i % ne10;
        மாறிலி src1_t * y_ptr = (மாறிலி src1_t *)((மாறிலி எழுத்து *)y + i10*nb10);
        z[i] = f32_to_dst(op(src0_to_f32(x[i]), src1_to_f32(*y_ptr)));
    }
}

template <தசம (*op)(தசம, தசம), typename src0_t, typename src1_t, typename dst_t>
நிலையான வெறுமை apply_இருமம்_op(மாறிலி ggml_compute_params * params, ggml_tensor * dst) {
    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_இருமம்_OP_LOCALS

    GGML_ASSERT( nb0 == அளவு(dst_t));
    GGML_ASSERT(nb00 == அளவு(src0_t));

    மாறிலி auto [ir0, ir1] = get_thread_range(params, src0);
    மாறிலி தருக்க is_src1_contiguous_rows = ggml_is_contiguous_rows(src1);

#ifdef GGML_USE_ACCELERATE
    vDSP_fn_t vDSP_op = nullptr;
    // TODO - avoid the f32-only check using type 'trait' lookup tables and row-based src-to-float conversion functions
    எனில் (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        எனில் (op == op_add) {
            vDSP_op = vDSP_vadd;
        } இல்லையெனில் எனில் (op == op_sub) {
            vDSP_op = vDSP_vsub;
        } இல்லையெனில் எனில் (op == op_mul) {
            vDSP_op = vDSP_vmul;
        } இல்லையெனில் எனில் (op == op_div) {
            vDSP_op = vDSP_vdiv;
        }
    }
#endif

    க்கு (int64_t ir = ir0; ir < ir1; ++ir) {
        மாறிலி int64_t i03 = ir/(ne02*ne01);
        மாறிலி int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        மாறிலி int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        மாறிலி int64_t i13 = i03 % ne13;
        மாறிலி int64_t i12 = i02 % ne12;
        மாறிலி int64_t i11 = i01 % ne11;

        dst_t        * dst_ptr  = (dst_t  *)       ((எழுத்து *)       dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
        மாறிலி src0_t * src0_ptr = (மாறிலி src0_t *) ((மாறிலி எழுத்து *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);
        மாறிலி src1_t * src1_ptr = (மாறிலி src1_t *) ((மாறிலி எழுத்து *) src1->data + i13*nb13 + i12*nb12 + i11*nb11);

        எனில் (is_src1_contiguous_rows) {
            // src1 is broadcastable across src0 and dst in i1, i2, i3
            மாறிலி int64_t nr0 = ne00 / ne10;

            க்கு (int64_t r = 0; r < nr0; ++r) {
#ifdef GGML_USE_ACCELERATE
                எனில் constexpr (std::is_same_v<src0_t, தசம> && std::is_same_v<src1_t, தசம> && std::is_same_v<dst_t, தசம>) {
                    எனில் (vDSP_op != nullptr) {
                        vDSP_op(src1_ptr, 1, src0_ptr + r*ne10, 1, dst_ptr + r*ne10, 1, ne10);
                        தொடர்;
                    }
                }
#endif
                vec_இருமம்_op_contiguous<op>(ne10, dst_ptr + r*ne10, src0_ptr + r*ne10, src1_ptr);
            }
        } இல்லையெனில் {
            vec_இருமம்_op_non_contiguous<op>(ne0, ne10, nb10, dst_ptr, src0_ptr, src1_ptr);
        }
    }
}

// TODO: Use the 'traits' lookup table (for type conversion fns), instead of a mass of 'if' conditions with long templates
template <தசம (*op)(தசம, தசம)>
நிலையான வெறுமை இருமம்_op(மாறிலி ggml_compute_params * params, ggml_tensor * dst) {
    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];

    /*  */ எனில் (src0->type == GGML_TYPE_F32  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_இருமம்_op<op, தசம, தசம, தசம>(params, dst);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_இருமம்_op<op, ggml_fp16_t, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_இருமம்_op<op, ggml_bf16_t, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_BF16) {
        apply_இருமம்_op<op, ggml_bf16_t, தசம, ggml_bf16_t>(params, dst);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) {
        apply_இருமம்_op<op, ggml_bf16_t, தசம, தசம>(params, dst);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F16) {
        apply_இருமம்_op<op, ggml_fp16_t, தசம, ggml_fp16_t>(params, dst);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) {
        apply_இருமம்_op<op, ggml_fp16_t, தசம, தசம>(params, dst);
    } இல்லையெனில் {
        GGML_ABORT("%s: unsupported types: dst: %s, src0: %s, src1: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type), ggml_type_name(src1->type));
    }
}

வெறுமை ggml_compute_forward_add_non_quantized(மாறிலி ggml_compute_params * params, ggml_tensor * dst) {
    இருமம்_op<op_add>(params, dst);
}

வெறுமை ggml_compute_forward_sub(மாறிலி ggml_compute_params * params, ggml_tensor * dst) {
    இருமம்_op<op_sub>(params, dst);
}

வெறுமை ggml_compute_forward_mul(மாறிலி ggml_compute_params * params, ggml_tensor * dst) {
    இருமம்_op<op_mul>(params, dst);
}

வெறுமை ggml_compute_forward_div(மாறிலி ggml_compute_params * params, ggml_tensor * dst) {
    இருமம்_op<op_div>(params, dst);
}
