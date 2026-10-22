#অন্তর্ভাৱ "binary-ops.h"

#যদি defined(GGML_USE_ACCELERATE)
#অন্তর্ভাৱ <Accelerate/Accelerate.h>

using vDSP_fn_t = শূন্য (*)(ধ্রুৱ ভিন্ন *, vDSP_Stride, ধ্রুৱ ভিন্ন *, vDSP_Stride, ভিন্ন *, vDSP_Stride, vDSP_Length);
#endif

স্থির অন্তঃস্থ ভিন্ন op_add(ভিন্ন a, ভিন্ন b) {
    ঘূৰাই a + b;
}

স্থির অন্তঃস্থ ভিন্ন op_sub(ভিন্ন a, ভিন্ন b) {
    ঘূৰাই a - b;
}

স্থির অন্তঃস্থ ভিন্ন op_mul(ভিন্ন a, ভিন্ন b) {
    ঘূৰাই a * b;
}

স্থির অন্তঃস্থ ভিন্ন op_div(ভিন্ন a, ভিন্ন b) {
    ঘূৰাই a / b;
}

template <ভিন্ন (*op)(ভিন্ন, ভিন্ন), typename src0_t, typename src1_t, typename dst_t>
স্থির অন্তঃস্থ শূন্য vec_দ্বিমিক_op_contiguous(ধ্রুৱ int64_t n, dst_t * z, ধ্রুৱ src0_t * x, ধ্রুৱ src1_t * y) {
    constexpr auto src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr auto src1_to_f32 = type_conversion_table<src1_t>::to_f32;
    constexpr auto f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    বাবে (পূর্ণাঙ্ক i = 0; i < n; i++) {
        z[i] = f32_to_dst(op(src0_to_f32(x[i]), src1_to_f32(y[i])));
    }
}

template <ভিন্ন (*op)(ভিন্ন, ভিন্ন), typename src0_t, typename src1_t, typename dst_t>
স্থির অন্তঃস্থ শূন্য vec_দ্বিমিক_op_non_contiguous(ধ্রুৱ int64_t n, ধ্রুৱ int64_t ne10, ধ্রুৱ int64_t nb10, dst_t * z, ধ্রুৱ src0_t * x, ধ্রুৱ src1_t * y) {
    constexpr auto src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr auto src1_to_f32 = type_conversion_table<src1_t>::to_f32;
    constexpr auto f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    বাবে (পূর্ণাঙ্ক i = 0; i < n; i++) {
        পূর্ণাঙ্ক i10 = i % ne10;
        ধ্রুৱ src1_t * y_ptr = (ধ্রুৱ src1_t *)((ধ্রুৱ ৱর্ণ *)y + i10*nb10);
        z[i] = f32_to_dst(op(src0_to_f32(x[i]), src1_to_f32(*y_ptr)));
    }
}

template <ভিন্ন (*op)(ভিন্ন, ভিন্ন), typename src0_t, typename src1_t, typename dst_t>
স্থির শূন্য apply_দ্বিমিক_op(ধ্রুৱ ggml_compute_params * params, ggml_tensor * dst) {
    ধ্রুৱ ggml_tensor * src0 = dst->src[0];
    ধ্রুৱ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_দ্বিমিক_OP_LOCALS

    GGML_ASSERT( nb0 == পরিমাণ(dst_t));
    GGML_ASSERT(nb00 == পরিমাণ(src0_t));

    ধ্রুৱ auto [ir0, ir1] = get_thread_range(params, src0);
    ধ্রুৱ তার্কিক is_src1_contiguous_rows = ggml_is_contiguous_rows(src1);

#ifdef GGML_USE_ACCELERATE
    vDSP_fn_t vDSP_op = nullptr;
    // TODO - avoid the f32-only check using type 'trait' lookup tables and row-based src-to-float conversion functions
    যদি (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        যদি (op == op_add) {
            vDSP_op = vDSP_vadd;
        } অন্যথা যদি (op == op_sub) {
            vDSP_op = vDSP_vsub;
        } অন্যথা যদি (op == op_mul) {
            vDSP_op = vDSP_vmul;
        } অন্যথা যদি (op == op_div) {
            vDSP_op = vDSP_vdiv;
        }
    }
#endif

    বাবে (int64_t ir = ir0; ir < ir1; ++ir) {
        ধ্রুৱ int64_t i03 = ir/(ne02*ne01);
        ধ্রুৱ int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        ধ্রুৱ int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        ধ্রুৱ int64_t i13 = i03 % ne13;
        ধ্রুৱ int64_t i12 = i02 % ne12;
        ধ্রুৱ int64_t i11 = i01 % ne11;

        dst_t        * dst_ptr  = (dst_t  *)       ((ৱর্ণ *)       dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
        ধ্রুৱ src0_t * src0_ptr = (ধ্রুৱ src0_t *) ((ধ্রুৱ ৱর্ণ *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);
        ধ্রুৱ src1_t * src1_ptr = (ধ্রুৱ src1_t *) ((ধ্রুৱ ৱর্ণ *) src1->data + i13*nb13 + i12*nb12 + i11*nb11);

        যদি (is_src1_contiguous_rows) {
            // src1 is broadcastable across src0 and dst in i1, i2, i3
            ধ্রুৱ int64_t nr0 = ne00 / ne10;

            বাবে (int64_t r = 0; r < nr0; ++r) {
#ifdef GGML_USE_ACCELERATE
                যদি constexpr (std::is_same_v<src0_t, ভিন্ন> && std::is_same_v<src1_t, ভিন্ন> && std::is_same_v<dst_t, ভিন্ন>) {
                    যদি (vDSP_op != nullptr) {
                        vDSP_op(src1_ptr, 1, src0_ptr + r*ne10, 1, dst_ptr + r*ne10, 1, ne10);
                        অনুৱর্ত;
                    }
                }
#endif
                vec_দ্বিমিক_op_contiguous<op>(ne10, dst_ptr + r*ne10, src0_ptr + r*ne10, src1_ptr);
            }
        } অন্যথা {
            vec_দ্বিমিক_op_non_contiguous<op>(ne0, ne10, nb10, dst_ptr, src0_ptr, src1_ptr);
        }
    }
}

// TODO: Use the 'traits' lookup table (for type conversion fns), instead of a mass of 'if' conditions with long templates
template <ভিন্ন (*op)(ভিন্ন, ভিন্ন)>
স্থির শূন্য দ্বিমিক_op(ধ্রুৱ ggml_compute_params * params, ggml_tensor * dst) {
    ধ্রুৱ ggml_tensor * src0 = dst->src[0];
    ধ্রুৱ ggml_tensor * src1 = dst->src[1];

    /*  */ যদি (src0->type == GGML_TYPE_F32  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_দ্বিমিক_op<op, ভিন্ন, ভিন্ন, ভিন্ন>(params, dst);
    } অন্যথা যদি (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_দ্বিমিক_op<op, ggml_fp16_t, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } অন্যথা যদি (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_দ্বিমিক_op<op, ggml_bf16_t, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } অন্যথা যদি (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_BF16) {
        apply_দ্বিমিক_op<op, ggml_bf16_t, ভিন্ন, ggml_bf16_t>(params, dst);
    } অন্যথা যদি (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) {
        apply_দ্বিমিক_op<op, ggml_bf16_t, ভিন্ন, ভিন্ন>(params, dst);
    } অন্যথা যদি (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F16) {
        apply_দ্বিমিক_op<op, ggml_fp16_t, ভিন্ন, ggml_fp16_t>(params, dst);
    } অন্যথা যদি (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) {
        apply_দ্বিমিক_op<op, ggml_fp16_t, ভিন্ন, ভিন্ন>(params, dst);
    } অন্যথা {
        GGML_ABORT("%s: unsupported types: dst: %s, src0: %s, src1: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type), ggml_type_name(src1->type));
    }
}

শূন্য ggml_compute_forward_add_non_quantized(ধ্রুৱ ggml_compute_params * params, ggml_tensor * dst) {
    দ্বিমিক_op<op_add>(params, dst);
}

শূন্য ggml_compute_forward_sub(ধ্রুৱ ggml_compute_params * params, ggml_tensor * dst) {
    দ্বিমিক_op<op_sub>(params, dst);
}

শূন্য ggml_compute_forward_mul(ধ্রুৱ ggml_compute_params * params, ggml_tensor * dst) {
    দ্বিমিক_op<op_mul>(params, dst);
}

শূন্য ggml_compute_forward_div(ধ্রুৱ ggml_compute_params * params, ggml_tensor * dst) {
    দ্বিমিক_op<op_div>(params, dst);
}
