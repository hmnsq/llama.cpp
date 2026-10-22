#അന്തര്ഭാവ "binary-ops.h"

#എങ്കിൽ defined(GGML_USE_ACCELERATE)
#അന്തര്ഭാവ <Accelerate/Accelerate.h>

using vDSP_fn_t = ശൂന്യം (*)(സ്ഥിരം ദശാംശം *, vDSP_Stride, സ്ഥിരം ദശാംശം *, vDSP_Stride, ദശാംശം *, vDSP_Stride, vDSP_Length);
#endif

സ്ഥിതം വരിയിൽ ദശാംശം op_add(ദശാംശം a, ദശാംശം b) {
    തിരികെ a + b;
}

സ്ഥിതം വരിയിൽ ദശാംശം op_sub(ദശാംശം a, ദശാംശം b) {
    തിരികെ a - b;
}

സ്ഥിതം വരിയിൽ ദശാംശം op_mul(ദശാംശം a, ദശാംശം b) {
    തിരികെ a * b;
}

സ്ഥിതം വരിയിൽ ദശാംശം op_div(ദശാംശം a, ദശാംശം b) {
    തിരികെ a / b;
}

template <ദശാംശം (*op)(ദശാംശം, ദശാംശം), typename src0_t, typename src1_t, typename dst_t>
സ്ഥിതം വരിയിൽ ശൂന്യം vec_ദ്വിമാന_op_contiguous(സ്ഥിരം int64_t n, dst_t * z, സ്ഥിരം src0_t * x, സ്ഥിരം src1_t * y) {
    constexpr auto src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr auto src1_to_f32 = type_conversion_table<src1_t>::to_f32;
    constexpr auto f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n; i++) {
        z[i] = f32_to_dst(op(src0_to_f32(x[i]), src1_to_f32(y[i])));
    }
}

template <ദശാംശം (*op)(ദശാംശം, ദശാംശം), typename src0_t, typename src1_t, typename dst_t>
സ്ഥിതം വരിയിൽ ശൂന്യം vec_ദ്വിമാന_op_non_contiguous(സ്ഥിരം int64_t n, സ്ഥിരം int64_t ne10, സ്ഥിരം int64_t nb10, dst_t * z, സ്ഥിരം src0_t * x, സ്ഥിരം src1_t * y) {
    constexpr auto src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr auto src1_to_f32 = type_conversion_table<src1_t>::to_f32;
    constexpr auto f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n; i++) {
        പൂർണ്ണസംഖ്യ i10 = i % ne10;
        സ്ഥിരം src1_t * y_ptr = (സ്ഥിരം src1_t *)((സ്ഥിരം അക്ഷരം *)y + i10*nb10);
        z[i] = f32_to_dst(op(src0_to_f32(x[i]), src1_to_f32(*y_ptr)));
    }
}

template <ദശാംശം (*op)(ദശാംശം, ദശാംശം), typename src0_t, typename src1_t, typename dst_t>
സ്ഥിതം ശൂന്യം apply_ദ്വിമാന_op(സ്ഥിരം ggml_compute_params * params, ggml_tensor * dst) {
    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_ദ്വിമാന_OP_LOCALS

    GGML_ASSERT( nb0 == വലുപ്പം(dst_t));
    GGML_ASSERT(nb00 == വലുപ്പം(src0_t));

    സ്ഥിരം auto [ir0, ir1] = get_thread_range(params, src0);
    സ്ഥിരം തർക്കം is_src1_contiguous_rows = ggml_is_contiguous_rows(src1);

#ifdef GGML_USE_ACCELERATE
    vDSP_fn_t vDSP_op = nullptr;
    // TODO - avoid the f32-only check using type 'trait' lookup tables and row-based src-to-float conversion functions
    എങ്കിൽ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        എങ്കിൽ (op == op_add) {
            vDSP_op = vDSP_vadd;
        } അല്ലെങ്കിൽ എങ്കിൽ (op == op_sub) {
            vDSP_op = vDSP_vsub;
        } അല്ലെങ്കിൽ എങ്കിൽ (op == op_mul) {
            vDSP_op = vDSP_vmul;
        } അല്ലെങ്കിൽ എങ്കിൽ (op == op_div) {
            vDSP_op = vDSP_vdiv;
        }
    }
#endif

    വേണ്ടി (int64_t ir = ir0; ir < ir1; ++ir) {
        സ്ഥിരം int64_t i03 = ir/(ne02*ne01);
        സ്ഥിരം int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        സ്ഥിരം int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        സ്ഥിരം int64_t i13 = i03 % ne13;
        സ്ഥിരം int64_t i12 = i02 % ne12;
        സ്ഥിരം int64_t i11 = i01 % ne11;

        dst_t        * dst_ptr  = (dst_t  *)       ((അക്ഷരം *)       dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
        സ്ഥിരം src0_t * src0_ptr = (സ്ഥിരം src0_t *) ((സ്ഥിരം അക്ഷരം *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);
        സ്ഥിരം src1_t * src1_ptr = (സ്ഥിരം src1_t *) ((സ്ഥിരം അക്ഷരം *) src1->data + i13*nb13 + i12*nb12 + i11*nb11);

        എങ്കിൽ (is_src1_contiguous_rows) {
            // src1 is broadcastable across src0 and dst in i1, i2, i3
            സ്ഥിരം int64_t nr0 = ne00 / ne10;

            വേണ്ടി (int64_t r = 0; r < nr0; ++r) {
#ifdef GGML_USE_ACCELERATE
                എങ്കിൽ constexpr (std::is_same_v<src0_t, ദശാംശം> && std::is_same_v<src1_t, ദശാംശം> && std::is_same_v<dst_t, ദശാംശം>) {
                    എങ്കിൽ (vDSP_op != nullptr) {
                        vDSP_op(src1_ptr, 1, src0_ptr + r*ne10, 1, dst_ptr + r*ne10, 1, ne10);
                        തുടരുക;
                    }
                }
#endif
                vec_ദ്വിമാന_op_contiguous<op>(ne10, dst_ptr + r*ne10, src0_ptr + r*ne10, src1_ptr);
            }
        } അല്ലെങ്കിൽ {
            vec_ദ്വിമാന_op_non_contiguous<op>(ne0, ne10, nb10, dst_ptr, src0_ptr, src1_ptr);
        }
    }
}

// TODO: Use the 'traits' lookup table (for type conversion fns), instead of a mass of 'if' conditions with long templates
template <ദശാംശം (*op)(ദശാംശം, ദശാംശം)>
സ്ഥിതം ശൂന്യം ദ്വിമാന_op(സ്ഥിരം ggml_compute_params * params, ggml_tensor * dst) {
    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];

    /*  */ എങ്കിൽ (src0->type == GGML_TYPE_F32  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_ദ്വിമാന_op<op, ദശാംശം, ദശാംശം, ദശാംശം>(params, dst);
    } അല്ലെങ്കിൽ എങ്കിൽ (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_ദ്വിമാന_op<op, ggml_fp16_t, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } അല്ലെങ്കിൽ എങ്കിൽ (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_ദ്വിമാന_op<op, ggml_bf16_t, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } അല്ലെങ്കിൽ എങ്കിൽ (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_BF16) {
        apply_ദ്വിമാന_op<op, ggml_bf16_t, ദശാംശം, ggml_bf16_t>(params, dst);
    } അല്ലെങ്കിൽ എങ്കിൽ (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) {
        apply_ദ്വിമാന_op<op, ggml_bf16_t, ദശാംശം, ദശാംശം>(params, dst);
    } അല്ലെങ്കിൽ എങ്കിൽ (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F16) {
        apply_ദ്വിമാന_op<op, ggml_fp16_t, ദശാംശം, ggml_fp16_t>(params, dst);
    } അല്ലെങ്കിൽ എങ്കിൽ (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) {
        apply_ദ്വിമാന_op<op, ggml_fp16_t, ദശാംശം, ദശാംശം>(params, dst);
    } അല്ലെങ്കിൽ {
        GGML_ABORT("%s: unsupported types: dst: %s, src0: %s, src1: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type), ggml_type_name(src1->type));
    }
}

ശൂന്യം ggml_compute_forward_add_non_quantized(സ്ഥിരം ggml_compute_params * params, ggml_tensor * dst) {
    ദ്വിമാന_op<op_add>(params, dst);
}

ശൂന്യം ggml_compute_forward_sub(സ്ഥിരം ggml_compute_params * params, ggml_tensor * dst) {
    ദ്വിമാന_op<op_sub>(params, dst);
}

ശൂന്യം ggml_compute_forward_mul(സ്ഥിരം ggml_compute_params * params, ggml_tensor * dst) {
    ദ്വിമാന_op<op_mul>(params, dst);
}

ശൂന്യം ggml_compute_forward_div(സ്ഥിരം ggml_compute_params * params, ggml_tensor * dst) {
    ദ്വിമാന_op<op_div>(params, dst);
}
