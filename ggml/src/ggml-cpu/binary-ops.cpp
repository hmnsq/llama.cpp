#અન્તર્ભાવ "binary-ops.h"

#જો defined(GGML_USE_ACCELERATE)
#અન્તર્ભાવ <Accelerate/Accelerate.h>

using vDSP_fn_t = શૂન્ય (*)(અચલ દશાંશ *, vDSP_Stride, અચલ દશાંશ *, vDSP_Stride, દશાંશ *, vDSP_Stride, vDSP_Length);
#endif

સ્થિર અંતઃરેખ દશાંશ op_add(દશાંશ a, દશાંશ b) {
    પાછા a + b;
}

સ્થિર અંતઃરેખ દશાંશ op_sub(દશાંશ a, દશાંશ b) {
    પાછા a - b;
}

સ્થિર અંતઃરેખ દશાંશ op_mul(દશાંશ a, દશાંશ b) {
    પાછા a * b;
}

સ્થિર અંતઃરેખ દશાંશ op_div(દશાંશ a, દશાંશ b) {
    પાછા a / b;
}

template <દશાંશ (*op)(દશાંશ, દશાંશ), typename src0_t, typename src1_t, typename dst_t>
સ્થિર અંતઃરેખ શૂન્ય vec_દ્વિઅંકી_op_contiguous(અચલ int64_t n, dst_t * z, અચલ src0_t * x, અચલ src1_t * y) {
    constexpr auto src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr auto src1_to_f32 = type_conversion_table<src1_t>::to_f32;
    constexpr auto f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    માટે (પૂર્ણાંક i = 0; i < n; i++) {
        z[i] = f32_to_dst(op(src0_to_f32(x[i]), src1_to_f32(y[i])));
    }
}

template <દશાંશ (*op)(દશાંશ, દશાંશ), typename src0_t, typename src1_t, typename dst_t>
સ્થિર અંતઃરેખ શૂન્ય vec_દ્વિઅંકી_op_non_contiguous(અચલ int64_t n, અચલ int64_t ne10, અચલ int64_t nb10, dst_t * z, અચલ src0_t * x, અચલ src1_t * y) {
    constexpr auto src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr auto src1_to_f32 = type_conversion_table<src1_t>::to_f32;
    constexpr auto f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    માટે (પૂર્ણાંક i = 0; i < n; i++) {
        પૂર્ણાંક i10 = i % ne10;
        અચલ src1_t * y_ptr = (અચલ src1_t *)((અચલ અક્ષર *)y + i10*nb10);
        z[i] = f32_to_dst(op(src0_to_f32(x[i]), src1_to_f32(*y_ptr)));
    }
}

template <દશાંશ (*op)(દશાંશ, દશાંશ), typename src0_t, typename src1_t, typename dst_t>
સ્થિર શૂન્ય apply_દ્વિઅંકી_op(અચલ ggml_compute_params * params, ggml_tensor * dst) {
    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_દ્વિઅંકી_OP_LOCALS

    GGML_ASSERT( nb0 == કદ(dst_t));
    GGML_ASSERT(nb00 == કદ(src0_t));

    અચલ auto [ir0, ir1] = get_thread_range(params, src0);
    અચલ તાર્કિક is_src1_contiguous_rows = ggml_is_contiguous_rows(src1);

#ifdef GGML_USE_ACCELERATE
    vDSP_fn_t vDSP_op = nullptr;
    // TODO - avoid the f32-only check using type 'trait' lookup tables and row-based src-to-float conversion functions
    જો (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        જો (op == op_add) {
            vDSP_op = vDSP_vadd;
        } નહીંતર જો (op == op_sub) {
            vDSP_op = vDSP_vsub;
        } નહીંતર જો (op == op_mul) {
            vDSP_op = vDSP_vmul;
        } નહીંતર જો (op == op_div) {
            vDSP_op = vDSP_vdiv;
        }
    }
#endif

    માટે (int64_t ir = ir0; ir < ir1; ++ir) {
        અચલ int64_t i03 = ir/(ne02*ne01);
        અચલ int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        અચલ int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        અચલ int64_t i13 = i03 % ne13;
        અચલ int64_t i12 = i02 % ne12;
        અચલ int64_t i11 = i01 % ne11;

        dst_t        * dst_ptr  = (dst_t  *)       ((અક્ષર *)       dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
        અચલ src0_t * src0_ptr = (અચલ src0_t *) ((અચલ અક્ષર *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);
        અચલ src1_t * src1_ptr = (અચલ src1_t *) ((અચલ અક્ષર *) src1->data + i13*nb13 + i12*nb12 + i11*nb11);

        જો (is_src1_contiguous_rows) {
            // src1 is broadcastable across src0 and dst in i1, i2, i3
            અચલ int64_t nr0 = ne00 / ne10;

            માટે (int64_t r = 0; r < nr0; ++r) {
#ifdef GGML_USE_ACCELERATE
                જો constexpr (std::is_same_v<src0_t, દશાંશ> && std::is_same_v<src1_t, દશાંશ> && std::is_same_v<dst_t, દશાંશ>) {
                    જો (vDSP_op != nullptr) {
                        vDSP_op(src1_ptr, 1, src0_ptr + r*ne10, 1, dst_ptr + r*ne10, 1, ne10);
                        ચાલુ;
                    }
                }
#endif
                vec_દ્વિઅંકી_op_contiguous<op>(ne10, dst_ptr + r*ne10, src0_ptr + r*ne10, src1_ptr);
            }
        } નહીંતર {
            vec_દ્વિઅંકી_op_non_contiguous<op>(ne0, ne10, nb10, dst_ptr, src0_ptr, src1_ptr);
        }
    }
}

// TODO: Use the 'traits' lookup table (for type conversion fns), instead of a mass of 'if' conditions with long templates
template <દશાંશ (*op)(દશાંશ, દશાંશ)>
સ્થિર શૂન્ય દ્વિઅંકી_op(અચલ ggml_compute_params * params, ggml_tensor * dst) {
    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];

    /*  */ જો (src0->type == GGML_TYPE_F32  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_દ્વિઅંકી_op<op, દશાંશ, દશાંશ, દશાંશ>(params, dst);
    } નહીંતર જો (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_દ્વિઅંકી_op<op, ggml_fp16_t, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } નહીંતર જો (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_દ્વિઅંકી_op<op, ggml_bf16_t, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } નહીંતર જો (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_BF16) {
        apply_દ્વિઅંકી_op<op, ggml_bf16_t, દશાંશ, ggml_bf16_t>(params, dst);
    } નહીંતર જો (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) {
        apply_દ્વિઅંકી_op<op, ggml_bf16_t, દશાંશ, દશાંશ>(params, dst);
    } નહીંતર જો (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F16) {
        apply_દ્વિઅંકી_op<op, ggml_fp16_t, દશાંશ, ggml_fp16_t>(params, dst);
    } નહીંતર જો (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) {
        apply_દ્વિઅંકી_op<op, ggml_fp16_t, દશાંશ, દશાંશ>(params, dst);
    } નહીંતર {
        GGML_ABORT("%s: unsupported types: dst: %s, src0: %s, src1: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type), ggml_type_name(src1->type));
    }
}

શૂન્ય ggml_compute_forward_add_non_quantized(અચલ ggml_compute_params * params, ggml_tensor * dst) {
    દ્વિઅંકી_op<op_add>(params, dst);
}

શૂન્ય ggml_compute_forward_sub(અચલ ggml_compute_params * params, ggml_tensor * dst) {
    દ્વિઅંકી_op<op_sub>(params, dst);
}

શૂન્ય ggml_compute_forward_mul(અચલ ggml_compute_params * params, ggml_tensor * dst) {
    દ્વિઅંકી_op<op_mul>(params, dst);
}

શૂન્ય ggml_compute_forward_div(અચલ ggml_compute_params * params, ggml_tensor * dst) {
    દ્વિઅંકી_op<op_div>(params, dst);
}
