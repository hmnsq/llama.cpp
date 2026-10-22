#అన్తర్భావ "binary-ops.h"

#ఒకవేళ defined(GGML_USE_ACCELERATE)
#అన్తర్భావ <Accelerate/Accelerate.h>

ఉపయోగించి vDSP_fn_t = శూన్య (*)(స్థిరరాశి దశాంశం *, vDSP_Stride, స్థిరరాశి దశాంశం *, vDSP_Stride, దశాంశం *, vDSP_Stride, v_DSP_నిడివి);
#అయితే_ముగింపు

స్థిర అంతఃవరుస దశాంశం op_కూడు(దశాంశం a, దశాంశం b) {
    తిరిగి a + b;
}

స్థిర అంతఃవరుస దశాంశం op_sub(దశాంశం a, దశాంశం b) {
    తిరిగి a - b;
}

స్థిర అంతఃవరుస దశాంశం op_mul(దశాంశం a, దశాంశం b) {
    తిరిగి a * b;
}

స్థిర అంతఃవరుస దశాంశం op_div(దశాంశం a, దశాంశం b) {
    తిరిగి a / b;
}

మూస <దశాంశం (*op)(దశాంశం, దశాంశం), రకం_పేరు src0_t, రకం_పేరు src1_t, రకం_పేరు dst_t>
స్థిర అంతఃవరుస శూన్య vec_ద్విమాన_op_contiguous(స్థిరరాశి int64_t n, dst_t * z, స్థిరరాశి src0_t * x, స్థిరరాశి src1_t * y) {
    constexpr స్వయంచాలక src0_గమ్యం_f32 = type_conversion_పట్టిక<src0_t>::గమ్యం_f32;
    constexpr స్వయంచాలక src1_గమ్యం_f32 = type_conversion_పట్టిక<src1_t>::గమ్యం_f32;
    constexpr స్వయంచాలక f32_గమ్యం_dst  = type_conversion_పట్టిక<dst_t >::from_f32;

    కొరకు (పూర్ణాంకం i = 0; i < n; i++) {
        z[i] = f32_గమ్యం_dst(op(src0_గమ్యం_f32(x[i]), src1_గమ్యం_f32(y[i])));
    }
}

మూస <దశాంశం (*op)(దశాంశం, దశాంశం), రకం_పేరు src0_t, రకం_పేరు src1_t, రకం_పేరు dst_t>
స్థిర అంతఃవరుస శూన్య vec_ద్విమాన_op_non_contiguous(స్థిరరాశి int64_t n, స్థిరరాశి int64_t ne10, స్థిరరాశి int64_t nb10, dst_t * z, స్థిరరాశి src0_t * x, స్థిరరాశి src1_t * y) {
    constexpr స్వయంచాలక src0_గమ్యం_f32 = type_conversion_పట్టిక<src0_t>::గమ్యం_f32;
    constexpr స్వయంచాలక src1_గమ్యం_f32 = type_conversion_పట్టిక<src1_t>::గమ్యం_f32;
    constexpr స్వయంచాలక f32_గమ్యం_dst  = type_conversion_పట్టిక<dst_t >::from_f32;

    కొరకు (పూర్ణాంకం i = 0; i < n; i++) {
        పూర్ణాంకం i10 = i % ne10;
        స్థిరరాశి src1_t * y_ptr = (స్థిరరాశి src1_t *)((స్థిరరాశి అక్షరం *)y + i10*nb10);
        z[i] = f32_గమ్యం_dst(op(src0_గమ్యం_f32(x[i]), src1_గమ్యం_f32(*y_ptr)));
    }
}

మూస <దశాంశం (*op)(దశాంశం, దశాంశం), రకం_పేరు src0_t, రకం_పేరు src1_t, రకం_పేరు dst_t>
స్థిర శూన్య apply_ద్విమాన_op(స్థిరరాశి ggml_compute_params * params, ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_ద్విమాన_OP_LOCALS

    GGML_ASSERT( nb0 == పరిమాణం(dst_t));
    GGML_ASSERT(nb00 == పరిమాణం(src0_t));

    స్థిరరాశి స్వయంచాలక [ir0, ir1] = get_thread_range(params, src0);
    స్థిరరాశి తార్కిక is_src1_contiguous_అడ్డువరుసలు = ggml_is_contiguous_అడ్డువరుసలు(src1);

#నిర్వచితమైతే GGML_USE_ACCELERATE
    vDSP_fn_t vDSP_op = శూన్యసూచిక;
    // TODO - avoid the f32-only check using type 'trait' lookup tables and row-based src-to-float conversion functions
    ఒకవేళ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        ఒకవేళ (op == op_కూడు) {
            vDSP_op = vDSP_vadd;
        } లేకపోతే ఒకవేళ (op == op_sub) {
            vDSP_op = vDSP_vsub;
        } లేకపోతే ఒకవేళ (op == op_mul) {
            vDSP_op = vDSP_vmul;
        } లేకపోతే ఒకవేళ (op == op_div) {
            vDSP_op = vDSP_vdiv;
        }
    }
#అయితే_ముగింపు

    కొరకు (int64_t ir = ir0; ir < ir1; ++ir) {
        స్థిరరాశి int64_t i03 = ir/(ne02*ne01);
        స్థిరరాశి int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        స్థిరరాశి int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        స్థిరరాశి int64_t i13 = i03 % ne13;
        స్థిరరాశి int64_t i12 = i02 % ne12;
        స్థిరరాశి int64_t i11 = i01 % ne11;

        dst_t        * dst_ptr  = (dst_t  *)       ((అక్షరం *)       dst->దత్తాంశం  + i03*nb3  + i02*nb2  + i01*nb1 );
        స్థిరరాశి src0_t * src0_ptr = (స్థిరరాశి src0_t *) ((స్థిరరాశి అక్షరం *) src0->దత్తాంశం + i03*nb03 + i02*nb02 + i01*nb01);
        స్థిరరాశి src1_t * src1_ptr = (స్థిరరాశి src1_t *) ((స్థిరరాశి అక్షరం *) src1->దత్తాంశం + i13*nb13 + i12*nb12 + i11*nb11);

        ఒకవేళ (is_src1_contiguous_అడ్డువరుసలు) {
            // src1 is broadcastable across src0 and dst in i1, i2, i3
            స్థిరరాశి int64_t nr0 = ne00 / ne10;

            కొరకు (int64_t r = 0; r < nr0; ++r) {
#నిర్వచితమైతే GGML_USE_ACCELERATE
                ఒకవేళ constexpr (std::is_same_v<src0_t, దశాంశం> && std::is_same_v<src1_t, దశాంశం> && std::is_same_v<dst_t, దశాంశం>) {
                    ఒకవేళ (vDSP_op != శూన్యసూచిక) {
                        vDSP_op(src1_ptr, 1, src0_ptr + r*ne10, 1, dst_ptr + r*ne10, 1, ne10);
                        కొనసాగు;
                    }
                }
#అయితే_ముగింపు
                vec_ద్విమాన_op_contiguous<op>(ne10, dst_ptr + r*ne10, src0_ptr + r*ne10, src1_ptr);
            }
        } లేకపోతే {
            vec_ద్విమాన_op_non_contiguous<op>(ne0, ne10, nb10, dst_ptr, src0_ptr, src1_ptr);
        }
    }
}

// TODO: Use the 'traits' lookup table (for type conversion fns), instead of a mass of 'if' conditions with long templates
మూస <దశాంశం (*op)(దశాంశం, దశాంశం)>
స్థిర శూన్య ద్విమాన_op(స్థిరరాశి ggml_compute_params * params, ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    /*  */ ఒకవేళ (src0->type == GGML_TYPE_F32  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_ద్విమాన_op<op, దశాంశం, దశాంశం, దశాంశం>(params, dst);
    } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_ద్విమాన_op<op, ggml_fp16_t, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_ద్విమాన_op<op, ggml_bf16_t, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_BF16) {
        apply_ద్విమాన_op<op, ggml_bf16_t, దశాంశం, ggml_bf16_t>(params, dst);
    } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) {
        apply_ద్విమాన_op<op, ggml_bf16_t, దశాంశం, దశాంశం>(params, dst);
    } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F16) {
        apply_ద్విమాన_op<op, ggml_fp16_t, దశాంశం, ggml_fp16_t>(params, dst);
    } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) {
        apply_ద్విమాన_op<op, ggml_fp16_t, దశాంశం, దశాంశం>(params, dst);
    } లేకపోతే {
        GGML_ABORT("%s: unsupported types: dst: %s, src0: %s, src1: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type), ggml_type_name(src1->type));
    }
}

శూన్య ggml_compute_forward_కూడు_non_quantized(స్థిరరాశి ggml_compute_params * params, ggml_tensor * dst) {
    ద్విమాన_op<op_కూడు>(params, dst);
}

శూన్య ggml_compute_forward_sub(స్థిరరాశి ggml_compute_params * params, ggml_tensor * dst) {
    ద్విమాన_op<op_sub>(params, dst);
}

శూన్య ggml_compute_forward_mul(స్థిరరాశి ggml_compute_params * params, ggml_tensor * dst) {
    ద్విమాన_op<op_mul>(params, dst);
}

శూన్య ggml_compute_forward_div(స్థిరరాశి ggml_compute_params * params, ggml_tensor * dst) {
    ద్విమాన_op<op_div>(params, dst);
}
