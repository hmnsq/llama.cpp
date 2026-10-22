#অন্তর্ভাব "ggml-et-uberkernel-common.h"
#অন্তর্ভাব "ggml-et-uberkernel-kernel-map.h"
#অন্তর্ভাব "ggml_tensor.h"
#অন্তর্ভাব "math_fp.h"
#অন্তর্ভাব "platform.h"

#অন্তর্ভাব <stdint.h>

কাঠামো ggml_et_glu_params;
কাঠামো ggml_et_unary_params;
কাঠামো ggml_et_rope_params;
কাঠামো ggml_et_rms_norm_params;
কাঠামো ggml_et_rms_norm_mul_params;
কাঠামো ggml_et_softmax_params;
কাঠামো ggml_et_set_rows_params;
কাঠামো ggml_et_get_rows_params;
কাঠামো ggml_et_cont_params;
কাঠামো ggml_et_concat_params;
কাঠামো ggml_et_cumsum_params;
কাঠামো ggml_et_diag_params;
কাঠামো ggml_et_fill_params;
কাঠামো ggml_et_flash_attn_ext_params;
কাঠামো ggml_et_gated_delta_net_params;
কাঠামো ggml_et_group_norm_params;
কাঠামো ggml_et_im2col_params;
কাঠামো ggml_et_l2_norm_params;
কাঠামো ggml_et_mul_mat_id_params;
কাঠামো ggml_et_norm_params;
কাঠামো ggml_et_pad_params;
কাঠামো ggml_et_repeat_params;
কাঠামো ggml_et_rwkv_wkv6_params;
কাঠামো ggml_et_rwkv_wkv7_params;
কাঠামো ggml_et_scale_params;
কাঠামো ggml_et_set_params;
কাঠামো ggml_et_solve_tri_params;
কাঠামো ggml_et_sqr_params;
কাঠামো ggml_et_ssm_conv_params;
কাঠামো ggml_et_ssm_scan_params;
কাঠামো ggml_et_যোগ_rows_params;
কাঠামো ggml_et_tri_params;

বাহ্যিক পূর্ণসংখ্যা el_map_f32_entry(কাঠামো ggml_et_দ্বিমিক_params *, শূন্য *);
বাহ্যিক পূর্ণসংখ্যা glu_f32_entry(কাঠামো ggml_et_glu_params *, শূন্য *);
বাহ্যিক পূর্ণসংখ্যা unary_f32_entry(কাঠামো ggml_et_unary_params *, শূন্য *);
বাহ্যিক পূর্ণসংখ্যা rope_f32_entry(কাঠামো ggml_et_rope_params *, শূন্য *);
বাহ্যিক পূর্ণসংখ্যা rms_norm_f32_entry(কাঠামো ggml_et_rms_norm_params *, শূন্য *);
বাহ্যিক পূর্ণসংখ্যা rms_norm_mul_f32_entry(কাঠামো ggml_et_rms_norm_mul_params *, শূন্য *);
বাহ্যিক পূর্ণসংখ্যা softmax_f32_entry(কাঠামো ggml_et_softmax_params *, শূন্য *);
বাহ্যিক পূর্ণসংখ্যা set_rows_f32_entry(কাঠামো ggml_et_set_rows_params *, শূন্য *);
বাহ্যিক পূর্ণসংখ্যা get_rows_f32_entry(কাঠামো ggml_et_get_rows_params *, শূন্য *);
বাহ্যিক পূর্ণসংখ্যা cont_f32_entry(কাঠামো ggml_et_cont_params *, শূন্য *);
বাহ্যিক পূর্ণসংখ্যা cont_f16_entry(কাঠামো ggml_et_cont_params *, শূন্য *);
বাহ্যিক পূর্ণসংখ্যা cpy_f32_f16_entry(কাঠামো ggml_et_cont_params *, শূন্য *);
বাহ্যিক পূর্ণসংখ্যা concat_f32_entry(কাঠামো ggml_et_concat_params *, শূন্য *);
বাহ্যিক পূর্ণসংখ্যা cumsum_f32_entry(কাঠামো ggml_et_cumsum_params *, শূন্য *);
বাহ্যিক পূর্ণসংখ্যা diag_f32_entry(কাঠামো ggml_et_diag_params *, শূন্য *);
বাহ্যিক পূর্ণসংখ্যা fill_f32_entry(কাঠামো ggml_et_fill_params *, শূন্য *);
বাহ্যিক পূর্ণসংখ্যা flash_attn_ext_f32_entry(কাঠামো ggml_et_flash_attn_ext_params *, শূন্য *);
বাহ্যিক পূর্ণসংখ্যা flash_attn_ext_f16_me_entry(কাঠামো ggml_et_flash_attn_ext_params *, শূন্য *);
বাহ্যিক পূর্ণসংখ্যা gated_delta_net_f32_entry(কাঠামো ggml_et_gated_delta_net_params *, শূন্য *);
বাহ্যিক পূর্ণসংখ্যা group_norm_f32_entry(কাঠামো ggml_et_group_norm_params *, শূন্য *);
বাহ্যিক পূর্ণসংখ্যা im2col_entry(কাঠামো ggml_et_im2col_params *, শূন্য *);
বাহ্যিক পূর্ণসংখ্যা l2_norm_f32_entry(কাঠামো ggml_et_l2_norm_params *, শূন্য *);
বাহ্যিক পূর্ণসংখ্যা mul_mat_id_f32_entry(কাঠামো ggml_et_mul_mat_id_params *, শূন্য *);
বাহ্যিক পূর্ণসংখ্যা norm_f32_entry(কাঠামো ggml_et_norm_params *, শূন্য *);
বাহ্যিক পূর্ণসংখ্যা pad_f32_entry(কাঠামো ggml_et_pad_params *, শূন্য *);
বাহ্যিক পূর্ণসংখ্যা repeat_f32_entry(কাঠামো ggml_et_repeat_params *, শূন্য *);
বাহ্যিক পূর্ণসংখ্যা rwkv_wkv6_f32_entry(কাঠামো ggml_et_rwkv_wkv6_params *, শূন্য *);
বাহ্যিক পূর্ণসংখ্যা rwkv_wkv7_f32_entry(কাঠামো ggml_et_rwkv_wkv7_params *, শূন্য *);
বাহ্যিক পূর্ণসংখ্যা scale_f32_entry(কাঠামো ggml_et_scale_params *, শূন্য *);
বাহ্যিক পূর্ণসংখ্যা set_f32_entry(কাঠামো ggml_et_set_params *, শূন্য *);
বাহ্যিক পূর্ণসংখ্যা solve_tri_f32_entry(কাঠামো ggml_et_solve_tri_params *, শূন্য *);
বাহ্যিক পূর্ণসংখ্যা sqr_f32_entry(কাঠামো ggml_et_sqr_params *, শূন্য *);
বাহ্যিক পূর্ণসংখ্যা ssm_conv_f32_entry(কাঠামো ggml_et_ssm_conv_params *, শূন্য *);
বাহ্যিক পূর্ণসংখ্যা ssm_scan_f32_entry(কাঠামো ggml_et_ssm_scan_params *, শূন্য *);
বাহ্যিক পূর্ণসংখ্যা যোগ_rows_f32_entry(কাঠামো ggml_et_যোগ_rows_params *, শূন্য *);
বাহ্যিক পূর্ণসংখ্যা tri_f32_entry(কাঠামো ggml_et_tri_params *, শূন্য *);
বাহ্যিক পূর্ণসংখ্যা mul_mat_f16_entry(কাঠামো ggml_et_দ্বিমিক_params *, শূন্য *);
বাহ্যিক পূর্ণসংখ্যা mul_mat_f16_matrix_engine_entry(কাঠামো ggml_et_দ্বিমিক_params *, শূন্য *);
বাহ্যিক পূর্ণসংখ্যা mul_mat_f32_entry(কাঠামো ggml_et_দ্বিমিক_params *, শূন্য *);
বাহ্যিক পূর্ণসংখ্যা mul_mat_f32_matrix_engine_entry(কাঠামো ggml_et_দ্বিমিক_params *, শূন্য *);
বাহ্যিক পূর্ণসংখ্যা mul_mat_Q8_0_entry(কাঠামো ggml_et_mm_q8_params *, শূন্য *);
বাহ্যিক পূর্ণসংখ্যা mul_mat_Q4_0_entry(কাঠামো ggml_et_দ্বিমিক_params *, শূন্য *);

স্থির অন্তঃরেখ size_t tensor_bytes(ধ্রুবক কাঠামো ggml_tensor * t) {
    ফেরত (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

কাঠামো uber_glu_params {
    কাঠামো ggml_tensor src0;
    কাঠামো ggml_tensor src1;
    কাঠামো ggml_tensor dst;
    // trailing scalars omitted — not needed for eviction
};

কাঠামো uber_unary_params {
    কাঠামো ggml_tensor src0;
    কাঠামো ggml_tensor dst;
};

কাঠামো uber_rope_params {
    কাঠামো ggml_tensor src0;
    কাঠামো ggml_tensor src1;
    কাঠামো ggml_tensor src2;
    কাঠামো ggml_tensor dst;
};

কাঠামো uber_rms_norm_params {
    কাঠামো ggml_tensor src0;
    কাঠামো ggml_tensor dst;
};

কাঠামো uber_rms_norm_mul_params {
    কাঠামো ggml_tensor src0;
    কাঠামো ggml_tensor src1;
    কাঠামো ggml_tensor dst;
};

কাঠামো uber_softmax_params {
    কাঠামো ggml_tensor src0;
    কাঠামো ggml_tensor src1;
    কাঠামো ggml_tensor src2;
    কাঠামো ggml_tensor dst;
};

কাঠামো uber_set_rows_params {
    কাঠামো ggml_tensor src0;
    কাঠামো ggml_tensor src1;
    কাঠামো ggml_tensor dst;
};

কাঠামো uber_get_rows_params {
    কাঠামো ggml_tensor src0;
    কাঠামো ggml_tensor src1;
    কাঠামো ggml_tensor dst;
};

কাঠামো uber_cont_params {
    কাঠামো ggml_tensor src0;
    কাঠামো ggml_tensor dst;
};

// src0 + src1 + dst (no trailing scalars needed for eviction)
কাঠামো uber_concat_params {
    কাঠামো ggml_tensor src0;
    কাঠামো ggml_tensor src1;
    কাঠামো ggml_tensor dst;
};

কাঠামো uber_ssm_conv_params {
    কাঠামো ggml_tensor src0;
    কাঠামো ggml_tensor src1;
    কাঠামো ggml_tensor dst;
};

কাঠামো uber_solve_tri_params {
    কাঠামো ggml_tensor src0;
    কাঠামো ggml_tensor src1;
    কাঠামো ggml_tensor dst;
};

কাঠামো uber_mul_mat_id_params {
    কাঠামো ggml_tensor src0;
    কাঠামো ggml_tensor src1;
    কাঠামো ggml_tensor src2;
    কাঠামো ggml_tensor dst;
};

// flash_attn_ext: Q=src0, K=src1, V=src2, mask=src3, dst (mask optional)
কাঠামো uber_flash_attn_ext_params {
    কাঠামো ggml_tensor src0;
    কাঠামো ggml_tensor src1;
    কাঠামো ggml_tensor src2;
    কাঠামো ggml_tensor mask;
    কাঠামো ggml_tensor dst;
};

// ssm_scan: 7 source tensors + dst
কাঠামো uber_ssm_scan_params {
    কাঠামো ggml_tensor src0;
    কাঠামো ggml_tensor src1;
    কাঠামো ggml_tensor src2;
    কাঠামো ggml_tensor src3;
    কাঠামো ggml_tensor src4;
    কাঠামো ggml_tensor src5;
    কাঠামো ggml_tensor src6;
    কাঠামো ggml_tensor dst;
};

// gated_delta_net: q,k,v,g,beta,state_in,dst
কাঠামো uber_gated_delta_net_params {
    কাঠামো ggml_tensor q;
    কাঠামো ggml_tensor k;
    কাঠামো ggml_tensor v;
    কাঠামো ggml_tensor g;
    কাঠামো ggml_tensor beta;
    কাঠামো ggml_tensor state_in;
    কাঠামো ggml_tensor dst;
};

স্থির শূন্য copy_f32_to_f16_row(uint16_t * dst, ধ্রুবক দশমিক * src, int64_t অঙ্ক_elements) {
    জন্য (int64_t i = 0; i < অঙ্ক_elements; i++) {
        dst[i] = fp32_to_fp16(src[i]);
    }
}

স্থির শূন্য copy_f32_row(দশমিক * dst, ধ্রুবক দশমিক * src, int64_t অঙ্ক_elements) {
    জন্য (int64_t i = 0; i < অঙ্ক_elements; i++) {
        dst[i] = src[i];
    }
}

স্থির শূন্য evict_region_past_l2_local(ধ্রুবক শূন্য * addr, size_t bytes) {
    যদি (!addr || bytes == 0) {
        ফেরত;
    }

    ধ্রুবক uint64_t CL     = 64;
    uint64_t       base   = (uint64_t) addr & ~(CL - 1);
    uint64_t       end    = ((uint64_t) addr + bytes + CL - 1) & ~(CL - 1);
    uint64_t       nlines = (end - base) / CL;
    cache_ops_priv_evict_sw(0, /*to_L2*/ 3, 0, 0, CL);
}

পূর্ণসংখ্যা entry_point(কাঠামো ggml_et_uberkernel_params * params, শূন্য * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    যদি (!kernel_env || !params) {
        ফেরত -1;
    }

    কাঠামো ggml_et_uberkernel_inst * insts       = (কাঠামো ggml_et_uberkernel_inst *) (uintptr_t) params->insts;
    uint8_t *                        params_blob = (uint8_t *) (uintptr_t) params->params_blob;

    যদি (!insts || !params_blob || params->inst_stride < আকার(কাঠামো ggml_et_uberkernel_inst)) {
        ফেরত -1;
    }

    জন্য (uint32_t i = 0; i < params->অঙ্ক_insts; ++i) {
        কাঠামো ggml_et_uberkernel_inst * inst =
            (কাঠামো ggml_et_uberkernel_inst *) ((uint8_t *) insts + (i * params->inst_stride));
        শূন্য * inst_params = params_blob + inst->params_offset;
        পূর্ণসংখ্যা    rc          = -1;

        et_barrier_global(32ULL);

        নির্বাচন (inst->kernel_id) {
            ক্ষেত্র GGML_ET_UBERKERNEL_KERNEL_EL_MAP_F32:
                {
                    কাঠামো ggml_et_দ্বিমিক_params * p = (কাঠামো ggml_et_দ্বিমিক_params *) inst_params;
                    rc                               = el_map_f32_entry(p, env);
                    বিরতি;
                }
            // case GGML_ET_UBERKERNEL_KERNEL_UNARY_F32: {
            //     // struct uber_unary_params *p = (struct uber_unary_params *) inst_params;
            //     // et_barrier(ET_BARRIER_GLOBAL);
            //     rc = unary_f32_entry((struct ggml_et_unary_params *) inst_params, env);
            //     break;
            // }
            // case GGML_ET_UBERKERNEL_KERNEL_CPY_F32_F16: {
            //     struct uber_unary_params *p = (struct uber_unary_params *) inst_params;
            //     // evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
            //     rc = cpy_f32_f16_entry((struct ggml_et_cont_params *) inst_params, env);
            //     break;
            // }
            // case GGML_ET_UBERKERNEL_KERNEL_GET_ROWS_F32: {
            //     struct uber_get_rows_params *p = (struct uber_get_rows_params *) inst_params;
            //     rc = get_rows_f32_entry((struct ggml_et_get_rows_params *) inst_params, env);
            //     break;
            // }
            // case GGML_ET_UBERKERNEL_KERNEL_CONT_F32: {
            //     struct uber_cont_params *p = (struct uber_cont_params *) inst_params;
            //     // evict_region_past_l2_local(p->src0.data, tensor_bytes(&p->src0));
            //     // evict_region_past_l2(p->dst.data, tensor_bytes(&p->dst));
            //     rc = cont_f32_entry((struct ggml_et_cont_params *) inst_params, env);
            //     break;
            // }
            ক্ষেত্র GGML_ET_UBERKERNEL_KERNEL_GLU_F32:
                {
                    rc = glu_f32_entry((কাঠামো ggml_et_glu_params *) inst_params, env);
                    বিরতি;
                }
            ক্ষেত্র GGML_ET_UBERKERNEL_KERNEL_ROPE_F32:
                {
                    rc = rope_f32_entry((কাঠামো ggml_et_rope_params *) inst_params, env);
                    বিরতি;
                }
            ক্ষেত্র GGML_ET_UBERKERNEL_KERNEL_RMS_NORM_F32:
                {
                    // struct ggml_et_rms_norm_params *p = (struct ggml_et_rms_norm_params *) inst_params;
                    // evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
                    rc = rms_norm_f32_entry((কাঠামো ggml_et_rms_norm_params *) inst_params, env);
                    বিরতি;
                }
            ক্ষেত্র GGML_ET_UBERKERNEL_KERNEL_RMS_NORM_MUL_F32:
                {
                    কাঠামো uber_rms_norm_mul_params * p = (কাঠামো uber_rms_norm_mul_params *) inst_params;
                    evict_region_past_l2(p->src0.তথ্য, tensor_bytes(&p->src0));
                    evict_region_past_l2(p->src1.তথ্য, tensor_bytes(&p->src1));
                    rc = rms_norm_mul_f32_entry((কাঠামো ggml_et_rms_norm_mul_params *) inst_params, env);
                    বিরতি;
                }
            ক্ষেত্র GGML_ET_UBERKERNEL_KERNEL_SOFTMAX_F32:
                {
                    rc = softmax_f32_entry((কাঠামো ggml_et_softmax_params *) inst_params, env);
                    বিরতি;
                }
            ক্ষেত্র GGML_ET_UBERKERNEL_KERNEL_SET_ROWS_F32:
                {
                    rc = set_rows_f32_entry((কাঠামো ggml_et_set_rows_params *) inst_params, env);
                    বিরতি;
                }

            // Single-source ops (src0 → dst)
            ক্ষেত্র GGML_ET_UBERKERNEL_KERNEL_SQR_F32:
                {
                    rc = sqr_f32_entry((কাঠামো ggml_et_sqr_params *) inst_params, env);
                    বিরতি;
                }
            ক্ষেত্র GGML_ET_UBERKERNEL_KERNEL_SCALE_F32:
                {
                    rc = scale_f32_entry((কাঠামো ggml_et_scale_params *) inst_params, env);
                    বিরতি;
                }
            ক্ষেত্র GGML_ET_UBERKERNEL_KERNEL_যোগ_ROWS_F32:
                {
                    rc = যোগ_rows_f32_entry((কাঠামো ggml_et_যোগ_rows_params *) inst_params, env);
                    বিরতি;
                }
            ক্ষেত্র GGML_ET_UBERKERNEL_KERNEL_CUMSUM_F32:
                {
                    rc = cumsum_f32_entry((কাঠামো ggml_et_cumsum_params *) inst_params, env);
                    বিরতি;
                }
            ক্ষেত্র GGML_ET_UBERKERNEL_KERNEL_NORM_F32:
                {
                    rc = norm_f32_entry((কাঠামো ggml_et_norm_params *) inst_params, env);
                    বিরতি;
                }
            ক্ষেত্র GGML_ET_UBERKERNEL_KERNEL_L2_NORM_F32:
                {
                    rc = l2_norm_f32_entry((কাঠামো ggml_et_l2_norm_params *) inst_params, env);
                    বিরতি;
                }
            ক্ষেত্র GGML_ET_UBERKERNEL_KERNEL_GROUP_NORM_F32:
                {
                    rc = group_norm_f32_entry((কাঠামো ggml_et_group_norm_params *) inst_params, env);
                    বিরতি;
                }
            ক্ষেত্র GGML_ET_UBERKERNEL_KERNEL_REPEAT_F32:
                {
                    rc = repeat_f32_entry((কাঠামো ggml_et_repeat_params *) inst_params, env);
                    বিরতি;
                }
            ক্ষেত্র GGML_ET_UBERKERNEL_KERNEL_DIAG_F32:
                {
                    rc = diag_f32_entry((কাঠামো ggml_et_diag_params *) inst_params, env);
                    বিরতি;
                }
            ক্ষেত্র GGML_ET_UBERKERNEL_KERNEL_TRI_F32:
                {
                    rc = tri_f32_entry((কাঠামো ggml_et_tri_params *) inst_params, env);
                    বিরতি;
                }
            ক্ষেত্র GGML_ET_UBERKERNEL_KERNEL_PAD_F32:
                {
                    rc = pad_f32_entry((কাঠামো ggml_et_pad_params *) inst_params, env);
                    বিরতি;
                }
            ক্ষেত্র GGML_ET_UBERKERNEL_KERNEL_CONT_F16:
                {
                    rc = cont_f16_entry((কাঠামো ggml_et_cont_params *) inst_params, env);
                    বিরতি;
                }
            ক্ষেত্র GGML_ET_UBERKERNEL_KERNEL_FILL_F32:
                {
                    rc = fill_f32_entry((কাঠামো ggml_et_fill_params *) inst_params, env);
                    বিরতি;
                }
            ক্ষেত্র GGML_ET_UBERKERNEL_KERNEL_SET_F32:
                {
                    rc = set_f32_entry((কাঠামো ggml_et_set_params *) inst_params, env);
                    বিরতি;
                }

            // Two-source ops
            ক্ষেত্র GGML_ET_UBERKERNEL_KERNEL_CONCAT_F32:
                {
                    rc = concat_f32_entry((কাঠামো ggml_et_concat_params *) inst_params, env);
                    বিরতি;
                }
            // case GGML_ET_UBERKERNEL_KERNEL_SSM_CONV_F32: {
            //     rc = ssm_conv_f32_entry((struct ggml_et_ssm_conv_params *) inst_params, env);
            //     break;
            // }
            ক্ষেত্র GGML_ET_UBERKERNEL_KERNEL_SOLVE_TRI_F32:
                {
                    rc = solve_tri_f32_entry((কাঠামো ggml_et_solve_tri_params *) inst_params, env);
                    বিরতি;
                }
            ক্ষেত্র GGML_ET_UBERKERNEL_KERNEL_IM2COL:
                {
                    rc = im2col_entry((কাঠামো ggml_et_im2col_params *) inst_params, env);
                    বিরতি;
                }

            // Three-source ops
            ক্ষেত্র GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_ID_F32:
                {
                    rc = mul_mat_id_f32_entry((কাঠামো ggml_et_mul_mat_id_params *) inst_params, env);
                    বিরতি;
                }
            ক্ষেত্র GGML_ET_UBERKERNEL_KERNEL_FLASH_ATTN_EXT_F32:
                {
                    rc = flash_attn_ext_f32_entry((কাঠামো ggml_et_flash_attn_ext_params *) inst_params, env);
                    বিরতি;
                }
            ক্ষেত্র GGML_ET_UBERKERNEL_KERNEL_FLASH_ATTN_EXT_F16_ME:
                {
                    rc = flash_attn_ext_f16_me_entry((কাঠামো ggml_et_flash_attn_ext_params *) inst_params, env);
                    বিরতি;
                }

            ক্ষেত্র GGML_ET_UBERKERNEL_KERNEL_GATED_DELTA_NET_F32:
                {
                    rc = gated_delta_net_f32_entry((কাঠামো ggml_et_gated_delta_net_params *) inst_params, env);
                    বিরতি;
                }
            ক্ষেত্র GGML_ET_UBERKERNEL_KERNEL_SSM_SCAN_F32:
                {
                    rc = ssm_scan_f32_entry((কাঠামো ggml_et_ssm_scan_params *) inst_params, env);
                    বিরতি;
                }
            // rwkv: raw float* params, no ggml_tensor fields to evict via
            ক্ষেত্র GGML_ET_UBERKERNEL_KERNEL_RWKV_WKV6_F32:
                {
                    rc = rwkv_wkv6_f32_entry((কাঠামো ggml_et_rwkv_wkv6_params *) inst_params, env);
                    বিরতি;
                }

            ক্ষেত্র GGML_ET_UBERKERNEL_KERNEL_RWKV_WKV7_F32:
                {
                    rc = rwkv_wkv7_f32_entry((কাঠামো ggml_et_rwkv_wkv7_params *) inst_params, env);
                    বিরতি;
                }

            // MUL_MAT: evict src1 (activations); src0=weights is
            //  read-only so never stale from a prior uberkernel op
            ক্ষেত্র GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F16:
                {
                    কাঠামো ggml_et_দ্বিমিক_params * p = (কাঠামো ggml_et_দ্বিমিক_params *) inst_params;
                    rc                               = mul_mat_f16_entry(p, env);
                    বিরতি;
                }
            ক্ষেত্র GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F16_MATRIX_ENGINE:
                {
                    কাঠামো ggml_et_দ্বিমিক_params * p = (কাঠামো ggml_et_দ্বিমিক_params *) inst_params;
                    rc                               = mul_mat_f16_matrix_engine_entry(p, env);
                    বিরতি;
                }
            ক্ষেত্র GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F32:
                {
                    কাঠামো ggml_et_দ্বিমিক_params * p = (কাঠামো ggml_et_দ্বিমিক_params *) inst_params;
                    rc                               = mul_mat_f32_entry(p, env);
                    বিরতি;
                }
            ক্ষেত্র GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F32_MATRIX_ENGINE:
                {
                    কাঠামো ggml_et_দ্বিমিক_params * p = (কাঠামো ggml_et_দ্বিমিক_params *) inst_params;
                    rc                               = mul_mat_f32_matrix_engine_entry(p, env);
                    বিরতি;
                }
            ক্ষেত্র GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_Q8_0:
                {
                    কাঠামো ggml_et_mm_q8_params * p = (কাঠামো ggml_et_mm_q8_params *) inst_params;
                    // evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
                    rc                              = mul_mat_Q8_0_entry(p, env);
                    বিরতি;
                }
            ক্ষেত্র GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_Q4_0:
                {
                    কাঠামো ggml_et_দ্বিমিক_params * p = (কাঠামো ggml_et_দ্বিমিক_params *) inst_params;
                    rc                               = mul_mat_Q4_0_entry(p, env);
                    বিরতি;
                }

            পূর্বনির্ধারিত:
                ফেরত -1;
        }

        যদি (rc != 0) {
            ফেরত rc;
        }
    }

    ফেরত 0;
}
