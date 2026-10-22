#ଅନ୍ତର୍ଭାଵ "ggml-et-uberkernel-common.h"
#ଅନ୍ତର୍ଭାଵ "ggml-et-uberkernel-kernel-map.h"
#ଅନ୍ତର୍ଭାଵ "ggml_tensor.h"
#ଅନ୍ତର୍ଭାଵ "math_fp.h"
#ଅନ୍ତର୍ଭାଵ "platform.h"

#ଅନ୍ତର୍ଭାଵ <stdint.h>

ସଂରଚନା ggml_et_glu_params;
ସଂରଚନା ggml_et_unary_params;
ସଂରଚନା ggml_et_rope_params;
ସଂରଚନା ggml_et_rms_norm_params;
ସଂରଚନା ggml_et_rms_norm_mul_params;
ସଂରଚନା ggml_et_softmax_params;
ସଂରଚନା ggml_et_set_rows_params;
ସଂରଚନା ggml_et_get_rows_params;
ସଂରଚନା ggml_et_cont_params;
ସଂରଚନା ggml_et_concat_params;
ସଂରଚନା ggml_et_cumsum_params;
ସଂରଚନା ggml_et_diag_params;
ସଂରଚନା ggml_et_fill_params;
ସଂରଚନା ggml_et_flash_attn_ext_params;
ସଂରଚନା ggml_et_gated_delta_net_params;
ସଂରଚନା ggml_et_group_norm_params;
ସଂରଚନା ggml_et_im2col_params;
ସଂରଚନା ggml_et_l2_norm_params;
ସଂରଚନା ggml_et_mul_mat_id_params;
ସଂରଚନା ggml_et_norm_params;
ସଂରଚନା ggml_et_pad_params;
ସଂରଚନା ggml_et_repeat_params;
ସଂରଚନା ggml_et_rwkv_wkv6_params;
ସଂରଚନା ggml_et_rwkv_wkv7_params;
ସଂରଚନା ggml_et_scale_params;
ସଂରଚନା ggml_et_set_params;
ସଂରଚନା ggml_et_solve_tri_params;
ସଂରଚନା ggml_et_sqr_params;
ସଂରଚନା ggml_et_ssm_conv_params;
ସଂରଚନା ggml_et_ssm_scan_params;
ସଂରଚନା ggml_et_sum_rows_params;
ସଂରଚନା ggml_et_tri_params;

ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା el_map_f32_entry(ସଂରଚନା ggml_et_ଦ୍ୱିମିକ_params *, ଶୂନ୍ୟ *);
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା glu_f32_entry(ସଂରଚନା ggml_et_glu_params *, ଶୂନ୍ୟ *);
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା unary_f32_entry(ସଂରଚନା ggml_et_unary_params *, ଶୂନ୍ୟ *);
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା rope_f32_entry(ସଂରଚନା ggml_et_rope_params *, ଶୂନ୍ୟ *);
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା rms_norm_f32_entry(ସଂରଚନା ggml_et_rms_norm_params *, ଶୂନ୍ୟ *);
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା rms_norm_mul_f32_entry(ସଂରଚନା ggml_et_rms_norm_mul_params *, ଶୂନ୍ୟ *);
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା softmax_f32_entry(ସଂରଚନା ggml_et_softmax_params *, ଶୂନ୍ୟ *);
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା set_rows_f32_entry(ସଂରଚନା ggml_et_set_rows_params *, ଶୂନ୍ୟ *);
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା get_rows_f32_entry(ସଂରଚନା ggml_et_get_rows_params *, ଶୂନ୍ୟ *);
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା cont_f32_entry(ସଂରଚନା ggml_et_cont_params *, ଶୂନ୍ୟ *);
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା cont_f16_entry(ସଂରଚନା ggml_et_cont_params *, ଶୂନ୍ୟ *);
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା cpy_f32_f16_entry(ସଂରଚନା ggml_et_cont_params *, ଶୂନ୍ୟ *);
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା concat_f32_entry(ସଂରଚନା ggml_et_concat_params *, ଶୂନ୍ୟ *);
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା cumsum_f32_entry(ସଂରଚନା ggml_et_cumsum_params *, ଶୂନ୍ୟ *);
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା diag_f32_entry(ସଂରଚନା ggml_et_diag_params *, ଶୂନ୍ୟ *);
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା fill_f32_entry(ସଂରଚନା ggml_et_fill_params *, ଶୂନ୍ୟ *);
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା flash_attn_ext_f32_entry(ସଂରଚନା ggml_et_flash_attn_ext_params *, ଶୂନ୍ୟ *);
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା flash_attn_ext_f16_me_entry(ସଂରଚନା ggml_et_flash_attn_ext_params *, ଶୂନ୍ୟ *);
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା gated_delta_net_f32_entry(ସଂରଚନା ggml_et_gated_delta_net_params *, ଶୂନ୍ୟ *);
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା group_norm_f32_entry(ସଂରଚନା ggml_et_group_norm_params *, ଶୂନ୍ୟ *);
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା im2col_entry(ସଂରଚନା ggml_et_im2col_params *, ଶୂନ୍ୟ *);
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା l2_norm_f32_entry(ସଂରଚନା ggml_et_l2_norm_params *, ଶୂନ୍ୟ *);
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା mul_mat_id_f32_entry(ସଂରଚନା ggml_et_mul_mat_id_params *, ଶୂନ୍ୟ *);
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା norm_f32_entry(ସଂରଚନା ggml_et_norm_params *, ଶୂନ୍ୟ *);
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା pad_f32_entry(ସଂରଚନା ggml_et_pad_params *, ଶୂନ୍ୟ *);
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା repeat_f32_entry(ସଂରଚନା ggml_et_repeat_params *, ଶୂନ୍ୟ *);
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା rwkv_wkv6_f32_entry(ସଂରଚନା ggml_et_rwkv_wkv6_params *, ଶୂନ୍ୟ *);
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା rwkv_wkv7_f32_entry(ସଂରଚନା ggml_et_rwkv_wkv7_params *, ଶୂନ୍ୟ *);
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା scale_f32_entry(ସଂରଚନା ggml_et_scale_params *, ଶୂନ୍ୟ *);
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା set_f32_entry(ସଂରଚନା ggml_et_set_params *, ଶୂନ୍ୟ *);
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା solve_tri_f32_entry(ସଂରଚନା ggml_et_solve_tri_params *, ଶୂନ୍ୟ *);
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା sqr_f32_entry(ସଂରଚନା ggml_et_sqr_params *, ଶୂନ୍ୟ *);
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ssm_conv_f32_entry(ସଂରଚନା ggml_et_ssm_conv_params *, ଶୂନ୍ୟ *);
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ssm_scan_f32_entry(ସଂରଚନା ggml_et_ssm_scan_params *, ଶୂନ୍ୟ *);
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା sum_rows_f32_entry(ସଂରଚନା ggml_et_sum_rows_params *, ଶୂନ୍ୟ *);
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା tri_f32_entry(ସଂରଚନା ggml_et_tri_params *, ଶୂନ୍ୟ *);
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା mul_mat_f16_entry(ସଂରଚନା ggml_et_ଦ୍ୱିମିକ_params *, ଶୂନ୍ୟ *);
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା mul_mat_f16_matrix_engine_entry(ସଂରଚନା ggml_et_ଦ୍ୱିମିକ_params *, ଶୂନ୍ୟ *);
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା mul_mat_f32_entry(ସଂରଚନା ggml_et_ଦ୍ୱିମିକ_params *, ଶୂନ୍ୟ *);
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା mul_mat_f32_matrix_engine_entry(ସଂରଚନା ggml_et_ଦ୍ୱିମିକ_params *, ଶୂନ୍ୟ *);
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା mul_mat_Q8_0_entry(ସଂରଚନା ggml_et_mm_q8_params *, ଶୂନ୍ୟ *);
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା mul_mat_Q4_0_entry(ସଂରଚନା ggml_et_ଦ୍ୱିମିକ_params *, ଶୂନ୍ୟ *);

ସ୍ଥିର ଧାଡ଼ିଭିତରେ size_t tensor_bytes(ଅଚଳ ସଂରଚନା ggml_tensor * t) {
    ଫେରନ୍ତୁ (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

ସଂରଚନା uber_glu_params {
    ସଂରଚନା ggml_tensor src0;
    ସଂରଚନା ggml_tensor src1;
    ସଂରଚନା ggml_tensor dst;
    // trailing scalars omitted — not needed for eviction
};

ସଂରଚନା uber_unary_params {
    ସଂରଚନା ggml_tensor src0;
    ସଂରଚନା ggml_tensor dst;
};

ସଂରଚନା uber_rope_params {
    ସଂରଚନା ggml_tensor src0;
    ସଂରଚନା ggml_tensor src1;
    ସଂରଚନା ggml_tensor src2;
    ସଂରଚନା ggml_tensor dst;
};

ସଂରଚନା uber_rms_norm_params {
    ସଂରଚନା ggml_tensor src0;
    ସଂରଚନା ggml_tensor dst;
};

ସଂରଚନା uber_rms_norm_mul_params {
    ସଂରଚନା ggml_tensor src0;
    ସଂରଚନା ggml_tensor src1;
    ସଂରଚନା ggml_tensor dst;
};

ସଂରଚନା uber_softmax_params {
    ସଂରଚନା ggml_tensor src0;
    ସଂରଚନା ggml_tensor src1;
    ସଂରଚନା ggml_tensor src2;
    ସଂରଚନା ggml_tensor dst;
};

ସଂରଚନା uber_set_rows_params {
    ସଂରଚନା ggml_tensor src0;
    ସଂରଚନା ggml_tensor src1;
    ସଂରଚନା ggml_tensor dst;
};

ସଂରଚନା uber_get_rows_params {
    ସଂରଚନା ggml_tensor src0;
    ସଂରଚନା ggml_tensor src1;
    ସଂରଚନା ggml_tensor dst;
};

ସଂରଚନା uber_cont_params {
    ସଂରଚନା ggml_tensor src0;
    ସଂରଚନା ggml_tensor dst;
};

// src0 + src1 + dst (no trailing scalars needed for eviction)
ସଂରଚନା uber_concat_params {
    ସଂରଚନା ggml_tensor src0;
    ସଂରଚନା ggml_tensor src1;
    ସଂରଚନା ggml_tensor dst;
};

ସଂରଚନା uber_ssm_conv_params {
    ସଂରଚନା ggml_tensor src0;
    ସଂରଚନା ggml_tensor src1;
    ସଂରଚନା ggml_tensor dst;
};

ସଂରଚନା uber_solve_tri_params {
    ସଂରଚନା ggml_tensor src0;
    ସଂରଚନା ggml_tensor src1;
    ସଂରଚନା ggml_tensor dst;
};

ସଂରଚନା uber_mul_mat_id_params {
    ସଂରଚନା ggml_tensor src0;
    ସଂରଚନା ggml_tensor src1;
    ସଂରଚନା ggml_tensor src2;
    ସଂରଚନା ggml_tensor dst;
};

// flash_attn_ext: Q=src0, K=src1, V=src2, mask=src3, dst (mask optional)
ସଂରଚନା uber_flash_attn_ext_params {
    ସଂରଚନା ggml_tensor src0;
    ସଂରଚନା ggml_tensor src1;
    ସଂରଚନା ggml_tensor src2;
    ସଂରଚନା ggml_tensor mask;
    ସଂରଚନା ggml_tensor dst;
};

// ssm_scan: 7 source tensors + dst
ସଂରଚନା uber_ssm_scan_params {
    ସଂରଚନା ggml_tensor src0;
    ସଂରଚନା ggml_tensor src1;
    ସଂରଚନା ggml_tensor src2;
    ସଂରଚନା ggml_tensor src3;
    ସଂରଚନା ggml_tensor src4;
    ସଂରଚନା ggml_tensor src5;
    ସଂରଚନା ggml_tensor src6;
    ସଂରଚନା ggml_tensor dst;
};

// gated_delta_net: q,k,v,g,beta,state_in,dst
ସଂରଚନା uber_gated_delta_net_params {
    ସଂରଚନା ggml_tensor q;
    ସଂରଚନା ggml_tensor k;
    ସଂରଚନା ggml_tensor v;
    ସଂରଚନା ggml_tensor g;
    ସଂରଚନା ggml_tensor beta;
    ସଂରଚନା ggml_tensor state_in;
    ସଂରଚନା ggml_tensor dst;
};

ସ୍ଥିର ଶୂନ୍ୟ copy_f32_to_f16_row(uint16_t * dst, ଅଚଳ ଦଶମିକ * src, int64_t ଅଙ୍କ_elements) {
    ପାଇଁ (int64_t i = 0; i < ଅଙ୍କ_elements; i++) {
        dst[i] = fp32_to_fp16(src[i]);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ copy_f32_row(ଦଶମିକ * dst, ଅଚଳ ଦଶମିକ * src, int64_t ଅଙ୍କ_elements) {
    ପାଇଁ (int64_t i = 0; i < ଅଙ୍କ_elements; i++) {
        dst[i] = src[i];
    }
}

ସ୍ଥିର ଶୂନ୍ୟ evict_region_past_l2_local(ଅଚଳ ଶୂନ୍ୟ * addr, size_t bytes) {
    ଯଦି (!addr || bytes == 0) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ uint64_t CL     = 64;
    uint64_t       base   = (uint64_t) addr & ~(CL - 1);
    uint64_t       end    = ((uint64_t) addr + bytes + CL - 1) & ~(CL - 1);
    uint64_t       nlines = (end - base) / CL;
    cache_ops_priv_evict_sw(0, /*to_L2*/ 3, 0, 0, CL);
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା entry_point(ସଂରଚନା ggml_et_uberkernel_params * params, ଶୂନ୍ୟ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ଯଦି (!kernel_env || !params) {
        ଫେରନ୍ତୁ -1;
    }

    ସଂରଚନା ggml_et_uberkernel_inst * insts       = (ସଂରଚନା ggml_et_uberkernel_inst *) (uintptr_t) params->insts;
    uint8_t *                        params_blob = (uint8_t *) (uintptr_t) params->params_blob;

    ଯଦି (!insts || !params_blob || params->inst_stride < ଆକାର(ସଂରଚନା ggml_et_uberkernel_inst)) {
        ଫେରନ୍ତୁ -1;
    }

    ପାଇଁ (uint32_t i = 0; i < params->ଅଙ୍କ_insts; ++i) {
        ସଂରଚନା ggml_et_uberkernel_inst * inst =
            (ସଂରଚନା ggml_et_uberkernel_inst *) ((uint8_t *) insts + (i * params->inst_stride));
        ଶୂନ୍ୟ * inst_params = params_blob + inst->params_offset;
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା    rc          = -1;

        et_barrier_global(32ULL);

        ବାଛ (inst->kernel_id) {
            ସ୍ଥିତି GGML_ET_UBERKERNEL_KERNEL_EL_MAP_F32:
                {
                    ସଂରଚନା ggml_et_ଦ୍ୱିମିକ_params * p = (ସଂରଚନା ggml_et_ଦ୍ୱିମିକ_params *) inst_params;
                    rc                               = el_map_f32_entry(p, env);
                    ଭାଙ୍ଗ;
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
            ସ୍ଥିତି GGML_ET_UBERKERNEL_KERNEL_GLU_F32:
                {
                    rc = glu_f32_entry((ସଂରଚନା ggml_et_glu_params *) inst_params, env);
                    ଭାଙ୍ଗ;
                }
            ସ୍ଥିତି GGML_ET_UBERKERNEL_KERNEL_ROPE_F32:
                {
                    rc = rope_f32_entry((ସଂରଚନା ggml_et_rope_params *) inst_params, env);
                    ଭାଙ୍ଗ;
                }
            ସ୍ଥିତି GGML_ET_UBERKERNEL_KERNEL_RMS_NORM_F32:
                {
                    // struct ggml_et_rms_norm_params *p = (struct ggml_et_rms_norm_params *) inst_params;
                    // evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
                    rc = rms_norm_f32_entry((ସଂରଚନା ggml_et_rms_norm_params *) inst_params, env);
                    ଭାଙ୍ଗ;
                }
            ସ୍ଥିତି GGML_ET_UBERKERNEL_KERNEL_RMS_NORM_MUL_F32:
                {
                    ସଂରଚନା uber_rms_norm_mul_params * p = (ସଂରଚନା uber_rms_norm_mul_params *) inst_params;
                    evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
                    evict_region_past_l2(p->src1.data, tensor_bytes(&p->src1));
                    rc = rms_norm_mul_f32_entry((ସଂରଚନା ggml_et_rms_norm_mul_params *) inst_params, env);
                    ଭାଙ୍ଗ;
                }
            ସ୍ଥିତି GGML_ET_UBERKERNEL_KERNEL_SOFTMAX_F32:
                {
                    rc = softmax_f32_entry((ସଂରଚନା ggml_et_softmax_params *) inst_params, env);
                    ଭାଙ୍ଗ;
                }
            ସ୍ଥିତି GGML_ET_UBERKERNEL_KERNEL_SET_ROWS_F32:
                {
                    rc = set_rows_f32_entry((ସଂରଚନା ggml_et_set_rows_params *) inst_params, env);
                    ଭାଙ୍ଗ;
                }

            // Single-source ops (src0 → dst)
            ସ୍ଥିତି GGML_ET_UBERKERNEL_KERNEL_SQR_F32:
                {
                    rc = sqr_f32_entry((ସଂରଚନା ggml_et_sqr_params *) inst_params, env);
                    ଭାଙ୍ଗ;
                }
            ସ୍ଥିତି GGML_ET_UBERKERNEL_KERNEL_SCALE_F32:
                {
                    rc = scale_f32_entry((ସଂରଚନା ggml_et_scale_params *) inst_params, env);
                    ଭାଙ୍ଗ;
                }
            ସ୍ଥିତି GGML_ET_UBERKERNEL_KERNEL_SUM_ROWS_F32:
                {
                    rc = sum_rows_f32_entry((ସଂରଚନା ggml_et_sum_rows_params *) inst_params, env);
                    ଭାଙ୍ଗ;
                }
            ସ୍ଥିତି GGML_ET_UBERKERNEL_KERNEL_CUMSUM_F32:
                {
                    rc = cumsum_f32_entry((ସଂରଚନା ggml_et_cumsum_params *) inst_params, env);
                    ଭାଙ୍ଗ;
                }
            ସ୍ଥିତି GGML_ET_UBERKERNEL_KERNEL_NORM_F32:
                {
                    rc = norm_f32_entry((ସଂରଚନା ggml_et_norm_params *) inst_params, env);
                    ଭାଙ୍ଗ;
                }
            ସ୍ଥିତି GGML_ET_UBERKERNEL_KERNEL_L2_NORM_F32:
                {
                    rc = l2_norm_f32_entry((ସଂରଚନା ggml_et_l2_norm_params *) inst_params, env);
                    ଭାଙ୍ଗ;
                }
            ସ୍ଥିତି GGML_ET_UBERKERNEL_KERNEL_GROUP_NORM_F32:
                {
                    rc = group_norm_f32_entry((ସଂରଚନା ggml_et_group_norm_params *) inst_params, env);
                    ଭାଙ୍ଗ;
                }
            ସ୍ଥିତି GGML_ET_UBERKERNEL_KERNEL_REPEAT_F32:
                {
                    rc = repeat_f32_entry((ସଂରଚନା ggml_et_repeat_params *) inst_params, env);
                    ଭାଙ୍ଗ;
                }
            ସ୍ଥିତି GGML_ET_UBERKERNEL_KERNEL_DIAG_F32:
                {
                    rc = diag_f32_entry((ସଂରଚନା ggml_et_diag_params *) inst_params, env);
                    ଭାଙ୍ଗ;
                }
            ସ୍ଥିତି GGML_ET_UBERKERNEL_KERNEL_TRI_F32:
                {
                    rc = tri_f32_entry((ସଂରଚନା ggml_et_tri_params *) inst_params, env);
                    ଭାଙ୍ଗ;
                }
            ସ୍ଥିତି GGML_ET_UBERKERNEL_KERNEL_PAD_F32:
                {
                    rc = pad_f32_entry((ସଂରଚନା ggml_et_pad_params *) inst_params, env);
                    ଭାଙ୍ଗ;
                }
            ସ୍ଥିତି GGML_ET_UBERKERNEL_KERNEL_CONT_F16:
                {
                    rc = cont_f16_entry((ସଂରଚନା ggml_et_cont_params *) inst_params, env);
                    ଭାଙ୍ଗ;
                }
            ସ୍ଥିତି GGML_ET_UBERKERNEL_KERNEL_FILL_F32:
                {
                    rc = fill_f32_entry((ସଂରଚନା ggml_et_fill_params *) inst_params, env);
                    ଭାଙ୍ଗ;
                }
            ସ୍ଥିତି GGML_ET_UBERKERNEL_KERNEL_SET_F32:
                {
                    rc = set_f32_entry((ସଂରଚନା ggml_et_set_params *) inst_params, env);
                    ଭାଙ୍ଗ;
                }

            // Two-source ops
            ସ୍ଥିତି GGML_ET_UBERKERNEL_KERNEL_CONCAT_F32:
                {
                    rc = concat_f32_entry((ସଂରଚନା ggml_et_concat_params *) inst_params, env);
                    ଭାଙ୍ଗ;
                }
            // case GGML_ET_UBERKERNEL_KERNEL_SSM_CONV_F32: {
            //     rc = ssm_conv_f32_entry((struct ggml_et_ssm_conv_params *) inst_params, env);
            //     break;
            // }
            ସ୍ଥିତି GGML_ET_UBERKERNEL_KERNEL_SOLVE_TRI_F32:
                {
                    rc = solve_tri_f32_entry((ସଂରଚନା ggml_et_solve_tri_params *) inst_params, env);
                    ଭାଙ୍ଗ;
                }
            ସ୍ଥିତି GGML_ET_UBERKERNEL_KERNEL_IM2COL:
                {
                    rc = im2col_entry((ସଂରଚନା ggml_et_im2col_params *) inst_params, env);
                    ଭାଙ୍ଗ;
                }

            // Three-source ops
            ସ୍ଥିତି GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_ID_F32:
                {
                    rc = mul_mat_id_f32_entry((ସଂରଚନା ggml_et_mul_mat_id_params *) inst_params, env);
                    ଭାଙ୍ଗ;
                }
            ସ୍ଥିତି GGML_ET_UBERKERNEL_KERNEL_FLASH_ATTN_EXT_F32:
                {
                    rc = flash_attn_ext_f32_entry((ସଂରଚନା ggml_et_flash_attn_ext_params *) inst_params, env);
                    ଭାଙ୍ଗ;
                }
            ସ୍ଥିତି GGML_ET_UBERKERNEL_KERNEL_FLASH_ATTN_EXT_F16_ME:
                {
                    rc = flash_attn_ext_f16_me_entry((ସଂରଚନା ggml_et_flash_attn_ext_params *) inst_params, env);
                    ଭାଙ୍ଗ;
                }

            ସ୍ଥିତି GGML_ET_UBERKERNEL_KERNEL_GATED_DELTA_NET_F32:
                {
                    rc = gated_delta_net_f32_entry((ସଂରଚନା ggml_et_gated_delta_net_params *) inst_params, env);
                    ଭାଙ୍ଗ;
                }
            ସ୍ଥିତି GGML_ET_UBERKERNEL_KERNEL_SSM_SCAN_F32:
                {
                    rc = ssm_scan_f32_entry((ସଂରଚନା ggml_et_ssm_scan_params *) inst_params, env);
                    ଭାଙ୍ଗ;
                }
            // rwkv: raw float* params, no ggml_tensor fields to evict via
            ସ୍ଥିତି GGML_ET_UBERKERNEL_KERNEL_RWKV_WKV6_F32:
                {
                    rc = rwkv_wkv6_f32_entry((ସଂରଚନା ggml_et_rwkv_wkv6_params *) inst_params, env);
                    ଭାଙ୍ଗ;
                }

            ସ୍ଥିତି GGML_ET_UBERKERNEL_KERNEL_RWKV_WKV7_F32:
                {
                    rc = rwkv_wkv7_f32_entry((ସଂରଚନା ggml_et_rwkv_wkv7_params *) inst_params, env);
                    ଭାଙ୍ଗ;
                }

            // MUL_MAT: evict src1 (activations); src0=weights is
            //  read-only so never stale from a prior uberkernel op
            ସ୍ଥିତି GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F16:
                {
                    ସଂରଚନା ggml_et_ଦ୍ୱିମିକ_params * p = (ସଂରଚନା ggml_et_ଦ୍ୱିମିକ_params *) inst_params;
                    rc                               = mul_mat_f16_entry(p, env);
                    ଭାଙ୍ଗ;
                }
            ସ୍ଥିତି GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F16_MATRIX_ENGINE:
                {
                    ସଂରଚନା ggml_et_ଦ୍ୱିମିକ_params * p = (ସଂରଚନା ggml_et_ଦ୍ୱିମିକ_params *) inst_params;
                    rc                               = mul_mat_f16_matrix_engine_entry(p, env);
                    ଭାଙ୍ଗ;
                }
            ସ୍ଥିତି GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F32:
                {
                    ସଂରଚନା ggml_et_ଦ୍ୱିମିକ_params * p = (ସଂରଚନା ggml_et_ଦ୍ୱିମିକ_params *) inst_params;
                    rc                               = mul_mat_f32_entry(p, env);
                    ଭାଙ୍ଗ;
                }
            ସ୍ଥିତି GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F32_MATRIX_ENGINE:
                {
                    ସଂରଚନା ggml_et_ଦ୍ୱିମିକ_params * p = (ସଂରଚନା ggml_et_ଦ୍ୱିମିକ_params *) inst_params;
                    rc                               = mul_mat_f32_matrix_engine_entry(p, env);
                    ଭାଙ୍ଗ;
                }
            ସ୍ଥିତି GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_Q8_0:
                {
                    ସଂରଚନା ggml_et_mm_q8_params * p = (ସଂରଚନା ggml_et_mm_q8_params *) inst_params;
                    // evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
                    rc                              = mul_mat_Q8_0_entry(p, env);
                    ଭାଙ୍ଗ;
                }
            ସ୍ଥିତି GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_Q4_0:
                {
                    ସଂରଚନା ggml_et_ଦ୍ୱିମିକ_params * p = (ସଂରଚନା ggml_et_ଦ୍ୱିମିକ_params *) inst_params;
                    rc                               = mul_mat_Q4_0_entry(p, env);
                    ଭାଙ୍ଗ;
                }

            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                ଫେରନ୍ତୁ -1;
        }

        ଯଦି (rc != 0) {
            ଫେରନ୍ତୁ rc;
        }
    }

    ଫେରନ୍ତୁ 0;
}
