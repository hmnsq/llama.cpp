#समावेश "ggml-et-uberkernel-common.h"
#समावेश "ggml-et-uberkernel-kernel-map.h"
#समावेश "ggml_tensor.h"
#समावेश "math_fp.h"
#समावेश "platform.h"

#समावेश <stdint.h>

काष्ठा ggml_et_glu_params;
काष्ठा ggml_et_unary_params;
काष्ठा ggml_et_rope_params;
काष्ठा ggml_et_rms_norm_params;
काष्ठा ggml_et_rms_norm_mul_params;
काष्ठा ggml_et_softmax_params;
काष्ठा ggml_et_set_rows_params;
काष्ठा ggml_et_get_rows_params;
काष्ठा ggml_et_cont_params;
काष्ठा ggml_et_concat_params;
काष्ठा ggml_et_cumsum_params;
काष्ठा ggml_et_diag_params;
काष्ठा ggml_et_fill_params;
काष्ठा ggml_et_flash_attn_ext_params;
काष्ठा ggml_et_gated_delta_net_params;
काष्ठा ggml_et_group_norm_params;
काष्ठा ggml_et_im2col_params;
काष्ठा ggml_et_l2_norm_params;
काष्ठा ggml_et_mul_mat_id_params;
काष्ठा ggml_et_norm_params;
काष्ठा ggml_et_pad_params;
काष्ठा ggml_et_repeat_params;
काष्ठा ggml_et_rwkv_wkv6_params;
काष्ठा ggml_et_rwkv_wkv7_params;
काष्ठा ggml_et_scale_params;
काष्ठा ggml_et_set_params;
काष्ठा ggml_et_solve_tri_params;
काष्ठा ggml_et_sqr_params;
काष्ठा ggml_et_ssm_conv_params;
काष्ठा ggml_et_ssm_scan_params;
काष्ठा ggml_et_योग_rows_params;
काष्ठा ggml_et_tri_params;

बाह्य पूर्णांक el_map_f32_entry(काष्ठा ggml_et_द्विआधारी_params *, व्योम *);
बाह्य पूर्णांक glu_f32_entry(काष्ठा ggml_et_glu_params *, व्योम *);
बाह्य पूर्णांक unary_f32_entry(काष्ठा ggml_et_unary_params *, व्योम *);
बाह्य पूर्णांक rope_f32_entry(काष्ठा ggml_et_rope_params *, व्योम *);
बाह्य पूर्णांक rms_norm_f32_entry(काष्ठा ggml_et_rms_norm_params *, व्योम *);
बाह्य पूर्णांक rms_norm_mul_f32_entry(काष्ठा ggml_et_rms_norm_mul_params *, व्योम *);
बाह्य पूर्णांक softmax_f32_entry(काष्ठा ggml_et_softmax_params *, व्योम *);
बाह्य पूर्णांक set_rows_f32_entry(काष्ठा ggml_et_set_rows_params *, व्योम *);
बाह्य पूर्णांक get_rows_f32_entry(काष्ठा ggml_et_get_rows_params *, व्योम *);
बाह्य पूर्णांक cont_f32_entry(काष्ठा ggml_et_cont_params *, व्योम *);
बाह्य पूर्णांक cont_f16_entry(काष्ठा ggml_et_cont_params *, व्योम *);
बाह्य पूर्णांक cpy_f32_f16_entry(काष्ठा ggml_et_cont_params *, व्योम *);
बाह्य पूर्णांक concat_f32_entry(काष्ठा ggml_et_concat_params *, व्योम *);
बाह्य पूर्णांक cumsum_f32_entry(काष्ठा ggml_et_cumsum_params *, व्योम *);
बाह्य पूर्णांक diag_f32_entry(काष्ठा ggml_et_diag_params *, व्योम *);
बाह्य पूर्णांक fill_f32_entry(काष्ठा ggml_et_fill_params *, व्योम *);
बाह्य पूर्णांक flash_attn_ext_f32_entry(काष्ठा ggml_et_flash_attn_ext_params *, व्योम *);
बाह्य पूर्णांक flash_attn_ext_f16_me_entry(काष्ठा ggml_et_flash_attn_ext_params *, व्योम *);
बाह्य पूर्णांक gated_delta_net_f32_entry(काष्ठा ggml_et_gated_delta_net_params *, व्योम *);
बाह्य पूर्णांक group_norm_f32_entry(काष्ठा ggml_et_group_norm_params *, व्योम *);
बाह्य पूर्णांक im2col_entry(काष्ठा ggml_et_im2col_params *, व्योम *);
बाह्य पूर्णांक l2_norm_f32_entry(काष्ठा ggml_et_l2_norm_params *, व्योम *);
बाह्य पूर्णांक mul_mat_id_f32_entry(काष्ठा ggml_et_mul_mat_id_params *, व्योम *);
बाह्य पूर्णांक norm_f32_entry(काष्ठा ggml_et_norm_params *, व्योम *);
बाह्य पूर्णांक pad_f32_entry(काष्ठा ggml_et_pad_params *, व्योम *);
बाह्य पूर्णांक repeat_f32_entry(काष्ठा ggml_et_repeat_params *, व्योम *);
बाह्य पूर्णांक rwkv_wkv6_f32_entry(काष्ठा ggml_et_rwkv_wkv6_params *, व्योम *);
बाह्य पूर्णांक rwkv_wkv7_f32_entry(काष्ठा ggml_et_rwkv_wkv7_params *, व्योम *);
बाह्य पूर्णांक scale_f32_entry(काष्ठा ggml_et_scale_params *, व्योम *);
बाह्य पूर्णांक set_f32_entry(काष्ठा ggml_et_set_params *, व्योम *);
बाह्य पूर्णांक solve_tri_f32_entry(काष्ठा ggml_et_solve_tri_params *, व्योम *);
बाह्य पूर्णांक sqr_f32_entry(काष्ठा ggml_et_sqr_params *, व्योम *);
बाह्य पूर्णांक ssm_conv_f32_entry(काष्ठा ggml_et_ssm_conv_params *, व्योम *);
बाह्य पूर्णांक ssm_scan_f32_entry(काष्ठा ggml_et_ssm_scan_params *, व्योम *);
बाह्य पूर्णांक योग_rows_f32_entry(काष्ठा ggml_et_योग_rows_params *, व्योम *);
बाह्य पूर्णांक tri_f32_entry(काष्ठा ggml_et_tri_params *, व्योम *);
बाह्य पूर्णांक mul_mat_f16_entry(काष्ठा ggml_et_द्विआधारी_params *, व्योम *);
बाह्य पूर्णांक mul_mat_f16_matrix_engine_entry(काष्ठा ggml_et_द्विआधारी_params *, व्योम *);
बाह्य पूर्णांक mul_mat_f32_entry(काष्ठा ggml_et_द्विआधारी_params *, व्योम *);
बाह्य पूर्णांक mul_mat_f32_matrix_engine_entry(काष्ठा ggml_et_द्विआधारी_params *, व्योम *);
बाह्य पूर्णांक mul_mat_Q8_0_entry(काष्ठा ggml_et_mm_q8_params *, व्योम *);
बाह्य पूर्णांक mul_mat_Q4_0_entry(काष्ठा ggml_et_द्विआधारी_params *, व्योम *);

अटल अंतरभूत माप_प्रकार tensor_bytes(स्थिर काष्ठा ggml_tensor * t) {
    वापस (माप_प्रकार) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

काष्ठा uber_glu_params {
    काष्ठा ggml_tensor src0;
    काष्ठा ggml_tensor src1;
    काष्ठा ggml_tensor dst;
    // trailing scalars omitted — not needed for eviction
};

काष्ठा uber_unary_params {
    काष्ठा ggml_tensor src0;
    काष्ठा ggml_tensor dst;
};

काष्ठा uber_rope_params {
    काष्ठा ggml_tensor src0;
    काष्ठा ggml_tensor src1;
    काष्ठा ggml_tensor src2;
    काष्ठा ggml_tensor dst;
};

काष्ठा uber_rms_norm_params {
    काष्ठा ggml_tensor src0;
    काष्ठा ggml_tensor dst;
};

काष्ठा uber_rms_norm_mul_params {
    काष्ठा ggml_tensor src0;
    काष्ठा ggml_tensor src1;
    काष्ठा ggml_tensor dst;
};

काष्ठा uber_softmax_params {
    काष्ठा ggml_tensor src0;
    काष्ठा ggml_tensor src1;
    काष्ठा ggml_tensor src2;
    काष्ठा ggml_tensor dst;
};

काष्ठा uber_set_rows_params {
    काष्ठा ggml_tensor src0;
    काष्ठा ggml_tensor src1;
    काष्ठा ggml_tensor dst;
};

काष्ठा uber_get_rows_params {
    काष्ठा ggml_tensor src0;
    काष्ठा ggml_tensor src1;
    काष्ठा ggml_tensor dst;
};

काष्ठा uber_cont_params {
    काष्ठा ggml_tensor src0;
    काष्ठा ggml_tensor dst;
};

// src0 + src1 + dst (no trailing scalars needed for eviction)
काष्ठा uber_concat_params {
    काष्ठा ggml_tensor src0;
    काष्ठा ggml_tensor src1;
    काष्ठा ggml_tensor dst;
};

काष्ठा uber_ssm_conv_params {
    काष्ठा ggml_tensor src0;
    काष्ठा ggml_tensor src1;
    काष्ठा ggml_tensor dst;
};

काष्ठा uber_solve_tri_params {
    काष्ठा ggml_tensor src0;
    काष्ठा ggml_tensor src1;
    काष्ठा ggml_tensor dst;
};

काष्ठा uber_mul_mat_id_params {
    काष्ठा ggml_tensor src0;
    काष्ठा ggml_tensor src1;
    काष्ठा ggml_tensor src2;
    काष्ठा ggml_tensor dst;
};

// flash_attn_ext: Q=src0, K=src1, V=src2, mask=src3, dst (mask optional)
काष्ठा uber_flash_attn_ext_params {
    काष्ठा ggml_tensor src0;
    काष्ठा ggml_tensor src1;
    काष्ठा ggml_tensor src2;
    काष्ठा ggml_tensor mask;
    काष्ठा ggml_tensor dst;
};

// ssm_scan: 7 source tensors + dst
काष्ठा uber_ssm_scan_params {
    काष्ठा ggml_tensor src0;
    काष्ठा ggml_tensor src1;
    काष्ठा ggml_tensor src2;
    काष्ठा ggml_tensor src3;
    काष्ठा ggml_tensor src4;
    काष्ठा ggml_tensor src5;
    काष्ठा ggml_tensor src6;
    काष्ठा ggml_tensor dst;
};

// gated_delta_net: q,k,v,g,beta,state_in,dst
काष्ठा uber_gated_delta_net_params {
    काष्ठा ggml_tensor q;
    काष्ठा ggml_tensor k;
    काष्ठा ggml_tensor v;
    काष्ठा ggml_tensor g;
    काष्ठा ggml_tensor beta;
    काष्ठा ggml_tensor state_in;
    काष्ठा ggml_tensor dst;
};

अटल व्योम copy_f32_to_f16_row(uint16_t * dst, स्थिर भग्न * src, int64_t अंक_elements) {
    क्रम (int64_t i = 0; i < अंक_elements; i++) {
        dst[i] = fp32_to_fp16(src[i]);
    }
}

अटल व्योम copy_f32_row(भग्न * dst, स्थिर भग्न * src, int64_t अंक_elements) {
    क्रम (int64_t i = 0; i < अंक_elements; i++) {
        dst[i] = src[i];
    }
}

अटल व्योम evict_region_past_l2_local(स्थिर व्योम * addr, माप_प्रकार bytes) {
    अगर (!addr || bytes == 0) {
        वापस;
    }

    स्थिर uint64_t CL     = 64;
    uint64_t       base   = (uint64_t) addr & ~(CL - 1);
    uint64_t       end    = ((uint64_t) addr + bytes + CL - 1) & ~(CL - 1);
    uint64_t       nlines = (end - base) / CL;
    cache_ops_priv_evict_sw(0, /*to_L2*/ 3, 0, 0, CL);
}

पूर्णांक entry_point(काष्ठा ggml_et_uberkernel_params * params, व्योम * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    अगर (!kernel_env || !params) {
        वापस -1;
    }

    काष्ठा ggml_et_uberkernel_inst * insts       = (काष्ठा ggml_et_uberkernel_inst *) (uintptr_t) params->insts;
    uint8_t *                        params_blob = (uint8_t *) (uintptr_t) params->params_blob;

    अगर (!insts || !params_blob || params->inst_stride < माप(काष्ठा ggml_et_uberkernel_inst)) {
        वापस -1;
    }

    क्रम (uint32_t i = 0; i < params->अंक_insts; ++i) {
        काष्ठा ggml_et_uberkernel_inst * inst =
            (काष्ठा ggml_et_uberkernel_inst *) ((uint8_t *) insts + (i * params->inst_stride));
        व्योम * inst_params = params_blob + inst->params_offset;
        पूर्णांक    rc          = -1;

        et_barrier_global(32ULL);

        चयन (inst->kernel_id) {
            हाल GGML_ET_UBERKERNEL_KERNEL_EL_MAP_F32:
                {
                    काष्ठा ggml_et_द्विआधारी_params * p = (काष्ठा ggml_et_द्विआधारी_params *) inst_params;
                    rc                               = el_map_f32_entry(p, env);
                    अवरोध;
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
            हाल GGML_ET_UBERKERNEL_KERNEL_GLU_F32:
                {
                    rc = glu_f32_entry((काष्ठा ggml_et_glu_params *) inst_params, env);
                    अवरोध;
                }
            हाल GGML_ET_UBERKERNEL_KERNEL_ROPE_F32:
                {
                    rc = rope_f32_entry((काष्ठा ggml_et_rope_params *) inst_params, env);
                    अवरोध;
                }
            हाल GGML_ET_UBERKERNEL_KERNEL_RMS_NORM_F32:
                {
                    // struct ggml_et_rms_norm_params *p = (struct ggml_et_rms_norm_params *) inst_params;
                    // evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
                    rc = rms_norm_f32_entry((काष्ठा ggml_et_rms_norm_params *) inst_params, env);
                    अवरोध;
                }
            हाल GGML_ET_UBERKERNEL_KERNEL_RMS_NORM_MUL_F32:
                {
                    काष्ठा uber_rms_norm_mul_params * p = (काष्ठा uber_rms_norm_mul_params *) inst_params;
                    evict_region_past_l2(p->src0.तथ्य, tensor_bytes(&p->src0));
                    evict_region_past_l2(p->src1.तथ्य, tensor_bytes(&p->src1));
                    rc = rms_norm_mul_f32_entry((काष्ठा ggml_et_rms_norm_mul_params *) inst_params, env);
                    अवरोध;
                }
            हाल GGML_ET_UBERKERNEL_KERNEL_SOFTMAX_F32:
                {
                    rc = softmax_f32_entry((काष्ठा ggml_et_softmax_params *) inst_params, env);
                    अवरोध;
                }
            हाल GGML_ET_UBERKERNEL_KERNEL_SET_ROWS_F32:
                {
                    rc = set_rows_f32_entry((काष्ठा ggml_et_set_rows_params *) inst_params, env);
                    अवरोध;
                }

            // Single-source ops (src0 → dst)
            हाल GGML_ET_UBERKERNEL_KERNEL_SQR_F32:
                {
                    rc = sqr_f32_entry((काष्ठा ggml_et_sqr_params *) inst_params, env);
                    अवरोध;
                }
            हाल GGML_ET_UBERKERNEL_KERNEL_SCALE_F32:
                {
                    rc = scale_f32_entry((काष्ठा ggml_et_scale_params *) inst_params, env);
                    अवरोध;
                }
            हाल GGML_ET_UBERKERNEL_KERNEL_योग_ROWS_F32:
                {
                    rc = योग_rows_f32_entry((काष्ठा ggml_et_योग_rows_params *) inst_params, env);
                    अवरोध;
                }
            हाल GGML_ET_UBERKERNEL_KERNEL_CUMSUM_F32:
                {
                    rc = cumsum_f32_entry((काष्ठा ggml_et_cumsum_params *) inst_params, env);
                    अवरोध;
                }
            हाल GGML_ET_UBERKERNEL_KERNEL_NORM_F32:
                {
                    rc = norm_f32_entry((काष्ठा ggml_et_norm_params *) inst_params, env);
                    अवरोध;
                }
            हाल GGML_ET_UBERKERNEL_KERNEL_L2_NORM_F32:
                {
                    rc = l2_norm_f32_entry((काष्ठा ggml_et_l2_norm_params *) inst_params, env);
                    अवरोध;
                }
            हाल GGML_ET_UBERKERNEL_KERNEL_GROUP_NORM_F32:
                {
                    rc = group_norm_f32_entry((काष्ठा ggml_et_group_norm_params *) inst_params, env);
                    अवरोध;
                }
            हाल GGML_ET_UBERKERNEL_KERNEL_REPEAT_F32:
                {
                    rc = repeat_f32_entry((काष्ठा ggml_et_repeat_params *) inst_params, env);
                    अवरोध;
                }
            हाल GGML_ET_UBERKERNEL_KERNEL_DIAG_F32:
                {
                    rc = diag_f32_entry((काष्ठा ggml_et_diag_params *) inst_params, env);
                    अवरोध;
                }
            हाल GGML_ET_UBERKERNEL_KERNEL_TRI_F32:
                {
                    rc = tri_f32_entry((काष्ठा ggml_et_tri_params *) inst_params, env);
                    अवरोध;
                }
            हाल GGML_ET_UBERKERNEL_KERNEL_PAD_F32:
                {
                    rc = pad_f32_entry((काष्ठा ggml_et_pad_params *) inst_params, env);
                    अवरोध;
                }
            हाल GGML_ET_UBERKERNEL_KERNEL_CONT_F16:
                {
                    rc = cont_f16_entry((काष्ठा ggml_et_cont_params *) inst_params, env);
                    अवरोध;
                }
            हाल GGML_ET_UBERKERNEL_KERNEL_FILL_F32:
                {
                    rc = fill_f32_entry((काष्ठा ggml_et_fill_params *) inst_params, env);
                    अवरोध;
                }
            हाल GGML_ET_UBERKERNEL_KERNEL_SET_F32:
                {
                    rc = set_f32_entry((काष्ठा ggml_et_set_params *) inst_params, env);
                    अवरोध;
                }

            // Two-source ops
            हाल GGML_ET_UBERKERNEL_KERNEL_CONCAT_F32:
                {
                    rc = concat_f32_entry((काष्ठा ggml_et_concat_params *) inst_params, env);
                    अवरोध;
                }
            // case GGML_ET_UBERKERNEL_KERNEL_SSM_CONV_F32: {
            //     rc = ssm_conv_f32_entry((struct ggml_et_ssm_conv_params *) inst_params, env);
            //     break;
            // }
            हाल GGML_ET_UBERKERNEL_KERNEL_SOLVE_TRI_F32:
                {
                    rc = solve_tri_f32_entry((काष्ठा ggml_et_solve_tri_params *) inst_params, env);
                    अवरोध;
                }
            हाल GGML_ET_UBERKERNEL_KERNEL_IM2COL:
                {
                    rc = im2col_entry((काष्ठा ggml_et_im2col_params *) inst_params, env);
                    अवरोध;
                }

            // Three-source ops
            हाल GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_ID_F32:
                {
                    rc = mul_mat_id_f32_entry((काष्ठा ggml_et_mul_mat_id_params *) inst_params, env);
                    अवरोध;
                }
            हाल GGML_ET_UBERKERNEL_KERNEL_FLASH_ATTN_EXT_F32:
                {
                    rc = flash_attn_ext_f32_entry((काष्ठा ggml_et_flash_attn_ext_params *) inst_params, env);
                    अवरोध;
                }
            हाल GGML_ET_UBERKERNEL_KERNEL_FLASH_ATTN_EXT_F16_ME:
                {
                    rc = flash_attn_ext_f16_me_entry((काष्ठा ggml_et_flash_attn_ext_params *) inst_params, env);
                    अवरोध;
                }

            हाल GGML_ET_UBERKERNEL_KERNEL_GATED_DELTA_NET_F32:
                {
                    rc = gated_delta_net_f32_entry((काष्ठा ggml_et_gated_delta_net_params *) inst_params, env);
                    अवरोध;
                }
            हाल GGML_ET_UBERKERNEL_KERNEL_SSM_SCAN_F32:
                {
                    rc = ssm_scan_f32_entry((काष्ठा ggml_et_ssm_scan_params *) inst_params, env);
                    अवरोध;
                }
            // rwkv: raw float* params, no ggml_tensor fields to evict via
            हाल GGML_ET_UBERKERNEL_KERNEL_RWKV_WKV6_F32:
                {
                    rc = rwkv_wkv6_f32_entry((काष्ठा ggml_et_rwkv_wkv6_params *) inst_params, env);
                    अवरोध;
                }

            हाल GGML_ET_UBERKERNEL_KERNEL_RWKV_WKV7_F32:
                {
                    rc = rwkv_wkv7_f32_entry((काष्ठा ggml_et_rwkv_wkv7_params *) inst_params, env);
                    अवरोध;
                }

            // MUL_MAT: evict src1 (activations); src0=weights is
            //  read-only so never stale from a prior uberkernel op
            हाल GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F16:
                {
                    काष्ठा ggml_et_द्विआधारी_params * p = (काष्ठा ggml_et_द्विआधारी_params *) inst_params;
                    rc                               = mul_mat_f16_entry(p, env);
                    अवरोध;
                }
            हाल GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F16_MATRIX_ENGINE:
                {
                    काष्ठा ggml_et_द्विआधारी_params * p = (काष्ठा ggml_et_द्विआधारी_params *) inst_params;
                    rc                               = mul_mat_f16_matrix_engine_entry(p, env);
                    अवरोध;
                }
            हाल GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F32:
                {
                    काष्ठा ggml_et_द्विआधारी_params * p = (काष्ठा ggml_et_द्विआधारी_params *) inst_params;
                    rc                               = mul_mat_f32_entry(p, env);
                    अवरोध;
                }
            हाल GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F32_MATRIX_ENGINE:
                {
                    काष्ठा ggml_et_द्विआधारी_params * p = (काष्ठा ggml_et_द्विआधारी_params *) inst_params;
                    rc                               = mul_mat_f32_matrix_engine_entry(p, env);
                    अवरोध;
                }
            हाल GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_Q8_0:
                {
                    काष्ठा ggml_et_mm_q8_params * p = (काष्ठा ggml_et_mm_q8_params *) inst_params;
                    // evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
                    rc                              = mul_mat_Q8_0_entry(p, env);
                    अवरोध;
                }
            हाल GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_Q4_0:
                {
                    काष्ठा ggml_et_द्विआधारी_params * p = (काष्ठा ggml_et_द्विआधारी_params *) inst_params;
                    rc                               = mul_mat_Q4_0_entry(p, env);
                    अवरोध;
                }

            शेष:
                वापस -1;
        }

        अगर (rc != 0) {
            वापस rc;
        }
    }

    वापस 0;
}
