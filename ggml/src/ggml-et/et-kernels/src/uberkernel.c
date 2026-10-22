#अन्तर्भाव "ggml-et-uberkernel-common.h"
#अन्तर्भाव "ggml-et-uberkernel-kernel-map.h"
#अन्तर्भाव "ggml_tensor.h"
#अन्तर्भाव "math_fp.h"
#अन्तर्भाव "platform.h"

#अन्तर्भाव <stdint.h>

संरचना ggml_et_glu_params;
संरचना ggml_et_unary_params;
संरचना ggml_et_rope_params;
संरचना ggml_et_rms_norm_params;
संरचना ggml_et_rms_norm_mul_params;
संरचना ggml_et_softmax_params;
संरचना ggml_et_set_rows_params;
संरचना ggml_et_get_rows_params;
संरचना ggml_et_cont_params;
संरचना ggml_et_concat_params;
संरचना ggml_et_cumsum_params;
संरचना ggml_et_diag_params;
संरचना ggml_et_fill_params;
संरचना ggml_et_flash_attn_ext_params;
संरचना ggml_et_gated_delta_net_params;
संरचना ggml_et_group_norm_params;
संरचना ggml_et_im2col_params;
संरचना ggml_et_l2_norm_params;
संरचना ggml_et_mul_mat_id_params;
संरचना ggml_et_norm_params;
संरचना ggml_et_pad_params;
संरचना ggml_et_repeat_params;
संरचना ggml_et_rwkv_wkv6_params;
संरचना ggml_et_rwkv_wkv7_params;
संरचना ggml_et_scale_params;
संरचना ggml_et_set_params;
संरचना ggml_et_solve_tri_params;
संरचना ggml_et_sqr_params;
संरचना ggml_et_ssm_conv_params;
संरचना ggml_et_ssm_scan_params;
संरचना ggml_et_sum_rows_params;
संरचना ggml_et_tri_params;

बाह्य पूर्णांक el_map_f32_entry(संरचना ggml_et_द्विमान_params *, शून्य *);
बाह्य पूर्णांक glu_f32_entry(संरचना ggml_et_glu_params *, शून्य *);
बाह्य पूर्णांक unary_f32_entry(संरचना ggml_et_unary_params *, शून्य *);
बाह्य पूर्णांक rope_f32_entry(संरचना ggml_et_rope_params *, शून्य *);
बाह्य पूर्णांक rms_norm_f32_entry(संरचना ggml_et_rms_norm_params *, शून्य *);
बाह्य पूर्णांक rms_norm_mul_f32_entry(संरचना ggml_et_rms_norm_mul_params *, शून्य *);
बाह्य पूर्णांक softmax_f32_entry(संरचना ggml_et_softmax_params *, शून्य *);
बाह्य पूर्णांक set_rows_f32_entry(संरचना ggml_et_set_rows_params *, शून्य *);
बाह्य पूर्णांक get_rows_f32_entry(संरचना ggml_et_get_rows_params *, शून्य *);
बाह्य पूर्णांक cont_f32_entry(संरचना ggml_et_cont_params *, शून्य *);
बाह्य पूर्णांक cont_f16_entry(संरचना ggml_et_cont_params *, शून्य *);
बाह्य पूर्णांक cpy_f32_f16_entry(संरचना ggml_et_cont_params *, शून्य *);
बाह्य पूर्णांक concat_f32_entry(संरचना ggml_et_concat_params *, शून्य *);
बाह्य पूर्णांक cumsum_f32_entry(संरचना ggml_et_cumsum_params *, शून्य *);
बाह्य पूर्णांक diag_f32_entry(संरचना ggml_et_diag_params *, शून्य *);
बाह्य पूर्णांक fill_f32_entry(संरचना ggml_et_fill_params *, शून्य *);
बाह्य पूर्णांक flash_attn_ext_f32_entry(संरचना ggml_et_flash_attn_ext_params *, शून्य *);
बाह्य पूर्णांक flash_attn_ext_f16_me_entry(संरचना ggml_et_flash_attn_ext_params *, शून्य *);
बाह्य पूर्णांक gated_delta_net_f32_entry(संरचना ggml_et_gated_delta_net_params *, शून्य *);
बाह्य पूर्णांक group_norm_f32_entry(संरचना ggml_et_group_norm_params *, शून्य *);
बाह्य पूर्णांक im2col_entry(संरचना ggml_et_im2col_params *, शून्य *);
बाह्य पूर्णांक l2_norm_f32_entry(संरचना ggml_et_l2_norm_params *, शून्य *);
बाह्य पूर्णांक mul_mat_id_f32_entry(संरचना ggml_et_mul_mat_id_params *, शून्य *);
बाह्य पूर्णांक norm_f32_entry(संरचना ggml_et_norm_params *, शून्य *);
बाह्य पूर्णांक pad_f32_entry(संरचना ggml_et_pad_params *, शून्य *);
बाह्य पूर्णांक repeat_f32_entry(संरचना ggml_et_repeat_params *, शून्य *);
बाह्य पूर्णांक rwkv_wkv6_f32_entry(संरचना ggml_et_rwkv_wkv6_params *, शून्य *);
बाह्य पूर्णांक rwkv_wkv7_f32_entry(संरचना ggml_et_rwkv_wkv7_params *, शून्य *);
बाह्य पूर्णांक scale_f32_entry(संरचना ggml_et_scale_params *, शून्य *);
बाह्य पूर्णांक set_f32_entry(संरचना ggml_et_set_params *, शून्य *);
बाह्य पूर्णांक solve_tri_f32_entry(संरचना ggml_et_solve_tri_params *, शून्य *);
बाह्य पूर्णांक sqr_f32_entry(संरचना ggml_et_sqr_params *, शून्य *);
बाह्य पूर्णांक ssm_conv_f32_entry(संरचना ggml_et_ssm_conv_params *, शून्य *);
बाह्य पूर्णांक ssm_scan_f32_entry(संरचना ggml_et_ssm_scan_params *, शून्य *);
बाह्य पूर्णांक sum_rows_f32_entry(संरचना ggml_et_sum_rows_params *, शून्य *);
बाह्य पूर्णांक tri_f32_entry(संरचना ggml_et_tri_params *, शून्य *);
बाह्य पूर्णांक mul_mat_f16_entry(संरचना ggml_et_द्विमान_params *, शून्य *);
बाह्य पूर्णांक mul_mat_f16_matrix_engine_entry(संरचना ggml_et_द्विमान_params *, शून्य *);
बाह्य पूर्णांक mul_mat_f32_entry(संरचना ggml_et_द्विमान_params *, शून्य *);
बाह्य पूर्णांक mul_mat_f32_matrix_engine_entry(संरचना ggml_et_द्विमान_params *, शून्य *);
बाह्य पूर्णांक mul_mat_Q8_0_entry(संरचना ggml_et_mm_q8_params *, शून्य *);
बाह्य पूर्णांक mul_mat_Q4_0_entry(संरचना ggml_et_द्विमान_params *, शून्य *);

स्थिर ओळीत माप_प्रकार tensor_bytes(स्थिरांक संरचना ggml_tensor * t) {
    परत (माप_प्रकार) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

संरचना uber_glu_params {
    संरचना ggml_tensor src0;
    संरचना ggml_tensor src1;
    संरचना ggml_tensor dst;
    // trailing scalars omitted — not needed for eviction
};

संरचना uber_unary_params {
    संरचना ggml_tensor src0;
    संरचना ggml_tensor dst;
};

संरचना uber_rope_params {
    संरचना ggml_tensor src0;
    संरचना ggml_tensor src1;
    संरचना ggml_tensor src2;
    संरचना ggml_tensor dst;
};

संरचना uber_rms_norm_params {
    संरचना ggml_tensor src0;
    संरचना ggml_tensor dst;
};

संरचना uber_rms_norm_mul_params {
    संरचना ggml_tensor src0;
    संरचना ggml_tensor src1;
    संरचना ggml_tensor dst;
};

संरचना uber_softmax_params {
    संरचना ggml_tensor src0;
    संरचना ggml_tensor src1;
    संरचना ggml_tensor src2;
    संरचना ggml_tensor dst;
};

संरचना uber_set_rows_params {
    संरचना ggml_tensor src0;
    संरचना ggml_tensor src1;
    संरचना ggml_tensor dst;
};

संरचना uber_get_rows_params {
    संरचना ggml_tensor src0;
    संरचना ggml_tensor src1;
    संरचना ggml_tensor dst;
};

संरचना uber_cont_params {
    संरचना ggml_tensor src0;
    संरचना ggml_tensor dst;
};

// src0 + src1 + dst (no trailing scalars needed for eviction)
संरचना uber_concat_params {
    संरचना ggml_tensor src0;
    संरचना ggml_tensor src1;
    संरचना ggml_tensor dst;
};

संरचना uber_ssm_conv_params {
    संरचना ggml_tensor src0;
    संरचना ggml_tensor src1;
    संरचना ggml_tensor dst;
};

संरचना uber_solve_tri_params {
    संरचना ggml_tensor src0;
    संरचना ggml_tensor src1;
    संरचना ggml_tensor dst;
};

संरचना uber_mul_mat_id_params {
    संरचना ggml_tensor src0;
    संरचना ggml_tensor src1;
    संरचना ggml_tensor src2;
    संरचना ggml_tensor dst;
};

// flash_attn_ext: Q=src0, K=src1, V=src2, mask=src3, dst (mask optional)
संरचना uber_flash_attn_ext_params {
    संरचना ggml_tensor src0;
    संरचना ggml_tensor src1;
    संरचना ggml_tensor src2;
    संरचना ggml_tensor mask;
    संरचना ggml_tensor dst;
};

// ssm_scan: 7 source tensors + dst
संरचना uber_ssm_scan_params {
    संरचना ggml_tensor src0;
    संरचना ggml_tensor src1;
    संरचना ggml_tensor src2;
    संरचना ggml_tensor src3;
    संरचना ggml_tensor src4;
    संरचना ggml_tensor src5;
    संरचना ggml_tensor src6;
    संरचना ggml_tensor dst;
};

// gated_delta_net: q,k,v,g,beta,state_in,dst
संरचना uber_gated_delta_net_params {
    संरचना ggml_tensor q;
    संरचना ggml_tensor k;
    संरचना ggml_tensor v;
    संरचना ggml_tensor g;
    संरचना ggml_tensor beta;
    संरचना ggml_tensor state_in;
    संरचना ggml_tensor dst;
};

स्थिर शून्य copy_f32_to_f16_row(uint16_t * dst, स्थिरांक दशांश * src, int64_t अंक_elements) {
    साठी (int64_t i = 0; i < अंक_elements; i++) {
        dst[i] = fp32_to_fp16(src[i]);
    }
}

स्थिर शून्य copy_f32_row(दशांश * dst, स्थिरांक दशांश * src, int64_t अंक_elements) {
    साठी (int64_t i = 0; i < अंक_elements; i++) {
        dst[i] = src[i];
    }
}

स्थिर शून्य evict_region_past_l2_local(स्थिरांक शून्य * addr, माप_प्रकार bytes) {
    जर (!addr || bytes == 0) {
        परत;
    }

    स्थिरांक uint64_t CL     = 64;
    uint64_t       base   = (uint64_t) addr & ~(CL - 1);
    uint64_t       end    = ((uint64_t) addr + bytes + CL - 1) & ~(CL - 1);
    uint64_t       nlines = (end - base) / CL;
    cache_ops_priv_evict_sw(0, /*to_L2*/ 3, 0, 0, CL);
}

पूर्णांक entry_point(संरचना ggml_et_uberkernel_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    जर (!kernel_env || !params) {
        परत -1;
    }

    संरचना ggml_et_uberkernel_inst * insts       = (संरचना ggml_et_uberkernel_inst *) (uintptr_t) params->insts;
    uint8_t *                        params_blob = (uint8_t *) (uintptr_t) params->params_blob;

    जर (!insts || !params_blob || params->inst_stride < आकार(संरचना ggml_et_uberkernel_inst)) {
        परत -1;
    }

    साठी (uint32_t i = 0; i < params->अंक_insts; ++i) {
        संरचना ggml_et_uberkernel_inst * inst =
            (संरचना ggml_et_uberkernel_inst *) ((uint8_t *) insts + (i * params->inst_stride));
        शून्य * inst_params = params_blob + inst->params_offset;
        पूर्णांक    rc          = -1;

        et_barrier_global(32ULL);

        निवड (inst->kernel_id) {
            स्थिती GGML_ET_UBERKERNEL_KERNEL_EL_MAP_F32:
                {
                    संरचना ggml_et_द्विमान_params * p = (संरचना ggml_et_द्विमान_params *) inst_params;
                    rc                               = el_map_f32_entry(p, env);
                    थांबा;
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
            स्थिती GGML_ET_UBERKERNEL_KERNEL_GLU_F32:
                {
                    rc = glu_f32_entry((संरचना ggml_et_glu_params *) inst_params, env);
                    थांबा;
                }
            स्थिती GGML_ET_UBERKERNEL_KERNEL_ROPE_F32:
                {
                    rc = rope_f32_entry((संरचना ggml_et_rope_params *) inst_params, env);
                    थांबा;
                }
            स्थिती GGML_ET_UBERKERNEL_KERNEL_RMS_NORM_F32:
                {
                    // struct ggml_et_rms_norm_params *p = (struct ggml_et_rms_norm_params *) inst_params;
                    // evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
                    rc = rms_norm_f32_entry((संरचना ggml_et_rms_norm_params *) inst_params, env);
                    थांबा;
                }
            स्थिती GGML_ET_UBERKERNEL_KERNEL_RMS_NORM_MUL_F32:
                {
                    संरचना uber_rms_norm_mul_params * p = (संरचना uber_rms_norm_mul_params *) inst_params;
                    evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
                    evict_region_past_l2(p->src1.data, tensor_bytes(&p->src1));
                    rc = rms_norm_mul_f32_entry((संरचना ggml_et_rms_norm_mul_params *) inst_params, env);
                    थांबा;
                }
            स्थिती GGML_ET_UBERKERNEL_KERNEL_SOFTMAX_F32:
                {
                    rc = softmax_f32_entry((संरचना ggml_et_softmax_params *) inst_params, env);
                    थांबा;
                }
            स्थिती GGML_ET_UBERKERNEL_KERNEL_SET_ROWS_F32:
                {
                    rc = set_rows_f32_entry((संरचना ggml_et_set_rows_params *) inst_params, env);
                    थांबा;
                }

            // Single-source ops (src0 → dst)
            स्थिती GGML_ET_UBERKERNEL_KERNEL_SQR_F32:
                {
                    rc = sqr_f32_entry((संरचना ggml_et_sqr_params *) inst_params, env);
                    थांबा;
                }
            स्थिती GGML_ET_UBERKERNEL_KERNEL_SCALE_F32:
                {
                    rc = scale_f32_entry((संरचना ggml_et_scale_params *) inst_params, env);
                    थांबा;
                }
            स्थिती GGML_ET_UBERKERNEL_KERNEL_SUM_ROWS_F32:
                {
                    rc = sum_rows_f32_entry((संरचना ggml_et_sum_rows_params *) inst_params, env);
                    थांबा;
                }
            स्थिती GGML_ET_UBERKERNEL_KERNEL_CUMSUM_F32:
                {
                    rc = cumsum_f32_entry((संरचना ggml_et_cumsum_params *) inst_params, env);
                    थांबा;
                }
            स्थिती GGML_ET_UBERKERNEL_KERNEL_NORM_F32:
                {
                    rc = norm_f32_entry((संरचना ggml_et_norm_params *) inst_params, env);
                    थांबा;
                }
            स्थिती GGML_ET_UBERKERNEL_KERNEL_L2_NORM_F32:
                {
                    rc = l2_norm_f32_entry((संरचना ggml_et_l2_norm_params *) inst_params, env);
                    थांबा;
                }
            स्थिती GGML_ET_UBERKERNEL_KERNEL_GROUP_NORM_F32:
                {
                    rc = group_norm_f32_entry((संरचना ggml_et_group_norm_params *) inst_params, env);
                    थांबा;
                }
            स्थिती GGML_ET_UBERKERNEL_KERNEL_REPEAT_F32:
                {
                    rc = repeat_f32_entry((संरचना ggml_et_repeat_params *) inst_params, env);
                    थांबा;
                }
            स्थिती GGML_ET_UBERKERNEL_KERNEL_DIAG_F32:
                {
                    rc = diag_f32_entry((संरचना ggml_et_diag_params *) inst_params, env);
                    थांबा;
                }
            स्थिती GGML_ET_UBERKERNEL_KERNEL_TRI_F32:
                {
                    rc = tri_f32_entry((संरचना ggml_et_tri_params *) inst_params, env);
                    थांबा;
                }
            स्थिती GGML_ET_UBERKERNEL_KERNEL_PAD_F32:
                {
                    rc = pad_f32_entry((संरचना ggml_et_pad_params *) inst_params, env);
                    थांबा;
                }
            स्थिती GGML_ET_UBERKERNEL_KERNEL_CONT_F16:
                {
                    rc = cont_f16_entry((संरचना ggml_et_cont_params *) inst_params, env);
                    थांबा;
                }
            स्थिती GGML_ET_UBERKERNEL_KERNEL_FILL_F32:
                {
                    rc = fill_f32_entry((संरचना ggml_et_fill_params *) inst_params, env);
                    थांबा;
                }
            स्थिती GGML_ET_UBERKERNEL_KERNEL_SET_F32:
                {
                    rc = set_f32_entry((संरचना ggml_et_set_params *) inst_params, env);
                    थांबा;
                }

            // Two-source ops
            स्थिती GGML_ET_UBERKERNEL_KERNEL_CONCAT_F32:
                {
                    rc = concat_f32_entry((संरचना ggml_et_concat_params *) inst_params, env);
                    थांबा;
                }
            // case GGML_ET_UBERKERNEL_KERNEL_SSM_CONV_F32: {
            //     rc = ssm_conv_f32_entry((struct ggml_et_ssm_conv_params *) inst_params, env);
            //     break;
            // }
            स्थिती GGML_ET_UBERKERNEL_KERNEL_SOLVE_TRI_F32:
                {
                    rc = solve_tri_f32_entry((संरचना ggml_et_solve_tri_params *) inst_params, env);
                    थांबा;
                }
            स्थिती GGML_ET_UBERKERNEL_KERNEL_IM2COL:
                {
                    rc = im2col_entry((संरचना ggml_et_im2col_params *) inst_params, env);
                    थांबा;
                }

            // Three-source ops
            स्थिती GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_ID_F32:
                {
                    rc = mul_mat_id_f32_entry((संरचना ggml_et_mul_mat_id_params *) inst_params, env);
                    थांबा;
                }
            स्थिती GGML_ET_UBERKERNEL_KERNEL_FLASH_ATTN_EXT_F32:
                {
                    rc = flash_attn_ext_f32_entry((संरचना ggml_et_flash_attn_ext_params *) inst_params, env);
                    थांबा;
                }
            स्थिती GGML_ET_UBERKERNEL_KERNEL_FLASH_ATTN_EXT_F16_ME:
                {
                    rc = flash_attn_ext_f16_me_entry((संरचना ggml_et_flash_attn_ext_params *) inst_params, env);
                    थांबा;
                }

            स्थिती GGML_ET_UBERKERNEL_KERNEL_GATED_DELTA_NET_F32:
                {
                    rc = gated_delta_net_f32_entry((संरचना ggml_et_gated_delta_net_params *) inst_params, env);
                    थांबा;
                }
            स्थिती GGML_ET_UBERKERNEL_KERNEL_SSM_SCAN_F32:
                {
                    rc = ssm_scan_f32_entry((संरचना ggml_et_ssm_scan_params *) inst_params, env);
                    थांबा;
                }
            // rwkv: raw float* params, no ggml_tensor fields to evict via
            स्थिती GGML_ET_UBERKERNEL_KERNEL_RWKV_WKV6_F32:
                {
                    rc = rwkv_wkv6_f32_entry((संरचना ggml_et_rwkv_wkv6_params *) inst_params, env);
                    थांबा;
                }

            स्थिती GGML_ET_UBERKERNEL_KERNEL_RWKV_WKV7_F32:
                {
                    rc = rwkv_wkv7_f32_entry((संरचना ggml_et_rwkv_wkv7_params *) inst_params, env);
                    थांबा;
                }

            // MUL_MAT: evict src1 (activations); src0=weights is
            //  read-only so never stale from a prior uberkernel op
            स्थिती GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F16:
                {
                    संरचना ggml_et_द्विमान_params * p = (संरचना ggml_et_द्विमान_params *) inst_params;
                    rc                               = mul_mat_f16_entry(p, env);
                    थांबा;
                }
            स्थिती GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F16_MATRIX_ENGINE:
                {
                    संरचना ggml_et_द्विमान_params * p = (संरचना ggml_et_द्विमान_params *) inst_params;
                    rc                               = mul_mat_f16_matrix_engine_entry(p, env);
                    थांबा;
                }
            स्थिती GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F32:
                {
                    संरचना ggml_et_द्विमान_params * p = (संरचना ggml_et_द्विमान_params *) inst_params;
                    rc                               = mul_mat_f32_entry(p, env);
                    थांबा;
                }
            स्थिती GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F32_MATRIX_ENGINE:
                {
                    संरचना ggml_et_द्विमान_params * p = (संरचना ggml_et_द्विमान_params *) inst_params;
                    rc                               = mul_mat_f32_matrix_engine_entry(p, env);
                    थांबा;
                }
            स्थिती GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_Q8_0:
                {
                    संरचना ggml_et_mm_q8_params * p = (संरचना ggml_et_mm_q8_params *) inst_params;
                    // evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
                    rc                              = mul_mat_Q8_0_entry(p, env);
                    थांबा;
                }
            स्थिती GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_Q4_0:
                {
                    संरचना ggml_et_द्विमान_params * p = (संरचना ggml_et_द्विमान_params *) inst_params;
                    rc                               = mul_mat_Q4_0_entry(p, env);
                    थांबा;
                }

            पूर्वनिर्धारित:
                परत -1;
        }

        जर (rc != 0) {
            परत rc;
        }
    }

    परत 0;
}
