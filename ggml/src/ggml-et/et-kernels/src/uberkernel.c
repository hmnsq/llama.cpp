#അന്തര്ഭാവ "ggml-et-uberkernel-common.h"
#അന്തര്ഭാവ "ggml-et-uberkernel-kernel-map.h"
#അന്തര്ഭാവ "ggml_tensor.h"
#അന്തര്ഭാവ "math_fp.h"
#അന്തര്ഭാവ "platform.h"

#അന്തര്ഭാവ <stdint.h>

ഘടന ggml_et_glu_params;
ഘടന ggml_et_unary_params;
ഘടന ggml_et_rope_params;
ഘടന ggml_et_rms_norm_params;
ഘടന ggml_et_rms_norm_mul_params;
ഘടന ggml_et_softmax_params;
ഘടന ggml_et_set_rows_params;
ഘടന ggml_et_get_rows_params;
ഘടന ggml_et_cont_params;
ഘടന ggml_et_concat_params;
ഘടന ggml_et_cumsum_params;
ഘടന ggml_et_diag_params;
ഘടന ggml_et_fill_params;
ഘടന ggml_et_flash_attn_ext_params;
ഘടന ggml_et_gated_delta_net_params;
ഘടന ggml_et_group_norm_params;
ഘടന ggml_et_im2col_params;
ഘടന ggml_et_l2_norm_params;
ഘടന ggml_et_mul_mat_id_params;
ഘടന ggml_et_norm_params;
ഘടന ggml_et_pad_params;
ഘടന ggml_et_repeat_params;
ഘടന ggml_et_rwkv_wkv6_params;
ഘടന ggml_et_rwkv_wkv7_params;
ഘടന ggml_et_scale_params;
ഘടന ggml_et_set_params;
ഘടന ggml_et_solve_tri_params;
ഘടന ggml_et_sqr_params;
ഘടന ggml_et_ssm_conv_params;
ഘടന ggml_et_ssm_scan_params;
ഘടന ggml_et_sum_rows_params;
ഘടന ggml_et_tri_params;

ബാഹ്യ പൂർണ്ണസംഖ്യ el_map_f32_entry(ഘടന ggml_et_ദ്വിമാന_params *, ശൂന്യം *);
ബാഹ്യ പൂർണ്ണസംഖ്യ glu_f32_entry(ഘടന ggml_et_glu_params *, ശൂന്യം *);
ബാഹ്യ പൂർണ്ണസംഖ്യ unary_f32_entry(ഘടന ggml_et_unary_params *, ശൂന്യം *);
ബാഹ്യ പൂർണ്ണസംഖ്യ rope_f32_entry(ഘടന ggml_et_rope_params *, ശൂന്യം *);
ബാഹ്യ പൂർണ്ണസംഖ്യ rms_norm_f32_entry(ഘടന ggml_et_rms_norm_params *, ശൂന്യം *);
ബാഹ്യ പൂർണ്ണസംഖ്യ rms_norm_mul_f32_entry(ഘടന ggml_et_rms_norm_mul_params *, ശൂന്യം *);
ബാഹ്യ പൂർണ്ണസംഖ്യ softmax_f32_entry(ഘടന ggml_et_softmax_params *, ശൂന്യം *);
ബാഹ്യ പൂർണ്ണസംഖ്യ set_rows_f32_entry(ഘടന ggml_et_set_rows_params *, ശൂന്യം *);
ബാഹ്യ പൂർണ്ണസംഖ്യ get_rows_f32_entry(ഘടന ggml_et_get_rows_params *, ശൂന്യം *);
ബാഹ്യ പൂർണ്ണസംഖ്യ cont_f32_entry(ഘടന ggml_et_cont_params *, ശൂന്യം *);
ബാഹ്യ പൂർണ്ണസംഖ്യ cont_f16_entry(ഘടന ggml_et_cont_params *, ശൂന്യം *);
ബാഹ്യ പൂർണ്ണസംഖ്യ cpy_f32_f16_entry(ഘടന ggml_et_cont_params *, ശൂന്യം *);
ബാഹ്യ പൂർണ്ണസംഖ്യ concat_f32_entry(ഘടന ggml_et_concat_params *, ശൂന്യം *);
ബാഹ്യ പൂർണ്ണസംഖ്യ cumsum_f32_entry(ഘടന ggml_et_cumsum_params *, ശൂന്യം *);
ബാഹ്യ പൂർണ്ണസംഖ്യ diag_f32_entry(ഘടന ggml_et_diag_params *, ശൂന്യം *);
ബാഹ്യ പൂർണ്ണസംഖ്യ fill_f32_entry(ഘടന ggml_et_fill_params *, ശൂന്യം *);
ബാഹ്യ പൂർണ്ണസംഖ്യ flash_attn_ext_f32_entry(ഘടന ggml_et_flash_attn_ext_params *, ശൂന്യം *);
ബാഹ്യ പൂർണ്ണസംഖ്യ flash_attn_ext_f16_me_entry(ഘടന ggml_et_flash_attn_ext_params *, ശൂന്യം *);
ബാഹ്യ പൂർണ്ണസംഖ്യ gated_delta_net_f32_entry(ഘടന ggml_et_gated_delta_net_params *, ശൂന്യം *);
ബാഹ്യ പൂർണ്ണസംഖ്യ group_norm_f32_entry(ഘടന ggml_et_group_norm_params *, ശൂന്യം *);
ബാഹ്യ പൂർണ്ണസംഖ്യ im2col_entry(ഘടന ggml_et_im2col_params *, ശൂന്യം *);
ബാഹ്യ പൂർണ്ണസംഖ്യ l2_norm_f32_entry(ഘടന ggml_et_l2_norm_params *, ശൂന്യം *);
ബാഹ്യ പൂർണ്ണസംഖ്യ mul_mat_id_f32_entry(ഘടന ggml_et_mul_mat_id_params *, ശൂന്യം *);
ബാഹ്യ പൂർണ്ണസംഖ്യ norm_f32_entry(ഘടന ggml_et_norm_params *, ശൂന്യം *);
ബാഹ്യ പൂർണ്ണസംഖ്യ pad_f32_entry(ഘടന ggml_et_pad_params *, ശൂന്യം *);
ബാഹ്യ പൂർണ്ണസംഖ്യ repeat_f32_entry(ഘടന ggml_et_repeat_params *, ശൂന്യം *);
ബാഹ്യ പൂർണ്ണസംഖ്യ rwkv_wkv6_f32_entry(ഘടന ggml_et_rwkv_wkv6_params *, ശൂന്യം *);
ബാഹ്യ പൂർണ്ണസംഖ്യ rwkv_wkv7_f32_entry(ഘടന ggml_et_rwkv_wkv7_params *, ശൂന്യം *);
ബാഹ്യ പൂർണ്ണസംഖ്യ scale_f32_entry(ഘടന ggml_et_scale_params *, ശൂന്യം *);
ബാഹ്യ പൂർണ്ണസംഖ്യ set_f32_entry(ഘടന ggml_et_set_params *, ശൂന്യം *);
ബാഹ്യ പൂർണ്ണസംഖ്യ solve_tri_f32_entry(ഘടന ggml_et_solve_tri_params *, ശൂന്യം *);
ബാഹ്യ പൂർണ്ണസംഖ്യ sqr_f32_entry(ഘടന ggml_et_sqr_params *, ശൂന്യം *);
ബാഹ്യ പൂർണ്ണസംഖ്യ ssm_conv_f32_entry(ഘടന ggml_et_ssm_conv_params *, ശൂന്യം *);
ബാഹ്യ പൂർണ്ണസംഖ്യ ssm_scan_f32_entry(ഘടന ggml_et_ssm_scan_params *, ശൂന്യം *);
ബാഹ്യ പൂർണ്ണസംഖ്യ sum_rows_f32_entry(ഘടന ggml_et_sum_rows_params *, ശൂന്യം *);
ബാഹ്യ പൂർണ്ണസംഖ്യ tri_f32_entry(ഘടന ggml_et_tri_params *, ശൂന്യം *);
ബാഹ്യ പൂർണ്ണസംഖ്യ mul_mat_f16_entry(ഘടന ggml_et_ദ്വിമാന_params *, ശൂന്യം *);
ബാഹ്യ പൂർണ്ണസംഖ്യ mul_mat_f16_matrix_engine_entry(ഘടന ggml_et_ദ്വിമാന_params *, ശൂന്യം *);
ബാഹ്യ പൂർണ്ണസംഖ്യ mul_mat_f32_entry(ഘടന ggml_et_ദ്വിമാന_params *, ശൂന്യം *);
ബാഹ്യ പൂർണ്ണസംഖ്യ mul_mat_f32_matrix_engine_entry(ഘടന ggml_et_ദ്വിമാന_params *, ശൂന്യം *);
ബാഹ്യ പൂർണ്ണസംഖ്യ mul_mat_Q8_0_entry(ഘടന ggml_et_mm_q8_params *, ശൂന്യം *);
ബാഹ്യ പൂർണ്ണസംഖ്യ mul_mat_Q4_0_entry(ഘടന ggml_et_ദ്വിമാന_params *, ശൂന്യം *);

സ്ഥിതം വരിയിൽ size_t tensor_bytes(സ്ഥിരം ഘടന ggml_tensor * t) {
    തിരികെ (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

ഘടന uber_glu_params {
    ഘടന ggml_tensor src0;
    ഘടന ggml_tensor src1;
    ഘടന ggml_tensor dst;
    // trailing scalars omitted — not needed for eviction
};

ഘടന uber_unary_params {
    ഘടന ggml_tensor src0;
    ഘടന ggml_tensor dst;
};

ഘടന uber_rope_params {
    ഘടന ggml_tensor src0;
    ഘടന ggml_tensor src1;
    ഘടന ggml_tensor src2;
    ഘടന ggml_tensor dst;
};

ഘടന uber_rms_norm_params {
    ഘടന ggml_tensor src0;
    ഘടന ggml_tensor dst;
};

ഘടന uber_rms_norm_mul_params {
    ഘടന ggml_tensor src0;
    ഘടന ggml_tensor src1;
    ഘടന ggml_tensor dst;
};

ഘടന uber_softmax_params {
    ഘടന ggml_tensor src0;
    ഘടന ggml_tensor src1;
    ഘടന ggml_tensor src2;
    ഘടന ggml_tensor dst;
};

ഘടന uber_set_rows_params {
    ഘടന ggml_tensor src0;
    ഘടന ggml_tensor src1;
    ഘടന ggml_tensor dst;
};

ഘടന uber_get_rows_params {
    ഘടന ggml_tensor src0;
    ഘടന ggml_tensor src1;
    ഘടന ggml_tensor dst;
};

ഘടന uber_cont_params {
    ഘടന ggml_tensor src0;
    ഘടന ggml_tensor dst;
};

// src0 + src1 + dst (no trailing scalars needed for eviction)
ഘടന uber_concat_params {
    ഘടന ggml_tensor src0;
    ഘടന ggml_tensor src1;
    ഘടന ggml_tensor dst;
};

ഘടന uber_ssm_conv_params {
    ഘടന ggml_tensor src0;
    ഘടന ggml_tensor src1;
    ഘടന ggml_tensor dst;
};

ഘടന uber_solve_tri_params {
    ഘടന ggml_tensor src0;
    ഘടന ggml_tensor src1;
    ഘടന ggml_tensor dst;
};

ഘടന uber_mul_mat_id_params {
    ഘടന ggml_tensor src0;
    ഘടന ggml_tensor src1;
    ഘടന ggml_tensor src2;
    ഘടന ggml_tensor dst;
};

// flash_attn_ext: Q=src0, K=src1, V=src2, mask=src3, dst (mask optional)
ഘടന uber_flash_attn_ext_params {
    ഘടന ggml_tensor src0;
    ഘടന ggml_tensor src1;
    ഘടന ggml_tensor src2;
    ഘടന ggml_tensor mask;
    ഘടന ggml_tensor dst;
};

// ssm_scan: 7 source tensors + dst
ഘടന uber_ssm_scan_params {
    ഘടന ggml_tensor src0;
    ഘടന ggml_tensor src1;
    ഘടന ggml_tensor src2;
    ഘടന ggml_tensor src3;
    ഘടന ggml_tensor src4;
    ഘടന ggml_tensor src5;
    ഘടന ggml_tensor src6;
    ഘടന ggml_tensor dst;
};

// gated_delta_net: q,k,v,g,beta,state_in,dst
ഘടന uber_gated_delta_net_params {
    ഘടന ggml_tensor q;
    ഘടന ggml_tensor k;
    ഘടന ggml_tensor v;
    ഘടന ggml_tensor g;
    ഘടന ggml_tensor beta;
    ഘടന ggml_tensor state_in;
    ഘടന ggml_tensor dst;
};

സ്ഥിതം ശൂന്യം copy_f32_to_f16_row(uint16_t * dst, സ്ഥിരം ദശാംശം * src, int64_t അക്കം_elements) {
    വേണ്ടി (int64_t i = 0; i < അക്കം_elements; i++) {
        dst[i] = fp32_to_fp16(src[i]);
    }
}

സ്ഥിതം ശൂന്യം copy_f32_row(ദശാംശം * dst, സ്ഥിരം ദശാംശം * src, int64_t അക്കം_elements) {
    വേണ്ടി (int64_t i = 0; i < അക്കം_elements; i++) {
        dst[i] = src[i];
    }
}

സ്ഥിതം ശൂന്യം evict_region_past_l2_local(സ്ഥിരം ശൂന്യം * addr, size_t bytes) {
    എങ്കിൽ (!addr || bytes == 0) {
        തിരികെ;
    }

    സ്ഥിരം uint64_t CL     = 64;
    uint64_t       base   = (uint64_t) addr & ~(CL - 1);
    uint64_t       end    = ((uint64_t) addr + bytes + CL - 1) & ~(CL - 1);
    uint64_t       nlines = (end - base) / CL;
    cache_ops_priv_evict_sw(0, /*to_L2*/ 3, 0, 0, CL);
}

പൂർണ്ണസംഖ്യ entry_point(ഘടന ggml_et_uberkernel_params * params, ശൂന്യം * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    എങ്കിൽ (!kernel_env || !params) {
        തിരികെ -1;
    }

    ഘടന ggml_et_uberkernel_inst * insts       = (ഘടന ggml_et_uberkernel_inst *) (uintptr_t) params->insts;
    uint8_t *                        params_blob = (uint8_t *) (uintptr_t) params->params_blob;

    എങ്കിൽ (!insts || !params_blob || params->inst_stride < വലുപ്പം(ഘടന ggml_et_uberkernel_inst)) {
        തിരികെ -1;
    }

    വേണ്ടി (uint32_t i = 0; i < params->അക്കം_insts; ++i) {
        ഘടന ggml_et_uberkernel_inst * inst =
            (ഘടന ggml_et_uberkernel_inst *) ((uint8_t *) insts + (i * params->inst_stride));
        ശൂന്യം * inst_params = params_blob + inst->params_offset;
        പൂർണ്ണസംഖ്യ    rc          = -1;

        et_barrier_global(32ULL);

        തിരഞ്ഞെടുപ്പ് (inst->kernel_id) {
            അവസ്ഥ GGML_ET_UBERKERNEL_KERNEL_EL_MAP_F32:
                {
                    ഘടന ggml_et_ദ്വിമാന_params * p = (ഘടന ggml_et_ദ്വിമാന_params *) inst_params;
                    rc                               = el_map_f32_entry(p, env);
                    നിർത്തുക;
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
            അവസ്ഥ GGML_ET_UBERKERNEL_KERNEL_GLU_F32:
                {
                    rc = glu_f32_entry((ഘടന ggml_et_glu_params *) inst_params, env);
                    നിർത്തുക;
                }
            അവസ്ഥ GGML_ET_UBERKERNEL_KERNEL_ROPE_F32:
                {
                    rc = rope_f32_entry((ഘടന ggml_et_rope_params *) inst_params, env);
                    നിർത്തുക;
                }
            അവസ്ഥ GGML_ET_UBERKERNEL_KERNEL_RMS_NORM_F32:
                {
                    // struct ggml_et_rms_norm_params *p = (struct ggml_et_rms_norm_params *) inst_params;
                    // evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
                    rc = rms_norm_f32_entry((ഘടന ggml_et_rms_norm_params *) inst_params, env);
                    നിർത്തുക;
                }
            അവസ്ഥ GGML_ET_UBERKERNEL_KERNEL_RMS_NORM_MUL_F32:
                {
                    ഘടന uber_rms_norm_mul_params * p = (ഘടന uber_rms_norm_mul_params *) inst_params;
                    evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
                    evict_region_past_l2(p->src1.data, tensor_bytes(&p->src1));
                    rc = rms_norm_mul_f32_entry((ഘടന ggml_et_rms_norm_mul_params *) inst_params, env);
                    നിർത്തുക;
                }
            അവസ്ഥ GGML_ET_UBERKERNEL_KERNEL_SOFTMAX_F32:
                {
                    rc = softmax_f32_entry((ഘടന ggml_et_softmax_params *) inst_params, env);
                    നിർത്തുക;
                }
            അവസ്ഥ GGML_ET_UBERKERNEL_KERNEL_SET_ROWS_F32:
                {
                    rc = set_rows_f32_entry((ഘടന ggml_et_set_rows_params *) inst_params, env);
                    നിർത്തുക;
                }

            // Single-source ops (src0 → dst)
            അവസ്ഥ GGML_ET_UBERKERNEL_KERNEL_SQR_F32:
                {
                    rc = sqr_f32_entry((ഘടന ggml_et_sqr_params *) inst_params, env);
                    നിർത്തുക;
                }
            അവസ്ഥ GGML_ET_UBERKERNEL_KERNEL_SCALE_F32:
                {
                    rc = scale_f32_entry((ഘടന ggml_et_scale_params *) inst_params, env);
                    നിർത്തുക;
                }
            അവസ്ഥ GGML_ET_UBERKERNEL_KERNEL_SUM_ROWS_F32:
                {
                    rc = sum_rows_f32_entry((ഘടന ggml_et_sum_rows_params *) inst_params, env);
                    നിർത്തുക;
                }
            അവസ്ഥ GGML_ET_UBERKERNEL_KERNEL_CUMSUM_F32:
                {
                    rc = cumsum_f32_entry((ഘടന ggml_et_cumsum_params *) inst_params, env);
                    നിർത്തുക;
                }
            അവസ്ഥ GGML_ET_UBERKERNEL_KERNEL_NORM_F32:
                {
                    rc = norm_f32_entry((ഘടന ggml_et_norm_params *) inst_params, env);
                    നിർത്തുക;
                }
            അവസ്ഥ GGML_ET_UBERKERNEL_KERNEL_L2_NORM_F32:
                {
                    rc = l2_norm_f32_entry((ഘടന ggml_et_l2_norm_params *) inst_params, env);
                    നിർത്തുക;
                }
            അവസ്ഥ GGML_ET_UBERKERNEL_KERNEL_GROUP_NORM_F32:
                {
                    rc = group_norm_f32_entry((ഘടന ggml_et_group_norm_params *) inst_params, env);
                    നിർത്തുക;
                }
            അവസ്ഥ GGML_ET_UBERKERNEL_KERNEL_REPEAT_F32:
                {
                    rc = repeat_f32_entry((ഘടന ggml_et_repeat_params *) inst_params, env);
                    നിർത്തുക;
                }
            അവസ്ഥ GGML_ET_UBERKERNEL_KERNEL_DIAG_F32:
                {
                    rc = diag_f32_entry((ഘടന ggml_et_diag_params *) inst_params, env);
                    നിർത്തുക;
                }
            അവസ്ഥ GGML_ET_UBERKERNEL_KERNEL_TRI_F32:
                {
                    rc = tri_f32_entry((ഘടന ggml_et_tri_params *) inst_params, env);
                    നിർത്തുക;
                }
            അവസ്ഥ GGML_ET_UBERKERNEL_KERNEL_PAD_F32:
                {
                    rc = pad_f32_entry((ഘടന ggml_et_pad_params *) inst_params, env);
                    നിർത്തുക;
                }
            അവസ്ഥ GGML_ET_UBERKERNEL_KERNEL_CONT_F16:
                {
                    rc = cont_f16_entry((ഘടന ggml_et_cont_params *) inst_params, env);
                    നിർത്തുക;
                }
            അവസ്ഥ GGML_ET_UBERKERNEL_KERNEL_FILL_F32:
                {
                    rc = fill_f32_entry((ഘടന ggml_et_fill_params *) inst_params, env);
                    നിർത്തുക;
                }
            അവസ്ഥ GGML_ET_UBERKERNEL_KERNEL_SET_F32:
                {
                    rc = set_f32_entry((ഘടന ggml_et_set_params *) inst_params, env);
                    നിർത്തുക;
                }

            // Two-source ops
            അവസ്ഥ GGML_ET_UBERKERNEL_KERNEL_CONCAT_F32:
                {
                    rc = concat_f32_entry((ഘടന ggml_et_concat_params *) inst_params, env);
                    നിർത്തുക;
                }
            // case GGML_ET_UBERKERNEL_KERNEL_SSM_CONV_F32: {
            //     rc = ssm_conv_f32_entry((struct ggml_et_ssm_conv_params *) inst_params, env);
            //     break;
            // }
            അവസ്ഥ GGML_ET_UBERKERNEL_KERNEL_SOLVE_TRI_F32:
                {
                    rc = solve_tri_f32_entry((ഘടന ggml_et_solve_tri_params *) inst_params, env);
                    നിർത്തുക;
                }
            അവസ്ഥ GGML_ET_UBERKERNEL_KERNEL_IM2COL:
                {
                    rc = im2col_entry((ഘടന ggml_et_im2col_params *) inst_params, env);
                    നിർത്തുക;
                }

            // Three-source ops
            അവസ്ഥ GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_ID_F32:
                {
                    rc = mul_mat_id_f32_entry((ഘടന ggml_et_mul_mat_id_params *) inst_params, env);
                    നിർത്തുക;
                }
            അവസ്ഥ GGML_ET_UBERKERNEL_KERNEL_FLASH_ATTN_EXT_F32:
                {
                    rc = flash_attn_ext_f32_entry((ഘടന ggml_et_flash_attn_ext_params *) inst_params, env);
                    നിർത്തുക;
                }
            അവസ്ഥ GGML_ET_UBERKERNEL_KERNEL_FLASH_ATTN_EXT_F16_ME:
                {
                    rc = flash_attn_ext_f16_me_entry((ഘടന ggml_et_flash_attn_ext_params *) inst_params, env);
                    നിർത്തുക;
                }

            അവസ്ഥ GGML_ET_UBERKERNEL_KERNEL_GATED_DELTA_NET_F32:
                {
                    rc = gated_delta_net_f32_entry((ഘടന ggml_et_gated_delta_net_params *) inst_params, env);
                    നിർത്തുക;
                }
            അവസ്ഥ GGML_ET_UBERKERNEL_KERNEL_SSM_SCAN_F32:
                {
                    rc = ssm_scan_f32_entry((ഘടന ggml_et_ssm_scan_params *) inst_params, env);
                    നിർത്തുക;
                }
            // rwkv: raw float* params, no ggml_tensor fields to evict via
            അവസ്ഥ GGML_ET_UBERKERNEL_KERNEL_RWKV_WKV6_F32:
                {
                    rc = rwkv_wkv6_f32_entry((ഘടന ggml_et_rwkv_wkv6_params *) inst_params, env);
                    നിർത്തുക;
                }

            അവസ്ഥ GGML_ET_UBERKERNEL_KERNEL_RWKV_WKV7_F32:
                {
                    rc = rwkv_wkv7_f32_entry((ഘടന ggml_et_rwkv_wkv7_params *) inst_params, env);
                    നിർത്തുക;
                }

            // MUL_MAT: evict src1 (activations); src0=weights is
            //  read-only so never stale from a prior uberkernel op
            അവസ്ഥ GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F16:
                {
                    ഘടന ggml_et_ദ്വിമാന_params * p = (ഘടന ggml_et_ദ്വിമാന_params *) inst_params;
                    rc                               = mul_mat_f16_entry(p, env);
                    നിർത്തുക;
                }
            അവസ്ഥ GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F16_MATRIX_ENGINE:
                {
                    ഘടന ggml_et_ദ്വിമാന_params * p = (ഘടന ggml_et_ദ്വിമാന_params *) inst_params;
                    rc                               = mul_mat_f16_matrix_engine_entry(p, env);
                    നിർത്തുക;
                }
            അവസ്ഥ GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F32:
                {
                    ഘടന ggml_et_ദ്വിമാന_params * p = (ഘടന ggml_et_ദ്വിമാന_params *) inst_params;
                    rc                               = mul_mat_f32_entry(p, env);
                    നിർത്തുക;
                }
            അവസ്ഥ GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F32_MATRIX_ENGINE:
                {
                    ഘടന ggml_et_ദ്വിമാന_params * p = (ഘടന ggml_et_ദ്വിമാന_params *) inst_params;
                    rc                               = mul_mat_f32_matrix_engine_entry(p, env);
                    നിർത്തുക;
                }
            അവസ്ഥ GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_Q8_0:
                {
                    ഘടന ggml_et_mm_q8_params * p = (ഘടന ggml_et_mm_q8_params *) inst_params;
                    // evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
                    rc                              = mul_mat_Q8_0_entry(p, env);
                    നിർത്തുക;
                }
            അവസ്ഥ GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_Q4_0:
                {
                    ഘടന ggml_et_ദ്വിമാന_params * p = (ഘടന ggml_et_ദ്വിമാന_params *) inst_params;
                    rc                               = mul_mat_Q4_0_entry(p, env);
                    നിർത്തുക;
                }

            സ്ഥിരസ്ഥിതി:
                തിരികെ -1;
        }

        എങ്കിൽ (rc != 0) {
            തിരികെ rc;
        }
    }

    തിരികെ 0;
}
