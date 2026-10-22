#અન્તર્ભાવ "ggml-et-uberkernel-common.h"
#અન્તર્ભાવ "ggml-et-uberkernel-kernel-map.h"
#અન્તર્ભાવ "ggml_tensor.h"
#અન્તર્ભાવ "math_fp.h"
#અન્તર્ભાવ "platform.h"

#અન્તર્ભાવ <stdint.h>

સંરચના ggml_et_glu_params;
સંરચના ggml_et_unary_params;
સંરચના ggml_et_rope_params;
સંરચના ggml_et_rms_norm_params;
સંરચના ggml_et_rms_norm_mul_params;
સંરચના ggml_et_softmax_params;
સંરચના ggml_et_set_rows_params;
સંરચના ggml_et_get_rows_params;
સંરચના ggml_et_cont_params;
સંરચના ggml_et_concat_params;
સંરચના ggml_et_cumsum_params;
સંરચના ggml_et_diag_params;
સંરચના ggml_et_fill_params;
સંરચના ggml_et_flash_attn_ext_params;
સંરચના ggml_et_gated_delta_net_params;
સંરચના ggml_et_group_norm_params;
સંરચના ggml_et_im2col_params;
સંરચના ggml_et_l2_norm_params;
સંરચના ggml_et_mul_mat_id_params;
સંરચના ggml_et_norm_params;
સંરચના ggml_et_pad_params;
સંરચના ggml_et_repeat_params;
સંરચના ggml_et_rwkv_wkv6_params;
સંરચના ggml_et_rwkv_wkv7_params;
સંરચના ggml_et_scale_params;
સંરચના ggml_et_set_params;
સંરચના ggml_et_solve_tri_params;
સંરચના ggml_et_sqr_params;
સંરચના ggml_et_ssm_conv_params;
સંરચના ggml_et_ssm_scan_params;
સંરચના ggml_et_sum_rows_params;
સંરચના ggml_et_tri_params;

બાહ્ય પૂર્ણાંક el_map_f32_entry(સંરચના ggml_et_દ્વિઅંકી_params *, શૂન્ય *);
બાહ્ય પૂર્ણાંક glu_f32_entry(સંરચના ggml_et_glu_params *, શૂન્ય *);
બાહ્ય પૂર્ણાંક unary_f32_entry(સંરચના ggml_et_unary_params *, શૂન્ય *);
બાહ્ય પૂર્ણાંક rope_f32_entry(સંરચના ggml_et_rope_params *, શૂન્ય *);
બાહ્ય પૂર્ણાંક rms_norm_f32_entry(સંરચના ggml_et_rms_norm_params *, શૂન્ય *);
બાહ્ય પૂર્ણાંક rms_norm_mul_f32_entry(સંરચના ggml_et_rms_norm_mul_params *, શૂન્ય *);
બાહ્ય પૂર્ણાંક softmax_f32_entry(સંરચના ggml_et_softmax_params *, શૂન્ય *);
બાહ્ય પૂર્ણાંક set_rows_f32_entry(સંરચના ggml_et_set_rows_params *, શૂન્ય *);
બાહ્ય પૂર્ણાંક get_rows_f32_entry(સંરચના ggml_et_get_rows_params *, શૂન્ય *);
બાહ્ય પૂર્ણાંક cont_f32_entry(સંરચના ggml_et_cont_params *, શૂન્ય *);
બાહ્ય પૂર્ણાંક cont_f16_entry(સંરચના ggml_et_cont_params *, શૂન્ય *);
બાહ્ય પૂર્ણાંક cpy_f32_f16_entry(સંરચના ggml_et_cont_params *, શૂન્ય *);
બાહ્ય પૂર્ણાંક concat_f32_entry(સંરચના ggml_et_concat_params *, શૂન્ય *);
બાહ્ય પૂર્ણાંક cumsum_f32_entry(સંરચના ggml_et_cumsum_params *, શૂન્ય *);
બાહ્ય પૂર્ણાંક diag_f32_entry(સંરચના ggml_et_diag_params *, શૂન્ય *);
બાહ્ય પૂર્ણાંક fill_f32_entry(સંરચના ggml_et_fill_params *, શૂન્ય *);
બાહ્ય પૂર્ણાંક flash_attn_ext_f32_entry(સંરચના ggml_et_flash_attn_ext_params *, શૂન્ય *);
બાહ્ય પૂર્ણાંક flash_attn_ext_f16_me_entry(સંરચના ggml_et_flash_attn_ext_params *, શૂન્ય *);
બાહ્ય પૂર્ણાંક gated_delta_net_f32_entry(સંરચના ggml_et_gated_delta_net_params *, શૂન્ય *);
બાહ્ય પૂર્ણાંક group_norm_f32_entry(સંરચના ggml_et_group_norm_params *, શૂન્ય *);
બાહ્ય પૂર્ણાંક im2col_entry(સંરચના ggml_et_im2col_params *, શૂન્ય *);
બાહ્ય પૂર્ણાંક l2_norm_f32_entry(સંરચના ggml_et_l2_norm_params *, શૂન્ય *);
બાહ્ય પૂર્ણાંક mul_mat_id_f32_entry(સંરચના ggml_et_mul_mat_id_params *, શૂન્ય *);
બાહ્ય પૂર્ણાંક norm_f32_entry(સંરચના ggml_et_norm_params *, શૂન્ય *);
બાહ્ય પૂર્ણાંક pad_f32_entry(સંરચના ggml_et_pad_params *, શૂન્ય *);
બાહ્ય પૂર્ણાંક repeat_f32_entry(સંરચના ggml_et_repeat_params *, શૂન્ય *);
બાહ્ય પૂર્ણાંક rwkv_wkv6_f32_entry(સંરચના ggml_et_rwkv_wkv6_params *, શૂન્ય *);
બાહ્ય પૂર્ણાંક rwkv_wkv7_f32_entry(સંરચના ggml_et_rwkv_wkv7_params *, શૂન્ય *);
બાહ્ય પૂર્ણાંક scale_f32_entry(સંરચના ggml_et_scale_params *, શૂન્ય *);
બાહ્ય પૂર્ણાંક set_f32_entry(સંરચના ggml_et_set_params *, શૂન્ય *);
બાહ્ય પૂર્ણાંક solve_tri_f32_entry(સંરચના ggml_et_solve_tri_params *, શૂન્ય *);
બાહ્ય પૂર્ણાંક sqr_f32_entry(સંરચના ggml_et_sqr_params *, શૂન્ય *);
બાહ્ય પૂર્ણાંક ssm_conv_f32_entry(સંરચના ggml_et_ssm_conv_params *, શૂન્ય *);
બાહ્ય પૂર્ણાંક ssm_scan_f32_entry(સંરચના ggml_et_ssm_scan_params *, શૂન્ય *);
બાહ્ય પૂર્ણાંક sum_rows_f32_entry(સંરચના ggml_et_sum_rows_params *, શૂન્ય *);
બાહ્ય પૂર્ણાંક tri_f32_entry(સંરચના ggml_et_tri_params *, શૂન્ય *);
બાહ્ય પૂર્ણાંક mul_mat_f16_entry(સંરચના ggml_et_દ્વિઅંકી_params *, શૂન્ય *);
બાહ્ય પૂર્ણાંક mul_mat_f16_matrix_engine_entry(સંરચના ggml_et_દ્વિઅંકી_params *, શૂન્ય *);
બાહ્ય પૂર્ણાંક mul_mat_f32_entry(સંરચના ggml_et_દ્વિઅંકી_params *, શૂન્ય *);
બાહ્ય પૂર્ણાંક mul_mat_f32_matrix_engine_entry(સંરચના ggml_et_દ્વિઅંકી_params *, શૂન્ય *);
બાહ્ય પૂર્ણાંક mul_mat_Q8_0_entry(સંરચના ggml_et_mm_q8_params *, શૂન્ય *);
બાહ્ય પૂર્ણાંક mul_mat_Q4_0_entry(સંરચના ggml_et_દ્વિઅંકી_params *, શૂન્ય *);

સ્થિર અંતઃરેખ size_t tensor_bytes(અચલ સંરચના ggml_tensor * t) {
    પાછા (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

સંરચના uber_glu_params {
    સંરચના ggml_tensor src0;
    સંરચના ggml_tensor src1;
    સંરચના ggml_tensor dst;
    // trailing scalars omitted — not needed for eviction
};

સંરચના uber_unary_params {
    સંરચના ggml_tensor src0;
    સંરચના ggml_tensor dst;
};

સંરચના uber_rope_params {
    સંરચના ggml_tensor src0;
    સંરચના ggml_tensor src1;
    સંરચના ggml_tensor src2;
    સંરચના ggml_tensor dst;
};

સંરચના uber_rms_norm_params {
    સંરચના ggml_tensor src0;
    સંરચના ggml_tensor dst;
};

સંરચના uber_rms_norm_mul_params {
    સંરચના ggml_tensor src0;
    સંરચના ggml_tensor src1;
    સંરચના ggml_tensor dst;
};

સંરચના uber_softmax_params {
    સંરચના ggml_tensor src0;
    સંરચના ggml_tensor src1;
    સંરચના ggml_tensor src2;
    સંરચના ggml_tensor dst;
};

સંરચના uber_set_rows_params {
    સંરચના ggml_tensor src0;
    સંરચના ggml_tensor src1;
    સંરચના ggml_tensor dst;
};

સંરચના uber_get_rows_params {
    સંરચના ggml_tensor src0;
    સંરચના ggml_tensor src1;
    સંરચના ggml_tensor dst;
};

સંરચના uber_cont_params {
    સંરચના ggml_tensor src0;
    સંરચના ggml_tensor dst;
};

// src0 + src1 + dst (no trailing scalars needed for eviction)
સંરચના uber_concat_params {
    સંરચના ggml_tensor src0;
    સંરચના ggml_tensor src1;
    સંરચના ggml_tensor dst;
};

સંરચના uber_ssm_conv_params {
    સંરચના ggml_tensor src0;
    સંરચના ggml_tensor src1;
    સંરચના ggml_tensor dst;
};

સંરચના uber_solve_tri_params {
    સંરચના ggml_tensor src0;
    સંરચના ggml_tensor src1;
    સંરચના ggml_tensor dst;
};

સંરચના uber_mul_mat_id_params {
    સંરચના ggml_tensor src0;
    સંરચના ggml_tensor src1;
    સંરચના ggml_tensor src2;
    સંરચના ggml_tensor dst;
};

// flash_attn_ext: Q=src0, K=src1, V=src2, mask=src3, dst (mask optional)
સંરચના uber_flash_attn_ext_params {
    સંરચના ggml_tensor src0;
    સંરચના ggml_tensor src1;
    સંરચના ggml_tensor src2;
    સંરચના ggml_tensor mask;
    સંરચના ggml_tensor dst;
};

// ssm_scan: 7 source tensors + dst
સંરચના uber_ssm_scan_params {
    સંરચના ggml_tensor src0;
    સંરચના ggml_tensor src1;
    સંરચના ggml_tensor src2;
    સંરચના ggml_tensor src3;
    સંરચના ggml_tensor src4;
    સંરચના ggml_tensor src5;
    સંરચના ggml_tensor src6;
    સંરચના ggml_tensor dst;
};

// gated_delta_net: q,k,v,g,beta,state_in,dst
સંરચના uber_gated_delta_net_params {
    સંરચના ggml_tensor q;
    સંરચના ggml_tensor k;
    સંરચના ggml_tensor v;
    સંરચના ggml_tensor g;
    સંરચના ggml_tensor beta;
    સંરચના ggml_tensor state_in;
    સંરચના ggml_tensor dst;
};

સ્થિર શૂન્ય copy_f32_to_f16_row(uint16_t * dst, અચલ દશાંશ * src, int64_t અંક_elements) {
    માટે (int64_t i = 0; i < અંક_elements; i++) {
        dst[i] = fp32_to_fp16(src[i]);
    }
}

સ્થિર શૂન્ય copy_f32_row(દશાંશ * dst, અચલ દશાંશ * src, int64_t અંક_elements) {
    માટે (int64_t i = 0; i < અંક_elements; i++) {
        dst[i] = src[i];
    }
}

સ્થિર શૂન્ય evict_region_past_l2_local(અચલ શૂન્ય * addr, size_t bytes) {
    જો (!addr || bytes == 0) {
        પાછા;
    }

    અચલ uint64_t CL     = 64;
    uint64_t       base   = (uint64_t) addr & ~(CL - 1);
    uint64_t       end    = ((uint64_t) addr + bytes + CL - 1) & ~(CL - 1);
    uint64_t       nlines = (end - base) / CL;
    cache_ops_priv_evict_sw(0, /*to_L2*/ 3, 0, 0, CL);
}

પૂર્ણાંક entry_point(સંરચના ggml_et_uberkernel_params * params, શૂન્ય * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    જો (!kernel_env || !params) {
        પાછા -1;
    }

    સંરચના ggml_et_uberkernel_inst * insts       = (સંરચના ggml_et_uberkernel_inst *) (uintptr_t) params->insts;
    uint8_t *                        params_blob = (uint8_t *) (uintptr_t) params->params_blob;

    જો (!insts || !params_blob || params->inst_stride < કદ(સંરચના ggml_et_uberkernel_inst)) {
        પાછા -1;
    }

    માટે (uint32_t i = 0; i < params->અંક_insts; ++i) {
        સંરચના ggml_et_uberkernel_inst * inst =
            (સંરચના ggml_et_uberkernel_inst *) ((uint8_t *) insts + (i * params->inst_stride));
        શૂન્ય * inst_params = params_blob + inst->params_offset;
        પૂર્ણાંક    rc          = -1;

        et_barrier_global(32ULL);

        પસંદગી (inst->kernel_id) {
            સ્થિતિ GGML_ET_UBERKERNEL_KERNEL_EL_MAP_F32:
                {
                    સંરચના ggml_et_દ્વિઅંકી_params * p = (સંરચના ggml_et_દ્વિઅંકી_params *) inst_params;
                    rc                               = el_map_f32_entry(p, env);
                    તોડો;
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
            સ્થિતિ GGML_ET_UBERKERNEL_KERNEL_GLU_F32:
                {
                    rc = glu_f32_entry((સંરચના ggml_et_glu_params *) inst_params, env);
                    તોડો;
                }
            સ્થિતિ GGML_ET_UBERKERNEL_KERNEL_ROPE_F32:
                {
                    rc = rope_f32_entry((સંરચના ggml_et_rope_params *) inst_params, env);
                    તોડો;
                }
            સ્થિતિ GGML_ET_UBERKERNEL_KERNEL_RMS_NORM_F32:
                {
                    // struct ggml_et_rms_norm_params *p = (struct ggml_et_rms_norm_params *) inst_params;
                    // evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
                    rc = rms_norm_f32_entry((સંરચના ggml_et_rms_norm_params *) inst_params, env);
                    તોડો;
                }
            સ્થિતિ GGML_ET_UBERKERNEL_KERNEL_RMS_NORM_MUL_F32:
                {
                    સંરચના uber_rms_norm_mul_params * p = (સંરચના uber_rms_norm_mul_params *) inst_params;
                    evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
                    evict_region_past_l2(p->src1.data, tensor_bytes(&p->src1));
                    rc = rms_norm_mul_f32_entry((સંરચના ggml_et_rms_norm_mul_params *) inst_params, env);
                    તોડો;
                }
            સ્થિતિ GGML_ET_UBERKERNEL_KERNEL_SOFTMAX_F32:
                {
                    rc = softmax_f32_entry((સંરચના ggml_et_softmax_params *) inst_params, env);
                    તોડો;
                }
            સ્થિતિ GGML_ET_UBERKERNEL_KERNEL_SET_ROWS_F32:
                {
                    rc = set_rows_f32_entry((સંરચના ggml_et_set_rows_params *) inst_params, env);
                    તોડો;
                }

            // Single-source ops (src0 → dst)
            સ્થિતિ GGML_ET_UBERKERNEL_KERNEL_SQR_F32:
                {
                    rc = sqr_f32_entry((સંરચના ggml_et_sqr_params *) inst_params, env);
                    તોડો;
                }
            સ્થિતિ GGML_ET_UBERKERNEL_KERNEL_SCALE_F32:
                {
                    rc = scale_f32_entry((સંરચના ggml_et_scale_params *) inst_params, env);
                    તોડો;
                }
            સ્થિતિ GGML_ET_UBERKERNEL_KERNEL_SUM_ROWS_F32:
                {
                    rc = sum_rows_f32_entry((સંરચના ggml_et_sum_rows_params *) inst_params, env);
                    તોડો;
                }
            સ્થિતિ GGML_ET_UBERKERNEL_KERNEL_CUMSUM_F32:
                {
                    rc = cumsum_f32_entry((સંરચના ggml_et_cumsum_params *) inst_params, env);
                    તોડો;
                }
            સ્થિતિ GGML_ET_UBERKERNEL_KERNEL_NORM_F32:
                {
                    rc = norm_f32_entry((સંરચના ggml_et_norm_params *) inst_params, env);
                    તોડો;
                }
            સ્થિતિ GGML_ET_UBERKERNEL_KERNEL_L2_NORM_F32:
                {
                    rc = l2_norm_f32_entry((સંરચના ggml_et_l2_norm_params *) inst_params, env);
                    તોડો;
                }
            સ્થિતિ GGML_ET_UBERKERNEL_KERNEL_GROUP_NORM_F32:
                {
                    rc = group_norm_f32_entry((સંરચના ggml_et_group_norm_params *) inst_params, env);
                    તોડો;
                }
            સ્થિતિ GGML_ET_UBERKERNEL_KERNEL_REPEAT_F32:
                {
                    rc = repeat_f32_entry((સંરચના ggml_et_repeat_params *) inst_params, env);
                    તોડો;
                }
            સ્થિતિ GGML_ET_UBERKERNEL_KERNEL_DIAG_F32:
                {
                    rc = diag_f32_entry((સંરચના ggml_et_diag_params *) inst_params, env);
                    તોડો;
                }
            સ્થિતિ GGML_ET_UBERKERNEL_KERNEL_TRI_F32:
                {
                    rc = tri_f32_entry((સંરચના ggml_et_tri_params *) inst_params, env);
                    તોડો;
                }
            સ્થિતિ GGML_ET_UBERKERNEL_KERNEL_PAD_F32:
                {
                    rc = pad_f32_entry((સંરચના ggml_et_pad_params *) inst_params, env);
                    તોડો;
                }
            સ્થિતિ GGML_ET_UBERKERNEL_KERNEL_CONT_F16:
                {
                    rc = cont_f16_entry((સંરચના ggml_et_cont_params *) inst_params, env);
                    તોડો;
                }
            સ્થિતિ GGML_ET_UBERKERNEL_KERNEL_FILL_F32:
                {
                    rc = fill_f32_entry((સંરચના ggml_et_fill_params *) inst_params, env);
                    તોડો;
                }
            સ્થિતિ GGML_ET_UBERKERNEL_KERNEL_SET_F32:
                {
                    rc = set_f32_entry((સંરચના ggml_et_set_params *) inst_params, env);
                    તોડો;
                }

            // Two-source ops
            સ્થિતિ GGML_ET_UBERKERNEL_KERNEL_CONCAT_F32:
                {
                    rc = concat_f32_entry((સંરચના ggml_et_concat_params *) inst_params, env);
                    તોડો;
                }
            // case GGML_ET_UBERKERNEL_KERNEL_SSM_CONV_F32: {
            //     rc = ssm_conv_f32_entry((struct ggml_et_ssm_conv_params *) inst_params, env);
            //     break;
            // }
            સ્થિતિ GGML_ET_UBERKERNEL_KERNEL_SOLVE_TRI_F32:
                {
                    rc = solve_tri_f32_entry((સંરચના ggml_et_solve_tri_params *) inst_params, env);
                    તોડો;
                }
            સ્થિતિ GGML_ET_UBERKERNEL_KERNEL_IM2COL:
                {
                    rc = im2col_entry((સંરચના ggml_et_im2col_params *) inst_params, env);
                    તોડો;
                }

            // Three-source ops
            સ્થિતિ GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_ID_F32:
                {
                    rc = mul_mat_id_f32_entry((સંરચના ggml_et_mul_mat_id_params *) inst_params, env);
                    તોડો;
                }
            સ્થિતિ GGML_ET_UBERKERNEL_KERNEL_FLASH_ATTN_EXT_F32:
                {
                    rc = flash_attn_ext_f32_entry((સંરચના ggml_et_flash_attn_ext_params *) inst_params, env);
                    તોડો;
                }
            સ્થિતિ GGML_ET_UBERKERNEL_KERNEL_FLASH_ATTN_EXT_F16_ME:
                {
                    rc = flash_attn_ext_f16_me_entry((સંરચના ggml_et_flash_attn_ext_params *) inst_params, env);
                    તોડો;
                }

            સ્થિતિ GGML_ET_UBERKERNEL_KERNEL_GATED_DELTA_NET_F32:
                {
                    rc = gated_delta_net_f32_entry((સંરચના ggml_et_gated_delta_net_params *) inst_params, env);
                    તોડો;
                }
            સ્થિતિ GGML_ET_UBERKERNEL_KERNEL_SSM_SCAN_F32:
                {
                    rc = ssm_scan_f32_entry((સંરચના ggml_et_ssm_scan_params *) inst_params, env);
                    તોડો;
                }
            // rwkv: raw float* params, no ggml_tensor fields to evict via
            સ્થિતિ GGML_ET_UBERKERNEL_KERNEL_RWKV_WKV6_F32:
                {
                    rc = rwkv_wkv6_f32_entry((સંરચના ggml_et_rwkv_wkv6_params *) inst_params, env);
                    તોડો;
                }

            સ્થિતિ GGML_ET_UBERKERNEL_KERNEL_RWKV_WKV7_F32:
                {
                    rc = rwkv_wkv7_f32_entry((સંરચના ggml_et_rwkv_wkv7_params *) inst_params, env);
                    તોડો;
                }

            // MUL_MAT: evict src1 (activations); src0=weights is
            //  read-only so never stale from a prior uberkernel op
            સ્થિતિ GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F16:
                {
                    સંરચના ggml_et_દ્વિઅંકી_params * p = (સંરચના ggml_et_દ્વિઅંકી_params *) inst_params;
                    rc                               = mul_mat_f16_entry(p, env);
                    તોડો;
                }
            સ્થિતિ GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F16_MATRIX_ENGINE:
                {
                    સંરચના ggml_et_દ્વિઅંકી_params * p = (સંરચના ggml_et_દ્વિઅંકી_params *) inst_params;
                    rc                               = mul_mat_f16_matrix_engine_entry(p, env);
                    તોડો;
                }
            સ્થિતિ GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F32:
                {
                    સંરચના ggml_et_દ્વિઅંકી_params * p = (સંરચના ggml_et_દ્વિઅંકી_params *) inst_params;
                    rc                               = mul_mat_f32_entry(p, env);
                    તોડો;
                }
            સ્થિતિ GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F32_MATRIX_ENGINE:
                {
                    સંરચના ggml_et_દ્વિઅંકી_params * p = (સંરચના ggml_et_દ્વિઅંકી_params *) inst_params;
                    rc                               = mul_mat_f32_matrix_engine_entry(p, env);
                    તોડો;
                }
            સ્થિતિ GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_Q8_0:
                {
                    સંરચના ggml_et_mm_q8_params * p = (સંરચના ggml_et_mm_q8_params *) inst_params;
                    // evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
                    rc                              = mul_mat_Q8_0_entry(p, env);
                    તોડો;
                }
            સ્થિતિ GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_Q4_0:
                {
                    સંરચના ggml_et_દ્વિઅંકી_params * p = (સંરચના ggml_et_દ્વિઅંકી_params *) inst_params;
                    rc                               = mul_mat_Q4_0_entry(p, env);
                    તોડો;
                }

            મૂળભૂત:
                પાછા -1;
        }

        જો (rc != 0) {
            પાછા rc;
        }
    }

    પાછા 0;
}
