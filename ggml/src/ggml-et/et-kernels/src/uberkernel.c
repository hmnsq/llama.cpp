#அந்தர்பாவ "ggml-et-uberkernel-common.h"
#அந்தர்பாவ "ggml-et-uberkernel-kernel-map.h"
#அந்தர்பாவ "ggml_tensor.h"
#அந்தர்பாவ "math_fp.h"
#அந்தர்பாவ "platform.h"

#அந்தர்பாவ <stdint.h>

கட்டமைப்பு ggml_et_glu_params;
கட்டமைப்பு ggml_et_unary_params;
கட்டமைப்பு ggml_et_rope_params;
கட்டமைப்பு ggml_et_rms_norm_params;
கட்டமைப்பு ggml_et_rms_norm_mul_params;
கட்டமைப்பு ggml_et_softmax_params;
கட்டமைப்பு ggml_et_set_rows_params;
கட்டமைப்பு ggml_et_get_rows_params;
கட்டமைப்பு ggml_et_cont_params;
கட்டமைப்பு ggml_et_concat_params;
கட்டமைப்பு ggml_et_cumsum_params;
கட்டமைப்பு ggml_et_diag_params;
கட்டமைப்பு ggml_et_fill_params;
கட்டமைப்பு ggml_et_flash_attn_ext_params;
கட்டமைப்பு ggml_et_gated_delta_net_params;
கட்டமைப்பு ggml_et_group_norm_params;
கட்டமைப்பு ggml_et_im2col_params;
கட்டமைப்பு ggml_et_l2_norm_params;
கட்டமைப்பு ggml_et_mul_mat_id_params;
கட்டமைப்பு ggml_et_norm_params;
கட்டமைப்பு ggml_et_pad_params;
கட்டமைப்பு ggml_et_repeat_params;
கட்டமைப்பு ggml_et_rwkv_wkv6_params;
கட்டமைப்பு ggml_et_rwkv_wkv7_params;
கட்டமைப்பு ggml_et_scale_params;
கட்டமைப்பு ggml_et_set_params;
கட்டமைப்பு ggml_et_solve_tri_params;
கட்டமைப்பு ggml_et_sqr_params;
கட்டமைப்பு ggml_et_ssm_conv_params;
கட்டமைப்பு ggml_et_ssm_scan_params;
கட்டமைப்பு ggml_et_sum_rows_params;
கட்டமைப்பு ggml_et_tri_params;

வெளி முழு el_map_f32_entry(கட்டமைப்பு ggml_et_இருமம்_params *, வெறுமை *);
வெளி முழு glu_f32_entry(கட்டமைப்பு ggml_et_glu_params *, வெறுமை *);
வெளி முழு unary_f32_entry(கட்டமைப்பு ggml_et_unary_params *, வெறுமை *);
வெளி முழு rope_f32_entry(கட்டமைப்பு ggml_et_rope_params *, வெறுமை *);
வெளி முழு rms_norm_f32_entry(கட்டமைப்பு ggml_et_rms_norm_params *, வெறுமை *);
வெளி முழு rms_norm_mul_f32_entry(கட்டமைப்பு ggml_et_rms_norm_mul_params *, வெறுமை *);
வெளி முழு softmax_f32_entry(கட்டமைப்பு ggml_et_softmax_params *, வெறுமை *);
வெளி முழு set_rows_f32_entry(கட்டமைப்பு ggml_et_set_rows_params *, வெறுமை *);
வெளி முழு get_rows_f32_entry(கட்டமைப்பு ggml_et_get_rows_params *, வெறுமை *);
வெளி முழு cont_f32_entry(கட்டமைப்பு ggml_et_cont_params *, வெறுமை *);
வெளி முழு cont_f16_entry(கட்டமைப்பு ggml_et_cont_params *, வெறுமை *);
வெளி முழு cpy_f32_f16_entry(கட்டமைப்பு ggml_et_cont_params *, வெறுமை *);
வெளி முழு concat_f32_entry(கட்டமைப்பு ggml_et_concat_params *, வெறுமை *);
வெளி முழு cumsum_f32_entry(கட்டமைப்பு ggml_et_cumsum_params *, வெறுமை *);
வெளி முழு diag_f32_entry(கட்டமைப்பு ggml_et_diag_params *, வெறுமை *);
வெளி முழு fill_f32_entry(கட்டமைப்பு ggml_et_fill_params *, வெறுமை *);
வெளி முழு flash_attn_ext_f32_entry(கட்டமைப்பு ggml_et_flash_attn_ext_params *, வெறுமை *);
வெளி முழு flash_attn_ext_f16_me_entry(கட்டமைப்பு ggml_et_flash_attn_ext_params *, வெறுமை *);
வெளி முழு gated_delta_net_f32_entry(கட்டமைப்பு ggml_et_gated_delta_net_params *, வெறுமை *);
வெளி முழு group_norm_f32_entry(கட்டமைப்பு ggml_et_group_norm_params *, வெறுமை *);
வெளி முழு im2col_entry(கட்டமைப்பு ggml_et_im2col_params *, வெறுமை *);
வெளி முழு l2_norm_f32_entry(கட்டமைப்பு ggml_et_l2_norm_params *, வெறுமை *);
வெளி முழு mul_mat_id_f32_entry(கட்டமைப்பு ggml_et_mul_mat_id_params *, வெறுமை *);
வெளி முழு norm_f32_entry(கட்டமைப்பு ggml_et_norm_params *, வெறுமை *);
வெளி முழு pad_f32_entry(கட்டமைப்பு ggml_et_pad_params *, வெறுமை *);
வெளி முழு repeat_f32_entry(கட்டமைப்பு ggml_et_repeat_params *, வெறுமை *);
வெளி முழு rwkv_wkv6_f32_entry(கட்டமைப்பு ggml_et_rwkv_wkv6_params *, வெறுமை *);
வெளி முழு rwkv_wkv7_f32_entry(கட்டமைப்பு ggml_et_rwkv_wkv7_params *, வெறுமை *);
வெளி முழு scale_f32_entry(கட்டமைப்பு ggml_et_scale_params *, வெறுமை *);
வெளி முழு set_f32_entry(கட்டமைப்பு ggml_et_set_params *, வெறுமை *);
வெளி முழு solve_tri_f32_entry(கட்டமைப்பு ggml_et_solve_tri_params *, வெறுமை *);
வெளி முழு sqr_f32_entry(கட்டமைப்பு ggml_et_sqr_params *, வெறுமை *);
வெளி முழு ssm_conv_f32_entry(கட்டமைப்பு ggml_et_ssm_conv_params *, வெறுமை *);
வெளி முழு ssm_scan_f32_entry(கட்டமைப்பு ggml_et_ssm_scan_params *, வெறுமை *);
வெளி முழு sum_rows_f32_entry(கட்டமைப்பு ggml_et_sum_rows_params *, வெறுமை *);
வெளி முழு tri_f32_entry(கட்டமைப்பு ggml_et_tri_params *, வெறுமை *);
வெளி முழு mul_mat_f16_entry(கட்டமைப்பு ggml_et_இருமம்_params *, வெறுமை *);
வெளி முழு mul_mat_f16_matrix_engine_entry(கட்டமைப்பு ggml_et_இருமம்_params *, வெறுமை *);
வெளி முழு mul_mat_f32_entry(கட்டமைப்பு ggml_et_இருமம்_params *, வெறுமை *);
வெளி முழு mul_mat_f32_matrix_engine_entry(கட்டமைப்பு ggml_et_இருமம்_params *, வெறுமை *);
வெளி முழு mul_mat_Q8_0_entry(கட்டமைப்பு ggml_et_mm_q8_params *, வெறுமை *);
வெளி முழு mul_mat_Q4_0_entry(கட்டமைப்பு ggml_et_இருமம்_params *, வெறுமை *);

நிலையான உள்வரி size_t tensor_bytes(மாறிலி கட்டமைப்பு ggml_tensor * t) {
    திரும்பு (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

கட்டமைப்பு uber_glu_params {
    கட்டமைப்பு ggml_tensor src0;
    கட்டமைப்பு ggml_tensor src1;
    கட்டமைப்பு ggml_tensor dst;
    // trailing scalars omitted — not needed for eviction
};

கட்டமைப்பு uber_unary_params {
    கட்டமைப்பு ggml_tensor src0;
    கட்டமைப்பு ggml_tensor dst;
};

கட்டமைப்பு uber_rope_params {
    கட்டமைப்பு ggml_tensor src0;
    கட்டமைப்பு ggml_tensor src1;
    கட்டமைப்பு ggml_tensor src2;
    கட்டமைப்பு ggml_tensor dst;
};

கட்டமைப்பு uber_rms_norm_params {
    கட்டமைப்பு ggml_tensor src0;
    கட்டமைப்பு ggml_tensor dst;
};

கட்டமைப்பு uber_rms_norm_mul_params {
    கட்டமைப்பு ggml_tensor src0;
    கட்டமைப்பு ggml_tensor src1;
    கட்டமைப்பு ggml_tensor dst;
};

கட்டமைப்பு uber_softmax_params {
    கட்டமைப்பு ggml_tensor src0;
    கட்டமைப்பு ggml_tensor src1;
    கட்டமைப்பு ggml_tensor src2;
    கட்டமைப்பு ggml_tensor dst;
};

கட்டமைப்பு uber_set_rows_params {
    கட்டமைப்பு ggml_tensor src0;
    கட்டமைப்பு ggml_tensor src1;
    கட்டமைப்பு ggml_tensor dst;
};

கட்டமைப்பு uber_get_rows_params {
    கட்டமைப்பு ggml_tensor src0;
    கட்டமைப்பு ggml_tensor src1;
    கட்டமைப்பு ggml_tensor dst;
};

கட்டமைப்பு uber_cont_params {
    கட்டமைப்பு ggml_tensor src0;
    கட்டமைப்பு ggml_tensor dst;
};

// src0 + src1 + dst (no trailing scalars needed for eviction)
கட்டமைப்பு uber_concat_params {
    கட்டமைப்பு ggml_tensor src0;
    கட்டமைப்பு ggml_tensor src1;
    கட்டமைப்பு ggml_tensor dst;
};

கட்டமைப்பு uber_ssm_conv_params {
    கட்டமைப்பு ggml_tensor src0;
    கட்டமைப்பு ggml_tensor src1;
    கட்டமைப்பு ggml_tensor dst;
};

கட்டமைப்பு uber_solve_tri_params {
    கட்டமைப்பு ggml_tensor src0;
    கட்டமைப்பு ggml_tensor src1;
    கட்டமைப்பு ggml_tensor dst;
};

கட்டமைப்பு uber_mul_mat_id_params {
    கட்டமைப்பு ggml_tensor src0;
    கட்டமைப்பு ggml_tensor src1;
    கட்டமைப்பு ggml_tensor src2;
    கட்டமைப்பு ggml_tensor dst;
};

// flash_attn_ext: Q=src0, K=src1, V=src2, mask=src3, dst (mask optional)
கட்டமைப்பு uber_flash_attn_ext_params {
    கட்டமைப்பு ggml_tensor src0;
    கட்டமைப்பு ggml_tensor src1;
    கட்டமைப்பு ggml_tensor src2;
    கட்டமைப்பு ggml_tensor mask;
    கட்டமைப்பு ggml_tensor dst;
};

// ssm_scan: 7 source tensors + dst
கட்டமைப்பு uber_ssm_scan_params {
    கட்டமைப்பு ggml_tensor src0;
    கட்டமைப்பு ggml_tensor src1;
    கட்டமைப்பு ggml_tensor src2;
    கட்டமைப்பு ggml_tensor src3;
    கட்டமைப்பு ggml_tensor src4;
    கட்டமைப்பு ggml_tensor src5;
    கட்டமைப்பு ggml_tensor src6;
    கட்டமைப்பு ggml_tensor dst;
};

// gated_delta_net: q,k,v,g,beta,state_in,dst
கட்டமைப்பு uber_gated_delta_net_params {
    கட்டமைப்பு ggml_tensor q;
    கட்டமைப்பு ggml_tensor k;
    கட்டமைப்பு ggml_tensor v;
    கட்டமைப்பு ggml_tensor g;
    கட்டமைப்பு ggml_tensor beta;
    கட்டமைப்பு ggml_tensor state_in;
    கட்டமைப்பு ggml_tensor dst;
};

நிலையான வெறுமை copy_f32_to_f16_row(uint16_t * dst, மாறிலி தசம * src, int64_t இலக்கம்_elements) {
    க்கு (int64_t i = 0; i < இலக்கம்_elements; i++) {
        dst[i] = fp32_to_fp16(src[i]);
    }
}

நிலையான வெறுமை copy_f32_row(தசம * dst, மாறிலி தசம * src, int64_t இலக்கம்_elements) {
    க்கு (int64_t i = 0; i < இலக்கம்_elements; i++) {
        dst[i] = src[i];
    }
}

நிலையான வெறுமை evict_region_past_l2_local(மாறிலி வெறுமை * addr, size_t bytes) {
    எனில் (!addr || bytes == 0) {
        திரும்பு;
    }

    மாறிலி uint64_t CL     = 64;
    uint64_t       base   = (uint64_t) addr & ~(CL - 1);
    uint64_t       end    = ((uint64_t) addr + bytes + CL - 1) & ~(CL - 1);
    uint64_t       nlines = (end - base) / CL;
    cache_ops_priv_evict_sw(0, /*to_L2*/ 3, 0, 0, CL);
}

முழு entry_point(கட்டமைப்பு ggml_et_uberkernel_params * params, வெறுமை * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    எனில் (!kernel_env || !params) {
        திரும்பு -1;
    }

    கட்டமைப்பு ggml_et_uberkernel_inst * insts       = (கட்டமைப்பு ggml_et_uberkernel_inst *) (uintptr_t) params->insts;
    uint8_t *                        params_blob = (uint8_t *) (uintptr_t) params->params_blob;

    எனில் (!insts || !params_blob || params->inst_stride < அளவு(கட்டமைப்பு ggml_et_uberkernel_inst)) {
        திரும்பு -1;
    }

    க்கு (uint32_t i = 0; i < params->இலக்கம்_insts; ++i) {
        கட்டமைப்பு ggml_et_uberkernel_inst * inst =
            (கட்டமைப்பு ggml_et_uberkernel_inst *) ((uint8_t *) insts + (i * params->inst_stride));
        வெறுமை * inst_params = params_blob + inst->params_offset;
        முழு    rc          = -1;

        et_barrier_global(32ULL);

        தேர்வு (inst->kernel_id) {
            நிலை GGML_ET_UBERKERNEL_KERNEL_EL_MAP_F32:
                {
                    கட்டமைப்பு ggml_et_இருமம்_params * p = (கட்டமைப்பு ggml_et_இருமம்_params *) inst_params;
                    rc                               = el_map_f32_entry(p, env);
                    நிறுத்து;
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
            நிலை GGML_ET_UBERKERNEL_KERNEL_GLU_F32:
                {
                    rc = glu_f32_entry((கட்டமைப்பு ggml_et_glu_params *) inst_params, env);
                    நிறுத்து;
                }
            நிலை GGML_ET_UBERKERNEL_KERNEL_ROPE_F32:
                {
                    rc = rope_f32_entry((கட்டமைப்பு ggml_et_rope_params *) inst_params, env);
                    நிறுத்து;
                }
            நிலை GGML_ET_UBERKERNEL_KERNEL_RMS_NORM_F32:
                {
                    // struct ggml_et_rms_norm_params *p = (struct ggml_et_rms_norm_params *) inst_params;
                    // evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
                    rc = rms_norm_f32_entry((கட்டமைப்பு ggml_et_rms_norm_params *) inst_params, env);
                    நிறுத்து;
                }
            நிலை GGML_ET_UBERKERNEL_KERNEL_RMS_NORM_MUL_F32:
                {
                    கட்டமைப்பு uber_rms_norm_mul_params * p = (கட்டமைப்பு uber_rms_norm_mul_params *) inst_params;
                    evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
                    evict_region_past_l2(p->src1.data, tensor_bytes(&p->src1));
                    rc = rms_norm_mul_f32_entry((கட்டமைப்பு ggml_et_rms_norm_mul_params *) inst_params, env);
                    நிறுத்து;
                }
            நிலை GGML_ET_UBERKERNEL_KERNEL_SOFTMAX_F32:
                {
                    rc = softmax_f32_entry((கட்டமைப்பு ggml_et_softmax_params *) inst_params, env);
                    நிறுத்து;
                }
            நிலை GGML_ET_UBERKERNEL_KERNEL_SET_ROWS_F32:
                {
                    rc = set_rows_f32_entry((கட்டமைப்பு ggml_et_set_rows_params *) inst_params, env);
                    நிறுத்து;
                }

            // Single-source ops (src0 → dst)
            நிலை GGML_ET_UBERKERNEL_KERNEL_SQR_F32:
                {
                    rc = sqr_f32_entry((கட்டமைப்பு ggml_et_sqr_params *) inst_params, env);
                    நிறுத்து;
                }
            நிலை GGML_ET_UBERKERNEL_KERNEL_SCALE_F32:
                {
                    rc = scale_f32_entry((கட்டமைப்பு ggml_et_scale_params *) inst_params, env);
                    நிறுத்து;
                }
            நிலை GGML_ET_UBERKERNEL_KERNEL_SUM_ROWS_F32:
                {
                    rc = sum_rows_f32_entry((கட்டமைப்பு ggml_et_sum_rows_params *) inst_params, env);
                    நிறுத்து;
                }
            நிலை GGML_ET_UBERKERNEL_KERNEL_CUMSUM_F32:
                {
                    rc = cumsum_f32_entry((கட்டமைப்பு ggml_et_cumsum_params *) inst_params, env);
                    நிறுத்து;
                }
            நிலை GGML_ET_UBERKERNEL_KERNEL_NORM_F32:
                {
                    rc = norm_f32_entry((கட்டமைப்பு ggml_et_norm_params *) inst_params, env);
                    நிறுத்து;
                }
            நிலை GGML_ET_UBERKERNEL_KERNEL_L2_NORM_F32:
                {
                    rc = l2_norm_f32_entry((கட்டமைப்பு ggml_et_l2_norm_params *) inst_params, env);
                    நிறுத்து;
                }
            நிலை GGML_ET_UBERKERNEL_KERNEL_GROUP_NORM_F32:
                {
                    rc = group_norm_f32_entry((கட்டமைப்பு ggml_et_group_norm_params *) inst_params, env);
                    நிறுத்து;
                }
            நிலை GGML_ET_UBERKERNEL_KERNEL_REPEAT_F32:
                {
                    rc = repeat_f32_entry((கட்டமைப்பு ggml_et_repeat_params *) inst_params, env);
                    நிறுத்து;
                }
            நிலை GGML_ET_UBERKERNEL_KERNEL_DIAG_F32:
                {
                    rc = diag_f32_entry((கட்டமைப்பு ggml_et_diag_params *) inst_params, env);
                    நிறுத்து;
                }
            நிலை GGML_ET_UBERKERNEL_KERNEL_TRI_F32:
                {
                    rc = tri_f32_entry((கட்டமைப்பு ggml_et_tri_params *) inst_params, env);
                    நிறுத்து;
                }
            நிலை GGML_ET_UBERKERNEL_KERNEL_PAD_F32:
                {
                    rc = pad_f32_entry((கட்டமைப்பு ggml_et_pad_params *) inst_params, env);
                    நிறுத்து;
                }
            நிலை GGML_ET_UBERKERNEL_KERNEL_CONT_F16:
                {
                    rc = cont_f16_entry((கட்டமைப்பு ggml_et_cont_params *) inst_params, env);
                    நிறுத்து;
                }
            நிலை GGML_ET_UBERKERNEL_KERNEL_FILL_F32:
                {
                    rc = fill_f32_entry((கட்டமைப்பு ggml_et_fill_params *) inst_params, env);
                    நிறுத்து;
                }
            நிலை GGML_ET_UBERKERNEL_KERNEL_SET_F32:
                {
                    rc = set_f32_entry((கட்டமைப்பு ggml_et_set_params *) inst_params, env);
                    நிறுத்து;
                }

            // Two-source ops
            நிலை GGML_ET_UBERKERNEL_KERNEL_CONCAT_F32:
                {
                    rc = concat_f32_entry((கட்டமைப்பு ggml_et_concat_params *) inst_params, env);
                    நிறுத்து;
                }
            // case GGML_ET_UBERKERNEL_KERNEL_SSM_CONV_F32: {
            //     rc = ssm_conv_f32_entry((struct ggml_et_ssm_conv_params *) inst_params, env);
            //     break;
            // }
            நிலை GGML_ET_UBERKERNEL_KERNEL_SOLVE_TRI_F32:
                {
                    rc = solve_tri_f32_entry((கட்டமைப்பு ggml_et_solve_tri_params *) inst_params, env);
                    நிறுத்து;
                }
            நிலை GGML_ET_UBERKERNEL_KERNEL_IM2COL:
                {
                    rc = im2col_entry((கட்டமைப்பு ggml_et_im2col_params *) inst_params, env);
                    நிறுத்து;
                }

            // Three-source ops
            நிலை GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_ID_F32:
                {
                    rc = mul_mat_id_f32_entry((கட்டமைப்பு ggml_et_mul_mat_id_params *) inst_params, env);
                    நிறுத்து;
                }
            நிலை GGML_ET_UBERKERNEL_KERNEL_FLASH_ATTN_EXT_F32:
                {
                    rc = flash_attn_ext_f32_entry((கட்டமைப்பு ggml_et_flash_attn_ext_params *) inst_params, env);
                    நிறுத்து;
                }
            நிலை GGML_ET_UBERKERNEL_KERNEL_FLASH_ATTN_EXT_F16_ME:
                {
                    rc = flash_attn_ext_f16_me_entry((கட்டமைப்பு ggml_et_flash_attn_ext_params *) inst_params, env);
                    நிறுத்து;
                }

            நிலை GGML_ET_UBERKERNEL_KERNEL_GATED_DELTA_NET_F32:
                {
                    rc = gated_delta_net_f32_entry((கட்டமைப்பு ggml_et_gated_delta_net_params *) inst_params, env);
                    நிறுத்து;
                }
            நிலை GGML_ET_UBERKERNEL_KERNEL_SSM_SCAN_F32:
                {
                    rc = ssm_scan_f32_entry((கட்டமைப்பு ggml_et_ssm_scan_params *) inst_params, env);
                    நிறுத்து;
                }
            // rwkv: raw float* params, no ggml_tensor fields to evict via
            நிலை GGML_ET_UBERKERNEL_KERNEL_RWKV_WKV6_F32:
                {
                    rc = rwkv_wkv6_f32_entry((கட்டமைப்பு ggml_et_rwkv_wkv6_params *) inst_params, env);
                    நிறுத்து;
                }

            நிலை GGML_ET_UBERKERNEL_KERNEL_RWKV_WKV7_F32:
                {
                    rc = rwkv_wkv7_f32_entry((கட்டமைப்பு ggml_et_rwkv_wkv7_params *) inst_params, env);
                    நிறுத்து;
                }

            // MUL_MAT: evict src1 (activations); src0=weights is
            //  read-only so never stale from a prior uberkernel op
            நிலை GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F16:
                {
                    கட்டமைப்பு ggml_et_இருமம்_params * p = (கட்டமைப்பு ggml_et_இருமம்_params *) inst_params;
                    rc                               = mul_mat_f16_entry(p, env);
                    நிறுத்து;
                }
            நிலை GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F16_MATRIX_ENGINE:
                {
                    கட்டமைப்பு ggml_et_இருமம்_params * p = (கட்டமைப்பு ggml_et_இருமம்_params *) inst_params;
                    rc                               = mul_mat_f16_matrix_engine_entry(p, env);
                    நிறுத்து;
                }
            நிலை GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F32:
                {
                    கட்டமைப்பு ggml_et_இருமம்_params * p = (கட்டமைப்பு ggml_et_இருமம்_params *) inst_params;
                    rc                               = mul_mat_f32_entry(p, env);
                    நிறுத்து;
                }
            நிலை GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F32_MATRIX_ENGINE:
                {
                    கட்டமைப்பு ggml_et_இருமம்_params * p = (கட்டமைப்பு ggml_et_இருமம்_params *) inst_params;
                    rc                               = mul_mat_f32_matrix_engine_entry(p, env);
                    நிறுத்து;
                }
            நிலை GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_Q8_0:
                {
                    கட்டமைப்பு ggml_et_mm_q8_params * p = (கட்டமைப்பு ggml_et_mm_q8_params *) inst_params;
                    // evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
                    rc                              = mul_mat_Q8_0_entry(p, env);
                    நிறுத்து;
                }
            நிலை GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_Q4_0:
                {
                    கட்டமைப்பு ggml_et_இருமம்_params * p = (கட்டமைப்பு ggml_et_இருமம்_params *) inst_params;
                    rc                               = mul_mat_Q4_0_entry(p, env);
                    நிறுத்து;
                }

            இயல்பு:
                திரும்பு -1;
        }

        எனில் (rc != 0) {
            திரும்பு rc;
        }
    }

    திரும்பு 0;
}
