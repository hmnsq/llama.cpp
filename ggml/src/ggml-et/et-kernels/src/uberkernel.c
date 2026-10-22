#شامل "ggml-et-uberkernel-common.h"
#شامل "ggml-et-uberkernel-kernel-map.h"
#شامل "ggml_tensor.h"
#شامل "math_fp.h"
#شامل "platform.h"

#شامل <stdint.h>

ساختار ggml_et_glu_params;
ساختار ggml_et_unary_params;
ساختار ggml_et_rope_params;
ساختار ggml_et_rms_norm_params;
ساختار ggml_et_rms_norm_mul_params;
ساختار ggml_et_softmax_params;
ساختار ggml_et_set_rows_params;
ساختار ggml_et_get_rows_params;
ساختار ggml_et_cont_params;
ساختار ggml_et_concat_params;
ساختار ggml_et_cumsum_params;
ساختار ggml_et_diag_params;
ساختار ggml_et_fill_params;
ساختار ggml_et_flash_attn_ext_params;
ساختار ggml_et_gated_delta_net_params;
ساختار ggml_et_group_norm_params;
ساختار ggml_et_im2col_params;
ساختار ggml_et_l2_norm_params;
ساختار ggml_et_mul_mat_id_params;
ساختار ggml_et_norm_params;
ساختار ggml_et_pad_params;
ساختار ggml_et_repeat_params;
ساختار ggml_et_rwkv_wkv6_params;
ساختار ggml_et_rwkv_wkv7_params;
ساختار ggml_et_scale_params;
ساختار ggml_et_set_params;
ساختار ggml_et_solve_tri_params;
ساختار ggml_et_sqr_params;
ساختار ggml_et_ssm_conv_params;
ساختار ggml_et_ssm_scan_params;
ساختار ggml_et_sum_rows_params;
ساختار ggml_et_tri_params;

بیرونی صحیح el_map_f32_entry(ساختار ggml_et_دودویی_params *, تهی *);
بیرونی صحیح glu_f32_entry(ساختار ggml_et_glu_params *, تهی *);
بیرونی صحیح unary_f32_entry(ساختار ggml_et_unary_params *, تهی *);
بیرونی صحیح rope_f32_entry(ساختار ggml_et_rope_params *, تهی *);
بیرونی صحیح rms_norm_f32_entry(ساختار ggml_et_rms_norm_params *, تهی *);
بیرونی صحیح rms_norm_mul_f32_entry(ساختار ggml_et_rms_norm_mul_params *, تهی *);
بیرونی صحیح softmax_f32_entry(ساختار ggml_et_softmax_params *, تهی *);
بیرونی صحیح set_rows_f32_entry(ساختار ggml_et_set_rows_params *, تهی *);
بیرونی صحیح get_rows_f32_entry(ساختار ggml_et_get_rows_params *, تهی *);
بیرونی صحیح cont_f32_entry(ساختار ggml_et_cont_params *, تهی *);
بیرونی صحیح cont_f16_entry(ساختار ggml_et_cont_params *, تهی *);
بیرونی صحیح cpy_f32_f16_entry(ساختار ggml_et_cont_params *, تهی *);
بیرونی صحیح concat_f32_entry(ساختار ggml_et_concat_params *, تهی *);
بیرونی صحیح cumsum_f32_entry(ساختار ggml_et_cumsum_params *, تهی *);
بیرونی صحیح diag_f32_entry(ساختار ggml_et_diag_params *, تهی *);
بیرونی صحیح fill_f32_entry(ساختار ggml_et_fill_params *, تهی *);
بیرونی صحیح flash_attn_ext_f32_entry(ساختار ggml_et_flash_attn_ext_params *, تهی *);
بیرونی صحیح flash_attn_ext_f16_me_entry(ساختار ggml_et_flash_attn_ext_params *, تهی *);
بیرونی صحیح gated_delta_net_f32_entry(ساختار ggml_et_gated_delta_net_params *, تهی *);
بیرونی صحیح group_norm_f32_entry(ساختار ggml_et_group_norm_params *, تهی *);
بیرونی صحیح im2col_entry(ساختار ggml_et_im2col_params *, تهی *);
بیرونی صحیح l2_norm_f32_entry(ساختار ggml_et_l2_norm_params *, تهی *);
بیرونی صحیح mul_mat_id_f32_entry(ساختار ggml_et_mul_mat_id_params *, تهی *);
بیرونی صحیح norm_f32_entry(ساختار ggml_et_norm_params *, تهی *);
بیرونی صحیح pad_f32_entry(ساختار ggml_et_pad_params *, تهی *);
بیرونی صحیح repeat_f32_entry(ساختار ggml_et_repeat_params *, تهی *);
بیرونی صحیح rwkv_wkv6_f32_entry(ساختار ggml_et_rwkv_wkv6_params *, تهی *);
بیرونی صحیح rwkv_wkv7_f32_entry(ساختار ggml_et_rwkv_wkv7_params *, تهی *);
بیرونی صحیح scale_f32_entry(ساختار ggml_et_scale_params *, تهی *);
بیرونی صحیح set_f32_entry(ساختار ggml_et_set_params *, تهی *);
بیرونی صحیح solve_tri_f32_entry(ساختار ggml_et_solve_tri_params *, تهی *);
بیرونی صحیح sqr_f32_entry(ساختار ggml_et_sqr_params *, تهی *);
بیرونی صحیح ssm_conv_f32_entry(ساختار ggml_et_ssm_conv_params *, تهی *);
بیرونی صحیح ssm_scan_f32_entry(ساختار ggml_et_ssm_scan_params *, تهی *);
بیرونی صحیح sum_rows_f32_entry(ساختار ggml_et_sum_rows_params *, تهی *);
بیرونی صحیح tri_f32_entry(ساختار ggml_et_tri_params *, تهی *);
بیرونی صحیح mul_mat_f16_entry(ساختار ggml_et_دودویی_params *, تهی *);
بیرونی صحیح mul_mat_f16_matrix_engine_entry(ساختار ggml_et_دودویی_params *, تهی *);
بیرونی صحیح mul_mat_f32_entry(ساختار ggml_et_دودویی_params *, تهی *);
بیرونی صحیح mul_mat_f32_matrix_engine_entry(ساختار ggml_et_دودویی_params *, تهی *);
بیرونی صحیح mul_mat_Q8_0_entry(ساختار ggml_et_mm_q8_params *, تهی *);
بیرونی صحیح mul_mat_Q4_0_entry(ساختار ggml_et_دودویی_params *, تهی *);

ایستا درونخطی size_t tensor_bytes(ثابت ساختار ggml_tensor * t) {
    بازگشت (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

ساختار uber_glu_params {
    ساختار ggml_tensor src0;
    ساختار ggml_tensor src1;
    ساختار ggml_tensor dst;
    // trailing scalars omitted — not needed for eviction
};

ساختار uber_unary_params {
    ساختار ggml_tensor src0;
    ساختار ggml_tensor dst;
};

ساختار uber_rope_params {
    ساختار ggml_tensor src0;
    ساختار ggml_tensor src1;
    ساختار ggml_tensor src2;
    ساختار ggml_tensor dst;
};

ساختار uber_rms_norm_params {
    ساختار ggml_tensor src0;
    ساختار ggml_tensor dst;
};

ساختار uber_rms_norm_mul_params {
    ساختار ggml_tensor src0;
    ساختار ggml_tensor src1;
    ساختار ggml_tensor dst;
};

ساختار uber_softmax_params {
    ساختار ggml_tensor src0;
    ساختار ggml_tensor src1;
    ساختار ggml_tensor src2;
    ساختار ggml_tensor dst;
};

ساختار uber_set_rows_params {
    ساختار ggml_tensor src0;
    ساختار ggml_tensor src1;
    ساختار ggml_tensor dst;
};

ساختار uber_get_rows_params {
    ساختار ggml_tensor src0;
    ساختار ggml_tensor src1;
    ساختار ggml_tensor dst;
};

ساختار uber_cont_params {
    ساختار ggml_tensor src0;
    ساختار ggml_tensor dst;
};

// src0 + src1 + dst (no trailing scalars needed for eviction)
ساختار uber_concat_params {
    ساختار ggml_tensor src0;
    ساختار ggml_tensor src1;
    ساختار ggml_tensor dst;
};

ساختار uber_ssm_conv_params {
    ساختار ggml_tensor src0;
    ساختار ggml_tensor src1;
    ساختار ggml_tensor dst;
};

ساختار uber_solve_tri_params {
    ساختار ggml_tensor src0;
    ساختار ggml_tensor src1;
    ساختار ggml_tensor dst;
};

ساختار uber_mul_mat_id_params {
    ساختار ggml_tensor src0;
    ساختار ggml_tensor src1;
    ساختار ggml_tensor src2;
    ساختار ggml_tensor dst;
};

// flash_attn_ext: Q=src0, K=src1, V=src2, mask=src3, dst (mask optional)
ساختار uber_flash_attn_ext_params {
    ساختار ggml_tensor src0;
    ساختار ggml_tensor src1;
    ساختار ggml_tensor src2;
    ساختار ggml_tensor mask;
    ساختار ggml_tensor dst;
};

// ssm_scan: 7 source tensors + dst
ساختار uber_ssm_scan_params {
    ساختار ggml_tensor src0;
    ساختار ggml_tensor src1;
    ساختار ggml_tensor src2;
    ساختار ggml_tensor src3;
    ساختار ggml_tensor src4;
    ساختار ggml_tensor src5;
    ساختار ggml_tensor src6;
    ساختار ggml_tensor dst;
};

// gated_delta_net: q,k,v,g,beta,state_in,dst
ساختار uber_gated_delta_net_params {
    ساختار ggml_tensor q;
    ساختار ggml_tensor k;
    ساختار ggml_tensor v;
    ساختار ggml_tensor g;
    ساختار ggml_tensor beta;
    ساختار ggml_tensor state_in;
    ساختار ggml_tensor dst;
};

ایستا تهی copy_f32_to_f16_row(uint16_t * dst, ثابت اعشاری * src, int64_t رقم_elements) {
    برای (int64_t i = 0; i < رقم_elements; i++) {
        dst[i] = fp32_to_fp16(src[i]);
    }
}

ایستا تهی copy_f32_row(اعشاری * dst, ثابت اعشاری * src, int64_t رقم_elements) {
    برای (int64_t i = 0; i < رقم_elements; i++) {
        dst[i] = src[i];
    }
}

ایستا تهی evict_region_past_l2_local(ثابت تهی * addr, size_t bytes) {
    اگر (!addr || bytes == 0) {
        بازگشت;
    }

    ثابت uint64_t CL     = 64;
    uint64_t       base   = (uint64_t) addr & ~(CL - 1);
    uint64_t       end    = ((uint64_t) addr + bytes + CL - 1) & ~(CL - 1);
    uint64_t       nlines = (end - base) / CL;
    cache_ops_priv_evict_sw(0, /*to_L2*/ 3, 0, 0, CL);
}

صحیح entry_point(ساختار ggml_et_uberkernel_params * params, تهی * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    اگر (!kernel_env || !params) {
        بازگشت -1;
    }

    ساختار ggml_et_uberkernel_inst * insts       = (ساختار ggml_et_uberkernel_inst *) (uintptr_t) params->insts;
    uint8_t *                        params_blob = (uint8_t *) (uintptr_t) params->params_blob;

    اگر (!insts || !params_blob || params->inst_stride < اندازه(ساختار ggml_et_uberkernel_inst)) {
        بازگشت -1;
    }

    برای (uint32_t i = 0; i < params->رقم_insts; ++i) {
        ساختار ggml_et_uberkernel_inst * inst =
            (ساختار ggml_et_uberkernel_inst *) ((uint8_t *) insts + (i * params->inst_stride));
        تهی * inst_params = params_blob + inst->params_offset;
        صحیح    rc          = -1;

        et_barrier_global(32ULL);

        انتخاب (inst->kernel_id) {
            حالت GGML_ET_UBERKERNEL_KERNEL_EL_MAP_F32:
                {
                    ساختار ggml_et_دودویی_params * p = (ساختار ggml_et_دودویی_params *) inst_params;
                    rc                               = el_map_f32_entry(p, env);
                    بشکن;
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
            حالت GGML_ET_UBERKERNEL_KERNEL_GLU_F32:
                {
                    rc = glu_f32_entry((ساختار ggml_et_glu_params *) inst_params, env);
                    بشکن;
                }
            حالت GGML_ET_UBERKERNEL_KERNEL_ROPE_F32:
                {
                    rc = rope_f32_entry((ساختار ggml_et_rope_params *) inst_params, env);
                    بشکن;
                }
            حالت GGML_ET_UBERKERNEL_KERNEL_RMS_NORM_F32:
                {
                    // struct ggml_et_rms_norm_params *p = (struct ggml_et_rms_norm_params *) inst_params;
                    // evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
                    rc = rms_norm_f32_entry((ساختار ggml_et_rms_norm_params *) inst_params, env);
                    بشکن;
                }
            حالت GGML_ET_UBERKERNEL_KERNEL_RMS_NORM_MUL_F32:
                {
                    ساختار uber_rms_norm_mul_params * p = (ساختار uber_rms_norm_mul_params *) inst_params;
                    evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
                    evict_region_past_l2(p->src1.data, tensor_bytes(&p->src1));
                    rc = rms_norm_mul_f32_entry((ساختار ggml_et_rms_norm_mul_params *) inst_params, env);
                    بشکن;
                }
            حالت GGML_ET_UBERKERNEL_KERNEL_SOFTMAX_F32:
                {
                    rc = softmax_f32_entry((ساختار ggml_et_softmax_params *) inst_params, env);
                    بشکن;
                }
            حالت GGML_ET_UBERKERNEL_KERNEL_SET_ROWS_F32:
                {
                    rc = set_rows_f32_entry((ساختار ggml_et_set_rows_params *) inst_params, env);
                    بشکن;
                }

            // Single-source ops (src0 → dst)
            حالت GGML_ET_UBERKERNEL_KERNEL_SQR_F32:
                {
                    rc = sqr_f32_entry((ساختار ggml_et_sqr_params *) inst_params, env);
                    بشکن;
                }
            حالت GGML_ET_UBERKERNEL_KERNEL_SCALE_F32:
                {
                    rc = scale_f32_entry((ساختار ggml_et_scale_params *) inst_params, env);
                    بشکن;
                }
            حالت GGML_ET_UBERKERNEL_KERNEL_SUM_ROWS_F32:
                {
                    rc = sum_rows_f32_entry((ساختار ggml_et_sum_rows_params *) inst_params, env);
                    بشکن;
                }
            حالت GGML_ET_UBERKERNEL_KERNEL_CUMSUM_F32:
                {
                    rc = cumsum_f32_entry((ساختار ggml_et_cumsum_params *) inst_params, env);
                    بشکن;
                }
            حالت GGML_ET_UBERKERNEL_KERNEL_NORM_F32:
                {
                    rc = norm_f32_entry((ساختار ggml_et_norm_params *) inst_params, env);
                    بشکن;
                }
            حالت GGML_ET_UBERKERNEL_KERNEL_L2_NORM_F32:
                {
                    rc = l2_norm_f32_entry((ساختار ggml_et_l2_norm_params *) inst_params, env);
                    بشکن;
                }
            حالت GGML_ET_UBERKERNEL_KERNEL_GROUP_NORM_F32:
                {
                    rc = group_norm_f32_entry((ساختار ggml_et_group_norm_params *) inst_params, env);
                    بشکن;
                }
            حالت GGML_ET_UBERKERNEL_KERNEL_REPEAT_F32:
                {
                    rc = repeat_f32_entry((ساختار ggml_et_repeat_params *) inst_params, env);
                    بشکن;
                }
            حالت GGML_ET_UBERKERNEL_KERNEL_DIAG_F32:
                {
                    rc = diag_f32_entry((ساختار ggml_et_diag_params *) inst_params, env);
                    بشکن;
                }
            حالت GGML_ET_UBERKERNEL_KERNEL_TRI_F32:
                {
                    rc = tri_f32_entry((ساختار ggml_et_tri_params *) inst_params, env);
                    بشکن;
                }
            حالت GGML_ET_UBERKERNEL_KERNEL_PAD_F32:
                {
                    rc = pad_f32_entry((ساختار ggml_et_pad_params *) inst_params, env);
                    بشکن;
                }
            حالت GGML_ET_UBERKERNEL_KERNEL_CONT_F16:
                {
                    rc = cont_f16_entry((ساختار ggml_et_cont_params *) inst_params, env);
                    بشکن;
                }
            حالت GGML_ET_UBERKERNEL_KERNEL_FILL_F32:
                {
                    rc = fill_f32_entry((ساختار ggml_et_fill_params *) inst_params, env);
                    بشکن;
                }
            حالت GGML_ET_UBERKERNEL_KERNEL_SET_F32:
                {
                    rc = set_f32_entry((ساختار ggml_et_set_params *) inst_params, env);
                    بشکن;
                }

            // Two-source ops
            حالت GGML_ET_UBERKERNEL_KERNEL_CONCAT_F32:
                {
                    rc = concat_f32_entry((ساختار ggml_et_concat_params *) inst_params, env);
                    بشکن;
                }
            // case GGML_ET_UBERKERNEL_KERNEL_SSM_CONV_F32: {
            //     rc = ssm_conv_f32_entry((struct ggml_et_ssm_conv_params *) inst_params, env);
            //     break;
            // }
            حالت GGML_ET_UBERKERNEL_KERNEL_SOLVE_TRI_F32:
                {
                    rc = solve_tri_f32_entry((ساختار ggml_et_solve_tri_params *) inst_params, env);
                    بشکن;
                }
            حالت GGML_ET_UBERKERNEL_KERNEL_IM2COL:
                {
                    rc = im2col_entry((ساختار ggml_et_im2col_params *) inst_params, env);
                    بشکن;
                }

            // Three-source ops
            حالت GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_ID_F32:
                {
                    rc = mul_mat_id_f32_entry((ساختار ggml_et_mul_mat_id_params *) inst_params, env);
                    بشکن;
                }
            حالت GGML_ET_UBERKERNEL_KERNEL_FLASH_ATTN_EXT_F32:
                {
                    rc = flash_attn_ext_f32_entry((ساختار ggml_et_flash_attn_ext_params *) inst_params, env);
                    بشکن;
                }
            حالت GGML_ET_UBERKERNEL_KERNEL_FLASH_ATTN_EXT_F16_ME:
                {
                    rc = flash_attn_ext_f16_me_entry((ساختار ggml_et_flash_attn_ext_params *) inst_params, env);
                    بشکن;
                }

            حالت GGML_ET_UBERKERNEL_KERNEL_GATED_DELTA_NET_F32:
                {
                    rc = gated_delta_net_f32_entry((ساختار ggml_et_gated_delta_net_params *) inst_params, env);
                    بشکن;
                }
            حالت GGML_ET_UBERKERNEL_KERNEL_SSM_SCAN_F32:
                {
                    rc = ssm_scan_f32_entry((ساختار ggml_et_ssm_scan_params *) inst_params, env);
                    بشکن;
                }
            // rwkv: raw float* params, no ggml_tensor fields to evict via
            حالت GGML_ET_UBERKERNEL_KERNEL_RWKV_WKV6_F32:
                {
                    rc = rwkv_wkv6_f32_entry((ساختار ggml_et_rwkv_wkv6_params *) inst_params, env);
                    بشکن;
                }

            حالت GGML_ET_UBERKERNEL_KERNEL_RWKV_WKV7_F32:
                {
                    rc = rwkv_wkv7_f32_entry((ساختار ggml_et_rwkv_wkv7_params *) inst_params, env);
                    بشکن;
                }

            // MUL_MAT: evict src1 (activations); src0=weights is
            //  read-only so never stale from a prior uberkernel op
            حالت GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F16:
                {
                    ساختار ggml_et_دودویی_params * p = (ساختار ggml_et_دودویی_params *) inst_params;
                    rc                               = mul_mat_f16_entry(p, env);
                    بشکن;
                }
            حالت GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F16_MATRIX_ENGINE:
                {
                    ساختار ggml_et_دودویی_params * p = (ساختار ggml_et_دودویی_params *) inst_params;
                    rc                               = mul_mat_f16_matrix_engine_entry(p, env);
                    بشکن;
                }
            حالت GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F32:
                {
                    ساختار ggml_et_دودویی_params * p = (ساختار ggml_et_دودویی_params *) inst_params;
                    rc                               = mul_mat_f32_entry(p, env);
                    بشکن;
                }
            حالت GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F32_MATRIX_ENGINE:
                {
                    ساختار ggml_et_دودویی_params * p = (ساختار ggml_et_دودویی_params *) inst_params;
                    rc                               = mul_mat_f32_matrix_engine_entry(p, env);
                    بشکن;
                }
            حالت GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_Q8_0:
                {
                    ساختار ggml_et_mm_q8_params * p = (ساختار ggml_et_mm_q8_params *) inst_params;
                    // evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
                    rc                              = mul_mat_Q8_0_entry(p, env);
                    بشکن;
                }
            حالت GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_Q4_0:
                {
                    ساختار ggml_et_دودویی_params * p = (ساختار ggml_et_دودویی_params *) inst_params;
                    rc                               = mul_mat_Q4_0_entry(p, env);
                    بشکن;
                }

            پیشفرض:
                بازگشت -1;
        }

        اگر (rc != 0) {
            بازگشت rc;
        }
    }

    بازگشت 0;
}
