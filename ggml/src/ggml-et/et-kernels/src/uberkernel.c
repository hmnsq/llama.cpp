#ضمن "ggml-et-uberkernel-common.h"
#ضمن "ggml-et-uberkernel-kernel-map.h"
#ضمن "ggml_tensor.h"
#ضمن "math_fp.h"
#ضمن "platform.h"

#ضمن <stdint.h>

بنية ggml_et_glu_params;
بنية ggml_et_unary_params;
بنية ggml_et_rope_params;
بنية ggml_et_rms_norm_params;
بنية ggml_et_rms_norm_mul_params;
بنية ggml_et_softmax_params;
بنية ggml_et_set_rows_params;
بنية ggml_et_get_rows_params;
بنية ggml_et_cont_params;
بنية ggml_et_concat_params;
بنية ggml_et_cumsum_params;
بنية ggml_et_diag_params;
بنية ggml_et_fill_params;
بنية ggml_et_flash_attn_ext_params;
بنية ggml_et_gated_delta_net_params;
بنية ggml_et_group_norm_params;
بنية ggml_et_im2col_params;
بنية ggml_et_l2_norm_params;
بنية ggml_et_mul_mat_id_params;
بنية ggml_et_norm_params;
بنية ggml_et_pad_params;
بنية ggml_et_repeat_params;
بنية ggml_et_rwkv_wkv6_params;
بنية ggml_et_rwkv_wkv7_params;
بنية ggml_et_scale_params;
بنية ggml_et_set_params;
بنية ggml_et_solve_tri_params;
بنية ggml_et_sqr_params;
بنية ggml_et_ssm_conv_params;
بنية ggml_et_ssm_scan_params;
بنية ggml_et_sum_rows_params;
بنية ggml_et_tri_params;

خارجي صحيح el_map_f32_entry(بنية ggml_et_ثنائی_params *, فراغ *);
خارجي صحيح glu_f32_entry(بنية ggml_et_glu_params *, فراغ *);
خارجي صحيح unary_f32_entry(بنية ggml_et_unary_params *, فراغ *);
خارجي صحيح rope_f32_entry(بنية ggml_et_rope_params *, فراغ *);
خارجي صحيح rms_norm_f32_entry(بنية ggml_et_rms_norm_params *, فراغ *);
خارجي صحيح rms_norm_mul_f32_entry(بنية ggml_et_rms_norm_mul_params *, فراغ *);
خارجي صحيح softmax_f32_entry(بنية ggml_et_softmax_params *, فراغ *);
خارجي صحيح set_rows_f32_entry(بنية ggml_et_set_rows_params *, فراغ *);
خارجي صحيح get_rows_f32_entry(بنية ggml_et_get_rows_params *, فراغ *);
خارجي صحيح cont_f32_entry(بنية ggml_et_cont_params *, فراغ *);
خارجي صحيح cont_f16_entry(بنية ggml_et_cont_params *, فراغ *);
خارجي صحيح cpy_f32_f16_entry(بنية ggml_et_cont_params *, فراغ *);
خارجي صحيح concat_f32_entry(بنية ggml_et_concat_params *, فراغ *);
خارجي صحيح cumsum_f32_entry(بنية ggml_et_cumsum_params *, فراغ *);
خارجي صحيح diag_f32_entry(بنية ggml_et_diag_params *, فراغ *);
خارجي صحيح fill_f32_entry(بنية ggml_et_fill_params *, فراغ *);
خارجي صحيح flash_attn_ext_f32_entry(بنية ggml_et_flash_attn_ext_params *, فراغ *);
خارجي صحيح flash_attn_ext_f16_me_entry(بنية ggml_et_flash_attn_ext_params *, فراغ *);
خارجي صحيح gated_delta_net_f32_entry(بنية ggml_et_gated_delta_net_params *, فراغ *);
خارجي صحيح group_norm_f32_entry(بنية ggml_et_group_norm_params *, فراغ *);
خارجي صحيح im2col_entry(بنية ggml_et_im2col_params *, فراغ *);
خارجي صحيح l2_norm_f32_entry(بنية ggml_et_l2_norm_params *, فراغ *);
خارجي صحيح mul_mat_id_f32_entry(بنية ggml_et_mul_mat_id_params *, فراغ *);
خارجي صحيح norm_f32_entry(بنية ggml_et_norm_params *, فراغ *);
خارجي صحيح pad_f32_entry(بنية ggml_et_pad_params *, فراغ *);
خارجي صحيح repeat_f32_entry(بنية ggml_et_repeat_params *, فراغ *);
خارجي صحيح rwkv_wkv6_f32_entry(بنية ggml_et_rwkv_wkv6_params *, فراغ *);
خارجي صحيح rwkv_wkv7_f32_entry(بنية ggml_et_rwkv_wkv7_params *, فراغ *);
خارجي صحيح scale_f32_entry(بنية ggml_et_scale_params *, فراغ *);
خارجي صحيح set_f32_entry(بنية ggml_et_set_params *, فراغ *);
خارجي صحيح solve_tri_f32_entry(بنية ggml_et_solve_tri_params *, فراغ *);
خارجي صحيح sqr_f32_entry(بنية ggml_et_sqr_params *, فراغ *);
خارجي صحيح ssm_conv_f32_entry(بنية ggml_et_ssm_conv_params *, فراغ *);
خارجي صحيح ssm_scan_f32_entry(بنية ggml_et_ssm_scan_params *, فراغ *);
خارجي صحيح sum_rows_f32_entry(بنية ggml_et_sum_rows_params *, فراغ *);
خارجي صحيح tri_f32_entry(بنية ggml_et_tri_params *, فراغ *);
خارجي صحيح mul_mat_f16_entry(بنية ggml_et_ثنائی_params *, فراغ *);
خارجي صحيح mul_mat_f16_matrix_engine_entry(بنية ggml_et_ثنائی_params *, فراغ *);
خارجي صحيح mul_mat_f32_entry(بنية ggml_et_ثنائی_params *, فراغ *);
خارجي صحيح mul_mat_f32_matrix_engine_entry(بنية ggml_et_ثنائی_params *, فراغ *);
خارجي صحيح mul_mat_Q8_0_entry(بنية ggml_et_mm_q8_params *, فراغ *);
خارجي صحيح mul_mat_Q4_0_entry(بنية ggml_et_ثنائی_params *, فراغ *);

ساكن مضمن size_t tensor_bytes(ثابت بنية ggml_tensor * t) {
    أرجع (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

بنية uber_glu_params {
    بنية ggml_tensor src0;
    بنية ggml_tensor src1;
    بنية ggml_tensor dst;
    // trailing scalars omitted — not needed for eviction
};

بنية uber_unary_params {
    بنية ggml_tensor src0;
    بنية ggml_tensor dst;
};

بنية uber_rope_params {
    بنية ggml_tensor src0;
    بنية ggml_tensor src1;
    بنية ggml_tensor src2;
    بنية ggml_tensor dst;
};

بنية uber_rms_norm_params {
    بنية ggml_tensor src0;
    بنية ggml_tensor dst;
};

بنية uber_rms_norm_mul_params {
    بنية ggml_tensor src0;
    بنية ggml_tensor src1;
    بنية ggml_tensor dst;
};

بنية uber_softmax_params {
    بنية ggml_tensor src0;
    بنية ggml_tensor src1;
    بنية ggml_tensor src2;
    بنية ggml_tensor dst;
};

بنية uber_set_rows_params {
    بنية ggml_tensor src0;
    بنية ggml_tensor src1;
    بنية ggml_tensor dst;
};

بنية uber_get_rows_params {
    بنية ggml_tensor src0;
    بنية ggml_tensor src1;
    بنية ggml_tensor dst;
};

بنية uber_cont_params {
    بنية ggml_tensor src0;
    بنية ggml_tensor dst;
};

// src0 + src1 + dst (no trailing scalars needed for eviction)
بنية uber_concat_params {
    بنية ggml_tensor src0;
    بنية ggml_tensor src1;
    بنية ggml_tensor dst;
};

بنية uber_ssm_conv_params {
    بنية ggml_tensor src0;
    بنية ggml_tensor src1;
    بنية ggml_tensor dst;
};

بنية uber_solve_tri_params {
    بنية ggml_tensor src0;
    بنية ggml_tensor src1;
    بنية ggml_tensor dst;
};

بنية uber_mul_mat_id_params {
    بنية ggml_tensor src0;
    بنية ggml_tensor src1;
    بنية ggml_tensor src2;
    بنية ggml_tensor dst;
};

// flash_attn_ext: Q=src0, K=src1, V=src2, mask=src3, dst (mask optional)
بنية uber_flash_attn_ext_params {
    بنية ggml_tensor src0;
    بنية ggml_tensor src1;
    بنية ggml_tensor src2;
    بنية ggml_tensor mask;
    بنية ggml_tensor dst;
};

// ssm_scan: 7 source tensors + dst
بنية uber_ssm_scan_params {
    بنية ggml_tensor src0;
    بنية ggml_tensor src1;
    بنية ggml_tensor src2;
    بنية ggml_tensor src3;
    بنية ggml_tensor src4;
    بنية ggml_tensor src5;
    بنية ggml_tensor src6;
    بنية ggml_tensor dst;
};

// gated_delta_net: q,k,v,g,beta,state_in,dst
بنية uber_gated_delta_net_params {
    بنية ggml_tensor q;
    بنية ggml_tensor k;
    بنية ggml_tensor v;
    بنية ggml_tensor g;
    بنية ggml_tensor beta;
    بنية ggml_tensor state_in;
    بنية ggml_tensor dst;
};

ساكن فراغ copy_f32_to_f16_row(uint16_t * dst, ثابت عائم * src, int64_t عدد_elements) {
    لكل (int64_t i = 0; i < عدد_elements; i++) {
        dst[i] = fp32_to_fp16(src[i]);
    }
}

ساكن فراغ copy_f32_row(عائم * dst, ثابت عائم * src, int64_t عدد_elements) {
    لكل (int64_t i = 0; i < عدد_elements; i++) {
        dst[i] = src[i];
    }
}

ساكن فراغ evict_region_past_l2_local(ثابت فراغ * addr, size_t bytes) {
    إذا (!addr || bytes == 0) {
        أرجع;
    }

    ثابت uint64_t CL     = 64;
    uint64_t       base   = (uint64_t) addr & ~(CL - 1);
    uint64_t       end    = ((uint64_t) addr + bytes + CL - 1) & ~(CL - 1);
    uint64_t       nlines = (end - base) / CL;
    cache_ops_priv_evict_sw(0, /*to_L2*/ 3, 0, 0, CL);
}

صحيح entry_point(بنية ggml_et_uberkernel_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    إذا (!kernel_env || !params) {
        أرجع -1;
    }

    بنية ggml_et_uberkernel_inst * insts       = (بنية ggml_et_uberkernel_inst *) (uintptr_t) params->insts;
    uint8_t *                        params_blob = (uint8_t *) (uintptr_t) params->params_blob;

    إذا (!insts || !params_blob || params->inst_stride < حجم(بنية ggml_et_uberkernel_inst)) {
        أرجع -1;
    }

    لكل (uint32_t i = 0; i < params->عدد_insts; ++i) {
        بنية ggml_et_uberkernel_inst * inst =
            (بنية ggml_et_uberkernel_inst *) ((uint8_t *) insts + (i * params->inst_stride));
        فراغ * inst_params = params_blob + inst->params_offset;
        صحيح    rc          = -1;

        et_barrier_global(32ULL);

        بدل (inst->kernel_id) {
            حالة GGML_ET_UBERKERNEL_KERNEL_EL_MAP_F32:
                {
                    بنية ggml_et_ثنائی_params * p = (بنية ggml_et_ثنائی_params *) inst_params;
                    rc                               = el_map_f32_entry(p, env);
                    اكسر;
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
            حالة GGML_ET_UBERKERNEL_KERNEL_GLU_F32:
                {
                    rc = glu_f32_entry((بنية ggml_et_glu_params *) inst_params, env);
                    اكسر;
                }
            حالة GGML_ET_UBERKERNEL_KERNEL_ROPE_F32:
                {
                    rc = rope_f32_entry((بنية ggml_et_rope_params *) inst_params, env);
                    اكسر;
                }
            حالة GGML_ET_UBERKERNEL_KERNEL_RMS_NORM_F32:
                {
                    // struct ggml_et_rms_norm_params *p = (struct ggml_et_rms_norm_params *) inst_params;
                    // evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
                    rc = rms_norm_f32_entry((بنية ggml_et_rms_norm_params *) inst_params, env);
                    اكسر;
                }
            حالة GGML_ET_UBERKERNEL_KERNEL_RMS_NORM_MUL_F32:
                {
                    بنية uber_rms_norm_mul_params * p = (بنية uber_rms_norm_mul_params *) inst_params;
                    evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
                    evict_region_past_l2(p->src1.data, tensor_bytes(&p->src1));
                    rc = rms_norm_mul_f32_entry((بنية ggml_et_rms_norm_mul_params *) inst_params, env);
                    اكسر;
                }
            حالة GGML_ET_UBERKERNEL_KERNEL_SOFTMAX_F32:
                {
                    rc = softmax_f32_entry((بنية ggml_et_softmax_params *) inst_params, env);
                    اكسر;
                }
            حالة GGML_ET_UBERKERNEL_KERNEL_SET_ROWS_F32:
                {
                    rc = set_rows_f32_entry((بنية ggml_et_set_rows_params *) inst_params, env);
                    اكسر;
                }

            // Single-source ops (src0 → dst)
            حالة GGML_ET_UBERKERNEL_KERNEL_SQR_F32:
                {
                    rc = sqr_f32_entry((بنية ggml_et_sqr_params *) inst_params, env);
                    اكسر;
                }
            حالة GGML_ET_UBERKERNEL_KERNEL_SCALE_F32:
                {
                    rc = scale_f32_entry((بنية ggml_et_scale_params *) inst_params, env);
                    اكسر;
                }
            حالة GGML_ET_UBERKERNEL_KERNEL_SUM_ROWS_F32:
                {
                    rc = sum_rows_f32_entry((بنية ggml_et_sum_rows_params *) inst_params, env);
                    اكسر;
                }
            حالة GGML_ET_UBERKERNEL_KERNEL_CUMSUM_F32:
                {
                    rc = cumsum_f32_entry((بنية ggml_et_cumsum_params *) inst_params, env);
                    اكسر;
                }
            حالة GGML_ET_UBERKERNEL_KERNEL_NORM_F32:
                {
                    rc = norm_f32_entry((بنية ggml_et_norm_params *) inst_params, env);
                    اكسر;
                }
            حالة GGML_ET_UBERKERNEL_KERNEL_L2_NORM_F32:
                {
                    rc = l2_norm_f32_entry((بنية ggml_et_l2_norm_params *) inst_params, env);
                    اكسر;
                }
            حالة GGML_ET_UBERKERNEL_KERNEL_GROUP_NORM_F32:
                {
                    rc = group_norm_f32_entry((بنية ggml_et_group_norm_params *) inst_params, env);
                    اكسر;
                }
            حالة GGML_ET_UBERKERNEL_KERNEL_REPEAT_F32:
                {
                    rc = repeat_f32_entry((بنية ggml_et_repeat_params *) inst_params, env);
                    اكسر;
                }
            حالة GGML_ET_UBERKERNEL_KERNEL_DIAG_F32:
                {
                    rc = diag_f32_entry((بنية ggml_et_diag_params *) inst_params, env);
                    اكسر;
                }
            حالة GGML_ET_UBERKERNEL_KERNEL_TRI_F32:
                {
                    rc = tri_f32_entry((بنية ggml_et_tri_params *) inst_params, env);
                    اكسر;
                }
            حالة GGML_ET_UBERKERNEL_KERNEL_PAD_F32:
                {
                    rc = pad_f32_entry((بنية ggml_et_pad_params *) inst_params, env);
                    اكسر;
                }
            حالة GGML_ET_UBERKERNEL_KERNEL_CONT_F16:
                {
                    rc = cont_f16_entry((بنية ggml_et_cont_params *) inst_params, env);
                    اكسر;
                }
            حالة GGML_ET_UBERKERNEL_KERNEL_FILL_F32:
                {
                    rc = fill_f32_entry((بنية ggml_et_fill_params *) inst_params, env);
                    اكسر;
                }
            حالة GGML_ET_UBERKERNEL_KERNEL_SET_F32:
                {
                    rc = set_f32_entry((بنية ggml_et_set_params *) inst_params, env);
                    اكسر;
                }

            // Two-source ops
            حالة GGML_ET_UBERKERNEL_KERNEL_CONCAT_F32:
                {
                    rc = concat_f32_entry((بنية ggml_et_concat_params *) inst_params, env);
                    اكسر;
                }
            // case GGML_ET_UBERKERNEL_KERNEL_SSM_CONV_F32: {
            //     rc = ssm_conv_f32_entry((struct ggml_et_ssm_conv_params *) inst_params, env);
            //     break;
            // }
            حالة GGML_ET_UBERKERNEL_KERNEL_SOLVE_TRI_F32:
                {
                    rc = solve_tri_f32_entry((بنية ggml_et_solve_tri_params *) inst_params, env);
                    اكسر;
                }
            حالة GGML_ET_UBERKERNEL_KERNEL_IM2COL:
                {
                    rc = im2col_entry((بنية ggml_et_im2col_params *) inst_params, env);
                    اكسر;
                }

            // Three-source ops
            حالة GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_ID_F32:
                {
                    rc = mul_mat_id_f32_entry((بنية ggml_et_mul_mat_id_params *) inst_params, env);
                    اكسر;
                }
            حالة GGML_ET_UBERKERNEL_KERNEL_FLASH_ATTN_EXT_F32:
                {
                    rc = flash_attn_ext_f32_entry((بنية ggml_et_flash_attn_ext_params *) inst_params, env);
                    اكسر;
                }
            حالة GGML_ET_UBERKERNEL_KERNEL_FLASH_ATTN_EXT_F16_ME:
                {
                    rc = flash_attn_ext_f16_me_entry((بنية ggml_et_flash_attn_ext_params *) inst_params, env);
                    اكسر;
                }

            حالة GGML_ET_UBERKERNEL_KERNEL_GATED_DELTA_NET_F32:
                {
                    rc = gated_delta_net_f32_entry((بنية ggml_et_gated_delta_net_params *) inst_params, env);
                    اكسر;
                }
            حالة GGML_ET_UBERKERNEL_KERNEL_SSM_SCAN_F32:
                {
                    rc = ssm_scan_f32_entry((بنية ggml_et_ssm_scan_params *) inst_params, env);
                    اكسر;
                }
            // rwkv: raw float* params, no ggml_tensor fields to evict via
            حالة GGML_ET_UBERKERNEL_KERNEL_RWKV_WKV6_F32:
                {
                    rc = rwkv_wkv6_f32_entry((بنية ggml_et_rwkv_wkv6_params *) inst_params, env);
                    اكسر;
                }

            حالة GGML_ET_UBERKERNEL_KERNEL_RWKV_WKV7_F32:
                {
                    rc = rwkv_wkv7_f32_entry((بنية ggml_et_rwkv_wkv7_params *) inst_params, env);
                    اكسر;
                }

            // MUL_MAT: evict src1 (activations); src0=weights is
            //  read-only so never stale from a prior uberkernel op
            حالة GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F16:
                {
                    بنية ggml_et_ثنائی_params * p = (بنية ggml_et_ثنائی_params *) inst_params;
                    rc                               = mul_mat_f16_entry(p, env);
                    اكسر;
                }
            حالة GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F16_MATRIX_ENGINE:
                {
                    بنية ggml_et_ثنائی_params * p = (بنية ggml_et_ثنائی_params *) inst_params;
                    rc                               = mul_mat_f16_matrix_engine_entry(p, env);
                    اكسر;
                }
            حالة GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F32:
                {
                    بنية ggml_et_ثنائی_params * p = (بنية ggml_et_ثنائی_params *) inst_params;
                    rc                               = mul_mat_f32_entry(p, env);
                    اكسر;
                }
            حالة GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F32_MATRIX_ENGINE:
                {
                    بنية ggml_et_ثنائی_params * p = (بنية ggml_et_ثنائی_params *) inst_params;
                    rc                               = mul_mat_f32_matrix_engine_entry(p, env);
                    اكسر;
                }
            حالة GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_Q8_0:
                {
                    بنية ggml_et_mm_q8_params * p = (بنية ggml_et_mm_q8_params *) inst_params;
                    // evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
                    rc                              = mul_mat_Q8_0_entry(p, env);
                    اكسر;
                }
            حالة GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_Q4_0:
                {
                    بنية ggml_et_ثنائی_params * p = (بنية ggml_et_ثنائی_params *) inst_params;
                    rc                               = mul_mat_Q4_0_entry(p, env);
                    اكسر;
                }

            افتراضي:
                أرجع -1;
        }

        إذا (rc != 0) {
            أرجع rc;
        }
    }

    أرجع 0;
}
