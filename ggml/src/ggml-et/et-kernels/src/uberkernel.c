#ಅನ್ತರ್ಭಾವ "ggml-et-uberkernel-common.h"
#ಅನ್ತರ್ಭಾವ "ggml-et-uberkernel-kernel-map.h"
#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "math_fp.h"
#ಅನ್ತರ್ಭಾವ "platform.h"

#ಅನ್ತರ್ಭಾವ <stdint.h>

ಸಂರಚನಾ ggml_et_glu_params;
ಸಂರಚನಾ ggml_et_unary_params;
ಸಂರಚನಾ ggml_et_rope_params;
ಸಂರಚನಾ ggml_et_rms_norm_params;
ಸಂರಚನಾ ggml_et_rms_norm_mul_params;
ಸಂರಚನಾ ggml_et_softmax_params;
ಸಂರಚನಾ ggml_et_set_rows_params;
ಸಂರಚನಾ ggml_et_get_rows_params;
ಸಂರಚನಾ ggml_et_cont_params;
ಸಂರಚನಾ ggml_et_concat_params;
ಸಂರಚನಾ ggml_et_cumsum_params;
ಸಂರಚನಾ ggml_et_diag_params;
ಸಂರಚನಾ ggml_et_fill_params;
ಸಂರಚನಾ ggml_et_flash_attn_ext_params;
ಸಂರಚನಾ ggml_et_gated_delta_net_params;
ಸಂರಚನಾ ggml_et_group_norm_params;
ಸಂರಚನಾ ggml_et_im2col_params;
ಸಂರಚನಾ ggml_et_l2_norm_params;
ಸಂರಚನಾ ggml_et_mul_mat_id_params;
ಸಂರಚನಾ ggml_et_norm_params;
ಸಂರಚನಾ ggml_et_pad_params;
ಸಂರಚನಾ ggml_et_repeat_params;
ಸಂರಚನಾ ggml_et_rwkv_wkv6_params;
ಸಂರಚನಾ ggml_et_rwkv_wkv7_params;
ಸಂರಚನಾ ggml_et_scale_params;
ಸಂರಚನಾ ggml_et_set_params;
ಸಂರಚನಾ ggml_et_solve_tri_params;
ಸಂರಚನಾ ggml_et_sqr_params;
ಸಂರಚನಾ ggml_et_ssm_conv_params;
ಸಂರಚನಾ ggml_et_ssm_scan_params;
ಸಂರಚನಾ ggml_et_sum_rows_params;
ಸಂರಚನಾ ggml_et_tri_params;

ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ el_map_f32_entry(ಸಂರಚನಾ ggml_et_ದ್ವಿಮಾನ_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ glu_f32_entry(ಸಂರಚನಾ ggml_et_glu_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ unary_f32_entry(ಸಂರಚನಾ ggml_et_unary_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ rope_f32_entry(ಸಂರಚನಾ ggml_et_rope_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ rms_norm_f32_entry(ಸಂರಚನಾ ggml_et_rms_norm_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ rms_norm_mul_f32_entry(ಸಂರಚನಾ ggml_et_rms_norm_mul_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ softmax_f32_entry(ಸಂರಚನಾ ggml_et_softmax_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ set_rows_f32_entry(ಸಂರಚನಾ ggml_et_set_rows_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ get_rows_f32_entry(ಸಂರಚನಾ ggml_et_get_rows_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ cont_f32_entry(ಸಂರಚನಾ ggml_et_cont_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ cont_f16_entry(ಸಂರಚನಾ ggml_et_cont_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ cpy_f32_f16_entry(ಸಂರಚನಾ ggml_et_cont_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ concat_f32_entry(ಸಂರಚನಾ ggml_et_concat_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ cumsum_f32_entry(ಸಂರಚನಾ ggml_et_cumsum_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ diag_f32_entry(ಸಂರಚನಾ ggml_et_diag_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ fill_f32_entry(ಸಂರಚನಾ ggml_et_fill_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ flash_attn_ext_f32_entry(ಸಂರಚನಾ ggml_et_flash_attn_ext_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ flash_attn_ext_f16_me_entry(ಸಂರಚನಾ ggml_et_flash_attn_ext_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ gated_delta_net_f32_entry(ಸಂರಚನಾ ggml_et_gated_delta_net_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ group_norm_f32_entry(ಸಂರಚನಾ ggml_et_group_norm_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ im2col_entry(ಸಂರಚನಾ ggml_et_im2col_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ l2_norm_f32_entry(ಸಂರಚನಾ ggml_et_l2_norm_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ mul_mat_id_f32_entry(ಸಂರಚನಾ ggml_et_mul_mat_id_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ norm_f32_entry(ಸಂರಚನಾ ggml_et_norm_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ pad_f32_entry(ಸಂರಚನಾ ggml_et_pad_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ repeat_f32_entry(ಸಂರಚನಾ ggml_et_repeat_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ rwkv_wkv6_f32_entry(ಸಂರಚನಾ ggml_et_rwkv_wkv6_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ rwkv_wkv7_f32_entry(ಸಂರಚನಾ ggml_et_rwkv_wkv7_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ scale_f32_entry(ಸಂರಚನಾ ggml_et_scale_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ set_f32_entry(ಸಂರಚನಾ ggml_et_set_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ solve_tri_f32_entry(ಸಂರಚನಾ ggml_et_solve_tri_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ sqr_f32_entry(ಸಂರಚನಾ ggml_et_sqr_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ ssm_conv_f32_entry(ಸಂರಚನಾ ggml_et_ssm_conv_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ ssm_scan_f32_entry(ಸಂರಚನಾ ggml_et_ssm_scan_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ sum_rows_f32_entry(ಸಂರಚನಾ ggml_et_sum_rows_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ tri_f32_entry(ಸಂರಚನಾ ggml_et_tri_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ mul_mat_f16_entry(ಸಂರಚನಾ ggml_et_ದ್ವಿಮಾನ_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ mul_mat_f16_matrix_engine_entry(ಸಂರಚನಾ ggml_et_ದ್ವಿಮಾನ_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ mul_mat_f32_entry(ಸಂರಚನಾ ggml_et_ದ್ವಿಮಾನ_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ mul_mat_f32_matrix_engine_entry(ಸಂರಚನಾ ggml_et_ದ್ವಿಮಾನ_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ mul_mat_Q8_0_entry(ಸಂರಚನಾ ggml_et_mm_q8_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ mul_mat_Q4_0_entry(ಸಂರಚನಾ ggml_et_ದ್ವಿಮಾನ_params *, ಶೂನ್ಯ *);

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ size_t tensor_bytes(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * t) {
    ಪ್ರತ್ಯಾಗಮ (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

ಸಂರಚನಾ uber_glu_params {
    ಸಂರಚನಾ ggml_tensor src0;
    ಸಂರಚನಾ ggml_tensor src1;
    ಸಂರಚನಾ ggml_tensor dst;
    // trailing scalars omitted — not needed for eviction
};

ಸಂರಚನಾ uber_unary_params {
    ಸಂರಚನಾ ggml_tensor src0;
    ಸಂರಚನಾ ggml_tensor dst;
};

ಸಂರಚನಾ uber_rope_params {
    ಸಂರಚನಾ ggml_tensor src0;
    ಸಂರಚನಾ ggml_tensor src1;
    ಸಂರಚನಾ ggml_tensor src2;
    ಸಂರಚನಾ ggml_tensor dst;
};

ಸಂರಚನಾ uber_rms_norm_params {
    ಸಂರಚನಾ ggml_tensor src0;
    ಸಂರಚನಾ ggml_tensor dst;
};

ಸಂರಚನಾ uber_rms_norm_mul_params {
    ಸಂರಚನಾ ggml_tensor src0;
    ಸಂರಚನಾ ggml_tensor src1;
    ಸಂರಚನಾ ggml_tensor dst;
};

ಸಂರಚನಾ uber_softmax_params {
    ಸಂರಚನಾ ggml_tensor src0;
    ಸಂರಚನಾ ggml_tensor src1;
    ಸಂರಚನಾ ggml_tensor src2;
    ಸಂರಚನಾ ggml_tensor dst;
};

ಸಂರಚನಾ uber_set_rows_params {
    ಸಂರಚನಾ ggml_tensor src0;
    ಸಂರಚನಾ ggml_tensor src1;
    ಸಂರಚನಾ ggml_tensor dst;
};

ಸಂರಚನಾ uber_get_rows_params {
    ಸಂರಚನಾ ggml_tensor src0;
    ಸಂರಚನಾ ggml_tensor src1;
    ಸಂರಚನಾ ggml_tensor dst;
};

ಸಂರಚನಾ uber_cont_params {
    ಸಂರಚನಾ ggml_tensor src0;
    ಸಂರಚನಾ ggml_tensor dst;
};

// src0 + src1 + dst (no trailing scalars needed for eviction)
ಸಂರಚನಾ uber_concat_params {
    ಸಂರಚನಾ ggml_tensor src0;
    ಸಂರಚನಾ ggml_tensor src1;
    ಸಂರಚನಾ ggml_tensor dst;
};

ಸಂರಚನಾ uber_ssm_conv_params {
    ಸಂರಚನಾ ggml_tensor src0;
    ಸಂರಚನಾ ggml_tensor src1;
    ಸಂರಚನಾ ggml_tensor dst;
};

ಸಂರಚನಾ uber_solve_tri_params {
    ಸಂರಚನಾ ggml_tensor src0;
    ಸಂರಚನಾ ggml_tensor src1;
    ಸಂರಚನಾ ggml_tensor dst;
};

ಸಂರಚನಾ uber_mul_mat_id_params {
    ಸಂರಚನಾ ggml_tensor src0;
    ಸಂರಚನಾ ggml_tensor src1;
    ಸಂರಚನಾ ggml_tensor src2;
    ಸಂರಚನಾ ggml_tensor dst;
};

// flash_attn_ext: Q=src0, K=src1, V=src2, mask=src3, dst (mask optional)
ಸಂರಚನಾ uber_flash_attn_ext_params {
    ಸಂರಚನಾ ggml_tensor src0;
    ಸಂರಚನಾ ggml_tensor src1;
    ಸಂರಚನಾ ggml_tensor src2;
    ಸಂರಚನಾ ggml_tensor mask;
    ಸಂರಚನಾ ggml_tensor dst;
};

// ssm_scan: 7 source tensors + dst
ಸಂರಚನಾ uber_ssm_scan_params {
    ಸಂರಚನಾ ggml_tensor src0;
    ಸಂರಚನಾ ggml_tensor src1;
    ಸಂರಚನಾ ggml_tensor src2;
    ಸಂರಚನಾ ggml_tensor src3;
    ಸಂರಚನಾ ggml_tensor src4;
    ಸಂರಚನಾ ggml_tensor src5;
    ಸಂರಚನಾ ggml_tensor src6;
    ಸಂರಚನಾ ggml_tensor dst;
};

// gated_delta_net: q,k,v,g,beta,state_in,dst
ಸಂರಚನಾ uber_gated_delta_net_params {
    ಸಂರಚನಾ ggml_tensor q;
    ಸಂರಚನಾ ggml_tensor k;
    ಸಂರಚನಾ ggml_tensor v;
    ಸಂರಚನಾ ggml_tensor g;
    ಸಂರಚನಾ ggml_tensor beta;
    ಸಂರಚನಾ ggml_tensor state_in;
    ಸಂರಚನಾ ggml_tensor dst;
};

ಸ್ಥಿರ ಶೂನ್ಯ copy_f32_to_f16_row(uint16_t * dst, ಧ್ರುವ ಭಿನ್ನ * src, int64_t ಅಂಕೆ_elements) {
    ಪ್ರತಿ (int64_t i = 0; i < ಅಂಕೆ_elements; i++) {
        dst[i] = fp32_to_fp16(src[i]);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ copy_f32_row(ಭಿನ್ನ * dst, ಧ್ರುವ ಭಿನ್ನ * src, int64_t ಅಂಕೆ_elements) {
    ಪ್ರತಿ (int64_t i = 0; i < ಅಂಕೆ_elements; i++) {
        dst[i] = src[i];
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ evict_region_past_l2_local(ಧ್ರುವ ಶೂನ್ಯ * addr, size_t bytes) {
    ಯದಿ (!addr || bytes == 0) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಧ್ರುವ uint64_t CL     = 64;
    uint64_t       base   = (uint64_t) addr & ~(CL - 1);
    uint64_t       end    = ((uint64_t) addr + bytes + CL - 1) & ~(CL - 1);
    uint64_t       nlines = (end - base) / CL;
    cache_ops_priv_evict_sw(0, /*to_L2*/ 3, 0, 0, CL);
}

ಪೂರ್ಣಾಙ್ಕ entry_point(ಸಂರಚನಾ ggml_et_uberkernel_params * params, ಶೂನ್ಯ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ಯದಿ (!kernel_env || !params) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಸಂರಚನಾ ggml_et_uberkernel_inst * insts       = (ಸಂರಚನಾ ggml_et_uberkernel_inst *) (uintptr_t) params->insts;
    uint8_t *                        params_blob = (uint8_t *) (uintptr_t) params->params_blob;

    ಯದಿ (!insts || !params_blob || params->inst_stride < ಪರಿಮಾಣ(ಸಂರಚನಾ ggml_et_uberkernel_inst)) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಪ್ರತಿ (uint32_t i = 0; i < params->ಅಂಕೆ_insts; ++i) {
        ಸಂರಚನಾ ggml_et_uberkernel_inst * inst =
            (ಸಂರಚನಾ ggml_et_uberkernel_inst *) ((uint8_t *) insts + (i * params->inst_stride));
        ಶೂನ್ಯ * inst_params = params_blob + inst->params_offset;
        ಪೂರ್ಣಾಙ್ಕ    rc          = -1;

        et_barrier_global(32ULL);

        ವಿಕಲ್ಪ (inst->kernel_id) {
            ಪಕ್ಷ GGML_ET_UBERKERNEL_KERNEL_EL_MAP_F32:
                {
                    ಸಂರಚನಾ ggml_et_ದ್ವಿಮಾನ_params * p = (ಸಂರಚನಾ ggml_et_ದ್ವಿಮಾನ_params *) inst_params;
                    rc                               = el_map_f32_entry(p, env);
                    ಭಙ್ಗ;
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
            ಪಕ್ಷ GGML_ET_UBERKERNEL_KERNEL_GLU_F32:
                {
                    rc = glu_f32_entry((ಸಂರಚನಾ ggml_et_glu_params *) inst_params, env);
                    ಭಙ್ಗ;
                }
            ಪಕ್ಷ GGML_ET_UBERKERNEL_KERNEL_ROPE_F32:
                {
                    rc = rope_f32_entry((ಸಂರಚನಾ ggml_et_rope_params *) inst_params, env);
                    ಭಙ್ಗ;
                }
            ಪಕ್ಷ GGML_ET_UBERKERNEL_KERNEL_RMS_NORM_F32:
                {
                    // struct ggml_et_rms_norm_params *p = (struct ggml_et_rms_norm_params *) inst_params;
                    // evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
                    rc = rms_norm_f32_entry((ಸಂರಚನಾ ggml_et_rms_norm_params *) inst_params, env);
                    ಭಙ್ಗ;
                }
            ಪಕ್ಷ GGML_ET_UBERKERNEL_KERNEL_RMS_NORM_MUL_F32:
                {
                    ಸಂರಚನಾ uber_rms_norm_mul_params * p = (ಸಂರಚನಾ uber_rms_norm_mul_params *) inst_params;
                    evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
                    evict_region_past_l2(p->src1.data, tensor_bytes(&p->src1));
                    rc = rms_norm_mul_f32_entry((ಸಂರಚನಾ ggml_et_rms_norm_mul_params *) inst_params, env);
                    ಭಙ್ಗ;
                }
            ಪಕ್ಷ GGML_ET_UBERKERNEL_KERNEL_SOFTMAX_F32:
                {
                    rc = softmax_f32_entry((ಸಂರಚನಾ ggml_et_softmax_params *) inst_params, env);
                    ಭಙ್ಗ;
                }
            ಪಕ್ಷ GGML_ET_UBERKERNEL_KERNEL_SET_ROWS_F32:
                {
                    rc = set_rows_f32_entry((ಸಂರಚನಾ ggml_et_set_rows_params *) inst_params, env);
                    ಭಙ್ಗ;
                }

            // Single-source ops (src0 → dst)
            ಪಕ್ಷ GGML_ET_UBERKERNEL_KERNEL_SQR_F32:
                {
                    rc = sqr_f32_entry((ಸಂರಚನಾ ggml_et_sqr_params *) inst_params, env);
                    ಭಙ್ಗ;
                }
            ಪಕ್ಷ GGML_ET_UBERKERNEL_KERNEL_SCALE_F32:
                {
                    rc = scale_f32_entry((ಸಂರಚನಾ ggml_et_scale_params *) inst_params, env);
                    ಭಙ್ಗ;
                }
            ಪಕ್ಷ GGML_ET_UBERKERNEL_KERNEL_SUM_ROWS_F32:
                {
                    rc = sum_rows_f32_entry((ಸಂರಚನಾ ggml_et_sum_rows_params *) inst_params, env);
                    ಭಙ್ಗ;
                }
            ಪಕ್ಷ GGML_ET_UBERKERNEL_KERNEL_CUMSUM_F32:
                {
                    rc = cumsum_f32_entry((ಸಂರಚನಾ ggml_et_cumsum_params *) inst_params, env);
                    ಭಙ್ಗ;
                }
            ಪಕ್ಷ GGML_ET_UBERKERNEL_KERNEL_NORM_F32:
                {
                    rc = norm_f32_entry((ಸಂರಚನಾ ggml_et_norm_params *) inst_params, env);
                    ಭಙ್ಗ;
                }
            ಪಕ್ಷ GGML_ET_UBERKERNEL_KERNEL_L2_NORM_F32:
                {
                    rc = l2_norm_f32_entry((ಸಂರಚನಾ ggml_et_l2_norm_params *) inst_params, env);
                    ಭಙ್ಗ;
                }
            ಪಕ್ಷ GGML_ET_UBERKERNEL_KERNEL_GROUP_NORM_F32:
                {
                    rc = group_norm_f32_entry((ಸಂರಚನಾ ggml_et_group_norm_params *) inst_params, env);
                    ಭಙ್ಗ;
                }
            ಪಕ್ಷ GGML_ET_UBERKERNEL_KERNEL_REPEAT_F32:
                {
                    rc = repeat_f32_entry((ಸಂರಚನಾ ggml_et_repeat_params *) inst_params, env);
                    ಭಙ್ಗ;
                }
            ಪಕ್ಷ GGML_ET_UBERKERNEL_KERNEL_DIAG_F32:
                {
                    rc = diag_f32_entry((ಸಂರಚನಾ ggml_et_diag_params *) inst_params, env);
                    ಭಙ್ಗ;
                }
            ಪಕ್ಷ GGML_ET_UBERKERNEL_KERNEL_TRI_F32:
                {
                    rc = tri_f32_entry((ಸಂರಚನಾ ggml_et_tri_params *) inst_params, env);
                    ಭಙ್ಗ;
                }
            ಪಕ್ಷ GGML_ET_UBERKERNEL_KERNEL_PAD_F32:
                {
                    rc = pad_f32_entry((ಸಂರಚನಾ ggml_et_pad_params *) inst_params, env);
                    ಭಙ್ಗ;
                }
            ಪಕ್ಷ GGML_ET_UBERKERNEL_KERNEL_CONT_F16:
                {
                    rc = cont_f16_entry((ಸಂರಚನಾ ggml_et_cont_params *) inst_params, env);
                    ಭಙ್ಗ;
                }
            ಪಕ್ಷ GGML_ET_UBERKERNEL_KERNEL_FILL_F32:
                {
                    rc = fill_f32_entry((ಸಂರಚನಾ ggml_et_fill_params *) inst_params, env);
                    ಭಙ್ಗ;
                }
            ಪಕ್ಷ GGML_ET_UBERKERNEL_KERNEL_SET_F32:
                {
                    rc = set_f32_entry((ಸಂರಚನಾ ggml_et_set_params *) inst_params, env);
                    ಭಙ್ಗ;
                }

            // Two-source ops
            ಪಕ್ಷ GGML_ET_UBERKERNEL_KERNEL_CONCAT_F32:
                {
                    rc = concat_f32_entry((ಸಂರಚನಾ ggml_et_concat_params *) inst_params, env);
                    ಭಙ್ಗ;
                }
            // case GGML_ET_UBERKERNEL_KERNEL_SSM_CONV_F32: {
            //     rc = ssm_conv_f32_entry((struct ggml_et_ssm_conv_params *) inst_params, env);
            //     break;
            // }
            ಪಕ್ಷ GGML_ET_UBERKERNEL_KERNEL_SOLVE_TRI_F32:
                {
                    rc = solve_tri_f32_entry((ಸಂರಚನಾ ggml_et_solve_tri_params *) inst_params, env);
                    ಭಙ್ಗ;
                }
            ಪಕ್ಷ GGML_ET_UBERKERNEL_KERNEL_IM2COL:
                {
                    rc = im2col_entry((ಸಂರಚನಾ ggml_et_im2col_params *) inst_params, env);
                    ಭಙ್ಗ;
                }

            // Three-source ops
            ಪಕ್ಷ GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_ID_F32:
                {
                    rc = mul_mat_id_f32_entry((ಸಂರಚನಾ ggml_et_mul_mat_id_params *) inst_params, env);
                    ಭಙ್ಗ;
                }
            ಪಕ್ಷ GGML_ET_UBERKERNEL_KERNEL_FLASH_ATTN_EXT_F32:
                {
                    rc = flash_attn_ext_f32_entry((ಸಂರಚನಾ ggml_et_flash_attn_ext_params *) inst_params, env);
                    ಭಙ್ಗ;
                }
            ಪಕ್ಷ GGML_ET_UBERKERNEL_KERNEL_FLASH_ATTN_EXT_F16_ME:
                {
                    rc = flash_attn_ext_f16_me_entry((ಸಂರಚನಾ ggml_et_flash_attn_ext_params *) inst_params, env);
                    ಭಙ್ಗ;
                }

            ಪಕ್ಷ GGML_ET_UBERKERNEL_KERNEL_GATED_DELTA_NET_F32:
                {
                    rc = gated_delta_net_f32_entry((ಸಂರಚನಾ ggml_et_gated_delta_net_params *) inst_params, env);
                    ಭಙ್ಗ;
                }
            ಪಕ್ಷ GGML_ET_UBERKERNEL_KERNEL_SSM_SCAN_F32:
                {
                    rc = ssm_scan_f32_entry((ಸಂರಚನಾ ggml_et_ssm_scan_params *) inst_params, env);
                    ಭಙ್ಗ;
                }
            // rwkv: raw float* params, no ggml_tensor fields to evict via
            ಪಕ್ಷ GGML_ET_UBERKERNEL_KERNEL_RWKV_WKV6_F32:
                {
                    rc = rwkv_wkv6_f32_entry((ಸಂರಚನಾ ggml_et_rwkv_wkv6_params *) inst_params, env);
                    ಭಙ್ಗ;
                }

            ಪಕ್ಷ GGML_ET_UBERKERNEL_KERNEL_RWKV_WKV7_F32:
                {
                    rc = rwkv_wkv7_f32_entry((ಸಂರಚನಾ ggml_et_rwkv_wkv7_params *) inst_params, env);
                    ಭಙ್ಗ;
                }

            // MUL_MAT: evict src1 (activations); src0=weights is
            //  read-only so never stale from a prior uberkernel op
            ಪಕ್ಷ GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F16:
                {
                    ಸಂರಚನಾ ggml_et_ದ್ವಿಮಾನ_params * p = (ಸಂರಚನಾ ggml_et_ದ್ವಿಮಾನ_params *) inst_params;
                    rc                               = mul_mat_f16_entry(p, env);
                    ಭಙ್ಗ;
                }
            ಪಕ್ಷ GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F16_MATRIX_ENGINE:
                {
                    ಸಂರಚನಾ ggml_et_ದ್ವಿಮಾನ_params * p = (ಸಂರಚನಾ ggml_et_ದ್ವಿಮಾನ_params *) inst_params;
                    rc                               = mul_mat_f16_matrix_engine_entry(p, env);
                    ಭಙ್ಗ;
                }
            ಪಕ್ಷ GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F32:
                {
                    ಸಂರಚನಾ ggml_et_ದ್ವಿಮಾನ_params * p = (ಸಂರಚನಾ ggml_et_ದ್ವಿಮಾನ_params *) inst_params;
                    rc                               = mul_mat_f32_entry(p, env);
                    ಭಙ್ಗ;
                }
            ಪಕ್ಷ GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F32_MATRIX_ENGINE:
                {
                    ಸಂರಚನಾ ggml_et_ದ್ವಿಮಾನ_params * p = (ಸಂರಚನಾ ggml_et_ದ್ವಿಮಾನ_params *) inst_params;
                    rc                               = mul_mat_f32_matrix_engine_entry(p, env);
                    ಭಙ್ಗ;
                }
            ಪಕ್ಷ GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_Q8_0:
                {
                    ಸಂರಚನಾ ggml_et_mm_q8_params * p = (ಸಂರಚನಾ ggml_et_mm_q8_params *) inst_params;
                    // evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
                    rc                              = mul_mat_Q8_0_entry(p, env);
                    ಭಙ್ಗ;
                }
            ಪಕ್ಷ GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_Q4_0:
                {
                    ಸಂರಚನಾ ggml_et_ದ್ವಿಮಾನ_params * p = (ಸಂರಚನಾ ggml_et_ದ್ವಿಮಾನ_params *) inst_params;
                    rc                               = mul_mat_Q4_0_entry(p, env);
                    ಭಙ್ಗ;
                }

            ಸ್ವಭಾವ:
                ಪ್ರತ್ಯಾಗಮ -1;
        }

        ಯದಿ (rc != 0) {
            ಪ್ರತ್ಯಾಗಮ rc;
        }
    }

    ಪ್ರತ್ಯಾಗಮ 0;
}
