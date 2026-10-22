#ಅನ್ತರ್ಭಾವ "ggml-et-uberkernel-common.h"
#ಅನ್ತರ್ಭಾವ "ggml-et-uberkernel-kernel-map.h"
#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "math_fp.h"
#ಅನ್ತರ್ಭಾವ "platform.h"

#ಅನ್ತರ್ಭಾವ <stdint.h>

ರಚನೆ ggml_et_glu_params;
ರಚನೆ ggml_et_unary_params;
ರಚನೆ ggml_et_rope_params;
ರಚನೆ ggml_et_rms_norm_params;
ರಚನೆ ggml_et_rms_norm_mul_params;
ರಚನೆ ggml_et_softmax_params;
ರಚನೆ ggml_et_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_params;
ರಚನೆ ggml_et_get_ಅಡ್ಡಸಾಲುಗಳು_params;
ರಚನೆ ggml_et_cont_params;
ರಚನೆ ggml_et_concat_params;
ರಚನೆ ggml_et_cumsum_params;
ರಚನೆ ggml_et_diag_params;
ರಚನೆ ggml_et_fill_params;
ರಚನೆ ggml_et_flash_attn_ext_params;
ರಚನೆ ggml_et_gated_delta_net_params;
ರಚನೆ ggml_et_ಗುಂಪು_norm_params;
ರಚನೆ ggml_et_im2col_params;
ರಚನೆ ggml_et_l2_norm_params;
ರಚನೆ ggml_et_mul_mat_id_params;
ರಚನೆ ggml_et_norm_params;
ರಚನೆ ggml_et_pad_params;
ರಚನೆ ggml_et_repeat_params;
ರಚನೆ ggml_et_rwkv_wkv6_params;
ರಚನೆ ggml_et_rwkv_wkv7_params;
ರಚನೆ ggml_et_scale_params;
ರಚನೆ ggml_et_ಗಣ_params;
ರಚನೆ ggml_et_solve_tri_params;
ರಚನೆ ggml_et_sqr_params;
ರಚನೆ ggml_et_ssm_conv_params;
ರಚನೆ ggml_et_ssm_ಪರಿಶೀಲನೆ_params;
ರಚನೆ ggml_et_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_params;
ರಚನೆ ggml_et_tri_params;

ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ el_map_f32_ನಮೂದು(ರಚನೆ ggml_et_ದ್ವಿಮಾನ_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ glu_f32_ನಮೂದು(ರಚನೆ ggml_et_glu_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ unary_f32_ನಮೂದು(ರಚನೆ ggml_et_unary_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ rope_f32_ನಮೂದು(ರಚನೆ ggml_et_rope_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ rms_norm_f32_ನಮೂದು(ರಚನೆ ggml_et_rms_norm_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ rms_norm_mul_f32_ನಮೂದು(ರಚನೆ ggml_et_rms_norm_mul_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ softmax_f32_ನಮೂದು(ರಚನೆ ggml_et_softmax_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_f32_ನಮೂದು(ರಚನೆ ggml_et_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ get_ಅಡ್ಡಸಾಲುಗಳು_f32_ನಮೂದು(ರಚನೆ ggml_et_get_ಅಡ್ಡಸಾಲುಗಳು_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ cont_f32_ನಮೂದು(ರಚನೆ ggml_et_cont_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ cont_f16_ನಮೂದು(ರಚನೆ ggml_et_cont_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ cpy_f32_f16_ನಮೂದು(ರಚನೆ ggml_et_cont_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ concat_f32_ನಮೂದು(ರಚನೆ ggml_et_concat_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ cumsum_f32_ನಮೂದು(ರಚನೆ ggml_et_cumsum_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ diag_f32_ನಮೂದು(ರಚನೆ ggml_et_diag_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ fill_f32_ನಮೂದು(ರಚನೆ ggml_et_fill_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ flash_attn_ext_f32_ನಮೂದು(ರಚನೆ ggml_et_flash_attn_ext_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ flash_attn_ext_f16_me_ನಮೂದು(ರಚನೆ ggml_et_flash_attn_ext_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ gated_delta_net_f32_ನಮೂದು(ರಚನೆ ggml_et_gated_delta_net_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ ಗುಂಪು_norm_f32_ನಮೂದು(ರಚನೆ ggml_et_ಗುಂಪು_norm_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ im2col_ನಮೂದು(ರಚನೆ ggml_et_im2col_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ l2_norm_f32_ನಮೂದು(ರಚನೆ ggml_et_l2_norm_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ mul_mat_id_f32_ನಮೂದು(ರಚನೆ ggml_et_mul_mat_id_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ norm_f32_ನಮೂದು(ರಚನೆ ggml_et_norm_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ pad_f32_ನಮೂದು(ರಚನೆ ggml_et_pad_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ repeat_f32_ನಮೂದು(ರಚನೆ ggml_et_repeat_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ rwkv_wkv6_f32_ನಮೂದು(ರಚನೆ ggml_et_rwkv_wkv6_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ rwkv_wkv7_f32_ನಮೂದು(ರಚನೆ ggml_et_rwkv_wkv7_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ scale_f32_ನಮೂದು(ರಚನೆ ggml_et_scale_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ ಗಣ_f32_ನಮೂದು(ರಚನೆ ggml_et_ಗಣ_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ solve_tri_f32_ನಮೂದು(ರಚನೆ ggml_et_solve_tri_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ sqr_f32_ನಮೂದು(ರಚನೆ ggml_et_sqr_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ ssm_conv_f32_ನಮೂದು(ರಚನೆ ggml_et_ssm_conv_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ ssm_ಪರಿಶೀಲನೆ_f32_ನಮೂದು(ರಚನೆ ggml_et_ssm_ಪರಿಶೀಲನೆ_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_f32_ನಮೂದು(ರಚನೆ ggml_et_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ tri_f32_ನಮೂದು(ರಚನೆ ggml_et_tri_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ mul_mat_f16_ನಮೂದು(ರಚನೆ ggml_et_ದ್ವಿಮಾನ_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ mul_mat_f16_ಮಾತೃಕೆ_engine_ನಮೂದು(ರಚನೆ ggml_et_ದ್ವಿಮಾನ_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ mul_mat_f32_ನಮೂದು(ರಚನೆ ggml_et_ದ್ವಿಮಾನ_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ mul_mat_f32_ಮಾತೃಕೆ_engine_ನಮೂದು(ರಚನೆ ggml_et_ದ್ವಿಮಾನ_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ mul_mat_Q8_0_ನಮೂದು(ರಚನೆ ggml_et_mm_q8_params *, ಶೂನ್ಯ *);
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ mul_mat_Q4_0_ನಮೂದು(ರಚನೆ ggml_et_ದ್ವಿಮಾನ_params *, ಶೂನ್ಯ *);

ಸ್ಥಿರ ಸಾಲೊಳಗೆ size_t tensor_ಬೈಟುಗಳು(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * t) {
    ಹಿಂತಿರುಗಿ (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

ರಚನೆ uber_glu_params {
    ರಚನೆ ggml_tensor src0;
    ರಚನೆ ggml_tensor src1;
    ರಚನೆ ggml_tensor dst;
    // trailing scalars omitted — not needed for eviction
};

ರಚನೆ uber_unary_params {
    ರಚನೆ ggml_tensor src0;
    ರಚನೆ ggml_tensor dst;
};

ರಚನೆ uber_rope_params {
    ರಚನೆ ggml_tensor src0;
    ರಚನೆ ggml_tensor src1;
    ರಚನೆ ggml_tensor src2;
    ರಚನೆ ggml_tensor dst;
};

ರಚನೆ uber_rms_norm_params {
    ರಚನೆ ggml_tensor src0;
    ರಚನೆ ggml_tensor dst;
};

ರಚನೆ uber_rms_norm_mul_params {
    ರಚನೆ ggml_tensor src0;
    ರಚನೆ ggml_tensor src1;
    ರಚನೆ ggml_tensor dst;
};

ರಚನೆ uber_softmax_params {
    ರಚನೆ ggml_tensor src0;
    ರಚನೆ ggml_tensor src1;
    ರಚನೆ ggml_tensor src2;
    ರಚನೆ ggml_tensor dst;
};

ರಚನೆ uber_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_params {
    ರಚನೆ ggml_tensor src0;
    ರಚನೆ ggml_tensor src1;
    ರಚನೆ ggml_tensor dst;
};

ರಚನೆ uber_get_ಅಡ್ಡಸಾಲುಗಳು_params {
    ರಚನೆ ggml_tensor src0;
    ರಚನೆ ggml_tensor src1;
    ರಚನೆ ggml_tensor dst;
};

ರಚನೆ uber_cont_params {
    ರಚನೆ ggml_tensor src0;
    ರಚನೆ ggml_tensor dst;
};

// src0 + src1 + dst (no trailing scalars needed for eviction)
ರಚನೆ uber_concat_params {
    ರಚನೆ ggml_tensor src0;
    ರಚನೆ ggml_tensor src1;
    ರಚನೆ ggml_tensor dst;
};

ರಚನೆ uber_ssm_conv_params {
    ರಚನೆ ggml_tensor src0;
    ರಚನೆ ggml_tensor src1;
    ರಚನೆ ggml_tensor dst;
};

ರಚನೆ uber_solve_tri_params {
    ರಚನೆ ggml_tensor src0;
    ರಚನೆ ggml_tensor src1;
    ರಚನೆ ggml_tensor dst;
};

ರಚನೆ uber_mul_mat_id_params {
    ರಚನೆ ggml_tensor src0;
    ರಚನೆ ggml_tensor src1;
    ರಚನೆ ggml_tensor src2;
    ರಚನೆ ggml_tensor dst;
};

// flash_attn_ext: Q=src0, K=src1, V=src2, mask=src3, dst (mask optional)
ರಚನೆ uber_flash_attn_ext_params {
    ರಚನೆ ggml_tensor src0;
    ರಚನೆ ggml_tensor src1;
    ರಚನೆ ggml_tensor src2;
    ರಚನೆ ggml_tensor ಮುಖವಾಡ;
    ರಚನೆ ggml_tensor dst;
};

// ssm_scan: 7 source tensors + dst
ರಚನೆ uber_ssm_ಪರಿಶೀಲನೆ_params {
    ರಚನೆ ggml_tensor src0;
    ರಚನೆ ggml_tensor src1;
    ರಚನೆ ggml_tensor src2;
    ರಚನೆ ggml_tensor src3;
    ರಚನೆ ggml_tensor src4;
    ರಚನೆ ggml_tensor src5;
    ರಚನೆ ggml_tensor src6;
    ರಚನೆ ggml_tensor dst;
};

// gated_delta_net: q,k,v,g,beta,state_in,dst
ರಚನೆ uber_gated_delta_net_params {
    ರಚನೆ ggml_tensor q;
    ರಚನೆ ggml_tensor k;
    ರಚನೆ ggml_tensor v;
    ರಚನೆ ggml_tensor g;
    ರಚನೆ ggml_tensor beta;
    ರಚನೆ ggml_tensor state_ಒಳ;
    ರಚನೆ ggml_tensor dst;
};

ಸ್ಥಿರ ಶೂನ್ಯ copy_f32_ಗಮ್ಯ_f16_ಅಡ್ಡಸಾಲು(uint16_t * dst, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src, int64_t ಅಂಕೆ_elements) {
    ಗಾಗಿ (int64_t i = 0; i < ಅಂಕೆ_elements; i++) {
        dst[i] = fp32_ಗಮ್ಯ_fp16(src[i]);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ copy_f32_ಅಡ್ಡಸಾಲು(ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src, int64_t ಅಂಕೆ_elements) {
    ಗಾಗಿ (int64_t i = 0; i < ಅಂಕೆ_elements; i++) {
        dst[i] = src[i];
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ evict_region_past_l2_local(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * addr, size_t ಬೈಟುಗಳು) {
    ಆದರೆ (!addr || ಬೈಟುಗಳು == 0) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ uint64_t CL     = 64;
    uint64_t       ತಳ   = (uint64_t) addr & ~(CL - 1);
    uint64_t       ಕೊನೆ    = ((uint64_t) addr + ಬೈಟುಗಳು + CL - 1) & ~(CL - 1);
    uint64_t       nlines = (ಕೊನೆ - ತಳ) / CL;
    cache_ops_priv_evict_sw(0, /*to_L2*/ 3, 0, 0, CL);
}

ಪೂರ್ಣಾಂಕ ನಮೂದು_ಬಿಂದು(ರಚನೆ ggml_et_uberkernel_params * params, ಶೂನ್ಯ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ಆದರೆ (!kernel_env || !params) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ರಚನೆ ggml_et_uberkernel_inst * insts       = (ರಚನೆ ggml_et_uberkernel_inst *) (uintptr_t) params->insts;
    uint8_t *                        params_blob = (uint8_t *) (uintptr_t) params->params_blob;

    ಆದರೆ (!insts || !params_blob || params->inst_stride < ಗಾತ್ರ(ರಚನೆ ggml_et_uberkernel_inst)) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಗಾಗಿ (uint32_t i = 0; i < params->ಅಂಕೆ_insts; ++i) {
        ರಚನೆ ggml_et_uberkernel_inst * inst =
            (ರಚನೆ ggml_et_uberkernel_inst *) ((uint8_t *) insts + (i * params->inst_stride));
        ಶೂನ್ಯ * inst_params = params_blob + inst->params_ಸ್ಥಾನಾಂತರ;
        ಪೂರ್ಣಾಂಕ    rc          = -1;

        et_barrier_global(32ULL);

        ಆಯ್ಕೆ (inst->kernel_id) {
            ಸ್ಥಿತಿ GGML_ET_UBERKERNEL_KERNEL_EL_MAP_F32:
                {
                    ರಚನೆ ggml_et_ದ್ವಿಮಾನ_params * p = (ರಚನೆ ggml_et_ದ್ವಿಮಾನ_params *) inst_params;
                    rc                               = el_map_f32_ನಮೂದು(p, env);
                    ನಿಲ್ಲಿಸು;
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
            ಸ್ಥಿತಿ GGML_ET_UBERKERNEL_KERNEL_GLU_F32:
                {
                    rc = glu_f32_ನಮೂದು((ರಚನೆ ggml_et_glu_params *) inst_params, env);
                    ನಿಲ್ಲಿಸು;
                }
            ಸ್ಥಿತಿ GGML_ET_UBERKERNEL_KERNEL_ROPE_F32:
                {
                    rc = rope_f32_ನಮೂದು((ರಚನೆ ggml_et_rope_params *) inst_params, env);
                    ನಿಲ್ಲಿಸು;
                }
            ಸ್ಥಿತಿ GGML_ET_UBERKERNEL_KERNEL_RMS_NORM_F32:
                {
                    // struct ggml_et_rms_norm_params *p = (struct ggml_et_rms_norm_params *) inst_params;
                    // evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
                    rc = rms_norm_f32_ನಮೂದು((ರಚನೆ ggml_et_rms_norm_params *) inst_params, env);
                    ನಿಲ್ಲಿಸು;
                }
            ಸ್ಥಿತಿ GGML_ET_UBERKERNEL_KERNEL_RMS_NORM_MUL_F32:
                {
                    ರಚನೆ uber_rms_norm_mul_params * p = (ರಚನೆ uber_rms_norm_mul_params *) inst_params;
                    evict_region_past_l2(p->src0.ದತ್ತಾಂಶ, tensor_ಬೈಟುಗಳು(&p->src0));
                    evict_region_past_l2(p->src1.ದತ್ತಾಂಶ, tensor_ಬೈಟುಗಳು(&p->src1));
                    rc = rms_norm_mul_f32_ನಮೂದು((ರಚನೆ ggml_et_rms_norm_mul_params *) inst_params, env);
                    ನಿಲ್ಲಿಸು;
                }
            ಸ್ಥಿತಿ GGML_ET_UBERKERNEL_KERNEL_SOFTMAX_F32:
                {
                    rc = softmax_f32_ನಮೂದು((ರಚನೆ ggml_et_softmax_params *) inst_params, env);
                    ನಿಲ್ಲಿಸು;
                }
            ಸ್ಥಿತಿ GGML_ET_UBERKERNEL_KERNEL_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_F32:
                {
                    rc = ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_f32_ನಮೂದು((ರಚನೆ ggml_et_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_params *) inst_params, env);
                    ನಿಲ್ಲಿಸು;
                }

            // Single-source ops (src0 → dst)
            ಸ್ಥಿತಿ GGML_ET_UBERKERNEL_KERNEL_SQR_F32:
                {
                    rc = sqr_f32_ನಮೂದು((ರಚನೆ ggml_et_sqr_params *) inst_params, env);
                    ನಿಲ್ಲಿಸು;
                }
            ಸ್ಥಿತಿ GGML_ET_UBERKERNEL_KERNEL_SCALE_F32:
                {
                    rc = scale_f32_ನಮೂದು((ರಚನೆ ggml_et_scale_params *) inst_params, env);
                    ನಿಲ್ಲಿಸು;
                }
            ಸ್ಥಿತಿ GGML_ET_UBERKERNEL_KERNEL_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_F32:
                {
                    rc = ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_f32_ನಮೂದು((ರಚನೆ ggml_et_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_params *) inst_params, env);
                    ನಿಲ್ಲಿಸು;
                }
            ಸ್ಥಿತಿ GGML_ET_UBERKERNEL_KERNEL_CUMSUM_F32:
                {
                    rc = cumsum_f32_ನಮೂದು((ರಚನೆ ggml_et_cumsum_params *) inst_params, env);
                    ನಿಲ್ಲಿಸು;
                }
            ಸ್ಥಿತಿ GGML_ET_UBERKERNEL_KERNEL_NORM_F32:
                {
                    rc = norm_f32_ನಮೂದು((ರಚನೆ ggml_et_norm_params *) inst_params, env);
                    ನಿಲ್ಲಿಸು;
                }
            ಸ್ಥಿತಿ GGML_ET_UBERKERNEL_KERNEL_L2_NORM_F32:
                {
                    rc = l2_norm_f32_ನಮೂದು((ರಚನೆ ggml_et_l2_norm_params *) inst_params, env);
                    ನಿಲ್ಲಿಸು;
                }
            ಸ್ಥಿತಿ GGML_ET_UBERKERNEL_KERNEL_ಗುಂಪು_NORM_F32:
                {
                    rc = ಗುಂಪು_norm_f32_ನಮೂದು((ರಚನೆ ggml_et_ಗುಂಪು_norm_params *) inst_params, env);
                    ನಿಲ್ಲಿಸು;
                }
            ಸ್ಥಿತಿ GGML_ET_UBERKERNEL_KERNEL_REPEAT_F32:
                {
                    rc = repeat_f32_ನಮೂದು((ರಚನೆ ggml_et_repeat_params *) inst_params, env);
                    ನಿಲ್ಲಿಸು;
                }
            ಸ್ಥಿತಿ GGML_ET_UBERKERNEL_KERNEL_DIAG_F32:
                {
                    rc = diag_f32_ನಮೂದು((ರಚನೆ ggml_et_diag_params *) inst_params, env);
                    ನಿಲ್ಲಿಸು;
                }
            ಸ್ಥಿತಿ GGML_ET_UBERKERNEL_KERNEL_TRI_F32:
                {
                    rc = tri_f32_ನಮೂದು((ರಚನೆ ggml_et_tri_params *) inst_params, env);
                    ನಿಲ್ಲಿಸು;
                }
            ಸ್ಥಿತಿ GGML_ET_UBERKERNEL_KERNEL_PAD_F32:
                {
                    rc = pad_f32_ನಮೂದು((ರಚನೆ ggml_et_pad_params *) inst_params, env);
                    ನಿಲ್ಲಿಸು;
                }
            ಸ್ಥಿತಿ GGML_ET_UBERKERNEL_KERNEL_CONT_F16:
                {
                    rc = cont_f16_ನಮೂದು((ರಚನೆ ggml_et_cont_params *) inst_params, env);
                    ನಿಲ್ಲಿಸು;
                }
            ಸ್ಥಿತಿ GGML_ET_UBERKERNEL_KERNEL_FILL_F32:
                {
                    rc = fill_f32_ನಮೂದು((ರಚನೆ ggml_et_fill_params *) inst_params, env);
                    ನಿಲ್ಲಿಸು;
                }
            ಸ್ಥಿತಿ GGML_ET_UBERKERNEL_KERNEL_ಗಣ_F32:
                {
                    rc = ಗಣ_f32_ನಮೂದು((ರಚನೆ ggml_et_ಗಣ_params *) inst_params, env);
                    ನಿಲ್ಲಿಸು;
                }

            // Two-source ops
            ಸ್ಥಿತಿ GGML_ET_UBERKERNEL_KERNEL_CONCAT_F32:
                {
                    rc = concat_f32_ನಮೂದು((ರಚನೆ ggml_et_concat_params *) inst_params, env);
                    ನಿಲ್ಲಿಸು;
                }
            // case GGML_ET_UBERKERNEL_KERNEL_SSM_CONV_F32: {
            //     rc = ssm_conv_f32_entry((struct ggml_et_ssm_conv_params *) inst_params, env);
            //     break;
            // }
            ಸ್ಥಿತಿ GGML_ET_UBERKERNEL_KERNEL_SOLVE_TRI_F32:
                {
                    rc = solve_tri_f32_ನಮೂದು((ರಚನೆ ggml_et_solve_tri_params *) inst_params, env);
                    ನಿಲ್ಲಿಸು;
                }
            ಸ್ಥಿತಿ GGML_ET_UBERKERNEL_KERNEL_IM2_ಲಂಬಸಾಲು:
                {
                    rc = im2col_ನಮೂದು((ರಚನೆ ggml_et_im2col_params *) inst_params, env);
                    ನಿಲ್ಲಿಸು;
                }

            // Three-source ops
            ಸ್ಥಿತಿ GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_ID_F32:
                {
                    rc = mul_mat_id_f32_ನಮೂದು((ರಚನೆ ggml_et_mul_mat_id_params *) inst_params, env);
                    ನಿಲ್ಲಿಸು;
                }
            ಸ್ಥಿತಿ GGML_ET_UBERKERNEL_KERNEL_FLASH_ATTN_EXT_F32:
                {
                    rc = flash_attn_ext_f32_ನಮೂದು((ರಚನೆ ggml_et_flash_attn_ext_params *) inst_params, env);
                    ನಿಲ್ಲಿಸು;
                }
            ಸ್ಥಿತಿ GGML_ET_UBERKERNEL_KERNEL_FLASH_ATTN_EXT_F16_ME:
                {
                    rc = flash_attn_ext_f16_me_ನಮೂದು((ರಚನೆ ggml_et_flash_attn_ext_params *) inst_params, env);
                    ನಿಲ್ಲಿಸು;
                }

            ಸ್ಥಿತಿ GGML_ET_UBERKERNEL_KERNEL_GATED_DELTA_NET_F32:
                {
                    rc = gated_delta_net_f32_ನಮೂದು((ರಚನೆ ggml_et_gated_delta_net_params *) inst_params, env);
                    ನಿಲ್ಲಿಸು;
                }
            ಸ್ಥಿತಿ GGML_ET_UBERKERNEL_KERNEL_SSM_ಪರಿಶೀಲನೆ_F32:
                {
                    rc = ssm_ಪರಿಶೀಲನೆ_f32_ನಮೂದು((ರಚನೆ ggml_et_ssm_ಪರಿಶೀಲನೆ_params *) inst_params, env);
                    ನಿಲ್ಲಿಸು;
                }
            // rwkv: raw float* params, no ggml_tensor fields to evict via
            ಸ್ಥಿತಿ GGML_ET_UBERKERNEL_KERNEL_RWKV_WKV6_F32:
                {
                    rc = rwkv_wkv6_f32_ನಮೂದು((ರಚನೆ ggml_et_rwkv_wkv6_params *) inst_params, env);
                    ನಿಲ್ಲಿಸು;
                }

            ಸ್ಥಿತಿ GGML_ET_UBERKERNEL_KERNEL_RWKV_WKV7_F32:
                {
                    rc = rwkv_wkv7_f32_ನಮೂದು((ರಚನೆ ggml_et_rwkv_wkv7_params *) inst_params, env);
                    ನಿಲ್ಲಿಸು;
                }

            // MUL_MAT: evict src1 (activations); src0=weights is
            //  read-only so never stale from a prior uberkernel op
            ಸ್ಥಿತಿ GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F16:
                {
                    ರಚನೆ ggml_et_ದ್ವಿಮಾನ_params * p = (ರಚನೆ ggml_et_ದ್ವಿಮಾನ_params *) inst_params;
                    rc                               = mul_mat_f16_ನಮೂದು(p, env);
                    ನಿಲ್ಲಿಸು;
                }
            ಸ್ಥಿತಿ GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F16_ಮಾತೃಕೆ_ENGINE:
                {
                    ರಚನೆ ggml_et_ದ್ವಿಮಾನ_params * p = (ರಚನೆ ggml_et_ದ್ವಿಮಾನ_params *) inst_params;
                    rc                               = mul_mat_f16_ಮಾತೃಕೆ_engine_ನಮೂದು(p, env);
                    ನಿಲ್ಲಿಸು;
                }
            ಸ್ಥಿತಿ GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F32:
                {
                    ರಚನೆ ggml_et_ದ್ವಿಮಾನ_params * p = (ರಚನೆ ggml_et_ದ್ವಿಮಾನ_params *) inst_params;
                    rc                               = mul_mat_f32_ನಮೂದು(p, env);
                    ನಿಲ್ಲಿಸು;
                }
            ಸ್ಥಿತಿ GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F32_ಮಾತೃಕೆ_ENGINE:
                {
                    ರಚನೆ ggml_et_ದ್ವಿಮಾನ_params * p = (ರಚನೆ ggml_et_ದ್ವಿಮಾನ_params *) inst_params;
                    rc                               = mul_mat_f32_ಮಾತೃಕೆ_engine_ನಮೂದು(p, env);
                    ನಿಲ್ಲಿಸು;
                }
            ಸ್ಥಿತಿ GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_Q8_0:
                {
                    ರಚನೆ ggml_et_mm_q8_params * p = (ರಚನೆ ggml_et_mm_q8_params *) inst_params;
                    // evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
                    rc                              = mul_mat_Q8_0_ನಮೂದು(p, env);
                    ನಿಲ್ಲಿಸು;
                }
            ಸ್ಥಿತಿ GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_Q4_0:
                {
                    ರಚನೆ ggml_et_ದ್ವಿಮಾನ_params * p = (ರಚನೆ ggml_et_ದ್ವಿಮಾನ_params *) inst_params;
                    rc                               = mul_mat_Q4_0_ನಮೂದು(p, env);
                    ನಿಲ್ಲಿಸು;
                }

            ಪೂರ್ವನಿಯೋಜಿತ:
                ಹಿಂತಿರುಗಿ -1;
        }

        ಆದರೆ (rc != 0) {
            ಹಿಂತಿರುಗಿ rc;
        }
    }

    ಹಿಂತಿರುಗಿ 0;
}
