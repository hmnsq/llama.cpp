#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-et-uberkernel-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-et-uberkernel-kernel-map.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml_tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "math_fp.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "platform.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

ਬਣਤਰ ggml_et_glu_params;
ਬਣਤਰ ggml_et_unary_params;
ਬਣਤਰ ggml_et_rope_params;
ਬਣਤਰ ggml_et_rms_norm_params;
ਬਣਤਰ ggml_et_rms_norm_mul_params;
ਬਣਤਰ ggml_et_softmax_params;
ਬਣਤਰ ggml_et_ਸੈੱਟ_ਪੰਕਤੀਆਂ_params;
ਬਣਤਰ ggml_et_get_ਪੰਕਤੀਆਂ_params;
ਬਣਤਰ ggml_et_cont_params;
ਬਣਤਰ ggml_et_concat_params;
ਬਣਤਰ ggml_et_cumsum_params;
ਬਣਤਰ ggml_et_diag_params;
ਬਣਤਰ ggml_et_fill_params;
ਬਣਤਰ ggml_et_flash_attn_ext_params;
ਬਣਤਰ ggml_et_gated_delta_net_params;
ਬਣਤਰ ggml_et_ਸਮੂਹ_norm_params;
ਬਣਤਰ ggml_et_im2col_params;
ਬਣਤਰ ggml_et_l2_norm_params;
ਬਣਤਰ ggml_et_mul_mat_id_params;
ਬਣਤਰ ggml_et_norm_params;
ਬਣਤਰ ggml_et_pad_params;
ਬਣਤਰ ggml_et_repeat_params;
ਬਣਤਰ ggml_et_rwkv_wkv6_params;
ਬਣਤਰ ggml_et_rwkv_wkv7_params;
ਬਣਤਰ ggml_et_scale_params;
ਬਣਤਰ ggml_et_ਸੈੱਟ_params;
ਬਣਤਰ ggml_et_solve_tri_params;
ਬਣਤਰ ggml_et_sqr_params;
ਬਣਤਰ ggml_et_ssm_conv_params;
ਬਣਤਰ ggml_et_ssm_ਜਾਂਚ_params;
ਬਣਤਰ ggml_et_ਜੋੜ_ਪੰਕਤੀਆਂ_params;
ਬਣਤਰ ggml_et_tri_params;

ਬਾਹਰੀ ਪੂਰਨ el_map_f32_ਇੰਦਰਾਜ(ਬਣਤਰ ggml_et_ਦੁਆਧਾਰੀ_params *, ਖਾਲੀ *);
ਬਾਹਰੀ ਪੂਰਨ glu_f32_ਇੰਦਰਾਜ(ਬਣਤਰ ggml_et_glu_params *, ਖਾਲੀ *);
ਬਾਹਰੀ ਪੂਰਨ unary_f32_ਇੰਦਰਾਜ(ਬਣਤਰ ggml_et_unary_params *, ਖਾਲੀ *);
ਬਾਹਰੀ ਪੂਰਨ rope_f32_ਇੰਦਰਾਜ(ਬਣਤਰ ggml_et_rope_params *, ਖਾਲੀ *);
ਬਾਹਰੀ ਪੂਰਨ rms_norm_f32_ਇੰਦਰਾਜ(ਬਣਤਰ ggml_et_rms_norm_params *, ਖਾਲੀ *);
ਬਾਹਰੀ ਪੂਰਨ rms_norm_mul_f32_ਇੰਦਰਾਜ(ਬਣਤਰ ggml_et_rms_norm_mul_params *, ਖਾਲੀ *);
ਬਾਹਰੀ ਪੂਰਨ softmax_f32_ਇੰਦਰਾਜ(ਬਣਤਰ ggml_et_softmax_params *, ਖਾਲੀ *);
ਬਾਹਰੀ ਪੂਰਨ ਸੈੱਟ_ਪੰਕਤੀਆਂ_f32_ਇੰਦਰਾਜ(ਬਣਤਰ ggml_et_ਸੈੱਟ_ਪੰਕਤੀਆਂ_params *, ਖਾਲੀ *);
ਬਾਹਰੀ ਪੂਰਨ get_ਪੰਕਤੀਆਂ_f32_ਇੰਦਰਾਜ(ਬਣਤਰ ggml_et_get_ਪੰਕਤੀਆਂ_params *, ਖਾਲੀ *);
ਬਾਹਰੀ ਪੂਰਨ cont_f32_ਇੰਦਰਾਜ(ਬਣਤਰ ggml_et_cont_params *, ਖਾਲੀ *);
ਬਾਹਰੀ ਪੂਰਨ cont_f16_ਇੰਦਰਾਜ(ਬਣਤਰ ggml_et_cont_params *, ਖਾਲੀ *);
ਬਾਹਰੀ ਪੂਰਨ cpy_f32_f16_ਇੰਦਰਾਜ(ਬਣਤਰ ggml_et_cont_params *, ਖਾਲੀ *);
ਬਾਹਰੀ ਪੂਰਨ concat_f32_ਇੰਦਰਾਜ(ਬਣਤਰ ggml_et_concat_params *, ਖਾਲੀ *);
ਬਾਹਰੀ ਪੂਰਨ cumsum_f32_ਇੰਦਰਾਜ(ਬਣਤਰ ggml_et_cumsum_params *, ਖਾਲੀ *);
ਬਾਹਰੀ ਪੂਰਨ diag_f32_ਇੰਦਰਾਜ(ਬਣਤਰ ggml_et_diag_params *, ਖਾਲੀ *);
ਬਾਹਰੀ ਪੂਰਨ fill_f32_ਇੰਦਰਾਜ(ਬਣਤਰ ggml_et_fill_params *, ਖਾਲੀ *);
ਬਾਹਰੀ ਪੂਰਨ flash_attn_ext_f32_ਇੰਦਰਾਜ(ਬਣਤਰ ggml_et_flash_attn_ext_params *, ਖਾਲੀ *);
ਬਾਹਰੀ ਪੂਰਨ flash_attn_ext_f16_me_ਇੰਦਰਾਜ(ਬਣਤਰ ggml_et_flash_attn_ext_params *, ਖਾਲੀ *);
ਬਾਹਰੀ ਪੂਰਨ gated_delta_net_f32_ਇੰਦਰਾਜ(ਬਣਤਰ ggml_et_gated_delta_net_params *, ਖਾਲੀ *);
ਬਾਹਰੀ ਪੂਰਨ ਸਮੂਹ_norm_f32_ਇੰਦਰਾਜ(ਬਣਤਰ ggml_et_ਸਮੂਹ_norm_params *, ਖਾਲੀ *);
ਬਾਹਰੀ ਪੂਰਨ im2col_ਇੰਦਰਾਜ(ਬਣਤਰ ggml_et_im2col_params *, ਖਾਲੀ *);
ਬਾਹਰੀ ਪੂਰਨ l2_norm_f32_ਇੰਦਰਾਜ(ਬਣਤਰ ggml_et_l2_norm_params *, ਖਾਲੀ *);
ਬਾਹਰੀ ਪੂਰਨ mul_mat_id_f32_ਇੰਦਰਾਜ(ਬਣਤਰ ggml_et_mul_mat_id_params *, ਖਾਲੀ *);
ਬਾਹਰੀ ਪੂਰਨ norm_f32_ਇੰਦਰਾਜ(ਬਣਤਰ ggml_et_norm_params *, ਖਾਲੀ *);
ਬਾਹਰੀ ਪੂਰਨ pad_f32_ਇੰਦਰਾਜ(ਬਣਤਰ ggml_et_pad_params *, ਖਾਲੀ *);
ਬਾਹਰੀ ਪੂਰਨ repeat_f32_ਇੰਦਰਾਜ(ਬਣਤਰ ggml_et_repeat_params *, ਖਾਲੀ *);
ਬਾਹਰੀ ਪੂਰਨ rwkv_wkv6_f32_ਇੰਦਰਾਜ(ਬਣਤਰ ggml_et_rwkv_wkv6_params *, ਖਾਲੀ *);
ਬਾਹਰੀ ਪੂਰਨ rwkv_wkv7_f32_ਇੰਦਰਾਜ(ਬਣਤਰ ggml_et_rwkv_wkv7_params *, ਖਾਲੀ *);
ਬਾਹਰੀ ਪੂਰਨ scale_f32_ਇੰਦਰਾਜ(ਬਣਤਰ ggml_et_scale_params *, ਖਾਲੀ *);
ਬਾਹਰੀ ਪੂਰਨ ਸੈੱਟ_f32_ਇੰਦਰਾਜ(ਬਣਤਰ ggml_et_ਸੈੱਟ_params *, ਖਾਲੀ *);
ਬਾਹਰੀ ਪੂਰਨ solve_tri_f32_ਇੰਦਰਾਜ(ਬਣਤਰ ggml_et_solve_tri_params *, ਖਾਲੀ *);
ਬਾਹਰੀ ਪੂਰਨ sqr_f32_ਇੰਦਰਾਜ(ਬਣਤਰ ggml_et_sqr_params *, ਖਾਲੀ *);
ਬਾਹਰੀ ਪੂਰਨ ssm_conv_f32_ਇੰਦਰਾਜ(ਬਣਤਰ ggml_et_ssm_conv_params *, ਖਾਲੀ *);
ਬਾਹਰੀ ਪੂਰਨ ssm_ਜਾਂਚ_f32_ਇੰਦਰਾਜ(ਬਣਤਰ ggml_et_ssm_ਜਾਂਚ_params *, ਖਾਲੀ *);
ਬਾਹਰੀ ਪੂਰਨ ਜੋੜ_ਪੰਕਤੀਆਂ_f32_ਇੰਦਰਾਜ(ਬਣਤਰ ggml_et_ਜੋੜ_ਪੰਕਤੀਆਂ_params *, ਖਾਲੀ *);
ਬਾਹਰੀ ਪੂਰਨ tri_f32_ਇੰਦਰਾਜ(ਬਣਤਰ ggml_et_tri_params *, ਖਾਲੀ *);
ਬਾਹਰੀ ਪੂਰਨ mul_mat_f16_ਇੰਦਰਾਜ(ਬਣਤਰ ggml_et_ਦੁਆਧਾਰੀ_params *, ਖਾਲੀ *);
ਬਾਹਰੀ ਪੂਰਨ mul_mat_f16_ਮੈਟ੍ਰਿਕਸ_engine_ਇੰਦਰਾਜ(ਬਣਤਰ ggml_et_ਦੁਆਧਾਰੀ_params *, ਖਾਲੀ *);
ਬਾਹਰੀ ਪੂਰਨ mul_mat_f32_ਇੰਦਰਾਜ(ਬਣਤਰ ggml_et_ਦੁਆਧਾਰੀ_params *, ਖਾਲੀ *);
ਬਾਹਰੀ ਪੂਰਨ mul_mat_f32_ਮੈਟ੍ਰਿਕਸ_engine_ਇੰਦਰਾਜ(ਬਣਤਰ ggml_et_ਦੁਆਧਾਰੀ_params *, ਖਾਲੀ *);
ਬਾਹਰੀ ਪੂਰਨ mul_mat_Q8_0_ਇੰਦਰਾਜ(ਬਣਤਰ ggml_et_mm_q8_params *, ਖਾਲੀ *);
ਬਾਹਰੀ ਪੂਰਨ mul_mat_Q4_0_ਇੰਦਰਾਜ(ਬਣਤਰ ggml_et_ਦੁਆਧਾਰੀ_params *, ਖਾਲੀ *);

ਸਥਿਰ ਸਤਰਅੰਦਰ size_t tensor_ਬਾਈਟਾਂ(ਅਚਲ ਬਣਤਰ ggml_tensor * t) {
    ਵਾਪਸ (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

ਬਣਤਰ uber_glu_params {
    ਬਣਤਰ ggml_tensor src0;
    ਬਣਤਰ ggml_tensor src1;
    ਬਣਤਰ ggml_tensor dst;
    // trailing scalars omitted — not needed for eviction
};

ਬਣਤਰ uber_unary_params {
    ਬਣਤਰ ggml_tensor src0;
    ਬਣਤਰ ggml_tensor dst;
};

ਬਣਤਰ uber_rope_params {
    ਬਣਤਰ ggml_tensor src0;
    ਬਣਤਰ ggml_tensor src1;
    ਬਣਤਰ ggml_tensor src2;
    ਬਣਤਰ ggml_tensor dst;
};

ਬਣਤਰ uber_rms_norm_params {
    ਬਣਤਰ ggml_tensor src0;
    ਬਣਤਰ ggml_tensor dst;
};

ਬਣਤਰ uber_rms_norm_mul_params {
    ਬਣਤਰ ggml_tensor src0;
    ਬਣਤਰ ggml_tensor src1;
    ਬਣਤਰ ggml_tensor dst;
};

ਬਣਤਰ uber_softmax_params {
    ਬਣਤਰ ggml_tensor src0;
    ਬਣਤਰ ggml_tensor src1;
    ਬਣਤਰ ggml_tensor src2;
    ਬਣਤਰ ggml_tensor dst;
};

ਬਣਤਰ uber_ਸੈੱਟ_ਪੰਕਤੀਆਂ_params {
    ਬਣਤਰ ggml_tensor src0;
    ਬਣਤਰ ggml_tensor src1;
    ਬਣਤਰ ggml_tensor dst;
};

ਬਣਤਰ uber_get_ਪੰਕਤੀਆਂ_params {
    ਬਣਤਰ ggml_tensor src0;
    ਬਣਤਰ ggml_tensor src1;
    ਬਣਤਰ ggml_tensor dst;
};

ਬਣਤਰ uber_cont_params {
    ਬਣਤਰ ggml_tensor src0;
    ਬਣਤਰ ggml_tensor dst;
};

// src0 + src1 + dst (no trailing scalars needed for eviction)
ਬਣਤਰ uber_concat_params {
    ਬਣਤਰ ggml_tensor src0;
    ਬਣਤਰ ggml_tensor src1;
    ਬਣਤਰ ggml_tensor dst;
};

ਬਣਤਰ uber_ssm_conv_params {
    ਬਣਤਰ ggml_tensor src0;
    ਬਣਤਰ ggml_tensor src1;
    ਬਣਤਰ ggml_tensor dst;
};

ਬਣਤਰ uber_solve_tri_params {
    ਬਣਤਰ ggml_tensor src0;
    ਬਣਤਰ ggml_tensor src1;
    ਬਣਤਰ ggml_tensor dst;
};

ਬਣਤਰ uber_mul_mat_id_params {
    ਬਣਤਰ ggml_tensor src0;
    ਬਣਤਰ ggml_tensor src1;
    ਬਣਤਰ ggml_tensor src2;
    ਬਣਤਰ ggml_tensor dst;
};

// flash_attn_ext: Q=src0, K=src1, V=src2, mask=src3, dst (mask optional)
ਬਣਤਰ uber_flash_attn_ext_params {
    ਬਣਤਰ ggml_tensor src0;
    ਬਣਤਰ ggml_tensor src1;
    ਬਣਤਰ ggml_tensor src2;
    ਬਣਤਰ ggml_tensor ਨਕਾਬ;
    ਬਣਤਰ ggml_tensor dst;
};

// ssm_scan: 7 source tensors + dst
ਬਣਤਰ uber_ssm_ਜਾਂਚ_params {
    ਬਣਤਰ ggml_tensor src0;
    ਬਣਤਰ ggml_tensor src1;
    ਬਣਤਰ ggml_tensor src2;
    ਬਣਤਰ ggml_tensor src3;
    ਬਣਤਰ ggml_tensor src4;
    ਬਣਤਰ ggml_tensor src5;
    ਬਣਤਰ ggml_tensor src6;
    ਬਣਤਰ ggml_tensor dst;
};

// gated_delta_net: q,k,v,g,beta,state_in,dst
ਬਣਤਰ uber_gated_delta_net_params {
    ਬਣਤਰ ggml_tensor q;
    ਬਣਤਰ ggml_tensor k;
    ਬਣਤਰ ggml_tensor v;
    ਬਣਤਰ ggml_tensor g;
    ਬਣਤਰ ggml_tensor beta;
    ਬਣਤਰ ggml_tensor state_ਵਿੱਚ;
    ਬਣਤਰ ggml_tensor dst;
};

ਸਥਿਰ ਖਾਲੀ copy_f32_ਮੰਜ਼ਿਲ_f16_ਪੰਕਤੀ(uint16_t * dst, ਅਚਲ ਦਸ਼ਮਲਵ * src, int64_t ਅੰਕ_elements) {
    ਲਈ (int64_t i = 0; i < ਅੰਕ_elements; i++) {
        dst[i] = fp32_ਮੰਜ਼ਿਲ_fp16(src[i]);
    }
}

ਸਥਿਰ ਖਾਲੀ copy_f32_ਪੰਕਤੀ(ਦਸ਼ਮਲਵ * dst, ਅਚਲ ਦਸ਼ਮਲਵ * src, int64_t ਅੰਕ_elements) {
    ਲਈ (int64_t i = 0; i < ਅੰਕ_elements; i++) {
        dst[i] = src[i];
    }
}

ਸਥਿਰ ਖਾਲੀ evict_region_past_l2_local(ਅਚਲ ਖਾਲੀ * addr, size_t ਬਾਈਟਾਂ) {
    ਜੇ (!addr || ਬਾਈਟਾਂ == 0) {
        ਵਾਪਸ;
    }

    ਅਚਲ uint64_t CL     = 64;
    uint64_t       ਅਧਾਰ   = (uint64_t) addr & ~(CL - 1);
    uint64_t       ਸਿਰਾ    = ((uint64_t) addr + ਬਾਈਟਾਂ + CL - 1) & ~(CL - 1);
    uint64_t       nlines = (ਸਿਰਾ - ਅਧਾਰ) / CL;
    cache_ops_priv_evict_sw(0, /*to_L2*/ 3, 0, 0, CL);
}

ਪੂਰਨ ਇੰਦਰਾਜ_ਬਿੰਦੂ(ਬਣਤਰ ggml_et_uberkernel_params * params, ਖਾਲੀ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ਜੇ (!kernel_env || !params) {
        ਵਾਪਸ -1;
    }

    ਬਣਤਰ ggml_et_uberkernel_inst * insts       = (ਬਣਤਰ ggml_et_uberkernel_inst *) (uintptr_t) params->insts;
    uint8_t *                        params_blob = (uint8_t *) (uintptr_t) params->params_blob;

    ਜੇ (!insts || !params_blob || params->inst_stride < ਆਕਾਰ(ਬਣਤਰ ggml_et_uberkernel_inst)) {
        ਵਾਪਸ -1;
    }

    ਲਈ (uint32_t i = 0; i < params->ਅੰਕ_insts; ++i) {
        ਬਣਤਰ ggml_et_uberkernel_inst * inst =
            (ਬਣਤਰ ggml_et_uberkernel_inst *) ((uint8_t *) insts + (i * params->inst_stride));
        ਖਾਲੀ * inst_params = params_blob + inst->params_ਵਿੱਥ;
        ਪੂਰਨ    rc          = -1;

        et_barrier_global(32ULL);

        ਚੋਣ (inst->kernel_id) {
            ਹਾਲਤ GGML_ET_UBERKERNEL_KERNEL_EL_MAP_F32:
                {
                    ਬਣਤਰ ggml_et_ਦੁਆਧਾਰੀ_params * p = (ਬਣਤਰ ggml_et_ਦੁਆਧਾਰੀ_params *) inst_params;
                    rc                               = el_map_f32_ਇੰਦਰਾਜ(p, env);
                    ਰੋਕੋ;
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
            ਹਾਲਤ GGML_ET_UBERKERNEL_KERNEL_GLU_F32:
                {
                    rc = glu_f32_ਇੰਦਰਾਜ((ਬਣਤਰ ggml_et_glu_params *) inst_params, env);
                    ਰੋਕੋ;
                }
            ਹਾਲਤ GGML_ET_UBERKERNEL_KERNEL_ROPE_F32:
                {
                    rc = rope_f32_ਇੰਦਰਾਜ((ਬਣਤਰ ggml_et_rope_params *) inst_params, env);
                    ਰੋਕੋ;
                }
            ਹਾਲਤ GGML_ET_UBERKERNEL_KERNEL_RMS_NORM_F32:
                {
                    // struct ggml_et_rms_norm_params *p = (struct ggml_et_rms_norm_params *) inst_params;
                    // evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
                    rc = rms_norm_f32_ਇੰਦਰਾਜ((ਬਣਤਰ ggml_et_rms_norm_params *) inst_params, env);
                    ਰੋਕੋ;
                }
            ਹਾਲਤ GGML_ET_UBERKERNEL_KERNEL_RMS_NORM_MUL_F32:
                {
                    ਬਣਤਰ uber_rms_norm_mul_params * p = (ਬਣਤਰ uber_rms_norm_mul_params *) inst_params;
                    evict_region_past_l2(p->src0.ਅੰਕੜੇ, tensor_ਬਾਈਟਾਂ(&p->src0));
                    evict_region_past_l2(p->src1.ਅੰਕੜੇ, tensor_ਬਾਈਟਾਂ(&p->src1));
                    rc = rms_norm_mul_f32_ਇੰਦਰਾਜ((ਬਣਤਰ ggml_et_rms_norm_mul_params *) inst_params, env);
                    ਰੋਕੋ;
                }
            ਹਾਲਤ GGML_ET_UBERKERNEL_KERNEL_SOFTMAX_F32:
                {
                    rc = softmax_f32_ਇੰਦਰਾਜ((ਬਣਤਰ ggml_et_softmax_params *) inst_params, env);
                    ਰੋਕੋ;
                }
            ਹਾਲਤ GGML_ET_UBERKERNEL_KERNEL_ਸੈੱਟ_ਪੰਕਤੀਆਂ_F32:
                {
                    rc = ਸੈੱਟ_ਪੰਕਤੀਆਂ_f32_ਇੰਦਰਾਜ((ਬਣਤਰ ggml_et_ਸੈੱਟ_ਪੰਕਤੀਆਂ_params *) inst_params, env);
                    ਰੋਕੋ;
                }

            // Single-source ops (src0 → dst)
            ਹਾਲਤ GGML_ET_UBERKERNEL_KERNEL_SQR_F32:
                {
                    rc = sqr_f32_ਇੰਦਰਾਜ((ਬਣਤਰ ggml_et_sqr_params *) inst_params, env);
                    ਰੋਕੋ;
                }
            ਹਾਲਤ GGML_ET_UBERKERNEL_KERNEL_SCALE_F32:
                {
                    rc = scale_f32_ਇੰਦਰਾਜ((ਬਣਤਰ ggml_et_scale_params *) inst_params, env);
                    ਰੋਕੋ;
                }
            ਹਾਲਤ GGML_ET_UBERKERNEL_KERNEL_ਜੋੜ_ਪੰਕਤੀਆਂ_F32:
                {
                    rc = ਜੋੜ_ਪੰਕਤੀਆਂ_f32_ਇੰਦਰਾਜ((ਬਣਤਰ ggml_et_ਜੋੜ_ਪੰਕਤੀਆਂ_params *) inst_params, env);
                    ਰੋਕੋ;
                }
            ਹਾਲਤ GGML_ET_UBERKERNEL_KERNEL_CUMSUM_F32:
                {
                    rc = cumsum_f32_ਇੰਦਰਾਜ((ਬਣਤਰ ggml_et_cumsum_params *) inst_params, env);
                    ਰੋਕੋ;
                }
            ਹਾਲਤ GGML_ET_UBERKERNEL_KERNEL_NORM_F32:
                {
                    rc = norm_f32_ਇੰਦਰਾਜ((ਬਣਤਰ ggml_et_norm_params *) inst_params, env);
                    ਰੋਕੋ;
                }
            ਹਾਲਤ GGML_ET_UBERKERNEL_KERNEL_L2_NORM_F32:
                {
                    rc = l2_norm_f32_ਇੰਦਰਾਜ((ਬਣਤਰ ggml_et_l2_norm_params *) inst_params, env);
                    ਰੋਕੋ;
                }
            ਹਾਲਤ GGML_ET_UBERKERNEL_KERNEL_ਸਮੂਹ_NORM_F32:
                {
                    rc = ਸਮੂਹ_norm_f32_ਇੰਦਰਾਜ((ਬਣਤਰ ggml_et_ਸਮੂਹ_norm_params *) inst_params, env);
                    ਰੋਕੋ;
                }
            ਹਾਲਤ GGML_ET_UBERKERNEL_KERNEL_REPEAT_F32:
                {
                    rc = repeat_f32_ਇੰਦਰਾਜ((ਬਣਤਰ ggml_et_repeat_params *) inst_params, env);
                    ਰੋਕੋ;
                }
            ਹਾਲਤ GGML_ET_UBERKERNEL_KERNEL_DIAG_F32:
                {
                    rc = diag_f32_ਇੰਦਰਾਜ((ਬਣਤਰ ggml_et_diag_params *) inst_params, env);
                    ਰੋਕੋ;
                }
            ਹਾਲਤ GGML_ET_UBERKERNEL_KERNEL_TRI_F32:
                {
                    rc = tri_f32_ਇੰਦਰਾਜ((ਬਣਤਰ ggml_et_tri_params *) inst_params, env);
                    ਰੋਕੋ;
                }
            ਹਾਲਤ GGML_ET_UBERKERNEL_KERNEL_PAD_F32:
                {
                    rc = pad_f32_ਇੰਦਰਾਜ((ਬਣਤਰ ggml_et_pad_params *) inst_params, env);
                    ਰੋਕੋ;
                }
            ਹਾਲਤ GGML_ET_UBERKERNEL_KERNEL_CONT_F16:
                {
                    rc = cont_f16_ਇੰਦਰਾਜ((ਬਣਤਰ ggml_et_cont_params *) inst_params, env);
                    ਰੋਕੋ;
                }
            ਹਾਲਤ GGML_ET_UBERKERNEL_KERNEL_FILL_F32:
                {
                    rc = fill_f32_ਇੰਦਰਾਜ((ਬਣਤਰ ggml_et_fill_params *) inst_params, env);
                    ਰੋਕੋ;
                }
            ਹਾਲਤ GGML_ET_UBERKERNEL_KERNEL_ਸੈੱਟ_F32:
                {
                    rc = ਸੈੱਟ_f32_ਇੰਦਰਾਜ((ਬਣਤਰ ggml_et_ਸੈੱਟ_params *) inst_params, env);
                    ਰੋਕੋ;
                }

            // Two-source ops
            ਹਾਲਤ GGML_ET_UBERKERNEL_KERNEL_CONCAT_F32:
                {
                    rc = concat_f32_ਇੰਦਰਾਜ((ਬਣਤਰ ggml_et_concat_params *) inst_params, env);
                    ਰੋਕੋ;
                }
            // case GGML_ET_UBERKERNEL_KERNEL_SSM_CONV_F32: {
            //     rc = ssm_conv_f32_entry((struct ggml_et_ssm_conv_params *) inst_params, env);
            //     break;
            // }
            ਹਾਲਤ GGML_ET_UBERKERNEL_KERNEL_SOLVE_TRI_F32:
                {
                    rc = solve_tri_f32_ਇੰਦਰਾਜ((ਬਣਤਰ ggml_et_solve_tri_params *) inst_params, env);
                    ਰੋਕੋ;
                }
            ਹਾਲਤ GGML_ET_UBERKERNEL_KERNEL_IM2_ਖਾਨਾ:
                {
                    rc = im2col_ਇੰਦਰਾਜ((ਬਣਤਰ ggml_et_im2col_params *) inst_params, env);
                    ਰੋਕੋ;
                }

            // Three-source ops
            ਹਾਲਤ GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_ID_F32:
                {
                    rc = mul_mat_id_f32_ਇੰਦਰਾਜ((ਬਣਤਰ ggml_et_mul_mat_id_params *) inst_params, env);
                    ਰੋਕੋ;
                }
            ਹਾਲਤ GGML_ET_UBERKERNEL_KERNEL_FLASH_ATTN_EXT_F32:
                {
                    rc = flash_attn_ext_f32_ਇੰਦਰਾਜ((ਬਣਤਰ ggml_et_flash_attn_ext_params *) inst_params, env);
                    ਰੋਕੋ;
                }
            ਹਾਲਤ GGML_ET_UBERKERNEL_KERNEL_FLASH_ATTN_EXT_F16_ME:
                {
                    rc = flash_attn_ext_f16_me_ਇੰਦਰਾਜ((ਬਣਤਰ ggml_et_flash_attn_ext_params *) inst_params, env);
                    ਰੋਕੋ;
                }

            ਹਾਲਤ GGML_ET_UBERKERNEL_KERNEL_GATED_DELTA_NET_F32:
                {
                    rc = gated_delta_net_f32_ਇੰਦਰਾਜ((ਬਣਤਰ ggml_et_gated_delta_net_params *) inst_params, env);
                    ਰੋਕੋ;
                }
            ਹਾਲਤ GGML_ET_UBERKERNEL_KERNEL_SSM_ਜਾਂਚ_F32:
                {
                    rc = ssm_ਜਾਂਚ_f32_ਇੰਦਰਾਜ((ਬਣਤਰ ggml_et_ssm_ਜਾਂਚ_params *) inst_params, env);
                    ਰੋਕੋ;
                }
            // rwkv: raw float* params, no ggml_tensor fields to evict via
            ਹਾਲਤ GGML_ET_UBERKERNEL_KERNEL_RWKV_WKV6_F32:
                {
                    rc = rwkv_wkv6_f32_ਇੰਦਰਾਜ((ਬਣਤਰ ggml_et_rwkv_wkv6_params *) inst_params, env);
                    ਰੋਕੋ;
                }

            ਹਾਲਤ GGML_ET_UBERKERNEL_KERNEL_RWKV_WKV7_F32:
                {
                    rc = rwkv_wkv7_f32_ਇੰਦਰਾਜ((ਬਣਤਰ ggml_et_rwkv_wkv7_params *) inst_params, env);
                    ਰੋਕੋ;
                }

            // MUL_MAT: evict src1 (activations); src0=weights is
            //  read-only so never stale from a prior uberkernel op
            ਹਾਲਤ GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F16:
                {
                    ਬਣਤਰ ggml_et_ਦੁਆਧਾਰੀ_params * p = (ਬਣਤਰ ggml_et_ਦੁਆਧਾਰੀ_params *) inst_params;
                    rc                               = mul_mat_f16_ਇੰਦਰਾਜ(p, env);
                    ਰੋਕੋ;
                }
            ਹਾਲਤ GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F16_ਮੈਟ੍ਰਿਕਸ_ENGINE:
                {
                    ਬਣਤਰ ggml_et_ਦੁਆਧਾਰੀ_params * p = (ਬਣਤਰ ggml_et_ਦੁਆਧਾਰੀ_params *) inst_params;
                    rc                               = mul_mat_f16_ਮੈਟ੍ਰਿਕਸ_engine_ਇੰਦਰਾਜ(p, env);
                    ਰੋਕੋ;
                }
            ਹਾਲਤ GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F32:
                {
                    ਬਣਤਰ ggml_et_ਦੁਆਧਾਰੀ_params * p = (ਬਣਤਰ ggml_et_ਦੁਆਧਾਰੀ_params *) inst_params;
                    rc                               = mul_mat_f32_ਇੰਦਰਾਜ(p, env);
                    ਰੋਕੋ;
                }
            ਹਾਲਤ GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F32_ਮੈਟ੍ਰਿਕਸ_ENGINE:
                {
                    ਬਣਤਰ ggml_et_ਦੁਆਧਾਰੀ_params * p = (ਬਣਤਰ ggml_et_ਦੁਆਧਾਰੀ_params *) inst_params;
                    rc                               = mul_mat_f32_ਮੈਟ੍ਰਿਕਸ_engine_ਇੰਦਰਾਜ(p, env);
                    ਰੋਕੋ;
                }
            ਹਾਲਤ GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_Q8_0:
                {
                    ਬਣਤਰ ggml_et_mm_q8_params * p = (ਬਣਤਰ ggml_et_mm_q8_params *) inst_params;
                    // evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
                    rc                              = mul_mat_Q8_0_ਇੰਦਰਾਜ(p, env);
                    ਰੋਕੋ;
                }
            ਹਾਲਤ GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_Q4_0:
                {
                    ਬਣਤਰ ggml_et_ਦੁਆਧਾਰੀ_params * p = (ਬਣਤਰ ggml_et_ਦੁਆਧਾਰੀ_params *) inst_params;
                    rc                               = mul_mat_Q4_0_ਇੰਦਰਾਜ(p, env);
                    ਰੋਕੋ;
                }

            ਮੂਲ:
                ਵਾਪਸ -1;
        }

        ਜੇ (rc != 0) {
            ਵਾਪਸ rc;
        }
    }

    ਵਾਪਸ 0;
}
