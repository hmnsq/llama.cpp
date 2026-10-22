#ಸೂಚನೆ once

#ಅನ್ತರ್ಭಾವ "ggml-metal-device.h"

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __cplusplus
ಬಾಹ್ಯ "C" {
#ಆದರೆ_ಅಂತ್ಯ

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ ggml_metal_op * ggml_metal_op_t;

ರಚನೆ ggml_metal_fusion; // forward decl (ggml-metal-device.h)

ggml_metal_op_t ggml_metal_op_init(
        ggml_metal_device_t dev,
        ggml_metal_cmd_buf_t cmd_buf,
        ರಚನೆ ggml_cgraph * gf,
        ರಚನೆ ggml_metal_fusion_info * finfo,
        ಪೂರ್ಣಾಂಕ  idx_ಆರಂಭ,
        ಪೂರ್ಣಾಂಕ  idx_ಕೊನೆ,
        bool use_concurrency,
        bool use_capture,
        ಪೂರ್ಣಾಂಕ  debug_ನಕ್ಷೆ);

ಶೂನ್ಯ ggml_metal_op_free(ggml_metal_op_t ctx);

ಪೂರ್ಣಾಂಕ ggml_metal_op_n_nodes(ggml_metal_op_t ctx);

ಪೂರ್ಣಾಂಕ ggml_metal_op_encode(ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);

//
// available ops:
//

// tokens per expert
size_t ggml_metal_op_mul_mat_id_extra_tpe(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);

// id map [n_tokens, n_expert]
size_t ggml_metal_op_mul_mat_id_extra_ids(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);
size_t ggml_metal_op_mul_mat_id_extra_amax(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);

// return true if we should use the FA vector kernel for this op
bool ggml_metal_op_flash_attn_ext_use_vec(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);

size_t ggml_metal_op_flash_attn_ext_extra_pad(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);
size_t ggml_metal_op_flash_attn_ext_extra_blk(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);
size_t ggml_metal_op_flash_attn_ext_extra_tmp(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);
size_t ggml_metal_op_flash_attn_ext_extra_kv_f16(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);
size_t ggml_metal_op_flash_attn_ext_extra_idx(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);

size_t ggml_metal_op_argmax_extra(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);

ಪೂರ್ಣಾಂಕ ggml_metal_op_concat            (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_repeat            (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_acc               (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_unary             (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_glu               (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_ಮೊತ್ತ               (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು          (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_cumsum            (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_get_ಅಡ್ಡಸಾಲುಗಳು          (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು          (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_diag              (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_lightning_indexer (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_dsv4_hc           (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_soft_max          (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_ssm_conv          (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_ssm_ಪರಿಶೀಲನೆ          (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_rwkv              (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_gated_delta_net   (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_solve_tri         (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_ಗಣ               (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_cpy               (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_pool_1d           (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_pool_2d           (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_fwht              (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_mul_mat           (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_mul_mat_id        (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_ಕೂಡಿಸು_id            (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_flash_attn_ext    (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_bin               (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_silu_ಹಿಂಭಾಗ         (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_l2_norm           (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_ಗುಂಪು_norm        (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_norm              (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_rope              (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_im2col            (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_conv_2d           (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_conv_2d_dw        (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_conv_3d           (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_conv_transpose_1d (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_conv_transpose_2d (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_col2im_1d         (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_snake_fused       (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_upscale           (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_pad               (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_pad_reflect_1d    (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_roll              (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_arange            (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_timestep_embedding(ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_argmax            (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_argsort           (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_ತುದಿ_k             (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_topk_moe          (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_moe_reduce        (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_tri               (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_opt_ಹೆಜ್ಜೆ_adamw    (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_opt_ಹೆಜ್ಜೆ_sgd      (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);
ಪೂರ್ಣಾಂಕ ggml_metal_op_ಎಣಿಕೆ_equal       (ggml_metal_op_t ctx, ಪೂರ್ಣಾಂಕ idx);

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __cplusplus
}
#ಆದರೆ_ಅಂತ್ಯ
