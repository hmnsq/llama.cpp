#pragma once

#अन्तर्भाव "ggml-metal-device.h"

#ifdef __cplusplus
बाह्य "C" {
#endif

प्रकारनिर्देश संरचना ggml_metal_op * ggml_metal_op_t;

संरचना ggml_metal_fusion; // forward decl (ggml-metal-device.h)

ggml_metal_op_t ggml_metal_op_init(
        ggml_metal_device_t dev,
        ggml_metal_cmd_buf_t cmd_buf,
        संरचना ggml_cgraph * gf,
        संरचना ggml_metal_fusion_info * finfo,
        पूर्णाङ्क  idx_start,
        पूर्णाङ्क  idx_end,
        bool use_concurrency,
        bool use_capture,
        पूर्णाङ्क  debug_graph);

शून्य ggml_metal_op_free(ggml_metal_op_t ctx);

पूर्णाङ्क ggml_metal_op_n_nodes(ggml_metal_op_t ctx);

पूर्णाङ्क ggml_metal_op_encode(ggml_metal_op_t ctx, पूर्णाङ्क idx);

//
// available ops:
//

// tokens per expert
माप_प्रकार ggml_metal_op_mul_mat_id_extra_tpe(अचर संरचना ggml_tensor * op);

// id map [n_tokens, n_expert]
माप_प्रकार ggml_metal_op_mul_mat_id_extra_ids(अचर संरचना ggml_tensor * op);
माप_प्रकार ggml_metal_op_mul_mat_id_extra_amax(अचर संरचना ggml_tensor * op);

// return true if we should use the FA vector kernel for this op
bool ggml_metal_op_flash_attn_ext_use_vec(अचर संरचना ggml_tensor * op);

माप_प्रकार ggml_metal_op_flash_attn_ext_extra_pad(अचर संरचना ggml_tensor * op);
माप_प्रकार ggml_metal_op_flash_attn_ext_extra_blk(अचर संरचना ggml_tensor * op);
माप_प्रकार ggml_metal_op_flash_attn_ext_extra_tmp(अचर संरचना ggml_tensor * op);
माप_प्रकार ggml_metal_op_flash_attn_ext_extra_kv_f16(अचर संरचना ggml_tensor * op);
माप_प्रकार ggml_metal_op_flash_attn_ext_extra_idx(अचर संरचना ggml_tensor * op);

माप_प्रकार ggml_metal_op_argmax_extra(अचर संरचना ggml_tensor * op);

पूर्णाङ्क ggml_metal_op_concat            (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_repeat            (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_acc               (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_unary             (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_glu               (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_sum               (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_sum_rows          (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_cumsum            (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_get_rows          (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_set_rows          (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_diag              (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_lightning_indexer (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_dsv4_hc           (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_soft_max          (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_ssm_conv          (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_ssm_scan          (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_rwkv              (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_gated_delta_net   (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_solve_tri         (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_set               (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_cpy               (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_pool_1d           (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_pool_2d           (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_fwht              (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_mul_mat           (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_mul_mat_id        (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_add_id            (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_flash_attn_ext    (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_bin               (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_silu_back         (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_l2_norm           (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_group_norm        (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_norm              (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_rope              (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_im2col            (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_conv_2d           (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_conv_2d_dw        (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_conv_3d           (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_conv_transpose_1d (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_conv_transpose_2d (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_col2im_1d         (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_snake_fused       (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_upscale           (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_pad               (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_pad_reflect_1d    (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_roll              (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_arange            (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_timestep_embedding(ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_argmax            (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_argsort           (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_शीर्ष_k             (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_topk_moe          (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_moe_reduce        (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_tri               (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_opt_step_adamw    (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_opt_step_sgd      (ggml_metal_op_t ctx, पूर्णाङ्क idx);
पूर्णाङ्क ggml_metal_op_गन्ती_equal       (ggml_metal_op_t ctx, पूर्णाङ्क idx);

#ifdef __cplusplus
}
#endif
