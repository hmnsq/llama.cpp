#pragma once

#અન્તર્ભાવ "ggml.h"

//
// cache line
//
// TODO: rework CACHE_LINE_SIZE so std::hardware_destructive_interference_size
// can be used consistently between C and C++ TUs; the previous macro form
// diverged based on include order and undersized the work buffer.
// ref: https://github.com/ggml-org/llama.cpp/pull/28882
#જો defined(__POWER9_VECTOR__)
#define CACHE_લીટી_SIZE 128
#અથવાજો defined(__VXE__) || defined(__VXE2__)
#define CACHE_લીટી_SIZE 256
#નહીંતર
#define CACHE_લીટી_SIZE 64
#endif

સ્થિર અચલ size_t CACHE_લીટી_SIZE_F32 = CACHE_લીટી_SIZE/કદ(દશાંશ);

// Work buffer size for im2col operations in CONV2D
#define GGML_IM2COL_WORK_SIZE (16 * 1024 * 1024)

#ifdef __cplusplus
બાહ્ય "C" {
#endif

શૂન્ય ggml_compute_forward_dup(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_add(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_add_id(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_add1(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_acc(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_sum(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_sum_rows(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_cumsum(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_mean(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_argmax(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_ગણતરી_equal(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_repeat(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_repeat_back(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_concat(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_silu_back(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_norm(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_rms_norm(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_rms_norm_mul_fused(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst_rms_norm, સંરચના ggml_tensor * dst_mul);
શૂન્ય ggml_compute_forward_rms_norm_back(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_group_norm(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_l2_norm(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_out_prod(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_scale(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_set(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_cpy(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_cont(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_get_rows(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_get_rows_back(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_set_rows(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_diag(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_diag_mask_inf(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_diag_mask_zero(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_soft_max(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_soft_max_ext_back(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_rope(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_rope_back(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_clamp(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_conv_transpose_1d(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_im2col(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_im2col_back_f32(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_im2col_3d(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_col2im_1d(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_conv_2d(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_conv_3d(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_conv_transpose_2d(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_conv_2d_dw(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_pool_1d(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_pool_2d(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_pool_2d_back(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_upscale(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_pad(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_pad_reflect_1d(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_roll(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_arange(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_timestep_embedding(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_argsort(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_ટોચ_k(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_leaky_relu(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_tri(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_fill(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_flash_attn_ext(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_flash_attn_back(
        અચલ સંરચના ggml_compute_params * params,
        અચલ bool masked,
        સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_ssm_conv(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_ssm_scan(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_win_part(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_win_unpart(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_unary(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_glu(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_get_rel_pos(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_add_rel_pos(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_rwkv_wkv6(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_rwkv_wkv7(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_solve_tri(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_gla(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_gated_delta_net(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_lightning_indexer(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_dsv4_hc_comb(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_dsv4_hc_pre(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_dsv4_hc_post(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_ટોચ_k_mask(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_map_custom1(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_map_custom2(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_map_custom3(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_custom(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_cross_entropy_loss(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_cross_entropy_loss_back(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_opt_step_adamw(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_mul_mat(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_fwht(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_opt_step_sgd(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
#ifdef __cplusplus
}
#endif
