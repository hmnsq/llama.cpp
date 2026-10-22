#pragma once

#അന്തര്ഭാവ "ggml.h"

//
// cache line
//
// TODO: rework CACHE_LINE_SIZE so std::hardware_destructive_interference_size
// can be used consistently between C and C++ TUs; the previous macro form
// diverged based on include order and undersized the work buffer.
// ref: https://github.com/ggml-org/llama.cpp/pull/28882
#എങ്കിൽ defined(__POWER9_VECTOR__)
#define CACHE_വരി_SIZE 128
#അല്ലെങ്കിലെങ്കിൽ defined(__VXE__) || defined(__VXE2__)
#define CACHE_വരി_SIZE 256
#അല്ലെങ്കിൽ
#define CACHE_വരി_SIZE 64
#endif

സ്ഥിതം സ്ഥിരം size_t CACHE_വരി_SIZE_F32 = CACHE_വരി_SIZE/വലുപ്പം(ദശാംശം);

// Work buffer size for im2col operations in CONV2D
#define GGML_IM2COL_WORK_SIZE (16 * 1024 * 1024)

#ifdef __cplusplus
ബാഹ്യ "C" {
#endif

ശൂന്യം ggml_compute_forward_dup(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_add(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_add_id(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_add1(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_acc(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_sum(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_sum_rows(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_cumsum(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_mean(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_argmax(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_എണ്ണം_equal(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_repeat(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_repeat_back(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_concat(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_silu_back(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_norm(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_rms_norm(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_rms_norm_mul_fused(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst_rms_norm, ഘടന ggml_tensor * dst_mul);
ശൂന്യം ggml_compute_forward_rms_norm_back(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_group_norm(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_l2_norm(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_out_prod(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_scale(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_set(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_cpy(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_cont(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_get_rows(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_get_rows_back(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_set_rows(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_diag(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_diag_mask_inf(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_diag_mask_zero(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_soft_max(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_soft_max_ext_back(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_rope(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_rope_back(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_clamp(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_conv_transpose_1d(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_im2col(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_im2col_back_f32(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_im2col_3d(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_col2im_1d(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_conv_2d(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_conv_3d(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_conv_transpose_2d(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_conv_2d_dw(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_pool_1d(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_pool_2d(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_pool_2d_back(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_upscale(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_pad(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_pad_reflect_1d(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_roll(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_arange(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_timestep_embedding(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_argsort(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_മുകൾ_k(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_leaky_relu(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_tri(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_fill(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_flash_attn_ext(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_flash_attn_back(
        സ്ഥിരം ഘടന ggml_compute_params * params,
        സ്ഥിരം bool masked,
        ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_ssm_conv(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_ssm_scan(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_win_part(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_win_unpart(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_unary(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_glu(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_get_rel_pos(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_add_rel_pos(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_rwkv_wkv6(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_rwkv_wkv7(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_solve_tri(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_gla(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_gated_delta_net(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_lightning_indexer(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_dsv4_hc_comb(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_dsv4_hc_pre(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_dsv4_hc_post(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_മുകൾ_k_mask(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_map_custom1(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_map_custom2(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_map_custom3(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_custom(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_cross_entropy_loss(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_cross_entropy_loss_back(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_opt_step_adamw(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_mul_mat(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_fwht(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_opt_step_sgd(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
#ifdef __cplusplus
}
#endif
