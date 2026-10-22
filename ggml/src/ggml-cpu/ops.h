#pragma once

#ಅನ್ತರ್ಭಾವ "ggml.h"

//
// cache line
//
// TODO: rework CACHE_LINE_SIZE so std::hardware_destructive_interference_size
// can be used consistently between C and C++ TUs; the previous macro form
// diverged based on include order and undersized the work buffer.
// ref: https://github.com/ggml-org/llama.cpp/pull/28882
#ಯದಿ defined(__POWER9_VECTOR__)
#define CACHE_ಸಾಲು_SIZE 128
#ನೋಚೇದ್ಯದಿ defined(__VXE__) || defined(__VXE2__)
#define CACHE_ಸಾಲು_SIZE 256
#ಅನ್ಯಥಾ
#define CACHE_ಸಾಲು_SIZE 64
#endif

ಸ್ಥಿರ ಧ್ರುವ size_t CACHE_ಸಾಲು_SIZE_F32 = CACHE_ಸಾಲು_SIZE/ಪರಿಮಾಣ(ಭಿನ್ನ);

// Work buffer size for im2col operations in CONV2D
#define GGML_IM2COL_WORK_SIZE (16 * 1024 * 1024)

#ifdef __cplusplus
ಬಾಹ್ಯ "C" {
#endif

ಶೂನ್ಯ ggml_compute_forward_dup(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_add(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_add_id(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_add1(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_acc(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_sum(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_sum_rows(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_cumsum(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_mean(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_argmax(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_ಎಣಿಕೆ_equal(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_repeat(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_repeat_back(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_concat(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_silu_back(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_norm(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_rms_norm(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_rms_norm_mul_fused(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst_rms_norm, ಸಂರಚನಾ ggml_tensor * dst_mul);
ಶೂನ್ಯ ggml_compute_forward_rms_norm_back(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_group_norm(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_l2_norm(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_out_prod(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_scale(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_set(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_cpy(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_cont(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_get_rows(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_get_rows_back(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_set_rows(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_diag(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_diag_mask_inf(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_diag_mask_zero(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_soft_max(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_soft_max_ext_back(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_rope(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_rope_back(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_clamp(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_conv_transpose_1d(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_im2col(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_im2col_back_f32(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_im2col_3d(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_col2im_1d(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_conv_2d(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_conv_3d(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_conv_transpose_2d(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_conv_2d_dw(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_pool_1d(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_pool_2d(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_pool_2d_back(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_upscale(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_pad(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_pad_reflect_1d(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_roll(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_arange(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_timestep_embedding(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_argsort(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_ತುದಿ_k(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_leaky_relu(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_tri(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_fill(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_flash_attn_ext(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_flash_attn_back(
        ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params,
        ಧ್ರುವ bool masked,
        ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_ssm_conv(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_ssm_scan(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_win_part(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_win_unpart(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_unary(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_glu(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_get_rel_pos(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_add_rel_pos(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_rwkv_wkv6(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_rwkv_wkv7(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_solve_tri(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_gla(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_gated_delta_net(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_lightning_indexer(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_dsv4_hc_comb(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_dsv4_hc_pre(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_dsv4_hc_post(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_ತುದಿ_k_mask(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_map_custom1(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_map_custom2(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_map_custom3(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_custom(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_cross_entropy_loss(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_cross_entropy_loss_back(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_opt_step_adamw(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_mul_mat(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_fwht(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_opt_step_sgd(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
#ifdef __cplusplus
}
#endif
