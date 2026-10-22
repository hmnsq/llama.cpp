#pragma once

#شامل "ggml.h"

//
// cache line
//
// TODO: rework CACHE_LINE_SIZE so std::hardware_destructive_interference_size
// can be used consistently between C and C++ TUs; the previous macro form
// diverged based on include order and undersized the work buffer.
// ref: https://github.com/ggml-org/llama.cpp/pull/28882
#اگر defined(__POWER9_VECTOR__)
#define CACHE_سطر_SIZE 128
#وگرنهاگر defined(__VXE__) || defined(__VXE2__)
#define CACHE_سطر_SIZE 256
#وگرنه
#define CACHE_سطر_SIZE 64
#endif

ایستا ثابت size_t CACHE_سطر_SIZE_F32 = CACHE_سطر_SIZE/اندازه(اعشاری);

// Work buffer size for im2col operations in CONV2D
#define GGML_IM2COL_WORK_SIZE (16 * 1024 * 1024)

#ifdef __cplusplus
بیرونی "C" {
#endif

تهی ggml_compute_forward_dup(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_add(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_add_id(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_add1(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_acc(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_sum(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_sum_rows(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_cumsum(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_mean(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_argmax(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_شمار_equal(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_repeat(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_repeat_back(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_concat(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_silu_back(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_norm(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_rms_norm(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_rms_norm_mul_fused(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst_rms_norm, ساختار ggml_tensor * dst_mul);
تهی ggml_compute_forward_rms_norm_back(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_group_norm(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_l2_norm(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_out_prod(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_scale(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_set(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_cpy(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_cont(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_get_rows(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_get_rows_back(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_set_rows(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_diag(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_diag_mask_inf(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_diag_mask_zero(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_soft_max(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_soft_max_ext_back(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_rope(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_rope_back(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_clamp(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_conv_transpose_1d(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_im2col(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_im2col_back_f32(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_im2col_3d(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_col2im_1d(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_conv_2d(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_conv_3d(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_conv_transpose_2d(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_conv_2d_dw(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_pool_1d(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_pool_2d(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_pool_2d_back(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_upscale(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_pad(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_pad_reflect_1d(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_roll(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_arange(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_timestep_embedding(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_argsort(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_بالاترین_k(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_leaky_relu(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_tri(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_fill(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_flash_attn_ext(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_flash_attn_back(
        ثابت ساختار ggml_compute_params * params,
        ثابت bool masked,
        ساختار ggml_tensor * dst);
تهی ggml_compute_forward_ssm_conv(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_ssm_scan(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_win_part(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_win_unpart(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_unary(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_glu(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_get_rel_pos(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_add_rel_pos(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_rwkv_wkv6(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_rwkv_wkv7(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_solve_tri(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_gla(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_gated_delta_net(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_lightning_indexer(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_dsv4_hc_comb(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_dsv4_hc_pre(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_dsv4_hc_post(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_بالاترین_k_mask(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_map_custom1(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_map_custom2(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_map_custom3(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_custom(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_cross_entropy_loss(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_cross_entropy_loss_back(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_opt_step_adamw(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_mul_mat(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_fwht(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_opt_step_sgd(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
#ifdef __cplusplus
}
#endif
