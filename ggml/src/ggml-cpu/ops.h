#pragma once

#অন্তর্ভাব "ggml.h"

//
// cache line
//
// TODO: rework CACHE_LINE_SIZE so std::hardware_destructive_interference_size
// can be used consistently between C and C++ TUs; the previous macro form
// diverged based on include order and undersized the work buffer.
// ref: https://github.com/ggml-org/llama.cpp/pull/28882
#যদি defined(__POWER9_VECTOR__)
#define CACHE_সারি_আকার 128
#নতুবা_যদি defined(__VXE__) || defined(__VXE2__)
#define CACHE_সারি_আকার 256
#অন্যথা
#define CACHE_সারি_আকার 64
#endif

স্থির ধ্রুবক size_t CACHE_সারি_আকার_F32 = CACHE_সারি_আকার/আকার(দশমিক);

// Work buffer size for im2col operations in CONV2D
#define GGML_IM2_COL_WORK_আকার (16 * 1024 * 1024)

#ifdef __cplusplus
বাহ্যিক "C" {
#endif

শূন্য ggml_compute_forward_dup(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_add(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_add_id(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_add1(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_acc(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_যোগ(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_যোগ_rows(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_cumsum(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_mean(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_argmax(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_গুনতি_equal(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_repeat(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_repeat_back(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_concat(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_silu_back(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_norm(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_rms_norm(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_rms_norm_mul_fused(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst_rms_norm, কাঠামো ggml_tensor * dst_mul);
শূন্য ggml_compute_forward_rms_norm_back(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_group_norm(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_l2_norm(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_out_prod(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_scale(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_set(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_cpy(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_cont(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_get_rows(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_get_rows_back(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_set_rows(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_diag(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_diag_mask_inf(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_diag_mask_zero(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_soft_max(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_soft_max_ext_back(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_rope(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_rope_back(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_clamp(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_conv_transpose_1d(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_im2col(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_im2col_back_f32(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_im2col_3d(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_col2im_1d(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_conv_2d(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_conv_3d(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_conv_transpose_2d(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_conv_2d_dw(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_pool_1d(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_pool_2d(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_pool_2d_back(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_upscale(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_pad(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_pad_reflect_1d(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_roll(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_arange(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_timestep_embedding(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_argsort(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_শীর্ষ_k(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_leaky_relu(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_tri(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_fill(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_flash_attn_ext(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_flash_attn_back(
        ধ্রুবক কাঠামো ggml_compute_params * params,
        ধ্রুবক bool masked,
        কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_ssm_conv(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_ssm_scan(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_win_part(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_win_unpart(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_unary(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_glu(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_get_rel_pos(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_add_rel_pos(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_rwkv_wkv6(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_rwkv_wkv7(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_solve_tri(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_gla(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_gated_delta_net(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_lightning_indexer(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_dsv4_hc_comb(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_dsv4_hc_pre(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_dsv4_hc_post(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_শীর্ষ_k_mask(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_map_custom1(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_map_custom2(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_map_custom3(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_custom(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_cross_entropy_loss(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_cross_entropy_loss_back(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_opt_step_adamw(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_mul_mat(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_fwht(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_opt_step_sgd(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
#ifdef __cplusplus
}
#endif
