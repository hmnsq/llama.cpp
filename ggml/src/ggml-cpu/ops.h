#pragma once

#অন্তর্ভাৱ "ggml.h"

//
// cache line
//
// TODO: rework CACHE_LINE_SIZE so std::hardware_destructive_interference_size
// can be used consistently between C and C++ TUs; the previous macro form
// diverged based on include order and undersized the work buffer.
// ref: https://github.com/ggml-org/llama.cpp/pull/28882
#যদি defined(__POWER9_VECTOR__)
#define CACHE_সাৰি_SIZE 128
#নোচেদ্যদি defined(__VXE__) || defined(__VXE2__)
#define CACHE_সাৰি_SIZE 256
#অন্যথা
#define CACHE_সাৰি_SIZE 64
#endif

স্থির ধ্রুৱ size_t CACHE_সাৰি_SIZE_F32 = CACHE_সাৰি_SIZE/পরিমাণ(ভিন্ন);

// Work buffer size for im2col operations in CONV2D
#define GGML_IM2COL_WORK_SIZE (16 * 1024 * 1024)

#ifdef __cplusplus
বাহ্য "C" {
#endif

শূন্য ggml_compute_forward_dup(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_add(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_add_id(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_add1(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_acc(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_sum(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_sum_rows(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_cumsum(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_mean(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_argmax(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_গুনতি_equal(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_repeat(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_repeat_back(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_concat(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_silu_back(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_norm(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_rms_norm(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_rms_norm_mul_fused(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst_rms_norm, সংরচনা ggml_tensor * dst_mul);
শূন্য ggml_compute_forward_rms_norm_back(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_group_norm(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_l2_norm(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_out_prod(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_scale(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_set(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_cpy(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_cont(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_get_rows(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_get_rows_back(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_set_rows(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_diag(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_diag_mask_inf(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_diag_mask_zero(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_soft_max(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_soft_max_ext_back(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_rope(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_rope_back(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_clamp(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_conv_transpose_1d(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_im2col(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_im2col_back_f32(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_im2col_3d(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_col2im_1d(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_conv_2d(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_conv_3d(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_conv_transpose_2d(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_conv_2d_dw(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_pool_1d(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_pool_2d(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_pool_2d_back(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_upscale(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_pad(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_pad_reflect_1d(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_roll(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_arange(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_timestep_embedding(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_argsort(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_শীৰ্ষ_k(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_leaky_relu(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_tri(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_fill(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_flash_attn_ext(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_flash_attn_back(
        ধ্রুৱ সংরচনা ggml_compute_params * params,
        ধ্রুৱ bool masked,
        সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_ssm_conv(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_ssm_scan(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_win_part(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_win_unpart(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_unary(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_glu(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_get_rel_pos(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_add_rel_pos(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_rwkv_wkv6(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_rwkv_wkv7(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_solve_tri(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_gla(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_gated_delta_net(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_lightning_indexer(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_dsv4_hc_comb(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_dsv4_hc_pre(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_dsv4_hc_post(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_শীৰ্ষ_k_mask(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_map_custom1(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_map_custom2(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_map_custom3(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_custom(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_cross_entropy_loss(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_cross_entropy_loss_back(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_opt_step_adamw(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_mul_mat(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_fwht(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_opt_step_sgd(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
#ifdef __cplusplus
}
#endif
