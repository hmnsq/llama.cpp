#pragma once

#ضمن "ggml.h"

//
// cache line
//
// TODO: rework CACHE_LINE_SIZE so std::hardware_destructive_interference_size
// can be used consistently between C and C++ TUs; the previous macro form
// diverged based on include order and undersized the work buffer.
// ref: https://github.com/ggml-org/llama.cpp/pull/28882
#إذا defined(__POWER9_VECTOR__)
#define CACHE_سطر_SIZE 128
#وإلاإذا defined(__VXE__) || defined(__VXE2__)
#define CACHE_سطر_SIZE 256
#وإلا
#define CACHE_سطر_SIZE 64
#endif

ساكن ثابت size_t CACHE_سطر_SIZE_F32 = CACHE_سطر_SIZE/حجم(عائم);

// Work buffer size for im2col operations in CONV2D
#define GGML_IM2COL_WORK_SIZE (16 * 1024 * 1024)

#ifdef __cplusplus
خارجي "C" {
#endif

فراغ ggml_compute_forward_dup(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_add(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_add_id(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_add1(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_acc(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_sum(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_sum_rows(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_cumsum(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_mean(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_argmax(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_تعداد_equal(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_repeat(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_repeat_back(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_concat(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_silu_back(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_norm(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_rms_norm(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_rms_norm_mul_fused(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst_rms_norm, بنية ggml_tensor * dst_mul);
فراغ ggml_compute_forward_rms_norm_back(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_group_norm(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_l2_norm(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_out_prod(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_scale(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_set(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_cpy(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_cont(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_get_rows(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_get_rows_back(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_set_rows(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_diag(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_diag_mask_inf(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_diag_mask_zero(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_soft_max(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_soft_max_ext_back(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_rope(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_rope_back(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_clamp(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_conv_transpose_1d(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_im2col(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_im2col_back_f32(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_im2col_3d(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_col2im_1d(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_conv_2d(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_conv_3d(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_conv_transpose_2d(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_conv_2d_dw(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_pool_1d(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_pool_2d(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_pool_2d_back(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_upscale(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_pad(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_pad_reflect_1d(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_roll(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_arange(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_timestep_embedding(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_argsort(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_قمة_k(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_leaky_relu(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_tri(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_fill(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_flash_attn_ext(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_flash_attn_back(
        ثابت بنية ggml_compute_params * params,
        ثابت bool masked,
        بنية ggml_tensor * dst);
فراغ ggml_compute_forward_ssm_conv(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_ssm_scan(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_win_part(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_win_unpart(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_unary(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_glu(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_get_rel_pos(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_add_rel_pos(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_rwkv_wkv6(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_rwkv_wkv7(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_solve_tri(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_gla(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_gated_delta_net(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_lightning_indexer(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_dsv4_hc_comb(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_dsv4_hc_pre(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_dsv4_hc_post(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_قمة_k_mask(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_map_custom1(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_map_custom2(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_map_custom3(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_custom(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_cross_entropy_loss(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_cross_entropy_loss_back(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_opt_step_adamw(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_mul_mat(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_fwht(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_opt_step_sgd(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
#ifdef __cplusplus
}
#endif
