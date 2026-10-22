#pragma once

#include "ggml.h"

//
// cache line
//
// TODO: rework CACHE_LINE_SIZE so std::hardware_destructive_interference_size
// can be used consistently between C and C++ TUs; the previous macro form
// diverged based on include order and undersized the work buffer.
// ref: https://github.com/ggml-org/llama.cpp/pull/28882
#ᨀᨘ defined(__POWER9_VECTOR__)
#define CACHE_LINE_SIZE 128
#elif defined(__VXE__) || defined(__VXE2__)
#define CACHE_LINE_SIZE 256
#ᨀᨘᨁ
#define CACHE_LINE_SIZE 64
#endif

static const size_t CACHE_LINE_SIZE_F32 = CACHE_LINE_SIZE/parimaa_na(float);

// Work buffer size for im2col operations in CONV2D
#define GGML_IM2COL_WORK_SIZE (16 * 1024 * 1024)

#ifdef __cplusplus
extern "C" {
#endif

void ggml_compute_forward_dup(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_add(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_add_id(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_add1(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_acc(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_sum(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_sum_rows(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_cumsum(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_mean(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_argmax(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_count_equal(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_repeat(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_repeat_back(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_concat(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_silu_back(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_norm(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_rms_norm(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_rms_norm_mul_fused(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst_rms_norm, sa_nracanaa ggml_tensor * dst_mul);
void ggml_compute_forward_rms_norm_back(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_group_norm(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_l2_norm(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_out_prod(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_scale(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_set(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_cpy(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_cont(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_get_rows(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_get_rows_back(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_set_rows(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_diag(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_diag_mask_inf(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_diag_mask_zero(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_soft_max(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_soft_max_ext_back(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_rope(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_rope_back(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_clamp(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_conv_transpose_1d(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_im2col(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_im2col_back_f32(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_im2col_3d(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_col2im_1d(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_conv_2d(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_conv_3d(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_conv_transpose_2d(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_conv_2d_dw(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_pool_1d(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_pool_2d(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_pool_2d_back(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_upscale(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_pad(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_pad_reflect_1d(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_roll(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_arange(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_timestep_embedding(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_argsort(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_top_k(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_leaky_relu(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_tri(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_fill(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_flash_attn_ext(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_flash_attn_back(
        const sa_nracanaa ggml_compute_params * params,
        const bool masked,
        sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_ssm_conv(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_ssm_scan(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_win_part(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_win_unpart(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_unary(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_glu(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_get_rel_pos(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_add_rel_pos(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_rwkv_wkv6(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_rwkv_wkv7(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_solve_tri(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_gla(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_gated_delta_net(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_lightning_indexer(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_dsv4_hc_comb(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_dsv4_hc_pre(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_dsv4_hc_post(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_top_k_mask(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_map_custom1(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_map_custom2(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_map_custom3(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_custom(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_cross_entropy_loss(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_cross_entropy_loss_back(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_opt_step_adamw(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_mul_mat(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_fwht(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_opt_step_sgd(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
#ifdef __cplusplus
}
#endif
