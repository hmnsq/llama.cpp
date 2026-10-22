#pragma once

#ଅନ୍ତର୍ଭାଵ "ggml.h"

//
// cache line
//
// TODO: rework CACHE_LINE_SIZE so std::hardware_destructive_interference_size
// can be used consistently between C and C++ TUs; the previous macro form
// diverged based on include order and undersized the work buffer.
// ref: https://github.com/ggml-org/llama.cpp/pull/28882
#ଯଦି defined(__POWER9_VECTOR__)
#define CACHE_ଧାଡ଼ି_SIZE 128
#ନଚେତଯଦି defined(__VXE__) || defined(__VXE2__)
#define CACHE_ଧାଡ଼ି_SIZE 256
#ଅନ୍ୟଥା
#define CACHE_ଧାଡ଼ି_SIZE 64
#endif

ସ୍ଥିର ଅଚଳ size_t CACHE_ଧାଡ଼ି_SIZE_F32 = CACHE_ଧାଡ଼ି_SIZE/ଆକାର(ଦଶମିକ);

// Work buffer size for im2col operations in CONV2D
#define GGML_IM2COL_WORK_SIZE (16 * 1024 * 1024)

#ifdef __cplusplus
ବାହ୍ୟ "C" {
#endif

ଶୂନ୍ୟ ggml_compute_forward_dup(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_add(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_add_id(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_add1(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_acc(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_sum(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_sum_rows(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_cumsum(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_mean(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_argmax(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_ଗଣନା_equal(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_repeat(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_repeat_back(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_concat(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_silu_back(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_norm(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_rms_norm(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_rms_norm_mul_fused(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst_rms_norm, ସଂରଚନା ggml_tensor * dst_mul);
ଶୂନ୍ୟ ggml_compute_forward_rms_norm_back(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_group_norm(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_l2_norm(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_out_prod(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_scale(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_set(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_cpy(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_cont(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_get_rows(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_get_rows_back(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_set_rows(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_diag(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_diag_mask_inf(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_diag_mask_zero(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_soft_max(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_soft_max_ext_back(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_rope(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_rope_back(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_clamp(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_conv_transpose_1d(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_im2col(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_im2col_back_f32(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_im2col_3d(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_col2im_1d(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_conv_2d(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_conv_3d(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_conv_transpose_2d(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_conv_2d_dw(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_pool_1d(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_pool_2d(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_pool_2d_back(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_upscale(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_pad(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_pad_reflect_1d(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_roll(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_arange(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_timestep_embedding(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_argsort(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_ଶୀର୍ଷ_k(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_leaky_relu(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_tri(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_fill(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_flash_attn_ext(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_flash_attn_back(
        ଅଚଳ ସଂରଚନା ggml_compute_params * params,
        ଅଚଳ bool masked,
        ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_ssm_conv(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_ssm_scan(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_win_part(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_win_unpart(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_unary(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_glu(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_get_rel_pos(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_add_rel_pos(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_rwkv_wkv6(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_rwkv_wkv7(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_solve_tri(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_gla(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_gated_delta_net(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_lightning_indexer(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_dsv4_hc_comb(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_dsv4_hc_pre(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_dsv4_hc_post(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_ଶୀର୍ଷ_k_mask(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_map_custom1(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_map_custom2(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_map_custom3(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_custom(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_cross_entropy_loss(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_cross_entropy_loss_back(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_opt_step_adamw(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_mul_mat(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_fwht(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_opt_step_sgd(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
#ifdef __cplusplus
}
#endif
