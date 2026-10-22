#pragma once

#அந்தர்பாவ "ggml.h"

//
// cache line
//
// TODO: rework CACHE_LINE_SIZE so std::hardware_destructive_interference_size
// can be used consistently between C and C++ TUs; the previous macro form
// diverged based on include order and undersized the work buffer.
// ref: https://github.com/ggml-org/llama.cpp/pull/28882
#எனில் defined(__POWER9_VECTOR__)
#define CACHE_வரி_SIZE 128
#அல்லதெனில் defined(__VXE__) || defined(__VXE2__)
#define CACHE_வரி_SIZE 256
#இல்லையெனில்
#define CACHE_வரி_SIZE 64
#endif

நிலையான மாறிலி size_t CACHE_வரி_SIZE_F32 = CACHE_வரி_SIZE/அளவு(தசம);

// Work buffer size for im2col operations in CONV2D
#define GGML_IM2COL_WORK_SIZE (16 * 1024 * 1024)

#ifdef __cplusplus
வெளி "C" {
#endif

வெறுமை ggml_compute_forward_dup(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_add(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_add_id(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_add1(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_acc(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_sum(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_sum_rows(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_cumsum(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_mean(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_argmax(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_மொத்தம்_equal(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_repeat(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_repeat_back(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_concat(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_silu_back(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_norm(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_rms_norm(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_rms_norm_mul_fused(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst_rms_norm, கட்டமைப்பு ggml_tensor * dst_mul);
வெறுமை ggml_compute_forward_rms_norm_back(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_group_norm(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_l2_norm(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_out_prod(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_scale(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_set(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_cpy(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_cont(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_get_rows(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_get_rows_back(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_set_rows(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_diag(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_diag_mask_inf(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_diag_mask_zero(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_soft_max(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_soft_max_ext_back(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_rope(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_rope_back(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_clamp(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_conv_transpose_1d(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_im2col(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_im2col_back_f32(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_im2col_3d(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_col2im_1d(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_conv_2d(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_conv_3d(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_conv_transpose_2d(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_conv_2d_dw(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_pool_1d(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_pool_2d(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_pool_2d_back(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_upscale(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_pad(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_pad_reflect_1d(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_roll(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_arange(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_timestep_embedding(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_argsort(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_உச்சி_k(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_leaky_relu(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_tri(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_fill(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_flash_attn_ext(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_flash_attn_back(
        மாறிலி கட்டமைப்பு ggml_compute_params * params,
        மாறிலி bool masked,
        கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_ssm_conv(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_ssm_scan(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_win_part(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_win_unpart(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_unary(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_glu(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_get_rel_pos(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_add_rel_pos(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_rwkv_wkv6(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_rwkv_wkv7(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_solve_tri(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_gla(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_gated_delta_net(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_lightning_indexer(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_dsv4_hc_comb(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_dsv4_hc_pre(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_dsv4_hc_post(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_உச்சி_k_mask(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_map_custom1(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_map_custom2(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_map_custom3(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_custom(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_cross_entropy_loss(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_cross_entropy_loss_back(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_opt_step_adamw(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_mul_mat(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_fwht(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_opt_step_sgd(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
#ifdef __cplusplus
}
#endif
