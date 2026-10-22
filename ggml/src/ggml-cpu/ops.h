#pragma once

#अन्तर्भाव "ggml.h"

//
// cache line
//
// TODO: rework CACHE_LINE_SIZE so std::hardware_destructive_interference_size
// can be used consistently between C and C++ TUs; the previous macro form
// diverged based on include order and undersized the work buffer.
// ref: https://github.com/ggml-org/llama.cpp/pull/28882
#जर defined(__POWER9_VECTOR__)
#define CACHE_ओळ_SIZE 128
#किंवाजर defined(__VXE__) || defined(__VXE2__)
#define CACHE_ओळ_SIZE 256
#अन्यथा
#define CACHE_ओळ_SIZE 64
#endif

स्थिर स्थिरांक माप_प्रकार CACHE_ओळ_SIZE_F32 = CACHE_ओळ_SIZE/आकार(दशांश);

// Work buffer size for im2col operations in CONV2D
#define GGML_IM2COL_WORK_SIZE (16 * 1024 * 1024)

#ifdef __cplusplus
बाह्य "C" {
#endif

शून्य ggml_compute_forward_dup(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_add(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_add_id(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_add1(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_acc(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_sum(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_sum_rows(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_cumsum(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_mean(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_argmax(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_मोजणी_equal(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_repeat(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_repeat_back(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_concat(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_silu_back(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_norm(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_rms_norm(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_rms_norm_mul_fused(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst_rms_norm, संरचना ggml_tensor * dst_mul);
शून्य ggml_compute_forward_rms_norm_back(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_group_norm(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_l2_norm(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_out_prod(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_scale(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_set(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_cpy(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_cont(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_get_rows(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_get_rows_back(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_set_rows(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_diag(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_diag_mask_inf(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_diag_mask_zero(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_soft_max(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_soft_max_ext_back(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_rope(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_rope_back(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_clamp(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_conv_transpose_1d(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_im2col(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_im2col_back_f32(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_im2col_3d(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_col2im_1d(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_conv_2d(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_conv_3d(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_conv_transpose_2d(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_conv_2d_dw(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_pool_1d(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_pool_2d(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_pool_2d_back(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_upscale(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_pad(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_pad_reflect_1d(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_roll(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_arange(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_timestep_embedding(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_argsort(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_शीर्ष_k(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_leaky_relu(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_tri(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_fill(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_flash_attn_ext(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_flash_attn_back(
        स्थिरांक संरचना ggml_compute_params * params,
        स्थिरांक bool masked,
        संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_ssm_conv(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_ssm_scan(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_win_part(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_win_unpart(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_unary(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_glu(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_get_rel_pos(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_add_rel_pos(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_rwkv_wkv6(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_rwkv_wkv7(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_solve_tri(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_gla(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_gated_delta_net(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_lightning_indexer(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_dsv4_hc_comb(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_dsv4_hc_pre(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_dsv4_hc_post(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_शीर्ष_k_mask(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_map_custom1(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_map_custom2(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_map_custom3(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_custom(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_cross_entropy_loss(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_cross_entropy_loss_back(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_opt_step_adamw(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_mul_mat(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_fwht(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_opt_step_sgd(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
#ifdef __cplusplus
}
#endif
