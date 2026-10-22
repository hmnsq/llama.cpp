#ಸೂಚನೆ once

#ಅನ್ತರ್ಭಾವ "ggml.h"

//
// cache line
//
// TODO: rework CACHE_LINE_SIZE so std::hardware_destructive_interference_size
// can be used consistently between C and C++ TUs; the previous macro form
// diverged based on include order and undersized the work buffer.
// ref: https://github.com/ggml-org/llama.cpp/pull/28882
#ಆದರೆ defined(__POWER9_VECTOR__)
#ವ್ಯಾಖ್ಯಾನಿಸು CACHE_ಸಾಲು_ಗಾತ್ರ 128
#ಅಥವಾಆದರೆ defined(__VXE__) || defined(__VXE2__)
#ವ್ಯಾಖ್ಯಾನಿಸು CACHE_ಸಾಲು_ಗಾತ್ರ 256
#ಇಲ್ಲದಿದ್ದರೆ
#ವ್ಯಾಖ್ಯಾನಿಸು CACHE_ಸಾಲು_ಗಾತ್ರ 64
#ಆದರೆ_ಅಂತ್ಯ

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ size_t CACHE_ಸಾಲು_ಗಾತ್ರ_F32 = CACHE_ಸಾಲು_ಗಾತ್ರ/ಗಾತ್ರ(ದಶಮಾಂಶ);

// Work buffer size for im2col operations in CONV2D
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_IM2_ಲಂಬಸಾಲು_WORK_ಗಾತ್ರ (16 * 1024 * 1024)

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __cplusplus
ಬಾಹ್ಯ "C" {
#ಆದರೆ_ಅಂತ್ಯ

ಶೂನ್ಯ ggml_compute_forward_dup(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_ಕೂಡಿಸು(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_ಕೂಡಿಸು_id(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_add1(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_acc(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_ಮೊತ್ತ(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_cumsum(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_ಸರಾಸರಿ(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_argmax(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_ಎಣಿಕೆ_equal(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_repeat(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_repeat_ಹಿಂಭಾಗ(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_concat(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_silu_ಹಿಂಭಾಗ(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_norm(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_rms_norm(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_rms_norm_mul_fused(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst_rms_norm, ರಚನೆ ggml_tensor * dst_mul);
ಶೂನ್ಯ ggml_compute_forward_rms_norm_ಹಿಂಭಾಗ(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_ಗುಂಪು_norm(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_l2_norm(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_ಹೊರಹೋಗು_prod(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_scale(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_ಗಣ(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_cpy(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_cont(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_get_ಅಡ್ಡಸಾಲುಗಳು(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_get_ಅಡ್ಡಸಾಲುಗಳು_ಹಿಂಭಾಗ(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_diag(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_diag_ಮುಖವಾಡ_inf(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_diag_ಮುಖವಾಡ_zero(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_soft_max(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_soft_max_ext_ಹಿಂಭಾಗ(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_rope(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_rope_ಹಿಂಭಾಗ(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_clamp(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_conv_transpose_1d(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_im2col(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_im2col_ಹಿಂಭಾಗ_f32(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_im2col_3d(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_col2im_1d(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_conv_2d(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_conv_3d(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_conv_transpose_2d(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_conv_2d_dw(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_pool_1d(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_pool_2d(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_pool_2d_ಹಿಂಭಾಗ(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_upscale(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_pad(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_pad_reflect_1d(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_roll(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_arange(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_timestep_embedding(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_argsort(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_ತುದಿ_k(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_leaky_relu(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_tri(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_fill(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_flash_attn_ext(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_flash_attn_ಹಿಂಭಾಗ(
        ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params,
        ಸ್ಥಿರಾಂಕ bool masked,
        ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_ssm_conv(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_ssm_ಪರಿಶೀಲನೆ(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_win_part(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_win_unpart(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_unary(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_glu(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_get_rel_pos(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_ಕೂಡಿಸು_rel_pos(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_rwkv_wkv6(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_rwkv_wkv7(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_solve_tri(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_gla(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_gated_delta_net(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_lightning_indexer(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_dsv4_hc_comb(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_dsv4_hc_pre(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_dsv4_hc_post(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_ತುದಿ_k_ಮುಖವಾಡ(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_map_custom1(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_map_custom2(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_map_custom3(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_custom(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_cross_entropy_loss(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_cross_entropy_loss_ಹಿಂಭಾಗ(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_opt_ಹೆಜ್ಜೆ_adamw(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_mul_mat(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_fwht(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_opt_ಹೆಜ್ಜೆ_sgd(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __cplusplus
}
#ಆದರೆ_ಅಂತ್ಯ
