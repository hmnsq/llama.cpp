#ਹਦਾਇਤ once

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"

//
// cache line
//
// TODO: rework CACHE_LINE_SIZE so std::hardware_destructive_interference_size
// can be used consistently between C and C++ TUs; the previous macro form
// diverged based on include order and undersized the work buffer.
// ref: https://github.com/ggml-org/llama.cpp/pull/28882
#ਜੇ defined(__POWER9_VECTOR__)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CACHE_ਸਤਰ_ਮਾਪ 128
#ਜਾਂਜੇ defined(__VXE__) || defined(__VXE2__)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CACHE_ਸਤਰ_ਮਾਪ 256
#ਨਹੀਂਤਾਂ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CACHE_ਸਤਰ_ਮਾਪ 64
#ਜੇ_ਅੰਤ

ਸਥਿਰ ਅਚਲ size_t CACHE_ਸਤਰ_ਮਾਪ_F32 = CACHE_ਸਤਰ_ਮਾਪ/ਆਕਾਰ(ਦਸ਼ਮਲਵ);

// Work buffer size for im2col operations in CONV2D
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_IM2_ਖਾਨਾ_WORK_ਮਾਪ (16 * 1024 * 1024)

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __cplusplus
ਬਾਹਰੀ "C" {
#ਜੇ_ਅੰਤ

ਖਾਲੀ ggml_compute_forward_dup(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_ਜੋੜੋ(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_ਜੋੜੋ_id(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_add1(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_acc(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_ਜੋੜ(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_ਜੋੜ_ਪੰਕਤੀਆਂ(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_cumsum(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_ਔਸਤ(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_argmax(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_ਗਿਣਤੀ_equal(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_repeat(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_repeat_ਪਿੱਠ(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_concat(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_silu_ਪਿੱਠ(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_norm(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_rms_norm(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_rms_norm_mul_fused(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst_rms_norm, ਬਣਤਰ ggml_tensor * dst_mul);
ਖਾਲੀ ggml_compute_forward_rms_norm_ਪਿੱਠ(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_ਸਮੂਹ_norm(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_l2_norm(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_ਬਾਹਰਲਾ_prod(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_scale(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_ਸੈੱਟ(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_cpy(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_cont(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_get_ਪੰਕਤੀਆਂ(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_get_ਪੰਕਤੀਆਂ_ਪਿੱਠ(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_ਸੈੱਟ_ਪੰਕਤੀਆਂ(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_diag(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_diag_ਨਕਾਬ_inf(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_diag_ਨਕਾਬ_zero(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_soft_max(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_soft_max_ext_ਪਿੱਠ(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_rope(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_rope_ਪਿੱਠ(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_clamp(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_conv_transpose_1d(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_im2col(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_im2col_ਪਿੱਠ_f32(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_im2col_3d(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_col2im_1d(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_conv_2d(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_conv_3d(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_conv_transpose_2d(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_conv_2d_dw(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_pool_1d(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_pool_2d(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_pool_2d_ਪਿੱਠ(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_upscale(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_pad(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_pad_reflect_1d(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_roll(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_arange(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_timestep_embedding(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_argsort(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_ਸਿਖਰ_k(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_leaky_relu(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_tri(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_fill(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_flash_attn_ext(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_flash_attn_ਪਿੱਠ(
        ਅਚਲ ਬਣਤਰ ggml_compute_params * params,
        ਅਚਲ bool masked,
        ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_ssm_conv(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_ssm_ਜਾਂਚ(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_win_part(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_win_unpart(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_unary(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_glu(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_get_rel_pos(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_ਜੋੜੋ_rel_pos(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_rwkv_wkv6(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_rwkv_wkv7(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_solve_tri(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_gla(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_gated_delta_net(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_lightning_indexer(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_dsv4_hc_comb(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_dsv4_hc_pre(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_dsv4_hc_post(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_ਸਿਖਰ_k_ਨਕਾਬ(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_map_custom1(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_map_custom2(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_map_custom3(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_custom(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_cross_entropy_loss(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_cross_entropy_loss_ਪਿੱਠ(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_opt_ਕਦਮ_adamw(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_mul_mat(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_fwht(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ ggml_compute_forward_opt_ਕਦਮ_sgd(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * dst);
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __cplusplus
}
#ਜੇ_ਅੰਤ
