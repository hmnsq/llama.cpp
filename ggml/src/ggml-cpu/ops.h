#هدايت once

#ضمن "ggml.h"

//
// cache line
//
// TODO: rework CACHE_LINE_SIZE so std::hardware_destructive_interference_size
// can be used consistently between C and C++ TUs; the previous macro form
// diverged based on include order and undersized the work buffer.
// ref: https://github.com/ggml-org/llama.cpp/pull/28882
#إذا defined(__POWER9_VECTOR__)
#وصف_ڪريو CACHE_سٽ_سائيز 128
#وإلاإذا defined(__VXE__) || defined(__VXE2__)
#وصف_ڪريو CACHE_سٽ_سائيز 256
#وإلا
#وصف_ڪريو CACHE_سٽ_سائيز 64
#جيڪڏهن_پورو

ساكن ثابت size_t CACHE_سٽ_سائيز_F32 = CACHE_سٽ_سائيز/حجم(عائم);

// Work buffer size for im2col operations in CONV2D
#وصف_ڪريو GGML_IM2_کانو_WORK_سائيز (16 * 1024 * 1024)

#جيڪڏهن_وصف_ٿيل __cplusplus
خارجي "C" {
#جيڪڏهن_پورو

فراغ ggml_compute_forward_dup(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_جوڙيو(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_جوڙيو_id(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_add1(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_acc(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_جوڙ(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_جوڙ_صفون(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_cumsum(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_سراسري(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_argmax(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_ڳڻپ_equal(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_repeat(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_repeat_پٺ(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_concat(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_silu_پٺ(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_norm(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_rms_norm(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_rms_norm_mul_fused(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst_rms_norm, بنية ggml_tensor * dst_mul);
فراغ ggml_compute_forward_rms_norm_پٺ(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_گروهه_norm(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_l2_norm(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_ٻاهريون_prod(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_scale(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_سيٽ(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_cpy(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_cont(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_get_صفون(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_get_صفون_پٺ(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_سيٽ_صفون(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_diag(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_diag_نقاب_inf(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_diag_نقاب_zero(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_soft_max(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_soft_max_ext_پٺ(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_rope(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_rope_پٺ(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_clamp(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_conv_transpose_1d(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_im2col(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_im2col_پٺ_f32(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_im2col_3d(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_col2im_1d(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_conv_2d(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_conv_3d(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_conv_transpose_2d(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_conv_2d_dw(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_pool_1d(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_pool_2d(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_pool_2d_پٺ(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_upscale(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_pad(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_pad_reflect_1d(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_roll(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_arange(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_timestep_embedding(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_argsort(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_چوٽي_k(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_leaky_relu(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_tri(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_fill(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_flash_attn_ext(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_flash_attn_پٺ(
        ثابت بنية ggml_compute_params * params,
        ثابت bool masked,
        بنية ggml_tensor * dst);
فراغ ggml_compute_forward_ssm_conv(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_ssm_جاچ(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_win_part(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_win_unpart(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_unary(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_glu(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_get_rel_pos(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_جوڙيو_rel_pos(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_rwkv_wkv6(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_rwkv_wkv7(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_solve_tri(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_gla(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_gated_delta_net(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_lightning_indexer(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_dsv4_hc_comb(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_dsv4_hc_pre(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_dsv4_hc_post(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_چوٽي_k_نقاب(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_map_custom1(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_map_custom2(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_map_custom3(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_custom(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_cross_entropy_loss(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_cross_entropy_loss_پٺ(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_opt_قدم_adamw(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_mul_mat(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_fwht(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_opt_قدم_sgd(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
#جيڪڏهن_وصف_ٿيل __cplusplus
}
#جيڪڏهن_پورو
