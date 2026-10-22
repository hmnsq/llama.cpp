#ہدایت once

#شامل "ggml.h"

//
// cache line
//
// TODO: rework CACHE_LINE_SIZE so std::hardware_destructive_interference_size
// can be used consistently between C and C++ TUs; the previous macro form
// diverged based on include order and undersized the work buffer.
// ref: https://github.com/ggml-org/llama.cpp/pull/28882
#اگر defined(__POWER9_VECTOR__)
#تعریف_کرو CACHE_سطر_سائز 128
#وگرنہ_اگر defined(__VXE__) || defined(__VXE2__)
#تعریف_کرو CACHE_سطر_سائز 256
#ورنہ
#تعریف_کرو CACHE_سطر_سائز 64
#اگر_ختم

ساکن ثابت size_t CACHE_سطر_سائز_F32 = CACHE_سطر_سائز/حجم(اعشاری);

// Work buffer size for im2col operations in CONV2D
#تعریف_کرو GGML_IM2_ستون_WORK_سائز (16 * 1024 * 1024)

#اگر_تعریف_شدہ __cplusplus
خارجی "C" {
#اگر_ختم

خالی ggml_compute_forward_dup(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_جوڑو(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_جوڑو_id(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_add1(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_acc(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_مجموع(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_مجموع_صفیں(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_cumsum(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_اوسط(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_argmax(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_گنتی_equal(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_repeat(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_repeat_عقب(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_concat(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_silu_عقب(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_norm(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_rms_norm(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_rms_norm_mul_fused(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst_rms_norm, ساخت ggml_tensor * dst_mul);
خالی ggml_compute_forward_rms_norm_عقب(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_گروہ_norm(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_l2_norm(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_برآمد_prod(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_scale(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_سیٹ(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_cpy(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_cont(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_get_صفیں(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_get_صفیں_عقب(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_سیٹ_صفیں(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_diag(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_diag_نقاب_inf(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_diag_نقاب_zero(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_soft_max(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_soft_max_ext_عقب(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_rope(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_rope_عقب(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_clamp(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_conv_transpose_1d(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_im2col(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_im2col_عقب_f32(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_im2col_3d(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_col2im_1d(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_conv_2d(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_conv_3d(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_conv_transpose_2d(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_conv_2d_dw(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_pool_1d(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_pool_2d(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_pool_2d_عقب(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_upscale(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_pad(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_pad_reflect_1d(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_roll(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_arange(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_timestep_embedding(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_argsort(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_چوٹی_k(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_leaky_relu(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_tri(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_fill(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_flash_attn_ext(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_flash_attn_عقب(
        ثابت ساخت ggml_compute_params * params,
        ثابت bool masked,
        ساخت ggml_tensor * dst);
خالی ggml_compute_forward_ssm_conv(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_ssm_جائزہ(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_win_part(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_win_unpart(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_unary(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_glu(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_get_rel_pos(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_جوڑو_rel_pos(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_rwkv_wkv6(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_rwkv_wkv7(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_solve_tri(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_gla(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_gated_delta_net(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_lightning_indexer(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_dsv4_hc_comb(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_dsv4_hc_pre(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_dsv4_hc_post(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_چوٹی_k_نقاب(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_map_custom1(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_map_custom2(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_map_custom3(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_custom(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_cross_entropy_loss(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_cross_entropy_loss_عقب(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_opt_قدم_adamw(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_mul_mat(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_fwht(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
خالی ggml_compute_forward_opt_قدم_sgd(ثابت ساخت ggml_compute_params * params, ساخت ggml_tensor * dst);
#اگر_تعریف_شدہ __cplusplus
}
#اگر_ختم
