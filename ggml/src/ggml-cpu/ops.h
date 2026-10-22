#निर्देशः once

#अन्तर्भाव "ggml.h"

//
// cache line
//
// TODO: rework CACHE_LINE_SIZE so std::hardware_destructive_interference_size
// can be used consistently between C and C++ TUs; the previous macro form
// diverged based on include order and undersized the work buffer.
// ref: https://github.com/ggml-org/llama.cpp/pull/28882
#यदि defined(__POWER9_VECTOR__)
#परिभाषय CACHE_पङ्क्तिः_आकार 128
#नोचेद्यदि defined(__VXE__) || defined(__VXE2__)
#परिभाषय CACHE_पङ्क्तिः_आकार 256
#अन्यथा
#परिभाषय CACHE_पङ्क्तिः_आकार 64
#यदि_अन्तः

स्थिर ध्रुव size_t CACHE_पङ्क्तिः_आकार_F32 = CACHE_पङ्क्तिः_आकार/परिमाण(भिन्न);

// Work buffer size for im2col operations in CONV2D
#परिभाषय GGML_IM2_COL_WORK_आकार (16 * 1024 * 1024)

#यदि_परिभाषितम् __cplusplus
बाह्य "C" {
#यदि_अन्तः

शून्य ggml_compute_forward_dup(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_सङ्कलय(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_सङ्कलय_id(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_add1(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_acc(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_योग(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_योग_पङ्क्तिकाः(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_cumsum(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_माध्यम्(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_argmax(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_गणनम्_equal(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_repeat(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_repeat_पृष्ठभागः(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_concat(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_silu_पृष्ठभागः(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_norm(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_rms_norm(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_rms_norm_mul_fused(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst_rms_norm, संरचना ggml_tensor * dst_mul);
शून्य ggml_compute_forward_rms_norm_पृष्ठभागः(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_गणः_norm(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_l2_norm(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_बहिर्गमनम्_prod(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_scale(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_समुच्चयः(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_cpy(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_cont(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_get_पङ्क्तिकाः(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_get_पङ्क्तिकाः_पृष्ठभागः(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_समुच्चयः_पङ्क्तिकाः(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_diag(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_diag_आवरणम्_inf(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_diag_आवरणम्_zero(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_soft_max(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_soft_max_ext_पृष्ठभागः(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_rope(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_rope_पृष्ठभागः(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_clamp(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_conv_transpose_1d(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_im2col(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_im2col_पृष्ठभागः_f32(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_im2col_3d(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_col2im_1d(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_conv_2d(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_conv_3d(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_conv_transpose_2d(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_conv_2d_dw(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_pool_1d(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_pool_2d(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_pool_2d_पृष्ठभागः(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_upscale(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_pad(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_pad_reflect_1d(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_roll(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_arange(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_timestep_embedding(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_argsort(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_शिखरम्_k(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_leaky_relu(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_tri(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_fill(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_flash_attn_ext(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_flash_attn_पृष्ठभागः(
        ध्रुव संरचना ggml_compute_params * params,
        ध्रुव bool masked,
        संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_ssm_conv(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_ssm_अवलोकनम्(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_win_part(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_win_unpart(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_unary(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_glu(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_get_rel_pos(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_सङ्कलय_rel_pos(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_rwkv_wkv6(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_rwkv_wkv7(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_solve_tri(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_gla(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_gated_delta_net(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_lightning_indexer(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_dsv4_hc_comb(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_dsv4_hc_pre(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_dsv4_hc_post(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_शिखरम्_k_आवरणम्(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_map_custom1(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_map_custom2(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_map_custom3(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_custom(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_cross_entropy_loss(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_cross_entropy_loss_पृष्ठभागः(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_opt_चरणः_adamw(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_mul_mat(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_fwht(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_opt_चरणः_sgd(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
#यदि_परिभाषितम् __cplusplus
}
#यदि_अन्तः
