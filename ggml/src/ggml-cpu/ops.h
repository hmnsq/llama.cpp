#आशय once

#समावेश "ggml.h"

//
// cache line
//
// TODO: rework CACHE_LINE_SIZE so std::hardware_destructive_interference_size
// can be used consistently between C and C++ TUs; the previous macro form
// diverged based on include order and undersized the work buffer.
// ref: https://github.com/ggml-org/llama.cpp/pull/28882
#अगर defined(__POWER9_VECTOR__)
#घोषणा CACHE_पंक्ति_आकार 128
#या_अगर defined(__VXE__) || defined(__VXE2__)
#घोषणा CACHE_पंक्ति_आकार 256
#अन्यथा
#घोषणा CACHE_पंक्ति_आकार 64
#पूर्ण_अगर

अटल स्थिर माप_प्रकार CACHE_पंक्ति_आकार_F32 = CACHE_पंक्ति_आकार/माप(भग्न);

// Work buffer size for im2col operations in CONV2D
#घोषणा GGML_IM2_COL_WORK_आकार (16 * 1024 * 1024)

#अगर_घोषित __cplusplus
बाह्य "C" {
#पूर्ण_अगर

व्योम ggml_compute_forward_dup(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_add(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_add_id(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_add1(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_acc(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_योग(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_योग_rows(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_cumsum(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_mean(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_argmax(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_गिनती_equal(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_repeat(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_repeat_back(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_concat(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_silu_back(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_norm(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_rms_norm(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_rms_norm_mul_fused(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst_rms_norm, काष्ठा ggml_tensor * dst_mul);
व्योम ggml_compute_forward_rms_norm_back(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_group_norm(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_l2_norm(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_out_prod(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_scale(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_set(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_cpy(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_cont(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_get_rows(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_get_rows_back(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_set_rows(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_diag(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_diag_mask_inf(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_diag_mask_zero(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_soft_max(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_soft_max_ext_back(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_rope(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_rope_back(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_clamp(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_conv_transpose_1d(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_im2col(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_im2col_back_f32(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_im2col_3d(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_col2im_1d(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_conv_2d(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_conv_3d(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_conv_transpose_2d(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_conv_2d_dw(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_pool_1d(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_pool_2d(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_pool_2d_back(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_upscale(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_pad(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_pad_reflect_1d(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_roll(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_arange(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_timestep_embedding(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_argsort(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_शीर्ष_k(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_leaky_relu(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_tri(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_fill(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_flash_attn_ext(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_flash_attn_back(
        स्थिर काष्ठा ggml_compute_params * params,
        स्थिर bool masked,
        काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_ssm_conv(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_ssm_scan(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_win_part(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_win_unpart(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_unary(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_glu(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_get_rel_pos(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_add_rel_pos(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_rwkv_wkv6(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_rwkv_wkv7(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_solve_tri(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_gla(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_gated_delta_net(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_lightning_indexer(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_dsv4_hc_comb(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_dsv4_hc_pre(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_dsv4_hc_post(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_शीर्ष_k_mask(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_map_custom1(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_map_custom2(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_map_custom3(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_custom(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_cross_entropy_loss(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_cross_entropy_loss_back(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_opt_step_adamw(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_mul_mat(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_fwht(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
व्योम ggml_compute_forward_opt_step_sgd(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
#अगर_घोषित __cplusplus
}
#पूर्ण_अगर
