#సూచన once

#అన్తర్భావ "ggml.h"

//
// cache line
//
// TODO: rework CACHE_LINE_SIZE so std::hardware_destructive_interference_size
// can be used consistently between C and C++ TUs; the previous macro form
// diverged based on include order and undersized the work buffer.
// ref: https://github.com/ggml-org/llama.cpp/pull/28882
#ఒకవేళ defined(__POWER9_VECTOR__)
#నిర్వచించు CACHE_పంక్తి_కొలత 128
#లేదావేళ defined(__VXE__) || defined(__VXE2__)
#నిర్వచించు CACHE_పంక్తి_కొలత 256
#లేకపోతే
#నిర్వచించు CACHE_పంక్తి_కొలత 64
#అయితే_ముగింపు

స్థిర స్థిరరాశి size_t CACHE_పంక్తి_కొలత_F32 = CACHE_పంక్తి_కొలత/పరిమాణం(దశాంశం);

// Work buffer size for im2col operations in CONV2D
#నిర్వచించు GGML_IM2_నిలువు_WORK_కొలత (16 * 1024 * 1024)

#నిర్వచితమైతే __cplusplus
బాహ్య "C" {
#అయితే_ముగింపు

శూన్య ggml_compute_forward_dup(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_కూడు(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_కూడు_id(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_add1(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_acc(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_కూడిక(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_కూడిక_అడ్డువరుసలు(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_cumsum(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_సగటు(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_argmax(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_లెక్క_equal(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_repeat(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_repeat_వెనుక(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_concat(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_silu_వెనుక(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_norm(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_rms_norm(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_rms_norm_mul_fused(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst_rms_norm, నిర్మాణం ggml_tensor * dst_mul);
శూన్య ggml_compute_forward_rms_norm_వెనుక(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_సమూహం_norm(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_l2_norm(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_బహిర్గతం_prod(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_scale(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_సమితి(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_cpy(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_cont(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_get_అడ్డువరుసలు(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_get_అడ్డువరుసలు_వెనుక(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_సమితి_అడ్డువరుసలు(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_diag(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_diag_ముసుగు_inf(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_diag_ముసుగు_zero(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_soft_max(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_soft_max_ext_వెనుక(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_rope(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_rope_వెనుక(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_clamp(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_conv_transpose_1d(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_im2col(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_im2col_వెనుక_f32(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_im2col_3d(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_col2im_1d(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_conv_2d(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_conv_3d(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_conv_transpose_2d(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_conv_2d_dw(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_pool_1d(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_pool_2d(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_pool_2d_వెనుక(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_upscale(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_pad(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_pad_reflect_1d(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_roll(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_arange(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_timestep_embedding(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_argsort(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_శిఖరం_k(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_leaky_relu(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_tri(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_fill(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_flash_attn_ext(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_flash_attn_వెనుక(
        స్థిరరాశి నిర్మాణం ggml_compute_params * params,
        స్థిరరాశి bool masked,
        నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_ssm_conv(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_ssm_పరిశీలన(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_win_part(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_win_unpart(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_unary(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_glu(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_get_rel_pos(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_కూడు_rel_pos(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_rwkv_wkv6(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_rwkv_wkv7(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_solve_tri(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_gla(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_gated_delta_net(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_lightning_indexer(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_dsv4_hc_comb(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_dsv4_hc_pre(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_dsv4_hc_post(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_శిఖరం_k_ముసుగు(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_map_custom1(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_map_custom2(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_map_custom3(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_custom(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_cross_entropy_loss(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_cross_entropy_loss_వెనుక(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_opt_అడుగు_adamw(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_mul_mat(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_fwht(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
శూన్య ggml_compute_forward_opt_అడుగు_sgd(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst);
#నిర్వచితమైతే __cplusplus
}
#అయితే_ముగింపు
