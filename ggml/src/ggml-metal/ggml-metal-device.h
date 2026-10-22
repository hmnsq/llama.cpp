#ਹਦਾਇਤ once

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __cplusplus
ਬਾਹਰੀ "C" {
#ਜੇ_ਅੰਤ

ਬਣਤਰ ggml_metal_ਬਫਰ_id {
    ਖਾਲੀ * metal; // id<MTLBuffer>
    size_t offs;
};

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ ggml_metal_device * ggml_metal_device_t;

//
// MTLFunctionConstantValues wrapper
//

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ ggml_metal_cv * ggml_metal_cv_t;

ggml_metal_cv_t ggml_metal_cv_init(ਖਾਲੀ);
ਖਾਲੀ ggml_metal_cv_free(ggml_metal_cv_t cv);

ਖਾਲੀ ggml_metal_cv_ਸੈੱਟ_int16(ggml_metal_cv_t cv, int16_t ਮੁੱਲ, int32_t idx);
ਖਾਲੀ ggml_metal_cv_ਸੈੱਟ_int32(ggml_metal_cv_t cv, int32_t ਮੁੱਲ, int32_t idx);
ਖਾਲੀ ggml_metal_cv_ਸੈੱਟ_bool (ggml_metal_cv_t cv, bool    ਮੁੱਲ, int32_t idx);

//
// MTLComputePipelineState wrapper
//

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ ggml_metal_pipeline * ggml_metal_pipeline_t;

ggml_metal_pipeline_t ggml_metal_pipeline_init(ਖਾਲੀ);
ਖਾਲੀ ggml_metal_pipeline_free(ggml_metal_pipeline_t pipeline);

// a collection of pipelines
ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ ggml_metal_pipelines * ggml_metal_pipelines_t;

ggml_metal_pipelines_t ggml_metal_pipelines_init(ਖਾਲੀ);
ਖਾਲੀ ggml_metal_pipelines_free(ggml_metal_pipelines_t ppls);

ਖਾਲੀ                  ggml_metal_pipelines_ਜੋੜੋ(ggml_metal_pipelines_t ppls, ਅਚਲ ਅੱਖਰ * name, ggml_metal_pipeline_t pipeline);
ggml_metal_pipeline_t ggml_metal_pipelines_get(ggml_metal_pipelines_t ppls, ਅਚਲ ਅੱਖਰ * name);

ਬਣਤਰ ggml_metal_pipeline_with_params {
    ggml_metal_pipeline_t pipeline;

    ਪੂਰਨ nsg;

    ਪੂਰਨ nr0;
    ਪੂਰਨ nr1;

    size_t smem;

    bool c4;
    bool cnt;
};

ਪੂਰਨ ggml_metal_pipeline_max_theads_ਪ੍ਰਤੀ_threadgroup(ਬਣਤਰ ggml_metal_pipeline_with_params pipeline);

//
// MTLCommandBuffer wrapper
//

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਖਾਲੀ * ggml_metal_cmd_buf_t;

//
// MTLComputeCommandEncoder wrapper
//

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ ggml_metal_encoder * ggml_metal_encoder_t;

ggml_metal_encoder_t ggml_metal_encoder_init(ggml_metal_cmd_buf_t cmd_buf_raw, bool concurrent);
ਖਾਲੀ ggml_metal_encoder_free(ggml_metal_encoder_t encoder);

ਖਾਲੀ ggml_metal_encoder_debug_ਸਮੂਹ_ਧੱਕੋ(ggml_metal_encoder_t encoder, ਅਚਲ ਅੱਖਰ * name);
ਖਾਲੀ ggml_metal_encoder_debug_ਸਮੂਹ_ਕੱਢੋ (ggml_metal_encoder_t encoder);

ਖਾਲੀ ggml_metal_encoder_ਸੈੱਟ_pipeline(ggml_metal_encoder_t encoder, ਬਣਤਰ ggml_metal_pipeline_with_params pipeline);

ਖਾਲੀ ggml_metal_encoder_ਸੈੱਟ_ਬਾਈਟਾਂ (ggml_metal_encoder_t encoder, ਖਾਲੀ * ਅੰਕੜੇ, size_t ਮਾਪ, ਪੂਰਨ idx);
ਖਾਲੀ ggml_metal_encoder_ਸੈੱਟ_ਬਫਰ(ggml_metal_encoder_t encoder, ਬਣਤਰ ggml_metal_ਬਫਰ_id ਬਫਰ, ਪੂਰਨ idx);

ਖਾਲੀ ggml_metal_encoder_ਸੈੱਟ_threadgroup_ਸਮ੍ਰਿਤੀ_ਮਾਪ(ggml_metal_encoder_t encoder, size_t ਮਾਪ, ਪੂਰਨ idx);

ਖਾਲੀ ggml_metal_encoder_dispatch_threadgroups(ggml_metal_encoder_t encoder, ਪੂਰਨ tg0, ਪੂਰਨ tg1, ਪੂਰਨ tg2, ਪੂਰਨ tptg0, ਪੂਰਨ tptg1, ਪੂਰਨ tptg2);

ਖਾਲੀ ggml_metal_encoder_ਸਮ੍ਰਿਤੀ_barrier(ggml_metal_encoder_t encoder);

ਖਾਲੀ ggml_metal_encoder_ਸਿਰਾ_encoding(ggml_metal_encoder_t encoder);

//
// MTLLibrary wrapper
//

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ ggml_metal_library * ggml_metal_library_t;

ggml_metal_library_t ggml_metal_library_init            (ggml_metal_device_t dev);
ggml_metal_library_t ggml_metal_library_init_from_ਸਰੋਤ(ggml_metal_device_t dev, ਅਚਲ ਅੱਖਰ * ਸਰੋਤ, bool verbose);

ਖਾਲੀ ggml_metal_library_free(ggml_metal_library_t lib);

ggml_metal_device_t ggml_metal_library_get_device(ggml_metal_library_t lib);

ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline    (ggml_metal_library_t lib, ਅਚਲ ਅੱਖਰ * name);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_compile_pipeline(ggml_metal_library_t lib, ਅਚਲ ਅੱਖਰ * ਅਧਾਰ, ਅਚਲ ਅੱਖਰ * name, ggml_metal_cv_t cv);

ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ਅਧਾਰ              (ggml_metal_library_t lib, ਗਿਣਤੀ ggml_op op);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cpy               (ggml_metal_library_t lib, ਗਿਣਤੀ ggml_type tsrc, ਗਿਣਤੀ ggml_type tdst);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_1d           (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op, ਗਿਣਤੀ ggml_op_pool op_pool);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_2d           (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op, ਗਿਣਤੀ ggml_op_pool op_pool);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_get_ਪੰਕਤੀਆਂ          (ggml_metal_library_t lib, ਗਿਣਤੀ ggml_type tsrc);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ਸੈੱਟ_ਪੰਕਤੀਆਂ          (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_diag              (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_repeat            (ggml_metal_library_t lib, ਗਿਣਤੀ ggml_type tsrc);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_concat            (ggml_metal_library_t lib, ਗਿਣਤੀ ggml_type tsrc);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_unary             (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_silu_ਪਿੱਠ         (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_glu               (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ਜੋੜ               (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ਜੋੜ_ਪੰਕਤੀਆਂ          (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cumsum_blk        (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cumsum_ਜੋੜੋ        (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_tri               (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_soft_max          (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_lightning_indexer (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_hc           (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv          (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op, int32_t nc, bool use_silu);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv_batched  (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op, ਪੂਰਨ ssm_conv_bs, int32_t nc, bool use_silu);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_ਜਾਂਚ          (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op, bool ਪੂਛ);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_ਜਾਂਚ_ssd_mma  (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rwkv              (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_gated_delta_net   (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_solve_tri         (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_ext        (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op, ਪੂਰਨ nsg, ਪੂਰਨ nxpsg, ਪੂਰਨ r1ptg);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_mma        (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op, ਪੂਰਨ nsg, ਪੂਰਨ nt, ਪੂਰਨ rt, bool ਜੋੜੋ_2);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_mma_auto   (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op, bool ਜੋੜੋ_2);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm            (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv            (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_map0    (ggml_metal_library_t lib, ਪੂਰਨ ne02, ਪੂਰਨ ne20);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_amax(ggml_metal_library_t lib);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_amax_part(ggml_metal_library_t lib);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id         (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_id         (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax            (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax_part       (ggml_metal_library_t lib);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax_final      (ggml_metal_library_t lib);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argsort           (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argsort_ਮਿਲਾਓ     (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_fwht              (ggml_metal_library_t lib, ਪੂਰਨ n, ਗਿਣਤੀ ggml_type tsrc);

ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ਸਿਖਰ_k             (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ਸਿਖਰ_k_radix       (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ਸਿਖਰ_k_ਮਿਲਾਓ       (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_topk_moe          (ggml_metal_library_t lib, int32_t n_expert, int32_t ਸਿਖਰ_k, bool with_norm);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_moe_reduce        (ggml_metal_library_t lib, int32_t n_expert_ਵਰਤਿਆ);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin               (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op, int32_t n_fuse);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin_one           (ggml_metal_library_t lib, ਗਿਣਤੀ ggml_op op);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_l2_norm           (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ਸਮੂਹ_norm        (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_norm              (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op, int32_t n_fuse);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_norm_scale        (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rope              (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_im2col            (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_1d (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_2d (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_col2im_1d         (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_snake             (ggml_metal_library_t lib, ਗਿਣਤੀ ggml_type type);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_2d           (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_2d_dw        (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op, bool tiled);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_3d           (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_upscale           (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad               (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad_reflect_1d    (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_roll              (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_arange            (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_timestep_embedding(ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_ਕਦਮ_adamw    (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_ਕਦਮ_sgd      (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_memset            (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op);
ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ਗਿਣਤੀ_equal       (ggml_metal_library_t lib, ਅਚਲ ਬਣਤਰ ggml_tensor * op);

ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_pad(
        ggml_metal_library_t lib,
        ਅਚਲ ਬਣਤਰ ggml_tensor * op,
        bool    has_ਨਕਾਬ,
        int32_t ncpsg);

ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_kv_f16(
        ggml_metal_library_t lib,
        ਅਚਲ ਬਣਤਰ ggml_tensor * op);

ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_blk(
        ggml_metal_library_t lib,
        ਅਚਲ ਬਣਤਰ ggml_tensor * op,
        int32_t nqptg,
        int32_t ncpsg);

ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_tensor(
        ggml_metal_library_t lib,
        ਅਚਲ ਬਣਤਰ ggml_tensor * op,
        bool has_ਨਕਾਬ,
        bool has_sinks,
        bool has_bias,
        bool has_scap);

ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext(
        ggml_metal_library_t lib,
        ਅਚਲ ਬਣਤਰ ggml_tensor * op,
        bool    has_ਨਕਾਬ,
        bool    has_sinks,
        bool    has_bias,
        bool    has_scap,
        bool    has_kvpad,
        int32_t nsg,
        bool    use_kv_f16,
        int32_t ns10,
        int32_t ns20);

ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec_idx(
        ggml_metal_library_t lib,
        ਅਚਲ ਬਣਤਰ ggml_tensor * op);

ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec(
        ggml_metal_library_t lib,
        ਅਚਲ ਬਣਤਰ ggml_tensor * op,
        bool    has_ਨਕਾਬ,
        bool    has_sinks,
        bool    has_bias,
        bool    has_scap,
        bool    has_kvpad,
        bool    has_sparse,
        int32_t nqpsg,
        int32_t ne,
        int32_t nsg,
        int32_t nwg,
        bool    use_kv_f16,
        int32_t ns10,
        int32_t ns20);

ਬਣਤਰ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce(
        ggml_metal_library_t lib,
        ਅਚਲ ਬਣਤਰ ggml_tensor * op,
        int32_t dv,
        int32_t nwg);

// MTLResidencySet wrapper

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਖਾਲੀ * ggml_metal_rset_t;

// a collection of residency sets (non-owning)
ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ ggml_metal_rsets * ggml_metal_rsets_t;

ggml_metal_rsets_t ggml_metal_rsets_init(ggml_metal_device_t dev);
ਖਾਲੀ ggml_metal_rsets_free(ggml_metal_rsets_t rsets);

//
// device
//

ਗਿਣਤੀ ggml_metal_device_id {
    GGML_METAL_DEVICE_GENERIC = 0,

    GGML_METAL_DEVICE_M1,
    GGML_METAL_DEVICE_M1_PRO,
    GGML_METAL_DEVICE_M1_MAX,
    GGML_METAL_DEVICE_M1_ULTRA,
    GGML_METAL_DEVICE_M2,
    GGML_METAL_DEVICE_M2_PRO,
    GGML_METAL_DEVICE_M2_MAX,
    GGML_METAL_DEVICE_M2_ULTRA,
    GGML_METAL_DEVICE_M3,
    GGML_METAL_DEVICE_M3_PRO,
    GGML_METAL_DEVICE_M3_MAX,
    GGML_METAL_DEVICE_M3_ULTRA,
    GGML_METAL_DEVICE_M4,
    GGML_METAL_DEVICE_M4_PRO,
    GGML_METAL_DEVICE_M4_MAX,
    GGML_METAL_DEVICE_M5,
    GGML_METAL_DEVICE_M5_PRO,
    GGML_METAL_DEVICE_M5_MAX,
    GGML_METAL_DEVICE_M5_ULTRA,
    GGML_METAL_DEVICE_A18_PRO,
};

ਅਚਲ ਅੱਖਰ * ggml_metal_device_id_token(ਗਿਣਤੀ ggml_metal_device_id id);

ਬਣਤਰ ggml_metal_device_props {
    ਪੂਰਨ device;
    ਪੂਰਨ device_phys;
    ਪੂਰਨ device_virt;
    ਅੱਖਰ name[128];
    ਅੱਖਰ desc[128];

    size_t max_ਬਫਰ_ਮਾਪ;
    size_t max_working_ਸੈੱਟ_ਮਾਪ;
    size_t max_theadgroup_ਸਮ੍ਰਿਤੀ_ਮਾਪ;

    bool has_simdgroup_reduction;
    bool has_simdgroup_mm;
    bool has_unified_ਸਮ੍ਰਿਤੀ;
    bool has_bfloat;
    bool has_tensor;
    bool use_residency_sets;
    bool use_shared_buffers;

    ਗਿਣਤੀ ggml_metal_device_id device_id;
    ਪੂਰਨ gpu_family;

    ਪੂਰਨ op_offload_min_batch_ਮਾਪ;
};

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ ggml_metal_event * ggml_metal_event_t;

ਖਾਲੀ ggml_metal_event_encode_signal(ggml_metal_event_t ev, ggml_metal_cmd_buf_t cmd_buf);
ਖਾਲੀ ggml_metal_event_encode_wait  (ggml_metal_event_t ev, ggml_metal_cmd_buf_t cmd_buf);

ggml_metal_device_t ggml_metal_device_init(ਪੂਰਨ device, ਪੂਰਨ n_devices);
ਖਾਲੀ ggml_metal_device_free(ggml_metal_device_t dev);

ggml_metal_device_t ggml_metal_device_get(ਪੂਰਨ device, ਪੂਰਨ n_devices);

ਖਾਲੀ * ggml_metal_device_get_obj  (ggml_metal_device_t dev); // id<MTLDevice>
ਖਾਲੀ * ggml_metal_device_get_ਕਤਾਰ(ggml_metal_device_t dev); // id<MTLCommandQueue>

ggml_metal_library_t ggml_metal_device_get_library(ggml_metal_device_t dev);

ਖਾਲੀ ggml_metal_device_rsets_ਜੋੜੋ(ggml_metal_device_t dev, ggml_metal_rset_t rset);
ਖਾਲੀ ggml_metal_device_rsets_rm (ggml_metal_device_t dev, ggml_metal_rset_t rset);

ਖਾਲੀ ggml_metal_device_rsets_keep_alive(ggml_metal_device_t dev);

ggml_metal_event_t ggml_metal_device_event_init(ggml_metal_device_t dev);
ਖਾਲੀ ggml_metal_device_event_free(ggml_metal_device_t dev, ggml_metal_event_t ev);
ਖਾਲੀ ggml_metal_device_event_synchronize(ggml_metal_device_t dev, ggml_metal_event_t ev);

ਖਾਲੀ ggml_metal_device_get_ਸਮ੍ਰਿਤੀ(ggml_metal_device_t dev, size_t * ਮੁਕਤ, size_t * ਕੁੱਲ);
bool ggml_metal_device_supports_op(ggml_metal_device_t dev, ਅਚਲ ਬਣਤਰ ggml_tensor * op);

ਅਚਲ ਬਣਤਰ ggml_metal_device_props * ggml_metal_device_get_props(ggml_metal_device_t dev);

ਬਣਤਰ ggml_metal_fusion_info;

// the device-owned fusion debugging context (NULL unless fusion debugging is enabled)
ਬਣਤਰ ggml_metal_fusion_info * ggml_metal_device_get_fusion_info(ggml_metal_device_t dev);

//
// device buffers
//

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ ggml_metal_ਬਫਰ * ggml_metal_ਬਫਰ_t;

ggml_metal_ਬਫਰ_t ggml_metal_ਬਫਰ_init(ggml_metal_device_t dev, size_t ਮਾਪ, bool shared);
ggml_metal_ਬਫਰ_t ggml_metal_ਬਫਰ_map (ggml_metal_device_t dev, ਖਾਲੀ * ptr, size_t ਮਾਪ, size_t max_tensor_ਮਾਪ);

ਖਾਲੀ   ggml_metal_ਬਫਰ_free     (ggml_metal_ਬਫਰ_t buf);
ਖਾਲੀ * ggml_metal_ਬਫਰ_get_ਅਧਾਰ (ggml_metal_ਬਫਰ_t buf);
bool   ggml_metal_ਬਫਰ_is_shared(ggml_metal_ਬਫਰ_t buf);

ਖਾਲੀ   ggml_metal_ਬਫਰ_memset_tensor(ggml_metal_ਬਫਰ_t buf, ਬਣਤਰ ggml_tensor * tensor, uint8_t ਮੁੱਲ, size_t ਵਿੱਥ, size_t ਮਾਪ);
ਖਾਲੀ   ggml_metal_ਬਫਰ_ਸੈੱਟ_tensor   (ggml_metal_ਬਫਰ_t buf, ਬਣਤਰ ggml_tensor * tensor, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t ਮਾਪ);
ਖਾਲੀ   ggml_metal_ਬਫਰ_get_tensor   (ggml_metal_ਬਫਰ_t buf, ਅਚਲ ਬਣਤਰ ggml_tensor * tensor, ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t ਮਾਪ);
bool   ggml_metal_ਬਫਰ_cpy_tensor   (ggml_metal_ਬਫਰ_t buf, ਅਚਲ ਬਣਤਰ ggml_tensor * src, ਬਣਤਰ ggml_tensor * dst);
ਖਾਲੀ   ggml_metal_ਬਫਰ_clear        (ggml_metal_ਬਫਰ_t buf, uint8_t ਮੁੱਲ);

// finds the Metal buffer that contains the tensor data on the GPU device
// the assumption is that there is 1-to-1 mapping between the host and device memory buffers, so we can find the
// Metal buffer based on the host memory pointer
//
ਬਣਤਰ ggml_metal_ਬਫਰ_id ggml_metal_ਬਫਰ_get_id(ggml_metal_ਬਫਰ_t buf, ਅਚਲ ਬਣਤਰ ggml_tensor * t);

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __cplusplus
}
#ਜੇ_ਅੰਤ
