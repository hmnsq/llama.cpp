#సూచన once

#అన్తర్భావ "ggml.h"

#నిర్వచితమైతే __cplusplus
బాహ్య "C" {
#అయితే_ముగింపు

నిర్మాణం ggml_metal_బఫరు_id {
    శూన్య * metal; // id<MTLBuffer>
    size_t offs;
};

రకనిర్వచనం నిర్మాణం ggml_metal_device * ggml_metal_device_t;

//
// MTLFunctionConstantValues wrapper
//

రకనిర్వచనం నిర్మాణం ggml_metal_cv * ggml_metal_cv_t;

ggml_metal_cv_t ggml_metal_cv_init(శూన్య);
శూన్య ggml_metal_cv_free(ggml_metal_cv_t cv);

శూన్య ggml_metal_cv_సమితి_int16(ggml_metal_cv_t cv, int16_t విలువ, int32_t idx);
శూన్య ggml_metal_cv_సమితి_int32(ggml_metal_cv_t cv, int32_t విలువ, int32_t idx);
శూన్య ggml_metal_cv_సమితి_bool (ggml_metal_cv_t cv, bool    విలువ, int32_t idx);

//
// MTLComputePipelineState wrapper
//

రకనిర్వచనం నిర్మాణం ggml_metal_pipeline * ggml_metal_pipeline_t;

ggml_metal_pipeline_t ggml_metal_pipeline_init(శూన్య);
శూన్య ggml_metal_pipeline_free(ggml_metal_pipeline_t pipeline);

// a collection of pipelines
రకనిర్వచనం నిర్మాణం ggml_metal_pipelines * ggml_metal_pipelines_t;

ggml_metal_pipelines_t ggml_metal_pipelines_init(శూన్య);
శూన్య ggml_metal_pipelines_free(ggml_metal_pipelines_t ppls);

శూన్య                  ggml_metal_pipelines_కూడు(ggml_metal_pipelines_t ppls, స్థిరరాశి అక్షరం * name, ggml_metal_pipeline_t pipeline);
ggml_metal_pipeline_t ggml_metal_pipelines_get(ggml_metal_pipelines_t ppls, స్థిరరాశి అక్షరం * name);

నిర్మాణం ggml_metal_pipeline_with_params {
    ggml_metal_pipeline_t pipeline;

    పూర్ణాంకం nsg;

    పూర్ణాంకం nr0;
    పూర్ణాంకం nr1;

    size_t smem;

    bool c4;
    bool cnt;
};

పూర్ణాంకం ggml_metal_pipeline_max_theads_చొప్పున_threadgroup(నిర్మాణం ggml_metal_pipeline_with_params pipeline);

//
// MTLCommandBuffer wrapper
//

రకనిర్వచనం శూన్య * ggml_metal_cmd_buf_t;

//
// MTLComputeCommandEncoder wrapper
//

రకనిర్వచనం నిర్మాణం ggml_metal_encoder * ggml_metal_encoder_t;

ggml_metal_encoder_t ggml_metal_encoder_init(ggml_metal_cmd_buf_t cmd_buf_raw, bool concurrent);
శూన్య ggml_metal_encoder_free(ggml_metal_encoder_t encoder);

శూన్య ggml_metal_encoder_debug_సమూహం_నెట్టు(ggml_metal_encoder_t encoder, స్థిరరాశి అక్షరం * name);
శూన్య ggml_metal_encoder_debug_సమూహం_తీయి (ggml_metal_encoder_t encoder);

శూన్య ggml_metal_encoder_సమితి_pipeline(ggml_metal_encoder_t encoder, నిర్మాణం ggml_metal_pipeline_with_params pipeline);

శూన్య ggml_metal_encoder_సమితి_బైట్లు (ggml_metal_encoder_t encoder, శూన్య * దత్తాంశం, size_t కొలత, పూర్ణాంకం idx);
శూన్య ggml_metal_encoder_సమితి_బఫరు(ggml_metal_encoder_t encoder, నిర్మాణం ggml_metal_బఫరు_id బఫరు, పూర్ణాంకం idx);

శూన్య ggml_metal_encoder_సమితి_threadgroup_జ్ఞాపకం_కొలత(ggml_metal_encoder_t encoder, size_t కొలత, పూర్ణాంకం idx);

శూన్య ggml_metal_encoder_dispatch_threadgroups(ggml_metal_encoder_t encoder, పూర్ణాంకం tg0, పూర్ణాంకం tg1, పూర్ణాంకం tg2, పూర్ణాంకం tptg0, పూర్ణాంకం tptg1, పూర్ణాంకం tptg2);

శూన్య ggml_metal_encoder_జ్ఞాపకం_barrier(ggml_metal_encoder_t encoder);

శూన్య ggml_metal_encoder_చివర_encoding(ggml_metal_encoder_t encoder);

//
// MTLLibrary wrapper
//

రకనిర్వచనం నిర్మాణం ggml_metal_library * ggml_metal_library_t;

ggml_metal_library_t ggml_metal_library_init            (ggml_metal_device_t dev);
ggml_metal_library_t ggml_metal_library_init_from_ఆధారం(ggml_metal_device_t dev, స్థిరరాశి అక్షరం * ఆధారం, bool verbose);

శూన్య ggml_metal_library_free(ggml_metal_library_t lib);

ggml_metal_device_t ggml_metal_library_get_device(ggml_metal_library_t lib);

నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline    (ggml_metal_library_t lib, స్థిరరాశి అక్షరం * name);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_compile_pipeline(ggml_metal_library_t lib, స్థిరరాశి అక్షరం * పునాది, స్థిరరాశి అక్షరం * name, ggml_metal_cv_t cv);

నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_పునాది              (ggml_metal_library_t lib, గణన ggml_op op);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cpy               (ggml_metal_library_t lib, గణన ggml_type tsrc, గణన ggml_type tdst);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_1d           (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op, గణన ggml_op_pool op_pool);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_2d           (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op, గణన ggml_op_pool op_pool);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_get_అడ్డువరుసలు          (ggml_metal_library_t lib, గణన ggml_type tsrc);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_సమితి_అడ్డువరుసలు          (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_diag              (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_repeat            (ggml_metal_library_t lib, గణన ggml_type tsrc);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_concat            (ggml_metal_library_t lib, గణన ggml_type tsrc);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_unary             (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_silu_వెనుక         (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_glu               (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_కూడిక               (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_కూడిక_అడ్డువరుసలు          (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cumsum_blk        (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cumsum_కూడు        (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_tri               (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_soft_max          (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_lightning_indexer (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_hc           (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv          (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op, int32_t nc, bool use_silu);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv_batched  (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op, పూర్ణాంకం ssm_conv_bs, int32_t nc, bool use_silu);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_పరిశీలన          (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op, bool తోక);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_పరిశీలన_ssd_mma  (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rwkv              (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_gated_delta_net   (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_solve_tri         (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_ext        (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op, పూర్ణాంకం nsg, పూర్ణాంకం nxpsg, పూర్ణాంకం r1ptg);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_mma        (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op, పూర్ణాంకం nsg, పూర్ణాంకం nt, పూర్ణాంకం rt, bool కూడు_2);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_mma_auto   (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op, bool కూడు_2);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm            (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv            (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_map0    (ggml_metal_library_t lib, పూర్ణాంకం ne02, పూర్ణాంకం ne20);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_amax(ggml_metal_library_t lib);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_amax_part(ggml_metal_library_t lib);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id         (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_id         (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax            (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax_part       (ggml_metal_library_t lib);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax_final      (ggml_metal_library_t lib);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argsort           (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argsort_విలీనం     (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_fwht              (ggml_metal_library_t lib, పూర్ణాంకం n, గణన ggml_type tsrc);

నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_శిఖరం_k             (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_శిఖరం_k_radix       (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_శిఖరం_k_విలీనం       (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_topk_moe          (ggml_metal_library_t lib, int32_t n_expert, int32_t శిఖరం_k, bool with_norm);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_moe_reduce        (ggml_metal_library_t lib, int32_t n_expert_వాడిన);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin               (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op, int32_t n_fuse);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin_one           (ggml_metal_library_t lib, గణన ggml_op op);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_l2_norm           (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_సమూహం_norm        (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_norm              (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op, int32_t n_fuse);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_norm_scale        (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rope              (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_im2col            (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_1d (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_2d (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_col2im_1d         (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_snake             (ggml_metal_library_t lib, గణన ggml_type type);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_2d           (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_2d_dw        (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op, bool tiled);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_3d           (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_upscale           (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad               (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad_reflect_1d    (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_roll              (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_arange            (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_timestep_embedding(ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_అడుగు_adamw    (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_అడుగు_sgd      (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_memset            (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op);
నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_లెక్క_equal       (ggml_metal_library_t lib, స్థిరరాశి నిర్మాణం ggml_tensor * op);

నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_pad(
        ggml_metal_library_t lib,
        స్థిరరాశి నిర్మాణం ggml_tensor * op,
        bool    has_ముసుగు,
        int32_t ncpsg);

నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_kv_f16(
        ggml_metal_library_t lib,
        స్థిరరాశి నిర్మాణం ggml_tensor * op);

నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_blk(
        ggml_metal_library_t lib,
        స్థిరరాశి నిర్మాణం ggml_tensor * op,
        int32_t nqptg,
        int32_t ncpsg);

నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_tensor(
        ggml_metal_library_t lib,
        స్థిరరాశి నిర్మాణం ggml_tensor * op,
        bool has_ముసుగు,
        bool has_sinks,
        bool has_bias,
        bool has_scap);

నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext(
        ggml_metal_library_t lib,
        స్థిరరాశి నిర్మాణం ggml_tensor * op,
        bool    has_ముసుగు,
        bool    has_sinks,
        bool    has_bias,
        bool    has_scap,
        bool    has_kvpad,
        int32_t nsg,
        bool    use_kv_f16,
        int32_t ns10,
        int32_t ns20);

నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec_idx(
        ggml_metal_library_t lib,
        స్థిరరాశి నిర్మాణం ggml_tensor * op);

నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec(
        ggml_metal_library_t lib,
        స్థిరరాశి నిర్మాణం ggml_tensor * op,
        bool    has_ముసుగు,
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

నిర్మాణం ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce(
        ggml_metal_library_t lib,
        స్థిరరాశి నిర్మాణం ggml_tensor * op,
        int32_t dv,
        int32_t nwg);

// MTLResidencySet wrapper

రకనిర్వచనం శూన్య * ggml_metal_rset_t;

// a collection of residency sets (non-owning)
రకనిర్వచనం నిర్మాణం ggml_metal_rsets * ggml_metal_rsets_t;

ggml_metal_rsets_t ggml_metal_rsets_init(ggml_metal_device_t dev);
శూన్య ggml_metal_rsets_free(ggml_metal_rsets_t rsets);

//
// device
//

గణన ggml_metal_device_id {
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

స్థిరరాశి అక్షరం * ggml_metal_device_id_token(గణన ggml_metal_device_id id);

నిర్మాణం ggml_metal_device_props {
    పూర్ణాంకం device;
    పూర్ణాంకం device_phys;
    పూర్ణాంకం device_virt;
    అక్షరం name[128];
    అక్షరం desc[128];

    size_t max_బఫరు_కొలత;
    size_t max_working_సమితి_కొలత;
    size_t max_theadgroup_జ్ఞాపకం_కొలత;

    bool has_simdgroup_reduction;
    bool has_simdgroup_mm;
    bool has_unified_జ్ఞాపకం;
    bool has_bfloat;
    bool has_tensor;
    bool use_residency_sets;
    bool use_shared_buffers;

    గణన ggml_metal_device_id device_id;
    పూర్ణాంకం gpu_family;

    పూర్ణాంకం op_offload_min_batch_కొలత;
};

రకనిర్వచనం నిర్మాణం ggml_metal_event * ggml_metal_event_t;

శూన్య ggml_metal_event_encode_signal(ggml_metal_event_t ev, ggml_metal_cmd_buf_t cmd_buf);
శూన్య ggml_metal_event_encode_wait  (ggml_metal_event_t ev, ggml_metal_cmd_buf_t cmd_buf);

ggml_metal_device_t ggml_metal_device_init(పూర్ణాంకం device, పూర్ణాంకం n_devices);
శూన్య ggml_metal_device_free(ggml_metal_device_t dev);

ggml_metal_device_t ggml_metal_device_get(పూర్ణాంకం device, పూర్ణాంకం n_devices);

శూన్య * ggml_metal_device_get_obj  (ggml_metal_device_t dev); // id<MTLDevice>
శూన్య * ggml_metal_device_get_వరుస(ggml_metal_device_t dev); // id<MTLCommandQueue>

ggml_metal_library_t ggml_metal_device_get_library(ggml_metal_device_t dev);

శూన్య ggml_metal_device_rsets_కూడు(ggml_metal_device_t dev, ggml_metal_rset_t rset);
శూన్య ggml_metal_device_rsets_rm (ggml_metal_device_t dev, ggml_metal_rset_t rset);

శూన్య ggml_metal_device_rsets_keep_alive(ggml_metal_device_t dev);

ggml_metal_event_t ggml_metal_device_event_init(ggml_metal_device_t dev);
శూన్య ggml_metal_device_event_free(ggml_metal_device_t dev, ggml_metal_event_t ev);
శూన్య ggml_metal_device_event_synchronize(ggml_metal_device_t dev, ggml_metal_event_t ev);

శూన్య ggml_metal_device_get_జ్ఞాపకం(ggml_metal_device_t dev, size_t * విడుదల, size_t * మొత్తం);
bool ggml_metal_device_supports_op(ggml_metal_device_t dev, స్థిరరాశి నిర్మాణం ggml_tensor * op);

స్థిరరాశి నిర్మాణం ggml_metal_device_props * ggml_metal_device_get_props(ggml_metal_device_t dev);

నిర్మాణం ggml_metal_fusion_info;

// the device-owned fusion debugging context (NULL unless fusion debugging is enabled)
నిర్మాణం ggml_metal_fusion_info * ggml_metal_device_get_fusion_info(ggml_metal_device_t dev);

//
// device buffers
//

రకనిర్వచనం నిర్మాణం ggml_metal_బఫరు * ggml_metal_బఫరు_t;

ggml_metal_బఫరు_t ggml_metal_బఫరు_init(ggml_metal_device_t dev, size_t కొలత, bool shared);
ggml_metal_బఫరు_t ggml_metal_బఫరు_map (ggml_metal_device_t dev, శూన్య * ptr, size_t కొలత, size_t max_tensor_కొలత);

శూన్య   ggml_metal_బఫరు_free     (ggml_metal_బఫరు_t buf);
శూన్య * ggml_metal_బఫరు_get_పునాది (ggml_metal_బఫరు_t buf);
bool   ggml_metal_బఫరు_is_shared(ggml_metal_బఫరు_t buf);

శూన్య   ggml_metal_బఫరు_memset_tensor(ggml_metal_బఫరు_t buf, నిర్మాణం ggml_tensor * tensor, uint8_t విలువ, size_t స్థానభ్రంశం, size_t కొలత);
శూన్య   ggml_metal_బఫరు_సమితి_tensor   (ggml_metal_బఫరు_t buf, నిర్మాణం ggml_tensor * tensor, స్థిరరాశి శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t కొలత);
శూన్య   ggml_metal_బఫరు_get_tensor   (ggml_metal_బఫరు_t buf, స్థిరరాశి నిర్మాణం ggml_tensor * tensor, శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t కొలత);
bool   ggml_metal_బఫరు_cpy_tensor   (ggml_metal_బఫరు_t buf, స్థిరరాశి నిర్మాణం ggml_tensor * src, నిర్మాణం ggml_tensor * dst);
శూన్య   ggml_metal_బఫరు_clear        (ggml_metal_బఫరు_t buf, uint8_t విలువ);

// finds the Metal buffer that contains the tensor data on the GPU device
// the assumption is that there is 1-to-1 mapping between the host and device memory buffers, so we can find the
// Metal buffer based on the host memory pointer
//
నిర్మాణం ggml_metal_బఫరు_id ggml_metal_బఫరు_get_id(ggml_metal_బఫరు_t buf, స్థిరరాశి నిర్మాణం ggml_tensor * t);

#నిర్వచితమైతే __cplusplus
}
#అయితే_ముగింపు
