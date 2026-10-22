#निर्देशः once

#अन्तर्भाव "ggml.h"

#यदि_परिभाषितम् __cplusplus
बाह्य "C" {
#यदि_अन्तः

संरचना ggml_metal_बफर_id {
    शून्य * metal; // id<MTLBuffer>
    size_t offs;
};

प्रकारनिर्देश संरचना ggml_metal_device * ggml_metal_device_t;

//
// MTLFunctionConstantValues wrapper
//

प्रकारनिर्देश संरचना ggml_metal_cv * ggml_metal_cv_t;

ggml_metal_cv_t ggml_metal_cv_init(शून्य);
शून्य ggml_metal_cv_free(ggml_metal_cv_t cv);

शून्य ggml_metal_cv_समुच्चयः_int16(ggml_metal_cv_t cv, int16_t मानम्, int32_t idx);
शून्य ggml_metal_cv_समुच्चयः_int32(ggml_metal_cv_t cv, int32_t मानम्, int32_t idx);
शून्य ggml_metal_cv_समुच्चयः_bool (ggml_metal_cv_t cv, bool    मानम्, int32_t idx);

//
// MTLComputePipelineState wrapper
//

प्रकारनिर्देश संरचना ggml_metal_pipeline * ggml_metal_pipeline_t;

ggml_metal_pipeline_t ggml_metal_pipeline_init(शून्य);
शून्य ggml_metal_pipeline_free(ggml_metal_pipeline_t pipeline);

// a collection of pipelines
प्रकारनिर्देश संरचना ggml_metal_pipelines * ggml_metal_pipelines_t;

ggml_metal_pipelines_t ggml_metal_pipelines_init(शून्य);
शून्य ggml_metal_pipelines_free(ggml_metal_pipelines_t ppls);

शून्य                  ggml_metal_pipelines_सङ्कलय(ggml_metal_pipelines_t ppls, ध्रुव वर्ण * name, ggml_metal_pipeline_t pipeline);
ggml_metal_pipeline_t ggml_metal_pipelines_get(ggml_metal_pipelines_t ppls, ध्रुव वर्ण * name);

संरचना ggml_metal_pipeline_with_params {
    ggml_metal_pipeline_t pipeline;

    पूर्णाङ्क nsg;

    पूर्णाङ्क nr0;
    पूर्णाङ्क nr1;

    size_t smem;

    bool c4;
    bool cnt;
};

पूर्णाङ्क ggml_metal_pipeline_max_theads_प्रत्येकम्_threadgroup(संरचना ggml_metal_pipeline_with_params pipeline);

//
// MTLCommandBuffer wrapper
//

प्रकारनिर्देश शून्य * ggml_metal_cmd_buf_t;

//
// MTLComputeCommandEncoder wrapper
//

प्रकारनिर्देश संरचना ggml_metal_encoder * ggml_metal_encoder_t;

ggml_metal_encoder_t ggml_metal_encoder_init(ggml_metal_cmd_buf_t cmd_buf_raw, bool concurrent);
शून्य ggml_metal_encoder_free(ggml_metal_encoder_t encoder);

शून्य ggml_metal_encoder_debug_गणः_प्रक्षिप(ggml_metal_encoder_t encoder, ध्रुव वर्ण * name);
शून्य ggml_metal_encoder_debug_गणः_निष्कासय (ggml_metal_encoder_t encoder);

शून्य ggml_metal_encoder_समुच्चयः_pipeline(ggml_metal_encoder_t encoder, संरचना ggml_metal_pipeline_with_params pipeline);

शून्य ggml_metal_encoder_समुच्चयः_अष्टकानि (ggml_metal_encoder_t encoder, शून्य * दत्त, size_t आकार, पूर्णाङ्क idx);
शून्य ggml_metal_encoder_समुच्चयः_बफर(ggml_metal_encoder_t encoder, संरचना ggml_metal_बफर_id बफर, पूर्णाङ्क idx);

शून्य ggml_metal_encoder_समुच्चयः_threadgroup_स्मृतिः_आकार(ggml_metal_encoder_t encoder, size_t आकार, पूर्णाङ्क idx);

शून्य ggml_metal_encoder_dispatch_threadgroups(ggml_metal_encoder_t encoder, पूर्णाङ्क tg0, पूर्णाङ्क tg1, पूर्णाङ्क tg2, पूर्णाङ्क tptg0, पूर्णाङ्क tptg1, पूर्णाङ्क tptg2);

शून्य ggml_metal_encoder_स्मृतिः_barrier(ggml_metal_encoder_t encoder);

शून्य ggml_metal_encoder_अन्तः_encoding(ggml_metal_encoder_t encoder);

//
// MTLLibrary wrapper
//

प्रकारनिर्देश संरचना ggml_metal_library * ggml_metal_library_t;

ggml_metal_library_t ggml_metal_library_init            (ggml_metal_device_t dev);
ggml_metal_library_t ggml_metal_library_init_from_source(ggml_metal_device_t dev, ध्रुव वर्ण * source, bool verbose);

शून्य ggml_metal_library_free(ggml_metal_library_t lib);

ggml_metal_device_t ggml_metal_library_get_device(ggml_metal_library_t lib);

संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline    (ggml_metal_library_t lib, ध्रुव वर्ण * name);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_compile_pipeline(ggml_metal_library_t lib, ध्रुव वर्ण * आधारः, ध्रुव वर्ण * name, ggml_metal_cv_t cv);

संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_आधारः              (ggml_metal_library_t lib, गणना ggml_op op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cpy               (ggml_metal_library_t lib, गणना ggml_type tsrc, गणना ggml_type tdst);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_1d           (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op, गणना ggml_op_pool op_pool);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_2d           (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op, गणना ggml_op_pool op_pool);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_get_पङ्क्तिकाः          (ggml_metal_library_t lib, गणना ggml_type tsrc);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_समुच्चयः_पङ्क्तिकाः          (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_diag              (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_repeat            (ggml_metal_library_t lib, गणना ggml_type tsrc);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_concat            (ggml_metal_library_t lib, गणना ggml_type tsrc);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_unary             (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_silu_पृष्ठभागः         (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_glu               (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_योग               (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_योग_पङ्क्तिकाः          (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cumsum_blk        (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cumsum_सङ्कलय        (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_tri               (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_soft_max          (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_lightning_indexer (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_hc           (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv          (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op, int32_t nc, bool use_silu);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv_batched  (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op, पूर्णाङ्क ssm_conv_bs, int32_t nc, bool use_silu);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_अवलोकनम्          (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op, bool पुच्छ);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_अवलोकनम्_ssd_mma  (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rwkv              (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_gated_delta_net   (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_solve_tri         (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_ext        (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op, पूर्णाङ्क nsg, पूर्णाङ्क nxpsg, पूर्णाङ्क r1ptg);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_mma        (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op, पूर्णाङ्क nsg, पूर्णाङ्क nt, पूर्णाङ्क rt, bool सङ्कलय_2);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_mma_auto   (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op, bool सङ्कलय_2);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm            (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv            (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_map0    (ggml_metal_library_t lib, पूर्णाङ्क ne02, पूर्णाङ्क ne20);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_amax(ggml_metal_library_t lib);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_amax_part(ggml_metal_library_t lib);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id         (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_id         (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax            (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax_part       (ggml_metal_library_t lib);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax_final      (ggml_metal_library_t lib);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argsort           (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argsort_merge     (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_fwht              (ggml_metal_library_t lib, पूर्णाङ्क n, गणना ggml_type tsrc);

संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_शिखरम्_k             (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_शिखरम्_k_radix       (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_शिखरम्_k_merge       (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_topk_moe          (ggml_metal_library_t lib, int32_t n_expert, int32_t शिखरम्_k, bool with_norm);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_moe_reduce        (ggml_metal_library_t lib, int32_t n_expert_प्रयुक्तम्);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin               (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op, int32_t n_fuse);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin_one           (ggml_metal_library_t lib, गणना ggml_op op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_l2_norm           (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_गणः_norm        (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_norm              (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op, int32_t n_fuse);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_norm_scale        (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rope              (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_im2col            (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_1d (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_2d (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_col2im_1d         (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_snake             (ggml_metal_library_t lib, गणना ggml_type type);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_2d           (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_2d_dw        (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op, bool tiled);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_3d           (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_upscale           (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad               (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad_reflect_1d    (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_roll              (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_arange            (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_timestep_embedding(ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_चरणः_adamw    (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_चरणः_sgd      (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_memset            (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_गणनम्_equal       (ggml_metal_library_t lib, ध्रुव संरचना ggml_tensor * op);

संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_pad(
        ggml_metal_library_t lib,
        ध्रुव संरचना ggml_tensor * op,
        bool    has_आवरणम्,
        int32_t ncpsg);

संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_kv_f16(
        ggml_metal_library_t lib,
        ध्रुव संरचना ggml_tensor * op);

संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_blk(
        ggml_metal_library_t lib,
        ध्रुव संरचना ggml_tensor * op,
        int32_t nqptg,
        int32_t ncpsg);

संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_tensor(
        ggml_metal_library_t lib,
        ध्रुव संरचना ggml_tensor * op,
        bool has_आवरणम्,
        bool has_sinks,
        bool has_bias,
        bool has_scap);

संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext(
        ggml_metal_library_t lib,
        ध्रुव संरचना ggml_tensor * op,
        bool    has_आवरणम्,
        bool    has_sinks,
        bool    has_bias,
        bool    has_scap,
        bool    has_kvpad,
        int32_t nsg,
        bool    use_kv_f16,
        int32_t ns10,
        int32_t ns20);

संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec_idx(
        ggml_metal_library_t lib,
        ध्रुव संरचना ggml_tensor * op);

संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec(
        ggml_metal_library_t lib,
        ध्रुव संरचना ggml_tensor * op,
        bool    has_आवरणम्,
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

संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce(
        ggml_metal_library_t lib,
        ध्रुव संरचना ggml_tensor * op,
        int32_t dv,
        int32_t nwg);

// MTLResidencySet wrapper

प्रकारनिर्देश शून्य * ggml_metal_rset_t;

// a collection of residency sets (non-owning)
प्रकारनिर्देश संरचना ggml_metal_rsets * ggml_metal_rsets_t;

ggml_metal_rsets_t ggml_metal_rsets_init(ggml_metal_device_t dev);
शून्य ggml_metal_rsets_free(ggml_metal_rsets_t rsets);

//
// device
//

गणना ggml_metal_device_id {
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

ध्रुव वर्ण * ggml_metal_device_id_token(गणना ggml_metal_device_id id);

संरचना ggml_metal_device_props {
    पूर्णाङ्क device;
    पूर्णाङ्क device_phys;
    पूर्णाङ्क device_virt;
    वर्ण name[128];
    वर्ण desc[128];

    size_t max_बफर_आकार;
    size_t max_working_समुच्चयः_आकार;
    size_t max_theadgroup_स्मृतिः_आकार;

    bool has_simdgroup_reduction;
    bool has_simdgroup_mm;
    bool has_unified_स्मृतिः;
    bool has_bfloat;
    bool has_tensor;
    bool use_residency_sets;
    bool use_shared_buffers;

    गणना ggml_metal_device_id device_id;
    पूर्णाङ्क gpu_family;

    पूर्णाङ्क op_offload_min_batch_आकार;
};

प्रकारनिर्देश संरचना ggml_metal_event * ggml_metal_event_t;

शून्य ggml_metal_event_encode_signal(ggml_metal_event_t ev, ggml_metal_cmd_buf_t cmd_buf);
शून्य ggml_metal_event_encode_wait  (ggml_metal_event_t ev, ggml_metal_cmd_buf_t cmd_buf);

ggml_metal_device_t ggml_metal_device_init(पूर्णाङ्क device, पूर्णाङ्क n_devices);
शून्य ggml_metal_device_free(ggml_metal_device_t dev);

ggml_metal_device_t ggml_metal_device_get(पूर्णाङ्क device, पूर्णाङ्क n_devices);

शून्य * ggml_metal_device_get_obj  (ggml_metal_device_t dev); // id<MTLDevice>
शून्य * ggml_metal_device_get_queue(ggml_metal_device_t dev); // id<MTLCommandQueue>

ggml_metal_library_t ggml_metal_device_get_library(ggml_metal_device_t dev);

शून्य ggml_metal_device_rsets_सङ्कलय(ggml_metal_device_t dev, ggml_metal_rset_t rset);
शून्य ggml_metal_device_rsets_rm (ggml_metal_device_t dev, ggml_metal_rset_t rset);

शून्य ggml_metal_device_rsets_keep_alive(ggml_metal_device_t dev);

ggml_metal_event_t ggml_metal_device_event_init(ggml_metal_device_t dev);
शून्य ggml_metal_device_event_free(ggml_metal_device_t dev, ggml_metal_event_t ev);
शून्य ggml_metal_device_event_synchronize(ggml_metal_device_t dev, ggml_metal_event_t ev);

शून्य ggml_metal_device_get_स्मृतिः(ggml_metal_device_t dev, size_t * मुक्त, size_t * योग);
bool ggml_metal_device_supports_op(ggml_metal_device_t dev, ध्रुव संरचना ggml_tensor * op);

ध्रुव संरचना ggml_metal_device_props * ggml_metal_device_get_props(ggml_metal_device_t dev);

संरचना ggml_metal_fusion_info;

// the device-owned fusion debugging context (NULL unless fusion debugging is enabled)
संरचना ggml_metal_fusion_info * ggml_metal_device_get_fusion_info(ggml_metal_device_t dev);

//
// device buffers
//

प्रकारनिर्देश संरचना ggml_metal_बफर * ggml_metal_बफर_t;

ggml_metal_बफर_t ggml_metal_बफर_init(ggml_metal_device_t dev, size_t आकार, bool shared);
ggml_metal_बफर_t ggml_metal_बफर_map (ggml_metal_device_t dev, शून्य * ptr, size_t आकार, size_t max_tensor_आकार);

शून्य   ggml_metal_बफर_free     (ggml_metal_बफर_t buf);
शून्य * ggml_metal_बफर_get_आधारः (ggml_metal_बफर_t buf);
bool   ggml_metal_बफर_is_shared(ggml_metal_बफर_t buf);

शून्य   ggml_metal_बफर_memset_tensor(ggml_metal_बफर_t buf, संरचना ggml_tensor * tensor, uint8_t मानम्, size_t विस्थापनम्, size_t आकार);
शून्य   ggml_metal_बफर_समुच्चयः_tensor   (ggml_metal_बफर_t buf, संरचना ggml_tensor * tensor, ध्रुव शून्य * दत्त, size_t विस्थापनम्, size_t आकार);
शून्य   ggml_metal_बफर_get_tensor   (ggml_metal_बफर_t buf, ध्रुव संरचना ggml_tensor * tensor, शून्य * दत्त, size_t विस्थापनम्, size_t आकार);
bool   ggml_metal_बफर_cpy_tensor   (ggml_metal_बफर_t buf, ध्रुव संरचना ggml_tensor * src, संरचना ggml_tensor * dst);
शून्य   ggml_metal_बफर_clear        (ggml_metal_बफर_t buf, uint8_t मानम्);

// finds the Metal buffer that contains the tensor data on the GPU device
// the assumption is that there is 1-to-1 mapping between the host and device memory buffers, so we can find the
// Metal buffer based on the host memory pointer
//
संरचना ggml_metal_बफर_id ggml_metal_बफर_get_id(ggml_metal_बफर_t buf, ध्रुव संरचना ggml_tensor * t);

#यदि_परिभाषितम् __cplusplus
}
#यदि_अन्तः
