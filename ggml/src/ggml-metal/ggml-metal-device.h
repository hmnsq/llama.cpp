#आशय once

#समावेश "ggml.h"

#अगर_घोषित __cplusplus
बाह्य "C" {
#पूर्ण_अगर

काष्ठा ggml_metal_बफर_id {
    व्योम * metal; // id<MTLBuffer>
    माप_प्रकार offs;
};

प्रकार काष्ठा ggml_metal_device * ggml_metal_device_t;

//
// MTLFunctionConstantValues wrapper
//

प्रकार काष्ठा ggml_metal_cv * ggml_metal_cv_t;

ggml_metal_cv_t ggml_metal_cv_init(व्योम);
व्योम ggml_metal_cv_free(ggml_metal_cv_t cv);

व्योम ggml_metal_cv_set_int16(ggml_metal_cv_t cv, int16_t मूल्य, int32_t idx);
व्योम ggml_metal_cv_set_int32(ggml_metal_cv_t cv, int32_t मूल्य, int32_t idx);
व्योम ggml_metal_cv_set_bool (ggml_metal_cv_t cv, bool    मूल्य, int32_t idx);

//
// MTLComputePipelineState wrapper
//

प्रकार काष्ठा ggml_metal_pipeline * ggml_metal_pipeline_t;

ggml_metal_pipeline_t ggml_metal_pipeline_init(व्योम);
व्योम ggml_metal_pipeline_free(ggml_metal_pipeline_t pipeline);

// a collection of pipelines
प्रकार काष्ठा ggml_metal_pipelines * ggml_metal_pipelines_t;

ggml_metal_pipelines_t ggml_metal_pipelines_init(व्योम);
व्योम ggml_metal_pipelines_free(ggml_metal_pipelines_t ppls);

व्योम                  ggml_metal_pipelines_add(ggml_metal_pipelines_t ppls, स्थिर अक्षर * name, ggml_metal_pipeline_t pipeline);
ggml_metal_pipeline_t ggml_metal_pipelines_get(ggml_metal_pipelines_t ppls, स्थिर अक्षर * name);

काष्ठा ggml_metal_pipeline_with_params {
    ggml_metal_pipeline_t pipeline;

    पूर्णांक nsg;

    पूर्णांक nr0;
    पूर्णांक nr1;

    माप_प्रकार smem;

    bool c4;
    bool cnt;
};

पूर्णांक ggml_metal_pipeline_max_theads_per_threadgroup(काष्ठा ggml_metal_pipeline_with_params pipeline);

//
// MTLCommandBuffer wrapper
//

प्रकार व्योम * ggml_metal_cmd_buf_t;

//
// MTLComputeCommandEncoder wrapper
//

प्रकार काष्ठा ggml_metal_encoder * ggml_metal_encoder_t;

ggml_metal_encoder_t ggml_metal_encoder_init(ggml_metal_cmd_buf_t cmd_buf_raw, bool concurrent);
व्योम ggml_metal_encoder_free(ggml_metal_encoder_t encoder);

व्योम ggml_metal_encoder_debug_group_धकेलो(ggml_metal_encoder_t encoder, स्थिर अक्षर * name);
व्योम ggml_metal_encoder_debug_group_निकालो (ggml_metal_encoder_t encoder);

व्योम ggml_metal_encoder_set_pipeline(ggml_metal_encoder_t encoder, काष्ठा ggml_metal_pipeline_with_params pipeline);

व्योम ggml_metal_encoder_set_bytes (ggml_metal_encoder_t encoder, व्योम * तथ्य, माप_प्रकार आकार, पूर्णांक idx);
व्योम ggml_metal_encoder_set_बफर(ggml_metal_encoder_t encoder, काष्ठा ggml_metal_बफर_id बफर, पूर्णांक idx);

व्योम ggml_metal_encoder_set_threadgroup_memory_आकार(ggml_metal_encoder_t encoder, माप_प्रकार आकार, पूर्णांक idx);

व्योम ggml_metal_encoder_dispatch_threadgroups(ggml_metal_encoder_t encoder, पूर्णांक tg0, पूर्णांक tg1, पूर्णांक tg2, पूर्णांक tptg0, पूर्णांक tptg1, पूर्णांक tptg2);

व्योम ggml_metal_encoder_memory_barrier(ggml_metal_encoder_t encoder);

व्योम ggml_metal_encoder_end_encoding(ggml_metal_encoder_t encoder);

//
// MTLLibrary wrapper
//

प्रकार काष्ठा ggml_metal_library * ggml_metal_library_t;

ggml_metal_library_t ggml_metal_library_init            (ggml_metal_device_t dev);
ggml_metal_library_t ggml_metal_library_init_from_source(ggml_metal_device_t dev, स्थिर अक्षर * source, bool verbose);

व्योम ggml_metal_library_free(ggml_metal_library_t lib);

ggml_metal_device_t ggml_metal_library_get_device(ggml_metal_library_t lib);

काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline    (ggml_metal_library_t lib, स्थिर अक्षर * name);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_compile_pipeline(ggml_metal_library_t lib, स्थिर अक्षर * base, स्थिर अक्षर * name, ggml_metal_cv_t cv);

काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_base              (ggml_metal_library_t lib, क्रमागत ggml_op op);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cpy               (ggml_metal_library_t lib, क्रमागत ggml_type tsrc, क्रमागत ggml_type tdst);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_1d           (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op, क्रमागत ggml_op_pool op_pool);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_2d           (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op, क्रमागत ggml_op_pool op_pool);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_get_rows          (ggml_metal_library_t lib, क्रमागत ggml_type tsrc);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_set_rows          (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_diag              (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_repeat            (ggml_metal_library_t lib, क्रमागत ggml_type tsrc);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_concat            (ggml_metal_library_t lib, क्रमागत ggml_type tsrc);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_unary             (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_silu_back         (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_glu               (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_योग               (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_योग_rows          (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cumsum_blk        (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cumsum_add        (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_tri               (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_soft_max          (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_lightning_indexer (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_hc           (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv          (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op, int32_t nc, bool use_silu);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv_batched  (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op, पूर्णांक ssm_conv_bs, int32_t nc, bool use_silu);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_scan          (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op, bool पुच्छ);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_scan_ssd_mma  (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rwkv              (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_gated_delta_net   (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_solve_tri         (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_ext        (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op, पूर्णांक nsg, पूर्णांक nxpsg, पूर्णांक r1ptg);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_mma        (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op, पूर्णांक nsg, पूर्णांक nt, पूर्णांक rt, bool add);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_mma_auto   (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op, bool add);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm            (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv            (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_map0    (ggml_metal_library_t lib, पूर्णांक ne02, पूर्णांक ne20);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_amax(ggml_metal_library_t lib);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_amax_part(ggml_metal_library_t lib);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id         (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_id         (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax            (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax_part       (ggml_metal_library_t lib);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax_final      (ggml_metal_library_t lib);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argsort           (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argsort_merge     (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_fwht              (ggml_metal_library_t lib, पूर्णांक n, क्रमागत ggml_type tsrc);

काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_शीर्ष_k             (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_शीर्ष_k_radix       (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_शीर्ष_k_merge       (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_topk_moe          (ggml_metal_library_t lib, int32_t n_expert, int32_t शीर्ष_k, bool with_norm);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_moe_reduce        (ggml_metal_library_t lib, int32_t n_expert_used);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin               (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op, int32_t n_fuse);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin_one           (ggml_metal_library_t lib, क्रमागत ggml_op op);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_l2_norm           (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_group_norm        (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_norm              (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op, int32_t n_fuse);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_norm_scale        (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rope              (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_im2col            (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_1d (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_2d (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_col2im_1d         (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_snake             (ggml_metal_library_t lib, क्रमागत ggml_type type);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_2d           (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_2d_dw        (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op, bool tiled);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_3d           (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_upscale           (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad               (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad_reflect_1d    (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_roll              (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_arange            (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_timestep_embedding(ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_step_adamw    (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_step_sgd      (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_memset            (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op);
काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_गिनती_equal       (ggml_metal_library_t lib, स्थिर काष्ठा ggml_tensor * op);

काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_pad(
        ggml_metal_library_t lib,
        स्थिर काष्ठा ggml_tensor * op,
        bool    has_mask,
        int32_t ncpsg);

काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_kv_f16(
        ggml_metal_library_t lib,
        स्थिर काष्ठा ggml_tensor * op);

काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_blk(
        ggml_metal_library_t lib,
        स्थिर काष्ठा ggml_tensor * op,
        int32_t nqptg,
        int32_t ncpsg);

काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_tensor(
        ggml_metal_library_t lib,
        स्थिर काष्ठा ggml_tensor * op,
        bool has_mask,
        bool has_sinks,
        bool has_bias,
        bool has_scap);

काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext(
        ggml_metal_library_t lib,
        स्थिर काष्ठा ggml_tensor * op,
        bool    has_mask,
        bool    has_sinks,
        bool    has_bias,
        bool    has_scap,
        bool    has_kvpad,
        int32_t nsg,
        bool    use_kv_f16,
        int32_t ns10,
        int32_t ns20);

काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec_idx(
        ggml_metal_library_t lib,
        स्थिर काष्ठा ggml_tensor * op);

काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec(
        ggml_metal_library_t lib,
        स्थिर काष्ठा ggml_tensor * op,
        bool    has_mask,
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

काष्ठा ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce(
        ggml_metal_library_t lib,
        स्थिर काष्ठा ggml_tensor * op,
        int32_t dv,
        int32_t nwg);

// MTLResidencySet wrapper

प्रकार व्योम * ggml_metal_rset_t;

// a collection of residency sets (non-owning)
प्रकार काष्ठा ggml_metal_rsets * ggml_metal_rsets_t;

ggml_metal_rsets_t ggml_metal_rsets_init(ggml_metal_device_t dev);
व्योम ggml_metal_rsets_free(ggml_metal_rsets_t rsets);

//
// device
//

क्रमागत ggml_metal_device_id {
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

स्थिर अक्षर * ggml_metal_device_id_token(क्रमागत ggml_metal_device_id id);

काष्ठा ggml_metal_device_props {
    पूर्णांक device;
    पूर्णांक device_phys;
    पूर्णांक device_virt;
    अक्षर name[128];
    अक्षर desc[128];

    माप_प्रकार max_बफर_आकार;
    माप_प्रकार max_working_set_आकार;
    माप_प्रकार max_theadgroup_memory_आकार;

    bool has_simdgroup_reduction;
    bool has_simdgroup_mm;
    bool has_unified_memory;
    bool has_bfloat;
    bool has_tensor;
    bool use_residency_sets;
    bool use_shared_buffers;

    क्रमागत ggml_metal_device_id device_id;
    पूर्णांक gpu_family;

    पूर्णांक op_offload_min_batch_आकार;
};

प्रकार काष्ठा ggml_metal_event * ggml_metal_event_t;

व्योम ggml_metal_event_encode_signal(ggml_metal_event_t ev, ggml_metal_cmd_buf_t cmd_buf);
व्योम ggml_metal_event_encode_wait  (ggml_metal_event_t ev, ggml_metal_cmd_buf_t cmd_buf);

ggml_metal_device_t ggml_metal_device_init(पूर्णांक device, पूर्णांक n_devices);
व्योम ggml_metal_device_free(ggml_metal_device_t dev);

ggml_metal_device_t ggml_metal_device_get(पूर्णांक device, पूर्णांक n_devices);

व्योम * ggml_metal_device_get_obj  (ggml_metal_device_t dev); // id<MTLDevice>
व्योम * ggml_metal_device_get_queue(ggml_metal_device_t dev); // id<MTLCommandQueue>

ggml_metal_library_t ggml_metal_device_get_library(ggml_metal_device_t dev);

व्योम ggml_metal_device_rsets_add(ggml_metal_device_t dev, ggml_metal_rset_t rset);
व्योम ggml_metal_device_rsets_rm (ggml_metal_device_t dev, ggml_metal_rset_t rset);

व्योम ggml_metal_device_rsets_keep_alive(ggml_metal_device_t dev);

ggml_metal_event_t ggml_metal_device_event_init(ggml_metal_device_t dev);
व्योम ggml_metal_device_event_free(ggml_metal_device_t dev, ggml_metal_event_t ev);
व्योम ggml_metal_device_event_synchronize(ggml_metal_device_t dev, ggml_metal_event_t ev);

व्योम ggml_metal_device_get_memory(ggml_metal_device_t dev, माप_प्रकार * मुक्त, माप_प्रकार * योग);
bool ggml_metal_device_supports_op(ggml_metal_device_t dev, स्थिर काष्ठा ggml_tensor * op);

स्थिर काष्ठा ggml_metal_device_props * ggml_metal_device_get_props(ggml_metal_device_t dev);

काष्ठा ggml_metal_fusion_info;

// the device-owned fusion debugging context (NULL unless fusion debugging is enabled)
काष्ठा ggml_metal_fusion_info * ggml_metal_device_get_fusion_info(ggml_metal_device_t dev);

//
// device buffers
//

प्रकार काष्ठा ggml_metal_बफर * ggml_metal_बफर_t;

ggml_metal_बफर_t ggml_metal_बफर_init(ggml_metal_device_t dev, माप_प्रकार आकार, bool shared);
ggml_metal_बफर_t ggml_metal_बफर_map (ggml_metal_device_t dev, व्योम * ptr, माप_प्रकार आकार, माप_प्रकार max_tensor_आकार);

व्योम   ggml_metal_बफर_free     (ggml_metal_बफर_t buf);
व्योम * ggml_metal_बफर_get_base (ggml_metal_बफर_t buf);
bool   ggml_metal_बफर_is_shared(ggml_metal_बफर_t buf);

व्योम   ggml_metal_बफर_memset_tensor(ggml_metal_बफर_t buf, काष्ठा ggml_tensor * tensor, uint8_t मूल्य, माप_प्रकार offset, माप_प्रकार आकार);
व्योम   ggml_metal_बफर_set_tensor   (ggml_metal_बफर_t buf, काष्ठा ggml_tensor * tensor, स्थिर व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार आकार);
व्योम   ggml_metal_बफर_get_tensor   (ggml_metal_बफर_t buf, स्थिर काष्ठा ggml_tensor * tensor, व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार आकार);
bool   ggml_metal_बफर_cpy_tensor   (ggml_metal_बफर_t buf, स्थिर काष्ठा ggml_tensor * src, काष्ठा ggml_tensor * dst);
व्योम   ggml_metal_बफर_clear        (ggml_metal_बफर_t buf, uint8_t मूल्य);

// finds the Metal buffer that contains the tensor data on the GPU device
// the assumption is that there is 1-to-1 mapping between the host and device memory buffers, so we can find the
// Metal buffer based on the host memory pointer
//
काष्ठा ggml_metal_बफर_id ggml_metal_बफर_get_id(ggml_metal_बफर_t buf, स्थिर काष्ठा ggml_tensor * t);

#अगर_घोषित __cplusplus
}
#पूर्ण_अगर
