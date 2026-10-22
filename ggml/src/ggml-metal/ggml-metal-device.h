#pragma once

#अन्तर्भाव "ggml.h"

#ifdef __cplusplus
बाह्य "C" {
#endif

संरचना ggml_metal_buffer_id {
    शून्य * metal; // id<MTLBuffer>
    माप_प्रकार offs;
};

प्रकारव्याख्या संरचना ggml_metal_device * ggml_metal_device_t;

//
// MTLFunctionConstantValues wrapper
//

प्रकारव्याख्या संरचना ggml_metal_cv * ggml_metal_cv_t;

ggml_metal_cv_t ggml_metal_cv_init(शून्य);
शून्य ggml_metal_cv_free(ggml_metal_cv_t cv);

शून्य ggml_metal_cv_set_int16(ggml_metal_cv_t cv, int16_t मूल्य, int32_t idx);
शून्य ggml_metal_cv_set_int32(ggml_metal_cv_t cv, int32_t मूल्य, int32_t idx);
शून्य ggml_metal_cv_set_bool (ggml_metal_cv_t cv, bool    मूल्य, int32_t idx);

//
// MTLComputePipelineState wrapper
//

प्रकारव्याख्या संरचना ggml_metal_pipeline * ggml_metal_pipeline_t;

ggml_metal_pipeline_t ggml_metal_pipeline_init(शून्य);
शून्य ggml_metal_pipeline_free(ggml_metal_pipeline_t pipeline);

// a collection of pipelines
प्रकारव्याख्या संरचना ggml_metal_pipelines * ggml_metal_pipelines_t;

ggml_metal_pipelines_t ggml_metal_pipelines_init(शून्य);
शून्य ggml_metal_pipelines_free(ggml_metal_pipelines_t ppls);

शून्य                  ggml_metal_pipelines_add(ggml_metal_pipelines_t ppls, स्थिरांक अक्षर * name, ggml_metal_pipeline_t pipeline);
ggml_metal_pipeline_t ggml_metal_pipelines_get(ggml_metal_pipelines_t ppls, स्थिरांक अक्षर * name);

संरचना ggml_metal_pipeline_with_params {
    ggml_metal_pipeline_t pipeline;

    पूर्णांक nsg;

    पूर्णांक nr0;
    पूर्णांक nr1;

    माप_प्रकार smem;

    bool c4;
    bool cnt;
};

पूर्णांक ggml_metal_pipeline_max_theads_per_threadgroup(संरचना ggml_metal_pipeline_with_params pipeline);

//
// MTLCommandBuffer wrapper
//

प्रकारव्याख्या शून्य * ggml_metal_cmd_buf_t;

//
// MTLComputeCommandEncoder wrapper
//

प्रकारव्याख्या संरचना ggml_metal_encoder * ggml_metal_encoder_t;

ggml_metal_encoder_t ggml_metal_encoder_init(ggml_metal_cmd_buf_t cmd_buf_raw, bool concurrent);
शून्य ggml_metal_encoder_free(ggml_metal_encoder_t encoder);

शून्य ggml_metal_encoder_debug_group_ढकला(ggml_metal_encoder_t encoder, स्थिरांक अक्षर * name);
शून्य ggml_metal_encoder_debug_group_काढा (ggml_metal_encoder_t encoder);

शून्य ggml_metal_encoder_set_pipeline(ggml_metal_encoder_t encoder, संरचना ggml_metal_pipeline_with_params pipeline);

शून्य ggml_metal_encoder_set_bytes (ggml_metal_encoder_t encoder, शून्य * data, माप_प्रकार size, पूर्णांक idx);
शून्य ggml_metal_encoder_set_buffer(ggml_metal_encoder_t encoder, संरचना ggml_metal_buffer_id buffer, पूर्णांक idx);

शून्य ggml_metal_encoder_set_threadgroup_memory_size(ggml_metal_encoder_t encoder, माप_प्रकार size, पूर्णांक idx);

शून्य ggml_metal_encoder_dispatch_threadgroups(ggml_metal_encoder_t encoder, पूर्णांक tg0, पूर्णांक tg1, पूर्णांक tg2, पूर्णांक tptg0, पूर्णांक tptg1, पूर्णांक tptg2);

शून्य ggml_metal_encoder_memory_barrier(ggml_metal_encoder_t encoder);

शून्य ggml_metal_encoder_end_encoding(ggml_metal_encoder_t encoder);

//
// MTLLibrary wrapper
//

प्रकारव्याख्या संरचना ggml_metal_library * ggml_metal_library_t;

ggml_metal_library_t ggml_metal_library_init            (ggml_metal_device_t dev);
ggml_metal_library_t ggml_metal_library_init_from_source(ggml_metal_device_t dev, स्थिरांक अक्षर * source, bool verbose);

शून्य ggml_metal_library_free(ggml_metal_library_t lib);

ggml_metal_device_t ggml_metal_library_get_device(ggml_metal_library_t lib);

संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline    (ggml_metal_library_t lib, स्थिरांक अक्षर * name);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_compile_pipeline(ggml_metal_library_t lib, स्थिरांक अक्षर * base, स्थिरांक अक्षर * name, ggml_metal_cv_t cv);

संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_base              (ggml_metal_library_t lib, गणना ggml_op op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cpy               (ggml_metal_library_t lib, गणना ggml_type tsrc, गणना ggml_type tdst);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_1d           (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op, गणना ggml_op_pool op_pool);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_2d           (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op, गणना ggml_op_pool op_pool);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_get_rows          (ggml_metal_library_t lib, गणना ggml_type tsrc);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_set_rows          (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_diag              (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_repeat            (ggml_metal_library_t lib, गणना ggml_type tsrc);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_concat            (ggml_metal_library_t lib, गणना ggml_type tsrc);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_unary             (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_silu_back         (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_glu               (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_sum               (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_sum_rows          (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cumsum_blk        (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cumsum_add        (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_tri               (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_soft_max          (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_lightning_indexer (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_hc           (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv          (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op, int32_t nc, bool use_silu);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv_batched  (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op, पूर्णांक ssm_conv_bs, int32_t nc, bool use_silu);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_scan          (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op, bool tail);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_scan_ssd_mma  (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rwkv              (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_gated_delta_net   (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_solve_tri         (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_ext        (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op, पूर्णांक nsg, पूर्णांक nxpsg, पूर्णांक r1ptg);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_mma        (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op, पूर्णांक nsg, पूर्णांक nt, पूर्णांक rt, bool add);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_mma_auto   (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op, bool add);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm            (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv            (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_map0    (ggml_metal_library_t lib, पूर्णांक ne02, पूर्णांक ne20);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_amax(ggml_metal_library_t lib);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_amax_part(ggml_metal_library_t lib);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id         (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_id         (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax            (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax_part       (ggml_metal_library_t lib);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax_final      (ggml_metal_library_t lib);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argsort           (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argsort_merge     (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_fwht              (ggml_metal_library_t lib, पूर्णांक n, गणना ggml_type tsrc);

संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_शीर्ष_k             (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_शीर्ष_k_radix       (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_शीर्ष_k_merge       (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_topk_moe          (ggml_metal_library_t lib, int32_t n_expert, int32_t शीर्ष_k, bool with_norm);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_moe_reduce        (ggml_metal_library_t lib, int32_t n_expert_used);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin               (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op, int32_t n_fuse);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin_one           (ggml_metal_library_t lib, गणना ggml_op op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_l2_norm           (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_group_norm        (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_norm              (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op, int32_t n_fuse);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_norm_scale        (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rope              (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_im2col            (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_1d (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_2d (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_col2im_1d         (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_snake             (ggml_metal_library_t lib, गणना ggml_type type);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_2d           (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_2d_dw        (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op, bool tiled);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_3d           (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_upscale           (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad               (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad_reflect_1d    (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_roll              (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_arange            (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_timestep_embedding(ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_step_adamw    (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_step_sgd      (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_memset            (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op);
संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_मोजणी_equal       (ggml_metal_library_t lib, स्थिरांक संरचना ggml_tensor * op);

संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_pad(
        ggml_metal_library_t lib,
        स्थिरांक संरचना ggml_tensor * op,
        bool    has_mask,
        int32_t ncpsg);

संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_kv_f16(
        ggml_metal_library_t lib,
        स्थिरांक संरचना ggml_tensor * op);

संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_blk(
        ggml_metal_library_t lib,
        स्थिरांक संरचना ggml_tensor * op,
        int32_t nqptg,
        int32_t ncpsg);

संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_tensor(
        ggml_metal_library_t lib,
        स्थिरांक संरचना ggml_tensor * op,
        bool has_mask,
        bool has_sinks,
        bool has_bias,
        bool has_scap);

संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext(
        ggml_metal_library_t lib,
        स्थिरांक संरचना ggml_tensor * op,
        bool    has_mask,
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
        स्थिरांक संरचना ggml_tensor * op);

संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec(
        ggml_metal_library_t lib,
        स्थिरांक संरचना ggml_tensor * op,
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

संरचना ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce(
        ggml_metal_library_t lib,
        स्थिरांक संरचना ggml_tensor * op,
        int32_t dv,
        int32_t nwg);

// MTLResidencySet wrapper

प्रकारव्याख्या शून्य * ggml_metal_rset_t;

// a collection of residency sets (non-owning)
प्रकारव्याख्या संरचना ggml_metal_rsets * ggml_metal_rsets_t;

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

स्थिरांक अक्षर * ggml_metal_device_id_token(गणना ggml_metal_device_id id);

संरचना ggml_metal_device_props {
    पूर्णांक device;
    पूर्णांक device_phys;
    पूर्णांक device_virt;
    अक्षर name[128];
    अक्षर desc[128];

    माप_प्रकार max_buffer_size;
    माप_प्रकार max_working_set_size;
    माप_प्रकार max_theadgroup_memory_size;

    bool has_simdgroup_reduction;
    bool has_simdgroup_mm;
    bool has_unified_memory;
    bool has_bfloat;
    bool has_tensor;
    bool use_residency_sets;
    bool use_shared_buffers;

    गणना ggml_metal_device_id device_id;
    पूर्णांक gpu_family;

    पूर्णांक op_offload_min_batch_size;
};

प्रकारव्याख्या संरचना ggml_metal_event * ggml_metal_event_t;

शून्य ggml_metal_event_encode_signal(ggml_metal_event_t ev, ggml_metal_cmd_buf_t cmd_buf);
शून्य ggml_metal_event_encode_wait  (ggml_metal_event_t ev, ggml_metal_cmd_buf_t cmd_buf);

ggml_metal_device_t ggml_metal_device_init(पूर्णांक device, पूर्णांक n_devices);
शून्य ggml_metal_device_free(ggml_metal_device_t dev);

ggml_metal_device_t ggml_metal_device_get(पूर्णांक device, पूर्णांक n_devices);

शून्य * ggml_metal_device_get_obj  (ggml_metal_device_t dev); // id<MTLDevice>
शून्य * ggml_metal_device_get_queue(ggml_metal_device_t dev); // id<MTLCommandQueue>

ggml_metal_library_t ggml_metal_device_get_library(ggml_metal_device_t dev);

शून्य ggml_metal_device_rsets_add(ggml_metal_device_t dev, ggml_metal_rset_t rset);
शून्य ggml_metal_device_rsets_rm (ggml_metal_device_t dev, ggml_metal_rset_t rset);

शून्य ggml_metal_device_rsets_keep_alive(ggml_metal_device_t dev);

ggml_metal_event_t ggml_metal_device_event_init(ggml_metal_device_t dev);
शून्य ggml_metal_device_event_free(ggml_metal_device_t dev, ggml_metal_event_t ev);
शून्य ggml_metal_device_event_synchronize(ggml_metal_device_t dev, ggml_metal_event_t ev);

शून्य ggml_metal_device_get_memory(ggml_metal_device_t dev, माप_प्रकार * मुक्त, माप_प्रकार * total);
bool ggml_metal_device_supports_op(ggml_metal_device_t dev, स्थिरांक संरचना ggml_tensor * op);

स्थिरांक संरचना ggml_metal_device_props * ggml_metal_device_get_props(ggml_metal_device_t dev);

संरचना ggml_metal_fusion_info;

// the device-owned fusion debugging context (NULL unless fusion debugging is enabled)
संरचना ggml_metal_fusion_info * ggml_metal_device_get_fusion_info(ggml_metal_device_t dev);

//
// device buffers
//

प्रकारव्याख्या संरचना ggml_metal_buffer * ggml_metal_buffer_t;

ggml_metal_buffer_t ggml_metal_buffer_init(ggml_metal_device_t dev, माप_प्रकार size, bool shared);
ggml_metal_buffer_t ggml_metal_buffer_map (ggml_metal_device_t dev, शून्य * ptr, माप_प्रकार size, माप_प्रकार max_tensor_size);

शून्य   ggml_metal_buffer_free     (ggml_metal_buffer_t buf);
शून्य * ggml_metal_buffer_get_base (ggml_metal_buffer_t buf);
bool   ggml_metal_buffer_is_shared(ggml_metal_buffer_t buf);

शून्य   ggml_metal_buffer_memset_tensor(ggml_metal_buffer_t buf, संरचना ggml_tensor * tensor, uint8_t मूल्य, माप_प्रकार offset, माप_प्रकार size);
शून्य   ggml_metal_buffer_set_tensor   (ggml_metal_buffer_t buf, संरचना ggml_tensor * tensor, स्थिरांक शून्य * data, माप_प्रकार offset, माप_प्रकार size);
शून्य   ggml_metal_buffer_get_tensor   (ggml_metal_buffer_t buf, स्थिरांक संरचना ggml_tensor * tensor, शून्य * data, माप_प्रकार offset, माप_प्रकार size);
bool   ggml_metal_buffer_cpy_tensor   (ggml_metal_buffer_t buf, स्थिरांक संरचना ggml_tensor * src, संरचना ggml_tensor * dst);
शून्य   ggml_metal_buffer_clear        (ggml_metal_buffer_t buf, uint8_t मूल्य);

// finds the Metal buffer that contains the tensor data on the GPU device
// the assumption is that there is 1-to-1 mapping between the host and device memory buffers, so we can find the
// Metal buffer based on the host memory pointer
//
संरचना ggml_metal_buffer_id ggml_metal_buffer_get_id(ggml_metal_buffer_t buf, स्थिरांक संरचना ggml_tensor * t);

#ifdef __cplusplus
}
#endif
