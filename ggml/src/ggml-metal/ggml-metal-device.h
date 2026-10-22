#pragma once

#شامل "ggml.h"

#ifdef __cplusplus
بیرونی "C" {
#endif

ساختار ggml_metal_buffer_id {
    تهی * metal; // id<MTLBuffer>
    size_t offs;
};

تعریفنوع ساختار ggml_metal_device * ggml_metal_device_t;

//
// MTLFunctionConstantValues wrapper
//

تعریفنوع ساختار ggml_metal_cv * ggml_metal_cv_t;

ggml_metal_cv_t ggml_metal_cv_init(تهی);
تهی ggml_metal_cv_free(ggml_metal_cv_t cv);

تهی ggml_metal_cv_set_int16(ggml_metal_cv_t cv, int16_t مقدار, int32_t idx);
تهی ggml_metal_cv_set_int32(ggml_metal_cv_t cv, int32_t مقدار, int32_t idx);
تهی ggml_metal_cv_set_bool (ggml_metal_cv_t cv, bool    مقدار, int32_t idx);

//
// MTLComputePipelineState wrapper
//

تعریفنوع ساختار ggml_metal_pipeline * ggml_metal_pipeline_t;

ggml_metal_pipeline_t ggml_metal_pipeline_init(تهی);
تهی ggml_metal_pipeline_free(ggml_metal_pipeline_t pipeline);

// a collection of pipelines
تعریفنوع ساختار ggml_metal_pipelines * ggml_metal_pipelines_t;

ggml_metal_pipelines_t ggml_metal_pipelines_init(تهی);
تهی ggml_metal_pipelines_free(ggml_metal_pipelines_t ppls);

تهی                  ggml_metal_pipelines_add(ggml_metal_pipelines_t ppls, ثابت نویسه * name, ggml_metal_pipeline_t pipeline);
ggml_metal_pipeline_t ggml_metal_pipelines_get(ggml_metal_pipelines_t ppls, ثابت نویسه * name);

ساختار ggml_metal_pipeline_with_params {
    ggml_metal_pipeline_t pipeline;

    صحیح nsg;

    صحیح nr0;
    صحیح nr1;

    size_t smem;

    bool c4;
    bool cnt;
};

صحیح ggml_metal_pipeline_max_theads_per_threadgroup(ساختار ggml_metal_pipeline_with_params pipeline);

//
// MTLCommandBuffer wrapper
//

تعریفنوع تهی * ggml_metal_cmd_buf_t;

//
// MTLComputeCommandEncoder wrapper
//

تعریفنوع ساختار ggml_metal_encoder * ggml_metal_encoder_t;

ggml_metal_encoder_t ggml_metal_encoder_init(ggml_metal_cmd_buf_t cmd_buf_raw, bool concurrent);
تهی ggml_metal_encoder_free(ggml_metal_encoder_t encoder);

تهی ggml_metal_encoder_debug_group_افزودن(ggml_metal_encoder_t encoder, ثابت نویسه * name);
تهی ggml_metal_encoder_debug_group_برداشت (ggml_metal_encoder_t encoder);

تهی ggml_metal_encoder_set_pipeline(ggml_metal_encoder_t encoder, ساختار ggml_metal_pipeline_with_params pipeline);

تهی ggml_metal_encoder_set_bytes (ggml_metal_encoder_t encoder, تهی * data, size_t size, صحیح idx);
تهی ggml_metal_encoder_set_buffer(ggml_metal_encoder_t encoder, ساختار ggml_metal_buffer_id buffer, صحیح idx);

تهی ggml_metal_encoder_set_threadgroup_memory_size(ggml_metal_encoder_t encoder, size_t size, صحیح idx);

تهی ggml_metal_encoder_dispatch_threadgroups(ggml_metal_encoder_t encoder, صحیح tg0, صحیح tg1, صحیح tg2, صحیح tptg0, صحیح tptg1, صحیح tptg2);

تهی ggml_metal_encoder_memory_barrier(ggml_metal_encoder_t encoder);

تهی ggml_metal_encoder_end_encoding(ggml_metal_encoder_t encoder);

//
// MTLLibrary wrapper
//

تعریفنوع ساختار ggml_metal_library * ggml_metal_library_t;

ggml_metal_library_t ggml_metal_library_init            (ggml_metal_device_t dev);
ggml_metal_library_t ggml_metal_library_init_from_source(ggml_metal_device_t dev, ثابت نویسه * source, bool verbose);

تهی ggml_metal_library_free(ggml_metal_library_t lib);

ggml_metal_device_t ggml_metal_library_get_device(ggml_metal_library_t lib);

ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline    (ggml_metal_library_t lib, ثابت نویسه * name);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_compile_pipeline(ggml_metal_library_t lib, ثابت نویسه * base, ثابت نویسه * name, ggml_metal_cv_t cv);

ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_base              (ggml_metal_library_t lib, شمارش ggml_op op);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cpy               (ggml_metal_library_t lib, شمارش ggml_type tsrc, شمارش ggml_type tdst);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_1d           (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op, شمارش ggml_op_pool op_pool);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_2d           (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op, شمارش ggml_op_pool op_pool);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_get_rows          (ggml_metal_library_t lib, شمارش ggml_type tsrc);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_set_rows          (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_diag              (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_repeat            (ggml_metal_library_t lib, شمارش ggml_type tsrc);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_concat            (ggml_metal_library_t lib, شمارش ggml_type tsrc);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_unary             (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_silu_back         (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_glu               (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_sum               (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_sum_rows          (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cumsum_blk        (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cumsum_add        (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_tri               (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_soft_max          (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_lightning_indexer (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_hc           (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv          (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op, int32_t nc, bool use_silu);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv_batched  (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op, صحیح ssm_conv_bs, int32_t nc, bool use_silu);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_scan          (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op, bool tail);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_scan_ssd_mma  (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rwkv              (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_gated_delta_net   (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_solve_tri         (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_ext        (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op, صحیح nsg, صحیح nxpsg, صحیح r1ptg);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_mma        (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op, صحیح nsg, صحیح nt, صحیح rt, bool add);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_mma_auto   (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op, bool add);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm            (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv            (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_map0    (ggml_metal_library_t lib, صحیح ne02, صحیح ne20);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_amax(ggml_metal_library_t lib);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_amax_part(ggml_metal_library_t lib);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id         (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_id         (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax            (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax_part       (ggml_metal_library_t lib);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax_final      (ggml_metal_library_t lib);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argsort           (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argsort_merge     (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_fwht              (ggml_metal_library_t lib, صحیح n, شمارش ggml_type tsrc);

ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_بالاترین_k             (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_بالاترین_k_radix       (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_بالاترین_k_merge       (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_topk_moe          (ggml_metal_library_t lib, int32_t n_expert, int32_t بالاترین_k, bool with_norm);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_moe_reduce        (ggml_metal_library_t lib, int32_t n_expert_used);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin               (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op, int32_t n_fuse);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin_one           (ggml_metal_library_t lib, شمارش ggml_op op);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_l2_norm           (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_group_norm        (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_norm              (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op, int32_t n_fuse);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_norm_scale        (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rope              (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_im2col            (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_1d (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_2d (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_col2im_1d         (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_snake             (ggml_metal_library_t lib, شمارش ggml_type type);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_2d           (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_2d_dw        (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op, bool tiled);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_3d           (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_upscale           (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad               (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad_reflect_1d    (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_roll              (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_arange            (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_timestep_embedding(ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_step_adamw    (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_step_sgd      (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_memset            (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op);
ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_شمار_equal       (ggml_metal_library_t lib, ثابت ساختار ggml_tensor * op);

ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_pad(
        ggml_metal_library_t lib,
        ثابت ساختار ggml_tensor * op,
        bool    has_mask,
        int32_t ncpsg);

ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_kv_f16(
        ggml_metal_library_t lib,
        ثابت ساختار ggml_tensor * op);

ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_blk(
        ggml_metal_library_t lib,
        ثابت ساختار ggml_tensor * op,
        int32_t nqptg,
        int32_t ncpsg);

ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_tensor(
        ggml_metal_library_t lib,
        ثابت ساختار ggml_tensor * op,
        bool has_mask,
        bool has_sinks,
        bool has_bias,
        bool has_scap);

ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext(
        ggml_metal_library_t lib,
        ثابت ساختار ggml_tensor * op,
        bool    has_mask,
        bool    has_sinks,
        bool    has_bias,
        bool    has_scap,
        bool    has_kvpad,
        int32_t nsg,
        bool    use_kv_f16,
        int32_t ns10,
        int32_t ns20);

ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec_idx(
        ggml_metal_library_t lib,
        ثابت ساختار ggml_tensor * op);

ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec(
        ggml_metal_library_t lib,
        ثابت ساختار ggml_tensor * op,
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

ساختار ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce(
        ggml_metal_library_t lib,
        ثابت ساختار ggml_tensor * op,
        int32_t dv,
        int32_t nwg);

// MTLResidencySet wrapper

تعریفنوع تهی * ggml_metal_rset_t;

// a collection of residency sets (non-owning)
تعریفنوع ساختار ggml_metal_rsets * ggml_metal_rsets_t;

ggml_metal_rsets_t ggml_metal_rsets_init(ggml_metal_device_t dev);
تهی ggml_metal_rsets_free(ggml_metal_rsets_t rsets);

//
// device
//

شمارش ggml_metal_device_id {
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

ثابت نویسه * ggml_metal_device_id_token(شمارش ggml_metal_device_id id);

ساختار ggml_metal_device_props {
    صحیح device;
    صحیح device_phys;
    صحیح device_virt;
    نویسه name[128];
    نویسه desc[128];

    size_t max_buffer_size;
    size_t max_working_set_size;
    size_t max_theadgroup_memory_size;

    bool has_simdgroup_reduction;
    bool has_simdgroup_mm;
    bool has_unified_memory;
    bool has_bfloat;
    bool has_tensor;
    bool use_residency_sets;
    bool use_shared_buffers;

    شمارش ggml_metal_device_id device_id;
    صحیح gpu_family;

    صحیح op_offload_min_batch_size;
};

تعریفنوع ساختار ggml_metal_event * ggml_metal_event_t;

تهی ggml_metal_event_encode_signal(ggml_metal_event_t ev, ggml_metal_cmd_buf_t cmd_buf);
تهی ggml_metal_event_encode_wait  (ggml_metal_event_t ev, ggml_metal_cmd_buf_t cmd_buf);

ggml_metal_device_t ggml_metal_device_init(صحیح device, صحیح n_devices);
تهی ggml_metal_device_free(ggml_metal_device_t dev);

ggml_metal_device_t ggml_metal_device_get(صحیح device, صحیح n_devices);

تهی * ggml_metal_device_get_obj  (ggml_metal_device_t dev); // id<MTLDevice>
تهی * ggml_metal_device_get_queue(ggml_metal_device_t dev); // id<MTLCommandQueue>

ggml_metal_library_t ggml_metal_device_get_library(ggml_metal_device_t dev);

تهی ggml_metal_device_rsets_add(ggml_metal_device_t dev, ggml_metal_rset_t rset);
تهی ggml_metal_device_rsets_rm (ggml_metal_device_t dev, ggml_metal_rset_t rset);

تهی ggml_metal_device_rsets_keep_alive(ggml_metal_device_t dev);

ggml_metal_event_t ggml_metal_device_event_init(ggml_metal_device_t dev);
تهی ggml_metal_device_event_free(ggml_metal_device_t dev, ggml_metal_event_t ev);
تهی ggml_metal_device_event_synchronize(ggml_metal_device_t dev, ggml_metal_event_t ev);

تهی ggml_metal_device_get_memory(ggml_metal_device_t dev, size_t * آزاد, size_t * total);
bool ggml_metal_device_supports_op(ggml_metal_device_t dev, ثابت ساختار ggml_tensor * op);

ثابت ساختار ggml_metal_device_props * ggml_metal_device_get_props(ggml_metal_device_t dev);

ساختار ggml_metal_fusion_info;

// the device-owned fusion debugging context (NULL unless fusion debugging is enabled)
ساختار ggml_metal_fusion_info * ggml_metal_device_get_fusion_info(ggml_metal_device_t dev);

//
// device buffers
//

تعریفنوع ساختار ggml_metal_buffer * ggml_metal_buffer_t;

ggml_metal_buffer_t ggml_metal_buffer_init(ggml_metal_device_t dev, size_t size, bool shared);
ggml_metal_buffer_t ggml_metal_buffer_map (ggml_metal_device_t dev, تهی * ptr, size_t size, size_t max_tensor_size);

تهی   ggml_metal_buffer_free     (ggml_metal_buffer_t buf);
تهی * ggml_metal_buffer_get_base (ggml_metal_buffer_t buf);
bool   ggml_metal_buffer_is_shared(ggml_metal_buffer_t buf);

تهی   ggml_metal_buffer_memset_tensor(ggml_metal_buffer_t buf, ساختار ggml_tensor * tensor, uint8_t مقدار, size_t offset, size_t size);
تهی   ggml_metal_buffer_set_tensor   (ggml_metal_buffer_t buf, ساختار ggml_tensor * tensor, ثابت تهی * data, size_t offset, size_t size);
تهی   ggml_metal_buffer_get_tensor   (ggml_metal_buffer_t buf, ثابت ساختار ggml_tensor * tensor, تهی * data, size_t offset, size_t size);
bool   ggml_metal_buffer_cpy_tensor   (ggml_metal_buffer_t buf, ثابت ساختار ggml_tensor * src, ساختار ggml_tensor * dst);
تهی   ggml_metal_buffer_clear        (ggml_metal_buffer_t buf, uint8_t مقدار);

// finds the Metal buffer that contains the tensor data on the GPU device
// the assumption is that there is 1-to-1 mapping between the host and device memory buffers, so we can find the
// Metal buffer based on the host memory pointer
//
ساختار ggml_metal_buffer_id ggml_metal_buffer_get_id(ggml_metal_buffer_t buf, ثابت ساختار ggml_tensor * t);

#ifdef __cplusplus
}
#endif
