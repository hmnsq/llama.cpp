#pragma once

#অন্তর্ভাৱ "ggml.h"

#ifdef __cplusplus
বাহ্য "C" {
#endif

সংরচনা ggml_metal_buffer_id {
    শূন্য * metal; // id<MTLBuffer>
    size_t offs;
};

প্রকারনির্দেশ সংরচনা ggml_metal_device * ggml_metal_device_t;

//
// MTLFunctionConstantValues wrapper
//

প্রকারনির্দেশ সংরচনা ggml_metal_cv * ggml_metal_cv_t;

ggml_metal_cv_t ggml_metal_cv_init(শূন্য);
শূন্য ggml_metal_cv_free(ggml_metal_cv_t cv);

শূন্য ggml_metal_cv_set_int16(ggml_metal_cv_t cv, int16_t মান, int32_t idx);
শূন্য ggml_metal_cv_set_int32(ggml_metal_cv_t cv, int32_t মান, int32_t idx);
শূন্য ggml_metal_cv_set_bool (ggml_metal_cv_t cv, bool    মান, int32_t idx);

//
// MTLComputePipelineState wrapper
//

প্রকারনির্দেশ সংরচনা ggml_metal_pipeline * ggml_metal_pipeline_t;

ggml_metal_pipeline_t ggml_metal_pipeline_init(শূন্য);
শূন্য ggml_metal_pipeline_free(ggml_metal_pipeline_t pipeline);

// a collection of pipelines
প্রকারনির্দেশ সংরচনা ggml_metal_pipelines * ggml_metal_pipelines_t;

ggml_metal_pipelines_t ggml_metal_pipelines_init(শূন্য);
শূন্য ggml_metal_pipelines_free(ggml_metal_pipelines_t ppls);

শূন্য                  ggml_metal_pipelines_add(ggml_metal_pipelines_t ppls, ধ্রুৱ ৱর্ণ * name, ggml_metal_pipeline_t pipeline);
ggml_metal_pipeline_t ggml_metal_pipelines_get(ggml_metal_pipelines_t ppls, ধ্রুৱ ৱর্ণ * name);

সংরচনা ggml_metal_pipeline_with_params {
    ggml_metal_pipeline_t pipeline;

    পূর্ণাঙ্ক nsg;

    পূর্ণাঙ্ক nr0;
    পূর্ণাঙ্ক nr1;

    size_t smem;

    bool c4;
    bool cnt;
};

পূর্ণাঙ্ক ggml_metal_pipeline_max_theads_per_threadgroup(সংরচনা ggml_metal_pipeline_with_params pipeline);

//
// MTLCommandBuffer wrapper
//

প্রকারনির্দেশ শূন্য * ggml_metal_cmd_buf_t;

//
// MTLComputeCommandEncoder wrapper
//

প্রকারনির্দেশ সংরচনা ggml_metal_encoder * ggml_metal_encoder_t;

ggml_metal_encoder_t ggml_metal_encoder_init(ggml_metal_cmd_buf_t cmd_buf_raw, bool concurrent);
শূন্য ggml_metal_encoder_free(ggml_metal_encoder_t encoder);

শূন্য ggml_metal_encoder_debug_group_ঠেলো(ggml_metal_encoder_t encoder, ধ্রুৱ ৱর্ণ * name);
শূন্য ggml_metal_encoder_debug_group_বেৰ_কৰো (ggml_metal_encoder_t encoder);

শূন্য ggml_metal_encoder_set_pipeline(ggml_metal_encoder_t encoder, সংরচনা ggml_metal_pipeline_with_params pipeline);

শূন্য ggml_metal_encoder_set_bytes (ggml_metal_encoder_t encoder, শূন্য * data, size_t size, পূর্ণাঙ্ক idx);
শূন্য ggml_metal_encoder_set_buffer(ggml_metal_encoder_t encoder, সংরচনা ggml_metal_buffer_id buffer, পূর্ণাঙ্ক idx);

শূন্য ggml_metal_encoder_set_threadgroup_memory_size(ggml_metal_encoder_t encoder, size_t size, পূর্ণাঙ্ক idx);

শূন্য ggml_metal_encoder_dispatch_threadgroups(ggml_metal_encoder_t encoder, পূর্ণাঙ্ক tg0, পূর্ণাঙ্ক tg1, পূর্ণাঙ্ক tg2, পূর্ণাঙ্ক tptg0, পূর্ণাঙ্ক tptg1, পূর্ণাঙ্ক tptg2);

শূন্য ggml_metal_encoder_memory_barrier(ggml_metal_encoder_t encoder);

শূন্য ggml_metal_encoder_end_encoding(ggml_metal_encoder_t encoder);

//
// MTLLibrary wrapper
//

প্রকারনির্দেশ সংরচনা ggml_metal_library * ggml_metal_library_t;

ggml_metal_library_t ggml_metal_library_init            (ggml_metal_device_t dev);
ggml_metal_library_t ggml_metal_library_init_from_source(ggml_metal_device_t dev, ধ্রুৱ ৱর্ণ * source, bool verbose);

শূন্য ggml_metal_library_free(ggml_metal_library_t lib);

ggml_metal_device_t ggml_metal_library_get_device(ggml_metal_library_t lib);

সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline    (ggml_metal_library_t lib, ধ্রুৱ ৱর্ণ * name);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_compile_pipeline(ggml_metal_library_t lib, ধ্রুৱ ৱর্ণ * base, ধ্রুৱ ৱর্ণ * name, ggml_metal_cv_t cv);

সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_base              (ggml_metal_library_t lib, গণনা ggml_op op);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cpy               (ggml_metal_library_t lib, গণনা ggml_type tsrc, গণনা ggml_type tdst);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_1d           (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op, গণনা ggml_op_pool op_pool);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_2d           (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op, গণনা ggml_op_pool op_pool);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_get_rows          (ggml_metal_library_t lib, গণনা ggml_type tsrc);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_set_rows          (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_diag              (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_repeat            (ggml_metal_library_t lib, গণনা ggml_type tsrc);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_concat            (ggml_metal_library_t lib, গণনা ggml_type tsrc);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_unary             (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_silu_back         (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_glu               (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_sum               (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_sum_rows          (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cumsum_blk        (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cumsum_add        (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_tri               (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_soft_max          (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_lightning_indexer (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_hc           (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv          (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op, int32_t nc, bool use_silu);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv_batched  (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op, পূর্ণাঙ্ক ssm_conv_bs, int32_t nc, bool use_silu);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_scan          (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op, bool tail);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_scan_ssd_mma  (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rwkv              (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_gated_delta_net   (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_solve_tri         (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_ext        (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op, পূর্ণাঙ্ক nsg, পূর্ণাঙ্ক nxpsg, পূর্ণাঙ্ক r1ptg);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_mma        (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op, পূর্ণাঙ্ক nsg, পূর্ণাঙ্ক nt, পূর্ণাঙ্ক rt, bool add);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_mma_auto   (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op, bool add);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm            (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv            (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_map0    (ggml_metal_library_t lib, পূর্ণাঙ্ক ne02, পূর্ণাঙ্ক ne20);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_amax(ggml_metal_library_t lib);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_amax_part(ggml_metal_library_t lib);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id         (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_id         (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax            (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax_part       (ggml_metal_library_t lib);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax_final      (ggml_metal_library_t lib);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argsort           (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argsort_merge     (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_fwht              (ggml_metal_library_t lib, পূর্ণাঙ্ক n, গণনা ggml_type tsrc);

সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_শীৰ্ষ_k             (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_শীৰ্ষ_k_radix       (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_শীৰ্ষ_k_merge       (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_topk_moe          (ggml_metal_library_t lib, int32_t n_expert, int32_t শীৰ্ষ_k, bool with_norm);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_moe_reduce        (ggml_metal_library_t lib, int32_t n_expert_used);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin               (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op, int32_t n_fuse);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin_one           (ggml_metal_library_t lib, গণনা ggml_op op);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_l2_norm           (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_group_norm        (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_norm              (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op, int32_t n_fuse);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_norm_scale        (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rope              (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_im2col            (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_1d (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_2d (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_col2im_1d         (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_snake             (ggml_metal_library_t lib, গণনা ggml_type type);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_2d           (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_2d_dw        (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op, bool tiled);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_3d           (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_upscale           (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad               (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad_reflect_1d    (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_roll              (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_arange            (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_timestep_embedding(ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_step_adamw    (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_step_sgd      (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_memset            (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op);
সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_গুনতি_equal       (ggml_metal_library_t lib, ধ্রুৱ সংরচনা ggml_tensor * op);

সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_pad(
        ggml_metal_library_t lib,
        ধ্রুৱ সংরচনা ggml_tensor * op,
        bool    has_mask,
        int32_t ncpsg);

সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_kv_f16(
        ggml_metal_library_t lib,
        ধ্রুৱ সংরচনা ggml_tensor * op);

সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_blk(
        ggml_metal_library_t lib,
        ধ্রুৱ সংরচনা ggml_tensor * op,
        int32_t nqptg,
        int32_t ncpsg);

সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_tensor(
        ggml_metal_library_t lib,
        ধ্রুৱ সংরচনা ggml_tensor * op,
        bool has_mask,
        bool has_sinks,
        bool has_bias,
        bool has_scap);

সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext(
        ggml_metal_library_t lib,
        ধ্রুৱ সংরচনা ggml_tensor * op,
        bool    has_mask,
        bool    has_sinks,
        bool    has_bias,
        bool    has_scap,
        bool    has_kvpad,
        int32_t nsg,
        bool    use_kv_f16,
        int32_t ns10,
        int32_t ns20);

সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec_idx(
        ggml_metal_library_t lib,
        ধ্রুৱ সংরচনা ggml_tensor * op);

সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec(
        ggml_metal_library_t lib,
        ধ্রুৱ সংরচনা ggml_tensor * op,
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

সংরচনা ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce(
        ggml_metal_library_t lib,
        ধ্রুৱ সংরচনা ggml_tensor * op,
        int32_t dv,
        int32_t nwg);

// MTLResidencySet wrapper

প্রকারনির্দেশ শূন্য * ggml_metal_rset_t;

// a collection of residency sets (non-owning)
প্রকারনির্দেশ সংরচনা ggml_metal_rsets * ggml_metal_rsets_t;

ggml_metal_rsets_t ggml_metal_rsets_init(ggml_metal_device_t dev);
শূন্য ggml_metal_rsets_free(ggml_metal_rsets_t rsets);

//
// device
//

গণনা ggml_metal_device_id {
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

ধ্রুৱ ৱর্ণ * ggml_metal_device_id_token(গণনা ggml_metal_device_id id);

সংরচনা ggml_metal_device_props {
    পূর্ণাঙ্ক device;
    পূর্ণাঙ্ক device_phys;
    পূর্ণাঙ্ক device_virt;
    ৱর্ণ name[128];
    ৱর্ণ desc[128];

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

    গণনা ggml_metal_device_id device_id;
    পূর্ণাঙ্ক gpu_family;

    পূর্ণাঙ্ক op_offload_min_batch_size;
};

প্রকারনির্দেশ সংরচনা ggml_metal_event * ggml_metal_event_t;

শূন্য ggml_metal_event_encode_signal(ggml_metal_event_t ev, ggml_metal_cmd_buf_t cmd_buf);
শূন্য ggml_metal_event_encode_wait  (ggml_metal_event_t ev, ggml_metal_cmd_buf_t cmd_buf);

ggml_metal_device_t ggml_metal_device_init(পূর্ণাঙ্ক device, পূর্ণাঙ্ক n_devices);
শূন্য ggml_metal_device_free(ggml_metal_device_t dev);

ggml_metal_device_t ggml_metal_device_get(পূর্ণাঙ্ক device, পূর্ণাঙ্ক n_devices);

শূন্য * ggml_metal_device_get_obj  (ggml_metal_device_t dev); // id<MTLDevice>
শূন্য * ggml_metal_device_get_queue(ggml_metal_device_t dev); // id<MTLCommandQueue>

ggml_metal_library_t ggml_metal_device_get_library(ggml_metal_device_t dev);

শূন্য ggml_metal_device_rsets_add(ggml_metal_device_t dev, ggml_metal_rset_t rset);
শূন্য ggml_metal_device_rsets_rm (ggml_metal_device_t dev, ggml_metal_rset_t rset);

শূন্য ggml_metal_device_rsets_keep_alive(ggml_metal_device_t dev);

ggml_metal_event_t ggml_metal_device_event_init(ggml_metal_device_t dev);
শূন্য ggml_metal_device_event_free(ggml_metal_device_t dev, ggml_metal_event_t ev);
শূন্য ggml_metal_device_event_synchronize(ggml_metal_device_t dev, ggml_metal_event_t ev);

শূন্য ggml_metal_device_get_memory(ggml_metal_device_t dev, size_t * মুক্ত, size_t * total);
bool ggml_metal_device_supports_op(ggml_metal_device_t dev, ধ্রুৱ সংরচনা ggml_tensor * op);

ধ্রুৱ সংরচনা ggml_metal_device_props * ggml_metal_device_get_props(ggml_metal_device_t dev);

সংরচনা ggml_metal_fusion_info;

// the device-owned fusion debugging context (NULL unless fusion debugging is enabled)
সংরচনা ggml_metal_fusion_info * ggml_metal_device_get_fusion_info(ggml_metal_device_t dev);

//
// device buffers
//

প্রকারনির্দেশ সংরচনা ggml_metal_buffer * ggml_metal_buffer_t;

ggml_metal_buffer_t ggml_metal_buffer_init(ggml_metal_device_t dev, size_t size, bool shared);
ggml_metal_buffer_t ggml_metal_buffer_map (ggml_metal_device_t dev, শূন্য * ptr, size_t size, size_t max_tensor_size);

শূন্য   ggml_metal_buffer_free     (ggml_metal_buffer_t buf);
শূন্য * ggml_metal_buffer_get_base (ggml_metal_buffer_t buf);
bool   ggml_metal_buffer_is_shared(ggml_metal_buffer_t buf);

শূন্য   ggml_metal_buffer_memset_tensor(ggml_metal_buffer_t buf, সংরচনা ggml_tensor * tensor, uint8_t মান, size_t offset, size_t size);
শূন্য   ggml_metal_buffer_set_tensor   (ggml_metal_buffer_t buf, সংরচনা ggml_tensor * tensor, ধ্রুৱ শূন্য * data, size_t offset, size_t size);
শূন্য   ggml_metal_buffer_get_tensor   (ggml_metal_buffer_t buf, ধ্রুৱ সংরচনা ggml_tensor * tensor, শূন্য * data, size_t offset, size_t size);
bool   ggml_metal_buffer_cpy_tensor   (ggml_metal_buffer_t buf, ধ্রুৱ সংরচনা ggml_tensor * src, সংরচনা ggml_tensor * dst);
শূন্য   ggml_metal_buffer_clear        (ggml_metal_buffer_t buf, uint8_t মান);

// finds the Metal buffer that contains the tensor data on the GPU device
// the assumption is that there is 1-to-1 mapping between the host and device memory buffers, so we can find the
// Metal buffer based on the host memory pointer
//
সংরচনা ggml_metal_buffer_id ggml_metal_buffer_get_id(ggml_metal_buffer_t buf, ধ্রুৱ সংরচনা ggml_tensor * t);

#ifdef __cplusplus
}
#endif
