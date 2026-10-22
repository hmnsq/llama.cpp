#pragma once

#அந்தர்பாவ "ggml.h"

#ifdef __cplusplus
வெளி "C" {
#endif

கட்டமைப்பு ggml_metal_buffer_id {
    வெறுமை * metal; // id<MTLBuffer>
    size_t offs;
};

வகைவரை கட்டமைப்பு ggml_metal_device * ggml_metal_device_t;

//
// MTLFunctionConstantValues wrapper
//

வகைவரை கட்டமைப்பு ggml_metal_cv * ggml_metal_cv_t;

ggml_metal_cv_t ggml_metal_cv_init(வெறுமை);
வெறுமை ggml_metal_cv_free(ggml_metal_cv_t cv);

வெறுமை ggml_metal_cv_set_int16(ggml_metal_cv_t cv, int16_t மதிப்பு, int32_t idx);
வெறுமை ggml_metal_cv_set_int32(ggml_metal_cv_t cv, int32_t மதிப்பு, int32_t idx);
வெறுமை ggml_metal_cv_set_bool (ggml_metal_cv_t cv, bool    மதிப்பு, int32_t idx);

//
// MTLComputePipelineState wrapper
//

வகைவரை கட்டமைப்பு ggml_metal_pipeline * ggml_metal_pipeline_t;

ggml_metal_pipeline_t ggml_metal_pipeline_init(வெறுமை);
வெறுமை ggml_metal_pipeline_free(ggml_metal_pipeline_t pipeline);

// a collection of pipelines
வகைவரை கட்டமைப்பு ggml_metal_pipelines * ggml_metal_pipelines_t;

ggml_metal_pipelines_t ggml_metal_pipelines_init(வெறுமை);
வெறுமை ggml_metal_pipelines_free(ggml_metal_pipelines_t ppls);

வெறுமை                  ggml_metal_pipelines_add(ggml_metal_pipelines_t ppls, மாறிலி எழுத்து * name, ggml_metal_pipeline_t pipeline);
ggml_metal_pipeline_t ggml_metal_pipelines_get(ggml_metal_pipelines_t ppls, மாறிலி எழுத்து * name);

கட்டமைப்பு ggml_metal_pipeline_with_params {
    ggml_metal_pipeline_t pipeline;

    முழு nsg;

    முழு nr0;
    முழு nr1;

    size_t smem;

    bool c4;
    bool cnt;
};

முழு ggml_metal_pipeline_max_theads_per_threadgroup(கட்டமைப்பு ggml_metal_pipeline_with_params pipeline);

//
// MTLCommandBuffer wrapper
//

வகைவரை வெறுமை * ggml_metal_cmd_buf_t;

//
// MTLComputeCommandEncoder wrapper
//

வகைவரை கட்டமைப்பு ggml_metal_encoder * ggml_metal_encoder_t;

ggml_metal_encoder_t ggml_metal_encoder_init(ggml_metal_cmd_buf_t cmd_buf_raw, bool concurrent);
வெறுமை ggml_metal_encoder_free(ggml_metal_encoder_t encoder);

வெறுமை ggml_metal_encoder_debug_group_தள்ளு(ggml_metal_encoder_t encoder, மாறிலி எழுத்து * name);
வெறுமை ggml_metal_encoder_debug_group_எடு (ggml_metal_encoder_t encoder);

வெறுமை ggml_metal_encoder_set_pipeline(ggml_metal_encoder_t encoder, கட்டமைப்பு ggml_metal_pipeline_with_params pipeline);

வெறுமை ggml_metal_encoder_set_bytes (ggml_metal_encoder_t encoder, வெறுமை * data, size_t size, முழு idx);
வெறுமை ggml_metal_encoder_set_buffer(ggml_metal_encoder_t encoder, கட்டமைப்பு ggml_metal_buffer_id buffer, முழு idx);

வெறுமை ggml_metal_encoder_set_threadgroup_memory_size(ggml_metal_encoder_t encoder, size_t size, முழு idx);

வெறுமை ggml_metal_encoder_dispatch_threadgroups(ggml_metal_encoder_t encoder, முழு tg0, முழு tg1, முழு tg2, முழு tptg0, முழு tptg1, முழு tptg2);

வெறுமை ggml_metal_encoder_memory_barrier(ggml_metal_encoder_t encoder);

வெறுமை ggml_metal_encoder_end_encoding(ggml_metal_encoder_t encoder);

//
// MTLLibrary wrapper
//

வகைவரை கட்டமைப்பு ggml_metal_library * ggml_metal_library_t;

ggml_metal_library_t ggml_metal_library_init            (ggml_metal_device_t dev);
ggml_metal_library_t ggml_metal_library_init_from_source(ggml_metal_device_t dev, மாறிலி எழுத்து * source, bool verbose);

வெறுமை ggml_metal_library_free(ggml_metal_library_t lib);

ggml_metal_device_t ggml_metal_library_get_device(ggml_metal_library_t lib);

கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline    (ggml_metal_library_t lib, மாறிலி எழுத்து * name);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_compile_pipeline(ggml_metal_library_t lib, மாறிலி எழுத்து * base, மாறிலி எழுத்து * name, ggml_metal_cv_t cv);

கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_base              (ggml_metal_library_t lib, எண்ணிக்கை ggml_op op);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cpy               (ggml_metal_library_t lib, எண்ணிக்கை ggml_type tsrc, எண்ணிக்கை ggml_type tdst);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_1d           (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op, எண்ணிக்கை ggml_op_pool op_pool);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_2d           (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op, எண்ணிக்கை ggml_op_pool op_pool);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_get_rows          (ggml_metal_library_t lib, எண்ணிக்கை ggml_type tsrc);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_set_rows          (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_diag              (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_repeat            (ggml_metal_library_t lib, எண்ணிக்கை ggml_type tsrc);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_concat            (ggml_metal_library_t lib, எண்ணிக்கை ggml_type tsrc);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_unary             (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_silu_back         (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_glu               (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_sum               (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_sum_rows          (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cumsum_blk        (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cumsum_add        (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_tri               (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_soft_max          (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_lightning_indexer (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_hc           (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv          (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op, int32_t nc, bool use_silu);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv_batched  (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op, முழு ssm_conv_bs, int32_t nc, bool use_silu);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_scan          (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op, bool tail);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_scan_ssd_mma  (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rwkv              (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_gated_delta_net   (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_solve_tri         (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_ext        (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op, முழு nsg, முழு nxpsg, முழு r1ptg);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_mma        (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op, முழு nsg, முழு nt, முழு rt, bool add);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_mma_auto   (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op, bool add);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm            (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv            (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_map0    (ggml_metal_library_t lib, முழு ne02, முழு ne20);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_amax(ggml_metal_library_t lib);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_amax_part(ggml_metal_library_t lib);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id         (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_id         (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax            (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax_part       (ggml_metal_library_t lib);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax_final      (ggml_metal_library_t lib);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argsort           (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argsort_merge     (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_fwht              (ggml_metal_library_t lib, முழு n, எண்ணிக்கை ggml_type tsrc);

கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_உச்சி_k             (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_உச்சி_k_radix       (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_உச்சி_k_merge       (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_topk_moe          (ggml_metal_library_t lib, int32_t n_expert, int32_t உச்சி_k, bool with_norm);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_moe_reduce        (ggml_metal_library_t lib, int32_t n_expert_used);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin               (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op, int32_t n_fuse);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin_one           (ggml_metal_library_t lib, எண்ணிக்கை ggml_op op);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_l2_norm           (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_group_norm        (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_norm              (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op, int32_t n_fuse);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_norm_scale        (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rope              (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_im2col            (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_1d (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_2d (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_col2im_1d         (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_snake             (ggml_metal_library_t lib, எண்ணிக்கை ggml_type type);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_2d           (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_2d_dw        (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op, bool tiled);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_3d           (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_upscale           (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad               (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad_reflect_1d    (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_roll              (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_arange            (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_timestep_embedding(ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_step_adamw    (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_step_sgd      (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_memset            (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op);
கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_மொத்தம்_equal       (ggml_metal_library_t lib, மாறிலி கட்டமைப்பு ggml_tensor * op);

கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_pad(
        ggml_metal_library_t lib,
        மாறிலி கட்டமைப்பு ggml_tensor * op,
        bool    has_mask,
        int32_t ncpsg);

கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_kv_f16(
        ggml_metal_library_t lib,
        மாறிலி கட்டமைப்பு ggml_tensor * op);

கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_blk(
        ggml_metal_library_t lib,
        மாறிலி கட்டமைப்பு ggml_tensor * op,
        int32_t nqptg,
        int32_t ncpsg);

கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_tensor(
        ggml_metal_library_t lib,
        மாறிலி கட்டமைப்பு ggml_tensor * op,
        bool has_mask,
        bool has_sinks,
        bool has_bias,
        bool has_scap);

கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext(
        ggml_metal_library_t lib,
        மாறிலி கட்டமைப்பு ggml_tensor * op,
        bool    has_mask,
        bool    has_sinks,
        bool    has_bias,
        bool    has_scap,
        bool    has_kvpad,
        int32_t nsg,
        bool    use_kv_f16,
        int32_t ns10,
        int32_t ns20);

கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec_idx(
        ggml_metal_library_t lib,
        மாறிலி கட்டமைப்பு ggml_tensor * op);

கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec(
        ggml_metal_library_t lib,
        மாறிலி கட்டமைப்பு ggml_tensor * op,
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

கட்டமைப்பு ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce(
        ggml_metal_library_t lib,
        மாறிலி கட்டமைப்பு ggml_tensor * op,
        int32_t dv,
        int32_t nwg);

// MTLResidencySet wrapper

வகைவரை வெறுமை * ggml_metal_rset_t;

// a collection of residency sets (non-owning)
வகைவரை கட்டமைப்பு ggml_metal_rsets * ggml_metal_rsets_t;

ggml_metal_rsets_t ggml_metal_rsets_init(ggml_metal_device_t dev);
வெறுமை ggml_metal_rsets_free(ggml_metal_rsets_t rsets);

//
// device
//

எண்ணிக்கை ggml_metal_device_id {
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

மாறிலி எழுத்து * ggml_metal_device_id_token(எண்ணிக்கை ggml_metal_device_id id);

கட்டமைப்பு ggml_metal_device_props {
    முழு device;
    முழு device_phys;
    முழு device_virt;
    எழுத்து name[128];
    எழுத்து desc[128];

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

    எண்ணிக்கை ggml_metal_device_id device_id;
    முழு gpu_family;

    முழு op_offload_min_batch_size;
};

வகைவரை கட்டமைப்பு ggml_metal_event * ggml_metal_event_t;

வெறுமை ggml_metal_event_encode_signal(ggml_metal_event_t ev, ggml_metal_cmd_buf_t cmd_buf);
வெறுமை ggml_metal_event_encode_wait  (ggml_metal_event_t ev, ggml_metal_cmd_buf_t cmd_buf);

ggml_metal_device_t ggml_metal_device_init(முழு device, முழு n_devices);
வெறுமை ggml_metal_device_free(ggml_metal_device_t dev);

ggml_metal_device_t ggml_metal_device_get(முழு device, முழு n_devices);

வெறுமை * ggml_metal_device_get_obj  (ggml_metal_device_t dev); // id<MTLDevice>
வெறுமை * ggml_metal_device_get_queue(ggml_metal_device_t dev); // id<MTLCommandQueue>

ggml_metal_library_t ggml_metal_device_get_library(ggml_metal_device_t dev);

வெறுமை ggml_metal_device_rsets_add(ggml_metal_device_t dev, ggml_metal_rset_t rset);
வெறுமை ggml_metal_device_rsets_rm (ggml_metal_device_t dev, ggml_metal_rset_t rset);

வெறுமை ggml_metal_device_rsets_keep_alive(ggml_metal_device_t dev);

ggml_metal_event_t ggml_metal_device_event_init(ggml_metal_device_t dev);
வெறுமை ggml_metal_device_event_free(ggml_metal_device_t dev, ggml_metal_event_t ev);
வெறுமை ggml_metal_device_event_synchronize(ggml_metal_device_t dev, ggml_metal_event_t ev);

வெறுமை ggml_metal_device_get_memory(ggml_metal_device_t dev, size_t * விடுவி, size_t * total);
bool ggml_metal_device_supports_op(ggml_metal_device_t dev, மாறிலி கட்டமைப்பு ggml_tensor * op);

மாறிலி கட்டமைப்பு ggml_metal_device_props * ggml_metal_device_get_props(ggml_metal_device_t dev);

கட்டமைப்பு ggml_metal_fusion_info;

// the device-owned fusion debugging context (NULL unless fusion debugging is enabled)
கட்டமைப்பு ggml_metal_fusion_info * ggml_metal_device_get_fusion_info(ggml_metal_device_t dev);

//
// device buffers
//

வகைவரை கட்டமைப்பு ggml_metal_buffer * ggml_metal_buffer_t;

ggml_metal_buffer_t ggml_metal_buffer_init(ggml_metal_device_t dev, size_t size, bool shared);
ggml_metal_buffer_t ggml_metal_buffer_map (ggml_metal_device_t dev, வெறுமை * ptr, size_t size, size_t max_tensor_size);

வெறுமை   ggml_metal_buffer_free     (ggml_metal_buffer_t buf);
வெறுமை * ggml_metal_buffer_get_base (ggml_metal_buffer_t buf);
bool   ggml_metal_buffer_is_shared(ggml_metal_buffer_t buf);

வெறுமை   ggml_metal_buffer_memset_tensor(ggml_metal_buffer_t buf, கட்டமைப்பு ggml_tensor * tensor, uint8_t மதிப்பு, size_t offset, size_t size);
வெறுமை   ggml_metal_buffer_set_tensor   (ggml_metal_buffer_t buf, கட்டமைப்பு ggml_tensor * tensor, மாறிலி வெறுமை * data, size_t offset, size_t size);
வெறுமை   ggml_metal_buffer_get_tensor   (ggml_metal_buffer_t buf, மாறிலி கட்டமைப்பு ggml_tensor * tensor, வெறுமை * data, size_t offset, size_t size);
bool   ggml_metal_buffer_cpy_tensor   (ggml_metal_buffer_t buf, மாறிலி கட்டமைப்பு ggml_tensor * src, கட்டமைப்பு ggml_tensor * dst);
வெறுமை   ggml_metal_buffer_clear        (ggml_metal_buffer_t buf, uint8_t மதிப்பு);

// finds the Metal buffer that contains the tensor data on the GPU device
// the assumption is that there is 1-to-1 mapping between the host and device memory buffers, so we can find the
// Metal buffer based on the host memory pointer
//
கட்டமைப்பு ggml_metal_buffer_id ggml_metal_buffer_get_id(ggml_metal_buffer_t buf, மாறிலி கட்டமைப்பு ggml_tensor * t);

#ifdef __cplusplus
}
#endif
