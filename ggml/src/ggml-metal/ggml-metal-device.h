#رهنمود once

#ضمن "ggml.h"

#اگر_تعریف_شده __cplusplus
خارجي "C" {
#پایان_اگر

بنية ggml_metal_میانگیر_id {
    فراغ * metal; // id<MTLBuffer>
    size_t offs;
};

تعريفنوع بنية ggml_metal_device * ggml_metal_device_t;

//
// MTLFunctionConstantValues wrapper
//

تعريفنوع بنية ggml_metal_cv * ggml_metal_cv_t;

ggml_metal_cv_t ggml_metal_cv_init(فراغ);
فراغ ggml_metal_cv_free(ggml_metal_cv_t cv);

فراغ ggml_metal_cv_مجموعه_int16(ggml_metal_cv_t cv, int16_t مقدار, int32_t idx);
فراغ ggml_metal_cv_مجموعه_int32(ggml_metal_cv_t cv, int32_t مقدار, int32_t idx);
فراغ ggml_metal_cv_مجموعه_bool (ggml_metal_cv_t cv, bool    مقدار, int32_t idx);

//
// MTLComputePipelineState wrapper
//

تعريفنوع بنية ggml_metal_pipeline * ggml_metal_pipeline_t;

ggml_metal_pipeline_t ggml_metal_pipeline_init(فراغ);
فراغ ggml_metal_pipeline_free(ggml_metal_pipeline_t pipeline);

// a collection of pipelines
تعريفنوع بنية ggml_metal_pipelines * ggml_metal_pipelines_t;

ggml_metal_pipelines_t ggml_metal_pipelines_init(فراغ);
فراغ ggml_metal_pipelines_free(ggml_metal_pipelines_t ppls);

فراغ                  ggml_metal_pipelines_جمع_کن(ggml_metal_pipelines_t ppls, ثابت حرف * name, ggml_metal_pipeline_t pipeline);
ggml_metal_pipeline_t ggml_metal_pipelines_get(ggml_metal_pipelines_t ppls, ثابت حرف * name);

بنية ggml_metal_pipeline_with_params {
    ggml_metal_pipeline_t pipeline;

    صحيح nsg;

    صحيح nr0;
    صحيح nr1;

    size_t smem;

    bool c4;
    bool cnt;
};

صحيح ggml_metal_pipeline_max_theads_در_هر_threadgroup(بنية ggml_metal_pipeline_with_params pipeline);

//
// MTLCommandBuffer wrapper
//

تعريفنوع فراغ * ggml_metal_cmd_buf_t;

//
// MTLComputeCommandEncoder wrapper
//

تعريفنوع بنية ggml_metal_encoder * ggml_metal_encoder_t;

ggml_metal_encoder_t ggml_metal_encoder_init(ggml_metal_cmd_buf_t cmd_buf_raw, bool concurrent);
فراغ ggml_metal_encoder_free(ggml_metal_encoder_t encoder);

فراغ ggml_metal_encoder_debug_گروه_افزودن(ggml_metal_encoder_t encoder, ثابت حرف * name);
فراغ ggml_metal_encoder_debug_گروه_برداشت (ggml_metal_encoder_t encoder);

فراغ ggml_metal_encoder_مجموعه_pipeline(ggml_metal_encoder_t encoder, بنية ggml_metal_pipeline_with_params pipeline);

فراغ ggml_metal_encoder_مجموعه_بایتها (ggml_metal_encoder_t encoder, فراغ * داده, size_t اندازه, صحيح idx);
فراغ ggml_metal_encoder_مجموعه_میانگیر(ggml_metal_encoder_t encoder, بنية ggml_metal_میانگیر_id میانگیر, صحيح idx);

فراغ ggml_metal_encoder_مجموعه_threadgroup_حافظه_اندازه(ggml_metal_encoder_t encoder, size_t اندازه, صحيح idx);

فراغ ggml_metal_encoder_dispatch_threadgroups(ggml_metal_encoder_t encoder, صحيح tg0, صحيح tg1, صحيح tg2, صحيح tptg0, صحيح tptg1, صحيح tptg2);

فراغ ggml_metal_encoder_حافظه_barrier(ggml_metal_encoder_t encoder);

فراغ ggml_metal_encoder_انتها_encoding(ggml_metal_encoder_t encoder);

//
// MTLLibrary wrapper
//

تعريفنوع بنية ggml_metal_library * ggml_metal_library_t;

ggml_metal_library_t ggml_metal_library_init            (ggml_metal_device_t dev);
ggml_metal_library_t ggml_metal_library_init_from_منبع(ggml_metal_device_t dev, ثابت حرف * منبع, bool verbose);

فراغ ggml_metal_library_free(ggml_metal_library_t lib);

ggml_metal_device_t ggml_metal_library_get_device(ggml_metal_library_t lib);

بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline    (ggml_metal_library_t lib, ثابت حرف * name);
بنية ggml_metal_pipeline_with_params ggml_metal_library_compile_pipeline(ggml_metal_library_t lib, ثابت حرف * پایه, ثابت حرف * name, ggml_metal_cv_t cv);

بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_پایه              (ggml_metal_library_t lib, تعداد ggml_op op);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cpy               (ggml_metal_library_t lib, تعداد ggml_type tsrc, تعداد ggml_type tdst);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_1d           (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op, تعداد ggml_op_pool op_pool);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_2d           (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op, تعداد ggml_op_pool op_pool);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_get_ردیفها          (ggml_metal_library_t lib, تعداد ggml_type tsrc);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_مجموعه_ردیفها          (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_diag              (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_repeat            (ggml_metal_library_t lib, تعداد ggml_type tsrc);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_concat            (ggml_metal_library_t lib, تعداد ggml_type tsrc);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_unary             (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_silu_پشت         (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_glu               (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_جمع               (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_جمع_ردیفها          (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cumsum_blk        (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cumsum_جمع_کن        (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_tri               (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_soft_max          (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_lightning_indexer (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_hc           (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv          (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op, int32_t nc, bool use_silu);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv_batched  (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op, صحيح ssm_conv_bs, int32_t nc, bool use_silu);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_پیمایش          (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op, bool دم);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_پیمایش_ssd_mma  (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rwkv              (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_gated_delta_net   (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_solve_tri         (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_ext        (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op, صحيح nsg, صحيح nxpsg, صحيح r1ptg);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_mma        (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op, صحيح nsg, صحيح nt, صحيح rt, bool جمع_کن_2);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_mma_auto   (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op, bool جمع_کن_2);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm            (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv            (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_map0    (ggml_metal_library_t lib, صحيح ne02, صحيح ne20);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_amax(ggml_metal_library_t lib);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_amax_part(ggml_metal_library_t lib);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id         (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_id         (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax            (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax_part       (ggml_metal_library_t lib);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax_final      (ggml_metal_library_t lib);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argsort           (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argsort_ادغام     (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_fwht              (ggml_metal_library_t lib, صحيح n, تعداد ggml_type tsrc);

بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_بالاترین_k             (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_بالاترین_k_radix       (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_بالاترین_k_ادغام       (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_topk_moe          (ggml_metal_library_t lib, int32_t n_expert, int32_t بالاترین_k, bool with_norm);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_moe_reduce        (ggml_metal_library_t lib, int32_t n_expert_استفاده_شده);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin               (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op, int32_t n_fuse);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin_one           (ggml_metal_library_t lib, تعداد ggml_op op);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_l2_norm           (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_گروه_norm        (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_norm              (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op, int32_t n_fuse);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_norm_scale        (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rope              (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_im2col            (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_1d (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_2d (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_col2im_1d         (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_snake             (ggml_metal_library_t lib, تعداد ggml_type type);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_2d           (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_2d_dw        (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op, bool tiled);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_3d           (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_upscale           (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad               (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad_reflect_1d    (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_roll              (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_arange            (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_timestep_embedding(ggml_metal_library_t lib, ثابت بنية ggml_tensor * op);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_گام_adamw    (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_گام_sgd      (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_memset            (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op);
بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_شمار_equal       (ggml_metal_library_t lib, ثابت بنية ggml_tensor * op);

بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_pad(
        ggml_metal_library_t lib,
        ثابت بنية ggml_tensor * op,
        bool    has_ماسک,
        int32_t ncpsg);

بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_kv_f16(
        ggml_metal_library_t lib,
        ثابت بنية ggml_tensor * op);

بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_blk(
        ggml_metal_library_t lib,
        ثابت بنية ggml_tensor * op,
        int32_t nqptg,
        int32_t ncpsg);

بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_tensor(
        ggml_metal_library_t lib,
        ثابت بنية ggml_tensor * op,
        bool has_ماسک,
        bool has_sinks,
        bool has_bias,
        bool has_scap);

بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext(
        ggml_metal_library_t lib,
        ثابت بنية ggml_tensor * op,
        bool    has_ماسک,
        bool    has_sinks,
        bool    has_bias,
        bool    has_scap,
        bool    has_kvpad,
        int32_t nsg,
        bool    use_kv_f16,
        int32_t ns10,
        int32_t ns20);

بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec_idx(
        ggml_metal_library_t lib,
        ثابت بنية ggml_tensor * op);

بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec(
        ggml_metal_library_t lib,
        ثابت بنية ggml_tensor * op,
        bool    has_ماسک,
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

بنية ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce(
        ggml_metal_library_t lib,
        ثابت بنية ggml_tensor * op,
        int32_t dv,
        int32_t nwg);

// MTLResidencySet wrapper

تعريفنوع فراغ * ggml_metal_rset_t;

// a collection of residency sets (non-owning)
تعريفنوع بنية ggml_metal_rsets * ggml_metal_rsets_t;

ggml_metal_rsets_t ggml_metal_rsets_init(ggml_metal_device_t dev);
فراغ ggml_metal_rsets_free(ggml_metal_rsets_t rsets);

//
// device
//

تعداد ggml_metal_device_id {
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

ثابت حرف * ggml_metal_device_id_token(تعداد ggml_metal_device_id id);

بنية ggml_metal_device_props {
    صحيح device;
    صحيح device_phys;
    صحيح device_virt;
    حرف name[128];
    حرف desc[128];

    size_t max_میانگیر_اندازه;
    size_t max_working_مجموعه_اندازه;
    size_t max_theadgroup_حافظه_اندازه;

    bool has_simdgroup_reduction;
    bool has_simdgroup_mm;
    bool has_unified_حافظه;
    bool has_bfloat;
    bool has_tensor;
    bool use_residency_sets;
    bool use_shared_buffers;

    تعداد ggml_metal_device_id device_id;
    صحيح gpu_family;

    صحيح op_offload_min_batch_اندازه;
};

تعريفنوع بنية ggml_metal_event * ggml_metal_event_t;

فراغ ggml_metal_event_encode_signal(ggml_metal_event_t ev, ggml_metal_cmd_buf_t cmd_buf);
فراغ ggml_metal_event_encode_wait  (ggml_metal_event_t ev, ggml_metal_cmd_buf_t cmd_buf);

ggml_metal_device_t ggml_metal_device_init(صحيح device, صحيح n_devices);
فراغ ggml_metal_device_free(ggml_metal_device_t dev);

ggml_metal_device_t ggml_metal_device_get(صحيح device, صحيح n_devices);

فراغ * ggml_metal_device_get_obj  (ggml_metal_device_t dev); // id<MTLDevice>
فراغ * ggml_metal_device_get_صف(ggml_metal_device_t dev); // id<MTLCommandQueue>

ggml_metal_library_t ggml_metal_device_get_library(ggml_metal_device_t dev);

فراغ ggml_metal_device_rsets_جمع_کن(ggml_metal_device_t dev, ggml_metal_rset_t rset);
فراغ ggml_metal_device_rsets_rm (ggml_metal_device_t dev, ggml_metal_rset_t rset);

فراغ ggml_metal_device_rsets_keep_alive(ggml_metal_device_t dev);

ggml_metal_event_t ggml_metal_device_event_init(ggml_metal_device_t dev);
فراغ ggml_metal_device_event_free(ggml_metal_device_t dev, ggml_metal_event_t ev);
فراغ ggml_metal_device_event_synchronize(ggml_metal_device_t dev, ggml_metal_event_t ev);

فراغ ggml_metal_device_get_حافظه(ggml_metal_device_t dev, size_t * حرر, size_t * مجموع);
bool ggml_metal_device_supports_op(ggml_metal_device_t dev, ثابت بنية ggml_tensor * op);

ثابت بنية ggml_metal_device_props * ggml_metal_device_get_props(ggml_metal_device_t dev);

بنية ggml_metal_fusion_info;

// the device-owned fusion debugging context (NULL unless fusion debugging is enabled)
بنية ggml_metal_fusion_info * ggml_metal_device_get_fusion_info(ggml_metal_device_t dev);

//
// device buffers
//

تعريفنوع بنية ggml_metal_میانگیر * ggml_metal_میانگیر_t;

ggml_metal_میانگیر_t ggml_metal_میانگیر_init(ggml_metal_device_t dev, size_t اندازه, bool shared);
ggml_metal_میانگیر_t ggml_metal_میانگیر_map (ggml_metal_device_t dev, فراغ * ptr, size_t اندازه, size_t max_tensor_اندازه);

فراغ   ggml_metal_میانگیر_free     (ggml_metal_میانگیر_t buf);
فراغ * ggml_metal_میانگیر_get_پایه (ggml_metal_میانگیر_t buf);
bool   ggml_metal_میانگیر_is_shared(ggml_metal_میانگیر_t buf);

فراغ   ggml_metal_میانگیر_memset_tensor(ggml_metal_میانگیر_t buf, بنية ggml_tensor * tensor, uint8_t مقدار, size_t جابه_جایی, size_t اندازه);
فراغ   ggml_metal_میانگیر_مجموعه_tensor   (ggml_metal_میانگیر_t buf, بنية ggml_tensor * tensor, ثابت فراغ * داده, size_t جابه_جایی, size_t اندازه);
فراغ   ggml_metal_میانگیر_get_tensor   (ggml_metal_میانگیر_t buf, ثابت بنية ggml_tensor * tensor, فراغ * داده, size_t جابه_جایی, size_t اندازه);
bool   ggml_metal_میانگیر_cpy_tensor   (ggml_metal_میانگیر_t buf, ثابت بنية ggml_tensor * src, بنية ggml_tensor * dst);
فراغ   ggml_metal_میانگیر_clear        (ggml_metal_میانگیر_t buf, uint8_t مقدار);

// finds the Metal buffer that contains the tensor data on the GPU device
// the assumption is that there is 1-to-1 mapping between the host and device memory buffers, so we can find the
// Metal buffer based on the host memory pointer
//
بنية ggml_metal_میانگیر_id ggml_metal_میانگیر_get_id(ggml_metal_میانگیر_t buf, ثابت بنية ggml_tensor * t);

#اگر_تعریف_شده __cplusplus
}
#پایان_اگر
