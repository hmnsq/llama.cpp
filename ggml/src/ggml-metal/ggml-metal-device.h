#ہدایت once

#شامل "ggml.h"

#اگر_تعریف_شدہ __cplusplus
خارجی "C" {
#اگر_ختم

ساخت ggml_metal_بفر_id {
    خالی * metal; // id<MTLBuffer>
    size_t offs;
};

قسمتعریف ساخت ggml_metal_device * ggml_metal_device_t;

//
// MTLFunctionConstantValues wrapper
//

قسمتعریف ساخت ggml_metal_cv * ggml_metal_cv_t;

ggml_metal_cv_t ggml_metal_cv_init(خالی);
خالی ggml_metal_cv_free(ggml_metal_cv_t cv);

خالی ggml_metal_cv_سیٹ_int16(ggml_metal_cv_t cv, int16_t قدر, int32_t idx);
خالی ggml_metal_cv_سیٹ_int32(ggml_metal_cv_t cv, int32_t قدر, int32_t idx);
خالی ggml_metal_cv_سیٹ_bool (ggml_metal_cv_t cv, bool    قدر, int32_t idx);

//
// MTLComputePipelineState wrapper
//

قسمتعریف ساخت ggml_metal_pipeline * ggml_metal_pipeline_t;

ggml_metal_pipeline_t ggml_metal_pipeline_init(خالی);
خالی ggml_metal_pipeline_free(ggml_metal_pipeline_t pipeline);

// a collection of pipelines
قسمتعریف ساخت ggml_metal_pipelines * ggml_metal_pipelines_t;

ggml_metal_pipelines_t ggml_metal_pipelines_init(خالی);
خالی ggml_metal_pipelines_free(ggml_metal_pipelines_t ppls);

خالی                  ggml_metal_pipelines_جوڑو(ggml_metal_pipelines_t ppls, ثابت حرف * name, ggml_metal_pipeline_t pipeline);
ggml_metal_pipeline_t ggml_metal_pipelines_get(ggml_metal_pipelines_t ppls, ثابت حرف * name);

ساخت ggml_metal_pipeline_with_params {
    ggml_metal_pipeline_t pipeline;

    صحیح nsg;

    صحیح nr0;
    صحیح nr1;

    size_t smem;

    bool c4;
    bool cnt;
};

صحیح ggml_metal_pipeline_max_theads_فی_threadgroup(ساخت ggml_metal_pipeline_with_params pipeline);

//
// MTLCommandBuffer wrapper
//

قسمتعریف خالی * ggml_metal_cmd_buf_t;

//
// MTLComputeCommandEncoder wrapper
//

قسمتعریف ساخت ggml_metal_encoder * ggml_metal_encoder_t;

ggml_metal_encoder_t ggml_metal_encoder_init(ggml_metal_cmd_buf_t cmd_buf_raw, bool concurrent);
خالی ggml_metal_encoder_free(ggml_metal_encoder_t encoder);

خالی ggml_metal_encoder_debug_گروہ_ڈالو(ggml_metal_encoder_t encoder, ثابت حرف * name);
خالی ggml_metal_encoder_debug_گروہ_نکالو (ggml_metal_encoder_t encoder);

خالی ggml_metal_encoder_سیٹ_pipeline(ggml_metal_encoder_t encoder, ساخت ggml_metal_pipeline_with_params pipeline);

خالی ggml_metal_encoder_سیٹ_بائٹیں (ggml_metal_encoder_t encoder, خالی * ڈیٹا, size_t سائز, صحیح idx);
خالی ggml_metal_encoder_سیٹ_بفر(ggml_metal_encoder_t encoder, ساخت ggml_metal_بفر_id بفر, صحیح idx);

خالی ggml_metal_encoder_سیٹ_threadgroup_حافظہ_سائز(ggml_metal_encoder_t encoder, size_t سائز, صحیح idx);

خالی ggml_metal_encoder_dispatch_threadgroups(ggml_metal_encoder_t encoder, صحیح tg0, صحیح tg1, صحیح tg2, صحیح tptg0, صحیح tptg1, صحیح tptg2);

خالی ggml_metal_encoder_حافظہ_barrier(ggml_metal_encoder_t encoder);

خالی ggml_metal_encoder_انتہا_encoding(ggml_metal_encoder_t encoder);

//
// MTLLibrary wrapper
//

قسمتعریف ساخت ggml_metal_library * ggml_metal_library_t;

ggml_metal_library_t ggml_metal_library_init            (ggml_metal_device_t dev);
ggml_metal_library_t ggml_metal_library_init_from_ماخذ(ggml_metal_device_t dev, ثابت حرف * ماخذ, bool verbose);

خالی ggml_metal_library_free(ggml_metal_library_t lib);

ggml_metal_device_t ggml_metal_library_get_device(ggml_metal_library_t lib);

ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline    (ggml_metal_library_t lib, ثابت حرف * name);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_compile_pipeline(ggml_metal_library_t lib, ثابت حرف * بنیاد, ثابت حرف * name, ggml_metal_cv_t cv);

ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_بنیاد              (ggml_metal_library_t lib, شمار ggml_op op);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cpy               (ggml_metal_library_t lib, شمار ggml_type tsrc, شمار ggml_type tdst);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_1d           (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op, شمار ggml_op_pool op_pool);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_2d           (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op, شمار ggml_op_pool op_pool);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_get_صفیں          (ggml_metal_library_t lib, شمار ggml_type tsrc);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_سیٹ_صفیں          (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_diag              (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_repeat            (ggml_metal_library_t lib, شمار ggml_type tsrc);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_concat            (ggml_metal_library_t lib, شمار ggml_type tsrc);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_unary             (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_silu_عقب         (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_glu               (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_مجموع               (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_مجموع_صفیں          (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cumsum_blk        (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cumsum_جوڑو        (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_tri               (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_soft_max          (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_lightning_indexer (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_hc           (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv          (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op, int32_t nc, bool use_silu);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv_batched  (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op, صحیح ssm_conv_bs, int32_t nc, bool use_silu);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_جائزہ          (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op, bool دم);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_جائزہ_ssd_mma  (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rwkv              (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_gated_delta_net   (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_solve_tri         (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_ext        (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op, صحیح nsg, صحیح nxpsg, صحیح r1ptg);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_mma        (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op, صحیح nsg, صحیح nt, صحیح rt, bool جوڑو_2);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_mma_auto   (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op, bool جوڑو_2);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm            (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv            (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_map0    (ggml_metal_library_t lib, صحیح ne02, صحیح ne20);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_amax(ggml_metal_library_t lib);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_amax_part(ggml_metal_library_t lib);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id         (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_id         (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax            (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax_part       (ggml_metal_library_t lib);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax_final      (ggml_metal_library_t lib);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argsort           (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argsort_ملاؤ     (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_fwht              (ggml_metal_library_t lib, صحیح n, شمار ggml_type tsrc);

ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_چوٹی_k             (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_چوٹی_k_radix       (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_چوٹی_k_ملاؤ       (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_topk_moe          (ggml_metal_library_t lib, int32_t n_expert, int32_t چوٹی_k, bool with_norm);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_moe_reduce        (ggml_metal_library_t lib, int32_t n_expert_مستعمل);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin               (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op, int32_t n_fuse);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin_one           (ggml_metal_library_t lib, شمار ggml_op op);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_l2_norm           (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_گروہ_norm        (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_norm              (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op, int32_t n_fuse);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_norm_scale        (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rope              (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_im2col            (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_1d (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_2d (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_col2im_1d         (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_snake             (ggml_metal_library_t lib, شمار ggml_type type);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_2d           (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_2d_dw        (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op, bool tiled);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_3d           (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_upscale           (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad               (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad_reflect_1d    (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_roll              (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_arange            (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_timestep_embedding(ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_قدم_adamw    (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_قدم_sgd      (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_memset            (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op);
ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_گنتی_equal       (ggml_metal_library_t lib, ثابت ساخت ggml_tensor * op);

ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_pad(
        ggml_metal_library_t lib,
        ثابت ساخت ggml_tensor * op,
        bool    has_نقاب,
        int32_t ncpsg);

ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_kv_f16(
        ggml_metal_library_t lib,
        ثابت ساخت ggml_tensor * op);

ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_blk(
        ggml_metal_library_t lib,
        ثابت ساخت ggml_tensor * op,
        int32_t nqptg,
        int32_t ncpsg);

ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_tensor(
        ggml_metal_library_t lib,
        ثابت ساخت ggml_tensor * op,
        bool has_نقاب,
        bool has_sinks,
        bool has_bias,
        bool has_scap);

ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext(
        ggml_metal_library_t lib,
        ثابت ساخت ggml_tensor * op,
        bool    has_نقاب,
        bool    has_sinks,
        bool    has_bias,
        bool    has_scap,
        bool    has_kvpad,
        int32_t nsg,
        bool    use_kv_f16,
        int32_t ns10,
        int32_t ns20);

ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec_idx(
        ggml_metal_library_t lib,
        ثابت ساخت ggml_tensor * op);

ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec(
        ggml_metal_library_t lib,
        ثابت ساخت ggml_tensor * op,
        bool    has_نقاب,
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

ساخت ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce(
        ggml_metal_library_t lib,
        ثابت ساخت ggml_tensor * op,
        int32_t dv,
        int32_t nwg);

// MTLResidencySet wrapper

قسمتعریف خالی * ggml_metal_rset_t;

// a collection of residency sets (non-owning)
قسمتعریف ساخت ggml_metal_rsets * ggml_metal_rsets_t;

ggml_metal_rsets_t ggml_metal_rsets_init(ggml_metal_device_t dev);
خالی ggml_metal_rsets_free(ggml_metal_rsets_t rsets);

//
// device
//

شمار ggml_metal_device_id {
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

ثابت حرف * ggml_metal_device_id_token(شمار ggml_metal_device_id id);

ساخت ggml_metal_device_props {
    صحیح device;
    صحیح device_phys;
    صحیح device_virt;
    حرف name[128];
    حرف desc[128];

    size_t max_بفر_سائز;
    size_t max_working_سیٹ_سائز;
    size_t max_theadgroup_حافظہ_سائز;

    bool has_simdgroup_reduction;
    bool has_simdgroup_mm;
    bool has_unified_حافظہ;
    bool has_bfloat;
    bool has_tensor;
    bool use_residency_sets;
    bool use_shared_buffers;

    شمار ggml_metal_device_id device_id;
    صحیح gpu_family;

    صحیح op_offload_min_batch_سائز;
};

قسمتعریف ساخت ggml_metal_event * ggml_metal_event_t;

خالی ggml_metal_event_encode_signal(ggml_metal_event_t ev, ggml_metal_cmd_buf_t cmd_buf);
خالی ggml_metal_event_encode_wait  (ggml_metal_event_t ev, ggml_metal_cmd_buf_t cmd_buf);

ggml_metal_device_t ggml_metal_device_init(صحیح device, صحیح n_devices);
خالی ggml_metal_device_free(ggml_metal_device_t dev);

ggml_metal_device_t ggml_metal_device_get(صحیح device, صحیح n_devices);

خالی * ggml_metal_device_get_obj  (ggml_metal_device_t dev); // id<MTLDevice>
خالی * ggml_metal_device_get_قطار(ggml_metal_device_t dev); // id<MTLCommandQueue>

ggml_metal_library_t ggml_metal_device_get_library(ggml_metal_device_t dev);

خالی ggml_metal_device_rsets_جوڑو(ggml_metal_device_t dev, ggml_metal_rset_t rset);
خالی ggml_metal_device_rsets_rm (ggml_metal_device_t dev, ggml_metal_rset_t rset);

خالی ggml_metal_device_rsets_keep_alive(ggml_metal_device_t dev);

ggml_metal_event_t ggml_metal_device_event_init(ggml_metal_device_t dev);
خالی ggml_metal_device_event_free(ggml_metal_device_t dev, ggml_metal_event_t ev);
خالی ggml_metal_device_event_synchronize(ggml_metal_device_t dev, ggml_metal_event_t ev);

خالی ggml_metal_device_get_حافظہ(ggml_metal_device_t dev, size_t * آزاد, size_t * کل);
bool ggml_metal_device_supports_op(ggml_metal_device_t dev, ثابت ساخت ggml_tensor * op);

ثابت ساخت ggml_metal_device_props * ggml_metal_device_get_props(ggml_metal_device_t dev);

ساخت ggml_metal_fusion_info;

// the device-owned fusion debugging context (NULL unless fusion debugging is enabled)
ساخت ggml_metal_fusion_info * ggml_metal_device_get_fusion_info(ggml_metal_device_t dev);

//
// device buffers
//

قسمتعریف ساخت ggml_metal_بفر * ggml_metal_بفر_t;

ggml_metal_بفر_t ggml_metal_بفر_init(ggml_metal_device_t dev, size_t سائز, bool shared);
ggml_metal_بفر_t ggml_metal_بفر_map (ggml_metal_device_t dev, خالی * ptr, size_t سائز, size_t max_tensor_سائز);

خالی   ggml_metal_بفر_free     (ggml_metal_بفر_t buf);
خالی * ggml_metal_بفر_get_بنیاد (ggml_metal_بفر_t buf);
bool   ggml_metal_بفر_is_shared(ggml_metal_بفر_t buf);

خالی   ggml_metal_بفر_memset_tensor(ggml_metal_بفر_t buf, ساخت ggml_tensor * tensor, uint8_t قدر, size_t انحراف, size_t سائز);
خالی   ggml_metal_بفر_سیٹ_tensor   (ggml_metal_بفر_t buf, ساخت ggml_tensor * tensor, ثابت خالی * ڈیٹا, size_t انحراف, size_t سائز);
خالی   ggml_metal_بفر_get_tensor   (ggml_metal_بفر_t buf, ثابت ساخت ggml_tensor * tensor, خالی * ڈیٹا, size_t انحراف, size_t سائز);
bool   ggml_metal_بفر_cpy_tensor   (ggml_metal_بفر_t buf, ثابت ساخت ggml_tensor * src, ساخت ggml_tensor * dst);
خالی   ggml_metal_بفر_clear        (ggml_metal_بفر_t buf, uint8_t قدر);

// finds the Metal buffer that contains the tensor data on the GPU device
// the assumption is that there is 1-to-1 mapping between the host and device memory buffers, so we can find the
// Metal buffer based on the host memory pointer
//
ساخت ggml_metal_بفر_id ggml_metal_بفر_get_id(ggml_metal_بفر_t buf, ثابت ساخت ggml_tensor * t);

#اگر_تعریف_شدہ __cplusplus
}
#اگر_ختم
