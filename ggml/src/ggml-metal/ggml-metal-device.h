#ಸೂಚನೆ once

#ಅನ್ತರ್ಭಾವ "ggml.h"

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __cplusplus
ಬಾಹ್ಯ "C" {
#ಆದರೆ_ಅಂತ್ಯ

ರಚನೆ ggml_metal_ಬಫರ್_id {
    ಶೂನ್ಯ * metal; // id<MTLBuffer>
    size_t offs;
};

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ ggml_metal_device * ggml_metal_device_t;

//
// MTLFunctionConstantValues wrapper
//

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ ggml_metal_cv * ggml_metal_cv_t;

ggml_metal_cv_t ggml_metal_cv_init(ಶೂನ್ಯ);
ಶೂನ್ಯ ggml_metal_cv_free(ggml_metal_cv_t cv);

ಶೂನ್ಯ ggml_metal_cv_ಗಣ_int16(ggml_metal_cv_t cv, int16_t ಮೌಲ್ಯ, int32_t idx);
ಶೂನ್ಯ ggml_metal_cv_ಗಣ_int32(ggml_metal_cv_t cv, int32_t ಮೌಲ್ಯ, int32_t idx);
ಶೂನ್ಯ ggml_metal_cv_ಗಣ_bool (ggml_metal_cv_t cv, bool    ಮೌಲ್ಯ, int32_t idx);

//
// MTLComputePipelineState wrapper
//

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ ggml_metal_pipeline * ggml_metal_pipeline_t;

ggml_metal_pipeline_t ggml_metal_pipeline_init(ಶೂನ್ಯ);
ಶೂನ್ಯ ggml_metal_pipeline_free(ggml_metal_pipeline_t pipeline);

// a collection of pipelines
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ ggml_metal_pipelines * ggml_metal_pipelines_t;

ggml_metal_pipelines_t ggml_metal_pipelines_init(ಶೂನ್ಯ);
ಶೂನ್ಯ ggml_metal_pipelines_free(ggml_metal_pipelines_t ppls);

ಶೂನ್ಯ                  ggml_metal_pipelines_ಕೂಡಿಸು(ggml_metal_pipelines_t ppls, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * name, ggml_metal_pipeline_t pipeline);
ggml_metal_pipeline_t ggml_metal_pipelines_get(ggml_metal_pipelines_t ppls, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * name);

ರಚನೆ ggml_metal_pipeline_with_params {
    ggml_metal_pipeline_t pipeline;

    ಪೂರ್ಣಾಂಕ nsg;

    ಪೂರ್ಣಾಂಕ nr0;
    ಪೂರ್ಣಾಂಕ nr1;

    size_t smem;

    bool c4;
    bool cnt;
};

ಪೂರ್ಣಾಂಕ ggml_metal_pipeline_max_theads_ಪ್ರತಿ_threadgroup(ರಚನೆ ggml_metal_pipeline_with_params pipeline);

//
// MTLCommandBuffer wrapper
//

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಶೂನ್ಯ * ggml_metal_cmd_buf_t;

//
// MTLComputeCommandEncoder wrapper
//

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ ggml_metal_encoder * ggml_metal_encoder_t;

ggml_metal_encoder_t ggml_metal_encoder_init(ggml_metal_cmd_buf_t cmd_buf_raw, bool concurrent);
ಶೂನ್ಯ ggml_metal_encoder_free(ggml_metal_encoder_t encoder);

ಶೂನ್ಯ ggml_metal_encoder_debug_ಗುಂಪು_ತಳ್ಳು(ggml_metal_encoder_t encoder, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * name);
ಶೂನ್ಯ ggml_metal_encoder_debug_ಗುಂಪು_ತೆಗೆ (ggml_metal_encoder_t encoder);

ಶೂನ್ಯ ggml_metal_encoder_ಗಣ_pipeline(ggml_metal_encoder_t encoder, ರಚನೆ ggml_metal_pipeline_with_params pipeline);

ಶೂನ್ಯ ggml_metal_encoder_ಗಣ_ಬೈಟುಗಳು (ggml_metal_encoder_t encoder, ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಗಾತ್ರ_2, ಪೂರ್ಣಾಂಕ idx);
ಶೂನ್ಯ ggml_metal_encoder_ಗಣ_ಬಫರ್(ggml_metal_encoder_t encoder, ರಚನೆ ggml_metal_ಬಫರ್_id ಬಫರ್, ಪೂರ್ಣಾಂಕ idx);

ಶೂನ್ಯ ggml_metal_encoder_ಗಣ_threadgroup_ಸ್ಮರಣೆ_ಗಾತ್ರ(ggml_metal_encoder_t encoder, size_t ಗಾತ್ರ_2, ಪೂರ್ಣಾಂಕ idx);

ಶೂನ್ಯ ggml_metal_encoder_dispatch_threadgroups(ggml_metal_encoder_t encoder, ಪೂರ್ಣಾಂಕ tg0, ಪೂರ್ಣಾಂಕ tg1, ಪೂರ್ಣಾಂಕ tg2, ಪೂರ್ಣಾಂಕ tptg0, ಪೂರ್ಣಾಂಕ tptg1, ಪೂರ್ಣಾಂಕ tptg2);

ಶೂನ್ಯ ggml_metal_encoder_ಸ್ಮರಣೆ_barrier(ggml_metal_encoder_t encoder);

ಶೂನ್ಯ ggml_metal_encoder_ಕೊನೆ_encoding(ggml_metal_encoder_t encoder);

//
// MTLLibrary wrapper
//

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ ggml_metal_library * ggml_metal_library_t;

ggml_metal_library_t ggml_metal_library_init            (ggml_metal_device_t dev);
ggml_metal_library_t ggml_metal_library_init_from_ಮೂಲ(ggml_metal_device_t dev, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಮೂಲ, bool verbose);

ಶೂನ್ಯ ggml_metal_library_free(ggml_metal_library_t lib);

ggml_metal_device_t ggml_metal_library_get_device(ggml_metal_library_t lib);

ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline    (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * name);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_compile_pipeline(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ತಳ, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * name, ggml_metal_cv_t cv);

ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ತಳ              (ggml_metal_library_t lib, ಎಣಿಕೆ ggml_op op);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cpy               (ggml_metal_library_t lib, ಎಣಿಕೆ ggml_type tsrc, ಎಣಿಕೆ ggml_type tdst);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_1d           (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op, ಎಣಿಕೆ ggml_op_pool op_pool);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_2d           (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op, ಎಣಿಕೆ ggml_op_pool op_pool);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_get_ಅಡ್ಡಸಾಲುಗಳು          (ggml_metal_library_t lib, ಎಣಿಕೆ ggml_type tsrc);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು          (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_diag              (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_repeat            (ggml_metal_library_t lib, ಎಣಿಕೆ ggml_type tsrc);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_concat            (ggml_metal_library_t lib, ಎಣಿಕೆ ggml_type tsrc);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_unary             (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_silu_ಹಿಂಭಾಗ         (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_glu               (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ಮೊತ್ತ               (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು          (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cumsum_blk        (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cumsum_ಕೂಡಿಸು        (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_tri               (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_soft_max          (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_lightning_indexer (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_hc           (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv          (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op, int32_t nc, bool use_silu);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv_batched  (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op, ಪೂರ್ಣಾಂಕ ssm_conv_bs, int32_t nc, bool use_silu);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_ಪರಿಶೀಲನೆ          (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op, bool ಬಾಲ);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_ಪರಿಶೀಲನೆ_ssd_mma  (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rwkv              (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_gated_delta_net   (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_solve_tri         (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_ext        (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op, ಪೂರ್ಣಾಂಕ nsg, ಪೂರ್ಣಾಂಕ nxpsg, ಪೂರ್ಣಾಂಕ r1ptg);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_mma        (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op, ಪೂರ್ಣಾಂಕ nsg, ಪೂರ್ಣಾಂಕ nt, ಪೂರ್ಣಾಂಕ rt, bool ಕೂಡಿಸು_2);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_mma_auto   (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op, bool ಕೂಡಿಸು_2);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm            (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv            (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_map0    (ggml_metal_library_t lib, ಪೂರ್ಣಾಂಕ ne02, ಪೂರ್ಣಾಂಕ ne20);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_amax(ggml_metal_library_t lib);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_amax_part(ggml_metal_library_t lib);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id         (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_id         (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax            (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax_part       (ggml_metal_library_t lib);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax_final      (ggml_metal_library_t lib);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argsort           (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argsort_ವಿಲೀನ     (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_fwht              (ggml_metal_library_t lib, ಪೂರ್ಣಾಂಕ n, ಎಣಿಕೆ ggml_type tsrc);

ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ತುದಿ_k             (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ತುದಿ_k_radix       (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ತುದಿ_k_ವಿಲೀನ       (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_topk_moe          (ggml_metal_library_t lib, int32_t n_expert, int32_t ತುದಿ_k, bool with_norm);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_moe_reduce        (ggml_metal_library_t lib, int32_t n_expert_ಬಳಸಿದ);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin               (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op, int32_t n_fuse);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin_one           (ggml_metal_library_t lib, ಎಣಿಕೆ ggml_op op);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_l2_norm           (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ಗುಂಪು_norm        (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_norm              (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op, int32_t n_fuse);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_norm_scale        (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rope              (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_im2col            (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_1d (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_2d (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_col2im_1d         (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_snake             (ggml_metal_library_t lib, ಎಣಿಕೆ ggml_type type);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_2d           (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_2d_dw        (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op, bool tiled);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_3d           (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_upscale           (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad               (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad_reflect_1d    (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_roll              (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_arange            (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_timestep_embedding(ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_ಹೆಜ್ಜೆ_adamw    (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_ಹೆಜ್ಜೆ_sgd      (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_memset            (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);
ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ಎಣಿಕೆ_equal       (ggml_metal_library_t lib, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);

ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_pad(
        ggml_metal_library_t lib,
        ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op,
        bool    has_ಮುಖವಾಡ,
        int32_t ncpsg);

ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_kv_f16(
        ggml_metal_library_t lib,
        ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);

ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_blk(
        ggml_metal_library_t lib,
        ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op,
        int32_t nqptg,
        int32_t ncpsg);

ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_tensor(
        ggml_metal_library_t lib,
        ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op,
        bool has_ಮುಖವಾಡ,
        bool has_sinks,
        bool has_bias,
        bool has_scap);

ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext(
        ggml_metal_library_t lib,
        ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op,
        bool    has_ಮುಖವಾಡ,
        bool    has_sinks,
        bool    has_bias,
        bool    has_scap,
        bool    has_kvpad,
        int32_t nsg,
        bool    use_kv_f16,
        int32_t ns10,
        int32_t ns20);

ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec_idx(
        ggml_metal_library_t lib,
        ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);

ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec(
        ggml_metal_library_t lib,
        ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op,
        bool    has_ಮುಖವಾಡ,
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

ರಚನೆ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce(
        ggml_metal_library_t lib,
        ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op,
        int32_t dv,
        int32_t nwg);

// MTLResidencySet wrapper

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಶೂನ್ಯ * ggml_metal_rset_t;

// a collection of residency sets (non-owning)
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ ggml_metal_rsets * ggml_metal_rsets_t;

ggml_metal_rsets_t ggml_metal_rsets_init(ggml_metal_device_t dev);
ಶೂನ್ಯ ggml_metal_rsets_free(ggml_metal_rsets_t rsets);

//
// device
//

ಎಣಿಕೆ ggml_metal_device_id {
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

ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_metal_device_id_token(ಎಣಿಕೆ ggml_metal_device_id id);

ರಚನೆ ggml_metal_device_props {
    ಪೂರ್ಣಾಂಕ device;
    ಪೂರ್ಣಾಂಕ device_phys;
    ಪೂರ್ಣಾಂಕ device_virt;
    ಅಕ್ಷರ name[128];
    ಅಕ್ಷರ desc[128];

    size_t max_ಬಫರ್_ಗಾತ್ರ;
    size_t max_working_ಗಣ_ಗಾತ್ರ;
    size_t max_theadgroup_ಸ್ಮರಣೆ_ಗಾತ್ರ;

    bool has_simdgroup_reduction;
    bool has_simdgroup_mm;
    bool has_unified_ಸ್ಮರಣೆ;
    bool has_bfloat;
    bool has_tensor;
    bool use_residency_sets;
    bool use_shared_buffers;

    ಎಣಿಕೆ ggml_metal_device_id device_id;
    ಪೂರ್ಣಾಂಕ gpu_family;

    ಪೂರ್ಣಾಂಕ op_offload_min_batch_ಗಾತ್ರ;
};

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ ggml_metal_event * ggml_metal_event_t;

ಶೂನ್ಯ ggml_metal_event_encode_signal(ggml_metal_event_t ev, ggml_metal_cmd_buf_t cmd_buf);
ಶೂನ್ಯ ggml_metal_event_encode_wait  (ggml_metal_event_t ev, ggml_metal_cmd_buf_t cmd_buf);

ggml_metal_device_t ggml_metal_device_init(ಪೂರ್ಣಾಂಕ device, ಪೂರ್ಣಾಂಕ n_devices);
ಶೂನ್ಯ ggml_metal_device_free(ggml_metal_device_t dev);

ggml_metal_device_t ggml_metal_device_get(ಪೂರ್ಣಾಂಕ device, ಪೂರ್ಣಾಂಕ n_devices);

ಶೂನ್ಯ * ggml_metal_device_get_obj  (ggml_metal_device_t dev); // id<MTLDevice>
ಶೂನ್ಯ * ggml_metal_device_get_ಸರತಿ(ggml_metal_device_t dev); // id<MTLCommandQueue>

ggml_metal_library_t ggml_metal_device_get_library(ggml_metal_device_t dev);

ಶೂನ್ಯ ggml_metal_device_rsets_ಕೂಡಿಸು(ggml_metal_device_t dev, ggml_metal_rset_t rset);
ಶೂನ್ಯ ggml_metal_device_rsets_rm (ggml_metal_device_t dev, ggml_metal_rset_t rset);

ಶೂನ್ಯ ggml_metal_device_rsets_keep_alive(ggml_metal_device_t dev);

ggml_metal_event_t ggml_metal_device_event_init(ggml_metal_device_t dev);
ಶೂನ್ಯ ggml_metal_device_event_free(ggml_metal_device_t dev, ggml_metal_event_t ev);
ಶೂನ್ಯ ggml_metal_device_event_synchronize(ggml_metal_device_t dev, ggml_metal_event_t ev);

ಶೂನ್ಯ ggml_metal_device_get_ಸ್ಮರಣೆ(ggml_metal_device_t dev, size_t * ಬಿಡುಗಡೆ, size_t * ಒಟ್ಟು);
bool ggml_metal_device_supports_op(ggml_metal_device_t dev, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);

ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_metal_device_props * ggml_metal_device_get_props(ggml_metal_device_t dev);

ರಚನೆ ggml_metal_fusion_info;

// the device-owned fusion debugging context (NULL unless fusion debugging is enabled)
ರಚನೆ ggml_metal_fusion_info * ggml_metal_device_get_fusion_info(ggml_metal_device_t dev);

//
// device buffers
//

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ ggml_metal_ಬಫರ್ * ggml_metal_ಬಫರ್_t;

ggml_metal_ಬಫರ್_t ggml_metal_ಬಫರ್_init(ggml_metal_device_t dev, size_t ಗಾತ್ರ_2, bool shared);
ggml_metal_ಬಫರ್_t ggml_metal_ಬಫರ್_map (ggml_metal_device_t dev, ಶೂನ್ಯ * ptr, size_t ಗಾತ್ರ_2, size_t max_tensor_ಗಾತ್ರ);

ಶೂನ್ಯ   ggml_metal_ಬಫರ್_free     (ggml_metal_ಬಫರ್_t buf);
ಶೂನ್ಯ * ggml_metal_ಬಫರ್_get_ತಳ (ggml_metal_ಬಫರ್_t buf);
bool   ggml_metal_ಬಫರ್_is_shared(ggml_metal_ಬಫರ್_t buf);

ಶೂನ್ಯ   ggml_metal_ಬಫರ್_memset_tensor(ggml_metal_ಬಫರ್_t buf, ರಚನೆ ggml_tensor * tensor, uint8_t ಮೌಲ್ಯ, size_t ಸ್ಥಾನಾಂತರ, size_t ಗಾತ್ರ_2);
ಶೂನ್ಯ   ggml_metal_ಬಫರ್_ಗಣ_tensor   (ggml_metal_ಬಫರ್_t buf, ರಚನೆ ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t ಗಾತ್ರ_2);
ಶೂನ್ಯ   ggml_metal_ಬಫರ್_get_tensor   (ggml_metal_ಬಫರ್_t buf, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor, ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t ಗಾತ್ರ_2);
bool   ggml_metal_ಬಫರ್_cpy_tensor   (ggml_metal_ಬಫರ್_t buf, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src, ರಚನೆ ggml_tensor * dst);
ಶೂನ್ಯ   ggml_metal_ಬಫರ್_clear        (ggml_metal_ಬಫರ್_t buf, uint8_t ಮೌಲ್ಯ);

// finds the Metal buffer that contains the tensor data on the GPU device
// the assumption is that there is 1-to-1 mapping between the host and device memory buffers, so we can find the
// Metal buffer based on the host memory pointer
//
ರಚನೆ ggml_metal_ಬಫರ್_id ggml_metal_ಬಫರ್_get_id(ggml_metal_ಬಫರ್_t buf, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * t);

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __cplusplus
}
#ಆದರೆ_ಅಂತ್ಯ
