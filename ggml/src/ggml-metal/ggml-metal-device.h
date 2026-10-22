#pragma once

#ଅନ୍ତର୍ଭାଵ "ggml.h"

#ifdef __cplusplus
ବାହ୍ୟ "C" {
#endif

ସଂରଚନା ggml_metal_buffer_id {
    ଶୂନ୍ୟ * metal; // id<MTLBuffer>
    size_t offs;
};

ପ୍ରକାରବ୍ୟାଖ୍ୟା ସଂରଚନା ggml_metal_device * ggml_metal_device_t;

//
// MTLFunctionConstantValues wrapper
//

ପ୍ରକାରବ୍ୟାଖ୍ୟା ସଂରଚନା ggml_metal_cv * ggml_metal_cv_t;

ggml_metal_cv_t ggml_metal_cv_init(ଶୂନ୍ୟ);
ଶୂନ୍ୟ ggml_metal_cv_free(ggml_metal_cv_t cv);

ଶୂନ୍ୟ ggml_metal_cv_set_int16(ggml_metal_cv_t cv, int16_t ମୂଲ୍ୟ, int32_t idx);
ଶୂନ୍ୟ ggml_metal_cv_set_int32(ggml_metal_cv_t cv, int32_t ମୂଲ୍ୟ, int32_t idx);
ଶୂନ୍ୟ ggml_metal_cv_set_bool (ggml_metal_cv_t cv, bool    ମୂଲ୍ୟ, int32_t idx);

//
// MTLComputePipelineState wrapper
//

ପ୍ରକାରବ୍ୟାଖ୍ୟା ସଂରଚନା ggml_metal_pipeline * ggml_metal_pipeline_t;

ggml_metal_pipeline_t ggml_metal_pipeline_init(ଶୂନ୍ୟ);
ଶୂନ୍ୟ ggml_metal_pipeline_free(ggml_metal_pipeline_t pipeline);

// a collection of pipelines
ପ୍ରକାରବ୍ୟାଖ୍ୟା ସଂରଚନା ggml_metal_pipelines * ggml_metal_pipelines_t;

ggml_metal_pipelines_t ggml_metal_pipelines_init(ଶୂନ୍ୟ);
ଶୂନ୍ୟ ggml_metal_pipelines_free(ggml_metal_pipelines_t ppls);

ଶୂନ୍ୟ                  ggml_metal_pipelines_add(ggml_metal_pipelines_t ppls, ଅଚଳ ଅକ୍ଷର * name, ggml_metal_pipeline_t pipeline);
ggml_metal_pipeline_t ggml_metal_pipelines_get(ggml_metal_pipelines_t ppls, ଅଚଳ ଅକ୍ଷର * name);

ସଂରଚନା ggml_metal_pipeline_with_params {
    ggml_metal_pipeline_t pipeline;

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା nsg;

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr0;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା nr1;

    size_t smem;

    bool c4;
    bool cnt;
};

ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_metal_pipeline_max_theads_per_threadgroup(ସଂରଚନା ggml_metal_pipeline_with_params pipeline);

//
// MTLCommandBuffer wrapper
//

ପ୍ରକାରବ୍ୟାଖ୍ୟା ଶୂନ୍ୟ * ggml_metal_cmd_buf_t;

//
// MTLComputeCommandEncoder wrapper
//

ପ୍ରକାରବ୍ୟାଖ୍ୟା ସଂରଚନା ggml_metal_encoder * ggml_metal_encoder_t;

ggml_metal_encoder_t ggml_metal_encoder_init(ggml_metal_cmd_buf_t cmd_buf_raw, bool concurrent);
ଶୂନ୍ୟ ggml_metal_encoder_free(ggml_metal_encoder_t encoder);

ଶୂନ୍ୟ ggml_metal_encoder_debug_group_ଠେଲ(ggml_metal_encoder_t encoder, ଅଚଳ ଅକ୍ଷର * name);
ଶୂନ୍ୟ ggml_metal_encoder_debug_group_କାଢ଼ (ggml_metal_encoder_t encoder);

ଶୂନ୍ୟ ggml_metal_encoder_set_pipeline(ggml_metal_encoder_t encoder, ସଂରଚନା ggml_metal_pipeline_with_params pipeline);

ଶୂନ୍ୟ ggml_metal_encoder_set_bytes (ggml_metal_encoder_t encoder, ଶୂନ୍ୟ * data, size_t size, ପୂର୍ଣ୍ଣସଂଖ୍ୟା idx);
ଶୂନ୍ୟ ggml_metal_encoder_set_buffer(ggml_metal_encoder_t encoder, ସଂରଚନା ggml_metal_buffer_id buffer, ପୂର୍ଣ୍ଣସଂଖ୍ୟା idx);

ଶୂନ୍ୟ ggml_metal_encoder_set_threadgroup_memory_size(ggml_metal_encoder_t encoder, size_t size, ପୂର୍ଣ୍ଣସଂଖ୍ୟା idx);

ଶୂନ୍ୟ ggml_metal_encoder_dispatch_threadgroups(ggml_metal_encoder_t encoder, ପୂର୍ଣ୍ଣସଂଖ୍ୟା tg0, ପୂର୍ଣ୍ଣସଂଖ୍ୟା tg1, ପୂର୍ଣ୍ଣସଂଖ୍ୟା tg2, ପୂର୍ଣ୍ଣସଂଖ୍ୟା tptg0, ପୂର୍ଣ୍ଣସଂଖ୍ୟା tptg1, ପୂର୍ଣ୍ଣସଂଖ୍ୟା tptg2);

ଶୂନ୍ୟ ggml_metal_encoder_memory_barrier(ggml_metal_encoder_t encoder);

ଶୂନ୍ୟ ggml_metal_encoder_end_encoding(ggml_metal_encoder_t encoder);

//
// MTLLibrary wrapper
//

ପ୍ରକାରବ୍ୟାଖ୍ୟା ସଂରଚନା ggml_metal_library * ggml_metal_library_t;

ggml_metal_library_t ggml_metal_library_init            (ggml_metal_device_t dev);
ggml_metal_library_t ggml_metal_library_init_from_source(ggml_metal_device_t dev, ଅଚଳ ଅକ୍ଷର * source, bool verbose);

ଶୂନ୍ୟ ggml_metal_library_free(ggml_metal_library_t lib);

ggml_metal_device_t ggml_metal_library_get_device(ggml_metal_library_t lib);

ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline    (ggml_metal_library_t lib, ଅଚଳ ଅକ୍ଷର * name);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_compile_pipeline(ggml_metal_library_t lib, ଅଚଳ ଅକ୍ଷର * base, ଅଚଳ ଅକ୍ଷର * name, ggml_metal_cv_t cv);

ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_base              (ggml_metal_library_t lib, ଗଣନା ggml_op op);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cpy               (ggml_metal_library_t lib, ଗଣନା ggml_type tsrc, ଗଣନା ggml_type tdst);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_1d           (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op, ଗଣନା ggml_op_pool op_pool);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_2d           (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op, ଗଣନା ggml_op_pool op_pool);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_get_rows          (ggml_metal_library_t lib, ଗଣନା ggml_type tsrc);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_set_rows          (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_diag              (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_repeat            (ggml_metal_library_t lib, ଗଣନା ggml_type tsrc);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_concat            (ggml_metal_library_t lib, ଗଣନା ggml_type tsrc);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_unary             (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_silu_back         (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_glu               (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_sum               (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_sum_rows          (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cumsum_blk        (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cumsum_add        (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_tri               (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_soft_max          (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_lightning_indexer (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_hc           (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv          (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op, int32_t nc, bool use_silu);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv_batched  (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ssm_conv_bs, int32_t nc, bool use_silu);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_scan          (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op, bool tail);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_scan_ssd_mma  (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rwkv              (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_gated_delta_net   (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_solve_tri         (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_ext        (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nsg, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nxpsg, ପୂର୍ଣ୍ଣସଂଖ୍ୟା r1ptg);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_mma        (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nsg, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nt, ପୂର୍ଣ୍ଣସଂଖ୍ୟା rt, bool add);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_mma_auto   (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op, bool add);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm            (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv            (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_map0    (ggml_metal_library_t lib, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne20);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_amax(ggml_metal_library_t lib);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_amax_part(ggml_metal_library_t lib);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id         (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_id         (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax            (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax_part       (ggml_metal_library_t lib);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax_final      (ggml_metal_library_t lib);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argsort           (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argsort_merge     (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_fwht              (ggml_metal_library_t lib, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଗଣନା ggml_type tsrc);

ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ଶୀର୍ଷ_k             (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ଶୀର୍ଷ_k_radix       (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ଶୀର୍ଷ_k_merge       (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_topk_moe          (ggml_metal_library_t lib, int32_t n_expert, int32_t ଶୀର୍ଷ_k, bool with_norm);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_moe_reduce        (ggml_metal_library_t lib, int32_t n_expert_used);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin               (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op, int32_t n_fuse);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin_one           (ggml_metal_library_t lib, ଗଣନା ggml_op op);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_l2_norm           (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_group_norm        (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_norm              (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op, int32_t n_fuse);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_norm_scale        (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rope              (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_im2col            (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_1d (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_2d (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_col2im_1d         (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_snake             (ggml_metal_library_t lib, ଗଣନା ggml_type type);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_2d           (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_2d_dw        (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op, bool tiled);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_3d           (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_upscale           (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad               (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad_reflect_1d    (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_roll              (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_arange            (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_timestep_embedding(ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_step_adamw    (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_step_sgd      (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_memset            (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op);
ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ଗଣନା_equal       (ggml_metal_library_t lib, ଅଚଳ ସଂରଚନା ggml_tensor * op);

ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_pad(
        ggml_metal_library_t lib,
        ଅଚଳ ସଂରଚନା ggml_tensor * op,
        bool    has_mask,
        int32_t ncpsg);

ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_kv_f16(
        ggml_metal_library_t lib,
        ଅଚଳ ସଂରଚନା ggml_tensor * op);

ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_blk(
        ggml_metal_library_t lib,
        ଅଚଳ ସଂରଚନା ggml_tensor * op,
        int32_t nqptg,
        int32_t ncpsg);

ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_tensor(
        ggml_metal_library_t lib,
        ଅଚଳ ସଂରଚନା ggml_tensor * op,
        bool has_mask,
        bool has_sinks,
        bool has_bias,
        bool has_scap);

ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext(
        ggml_metal_library_t lib,
        ଅଚଳ ସଂରଚନା ggml_tensor * op,
        bool    has_mask,
        bool    has_sinks,
        bool    has_bias,
        bool    has_scap,
        bool    has_kvpad,
        int32_t nsg,
        bool    use_kv_f16,
        int32_t ns10,
        int32_t ns20);

ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec_idx(
        ggml_metal_library_t lib,
        ଅଚଳ ସଂରଚନା ggml_tensor * op);

ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec(
        ggml_metal_library_t lib,
        ଅଚଳ ସଂରଚନା ggml_tensor * op,
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

ସଂରଚନା ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce(
        ggml_metal_library_t lib,
        ଅଚଳ ସଂରଚନା ggml_tensor * op,
        int32_t dv,
        int32_t nwg);

// MTLResidencySet wrapper

ପ୍ରକାରବ୍ୟାଖ୍ୟା ଶୂନ୍ୟ * ggml_metal_rset_t;

// a collection of residency sets (non-owning)
ପ୍ରକାରବ୍ୟାଖ୍ୟା ସଂରଚନା ggml_metal_rsets * ggml_metal_rsets_t;

ggml_metal_rsets_t ggml_metal_rsets_init(ggml_metal_device_t dev);
ଶୂନ୍ୟ ggml_metal_rsets_free(ggml_metal_rsets_t rsets);

//
// device
//

ଗଣନା ggml_metal_device_id {
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

ଅଚଳ ଅକ୍ଷର * ggml_metal_device_id_token(ଗଣନା ggml_metal_device_id id);

ସଂରଚନା ggml_metal_device_props {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା device;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା device_phys;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା device_virt;
    ଅକ୍ଷର name[128];
    ଅକ୍ଷର desc[128];

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

    ଗଣନା ggml_metal_device_id device_id;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା gpu_family;

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_offload_min_batch_size;
};

ପ୍ରକାରବ୍ୟାଖ୍ୟା ସଂରଚନା ggml_metal_event * ggml_metal_event_t;

ଶୂନ୍ୟ ggml_metal_event_encode_signal(ggml_metal_event_t ev, ggml_metal_cmd_buf_t cmd_buf);
ଶୂନ୍ୟ ggml_metal_event_encode_wait  (ggml_metal_event_t ev, ggml_metal_cmd_buf_t cmd_buf);

ggml_metal_device_t ggml_metal_device_init(ପୂର୍ଣ୍ଣସଂଖ୍ୟା device, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_devices);
ଶୂନ୍ୟ ggml_metal_device_free(ggml_metal_device_t dev);

ggml_metal_device_t ggml_metal_device_get(ପୂର୍ଣ୍ଣସଂଖ୍ୟା device, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_devices);

ଶୂନ୍ୟ * ggml_metal_device_get_obj  (ggml_metal_device_t dev); // id<MTLDevice>
ଶୂନ୍ୟ * ggml_metal_device_get_queue(ggml_metal_device_t dev); // id<MTLCommandQueue>

ggml_metal_library_t ggml_metal_device_get_library(ggml_metal_device_t dev);

ଶୂନ୍ୟ ggml_metal_device_rsets_add(ggml_metal_device_t dev, ggml_metal_rset_t rset);
ଶୂନ୍ୟ ggml_metal_device_rsets_rm (ggml_metal_device_t dev, ggml_metal_rset_t rset);

ଶୂନ୍ୟ ggml_metal_device_rsets_keep_alive(ggml_metal_device_t dev);

ggml_metal_event_t ggml_metal_device_event_init(ggml_metal_device_t dev);
ଶୂନ୍ୟ ggml_metal_device_event_free(ggml_metal_device_t dev, ggml_metal_event_t ev);
ଶୂନ୍ୟ ggml_metal_device_event_synchronize(ggml_metal_device_t dev, ggml_metal_event_t ev);

ଶୂନ୍ୟ ggml_metal_device_get_memory(ggml_metal_device_t dev, size_t * ମୁକ୍ତ, size_t * total);
bool ggml_metal_device_supports_op(ggml_metal_device_t dev, ଅଚଳ ସଂରଚନା ggml_tensor * op);

ଅଚଳ ସଂରଚନା ggml_metal_device_props * ggml_metal_device_get_props(ggml_metal_device_t dev);

ସଂରଚନା ggml_metal_fusion_info;

// the device-owned fusion debugging context (NULL unless fusion debugging is enabled)
ସଂରଚନା ggml_metal_fusion_info * ggml_metal_device_get_fusion_info(ggml_metal_device_t dev);

//
// device buffers
//

ପ୍ରକାରବ୍ୟାଖ୍ୟା ସଂରଚନା ggml_metal_buffer * ggml_metal_buffer_t;

ggml_metal_buffer_t ggml_metal_buffer_init(ggml_metal_device_t dev, size_t size, bool shared);
ggml_metal_buffer_t ggml_metal_buffer_map (ggml_metal_device_t dev, ଶୂନ୍ୟ * ptr, size_t size, size_t max_tensor_size);

ଶୂନ୍ୟ   ggml_metal_buffer_free     (ggml_metal_buffer_t buf);
ଶୂନ୍ୟ * ggml_metal_buffer_get_base (ggml_metal_buffer_t buf);
bool   ggml_metal_buffer_is_shared(ggml_metal_buffer_t buf);

ଶୂନ୍ୟ   ggml_metal_buffer_memset_tensor(ggml_metal_buffer_t buf, ସଂରଚନା ggml_tensor * tensor, uint8_t ମୂଲ୍ୟ, size_t offset, size_t size);
ଶୂନ୍ୟ   ggml_metal_buffer_set_tensor   (ggml_metal_buffer_t buf, ସଂରଚନା ggml_tensor * tensor, ଅଚଳ ଶୂନ୍ୟ * data, size_t offset, size_t size);
ଶୂନ୍ୟ   ggml_metal_buffer_get_tensor   (ggml_metal_buffer_t buf, ଅଚଳ ସଂରଚନା ggml_tensor * tensor, ଶୂନ୍ୟ * data, size_t offset, size_t size);
bool   ggml_metal_buffer_cpy_tensor   (ggml_metal_buffer_t buf, ଅଚଳ ସଂରଚନା ggml_tensor * src, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ   ggml_metal_buffer_clear        (ggml_metal_buffer_t buf, uint8_t ମୂଲ୍ୟ);

// finds the Metal buffer that contains the tensor data on the GPU device
// the assumption is that there is 1-to-1 mapping between the host and device memory buffers, so we can find the
// Metal buffer based on the host memory pointer
//
ସଂରଚନା ggml_metal_buffer_id ggml_metal_buffer_get_id(ggml_metal_buffer_t buf, ଅଚଳ ସଂରଚନା ggml_tensor * t);

#ifdef __cplusplus
}
#endif
