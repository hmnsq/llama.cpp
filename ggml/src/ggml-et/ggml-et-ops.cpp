#अन्तर्भाव "ggml-et-ops.h"

#अन्तर्भाव "ggml-et-cpu-compare.h"
#अन्तर्भाव "ggml-et-kernels.h"
#अन्तर्भाव "ggml-impl.h"

#अन्तर्भाव <stdio.h>

#अन्तर्भाव <cstdint>

// CPU comparison configuration - can be enabled for debugging
स्थिर ggml_et_cpu_compare_config rope_cpu_compare_config = {
    /* .enabled = */ मिच्छा,
    /* .use_cpu_result = */ मिच्छा,  // Replace ET result with CPU result
    /* .log_differences = */ सच्च,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

स्थिर ggml_et_cpu_compare_config rms_norm_cpu_compare_config = {
    /* .enabled = */ मिच्छा,
    /* .use_cpu_result = */ मिच्छा,
    /* .log_differences = */ सच्च,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

स्थिर ggml_et_cpu_compare_config norm_cpu_compare_config = {
    /* .enabled = */ मिच्छा,
    /* .use_cpu_result = */ मिच्छा,
    /* .log_differences = */ सच्च,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

स्थिर ggml_et_cpu_compare_config l2_norm_cpu_compare_config = {
    /* .enabled = */ मिच्छा,
    /* .use_cpu_result = */ मिच्छा,
    /* .log_differences = */ सच्च,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

स्थिर ggml_et_cpu_compare_config group_norm_cpu_compare_config = {
    /* .enabled = */ मिच्छा,
    /* .use_cpu_result = */ मिच्छा,
    /* .log_differences = */ सच्च,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

स्थिर ggml_et_cpu_compare_config im2col_cpu_compare_config = {
    /* .enabled = */ मिच्छा,
    /* .use_cpu_result = */ मिच्छा,
    /* .log_differences = */ सच्च,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

स्थिर ggml_et_cpu_compare_config unary_cpu_compare_config = {
    /* .enabled = */ मिच्छा,
    /* .use_cpu_result = */ मिच्छा,
    /* .log_differences = */ सच्च,
    /* .tolerance = */ 1e-4f,
    /* .max_log_elements = */ 4096
};

स्थिर ggml_et_cpu_compare_config sum_rows_cpu_compare_config = {
    /* .enabled = */ मिच्छा,
    /* .use_cpu_result = */ मिच्छा,
    /* .log_differences = */ सच्च,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

स्थिर ggml_et_cpu_compare_config clamp_cpu_compare_config = {
    /* .enabled = */ मिच्छा,
    /* .use_cpu_result = */ मिच्छा,
    /* .log_differences = */ सच्च,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

स्थिर ggml_et_cpu_compare_config mean_cpu_compare_config = {
    /* .enabled = */ मिच्छा,
    /* .use_cpu_result = */ मिच्छा,
    /* .log_differences = */ सच्च,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

स्थिर ggml_et_cpu_compare_config sqr_cpu_compare_config = {
    /* .enabled = */ मिच्छा,
    /* .use_cpu_result = */ मिच्छा,
    /* .log_differences = */ सच्च,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

स्थिर ggml_et_cpu_compare_config elmap_cpu_compare_config = {
    /* .enabled = */ मिच्छा,
    /* .use_cpu_result = */ मिच्छा,
    /* .log_differences = */ सच्च,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

स्थिर ggml_et_cpu_compare_config glu_cpu_compare_config = {
    /* .enabled = */ मिच्छा,
    /* .use_cpu_result = */ मिच्छा,
    /* .log_differences = */ सच्च,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

स्थिर ggml_et_cpu_compare_config mul_mat_cpu_compare_config = {
    /* .enabled = */ मिच्छा,
    /* .use_cpu_result = */ मिच्छा,
    /* .log_differences = */ सच्च,
    /* .tolerance = */ 0.01,
    /* .max_log_elements = */ 4096
};

स्थिर ggml_et_cpu_compare_config mul_mat_id_cpu_compare_config = {
    /* .enabled = */ मिच्छा,
    /* .use_cpu_result = */ मिच्छा,
    /* .log_differences = */ सच्च,
    /* .tolerance = */ 0.01,
    /* .max_log_elements = */ 4096
};

स्थिर ggml_et_cpu_compare_config softmax_cpu_compare_config = {
    /* .enabled = */ मिच्छा,
    /* .use_cpu_result = */ मिच्छा,
    /* .log_differences = */ सच्च,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 1024
};

स्थिर ggml_et_cpu_compare_config get_rows_cpu_compare_config = {
    /* .enabled = */ मिच्छा,
    /* .use_cpu_result = */ मिच्छा,
    /* .log_differences = */ सच्च,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 2048
};

स्थिर ggml_et_cpu_compare_config pad_cpu_compare_config = {
    /* .enabled = */ मिच्छा,
    /* .use_cpu_result = */ मिच्छा,
    /* .log_differences = */ सच्च,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

स्थिर ggml_et_cpu_compare_config cont_cpu_compare_config = {
    /* .enabled = */ मिच्छा,
    /* .use_cpu_result = */ मिच्छा,
    /* .log_differences = */ सच्च,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

स्थिर ggml_et_cpu_compare_config concat_cpu_compare_config = {
    /* .enabled = */ मिच्छा,
    /* .use_cpu_result = */ मिच्छा,
    /* .log_differences = */ सच्च,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

स्थिर ggml_et_cpu_compare_config cumsum_cpu_compare_config = {
    /* .enabled = */ मिच्छा,
    /* .use_cpu_result = */ मिच्छा,
    /* .log_differences = */ सच्च,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

स्थिर ggml_et_cpu_compare_config repeat_cpu_compare_config = {
    /* .enabled = */ मिच्छा,
    /* .use_cpu_result = */ मिच्छा,
    /* .log_differences = */ सच्च,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

स्थिर ggml_et_cpu_compare_config ssm_conv_cpu_compare_config = {
    /* .enabled = */ मिच्छा,
    /* .use_cpu_result = */ मिच्छा,
    /* .log_differences = */ सच्च,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

स्थिर ggml_et_cpu_compare_config rwkv_wkv6_cpu_compare_config = {
    /* .enabled = */ मिच्छा,
    /* .use_cpu_result = */ मिच्छा,
    /* .log_differences = */ सच्च,
    /* .tolerance = */ 1e-4f,
    /* .max_log_elements = */ 4096
};

स्थिर ggml_et_cpu_compare_config rwkv_wkv7_cpu_compare_config = {
    /* .enabled = */ मिच्छा,
    /* .use_cpu_result = */ मिच्छा,
    /* .log_differences = */ सच्च,
    /* .tolerance = */ 1e-4f,
    /* .max_log_elements = */ 4096
};

स्थिर ggml_et_cpu_compare_config set_rows_cpu_compare_config = {
    /* .enabled = */ मिच्छा,
    /* .use_cpu_result = */ मिच्छा,
    /* .log_differences = */ सच्च,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 2048
};

तार्किक ggml_et_op_rms_norm_mul(ggml_backend_et_device_context * dev_ctx,
                             ध्रुव ggml_tensor *              rms_norm_ग्रन्थिः,
                             ध्रुव ggml_tensor *              mul_ग्रन्थिः) {
    ET_PERF_START();

    सचे (!dev_ctx || !rms_norm_ग्रन्थिः || !mul_ग्रन्थिः) {
        GGML_LOG_ERROR("ET: Invalid parameters for fused RMS_NORM_MUL operation\n");
        पटिनिवत्त मिच्छा;
    }

    सचे (!rms_norm_ग्रन्थिः->src[0]) {
        GGML_LOG_ERROR("ET: Fused RMS_NORM_MUL missing required input\n");
        पटिनिवत्त मिच्छा;
    }

    // Extract weights: the MUL operand that isn't the rms_norm output
    ध्रुव ggml_tensor * weights = (mul_ग्रन्थिः->src[0] == rms_norm_ग्रन्थिः) ? mul_ग्रन्थिः->src[1] : mul_ग्रन्थिः->src[0];

    सचे (!weights) {
        GGML_LOG_ERROR("ET: Fused RMS_NORM_MUL missing weights tensor\n");
        पटिनिवत्त मिच्छा;
    }

    भिन्न eps;
    memcpy(&eps, rms_norm_ग्रन्थिः->op_params, परिमाण(भिन्न));

    ggml_et_rms_norm_mul_params params;
    params.src0 = *rms_norm_ग्रन्थिः->src[0];  // input to normalize
    params.src1 = *weights;                // normalization weights
    params.dst  = *mul_ग्रन्थिः;               // final output
    params.eps  = eps;

    तार्किक kernel_result = ggml_et_launch_kernel(dev_ctx, "rms_norm_mul_f32", &params, परिमाण(params), 0xFFFFFFFF);

    ET_PERF_END_EXT("RMS_NORM_MUL", "rms_norm_mul_f32", mul_ग्रन्थिः, "eps=%.6f", (द्विगुण) eps);
    पटिनिवत्त kernel_result;
}

तार्किक ggml_et_op_scale(ggml_backend_et_device_context * dev_ctx, ध्रुव ggml_tensor * ग्रन्थिः) {
    ET_PERF_START();

    सचे (!dev_ctx || !ग्रन्थिः) {
        GGML_LOG_ERROR("ET: Invalid parameters for SCALE operation\n");
        पटिनिवत्त मिच्छा;
    }

    सचे (!ग्रन्थिः->src[0]) {
        GGML_LOG_ERROR("ET: SCALE operation missing required input\n");
        पटिनिवत्त मिच्छा;
    }

    सचे (ग्रन्थिः->type != GGML_TYPE_F32 || ग्रन्थिः->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: SCALE operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(ग्रन्थिः->type),
                       ggml_type_name(ग्रन्थिः->src[0]->type));
        पटिनिवत्त मिच्छा;
    }

    भिन्न scale, bias;
    memcpy(&scale, (ध्रुव भिन्न *) ग्रन्थिः->op_params + 0, परिमाण(भिन्न));
    memcpy(&bias, (ध्रुव भिन्न *) ग्रन्थिः->op_params + 1, परिमाण(भिन्न));

    ggml_et_scale_params params;
    params.src0  = *ग्रन्थिः->src[0];
    params.dst   = *ग्रन्थिः;
    params.scale = scale;
    params.bias  = bias;

    तार्किक kernel_result = ggml_et_launch_kernel(dev_ctx, "scale_f32", &params, परिमाण(params), 0xFFFFFFFF);

    ET_PERF_END_EXT("SCALE", "scale_f32", ग्रन्थिः, "scale=%.6f|bias=%.6f", (द्विगुण) scale, (द्विगुण) bias);
    पटिनिवत्त kernel_result;
}

तार्किक ggml_et_op_sqr(ggml_backend_et_device_context * dev_ctx, ध्रुव ggml_tensor * ग्रन्थिः) {
    ET_PERF_START();

    सचे (!dev_ctx || !ग्रन्थिः) {
        GGML_LOG_ERROR("ET: Invalid parameters for SQR operation\n");
        पटिनिवत्त मिच्छा;
    }

    सचे (!ग्रन्थिः->src[0]) {
        GGML_LOG_ERROR("ET: SQR operation missing required input\n");
        पटिनिवत्त मिच्छा;
    }

    सचे (ग्रन्थिः->type != GGML_TYPE_F32 || ग्रन्थिः->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: SQR operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(ग्रन्थिः->type),
                       ggml_type_name(ग्रन्थिः->src[0]->type));
        पटिनिवत्त मिच्छा;
    }

    ggml_et_sqr_params params;
    params.src0 = *ग्रन्थिः->src[0];  // F32 input tensor
    params.dst  = *ग्रन्थिः;          // F32 output tensor

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    तार्किक                    cpu_comparison_active = मिच्छा;
    सचे (sqr_cpu_compare_config.enabled) {
        सचे (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ग्रन्थिः, GGML_OP_SQR)) {
            cpu_comparison_active = सच्च;
        } अञ्ञथा {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for SQR operation\n");
        }
    }

    तार्किक kernel_result = ggml_et_launch_kernel(dev_ctx, "sqr_f32", &params, परिमाण(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    सचे (cpu_comparison_active) {
        सचे (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ग्रन्थिः, &sqr_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for SQR operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("SQR", "sqr_f32", ग्रन्थिः);
    पटिनिवत्त kernel_result;
}

तार्किक ggml_et_op_sum_rows(ggml_backend_et_device_context * dev_ctx, ध्रुव ggml_tensor * ग्रन्थिः) {
    ET_PERF_START();

    सचे (!dev_ctx || !ग्रन्थिः) {
        GGML_LOG_ERROR("ET: Invalid parameters for SUM_ROWS operation\n");
        पटिनिवत्त मिच्छा;
    }

    सचे (!ग्रन्थिः->src[0]) {
        GGML_LOG_ERROR("ET: SUM_ROWS operation missing required input\n");
        पटिनिवत्त मिच्छा;
    }

    सचे (ग्रन्थिः->type != GGML_TYPE_F32 || ग्रन्थिः->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: SUM_ROWS operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(ग्रन्थिः->type),
                       ggml_type_name(ग्रन्थिः->src[0]->type));
        पटिनिवत्त मिच्छा;
    }

    ggml_et_sum_rows_params params;
    params.src0 = *ग्रन्थिः->src[0];
    params.dst  = *ग्रन्थिः;

    // Phase 1: Initialize CPU comparison context
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    तार्किक                    cpu_comparison_active = मिच्छा;
    सचे (sum_rows_cpu_compare_config.enabled) {
        सचे (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ग्रन्थिः, GGML_OP_SUM_ROWS)) {
            cpu_comparison_active = सच्च;
        } अञ्ञथा {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for SUM_ROWS operation\n");
        }
    }

    तार्किक kernel_result = ggml_et_launch_kernel(dev_ctx, "sum_rows_f32", &params, परिमाण(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare
    सचे (cpu_comparison_active) {
        सचे (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ग्रन्थिः, &sum_rows_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for SUM_ROWS operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("SUM_ROWS", "sum_rows_f32", ग्रन्थिः);
    पटिनिवत्त kernel_result;
}

तार्किक ggml_et_op_mean(ggml_backend_et_device_context * dev_ctx, ध्रुव ggml_tensor * ग्रन्थिः) {
    ET_PERF_START();

    सचे (!dev_ctx || !ग्रन्थिः) {
        GGML_LOG_ERROR("ET: Invalid parameters for MEAN operation\n");
        पटिनिवत्त मिच्छा;
    }

    सचे (!ग्रन्थिः->src[0]) {
        GGML_LOG_ERROR("ET: MEAN operation missing required input\n");
        पटिनिवत्त मिच्छा;
    }

    सचे (ग्रन्थिः->type != GGML_TYPE_F32 || ग्रन्थिः->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: MEAN operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(ग्रन्थिः->type),
                       ggml_type_name(ग्रन्थिः->src[0]->type));
        पटिनिवत्त मिच्छा;
    }

    ggml_et_mean_params params;
    params.src0 = *ग्रन्थिः->src[0];
    params.dst  = *ग्रन्थिः;

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    तार्किक                    cpu_comparison_active = मिच्छा;
    सचे (mean_cpu_compare_config.enabled) {
        सचे (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ग्रन्थिः, GGML_OP_MEAN)) {
            cpu_comparison_active = सच्च;
        } अञ्ञथा {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for MEAN operation\n");
        }
    }

    तार्किक kernel_result = ggml_et_launch_kernel(dev_ctx, "mean_f32", &params, परिमाण(params), 0xFFFFFFFF);

    सचे (cpu_comparison_active) {
        सचे (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ग्रन्थिः, &mean_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for MEAN operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("MEAN", "mean_f32", ग्रन्थिः);
    पटिनिवत्त kernel_result;
}

तार्किक ggml_et_op_clamp(ggml_backend_et_device_context * dev_ctx, ध्रुव ggml_tensor * ग्रन्थिः) {
    ET_PERF_START();

    सचे (!dev_ctx || !ग्रन्थिः) {
        GGML_LOG_ERROR("ET: Invalid parameters for CLAMP operation\n");
        पटिनिवत्त मिच्छा;
    }

    सचे (!ग्रन्थिः->src[0]) {
        GGML_LOG_ERROR("ET: CLAMP operation missing required input\n");
        पटिनिवत्त मिच्छा;
    }

    सचे (ग्रन्थिः->type != GGML_TYPE_F32 || ग्रन्थिः->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: CLAMP operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(ग्रन्थिः->type),
                       ggml_type_name(ग्रन्थिः->src[0]->type));
        पटिनिवत्त मिच्छा;
    }

    ggml_et_clamp_params params;
    params.src0 = *ग्रन्थिः->src[0];
    params.dst  = *ग्रन्थिः;
    // op_params layout per ggml.c::ggml_clamp: { min, max } as floats
    memcpy(&params.min_val, (ध्रुव भिन्न *) ग्रन्थिः->op_params + 0, परिमाण(भिन्न));
    memcpy(&params.max_val, (ध्रुव भिन्न *) ग्रन्थिः->op_params + 1, परिमाण(भिन्न));

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    तार्किक                    cpu_comparison_active = मिच्छा;
    सचे (clamp_cpu_compare_config.enabled) {
        सचे (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ग्रन्थिः, GGML_OP_CLAMP)) {
            cpu_comparison_active = सच्च;
        } अञ्ञथा {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for CLAMP operation\n");
        }
    }

    तार्किक kernel_result = ggml_et_launch_kernel(dev_ctx, "clamp_f32", &params, परिमाण(params), 0xFFFFFFFF);

    सचे (cpu_comparison_active) {
        सचे (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ग्रन्थिः, &clamp_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for CLAMP operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("CLAMP", "clamp_f32", ग्रन्थिः);
    पटिनिवत्त kernel_result;
}

तार्किक ggml_et_op_unary(ggml_backend_et_device_context * dev_ctx, ध्रुव ggml_tensor * ग्रन्थिः) {
    ET_PERF_START();

    सचे (!dev_ctx || !ग्रन्थिः) {
        GGML_LOG_ERROR("ET: Invalid parameters for UNARY operation\n");
        पटिनिवत्त मिच्छा;
    }

    सचे (!ग्रन्थिः->src[0]) {
        GGML_LOG_ERROR("ET: UNARY operation missing required input\n");
        पटिनिवत्त मिच्छा;
    }

    सचे (ग्रन्थिः->type != GGML_TYPE_F32 || ग्रन्थिः->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: UNARY operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(ग्रन्थिः->type),
                       ggml_type_name(ग्रन्थिः->src[0]->type));
        पटिनिवत्त मिच्छा;
    }

    ध्रुव ggml_unary_op uop     = ggml_get_unary_op(ग्रन्थिः);
    ध्रुव वर्ण *        op_name = ggml_unary_op_name(uop);

    ggml_et_unary_params params;
    params.src0     = *ग्रन्थिः->src[0];  // F32 input tensor
    params.dst      = *ग्रन्थिः;          // F32 output tensor
    params.unary_op = (int32_t) uop;

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    तार्किक                    cpu_comparison_active = मिच्छा;
    सचे (unary_cpu_compare_config.enabled) {
        सचे (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ग्रन्थिः, GGML_OP_UNARY)) {
            cpu_comparison_active = सच्च;
        } अञ्ञथा {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for UNARY/%s operation\n", op_name);
        }
    }

    तार्किक kernel_result = ggml_et_launch_kernel(dev_ctx, "unary_f32", &params, परिमाण(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    सचे (cpu_comparison_active) {
        सचे (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ग्रन्थिः, &unary_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for UNARY/%s operation\n", op_name);
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END_EXT("UNARY", "unary_f32", ग्रन्थिः, "op=%s", op_name);
    पटिनिवत्त kernel_result;
}

तार्किक ggml_et_op_mul(ggml_backend_et_device_context * dev_ctx, ध्रुव ggml_tensor * ग्रन्थिः) {
    // Delegate to generic element map operation
    पटिनिवत्त ggml_et_op_elmap(dev_ctx, ग्रन्थिः);
}

तार्किक ggml_et_op_add(ggml_backend_et_device_context * dev_ctx, ध्रुव ggml_tensor * ग्रन्थिः) {
    // Delegate to generic element map operation
    पटिनिवत्त ggml_et_op_elmap(dev_ctx, ग्रन्थिः);
}

तार्किक ggml_et_op_sub(ggml_backend_et_device_context * dev_ctx, ध्रुव ggml_tensor * ग्रन्थिः) {
    // Delegate to generic element map operation
    पटिनिवत्त ggml_et_op_elmap(dev_ctx, ग्रन्थिः);
}

तार्किक ggml_et_op_elmap(ggml_backend_et_device_context * dev_ctx, ध्रुव ggml_tensor * ग्रन्थिः) {
    ET_PERF_START();

    सचे (!dev_ctx || !ग्रन्थिः) {
        GGML_LOG_ERROR("ET: Invalid parameters for element map operation\n");
        पटिनिवत्त मिच्छा;
    }

    सचे (!ग्रन्थिः->src[0] || !ग्रन्थिः->src[1]) {
        GGML_LOG_ERROR("ET: Element map operation missing required inputs\n");
        पटिनिवत्त मिच्छा;
    }

    सचे (ग्रन्थिः->type != GGML_TYPE_F32 || ग्रन्थिः->src[0]->type != GGML_TYPE_F32 || ग्रन्थिः->src[1]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: Element map operation with unsupported types: dst=%s src0=%s src1=%s\n",
                       ggml_type_name(ग्रन्थिः->type), ggml_type_name(ग्रन्थिः->src[0]->type),
                       ggml_type_name(ग्रन्थिः->src[1]->type));
        पटिनिवत्त मिच्छा;
    }

    ध्रुव वर्ण * op_name = ggml_op_name(ग्रन्थिः->op);

    ggml_et_elmap_params params;
    params.src0 = *ग्रन्थिः->src[0];
    params.src1 = *ग्रन्थिः->src[1];
    params.dst  = *ग्रन्थिः;  // F32 output tensor (op type stored in dst.op)

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    तार्किक                    cpu_comparison_active = मिच्छा;
    सचे (elmap_cpu_compare_config.enabled) {
        सचे (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ग्रन्थिः, ग्रन्थिः->op)) {
            cpu_comparison_active = सच्च;
        } अञ्ञथा {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for %s operation\n", op_name);
        }
    }

    // fprintf(stderr, "ET: el_map s0 [%ld, %ld, %ld, %ld] s1 [%ld, %ld, %ld, %ld]\n",
    //     node->src[0]->ne[0], node->src[0]->ne[1], node->src[0]->ne[2], node->src[0]->ne[3],
    //     node->src[1]->ne[0], node->src[1]->ne[1], node->src[1]->ne[2], node->src[1]->ne[3]);

    तार्किक kernel_result = ggml_et_launch_kernel(dev_ctx, "el_map_f32", &params, परिमाण(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    सचे (cpu_comparison_active) {
        सचे (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ग्रन्थिः, &elmap_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for %s operation\n", op_name);
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END(op_name, "el_map_f32", ग्रन्थिः);
    पटिनिवत्त kernel_result;
}

तार्किक ggml_et_op_glu(ggml_backend_et_device_context * dev_ctx, ध्रुव ggml_tensor * ग्रन्थिः) {
    ET_PERF_START();

    // Validate inputs
    सचे (!dev_ctx || !ग्रन्थिः) {
        GGML_LOG_ERROR("ET: Invalid parameters for GLU operation\n");
        पटिनिवत्त मिच्छा;
    }

    सचे (!ग्रन्थिः->src[0]) {
        GGML_LOG_ERROR("ET: GLU operation missing required input\n");
        पटिनिवत्त मिच्छा;
    }

    ध्रुव तार्किक is_split_mode = ग्रन्थिः->src[1] != nullptr;

    // Only support F32 (as validated by supports_op)
    सचे (ग्रन्थिः->type != GGML_TYPE_F32 || ग्रन्थिः->src[0]->type != GGML_TYPE_F32 ||
        (is_split_mode && ग्रन्थिः->src[1]->type != GGML_TYPE_F32)) {
        पटिनिवत्त मिच्छा;
    }

    // Extract GLU operation parameters from op_params
    int32_t glu_op_type = ggml_get_op_params_i32(ग्रन्थिः, 0);  // GLU variant (REGLU, GEGLU, SWIGLU, etc.)
    int32_t swapped     = ggml_get_op_params_i32(ग्रन्थिः, 1);  // Whether gate/value are swapped

    // Supported variants
    विकल्प (glu_op_type) {
        पक्ष GGML_GLU_OP_REGLU:
        पक्ष GGML_GLU_OP_GEGLU:
        पक्ष GGML_GLU_OP_SWIGLU:
        पक्ष GGML_GLU_OP_SWIGLU_OAI:
        पक्ष GGML_GLU_OP_SWIGLU_CLAMP:
        पक्ष GGML_GLU_OP_GEGLU_ERF:
        पक्ष GGML_GLU_OP_GEGLU_QUICK:
            भङ्ग;
        स्वभाव:
            GGML_LOG_ERROR("ET: GLU operation with unsupported variant: %s\n",
                           ggml_glu_op_name((ggml_glu_op) glu_op_type));
            पटिनिवत्त मिच्छा;
    }

    // Get GLU operation name for logging
    ध्रुव वर्ण * glu_op_name = ggml_glu_op_name((ggml_glu_op) glu_op_type);

    // Pack parameters. Single-tensor mode is encoded by zeroing src1.
    ggml_et_glu_params params = {};
    params.src0               = *ग्रन्थिः->src[0];
    सचे (is_split_mode) {
        params.src1 = *ग्रन्थिः->src[1];
    }
    params.dst         = *ग्रन्थिः;
    params.glu_op_type = glu_op_type;
    params.swapped     = swapped;
    params.alpha       = 0.0f;
    params.limit       = 0.0f;
    सचे (glu_op_type == GGML_GLU_OP_SWIGLU_OAI) {
        params.alpha = ggml_get_op_params_f32(ग्रन्थिः, 2);
    }
    सचे (glu_op_type == GGML_GLU_OP_SWIGLU_OAI || glu_op_type == GGML_GLU_OP_SWIGLU_CLAMP) {
        params.limit = ggml_get_op_params_f32(ग्रन्थिः, 3);
    }
    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    तार्किक                    cpu_comparison_active = मिच्छा;
    सचे (glu_cpu_compare_config.enabled) {
        सचे (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ग्रन्थिः, GGML_OP_GLU)) {
            cpu_comparison_active = सच्च;
        } अञ्ञथा {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for %s operation\n", glu_op_name);
        }
    }

    // Launch ET kernel
    तार्किक kernel_result = ggml_et_launch_kernel(dev_ctx, "glu_f32", &params, परिमाण(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    सचे (cpu_comparison_active) {
        सचे (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ग्रन्थिः, &glu_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for %s operation\n", glu_op_name);
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("GLU", "glu_f32", ग्रन्थिः);
    पटिनिवत्त kernel_result;
}

तार्किक ggml_et_op_mul_mat(ggml_backend_et_device_context * dev_ctx,
                        ध्रुव ggml_tensor *              ग्रन्थिः,
                        ध्रुव ggml_tensor *              add_ग्रन्थिः) {
    ET_PERF_START();

    सचे (!dev_ctx || !ग्रन्थिः) {
        GGML_LOG_ERROR("ET: Invalid parameters for MUL_MAT operation\n");
        पटिनिवत्त मिच्छा;
    }

    सचे (!ग्रन्थिः->src[0] || !ग्रन्थिः->src[1]) {
        GGML_LOG_ERROR("ET: MUL_MAT operation missing required inputs\n");
        पटिनिवत्त मिच्छा;
    }

    // Fused MM+ADD: when add_node is non-NULL the caller has already validated
    // (Q8_0 weights, F32 acts, exact-shape ADD with stride parity to dst) via
    // ggml_et_can_fuse({MUL_MAT, ADD}). The kernel writes dst = mm + bias and
    // the ADD's output replaces MM's as the actual dst.
    ध्रुव ggml_tensor * fused_dst   = add_ग्रन्थिः ? add_ग्रन्थिः : ग्रन्थिः;
    ध्रुव ggml_tensor * bias_tensor = nullptr;
    सचे (add_ग्रन्थिः) {
        bias_tensor = (add_ग्रन्थिः->src[0] == ग्रन्थिः) ? add_ग्रन्थिः->src[1] : add_ग्रन्थिः->src[0];
    }

    ध्रुव वर्ण * kernel_name;
    ध्रुव वर्ण * src0_type_name;

    सचे (ग्रन्थिः->type == GGML_TYPE_F32 && ग्रन्थिः->src[0]->type == GGML_TYPE_Q4_0 && ग्रन्थिः->src[1]->type == GGML_TYPE_F32 &&
        ग्रन्थिः->src[1]->ne[1] >= 53 &&      // N >= 53
        ग्रन्थिः->src[0]->ne[1] % 16 == 0 &&  // M % TILE_M
        ग्रन्थिः->src[0]->ne[0] % 32 == 0) {  // K % BLOCK_K (Q4_0 block)

        // Matrix engine for N >= 53; partial N (via n_cur-1) and errata padding are handled in-kernel.
        kernel_name    = "mul_mat_Q4_0_matrix_engine";
        src0_type_name = "Q4_0";

    } अञ्ञथा सचे (ग्रन्थिः->type == GGML_TYPE_F32 && ग्रन्थिः->src[0]->type == GGML_TYPE_Q4_0 &&
               ग्रन्थिः->src[1]->type == GGML_TYPE_F32) {
        kernel_name    = "mul_mat_Q4_0";  // N < 53, or M % 16 != 0 or K % 32 != 0
        src0_type_name = "Q4_0";

    } अञ्ञथा सचे (ग्रन्थिः->type == GGML_TYPE_F32 && ग्रन्थिः->src[0]->type == GGML_TYPE_Q8_0 &&
               ग्रन्थिः->src[1]->type == GGML_TYPE_F32) {
        kernel_name    = "mul_mat_Q8_0";
        src0_type_name = "Q8_0";

    } अञ्ञथा सचे (ग्रन्थिः->type == GGML_TYPE_F32 && ग्रन्थिः->src[0]->type == GGML_TYPE_F16 &&
               ग्रन्थिः->src[1]->type == GGML_TYPE_F16 && ग्रन्थिः->ne[0] % 16 == 0 && ग्रन्थिः->src[0]->ne[0] % 16 == 0 &&
               ग्रन्थिः->src[0]->ne[1] % 16 == 0 && ग्रन्थिः->src[1]->ne[0] != 1) {
        kernel_name    = "mul_mat_f16_matrix_engine";
        src0_type_name = "F16";

    } अञ्ञथा सचे (ग्रन्थिः->type == GGML_TYPE_F32 && ग्रन्थिः->src[0]->type == GGML_TYPE_F16 &&
               (ग्रन्थिः->src[1]->type == GGML_TYPE_F16 || ग्रन्थिः->src[1]->type == GGML_TYPE_F32)) {
        kernel_name    = "mul_mat_f16";
        src0_type_name = "F16";

    } अञ्ञथा सचे (ग्रन्थिः->type == GGML_TYPE_F32 && ग्रन्थिः->src[0]->type == GGML_TYPE_F32 &&
               ग्रन्थिः->src[1]->type == GGML_TYPE_F32 && ग्रन्थिः->ne[0] % 16 == 0 && ग्रन्थिः->src[0]->ne[0] % 16 == 0 &&
               ग्रन्थिः->src[0]->ne[1] % 16 == 0 && ग्रन्थिः->src[1]->ne[0] != 1) {  // GEMV is faster with the generic path

        kernel_name    = "mul_mat_f32_matrix_engine";
        src0_type_name = "F32";
    } अञ्ञथा सचे (ग्रन्थिः->type == GGML_TYPE_F32 && ग्रन्थिः->src[0]->type == GGML_TYPE_F32 &&
               (ग्रन्थिः->src[1]->type == GGML_TYPE_F16 || ग्रन्थिः->src[1]->type == GGML_TYPE_F32)) {
        kernel_name    = "mul_mat_f32";
        src0_type_name = "F32";
    } अञ्ञथा {
        GGML_LOG_ERROR("ET: MUL_MAT operation with unsupported types: dst=%s src0=%s src1=%s\n",
                       ggml_type_name(ग्रन्थिः->type), ggml_type_name(ग्रन्थिः->src[0]->type),
                       ggml_type_name(ग्रन्थिः->src[1]->type));
        पटिनिवत्त मिच्छा;
    }

    ggml_et_द्विमान_params params;
    params.src0 = *ग्रन्थिः->src[0];  // weight matrix
    params.src1 = *ग्रन्थिः->src[1];  // activation matrix
    params.dst  = *fused_dst;     // output (= add_node when fused, else node)

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    तार्किक                    cpu_comparison_active = मिच्छा;
    सचे (mul_mat_cpu_compare_config.enabled) {
        सचे (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, fused_dst, GGML_OP_MUL_MAT)) {
            cpu_comparison_active = सच्च;
        } अञ्ञथा {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for MUL_MAT operation\n");
        }
    }

    तार्किक kernel_result;
    सचे (ग्रन्थिः->src[0]->type == GGML_TYPE_Q8_0) {
        // Q8_0 kernel always takes the extended struct. bias.data is non-NULL
        // only on the fused path; otherwise the kernel skips the add entirely.
        ggml_et_mm_q8_params q8_params = {};
        q8_params.src0                 = params.src0;
        q8_params.src1                 = params.src1;
        q8_params.dst                  = params.dst;
        सचे (bias_tensor) {
            q8_params.bias = *bias_tensor;
        }
        kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &q8_params, परिमाण(q8_params), 0xFFFFFFFF);
    } अञ्ञथा {
        // Non-Q8 MM kernels don't yet support fused-add; the graph fuse check
        // already rejects non-Q8 pairs, so add_node is always nullptr here.
        kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, परिमाण(params), 0xFFFFFFFF);
    }

    // printf("Tensor error:");
    // if (params.src0.data != NULL)
    // {
    //     printf("Ptr OK\n");
    //     printf("node->data ptr = %p\n", node->data);
    //     // if (once < 100){
    //     //     // uint64_t * host_data = (uint64_t *) node->data;
    //     //     // printf("Tensor error: %lu\n", host_data[0]);

    //     //     // printf("Tensor error:");
    //     //     once++;
    //     // }
    // }

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    सचे (cpu_comparison_active) {
        सचे (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, fused_dst, &mul_mat_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for MUL_MAT operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    {
        // Calculate actual FLOPs including batch/sequence dimensions
        // dst shape: [M, N, ne2, ne3] where M=ne[1], N=ne[0]
        int64_t m   = ग्रन्थिः->ne[1];
        int64_t n   = ग्रन्थिः->ne[0];
        int64_t k   = ग्रन्थिः->src[0]->ne[0];
        int64_t ne2 = ग्रन्थिः->ne[2];
        int64_t ne3 = ग्रन्थिः->ne[3];

        // Total FLOPs = (batch_size) * M * N * (2*K - 1)
        // Each MxN matrix-matrix multiply does M*N*(2*K-1) FLOPs
        // Broadcasting is handled by repeating computation, so count actual operations
        int64_t batch_size  = ne2 * ne3;
        int64_t total_flops = batch_size * m * n * (2 * k - 1);

        वर्ण kernel_variant[64];
        snprintf(kernel_variant, परिमाण(kernel_variant), "%s_%sx%s", kernel_name, src0_type_name,
                 ggml_type_name(ग्रन्थिः->src[1]->type));
        ET_PERF_END_EXT("MUL_MAT", kernel_variant, ग्रन्थिः, "flops=%" PRId64, total_flops);
    }
    पटिनिवत्त kernel_result;
}

तार्किक ggml_et_op_mul_mat_id(ggml_backend_et_device_context * dev_ctx, ध्रुव ggml_tensor * ग्रन्थिः) {
    ET_PERF_START();
    सचे (!dev_ctx || !ग्रन्थिः) {
        GGML_LOG_ERROR("ET: Invalid parameters for MUL_MAT_ID operation\n");
        पटिनिवत्त मिच्छा;
    }

    सचे (!ग्रन्थिः->src[0] || !ग्रन्थिः->src[1] || !ग्रन्थिः->src[2]) {
        GGML_LOG_ERROR("ET: MUL_MAT_ID operation missing required inputs\n");
        पटिनिवत्त मिच्छा;
    }

    ध्रुव वर्ण * kernel_name;
    ध्रुव वर्ण * src0_type_name;

    // Support Q8_0/Q4_0/F16/F32 x F32 -> F32 matrix multiplication with expert selection
    सचे (ग्रन्थिः->type == GGML_TYPE_F32 && ग्रन्थिः->src[0]->type == GGML_TYPE_Q8_0 && ग्रन्थिः->src[1]->type == GGML_TYPE_F32 &&
        ग्रन्थिः->src[2]->type == GGML_TYPE_I32) {
        kernel_name    = "mul_mat_id_Q8_0";
        src0_type_name = "Q8_0";

    } अञ्ञथा सचे (ग्रन्थिः->type == GGML_TYPE_F32 && ग्रन्थिः->src[0]->type == GGML_TYPE_Q4_0 &&
               ग्रन्थिः->src[1]->type == GGML_TYPE_F32 && ग्रन्थिः->src[2]->type == GGML_TYPE_I32) {
        kernel_name    = "mul_mat_id_Q4_0";
        src0_type_name = "Q4_0";

    } अञ्ञथा सचे (ग्रन्थिः->type == GGML_TYPE_F32 && ग्रन्थिः->src[0]->type == GGML_TYPE_F16 &&
               ग्रन्थिः->src[1]->type == GGML_TYPE_F32 && ग्रन्थिः->src[2]->type == GGML_TYPE_I32) {
        kernel_name    = "mul_mat_id_f32";
        src0_type_name = "F16";

    } अञ्ञथा सचे (ग्रन्थिः->type == GGML_TYPE_F32 && ग्रन्थिः->src[0]->type == GGML_TYPE_F32 &&
               ग्रन्थिः->src[1]->type == GGML_TYPE_F32 && ग्रन्थिः->src[2]->type == GGML_TYPE_I32) {
        kernel_name    = "mul_mat_id_f32";
        src0_type_name = "F32";

    } अञ्ञथा {
        GGML_LOG_ERROR("ET: MUL_MAT_ID operation with unsupported types: dst=%s src0=%s src1=%s src2=%s\n",
                       ggml_type_name(ग्रन्थिः->type), ggml_type_name(ग्रन्थिः->src[0]->type),
                       ggml_type_name(ग्रन्थिः->src[1]->type), ggml_type_name(ग्रन्थिः->src[2]->type));
        पटिनिवत्त मिच्छा;
    }

    // Pack parameters - copy full tensor structures
    ggml_et_mul_mat_id_params params;
    params.src0 = *ग्रन्थिः->src[0];  // Expert weight matrices (Q8_0/F16/F32)
    params.src1 = *ग्रन्थिः->src[1];  // Activation matrix (F32)
    params.src2 = *ग्रन्थिः->src[2];  // Expert indices (I32)
    params.dst  = *ग्रन्थिः;          // Output matrix (F32)

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    तार्किक                    cpu_comparison_active = मिच्छा;
    सचे (mul_mat_id_cpu_compare_config.enabled) {
        सचे (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ग्रन्थिः, GGML_OP_MUL_MAT_ID)) {
            cpu_comparison_active = सच्च;
        } अञ्ञथा {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for MUL_MAT_ID operation\n");
        }
    }

    // Launch ET kernel
    तार्किक kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, परिमाण(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    सचे (cpu_comparison_active) {
        सचे (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ग्रन्थिः, &mul_mat_id_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for MUL_MAT_ID operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    // Calculate FLOPs (approximate - similar to MUL_MAT but with expert routing overhead)
    // Each expert computation is similar to a MUL_MAT, but we only compute for selected experts
    int64_t K             = ग्रन्थिः->src[0]->ne[0];
    int64_t M             = ग्रन्थिः->src[0]->ne[1];
    int64_t n_expert_used = ग्रन्थिः->src[2]->ne[0];
    int64_t batch         = ग्रन्थिः->src[2]->ne[1];

    int64_t total_flops = batch * n_expert_used * M * (2 * K - 1);

    वर्ण kernel_variant[64];
    snprintf(kernel_variant, परिमाण(kernel_variant), "%s_%sx%s", kernel_name, src0_type_name,
             ggml_type_name(ग्रन्थिः->src[1]->type));
    ET_PERF_END_EXT("MUL_MAT_ID", kernel_variant, ग्रन्थिः, "flops=%" PRId64 "|n_expert=%lld|n_expert_used=%lld",
                    total_flops, (long long) ग्रन्थिः->src[0]->ne[2], (long long) n_expert_used);

    पटिनिवत्त kernel_result;
}

तार्किक ggml_et_op_rope(ggml_backend_et_device_context * dev_ctx, ध्रुव ggml_tensor * ग्रन्थिः) {
    ET_PERF_START();

    सचे (!dev_ctx || !ग्रन्थिः) {
        GGML_LOG_ERROR("ET: Invalid parameters for ROPE operation\n");
        पटिनिवत्त मिच्छा;
    }

    सचे (!ग्रन्थिः->src[0] || !ग्रन्थिः->src[1]) {
        GGML_LOG_ERROR("ET: ROPE operation missing required inputs\n");
        पटिनिवत्त मिच्छा;
    }

    ध्रुव वर्ण * kernel_name;

    सचे (ग्रन्थिः->type == GGML_TYPE_F32 && ग्रन्थिः->src[0]->type == GGML_TYPE_F32 && ग्रन्थिः->src[1]->type == GGML_TYPE_I32) {
        kernel_name = "rope_f32";
    } अञ्ञथा {
        पटिनिवत्त मिच्छा;
    }

    // Pack parameters - copy full tensor structures and op_params
    ggml_et_rope_params params;
    params.src0 = *ग्रन्थिः->src[0];                       // F32 input tensor
    params.src1 = *ग्रन्थिः->src[1];                       // I32 position tensor
    सचे (ग्रन्थिः->src[2]) {
        params.src2 = *ग्रन्थिः->src[2];                   // F32 frequency factors (optional)
    } अञ्ञथा {
        memset(&params.src2, 0, परिमाण(params.src2));  // Zero if not provided
    }
    params.dst = *ग्रन्थिः;                                // F32 output tensor

    params.rope_params.n_past     = ((ध्रुव int32_t *) ग्रन्थिः->op_params)[0];
    params.rope_params.n_dims     = ((ध्रुव int32_t *) ग्रन्थिः->op_params)[1];
    params.rope_params.mode       = ((ध्रुव int32_t *) ग्रन्थिः->op_params)[2];
    params.rope_params.n_ctx      = ((ध्रुव int32_t *) ग्रन्थिः->op_params)[3];
    params.rope_params.n_ctx_orig = ((ध्रुव int32_t *) ग्रन्थिः->op_params)[4];
    memcpy(&params.rope_params.freq_base, (ध्रुव int32_t *) ग्रन्थिः->op_params + 5, परिमाण(भिन्न));
    memcpy(&params.rope_params.freq_scale, (ध्रुव int32_t *) ग्रन्थिः->op_params + 6, परिमाण(भिन्न));
    memcpy(&params.rope_params.ext_factor, (ध्रुव int32_t *) ग्रन्थिः->op_params + 7, परिमाण(भिन्न));
    memcpy(&params.rope_params.attn_factor, (ध्रुव int32_t *) ग्रन्थिः->op_params + 8, परिमाण(भिन्न));
    memcpy(&params.rope_params.beta_fast, (ध्रुव int32_t *) ग्रन्थिः->op_params + 9, परिमाण(भिन्न));
    memcpy(&params.rope_params.beta_slow, (ध्रुव int32_t *) ग्रन्थिः->op_params + 10, परिमाण(भिन्न));
    सचे (params.rope_params.mode & GGML_ROPE_TYPE_MROPE) {
        memcpy(params.rope_params.sections, (ध्रुव int32_t *) ग्रन्थिः->op_params + 11, परिमाण(int32_t) * 4);
    } अञ्ञथा {
        memset(params.rope_params.sections, 0, परिमाण(params.rope_params.sections));
    }

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    तार्किक                    cpu_comparison_active = मिच्छा;
    सचे (rope_cpu_compare_config.enabled) {
        GGML_LOG_DEBUG("ET: Initializing CPU comparison for ROPE operation\n");
        सचे (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ग्रन्थिः, GGML_OP_ROPE)) {
            cpu_comparison_active = सच्च;
        } अञ्ञथा {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for ROPE operation\n");
        }
    }

    तार्किक kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, परिमाण(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    सचे (cpu_comparison_active) {
        सचे (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ग्रन्थिः, &rope_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for ROPE operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END_EXT("ROPE", kernel_name, ग्रन्थिः, "mode=0x%x|n_dims=%d|freq_base=%.2f|freq_scale=%.2f",
                    params.rope_params.mode, params.rope_params.n_dims, (द्विगुण) params.rope_params.freq_base,
                    (द्विगुण) params.rope_params.freq_scale);
    पटिनिवत्त kernel_result;
}

तार्किक ggml_et_op_rms_norm(ggml_backend_et_device_context * dev_ctx, ध्रुव ggml_tensor * ग्रन्थिः) {
    ET_PERF_START();

    सचे (!dev_ctx || !ग्रन्थिः) {
        GGML_LOG_ERROR("ET: Invalid parameters for RMS_NORM operation\n");
        पटिनिवत्त मिच्छा;
    }

    सचे (!ग्रन्थिः->src[0]) {
        GGML_LOG_ERROR("ET: RMS_NORM operation missing required input\n");
        पटिनिवत्त मिच्छा;
    }

    ध्रुव वर्ण * kernel_name;

    सचे (ग्रन्थिः->type == GGML_TYPE_F32 && ग्रन्थिः->src[0]->type == GGML_TYPE_F32) {
        kernel_name = "rms_norm_f32";

    } अञ्ञथा {
        GGML_LOG_ERROR("ET: RMS_NORM operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(ग्रन्थिः->type),
                       ggml_type_name(ग्रन्थिः->src[0]->type));
        पटिनिवत्त मिच्छा;
    }

    भिन्न eps;
    memcpy(&eps, ग्रन्थिः->op_params, परिमाण(भिन्न));

    ggml_et_rms_norm_params params;
    params.src0 = *ग्रन्थिः->src[0];  // F32 input tensor
    params.dst  = *ग्रन्थिः;          // F32 output tensor
    params.eps  = eps;            // Epsilon parameter for numerical stability

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    तार्किक                    cpu_comparison_active = मिच्छा;
    सचे (rms_norm_cpu_compare_config.enabled) {
        सचे (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ग्रन्थिः, GGML_OP_RMS_NORM)) {
            cpu_comparison_active = सच्च;
        } अञ्ञथा {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for RMS_NORM operation\n");
        }
    }

    तार्किक kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, परिमाण(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    सचे (cpu_comparison_active) {
        सचे (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ग्रन्थिः, &rms_norm_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for RMS_NORM operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END_EXT("RMS_NORM", kernel_name, ग्रन्थिः, "eps=%.6f", (द्विगुण) eps);
    पटिनिवत्त kernel_result;
}

तार्किक ggml_et_op_norm(ggml_backend_et_device_context * dev_ctx, ध्रुव ggml_tensor * ग्रन्थिः) {
    ET_PERF_START();

    सचे (!dev_ctx || !ग्रन्थिः) {
        GGML_LOG_ERROR("ET: Invalid parameters for NORM operation\n");
        पटिनिवत्त मिच्छा;
    }

    सचे (!ग्रन्थिः->src[0]) {
        GGML_LOG_ERROR("ET: NORM operation missing required input\n");
        पटिनिवत्त मिच्छा;
    }

    ध्रुव वर्ण * kernel_name;

    सचे (ग्रन्थिः->type == GGML_TYPE_F32 && ग्रन्थिः->src[0]->type == GGML_TYPE_F32) {
        kernel_name = "norm_f32";

    } अञ्ञथा {
        GGML_LOG_ERROR("ET: NORM operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(ग्रन्थिः->type),
                       ggml_type_name(ग्रन्थिः->src[0]->type));
        पटिनिवत्त मिच्छा;
    }

    भिन्न eps;
    memcpy(&eps, ग्रन्थिः->op_params, परिमाण(भिन्न));

    ggml_et_norm_params params;
    params.src0 = *ग्रन्थिः->src[0];  // F32 input tensor
    params.dst  = *ग्रन्थिः;          // F32 output tensor
    params.eps  = eps;            // Epsilon parameter for numerical stability

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    तार्किक                    cpu_comparison_active = मिच्छा;
    सचे (norm_cpu_compare_config.enabled) {
        सचे (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ग्रन्थिः, GGML_OP_NORM)) {
            cpu_comparison_active = सच्च;
        } अञ्ञथा {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for NORM operation\n");
        }
    }

    तार्किक kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, परिमाण(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    सचे (cpu_comparison_active) {
        सचे (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ग्रन्थिः, &norm_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for NORM operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END_EXT("NORM", kernel_name, ग्रन्थिः, "eps=%.6f", (द्विगुण) eps);
    पटिनिवत्त kernel_result;
}

तार्किक ggml_et_op_l2_norm(ggml_backend_et_device_context * dev_ctx, ध्रुव ggml_tensor * ग्रन्थिः) {
    ET_PERF_START();

    सचे (!dev_ctx || !ग्रन्थिः) {
        GGML_LOG_ERROR("ET: Invalid parameters for L2_NORM operation\n");
        पटिनिवत्त मिच्छा;
    }

    सचे (!ग्रन्थिः->src[0]) {
        GGML_LOG_ERROR("ET: L2_NORM operation missing required input\n");
        पटिनिवत्त मिच्छा;
    }

    ध्रुव वर्ण * kernel_name;

    सचे (ग्रन्थिः->type == GGML_TYPE_F32 && ग्रन्थिः->src[0]->type == GGML_TYPE_F32) {
        kernel_name = "l2_norm_f32";

    } अञ्ञथा {
        GGML_LOG_ERROR("ET: L2_NORM operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(ग्रन्थिः->type),
                       ggml_type_name(ग्रन्थिः->src[0]->type));
        पटिनिवत्त मिच्छा;
    }

    भिन्न eps;
    memcpy(&eps, ग्रन्थिः->op_params, परिमाण(भिन्न));

    ggml_et_l2_norm_params params;
    params.src0 = *ग्रन्थिः->src[0];  // F32 input tensor
    params.dst  = *ग्रन्थिः;          // F32 output tensor
    params.eps  = eps;            // Epsilon parameter for numerical stability

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    तार्किक                    cpu_comparison_active = मिच्छा;
    सचे (l2_norm_cpu_compare_config.enabled) {
        सचे (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ग्रन्थिः, GGML_OP_L2_NORM)) {
            cpu_comparison_active = सच्च;
        } अञ्ञथा {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for L2_NORM operation\n");
        }
    }

    तार्किक kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, परिमाण(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    सचे (cpu_comparison_active) {
        सचे (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ग्रन्थिः, &l2_norm_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for L2_NORM operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END_EXT("L2_NORM", kernel_name, ग्रन्थिः, "eps=%.6f", (द्विगुण) eps);
    पटिनिवत्त kernel_result;
}

तार्किक ggml_et_op_group_norm(ggml_backend_et_device_context * dev_ctx, ध्रुव ggml_tensor * ग्रन्थिः) {
    ET_PERF_START();

    सचे (!dev_ctx || !ग्रन्थिः) {
        GGML_LOG_ERROR("ET: Invalid parameters for GROUP_NORM operation\n");
        पटिनिवत्त मिच्छा;
    }

    सचे (!ग्रन्थिः->src[0]) {
        GGML_LOG_ERROR("ET: GROUP_NORM operation missing required input\n");
        पटिनिवत्त मिच्छा;
    }

    सचे (ग्रन्थिः->type != GGML_TYPE_F32 || ग्रन्थिः->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: GROUP_NORM operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(ग्रन्थिः->type),
                       ggml_type_name(ग्रन्थिः->src[0]->type));
        पटिनिवत्त मिच्छा;
    }

    ध्रुव int32_t n_groups = ggml_get_op_params_i32(ग्रन्थिः, 0);
    भिन्न         eps;
    memcpy(&eps, (ध्रुव भिन्न *) ग्रन्थिः->op_params + 1, परिमाण(भिन्न));

    ggml_et_group_norm_params params;
    params.src0     = *ग्रन्थिः->src[0];
    params.dst      = *ग्रन्थिः;
    params.n_groups = n_groups;
    params.eps      = eps;

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    तार्किक                    cpu_comparison_active = मिच्छा;
    सचे (group_norm_cpu_compare_config.enabled) {
        सचे (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ग्रन्थिः, GGML_OP_GROUP_NORM)) {
            cpu_comparison_active = सच्च;
        } अञ्ञथा {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for GROUP_NORM operation\n");
        }
    }

    तार्किक kernel_result = ggml_et_launch_kernel(dev_ctx, "group_norm_f32", &params, परिमाण(params), 0xFFFFFFFF);

    सचे (cpu_comparison_active) {
        सचे (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ग्रन्थिः, &group_norm_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for GROUP_NORM operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END_EXT("GROUP_NORM", "group_norm_f32", ग्रन्थिः, "eps=%.6f|n_groups=%d", (द्विगुण) eps, n_groups);
    पटिनिवत्त kernel_result;
}

तार्किक ggml_et_op_im2col(ggml_backend_et_device_context * dev_ctx, ध्रुव ggml_tensor * ग्रन्थिः) {
    ET_PERF_START();

    सचे (!dev_ctx || !ग्रन्थिः) {
        GGML_LOG_ERROR("ET: Invalid parameters for IM2COL operation\n");
        पटिनिवत्त मिच्छा;
    }

    सचे (!ग्रन्थिः->src[0] || !ग्रन्थिः->src[1]) {
        GGML_LOG_ERROR("ET: IM2COL operation missing required inputs\n");
        पटिनिवत्त मिच्छा;
    }

    ध्रुव तार्किक supported_types =
        (ग्रन्थिः->type == GGML_TYPE_F32 && ग्रन्थिः->src[1]->type == GGML_TYPE_F32) ||
        (ग्रन्थिः->type == GGML_TYPE_F16 && (ग्रन्थिः->src[1]->type == GGML_TYPE_F16 || ग्रन्थिः->src[1]->type == GGML_TYPE_F32));

    सचे (!supported_types) {
        GGML_LOG_ERROR("ET: IM2COL operation with unsupported types: dst=%s src1=%s\n", ggml_type_name(ग्रन्थिः->type),
                       ggml_type_name(ग्रन्थिः->src[1]->type));
        पटिनिवत्त मिच्छा;
    }

    ggml_et_im2col_params params;
    params.src0 = *ग्रन्थिः->src[0];
    params.src1 = *ग्रन्थिः->src[1];
    params.dst  = *ग्रन्थिः;

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    तार्किक                    cpu_comparison_active = मिच्छा;
    सचे (im2col_cpu_compare_config.enabled) {
        सचे (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ग्रन्थिः, GGML_OP_IM2COL)) {
            cpu_comparison_active = सच्च;
        } अञ्ञथा {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for IM2COL operation\n");
        }
    }

    तार्किक kernel_result = ggml_et_launch_kernel(dev_ctx, "im2col", &params, परिमाण(params), 0xFFFFFFFF);

    सचे (cpu_comparison_active) {
        सचे (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ग्रन्थिः, &im2col_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for IM2COL operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("IM2COL", "im2col", ग्रन्थिः);
    पटिनिवत्त kernel_result;
}

तार्किक ggml_et_op_conv_2d(ggml_backend_et_device_context * dev_ctx, ध्रुव ggml_tensor * ग्रन्थिः) {
    ET_PERF_START();

    सचे (!dev_ctx || !ग्रन्थिः) {
        पटिनिवत्त मिच्छा;
    }
    सचे (!ग्रन्थिः->src[0] || !ग्रन्थिः->src[1]) {
        पटिनिवत्त मिच्छा;
    }
    सचे (!ग्रन्थिः->data || !ग्रन्थिः->src[0]->data || !ग्रन्थिः->src[1]->data) {
        पटिनिवत्त मिच्छा;
    }

    // Kernel constraints (mirror supports_op; recheck here as a guard).
    ध्रुव ggml_tensor * flt = ग्रन्थिः->src[0];  // [Kw, Kh, Cin, Cout]
    ध्रुव ggml_tensor * in  = ग्रन्थिः->src[1];  // [W,  H,  Cin, N]
    सचे (ग्रन्थिः->type != GGML_TYPE_F32 || flt->type != GGML_TYPE_F32 || in->type != GGML_TYPE_F32) {
        पटिनिवत्त मिच्छा;
    }

    ध्रुव int32_t s0 = ggml_get_op_params_i32(ग्रन्थिः, 0);
    ध्रुव int32_t s1 = ggml_get_op_params_i32(ग्रन्थिः, 1);
    ध्रुव int32_t p0 = ggml_get_op_params_i32(ग्रन्थिः, 2);
    ध्रुव int32_t p1 = ggml_get_op_params_i32(ग्रन्थिः, 3);
    ध्रुव int32_t d0 = ggml_get_op_params_i32(ग्रन्थिः, 4);
    ध्रुव int32_t d1 = ggml_get_op_params_i32(ग्रन्थिः, 5);

    सचे (s0 < 1 || s1 < 1) {
        पटिनिवत्त मिच्छा;
    }
    सचे (d0 != 1 || d1 != 1) {
        पटिनिवत्त मिच्छा;
    }
    सचे (flt->ne[2] % 16 != 0 || flt->ne[3] % 16 != 0) {
        पटिनिवत्त मिच्छा;
    }
    सचे (in->ne[3] != 1) {
        पटिनिवत्त मिच्छा;
    }
    सचे (ग्रन्थिः->ne[0] <= 0) {
        पटिनिवत्त मिच्छा;  // OW > 0 (any width OK; staging path handles non-16)
    }
    (शून्य) p0;
    (शून्य) p1;

    ggml_et_द्विमान_params params;
    params.src0 = *ग्रन्थिः->src[0];
    params.src1 = *ग्रन्थिः->src[1];
    params.dst  = *ग्रन्थिः;

    तार्किक kernel_result = ggml_et_launch_kernel(dev_ctx, "conv_2d_f32_me", &params, परिमाण(params), 0xFFFFFFFFu);

    ET_PERF_END("CONV_2D", "conv_2d_f32_me", ग्रन्थिः);
    पटिनिवत्त kernel_result;
}

तार्किक ggml_et_op_softmax(ggml_backend_et_device_context * dev_ctx, ध्रुव ggml_tensor * ग्रन्थिः) {
    ET_PERF_START();

    सचे (!dev_ctx || !ग्रन्थिः) {
        GGML_LOG_ERROR("ET: Invalid parameters for SOFTMAX operation\n");
        पटिनिवत्त मिच्छा;
    }

    सचे (!ग्रन्थिः->src[0]) {
        GGML_LOG_ERROR("ET: SOFTMAX operation missing required input\n");
        पटिनिवत्त मिच्छा;
    }

    ध्रुव वर्ण * kernel_name;

    सचे (ग्रन्थिः->type == GGML_TYPE_F32 && ग्रन्थिः->src[0]->type == GGML_TYPE_F32) {
        kernel_name = "softmax_f32";

    } अञ्ञथा {
        GGML_LOG_ERROR("ET: SOFTMAX operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(ग्रन्थिः->type),
                       ggml_type_name(ग्रन्थिः->src[0]->type));
        पटिनिवत्त मिच्छा;
    }

    // Validate contiguity requirements
    सचे (!ggml_is_contiguous(ग्रन्थिः)) {
        GGML_LOG_ERROR("ET: SOFTMAX operation requires contiguous destination tensor\n");
        पटिनिवत्त मिच्छा;
    }

    सचे (!ggml_is_contiguous(ग्रन्थिः->src[0])) {
        GGML_LOG_ERROR("ET: SOFTMAX operation requires contiguous source tensor\n");
        पटिनिवत्त मिच्छा;
    }

    // Check optional mask tensor
    सचे (ग्रन्थिः->src[1]) {
        सचे (ग्रन्थिः->src[1]->type != GGML_TYPE_F32) {
            GGML_LOG_ERROR("ET: SOFTMAX operation with unsupported mask type: %s (F32 required)\n",
                           ggml_type_name(ग्रन्थिः->src[1]->type));
            पटिनिवत्त मिच्छा;
        }
        सचे (!ggml_is_contiguous(ग्रन्थिः->src[1])) {
            GGML_LOG_ERROR("ET: SOFTMAX operation requires contiguous mask tensor\n");
            पटिनिवत्त मिच्छा;
        }
    }

    // Check optional sinks tensor
    सचे (ग्रन्थिः->src[2]) {
        सचे (ग्रन्थिः->src[2]->type != GGML_TYPE_F32) {
            GGML_LOG_ERROR("ET: SOFTMAX operation with unsupported sinks type: %s (F32 required)\n",
                           ggml_type_name(ग्रन्थिः->src[2]->type));
            पटिनिवत्त मिच्छा;
        }
        सचे (!ggml_is_contiguous(ग्रन्थिः->src[2])) {
            GGML_LOG_ERROR("ET: SOFTMAX operation requires contiguous sinks tensor\n");
            पटिनिवत्त मिच्छा;
        }
    }

    // Extract scale and max_bias from op_params
    भिन्न scale    = 1.0f;
    भिन्न max_bias = 0.0f;
    सचे (ग्रन्थिः->op_params) {
        memcpy(&scale, (ध्रुव भिन्न *) ग्रन्थिः->op_params + 0, परिमाण(भिन्न));
        memcpy(&max_bias, (ध्रुव भिन्न *) ग्रन्थिः->op_params + 1, परिमाण(भिन्न));
    }

    ggml_et_softmax_params params;
    params.src0 = *ग्रन्थिः->src[0];                       // F32 input tensor
    सचे (ग्रन्थिः->src[1]) {
        params.src1 = *ग्रन्थिः->src[1];                   // F32 mask tensor
    } अञ्ञथा {
        memset(&params.src1, 0, परिमाण(params.src1));  // Zero if no mask
    }
    सचे (ग्रन्थिः->src[2]) {
        params.src2 = *ग्रन्थिः->src[2];                   // F32 sinks tensor
    } अञ्ञथा {
        memset(&params.src2, 0, परिमाण(params.src2));  // Zero if no sinks
    }
    params.dst      = *ग्रन्थिः;                           // F32 output tensor
    params.scale    = scale;                           // Scale factor
    params.max_bias = max_bias;                        // ALiBi bias

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    तार्किक                    cpu_comparison_active = मिच्छा;
    सचे (softmax_cpu_compare_config.enabled) {
        सचे (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ग्रन्थिः, GGML_OP_SOFT_MAX)) {
            cpu_comparison_active = सच्च;
        } अञ्ञथा {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for SOFTMAX operation\n");
        }
    }

    तार्किक kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, परिमाण(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    सचे (cpu_comparison_active) {
        सचे (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ग्रन्थिः, &softmax_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for SOFTMAX operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END_EXT("SOFTMAX", kernel_name, ग्रन्थिः, "scale=%.6f|max_bias=%.6f|has_mask=%s", (द्विगुण) scale,
                    (द्विगुण) max_bias, ग्रन्थिः->src[1] ? "yes" : "no");
    पटिनिवत्त kernel_result;
}

तार्किक ggml_et_op_flash_attn_ext(ggml_backend_et_device_context * dev_ctx, ध्रुव ggml_tensor * ग्रन्थिः) {
    ET_PERF_START();

    सचे (!dev_ctx || !ग्रन्थिः) {
        GGML_LOG_ERROR("ET: Invalid parameters for FLASH_ATTN_EXT operation\n");
        पटिनिवत्त मिच्छा;
    }

    सचे (!ग्रन्थिः->src[0] || !ग्रन्थिः->src[1] || !ग्रन्थिः->src[2]) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT operation missing required inputs\n");
        पटिनिवत्त मिच्छा;
    }

    सचे (ग्रन्थिः->type != GGML_TYPE_F32 || ग्रन्थिः->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT requires F32 Q and dst, got dst=%s q=%s\n", ggml_type_name(ग्रन्थिः->type),
                       ggml_type_name(ग्रन्थिः->src[0]->type));
        पटिनिवत्त मिच्छा;
    }

    // K and V can be F16 or F32
    सचे ((ग्रन्थिः->src[1]->type != GGML_TYPE_F32 && ग्रन्थिः->src[1]->type != GGML_TYPE_F16) ||
        (ग्रन्थिः->src[2]->type != GGML_TYPE_F32 && ग्रन्थिः->src[2]->type != GGML_TYPE_F16)) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT K/V must be F16 or F32, got k=%s v=%s\n", ggml_type_name(ग्रन्थिः->src[1]->type),
                       ggml_type_name(ग्रन्थिः->src[2]->type));
        पटिनिवत्त मिच्छा;
    }

    सचे (ग्रन्थिः->src[4] != nullptr) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT baseline kernel does not support sinks\n");
        पटिनिवत्त मिच्छा;
    }

    // Mask is optional; if present must be F16 or F32
    सचे (ग्रन्थिः->src[3] != nullptr && ग्रन्थिः->src[3]->type != GGML_TYPE_F32 && ग्रन्थिः->src[3]->type != GGML_TYPE_F16) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT mask must be F16 or F32, got %s\n", ggml_type_name(ग्रन्थिः->src[3]->type));
        पटिनिवत्त मिच्छा;
    }

    // Q and dst must be row-contiguous F32
    सचे (!ggml_is_contiguous_rows(ग्रन्थिः) || !ggml_is_contiguous_rows(ग्रन्थिः->src[0])) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT requires row-contiguous Q and dst\n");
        पटिनिवत्त मिच्छा;
    }

    सचे (ग्रन्थिः->nb[0] != परिमाण(भिन्न) || ग्रन्थिः->src[0]->nb[0] != परिमाण(भिन्न)) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT requires contiguous F32 rows for Q and dst\n");
        पटिनिवत्त मिच्छा;
    }

    // K/V must have element-sized stride in dim 0
    ध्रुव size_t k_elem = ग्रन्थिः->src[1]->type == GGML_TYPE_F16 ? 2 : 4;
    ध्रुव size_t v_elem = ग्रन्थिः->src[2]->type == GGML_TYPE_F16 ? 2 : 4;
    सचे (ग्रन्थिः->src[1]->nb[0] != k_elem || ग्रन्थिः->src[2]->nb[0] != v_elem) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT K/V must have element-sized stride in dim 0\n");
        पटिनिवत्त मिच्छा;
    }

    भिन्न scale         = 1.0f;
    भिन्न max_bias      = 0.0f;
    भिन्न logit_softcap = 0.0f;
    memcpy(&scale, (ध्रुव भिन्न *) ग्रन्थिः->op_params + 0, परिमाण(scale));
    memcpy(&max_bias, (ध्रुव भिन्न *) ग्रन्थिः->op_params + 1, परिमाण(max_bias));
    memcpy(&logit_softcap, (ध्रुव भिन्न *) ग्रन्थिः->op_params + 2, परिमाण(logit_softcap));

    सचे (max_bias != 0.0f || logit_softcap != 0.0f) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT baseline kernel does not support max_bias or logit_softcap\n");
        पटिनिवत्त मिच्छा;
    }

    ध्रुव ggml_prec prec = ggml_flash_attn_ext_get_prec(ग्रन्थिः);
    सचे (prec != GGML_PREC_F32 && prec != GGML_PREC_DEFAULT) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT baseline kernel only supports F32 precision\n");
        पटिनिवत्त मिच्छा;
    }

    // dk must match between Q and K; dv must match between V and dst
    सचे (ग्रन्थिः->src[0]->ne[0] != ग्रन्थिः->src[1]->ne[0]) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT dk mismatch: Q=%lld K=%lld\n", (long long) ग्रन्थिः->src[0]->ne[0],
                       (long long) ग्रन्थिः->src[1]->ne[0]);
        पटिनिवत्त मिच्छा;
    }

    सचे (ग्रन्थिः->src[2]->ne[0] != ग्रन्थिः->ne[0]) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT dv mismatch: V=%lld dst=%lld\n", (long long) ग्रन्थिः->src[2]->ne[0],
                       (long long) ग्रन्थिः->ne[0]);
        पटिनिवत्त मिच्छा;
    }

    सचे (ग्रन्थिः->src[2]->ne[0] > 512) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT dv=%lld exceeds maximum 512\n", (long long) ग्रन्थिः->src[2]->ne[0]);
        पटिनिवत्त मिच्छा;
    }

    सचे (ग्रन्थिः->src[0]->ne[0] > 512) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT dk=%lld exceeds maximum 512\n", (long long) ग्रन्थिः->src[0]->ne[0]);
        पटिनिवत्त मिच्छा;
    }

    // GQA: n_head_q must be a multiple of n_head_kv
    ध्रुव int64_t nhq = ग्रन्थिः->src[0]->ne[2];
    ध्रुव int64_t nhk = ग्रन्थिः->src[1]->ne[2];
    सचे (nhq % nhk != 0) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT n_head_q (%lld) not divisible by n_head_kv (%lld)\n", (long long) nhq,
                       (long long) nhk);
        पटिनिवत्त मिच्छा;
    }

    // K and V must have matching sequence length, heads, and batch dims
    सचे (ग्रन्थिः->src[1]->ne[1] != ग्रन्थिः->src[2]->ne[1] || ग्रन्थिः->src[1]->ne[2] != ग्रन्थिः->src[2]->ne[2] ||
        ग्रन्थिः->src[1]->ne[3] != ग्रन्थिः->src[2]->ne[3]) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT K/V shape mismatch\n");
        पटिनिवत्त मिच्छा;
    }

    // dst layout checks: [dv, nhq, nq, no]
    सचे (ग्रन्थिः->src[0]->ne[1] != ग्रन्थिः->ne[2] || ग्रन्थिः->src[0]->ne[2] != ग्रन्थिः->ne[1] ||
        ग्रन्थिः->src[0]->ne[3] != ग्रन्थिः->ne[3]) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT dst shape mismatch\n");
        पटिनिवत्त मिच्छा;
    }

    // Batch dims: Q batch must match K batch
    सचे (ग्रन्थिः->src[0]->ne[3] != ग्रन्थिः->src[1]->ne[3]) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT batch dimension mismatch\n");
        पटिनिवत्त मिच्छा;
    }

    ggml_et_flash_attn_ext_params params;
    memset(&params, 0, परिमाण(params));
    params.src0 = *ग्रन्थिः->src[0];
    params.src1 = *ग्रन्थिः->src[1];
    params.src2 = *ग्रन्थिः->src[2];
    सचे (ग्रन्थिः->src[3] != nullptr) {
        params.mask     = *ग्रन्थिः->src[3];
        params.has_mask = 1;
    }
    params.dst   = *ग्रन्थिः;
    params.scale = scale;

    // Use matrix engine kernel when K/V are F16 and dk is a multiple of 32
    ध्रुव वर्ण * kernel_name;
    सचे (ग्रन्थिः->src[1]->type == GGML_TYPE_F16 && ग्रन्थिः->src[2]->type == GGML_TYPE_F16 && (ग्रन्थिः->src[0]->ne[0] % 32) == 0) {
        kernel_name = "flash_attn_ext_f16_me";
    } अञ्ञथा {
        kernel_name = "flash_attn_ext_f32";
    }

    ध्रुव तार्किक kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, परिमाण(params), 0xFFFFFFFF);

    ET_PERF_END_EXT("FLASH_ATTN_EXT", kernel_name, ग्रन्थिः, "scale=%.6f", (द्विगुण) scale);
    पटिनिवत्त kernel_result;
}

तार्किक ggml_et_op_get_rows(ggml_backend_et_device_context * dev_ctx, ध्रुव ggml_tensor * ग्रन्थिः) {
    ET_PERF_START();

    सचे (!dev_ctx || !ग्रन्थिः) {
        GGML_LOG_ERROR("ET: Invalid parameters for GET_ROWS operation\n");
        पटिनिवत्त मिच्छा;
    }

    सचे (!ग्रन्थिः->src[0] || !ग्रन्थिः->src[1]) {
        GGML_LOG_ERROR("ET: GET_ROWS operation missing required inputs\n");
        पटिनिवत्त मिच्छा;
    }

    ध्रुव वर्ण * kernel_name;

    सचे (ग्रन्थिः->type == GGML_TYPE_F32 && ग्रन्थिः->src[1]->type == GGML_TYPE_I32 &&
        (ग्रन्थिः->src[0]->type == GGML_TYPE_F32 || ग्रन्थिः->src[0]->type == GGML_TYPE_F16 ||
         ग्रन्थिः->src[0]->type == GGML_TYPE_Q4_0 || ग्रन्थिः->src[0]->type == GGML_TYPE_Q8_0 ||
         ग्रन्थिः->src[0]->type == GGML_TYPE_Q4_K)) {
        kernel_name = "get_rows_f32";

    } अञ्ञथा {
        GGML_LOG_ERROR("ET: GET_ROWS operation with unsupported types: dst=%s src0=%s src1=%s\n",
                       ggml_type_name(ग्रन्थिः->type), ggml_type_name(ग्रन्थिः->src[0]->type),
                       ggml_type_name(ग्रन्थिः->src[1]->type));
        पटिनिवत्त मिच्छा;
    }

    // Validate contiguity requirements
    सचे (!ggml_is_contiguous(ग्रन्थिः)) {
        GGML_LOG_ERROR("ET: GET_ROWS operation requires contiguous destination tensor\n");
        पटिनिवत्त मिच्छा;
    }

    सचे (!ggml_is_contiguous(ग्रन्थिः->src[0])) {
        GGML_LOG_ERROR("ET: GET_ROWS operation requires contiguous data tensor\n");
        पटिनिवत्त मिच्छा;
    }

    सचे (!ggml_is_contiguous(ग्रन्थिः->src[1])) {
        GGML_LOG_ERROR("ET: GET_ROWS operation requires contiguous indices tensor\n");
        पटिनिवत्त मिच्छा;
    }

    // Validate dimension constraints from ggml implementation
    सचे (ग्रन्थिः->src[0]->ne[2] != ग्रन्थिः->src[1]->ne[1] || ग्रन्थिः->src[1]->ne[3] != 1) {
        GGML_LOG_ERROR(
            "ET: GET_ROWS operation dimension constraint failed: src0.ne[2]=%lld != src1.ne[1]=%lld or src1.ne[3]=%lld "
            "!= 1\n",
            (long long) ग्रन्थिः->src[0]->ne[2], (long long) ग्रन्थिः->src[1]->ne[1], (long long) ग्रन्थिः->src[1]->ne[3]);
        पटिनिवत्त मिच्छा;
    }

    ggml_et_get_rows_params params;
    params.src0 = *ग्रन्थिः->src[0];  // Data tensor (F32 or Q8_0)
    params.src1 = *ग्रन्थिः->src[1];  // Indices tensor (I32)
    params.dst  = *ग्रन्थिः;          // Output tensor (F32)

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    तार्किक                    cpu_comparison_active = मिच्छा;
    सचे (get_rows_cpu_compare_config.enabled) {
        सचे (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ग्रन्थिः, GGML_OP_GET_ROWS)) {
            cpu_comparison_active = सच्च;
        } अञ्ञथा {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for GET_ROWS operation\n");
        }
    }

    तार्किक kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, परिमाण(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    सचे (cpu_comparison_active) {
        सचे (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ग्रन्थिः, &get_rows_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for GET_ROWS operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("GET_ROWS", kernel_name, ग्रन्थिः);
    पटिनिवत्त kernel_result;
}

तार्किक ggml_et_op_cont(ggml_backend_et_device_context * dev_ctx, ध्रुव ggml_tensor * ग्रन्थिः) {
    ET_PERF_START();

    // Validate source tensor exists
    सचे (!ग्रन्थिः->src[0]) {
        GGML_LOG_ERROR("ET: CONT operation missing source tensor\n");
        पटिनिवत्त मिच्छा;
    }

    // Validate types match (input and output must be same type)
    सचे (ग्रन्थिः->type != ग्रन्थिः->src[0]->type) {
        GGML_LOG_ERROR("ET: CONT operation type mismatch: src=%s dst=%s\n", ggml_type_name(ग्रन्थिः->src[0]->type),
                       ggml_type_name(ग्रन्थिः->type));
        पटिनिवत्त मिच्छा;
    }

    // Validate supported types
    सचे (ग्रन्थिः->type != GGML_TYPE_F32 && ग्रन्थिः->type != GGML_TYPE_F16) {
        GGML_LOG_ERROR("ET: CONT operation unsupported type: %s (only F32 and F16 supported)\n",
                       ggml_type_name(ग्रन्थिः->type));
        पटिनिवत्त मिच्छा;
    }

    // Validate contiguity - output must be contiguous, input can be non-contiguous
    सचे (!ggml_is_contiguous(ग्रन्थिः)) {
        GGML_LOG_ERROR("ET: CONT operation requires contiguous output tensor\n");
        पटिनिवत्त मिच्छा;
    }

    // Select kernel based on type
    ध्रुव वर्ण * kernel_name;
    सचे (ग्रन्थिः->type == GGML_TYPE_F32) {
        kernel_name = "cont_f32";
    } अञ्ञथा सचे (ग्रन्थिः->type == GGML_TYPE_F16) {
        kernel_name = "cont_f16";
    } अञ्ञथा {
        GGML_LOG_ERROR("ET: CONT operation with unsupported type: %s\n", ggml_type_name(ग्रन्थिः->type));
        पटिनिवत्त मिच्छा;
    }

    ggml_et_cont_params params;
    params.src0 = *ग्रन्थिः->src[0];  // Input tensor (potentially non-contiguous)
    params.dst  = *ग्रन्थिः;          // Output tensor (contiguous)

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    तार्किक                    cpu_comparison_active = मिच्छा;
    सचे (cont_cpu_compare_config.enabled) {
        सचे (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ग्रन्थिः, GGML_OP_CONT)) {
            cpu_comparison_active = सच्च;
        } अञ्ञथा {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for CONT operation\n");
        }
    }

    तार्किक kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, परिमाण(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    सचे (cpu_comparison_active) {
        सचे (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ग्रन्थिः, &cont_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for CONT operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("CONT", kernel_name, ग्रन्थिः);
    पटिनिवत्त kernel_result;
}

तार्किक ggml_et_op_cumsum(ggml_backend_et_device_context * dev_ctx, ध्रुव ggml_tensor * ग्रन्थिः) {
    ET_PERF_START();

    सचे (!dev_ctx || !ग्रन्थिः || !ग्रन्थिः->src[0]) {
        GGML_LOG_ERROR("ET: Invalid parameters for CUMSUM operation\n");
        पटिनिवत्त मिच्छा;
    }

    सचे (ग्रन्थिः->type != GGML_TYPE_F32 || ग्रन्थिः->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: CUMSUM operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(ग्रन्थिः->type),
                       ggml_type_name(ग्रन्थिः->src[0]->type));
        पटिनिवत्त मिच्छा;
    }

    ध्रुव वर्ण * kernel_name = "cumsum_f32";

    ggml_et_cumsum_params params;
    params.src0 = *ग्रन्थिः->src[0];
    params.dst  = *ग्रन्थिः;

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    तार्किक                    cpu_comparison_active = मिच्छा;
    सचे (cumsum_cpu_compare_config.enabled) {
        सचे (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ग्रन्थिः, GGML_OP_CUMSUM)) {
            cpu_comparison_active = सच्च;
        } अञ्ञथा {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for CUMSUM operation\n");
        }
    }

    तार्किक kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, परिमाण(params), 0xFFFFFFFF);

    सचे (cpu_comparison_active) {
        सचे (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ग्रन्थिः, &cumsum_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for CUMSUM operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("CUMSUM", kernel_name, ग्रन्थिः);
    पटिनिवत्त kernel_result;
}

तार्किक ggml_et_op_cpy(ggml_backend_et_device_context * dev_ctx, ध्रुव ggml_tensor * ग्रन्थिः) {
    ET_PERF_START();

    // CPY copies data from src[0] into the layout of dst (which matches src[1])
    // For same-type with contiguous dst, this is identical to CONT
    सचे (!ग्रन्थिः->src[0]) {
        GGML_LOG_ERROR("ET: CPY operation missing source tensor\n");
        पटिनिवत्त मिच्छा;
    }

    // Scalar / zero-element special path: if any dimension is 0, nothing to copy
    ध्रुव int64_t nelements = ग्रन्थिः->ne[0] * ग्रन्थिः->ne[1] * ग्रन्थिः->ne[2] * ग्रन्थिः->ne[3];
    सचे (nelements == 0) {
        GGML_LOG_DEBUG("ET: CPY no-op (zero elements): ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]\n",
                       ग्रन्थिः->ne[0], ग्रन्थिः->ne[1], ग्रन्थिः->ne[2], ग्रन्थिः->ne[3]);
        ET_PERF_END("CPY", "noop", ग्रन्थिः);
        पटिनिवत्त सच्च;
    }

    // Only F32 and F16 supported for dst
    सचे (ग्रन्थिः->type != GGML_TYPE_F32 && ग्रन्थिः->type != GGML_TYPE_F16) {
        GGML_LOG_ERROR("ET: CPY unsupported dst type: %s\n", ggml_type_name(ग्रन्थिः->type));
        पटिनिवत्त मिच्छा;
    }

    // Select kernel based on src/dst type combination
    ध्रुव वर्ण * kernel_name;
    सचे (ग्रन्थिः->src[0]->type == GGML_TYPE_F32 && ग्रन्थिः->type == GGML_TYPE_F32) {
        kernel_name = "cont_f32";
    } अञ्ञथा सचे (ग्रन्थिः->src[0]->type == GGML_TYPE_F16 && ग्रन्थिः->type == GGML_TYPE_F16) {
        kernel_name = "cont_f16";
    } अञ्ञथा सचे (ग्रन्थिः->src[0]->type == GGML_TYPE_F32 && ग्रन्थिः->type == GGML_TYPE_F16) {
        kernel_name = "cpy_f32_f16";
    } अञ्ञथा {
        GGML_LOG_ERROR("ET: CPY unsupported type combination: src=%s dst=%s\n", ggml_type_name(ग्रन्थिः->src[0]->type),
                       ggml_type_name(ग्रन्थिः->type));
        पटिनिवत्त मिच्छा;
    }

    ggml_et_cont_params params;
    params.src0 = *ग्रन्थिः->src[0];
    params.dst  = *ग्रन्थिः;

    // CPU comparison for debugging
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    तार्किक                    cpu_comparison_active = मिच्छा;
    सचे (cont_cpu_compare_config.enabled) {
        सचे (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ग्रन्थिः, GGML_OP_CPY)) {
            cpu_comparison_active = सच्च;
        } अञ्ञथा {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for CPY operation\n");
        }
    }

    तार्किक kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, परिमाण(params), 0xFFFFFFFF);

    सचे (cpu_comparison_active) {
        सचे (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ग्रन्थिः, &cont_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for CPY operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("CPY", kernel_name, ग्रन्थिः);
    पटिनिवत्त kernel_result;
}

तार्किक ggml_et_op_concat(ggml_backend_et_device_context * dev_ctx, ध्रुव ggml_tensor * ग्रन्थिः) {
    ET_PERF_START();

    सचे (!dev_ctx || !ग्रन्थिः) {
        GGML_LOG_ERROR("ET: Invalid parameters for CONCAT operation\n");
        पटिनिवत्त मिच्छा;
    }

    सचे (!ग्रन्थिः->src[0] || !ग्रन्थिः->src[1]) {
        GGML_LOG_ERROR("ET: CONCAT operation missing required inputs\n");
        पटिनिवत्त मिच्छा;
    }

    ध्रुव वर्ण * kernel_name;

    सचे (ग्रन्थिः->type == GGML_TYPE_F32 && ग्रन्थिः->src[0]->type == GGML_TYPE_F32 && ग्रन्थिः->src[1]->type == GGML_TYPE_F32) {
        kernel_name = "concat_f32";

    } अञ्ञथा {
        GGML_LOG_ERROR("ET: CONCAT operation with unsupported types: dst=%s src0=%s src1=%s\n",
                       ggml_type_name(ग्रन्थिः->type), ggml_type_name(ग्रन्थिः->src[0]->type),
                       ggml_type_name(ग्रन्थिः->src[1]->type));
        पटिनिवत्त मिच्छा;
    }

    int32_t dim;
    memcpy(&dim, ग्रन्थिः->op_params, परिमाण(int32_t));

    ggml_et_concat_params params;
    params.src0 = *ग्रन्थिः->src[0];
    params.src1 = *ग्रन्थिः->src[1];
    params.dst  = *ग्रन्थिः;
    params.dim  = dim;

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    तार्किक                    cpu_comparison_active = मिच्छा;
    सचे (concat_cpu_compare_config.enabled) {
        सचे (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ग्रन्थिः, GGML_OP_CONCAT)) {
            cpu_comparison_active = सच्च;
        } अञ्ञथा {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for CONCAT operation\n");
        }
    }

    तार्किक kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, परिमाण(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    सचे (cpu_comparison_active) {
        सचे (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ग्रन्थिः, &concat_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for CONCAT operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END_EXT("CONCAT", kernel_name, ग्रन्थिः, "dim=%d", dim);
    पटिनिवत्त kernel_result;
}

तार्किक ggml_et_op_repeat(ggml_backend_et_device_context * dev_ctx, ध्रुव ggml_tensor * ग्रन्थिः) {
    ET_PERF_START();

    सचे (!dev_ctx || !ग्रन्थिः) {
        GGML_LOG_ERROR("ET: Invalid parameters for REPEAT operation\n");
        पटिनिवत्त मिच्छा;
    }

    सचे (!ग्रन्थिः->src[0]) {
        GGML_LOG_ERROR("ET: REPEAT operation missing required input\n");
        पटिनिवत्त मिच्छा;
    }

    ध्रुव वर्ण * kernel_name;

    सचे (ग्रन्थिः->type == GGML_TYPE_F32 && ग्रन्थिः->src[0]->type == GGML_TYPE_F32) {
        // No-op REPEAT (every repeat factor is 1): the output is just a copy
        // of the input. Route to cont_f32, whose contiguous fast path handles
        // arbitrary sizes (including those rejected by repeat_f32's gate,
        // e.g. ne[0]=1).
        सचे (ggml_are_same_shape(ग्रन्थिः->src[0], ग्रन्थिः)) {
            kernel_name = "cont_f32";
        } अञ्ञथा {
            kernel_name = "repeat_f32";
        }

    } अञ्ञथा {
        GGML_LOG_ERROR("ET: REPEAT operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(ग्रन्थिः->type),
                       ggml_type_name(ग्रन्थिः->src[0]->type));
        पटिनिवत्त मिच्छा;
    }

    // ggml_et_cont_params and ggml_et_repeat_params have identical layouts
    // (just src0 + dst), so the same payload works for either kernel.
    ggml_et_repeat_params params;
    params.src0 = *ग्रन्थिः->src[0];
    params.dst  = *ग्रन्थिः;

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    तार्किक                    cpu_comparison_active = मिच्छा;
    सचे (repeat_cpu_compare_config.enabled) {
        सचे (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ग्रन्थिः, GGML_OP_REPEAT)) {
            cpu_comparison_active = सच्च;
        } अञ्ञथा {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for REPEAT operation\n");
        }
    }

    तार्किक kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, परिमाण(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    सचे (cpu_comparison_active) {
        सचे (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ग्रन्थिः, &repeat_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for REPEAT operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("REPEAT", kernel_name, ग्रन्थिः);
    पटिनिवत्त kernel_result;
}

तार्किक ggml_et_op_ssm_conv(ggml_backend_et_device_context * dev_ctx, ध्रुव ggml_tensor * ग्रन्थिः) {
    ET_PERF_START();

    सचे (!dev_ctx || !ग्रन्थिः || !ग्रन्थिः->src[0] || !ग्रन्थिः->src[1]) {
        GGML_LOG_ERROR("ET: Invalid parameters for SSM_CONV operation\n");
        पटिनिवत्त मिच्छा;
    }

    सचे (ग्रन्थिः->type != GGML_TYPE_F32 || ग्रन्थिः->src[0]->type != GGML_TYPE_F32 || ग्रन्थिः->src[1]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: SSM_CONV operation with unsupported types: dst=%s src0=%s src1=%s\n",
                       ggml_type_name(ग्रन्थिः->type), ggml_type_name(ग्रन्थिः->src[0]->type),
                       ggml_type_name(ग्रन्थिः->src[1]->type));
        पटिनिवत्त मिच्छा;
    }

    ध्रुव वर्ण * kernel_name = "ssm_conv_f32";

    ggml_et_ssm_conv_params params;
    params.src0 = *ग्रन्थिः->src[0];
    params.src1 = *ग्रन्थिः->src[1];
    params.dst  = *ग्रन्थिः;

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    तार्किक                    cpu_comparison_active = मिच्छा;
    सचे (ssm_conv_cpu_compare_config.enabled) {
        सचे (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ग्रन्थिः, GGML_OP_SSM_CONV)) {
            cpu_comparison_active = सच्च;
        } अञ्ञथा {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for SSM_CONV operation\n");
        }
    }

    तार्किक kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, परिमाण(params), 0xFFFFFFFF);

    सचे (cpu_comparison_active) {
        सचे (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ग्रन्थिः, &ssm_conv_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for SSM_CONV operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("SSM_CONV", kernel_name, ग्रन्थिः);
    पटिनिवत्त kernel_result;
}

तार्किक ggml_et_op_ssm_scan(ggml_backend_et_device_context * dev_ctx, ध्रुव ggml_tensor * ग्रन्थिः) {
    ET_PERF_START();

    सचे (!dev_ctx || !ग्रन्थिः) {
        GGML_LOG_ERROR("ET: Invalid parameters for SSM_SCAN operation\n");
        पटिनिवत्त मिच्छा;
    }

    अत्थाय (पूर्णाङ्क i = 0; i < 7; ++i) {
        सचे (!ग्रन्थिः->src[i]) {
            GGML_LOG_ERROR("ET: SSM_SCAN missing required input %d\n", i);
            पटिनिवत्त मिच्छा;
        }
    }

    सचे (ग्रन्थिः->type != GGML_TYPE_F32 || ग्रन्थिः->src[0]->type != GGML_TYPE_F32 || ग्रन्थिः->src[1]->type != GGML_TYPE_F32 ||
        ग्रन्थिः->src[2]->type != GGML_TYPE_F32 || ग्रन्थिः->src[3]->type != GGML_TYPE_F32 ||
        ग्रन्थिः->src[4]->type != GGML_TYPE_F32 || ग्रन्थिः->src[5]->type != GGML_TYPE_F32 ||
        ग्रन्थिः->src[6]->type != GGML_TYPE_I32) {
        GGML_LOG_ERROR("ET: SSM_SCAN operation with unsupported types\n");
        पटिनिवत्त मिच्छा;
    }

    ggml_et_ssm_scan_params params;
    params.src0 = *ग्रन्थिः->src[0];
    params.src1 = *ग्रन्थिः->src[1];
    params.src2 = *ग्रन्थिः->src[2];
    params.src3 = *ग्रन्थिः->src[3];
    params.src4 = *ग्रन्थिः->src[4];
    params.src5 = *ग्रन्थिः->src[5];
    params.src6 = *ग्रन्थिः->src[6];
    params.dst  = *ग्रन्थिः;
    params.K    = ggml_get_op_params_i32(ग्रन्थिः, 0);

    तार्किक kernel_result = ggml_et_launch_kernel(dev_ctx, "ssm_scan_f32", &params, परिमाण(params), 0xFFFFFFFF);

    ET_PERF_END("SSM_SCAN", "ssm_scan_f32", ग्रन्थिः);
    पटिनिवत्त kernel_result;
}

तार्किक ggml_et_op_rwkv_wkv6(ggml_backend_et_device_context * dev_ctx, ध्रुव ggml_tensor * ग्रन्थिः) {
    ET_PERF_START();

    सचे (!dev_ctx || !ग्रन्थिः) {
        GGML_LOG_ERROR("ET: Invalid parameters for RWKV_WKV6 operation\n");
        पटिनिवत्त मिच्छा;
    }

    // Validate all 6 source tensors exist
    अत्थाय (पूर्णाङ्क i = 0; i <= 5; i++) {
        सचे (!ग्रन्थिः->src[i]) {
            GGML_LOG_ERROR("ET: RWKV_WKV6 operation missing src[%d]\n", i);
            पटिनिवत्त मिच्छा;
        }
    }

    सचे (ग्रन्थिः->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: RWKV_WKV6 only supports F32, got %s\n", ggml_type_name(ग्रन्थिः->type));
        पटिनिवत्त मिच्छा;
    }

    ध्रुव वर्ण * kernel_name = "rwkv_wkv6_f32";

    ध्रुव int64_t S      = ग्रन्थिः->src[0]->ne[0];  // head_size
    ध्रुव int64_t H      = ग्रन्थिः->src[0]->ne[1];  // num heads
    ध्रुव int64_t T      = ग्रन्थिः->src[1]->ne[2];  // num tokens
    ध्रुव int64_t n_seqs = ग्रन्थिः->src[5]->ne[1];  // num sequences
    ध्रुव int64_t C      = S * H;

    ggml_et_rwkv_wkv6_params params;
    params.k        = (भिन्न *) ग्रन्थिः->src[0]->data;
    params.v        = (भिन्न *) ग्रन्थिः->src[1]->data;
    params.r        = (भिन्न *) ग्रन्थिः->src[2]->data;
    params.tf       = (भिन्न *) ग्रन्थिः->src[3]->data;
    params.td       = (भिन्न *) ग्रन्थिः->src[4]->data;
    params.state_in = (भिन्न *) ग्रन्थिः->src[5]->data;
    params.dst      = (भिन्न *) ग्रन्थिः->data;
    params.C        = (int32_t) C;
    params.H        = (int32_t) H;
    params.S        = (int32_t) S;
    params.T        = (int32_t) T;
    params.n_seqs   = (int32_t) n_seqs;

    // Phase 1: Initialize CPU comparison context
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    तार्किक                    cpu_comparison_active = मिच्छा;
    सचे (rwkv_wkv6_cpu_compare_config.enabled) {
        सचे (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ग्रन्थिः, GGML_OP_RWKV_WKV6)) {
            cpu_comparison_active = सच्च;
        } अञ्ञथा {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for RWKV_WKV6 operation\n");
        }
    }

    तार्किक kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, परिमाण(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare
    सचे (cpu_comparison_active) {
        सचे (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ग्रन्थिः, &rwkv_wkv6_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for RWKV_WKV6 operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END_EXT("RWKV_WKV6", kernel_name, ग्रन्थिः, "S=%d H=%d T=%d n_seqs=%d", (पूर्णाङ्क) S, (पूर्णाङ्क) H, (पूर्णाङ्क) T,
                    (पूर्णाङ्क) n_seqs);
    पटिनिवत्त kernel_result;
}

तार्किक ggml_et_op_rwkv_wkv7(ggml_backend_et_device_context * dev_ctx, ध्रुव ggml_tensor * ग्रन्थिः) {
    ET_PERF_START();

    सचे (!dev_ctx || !ग्रन्थिः) {
        GGML_LOG_ERROR("ET: Invalid parameters for RWKV_WKV7 operation\n");
        पटिनिवत्त मिच्छा;
    }

    // Validate all 7 source tensors exist
    अत्थाय (पूर्णाङ्क i = 0; i <= 6; i++) {
        सचे (!ग्रन्थिः->src[i]) {
            GGML_LOG_ERROR("ET: RWKV_WKV7 operation missing src[%d]\n", i);
            पटिनिवत्त मिच्छा;
        }
    }

    सचे (ग्रन्थिः->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: RWKV_WKV7 only supports F32, got %s\n", ggml_type_name(ग्रन्थिः->type));
        पटिनिवत्त मिच्छा;
    }

    ध्रुव वर्ण * kernel_name = "rwkv_wkv7_f32";

    ध्रुव int64_t S      = ग्रन्थिः->src[2]->ne[0];  // head_size
    ध्रुव int64_t H      = ग्रन्थिः->src[2]->ne[1];  // num heads
    ध्रुव int64_t T      = ग्रन्थिः->src[1]->ne[2];  // num tokens
    ध्रुव int64_t n_seqs = ग्रन्थिः->src[6]->ne[1];  // num sequences
    ध्रुव int64_t C      = S * H;

    ggml_et_rwkv_wkv7_params params;
    params.r        = (भिन्न *) ग्रन्थिः->src[0]->data;
    params.w        = (भिन्न *) ग्रन्थिः->src[1]->data;
    params.k        = (भिन्न *) ग्रन्थिः->src[2]->data;
    params.v        = (भिन्न *) ग्रन्थिः->src[3]->data;
    params.a        = (भिन्न *) ग्रन्थिः->src[4]->data;
    params.b        = (भिन्न *) ग्रन्थिः->src[5]->data;
    params.state_in = (भिन्न *) ग्रन्थिः->src[6]->data;
    params.dst      = (भिन्न *) ग्रन्थिः->data;
    params.C        = (int32_t) C;
    params.H        = (int32_t) H;
    params.S        = (int32_t) S;
    params.T        = (int32_t) T;
    params.n_seqs   = (int32_t) n_seqs;

    // Phase 1: Initialize CPU comparison context
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    तार्किक                    cpu_comparison_active = मिच्छा;
    सचे (rwkv_wkv7_cpu_compare_config.enabled) {
        सचे (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ग्रन्थिः, GGML_OP_RWKV_WKV7)) {
            cpu_comparison_active = सच्च;
        } अञ्ञथा {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for RWKV_WKV7 operation\n");
        }
    }

    तार्किक kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, परिमाण(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare
    सचे (cpu_comparison_active) {
        सचे (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ग्रन्थिः, &rwkv_wkv7_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for RWKV_WKV7 operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END_EXT("RWKV_WKV7", kernel_name, ग्रन्थिः, "S=%d H=%d T=%d n_seqs=%d", (पूर्णाङ्क) S, (पूर्णाङ्क) H, (पूर्णाङ्क) T,
                    (पूर्णाङ्क) n_seqs);
    पटिनिवत्त kernel_result;
}

स्थिर ggml_et_cpu_compare_config gated_delta_net_cpu_compare_config = {
    /* .enabled = */ मिच्छा,
    /* .use_cpu_result = */ मिच्छा,
    /* .log_differences = */ सच्च,
    /* .tolerance = */ 1e-4f,
    /* .max_log_elements = */ 4096
};

तार्किक ggml_et_op_gated_delta_net(ggml_backend_et_device_context * dev_ctx, ध्रुव ggml_tensor * ग्रन्थिः) {
    ET_PERF_START();

    सचे (!dev_ctx || !ग्रन्थिः) {
        GGML_LOG_ERROR("ET: Invalid parameters for GATED_DELTA_NET operation\n");
        पटिनिवत्त मिच्छा;
    }

    // Validate all 6 source tensors exist
    अत्थाय (पूर्णाङ्क i = 0; i <= 5; i++) {
        सचे (!ग्रन्थिः->src[i]) {
            GGML_LOG_ERROR("ET: GATED_DELTA_NET operation missing src[%d]\n", i);
            पटिनिवत्त मिच्छा;
        }
    }

    सचे (ग्रन्थिः->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: GATED_DELTA_NET only supports F32, got %s\n", ggml_type_name(ग्रन्थिः->type));
        पटिनिवत्त मिच्छा;
    }

    ध्रुव वर्ण * kernel_name = "gated_delta_net_f32";

    ध्रुव ggml_tensor * src_q     = ग्रन्थिः->src[0];
    ध्रुव ggml_tensor * src_k     = ग्रन्थिः->src[1];
    ध्रुव ggml_tensor * src_v     = ग्रन्थिः->src[2];
    ध्रुव ggml_tensor * src_g     = ग्रन्थिः->src[3];
    ध्रुव ggml_tensor * src_beta  = ग्रन्थिः->src[4];
    ध्रुव ggml_tensor * src_state = ग्रन्थिः->src[5];

    ध्रुव int64_t S_v      = src_v->ne[0];
    ध्रुव int64_t H        = src_v->ne[1];
    ध्रुव int64_t n_tokens = src_v->ne[2];
    ध्रुव int64_t n_seqs   = src_v->ne[3];
    ध्रुव int64_t H_q      = src_q->ne[1];
    ध्रुव int64_t H_k      = src_k->ne[1];
    ध्रुव int64_t n_seqs_q = src_q->ne[3];
    ध्रुव int64_t n_seqs_k = src_k->ne[3];

    ggml_et_gated_delta_net_params params;
    params.q        = *src_q;
    params.k        = *src_k;
    params.v        = *src_v;
    params.g        = *src_g;
    params.beta     = *src_beta;
    params.state_in = *src_state;
    params.dst      = *ग्रन्थिः;
    params.S_v      = (int32_t) S_v;
    params.H        = (int32_t) H;
    params.H_q      = (int32_t) H_q;
    params.H_k      = (int32_t) H_k;
    params.n_tokens = (int32_t) n_tokens;
    params.n_seqs   = (int32_t) n_seqs;
    params.n_seqs_q = (int32_t) n_seqs_q;
    params.n_seqs_k = (int32_t) n_seqs_k;
    params.kda      = (src_g->ne[0] == S_v) ? 1 : 0;
    params.K        = ggml_get_op_params_i32(ग्रन्थिः, 0);
    params.scale    = 1.0f / sqrtf((भिन्न) S_v);

    // CPU comparison for debugging
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    तार्किक                    cpu_comparison_active = मिच्छा;
    सचे (gated_delta_net_cpu_compare_config.enabled) {
        सचे (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ग्रन्थिः, GGML_OP_GATED_DELTA_NET)) {
            cpu_comparison_active = सच्च;
        } अञ्ञथा {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for GATED_DELTA_NET operation\n");
        }
    }

    तार्किक kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, परिमाण(params), 0xFFFFFFFF);

    सचे (cpu_comparison_active) {
        सचे (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ग्रन्थिः, &gated_delta_net_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for GATED_DELTA_NET operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END_EXT("GATED_DELTA_NET", kernel_name, ग्रन्थिः, "S_v=%d H=%d n_tokens=%d n_seqs=%d kda=%d", (पूर्णाङ्क) S_v,
                    (पूर्णाङ्क) H, (पूर्णाङ्क) n_tokens, (पूर्णाङ्क) n_seqs, params.kda);
    पटिनिवत्त kernel_result;
}

तार्किक ggml_et_op_set_rows(ggml_backend_et_device_context * dev_ctx, ध्रुव ggml_tensor * ग्रन्थिः) {
    ET_PERF_START();

    सचे (!dev_ctx || !ग्रन्थिः) {
        GGML_LOG_ERROR("ET: Invalid parameters for SET_ROWS operation\n");
        पटिनिवत्त मिच्छा;
    }

    सचे (!ग्रन्थिः->src[0] || !ग्रन्थिः->src[1] || !ग्रन्थिः->src[2]) {
        GGML_LOG_ERROR(
            "ET: SET_ROWS operation missing required inputs (needs src[0]=base, src[1]=indices, src[2]=data)\n");
        पटिनिवत्त मिच्छा;
    }

    ध्रुव वर्ण * kernel_name;

    // Support F32 data with I64 indices -> F32/F16 output (scatter operation)
    सचे (ग्रन्थिः->src[0]->type == GGML_TYPE_F32 && ग्रन्थिः->src[1]->type == GGML_TYPE_I64 &&
        (ग्रन्थिः->type == GGML_TYPE_F32 || ग्रन्थिः->type == GGML_TYPE_F16)) {
        सचे (ग्रन्थिः->type == GGML_TYPE_F32 || ग्रन्थिः->type == GGML_TYPE_F16) {
            kernel_name = "set_rows_f32";
        } अञ्ञथा {
            GGML_LOG_ERROR("ET: SET_ROWS unsupported output type: %s\n", ggml_type_name(ग्रन्थिः->type));
            पटिनिवत्त मिच्छा;
        }

    } अञ्ञथा {
        GGML_LOG_ERROR("ET: SET_ROWS operation with unsupported types: dst=%s src0=%s src1=%s\n",
                       ggml_type_name(ग्रन्थिः->type), ggml_type_name(ग्रन्थिः->src[0]->type),
                       ggml_type_name(ग्रन्थिः->src[1]->type));
        पटिनिवत्त मिच्छा;
    }

    // Validate contiguity requirements
    सचे (!ggml_is_contiguous_rows(ग्रन्थिः)) {
        GGML_LOG_ERROR("ET: SET_ROWS operation requires contiguous-rows destination tensor\n");
        पटिनिवत्त मिच्छा;
    }

    सचे (!ggml_is_contiguous_rows(ग्रन्थिः->src[0])) {
        GGML_LOG_ERROR("ET: SET_ROWS operation requires contiguous-rows source tensor\n");
        पटिनिवत्त मिच्छा;
    }

    सचे (!ggml_is_contiguous(ग्रन्थिः->src[1])) {
        GGML_LOG_ERROR("ET: SET_ROWS operation requires contiguous indices tensor\n");
        पटिनिवत्त मिच्छा;
    }

    // Validate dimension constraints from ggml implementation
    सचे (!(ग्रन्थिः->ne[0] == ग्रन्थिः->src[0]->ne[0] &&              // same number of columns
          ग्रन्थिः->ne[2] == ग्रन्थिः->src[0]->ne[2] &&              // same batch size
          ग्रन्थिः->ne[3] == ग्रन्थिः->src[0]->ne[3] &&              // same outer dimension
          ग्रन्थिः->src[0]->ne[1] == ग्रन्थिः->src[1]->ne[0] &&      // src rows = index count
          ग्रन्थिः->src[0]->ne[2] % ग्रन्थिः->src[1]->ne[1] == 0 &&  // batch constraint
          ग्रन्थिः->src[0]->ne[3] % ग्रन्थिः->src[1]->ne[2] == 0 &&  // outer constraint
          ग्रन्थिः->src[1]->ne[3] == 1)) {                       // indices constraint
        GGML_LOG_ERROR("ET: SET_ROWS operation dimension constraint failed\n");
        पटिनिवत्त मिच्छा;
    }

    ggml_et_set_rows_params params;
    params.src0 = *ग्रन्थिः->src[0];  // F32 source data tensor
    params.src1 = *ग्रन्थिः->src[1];  // I64 indices tensor
    params.dst  = *ग्रन्थिः;          // F32/F16 destination tensor

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    तार्किक                    cpu_comparison_active = मिच्छा;
    सचे (set_rows_cpu_compare_config.enabled) {
        सचे (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ग्रन्थिः, GGML_OP_SET_ROWS)) {
            cpu_comparison_active = सच्च;
        } अञ्ञथा {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for SET_ROWS operation\n");
        }
    }

    तार्किक kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, परिमाण(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    सचे (cpu_comparison_active) {
        सचे (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ग्रन्थिः, &set_rows_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for SET_ROWS operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("SET_ROWS", kernel_name, ग्रन्थिः);
    पटिनिवत्त kernel_result;
}

तार्किक ggml_et_op_fill(ggml_backend_et_device_context * dev_ctx, ध्रुव ggml_tensor * ग्रन्थिः) {
    ET_PERF_START();

    ggml_et_fill_params params;
    params.dst = *ग्रन्थिः;
    memcpy(&params.c, ग्रन्थिः->op_params, परिमाण(भिन्न));

    तार्किक kernel_result = ggml_et_launch_kernel(dev_ctx, "fill_f32", &params, परिमाण(params), 0xFFFFFFFF);

    ET_PERF_END("FILL", "fill_f32", ग्रन्थिः);
    पटिनिवत्त kernel_result;
}

तार्किक ggml_et_op_diag(ggml_backend_et_device_context * dev_ctx, ध्रुव ggml_tensor * ग्रन्थिः) {
    ET_PERF_START();

    सचे (!ग्रन्थिः->src[0]) {
        GGML_LOG_ERROR("ET: DIAG operation missing source tensor\n");
        पटिनिवत्त मिच्छा;
    }

    ggml_et_diag_params params;
    params.src0 = *ग्रन्थिः->src[0];
    params.dst  = *ग्रन्थिः;

    तार्किक kernel_result = ggml_et_launch_kernel(dev_ctx, "diag_f32", &params, परिमाण(params), 0xFFFFFFFF);

    ET_PERF_END("DIAG", "diag_f32", ग्रन्थिः);
    पटिनिवत्त kernel_result;
}

तार्किक ggml_et_op_tri(ggml_backend_et_device_context * dev_ctx, ध्रुव ggml_tensor * ग्रन्थिः) {
    ET_PERF_START();

    सचे (!ग्रन्थिः->src[0]) {
        GGML_LOG_ERROR("ET: TRI operation missing source tensor\n");
        पटिनिवत्त मिच्छा;
    }

    ggml_et_tri_params params;
    params.src0 = *ग्रन्थिः->src[0];
    params.dst  = *ग्रन्थिः;
    memcpy(&params.tri_type, ग्रन्थिः->op_params, परिमाण(int32_t));

    तार्किक kernel_result = ggml_et_launch_kernel(dev_ctx, "tri_f32", &params, परिमाण(params), 0xFFFFFFFF);

    ET_PERF_END("TRI", "tri_f32", ग्रन्थिः);
    पटिनिवत्त kernel_result;
}

तार्किक ggml_et_op_solve_tri(ggml_backend_et_device_context * dev_ctx, ध्रुव ggml_tensor * ग्रन्थिः) {
    ET_PERF_START();

    सचे (!ग्रन्थिः->src[0] || !ग्रन्थिः->src[1]) {
        GGML_LOG_ERROR("ET: SOLVE_TRI operation missing source tensor(s)\n");
        पटिनिवत्त मिच्छा;
    }

    ggml_et_solve_tri_params params;
    params.src0 = *ग्रन्थिः->src[0];  // A (lower-triangular)
    params.src1 = *ग्रन्थिः->src[1];  // B (RHS)
    params.dst  = *ग्रन्थिः;          // X (solution)

    तार्किक kernel_result = ggml_et_launch_kernel(dev_ctx, "solve_tri_f32", &params, परिमाण(params), 0xFFFFFFFF);

    ET_PERF_END("SOLVE_TRI", "solve_tri_f32", ग्रन्थिः);
    पटिनिवत्त kernel_result;
}

तार्किक ggml_et_op_set(ggml_backend_et_device_context * dev_ctx, ध्रुव ggml_tensor * ग्रन्थिः) {
    ET_PERF_START();

    सचे (!ग्रन्थिः->src[0] || !ग्रन्थिः->src[1]) {
        GGML_LOG_ERROR("ET: SET operation missing source tensor(s)\n");
        पटिनिवत्त मिच्छा;
    }

    ध्रुव तार्किक   inplace = (तार्किक) ((ध्रुव int32_t *) ग्रन्थिः->op_params)[4];
    ध्रुव size_t offset  = ((ध्रुव int32_t *) ग्रन्थिः->op_params)[3];
    ध्रुव size_t nb1     = ((ध्रुव int32_t *) ग्रन्थिः->op_params)[0];
    ध्रुव size_t nb2     = ((ध्रुव int32_t *) ग्रन्थिः->op_params)[1];
    ध्रुव size_t nb3     = ((ध्रुव int32_t *) ग्रन्थिः->op_params)[2];

    सचे (!inplace) {
        GGML_LOG_ERROR("ET: SET only supports inplace (inplace=%d)\n", inplace);
        पटिनिवत्त मिच्छा;
    }

    सचे (ग्रन्थिः->type != GGML_TYPE_F32 || ग्रन्थिः->src[0]->type != GGML_TYPE_F32 || ग्रन्थिः->src[1]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: SET only supports F32 (dst=%s src0=%s src1=%s)\n", ggml_type_name(ग्रन्थिः->type),
                       ggml_type_name(ग्रन्थिः->src[0]->type), ggml_type_name(ग्रन्थिः->src[1]->type));
        पटिनिवत्त मिच्छा;
    }

    सचे (!ggml_are_same_shape(ग्रन्थिः, ग्रन्थिः->src[0])) {
        GGML_LOG_ERROR("ET: SET requires same-shape src0 and dst\n");
        पटिनिवत्त मिच्छा;
    }

    सचे (!ggml_is_contiguous(ग्रन्थिः) || !ggml_is_contiguous(ग्रन्थिः->src[0]) || !ggml_is_contiguous(ग्रन्थिः->src[1])) {
        GGML_LOG_ERROR("ET: SET requires contiguous dst, src0, and src1\n");
        पटिनिवत्त मिच्छा;
    }

    ggml_et_set_params params;
    params.src1   = *ग्रन्थिः->src[1];
    params.dst    = *ग्रन्थिः;
    params.nb1    = (int32_t) nb1;
    params.nb2    = (int32_t) nb2;
    params.nb3    = (int32_t) nb3;
    params.offset = (int32_t) offset;

    तार्किक kernel_result = ggml_et_launch_kernel(dev_ctx, "set_f32", &params, परिमाण(params), 0xFFFFFFFF);

    ET_PERF_END("SET", "set_f32", ग्रन्थिः);
    पटिनिवत्त kernel_result;
}

तार्किक ggml_et_op_pad(ggml_backend_et_device_context * dev_ctx, ध्रुव ggml_tensor * ग्रन्थिः) {
    ET_PERF_START();

    सचे (!ग्रन्थिः->src[0]) {
        GGML_LOG_ERROR("ET: PAD operation missing source tensor\n");
        पटिनिवत्त मिच्छा;
    }

    सचे (ग्रन्थिः->type != GGML_TYPE_F32 || ग्रन्थिः->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: PAD only supports F32 (src=%s dst=%s)\n", ggml_type_name(ग्रन्थिः->src[0]->type),
                       ggml_type_name(ग्रन्थिः->type));
        पटिनिवत्त मिच्छा;
    }

    सचे (!ggml_is_contiguous(ग्रन्थिः)) {
        GGML_LOG_ERROR("ET: PAD requires contiguous output tensor\n");
        पटिनिवत्त मिच्छा;
    }

    सचे (ग्रन्थिः->src[0]->nb[0] != परिमाण(भिन्न)) {
        GGML_LOG_ERROR("ET: PAD requires element-contiguous src dim0 (nb[0]=%zu)\n", (size_t) ग्रन्थिः->src[0]->nb[0]);
        पटिनिवत्त मिच्छा;
    }

    // Extract padding parameters from op_params
    ध्रुव int32_t * op_params = (ध्रुव int32_t *) ग्रन्थिः->op_params;

    ggml_et_pad_params params;
    params.src0  = *ग्रन्थिः->src[0];
    params.dst   = *ग्रन्थिः;
    params.lp[0] = op_params[0];
    params.rp[0] = op_params[1];
    params.lp[1] = op_params[2];
    params.rp[1] = op_params[3];
    params.lp[2] = op_params[4];
    params.rp[2] = op_params[5];
    params.lp[3] = op_params[6];
    params.rp[3] = op_params[7];

    // v1: no dim0 padding
    सचे (params.lp[0] != 0 || params.rp[0] != 0) {
        GGML_LOG_ERROR("ET: PAD dim0 padding not supported (lp0=%d rp0=%d)\n", params.lp[0], params.rp[0]);
        पटिनिवत्त मिच्छा;
    }

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    तार्किक                    cpu_comparison_active = मिच्छा;
    सचे (pad_cpu_compare_config.enabled) {
        सचे (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ग्रन्थिः, GGML_OP_PAD)) {
            cpu_comparison_active = सच्च;
        } अञ्ञथा {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for PAD operation\n");
        }
    }

    तार्किक kernel_result = ggml_et_launch_kernel(dev_ctx, "pad_f32", &params, परिमाण(params), 0xFFFFFFFF);

    सचे (cpu_comparison_active) {
        सचे (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ग्रन्थिः, &pad_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for PAD operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("PAD", "pad_f32", ग्रन्थिः);
    पटिनिवत्त kernel_result;
}
