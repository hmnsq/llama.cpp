#அந்தர்பாவ "ggml-et-ops.h"

#அந்தர்பாவ "ggml-et-cpu-compare.h"
#அந்தர்பாவ "ggml-et-kernels.h"
#அந்தர்பாவ "ggml-impl.h"

#அந்தர்பாவ <stdio.h>

#அந்தர்பாவ <cstdint>

// CPU comparison configuration - can be enabled for debugging
நிலையான ggml_et_cpu_compare_config rope_cpu_compare_config = {
    /* .enabled = */ பொய்,
    /* .use_cpu_result = */ பொய்,  // Replace ET result with CPU result
    /* .log_differences = */ உண்மை,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

நிலையான ggml_et_cpu_compare_config rms_norm_cpu_compare_config = {
    /* .enabled = */ பொய்,
    /* .use_cpu_result = */ பொய்,
    /* .log_differences = */ உண்மை,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

நிலையான ggml_et_cpu_compare_config norm_cpu_compare_config = {
    /* .enabled = */ பொய்,
    /* .use_cpu_result = */ பொய்,
    /* .log_differences = */ உண்மை,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

நிலையான ggml_et_cpu_compare_config l2_norm_cpu_compare_config = {
    /* .enabled = */ பொய்,
    /* .use_cpu_result = */ பொய்,
    /* .log_differences = */ உண்மை,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

நிலையான ggml_et_cpu_compare_config group_norm_cpu_compare_config = {
    /* .enabled = */ பொய்,
    /* .use_cpu_result = */ பொய்,
    /* .log_differences = */ உண்மை,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

நிலையான ggml_et_cpu_compare_config im2col_cpu_compare_config = {
    /* .enabled = */ பொய்,
    /* .use_cpu_result = */ பொய்,
    /* .log_differences = */ உண்மை,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

நிலையான ggml_et_cpu_compare_config unary_cpu_compare_config = {
    /* .enabled = */ பொய்,
    /* .use_cpu_result = */ பொய்,
    /* .log_differences = */ உண்மை,
    /* .tolerance = */ 1e-4f,
    /* .max_log_elements = */ 4096
};

நிலையான ggml_et_cpu_compare_config sum_rows_cpu_compare_config = {
    /* .enabled = */ பொய்,
    /* .use_cpu_result = */ பொய்,
    /* .log_differences = */ உண்மை,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

நிலையான ggml_et_cpu_compare_config clamp_cpu_compare_config = {
    /* .enabled = */ பொய்,
    /* .use_cpu_result = */ பொய்,
    /* .log_differences = */ உண்மை,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

நிலையான ggml_et_cpu_compare_config mean_cpu_compare_config = {
    /* .enabled = */ பொய்,
    /* .use_cpu_result = */ பொய்,
    /* .log_differences = */ உண்மை,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

நிலையான ggml_et_cpu_compare_config sqr_cpu_compare_config = {
    /* .enabled = */ பொய்,
    /* .use_cpu_result = */ பொய்,
    /* .log_differences = */ உண்மை,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

நிலையான ggml_et_cpu_compare_config elmap_cpu_compare_config = {
    /* .enabled = */ பொய்,
    /* .use_cpu_result = */ பொய்,
    /* .log_differences = */ உண்மை,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

நிலையான ggml_et_cpu_compare_config glu_cpu_compare_config = {
    /* .enabled = */ பொய்,
    /* .use_cpu_result = */ பொய்,
    /* .log_differences = */ உண்மை,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

நிலையான ggml_et_cpu_compare_config mul_mat_cpu_compare_config = {
    /* .enabled = */ பொய்,
    /* .use_cpu_result = */ பொய்,
    /* .log_differences = */ உண்மை,
    /* .tolerance = */ 0.01,
    /* .max_log_elements = */ 4096
};

நிலையான ggml_et_cpu_compare_config mul_mat_id_cpu_compare_config = {
    /* .enabled = */ பொய்,
    /* .use_cpu_result = */ பொய்,
    /* .log_differences = */ உண்மை,
    /* .tolerance = */ 0.01,
    /* .max_log_elements = */ 4096
};

நிலையான ggml_et_cpu_compare_config softmax_cpu_compare_config = {
    /* .enabled = */ பொய்,
    /* .use_cpu_result = */ பொய்,
    /* .log_differences = */ உண்மை,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 1024
};

நிலையான ggml_et_cpu_compare_config get_rows_cpu_compare_config = {
    /* .enabled = */ பொய்,
    /* .use_cpu_result = */ பொய்,
    /* .log_differences = */ உண்மை,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 2048
};

நிலையான ggml_et_cpu_compare_config pad_cpu_compare_config = {
    /* .enabled = */ பொய்,
    /* .use_cpu_result = */ பொய்,
    /* .log_differences = */ உண்மை,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

நிலையான ggml_et_cpu_compare_config cont_cpu_compare_config = {
    /* .enabled = */ பொய்,
    /* .use_cpu_result = */ பொய்,
    /* .log_differences = */ உண்மை,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

நிலையான ggml_et_cpu_compare_config concat_cpu_compare_config = {
    /* .enabled = */ பொய்,
    /* .use_cpu_result = */ பொய்,
    /* .log_differences = */ உண்மை,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

நிலையான ggml_et_cpu_compare_config cumsum_cpu_compare_config = {
    /* .enabled = */ பொய்,
    /* .use_cpu_result = */ பொய்,
    /* .log_differences = */ உண்மை,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

நிலையான ggml_et_cpu_compare_config repeat_cpu_compare_config = {
    /* .enabled = */ பொய்,
    /* .use_cpu_result = */ பொய்,
    /* .log_differences = */ உண்மை,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

நிலையான ggml_et_cpu_compare_config ssm_conv_cpu_compare_config = {
    /* .enabled = */ பொய்,
    /* .use_cpu_result = */ பொய்,
    /* .log_differences = */ உண்மை,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

நிலையான ggml_et_cpu_compare_config rwkv_wkv6_cpu_compare_config = {
    /* .enabled = */ பொய்,
    /* .use_cpu_result = */ பொய்,
    /* .log_differences = */ உண்மை,
    /* .tolerance = */ 1e-4f,
    /* .max_log_elements = */ 4096
};

நிலையான ggml_et_cpu_compare_config rwkv_wkv7_cpu_compare_config = {
    /* .enabled = */ பொய்,
    /* .use_cpu_result = */ பொய்,
    /* .log_differences = */ உண்மை,
    /* .tolerance = */ 1e-4f,
    /* .max_log_elements = */ 4096
};

நிலையான ggml_et_cpu_compare_config set_rows_cpu_compare_config = {
    /* .enabled = */ பொய்,
    /* .use_cpu_result = */ பொய்,
    /* .log_differences = */ உண்மை,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 2048
};

தருக்க ggml_et_op_rms_norm_mul(ggml_backend_et_device_context * dev_ctx,
                             மாறிலி ggml_tensor *              rms_norm_முனை,
                             மாறிலி ggml_tensor *              mul_முனை) {
    ET_PERF_START();

    எனில் (!dev_ctx || !rms_norm_முனை || !mul_முனை) {
        GGML_LOG_ERROR("ET: Invalid parameters for fused RMS_NORM_MUL operation\n");
        திரும்பு பொய்;
    }

    எனில் (!rms_norm_முனை->src[0]) {
        GGML_LOG_ERROR("ET: Fused RMS_NORM_MUL missing required input\n");
        திரும்பு பொய்;
    }

    // Extract weights: the MUL operand that isn't the rms_norm output
    மாறிலி ggml_tensor * weights = (mul_முனை->src[0] == rms_norm_முனை) ? mul_முனை->src[1] : mul_முனை->src[0];

    எனில் (!weights) {
        GGML_LOG_ERROR("ET: Fused RMS_NORM_MUL missing weights tensor\n");
        திரும்பு பொய்;
    }

    தசம eps;
    memcpy(&eps, rms_norm_முனை->op_params, அளவு(தசம));

    ggml_et_rms_norm_mul_params params;
    params.src0 = *rms_norm_முனை->src[0];  // input to normalize
    params.src1 = *weights;                // normalization weights
    params.dst  = *mul_முனை;               // final output
    params.eps  = eps;

    தருக்க kernel_result = ggml_et_launch_kernel(dev_ctx, "rms_norm_mul_f32", &params, அளவு(params), 0xFFFFFFFF);

    ET_PERF_END_EXT("RMS_NORM_MUL", "rms_norm_mul_f32", mul_முனை, "eps=%.6f", (இரட்டை) eps);
    திரும்பு kernel_result;
}

தருக்க ggml_et_op_scale(ggml_backend_et_device_context * dev_ctx, மாறிலி ggml_tensor * முனை) {
    ET_PERF_START();

    எனில் (!dev_ctx || !முனை) {
        GGML_LOG_ERROR("ET: Invalid parameters for SCALE operation\n");
        திரும்பு பொய்;
    }

    எனில் (!முனை->src[0]) {
        GGML_LOG_ERROR("ET: SCALE operation missing required input\n");
        திரும்பு பொய்;
    }

    எனில் (முனை->type != GGML_TYPE_F32 || முனை->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: SCALE operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(முனை->type),
                       ggml_type_name(முனை->src[0]->type));
        திரும்பு பொய்;
    }

    தசம scale, bias;
    memcpy(&scale, (மாறிலி தசம *) முனை->op_params + 0, அளவு(தசம));
    memcpy(&bias, (மாறிலி தசம *) முனை->op_params + 1, அளவு(தசம));

    ggml_et_scale_params params;
    params.src0  = *முனை->src[0];
    params.dst   = *முனை;
    params.scale = scale;
    params.bias  = bias;

    தருக்க kernel_result = ggml_et_launch_kernel(dev_ctx, "scale_f32", &params, அளவு(params), 0xFFFFFFFF);

    ET_PERF_END_EXT("SCALE", "scale_f32", முனை, "scale=%.6f|bias=%.6f", (இரட்டை) scale, (இரட்டை) bias);
    திரும்பு kernel_result;
}

தருக்க ggml_et_op_sqr(ggml_backend_et_device_context * dev_ctx, மாறிலி ggml_tensor * முனை) {
    ET_PERF_START();

    எனில் (!dev_ctx || !முனை) {
        GGML_LOG_ERROR("ET: Invalid parameters for SQR operation\n");
        திரும்பு பொய்;
    }

    எனில் (!முனை->src[0]) {
        GGML_LOG_ERROR("ET: SQR operation missing required input\n");
        திரும்பு பொய்;
    }

    எனில் (முனை->type != GGML_TYPE_F32 || முனை->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: SQR operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(முனை->type),
                       ggml_type_name(முனை->src[0]->type));
        திரும்பு பொய்;
    }

    ggml_et_sqr_params params;
    params.src0 = *முனை->src[0];  // F32 input tensor
    params.dst  = *முனை;          // F32 output tensor

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    தருக்க                    cpu_comparison_active = பொய்;
    எனில் (sqr_cpu_compare_config.enabled) {
        எனில் (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, முனை, GGML_OP_SQR)) {
            cpu_comparison_active = உண்மை;
        } இல்லையெனில் {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for SQR operation\n");
        }
    }

    தருக்க kernel_result = ggml_et_launch_kernel(dev_ctx, "sqr_f32", &params, அளவு(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    எனில் (cpu_comparison_active) {
        எனில் (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, முனை, &sqr_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for SQR operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("SQR", "sqr_f32", முனை);
    திரும்பு kernel_result;
}

தருக்க ggml_et_op_sum_rows(ggml_backend_et_device_context * dev_ctx, மாறிலி ggml_tensor * முனை) {
    ET_PERF_START();

    எனில் (!dev_ctx || !முனை) {
        GGML_LOG_ERROR("ET: Invalid parameters for SUM_ROWS operation\n");
        திரும்பு பொய்;
    }

    எனில் (!முனை->src[0]) {
        GGML_LOG_ERROR("ET: SUM_ROWS operation missing required input\n");
        திரும்பு பொய்;
    }

    எனில் (முனை->type != GGML_TYPE_F32 || முனை->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: SUM_ROWS operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(முனை->type),
                       ggml_type_name(முனை->src[0]->type));
        திரும்பு பொய்;
    }

    ggml_et_sum_rows_params params;
    params.src0 = *முனை->src[0];
    params.dst  = *முனை;

    // Phase 1: Initialize CPU comparison context
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    தருக்க                    cpu_comparison_active = பொய்;
    எனில் (sum_rows_cpu_compare_config.enabled) {
        எனில் (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, முனை, GGML_OP_SUM_ROWS)) {
            cpu_comparison_active = உண்மை;
        } இல்லையெனில் {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for SUM_ROWS operation\n");
        }
    }

    தருக்க kernel_result = ggml_et_launch_kernel(dev_ctx, "sum_rows_f32", &params, அளவு(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare
    எனில் (cpu_comparison_active) {
        எனில் (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, முனை, &sum_rows_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for SUM_ROWS operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("SUM_ROWS", "sum_rows_f32", முனை);
    திரும்பு kernel_result;
}

தருக்க ggml_et_op_mean(ggml_backend_et_device_context * dev_ctx, மாறிலி ggml_tensor * முனை) {
    ET_PERF_START();

    எனில் (!dev_ctx || !முனை) {
        GGML_LOG_ERROR("ET: Invalid parameters for MEAN operation\n");
        திரும்பு பொய்;
    }

    எனில் (!முனை->src[0]) {
        GGML_LOG_ERROR("ET: MEAN operation missing required input\n");
        திரும்பு பொய்;
    }

    எனில் (முனை->type != GGML_TYPE_F32 || முனை->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: MEAN operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(முனை->type),
                       ggml_type_name(முனை->src[0]->type));
        திரும்பு பொய்;
    }

    ggml_et_mean_params params;
    params.src0 = *முனை->src[0];
    params.dst  = *முனை;

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    தருக்க                    cpu_comparison_active = பொய்;
    எனில் (mean_cpu_compare_config.enabled) {
        எனில் (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, முனை, GGML_OP_MEAN)) {
            cpu_comparison_active = உண்மை;
        } இல்லையெனில் {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for MEAN operation\n");
        }
    }

    தருக்க kernel_result = ggml_et_launch_kernel(dev_ctx, "mean_f32", &params, அளவு(params), 0xFFFFFFFF);

    எனில் (cpu_comparison_active) {
        எனில் (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, முனை, &mean_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for MEAN operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("MEAN", "mean_f32", முனை);
    திரும்பு kernel_result;
}

தருக்க ggml_et_op_clamp(ggml_backend_et_device_context * dev_ctx, மாறிலி ggml_tensor * முனை) {
    ET_PERF_START();

    எனில் (!dev_ctx || !முனை) {
        GGML_LOG_ERROR("ET: Invalid parameters for CLAMP operation\n");
        திரும்பு பொய்;
    }

    எனில் (!முனை->src[0]) {
        GGML_LOG_ERROR("ET: CLAMP operation missing required input\n");
        திரும்பு பொய்;
    }

    எனில் (முனை->type != GGML_TYPE_F32 || முனை->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: CLAMP operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(முனை->type),
                       ggml_type_name(முனை->src[0]->type));
        திரும்பு பொய்;
    }

    ggml_et_clamp_params params;
    params.src0 = *முனை->src[0];
    params.dst  = *முனை;
    // op_params layout per ggml.c::ggml_clamp: { min, max } as floats
    memcpy(&params.min_val, (மாறிலி தசம *) முனை->op_params + 0, அளவு(தசம));
    memcpy(&params.max_val, (மாறிலி தசம *) முனை->op_params + 1, அளவு(தசம));

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    தருக்க                    cpu_comparison_active = பொய்;
    எனில் (clamp_cpu_compare_config.enabled) {
        எனில் (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, முனை, GGML_OP_CLAMP)) {
            cpu_comparison_active = உண்மை;
        } இல்லையெனில் {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for CLAMP operation\n");
        }
    }

    தருக்க kernel_result = ggml_et_launch_kernel(dev_ctx, "clamp_f32", &params, அளவு(params), 0xFFFFFFFF);

    எனில் (cpu_comparison_active) {
        எனில் (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, முனை, &clamp_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for CLAMP operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("CLAMP", "clamp_f32", முனை);
    திரும்பு kernel_result;
}

தருக்க ggml_et_op_unary(ggml_backend_et_device_context * dev_ctx, மாறிலி ggml_tensor * முனை) {
    ET_PERF_START();

    எனில் (!dev_ctx || !முனை) {
        GGML_LOG_ERROR("ET: Invalid parameters for UNARY operation\n");
        திரும்பு பொய்;
    }

    எனில் (!முனை->src[0]) {
        GGML_LOG_ERROR("ET: UNARY operation missing required input\n");
        திரும்பு பொய்;
    }

    எனில் (முனை->type != GGML_TYPE_F32 || முனை->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: UNARY operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(முனை->type),
                       ggml_type_name(முனை->src[0]->type));
        திரும்பு பொய்;
    }

    மாறிலி ggml_unary_op uop     = ggml_get_unary_op(முனை);
    மாறிலி எழுத்து *        op_name = ggml_unary_op_name(uop);

    ggml_et_unary_params params;
    params.src0     = *முனை->src[0];  // F32 input tensor
    params.dst      = *முனை;          // F32 output tensor
    params.unary_op = (int32_t) uop;

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    தருக்க                    cpu_comparison_active = பொய்;
    எனில் (unary_cpu_compare_config.enabled) {
        எனில் (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, முனை, GGML_OP_UNARY)) {
            cpu_comparison_active = உண்மை;
        } இல்லையெனில் {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for UNARY/%s operation\n", op_name);
        }
    }

    தருக்க kernel_result = ggml_et_launch_kernel(dev_ctx, "unary_f32", &params, அளவு(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    எனில் (cpu_comparison_active) {
        எனில் (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, முனை, &unary_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for UNARY/%s operation\n", op_name);
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END_EXT("UNARY", "unary_f32", முனை, "op=%s", op_name);
    திரும்பு kernel_result;
}

தருக்க ggml_et_op_mul(ggml_backend_et_device_context * dev_ctx, மாறிலி ggml_tensor * முனை) {
    // Delegate to generic element map operation
    திரும்பு ggml_et_op_elmap(dev_ctx, முனை);
}

தருக்க ggml_et_op_add(ggml_backend_et_device_context * dev_ctx, மாறிலி ggml_tensor * முனை) {
    // Delegate to generic element map operation
    திரும்பு ggml_et_op_elmap(dev_ctx, முனை);
}

தருக்க ggml_et_op_sub(ggml_backend_et_device_context * dev_ctx, மாறிலி ggml_tensor * முனை) {
    // Delegate to generic element map operation
    திரும்பு ggml_et_op_elmap(dev_ctx, முனை);
}

தருக்க ggml_et_op_elmap(ggml_backend_et_device_context * dev_ctx, மாறிலி ggml_tensor * முனை) {
    ET_PERF_START();

    எனில் (!dev_ctx || !முனை) {
        GGML_LOG_ERROR("ET: Invalid parameters for element map operation\n");
        திரும்பு பொய்;
    }

    எனில் (!முனை->src[0] || !முனை->src[1]) {
        GGML_LOG_ERROR("ET: Element map operation missing required inputs\n");
        திரும்பு பொய்;
    }

    எனில் (முனை->type != GGML_TYPE_F32 || முனை->src[0]->type != GGML_TYPE_F32 || முனை->src[1]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: Element map operation with unsupported types: dst=%s src0=%s src1=%s\n",
                       ggml_type_name(முனை->type), ggml_type_name(முனை->src[0]->type),
                       ggml_type_name(முனை->src[1]->type));
        திரும்பு பொய்;
    }

    மாறிலி எழுத்து * op_name = ggml_op_name(முனை->op);

    ggml_et_elmap_params params;
    params.src0 = *முனை->src[0];
    params.src1 = *முனை->src[1];
    params.dst  = *முனை;  // F32 output tensor (op type stored in dst.op)

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    தருக்க                    cpu_comparison_active = பொய்;
    எனில் (elmap_cpu_compare_config.enabled) {
        எனில் (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, முனை, முனை->op)) {
            cpu_comparison_active = உண்மை;
        } இல்லையெனில் {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for %s operation\n", op_name);
        }
    }

    // fprintf(stderr, "ET: el_map s0 [%ld, %ld, %ld, %ld] s1 [%ld, %ld, %ld, %ld]\n",
    //     node->src[0]->ne[0], node->src[0]->ne[1], node->src[0]->ne[2], node->src[0]->ne[3],
    //     node->src[1]->ne[0], node->src[1]->ne[1], node->src[1]->ne[2], node->src[1]->ne[3]);

    தருக்க kernel_result = ggml_et_launch_kernel(dev_ctx, "el_map_f32", &params, அளவு(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    எனில் (cpu_comparison_active) {
        எனில் (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, முனை, &elmap_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for %s operation\n", op_name);
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END(op_name, "el_map_f32", முனை);
    திரும்பு kernel_result;
}

தருக்க ggml_et_op_glu(ggml_backend_et_device_context * dev_ctx, மாறிலி ggml_tensor * முனை) {
    ET_PERF_START();

    // Validate inputs
    எனில் (!dev_ctx || !முனை) {
        GGML_LOG_ERROR("ET: Invalid parameters for GLU operation\n");
        திரும்பு பொய்;
    }

    எனில் (!முனை->src[0]) {
        GGML_LOG_ERROR("ET: GLU operation missing required input\n");
        திரும்பு பொய்;
    }

    மாறிலி தருக்க is_split_mode = முனை->src[1] != nullptr;

    // Only support F32 (as validated by supports_op)
    எனில் (முனை->type != GGML_TYPE_F32 || முனை->src[0]->type != GGML_TYPE_F32 ||
        (is_split_mode && முனை->src[1]->type != GGML_TYPE_F32)) {
        திரும்பு பொய்;
    }

    // Extract GLU operation parameters from op_params
    int32_t glu_op_type = ggml_get_op_params_i32(முனை, 0);  // GLU variant (REGLU, GEGLU, SWIGLU, etc.)
    int32_t swapped     = ggml_get_op_params_i32(முனை, 1);  // Whether gate/value are swapped

    // Supported variants
    தேர்வு (glu_op_type) {
        நிலை GGML_GLU_OP_REGLU:
        நிலை GGML_GLU_OP_GEGLU:
        நிலை GGML_GLU_OP_SWIGLU:
        நிலை GGML_GLU_OP_SWIGLU_OAI:
        நிலை GGML_GLU_OP_SWIGLU_CLAMP:
        நிலை GGML_GLU_OP_GEGLU_ERF:
        நிலை GGML_GLU_OP_GEGLU_QUICK:
            நிறுத்து;
        இயல்பு:
            GGML_LOG_ERROR("ET: GLU operation with unsupported variant: %s\n",
                           ggml_glu_op_name((ggml_glu_op) glu_op_type));
            திரும்பு பொய்;
    }

    // Get GLU operation name for logging
    மாறிலி எழுத்து * glu_op_name = ggml_glu_op_name((ggml_glu_op) glu_op_type);

    // Pack parameters. Single-tensor mode is encoded by zeroing src1.
    ggml_et_glu_params params = {};
    params.src0               = *முனை->src[0];
    எனில் (is_split_mode) {
        params.src1 = *முனை->src[1];
    }
    params.dst         = *முனை;
    params.glu_op_type = glu_op_type;
    params.swapped     = swapped;
    params.alpha       = 0.0f;
    params.limit       = 0.0f;
    எனில் (glu_op_type == GGML_GLU_OP_SWIGLU_OAI) {
        params.alpha = ggml_get_op_params_f32(முனை, 2);
    }
    எனில் (glu_op_type == GGML_GLU_OP_SWIGLU_OAI || glu_op_type == GGML_GLU_OP_SWIGLU_CLAMP) {
        params.limit = ggml_get_op_params_f32(முனை, 3);
    }
    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    தருக்க                    cpu_comparison_active = பொய்;
    எனில் (glu_cpu_compare_config.enabled) {
        எனில் (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, முனை, GGML_OP_GLU)) {
            cpu_comparison_active = உண்மை;
        } இல்லையெனில் {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for %s operation\n", glu_op_name);
        }
    }

    // Launch ET kernel
    தருக்க kernel_result = ggml_et_launch_kernel(dev_ctx, "glu_f32", &params, அளவு(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    எனில் (cpu_comparison_active) {
        எனில் (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, முனை, &glu_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for %s operation\n", glu_op_name);
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("GLU", "glu_f32", முனை);
    திரும்பு kernel_result;
}

தருக்க ggml_et_op_mul_mat(ggml_backend_et_device_context * dev_ctx,
                        மாறிலி ggml_tensor *              முனை,
                        மாறிலி ggml_tensor *              add_முனை) {
    ET_PERF_START();

    எனில் (!dev_ctx || !முனை) {
        GGML_LOG_ERROR("ET: Invalid parameters for MUL_MAT operation\n");
        திரும்பு பொய்;
    }

    எனில் (!முனை->src[0] || !முனை->src[1]) {
        GGML_LOG_ERROR("ET: MUL_MAT operation missing required inputs\n");
        திரும்பு பொய்;
    }

    // Fused MM+ADD: when add_node is non-NULL the caller has already validated
    // (Q8_0 weights, F32 acts, exact-shape ADD with stride parity to dst) via
    // ggml_et_can_fuse({MUL_MAT, ADD}). The kernel writes dst = mm + bias and
    // the ADD's output replaces MM's as the actual dst.
    மாறிலி ggml_tensor * fused_dst   = add_முனை ? add_முனை : முனை;
    மாறிலி ggml_tensor * bias_tensor = nullptr;
    எனில் (add_முனை) {
        bias_tensor = (add_முனை->src[0] == முனை) ? add_முனை->src[1] : add_முனை->src[0];
    }

    மாறிலி எழுத்து * kernel_name;
    மாறிலி எழுத்து * src0_type_name;

    எனில் (முனை->type == GGML_TYPE_F32 && முனை->src[0]->type == GGML_TYPE_Q4_0 && முனை->src[1]->type == GGML_TYPE_F32 &&
        முனை->src[1]->ne[1] >= 53 &&      // N >= 53
        முனை->src[0]->ne[1] % 16 == 0 &&  // M % TILE_M
        முனை->src[0]->ne[0] % 32 == 0) {  // K % BLOCK_K (Q4_0 block)

        // Matrix engine for N >= 53; partial N (via n_cur-1) and errata padding are handled in-kernel.
        kernel_name    = "mul_mat_Q4_0_matrix_engine";
        src0_type_name = "Q4_0";

    } இல்லையெனில் எனில் (முனை->type == GGML_TYPE_F32 && முனை->src[0]->type == GGML_TYPE_Q4_0 &&
               முனை->src[1]->type == GGML_TYPE_F32) {
        kernel_name    = "mul_mat_Q4_0";  // N < 53, or M % 16 != 0 or K % 32 != 0
        src0_type_name = "Q4_0";

    } இல்லையெனில் எனில் (முனை->type == GGML_TYPE_F32 && முனை->src[0]->type == GGML_TYPE_Q8_0 &&
               முனை->src[1]->type == GGML_TYPE_F32) {
        kernel_name    = "mul_mat_Q8_0";
        src0_type_name = "Q8_0";

    } இல்லையெனில் எனில் (முனை->type == GGML_TYPE_F32 && முனை->src[0]->type == GGML_TYPE_F16 &&
               முனை->src[1]->type == GGML_TYPE_F16 && முனை->ne[0] % 16 == 0 && முனை->src[0]->ne[0] % 16 == 0 &&
               முனை->src[0]->ne[1] % 16 == 0 && முனை->src[1]->ne[0] != 1) {
        kernel_name    = "mul_mat_f16_matrix_engine";
        src0_type_name = "F16";

    } இல்லையெனில் எனில் (முனை->type == GGML_TYPE_F32 && முனை->src[0]->type == GGML_TYPE_F16 &&
               (முனை->src[1]->type == GGML_TYPE_F16 || முனை->src[1]->type == GGML_TYPE_F32)) {
        kernel_name    = "mul_mat_f16";
        src0_type_name = "F16";

    } இல்லையெனில் எனில் (முனை->type == GGML_TYPE_F32 && முனை->src[0]->type == GGML_TYPE_F32 &&
               முனை->src[1]->type == GGML_TYPE_F32 && முனை->ne[0] % 16 == 0 && முனை->src[0]->ne[0] % 16 == 0 &&
               முனை->src[0]->ne[1] % 16 == 0 && முனை->src[1]->ne[0] != 1) {  // GEMV is faster with the generic path

        kernel_name    = "mul_mat_f32_matrix_engine";
        src0_type_name = "F32";
    } இல்லையெனில் எனில் (முனை->type == GGML_TYPE_F32 && முனை->src[0]->type == GGML_TYPE_F32 &&
               (முனை->src[1]->type == GGML_TYPE_F16 || முனை->src[1]->type == GGML_TYPE_F32)) {
        kernel_name    = "mul_mat_f32";
        src0_type_name = "F32";
    } இல்லையெனில் {
        GGML_LOG_ERROR("ET: MUL_MAT operation with unsupported types: dst=%s src0=%s src1=%s\n",
                       ggml_type_name(முனை->type), ggml_type_name(முனை->src[0]->type),
                       ggml_type_name(முனை->src[1]->type));
        திரும்பு பொய்;
    }

    ggml_et_இருமம்_params params;
    params.src0 = *முனை->src[0];  // weight matrix
    params.src1 = *முனை->src[1];  // activation matrix
    params.dst  = *fused_dst;     // output (= add_node when fused, else node)

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    தருக்க                    cpu_comparison_active = பொய்;
    எனில் (mul_mat_cpu_compare_config.enabled) {
        எனில் (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, fused_dst, GGML_OP_MUL_MAT)) {
            cpu_comparison_active = உண்மை;
        } இல்லையெனில் {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for MUL_MAT operation\n");
        }
    }

    தருக்க kernel_result;
    எனில் (முனை->src[0]->type == GGML_TYPE_Q8_0) {
        // Q8_0 kernel always takes the extended struct. bias.data is non-NULL
        // only on the fused path; otherwise the kernel skips the add entirely.
        ggml_et_mm_q8_params q8_params = {};
        q8_params.src0                 = params.src0;
        q8_params.src1                 = params.src1;
        q8_params.dst                  = params.dst;
        எனில் (bias_tensor) {
            q8_params.bias = *bias_tensor;
        }
        kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &q8_params, அளவு(q8_params), 0xFFFFFFFF);
    } இல்லையெனில் {
        // Non-Q8 MM kernels don't yet support fused-add; the graph fuse check
        // already rejects non-Q8 pairs, so add_node is always nullptr here.
        kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, அளவு(params), 0xFFFFFFFF);
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
    எனில் (cpu_comparison_active) {
        எனில் (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, fused_dst, &mul_mat_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for MUL_MAT operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    {
        // Calculate actual FLOPs including batch/sequence dimensions
        // dst shape: [M, N, ne2, ne3] where M=ne[1], N=ne[0]
        int64_t m   = முனை->ne[1];
        int64_t n   = முனை->ne[0];
        int64_t k   = முனை->src[0]->ne[0];
        int64_t ne2 = முனை->ne[2];
        int64_t ne3 = முனை->ne[3];

        // Total FLOPs = (batch_size) * M * N * (2*K - 1)
        // Each MxN matrix-matrix multiply does M*N*(2*K-1) FLOPs
        // Broadcasting is handled by repeating computation, so count actual operations
        int64_t batch_size  = ne2 * ne3;
        int64_t total_flops = batch_size * m * n * (2 * k - 1);

        எழுத்து kernel_variant[64];
        snprintf(kernel_variant, அளவு(kernel_variant), "%s_%sx%s", kernel_name, src0_type_name,
                 ggml_type_name(முனை->src[1]->type));
        ET_PERF_END_EXT("MUL_MAT", kernel_variant, முனை, "flops=%" PRId64, total_flops);
    }
    திரும்பு kernel_result;
}

தருக்க ggml_et_op_mul_mat_id(ggml_backend_et_device_context * dev_ctx, மாறிலி ggml_tensor * முனை) {
    ET_PERF_START();
    எனில் (!dev_ctx || !முனை) {
        GGML_LOG_ERROR("ET: Invalid parameters for MUL_MAT_ID operation\n");
        திரும்பு பொய்;
    }

    எனில் (!முனை->src[0] || !முனை->src[1] || !முனை->src[2]) {
        GGML_LOG_ERROR("ET: MUL_MAT_ID operation missing required inputs\n");
        திரும்பு பொய்;
    }

    மாறிலி எழுத்து * kernel_name;
    மாறிலி எழுத்து * src0_type_name;

    // Support Q8_0/Q4_0/F16/F32 x F32 -> F32 matrix multiplication with expert selection
    எனில் (முனை->type == GGML_TYPE_F32 && முனை->src[0]->type == GGML_TYPE_Q8_0 && முனை->src[1]->type == GGML_TYPE_F32 &&
        முனை->src[2]->type == GGML_TYPE_I32) {
        kernel_name    = "mul_mat_id_Q8_0";
        src0_type_name = "Q8_0";

    } இல்லையெனில் எனில் (முனை->type == GGML_TYPE_F32 && முனை->src[0]->type == GGML_TYPE_Q4_0 &&
               முனை->src[1]->type == GGML_TYPE_F32 && முனை->src[2]->type == GGML_TYPE_I32) {
        kernel_name    = "mul_mat_id_Q4_0";
        src0_type_name = "Q4_0";

    } இல்லையெனில் எனில் (முனை->type == GGML_TYPE_F32 && முனை->src[0]->type == GGML_TYPE_F16 &&
               முனை->src[1]->type == GGML_TYPE_F32 && முனை->src[2]->type == GGML_TYPE_I32) {
        kernel_name    = "mul_mat_id_f32";
        src0_type_name = "F16";

    } இல்லையெனில் எனில் (முனை->type == GGML_TYPE_F32 && முனை->src[0]->type == GGML_TYPE_F32 &&
               முனை->src[1]->type == GGML_TYPE_F32 && முனை->src[2]->type == GGML_TYPE_I32) {
        kernel_name    = "mul_mat_id_f32";
        src0_type_name = "F32";

    } இல்லையெனில் {
        GGML_LOG_ERROR("ET: MUL_MAT_ID operation with unsupported types: dst=%s src0=%s src1=%s src2=%s\n",
                       ggml_type_name(முனை->type), ggml_type_name(முனை->src[0]->type),
                       ggml_type_name(முனை->src[1]->type), ggml_type_name(முனை->src[2]->type));
        திரும்பு பொய்;
    }

    // Pack parameters - copy full tensor structures
    ggml_et_mul_mat_id_params params;
    params.src0 = *முனை->src[0];  // Expert weight matrices (Q8_0/F16/F32)
    params.src1 = *முனை->src[1];  // Activation matrix (F32)
    params.src2 = *முனை->src[2];  // Expert indices (I32)
    params.dst  = *முனை;          // Output matrix (F32)

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    தருக்க                    cpu_comparison_active = பொய்;
    எனில் (mul_mat_id_cpu_compare_config.enabled) {
        எனில் (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, முனை, GGML_OP_MUL_MAT_ID)) {
            cpu_comparison_active = உண்மை;
        } இல்லையெனில் {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for MUL_MAT_ID operation\n");
        }
    }

    // Launch ET kernel
    தருக்க kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, அளவு(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    எனில் (cpu_comparison_active) {
        எனில் (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, முனை, &mul_mat_id_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for MUL_MAT_ID operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    // Calculate FLOPs (approximate - similar to MUL_MAT but with expert routing overhead)
    // Each expert computation is similar to a MUL_MAT, but we only compute for selected experts
    int64_t K             = முனை->src[0]->ne[0];
    int64_t M             = முனை->src[0]->ne[1];
    int64_t n_expert_used = முனை->src[2]->ne[0];
    int64_t batch         = முனை->src[2]->ne[1];

    int64_t total_flops = batch * n_expert_used * M * (2 * K - 1);

    எழுத்து kernel_variant[64];
    snprintf(kernel_variant, அளவு(kernel_variant), "%s_%sx%s", kernel_name, src0_type_name,
             ggml_type_name(முனை->src[1]->type));
    ET_PERF_END_EXT("MUL_MAT_ID", kernel_variant, முனை, "flops=%" PRId64 "|n_expert=%lld|n_expert_used=%lld",
                    total_flops, (long long) முனை->src[0]->ne[2], (long long) n_expert_used);

    திரும்பு kernel_result;
}

தருக்க ggml_et_op_rope(ggml_backend_et_device_context * dev_ctx, மாறிலி ggml_tensor * முனை) {
    ET_PERF_START();

    எனில் (!dev_ctx || !முனை) {
        GGML_LOG_ERROR("ET: Invalid parameters for ROPE operation\n");
        திரும்பு பொய்;
    }

    எனில் (!முனை->src[0] || !முனை->src[1]) {
        GGML_LOG_ERROR("ET: ROPE operation missing required inputs\n");
        திரும்பு பொய்;
    }

    மாறிலி எழுத்து * kernel_name;

    எனில் (முனை->type == GGML_TYPE_F32 && முனை->src[0]->type == GGML_TYPE_F32 && முனை->src[1]->type == GGML_TYPE_I32) {
        kernel_name = "rope_f32";
    } இல்லையெனில் {
        திரும்பு பொய்;
    }

    // Pack parameters - copy full tensor structures and op_params
    ggml_et_rope_params params;
    params.src0 = *முனை->src[0];                       // F32 input tensor
    params.src1 = *முனை->src[1];                       // I32 position tensor
    எனில் (முனை->src[2]) {
        params.src2 = *முனை->src[2];                   // F32 frequency factors (optional)
    } இல்லையெனில் {
        memset(&params.src2, 0, அளவு(params.src2));  // Zero if not provided
    }
    params.dst = *முனை;                                // F32 output tensor

    params.rope_params.n_past     = ((மாறிலி int32_t *) முனை->op_params)[0];
    params.rope_params.n_dims     = ((மாறிலி int32_t *) முனை->op_params)[1];
    params.rope_params.mode       = ((மாறிலி int32_t *) முனை->op_params)[2];
    params.rope_params.n_ctx      = ((மாறிலி int32_t *) முனை->op_params)[3];
    params.rope_params.n_ctx_orig = ((மாறிலி int32_t *) முனை->op_params)[4];
    memcpy(&params.rope_params.freq_base, (மாறிலி int32_t *) முனை->op_params + 5, அளவு(தசம));
    memcpy(&params.rope_params.freq_scale, (மாறிலி int32_t *) முனை->op_params + 6, அளவு(தசம));
    memcpy(&params.rope_params.ext_factor, (மாறிலி int32_t *) முனை->op_params + 7, அளவு(தசம));
    memcpy(&params.rope_params.attn_factor, (மாறிலி int32_t *) முனை->op_params + 8, அளவு(தசம));
    memcpy(&params.rope_params.beta_fast, (மாறிலி int32_t *) முனை->op_params + 9, அளவு(தசம));
    memcpy(&params.rope_params.beta_slow, (மாறிலி int32_t *) முனை->op_params + 10, அளவு(தசம));
    எனில் (params.rope_params.mode & GGML_ROPE_TYPE_MROPE) {
        memcpy(params.rope_params.sections, (மாறிலி int32_t *) முனை->op_params + 11, அளவு(int32_t) * 4);
    } இல்லையெனில் {
        memset(params.rope_params.sections, 0, அளவு(params.rope_params.sections));
    }

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    தருக்க                    cpu_comparison_active = பொய்;
    எனில் (rope_cpu_compare_config.enabled) {
        GGML_LOG_DEBUG("ET: Initializing CPU comparison for ROPE operation\n");
        எனில் (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, முனை, GGML_OP_ROPE)) {
            cpu_comparison_active = உண்மை;
        } இல்லையெனில் {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for ROPE operation\n");
        }
    }

    தருக்க kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, அளவு(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    எனில் (cpu_comparison_active) {
        எனில் (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, முனை, &rope_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for ROPE operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END_EXT("ROPE", kernel_name, முனை, "mode=0x%x|n_dims=%d|freq_base=%.2f|freq_scale=%.2f",
                    params.rope_params.mode, params.rope_params.n_dims, (இரட்டை) params.rope_params.freq_base,
                    (இரட்டை) params.rope_params.freq_scale);
    திரும்பு kernel_result;
}

தருக்க ggml_et_op_rms_norm(ggml_backend_et_device_context * dev_ctx, மாறிலி ggml_tensor * முனை) {
    ET_PERF_START();

    எனில் (!dev_ctx || !முனை) {
        GGML_LOG_ERROR("ET: Invalid parameters for RMS_NORM operation\n");
        திரும்பு பொய்;
    }

    எனில் (!முனை->src[0]) {
        GGML_LOG_ERROR("ET: RMS_NORM operation missing required input\n");
        திரும்பு பொய்;
    }

    மாறிலி எழுத்து * kernel_name;

    எனில் (முனை->type == GGML_TYPE_F32 && முனை->src[0]->type == GGML_TYPE_F32) {
        kernel_name = "rms_norm_f32";

    } இல்லையெனில் {
        GGML_LOG_ERROR("ET: RMS_NORM operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(முனை->type),
                       ggml_type_name(முனை->src[0]->type));
        திரும்பு பொய்;
    }

    தசம eps;
    memcpy(&eps, முனை->op_params, அளவு(தசம));

    ggml_et_rms_norm_params params;
    params.src0 = *முனை->src[0];  // F32 input tensor
    params.dst  = *முனை;          // F32 output tensor
    params.eps  = eps;            // Epsilon parameter for numerical stability

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    தருக்க                    cpu_comparison_active = பொய்;
    எனில் (rms_norm_cpu_compare_config.enabled) {
        எனில் (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, முனை, GGML_OP_RMS_NORM)) {
            cpu_comparison_active = உண்மை;
        } இல்லையெனில் {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for RMS_NORM operation\n");
        }
    }

    தருக்க kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, அளவு(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    எனில் (cpu_comparison_active) {
        எனில் (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, முனை, &rms_norm_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for RMS_NORM operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END_EXT("RMS_NORM", kernel_name, முனை, "eps=%.6f", (இரட்டை) eps);
    திரும்பு kernel_result;
}

தருக்க ggml_et_op_norm(ggml_backend_et_device_context * dev_ctx, மாறிலி ggml_tensor * முனை) {
    ET_PERF_START();

    எனில் (!dev_ctx || !முனை) {
        GGML_LOG_ERROR("ET: Invalid parameters for NORM operation\n");
        திரும்பு பொய்;
    }

    எனில் (!முனை->src[0]) {
        GGML_LOG_ERROR("ET: NORM operation missing required input\n");
        திரும்பு பொய்;
    }

    மாறிலி எழுத்து * kernel_name;

    எனில் (முனை->type == GGML_TYPE_F32 && முனை->src[0]->type == GGML_TYPE_F32) {
        kernel_name = "norm_f32";

    } இல்லையெனில் {
        GGML_LOG_ERROR("ET: NORM operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(முனை->type),
                       ggml_type_name(முனை->src[0]->type));
        திரும்பு பொய்;
    }

    தசம eps;
    memcpy(&eps, முனை->op_params, அளவு(தசம));

    ggml_et_norm_params params;
    params.src0 = *முனை->src[0];  // F32 input tensor
    params.dst  = *முனை;          // F32 output tensor
    params.eps  = eps;            // Epsilon parameter for numerical stability

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    தருக்க                    cpu_comparison_active = பொய்;
    எனில் (norm_cpu_compare_config.enabled) {
        எனில் (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, முனை, GGML_OP_NORM)) {
            cpu_comparison_active = உண்மை;
        } இல்லையெனில் {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for NORM operation\n");
        }
    }

    தருக்க kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, அளவு(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    எனில் (cpu_comparison_active) {
        எனில் (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, முனை, &norm_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for NORM operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END_EXT("NORM", kernel_name, முனை, "eps=%.6f", (இரட்டை) eps);
    திரும்பு kernel_result;
}

தருக்க ggml_et_op_l2_norm(ggml_backend_et_device_context * dev_ctx, மாறிலி ggml_tensor * முனை) {
    ET_PERF_START();

    எனில் (!dev_ctx || !முனை) {
        GGML_LOG_ERROR("ET: Invalid parameters for L2_NORM operation\n");
        திரும்பு பொய்;
    }

    எனில் (!முனை->src[0]) {
        GGML_LOG_ERROR("ET: L2_NORM operation missing required input\n");
        திரும்பு பொய்;
    }

    மாறிலி எழுத்து * kernel_name;

    எனில் (முனை->type == GGML_TYPE_F32 && முனை->src[0]->type == GGML_TYPE_F32) {
        kernel_name = "l2_norm_f32";

    } இல்லையெனில் {
        GGML_LOG_ERROR("ET: L2_NORM operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(முனை->type),
                       ggml_type_name(முனை->src[0]->type));
        திரும்பு பொய்;
    }

    தசம eps;
    memcpy(&eps, முனை->op_params, அளவு(தசம));

    ggml_et_l2_norm_params params;
    params.src0 = *முனை->src[0];  // F32 input tensor
    params.dst  = *முனை;          // F32 output tensor
    params.eps  = eps;            // Epsilon parameter for numerical stability

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    தருக்க                    cpu_comparison_active = பொய்;
    எனில் (l2_norm_cpu_compare_config.enabled) {
        எனில் (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, முனை, GGML_OP_L2_NORM)) {
            cpu_comparison_active = உண்மை;
        } இல்லையெனில் {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for L2_NORM operation\n");
        }
    }

    தருக்க kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, அளவு(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    எனில் (cpu_comparison_active) {
        எனில் (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, முனை, &l2_norm_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for L2_NORM operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END_EXT("L2_NORM", kernel_name, முனை, "eps=%.6f", (இரட்டை) eps);
    திரும்பு kernel_result;
}

தருக்க ggml_et_op_group_norm(ggml_backend_et_device_context * dev_ctx, மாறிலி ggml_tensor * முனை) {
    ET_PERF_START();

    எனில் (!dev_ctx || !முனை) {
        GGML_LOG_ERROR("ET: Invalid parameters for GROUP_NORM operation\n");
        திரும்பு பொய்;
    }

    எனில் (!முனை->src[0]) {
        GGML_LOG_ERROR("ET: GROUP_NORM operation missing required input\n");
        திரும்பு பொய்;
    }

    எனில் (முனை->type != GGML_TYPE_F32 || முனை->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: GROUP_NORM operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(முனை->type),
                       ggml_type_name(முனை->src[0]->type));
        திரும்பு பொய்;
    }

    மாறிலி int32_t n_groups = ggml_get_op_params_i32(முனை, 0);
    தசம         eps;
    memcpy(&eps, (மாறிலி தசம *) முனை->op_params + 1, அளவு(தசம));

    ggml_et_group_norm_params params;
    params.src0     = *முனை->src[0];
    params.dst      = *முனை;
    params.n_groups = n_groups;
    params.eps      = eps;

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    தருக்க                    cpu_comparison_active = பொய்;
    எனில் (group_norm_cpu_compare_config.enabled) {
        எனில் (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, முனை, GGML_OP_GROUP_NORM)) {
            cpu_comparison_active = உண்மை;
        } இல்லையெனில் {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for GROUP_NORM operation\n");
        }
    }

    தருக்க kernel_result = ggml_et_launch_kernel(dev_ctx, "group_norm_f32", &params, அளவு(params), 0xFFFFFFFF);

    எனில் (cpu_comparison_active) {
        எனில் (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, முனை, &group_norm_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for GROUP_NORM operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END_EXT("GROUP_NORM", "group_norm_f32", முனை, "eps=%.6f|n_groups=%d", (இரட்டை) eps, n_groups);
    திரும்பு kernel_result;
}

தருக்க ggml_et_op_im2col(ggml_backend_et_device_context * dev_ctx, மாறிலி ggml_tensor * முனை) {
    ET_PERF_START();

    எனில் (!dev_ctx || !முனை) {
        GGML_LOG_ERROR("ET: Invalid parameters for IM2COL operation\n");
        திரும்பு பொய்;
    }

    எனில் (!முனை->src[0] || !முனை->src[1]) {
        GGML_LOG_ERROR("ET: IM2COL operation missing required inputs\n");
        திரும்பு பொய்;
    }

    மாறிலி தருக்க supported_types =
        (முனை->type == GGML_TYPE_F32 && முனை->src[1]->type == GGML_TYPE_F32) ||
        (முனை->type == GGML_TYPE_F16 && (முனை->src[1]->type == GGML_TYPE_F16 || முனை->src[1]->type == GGML_TYPE_F32));

    எனில் (!supported_types) {
        GGML_LOG_ERROR("ET: IM2COL operation with unsupported types: dst=%s src1=%s\n", ggml_type_name(முனை->type),
                       ggml_type_name(முனை->src[1]->type));
        திரும்பு பொய்;
    }

    ggml_et_im2col_params params;
    params.src0 = *முனை->src[0];
    params.src1 = *முனை->src[1];
    params.dst  = *முனை;

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    தருக்க                    cpu_comparison_active = பொய்;
    எனில் (im2col_cpu_compare_config.enabled) {
        எனில் (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, முனை, GGML_OP_IM2COL)) {
            cpu_comparison_active = உண்மை;
        } இல்லையெனில் {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for IM2COL operation\n");
        }
    }

    தருக்க kernel_result = ggml_et_launch_kernel(dev_ctx, "im2col", &params, அளவு(params), 0xFFFFFFFF);

    எனில் (cpu_comparison_active) {
        எனில் (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, முனை, &im2col_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for IM2COL operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("IM2COL", "im2col", முனை);
    திரும்பு kernel_result;
}

தருக்க ggml_et_op_conv_2d(ggml_backend_et_device_context * dev_ctx, மாறிலி ggml_tensor * முனை) {
    ET_PERF_START();

    எனில் (!dev_ctx || !முனை) {
        திரும்பு பொய்;
    }
    எனில் (!முனை->src[0] || !முனை->src[1]) {
        திரும்பு பொய்;
    }
    எனில் (!முனை->data || !முனை->src[0]->data || !முனை->src[1]->data) {
        திரும்பு பொய்;
    }

    // Kernel constraints (mirror supports_op; recheck here as a guard).
    மாறிலி ggml_tensor * flt = முனை->src[0];  // [Kw, Kh, Cin, Cout]
    மாறிலி ggml_tensor * in  = முனை->src[1];  // [W,  H,  Cin, N]
    எனில் (முனை->type != GGML_TYPE_F32 || flt->type != GGML_TYPE_F32 || in->type != GGML_TYPE_F32) {
        திரும்பு பொய்;
    }

    மாறிலி int32_t s0 = ggml_get_op_params_i32(முனை, 0);
    மாறிலி int32_t s1 = ggml_get_op_params_i32(முனை, 1);
    மாறிலி int32_t p0 = ggml_get_op_params_i32(முனை, 2);
    மாறிலி int32_t p1 = ggml_get_op_params_i32(முனை, 3);
    மாறிலி int32_t d0 = ggml_get_op_params_i32(முனை, 4);
    மாறிலி int32_t d1 = ggml_get_op_params_i32(முனை, 5);

    எனில் (s0 < 1 || s1 < 1) {
        திரும்பு பொய்;
    }
    எனில் (d0 != 1 || d1 != 1) {
        திரும்பு பொய்;
    }
    எனில் (flt->ne[2] % 16 != 0 || flt->ne[3] % 16 != 0) {
        திரும்பு பொய்;
    }
    எனில் (in->ne[3] != 1) {
        திரும்பு பொய்;
    }
    எனில் (முனை->ne[0] <= 0) {
        திரும்பு பொய்;  // OW > 0 (any width OK; staging path handles non-16)
    }
    (வெறுமை) p0;
    (வெறுமை) p1;

    ggml_et_இருமம்_params params;
    params.src0 = *முனை->src[0];
    params.src1 = *முனை->src[1];
    params.dst  = *முனை;

    தருக்க kernel_result = ggml_et_launch_kernel(dev_ctx, "conv_2d_f32_me", &params, அளவு(params), 0xFFFFFFFFu);

    ET_PERF_END("CONV_2D", "conv_2d_f32_me", முனை);
    திரும்பு kernel_result;
}

தருக்க ggml_et_op_softmax(ggml_backend_et_device_context * dev_ctx, மாறிலி ggml_tensor * முனை) {
    ET_PERF_START();

    எனில் (!dev_ctx || !முனை) {
        GGML_LOG_ERROR("ET: Invalid parameters for SOFTMAX operation\n");
        திரும்பு பொய்;
    }

    எனில் (!முனை->src[0]) {
        GGML_LOG_ERROR("ET: SOFTMAX operation missing required input\n");
        திரும்பு பொய்;
    }

    மாறிலி எழுத்து * kernel_name;

    எனில் (முனை->type == GGML_TYPE_F32 && முனை->src[0]->type == GGML_TYPE_F32) {
        kernel_name = "softmax_f32";

    } இல்லையெனில் {
        GGML_LOG_ERROR("ET: SOFTMAX operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(முனை->type),
                       ggml_type_name(முனை->src[0]->type));
        திரும்பு பொய்;
    }

    // Validate contiguity requirements
    எனில் (!ggml_is_contiguous(முனை)) {
        GGML_LOG_ERROR("ET: SOFTMAX operation requires contiguous destination tensor\n");
        திரும்பு பொய்;
    }

    எனில் (!ggml_is_contiguous(முனை->src[0])) {
        GGML_LOG_ERROR("ET: SOFTMAX operation requires contiguous source tensor\n");
        திரும்பு பொய்;
    }

    // Check optional mask tensor
    எனில் (முனை->src[1]) {
        எனில் (முனை->src[1]->type != GGML_TYPE_F32) {
            GGML_LOG_ERROR("ET: SOFTMAX operation with unsupported mask type: %s (F32 required)\n",
                           ggml_type_name(முனை->src[1]->type));
            திரும்பு பொய்;
        }
        எனில் (!ggml_is_contiguous(முனை->src[1])) {
            GGML_LOG_ERROR("ET: SOFTMAX operation requires contiguous mask tensor\n");
            திரும்பு பொய்;
        }
    }

    // Check optional sinks tensor
    எனில் (முனை->src[2]) {
        எனில் (முனை->src[2]->type != GGML_TYPE_F32) {
            GGML_LOG_ERROR("ET: SOFTMAX operation with unsupported sinks type: %s (F32 required)\n",
                           ggml_type_name(முனை->src[2]->type));
            திரும்பு பொய்;
        }
        எனில் (!ggml_is_contiguous(முனை->src[2])) {
            GGML_LOG_ERROR("ET: SOFTMAX operation requires contiguous sinks tensor\n");
            திரும்பு பொய்;
        }
    }

    // Extract scale and max_bias from op_params
    தசம scale    = 1.0f;
    தசம max_bias = 0.0f;
    எனில் (முனை->op_params) {
        memcpy(&scale, (மாறிலி தசம *) முனை->op_params + 0, அளவு(தசம));
        memcpy(&max_bias, (மாறிலி தசம *) முனை->op_params + 1, அளவு(தசம));
    }

    ggml_et_softmax_params params;
    params.src0 = *முனை->src[0];                       // F32 input tensor
    எனில் (முனை->src[1]) {
        params.src1 = *முனை->src[1];                   // F32 mask tensor
    } இல்லையெனில் {
        memset(&params.src1, 0, அளவு(params.src1));  // Zero if no mask
    }
    எனில் (முனை->src[2]) {
        params.src2 = *முனை->src[2];                   // F32 sinks tensor
    } இல்லையெனில் {
        memset(&params.src2, 0, அளவு(params.src2));  // Zero if no sinks
    }
    params.dst      = *முனை;                           // F32 output tensor
    params.scale    = scale;                           // Scale factor
    params.max_bias = max_bias;                        // ALiBi bias

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    தருக்க                    cpu_comparison_active = பொய்;
    எனில் (softmax_cpu_compare_config.enabled) {
        எனில் (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, முனை, GGML_OP_SOFT_MAX)) {
            cpu_comparison_active = உண்மை;
        } இல்லையெனில் {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for SOFTMAX operation\n");
        }
    }

    தருக்க kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, அளவு(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    எனில் (cpu_comparison_active) {
        எனில் (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, முனை, &softmax_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for SOFTMAX operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END_EXT("SOFTMAX", kernel_name, முனை, "scale=%.6f|max_bias=%.6f|has_mask=%s", (இரட்டை) scale,
                    (இரட்டை) max_bias, முனை->src[1] ? "yes" : "no");
    திரும்பு kernel_result;
}

தருக்க ggml_et_op_flash_attn_ext(ggml_backend_et_device_context * dev_ctx, மாறிலி ggml_tensor * முனை) {
    ET_PERF_START();

    எனில் (!dev_ctx || !முனை) {
        GGML_LOG_ERROR("ET: Invalid parameters for FLASH_ATTN_EXT operation\n");
        திரும்பு பொய்;
    }

    எனில் (!முனை->src[0] || !முனை->src[1] || !முனை->src[2]) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT operation missing required inputs\n");
        திரும்பு பொய்;
    }

    எனில் (முனை->type != GGML_TYPE_F32 || முனை->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT requires F32 Q and dst, got dst=%s q=%s\n", ggml_type_name(முனை->type),
                       ggml_type_name(முனை->src[0]->type));
        திரும்பு பொய்;
    }

    // K and V can be F16 or F32
    எனில் ((முனை->src[1]->type != GGML_TYPE_F32 && முனை->src[1]->type != GGML_TYPE_F16) ||
        (முனை->src[2]->type != GGML_TYPE_F32 && முனை->src[2]->type != GGML_TYPE_F16)) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT K/V must be F16 or F32, got k=%s v=%s\n", ggml_type_name(முனை->src[1]->type),
                       ggml_type_name(முனை->src[2]->type));
        திரும்பு பொய்;
    }

    எனில் (முனை->src[4] != nullptr) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT baseline kernel does not support sinks\n");
        திரும்பு பொய்;
    }

    // Mask is optional; if present must be F16 or F32
    எனில் (முனை->src[3] != nullptr && முனை->src[3]->type != GGML_TYPE_F32 && முனை->src[3]->type != GGML_TYPE_F16) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT mask must be F16 or F32, got %s\n", ggml_type_name(முனை->src[3]->type));
        திரும்பு பொய்;
    }

    // Q and dst must be row-contiguous F32
    எனில் (!ggml_is_contiguous_rows(முனை) || !ggml_is_contiguous_rows(முனை->src[0])) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT requires row-contiguous Q and dst\n");
        திரும்பு பொய்;
    }

    எனில் (முனை->nb[0] != அளவு(தசம) || முனை->src[0]->nb[0] != அளவு(தசம)) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT requires contiguous F32 rows for Q and dst\n");
        திரும்பு பொய்;
    }

    // K/V must have element-sized stride in dim 0
    மாறிலி size_t k_elem = முனை->src[1]->type == GGML_TYPE_F16 ? 2 : 4;
    மாறிலி size_t v_elem = முனை->src[2]->type == GGML_TYPE_F16 ? 2 : 4;
    எனில் (முனை->src[1]->nb[0] != k_elem || முனை->src[2]->nb[0] != v_elem) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT K/V must have element-sized stride in dim 0\n");
        திரும்பு பொய்;
    }

    தசம scale         = 1.0f;
    தசம max_bias      = 0.0f;
    தசம logit_softcap = 0.0f;
    memcpy(&scale, (மாறிலி தசம *) முனை->op_params + 0, அளவு(scale));
    memcpy(&max_bias, (மாறிலி தசம *) முனை->op_params + 1, அளவு(max_bias));
    memcpy(&logit_softcap, (மாறிலி தசம *) முனை->op_params + 2, அளவு(logit_softcap));

    எனில் (max_bias != 0.0f || logit_softcap != 0.0f) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT baseline kernel does not support max_bias or logit_softcap\n");
        திரும்பு பொய்;
    }

    மாறிலி ggml_prec prec = ggml_flash_attn_ext_get_prec(முனை);
    எனில் (prec != GGML_PREC_F32 && prec != GGML_PREC_DEFAULT) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT baseline kernel only supports F32 precision\n");
        திரும்பு பொய்;
    }

    // dk must match between Q and K; dv must match between V and dst
    எனில் (முனை->src[0]->ne[0] != முனை->src[1]->ne[0]) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT dk mismatch: Q=%lld K=%lld\n", (long long) முனை->src[0]->ne[0],
                       (long long) முனை->src[1]->ne[0]);
        திரும்பு பொய்;
    }

    எனில் (முனை->src[2]->ne[0] != முனை->ne[0]) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT dv mismatch: V=%lld dst=%lld\n", (long long) முனை->src[2]->ne[0],
                       (long long) முனை->ne[0]);
        திரும்பு பொய்;
    }

    எனில் (முனை->src[2]->ne[0] > 512) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT dv=%lld exceeds maximum 512\n", (long long) முனை->src[2]->ne[0]);
        திரும்பு பொய்;
    }

    எனில் (முனை->src[0]->ne[0] > 512) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT dk=%lld exceeds maximum 512\n", (long long) முனை->src[0]->ne[0]);
        திரும்பு பொய்;
    }

    // GQA: n_head_q must be a multiple of n_head_kv
    மாறிலி int64_t nhq = முனை->src[0]->ne[2];
    மாறிலி int64_t nhk = முனை->src[1]->ne[2];
    எனில் (nhq % nhk != 0) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT n_head_q (%lld) not divisible by n_head_kv (%lld)\n", (long long) nhq,
                       (long long) nhk);
        திரும்பு பொய்;
    }

    // K and V must have matching sequence length, heads, and batch dims
    எனில் (முனை->src[1]->ne[1] != முனை->src[2]->ne[1] || முனை->src[1]->ne[2] != முனை->src[2]->ne[2] ||
        முனை->src[1]->ne[3] != முனை->src[2]->ne[3]) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT K/V shape mismatch\n");
        திரும்பு பொய்;
    }

    // dst layout checks: [dv, nhq, nq, no]
    எனில் (முனை->src[0]->ne[1] != முனை->ne[2] || முனை->src[0]->ne[2] != முனை->ne[1] ||
        முனை->src[0]->ne[3] != முனை->ne[3]) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT dst shape mismatch\n");
        திரும்பு பொய்;
    }

    // Batch dims: Q batch must match K batch
    எனில் (முனை->src[0]->ne[3] != முனை->src[1]->ne[3]) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT batch dimension mismatch\n");
        திரும்பு பொய்;
    }

    ggml_et_flash_attn_ext_params params;
    memset(&params, 0, அளவு(params));
    params.src0 = *முனை->src[0];
    params.src1 = *முனை->src[1];
    params.src2 = *முனை->src[2];
    எனில் (முனை->src[3] != nullptr) {
        params.mask     = *முனை->src[3];
        params.has_mask = 1;
    }
    params.dst   = *முனை;
    params.scale = scale;

    // Use matrix engine kernel when K/V are F16 and dk is a multiple of 32
    மாறிலி எழுத்து * kernel_name;
    எனில் (முனை->src[1]->type == GGML_TYPE_F16 && முனை->src[2]->type == GGML_TYPE_F16 && (முனை->src[0]->ne[0] % 32) == 0) {
        kernel_name = "flash_attn_ext_f16_me";
    } இல்லையெனில் {
        kernel_name = "flash_attn_ext_f32";
    }

    மாறிலி தருக்க kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, அளவு(params), 0xFFFFFFFF);

    ET_PERF_END_EXT("FLASH_ATTN_EXT", kernel_name, முனை, "scale=%.6f", (இரட்டை) scale);
    திரும்பு kernel_result;
}

தருக்க ggml_et_op_get_rows(ggml_backend_et_device_context * dev_ctx, மாறிலி ggml_tensor * முனை) {
    ET_PERF_START();

    எனில் (!dev_ctx || !முனை) {
        GGML_LOG_ERROR("ET: Invalid parameters for GET_ROWS operation\n");
        திரும்பு பொய்;
    }

    எனில் (!முனை->src[0] || !முனை->src[1]) {
        GGML_LOG_ERROR("ET: GET_ROWS operation missing required inputs\n");
        திரும்பு பொய்;
    }

    மாறிலி எழுத்து * kernel_name;

    எனில் (முனை->type == GGML_TYPE_F32 && முனை->src[1]->type == GGML_TYPE_I32 &&
        (முனை->src[0]->type == GGML_TYPE_F32 || முனை->src[0]->type == GGML_TYPE_F16 ||
         முனை->src[0]->type == GGML_TYPE_Q4_0 || முனை->src[0]->type == GGML_TYPE_Q8_0 ||
         முனை->src[0]->type == GGML_TYPE_Q4_K)) {
        kernel_name = "get_rows_f32";

    } இல்லையெனில் {
        GGML_LOG_ERROR("ET: GET_ROWS operation with unsupported types: dst=%s src0=%s src1=%s\n",
                       ggml_type_name(முனை->type), ggml_type_name(முனை->src[0]->type),
                       ggml_type_name(முனை->src[1]->type));
        திரும்பு பொய்;
    }

    // Validate contiguity requirements
    எனில் (!ggml_is_contiguous(முனை)) {
        GGML_LOG_ERROR("ET: GET_ROWS operation requires contiguous destination tensor\n");
        திரும்பு பொய்;
    }

    எனில் (!ggml_is_contiguous(முனை->src[0])) {
        GGML_LOG_ERROR("ET: GET_ROWS operation requires contiguous data tensor\n");
        திரும்பு பொய்;
    }

    எனில் (!ggml_is_contiguous(முனை->src[1])) {
        GGML_LOG_ERROR("ET: GET_ROWS operation requires contiguous indices tensor\n");
        திரும்பு பொய்;
    }

    // Validate dimension constraints from ggml implementation
    எனில் (முனை->src[0]->ne[2] != முனை->src[1]->ne[1] || முனை->src[1]->ne[3] != 1) {
        GGML_LOG_ERROR(
            "ET: GET_ROWS operation dimension constraint failed: src0.ne[2]=%lld != src1.ne[1]=%lld or src1.ne[3]=%lld "
            "!= 1\n",
            (long long) முனை->src[0]->ne[2], (long long) முனை->src[1]->ne[1], (long long) முனை->src[1]->ne[3]);
        திரும்பு பொய்;
    }

    ggml_et_get_rows_params params;
    params.src0 = *முனை->src[0];  // Data tensor (F32 or Q8_0)
    params.src1 = *முனை->src[1];  // Indices tensor (I32)
    params.dst  = *முனை;          // Output tensor (F32)

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    தருக்க                    cpu_comparison_active = பொய்;
    எனில் (get_rows_cpu_compare_config.enabled) {
        எனில் (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, முனை, GGML_OP_GET_ROWS)) {
            cpu_comparison_active = உண்மை;
        } இல்லையெனில் {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for GET_ROWS operation\n");
        }
    }

    தருக்க kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, அளவு(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    எனில் (cpu_comparison_active) {
        எனில் (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, முனை, &get_rows_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for GET_ROWS operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("GET_ROWS", kernel_name, முனை);
    திரும்பு kernel_result;
}

தருக்க ggml_et_op_cont(ggml_backend_et_device_context * dev_ctx, மாறிலி ggml_tensor * முனை) {
    ET_PERF_START();

    // Validate source tensor exists
    எனில் (!முனை->src[0]) {
        GGML_LOG_ERROR("ET: CONT operation missing source tensor\n");
        திரும்பு பொய்;
    }

    // Validate types match (input and output must be same type)
    எனில் (முனை->type != முனை->src[0]->type) {
        GGML_LOG_ERROR("ET: CONT operation type mismatch: src=%s dst=%s\n", ggml_type_name(முனை->src[0]->type),
                       ggml_type_name(முனை->type));
        திரும்பு பொய்;
    }

    // Validate supported types
    எனில் (முனை->type != GGML_TYPE_F32 && முனை->type != GGML_TYPE_F16) {
        GGML_LOG_ERROR("ET: CONT operation unsupported type: %s (only F32 and F16 supported)\n",
                       ggml_type_name(முனை->type));
        திரும்பு பொய்;
    }

    // Validate contiguity - output must be contiguous, input can be non-contiguous
    எனில் (!ggml_is_contiguous(முனை)) {
        GGML_LOG_ERROR("ET: CONT operation requires contiguous output tensor\n");
        திரும்பு பொய்;
    }

    // Select kernel based on type
    மாறிலி எழுத்து * kernel_name;
    எனில் (முனை->type == GGML_TYPE_F32) {
        kernel_name = "cont_f32";
    } இல்லையெனில் எனில் (முனை->type == GGML_TYPE_F16) {
        kernel_name = "cont_f16";
    } இல்லையெனில் {
        GGML_LOG_ERROR("ET: CONT operation with unsupported type: %s\n", ggml_type_name(முனை->type));
        திரும்பு பொய்;
    }

    ggml_et_cont_params params;
    params.src0 = *முனை->src[0];  // Input tensor (potentially non-contiguous)
    params.dst  = *முனை;          // Output tensor (contiguous)

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    தருக்க                    cpu_comparison_active = பொய்;
    எனில் (cont_cpu_compare_config.enabled) {
        எனில் (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, முனை, GGML_OP_CONT)) {
            cpu_comparison_active = உண்மை;
        } இல்லையெனில் {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for CONT operation\n");
        }
    }

    தருக்க kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, அளவு(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    எனில் (cpu_comparison_active) {
        எனில் (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, முனை, &cont_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for CONT operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("CONT", kernel_name, முனை);
    திரும்பு kernel_result;
}

தருக்க ggml_et_op_cumsum(ggml_backend_et_device_context * dev_ctx, மாறிலி ggml_tensor * முனை) {
    ET_PERF_START();

    எனில் (!dev_ctx || !முனை || !முனை->src[0]) {
        GGML_LOG_ERROR("ET: Invalid parameters for CUMSUM operation\n");
        திரும்பு பொய்;
    }

    எனில் (முனை->type != GGML_TYPE_F32 || முனை->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: CUMSUM operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(முனை->type),
                       ggml_type_name(முனை->src[0]->type));
        திரும்பு பொய்;
    }

    மாறிலி எழுத்து * kernel_name = "cumsum_f32";

    ggml_et_cumsum_params params;
    params.src0 = *முனை->src[0];
    params.dst  = *முனை;

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    தருக்க                    cpu_comparison_active = பொய்;
    எனில் (cumsum_cpu_compare_config.enabled) {
        எனில் (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, முனை, GGML_OP_CUMSUM)) {
            cpu_comparison_active = உண்மை;
        } இல்லையெனில் {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for CUMSUM operation\n");
        }
    }

    தருக்க kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, அளவு(params), 0xFFFFFFFF);

    எனில் (cpu_comparison_active) {
        எனில் (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, முனை, &cumsum_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for CUMSUM operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("CUMSUM", kernel_name, முனை);
    திரும்பு kernel_result;
}

தருக்க ggml_et_op_cpy(ggml_backend_et_device_context * dev_ctx, மாறிலி ggml_tensor * முனை) {
    ET_PERF_START();

    // CPY copies data from src[0] into the layout of dst (which matches src[1])
    // For same-type with contiguous dst, this is identical to CONT
    எனில் (!முனை->src[0]) {
        GGML_LOG_ERROR("ET: CPY operation missing source tensor\n");
        திரும்பு பொய்;
    }

    // Scalar / zero-element special path: if any dimension is 0, nothing to copy
    மாறிலி int64_t nelements = முனை->ne[0] * முனை->ne[1] * முனை->ne[2] * முனை->ne[3];
    எனில் (nelements == 0) {
        GGML_LOG_DEBUG("ET: CPY no-op (zero elements): ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]\n",
                       முனை->ne[0], முனை->ne[1], முனை->ne[2], முனை->ne[3]);
        ET_PERF_END("CPY", "noop", முனை);
        திரும்பு உண்மை;
    }

    // Only F32 and F16 supported for dst
    எனில் (முனை->type != GGML_TYPE_F32 && முனை->type != GGML_TYPE_F16) {
        GGML_LOG_ERROR("ET: CPY unsupported dst type: %s\n", ggml_type_name(முனை->type));
        திரும்பு பொய்;
    }

    // Select kernel based on src/dst type combination
    மாறிலி எழுத்து * kernel_name;
    எனில் (முனை->src[0]->type == GGML_TYPE_F32 && முனை->type == GGML_TYPE_F32) {
        kernel_name = "cont_f32";
    } இல்லையெனில் எனில் (முனை->src[0]->type == GGML_TYPE_F16 && முனை->type == GGML_TYPE_F16) {
        kernel_name = "cont_f16";
    } இல்லையெனில் எனில் (முனை->src[0]->type == GGML_TYPE_F32 && முனை->type == GGML_TYPE_F16) {
        kernel_name = "cpy_f32_f16";
    } இல்லையெனில் {
        GGML_LOG_ERROR("ET: CPY unsupported type combination: src=%s dst=%s\n", ggml_type_name(முனை->src[0]->type),
                       ggml_type_name(முனை->type));
        திரும்பு பொய்;
    }

    ggml_et_cont_params params;
    params.src0 = *முனை->src[0];
    params.dst  = *முனை;

    // CPU comparison for debugging
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    தருக்க                    cpu_comparison_active = பொய்;
    எனில் (cont_cpu_compare_config.enabled) {
        எனில் (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, முனை, GGML_OP_CPY)) {
            cpu_comparison_active = உண்மை;
        } இல்லையெனில் {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for CPY operation\n");
        }
    }

    தருக்க kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, அளவு(params), 0xFFFFFFFF);

    எனில் (cpu_comparison_active) {
        எனில் (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, முனை, &cont_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for CPY operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("CPY", kernel_name, முனை);
    திரும்பு kernel_result;
}

தருக்க ggml_et_op_concat(ggml_backend_et_device_context * dev_ctx, மாறிலி ggml_tensor * முனை) {
    ET_PERF_START();

    எனில் (!dev_ctx || !முனை) {
        GGML_LOG_ERROR("ET: Invalid parameters for CONCAT operation\n");
        திரும்பு பொய்;
    }

    எனில் (!முனை->src[0] || !முனை->src[1]) {
        GGML_LOG_ERROR("ET: CONCAT operation missing required inputs\n");
        திரும்பு பொய்;
    }

    மாறிலி எழுத்து * kernel_name;

    எனில் (முனை->type == GGML_TYPE_F32 && முனை->src[0]->type == GGML_TYPE_F32 && முனை->src[1]->type == GGML_TYPE_F32) {
        kernel_name = "concat_f32";

    } இல்லையெனில் {
        GGML_LOG_ERROR("ET: CONCAT operation with unsupported types: dst=%s src0=%s src1=%s\n",
                       ggml_type_name(முனை->type), ggml_type_name(முனை->src[0]->type),
                       ggml_type_name(முனை->src[1]->type));
        திரும்பு பொய்;
    }

    int32_t dim;
    memcpy(&dim, முனை->op_params, அளவு(int32_t));

    ggml_et_concat_params params;
    params.src0 = *முனை->src[0];
    params.src1 = *முனை->src[1];
    params.dst  = *முனை;
    params.dim  = dim;

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    தருக்க                    cpu_comparison_active = பொய்;
    எனில் (concat_cpu_compare_config.enabled) {
        எனில் (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, முனை, GGML_OP_CONCAT)) {
            cpu_comparison_active = உண்மை;
        } இல்லையெனில் {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for CONCAT operation\n");
        }
    }

    தருக்க kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, அளவு(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    எனில் (cpu_comparison_active) {
        எனில் (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, முனை, &concat_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for CONCAT operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END_EXT("CONCAT", kernel_name, முனை, "dim=%d", dim);
    திரும்பு kernel_result;
}

தருக்க ggml_et_op_repeat(ggml_backend_et_device_context * dev_ctx, மாறிலி ggml_tensor * முனை) {
    ET_PERF_START();

    எனில் (!dev_ctx || !முனை) {
        GGML_LOG_ERROR("ET: Invalid parameters for REPEAT operation\n");
        திரும்பு பொய்;
    }

    எனில் (!முனை->src[0]) {
        GGML_LOG_ERROR("ET: REPEAT operation missing required input\n");
        திரும்பு பொய்;
    }

    மாறிலி எழுத்து * kernel_name;

    எனில் (முனை->type == GGML_TYPE_F32 && முனை->src[0]->type == GGML_TYPE_F32) {
        // No-op REPEAT (every repeat factor is 1): the output is just a copy
        // of the input. Route to cont_f32, whose contiguous fast path handles
        // arbitrary sizes (including those rejected by repeat_f32's gate,
        // e.g. ne[0]=1).
        எனில் (ggml_are_same_shape(முனை->src[0], முனை)) {
            kernel_name = "cont_f32";
        } இல்லையெனில் {
            kernel_name = "repeat_f32";
        }

    } இல்லையெனில் {
        GGML_LOG_ERROR("ET: REPEAT operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(முனை->type),
                       ggml_type_name(முனை->src[0]->type));
        திரும்பு பொய்;
    }

    // ggml_et_cont_params and ggml_et_repeat_params have identical layouts
    // (just src0 + dst), so the same payload works for either kernel.
    ggml_et_repeat_params params;
    params.src0 = *முனை->src[0];
    params.dst  = *முனை;

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    தருக்க                    cpu_comparison_active = பொய்;
    எனில் (repeat_cpu_compare_config.enabled) {
        எனில் (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, முனை, GGML_OP_REPEAT)) {
            cpu_comparison_active = உண்மை;
        } இல்லையெனில் {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for REPEAT operation\n");
        }
    }

    தருக்க kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, அளவு(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    எனில் (cpu_comparison_active) {
        எனில் (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, முனை, &repeat_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for REPEAT operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("REPEAT", kernel_name, முனை);
    திரும்பு kernel_result;
}

தருக்க ggml_et_op_ssm_conv(ggml_backend_et_device_context * dev_ctx, மாறிலி ggml_tensor * முனை) {
    ET_PERF_START();

    எனில் (!dev_ctx || !முனை || !முனை->src[0] || !முனை->src[1]) {
        GGML_LOG_ERROR("ET: Invalid parameters for SSM_CONV operation\n");
        திரும்பு பொய்;
    }

    எனில் (முனை->type != GGML_TYPE_F32 || முனை->src[0]->type != GGML_TYPE_F32 || முனை->src[1]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: SSM_CONV operation with unsupported types: dst=%s src0=%s src1=%s\n",
                       ggml_type_name(முனை->type), ggml_type_name(முனை->src[0]->type),
                       ggml_type_name(முனை->src[1]->type));
        திரும்பு பொய்;
    }

    மாறிலி எழுத்து * kernel_name = "ssm_conv_f32";

    ggml_et_ssm_conv_params params;
    params.src0 = *முனை->src[0];
    params.src1 = *முனை->src[1];
    params.dst  = *முனை;

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    தருக்க                    cpu_comparison_active = பொய்;
    எனில் (ssm_conv_cpu_compare_config.enabled) {
        எனில் (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, முனை, GGML_OP_SSM_CONV)) {
            cpu_comparison_active = உண்மை;
        } இல்லையெனில் {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for SSM_CONV operation\n");
        }
    }

    தருக்க kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, அளவு(params), 0xFFFFFFFF);

    எனில் (cpu_comparison_active) {
        எனில் (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, முனை, &ssm_conv_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for SSM_CONV operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("SSM_CONV", kernel_name, முனை);
    திரும்பு kernel_result;
}

தருக்க ggml_et_op_ssm_scan(ggml_backend_et_device_context * dev_ctx, மாறிலி ggml_tensor * முனை) {
    ET_PERF_START();

    எனில் (!dev_ctx || !முனை) {
        GGML_LOG_ERROR("ET: Invalid parameters for SSM_SCAN operation\n");
        திரும்பு பொய்;
    }

    க்கு (முழு i = 0; i < 7; ++i) {
        எனில் (!முனை->src[i]) {
            GGML_LOG_ERROR("ET: SSM_SCAN missing required input %d\n", i);
            திரும்பு பொய்;
        }
    }

    எனில் (முனை->type != GGML_TYPE_F32 || முனை->src[0]->type != GGML_TYPE_F32 || முனை->src[1]->type != GGML_TYPE_F32 ||
        முனை->src[2]->type != GGML_TYPE_F32 || முனை->src[3]->type != GGML_TYPE_F32 ||
        முனை->src[4]->type != GGML_TYPE_F32 || முனை->src[5]->type != GGML_TYPE_F32 ||
        முனை->src[6]->type != GGML_TYPE_I32) {
        GGML_LOG_ERROR("ET: SSM_SCAN operation with unsupported types\n");
        திரும்பு பொய்;
    }

    ggml_et_ssm_scan_params params;
    params.src0 = *முனை->src[0];
    params.src1 = *முனை->src[1];
    params.src2 = *முனை->src[2];
    params.src3 = *முனை->src[3];
    params.src4 = *முனை->src[4];
    params.src5 = *முனை->src[5];
    params.src6 = *முனை->src[6];
    params.dst  = *முனை;
    params.K    = ggml_get_op_params_i32(முனை, 0);

    தருக்க kernel_result = ggml_et_launch_kernel(dev_ctx, "ssm_scan_f32", &params, அளவு(params), 0xFFFFFFFF);

    ET_PERF_END("SSM_SCAN", "ssm_scan_f32", முனை);
    திரும்பு kernel_result;
}

தருக்க ggml_et_op_rwkv_wkv6(ggml_backend_et_device_context * dev_ctx, மாறிலி ggml_tensor * முனை) {
    ET_PERF_START();

    எனில் (!dev_ctx || !முனை) {
        GGML_LOG_ERROR("ET: Invalid parameters for RWKV_WKV6 operation\n");
        திரும்பு பொய்;
    }

    // Validate all 6 source tensors exist
    க்கு (முழு i = 0; i <= 5; i++) {
        எனில் (!முனை->src[i]) {
            GGML_LOG_ERROR("ET: RWKV_WKV6 operation missing src[%d]\n", i);
            திரும்பு பொய்;
        }
    }

    எனில் (முனை->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: RWKV_WKV6 only supports F32, got %s\n", ggml_type_name(முனை->type));
        திரும்பு பொய்;
    }

    மாறிலி எழுத்து * kernel_name = "rwkv_wkv6_f32";

    மாறிலி int64_t S      = முனை->src[0]->ne[0];  // head_size
    மாறிலி int64_t H      = முனை->src[0]->ne[1];  // num heads
    மாறிலி int64_t T      = முனை->src[1]->ne[2];  // num tokens
    மாறிலி int64_t n_seqs = முனை->src[5]->ne[1];  // num sequences
    மாறிலி int64_t C      = S * H;

    ggml_et_rwkv_wkv6_params params;
    params.k        = (தசம *) முனை->src[0]->data;
    params.v        = (தசம *) முனை->src[1]->data;
    params.r        = (தசம *) முனை->src[2]->data;
    params.tf       = (தசம *) முனை->src[3]->data;
    params.td       = (தசம *) முனை->src[4]->data;
    params.state_in = (தசம *) முனை->src[5]->data;
    params.dst      = (தசம *) முனை->data;
    params.C        = (int32_t) C;
    params.H        = (int32_t) H;
    params.S        = (int32_t) S;
    params.T        = (int32_t) T;
    params.n_seqs   = (int32_t) n_seqs;

    // Phase 1: Initialize CPU comparison context
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    தருக்க                    cpu_comparison_active = பொய்;
    எனில் (rwkv_wkv6_cpu_compare_config.enabled) {
        எனில் (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, முனை, GGML_OP_RWKV_WKV6)) {
            cpu_comparison_active = உண்மை;
        } இல்லையெனில் {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for RWKV_WKV6 operation\n");
        }
    }

    தருக்க kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, அளவு(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare
    எனில் (cpu_comparison_active) {
        எனில் (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, முனை, &rwkv_wkv6_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for RWKV_WKV6 operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END_EXT("RWKV_WKV6", kernel_name, முனை, "S=%d H=%d T=%d n_seqs=%d", (முழு) S, (முழு) H, (முழு) T,
                    (முழு) n_seqs);
    திரும்பு kernel_result;
}

தருக்க ggml_et_op_rwkv_wkv7(ggml_backend_et_device_context * dev_ctx, மாறிலி ggml_tensor * முனை) {
    ET_PERF_START();

    எனில் (!dev_ctx || !முனை) {
        GGML_LOG_ERROR("ET: Invalid parameters for RWKV_WKV7 operation\n");
        திரும்பு பொய்;
    }

    // Validate all 7 source tensors exist
    க்கு (முழு i = 0; i <= 6; i++) {
        எனில் (!முனை->src[i]) {
            GGML_LOG_ERROR("ET: RWKV_WKV7 operation missing src[%d]\n", i);
            திரும்பு பொய்;
        }
    }

    எனில் (முனை->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: RWKV_WKV7 only supports F32, got %s\n", ggml_type_name(முனை->type));
        திரும்பு பொய்;
    }

    மாறிலி எழுத்து * kernel_name = "rwkv_wkv7_f32";

    மாறிலி int64_t S      = முனை->src[2]->ne[0];  // head_size
    மாறிலி int64_t H      = முனை->src[2]->ne[1];  // num heads
    மாறிலி int64_t T      = முனை->src[1]->ne[2];  // num tokens
    மாறிலி int64_t n_seqs = முனை->src[6]->ne[1];  // num sequences
    மாறிலி int64_t C      = S * H;

    ggml_et_rwkv_wkv7_params params;
    params.r        = (தசம *) முனை->src[0]->data;
    params.w        = (தசம *) முனை->src[1]->data;
    params.k        = (தசம *) முனை->src[2]->data;
    params.v        = (தசம *) முனை->src[3]->data;
    params.a        = (தசம *) முனை->src[4]->data;
    params.b        = (தசம *) முனை->src[5]->data;
    params.state_in = (தசம *) முனை->src[6]->data;
    params.dst      = (தசம *) முனை->data;
    params.C        = (int32_t) C;
    params.H        = (int32_t) H;
    params.S        = (int32_t) S;
    params.T        = (int32_t) T;
    params.n_seqs   = (int32_t) n_seqs;

    // Phase 1: Initialize CPU comparison context
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    தருக்க                    cpu_comparison_active = பொய்;
    எனில் (rwkv_wkv7_cpu_compare_config.enabled) {
        எனில் (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, முனை, GGML_OP_RWKV_WKV7)) {
            cpu_comparison_active = உண்மை;
        } இல்லையெனில் {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for RWKV_WKV7 operation\n");
        }
    }

    தருக்க kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, அளவு(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare
    எனில் (cpu_comparison_active) {
        எனில் (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, முனை, &rwkv_wkv7_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for RWKV_WKV7 operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END_EXT("RWKV_WKV7", kernel_name, முனை, "S=%d H=%d T=%d n_seqs=%d", (முழு) S, (முழு) H, (முழு) T,
                    (முழு) n_seqs);
    திரும்பு kernel_result;
}

நிலையான ggml_et_cpu_compare_config gated_delta_net_cpu_compare_config = {
    /* .enabled = */ பொய்,
    /* .use_cpu_result = */ பொய்,
    /* .log_differences = */ உண்மை,
    /* .tolerance = */ 1e-4f,
    /* .max_log_elements = */ 4096
};

தருக்க ggml_et_op_gated_delta_net(ggml_backend_et_device_context * dev_ctx, மாறிலி ggml_tensor * முனை) {
    ET_PERF_START();

    எனில் (!dev_ctx || !முனை) {
        GGML_LOG_ERROR("ET: Invalid parameters for GATED_DELTA_NET operation\n");
        திரும்பு பொய்;
    }

    // Validate all 6 source tensors exist
    க்கு (முழு i = 0; i <= 5; i++) {
        எனில் (!முனை->src[i]) {
            GGML_LOG_ERROR("ET: GATED_DELTA_NET operation missing src[%d]\n", i);
            திரும்பு பொய்;
        }
    }

    எனில் (முனை->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: GATED_DELTA_NET only supports F32, got %s\n", ggml_type_name(முனை->type));
        திரும்பு பொய்;
    }

    மாறிலி எழுத்து * kernel_name = "gated_delta_net_f32";

    மாறிலி ggml_tensor * src_q     = முனை->src[0];
    மாறிலி ggml_tensor * src_k     = முனை->src[1];
    மாறிலி ggml_tensor * src_v     = முனை->src[2];
    மாறிலி ggml_tensor * src_g     = முனை->src[3];
    மாறிலி ggml_tensor * src_beta  = முனை->src[4];
    மாறிலி ggml_tensor * src_state = முனை->src[5];

    மாறிலி int64_t S_v      = src_v->ne[0];
    மாறிலி int64_t H        = src_v->ne[1];
    மாறிலி int64_t n_tokens = src_v->ne[2];
    மாறிலி int64_t n_seqs   = src_v->ne[3];
    மாறிலி int64_t H_q      = src_q->ne[1];
    மாறிலி int64_t H_k      = src_k->ne[1];
    மாறிலி int64_t n_seqs_q = src_q->ne[3];
    மாறிலி int64_t n_seqs_k = src_k->ne[3];

    ggml_et_gated_delta_net_params params;
    params.q        = *src_q;
    params.k        = *src_k;
    params.v        = *src_v;
    params.g        = *src_g;
    params.beta     = *src_beta;
    params.state_in = *src_state;
    params.dst      = *முனை;
    params.S_v      = (int32_t) S_v;
    params.H        = (int32_t) H;
    params.H_q      = (int32_t) H_q;
    params.H_k      = (int32_t) H_k;
    params.n_tokens = (int32_t) n_tokens;
    params.n_seqs   = (int32_t) n_seqs;
    params.n_seqs_q = (int32_t) n_seqs_q;
    params.n_seqs_k = (int32_t) n_seqs_k;
    params.kda      = (src_g->ne[0] == S_v) ? 1 : 0;
    params.K        = ggml_get_op_params_i32(முனை, 0);
    params.scale    = 1.0f / sqrtf((தசம) S_v);

    // CPU comparison for debugging
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    தருக்க                    cpu_comparison_active = பொய்;
    எனில் (gated_delta_net_cpu_compare_config.enabled) {
        எனில் (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, முனை, GGML_OP_GATED_DELTA_NET)) {
            cpu_comparison_active = உண்மை;
        } இல்லையெனில் {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for GATED_DELTA_NET operation\n");
        }
    }

    தருக்க kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, அளவு(params), 0xFFFFFFFF);

    எனில் (cpu_comparison_active) {
        எனில் (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, முனை, &gated_delta_net_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for GATED_DELTA_NET operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END_EXT("GATED_DELTA_NET", kernel_name, முனை, "S_v=%d H=%d n_tokens=%d n_seqs=%d kda=%d", (முழு) S_v,
                    (முழு) H, (முழு) n_tokens, (முழு) n_seqs, params.kda);
    திரும்பு kernel_result;
}

தருக்க ggml_et_op_set_rows(ggml_backend_et_device_context * dev_ctx, மாறிலி ggml_tensor * முனை) {
    ET_PERF_START();

    எனில் (!dev_ctx || !முனை) {
        GGML_LOG_ERROR("ET: Invalid parameters for SET_ROWS operation\n");
        திரும்பு பொய்;
    }

    எனில் (!முனை->src[0] || !முனை->src[1] || !முனை->src[2]) {
        GGML_LOG_ERROR(
            "ET: SET_ROWS operation missing required inputs (needs src[0]=base, src[1]=indices, src[2]=data)\n");
        திரும்பு பொய்;
    }

    மாறிலி எழுத்து * kernel_name;

    // Support F32 data with I64 indices -> F32/F16 output (scatter operation)
    எனில் (முனை->src[0]->type == GGML_TYPE_F32 && முனை->src[1]->type == GGML_TYPE_I64 &&
        (முனை->type == GGML_TYPE_F32 || முனை->type == GGML_TYPE_F16)) {
        எனில் (முனை->type == GGML_TYPE_F32 || முனை->type == GGML_TYPE_F16) {
            kernel_name = "set_rows_f32";
        } இல்லையெனில் {
            GGML_LOG_ERROR("ET: SET_ROWS unsupported output type: %s\n", ggml_type_name(முனை->type));
            திரும்பு பொய்;
        }

    } இல்லையெனில் {
        GGML_LOG_ERROR("ET: SET_ROWS operation with unsupported types: dst=%s src0=%s src1=%s\n",
                       ggml_type_name(முனை->type), ggml_type_name(முனை->src[0]->type),
                       ggml_type_name(முனை->src[1]->type));
        திரும்பு பொய்;
    }

    // Validate contiguity requirements
    எனில் (!ggml_is_contiguous_rows(முனை)) {
        GGML_LOG_ERROR("ET: SET_ROWS operation requires contiguous-rows destination tensor\n");
        திரும்பு பொய்;
    }

    எனில் (!ggml_is_contiguous_rows(முனை->src[0])) {
        GGML_LOG_ERROR("ET: SET_ROWS operation requires contiguous-rows source tensor\n");
        திரும்பு பொய்;
    }

    எனில் (!ggml_is_contiguous(முனை->src[1])) {
        GGML_LOG_ERROR("ET: SET_ROWS operation requires contiguous indices tensor\n");
        திரும்பு பொய்;
    }

    // Validate dimension constraints from ggml implementation
    எனில் (!(முனை->ne[0] == முனை->src[0]->ne[0] &&              // same number of columns
          முனை->ne[2] == முனை->src[0]->ne[2] &&              // same batch size
          முனை->ne[3] == முனை->src[0]->ne[3] &&              // same outer dimension
          முனை->src[0]->ne[1] == முனை->src[1]->ne[0] &&      // src rows = index count
          முனை->src[0]->ne[2] % முனை->src[1]->ne[1] == 0 &&  // batch constraint
          முனை->src[0]->ne[3] % முனை->src[1]->ne[2] == 0 &&  // outer constraint
          முனை->src[1]->ne[3] == 1)) {                       // indices constraint
        GGML_LOG_ERROR("ET: SET_ROWS operation dimension constraint failed\n");
        திரும்பு பொய்;
    }

    ggml_et_set_rows_params params;
    params.src0 = *முனை->src[0];  // F32 source data tensor
    params.src1 = *முனை->src[1];  // I64 indices tensor
    params.dst  = *முனை;          // F32/F16 destination tensor

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    தருக்க                    cpu_comparison_active = பொய்;
    எனில் (set_rows_cpu_compare_config.enabled) {
        எனில் (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, முனை, GGML_OP_SET_ROWS)) {
            cpu_comparison_active = உண்மை;
        } இல்லையெனில் {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for SET_ROWS operation\n");
        }
    }

    தருக்க kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, அளவு(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    எனில் (cpu_comparison_active) {
        எனில் (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, முனை, &set_rows_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for SET_ROWS operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("SET_ROWS", kernel_name, முனை);
    திரும்பு kernel_result;
}

தருக்க ggml_et_op_fill(ggml_backend_et_device_context * dev_ctx, மாறிலி ggml_tensor * முனை) {
    ET_PERF_START();

    ggml_et_fill_params params;
    params.dst = *முனை;
    memcpy(&params.c, முனை->op_params, அளவு(தசம));

    தருக்க kernel_result = ggml_et_launch_kernel(dev_ctx, "fill_f32", &params, அளவு(params), 0xFFFFFFFF);

    ET_PERF_END("FILL", "fill_f32", முனை);
    திரும்பு kernel_result;
}

தருக்க ggml_et_op_diag(ggml_backend_et_device_context * dev_ctx, மாறிலி ggml_tensor * முனை) {
    ET_PERF_START();

    எனில் (!முனை->src[0]) {
        GGML_LOG_ERROR("ET: DIAG operation missing source tensor\n");
        திரும்பு பொய்;
    }

    ggml_et_diag_params params;
    params.src0 = *முனை->src[0];
    params.dst  = *முனை;

    தருக்க kernel_result = ggml_et_launch_kernel(dev_ctx, "diag_f32", &params, அளவு(params), 0xFFFFFFFF);

    ET_PERF_END("DIAG", "diag_f32", முனை);
    திரும்பு kernel_result;
}

தருக்க ggml_et_op_tri(ggml_backend_et_device_context * dev_ctx, மாறிலி ggml_tensor * முனை) {
    ET_PERF_START();

    எனில் (!முனை->src[0]) {
        GGML_LOG_ERROR("ET: TRI operation missing source tensor\n");
        திரும்பு பொய்;
    }

    ggml_et_tri_params params;
    params.src0 = *முனை->src[0];
    params.dst  = *முனை;
    memcpy(&params.tri_type, முனை->op_params, அளவு(int32_t));

    தருக்க kernel_result = ggml_et_launch_kernel(dev_ctx, "tri_f32", &params, அளவு(params), 0xFFFFFFFF);

    ET_PERF_END("TRI", "tri_f32", முனை);
    திரும்பு kernel_result;
}

தருக்க ggml_et_op_solve_tri(ggml_backend_et_device_context * dev_ctx, மாறிலி ggml_tensor * முனை) {
    ET_PERF_START();

    எனில் (!முனை->src[0] || !முனை->src[1]) {
        GGML_LOG_ERROR("ET: SOLVE_TRI operation missing source tensor(s)\n");
        திரும்பு பொய்;
    }

    ggml_et_solve_tri_params params;
    params.src0 = *முனை->src[0];  // A (lower-triangular)
    params.src1 = *முனை->src[1];  // B (RHS)
    params.dst  = *முனை;          // X (solution)

    தருக்க kernel_result = ggml_et_launch_kernel(dev_ctx, "solve_tri_f32", &params, அளவு(params), 0xFFFFFFFF);

    ET_PERF_END("SOLVE_TRI", "solve_tri_f32", முனை);
    திரும்பு kernel_result;
}

தருக்க ggml_et_op_set(ggml_backend_et_device_context * dev_ctx, மாறிலி ggml_tensor * முனை) {
    ET_PERF_START();

    எனில் (!முனை->src[0] || !முனை->src[1]) {
        GGML_LOG_ERROR("ET: SET operation missing source tensor(s)\n");
        திரும்பு பொய்;
    }

    மாறிலி தருக்க   inplace = (தருக்க) ((மாறிலி int32_t *) முனை->op_params)[4];
    மாறிலி size_t offset  = ((மாறிலி int32_t *) முனை->op_params)[3];
    மாறிலி size_t nb1     = ((மாறிலி int32_t *) முனை->op_params)[0];
    மாறிலி size_t nb2     = ((மாறிலி int32_t *) முனை->op_params)[1];
    மாறிலி size_t nb3     = ((மாறிலி int32_t *) முனை->op_params)[2];

    எனில் (!inplace) {
        GGML_LOG_ERROR("ET: SET only supports inplace (inplace=%d)\n", inplace);
        திரும்பு பொய்;
    }

    எனில் (முனை->type != GGML_TYPE_F32 || முனை->src[0]->type != GGML_TYPE_F32 || முனை->src[1]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: SET only supports F32 (dst=%s src0=%s src1=%s)\n", ggml_type_name(முனை->type),
                       ggml_type_name(முனை->src[0]->type), ggml_type_name(முனை->src[1]->type));
        திரும்பு பொய்;
    }

    எனில் (!ggml_are_same_shape(முனை, முனை->src[0])) {
        GGML_LOG_ERROR("ET: SET requires same-shape src0 and dst\n");
        திரும்பு பொய்;
    }

    எனில் (!ggml_is_contiguous(முனை) || !ggml_is_contiguous(முனை->src[0]) || !ggml_is_contiguous(முனை->src[1])) {
        GGML_LOG_ERROR("ET: SET requires contiguous dst, src0, and src1\n");
        திரும்பு பொய்;
    }

    ggml_et_set_params params;
    params.src1   = *முனை->src[1];
    params.dst    = *முனை;
    params.nb1    = (int32_t) nb1;
    params.nb2    = (int32_t) nb2;
    params.nb3    = (int32_t) nb3;
    params.offset = (int32_t) offset;

    தருக்க kernel_result = ggml_et_launch_kernel(dev_ctx, "set_f32", &params, அளவு(params), 0xFFFFFFFF);

    ET_PERF_END("SET", "set_f32", முனை);
    திரும்பு kernel_result;
}

தருக்க ggml_et_op_pad(ggml_backend_et_device_context * dev_ctx, மாறிலி ggml_tensor * முனை) {
    ET_PERF_START();

    எனில் (!முனை->src[0]) {
        GGML_LOG_ERROR("ET: PAD operation missing source tensor\n");
        திரும்பு பொய்;
    }

    எனில் (முனை->type != GGML_TYPE_F32 || முனை->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: PAD only supports F32 (src=%s dst=%s)\n", ggml_type_name(முனை->src[0]->type),
                       ggml_type_name(முனை->type));
        திரும்பு பொய்;
    }

    எனில் (!ggml_is_contiguous(முனை)) {
        GGML_LOG_ERROR("ET: PAD requires contiguous output tensor\n");
        திரும்பு பொய்;
    }

    எனில் (முனை->src[0]->nb[0] != அளவு(தசம)) {
        GGML_LOG_ERROR("ET: PAD requires element-contiguous src dim0 (nb[0]=%zu)\n", (size_t) முனை->src[0]->nb[0]);
        திரும்பு பொய்;
    }

    // Extract padding parameters from op_params
    மாறிலி int32_t * op_params = (மாறிலி int32_t *) முனை->op_params;

    ggml_et_pad_params params;
    params.src0  = *முனை->src[0];
    params.dst   = *முனை;
    params.lp[0] = op_params[0];
    params.rp[0] = op_params[1];
    params.lp[1] = op_params[2];
    params.rp[1] = op_params[3];
    params.lp[2] = op_params[4];
    params.rp[2] = op_params[5];
    params.lp[3] = op_params[6];
    params.rp[3] = op_params[7];

    // v1: no dim0 padding
    எனில் (params.lp[0] != 0 || params.rp[0] != 0) {
        GGML_LOG_ERROR("ET: PAD dim0 padding not supported (lp0=%d rp0=%d)\n", params.lp[0], params.rp[0]);
        திரும்பு பொய்;
    }

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    தருக்க                    cpu_comparison_active = பொய்;
    எனில் (pad_cpu_compare_config.enabled) {
        எனில் (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, முனை, GGML_OP_PAD)) {
            cpu_comparison_active = உண்மை;
        } இல்லையெனில் {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for PAD operation\n");
        }
    }

    தருக்க kernel_result = ggml_et_launch_kernel(dev_ctx, "pad_f32", &params, அளவு(params), 0xFFFFFFFF);

    எனில் (cpu_comparison_active) {
        எனில் (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, முனை, &pad_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for PAD operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("PAD", "pad_f32", முனை);
    திரும்பு kernel_result;
}
