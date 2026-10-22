#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-et-ops.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-et-cpu-compare.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-et-kernels.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <stdio.h>

#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>

// CPU comparison configuration - can be enabled for debugging
ਸਥਿਰ ggml_et_cpu_compare_config rope_cpu_compare_config = {
    /* .enabled = */ ਝੂਠ,
    /* .use_cpu_result = */ ਝੂਠ,  // Replace ET result with CPU result
    /* .log_differences = */ ਸੱਚ,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

ਸਥਿਰ ggml_et_cpu_compare_config rms_norm_cpu_compare_config = {
    /* .enabled = */ ਝੂਠ,
    /* .use_cpu_result = */ ਝੂਠ,
    /* .log_differences = */ ਸੱਚ,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

ਸਥਿਰ ggml_et_cpu_compare_config norm_cpu_compare_config = {
    /* .enabled = */ ਝੂਠ,
    /* .use_cpu_result = */ ਝੂਠ,
    /* .log_differences = */ ਸੱਚ,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

ਸਥਿਰ ggml_et_cpu_compare_config l2_norm_cpu_compare_config = {
    /* .enabled = */ ਝੂਠ,
    /* .use_cpu_result = */ ਝੂਠ,
    /* .log_differences = */ ਸੱਚ,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

ਸਥਿਰ ggml_et_cpu_compare_config ਸਮੂਹ_norm_cpu_compare_config = {
    /* .enabled = */ ਝੂਠ,
    /* .use_cpu_result = */ ਝੂਠ,
    /* .log_differences = */ ਸੱਚ,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

ਸਥਿਰ ggml_et_cpu_compare_config im2col_cpu_compare_config = {
    /* .enabled = */ ਝੂਠ,
    /* .use_cpu_result = */ ਝੂਠ,
    /* .log_differences = */ ਸੱਚ,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

ਸਥਿਰ ggml_et_cpu_compare_config unary_cpu_compare_config = {
    /* .enabled = */ ਝੂਠ,
    /* .use_cpu_result = */ ਝੂਠ,
    /* .log_differences = */ ਸੱਚ,
    /* .tolerance = */ 1e-4f,
    /* .max_log_elements = */ 4096
};

ਸਥਿਰ ggml_et_cpu_compare_config ਜੋੜ_ਪੰਕਤੀਆਂ_cpu_compare_config = {
    /* .enabled = */ ਝੂਠ,
    /* .use_cpu_result = */ ਝੂਠ,
    /* .log_differences = */ ਸੱਚ,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

ਸਥਿਰ ggml_et_cpu_compare_config clamp_cpu_compare_config = {
    /* .enabled = */ ਝੂਠ,
    /* .use_cpu_result = */ ਝੂਠ,
    /* .log_differences = */ ਸੱਚ,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

ਸਥਿਰ ggml_et_cpu_compare_config ਔਸਤ_cpu_compare_config = {
    /* .enabled = */ ਝੂਠ,
    /* .use_cpu_result = */ ਝੂਠ,
    /* .log_differences = */ ਸੱਚ,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

ਸਥਿਰ ggml_et_cpu_compare_config sqr_cpu_compare_config = {
    /* .enabled = */ ਝੂਠ,
    /* .use_cpu_result = */ ਝੂਠ,
    /* .log_differences = */ ਸੱਚ,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

ਸਥਿਰ ggml_et_cpu_compare_config elmap_cpu_compare_config = {
    /* .enabled = */ ਝੂਠ,
    /* .use_cpu_result = */ ਝੂਠ,
    /* .log_differences = */ ਸੱਚ,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

ਸਥਿਰ ggml_et_cpu_compare_config glu_cpu_compare_config = {
    /* .enabled = */ ਝੂਠ,
    /* .use_cpu_result = */ ਝੂਠ,
    /* .log_differences = */ ਸੱਚ,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

ਸਥਿਰ ggml_et_cpu_compare_config mul_mat_cpu_compare_config = {
    /* .enabled = */ ਝੂਠ,
    /* .use_cpu_result = */ ਝੂਠ,
    /* .log_differences = */ ਸੱਚ,
    /* .tolerance = */ 0.01,
    /* .max_log_elements = */ 4096
};

ਸਥਿਰ ggml_et_cpu_compare_config mul_mat_id_cpu_compare_config = {
    /* .enabled = */ ਝੂਠ,
    /* .use_cpu_result = */ ਝੂਠ,
    /* .log_differences = */ ਸੱਚ,
    /* .tolerance = */ 0.01,
    /* .max_log_elements = */ 4096
};

ਸਥਿਰ ggml_et_cpu_compare_config softmax_cpu_compare_config = {
    /* .enabled = */ ਝੂਠ,
    /* .use_cpu_result = */ ਝੂਠ,
    /* .log_differences = */ ਸੱਚ,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 1024
};

ਸਥਿਰ ggml_et_cpu_compare_config get_ਪੰਕਤੀਆਂ_cpu_compare_config = {
    /* .enabled = */ ਝੂਠ,
    /* .use_cpu_result = */ ਝੂਠ,
    /* .log_differences = */ ਸੱਚ,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 2048
};

ਸਥਿਰ ggml_et_cpu_compare_config pad_cpu_compare_config = {
    /* .enabled = */ ਝੂਠ,
    /* .use_cpu_result = */ ਝੂਠ,
    /* .log_differences = */ ਸੱਚ,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

ਸਥਿਰ ggml_et_cpu_compare_config cont_cpu_compare_config = {
    /* .enabled = */ ਝੂਠ,
    /* .use_cpu_result = */ ਝੂਠ,
    /* .log_differences = */ ਸੱਚ,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

ਸਥਿਰ ggml_et_cpu_compare_config concat_cpu_compare_config = {
    /* .enabled = */ ਝੂਠ,
    /* .use_cpu_result = */ ਝੂਠ,
    /* .log_differences = */ ਸੱਚ,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

ਸਥਿਰ ggml_et_cpu_compare_config cumsum_cpu_compare_config = {
    /* .enabled = */ ਝੂਠ,
    /* .use_cpu_result = */ ਝੂਠ,
    /* .log_differences = */ ਸੱਚ,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

ਸਥਿਰ ggml_et_cpu_compare_config repeat_cpu_compare_config = {
    /* .enabled = */ ਝੂਠ,
    /* .use_cpu_result = */ ਝੂਠ,
    /* .log_differences = */ ਸੱਚ,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

ਸਥਿਰ ggml_et_cpu_compare_config ssm_conv_cpu_compare_config = {
    /* .enabled = */ ਝੂਠ,
    /* .use_cpu_result = */ ਝੂਠ,
    /* .log_differences = */ ਸੱਚ,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

ਸਥਿਰ ggml_et_cpu_compare_config rwkv_wkv6_cpu_compare_config = {
    /* .enabled = */ ਝੂਠ,
    /* .use_cpu_result = */ ਝੂਠ,
    /* .log_differences = */ ਸੱਚ,
    /* .tolerance = */ 1e-4f,
    /* .max_log_elements = */ 4096
};

ਸਥਿਰ ggml_et_cpu_compare_config rwkv_wkv7_cpu_compare_config = {
    /* .enabled = */ ਝੂਠ,
    /* .use_cpu_result = */ ਝੂਠ,
    /* .log_differences = */ ਸੱਚ,
    /* .tolerance = */ 1e-4f,
    /* .max_log_elements = */ 4096
};

ਸਥਿਰ ggml_et_cpu_compare_config ਸੈੱਟ_ਪੰਕਤੀਆਂ_cpu_compare_config = {
    /* .enabled = */ ਝੂਠ,
    /* .use_cpu_result = */ ਝੂਠ,
    /* .log_differences = */ ਸੱਚ,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 2048
};

ਤਾਰਕਿਕ ggml_et_op_rms_norm_mul(ggml_backend_et_device_context * dev_ctx,
                             ਅਚਲ ggml_tensor *              rms_norm_ਗੰਢ,
                             ਅਚਲ ggml_tensor *              mul_ਗੰਢ) {
    ET_PERF_ਸ਼ੁਰੂ();

    ਜੇ (!dev_ctx || !rms_norm_ਗੰਢ || !mul_ਗੰਢ) {
        GGML_LOG_ERROR("ET: Invalid parameters for fused RMS_NORM_MUL operation\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (!rms_norm_ਗੰਢ->src[0]) {
        GGML_LOG_ERROR("ET: Fused RMS_NORM_MUL missing required input\n");
        ਵਾਪਸ ਝੂਠ;
    }

    // Extract weights: the MUL operand that isn't the rms_norm output
    ਅਚਲ ggml_tensor * weights = (mul_ਗੰਢ->src[0] == rms_norm_ਗੰਢ) ? mul_ਗੰਢ->src[1] : mul_ਗੰਢ->src[0];

    ਜੇ (!weights) {
        GGML_LOG_ERROR("ET: Fused RMS_NORM_MUL missing weights tensor\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਦਸ਼ਮਲਵ eps;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&eps, rms_norm_ਗੰਢ->op_params, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ggml_et_rms_norm_mul_params params;
    params.src0 = *rms_norm_ਗੰਢ->src[0];  // input to normalize
    params.src1 = *weights;                // normalization weights
    params.dst  = *mul_ਗੰਢ;               // final output
    params.eps  = eps;

    ਤਾਰਕਿਕ kernel_ਨਤੀਜਾ = ggml_et_launch_kernel(dev_ctx, "rms_norm_mul_f32", &params, ਆਕਾਰ(params), 0xFFFFFFFF);

    ET_PERF_ਸਿਰਾ_EXT("RMS_NORM_MUL", "rms_norm_mul_f32", mul_ਗੰਢ, "eps=%.6f", (ਦੁੱਗਣਾ) eps);
    ਵਾਪਸ kernel_ਨਤੀਜਾ;
}

ਤਾਰਕਿਕ ggml_et_op_scale(ggml_backend_et_device_context * dev_ctx, ਅਚਲ ggml_tensor * ਗੰਢ) {
    ET_PERF_ਸ਼ੁਰੂ();

    ਜੇ (!dev_ctx || !ਗੰਢ) {
        GGML_LOG_ERROR("ET: Invalid parameters for SCALE operation\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (!ਗੰਢ->src[0]) {
        GGML_LOG_ERROR("ET: SCALE operation missing required input\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (ਗੰਢ->type != GGML_TYPE_F32 || ਗੰਢ->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: SCALE operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(ਗੰਢ->type),
                       ggml_type_name(ਗੰਢ->src[0]->type));
        ਵਾਪਸ ਝੂਠ;
    }

    ਦਸ਼ਮਲਵ scale, bias;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&scale, (ਅਚਲ ਦਸ਼ਮਲਵ *) ਗੰਢ->op_params + 0, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&bias, (ਅਚਲ ਦਸ਼ਮਲਵ *) ਗੰਢ->op_params + 1, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ggml_et_scale_params params;
    params.src0  = *ਗੰਢ->src[0];
    params.dst   = *ਗੰਢ;
    params.scale = scale;
    params.bias  = bias;

    ਤਾਰਕਿਕ kernel_ਨਤੀਜਾ = ggml_et_launch_kernel(dev_ctx, "scale_f32", &params, ਆਕਾਰ(params), 0xFFFFFFFF);

    ET_PERF_ਸਿਰਾ_EXT("SCALE", "scale_f32", ਗੰਢ, "scale=%.6f|bias=%.6f", (ਦੁੱਗਣਾ) scale, (ਦੁੱਗਣਾ) bias);
    ਵਾਪਸ kernel_ਨਤੀਜਾ;
}

ਤਾਰਕਿਕ ggml_et_op_sqr(ggml_backend_et_device_context * dev_ctx, ਅਚਲ ggml_tensor * ਗੰਢ) {
    ET_PERF_ਸ਼ੁਰੂ();

    ਜੇ (!dev_ctx || !ਗੰਢ) {
        GGML_LOG_ERROR("ET: Invalid parameters for SQR operation\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (!ਗੰਢ->src[0]) {
        GGML_LOG_ERROR("ET: SQR operation missing required input\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (ਗੰਢ->type != GGML_TYPE_F32 || ਗੰਢ->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: SQR operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(ਗੰਢ->type),
                       ggml_type_name(ਗੰਢ->src[0]->type));
        ਵਾਪਸ ਝੂਠ;
    }

    ggml_et_sqr_params params;
    params.src0 = *ਗੰਢ->src[0];  // F32 input tensor
    params.dst  = *ਗੰਢ;          // F32 output tensor

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ਤਾਰਕਿਕ                    cpu_comparison_ਸਰਗਰਮ = ਝੂਠ;
    ਜੇ (sqr_cpu_compare_config.enabled) {
        ਜੇ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ਗੰਢ, GGML_OP_SQR)) {
            cpu_comparison_ਸਰਗਰਮ = ਸੱਚ;
        } ਨਹੀਂਤਾਂ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for SQR operation\n");
        }
    }

    ਤਾਰਕਿਕ kernel_ਨਤੀਜਾ = ggml_et_launch_kernel(dev_ctx, "sqr_f32", &params, ਆਕਾਰ(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ਜੇ (cpu_comparison_ਸਰਗਰਮ) {
        ਜੇ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ਗੰਢ, &sqr_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for SQR operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_ਸਿਰਾ("SQR", "sqr_f32", ਗੰਢ);
    ਵਾਪਸ kernel_ਨਤੀਜਾ;
}

ਤਾਰਕਿਕ ggml_et_op_ਜੋੜ_ਪੰਕਤੀਆਂ(ggml_backend_et_device_context * dev_ctx, ਅਚਲ ggml_tensor * ਗੰਢ) {
    ET_PERF_ਸ਼ੁਰੂ();

    ਜੇ (!dev_ctx || !ਗੰਢ) {
        GGML_LOG_ERROR("ET: Invalid parameters for SUM_ROWS operation\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (!ਗੰਢ->src[0]) {
        GGML_LOG_ERROR("ET: SUM_ROWS operation missing required input\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (ਗੰਢ->type != GGML_TYPE_F32 || ਗੰਢ->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: SUM_ROWS operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(ਗੰਢ->type),
                       ggml_type_name(ਗੰਢ->src[0]->type));
        ਵਾਪਸ ਝੂਠ;
    }

    ggml_et_ਜੋੜ_ਪੰਕਤੀਆਂ_params params;
    params.src0 = *ਗੰਢ->src[0];
    params.dst  = *ਗੰਢ;

    // Phase 1: Initialize CPU comparison context
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ਤਾਰਕਿਕ                    cpu_comparison_ਸਰਗਰਮ = ਝੂਠ;
    ਜੇ (ਜੋੜ_ਪੰਕਤੀਆਂ_cpu_compare_config.enabled) {
        ਜੇ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ਗੰਢ, GGML_OP_ਜੋੜ_ਪੰਕਤੀਆਂ)) {
            cpu_comparison_ਸਰਗਰਮ = ਸੱਚ;
        } ਨਹੀਂਤਾਂ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for SUM_ROWS operation\n");
        }
    }

    ਤਾਰਕਿਕ kernel_ਨਤੀਜਾ = ggml_et_launch_kernel(dev_ctx, "sum_rows_f32", &params, ਆਕਾਰ(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare
    ਜੇ (cpu_comparison_ਸਰਗਰਮ) {
        ਜੇ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ਗੰਢ, &ਜੋੜ_ਪੰਕਤੀਆਂ_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for SUM_ROWS operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_ਸਿਰਾ("SUM_ROWS", "sum_rows_f32", ਗੰਢ);
    ਵਾਪਸ kernel_ਨਤੀਜਾ;
}

ਤਾਰਕਿਕ ggml_et_op_ਔਸਤ(ggml_backend_et_device_context * dev_ctx, ਅਚਲ ggml_tensor * ਗੰਢ) {
    ET_PERF_ਸ਼ੁਰੂ();

    ਜੇ (!dev_ctx || !ਗੰਢ) {
        GGML_LOG_ERROR("ET: Invalid parameters for MEAN operation\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (!ਗੰਢ->src[0]) {
        GGML_LOG_ERROR("ET: MEAN operation missing required input\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (ਗੰਢ->type != GGML_TYPE_F32 || ਗੰਢ->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: MEAN operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(ਗੰਢ->type),
                       ggml_type_name(ਗੰਢ->src[0]->type));
        ਵਾਪਸ ਝੂਠ;
    }

    ggml_et_ਔਸਤ_params params;
    params.src0 = *ਗੰਢ->src[0];
    params.dst  = *ਗੰਢ;

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ਤਾਰਕਿਕ                    cpu_comparison_ਸਰਗਰਮ = ਝੂਠ;
    ਜੇ (ਔਸਤ_cpu_compare_config.enabled) {
        ਜੇ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ਗੰਢ, GGML_OP_ਔਸਤ)) {
            cpu_comparison_ਸਰਗਰਮ = ਸੱਚ;
        } ਨਹੀਂਤਾਂ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for MEAN operation\n");
        }
    }

    ਤਾਰਕਿਕ kernel_ਨਤੀਜਾ = ggml_et_launch_kernel(dev_ctx, "mean_f32", &params, ਆਕਾਰ(params), 0xFFFFFFFF);

    ਜੇ (cpu_comparison_ਸਰਗਰਮ) {
        ਜੇ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ਗੰਢ, &ਔਸਤ_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for MEAN operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_ਸਿਰਾ("MEAN", "mean_f32", ਗੰਢ);
    ਵਾਪਸ kernel_ਨਤੀਜਾ;
}

ਤਾਰਕਿਕ ggml_et_op_clamp(ggml_backend_et_device_context * dev_ctx, ਅਚਲ ggml_tensor * ਗੰਢ) {
    ET_PERF_ਸ਼ੁਰੂ();

    ਜੇ (!dev_ctx || !ਗੰਢ) {
        GGML_LOG_ERROR("ET: Invalid parameters for CLAMP operation\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (!ਗੰਢ->src[0]) {
        GGML_LOG_ERROR("ET: CLAMP operation missing required input\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (ਗੰਢ->type != GGML_TYPE_F32 || ਗੰਢ->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: CLAMP operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(ਗੰਢ->type),
                       ggml_type_name(ਗੰਢ->src[0]->type));
        ਵਾਪਸ ਝੂਠ;
    }

    ggml_et_clamp_params params;
    params.src0 = *ਗੰਢ->src[0];
    params.dst  = *ਗੰਢ;
    // op_params layout per ggml.c::ggml_clamp: { min, max } as floats
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&params.min_val, (ਅਚਲ ਦਸ਼ਮਲਵ *) ਗੰਢ->op_params + 0, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&params.max_val, (ਅਚਲ ਦਸ਼ਮਲਵ *) ਗੰਢ->op_params + 1, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ਤਾਰਕਿਕ                    cpu_comparison_ਸਰਗਰਮ = ਝੂਠ;
    ਜੇ (clamp_cpu_compare_config.enabled) {
        ਜੇ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ਗੰਢ, GGML_OP_CLAMP)) {
            cpu_comparison_ਸਰਗਰਮ = ਸੱਚ;
        } ਨਹੀਂਤਾਂ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for CLAMP operation\n");
        }
    }

    ਤਾਰਕਿਕ kernel_ਨਤੀਜਾ = ggml_et_launch_kernel(dev_ctx, "clamp_f32", &params, ਆਕਾਰ(params), 0xFFFFFFFF);

    ਜੇ (cpu_comparison_ਸਰਗਰਮ) {
        ਜੇ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ਗੰਢ, &clamp_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for CLAMP operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_ਸਿਰਾ("CLAMP", "clamp_f32", ਗੰਢ);
    ਵਾਪਸ kernel_ਨਤੀਜਾ;
}

ਤਾਰਕਿਕ ggml_et_op_unary(ggml_backend_et_device_context * dev_ctx, ਅਚਲ ggml_tensor * ਗੰਢ) {
    ET_PERF_ਸ਼ੁਰੂ();

    ਜੇ (!dev_ctx || !ਗੰਢ) {
        GGML_LOG_ERROR("ET: Invalid parameters for UNARY operation\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (!ਗੰਢ->src[0]) {
        GGML_LOG_ERROR("ET: UNARY operation missing required input\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (ਗੰਢ->type != GGML_TYPE_F32 || ਗੰਢ->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: UNARY operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(ਗੰਢ->type),
                       ggml_type_name(ਗੰਢ->src[0]->type));
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ ggml_unary_op uop     = ggml_get_unary_op(ਗੰਢ);
    ਅਚਲ ਅੱਖਰ *        op_name = ggml_unary_op_name(uop);

    ggml_et_unary_params params;
    params.src0     = *ਗੰਢ->src[0];  // F32 input tensor
    params.dst      = *ਗੰਢ;          // F32 output tensor
    params.unary_op = (int32_t) uop;

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ਤਾਰਕਿਕ                    cpu_comparison_ਸਰਗਰਮ = ਝੂਠ;
    ਜੇ (unary_cpu_compare_config.enabled) {
        ਜੇ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ਗੰਢ, GGML_OP_UNARY)) {
            cpu_comparison_ਸਰਗਰਮ = ਸੱਚ;
        } ਨਹੀਂਤਾਂ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for UNARY/%s operation\n", op_name);
        }
    }

    ਤਾਰਕਿਕ kernel_ਨਤੀਜਾ = ggml_et_launch_kernel(dev_ctx, "unary_f32", &params, ਆਕਾਰ(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ਜੇ (cpu_comparison_ਸਰਗਰਮ) {
        ਜੇ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ਗੰਢ, &unary_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for UNARY/%s operation\n", op_name);
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_ਸਿਰਾ_EXT("UNARY", "unary_f32", ਗੰਢ, "op=%s", op_name);
    ਵਾਪਸ kernel_ਨਤੀਜਾ;
}

ਤਾਰਕਿਕ ggml_et_op_mul(ggml_backend_et_device_context * dev_ctx, ਅਚਲ ggml_tensor * ਗੰਢ) {
    // Delegate to generic element map operation
    ਵਾਪਸ ggml_et_op_elmap(dev_ctx, ਗੰਢ);
}

ਤਾਰਕਿਕ ggml_et_op_ਜੋੜੋ(ggml_backend_et_device_context * dev_ctx, ਅਚਲ ggml_tensor * ਗੰਢ) {
    // Delegate to generic element map operation
    ਵਾਪਸ ggml_et_op_elmap(dev_ctx, ਗੰਢ);
}

ਤਾਰਕਿਕ ggml_et_op_sub(ggml_backend_et_device_context * dev_ctx, ਅਚਲ ggml_tensor * ਗੰਢ) {
    // Delegate to generic element map operation
    ਵਾਪਸ ggml_et_op_elmap(dev_ctx, ਗੰਢ);
}

ਤਾਰਕਿਕ ggml_et_op_elmap(ggml_backend_et_device_context * dev_ctx, ਅਚਲ ggml_tensor * ਗੰਢ) {
    ET_PERF_ਸ਼ੁਰੂ();

    ਜੇ (!dev_ctx || !ਗੰਢ) {
        GGML_LOG_ERROR("ET: Invalid parameters for element map operation\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (!ਗੰਢ->src[0] || !ਗੰਢ->src[1]) {
        GGML_LOG_ERROR("ET: Element map operation missing required inputs\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (ਗੰਢ->type != GGML_TYPE_F32 || ਗੰਢ->src[0]->type != GGML_TYPE_F32 || ਗੰਢ->src[1]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: Element map operation with unsupported types: dst=%s src0=%s src1=%s\n",
                       ggml_type_name(ਗੰਢ->type), ggml_type_name(ਗੰਢ->src[0]->type),
                       ggml_type_name(ਗੰਢ->src[1]->type));
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ ਅੱਖਰ * op_name = ggml_op_name(ਗੰਢ->op);

    ggml_et_elmap_params params;
    params.src0 = *ਗੰਢ->src[0];
    params.src1 = *ਗੰਢ->src[1];
    params.dst  = *ਗੰਢ;  // F32 output tensor (op type stored in dst.op)

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ਤਾਰਕਿਕ                    cpu_comparison_ਸਰਗਰਮ = ਝੂਠ;
    ਜੇ (elmap_cpu_compare_config.enabled) {
        ਜੇ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ਗੰਢ, ਗੰਢ->op)) {
            cpu_comparison_ਸਰਗਰਮ = ਸੱਚ;
        } ਨਹੀਂਤਾਂ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for %s operation\n", op_name);
        }
    }

    // fprintf(stderr, "ET: el_map s0 [%ld, %ld, %ld, %ld] s1 [%ld, %ld, %ld, %ld]\n",
    //     node->src[0]->ne[0], node->src[0]->ne[1], node->src[0]->ne[2], node->src[0]->ne[3],
    //     node->src[1]->ne[0], node->src[1]->ne[1], node->src[1]->ne[2], node->src[1]->ne[3]);

    ਤਾਰਕਿਕ kernel_ਨਤੀਜਾ = ggml_et_launch_kernel(dev_ctx, "el_map_f32", &params, ਆਕਾਰ(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ਜੇ (cpu_comparison_ਸਰਗਰਮ) {
        ਜੇ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ਗੰਢ, &elmap_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for %s operation\n", op_name);
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_ਸਿਰਾ(op_name, "el_map_f32", ਗੰਢ);
    ਵਾਪਸ kernel_ਨਤੀਜਾ;
}

ਤਾਰਕਿਕ ggml_et_op_glu(ggml_backend_et_device_context * dev_ctx, ਅਚਲ ggml_tensor * ਗੰਢ) {
    ET_PERF_ਸ਼ੁਰੂ();

    // Validate inputs
    ਜੇ (!dev_ctx || !ਗੰਢ) {
        GGML_LOG_ERROR("ET: Invalid parameters for GLU operation\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (!ਗੰਢ->src[0]) {
        GGML_LOG_ERROR("ET: GLU operation missing required input\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ ਤਾਰਕਿਕ is_ਵੰਡੋ_ਬਹੁਲਕ = ਗੰਢ->src[1] != ਖਾਲੀ_ਸੂਚਕ;

    // Only support F32 (as validated by supports_op)
    ਜੇ (ਗੰਢ->type != GGML_TYPE_F32 || ਗੰਢ->src[0]->type != GGML_TYPE_F32 ||
        (is_ਵੰਡੋ_ਬਹੁਲਕ && ਗੰਢ->src[1]->type != GGML_TYPE_F32)) {
        ਵਾਪਸ ਝੂਠ;
    }

    // Extract GLU operation parameters from op_params
    int32_t glu_op_type = ggml_get_op_params_i32(ਗੰਢ, 0);  // GLU variant (REGLU, GEGLU, SWIGLU, etc.)
    int32_t swapped     = ggml_get_op_params_i32(ਗੰਢ, 1);  // Whether gate/value are swapped

    // Supported variants
    ਚੋਣ (glu_op_type) {
        ਹਾਲਤ GGML_GLU_OP_REGLU:
        ਹਾਲਤ GGML_GLU_OP_GEGLU:
        ਹਾਲਤ GGML_GLU_OP_SWIGLU:
        ਹਾਲਤ GGML_GLU_OP_SWIGLU_OAI:
        ਹਾਲਤ GGML_GLU_OP_SWIGLU_CLAMP:
        ਹਾਲਤ GGML_GLU_OP_GEGLU_ERF:
        ਹਾਲਤ GGML_GLU_OP_GEGLU_ਤੇਜ਼:
            ਰੋਕੋ;
        ਮੂਲ:
            GGML_LOG_ERROR("ET: GLU operation with unsupported variant: %s\n",
                           ggml_glu_op_name((ggml_glu_op) glu_op_type));
            ਵਾਪਸ ਝੂਠ;
    }

    // Get GLU operation name for logging
    ਅਚਲ ਅੱਖਰ * glu_op_name = ggml_glu_op_name((ggml_glu_op) glu_op_type);

    // Pack parameters. Single-tensor mode is encoded by zeroing src1.
    ggml_et_glu_params params = {};
    params.src0               = *ਗੰਢ->src[0];
    ਜੇ (is_ਵੰਡੋ_ਬਹੁਲਕ) {
        params.src1 = *ਗੰਢ->src[1];
    }
    params.dst         = *ਗੰਢ;
    params.glu_op_type = glu_op_type;
    params.swapped     = swapped;
    params.alpha       = 0.0f;
    params.ਹੱਦਬੰਦੀ       = 0.0f;
    ਜੇ (glu_op_type == GGML_GLU_OP_SWIGLU_OAI) {
        params.alpha = ggml_get_op_params_f32(ਗੰਢ, 2);
    }
    ਜੇ (glu_op_type == GGML_GLU_OP_SWIGLU_OAI || glu_op_type == GGML_GLU_OP_SWIGLU_CLAMP) {
        params.ਹੱਦਬੰਦੀ = ggml_get_op_params_f32(ਗੰਢ, 3);
    }
    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ਤਾਰਕਿਕ                    cpu_comparison_ਸਰਗਰਮ = ਝੂਠ;
    ਜੇ (glu_cpu_compare_config.enabled) {
        ਜੇ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ਗੰਢ, GGML_OP_GLU)) {
            cpu_comparison_ਸਰਗਰਮ = ਸੱਚ;
        } ਨਹੀਂਤਾਂ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for %s operation\n", glu_op_name);
        }
    }

    // Launch ET kernel
    ਤਾਰਕਿਕ kernel_ਨਤੀਜਾ = ggml_et_launch_kernel(dev_ctx, "glu_f32", &params, ਆਕਾਰ(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ਜੇ (cpu_comparison_ਸਰਗਰਮ) {
        ਜੇ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ਗੰਢ, &glu_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for %s operation\n", glu_op_name);
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_ਸਿਰਾ("GLU", "glu_f32", ਗੰਢ);
    ਵਾਪਸ kernel_ਨਤੀਜਾ;
}

ਤਾਰਕਿਕ ggml_et_op_mul_mat(ggml_backend_et_device_context * dev_ctx,
                        ਅਚਲ ggml_tensor *              ਗੰਢ,
                        ਅਚਲ ggml_tensor *              ਜੋੜੋ_ਗੰਢ) {
    ET_PERF_ਸ਼ੁਰੂ();

    ਜੇ (!dev_ctx || !ਗੰਢ) {
        GGML_LOG_ERROR("ET: Invalid parameters for MUL_MAT operation\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (!ਗੰਢ->src[0] || !ਗੰਢ->src[1]) {
        GGML_LOG_ERROR("ET: MUL_MAT operation missing required inputs\n");
        ਵਾਪਸ ਝੂਠ;
    }

    // Fused MM+ADD: when add_node is non-NULL the caller has already validated
    // (Q8_0 weights, F32 acts, exact-shape ADD with stride parity to dst) via
    // ggml_et_can_fuse({MUL_MAT, ADD}). The kernel writes dst = mm + bias and
    // the ADD's output replaces MM's as the actual dst.
    ਅਚਲ ggml_tensor * fused_dst   = ਜੋੜੋ_ਗੰਢ ? ਜੋੜੋ_ਗੰਢ : ਗੰਢ;
    ਅਚਲ ggml_tensor * bias_tensor = ਖਾਲੀ_ਸੂਚਕ;
    ਜੇ (ਜੋੜੋ_ਗੰਢ) {
        bias_tensor = (ਜੋੜੋ_ਗੰਢ->src[0] == ਗੰਢ) ? ਜੋੜੋ_ਗੰਢ->src[1] : ਜੋੜੋ_ਗੰਢ->src[0];
    }

    ਅਚਲ ਅੱਖਰ * kernel_name;
    ਅਚਲ ਅੱਖਰ * src0_type_name;

    ਜੇ (ਗੰਢ->type == GGML_TYPE_F32 && ਗੰਢ->src[0]->type == GGML_TYPE_Q4_0 && ਗੰਢ->src[1]->type == GGML_TYPE_F32 &&
        ਗੰਢ->src[1]->ne[1] >= 53 &&      // N >= 53
        ਗੰਢ->src[0]->ne[1] % 16 == 0 &&  // M % TILE_M
        ਗੰਢ->src[0]->ne[0] % 32 == 0) {  // K % BLOCK_K (Q4_0 block)

        // Matrix engine for N >= 53; partial N (via n_cur-1) and errata padding are handled in-kernel.
        kernel_name    = "mul_mat_Q4_0_matrix_engine";
        src0_type_name = "Q4_0";

    } ਨਹੀਂਤਾਂ ਜੇ (ਗੰਢ->type == GGML_TYPE_F32 && ਗੰਢ->src[0]->type == GGML_TYPE_Q4_0 &&
               ਗੰਢ->src[1]->type == GGML_TYPE_F32) {
        kernel_name    = "mul_mat_Q4_0";  // N < 53, or M % 16 != 0 or K % 32 != 0
        src0_type_name = "Q4_0";

    } ਨਹੀਂਤਾਂ ਜੇ (ਗੰਢ->type == GGML_TYPE_F32 && ਗੰਢ->src[0]->type == GGML_TYPE_Q8_0 &&
               ਗੰਢ->src[1]->type == GGML_TYPE_F32) {
        kernel_name    = "mul_mat_Q8_0";
        src0_type_name = "Q8_0";

    } ਨਹੀਂਤਾਂ ਜੇ (ਗੰਢ->type == GGML_TYPE_F32 && ਗੰਢ->src[0]->type == GGML_TYPE_F16 &&
               ਗੰਢ->src[1]->type == GGML_TYPE_F16 && ਗੰਢ->ne[0] % 16 == 0 && ਗੰਢ->src[0]->ne[0] % 16 == 0 &&
               ਗੰਢ->src[0]->ne[1] % 16 == 0 && ਗੰਢ->src[1]->ne[0] != 1) {
        kernel_name    = "mul_mat_f16_matrix_engine";
        src0_type_name = "F16";

    } ਨਹੀਂਤਾਂ ਜੇ (ਗੰਢ->type == GGML_TYPE_F32 && ਗੰਢ->src[0]->type == GGML_TYPE_F16 &&
               (ਗੰਢ->src[1]->type == GGML_TYPE_F16 || ਗੰਢ->src[1]->type == GGML_TYPE_F32)) {
        kernel_name    = "mul_mat_f16";
        src0_type_name = "F16";

    } ਨਹੀਂਤਾਂ ਜੇ (ਗੰਢ->type == GGML_TYPE_F32 && ਗੰਢ->src[0]->type == GGML_TYPE_F32 &&
               ਗੰਢ->src[1]->type == GGML_TYPE_F32 && ਗੰਢ->ne[0] % 16 == 0 && ਗੰਢ->src[0]->ne[0] % 16 == 0 &&
               ਗੰਢ->src[0]->ne[1] % 16 == 0 && ਗੰਢ->src[1]->ne[0] != 1) {  // GEMV is faster with the generic path

        kernel_name    = "mul_mat_f32_matrix_engine";
        src0_type_name = "F32";
    } ਨਹੀਂਤਾਂ ਜੇ (ਗੰਢ->type == GGML_TYPE_F32 && ਗੰਢ->src[0]->type == GGML_TYPE_F32 &&
               (ਗੰਢ->src[1]->type == GGML_TYPE_F16 || ਗੰਢ->src[1]->type == GGML_TYPE_F32)) {
        kernel_name    = "mul_mat_f32";
        src0_type_name = "F32";
    } ਨਹੀਂਤਾਂ {
        GGML_LOG_ERROR("ET: MUL_MAT operation with unsupported types: dst=%s src0=%s src1=%s\n",
                       ggml_type_name(ਗੰਢ->type), ggml_type_name(ਗੰਢ->src[0]->type),
                       ggml_type_name(ਗੰਢ->src[1]->type));
        ਵਾਪਸ ਝੂਠ;
    }

    ggml_et_ਦੁਆਧਾਰੀ_params params;
    params.src0 = *ਗੰਢ->src[0];  // weight matrix
    params.src1 = *ਗੰਢ->src[1];  // activation matrix
    params.dst  = *fused_dst;     // output (= add_node when fused, else node)

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ਤਾਰਕਿਕ                    cpu_comparison_ਸਰਗਰਮ = ਝੂਠ;
    ਜੇ (mul_mat_cpu_compare_config.enabled) {
        ਜੇ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, fused_dst, GGML_OP_MUL_MAT)) {
            cpu_comparison_ਸਰਗਰਮ = ਸੱਚ;
        } ਨਹੀਂਤਾਂ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for MUL_MAT operation\n");
        }
    }

    ਤਾਰਕਿਕ kernel_ਨਤੀਜਾ;
    ਜੇ (ਗੰਢ->src[0]->type == GGML_TYPE_Q8_0) {
        // Q8_0 kernel always takes the extended struct. bias.data is non-NULL
        // only on the fused path; otherwise the kernel skips the add entirely.
        ggml_et_mm_q8_params q8_params = {};
        q8_params.src0                 = params.src0;
        q8_params.src1                 = params.src1;
        q8_params.dst                  = params.dst;
        ਜੇ (bias_tensor) {
            q8_params.bias = *bias_tensor;
        }
        kernel_ਨਤੀਜਾ = ggml_et_launch_kernel(dev_ctx, kernel_name, &q8_params, ਆਕਾਰ(q8_params), 0xFFFFFFFF);
    } ਨਹੀਂਤਾਂ {
        // Non-Q8 MM kernels don't yet support fused-add; the graph fuse check
        // already rejects non-Q8 pairs, so add_node is always nullptr here.
        kernel_ਨਤੀਜਾ = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ਆਕਾਰ(params), 0xFFFFFFFF);
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
    ਜੇ (cpu_comparison_ਸਰਗਰਮ) {
        ਜੇ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, fused_dst, &mul_mat_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for MUL_MAT operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    {
        // Calculate actual FLOPs including batch/sequence dimensions
        // dst shape: [M, N, ne2, ne3] where M=ne[1], N=ne[0]
        int64_t m   = ਗੰਢ->ne[1];
        int64_t n   = ਗੰਢ->ne[0];
        int64_t k   = ਗੰਢ->src[0]->ne[0];
        int64_t ne2 = ਗੰਢ->ne[2];
        int64_t ne3 = ਗੰਢ->ne[3];

        // Total FLOPs = (batch_size) * M * N * (2*K - 1)
        // Each MxN matrix-matrix multiply does M*N*(2*K-1) FLOPs
        // Broadcasting is handled by repeating computation, so count actual operations
        int64_t batch_ਮਾਪ  = ne2 * ne3;
        int64_t ਕੁੱਲ_flops = batch_ਮਾਪ * m * n * (2 * k - 1);

        ਅੱਖਰ kernel_variant[64];
        snprintf(kernel_variant, ਆਕਾਰ(kernel_variant), "%s_%sx%s", kernel_name, src0_type_name,
                 ggml_type_name(ਗੰਢ->src[1]->type));
        ET_PERF_ਸਿਰਾ_EXT("MUL_MAT", kernel_variant, ਗੰਢ, "flops=%" PRId64, ਕੁੱਲ_flops);
    }
    ਵਾਪਸ kernel_ਨਤੀਜਾ;
}

ਤਾਰਕਿਕ ggml_et_op_mul_mat_id(ggml_backend_et_device_context * dev_ctx, ਅਚਲ ggml_tensor * ਗੰਢ) {
    ET_PERF_ਸ਼ੁਰੂ();
    ਜੇ (!dev_ctx || !ਗੰਢ) {
        GGML_LOG_ERROR("ET: Invalid parameters for MUL_MAT_ID operation\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (!ਗੰਢ->src[0] || !ਗੰਢ->src[1] || !ਗੰਢ->src[2]) {
        GGML_LOG_ERROR("ET: MUL_MAT_ID operation missing required inputs\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ ਅੱਖਰ * kernel_name;
    ਅਚਲ ਅੱਖਰ * src0_type_name;

    // Support Q8_0/Q4_0/F16/F32 x F32 -> F32 matrix multiplication with expert selection
    ਜੇ (ਗੰਢ->type == GGML_TYPE_F32 && ਗੰਢ->src[0]->type == GGML_TYPE_Q8_0 && ਗੰਢ->src[1]->type == GGML_TYPE_F32 &&
        ਗੰਢ->src[2]->type == GGML_TYPE_I32) {
        kernel_name    = "mul_mat_id_Q8_0";
        src0_type_name = "Q8_0";

    } ਨਹੀਂਤਾਂ ਜੇ (ਗੰਢ->type == GGML_TYPE_F32 && ਗੰਢ->src[0]->type == GGML_TYPE_Q4_0 &&
               ਗੰਢ->src[1]->type == GGML_TYPE_F32 && ਗੰਢ->src[2]->type == GGML_TYPE_I32) {
        kernel_name    = "mul_mat_id_Q4_0";
        src0_type_name = "Q4_0";

    } ਨਹੀਂਤਾਂ ਜੇ (ਗੰਢ->type == GGML_TYPE_F32 && ਗੰਢ->src[0]->type == GGML_TYPE_F16 &&
               ਗੰਢ->src[1]->type == GGML_TYPE_F32 && ਗੰਢ->src[2]->type == GGML_TYPE_I32) {
        kernel_name    = "mul_mat_id_f32";
        src0_type_name = "F16";

    } ਨਹੀਂਤਾਂ ਜੇ (ਗੰਢ->type == GGML_TYPE_F32 && ਗੰਢ->src[0]->type == GGML_TYPE_F32 &&
               ਗੰਢ->src[1]->type == GGML_TYPE_F32 && ਗੰਢ->src[2]->type == GGML_TYPE_I32) {
        kernel_name    = "mul_mat_id_f32";
        src0_type_name = "F32";

    } ਨਹੀਂਤਾਂ {
        GGML_LOG_ERROR("ET: MUL_MAT_ID operation with unsupported types: dst=%s src0=%s src1=%s src2=%s\n",
                       ggml_type_name(ਗੰਢ->type), ggml_type_name(ਗੰਢ->src[0]->type),
                       ggml_type_name(ਗੰਢ->src[1]->type), ggml_type_name(ਗੰਢ->src[2]->type));
        ਵਾਪਸ ਝੂਠ;
    }

    // Pack parameters - copy full tensor structures
    ggml_et_mul_mat_id_params params;
    params.src0 = *ਗੰਢ->src[0];  // Expert weight matrices (Q8_0/F16/F32)
    params.src1 = *ਗੰਢ->src[1];  // Activation matrix (F32)
    params.src2 = *ਗੰਢ->src[2];  // Expert indices (I32)
    params.dst  = *ਗੰਢ;          // Output matrix (F32)

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ਤਾਰਕਿਕ                    cpu_comparison_ਸਰਗਰਮ = ਝੂਠ;
    ਜੇ (mul_mat_id_cpu_compare_config.enabled) {
        ਜੇ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ਗੰਢ, GGML_OP_MUL_MAT_ID)) {
            cpu_comparison_ਸਰਗਰਮ = ਸੱਚ;
        } ਨਹੀਂਤਾਂ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for MUL_MAT_ID operation\n");
        }
    }

    // Launch ET kernel
    ਤਾਰਕਿਕ kernel_ਨਤੀਜਾ = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ਆਕਾਰ(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ਜੇ (cpu_comparison_ਸਰਗਰਮ) {
        ਜੇ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ਗੰਢ, &mul_mat_id_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for MUL_MAT_ID operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    // Calculate FLOPs (approximate - similar to MUL_MAT but with expert routing overhead)
    // Each expert computation is similar to a MUL_MAT, but we only compute for selected experts
    int64_t K             = ਗੰਢ->src[0]->ne[0];
    int64_t M             = ਗੰਢ->src[0]->ne[1];
    int64_t n_expert_ਵਰਤਿਆ = ਗੰਢ->src[2]->ne[0];
    int64_t batch         = ਗੰਢ->src[2]->ne[1];

    int64_t ਕੁੱਲ_flops = batch * n_expert_ਵਰਤਿਆ * M * (2 * K - 1);

    ਅੱਖਰ kernel_variant[64];
    snprintf(kernel_variant, ਆਕਾਰ(kernel_variant), "%s_%sx%s", kernel_name, src0_type_name,
             ggml_type_name(ਗੰਢ->src[1]->type));
    ET_PERF_ਸਿਰਾ_EXT("MUL_MAT_ID", kernel_variant, ਗੰਢ, "flops=%" PRId64 "|n_expert=%lld|n_expert_used=%lld",
                    ਕੁੱਲ_flops, (ਲੰਮਾ ਲੰਮਾ) ਗੰਢ->src[0]->ne[2], (ਲੰਮਾ ਲੰਮਾ) n_expert_ਵਰਤਿਆ);

    ਵਾਪਸ kernel_ਨਤੀਜਾ;
}

ਤਾਰਕਿਕ ggml_et_op_rope(ggml_backend_et_device_context * dev_ctx, ਅਚਲ ggml_tensor * ਗੰਢ) {
    ET_PERF_ਸ਼ੁਰੂ();

    ਜੇ (!dev_ctx || !ਗੰਢ) {
        GGML_LOG_ERROR("ET: Invalid parameters for ROPE operation\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (!ਗੰਢ->src[0] || !ਗੰਢ->src[1]) {
        GGML_LOG_ERROR("ET: ROPE operation missing required inputs\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ ਅੱਖਰ * kernel_name;

    ਜੇ (ਗੰਢ->type == GGML_TYPE_F32 && ਗੰਢ->src[0]->type == GGML_TYPE_F32 && ਗੰਢ->src[1]->type == GGML_TYPE_I32) {
        kernel_name = "rope_f32";
    } ਨਹੀਂਤਾਂ {
        ਵਾਪਸ ਝੂਠ;
    }

    // Pack parameters - copy full tensor structures and op_params
    ggml_et_rope_params params;
    params.src0 = *ਗੰਢ->src[0];                       // F32 input tensor
    params.src1 = *ਗੰਢ->src[1];                       // I32 position tensor
    ਜੇ (ਗੰਢ->src[2]) {
        params.src2 = *ਗੰਢ->src[2];                   // F32 frequency factors (optional)
    } ਨਹੀਂਤਾਂ {
        memset(&params.src2, 0, ਆਕਾਰ(params.src2));  // Zero if not provided
    }
    params.dst = *ਗੰਢ;                                // F32 output tensor

    params.rope_params.n_past     = ((ਅਚਲ int32_t *) ਗੰਢ->op_params)[0];
    params.rope_params.n_dims     = ((ਅਚਲ int32_t *) ਗੰਢ->op_params)[1];
    params.rope_params.mode       = ((ਅਚਲ int32_t *) ਗੰਢ->op_params)[2];
    params.rope_params.n_ctx      = ((ਅਚਲ int32_t *) ਗੰਢ->op_params)[3];
    params.rope_params.n_ctx_orig = ((ਅਚਲ int32_t *) ਗੰਢ->op_params)[4];
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&params.rope_params.freq_ਅਧਾਰ, (ਅਚਲ int32_t *) ਗੰਢ->op_params + 5, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&params.rope_params.freq_scale, (ਅਚਲ int32_t *) ਗੰਢ->op_params + 6, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&params.rope_params.ext_ਗੁਣਕ, (ਅਚਲ int32_t *) ਗੰਢ->op_params + 7, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&params.rope_params.attn_ਗੁਣਕ, (ਅਚਲ int32_t *) ਗੰਢ->op_params + 8, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&params.rope_params.beta_fast, (ਅਚਲ int32_t *) ਗੰਢ->op_params + 9, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&params.rope_params.beta_slow, (ਅਚਲ int32_t *) ਗੰਢ->op_params + 10, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਜੇ (params.rope_params.mode & GGML_ROPE_TYPE_MROPE) {
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(params.rope_params.sections, (ਅਚਲ int32_t *) ਗੰਢ->op_params + 11, ਆਕਾਰ(int32_t) * 4);
    } ਨਹੀਂਤਾਂ {
        memset(params.rope_params.sections, 0, ਆਕਾਰ(params.rope_params.sections));
    }

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ਤਾਰਕਿਕ                    cpu_comparison_ਸਰਗਰਮ = ਝੂਠ;
    ਜੇ (rope_cpu_compare_config.enabled) {
        GGML_LOG_DEBUG("ET: Initializing CPU comparison for ROPE operation\n");
        ਜੇ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ਗੰਢ, GGML_OP_ROPE)) {
            cpu_comparison_ਸਰਗਰਮ = ਸੱਚ;
        } ਨਹੀਂਤਾਂ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for ROPE operation\n");
        }
    }

    ਤਾਰਕਿਕ kernel_ਨਤੀਜਾ = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ਆਕਾਰ(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ਜੇ (cpu_comparison_ਸਰਗਰਮ) {
        ਜੇ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ਗੰਢ, &rope_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for ROPE operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_ਸਿਰਾ_EXT("ROPE", kernel_name, ਗੰਢ, "mode=0x%x|n_dims=%d|freq_base=%.2f|freq_scale=%.2f",
                    params.rope_params.mode, params.rope_params.n_dims, (ਦੁੱਗਣਾ) params.rope_params.freq_ਅਧਾਰ,
                    (ਦੁੱਗਣਾ) params.rope_params.freq_scale);
    ਵਾਪਸ kernel_ਨਤੀਜਾ;
}

ਤਾਰਕਿਕ ggml_et_op_rms_norm(ggml_backend_et_device_context * dev_ctx, ਅਚਲ ggml_tensor * ਗੰਢ) {
    ET_PERF_ਸ਼ੁਰੂ();

    ਜੇ (!dev_ctx || !ਗੰਢ) {
        GGML_LOG_ERROR("ET: Invalid parameters for RMS_NORM operation\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (!ਗੰਢ->src[0]) {
        GGML_LOG_ERROR("ET: RMS_NORM operation missing required input\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ ਅੱਖਰ * kernel_name;

    ਜੇ (ਗੰਢ->type == GGML_TYPE_F32 && ਗੰਢ->src[0]->type == GGML_TYPE_F32) {
        kernel_name = "rms_norm_f32";

    } ਨਹੀਂਤਾਂ {
        GGML_LOG_ERROR("ET: RMS_NORM operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(ਗੰਢ->type),
                       ggml_type_name(ਗੰਢ->src[0]->type));
        ਵਾਪਸ ਝੂਠ;
    }

    ਦਸ਼ਮਲਵ eps;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&eps, ਗੰਢ->op_params, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ggml_et_rms_norm_params params;
    params.src0 = *ਗੰਢ->src[0];  // F32 input tensor
    params.dst  = *ਗੰਢ;          // F32 output tensor
    params.eps  = eps;            // Epsilon parameter for numerical stability

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ਤਾਰਕਿਕ                    cpu_comparison_ਸਰਗਰਮ = ਝੂਠ;
    ਜੇ (rms_norm_cpu_compare_config.enabled) {
        ਜੇ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ਗੰਢ, GGML_OP_RMS_NORM)) {
            cpu_comparison_ਸਰਗਰਮ = ਸੱਚ;
        } ਨਹੀਂਤਾਂ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for RMS_NORM operation\n");
        }
    }

    ਤਾਰਕਿਕ kernel_ਨਤੀਜਾ = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ਆਕਾਰ(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ਜੇ (cpu_comparison_ਸਰਗਰਮ) {
        ਜੇ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ਗੰਢ, &rms_norm_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for RMS_NORM operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_ਸਿਰਾ_EXT("RMS_NORM", kernel_name, ਗੰਢ, "eps=%.6f", (ਦੁੱਗਣਾ) eps);
    ਵਾਪਸ kernel_ਨਤੀਜਾ;
}

ਤਾਰਕਿਕ ggml_et_op_norm(ggml_backend_et_device_context * dev_ctx, ਅਚਲ ggml_tensor * ਗੰਢ) {
    ET_PERF_ਸ਼ੁਰੂ();

    ਜੇ (!dev_ctx || !ਗੰਢ) {
        GGML_LOG_ERROR("ET: Invalid parameters for NORM operation\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (!ਗੰਢ->src[0]) {
        GGML_LOG_ERROR("ET: NORM operation missing required input\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ ਅੱਖਰ * kernel_name;

    ਜੇ (ਗੰਢ->type == GGML_TYPE_F32 && ਗੰਢ->src[0]->type == GGML_TYPE_F32) {
        kernel_name = "norm_f32";

    } ਨਹੀਂਤਾਂ {
        GGML_LOG_ERROR("ET: NORM operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(ਗੰਢ->type),
                       ggml_type_name(ਗੰਢ->src[0]->type));
        ਵਾਪਸ ਝੂਠ;
    }

    ਦਸ਼ਮਲਵ eps;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&eps, ਗੰਢ->op_params, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ggml_et_norm_params params;
    params.src0 = *ਗੰਢ->src[0];  // F32 input tensor
    params.dst  = *ਗੰਢ;          // F32 output tensor
    params.eps  = eps;            // Epsilon parameter for numerical stability

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ਤਾਰਕਿਕ                    cpu_comparison_ਸਰਗਰਮ = ਝੂਠ;
    ਜੇ (norm_cpu_compare_config.enabled) {
        ਜੇ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ਗੰਢ, GGML_OP_NORM)) {
            cpu_comparison_ਸਰਗਰਮ = ਸੱਚ;
        } ਨਹੀਂਤਾਂ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for NORM operation\n");
        }
    }

    ਤਾਰਕਿਕ kernel_ਨਤੀਜਾ = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ਆਕਾਰ(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ਜੇ (cpu_comparison_ਸਰਗਰਮ) {
        ਜੇ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ਗੰਢ, &norm_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for NORM operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_ਸਿਰਾ_EXT("NORM", kernel_name, ਗੰਢ, "eps=%.6f", (ਦੁੱਗਣਾ) eps);
    ਵਾਪਸ kernel_ਨਤੀਜਾ;
}

ਤਾਰਕਿਕ ggml_et_op_l2_norm(ggml_backend_et_device_context * dev_ctx, ਅਚਲ ggml_tensor * ਗੰਢ) {
    ET_PERF_ਸ਼ੁਰੂ();

    ਜੇ (!dev_ctx || !ਗੰਢ) {
        GGML_LOG_ERROR("ET: Invalid parameters for L2_NORM operation\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (!ਗੰਢ->src[0]) {
        GGML_LOG_ERROR("ET: L2_NORM operation missing required input\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ ਅੱਖਰ * kernel_name;

    ਜੇ (ਗੰਢ->type == GGML_TYPE_F32 && ਗੰਢ->src[0]->type == GGML_TYPE_F32) {
        kernel_name = "l2_norm_f32";

    } ਨਹੀਂਤਾਂ {
        GGML_LOG_ERROR("ET: L2_NORM operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(ਗੰਢ->type),
                       ggml_type_name(ਗੰਢ->src[0]->type));
        ਵਾਪਸ ਝੂਠ;
    }

    ਦਸ਼ਮਲਵ eps;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&eps, ਗੰਢ->op_params, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ggml_et_l2_norm_params params;
    params.src0 = *ਗੰਢ->src[0];  // F32 input tensor
    params.dst  = *ਗੰਢ;          // F32 output tensor
    params.eps  = eps;            // Epsilon parameter for numerical stability

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ਤਾਰਕਿਕ                    cpu_comparison_ਸਰਗਰਮ = ਝੂਠ;
    ਜੇ (l2_norm_cpu_compare_config.enabled) {
        ਜੇ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ਗੰਢ, GGML_OP_L2_NORM)) {
            cpu_comparison_ਸਰਗਰਮ = ਸੱਚ;
        } ਨਹੀਂਤਾਂ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for L2_NORM operation\n");
        }
    }

    ਤਾਰਕਿਕ kernel_ਨਤੀਜਾ = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ਆਕਾਰ(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ਜੇ (cpu_comparison_ਸਰਗਰਮ) {
        ਜੇ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ਗੰਢ, &l2_norm_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for L2_NORM operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_ਸਿਰਾ_EXT("L2_NORM", kernel_name, ਗੰਢ, "eps=%.6f", (ਦੁੱਗਣਾ) eps);
    ਵਾਪਸ kernel_ਨਤੀਜਾ;
}

ਤਾਰਕਿਕ ggml_et_op_ਸਮੂਹ_norm(ggml_backend_et_device_context * dev_ctx, ਅਚਲ ggml_tensor * ਗੰਢ) {
    ET_PERF_ਸ਼ੁਰੂ();

    ਜੇ (!dev_ctx || !ਗੰਢ) {
        GGML_LOG_ERROR("ET: Invalid parameters for GROUP_NORM operation\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (!ਗੰਢ->src[0]) {
        GGML_LOG_ERROR("ET: GROUP_NORM operation missing required input\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (ਗੰਢ->type != GGML_TYPE_F32 || ਗੰਢ->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: GROUP_NORM operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(ਗੰਢ->type),
                       ggml_type_name(ਗੰਢ->src[0]->type));
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ int32_t n_groups = ggml_get_op_params_i32(ਗੰਢ, 0);
    ਦਸ਼ਮਲਵ         eps;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&eps, (ਅਚਲ ਦਸ਼ਮਲਵ *) ਗੰਢ->op_params + 1, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ggml_et_ਸਮੂਹ_norm_params params;
    params.src0     = *ਗੰਢ->src[0];
    params.dst      = *ਗੰਢ;
    params.n_groups = n_groups;
    params.eps      = eps;

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ਤਾਰਕਿਕ                    cpu_comparison_ਸਰਗਰਮ = ਝੂਠ;
    ਜੇ (ਸਮੂਹ_norm_cpu_compare_config.enabled) {
        ਜੇ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ਗੰਢ, GGML_OP_ਸਮੂਹ_NORM)) {
            cpu_comparison_ਸਰਗਰਮ = ਸੱਚ;
        } ਨਹੀਂਤਾਂ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for GROUP_NORM operation\n");
        }
    }

    ਤਾਰਕਿਕ kernel_ਨਤੀਜਾ = ggml_et_launch_kernel(dev_ctx, "group_norm_f32", &params, ਆਕਾਰ(params), 0xFFFFFFFF);

    ਜੇ (cpu_comparison_ਸਰਗਰਮ) {
        ਜੇ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ਗੰਢ, &ਸਮੂਹ_norm_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for GROUP_NORM operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_ਸਿਰਾ_EXT("GROUP_NORM", "group_norm_f32", ਗੰਢ, "eps=%.6f|n_groups=%d", (ਦੁੱਗਣਾ) eps, n_groups);
    ਵਾਪਸ kernel_ਨਤੀਜਾ;
}

ਤਾਰਕਿਕ ggml_et_op_im2col(ggml_backend_et_device_context * dev_ctx, ਅਚਲ ggml_tensor * ਗੰਢ) {
    ET_PERF_ਸ਼ੁਰੂ();

    ਜੇ (!dev_ctx || !ਗੰਢ) {
        GGML_LOG_ERROR("ET: Invalid parameters for IM2COL operation\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (!ਗੰਢ->src[0] || !ਗੰਢ->src[1]) {
        GGML_LOG_ERROR("ET: IM2COL operation missing required inputs\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ ਤਾਰਕਿਕ supported_types =
        (ਗੰਢ->type == GGML_TYPE_F32 && ਗੰਢ->src[1]->type == GGML_TYPE_F32) ||
        (ਗੰਢ->type == GGML_TYPE_F16 && (ਗੰਢ->src[1]->type == GGML_TYPE_F16 || ਗੰਢ->src[1]->type == GGML_TYPE_F32));

    ਜੇ (!supported_types) {
        GGML_LOG_ERROR("ET: IM2COL operation with unsupported types: dst=%s src1=%s\n", ggml_type_name(ਗੰਢ->type),
                       ggml_type_name(ਗੰਢ->src[1]->type));
        ਵਾਪਸ ਝੂਠ;
    }

    ggml_et_im2col_params params;
    params.src0 = *ਗੰਢ->src[0];
    params.src1 = *ਗੰਢ->src[1];
    params.dst  = *ਗੰਢ;

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ਤਾਰਕਿਕ                    cpu_comparison_ਸਰਗਰਮ = ਝੂਠ;
    ਜੇ (im2col_cpu_compare_config.enabled) {
        ਜੇ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ਗੰਢ, GGML_OP_IM2_ਖਾਨਾ)) {
            cpu_comparison_ਸਰਗਰਮ = ਸੱਚ;
        } ਨਹੀਂਤਾਂ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for IM2COL operation\n");
        }
    }

    ਤਾਰਕਿਕ kernel_ਨਤੀਜਾ = ggml_et_launch_kernel(dev_ctx, "im2col", &params, ਆਕਾਰ(params), 0xFFFFFFFF);

    ਜੇ (cpu_comparison_ਸਰਗਰਮ) {
        ਜੇ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ਗੰਢ, &im2col_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for IM2COL operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_ਸਿਰਾ("IM2COL", "im2col", ਗੰਢ);
    ਵਾਪਸ kernel_ਨਤੀਜਾ;
}

ਤਾਰਕਿਕ ggml_et_op_conv_2d(ggml_backend_et_device_context * dev_ctx, ਅਚਲ ggml_tensor * ਗੰਢ) {
    ET_PERF_ਸ਼ੁਰੂ();

    ਜੇ (!dev_ctx || !ਗੰਢ) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (!ਗੰਢ->src[0] || !ਗੰਢ->src[1]) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (!ਗੰਢ->ਅੰਕੜੇ || !ਗੰਢ->src[0]->ਅੰਕੜੇ || !ਗੰਢ->src[1]->ਅੰਕੜੇ) {
        ਵਾਪਸ ਝੂਠ;
    }

    // Kernel constraints (mirror supports_op; recheck here as a guard).
    ਅਚਲ ggml_tensor * flt = ਗੰਢ->src[0];  // [Kw, Kh, Cin, Cout]
    ਅਚਲ ggml_tensor * ਵਿੱਚ  = ਗੰਢ->src[1];  // [W,  H,  Cin, N]
    ਜੇ (ਗੰਢ->type != GGML_TYPE_F32 || flt->type != GGML_TYPE_F32 || ਵਿੱਚ->type != GGML_TYPE_F32) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ int32_t s0 = ggml_get_op_params_i32(ਗੰਢ, 0);
    ਅਚਲ int32_t s1 = ggml_get_op_params_i32(ਗੰਢ, 1);
    ਅਚਲ int32_t p0 = ggml_get_op_params_i32(ਗੰਢ, 2);
    ਅਚਲ int32_t p1 = ggml_get_op_params_i32(ਗੰਢ, 3);
    ਅਚਲ int32_t d0 = ggml_get_op_params_i32(ਗੰਢ, 4);
    ਅਚਲ int32_t d1 = ggml_get_op_params_i32(ਗੰਢ, 5);

    ਜੇ (s0 < 1 || s1 < 1) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (d0 != 1 || d1 != 1) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (flt->ne[2] % 16 != 0 || flt->ne[3] % 16 != 0) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (ਵਿੱਚ->ne[3] != 1) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (ਗੰਢ->ne[0] <= 0) {
        ਵਾਪਸ ਝੂਠ;  // OW > 0 (any width OK; staging path handles non-16)
    }
    (ਖਾਲੀ) p0;
    (ਖਾਲੀ) p1;

    ggml_et_ਦੁਆਧਾਰੀ_params params;
    params.src0 = *ਗੰਢ->src[0];
    params.src1 = *ਗੰਢ->src[1];
    params.dst  = *ਗੰਢ;

    ਤਾਰਕਿਕ kernel_ਨਤੀਜਾ = ggml_et_launch_kernel(dev_ctx, "conv_2d_f32_me", &params, ਆਕਾਰ(params), 0xFFFFFFFFu);

    ET_PERF_ਸਿਰਾ("CONV_2D", "conv_2d_f32_me", ਗੰਢ);
    ਵਾਪਸ kernel_ਨਤੀਜਾ;
}

ਤਾਰਕਿਕ ggml_et_op_softmax(ggml_backend_et_device_context * dev_ctx, ਅਚਲ ggml_tensor * ਗੰਢ) {
    ET_PERF_ਸ਼ੁਰੂ();

    ਜੇ (!dev_ctx || !ਗੰਢ) {
        GGML_LOG_ERROR("ET: Invalid parameters for SOFTMAX operation\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (!ਗੰਢ->src[0]) {
        GGML_LOG_ERROR("ET: SOFTMAX operation missing required input\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ ਅੱਖਰ * kernel_name;

    ਜੇ (ਗੰਢ->type == GGML_TYPE_F32 && ਗੰਢ->src[0]->type == GGML_TYPE_F32) {
        kernel_name = "softmax_f32";

    } ਨਹੀਂਤਾਂ {
        GGML_LOG_ERROR("ET: SOFTMAX operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(ਗੰਢ->type),
                       ggml_type_name(ਗੰਢ->src[0]->type));
        ਵਾਪਸ ਝੂਠ;
    }

    // Validate contiguity requirements
    ਜੇ (!ggml_is_contiguous(ਗੰਢ)) {
        GGML_LOG_ERROR("ET: SOFTMAX operation requires contiguous destination tensor\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (!ggml_is_contiguous(ਗੰਢ->src[0])) {
        GGML_LOG_ERROR("ET: SOFTMAX operation requires contiguous source tensor\n");
        ਵਾਪਸ ਝੂਠ;
    }

    // Check optional mask tensor
    ਜੇ (ਗੰਢ->src[1]) {
        ਜੇ (ਗੰਢ->src[1]->type != GGML_TYPE_F32) {
            GGML_LOG_ERROR("ET: SOFTMAX operation with unsupported mask type: %s (F32 required)\n",
                           ggml_type_name(ਗੰਢ->src[1]->type));
            ਵਾਪਸ ਝੂਠ;
        }
        ਜੇ (!ggml_is_contiguous(ਗੰਢ->src[1])) {
            GGML_LOG_ERROR("ET: SOFTMAX operation requires contiguous mask tensor\n");
            ਵਾਪਸ ਝੂਠ;
        }
    }

    // Check optional sinks tensor
    ਜੇ (ਗੰਢ->src[2]) {
        ਜੇ (ਗੰਢ->src[2]->type != GGML_TYPE_F32) {
            GGML_LOG_ERROR("ET: SOFTMAX operation with unsupported sinks type: %s (F32 required)\n",
                           ggml_type_name(ਗੰਢ->src[2]->type));
            ਵਾਪਸ ਝੂਠ;
        }
        ਜੇ (!ggml_is_contiguous(ਗੰਢ->src[2])) {
            GGML_LOG_ERROR("ET: SOFTMAX operation requires contiguous sinks tensor\n");
            ਵਾਪਸ ਝੂਠ;
        }
    }

    // Extract scale and max_bias from op_params
    ਦਸ਼ਮਲਵ scale    = 1.0f;
    ਦਸ਼ਮਲਵ max_bias = 0.0f;
    ਜੇ (ਗੰਢ->op_params) {
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(&scale, (ਅਚਲ ਦਸ਼ਮਲਵ *) ਗੰਢ->op_params + 0, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(&max_bias, (ਅਚਲ ਦਸ਼ਮਲਵ *) ਗੰਢ->op_params + 1, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    }

    ggml_et_softmax_params params;
    params.src0 = *ਗੰਢ->src[0];                       // F32 input tensor
    ਜੇ (ਗੰਢ->src[1]) {
        params.src1 = *ਗੰਢ->src[1];                   // F32 mask tensor
    } ਨਹੀਂਤਾਂ {
        memset(&params.src1, 0, ਆਕਾਰ(params.src1));  // Zero if no mask
    }
    ਜੇ (ਗੰਢ->src[2]) {
        params.src2 = *ਗੰਢ->src[2];                   // F32 sinks tensor
    } ਨਹੀਂਤਾਂ {
        memset(&params.src2, 0, ਆਕਾਰ(params.src2));  // Zero if no sinks
    }
    params.dst      = *ਗੰਢ;                           // F32 output tensor
    params.scale    = scale;                           // Scale factor
    params.max_bias = max_bias;                        // ALiBi bias

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ਤਾਰਕਿਕ                    cpu_comparison_ਸਰਗਰਮ = ਝੂਠ;
    ਜੇ (softmax_cpu_compare_config.enabled) {
        ਜੇ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ਗੰਢ, GGML_OP_SOFT_MAX)) {
            cpu_comparison_ਸਰਗਰਮ = ਸੱਚ;
        } ਨਹੀਂਤਾਂ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for SOFTMAX operation\n");
        }
    }

    ਤਾਰਕਿਕ kernel_ਨਤੀਜਾ = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ਆਕਾਰ(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ਜੇ (cpu_comparison_ਸਰਗਰਮ) {
        ਜੇ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ਗੰਢ, &softmax_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for SOFTMAX operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_ਸਿਰਾ_EXT("SOFTMAX", kernel_name, ਗੰਢ, "scale=%.6f|max_bias=%.6f|has_mask=%s", (ਦੁੱਗਣਾ) scale,
                    (ਦੁੱਗਣਾ) max_bias, ਗੰਢ->src[1] ? "yes" : "no");
    ਵਾਪਸ kernel_ਨਤੀਜਾ;
}

ਤਾਰਕਿਕ ggml_et_op_flash_attn_ext(ggml_backend_et_device_context * dev_ctx, ਅਚਲ ggml_tensor * ਗੰਢ) {
    ET_PERF_ਸ਼ੁਰੂ();

    ਜੇ (!dev_ctx || !ਗੰਢ) {
        GGML_LOG_ERROR("ET: Invalid parameters for FLASH_ATTN_EXT operation\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (!ਗੰਢ->src[0] || !ਗੰਢ->src[1] || !ਗੰਢ->src[2]) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT operation missing required inputs\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (ਗੰਢ->type != GGML_TYPE_F32 || ਗੰਢ->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT requires F32 Q and dst, got dst=%s q=%s\n", ggml_type_name(ਗੰਢ->type),
                       ggml_type_name(ਗੰਢ->src[0]->type));
        ਵਾਪਸ ਝੂਠ;
    }

    // K and V can be F16 or F32
    ਜੇ ((ਗੰਢ->src[1]->type != GGML_TYPE_F32 && ਗੰਢ->src[1]->type != GGML_TYPE_F16) ||
        (ਗੰਢ->src[2]->type != GGML_TYPE_F32 && ਗੰਢ->src[2]->type != GGML_TYPE_F16)) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT K/V must be F16 or F32, got k=%s v=%s\n", ggml_type_name(ਗੰਢ->src[1]->type),
                       ggml_type_name(ਗੰਢ->src[2]->type));
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (ਗੰਢ->src[4] != ਖਾਲੀ_ਸੂਚਕ) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT baseline kernel does not support sinks\n");
        ਵਾਪਸ ਝੂਠ;
    }

    // Mask is optional; if present must be F16 or F32
    ਜੇ (ਗੰਢ->src[3] != ਖਾਲੀ_ਸੂਚਕ && ਗੰਢ->src[3]->type != GGML_TYPE_F32 && ਗੰਢ->src[3]->type != GGML_TYPE_F16) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT mask must be F16 or F32, got %s\n", ggml_type_name(ਗੰਢ->src[3]->type));
        ਵਾਪਸ ਝੂਠ;
    }

    // Q and dst must be row-contiguous F32
    ਜੇ (!ggml_is_contiguous_ਪੰਕਤੀਆਂ(ਗੰਢ) || !ggml_is_contiguous_ਪੰਕਤੀਆਂ(ਗੰਢ->src[0])) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT requires row-contiguous Q and dst\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (ਗੰਢ->nb[0] != ਆਕਾਰ(ਦਸ਼ਮਲਵ) || ਗੰਢ->src[0]->nb[0] != ਆਕਾਰ(ਦਸ਼ਮਲਵ)) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT requires contiguous F32 rows for Q and dst\n");
        ਵਾਪਸ ਝੂਠ;
    }

    // K/V must have element-sized stride in dim 0
    ਅਚਲ size_t k_elem = ਗੰਢ->src[1]->type == GGML_TYPE_F16 ? 2 : 4;
    ਅਚਲ size_t v_elem = ਗੰਢ->src[2]->type == GGML_TYPE_F16 ? 2 : 4;
    ਜੇ (ਗੰਢ->src[1]->nb[0] != k_elem || ਗੰਢ->src[2]->nb[0] != v_elem) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT K/V must have element-sized stride in dim 0\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਦਸ਼ਮਲਵ scale         = 1.0f;
    ਦਸ਼ਮਲਵ max_bias      = 0.0f;
    ਦਸ਼ਮਲਵ logit_softcap = 0.0f;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&scale, (ਅਚਲ ਦਸ਼ਮਲਵ *) ਗੰਢ->op_params + 0, ਆਕਾਰ(scale));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&max_bias, (ਅਚਲ ਦਸ਼ਮਲਵ *) ਗੰਢ->op_params + 1, ਆਕਾਰ(max_bias));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&logit_softcap, (ਅਚਲ ਦਸ਼ਮਲਵ *) ਗੰਢ->op_params + 2, ਆਕਾਰ(logit_softcap));

    ਜੇ (max_bias != 0.0f || logit_softcap != 0.0f) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT baseline kernel does not support max_bias or logit_softcap\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ ggml_prec prec = ggml_flash_attn_ext_get_prec(ਗੰਢ);
    ਜੇ (prec != GGML_PREC_F32 && prec != GGML_PREC_DEFAULT) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT baseline kernel only supports F32 precision\n");
        ਵਾਪਸ ਝੂਠ;
    }

    // dk must match between Q and K; dv must match between V and dst
    ਜੇ (ਗੰਢ->src[0]->ne[0] != ਗੰਢ->src[1]->ne[0]) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT dk mismatch: Q=%lld K=%lld\n", (ਲੰਮਾ ਲੰਮਾ) ਗੰਢ->src[0]->ne[0],
                       (ਲੰਮਾ ਲੰਮਾ) ਗੰਢ->src[1]->ne[0]);
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (ਗੰਢ->src[2]->ne[0] != ਗੰਢ->ne[0]) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT dv mismatch: V=%lld dst=%lld\n", (ਲੰਮਾ ਲੰਮਾ) ਗੰਢ->src[2]->ne[0],
                       (ਲੰਮਾ ਲੰਮਾ) ਗੰਢ->ne[0]);
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (ਗੰਢ->src[2]->ne[0] > 512) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT dv=%lld exceeds maximum 512\n", (ਲੰਮਾ ਲੰਮਾ) ਗੰਢ->src[2]->ne[0]);
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (ਗੰਢ->src[0]->ne[0] > 512) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT dk=%lld exceeds maximum 512\n", (ਲੰਮਾ ਲੰਮਾ) ਗੰਢ->src[0]->ne[0]);
        ਵਾਪਸ ਝੂਠ;
    }

    // GQA: n_head_q must be a multiple of n_head_kv
    ਅਚਲ int64_t nhq = ਗੰਢ->src[0]->ne[2];
    ਅਚਲ int64_t nhk = ਗੰਢ->src[1]->ne[2];
    ਜੇ (nhq % nhk != 0) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT n_head_q (%lld) not divisible by n_head_kv (%lld)\n", (ਲੰਮਾ ਲੰਮਾ) nhq,
                       (ਲੰਮਾ ਲੰਮਾ) nhk);
        ਵਾਪਸ ਝੂਠ;
    }

    // K and V must have matching sequence length, heads, and batch dims
    ਜੇ (ਗੰਢ->src[1]->ne[1] != ਗੰਢ->src[2]->ne[1] || ਗੰਢ->src[1]->ne[2] != ਗੰਢ->src[2]->ne[2] ||
        ਗੰਢ->src[1]->ne[3] != ਗੰਢ->src[2]->ne[3]) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT K/V shape mismatch\n");
        ਵਾਪਸ ਝੂਠ;
    }

    // dst layout checks: [dv, nhq, nq, no]
    ਜੇ (ਗੰਢ->src[0]->ne[1] != ਗੰਢ->ne[2] || ਗੰਢ->src[0]->ne[2] != ਗੰਢ->ne[1] ||
        ਗੰਢ->src[0]->ne[3] != ਗੰਢ->ne[3]) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT dst shape mismatch\n");
        ਵਾਪਸ ਝੂਠ;
    }

    // Batch dims: Q batch must match K batch
    ਜੇ (ਗੰਢ->src[0]->ne[3] != ਗੰਢ->src[1]->ne[3]) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT batch dimension mismatch\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ggml_et_flash_attn_ext_params params;
    memset(&params, 0, ਆਕਾਰ(params));
    params.src0 = *ਗੰਢ->src[0];
    params.src1 = *ਗੰਢ->src[1];
    params.src2 = *ਗੰਢ->src[2];
    ਜੇ (ਗੰਢ->src[3] != ਖਾਲੀ_ਸੂਚਕ) {
        params.ਨਕਾਬ     = *ਗੰਢ->src[3];
        params.has_ਨਕਾਬ = 1;
    }
    params.dst   = *ਗੰਢ;
    params.scale = scale;

    // Use matrix engine kernel when K/V are F16 and dk is a multiple of 32
    ਅਚਲ ਅੱਖਰ * kernel_name;
    ਜੇ (ਗੰਢ->src[1]->type == GGML_TYPE_F16 && ਗੰਢ->src[2]->type == GGML_TYPE_F16 && (ਗੰਢ->src[0]->ne[0] % 32) == 0) {
        kernel_name = "flash_attn_ext_f16_me";
    } ਨਹੀਂਤਾਂ {
        kernel_name = "flash_attn_ext_f32";
    }

    ਅਚਲ ਤਾਰਕਿਕ kernel_ਨਤੀਜਾ = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ਆਕਾਰ(params), 0xFFFFFFFF);

    ET_PERF_ਸਿਰਾ_EXT("FLASH_ATTN_EXT", kernel_name, ਗੰਢ, "scale=%.6f", (ਦੁੱਗਣਾ) scale);
    ਵਾਪਸ kernel_ਨਤੀਜਾ;
}

ਤਾਰਕਿਕ ggml_et_op_get_ਪੰਕਤੀਆਂ(ggml_backend_et_device_context * dev_ctx, ਅਚਲ ggml_tensor * ਗੰਢ) {
    ET_PERF_ਸ਼ੁਰੂ();

    ਜੇ (!dev_ctx || !ਗੰਢ) {
        GGML_LOG_ERROR("ET: Invalid parameters for GET_ROWS operation\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (!ਗੰਢ->src[0] || !ਗੰਢ->src[1]) {
        GGML_LOG_ERROR("ET: GET_ROWS operation missing required inputs\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ ਅੱਖਰ * kernel_name;

    ਜੇ (ਗੰਢ->type == GGML_TYPE_F32 && ਗੰਢ->src[1]->type == GGML_TYPE_I32 &&
        (ਗੰਢ->src[0]->type == GGML_TYPE_F32 || ਗੰਢ->src[0]->type == GGML_TYPE_F16 ||
         ਗੰਢ->src[0]->type == GGML_TYPE_Q4_0 || ਗੰਢ->src[0]->type == GGML_TYPE_Q8_0 ||
         ਗੰਢ->src[0]->type == GGML_TYPE_Q4_K)) {
        kernel_name = "get_rows_f32";

    } ਨਹੀਂਤਾਂ {
        GGML_LOG_ERROR("ET: GET_ROWS operation with unsupported types: dst=%s src0=%s src1=%s\n",
                       ggml_type_name(ਗੰਢ->type), ggml_type_name(ਗੰਢ->src[0]->type),
                       ggml_type_name(ਗੰਢ->src[1]->type));
        ਵਾਪਸ ਝੂਠ;
    }

    // Validate contiguity requirements
    ਜੇ (!ggml_is_contiguous(ਗੰਢ)) {
        GGML_LOG_ERROR("ET: GET_ROWS operation requires contiguous destination tensor\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (!ggml_is_contiguous(ਗੰਢ->src[0])) {
        GGML_LOG_ERROR("ET: GET_ROWS operation requires contiguous data tensor\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (!ggml_is_contiguous(ਗੰਢ->src[1])) {
        GGML_LOG_ERROR("ET: GET_ROWS operation requires contiguous indices tensor\n");
        ਵਾਪਸ ਝੂਠ;
    }

    // Validate dimension constraints from ggml implementation
    ਜੇ (ਗੰਢ->src[0]->ne[2] != ਗੰਢ->src[1]->ne[1] || ਗੰਢ->src[1]->ne[3] != 1) {
        GGML_LOG_ERROR(
            "ET: GET_ROWS operation dimension constraint failed: src0.ne[2]=%lld != src1.ne[1]=%lld or src1.ne[3]=%lld "
            "!= 1\n",
            (ਲੰਮਾ ਲੰਮਾ) ਗੰਢ->src[0]->ne[2], (ਲੰਮਾ ਲੰਮਾ) ਗੰਢ->src[1]->ne[1], (ਲੰਮਾ ਲੰਮਾ) ਗੰਢ->src[1]->ne[3]);
        ਵਾਪਸ ਝੂਠ;
    }

    ggml_et_get_ਪੰਕਤੀਆਂ_params params;
    params.src0 = *ਗੰਢ->src[0];  // Data tensor (F32 or Q8_0)
    params.src1 = *ਗੰਢ->src[1];  // Indices tensor (I32)
    params.dst  = *ਗੰਢ;          // Output tensor (F32)

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ਤਾਰਕਿਕ                    cpu_comparison_ਸਰਗਰਮ = ਝੂਠ;
    ਜੇ (get_ਪੰਕਤੀਆਂ_cpu_compare_config.enabled) {
        ਜੇ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ਗੰਢ, GGML_OP_GET_ਪੰਕਤੀਆਂ)) {
            cpu_comparison_ਸਰਗਰਮ = ਸੱਚ;
        } ਨਹੀਂਤਾਂ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for GET_ROWS operation\n");
        }
    }

    ਤਾਰਕਿਕ kernel_ਨਤੀਜਾ = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ਆਕਾਰ(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ਜੇ (cpu_comparison_ਸਰਗਰਮ) {
        ਜੇ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ਗੰਢ, &get_ਪੰਕਤੀਆਂ_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for GET_ROWS operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_ਸਿਰਾ("GET_ROWS", kernel_name, ਗੰਢ);
    ਵਾਪਸ kernel_ਨਤੀਜਾ;
}

ਤਾਰਕਿਕ ggml_et_op_cont(ggml_backend_et_device_context * dev_ctx, ਅਚਲ ggml_tensor * ਗੰਢ) {
    ET_PERF_ਸ਼ੁਰੂ();

    // Validate source tensor exists
    ਜੇ (!ਗੰਢ->src[0]) {
        GGML_LOG_ERROR("ET: CONT operation missing source tensor\n");
        ਵਾਪਸ ਝੂਠ;
    }

    // Validate types match (input and output must be same type)
    ਜੇ (ਗੰਢ->type != ਗੰਢ->src[0]->type) {
        GGML_LOG_ERROR("ET: CONT operation type mismatch: src=%s dst=%s\n", ggml_type_name(ਗੰਢ->src[0]->type),
                       ggml_type_name(ਗੰਢ->type));
        ਵਾਪਸ ਝੂਠ;
    }

    // Validate supported types
    ਜੇ (ਗੰਢ->type != GGML_TYPE_F32 && ਗੰਢ->type != GGML_TYPE_F16) {
        GGML_LOG_ERROR("ET: CONT operation unsupported type: %s (only F32 and F16 supported)\n",
                       ggml_type_name(ਗੰਢ->type));
        ਵਾਪਸ ਝੂਠ;
    }

    // Validate contiguity - output must be contiguous, input can be non-contiguous
    ਜੇ (!ggml_is_contiguous(ਗੰਢ)) {
        GGML_LOG_ERROR("ET: CONT operation requires contiguous output tensor\n");
        ਵਾਪਸ ਝੂਠ;
    }

    // Select kernel based on type
    ਅਚਲ ਅੱਖਰ * kernel_name;
    ਜੇ (ਗੰਢ->type == GGML_TYPE_F32) {
        kernel_name = "cont_f32";
    } ਨਹੀਂਤਾਂ ਜੇ (ਗੰਢ->type == GGML_TYPE_F16) {
        kernel_name = "cont_f16";
    } ਨਹੀਂਤਾਂ {
        GGML_LOG_ERROR("ET: CONT operation with unsupported type: %s\n", ggml_type_name(ਗੰਢ->type));
        ਵਾਪਸ ਝੂਠ;
    }

    ggml_et_cont_params params;
    params.src0 = *ਗੰਢ->src[0];  // Input tensor (potentially non-contiguous)
    params.dst  = *ਗੰਢ;          // Output tensor (contiguous)

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ਤਾਰਕਿਕ                    cpu_comparison_ਸਰਗਰਮ = ਝੂਠ;
    ਜੇ (cont_cpu_compare_config.enabled) {
        ਜੇ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ਗੰਢ, GGML_OP_CONT)) {
            cpu_comparison_ਸਰਗਰਮ = ਸੱਚ;
        } ਨਹੀਂਤਾਂ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for CONT operation\n");
        }
    }

    ਤਾਰਕਿਕ kernel_ਨਤੀਜਾ = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ਆਕਾਰ(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ਜੇ (cpu_comparison_ਸਰਗਰਮ) {
        ਜੇ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ਗੰਢ, &cont_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for CONT operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_ਸਿਰਾ("CONT", kernel_name, ਗੰਢ);
    ਵਾਪਸ kernel_ਨਤੀਜਾ;
}

ਤਾਰਕਿਕ ggml_et_op_cumsum(ggml_backend_et_device_context * dev_ctx, ਅਚਲ ggml_tensor * ਗੰਢ) {
    ET_PERF_ਸ਼ੁਰੂ();

    ਜੇ (!dev_ctx || !ਗੰਢ || !ਗੰਢ->src[0]) {
        GGML_LOG_ERROR("ET: Invalid parameters for CUMSUM operation\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (ਗੰਢ->type != GGML_TYPE_F32 || ਗੰਢ->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: CUMSUM operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(ਗੰਢ->type),
                       ggml_type_name(ਗੰਢ->src[0]->type));
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ ਅੱਖਰ * kernel_name = "cumsum_f32";

    ggml_et_cumsum_params params;
    params.src0 = *ਗੰਢ->src[0];
    params.dst  = *ਗੰਢ;

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ਤਾਰਕਿਕ                    cpu_comparison_ਸਰਗਰਮ = ਝੂਠ;
    ਜੇ (cumsum_cpu_compare_config.enabled) {
        ਜੇ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ਗੰਢ, GGML_OP_CUMSUM)) {
            cpu_comparison_ਸਰਗਰਮ = ਸੱਚ;
        } ਨਹੀਂਤਾਂ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for CUMSUM operation\n");
        }
    }

    ਤਾਰਕਿਕ kernel_ਨਤੀਜਾ = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ਆਕਾਰ(params), 0xFFFFFFFF);

    ਜੇ (cpu_comparison_ਸਰਗਰਮ) {
        ਜੇ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ਗੰਢ, &cumsum_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for CUMSUM operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_ਸਿਰਾ("CUMSUM", kernel_name, ਗੰਢ);
    ਵਾਪਸ kernel_ਨਤੀਜਾ;
}

ਤਾਰਕਿਕ ggml_et_op_cpy(ggml_backend_et_device_context * dev_ctx, ਅਚਲ ggml_tensor * ਗੰਢ) {
    ET_PERF_ਸ਼ੁਰੂ();

    // CPY copies data from src[0] into the layout of dst (which matches src[1])
    // For same-type with contiguous dst, this is identical to CONT
    ਜੇ (!ਗੰਢ->src[0]) {
        GGML_LOG_ERROR("ET: CPY operation missing source tensor\n");
        ਵਾਪਸ ਝੂਠ;
    }

    // Scalar / zero-element special path: if any dimension is 0, nothing to copy
    ਅਚਲ int64_t nelements = ਗੰਢ->ne[0] * ਗੰਢ->ne[1] * ਗੰਢ->ne[2] * ਗੰਢ->ne[3];
    ਜੇ (nelements == 0) {
        GGML_LOG_DEBUG("ET: CPY no-op (zero elements): ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]\n",
                       ਗੰਢ->ne[0], ਗੰਢ->ne[1], ਗੰਢ->ne[2], ਗੰਢ->ne[3]);
        ET_PERF_ਸਿਰਾ("CPY", "noop", ਗੰਢ);
        ਵਾਪਸ ਸੱਚ;
    }

    // Only F32 and F16 supported for dst
    ਜੇ (ਗੰਢ->type != GGML_TYPE_F32 && ਗੰਢ->type != GGML_TYPE_F16) {
        GGML_LOG_ERROR("ET: CPY unsupported dst type: %s\n", ggml_type_name(ਗੰਢ->type));
        ਵਾਪਸ ਝੂਠ;
    }

    // Select kernel based on src/dst type combination
    ਅਚਲ ਅੱਖਰ * kernel_name;
    ਜੇ (ਗੰਢ->src[0]->type == GGML_TYPE_F32 && ਗੰਢ->type == GGML_TYPE_F32) {
        kernel_name = "cont_f32";
    } ਨਹੀਂਤਾਂ ਜੇ (ਗੰਢ->src[0]->type == GGML_TYPE_F16 && ਗੰਢ->type == GGML_TYPE_F16) {
        kernel_name = "cont_f16";
    } ਨਹੀਂਤਾਂ ਜੇ (ਗੰਢ->src[0]->type == GGML_TYPE_F32 && ਗੰਢ->type == GGML_TYPE_F16) {
        kernel_name = "cpy_f32_f16";
    } ਨਹੀਂਤਾਂ {
        GGML_LOG_ERROR("ET: CPY unsupported type combination: src=%s dst=%s\n", ggml_type_name(ਗੰਢ->src[0]->type),
                       ggml_type_name(ਗੰਢ->type));
        ਵਾਪਸ ਝੂਠ;
    }

    ggml_et_cont_params params;
    params.src0 = *ਗੰਢ->src[0];
    params.dst  = *ਗੰਢ;

    // CPU comparison for debugging
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ਤਾਰਕਿਕ                    cpu_comparison_ਸਰਗਰਮ = ਝੂਠ;
    ਜੇ (cont_cpu_compare_config.enabled) {
        ਜੇ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ਗੰਢ, GGML_OP_CPY)) {
            cpu_comparison_ਸਰਗਰਮ = ਸੱਚ;
        } ਨਹੀਂਤਾਂ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for CPY operation\n");
        }
    }

    ਤਾਰਕਿਕ kernel_ਨਤੀਜਾ = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ਆਕਾਰ(params), 0xFFFFFFFF);

    ਜੇ (cpu_comparison_ਸਰਗਰਮ) {
        ਜੇ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ਗੰਢ, &cont_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for CPY operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_ਸਿਰਾ("CPY", kernel_name, ਗੰਢ);
    ਵਾਪਸ kernel_ਨਤੀਜਾ;
}

ਤਾਰਕਿਕ ggml_et_op_concat(ggml_backend_et_device_context * dev_ctx, ਅਚਲ ggml_tensor * ਗੰਢ) {
    ET_PERF_ਸ਼ੁਰੂ();

    ਜੇ (!dev_ctx || !ਗੰਢ) {
        GGML_LOG_ERROR("ET: Invalid parameters for CONCAT operation\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (!ਗੰਢ->src[0] || !ਗੰਢ->src[1]) {
        GGML_LOG_ERROR("ET: CONCAT operation missing required inputs\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ ਅੱਖਰ * kernel_name;

    ਜੇ (ਗੰਢ->type == GGML_TYPE_F32 && ਗੰਢ->src[0]->type == GGML_TYPE_F32 && ਗੰਢ->src[1]->type == GGML_TYPE_F32) {
        kernel_name = "concat_f32";

    } ਨਹੀਂਤਾਂ {
        GGML_LOG_ERROR("ET: CONCAT operation with unsupported types: dst=%s src0=%s src1=%s\n",
                       ggml_type_name(ਗੰਢ->type), ggml_type_name(ਗੰਢ->src[0]->type),
                       ggml_type_name(ਗੰਢ->src[1]->type));
        ਵਾਪਸ ਝੂਠ;
    }

    int32_t dim;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&dim, ਗੰਢ->op_params, ਆਕਾਰ(int32_t));

    ggml_et_concat_params params;
    params.src0 = *ਗੰਢ->src[0];
    params.src1 = *ਗੰਢ->src[1];
    params.dst  = *ਗੰਢ;
    params.dim  = dim;

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ਤਾਰਕਿਕ                    cpu_comparison_ਸਰਗਰਮ = ਝੂਠ;
    ਜੇ (concat_cpu_compare_config.enabled) {
        ਜੇ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ਗੰਢ, GGML_OP_CONCAT)) {
            cpu_comparison_ਸਰਗਰਮ = ਸੱਚ;
        } ਨਹੀਂਤਾਂ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for CONCAT operation\n");
        }
    }

    ਤਾਰਕਿਕ kernel_ਨਤੀਜਾ = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ਆਕਾਰ(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ਜੇ (cpu_comparison_ਸਰਗਰਮ) {
        ਜੇ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ਗੰਢ, &concat_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for CONCAT operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_ਸਿਰਾ_EXT("CONCAT", kernel_name, ਗੰਢ, "dim=%d", dim);
    ਵਾਪਸ kernel_ਨਤੀਜਾ;
}

ਤਾਰਕਿਕ ggml_et_op_repeat(ggml_backend_et_device_context * dev_ctx, ਅਚਲ ggml_tensor * ਗੰਢ) {
    ET_PERF_ਸ਼ੁਰੂ();

    ਜੇ (!dev_ctx || !ਗੰਢ) {
        GGML_LOG_ERROR("ET: Invalid parameters for REPEAT operation\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (!ਗੰਢ->src[0]) {
        GGML_LOG_ERROR("ET: REPEAT operation missing required input\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ ਅੱਖਰ * kernel_name;

    ਜੇ (ਗੰਢ->type == GGML_TYPE_F32 && ਗੰਢ->src[0]->type == GGML_TYPE_F32) {
        // No-op REPEAT (every repeat factor is 1): the output is just a copy
        // of the input. Route to cont_f32, whose contiguous fast path handles
        // arbitrary sizes (including those rejected by repeat_f32's gate,
        // e.g. ne[0]=1).
        ਜੇ (ggml_are_same_shape(ਗੰਢ->src[0], ਗੰਢ)) {
            kernel_name = "cont_f32";
        } ਨਹੀਂਤਾਂ {
            kernel_name = "repeat_f32";
        }

    } ਨਹੀਂਤਾਂ {
        GGML_LOG_ERROR("ET: REPEAT operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(ਗੰਢ->type),
                       ggml_type_name(ਗੰਢ->src[0]->type));
        ਵਾਪਸ ਝੂਠ;
    }

    // ggml_et_cont_params and ggml_et_repeat_params have identical layouts
    // (just src0 + dst), so the same payload works for either kernel.
    ggml_et_repeat_params params;
    params.src0 = *ਗੰਢ->src[0];
    params.dst  = *ਗੰਢ;

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ਤਾਰਕਿਕ                    cpu_comparison_ਸਰਗਰਮ = ਝੂਠ;
    ਜੇ (repeat_cpu_compare_config.enabled) {
        ਜੇ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ਗੰਢ, GGML_OP_REPEAT)) {
            cpu_comparison_ਸਰਗਰਮ = ਸੱਚ;
        } ਨਹੀਂਤਾਂ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for REPEAT operation\n");
        }
    }

    ਤਾਰਕਿਕ kernel_ਨਤੀਜਾ = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ਆਕਾਰ(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ਜੇ (cpu_comparison_ਸਰਗਰਮ) {
        ਜੇ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ਗੰਢ, &repeat_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for REPEAT operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_ਸਿਰਾ("REPEAT", kernel_name, ਗੰਢ);
    ਵਾਪਸ kernel_ਨਤੀਜਾ;
}

ਤਾਰਕਿਕ ggml_et_op_ssm_conv(ggml_backend_et_device_context * dev_ctx, ਅਚਲ ggml_tensor * ਗੰਢ) {
    ET_PERF_ਸ਼ੁਰੂ();

    ਜੇ (!dev_ctx || !ਗੰਢ || !ਗੰਢ->src[0] || !ਗੰਢ->src[1]) {
        GGML_LOG_ERROR("ET: Invalid parameters for SSM_CONV operation\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (ਗੰਢ->type != GGML_TYPE_F32 || ਗੰਢ->src[0]->type != GGML_TYPE_F32 || ਗੰਢ->src[1]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: SSM_CONV operation with unsupported types: dst=%s src0=%s src1=%s\n",
                       ggml_type_name(ਗੰਢ->type), ggml_type_name(ਗੰਢ->src[0]->type),
                       ggml_type_name(ਗੰਢ->src[1]->type));
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ ਅੱਖਰ * kernel_name = "ssm_conv_f32";

    ggml_et_ssm_conv_params params;
    params.src0 = *ਗੰਢ->src[0];
    params.src1 = *ਗੰਢ->src[1];
    params.dst  = *ਗੰਢ;

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ਤਾਰਕਿਕ                    cpu_comparison_ਸਰਗਰਮ = ਝੂਠ;
    ਜੇ (ssm_conv_cpu_compare_config.enabled) {
        ਜੇ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ਗੰਢ, GGML_OP_SSM_CONV)) {
            cpu_comparison_ਸਰਗਰਮ = ਸੱਚ;
        } ਨਹੀਂਤਾਂ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for SSM_CONV operation\n");
        }
    }

    ਤਾਰਕਿਕ kernel_ਨਤੀਜਾ = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ਆਕਾਰ(params), 0xFFFFFFFF);

    ਜੇ (cpu_comparison_ਸਰਗਰਮ) {
        ਜੇ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ਗੰਢ, &ssm_conv_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for SSM_CONV operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_ਸਿਰਾ("SSM_CONV", kernel_name, ਗੰਢ);
    ਵਾਪਸ kernel_ਨਤੀਜਾ;
}

ਤਾਰਕਿਕ ggml_et_op_ssm_ਜਾਂਚ(ggml_backend_et_device_context * dev_ctx, ਅਚਲ ggml_tensor * ਗੰਢ) {
    ET_PERF_ਸ਼ੁਰੂ();

    ਜੇ (!dev_ctx || !ਗੰਢ) {
        GGML_LOG_ERROR("ET: Invalid parameters for SSM_SCAN operation\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਲਈ (ਪੂਰਨ i = 0; i < 7; ++i) {
        ਜੇ (!ਗੰਢ->src[i]) {
            GGML_LOG_ERROR("ET: SSM_SCAN missing required input %d\n", i);
            ਵਾਪਸ ਝੂਠ;
        }
    }

    ਜੇ (ਗੰਢ->type != GGML_TYPE_F32 || ਗੰਢ->src[0]->type != GGML_TYPE_F32 || ਗੰਢ->src[1]->type != GGML_TYPE_F32 ||
        ਗੰਢ->src[2]->type != GGML_TYPE_F32 || ਗੰਢ->src[3]->type != GGML_TYPE_F32 ||
        ਗੰਢ->src[4]->type != GGML_TYPE_F32 || ਗੰਢ->src[5]->type != GGML_TYPE_F32 ||
        ਗੰਢ->src[6]->type != GGML_TYPE_I32) {
        GGML_LOG_ERROR("ET: SSM_SCAN operation with unsupported types\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ggml_et_ssm_ਜਾਂਚ_params params;
    params.src0 = *ਗੰਢ->src[0];
    params.src1 = *ਗੰਢ->src[1];
    params.src2 = *ਗੰਢ->src[2];
    params.src3 = *ਗੰਢ->src[3];
    params.src4 = *ਗੰਢ->src[4];
    params.src5 = *ਗੰਢ->src[5];
    params.src6 = *ਗੰਢ->src[6];
    params.dst  = *ਗੰਢ;
    params.K    = ggml_get_op_params_i32(ਗੰਢ, 0);

    ਤਾਰਕਿਕ kernel_ਨਤੀਜਾ = ggml_et_launch_kernel(dev_ctx, "ssm_scan_f32", &params, ਆਕਾਰ(params), 0xFFFFFFFF);

    ET_PERF_ਸਿਰਾ("SSM_SCAN", "ssm_scan_f32", ਗੰਢ);
    ਵਾਪਸ kernel_ਨਤੀਜਾ;
}

ਤਾਰਕਿਕ ggml_et_op_rwkv_wkv6(ggml_backend_et_device_context * dev_ctx, ਅਚਲ ggml_tensor * ਗੰਢ) {
    ET_PERF_ਸ਼ੁਰੂ();

    ਜੇ (!dev_ctx || !ਗੰਢ) {
        GGML_LOG_ERROR("ET: Invalid parameters for RWKV_WKV6 operation\n");
        ਵਾਪਸ ਝੂਠ;
    }

    // Validate all 6 source tensors exist
    ਲਈ (ਪੂਰਨ i = 0; i <= 5; i++) {
        ਜੇ (!ਗੰਢ->src[i]) {
            GGML_LOG_ERROR("ET: RWKV_WKV6 operation missing src[%d]\n", i);
            ਵਾਪਸ ਝੂਠ;
        }
    }

    ਜੇ (ਗੰਢ->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: RWKV_WKV6 only supports F32, got %s\n", ggml_type_name(ਗੰਢ->type));
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ ਅੱਖਰ * kernel_name = "rwkv_wkv6_f32";

    ਅਚਲ int64_t S      = ਗੰਢ->src[0]->ne[0];  // head_size
    ਅਚਲ int64_t H      = ਗੰਢ->src[0]->ne[1];  // num heads
    ਅਚਲ int64_t T      = ਗੰਢ->src[1]->ne[2];  // num tokens
    ਅਚਲ int64_t n_seqs = ਗੰਢ->src[5]->ne[1];  // num sequences
    ਅਚਲ int64_t C      = S * H;

    ggml_et_rwkv_wkv6_params params;
    params.k        = (ਦਸ਼ਮਲਵ *) ਗੰਢ->src[0]->ਅੰਕੜੇ;
    params.v        = (ਦਸ਼ਮਲਵ *) ਗੰਢ->src[1]->ਅੰਕੜੇ;
    params.r        = (ਦਸ਼ਮਲਵ *) ਗੰਢ->src[2]->ਅੰਕੜੇ;
    params.tf       = (ਦਸ਼ਮਲਵ *) ਗੰਢ->src[3]->ਅੰਕੜੇ;
    params.td       = (ਦਸ਼ਮਲਵ *) ਗੰਢ->src[4]->ਅੰਕੜੇ;
    params.state_ਵਿੱਚ = (ਦਸ਼ਮਲਵ *) ਗੰਢ->src[5]->ਅੰਕੜੇ;
    params.dst      = (ਦਸ਼ਮਲਵ *) ਗੰਢ->ਅੰਕੜੇ;
    params.C        = (int32_t) C;
    params.H        = (int32_t) H;
    params.S        = (int32_t) S;
    params.T        = (int32_t) T;
    params.n_seqs   = (int32_t) n_seqs;

    // Phase 1: Initialize CPU comparison context
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ਤਾਰਕਿਕ                    cpu_comparison_ਸਰਗਰਮ = ਝੂਠ;
    ਜੇ (rwkv_wkv6_cpu_compare_config.enabled) {
        ਜੇ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ਗੰਢ, GGML_OP_RWKV_WKV6)) {
            cpu_comparison_ਸਰਗਰਮ = ਸੱਚ;
        } ਨਹੀਂਤਾਂ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for RWKV_WKV6 operation\n");
        }
    }

    ਤਾਰਕਿਕ kernel_ਨਤੀਜਾ = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ਆਕਾਰ(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare
    ਜੇ (cpu_comparison_ਸਰਗਰਮ) {
        ਜੇ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ਗੰਢ, &rwkv_wkv6_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for RWKV_WKV6 operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_ਸਿਰਾ_EXT("RWKV_WKV6", kernel_name, ਗੰਢ, "S=%d H=%d T=%d n_seqs=%d", (ਪੂਰਨ) S, (ਪੂਰਨ) H, (ਪੂਰਨ) T,
                    (ਪੂਰਨ) n_seqs);
    ਵਾਪਸ kernel_ਨਤੀਜਾ;
}

ਤਾਰਕਿਕ ggml_et_op_rwkv_wkv7(ggml_backend_et_device_context * dev_ctx, ਅਚਲ ggml_tensor * ਗੰਢ) {
    ET_PERF_ਸ਼ੁਰੂ();

    ਜੇ (!dev_ctx || !ਗੰਢ) {
        GGML_LOG_ERROR("ET: Invalid parameters for RWKV_WKV7 operation\n");
        ਵਾਪਸ ਝੂਠ;
    }

    // Validate all 7 source tensors exist
    ਲਈ (ਪੂਰਨ i = 0; i <= 6; i++) {
        ਜੇ (!ਗੰਢ->src[i]) {
            GGML_LOG_ERROR("ET: RWKV_WKV7 operation missing src[%d]\n", i);
            ਵਾਪਸ ਝੂਠ;
        }
    }

    ਜੇ (ਗੰਢ->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: RWKV_WKV7 only supports F32, got %s\n", ggml_type_name(ਗੰਢ->type));
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ ਅੱਖਰ * kernel_name = "rwkv_wkv7_f32";

    ਅਚਲ int64_t S      = ਗੰਢ->src[2]->ne[0];  // head_size
    ਅਚਲ int64_t H      = ਗੰਢ->src[2]->ne[1];  // num heads
    ਅਚਲ int64_t T      = ਗੰਢ->src[1]->ne[2];  // num tokens
    ਅਚਲ int64_t n_seqs = ਗੰਢ->src[6]->ne[1];  // num sequences
    ਅਚਲ int64_t C      = S * H;

    ggml_et_rwkv_wkv7_params params;
    params.r        = (ਦਸ਼ਮਲਵ *) ਗੰਢ->src[0]->ਅੰਕੜੇ;
    params.w        = (ਦਸ਼ਮਲਵ *) ਗੰਢ->src[1]->ਅੰਕੜੇ;
    params.k        = (ਦਸ਼ਮਲਵ *) ਗੰਢ->src[2]->ਅੰਕੜੇ;
    params.v        = (ਦਸ਼ਮਲਵ *) ਗੰਢ->src[3]->ਅੰਕੜੇ;
    params.a        = (ਦਸ਼ਮਲਵ *) ਗੰਢ->src[4]->ਅੰਕੜੇ;
    params.b        = (ਦਸ਼ਮਲਵ *) ਗੰਢ->src[5]->ਅੰਕੜੇ;
    params.state_ਵਿੱਚ = (ਦਸ਼ਮਲਵ *) ਗੰਢ->src[6]->ਅੰਕੜੇ;
    params.dst      = (ਦਸ਼ਮਲਵ *) ਗੰਢ->ਅੰਕੜੇ;
    params.C        = (int32_t) C;
    params.H        = (int32_t) H;
    params.S        = (int32_t) S;
    params.T        = (int32_t) T;
    params.n_seqs   = (int32_t) n_seqs;

    // Phase 1: Initialize CPU comparison context
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ਤਾਰਕਿਕ                    cpu_comparison_ਸਰਗਰਮ = ਝੂਠ;
    ਜੇ (rwkv_wkv7_cpu_compare_config.enabled) {
        ਜੇ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ਗੰਢ, GGML_OP_RWKV_WKV7)) {
            cpu_comparison_ਸਰਗਰਮ = ਸੱਚ;
        } ਨਹੀਂਤਾਂ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for RWKV_WKV7 operation\n");
        }
    }

    ਤਾਰਕਿਕ kernel_ਨਤੀਜਾ = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ਆਕਾਰ(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare
    ਜੇ (cpu_comparison_ਸਰਗਰਮ) {
        ਜੇ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ਗੰਢ, &rwkv_wkv7_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for RWKV_WKV7 operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_ਸਿਰਾ_EXT("RWKV_WKV7", kernel_name, ਗੰਢ, "S=%d H=%d T=%d n_seqs=%d", (ਪੂਰਨ) S, (ਪੂਰਨ) H, (ਪੂਰਨ) T,
                    (ਪੂਰਨ) n_seqs);
    ਵਾਪਸ kernel_ਨਤੀਜਾ;
}

ਸਥਿਰ ggml_et_cpu_compare_config gated_delta_net_cpu_compare_config = {
    /* .enabled = */ ਝੂਠ,
    /* .use_cpu_result = */ ਝੂਠ,
    /* .log_differences = */ ਸੱਚ,
    /* .tolerance = */ 1e-4f,
    /* .max_log_elements = */ 4096
};

ਤਾਰਕਿਕ ggml_et_op_gated_delta_net(ggml_backend_et_device_context * dev_ctx, ਅਚਲ ggml_tensor * ਗੰਢ) {
    ET_PERF_ਸ਼ੁਰੂ();

    ਜੇ (!dev_ctx || !ਗੰਢ) {
        GGML_LOG_ERROR("ET: Invalid parameters for GATED_DELTA_NET operation\n");
        ਵਾਪਸ ਝੂਠ;
    }

    // Validate all 6 source tensors exist
    ਲਈ (ਪੂਰਨ i = 0; i <= 5; i++) {
        ਜੇ (!ਗੰਢ->src[i]) {
            GGML_LOG_ERROR("ET: GATED_DELTA_NET operation missing src[%d]\n", i);
            ਵਾਪਸ ਝੂਠ;
        }
    }

    ਜੇ (ਗੰਢ->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: GATED_DELTA_NET only supports F32, got %s\n", ggml_type_name(ਗੰਢ->type));
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ ਅੱਖਰ * kernel_name = "gated_delta_net_f32";

    ਅਚਲ ggml_tensor * src_q     = ਗੰਢ->src[0];
    ਅਚਲ ggml_tensor * src_k     = ਗੰਢ->src[1];
    ਅਚਲ ggml_tensor * src_v     = ਗੰਢ->src[2];
    ਅਚਲ ggml_tensor * src_g     = ਗੰਢ->src[3];
    ਅਚਲ ggml_tensor * src_beta  = ਗੰਢ->src[4];
    ਅਚਲ ggml_tensor * src_state = ਗੰਢ->src[5];

    ਅਚਲ int64_t S_v      = src_v->ne[0];
    ਅਚਲ int64_t H        = src_v->ne[1];
    ਅਚਲ int64_t n_ਟੋਕਨ = src_v->ne[2];
    ਅਚਲ int64_t n_seqs   = src_v->ne[3];
    ਅਚਲ int64_t H_q      = src_q->ne[1];
    ਅਚਲ int64_t H_k      = src_k->ne[1];
    ਅਚਲ int64_t n_seqs_q = src_q->ne[3];
    ਅਚਲ int64_t n_seqs_k = src_k->ne[3];

    ggml_et_gated_delta_net_params params;
    params.q        = *src_q;
    params.k        = *src_k;
    params.v        = *src_v;
    params.g        = *src_g;
    params.beta     = *src_beta;
    params.state_ਵਿੱਚ = *src_state;
    params.dst      = *ਗੰਢ;
    params.S_v      = (int32_t) S_v;
    params.H        = (int32_t) H;
    params.H_q      = (int32_t) H_q;
    params.H_k      = (int32_t) H_k;
    params.n_ਟੋਕਨ = (int32_t) n_ਟੋਕਨ;
    params.n_seqs   = (int32_t) n_seqs;
    params.n_seqs_q = (int32_t) n_seqs_q;
    params.n_seqs_k = (int32_t) n_seqs_k;
    params.kda      = (src_g->ne[0] == S_v) ? 1 : 0;
    params.K        = ggml_get_op_params_i32(ਗੰਢ, 0);
    params.scale    = 1.0f / sqrtf((ਦਸ਼ਮਲਵ) S_v);

    // CPU comparison for debugging
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ਤਾਰਕਿਕ                    cpu_comparison_ਸਰਗਰਮ = ਝੂਠ;
    ਜੇ (gated_delta_net_cpu_compare_config.enabled) {
        ਜੇ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ਗੰਢ, GGML_OP_GATED_DELTA_NET)) {
            cpu_comparison_ਸਰਗਰਮ = ਸੱਚ;
        } ਨਹੀਂਤਾਂ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for GATED_DELTA_NET operation\n");
        }
    }

    ਤਾਰਕਿਕ kernel_ਨਤੀਜਾ = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ਆਕਾਰ(params), 0xFFFFFFFF);

    ਜੇ (cpu_comparison_ਸਰਗਰਮ) {
        ਜੇ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ਗੰਢ, &gated_delta_net_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for GATED_DELTA_NET operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_ਸਿਰਾ_EXT("GATED_DELTA_NET", kernel_name, ਗੰਢ, "S_v=%d H=%d n_tokens=%d n_seqs=%d kda=%d", (ਪੂਰਨ) S_v,
                    (ਪੂਰਨ) H, (ਪੂਰਨ) n_ਟੋਕਨ, (ਪੂਰਨ) n_seqs, params.kda);
    ਵਾਪਸ kernel_ਨਤੀਜਾ;
}

ਤਾਰਕਿਕ ggml_et_op_ਸੈੱਟ_ਪੰਕਤੀਆਂ(ggml_backend_et_device_context * dev_ctx, ਅਚਲ ggml_tensor * ਗੰਢ) {
    ET_PERF_ਸ਼ੁਰੂ();

    ਜੇ (!dev_ctx || !ਗੰਢ) {
        GGML_LOG_ERROR("ET: Invalid parameters for SET_ROWS operation\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (!ਗੰਢ->src[0] || !ਗੰਢ->src[1] || !ਗੰਢ->src[2]) {
        GGML_LOG_ERROR(
            "ET: SET_ROWS operation missing required inputs (needs src[0]=base, src[1]=indices, src[2]=data)\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ ਅੱਖਰ * kernel_name;

    // Support F32 data with I64 indices -> F32/F16 output (scatter operation)
    ਜੇ (ਗੰਢ->src[0]->type == GGML_TYPE_F32 && ਗੰਢ->src[1]->type == GGML_TYPE_I64 &&
        (ਗੰਢ->type == GGML_TYPE_F32 || ਗੰਢ->type == GGML_TYPE_F16)) {
        ਜੇ (ਗੰਢ->type == GGML_TYPE_F32 || ਗੰਢ->type == GGML_TYPE_F16) {
            kernel_name = "set_rows_f32";
        } ਨਹੀਂਤਾਂ {
            GGML_LOG_ERROR("ET: SET_ROWS unsupported output type: %s\n", ggml_type_name(ਗੰਢ->type));
            ਵਾਪਸ ਝੂਠ;
        }

    } ਨਹੀਂਤਾਂ {
        GGML_LOG_ERROR("ET: SET_ROWS operation with unsupported types: dst=%s src0=%s src1=%s\n",
                       ggml_type_name(ਗੰਢ->type), ggml_type_name(ਗੰਢ->src[0]->type),
                       ggml_type_name(ਗੰਢ->src[1]->type));
        ਵਾਪਸ ਝੂਠ;
    }

    // Validate contiguity requirements
    ਜੇ (!ggml_is_contiguous_ਪੰਕਤੀਆਂ(ਗੰਢ)) {
        GGML_LOG_ERROR("ET: SET_ROWS operation requires contiguous-rows destination tensor\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (!ggml_is_contiguous_ਪੰਕਤੀਆਂ(ਗੰਢ->src[0])) {
        GGML_LOG_ERROR("ET: SET_ROWS operation requires contiguous-rows source tensor\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (!ggml_is_contiguous(ਗੰਢ->src[1])) {
        GGML_LOG_ERROR("ET: SET_ROWS operation requires contiguous indices tensor\n");
        ਵਾਪਸ ਝੂਠ;
    }

    // Validate dimension constraints from ggml implementation
    ਜੇ (!(ਗੰਢ->ne[0] == ਗੰਢ->src[0]->ne[0] &&              // same number of columns
          ਗੰਢ->ne[2] == ਗੰਢ->src[0]->ne[2] &&              // same batch size
          ਗੰਢ->ne[3] == ਗੰਢ->src[0]->ne[3] &&              // same outer dimension
          ਗੰਢ->src[0]->ne[1] == ਗੰਢ->src[1]->ne[0] &&      // src rows = index count
          ਗੰਢ->src[0]->ne[2] % ਗੰਢ->src[1]->ne[1] == 0 &&  // batch constraint
          ਗੰਢ->src[0]->ne[3] % ਗੰਢ->src[1]->ne[2] == 0 &&  // outer constraint
          ਗੰਢ->src[1]->ne[3] == 1)) {                       // indices constraint
        GGML_LOG_ERROR("ET: SET_ROWS operation dimension constraint failed\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ggml_et_ਸੈੱਟ_ਪੰਕਤੀਆਂ_params params;
    params.src0 = *ਗੰਢ->src[0];  // F32 source data tensor
    params.src1 = *ਗੰਢ->src[1];  // I64 indices tensor
    params.dst  = *ਗੰਢ;          // F32/F16 destination tensor

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ਤਾਰਕਿਕ                    cpu_comparison_ਸਰਗਰਮ = ਝੂਠ;
    ਜੇ (ਸੈੱਟ_ਪੰਕਤੀਆਂ_cpu_compare_config.enabled) {
        ਜੇ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ਗੰਢ, GGML_OP_ਸੈੱਟ_ਪੰਕਤੀਆਂ)) {
            cpu_comparison_ਸਰਗਰਮ = ਸੱਚ;
        } ਨਹੀਂਤਾਂ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for SET_ROWS operation\n");
        }
    }

    ਤਾਰਕਿਕ kernel_ਨਤੀਜਾ = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ਆਕਾਰ(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ਜੇ (cpu_comparison_ਸਰਗਰਮ) {
        ਜੇ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ਗੰਢ, &ਸੈੱਟ_ਪੰਕਤੀਆਂ_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for SET_ROWS operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_ਸਿਰਾ("SET_ROWS", kernel_name, ਗੰਢ);
    ਵਾਪਸ kernel_ਨਤੀਜਾ;
}

ਤਾਰਕਿਕ ggml_et_op_fill(ggml_backend_et_device_context * dev_ctx, ਅਚਲ ggml_tensor * ਗੰਢ) {
    ET_PERF_ਸ਼ੁਰੂ();

    ggml_et_fill_params params;
    params.dst = *ਗੰਢ;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&params.c, ਗੰਢ->op_params, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਤਾਰਕਿਕ kernel_ਨਤੀਜਾ = ggml_et_launch_kernel(dev_ctx, "fill_f32", &params, ਆਕਾਰ(params), 0xFFFFFFFF);

    ET_PERF_ਸਿਰਾ("FILL", "fill_f32", ਗੰਢ);
    ਵਾਪਸ kernel_ਨਤੀਜਾ;
}

ਤਾਰਕਿਕ ggml_et_op_diag(ggml_backend_et_device_context * dev_ctx, ਅਚਲ ggml_tensor * ਗੰਢ) {
    ET_PERF_ਸ਼ੁਰੂ();

    ਜੇ (!ਗੰਢ->src[0]) {
        GGML_LOG_ERROR("ET: DIAG operation missing source tensor\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ggml_et_diag_params params;
    params.src0 = *ਗੰਢ->src[0];
    params.dst  = *ਗੰਢ;

    ਤਾਰਕਿਕ kernel_ਨਤੀਜਾ = ggml_et_launch_kernel(dev_ctx, "diag_f32", &params, ਆਕਾਰ(params), 0xFFFFFFFF);

    ET_PERF_ਸਿਰਾ("DIAG", "diag_f32", ਗੰਢ);
    ਵਾਪਸ kernel_ਨਤੀਜਾ;
}

ਤਾਰਕਿਕ ggml_et_op_tri(ggml_backend_et_device_context * dev_ctx, ਅਚਲ ggml_tensor * ਗੰਢ) {
    ET_PERF_ਸ਼ੁਰੂ();

    ਜੇ (!ਗੰਢ->src[0]) {
        GGML_LOG_ERROR("ET: TRI operation missing source tensor\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ggml_et_tri_params params;
    params.src0 = *ਗੰਢ->src[0];
    params.dst  = *ਗੰਢ;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&params.tri_type, ਗੰਢ->op_params, ਆਕਾਰ(int32_t));

    ਤਾਰਕਿਕ kernel_ਨਤੀਜਾ = ggml_et_launch_kernel(dev_ctx, "tri_f32", &params, ਆਕਾਰ(params), 0xFFFFFFFF);

    ET_PERF_ਸਿਰਾ("TRI", "tri_f32", ਗੰਢ);
    ਵਾਪਸ kernel_ਨਤੀਜਾ;
}

ਤਾਰਕਿਕ ggml_et_op_solve_tri(ggml_backend_et_device_context * dev_ctx, ਅਚਲ ggml_tensor * ਗੰਢ) {
    ET_PERF_ਸ਼ੁਰੂ();

    ਜੇ (!ਗੰਢ->src[0] || !ਗੰਢ->src[1]) {
        GGML_LOG_ERROR("ET: SOLVE_TRI operation missing source tensor(s)\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ggml_et_solve_tri_params params;
    params.src0 = *ਗੰਢ->src[0];  // A (lower-triangular)
    params.src1 = *ਗੰਢ->src[1];  // B (RHS)
    params.dst  = *ਗੰਢ;          // X (solution)

    ਤਾਰਕਿਕ kernel_ਨਤੀਜਾ = ggml_et_launch_kernel(dev_ctx, "solve_tri_f32", &params, ਆਕਾਰ(params), 0xFFFFFFFF);

    ET_PERF_ਸਿਰਾ("SOLVE_TRI", "solve_tri_f32", ਗੰਢ);
    ਵਾਪਸ kernel_ਨਤੀਜਾ;
}

ਤਾਰਕਿਕ ggml_et_op_ਸੈੱਟ(ggml_backend_et_device_context * dev_ctx, ਅਚਲ ggml_tensor * ਗੰਢ) {
    ET_PERF_ਸ਼ੁਰੂ();

    ਜੇ (!ਗੰਢ->src[0] || !ਗੰਢ->src[1]) {
        GGML_LOG_ERROR("ET: SET operation missing source tensor(s)\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ ਤਾਰਕਿਕ   inplace = (ਤਾਰਕਿਕ) ((ਅਚਲ int32_t *) ਗੰਢ->op_params)[4];
    ਅਚਲ size_t ਵਿੱਥ  = ((ਅਚਲ int32_t *) ਗੰਢ->op_params)[3];
    ਅਚਲ size_t nb1     = ((ਅਚਲ int32_t *) ਗੰਢ->op_params)[0];
    ਅਚਲ size_t nb2     = ((ਅਚਲ int32_t *) ਗੰਢ->op_params)[1];
    ਅਚਲ size_t nb3     = ((ਅਚਲ int32_t *) ਗੰਢ->op_params)[2];

    ਜੇ (!inplace) {
        GGML_LOG_ERROR("ET: SET only supports inplace (inplace=%d)\n", inplace);
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (ਗੰਢ->type != GGML_TYPE_F32 || ਗੰਢ->src[0]->type != GGML_TYPE_F32 || ਗੰਢ->src[1]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: SET only supports F32 (dst=%s src0=%s src1=%s)\n", ggml_type_name(ਗੰਢ->type),
                       ggml_type_name(ਗੰਢ->src[0]->type), ggml_type_name(ਗੰਢ->src[1]->type));
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (!ggml_are_same_shape(ਗੰਢ, ਗੰਢ->src[0])) {
        GGML_LOG_ERROR("ET: SET requires same-shape src0 and dst\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (!ggml_is_contiguous(ਗੰਢ) || !ggml_is_contiguous(ਗੰਢ->src[0]) || !ggml_is_contiguous(ਗੰਢ->src[1])) {
        GGML_LOG_ERROR("ET: SET requires contiguous dst, src0, and src1\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ggml_et_ਸੈੱਟ_params params;
    params.src1   = *ਗੰਢ->src[1];
    params.dst    = *ਗੰਢ;
    params.nb1    = (int32_t) nb1;
    params.nb2    = (int32_t) nb2;
    params.nb3    = (int32_t) nb3;
    params.ਵਿੱਥ = (int32_t) ਵਿੱਥ;

    ਤਾਰਕਿਕ kernel_ਨਤੀਜਾ = ggml_et_launch_kernel(dev_ctx, "set_f32", &params, ਆਕਾਰ(params), 0xFFFFFFFF);

    ET_PERF_ਸਿਰਾ("SET", "set_f32", ਗੰਢ);
    ਵਾਪਸ kernel_ਨਤੀਜਾ;
}

ਤਾਰਕਿਕ ggml_et_op_pad(ggml_backend_et_device_context * dev_ctx, ਅਚਲ ggml_tensor * ਗੰਢ) {
    ET_PERF_ਸ਼ੁਰੂ();

    ਜੇ (!ਗੰਢ->src[0]) {
        GGML_LOG_ERROR("ET: PAD operation missing source tensor\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (ਗੰਢ->type != GGML_TYPE_F32 || ਗੰਢ->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: PAD only supports F32 (src=%s dst=%s)\n", ggml_type_name(ਗੰਢ->src[0]->type),
                       ggml_type_name(ਗੰਢ->type));
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (!ggml_is_contiguous(ਗੰਢ)) {
        GGML_LOG_ERROR("ET: PAD requires contiguous output tensor\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (ਗੰਢ->src[0]->nb[0] != ਆਕਾਰ(ਦਸ਼ਮਲਵ)) {
        GGML_LOG_ERROR("ET: PAD requires element-contiguous src dim0 (nb[0]=%zu)\n", (size_t) ਗੰਢ->src[0]->nb[0]);
        ਵਾਪਸ ਝੂਠ;
    }

    // Extract padding parameters from op_params
    ਅਚਲ int32_t * op_params = (ਅਚਲ int32_t *) ਗੰਢ->op_params;

    ggml_et_pad_params params;
    params.src0  = *ਗੰਢ->src[0];
    params.dst   = *ਗੰਢ;
    params.lp[0] = op_params[0];
    params.rp[0] = op_params[1];
    params.lp[1] = op_params[2];
    params.rp[1] = op_params[3];
    params.lp[2] = op_params[4];
    params.rp[2] = op_params[5];
    params.lp[3] = op_params[6];
    params.rp[3] = op_params[7];

    // v1: no dim0 padding
    ਜੇ (params.lp[0] != 0 || params.rp[0] != 0) {
        GGML_LOG_ERROR("ET: PAD dim0 padding not supported (lp0=%d rp0=%d)\n", params.lp[0], params.rp[0]);
        ਵਾਪਸ ਝੂਠ;
    }

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ਤਾਰਕਿਕ                    cpu_comparison_ਸਰਗਰਮ = ਝੂਠ;
    ਜੇ (pad_cpu_compare_config.enabled) {
        ਜੇ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ਗੰਢ, GGML_OP_PAD)) {
            cpu_comparison_ਸਰਗਰਮ = ਸੱਚ;
        } ਨਹੀਂਤਾਂ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for PAD operation\n");
        }
    }

    ਤਾਰਕਿਕ kernel_ਨਤੀਜਾ = ggml_et_launch_kernel(dev_ctx, "pad_f32", &params, ਆਕਾਰ(params), 0xFFFFFFFF);

    ਜੇ (cpu_comparison_ਸਰਗਰਮ) {
        ਜੇ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ਗੰਢ, &pad_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for PAD operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_ਸਿਰਾ("PAD", "pad_f32", ਗੰਢ);
    ਵਾਪਸ kernel_ਨਤੀਜਾ;
}
