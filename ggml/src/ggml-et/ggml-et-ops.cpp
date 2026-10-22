#ضمن "ggml-et-ops.h"

#ضمن "ggml-et-cpu-compare.h"
#ضمن "ggml-et-kernels.h"
#ضمن "ggml-impl.h"

#ضمن <stdio.h>

#ضمن <cstdint>

// CPU comparison configuration - can be enabled for debugging
ساكن ggml_et_cpu_compare_config rope_cpu_compare_config = {
    /* .enabled = */ خطأ,
    /* .use_cpu_result = */ خطأ,  // Replace ET result with CPU result
    /* .log_differences = */ true,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

ساكن ggml_et_cpu_compare_config rms_norm_cpu_compare_config = {
    /* .enabled = */ خطأ,
    /* .use_cpu_result = */ خطأ,
    /* .log_differences = */ true,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

ساكن ggml_et_cpu_compare_config norm_cpu_compare_config = {
    /* .enabled = */ خطأ,
    /* .use_cpu_result = */ خطأ,
    /* .log_differences = */ true,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

ساكن ggml_et_cpu_compare_config l2_norm_cpu_compare_config = {
    /* .enabled = */ خطأ,
    /* .use_cpu_result = */ خطأ,
    /* .log_differences = */ true,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

ساكن ggml_et_cpu_compare_config group_norm_cpu_compare_config = {
    /* .enabled = */ خطأ,
    /* .use_cpu_result = */ خطأ,
    /* .log_differences = */ true,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

ساكن ggml_et_cpu_compare_config im2col_cpu_compare_config = {
    /* .enabled = */ خطأ,
    /* .use_cpu_result = */ خطأ,
    /* .log_differences = */ true,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

ساكن ggml_et_cpu_compare_config unary_cpu_compare_config = {
    /* .enabled = */ خطأ,
    /* .use_cpu_result = */ خطأ,
    /* .log_differences = */ true,
    /* .tolerance = */ 1e-4f,
    /* .max_log_elements = */ 4096
};

ساكن ggml_et_cpu_compare_config sum_rows_cpu_compare_config = {
    /* .enabled = */ خطأ,
    /* .use_cpu_result = */ خطأ,
    /* .log_differences = */ true,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

ساكن ggml_et_cpu_compare_config clamp_cpu_compare_config = {
    /* .enabled = */ خطأ,
    /* .use_cpu_result = */ خطأ,
    /* .log_differences = */ true,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

ساكن ggml_et_cpu_compare_config mean_cpu_compare_config = {
    /* .enabled = */ خطأ,
    /* .use_cpu_result = */ خطأ,
    /* .log_differences = */ true,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

ساكن ggml_et_cpu_compare_config sqr_cpu_compare_config = {
    /* .enabled = */ خطأ,
    /* .use_cpu_result = */ خطأ,
    /* .log_differences = */ true,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

ساكن ggml_et_cpu_compare_config elmap_cpu_compare_config = {
    /* .enabled = */ خطأ,
    /* .use_cpu_result = */ خطأ,
    /* .log_differences = */ true,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

ساكن ggml_et_cpu_compare_config glu_cpu_compare_config = {
    /* .enabled = */ خطأ,
    /* .use_cpu_result = */ خطأ,
    /* .log_differences = */ true,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

ساكن ggml_et_cpu_compare_config mul_mat_cpu_compare_config = {
    /* .enabled = */ خطأ,
    /* .use_cpu_result = */ خطأ,
    /* .log_differences = */ true,
    /* .tolerance = */ 0.01,
    /* .max_log_elements = */ 4096
};

ساكن ggml_et_cpu_compare_config mul_mat_id_cpu_compare_config = {
    /* .enabled = */ خطأ,
    /* .use_cpu_result = */ خطأ,
    /* .log_differences = */ true,
    /* .tolerance = */ 0.01,
    /* .max_log_elements = */ 4096
};

ساكن ggml_et_cpu_compare_config softmax_cpu_compare_config = {
    /* .enabled = */ خطأ,
    /* .use_cpu_result = */ خطأ,
    /* .log_differences = */ true,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 1024
};

ساكن ggml_et_cpu_compare_config get_rows_cpu_compare_config = {
    /* .enabled = */ خطأ,
    /* .use_cpu_result = */ خطأ,
    /* .log_differences = */ true,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 2048
};

ساكن ggml_et_cpu_compare_config pad_cpu_compare_config = {
    /* .enabled = */ خطأ,
    /* .use_cpu_result = */ خطأ,
    /* .log_differences = */ true,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

ساكن ggml_et_cpu_compare_config cont_cpu_compare_config = {
    /* .enabled = */ خطأ,
    /* .use_cpu_result = */ خطأ,
    /* .log_differences = */ true,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

ساكن ggml_et_cpu_compare_config concat_cpu_compare_config = {
    /* .enabled = */ خطأ,
    /* .use_cpu_result = */ خطأ,
    /* .log_differences = */ true,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

ساكن ggml_et_cpu_compare_config cumsum_cpu_compare_config = {
    /* .enabled = */ خطأ,
    /* .use_cpu_result = */ خطأ,
    /* .log_differences = */ true,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

ساكن ggml_et_cpu_compare_config repeat_cpu_compare_config = {
    /* .enabled = */ خطأ,
    /* .use_cpu_result = */ خطأ,
    /* .log_differences = */ true,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

ساكن ggml_et_cpu_compare_config ssm_conv_cpu_compare_config = {
    /* .enabled = */ خطأ,
    /* .use_cpu_result = */ خطأ,
    /* .log_differences = */ true,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

ساكن ggml_et_cpu_compare_config rwkv_wkv6_cpu_compare_config = {
    /* .enabled = */ خطأ,
    /* .use_cpu_result = */ خطأ,
    /* .log_differences = */ true,
    /* .tolerance = */ 1e-4f,
    /* .max_log_elements = */ 4096
};

ساكن ggml_et_cpu_compare_config rwkv_wkv7_cpu_compare_config = {
    /* .enabled = */ خطأ,
    /* .use_cpu_result = */ خطأ,
    /* .log_differences = */ true,
    /* .tolerance = */ 1e-4f,
    /* .max_log_elements = */ 4096
};

ساكن ggml_et_cpu_compare_config set_rows_cpu_compare_config = {
    /* .enabled = */ خطأ,
    /* .use_cpu_result = */ خطأ,
    /* .log_differences = */ true,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 2048
};

منطقي ggml_et_op_rms_norm_mul(ggml_backend_et_device_context * dev_ctx,
                             ثابت ggml_tensor *              rms_norm_كلي,
                             ثابت ggml_tensor *              mul_كلي) {
    ET_PERF_START();

    إذا (!dev_ctx || !rms_norm_كلي || !mul_كلي) {
        GGML_LOG_ERROR("ET: Invalid parameters for fused RMS_NORM_MUL operation\n");
        أرجع خطأ;
    }

    إذا (!rms_norm_كلي->src[0]) {
        GGML_LOG_ERROR("ET: Fused RMS_NORM_MUL missing required input\n");
        أرجع خطأ;
    }

    // Extract weights: the MUL operand that isn't the rms_norm output
    ثابت ggml_tensor * weights = (mul_كلي->src[0] == rms_norm_كلي) ? mul_كلي->src[1] : mul_كلي->src[0];

    إذا (!weights) {
        GGML_LOG_ERROR("ET: Fused RMS_NORM_MUL missing weights tensor\n");
        أرجع خطأ;
    }

    عائم eps;
    memcpy(&eps, rms_norm_كلي->op_params, حجم(عائم));

    ggml_et_rms_norm_mul_params params;
    params.src0 = *rms_norm_كلي->src[0];  // input to normalize
    params.src1 = *weights;                // normalization weights
    params.dst  = *mul_كلي;               // final output
    params.eps  = eps;

    منطقي kernel_result = ggml_et_launch_kernel(dev_ctx, "rms_norm_mul_f32", &params, حجم(params), 0xFFFFFFFF);

    ET_PERF_END_EXT("RMS_NORM_MUL", "rms_norm_mul_f32", mul_كلي, "eps=%.6f", (مزدوج) eps);
    أرجع kernel_result;
}

منطقي ggml_et_op_scale(ggml_backend_et_device_context * dev_ctx, ثابت ggml_tensor * كلي) {
    ET_PERF_START();

    إذا (!dev_ctx || !كلي) {
        GGML_LOG_ERROR("ET: Invalid parameters for SCALE operation\n");
        أرجع خطأ;
    }

    إذا (!كلي->src[0]) {
        GGML_LOG_ERROR("ET: SCALE operation missing required input\n");
        أرجع خطأ;
    }

    إذا (كلي->type != GGML_TYPE_F32 || كلي->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: SCALE operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(كلي->type),
                       ggml_type_name(كلي->src[0]->type));
        أرجع خطأ;
    }

    عائم scale, bias;
    memcpy(&scale, (ثابت عائم *) كلي->op_params + 0, حجم(عائم));
    memcpy(&bias, (ثابت عائم *) كلي->op_params + 1, حجم(عائم));

    ggml_et_scale_params params;
    params.src0  = *كلي->src[0];
    params.dst   = *كلي;
    params.scale = scale;
    params.bias  = bias;

    منطقي kernel_result = ggml_et_launch_kernel(dev_ctx, "scale_f32", &params, حجم(params), 0xFFFFFFFF);

    ET_PERF_END_EXT("SCALE", "scale_f32", كلي, "scale=%.6f|bias=%.6f", (مزدوج) scale, (مزدوج) bias);
    أرجع kernel_result;
}

منطقي ggml_et_op_sqr(ggml_backend_et_device_context * dev_ctx, ثابت ggml_tensor * كلي) {
    ET_PERF_START();

    إذا (!dev_ctx || !كلي) {
        GGML_LOG_ERROR("ET: Invalid parameters for SQR operation\n");
        أرجع خطأ;
    }

    إذا (!كلي->src[0]) {
        GGML_LOG_ERROR("ET: SQR operation missing required input\n");
        أرجع خطأ;
    }

    إذا (كلي->type != GGML_TYPE_F32 || كلي->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: SQR operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(كلي->type),
                       ggml_type_name(كلي->src[0]->type));
        أرجع خطأ;
    }

    ggml_et_sqr_params params;
    params.src0 = *كلي->src[0];  // F32 input tensor
    params.dst  = *كلي;          // F32 output tensor

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    منطقي                    cpu_comparison_active = خطأ;
    إذا (sqr_cpu_compare_config.enabled) {
        إذا (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, كلي, GGML_OP_SQR)) {
            cpu_comparison_active = true;
        } وإلا {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for SQR operation\n");
        }
    }

    منطقي kernel_result = ggml_et_launch_kernel(dev_ctx, "sqr_f32", &params, حجم(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    إذا (cpu_comparison_active) {
        إذا (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, كلي, &sqr_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for SQR operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("SQR", "sqr_f32", كلي);
    أرجع kernel_result;
}

منطقي ggml_et_op_sum_rows(ggml_backend_et_device_context * dev_ctx, ثابت ggml_tensor * كلي) {
    ET_PERF_START();

    إذا (!dev_ctx || !كلي) {
        GGML_LOG_ERROR("ET: Invalid parameters for SUM_ROWS operation\n");
        أرجع خطأ;
    }

    إذا (!كلي->src[0]) {
        GGML_LOG_ERROR("ET: SUM_ROWS operation missing required input\n");
        أرجع خطأ;
    }

    إذا (كلي->type != GGML_TYPE_F32 || كلي->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: SUM_ROWS operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(كلي->type),
                       ggml_type_name(كلي->src[0]->type));
        أرجع خطأ;
    }

    ggml_et_sum_rows_params params;
    params.src0 = *كلي->src[0];
    params.dst  = *كلي;

    // Phase 1: Initialize CPU comparison context
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    منطقي                    cpu_comparison_active = خطأ;
    إذا (sum_rows_cpu_compare_config.enabled) {
        إذا (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, كلي, GGML_OP_SUM_ROWS)) {
            cpu_comparison_active = true;
        } وإلا {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for SUM_ROWS operation\n");
        }
    }

    منطقي kernel_result = ggml_et_launch_kernel(dev_ctx, "sum_rows_f32", &params, حجم(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare
    إذا (cpu_comparison_active) {
        إذا (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, كلي, &sum_rows_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for SUM_ROWS operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("SUM_ROWS", "sum_rows_f32", كلي);
    أرجع kernel_result;
}

منطقي ggml_et_op_mean(ggml_backend_et_device_context * dev_ctx, ثابت ggml_tensor * كلي) {
    ET_PERF_START();

    إذا (!dev_ctx || !كلي) {
        GGML_LOG_ERROR("ET: Invalid parameters for MEAN operation\n");
        أرجع خطأ;
    }

    إذا (!كلي->src[0]) {
        GGML_LOG_ERROR("ET: MEAN operation missing required input\n");
        أرجع خطأ;
    }

    إذا (كلي->type != GGML_TYPE_F32 || كلي->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: MEAN operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(كلي->type),
                       ggml_type_name(كلي->src[0]->type));
        أرجع خطأ;
    }

    ggml_et_mean_params params;
    params.src0 = *كلي->src[0];
    params.dst  = *كلي;

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    منطقي                    cpu_comparison_active = خطأ;
    إذا (mean_cpu_compare_config.enabled) {
        إذا (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, كلي, GGML_OP_MEAN)) {
            cpu_comparison_active = true;
        } وإلا {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for MEAN operation\n");
        }
    }

    منطقي kernel_result = ggml_et_launch_kernel(dev_ctx, "mean_f32", &params, حجم(params), 0xFFFFFFFF);

    إذا (cpu_comparison_active) {
        إذا (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, كلي, &mean_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for MEAN operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("MEAN", "mean_f32", كلي);
    أرجع kernel_result;
}

منطقي ggml_et_op_clamp(ggml_backend_et_device_context * dev_ctx, ثابت ggml_tensor * كلي) {
    ET_PERF_START();

    إذا (!dev_ctx || !كلي) {
        GGML_LOG_ERROR("ET: Invalid parameters for CLAMP operation\n");
        أرجع خطأ;
    }

    إذا (!كلي->src[0]) {
        GGML_LOG_ERROR("ET: CLAMP operation missing required input\n");
        أرجع خطأ;
    }

    إذا (كلي->type != GGML_TYPE_F32 || كلي->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: CLAMP operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(كلي->type),
                       ggml_type_name(كلي->src[0]->type));
        أرجع خطأ;
    }

    ggml_et_clamp_params params;
    params.src0 = *كلي->src[0];
    params.dst  = *كلي;
    // op_params layout per ggml.c::ggml_clamp: { min, max } as floats
    memcpy(&params.min_val, (ثابت عائم *) كلي->op_params + 0, حجم(عائم));
    memcpy(&params.max_val, (ثابت عائم *) كلي->op_params + 1, حجم(عائم));

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    منطقي                    cpu_comparison_active = خطأ;
    إذا (clamp_cpu_compare_config.enabled) {
        إذا (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, كلي, GGML_OP_CLAMP)) {
            cpu_comparison_active = true;
        } وإلا {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for CLAMP operation\n");
        }
    }

    منطقي kernel_result = ggml_et_launch_kernel(dev_ctx, "clamp_f32", &params, حجم(params), 0xFFFFFFFF);

    إذا (cpu_comparison_active) {
        إذا (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, كلي, &clamp_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for CLAMP operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("CLAMP", "clamp_f32", كلي);
    أرجع kernel_result;
}

منطقي ggml_et_op_unary(ggml_backend_et_device_context * dev_ctx, ثابت ggml_tensor * كلي) {
    ET_PERF_START();

    إذا (!dev_ctx || !كلي) {
        GGML_LOG_ERROR("ET: Invalid parameters for UNARY operation\n");
        أرجع خطأ;
    }

    إذا (!كلي->src[0]) {
        GGML_LOG_ERROR("ET: UNARY operation missing required input\n");
        أرجع خطأ;
    }

    إذا (كلي->type != GGML_TYPE_F32 || كلي->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: UNARY operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(كلي->type),
                       ggml_type_name(كلي->src[0]->type));
        أرجع خطأ;
    }

    ثابت ggml_unary_op uop     = ggml_get_unary_op(كلي);
    ثابت حرف *        op_name = ggml_unary_op_name(uop);

    ggml_et_unary_params params;
    params.src0     = *كلي->src[0];  // F32 input tensor
    params.dst      = *كلي;          // F32 output tensor
    params.unary_op = (int32_t) uop;

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    منطقي                    cpu_comparison_active = خطأ;
    إذا (unary_cpu_compare_config.enabled) {
        إذا (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, كلي, GGML_OP_UNARY)) {
            cpu_comparison_active = true;
        } وإلا {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for UNARY/%s operation\n", op_name);
        }
    }

    منطقي kernel_result = ggml_et_launch_kernel(dev_ctx, "unary_f32", &params, حجم(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    إذا (cpu_comparison_active) {
        إذا (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, كلي, &unary_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for UNARY/%s operation\n", op_name);
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END_EXT("UNARY", "unary_f32", كلي, "op=%s", op_name);
    أرجع kernel_result;
}

منطقي ggml_et_op_mul(ggml_backend_et_device_context * dev_ctx, ثابت ggml_tensor * كلي) {
    // Delegate to generic element map operation
    أرجع ggml_et_op_elmap(dev_ctx, كلي);
}

منطقي ggml_et_op_add(ggml_backend_et_device_context * dev_ctx, ثابت ggml_tensor * كلي) {
    // Delegate to generic element map operation
    أرجع ggml_et_op_elmap(dev_ctx, كلي);
}

منطقي ggml_et_op_sub(ggml_backend_et_device_context * dev_ctx, ثابت ggml_tensor * كلي) {
    // Delegate to generic element map operation
    أرجع ggml_et_op_elmap(dev_ctx, كلي);
}

منطقي ggml_et_op_elmap(ggml_backend_et_device_context * dev_ctx, ثابت ggml_tensor * كلي) {
    ET_PERF_START();

    إذا (!dev_ctx || !كلي) {
        GGML_LOG_ERROR("ET: Invalid parameters for element map operation\n");
        أرجع خطأ;
    }

    إذا (!كلي->src[0] || !كلي->src[1]) {
        GGML_LOG_ERROR("ET: Element map operation missing required inputs\n");
        أرجع خطأ;
    }

    إذا (كلي->type != GGML_TYPE_F32 || كلي->src[0]->type != GGML_TYPE_F32 || كلي->src[1]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: Element map operation with unsupported types: dst=%s src0=%s src1=%s\n",
                       ggml_type_name(كلي->type), ggml_type_name(كلي->src[0]->type),
                       ggml_type_name(كلي->src[1]->type));
        أرجع خطأ;
    }

    ثابت حرف * op_name = ggml_op_name(كلي->op);

    ggml_et_elmap_params params;
    params.src0 = *كلي->src[0];
    params.src1 = *كلي->src[1];
    params.dst  = *كلي;  // F32 output tensor (op type stored in dst.op)

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    منطقي                    cpu_comparison_active = خطأ;
    إذا (elmap_cpu_compare_config.enabled) {
        إذا (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, كلي, كلي->op)) {
            cpu_comparison_active = true;
        } وإلا {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for %s operation\n", op_name);
        }
    }

    // fprintf(stderr, "ET: el_map s0 [%ld, %ld, %ld, %ld] s1 [%ld, %ld, %ld, %ld]\n",
    //     node->src[0]->ne[0], node->src[0]->ne[1], node->src[0]->ne[2], node->src[0]->ne[3],
    //     node->src[1]->ne[0], node->src[1]->ne[1], node->src[1]->ne[2], node->src[1]->ne[3]);

    منطقي kernel_result = ggml_et_launch_kernel(dev_ctx, "el_map_f32", &params, حجم(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    إذا (cpu_comparison_active) {
        إذا (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, كلي, &elmap_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for %s operation\n", op_name);
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END(op_name, "el_map_f32", كلي);
    أرجع kernel_result;
}

منطقي ggml_et_op_glu(ggml_backend_et_device_context * dev_ctx, ثابت ggml_tensor * كلي) {
    ET_PERF_START();

    // Validate inputs
    إذا (!dev_ctx || !كلي) {
        GGML_LOG_ERROR("ET: Invalid parameters for GLU operation\n");
        أرجع خطأ;
    }

    إذا (!كلي->src[0]) {
        GGML_LOG_ERROR("ET: GLU operation missing required input\n");
        أرجع خطأ;
    }

    ثابت منطقي is_split_mode = كلي->src[1] != nullptr;

    // Only support F32 (as validated by supports_op)
    إذا (كلي->type != GGML_TYPE_F32 || كلي->src[0]->type != GGML_TYPE_F32 ||
        (is_split_mode && كلي->src[1]->type != GGML_TYPE_F32)) {
        أرجع خطأ;
    }

    // Extract GLU operation parameters from op_params
    int32_t glu_op_type = ggml_get_op_params_i32(كلي, 0);  // GLU variant (REGLU, GEGLU, SWIGLU, etc.)
    int32_t swapped     = ggml_get_op_params_i32(كلي, 1);  // Whether gate/value are swapped

    // Supported variants
    بدل (glu_op_type) {
        حالة GGML_GLU_OP_REGLU:
        حالة GGML_GLU_OP_GEGLU:
        حالة GGML_GLU_OP_SWIGLU:
        حالة GGML_GLU_OP_SWIGLU_OAI:
        حالة GGML_GLU_OP_SWIGLU_CLAMP:
        حالة GGML_GLU_OP_GEGLU_ERF:
        حالة GGML_GLU_OP_GEGLU_QUICK:
            اكسر;
        افتراضي:
            GGML_LOG_ERROR("ET: GLU operation with unsupported variant: %s\n",
                           ggml_glu_op_name((ggml_glu_op) glu_op_type));
            أرجع خطأ;
    }

    // Get GLU operation name for logging
    ثابت حرف * glu_op_name = ggml_glu_op_name((ggml_glu_op) glu_op_type);

    // Pack parameters. Single-tensor mode is encoded by zeroing src1.
    ggml_et_glu_params params = {};
    params.src0               = *كلي->src[0];
    إذا (is_split_mode) {
        params.src1 = *كلي->src[1];
    }
    params.dst         = *كلي;
    params.glu_op_type = glu_op_type;
    params.swapped     = swapped;
    params.alpha       = 0.0f;
    params.limit       = 0.0f;
    إذا (glu_op_type == GGML_GLU_OP_SWIGLU_OAI) {
        params.alpha = ggml_get_op_params_f32(كلي, 2);
    }
    إذا (glu_op_type == GGML_GLU_OP_SWIGLU_OAI || glu_op_type == GGML_GLU_OP_SWIGLU_CLAMP) {
        params.limit = ggml_get_op_params_f32(كلي, 3);
    }
    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    منطقي                    cpu_comparison_active = خطأ;
    إذا (glu_cpu_compare_config.enabled) {
        إذا (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, كلي, GGML_OP_GLU)) {
            cpu_comparison_active = true;
        } وإلا {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for %s operation\n", glu_op_name);
        }
    }

    // Launch ET kernel
    منطقي kernel_result = ggml_et_launch_kernel(dev_ctx, "glu_f32", &params, حجم(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    إذا (cpu_comparison_active) {
        إذا (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, كلي, &glu_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for %s operation\n", glu_op_name);
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("GLU", "glu_f32", كلي);
    أرجع kernel_result;
}

منطقي ggml_et_op_mul_mat(ggml_backend_et_device_context * dev_ctx,
                        ثابت ggml_tensor *              كلي,
                        ثابت ggml_tensor *              add_كلي) {
    ET_PERF_START();

    إذا (!dev_ctx || !كلي) {
        GGML_LOG_ERROR("ET: Invalid parameters for MUL_MAT operation\n");
        أرجع خطأ;
    }

    إذا (!كلي->src[0] || !كلي->src[1]) {
        GGML_LOG_ERROR("ET: MUL_MAT operation missing required inputs\n");
        أرجع خطأ;
    }

    // Fused MM+ADD: when add_node is non-NULL the caller has already validated
    // (Q8_0 weights, F32 acts, exact-shape ADD with stride parity to dst) via
    // ggml_et_can_fuse({MUL_MAT, ADD}). The kernel writes dst = mm + bias and
    // the ADD's output replaces MM's as the actual dst.
    ثابت ggml_tensor * fused_dst   = add_كلي ? add_كلي : كلي;
    ثابت ggml_tensor * bias_tensor = nullptr;
    إذا (add_كلي) {
        bias_tensor = (add_كلي->src[0] == كلي) ? add_كلي->src[1] : add_كلي->src[0];
    }

    ثابت حرف * kernel_name;
    ثابت حرف * src0_type_name;

    إذا (كلي->type == GGML_TYPE_F32 && كلي->src[0]->type == GGML_TYPE_Q4_0 && كلي->src[1]->type == GGML_TYPE_F32 &&
        كلي->src[1]->ne[1] >= 53 &&      // N >= 53
        كلي->src[0]->ne[1] % 16 == 0 &&  // M % TILE_M
        كلي->src[0]->ne[0] % 32 == 0) {  // K % BLOCK_K (Q4_0 block)

        // Matrix engine for N >= 53; partial N (via n_cur-1) and errata padding are handled in-kernel.
        kernel_name    = "mul_mat_Q4_0_matrix_engine";
        src0_type_name = "Q4_0";

    } وإلا إذا (كلي->type == GGML_TYPE_F32 && كلي->src[0]->type == GGML_TYPE_Q4_0 &&
               كلي->src[1]->type == GGML_TYPE_F32) {
        kernel_name    = "mul_mat_Q4_0";  // N < 53, or M % 16 != 0 or K % 32 != 0
        src0_type_name = "Q4_0";

    } وإلا إذا (كلي->type == GGML_TYPE_F32 && كلي->src[0]->type == GGML_TYPE_Q8_0 &&
               كلي->src[1]->type == GGML_TYPE_F32) {
        kernel_name    = "mul_mat_Q8_0";
        src0_type_name = "Q8_0";

    } وإلا إذا (كلي->type == GGML_TYPE_F32 && كلي->src[0]->type == GGML_TYPE_F16 &&
               كلي->src[1]->type == GGML_TYPE_F16 && كلي->ne[0] % 16 == 0 && كلي->src[0]->ne[0] % 16 == 0 &&
               كلي->src[0]->ne[1] % 16 == 0 && كلي->src[1]->ne[0] != 1) {
        kernel_name    = "mul_mat_f16_matrix_engine";
        src0_type_name = "F16";

    } وإلا إذا (كلي->type == GGML_TYPE_F32 && كلي->src[0]->type == GGML_TYPE_F16 &&
               (كلي->src[1]->type == GGML_TYPE_F16 || كلي->src[1]->type == GGML_TYPE_F32)) {
        kernel_name    = "mul_mat_f16";
        src0_type_name = "F16";

    } وإلا إذا (كلي->type == GGML_TYPE_F32 && كلي->src[0]->type == GGML_TYPE_F32 &&
               كلي->src[1]->type == GGML_TYPE_F32 && كلي->ne[0] % 16 == 0 && كلي->src[0]->ne[0] % 16 == 0 &&
               كلي->src[0]->ne[1] % 16 == 0 && كلي->src[1]->ne[0] != 1) {  // GEMV is faster with the generic path

        kernel_name    = "mul_mat_f32_matrix_engine";
        src0_type_name = "F32";
    } وإلا إذا (كلي->type == GGML_TYPE_F32 && كلي->src[0]->type == GGML_TYPE_F32 &&
               (كلي->src[1]->type == GGML_TYPE_F16 || كلي->src[1]->type == GGML_TYPE_F32)) {
        kernel_name    = "mul_mat_f32";
        src0_type_name = "F32";
    } وإلا {
        GGML_LOG_ERROR("ET: MUL_MAT operation with unsupported types: dst=%s src0=%s src1=%s\n",
                       ggml_type_name(كلي->type), ggml_type_name(كلي->src[0]->type),
                       ggml_type_name(كلي->src[1]->type));
        أرجع خطأ;
    }

    ggml_et_نبيو_params params;
    params.src0 = *كلي->src[0];  // weight matrix
    params.src1 = *كلي->src[1];  // activation matrix
    params.dst  = *fused_dst;     // output (= add_node when fused, else node)

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    منطقي                    cpu_comparison_active = خطأ;
    إذا (mul_mat_cpu_compare_config.enabled) {
        إذا (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, fused_dst, GGML_OP_MUL_MAT)) {
            cpu_comparison_active = true;
        } وإلا {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for MUL_MAT operation\n");
        }
    }

    منطقي kernel_result;
    إذا (كلي->src[0]->type == GGML_TYPE_Q8_0) {
        // Q8_0 kernel always takes the extended struct. bias.data is non-NULL
        // only on the fused path; otherwise the kernel skips the add entirely.
        ggml_et_mm_q8_params q8_params = {};
        q8_params.src0                 = params.src0;
        q8_params.src1                 = params.src1;
        q8_params.dst                  = params.dst;
        إذا (bias_tensor) {
            q8_params.bias = *bias_tensor;
        }
        kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &q8_params, حجم(q8_params), 0xFFFFFFFF);
    } وإلا {
        // Non-Q8 MM kernels don't yet support fused-add; the graph fuse check
        // already rejects non-Q8 pairs, so add_node is always nullptr here.
        kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, حجم(params), 0xFFFFFFFF);
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
    إذا (cpu_comparison_active) {
        إذا (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, fused_dst, &mul_mat_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for MUL_MAT operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    {
        // Calculate actual FLOPs including batch/sequence dimensions
        // dst shape: [M, N, ne2, ne3] where M=ne[1], N=ne[0]
        int64_t m   = كلي->ne[1];
        int64_t n   = كلي->ne[0];
        int64_t k   = كلي->src[0]->ne[0];
        int64_t ne2 = كلي->ne[2];
        int64_t ne3 = كلي->ne[3];

        // Total FLOPs = (batch_size) * M * N * (2*K - 1)
        // Each MxN matrix-matrix multiply does M*N*(2*K-1) FLOPs
        // Broadcasting is handled by repeating computation, so count actual operations
        int64_t batch_size  = ne2 * ne3;
        int64_t total_flops = batch_size * m * n * (2 * k - 1);

        حرف kernel_variant[64];
        snprintf(kernel_variant, حجم(kernel_variant), "%s_%sx%s", kernel_name, src0_type_name,
                 ggml_type_name(كلي->src[1]->type));
        ET_PERF_END_EXT("MUL_MAT", kernel_variant, كلي, "flops=%" PRId64, total_flops);
    }
    أرجع kernel_result;
}

منطقي ggml_et_op_mul_mat_id(ggml_backend_et_device_context * dev_ctx, ثابت ggml_tensor * كلي) {
    ET_PERF_START();
    إذا (!dev_ctx || !كلي) {
        GGML_LOG_ERROR("ET: Invalid parameters for MUL_MAT_ID operation\n");
        أرجع خطأ;
    }

    إذا (!كلي->src[0] || !كلي->src[1] || !كلي->src[2]) {
        GGML_LOG_ERROR("ET: MUL_MAT_ID operation missing required inputs\n");
        أرجع خطأ;
    }

    ثابت حرف * kernel_name;
    ثابت حرف * src0_type_name;

    // Support Q8_0/Q4_0/F16/F32 x F32 -> F32 matrix multiplication with expert selection
    إذا (كلي->type == GGML_TYPE_F32 && كلي->src[0]->type == GGML_TYPE_Q8_0 && كلي->src[1]->type == GGML_TYPE_F32 &&
        كلي->src[2]->type == GGML_TYPE_I32) {
        kernel_name    = "mul_mat_id_Q8_0";
        src0_type_name = "Q8_0";

    } وإلا إذا (كلي->type == GGML_TYPE_F32 && كلي->src[0]->type == GGML_TYPE_Q4_0 &&
               كلي->src[1]->type == GGML_TYPE_F32 && كلي->src[2]->type == GGML_TYPE_I32) {
        kernel_name    = "mul_mat_id_Q4_0";
        src0_type_name = "Q4_0";

    } وإلا إذا (كلي->type == GGML_TYPE_F32 && كلي->src[0]->type == GGML_TYPE_F16 &&
               كلي->src[1]->type == GGML_TYPE_F32 && كلي->src[2]->type == GGML_TYPE_I32) {
        kernel_name    = "mul_mat_id_f32";
        src0_type_name = "F16";

    } وإلا إذا (كلي->type == GGML_TYPE_F32 && كلي->src[0]->type == GGML_TYPE_F32 &&
               كلي->src[1]->type == GGML_TYPE_F32 && كلي->src[2]->type == GGML_TYPE_I32) {
        kernel_name    = "mul_mat_id_f32";
        src0_type_name = "F32";

    } وإلا {
        GGML_LOG_ERROR("ET: MUL_MAT_ID operation with unsupported types: dst=%s src0=%s src1=%s src2=%s\n",
                       ggml_type_name(كلي->type), ggml_type_name(كلي->src[0]->type),
                       ggml_type_name(كلي->src[1]->type), ggml_type_name(كلي->src[2]->type));
        أرجع خطأ;
    }

    // Pack parameters - copy full tensor structures
    ggml_et_mul_mat_id_params params;
    params.src0 = *كلي->src[0];  // Expert weight matrices (Q8_0/F16/F32)
    params.src1 = *كلي->src[1];  // Activation matrix (F32)
    params.src2 = *كلي->src[2];  // Expert indices (I32)
    params.dst  = *كلي;          // Output matrix (F32)

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    منطقي                    cpu_comparison_active = خطأ;
    إذا (mul_mat_id_cpu_compare_config.enabled) {
        إذا (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, كلي, GGML_OP_MUL_MAT_ID)) {
            cpu_comparison_active = true;
        } وإلا {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for MUL_MAT_ID operation\n");
        }
    }

    // Launch ET kernel
    منطقي kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, حجم(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    إذا (cpu_comparison_active) {
        إذا (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, كلي, &mul_mat_id_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for MUL_MAT_ID operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    // Calculate FLOPs (approximate - similar to MUL_MAT but with expert routing overhead)
    // Each expert computation is similar to a MUL_MAT, but we only compute for selected experts
    int64_t K             = كلي->src[0]->ne[0];
    int64_t M             = كلي->src[0]->ne[1];
    int64_t n_expert_used = كلي->src[2]->ne[0];
    int64_t batch         = كلي->src[2]->ne[1];

    int64_t total_flops = batch * n_expert_used * M * (2 * K - 1);

    حرف kernel_variant[64];
    snprintf(kernel_variant, حجم(kernel_variant), "%s_%sx%s", kernel_name, src0_type_name,
             ggml_type_name(كلي->src[1]->type));
    ET_PERF_END_EXT("MUL_MAT_ID", kernel_variant, كلي, "flops=%" PRId64 "|n_expert=%lld|n_expert_used=%lld",
                    total_flops, (long long) كلي->src[0]->ne[2], (long long) n_expert_used);

    أرجع kernel_result;
}

منطقي ggml_et_op_rope(ggml_backend_et_device_context * dev_ctx, ثابت ggml_tensor * كلي) {
    ET_PERF_START();

    إذا (!dev_ctx || !كلي) {
        GGML_LOG_ERROR("ET: Invalid parameters for ROPE operation\n");
        أرجع خطأ;
    }

    إذا (!كلي->src[0] || !كلي->src[1]) {
        GGML_LOG_ERROR("ET: ROPE operation missing required inputs\n");
        أرجع خطأ;
    }

    ثابت حرف * kernel_name;

    إذا (كلي->type == GGML_TYPE_F32 && كلي->src[0]->type == GGML_TYPE_F32 && كلي->src[1]->type == GGML_TYPE_I32) {
        kernel_name = "rope_f32";
    } وإلا {
        أرجع خطأ;
    }

    // Pack parameters - copy full tensor structures and op_params
    ggml_et_rope_params params;
    params.src0 = *كلي->src[0];                       // F32 input tensor
    params.src1 = *كلي->src[1];                       // I32 position tensor
    إذا (كلي->src[2]) {
        params.src2 = *كلي->src[2];                   // F32 frequency factors (optional)
    } وإلا {
        memset(&params.src2, 0, حجم(params.src2));  // Zero if not provided
    }
    params.dst = *كلي;                                // F32 output tensor

    params.rope_params.n_past     = ((ثابت int32_t *) كلي->op_params)[0];
    params.rope_params.n_dims     = ((ثابت int32_t *) كلي->op_params)[1];
    params.rope_params.mode       = ((ثابت int32_t *) كلي->op_params)[2];
    params.rope_params.n_ctx      = ((ثابت int32_t *) كلي->op_params)[3];
    params.rope_params.n_ctx_orig = ((ثابت int32_t *) كلي->op_params)[4];
    memcpy(&params.rope_params.freq_base, (ثابت int32_t *) كلي->op_params + 5, حجم(عائم));
    memcpy(&params.rope_params.freq_scale, (ثابت int32_t *) كلي->op_params + 6, حجم(عائم));
    memcpy(&params.rope_params.ext_factor, (ثابت int32_t *) كلي->op_params + 7, حجم(عائم));
    memcpy(&params.rope_params.attn_factor, (ثابت int32_t *) كلي->op_params + 8, حجم(عائم));
    memcpy(&params.rope_params.beta_fast, (ثابت int32_t *) كلي->op_params + 9, حجم(عائم));
    memcpy(&params.rope_params.beta_slow, (ثابت int32_t *) كلي->op_params + 10, حجم(عائم));
    إذا (params.rope_params.mode & GGML_ROPE_TYPE_MROPE) {
        memcpy(params.rope_params.sections, (ثابت int32_t *) كلي->op_params + 11, حجم(int32_t) * 4);
    } وإلا {
        memset(params.rope_params.sections, 0, حجم(params.rope_params.sections));
    }

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    منطقي                    cpu_comparison_active = خطأ;
    إذا (rope_cpu_compare_config.enabled) {
        GGML_LOG_DEBUG("ET: Initializing CPU comparison for ROPE operation\n");
        إذا (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, كلي, GGML_OP_ROPE)) {
            cpu_comparison_active = true;
        } وإلا {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for ROPE operation\n");
        }
    }

    منطقي kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, حجم(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    إذا (cpu_comparison_active) {
        إذا (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, كلي, &rope_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for ROPE operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END_EXT("ROPE", kernel_name, كلي, "mode=0x%x|n_dims=%d|freq_base=%.2f|freq_scale=%.2f",
                    params.rope_params.mode, params.rope_params.n_dims, (مزدوج) params.rope_params.freq_base,
                    (مزدوج) params.rope_params.freq_scale);
    أرجع kernel_result;
}

منطقي ggml_et_op_rms_norm(ggml_backend_et_device_context * dev_ctx, ثابت ggml_tensor * كلي) {
    ET_PERF_START();

    إذا (!dev_ctx || !كلي) {
        GGML_LOG_ERROR("ET: Invalid parameters for RMS_NORM operation\n");
        أرجع خطأ;
    }

    إذا (!كلي->src[0]) {
        GGML_LOG_ERROR("ET: RMS_NORM operation missing required input\n");
        أرجع خطأ;
    }

    ثابت حرف * kernel_name;

    إذا (كلي->type == GGML_TYPE_F32 && كلي->src[0]->type == GGML_TYPE_F32) {
        kernel_name = "rms_norm_f32";

    } وإلا {
        GGML_LOG_ERROR("ET: RMS_NORM operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(كلي->type),
                       ggml_type_name(كلي->src[0]->type));
        أرجع خطأ;
    }

    عائم eps;
    memcpy(&eps, كلي->op_params, حجم(عائم));

    ggml_et_rms_norm_params params;
    params.src0 = *كلي->src[0];  // F32 input tensor
    params.dst  = *كلي;          // F32 output tensor
    params.eps  = eps;            // Epsilon parameter for numerical stability

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    منطقي                    cpu_comparison_active = خطأ;
    إذا (rms_norm_cpu_compare_config.enabled) {
        إذا (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, كلي, GGML_OP_RMS_NORM)) {
            cpu_comparison_active = true;
        } وإلا {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for RMS_NORM operation\n");
        }
    }

    منطقي kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, حجم(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    إذا (cpu_comparison_active) {
        إذا (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, كلي, &rms_norm_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for RMS_NORM operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END_EXT("RMS_NORM", kernel_name, كلي, "eps=%.6f", (مزدوج) eps);
    أرجع kernel_result;
}

منطقي ggml_et_op_norm(ggml_backend_et_device_context * dev_ctx, ثابت ggml_tensor * كلي) {
    ET_PERF_START();

    إذا (!dev_ctx || !كلي) {
        GGML_LOG_ERROR("ET: Invalid parameters for NORM operation\n");
        أرجع خطأ;
    }

    إذا (!كلي->src[0]) {
        GGML_LOG_ERROR("ET: NORM operation missing required input\n");
        أرجع خطأ;
    }

    ثابت حرف * kernel_name;

    إذا (كلي->type == GGML_TYPE_F32 && كلي->src[0]->type == GGML_TYPE_F32) {
        kernel_name = "norm_f32";

    } وإلا {
        GGML_LOG_ERROR("ET: NORM operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(كلي->type),
                       ggml_type_name(كلي->src[0]->type));
        أرجع خطأ;
    }

    عائم eps;
    memcpy(&eps, كلي->op_params, حجم(عائم));

    ggml_et_norm_params params;
    params.src0 = *كلي->src[0];  // F32 input tensor
    params.dst  = *كلي;          // F32 output tensor
    params.eps  = eps;            // Epsilon parameter for numerical stability

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    منطقي                    cpu_comparison_active = خطأ;
    إذا (norm_cpu_compare_config.enabled) {
        إذا (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, كلي, GGML_OP_NORM)) {
            cpu_comparison_active = true;
        } وإلا {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for NORM operation\n");
        }
    }

    منطقي kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, حجم(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    إذا (cpu_comparison_active) {
        إذا (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, كلي, &norm_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for NORM operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END_EXT("NORM", kernel_name, كلي, "eps=%.6f", (مزدوج) eps);
    أرجع kernel_result;
}

منطقي ggml_et_op_l2_norm(ggml_backend_et_device_context * dev_ctx, ثابت ggml_tensor * كلي) {
    ET_PERF_START();

    إذا (!dev_ctx || !كلي) {
        GGML_LOG_ERROR("ET: Invalid parameters for L2_NORM operation\n");
        أرجع خطأ;
    }

    إذا (!كلي->src[0]) {
        GGML_LOG_ERROR("ET: L2_NORM operation missing required input\n");
        أرجع خطأ;
    }

    ثابت حرف * kernel_name;

    إذا (كلي->type == GGML_TYPE_F32 && كلي->src[0]->type == GGML_TYPE_F32) {
        kernel_name = "l2_norm_f32";

    } وإلا {
        GGML_LOG_ERROR("ET: L2_NORM operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(كلي->type),
                       ggml_type_name(كلي->src[0]->type));
        أرجع خطأ;
    }

    عائم eps;
    memcpy(&eps, كلي->op_params, حجم(عائم));

    ggml_et_l2_norm_params params;
    params.src0 = *كلي->src[0];  // F32 input tensor
    params.dst  = *كلي;          // F32 output tensor
    params.eps  = eps;            // Epsilon parameter for numerical stability

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    منطقي                    cpu_comparison_active = خطأ;
    إذا (l2_norm_cpu_compare_config.enabled) {
        إذا (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, كلي, GGML_OP_L2_NORM)) {
            cpu_comparison_active = true;
        } وإلا {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for L2_NORM operation\n");
        }
    }

    منطقي kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, حجم(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    إذا (cpu_comparison_active) {
        إذا (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, كلي, &l2_norm_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for L2_NORM operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END_EXT("L2_NORM", kernel_name, كلي, "eps=%.6f", (مزدوج) eps);
    أرجع kernel_result;
}

منطقي ggml_et_op_group_norm(ggml_backend_et_device_context * dev_ctx, ثابت ggml_tensor * كلي) {
    ET_PERF_START();

    إذا (!dev_ctx || !كلي) {
        GGML_LOG_ERROR("ET: Invalid parameters for GROUP_NORM operation\n");
        أرجع خطأ;
    }

    إذا (!كلي->src[0]) {
        GGML_LOG_ERROR("ET: GROUP_NORM operation missing required input\n");
        أرجع خطأ;
    }

    إذا (كلي->type != GGML_TYPE_F32 || كلي->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: GROUP_NORM operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(كلي->type),
                       ggml_type_name(كلي->src[0]->type));
        أرجع خطأ;
    }

    ثابت int32_t n_groups = ggml_get_op_params_i32(كلي, 0);
    عائم         eps;
    memcpy(&eps, (ثابت عائم *) كلي->op_params + 1, حجم(عائم));

    ggml_et_group_norm_params params;
    params.src0     = *كلي->src[0];
    params.dst      = *كلي;
    params.n_groups = n_groups;
    params.eps      = eps;

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    منطقي                    cpu_comparison_active = خطأ;
    إذا (group_norm_cpu_compare_config.enabled) {
        إذا (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, كلي, GGML_OP_GROUP_NORM)) {
            cpu_comparison_active = true;
        } وإلا {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for GROUP_NORM operation\n");
        }
    }

    منطقي kernel_result = ggml_et_launch_kernel(dev_ctx, "group_norm_f32", &params, حجم(params), 0xFFFFFFFF);

    إذا (cpu_comparison_active) {
        إذا (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, كلي, &group_norm_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for GROUP_NORM operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END_EXT("GROUP_NORM", "group_norm_f32", كلي, "eps=%.6f|n_groups=%d", (مزدوج) eps, n_groups);
    أرجع kernel_result;
}

منطقي ggml_et_op_im2col(ggml_backend_et_device_context * dev_ctx, ثابت ggml_tensor * كلي) {
    ET_PERF_START();

    إذا (!dev_ctx || !كلي) {
        GGML_LOG_ERROR("ET: Invalid parameters for IM2COL operation\n");
        أرجع خطأ;
    }

    إذا (!كلي->src[0] || !كلي->src[1]) {
        GGML_LOG_ERROR("ET: IM2COL operation missing required inputs\n");
        أرجع خطأ;
    }

    ثابت منطقي supported_types =
        (كلي->type == GGML_TYPE_F32 && كلي->src[1]->type == GGML_TYPE_F32) ||
        (كلي->type == GGML_TYPE_F16 && (كلي->src[1]->type == GGML_TYPE_F16 || كلي->src[1]->type == GGML_TYPE_F32));

    إذا (!supported_types) {
        GGML_LOG_ERROR("ET: IM2COL operation with unsupported types: dst=%s src1=%s\n", ggml_type_name(كلي->type),
                       ggml_type_name(كلي->src[1]->type));
        أرجع خطأ;
    }

    ggml_et_im2col_params params;
    params.src0 = *كلي->src[0];
    params.src1 = *كلي->src[1];
    params.dst  = *كلي;

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    منطقي                    cpu_comparison_active = خطأ;
    إذا (im2col_cpu_compare_config.enabled) {
        إذا (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, كلي, GGML_OP_IM2COL)) {
            cpu_comparison_active = true;
        } وإلا {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for IM2COL operation\n");
        }
    }

    منطقي kernel_result = ggml_et_launch_kernel(dev_ctx, "im2col", &params, حجم(params), 0xFFFFFFFF);

    إذا (cpu_comparison_active) {
        إذا (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, كلي, &im2col_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for IM2COL operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("IM2COL", "im2col", كلي);
    أرجع kernel_result;
}

منطقي ggml_et_op_conv_2d(ggml_backend_et_device_context * dev_ctx, ثابت ggml_tensor * كلي) {
    ET_PERF_START();

    إذا (!dev_ctx || !كلي) {
        أرجع خطأ;
    }
    إذا (!كلي->src[0] || !كلي->src[1]) {
        أرجع خطأ;
    }
    إذا (!كلي->data || !كلي->src[0]->data || !كلي->src[1]->data) {
        أرجع خطأ;
    }

    // Kernel constraints (mirror supports_op; recheck here as a guard).
    ثابت ggml_tensor * flt = كلي->src[0];  // [Kw, Kh, Cin, Cout]
    ثابت ggml_tensor * in  = كلي->src[1];  // [W,  H,  Cin, N]
    إذا (كلي->type != GGML_TYPE_F32 || flt->type != GGML_TYPE_F32 || in->type != GGML_TYPE_F32) {
        أرجع خطأ;
    }

    ثابت int32_t s0 = ggml_get_op_params_i32(كلي, 0);
    ثابت int32_t s1 = ggml_get_op_params_i32(كلي, 1);
    ثابت int32_t p0 = ggml_get_op_params_i32(كلي, 2);
    ثابت int32_t p1 = ggml_get_op_params_i32(كلي, 3);
    ثابت int32_t d0 = ggml_get_op_params_i32(كلي, 4);
    ثابت int32_t d1 = ggml_get_op_params_i32(كلي, 5);

    إذا (s0 < 1 || s1 < 1) {
        أرجع خطأ;
    }
    إذا (d0 != 1 || d1 != 1) {
        أرجع خطأ;
    }
    إذا (flt->ne[2] % 16 != 0 || flt->ne[3] % 16 != 0) {
        أرجع خطأ;
    }
    إذا (in->ne[3] != 1) {
        أرجع خطأ;
    }
    إذا (كلي->ne[0] <= 0) {
        أرجع خطأ;  // OW > 0 (any width OK; staging path handles non-16)
    }
    (فراغ) p0;
    (فراغ) p1;

    ggml_et_نبيو_params params;
    params.src0 = *كلي->src[0];
    params.src1 = *كلي->src[1];
    params.dst  = *كلي;

    منطقي kernel_result = ggml_et_launch_kernel(dev_ctx, "conv_2d_f32_me", &params, حجم(params), 0xFFFFFFFFu);

    ET_PERF_END("CONV_2D", "conv_2d_f32_me", كلي);
    أرجع kernel_result;
}

منطقي ggml_et_op_softmax(ggml_backend_et_device_context * dev_ctx, ثابت ggml_tensor * كلي) {
    ET_PERF_START();

    إذا (!dev_ctx || !كلي) {
        GGML_LOG_ERROR("ET: Invalid parameters for SOFTMAX operation\n");
        أرجع خطأ;
    }

    إذا (!كلي->src[0]) {
        GGML_LOG_ERROR("ET: SOFTMAX operation missing required input\n");
        أرجع خطأ;
    }

    ثابت حرف * kernel_name;

    إذا (كلي->type == GGML_TYPE_F32 && كلي->src[0]->type == GGML_TYPE_F32) {
        kernel_name = "softmax_f32";

    } وإلا {
        GGML_LOG_ERROR("ET: SOFTMAX operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(كلي->type),
                       ggml_type_name(كلي->src[0]->type));
        أرجع خطأ;
    }

    // Validate contiguity requirements
    إذا (!ggml_is_contiguous(كلي)) {
        GGML_LOG_ERROR("ET: SOFTMAX operation requires contiguous destination tensor\n");
        أرجع خطأ;
    }

    إذا (!ggml_is_contiguous(كلي->src[0])) {
        GGML_LOG_ERROR("ET: SOFTMAX operation requires contiguous source tensor\n");
        أرجع خطأ;
    }

    // Check optional mask tensor
    إذا (كلي->src[1]) {
        إذا (كلي->src[1]->type != GGML_TYPE_F32) {
            GGML_LOG_ERROR("ET: SOFTMAX operation with unsupported mask type: %s (F32 required)\n",
                           ggml_type_name(كلي->src[1]->type));
            أرجع خطأ;
        }
        إذا (!ggml_is_contiguous(كلي->src[1])) {
            GGML_LOG_ERROR("ET: SOFTMAX operation requires contiguous mask tensor\n");
            أرجع خطأ;
        }
    }

    // Check optional sinks tensor
    إذا (كلي->src[2]) {
        إذا (كلي->src[2]->type != GGML_TYPE_F32) {
            GGML_LOG_ERROR("ET: SOFTMAX operation with unsupported sinks type: %s (F32 required)\n",
                           ggml_type_name(كلي->src[2]->type));
            أرجع خطأ;
        }
        إذا (!ggml_is_contiguous(كلي->src[2])) {
            GGML_LOG_ERROR("ET: SOFTMAX operation requires contiguous sinks tensor\n");
            أرجع خطأ;
        }
    }

    // Extract scale and max_bias from op_params
    عائم scale    = 1.0f;
    عائم max_bias = 0.0f;
    إذا (كلي->op_params) {
        memcpy(&scale, (ثابت عائم *) كلي->op_params + 0, حجم(عائم));
        memcpy(&max_bias, (ثابت عائم *) كلي->op_params + 1, حجم(عائم));
    }

    ggml_et_softmax_params params;
    params.src0 = *كلي->src[0];                       // F32 input tensor
    إذا (كلي->src[1]) {
        params.src1 = *كلي->src[1];                   // F32 mask tensor
    } وإلا {
        memset(&params.src1, 0, حجم(params.src1));  // Zero if no mask
    }
    إذا (كلي->src[2]) {
        params.src2 = *كلي->src[2];                   // F32 sinks tensor
    } وإلا {
        memset(&params.src2, 0, حجم(params.src2));  // Zero if no sinks
    }
    params.dst      = *كلي;                           // F32 output tensor
    params.scale    = scale;                           // Scale factor
    params.max_bias = max_bias;                        // ALiBi bias

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    منطقي                    cpu_comparison_active = خطأ;
    إذا (softmax_cpu_compare_config.enabled) {
        إذا (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, كلي, GGML_OP_SOFT_MAX)) {
            cpu_comparison_active = true;
        } وإلا {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for SOFTMAX operation\n");
        }
    }

    منطقي kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, حجم(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    إذا (cpu_comparison_active) {
        إذا (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, كلي, &softmax_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for SOFTMAX operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END_EXT("SOFTMAX", kernel_name, كلي, "scale=%.6f|max_bias=%.6f|has_mask=%s", (مزدوج) scale,
                    (مزدوج) max_bias, كلي->src[1] ? "yes" : "no");
    أرجع kernel_result;
}

منطقي ggml_et_op_flash_attn_ext(ggml_backend_et_device_context * dev_ctx, ثابت ggml_tensor * كلي) {
    ET_PERF_START();

    إذا (!dev_ctx || !كلي) {
        GGML_LOG_ERROR("ET: Invalid parameters for FLASH_ATTN_EXT operation\n");
        أرجع خطأ;
    }

    إذا (!كلي->src[0] || !كلي->src[1] || !كلي->src[2]) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT operation missing required inputs\n");
        أرجع خطأ;
    }

    إذا (كلي->type != GGML_TYPE_F32 || كلي->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT requires F32 Q and dst, got dst=%s q=%s\n", ggml_type_name(كلي->type),
                       ggml_type_name(كلي->src[0]->type));
        أرجع خطأ;
    }

    // K and V can be F16 or F32
    إذا ((كلي->src[1]->type != GGML_TYPE_F32 && كلي->src[1]->type != GGML_TYPE_F16) ||
        (كلي->src[2]->type != GGML_TYPE_F32 && كلي->src[2]->type != GGML_TYPE_F16)) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT K/V must be F16 or F32, got k=%s v=%s\n", ggml_type_name(كلي->src[1]->type),
                       ggml_type_name(كلي->src[2]->type));
        أرجع خطأ;
    }

    إذا (كلي->src[4] != nullptr) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT baseline kernel does not support sinks\n");
        أرجع خطأ;
    }

    // Mask is optional; if present must be F16 or F32
    إذا (كلي->src[3] != nullptr && كلي->src[3]->type != GGML_TYPE_F32 && كلي->src[3]->type != GGML_TYPE_F16) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT mask must be F16 or F32, got %s\n", ggml_type_name(كلي->src[3]->type));
        أرجع خطأ;
    }

    // Q and dst must be row-contiguous F32
    إذا (!ggml_is_contiguous_rows(كلي) || !ggml_is_contiguous_rows(كلي->src[0])) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT requires row-contiguous Q and dst\n");
        أرجع خطأ;
    }

    إذا (كلي->nb[0] != حجم(عائم) || كلي->src[0]->nb[0] != حجم(عائم)) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT requires contiguous F32 rows for Q and dst\n");
        أرجع خطأ;
    }

    // K/V must have element-sized stride in dim 0
    ثابت size_t k_elem = كلي->src[1]->type == GGML_TYPE_F16 ? 2 : 4;
    ثابت size_t v_elem = كلي->src[2]->type == GGML_TYPE_F16 ? 2 : 4;
    إذا (كلي->src[1]->nb[0] != k_elem || كلي->src[2]->nb[0] != v_elem) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT K/V must have element-sized stride in dim 0\n");
        أرجع خطأ;
    }

    عائم scale         = 1.0f;
    عائم max_bias      = 0.0f;
    عائم logit_softcap = 0.0f;
    memcpy(&scale, (ثابت عائم *) كلي->op_params + 0, حجم(scale));
    memcpy(&max_bias, (ثابت عائم *) كلي->op_params + 1, حجم(max_bias));
    memcpy(&logit_softcap, (ثابت عائم *) كلي->op_params + 2, حجم(logit_softcap));

    إذا (max_bias != 0.0f || logit_softcap != 0.0f) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT baseline kernel does not support max_bias or logit_softcap\n");
        أرجع خطأ;
    }

    ثابت ggml_prec prec = ggml_flash_attn_ext_get_prec(كلي);
    إذا (prec != GGML_PREC_F32 && prec != GGML_PREC_DEFAULT) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT baseline kernel only supports F32 precision\n");
        أرجع خطأ;
    }

    // dk must match between Q and K; dv must match between V and dst
    إذا (كلي->src[0]->ne[0] != كلي->src[1]->ne[0]) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT dk mismatch: Q=%lld K=%lld\n", (long long) كلي->src[0]->ne[0],
                       (long long) كلي->src[1]->ne[0]);
        أرجع خطأ;
    }

    إذا (كلي->src[2]->ne[0] != كلي->ne[0]) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT dv mismatch: V=%lld dst=%lld\n", (long long) كلي->src[2]->ne[0],
                       (long long) كلي->ne[0]);
        أرجع خطأ;
    }

    إذا (كلي->src[2]->ne[0] > 512) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT dv=%lld exceeds maximum 512\n", (long long) كلي->src[2]->ne[0]);
        أرجع خطأ;
    }

    إذا (كلي->src[0]->ne[0] > 512) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT dk=%lld exceeds maximum 512\n", (long long) كلي->src[0]->ne[0]);
        أرجع خطأ;
    }

    // GQA: n_head_q must be a multiple of n_head_kv
    ثابت int64_t nhq = كلي->src[0]->ne[2];
    ثابت int64_t nhk = كلي->src[1]->ne[2];
    إذا (nhq % nhk != 0) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT n_head_q (%lld) not divisible by n_head_kv (%lld)\n", (long long) nhq,
                       (long long) nhk);
        أرجع خطأ;
    }

    // K and V must have matching sequence length, heads, and batch dims
    إذا (كلي->src[1]->ne[1] != كلي->src[2]->ne[1] || كلي->src[1]->ne[2] != كلي->src[2]->ne[2] ||
        كلي->src[1]->ne[3] != كلي->src[2]->ne[3]) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT K/V shape mismatch\n");
        أرجع خطأ;
    }

    // dst layout checks: [dv, nhq, nq, no]
    إذا (كلي->src[0]->ne[1] != كلي->ne[2] || كلي->src[0]->ne[2] != كلي->ne[1] ||
        كلي->src[0]->ne[3] != كلي->ne[3]) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT dst shape mismatch\n");
        أرجع خطأ;
    }

    // Batch dims: Q batch must match K batch
    إذا (كلي->src[0]->ne[3] != كلي->src[1]->ne[3]) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT batch dimension mismatch\n");
        أرجع خطأ;
    }

    ggml_et_flash_attn_ext_params params;
    memset(&params, 0, حجم(params));
    params.src0 = *كلي->src[0];
    params.src1 = *كلي->src[1];
    params.src2 = *كلي->src[2];
    إذا (كلي->src[3] != nullptr) {
        params.mask     = *كلي->src[3];
        params.has_mask = 1;
    }
    params.dst   = *كلي;
    params.scale = scale;

    // Use matrix engine kernel when K/V are F16 and dk is a multiple of 32
    ثابت حرف * kernel_name;
    إذا (كلي->src[1]->type == GGML_TYPE_F16 && كلي->src[2]->type == GGML_TYPE_F16 && (كلي->src[0]->ne[0] % 32) == 0) {
        kernel_name = "flash_attn_ext_f16_me";
    } وإلا {
        kernel_name = "flash_attn_ext_f32";
    }

    ثابت منطقي kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, حجم(params), 0xFFFFFFFF);

    ET_PERF_END_EXT("FLASH_ATTN_EXT", kernel_name, كلي, "scale=%.6f", (مزدوج) scale);
    أرجع kernel_result;
}

منطقي ggml_et_op_get_rows(ggml_backend_et_device_context * dev_ctx, ثابت ggml_tensor * كلي) {
    ET_PERF_START();

    إذا (!dev_ctx || !كلي) {
        GGML_LOG_ERROR("ET: Invalid parameters for GET_ROWS operation\n");
        أرجع خطأ;
    }

    إذا (!كلي->src[0] || !كلي->src[1]) {
        GGML_LOG_ERROR("ET: GET_ROWS operation missing required inputs\n");
        أرجع خطأ;
    }

    ثابت حرف * kernel_name;

    إذا (كلي->type == GGML_TYPE_F32 && كلي->src[1]->type == GGML_TYPE_I32 &&
        (كلي->src[0]->type == GGML_TYPE_F32 || كلي->src[0]->type == GGML_TYPE_F16 ||
         كلي->src[0]->type == GGML_TYPE_Q4_0 || كلي->src[0]->type == GGML_TYPE_Q8_0 ||
         كلي->src[0]->type == GGML_TYPE_Q4_K)) {
        kernel_name = "get_rows_f32";

    } وإلا {
        GGML_LOG_ERROR("ET: GET_ROWS operation with unsupported types: dst=%s src0=%s src1=%s\n",
                       ggml_type_name(كلي->type), ggml_type_name(كلي->src[0]->type),
                       ggml_type_name(كلي->src[1]->type));
        أرجع خطأ;
    }

    // Validate contiguity requirements
    إذا (!ggml_is_contiguous(كلي)) {
        GGML_LOG_ERROR("ET: GET_ROWS operation requires contiguous destination tensor\n");
        أرجع خطأ;
    }

    إذا (!ggml_is_contiguous(كلي->src[0])) {
        GGML_LOG_ERROR("ET: GET_ROWS operation requires contiguous data tensor\n");
        أرجع خطأ;
    }

    إذا (!ggml_is_contiguous(كلي->src[1])) {
        GGML_LOG_ERROR("ET: GET_ROWS operation requires contiguous indices tensor\n");
        أرجع خطأ;
    }

    // Validate dimension constraints from ggml implementation
    إذا (كلي->src[0]->ne[2] != كلي->src[1]->ne[1] || كلي->src[1]->ne[3] != 1) {
        GGML_LOG_ERROR(
            "ET: GET_ROWS operation dimension constraint failed: src0.ne[2]=%lld != src1.ne[1]=%lld or src1.ne[3]=%lld "
            "!= 1\n",
            (long long) كلي->src[0]->ne[2], (long long) كلي->src[1]->ne[1], (long long) كلي->src[1]->ne[3]);
        أرجع خطأ;
    }

    ggml_et_get_rows_params params;
    params.src0 = *كلي->src[0];  // Data tensor (F32 or Q8_0)
    params.src1 = *كلي->src[1];  // Indices tensor (I32)
    params.dst  = *كلي;          // Output tensor (F32)

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    منطقي                    cpu_comparison_active = خطأ;
    إذا (get_rows_cpu_compare_config.enabled) {
        إذا (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, كلي, GGML_OP_GET_ROWS)) {
            cpu_comparison_active = true;
        } وإلا {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for GET_ROWS operation\n");
        }
    }

    منطقي kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, حجم(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    إذا (cpu_comparison_active) {
        إذا (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, كلي, &get_rows_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for GET_ROWS operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("GET_ROWS", kernel_name, كلي);
    أرجع kernel_result;
}

منطقي ggml_et_op_cont(ggml_backend_et_device_context * dev_ctx, ثابت ggml_tensor * كلي) {
    ET_PERF_START();

    // Validate source tensor exists
    إذا (!كلي->src[0]) {
        GGML_LOG_ERROR("ET: CONT operation missing source tensor\n");
        أرجع خطأ;
    }

    // Validate types match (input and output must be same type)
    إذا (كلي->type != كلي->src[0]->type) {
        GGML_LOG_ERROR("ET: CONT operation type mismatch: src=%s dst=%s\n", ggml_type_name(كلي->src[0]->type),
                       ggml_type_name(كلي->type));
        أرجع خطأ;
    }

    // Validate supported types
    إذا (كلي->type != GGML_TYPE_F32 && كلي->type != GGML_TYPE_F16) {
        GGML_LOG_ERROR("ET: CONT operation unsupported type: %s (only F32 and F16 supported)\n",
                       ggml_type_name(كلي->type));
        أرجع خطأ;
    }

    // Validate contiguity - output must be contiguous, input can be non-contiguous
    إذا (!ggml_is_contiguous(كلي)) {
        GGML_LOG_ERROR("ET: CONT operation requires contiguous output tensor\n");
        أرجع خطأ;
    }

    // Select kernel based on type
    ثابت حرف * kernel_name;
    إذا (كلي->type == GGML_TYPE_F32) {
        kernel_name = "cont_f32";
    } وإلا إذا (كلي->type == GGML_TYPE_F16) {
        kernel_name = "cont_f16";
    } وإلا {
        GGML_LOG_ERROR("ET: CONT operation with unsupported type: %s\n", ggml_type_name(كلي->type));
        أرجع خطأ;
    }

    ggml_et_cont_params params;
    params.src0 = *كلي->src[0];  // Input tensor (potentially non-contiguous)
    params.dst  = *كلي;          // Output tensor (contiguous)

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    منطقي                    cpu_comparison_active = خطأ;
    إذا (cont_cpu_compare_config.enabled) {
        إذا (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, كلي, GGML_OP_CONT)) {
            cpu_comparison_active = true;
        } وإلا {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for CONT operation\n");
        }
    }

    منطقي kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, حجم(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    إذا (cpu_comparison_active) {
        إذا (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, كلي, &cont_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for CONT operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("CONT", kernel_name, كلي);
    أرجع kernel_result;
}

منطقي ggml_et_op_cumsum(ggml_backend_et_device_context * dev_ctx, ثابت ggml_tensor * كلي) {
    ET_PERF_START();

    إذا (!dev_ctx || !كلي || !كلي->src[0]) {
        GGML_LOG_ERROR("ET: Invalid parameters for CUMSUM operation\n");
        أرجع خطأ;
    }

    إذا (كلي->type != GGML_TYPE_F32 || كلي->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: CUMSUM operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(كلي->type),
                       ggml_type_name(كلي->src[0]->type));
        أرجع خطأ;
    }

    ثابت حرف * kernel_name = "cumsum_f32";

    ggml_et_cumsum_params params;
    params.src0 = *كلي->src[0];
    params.dst  = *كلي;

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    منطقي                    cpu_comparison_active = خطأ;
    إذا (cumsum_cpu_compare_config.enabled) {
        إذا (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, كلي, GGML_OP_CUMSUM)) {
            cpu_comparison_active = true;
        } وإلا {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for CUMSUM operation\n");
        }
    }

    منطقي kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, حجم(params), 0xFFFFFFFF);

    إذا (cpu_comparison_active) {
        إذا (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, كلي, &cumsum_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for CUMSUM operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("CUMSUM", kernel_name, كلي);
    أرجع kernel_result;
}

منطقي ggml_et_op_cpy(ggml_backend_et_device_context * dev_ctx, ثابت ggml_tensor * كلي) {
    ET_PERF_START();

    // CPY copies data from src[0] into the layout of dst (which matches src[1])
    // For same-type with contiguous dst, this is identical to CONT
    إذا (!كلي->src[0]) {
        GGML_LOG_ERROR("ET: CPY operation missing source tensor\n");
        أرجع خطأ;
    }

    // Scalar / zero-element special path: if any dimension is 0, nothing to copy
    ثابت int64_t nelements = كلي->ne[0] * كلي->ne[1] * كلي->ne[2] * كلي->ne[3];
    إذا (nelements == 0) {
        GGML_LOG_DEBUG("ET: CPY no-op (zero elements): ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]\n",
                       كلي->ne[0], كلي->ne[1], كلي->ne[2], كلي->ne[3]);
        ET_PERF_END("CPY", "noop", كلي);
        أرجع true;
    }

    // Only F32 and F16 supported for dst
    إذا (كلي->type != GGML_TYPE_F32 && كلي->type != GGML_TYPE_F16) {
        GGML_LOG_ERROR("ET: CPY unsupported dst type: %s\n", ggml_type_name(كلي->type));
        أرجع خطأ;
    }

    // Select kernel based on src/dst type combination
    ثابت حرف * kernel_name;
    إذا (كلي->src[0]->type == GGML_TYPE_F32 && كلي->type == GGML_TYPE_F32) {
        kernel_name = "cont_f32";
    } وإلا إذا (كلي->src[0]->type == GGML_TYPE_F16 && كلي->type == GGML_TYPE_F16) {
        kernel_name = "cont_f16";
    } وإلا إذا (كلي->src[0]->type == GGML_TYPE_F32 && كلي->type == GGML_TYPE_F16) {
        kernel_name = "cpy_f32_f16";
    } وإلا {
        GGML_LOG_ERROR("ET: CPY unsupported type combination: src=%s dst=%s\n", ggml_type_name(كلي->src[0]->type),
                       ggml_type_name(كلي->type));
        أرجع خطأ;
    }

    ggml_et_cont_params params;
    params.src0 = *كلي->src[0];
    params.dst  = *كلي;

    // CPU comparison for debugging
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    منطقي                    cpu_comparison_active = خطأ;
    إذا (cont_cpu_compare_config.enabled) {
        إذا (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, كلي, GGML_OP_CPY)) {
            cpu_comparison_active = true;
        } وإلا {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for CPY operation\n");
        }
    }

    منطقي kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, حجم(params), 0xFFFFFFFF);

    إذا (cpu_comparison_active) {
        إذا (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, كلي, &cont_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for CPY operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("CPY", kernel_name, كلي);
    أرجع kernel_result;
}

منطقي ggml_et_op_concat(ggml_backend_et_device_context * dev_ctx, ثابت ggml_tensor * كلي) {
    ET_PERF_START();

    إذا (!dev_ctx || !كلي) {
        GGML_LOG_ERROR("ET: Invalid parameters for CONCAT operation\n");
        أرجع خطأ;
    }

    إذا (!كلي->src[0] || !كلي->src[1]) {
        GGML_LOG_ERROR("ET: CONCAT operation missing required inputs\n");
        أرجع خطأ;
    }

    ثابت حرف * kernel_name;

    إذا (كلي->type == GGML_TYPE_F32 && كلي->src[0]->type == GGML_TYPE_F32 && كلي->src[1]->type == GGML_TYPE_F32) {
        kernel_name = "concat_f32";

    } وإلا {
        GGML_LOG_ERROR("ET: CONCAT operation with unsupported types: dst=%s src0=%s src1=%s\n",
                       ggml_type_name(كلي->type), ggml_type_name(كلي->src[0]->type),
                       ggml_type_name(كلي->src[1]->type));
        أرجع خطأ;
    }

    int32_t dim;
    memcpy(&dim, كلي->op_params, حجم(int32_t));

    ggml_et_concat_params params;
    params.src0 = *كلي->src[0];
    params.src1 = *كلي->src[1];
    params.dst  = *كلي;
    params.dim  = dim;

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    منطقي                    cpu_comparison_active = خطأ;
    إذا (concat_cpu_compare_config.enabled) {
        إذا (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, كلي, GGML_OP_CONCAT)) {
            cpu_comparison_active = true;
        } وإلا {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for CONCAT operation\n");
        }
    }

    منطقي kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, حجم(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    إذا (cpu_comparison_active) {
        إذا (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, كلي, &concat_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for CONCAT operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END_EXT("CONCAT", kernel_name, كلي, "dim=%d", dim);
    أرجع kernel_result;
}

منطقي ggml_et_op_repeat(ggml_backend_et_device_context * dev_ctx, ثابت ggml_tensor * كلي) {
    ET_PERF_START();

    إذا (!dev_ctx || !كلي) {
        GGML_LOG_ERROR("ET: Invalid parameters for REPEAT operation\n");
        أرجع خطأ;
    }

    إذا (!كلي->src[0]) {
        GGML_LOG_ERROR("ET: REPEAT operation missing required input\n");
        أرجع خطأ;
    }

    ثابت حرف * kernel_name;

    إذا (كلي->type == GGML_TYPE_F32 && كلي->src[0]->type == GGML_TYPE_F32) {
        // No-op REPEAT (every repeat factor is 1): the output is just a copy
        // of the input. Route to cont_f32, whose contiguous fast path handles
        // arbitrary sizes (including those rejected by repeat_f32's gate,
        // e.g. ne[0]=1).
        إذا (ggml_are_same_shape(كلي->src[0], كلي)) {
            kernel_name = "cont_f32";
        } وإلا {
            kernel_name = "repeat_f32";
        }

    } وإلا {
        GGML_LOG_ERROR("ET: REPEAT operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(كلي->type),
                       ggml_type_name(كلي->src[0]->type));
        أرجع خطأ;
    }

    // ggml_et_cont_params and ggml_et_repeat_params have identical layouts
    // (just src0 + dst), so the same payload works for either kernel.
    ggml_et_repeat_params params;
    params.src0 = *كلي->src[0];
    params.dst  = *كلي;

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    منطقي                    cpu_comparison_active = خطأ;
    إذا (repeat_cpu_compare_config.enabled) {
        إذا (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, كلي, GGML_OP_REPEAT)) {
            cpu_comparison_active = true;
        } وإلا {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for REPEAT operation\n");
        }
    }

    منطقي kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, حجم(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    إذا (cpu_comparison_active) {
        إذا (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, كلي, &repeat_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for REPEAT operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("REPEAT", kernel_name, كلي);
    أرجع kernel_result;
}

منطقي ggml_et_op_ssm_conv(ggml_backend_et_device_context * dev_ctx, ثابت ggml_tensor * كلي) {
    ET_PERF_START();

    إذا (!dev_ctx || !كلي || !كلي->src[0] || !كلي->src[1]) {
        GGML_LOG_ERROR("ET: Invalid parameters for SSM_CONV operation\n");
        أرجع خطأ;
    }

    إذا (كلي->type != GGML_TYPE_F32 || كلي->src[0]->type != GGML_TYPE_F32 || كلي->src[1]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: SSM_CONV operation with unsupported types: dst=%s src0=%s src1=%s\n",
                       ggml_type_name(كلي->type), ggml_type_name(كلي->src[0]->type),
                       ggml_type_name(كلي->src[1]->type));
        أرجع خطأ;
    }

    ثابت حرف * kernel_name = "ssm_conv_f32";

    ggml_et_ssm_conv_params params;
    params.src0 = *كلي->src[0];
    params.src1 = *كلي->src[1];
    params.dst  = *كلي;

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    منطقي                    cpu_comparison_active = خطأ;
    إذا (ssm_conv_cpu_compare_config.enabled) {
        إذا (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, كلي, GGML_OP_SSM_CONV)) {
            cpu_comparison_active = true;
        } وإلا {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for SSM_CONV operation\n");
        }
    }

    منطقي kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, حجم(params), 0xFFFFFFFF);

    إذا (cpu_comparison_active) {
        إذا (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, كلي, &ssm_conv_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for SSM_CONV operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("SSM_CONV", kernel_name, كلي);
    أرجع kernel_result;
}

منطقي ggml_et_op_ssm_scan(ggml_backend_et_device_context * dev_ctx, ثابت ggml_tensor * كلي) {
    ET_PERF_START();

    إذا (!dev_ctx || !كلي) {
        GGML_LOG_ERROR("ET: Invalid parameters for SSM_SCAN operation\n");
        أرجع خطأ;
    }

    لكل (صحيح i = 0; i < 7; ++i) {
        إذا (!كلي->src[i]) {
            GGML_LOG_ERROR("ET: SSM_SCAN missing required input %d\n", i);
            أرجع خطأ;
        }
    }

    إذا (كلي->type != GGML_TYPE_F32 || كلي->src[0]->type != GGML_TYPE_F32 || كلي->src[1]->type != GGML_TYPE_F32 ||
        كلي->src[2]->type != GGML_TYPE_F32 || كلي->src[3]->type != GGML_TYPE_F32 ||
        كلي->src[4]->type != GGML_TYPE_F32 || كلي->src[5]->type != GGML_TYPE_F32 ||
        كلي->src[6]->type != GGML_TYPE_I32) {
        GGML_LOG_ERROR("ET: SSM_SCAN operation with unsupported types\n");
        أرجع خطأ;
    }

    ggml_et_ssm_scan_params params;
    params.src0 = *كلي->src[0];
    params.src1 = *كلي->src[1];
    params.src2 = *كلي->src[2];
    params.src3 = *كلي->src[3];
    params.src4 = *كلي->src[4];
    params.src5 = *كلي->src[5];
    params.src6 = *كلي->src[6];
    params.dst  = *كلي;
    params.K    = ggml_get_op_params_i32(كلي, 0);

    منطقي kernel_result = ggml_et_launch_kernel(dev_ctx, "ssm_scan_f32", &params, حجم(params), 0xFFFFFFFF);

    ET_PERF_END("SSM_SCAN", "ssm_scan_f32", كلي);
    أرجع kernel_result;
}

منطقي ggml_et_op_rwkv_wkv6(ggml_backend_et_device_context * dev_ctx, ثابت ggml_tensor * كلي) {
    ET_PERF_START();

    إذا (!dev_ctx || !كلي) {
        GGML_LOG_ERROR("ET: Invalid parameters for RWKV_WKV6 operation\n");
        أرجع خطأ;
    }

    // Validate all 6 source tensors exist
    لكل (صحيح i = 0; i <= 5; i++) {
        إذا (!كلي->src[i]) {
            GGML_LOG_ERROR("ET: RWKV_WKV6 operation missing src[%d]\n", i);
            أرجع خطأ;
        }
    }

    إذا (كلي->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: RWKV_WKV6 only supports F32, got %s\n", ggml_type_name(كلي->type));
        أرجع خطأ;
    }

    ثابت حرف * kernel_name = "rwkv_wkv6_f32";

    ثابت int64_t S      = كلي->src[0]->ne[0];  // head_size
    ثابت int64_t H      = كلي->src[0]->ne[1];  // num heads
    ثابت int64_t T      = كلي->src[1]->ne[2];  // num tokens
    ثابت int64_t n_seqs = كلي->src[5]->ne[1];  // num sequences
    ثابت int64_t C      = S * H;

    ggml_et_rwkv_wkv6_params params;
    params.k        = (عائم *) كلي->src[0]->data;
    params.v        = (عائم *) كلي->src[1]->data;
    params.r        = (عائم *) كلي->src[2]->data;
    params.tf       = (عائم *) كلي->src[3]->data;
    params.td       = (عائم *) كلي->src[4]->data;
    params.state_in = (عائم *) كلي->src[5]->data;
    params.dst      = (عائم *) كلي->data;
    params.C        = (int32_t) C;
    params.H        = (int32_t) H;
    params.S        = (int32_t) S;
    params.T        = (int32_t) T;
    params.n_seqs   = (int32_t) n_seqs;

    // Phase 1: Initialize CPU comparison context
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    منطقي                    cpu_comparison_active = خطأ;
    إذا (rwkv_wkv6_cpu_compare_config.enabled) {
        إذا (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, كلي, GGML_OP_RWKV_WKV6)) {
            cpu_comparison_active = true;
        } وإلا {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for RWKV_WKV6 operation\n");
        }
    }

    منطقي kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, حجم(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare
    إذا (cpu_comparison_active) {
        إذا (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, كلي, &rwkv_wkv6_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for RWKV_WKV6 operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END_EXT("RWKV_WKV6", kernel_name, كلي, "S=%d H=%d T=%d n_seqs=%d", (صحيح) S, (صحيح) H, (صحيح) T,
                    (صحيح) n_seqs);
    أرجع kernel_result;
}

منطقي ggml_et_op_rwkv_wkv7(ggml_backend_et_device_context * dev_ctx, ثابت ggml_tensor * كلي) {
    ET_PERF_START();

    إذا (!dev_ctx || !كلي) {
        GGML_LOG_ERROR("ET: Invalid parameters for RWKV_WKV7 operation\n");
        أرجع خطأ;
    }

    // Validate all 7 source tensors exist
    لكل (صحيح i = 0; i <= 6; i++) {
        إذا (!كلي->src[i]) {
            GGML_LOG_ERROR("ET: RWKV_WKV7 operation missing src[%d]\n", i);
            أرجع خطأ;
        }
    }

    إذا (كلي->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: RWKV_WKV7 only supports F32, got %s\n", ggml_type_name(كلي->type));
        أرجع خطأ;
    }

    ثابت حرف * kernel_name = "rwkv_wkv7_f32";

    ثابت int64_t S      = كلي->src[2]->ne[0];  // head_size
    ثابت int64_t H      = كلي->src[2]->ne[1];  // num heads
    ثابت int64_t T      = كلي->src[1]->ne[2];  // num tokens
    ثابت int64_t n_seqs = كلي->src[6]->ne[1];  // num sequences
    ثابت int64_t C      = S * H;

    ggml_et_rwkv_wkv7_params params;
    params.r        = (عائم *) كلي->src[0]->data;
    params.w        = (عائم *) كلي->src[1]->data;
    params.k        = (عائم *) كلي->src[2]->data;
    params.v        = (عائم *) كلي->src[3]->data;
    params.a        = (عائم *) كلي->src[4]->data;
    params.b        = (عائم *) كلي->src[5]->data;
    params.state_in = (عائم *) كلي->src[6]->data;
    params.dst      = (عائم *) كلي->data;
    params.C        = (int32_t) C;
    params.H        = (int32_t) H;
    params.S        = (int32_t) S;
    params.T        = (int32_t) T;
    params.n_seqs   = (int32_t) n_seqs;

    // Phase 1: Initialize CPU comparison context
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    منطقي                    cpu_comparison_active = خطأ;
    إذا (rwkv_wkv7_cpu_compare_config.enabled) {
        إذا (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, كلي, GGML_OP_RWKV_WKV7)) {
            cpu_comparison_active = true;
        } وإلا {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for RWKV_WKV7 operation\n");
        }
    }

    منطقي kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, حجم(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare
    إذا (cpu_comparison_active) {
        إذا (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, كلي, &rwkv_wkv7_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for RWKV_WKV7 operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END_EXT("RWKV_WKV7", kernel_name, كلي, "S=%d H=%d T=%d n_seqs=%d", (صحيح) S, (صحيح) H, (صحيح) T,
                    (صحيح) n_seqs);
    أرجع kernel_result;
}

ساكن ggml_et_cpu_compare_config gated_delta_net_cpu_compare_config = {
    /* .enabled = */ خطأ,
    /* .use_cpu_result = */ خطأ,
    /* .log_differences = */ true,
    /* .tolerance = */ 1e-4f,
    /* .max_log_elements = */ 4096
};

منطقي ggml_et_op_gated_delta_net(ggml_backend_et_device_context * dev_ctx, ثابت ggml_tensor * كلي) {
    ET_PERF_START();

    إذا (!dev_ctx || !كلي) {
        GGML_LOG_ERROR("ET: Invalid parameters for GATED_DELTA_NET operation\n");
        أرجع خطأ;
    }

    // Validate all 6 source tensors exist
    لكل (صحيح i = 0; i <= 5; i++) {
        إذا (!كلي->src[i]) {
            GGML_LOG_ERROR("ET: GATED_DELTA_NET operation missing src[%d]\n", i);
            أرجع خطأ;
        }
    }

    إذا (كلي->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: GATED_DELTA_NET only supports F32, got %s\n", ggml_type_name(كلي->type));
        أرجع خطأ;
    }

    ثابت حرف * kernel_name = "gated_delta_net_f32";

    ثابت ggml_tensor * src_q     = كلي->src[0];
    ثابت ggml_tensor * src_k     = كلي->src[1];
    ثابت ggml_tensor * src_v     = كلي->src[2];
    ثابت ggml_tensor * src_g     = كلي->src[3];
    ثابت ggml_tensor * src_beta  = كلي->src[4];
    ثابت ggml_tensor * src_state = كلي->src[5];

    ثابت int64_t S_v      = src_v->ne[0];
    ثابت int64_t H        = src_v->ne[1];
    ثابت int64_t n_tokens = src_v->ne[2];
    ثابت int64_t n_seqs   = src_v->ne[3];
    ثابت int64_t H_q      = src_q->ne[1];
    ثابت int64_t H_k      = src_k->ne[1];
    ثابت int64_t n_seqs_q = src_q->ne[3];
    ثابت int64_t n_seqs_k = src_k->ne[3];

    ggml_et_gated_delta_net_params params;
    params.q        = *src_q;
    params.k        = *src_k;
    params.v        = *src_v;
    params.g        = *src_g;
    params.beta     = *src_beta;
    params.state_in = *src_state;
    params.dst      = *كلي;
    params.S_v      = (int32_t) S_v;
    params.H        = (int32_t) H;
    params.H_q      = (int32_t) H_q;
    params.H_k      = (int32_t) H_k;
    params.n_tokens = (int32_t) n_tokens;
    params.n_seqs   = (int32_t) n_seqs;
    params.n_seqs_q = (int32_t) n_seqs_q;
    params.n_seqs_k = (int32_t) n_seqs_k;
    params.kda      = (src_g->ne[0] == S_v) ? 1 : 0;
    params.K        = ggml_get_op_params_i32(كلي, 0);
    params.scale    = 1.0f / sqrtf((عائم) S_v);

    // CPU comparison for debugging
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    منطقي                    cpu_comparison_active = خطأ;
    إذا (gated_delta_net_cpu_compare_config.enabled) {
        إذا (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, كلي, GGML_OP_GATED_DELTA_NET)) {
            cpu_comparison_active = true;
        } وإلا {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for GATED_DELTA_NET operation\n");
        }
    }

    منطقي kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, حجم(params), 0xFFFFFFFF);

    إذا (cpu_comparison_active) {
        إذا (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, كلي, &gated_delta_net_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for GATED_DELTA_NET operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END_EXT("GATED_DELTA_NET", kernel_name, كلي, "S_v=%d H=%d n_tokens=%d n_seqs=%d kda=%d", (صحيح) S_v,
                    (صحيح) H, (صحيح) n_tokens, (صحيح) n_seqs, params.kda);
    أرجع kernel_result;
}

منطقي ggml_et_op_set_rows(ggml_backend_et_device_context * dev_ctx, ثابت ggml_tensor * كلي) {
    ET_PERF_START();

    إذا (!dev_ctx || !كلي) {
        GGML_LOG_ERROR("ET: Invalid parameters for SET_ROWS operation\n");
        أرجع خطأ;
    }

    إذا (!كلي->src[0] || !كلي->src[1] || !كلي->src[2]) {
        GGML_LOG_ERROR(
            "ET: SET_ROWS operation missing required inputs (needs src[0]=base, src[1]=indices, src[2]=data)\n");
        أرجع خطأ;
    }

    ثابت حرف * kernel_name;

    // Support F32 data with I64 indices -> F32/F16 output (scatter operation)
    إذا (كلي->src[0]->type == GGML_TYPE_F32 && كلي->src[1]->type == GGML_TYPE_I64 &&
        (كلي->type == GGML_TYPE_F32 || كلي->type == GGML_TYPE_F16)) {
        إذا (كلي->type == GGML_TYPE_F32 || كلي->type == GGML_TYPE_F16) {
            kernel_name = "set_rows_f32";
        } وإلا {
            GGML_LOG_ERROR("ET: SET_ROWS unsupported output type: %s\n", ggml_type_name(كلي->type));
            أرجع خطأ;
        }

    } وإلا {
        GGML_LOG_ERROR("ET: SET_ROWS operation with unsupported types: dst=%s src0=%s src1=%s\n",
                       ggml_type_name(كلي->type), ggml_type_name(كلي->src[0]->type),
                       ggml_type_name(كلي->src[1]->type));
        أرجع خطأ;
    }

    // Validate contiguity requirements
    إذا (!ggml_is_contiguous_rows(كلي)) {
        GGML_LOG_ERROR("ET: SET_ROWS operation requires contiguous-rows destination tensor\n");
        أرجع خطأ;
    }

    إذا (!ggml_is_contiguous_rows(كلي->src[0])) {
        GGML_LOG_ERROR("ET: SET_ROWS operation requires contiguous-rows source tensor\n");
        أرجع خطأ;
    }

    إذا (!ggml_is_contiguous(كلي->src[1])) {
        GGML_LOG_ERROR("ET: SET_ROWS operation requires contiguous indices tensor\n");
        أرجع خطأ;
    }

    // Validate dimension constraints from ggml implementation
    إذا (!(كلي->ne[0] == كلي->src[0]->ne[0] &&              // same number of columns
          كلي->ne[2] == كلي->src[0]->ne[2] &&              // same batch size
          كلي->ne[3] == كلي->src[0]->ne[3] &&              // same outer dimension
          كلي->src[0]->ne[1] == كلي->src[1]->ne[0] &&      // src rows = index count
          كلي->src[0]->ne[2] % كلي->src[1]->ne[1] == 0 &&  // batch constraint
          كلي->src[0]->ne[3] % كلي->src[1]->ne[2] == 0 &&  // outer constraint
          كلي->src[1]->ne[3] == 1)) {                       // indices constraint
        GGML_LOG_ERROR("ET: SET_ROWS operation dimension constraint failed\n");
        أرجع خطأ;
    }

    ggml_et_set_rows_params params;
    params.src0 = *كلي->src[0];  // F32 source data tensor
    params.src1 = *كلي->src[1];  // I64 indices tensor
    params.dst  = *كلي;          // F32/F16 destination tensor

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    منطقي                    cpu_comparison_active = خطأ;
    إذا (set_rows_cpu_compare_config.enabled) {
        إذا (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, كلي, GGML_OP_SET_ROWS)) {
            cpu_comparison_active = true;
        } وإلا {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for SET_ROWS operation\n");
        }
    }

    منطقي kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, حجم(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    إذا (cpu_comparison_active) {
        إذا (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, كلي, &set_rows_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for SET_ROWS operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("SET_ROWS", kernel_name, كلي);
    أرجع kernel_result;
}

منطقي ggml_et_op_fill(ggml_backend_et_device_context * dev_ctx, ثابت ggml_tensor * كلي) {
    ET_PERF_START();

    ggml_et_fill_params params;
    params.dst = *كلي;
    memcpy(&params.c, كلي->op_params, حجم(عائم));

    منطقي kernel_result = ggml_et_launch_kernel(dev_ctx, "fill_f32", &params, حجم(params), 0xFFFFFFFF);

    ET_PERF_END("FILL", "fill_f32", كلي);
    أرجع kernel_result;
}

منطقي ggml_et_op_diag(ggml_backend_et_device_context * dev_ctx, ثابت ggml_tensor * كلي) {
    ET_PERF_START();

    إذا (!كلي->src[0]) {
        GGML_LOG_ERROR("ET: DIAG operation missing source tensor\n");
        أرجع خطأ;
    }

    ggml_et_diag_params params;
    params.src0 = *كلي->src[0];
    params.dst  = *كلي;

    منطقي kernel_result = ggml_et_launch_kernel(dev_ctx, "diag_f32", &params, حجم(params), 0xFFFFFFFF);

    ET_PERF_END("DIAG", "diag_f32", كلي);
    أرجع kernel_result;
}

منطقي ggml_et_op_tri(ggml_backend_et_device_context * dev_ctx, ثابت ggml_tensor * كلي) {
    ET_PERF_START();

    إذا (!كلي->src[0]) {
        GGML_LOG_ERROR("ET: TRI operation missing source tensor\n");
        أرجع خطأ;
    }

    ggml_et_tri_params params;
    params.src0 = *كلي->src[0];
    params.dst  = *كلي;
    memcpy(&params.tri_type, كلي->op_params, حجم(int32_t));

    منطقي kernel_result = ggml_et_launch_kernel(dev_ctx, "tri_f32", &params, حجم(params), 0xFFFFFFFF);

    ET_PERF_END("TRI", "tri_f32", كلي);
    أرجع kernel_result;
}

منطقي ggml_et_op_solve_tri(ggml_backend_et_device_context * dev_ctx, ثابت ggml_tensor * كلي) {
    ET_PERF_START();

    إذا (!كلي->src[0] || !كلي->src[1]) {
        GGML_LOG_ERROR("ET: SOLVE_TRI operation missing source tensor(s)\n");
        أرجع خطأ;
    }

    ggml_et_solve_tri_params params;
    params.src0 = *كلي->src[0];  // A (lower-triangular)
    params.src1 = *كلي->src[1];  // B (RHS)
    params.dst  = *كلي;          // X (solution)

    منطقي kernel_result = ggml_et_launch_kernel(dev_ctx, "solve_tri_f32", &params, حجم(params), 0xFFFFFFFF);

    ET_PERF_END("SOLVE_TRI", "solve_tri_f32", كلي);
    أرجع kernel_result;
}

منطقي ggml_et_op_set(ggml_backend_et_device_context * dev_ctx, ثابت ggml_tensor * كلي) {
    ET_PERF_START();

    إذا (!كلي->src[0] || !كلي->src[1]) {
        GGML_LOG_ERROR("ET: SET operation missing source tensor(s)\n");
        أرجع خطأ;
    }

    ثابت منطقي   inplace = (منطقي) ((ثابت int32_t *) كلي->op_params)[4];
    ثابت size_t offset  = ((ثابت int32_t *) كلي->op_params)[3];
    ثابت size_t nb1     = ((ثابت int32_t *) كلي->op_params)[0];
    ثابت size_t nb2     = ((ثابت int32_t *) كلي->op_params)[1];
    ثابت size_t nb3     = ((ثابت int32_t *) كلي->op_params)[2];

    إذا (!inplace) {
        GGML_LOG_ERROR("ET: SET only supports inplace (inplace=%d)\n", inplace);
        أرجع خطأ;
    }

    إذا (كلي->type != GGML_TYPE_F32 || كلي->src[0]->type != GGML_TYPE_F32 || كلي->src[1]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: SET only supports F32 (dst=%s src0=%s src1=%s)\n", ggml_type_name(كلي->type),
                       ggml_type_name(كلي->src[0]->type), ggml_type_name(كلي->src[1]->type));
        أرجع خطأ;
    }

    إذا (!ggml_are_same_shape(كلي, كلي->src[0])) {
        GGML_LOG_ERROR("ET: SET requires same-shape src0 and dst\n");
        أرجع خطأ;
    }

    إذا (!ggml_is_contiguous(كلي) || !ggml_is_contiguous(كلي->src[0]) || !ggml_is_contiguous(كلي->src[1])) {
        GGML_LOG_ERROR("ET: SET requires contiguous dst, src0, and src1\n");
        أرجع خطأ;
    }

    ggml_et_set_params params;
    params.src1   = *كلي->src[1];
    params.dst    = *كلي;
    params.nb1    = (int32_t) nb1;
    params.nb2    = (int32_t) nb2;
    params.nb3    = (int32_t) nb3;
    params.offset = (int32_t) offset;

    منطقي kernel_result = ggml_et_launch_kernel(dev_ctx, "set_f32", &params, حجم(params), 0xFFFFFFFF);

    ET_PERF_END("SET", "set_f32", كلي);
    أرجع kernel_result;
}

منطقي ggml_et_op_pad(ggml_backend_et_device_context * dev_ctx, ثابت ggml_tensor * كلي) {
    ET_PERF_START();

    إذا (!كلي->src[0]) {
        GGML_LOG_ERROR("ET: PAD operation missing source tensor\n");
        أرجع خطأ;
    }

    إذا (كلي->type != GGML_TYPE_F32 || كلي->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: PAD only supports F32 (src=%s dst=%s)\n", ggml_type_name(كلي->src[0]->type),
                       ggml_type_name(كلي->type));
        أرجع خطأ;
    }

    إذا (!ggml_is_contiguous(كلي)) {
        GGML_LOG_ERROR("ET: PAD requires contiguous output tensor\n");
        أرجع خطأ;
    }

    إذا (كلي->src[0]->nb[0] != حجم(عائم)) {
        GGML_LOG_ERROR("ET: PAD requires element-contiguous src dim0 (nb[0]=%zu)\n", (size_t) كلي->src[0]->nb[0]);
        أرجع خطأ;
    }

    // Extract padding parameters from op_params
    ثابت int32_t * op_params = (ثابت int32_t *) كلي->op_params;

    ggml_et_pad_params params;
    params.src0  = *كلي->src[0];
    params.dst   = *كلي;
    params.lp[0] = op_params[0];
    params.rp[0] = op_params[1];
    params.lp[1] = op_params[2];
    params.rp[1] = op_params[3];
    params.lp[2] = op_params[4];
    params.rp[2] = op_params[5];
    params.lp[3] = op_params[6];
    params.rp[3] = op_params[7];

    // v1: no dim0 padding
    إذا (params.lp[0] != 0 || params.rp[0] != 0) {
        GGML_LOG_ERROR("ET: PAD dim0 padding not supported (lp0=%d rp0=%d)\n", params.lp[0], params.rp[0]);
        أرجع خطأ;
    }

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    منطقي                    cpu_comparison_active = خطأ;
    إذا (pad_cpu_compare_config.enabled) {
        إذا (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, كلي, GGML_OP_PAD)) {
            cpu_comparison_active = true;
        } وإلا {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for PAD operation\n");
        }
    }

    منطقي kernel_result = ggml_et_launch_kernel(dev_ctx, "pad_f32", &params, حجم(params), 0xFFFFFFFF);

    إذا (cpu_comparison_active) {
        إذا (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, كلي, &pad_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for PAD operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("PAD", "pad_f32", كلي);
    أرجع kernel_result;
}
