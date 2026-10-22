#అన్తర్భావ "ggml-et-ops.h"

#అన్తర్భావ "ggml-et-cpu-compare.h"
#అన్తర్భావ "ggml-et-kernels.h"
#అన్తర్భావ "ggml-impl.h"

#అన్తర్భావ <stdio.h>

#అన్తర్భావ <cstdint>

// CPU comparison configuration - can be enabled for debugging
స్థిర ggml_et_cpu_compare_config rope_cpu_compare_config = {
    /* .enabled = */ అబద్ధం,
    /* .use_cpu_result = */ అబద్ధం,  // Replace ET result with CPU result
    /* .log_differences = */ నిజం,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

స్థిర ggml_et_cpu_compare_config rms_norm_cpu_compare_config = {
    /* .enabled = */ అబద్ధం,
    /* .use_cpu_result = */ అబద్ధం,
    /* .log_differences = */ నిజం,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

స్థిర ggml_et_cpu_compare_config norm_cpu_compare_config = {
    /* .enabled = */ అబద్ధం,
    /* .use_cpu_result = */ అబద్ధం,
    /* .log_differences = */ నిజం,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

స్థిర ggml_et_cpu_compare_config l2_norm_cpu_compare_config = {
    /* .enabled = */ అబద్ధం,
    /* .use_cpu_result = */ అబద్ధం,
    /* .log_differences = */ నిజం,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

స్థిర ggml_et_cpu_compare_config సమూహం_norm_cpu_compare_config = {
    /* .enabled = */ అబద్ధం,
    /* .use_cpu_result = */ అబద్ధం,
    /* .log_differences = */ నిజం,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

స్థిర ggml_et_cpu_compare_config im2col_cpu_compare_config = {
    /* .enabled = */ అబద్ధం,
    /* .use_cpu_result = */ అబద్ధం,
    /* .log_differences = */ నిజం,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

స్థిర ggml_et_cpu_compare_config unary_cpu_compare_config = {
    /* .enabled = */ అబద్ధం,
    /* .use_cpu_result = */ అబద్ధం,
    /* .log_differences = */ నిజం,
    /* .tolerance = */ 1e-4f,
    /* .max_log_elements = */ 4096
};

స్థిర ggml_et_cpu_compare_config కూడిక_అడ్డువరుసలు_cpu_compare_config = {
    /* .enabled = */ అబద్ధం,
    /* .use_cpu_result = */ అబద్ధం,
    /* .log_differences = */ నిజం,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

స్థిర ggml_et_cpu_compare_config clamp_cpu_compare_config = {
    /* .enabled = */ అబద్ధం,
    /* .use_cpu_result = */ అబద్ధం,
    /* .log_differences = */ నిజం,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

స్థిర ggml_et_cpu_compare_config సగటు_cpu_compare_config = {
    /* .enabled = */ అబద్ధం,
    /* .use_cpu_result = */ అబద్ధం,
    /* .log_differences = */ నిజం,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

స్థిర ggml_et_cpu_compare_config sqr_cpu_compare_config = {
    /* .enabled = */ అబద్ధం,
    /* .use_cpu_result = */ అబద్ధం,
    /* .log_differences = */ నిజం,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

స్థిర ggml_et_cpu_compare_config elmap_cpu_compare_config = {
    /* .enabled = */ అబద్ధం,
    /* .use_cpu_result = */ అబద్ధం,
    /* .log_differences = */ నిజం,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

స్థిర ggml_et_cpu_compare_config glu_cpu_compare_config = {
    /* .enabled = */ అబద్ధం,
    /* .use_cpu_result = */ అబద్ధం,
    /* .log_differences = */ నిజం,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

స్థిర ggml_et_cpu_compare_config mul_mat_cpu_compare_config = {
    /* .enabled = */ అబద్ధం,
    /* .use_cpu_result = */ అబద్ధం,
    /* .log_differences = */ నిజం,
    /* .tolerance = */ 0.01,
    /* .max_log_elements = */ 4096
};

స్థిర ggml_et_cpu_compare_config mul_mat_id_cpu_compare_config = {
    /* .enabled = */ అబద్ధం,
    /* .use_cpu_result = */ అబద్ధం,
    /* .log_differences = */ నిజం,
    /* .tolerance = */ 0.01,
    /* .max_log_elements = */ 4096
};

స్థిర ggml_et_cpu_compare_config softmax_cpu_compare_config = {
    /* .enabled = */ అబద్ధం,
    /* .use_cpu_result = */ అబద్ధం,
    /* .log_differences = */ నిజం,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 1024
};

స్థిర ggml_et_cpu_compare_config get_అడ్డువరుసలు_cpu_compare_config = {
    /* .enabled = */ అబద్ధం,
    /* .use_cpu_result = */ అబద్ధం,
    /* .log_differences = */ నిజం,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 2048
};

స్థిర ggml_et_cpu_compare_config pad_cpu_compare_config = {
    /* .enabled = */ అబద్ధం,
    /* .use_cpu_result = */ అబద్ధం,
    /* .log_differences = */ నిజం,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

స్థిర ggml_et_cpu_compare_config cont_cpu_compare_config = {
    /* .enabled = */ అబద్ధం,
    /* .use_cpu_result = */ అబద్ధం,
    /* .log_differences = */ నిజం,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

స్థిర ggml_et_cpu_compare_config concat_cpu_compare_config = {
    /* .enabled = */ అబద్ధం,
    /* .use_cpu_result = */ అబద్ధం,
    /* .log_differences = */ నిజం,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

స్థిర ggml_et_cpu_compare_config cumsum_cpu_compare_config = {
    /* .enabled = */ అబద్ధం,
    /* .use_cpu_result = */ అబద్ధం,
    /* .log_differences = */ నిజం,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

స్థిర ggml_et_cpu_compare_config repeat_cpu_compare_config = {
    /* .enabled = */ అబద్ధం,
    /* .use_cpu_result = */ అబద్ధం,
    /* .log_differences = */ నిజం,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

స్థిర ggml_et_cpu_compare_config ssm_conv_cpu_compare_config = {
    /* .enabled = */ అబద్ధం,
    /* .use_cpu_result = */ అబద్ధం,
    /* .log_differences = */ నిజం,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

స్థిర ggml_et_cpu_compare_config rwkv_wkv6_cpu_compare_config = {
    /* .enabled = */ అబద్ధం,
    /* .use_cpu_result = */ అబద్ధం,
    /* .log_differences = */ నిజం,
    /* .tolerance = */ 1e-4f,
    /* .max_log_elements = */ 4096
};

స్థిర ggml_et_cpu_compare_config rwkv_wkv7_cpu_compare_config = {
    /* .enabled = */ అబద్ధం,
    /* .use_cpu_result = */ అబద్ధం,
    /* .log_differences = */ నిజం,
    /* .tolerance = */ 1e-4f,
    /* .max_log_elements = */ 4096
};

స్థిర ggml_et_cpu_compare_config సమితి_అడ్డువరుసలు_cpu_compare_config = {
    /* .enabled = */ అబద్ధం,
    /* .use_cpu_result = */ అబద్ధం,
    /* .log_differences = */ నిజం,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 2048
};

తార్కిక ggml_et_op_rms_norm_mul(ggml_backend_et_device_context * dev_ctx,
                             స్థిరరాశి ggml_tensor *              rms_norm_కణుపు,
                             స్థిరరాశి ggml_tensor *              mul_కణుపు) {
    ET_PERF_ప్రారంభం();

    ఒకవేళ (!dev_ctx || !rms_norm_కణుపు || !mul_కణుపు) {
        GGML_LOG_ERROR("ET: Invalid parameters for fused RMS_NORM_MUL operation\n");
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (!rms_norm_కణుపు->src[0]) {
        GGML_LOG_ERROR("ET: Fused RMS_NORM_MUL missing required input\n");
        తిరిగి అబద్ధం;
    }

    // Extract weights: the MUL operand that isn't the rms_norm output
    స్థిరరాశి ggml_tensor * weights = (mul_కణుపు->src[0] == rms_norm_కణుపు) ? mul_కణుపు->src[1] : mul_కణుపు->src[0];

    ఒకవేళ (!weights) {
        GGML_LOG_ERROR("ET: Fused RMS_NORM_MUL missing weights tensor\n");
        తిరిగి అబద్ధం;
    }

    దశాంశం eps;
    జ్ఞాపకం_నకలు(&eps, rms_norm_కణుపు->op_params, పరిమాణం(దశాంశం));

    ggml_et_rms_norm_mul_params params;
    params.src0 = *rms_norm_కణుపు->src[0];  // input to normalize
    params.src1 = *weights;                // normalization weights
    params.dst  = *mul_కణుపు;               // final output
    params.eps  = eps;

    తార్కిక kernel_ఫలితం = ggml_et_launch_kernel(dev_ctx, "rms_norm_mul_f32", &params, పరిమాణం(params), 0xFFFFFFFF);

    ET_PERF_చివర_EXT("RMS_NORM_MUL", "rms_norm_mul_f32", mul_కణుపు, "eps=%.6f", (ద్విగుణ) eps);
    తిరిగి kernel_ఫలితం;
}

తార్కిక ggml_et_op_scale(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు) {
    ET_PERF_ప్రారంభం();

    ఒకవేళ (!dev_ctx || !కణుపు) {
        GGML_LOG_ERROR("ET: Invalid parameters for SCALE operation\n");
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (!కణుపు->src[0]) {
        GGML_LOG_ERROR("ET: SCALE operation missing required input\n");
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (కణుపు->type != GGML_TYPE_F32 || కణుపు->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: SCALE operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(కణుపు->type),
                       ggml_type_name(కణుపు->src[0]->type));
        తిరిగి అబద్ధం;
    }

    దశాంశం scale, bias;
    జ్ఞాపకం_నకలు(&scale, (స్థిరరాశి దశాంశం *) కణుపు->op_params + 0, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&bias, (స్థిరరాశి దశాంశం *) కణుపు->op_params + 1, పరిమాణం(దశాంశం));

    ggml_et_scale_params params;
    params.src0  = *కణుపు->src[0];
    params.dst   = *కణుపు;
    params.scale = scale;
    params.bias  = bias;

    తార్కిక kernel_ఫలితం = ggml_et_launch_kernel(dev_ctx, "scale_f32", &params, పరిమాణం(params), 0xFFFFFFFF);

    ET_PERF_చివర_EXT("SCALE", "scale_f32", కణుపు, "scale=%.6f|bias=%.6f", (ద్విగుణ) scale, (ద్విగుణ) bias);
    తిరిగి kernel_ఫలితం;
}

తార్కిక ggml_et_op_sqr(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు) {
    ET_PERF_ప్రారంభం();

    ఒకవేళ (!dev_ctx || !కణుపు) {
        GGML_LOG_ERROR("ET: Invalid parameters for SQR operation\n");
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (!కణుపు->src[0]) {
        GGML_LOG_ERROR("ET: SQR operation missing required input\n");
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (కణుపు->type != GGML_TYPE_F32 || కణుపు->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: SQR operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(కణుపు->type),
                       ggml_type_name(కణుపు->src[0]->type));
        తిరిగి అబద్ధం;
    }

    ggml_et_sqr_params params;
    params.src0 = *కణుపు->src[0];  // F32 input tensor
    params.dst  = *కణుపు;          // F32 output tensor

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    తార్కిక                    cpu_comparison_చురుకైన = అబద్ధం;
    ఒకవేళ (sqr_cpu_compare_config.enabled) {
        ఒకవేళ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, కణుపు, GGML_OP_SQR)) {
            cpu_comparison_చురుకైన = నిజం;
        } లేకపోతే {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for SQR operation\n");
        }
    }

    తార్కిక kernel_ఫలితం = ggml_et_launch_kernel(dev_ctx, "sqr_f32", &params, పరిమాణం(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ఒకవేళ (cpu_comparison_చురుకైన) {
        ఒకవేళ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, కణుపు, &sqr_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for SQR operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_చివర("SQR", "sqr_f32", కణుపు);
    తిరిగి kernel_ఫలితం;
}

తార్కిక ggml_et_op_కూడిక_అడ్డువరుసలు(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు) {
    ET_PERF_ప్రారంభం();

    ఒకవేళ (!dev_ctx || !కణుపు) {
        GGML_LOG_ERROR("ET: Invalid parameters for SUM_ROWS operation\n");
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (!కణుపు->src[0]) {
        GGML_LOG_ERROR("ET: SUM_ROWS operation missing required input\n");
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (కణుపు->type != GGML_TYPE_F32 || కణుపు->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: SUM_ROWS operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(కణుపు->type),
                       ggml_type_name(కణుపు->src[0]->type));
        తిరిగి అబద్ధం;
    }

    ggml_et_కూడిక_అడ్డువరుసలు_params params;
    params.src0 = *కణుపు->src[0];
    params.dst  = *కణుపు;

    // Phase 1: Initialize CPU comparison context
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    తార్కిక                    cpu_comparison_చురుకైన = అబద్ధం;
    ఒకవేళ (కూడిక_అడ్డువరుసలు_cpu_compare_config.enabled) {
        ఒకవేళ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, కణుపు, GGML_OP_కూడిక_అడ్డువరుసలు)) {
            cpu_comparison_చురుకైన = నిజం;
        } లేకపోతే {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for SUM_ROWS operation\n");
        }
    }

    తార్కిక kernel_ఫలితం = ggml_et_launch_kernel(dev_ctx, "sum_rows_f32", &params, పరిమాణం(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare
    ఒకవేళ (cpu_comparison_చురుకైన) {
        ఒకవేళ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, కణుపు, &కూడిక_అడ్డువరుసలు_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for SUM_ROWS operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_చివర("SUM_ROWS", "sum_rows_f32", కణుపు);
    తిరిగి kernel_ఫలితం;
}

తార్కిక ggml_et_op_సగటు(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు) {
    ET_PERF_ప్రారంభం();

    ఒకవేళ (!dev_ctx || !కణుపు) {
        GGML_LOG_ERROR("ET: Invalid parameters for MEAN operation\n");
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (!కణుపు->src[0]) {
        GGML_LOG_ERROR("ET: MEAN operation missing required input\n");
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (కణుపు->type != GGML_TYPE_F32 || కణుపు->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: MEAN operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(కణుపు->type),
                       ggml_type_name(కణుపు->src[0]->type));
        తిరిగి అబద్ధం;
    }

    ggml_et_సగటు_params params;
    params.src0 = *కణుపు->src[0];
    params.dst  = *కణుపు;

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    తార్కిక                    cpu_comparison_చురుకైన = అబద్ధం;
    ఒకవేళ (సగటు_cpu_compare_config.enabled) {
        ఒకవేళ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, కణుపు, GGML_OP_సగటు)) {
            cpu_comparison_చురుకైన = నిజం;
        } లేకపోతే {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for MEAN operation\n");
        }
    }

    తార్కిక kernel_ఫలితం = ggml_et_launch_kernel(dev_ctx, "mean_f32", &params, పరిమాణం(params), 0xFFFFFFFF);

    ఒకవేళ (cpu_comparison_చురుకైన) {
        ఒకవేళ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, కణుపు, &సగటు_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for MEAN operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_చివర("MEAN", "mean_f32", కణుపు);
    తిరిగి kernel_ఫలితం;
}

తార్కిక ggml_et_op_clamp(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు) {
    ET_PERF_ప్రారంభం();

    ఒకవేళ (!dev_ctx || !కణుపు) {
        GGML_LOG_ERROR("ET: Invalid parameters for CLAMP operation\n");
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (!కణుపు->src[0]) {
        GGML_LOG_ERROR("ET: CLAMP operation missing required input\n");
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (కణుపు->type != GGML_TYPE_F32 || కణుపు->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: CLAMP operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(కణుపు->type),
                       ggml_type_name(కణుపు->src[0]->type));
        తిరిగి అబద్ధం;
    }

    ggml_et_clamp_params params;
    params.src0 = *కణుపు->src[0];
    params.dst  = *కణుపు;
    // op_params layout per ggml.c::ggml_clamp: { min, max } as floats
    జ్ఞాపకం_నకలు(&params.min_val, (స్థిరరాశి దశాంశం *) కణుపు->op_params + 0, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&params.max_val, (స్థిరరాశి దశాంశం *) కణుపు->op_params + 1, పరిమాణం(దశాంశం));

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    తార్కిక                    cpu_comparison_చురుకైన = అబద్ధం;
    ఒకవేళ (clamp_cpu_compare_config.enabled) {
        ఒకవేళ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, కణుపు, GGML_OP_CLAMP)) {
            cpu_comparison_చురుకైన = నిజం;
        } లేకపోతే {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for CLAMP operation\n");
        }
    }

    తార్కిక kernel_ఫలితం = ggml_et_launch_kernel(dev_ctx, "clamp_f32", &params, పరిమాణం(params), 0xFFFFFFFF);

    ఒకవేళ (cpu_comparison_చురుకైన) {
        ఒకవేళ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, కణుపు, &clamp_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for CLAMP operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_చివర("CLAMP", "clamp_f32", కణుపు);
    తిరిగి kernel_ఫలితం;
}

తార్కిక ggml_et_op_unary(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు) {
    ET_PERF_ప్రారంభం();

    ఒకవేళ (!dev_ctx || !కణుపు) {
        GGML_LOG_ERROR("ET: Invalid parameters for UNARY operation\n");
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (!కణుపు->src[0]) {
        GGML_LOG_ERROR("ET: UNARY operation missing required input\n");
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (కణుపు->type != GGML_TYPE_F32 || కణుపు->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: UNARY operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(కణుపు->type),
                       ggml_type_name(కణుపు->src[0]->type));
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి ggml_unary_op uop     = ggml_get_unary_op(కణుపు);
    స్థిరరాశి అక్షరం *        op_name = ggml_unary_op_name(uop);

    ggml_et_unary_params params;
    params.src0     = *కణుపు->src[0];  // F32 input tensor
    params.dst      = *కణుపు;          // F32 output tensor
    params.unary_op = (int32_t) uop;

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    తార్కిక                    cpu_comparison_చురుకైన = అబద్ధం;
    ఒకవేళ (unary_cpu_compare_config.enabled) {
        ఒకవేళ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, కణుపు, GGML_OP_UNARY)) {
            cpu_comparison_చురుకైన = నిజం;
        } లేకపోతే {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for UNARY/%s operation\n", op_name);
        }
    }

    తార్కిక kernel_ఫలితం = ggml_et_launch_kernel(dev_ctx, "unary_f32", &params, పరిమాణం(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ఒకవేళ (cpu_comparison_చురుకైన) {
        ఒకవేళ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, కణుపు, &unary_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for UNARY/%s operation\n", op_name);
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_చివర_EXT("UNARY", "unary_f32", కణుపు, "op=%s", op_name);
    తిరిగి kernel_ఫలితం;
}

తార్కిక ggml_et_op_mul(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు) {
    // Delegate to generic element map operation
    తిరిగి ggml_et_op_elmap(dev_ctx, కణుపు);
}

తార్కిక ggml_et_op_కూడు(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు) {
    // Delegate to generic element map operation
    తిరిగి ggml_et_op_elmap(dev_ctx, కణుపు);
}

తార్కిక ggml_et_op_sub(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు) {
    // Delegate to generic element map operation
    తిరిగి ggml_et_op_elmap(dev_ctx, కణుపు);
}

తార్కిక ggml_et_op_elmap(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు) {
    ET_PERF_ప్రారంభం();

    ఒకవేళ (!dev_ctx || !కణుపు) {
        GGML_LOG_ERROR("ET: Invalid parameters for element map operation\n");
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (!కణుపు->src[0] || !కణుపు->src[1]) {
        GGML_LOG_ERROR("ET: Element map operation missing required inputs\n");
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (కణుపు->type != GGML_TYPE_F32 || కణుపు->src[0]->type != GGML_TYPE_F32 || కణుపు->src[1]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: Element map operation with unsupported types: dst=%s src0=%s src1=%s\n",
                       ggml_type_name(కణుపు->type), ggml_type_name(కణుపు->src[0]->type),
                       ggml_type_name(కణుపు->src[1]->type));
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి అక్షరం * op_name = ggml_op_name(కణుపు->op);

    ggml_et_elmap_params params;
    params.src0 = *కణుపు->src[0];
    params.src1 = *కణుపు->src[1];
    params.dst  = *కణుపు;  // F32 output tensor (op type stored in dst.op)

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    తార్కిక                    cpu_comparison_చురుకైన = అబద్ధం;
    ఒకవేళ (elmap_cpu_compare_config.enabled) {
        ఒకవేళ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, కణుపు, కణుపు->op)) {
            cpu_comparison_చురుకైన = నిజం;
        } లేకపోతే {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for %s operation\n", op_name);
        }
    }

    // fprintf(stderr, "ET: el_map s0 [%ld, %ld, %ld, %ld] s1 [%ld, %ld, %ld, %ld]\n",
    //     node->src[0]->ne[0], node->src[0]->ne[1], node->src[0]->ne[2], node->src[0]->ne[3],
    //     node->src[1]->ne[0], node->src[1]->ne[1], node->src[1]->ne[2], node->src[1]->ne[3]);

    తార్కిక kernel_ఫలితం = ggml_et_launch_kernel(dev_ctx, "el_map_f32", &params, పరిమాణం(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ఒకవేళ (cpu_comparison_చురుకైన) {
        ఒకవేళ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, కణుపు, &elmap_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for %s operation\n", op_name);
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_చివర(op_name, "el_map_f32", కణుపు);
    తిరిగి kernel_ఫలితం;
}

తార్కిక ggml_et_op_glu(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు) {
    ET_PERF_ప్రారంభం();

    // Validate inputs
    ఒకవేళ (!dev_ctx || !కణుపు) {
        GGML_LOG_ERROR("ET: Invalid parameters for GLU operation\n");
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (!కణుపు->src[0]) {
        GGML_LOG_ERROR("ET: GLU operation missing required input\n");
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి తార్కిక is_విభజించు_బహుళకం = కణుపు->src[1] != శూన్యసూచిక;

    // Only support F32 (as validated by supports_op)
    ఒకవేళ (కణుపు->type != GGML_TYPE_F32 || కణుపు->src[0]->type != GGML_TYPE_F32 ||
        (is_విభజించు_బహుళకం && కణుపు->src[1]->type != GGML_TYPE_F32)) {
        తిరిగి అబద్ధం;
    }

    // Extract GLU operation parameters from op_params
    int32_t glu_op_type = ggml_get_op_params_i32(కణుపు, 0);  // GLU variant (REGLU, GEGLU, SWIGLU, etc.)
    int32_t swapped     = ggml_get_op_params_i32(కణుపు, 1);  // Whether gate/value are swapped

    // Supported variants
    ఎంపిక (glu_op_type) {
        స్థితి GGML_GLU_OP_REGLU:
        స్థితి GGML_GLU_OP_GEGLU:
        స్థితి GGML_GLU_OP_SWIGLU:
        స్థితి GGML_GLU_OP_SWIGLU_OAI:
        స్థితి GGML_GLU_OP_SWIGLU_CLAMP:
        స్థితి GGML_GLU_OP_GEGLU_ERF:
        స్థితి GGML_GLU_OP_GEGLU_శీఘ్ర:
            ఆపు;
        అప్రమేయ:
            GGML_LOG_ERROR("ET: GLU operation with unsupported variant: %s\n",
                           ggml_glu_op_name((ggml_glu_op) glu_op_type));
            తిరిగి అబద్ధం;
    }

    // Get GLU operation name for logging
    స్థిరరాశి అక్షరం * glu_op_name = ggml_glu_op_name((ggml_glu_op) glu_op_type);

    // Pack parameters. Single-tensor mode is encoded by zeroing src1.
    ggml_et_glu_params params = {};
    params.src0               = *కణుపు->src[0];
    ఒకవేళ (is_విభజించు_బహుళకం) {
        params.src1 = *కణుపు->src[1];
    }
    params.dst         = *కణుపు;
    params.glu_op_type = glu_op_type;
    params.swapped     = swapped;
    params.alpha       = 0.0f;
    params.హద్దు       = 0.0f;
    ఒకవేళ (glu_op_type == GGML_GLU_OP_SWIGLU_OAI) {
        params.alpha = ggml_get_op_params_f32(కణుపు, 2);
    }
    ఒకవేళ (glu_op_type == GGML_GLU_OP_SWIGLU_OAI || glu_op_type == GGML_GLU_OP_SWIGLU_CLAMP) {
        params.హద్దు = ggml_get_op_params_f32(కణుపు, 3);
    }
    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    తార్కిక                    cpu_comparison_చురుకైన = అబద్ధం;
    ఒకవేళ (glu_cpu_compare_config.enabled) {
        ఒకవేళ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, కణుపు, GGML_OP_GLU)) {
            cpu_comparison_చురుకైన = నిజం;
        } లేకపోతే {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for %s operation\n", glu_op_name);
        }
    }

    // Launch ET kernel
    తార్కిక kernel_ఫలితం = ggml_et_launch_kernel(dev_ctx, "glu_f32", &params, పరిమాణం(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ఒకవేళ (cpu_comparison_చురుకైన) {
        ఒకవేళ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, కణుపు, &glu_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for %s operation\n", glu_op_name);
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_చివర("GLU", "glu_f32", కణుపు);
    తిరిగి kernel_ఫలితం;
}

తార్కిక ggml_et_op_mul_mat(ggml_backend_et_device_context * dev_ctx,
                        స్థిరరాశి ggml_tensor *              కణుపు,
                        స్థిరరాశి ggml_tensor *              కూడు_కణుపు) {
    ET_PERF_ప్రారంభం();

    ఒకవేళ (!dev_ctx || !కణుపు) {
        GGML_LOG_ERROR("ET: Invalid parameters for MUL_MAT operation\n");
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (!కణుపు->src[0] || !కణుపు->src[1]) {
        GGML_LOG_ERROR("ET: MUL_MAT operation missing required inputs\n");
        తిరిగి అబద్ధం;
    }

    // Fused MM+ADD: when add_node is non-NULL the caller has already validated
    // (Q8_0 weights, F32 acts, exact-shape ADD with stride parity to dst) via
    // ggml_et_can_fuse({MUL_MAT, ADD}). The kernel writes dst = mm + bias and
    // the ADD's output replaces MM's as the actual dst.
    స్థిరరాశి ggml_tensor * fused_dst   = కూడు_కణుపు ? కూడు_కణుపు : కణుపు;
    స్థిరరాశి ggml_tensor * bias_tensor = శూన్యసూచిక;
    ఒకవేళ (కూడు_కణుపు) {
        bias_tensor = (కూడు_కణుపు->src[0] == కణుపు) ? కూడు_కణుపు->src[1] : కూడు_కణుపు->src[0];
    }

    స్థిరరాశి అక్షరం * kernel_name;
    స్థిరరాశి అక్షరం * src0_type_name;

    ఒకవేళ (కణుపు->type == GGML_TYPE_F32 && కణుపు->src[0]->type == GGML_TYPE_Q4_0 && కణుపు->src[1]->type == GGML_TYPE_F32 &&
        కణుపు->src[1]->ne[1] >= 53 &&      // N >= 53
        కణుపు->src[0]->ne[1] % 16 == 0 &&  // M % TILE_M
        కణుపు->src[0]->ne[0] % 32 == 0) {  // K % BLOCK_K (Q4_0 block)

        // Matrix engine for N >= 53; partial N (via n_cur-1) and errata padding are handled in-kernel.
        kernel_name    = "mul_mat_Q4_0_matrix_engine";
        src0_type_name = "Q4_0";

    } లేకపోతే ఒకవేళ (కణుపు->type == GGML_TYPE_F32 && కణుపు->src[0]->type == GGML_TYPE_Q4_0 &&
               కణుపు->src[1]->type == GGML_TYPE_F32) {
        kernel_name    = "mul_mat_Q4_0";  // N < 53, or M % 16 != 0 or K % 32 != 0
        src0_type_name = "Q4_0";

    } లేకపోతే ఒకవేళ (కణుపు->type == GGML_TYPE_F32 && కణుపు->src[0]->type == GGML_TYPE_Q8_0 &&
               కణుపు->src[1]->type == GGML_TYPE_F32) {
        kernel_name    = "mul_mat_Q8_0";
        src0_type_name = "Q8_0";

    } లేకపోతే ఒకవేళ (కణుపు->type == GGML_TYPE_F32 && కణుపు->src[0]->type == GGML_TYPE_F16 &&
               కణుపు->src[1]->type == GGML_TYPE_F16 && కణుపు->ne[0] % 16 == 0 && కణుపు->src[0]->ne[0] % 16 == 0 &&
               కణుపు->src[0]->ne[1] % 16 == 0 && కణుపు->src[1]->ne[0] != 1) {
        kernel_name    = "mul_mat_f16_matrix_engine";
        src0_type_name = "F16";

    } లేకపోతే ఒకవేళ (కణుపు->type == GGML_TYPE_F32 && కణుపు->src[0]->type == GGML_TYPE_F16 &&
               (కణుపు->src[1]->type == GGML_TYPE_F16 || కణుపు->src[1]->type == GGML_TYPE_F32)) {
        kernel_name    = "mul_mat_f16";
        src0_type_name = "F16";

    } లేకపోతే ఒకవేళ (కణుపు->type == GGML_TYPE_F32 && కణుపు->src[0]->type == GGML_TYPE_F32 &&
               కణుపు->src[1]->type == GGML_TYPE_F32 && కణుపు->ne[0] % 16 == 0 && కణుపు->src[0]->ne[0] % 16 == 0 &&
               కణుపు->src[0]->ne[1] % 16 == 0 && కణుపు->src[1]->ne[0] != 1) {  // GEMV is faster with the generic path

        kernel_name    = "mul_mat_f32_matrix_engine";
        src0_type_name = "F32";
    } లేకపోతే ఒకవేళ (కణుపు->type == GGML_TYPE_F32 && కణుపు->src[0]->type == GGML_TYPE_F32 &&
               (కణుపు->src[1]->type == GGML_TYPE_F16 || కణుపు->src[1]->type == GGML_TYPE_F32)) {
        kernel_name    = "mul_mat_f32";
        src0_type_name = "F32";
    } లేకపోతే {
        GGML_LOG_ERROR("ET: MUL_MAT operation with unsupported types: dst=%s src0=%s src1=%s\n",
                       ggml_type_name(కణుపు->type), ggml_type_name(కణుపు->src[0]->type),
                       ggml_type_name(కణుపు->src[1]->type));
        తిరిగి అబద్ధం;
    }

    ggml_et_ద్విమాన_params params;
    params.src0 = *కణుపు->src[0];  // weight matrix
    params.src1 = *కణుపు->src[1];  // activation matrix
    params.dst  = *fused_dst;     // output (= add_node when fused, else node)

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    తార్కిక                    cpu_comparison_చురుకైన = అబద్ధం;
    ఒకవేళ (mul_mat_cpu_compare_config.enabled) {
        ఒకవేళ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, fused_dst, GGML_OP_MUL_MAT)) {
            cpu_comparison_చురుకైన = నిజం;
        } లేకపోతే {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for MUL_MAT operation\n");
        }
    }

    తార్కిక kernel_ఫలితం;
    ఒకవేళ (కణుపు->src[0]->type == GGML_TYPE_Q8_0) {
        // Q8_0 kernel always takes the extended struct. bias.data is non-NULL
        // only on the fused path; otherwise the kernel skips the add entirely.
        ggml_et_mm_q8_params q8_params = {};
        q8_params.src0                 = params.src0;
        q8_params.src1                 = params.src1;
        q8_params.dst                  = params.dst;
        ఒకవేళ (bias_tensor) {
            q8_params.bias = *bias_tensor;
        }
        kernel_ఫలితం = ggml_et_launch_kernel(dev_ctx, kernel_name, &q8_params, పరిమాణం(q8_params), 0xFFFFFFFF);
    } లేకపోతే {
        // Non-Q8 MM kernels don't yet support fused-add; the graph fuse check
        // already rejects non-Q8 pairs, so add_node is always nullptr here.
        kernel_ఫలితం = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, పరిమాణం(params), 0xFFFFFFFF);
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
    ఒకవేళ (cpu_comparison_చురుకైన) {
        ఒకవేళ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, fused_dst, &mul_mat_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for MUL_MAT operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    {
        // Calculate actual FLOPs including batch/sequence dimensions
        // dst shape: [M, N, ne2, ne3] where M=ne[1], N=ne[0]
        int64_t m   = కణుపు->ne[1];
        int64_t n   = కణుపు->ne[0];
        int64_t k   = కణుపు->src[0]->ne[0];
        int64_t ne2 = కణుపు->ne[2];
        int64_t ne3 = కణుపు->ne[3];

        // Total FLOPs = (batch_size) * M * N * (2*K - 1)
        // Each MxN matrix-matrix multiply does M*N*(2*K-1) FLOPs
        // Broadcasting is handled by repeating computation, so count actual operations
        int64_t batch_కొలత  = ne2 * ne3;
        int64_t మొత్తం_flops = batch_కొలత * m * n * (2 * k - 1);

        అక్షరం kernel_variant[64];
        snprintf(kernel_variant, పరిమాణం(kernel_variant), "%s_%sx%s", kernel_name, src0_type_name,
                 ggml_type_name(కణుపు->src[1]->type));
        ET_PERF_చివర_EXT("MUL_MAT", kernel_variant, కణుపు, "flops=%" PRId64, మొత్తం_flops);
    }
    తిరిగి kernel_ఫలితం;
}

తార్కిక ggml_et_op_mul_mat_id(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు) {
    ET_PERF_ప్రారంభం();
    ఒకవేళ (!dev_ctx || !కణుపు) {
        GGML_LOG_ERROR("ET: Invalid parameters for MUL_MAT_ID operation\n");
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (!కణుపు->src[0] || !కణుపు->src[1] || !కణుపు->src[2]) {
        GGML_LOG_ERROR("ET: MUL_MAT_ID operation missing required inputs\n");
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి అక్షరం * kernel_name;
    స్థిరరాశి అక్షరం * src0_type_name;

    // Support Q8_0/Q4_0/F16/F32 x F32 -> F32 matrix multiplication with expert selection
    ఒకవేళ (కణుపు->type == GGML_TYPE_F32 && కణుపు->src[0]->type == GGML_TYPE_Q8_0 && కణుపు->src[1]->type == GGML_TYPE_F32 &&
        కణుపు->src[2]->type == GGML_TYPE_I32) {
        kernel_name    = "mul_mat_id_Q8_0";
        src0_type_name = "Q8_0";

    } లేకపోతే ఒకవేళ (కణుపు->type == GGML_TYPE_F32 && కణుపు->src[0]->type == GGML_TYPE_Q4_0 &&
               కణుపు->src[1]->type == GGML_TYPE_F32 && కణుపు->src[2]->type == GGML_TYPE_I32) {
        kernel_name    = "mul_mat_id_Q4_0";
        src0_type_name = "Q4_0";

    } లేకపోతే ఒకవేళ (కణుపు->type == GGML_TYPE_F32 && కణుపు->src[0]->type == GGML_TYPE_F16 &&
               కణుపు->src[1]->type == GGML_TYPE_F32 && కణుపు->src[2]->type == GGML_TYPE_I32) {
        kernel_name    = "mul_mat_id_f32";
        src0_type_name = "F16";

    } లేకపోతే ఒకవేళ (కణుపు->type == GGML_TYPE_F32 && కణుపు->src[0]->type == GGML_TYPE_F32 &&
               కణుపు->src[1]->type == GGML_TYPE_F32 && కణుపు->src[2]->type == GGML_TYPE_I32) {
        kernel_name    = "mul_mat_id_f32";
        src0_type_name = "F32";

    } లేకపోతే {
        GGML_LOG_ERROR("ET: MUL_MAT_ID operation with unsupported types: dst=%s src0=%s src1=%s src2=%s\n",
                       ggml_type_name(కణుపు->type), ggml_type_name(కణుపు->src[0]->type),
                       ggml_type_name(కణుపు->src[1]->type), ggml_type_name(కణుపు->src[2]->type));
        తిరిగి అబద్ధం;
    }

    // Pack parameters - copy full tensor structures
    ggml_et_mul_mat_id_params params;
    params.src0 = *కణుపు->src[0];  // Expert weight matrices (Q8_0/F16/F32)
    params.src1 = *కణుపు->src[1];  // Activation matrix (F32)
    params.src2 = *కణుపు->src[2];  // Expert indices (I32)
    params.dst  = *కణుపు;          // Output matrix (F32)

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    తార్కిక                    cpu_comparison_చురుకైన = అబద్ధం;
    ఒకవేళ (mul_mat_id_cpu_compare_config.enabled) {
        ఒకవేళ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, కణుపు, GGML_OP_MUL_MAT_ID)) {
            cpu_comparison_చురుకైన = నిజం;
        } లేకపోతే {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for MUL_MAT_ID operation\n");
        }
    }

    // Launch ET kernel
    తార్కిక kernel_ఫలితం = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, పరిమాణం(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ఒకవేళ (cpu_comparison_చురుకైన) {
        ఒకవేళ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, కణుపు, &mul_mat_id_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for MUL_MAT_ID operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    // Calculate FLOPs (approximate - similar to MUL_MAT but with expert routing overhead)
    // Each expert computation is similar to a MUL_MAT, but we only compute for selected experts
    int64_t K             = కణుపు->src[0]->ne[0];
    int64_t M             = కణుపు->src[0]->ne[1];
    int64_t n_expert_వాడిన = కణుపు->src[2]->ne[0];
    int64_t batch         = కణుపు->src[2]->ne[1];

    int64_t మొత్తం_flops = batch * n_expert_వాడిన * M * (2 * K - 1);

    అక్షరం kernel_variant[64];
    snprintf(kernel_variant, పరిమాణం(kernel_variant), "%s_%sx%s", kernel_name, src0_type_name,
             ggml_type_name(కణుపు->src[1]->type));
    ET_PERF_చివర_EXT("MUL_MAT_ID", kernel_variant, కణుపు, "flops=%" PRId64 "|n_expert=%lld|n_expert_used=%lld",
                    మొత్తం_flops, (దీర్ఘ దీర్ఘ) కణుపు->src[0]->ne[2], (దీర్ఘ దీర్ఘ) n_expert_వాడిన);

    తిరిగి kernel_ఫలితం;
}

తార్కిక ggml_et_op_rope(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు) {
    ET_PERF_ప్రారంభం();

    ఒకవేళ (!dev_ctx || !కణుపు) {
        GGML_LOG_ERROR("ET: Invalid parameters for ROPE operation\n");
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (!కణుపు->src[0] || !కణుపు->src[1]) {
        GGML_LOG_ERROR("ET: ROPE operation missing required inputs\n");
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి అక్షరం * kernel_name;

    ఒకవేళ (కణుపు->type == GGML_TYPE_F32 && కణుపు->src[0]->type == GGML_TYPE_F32 && కణుపు->src[1]->type == GGML_TYPE_I32) {
        kernel_name = "rope_f32";
    } లేకపోతే {
        తిరిగి అబద్ధం;
    }

    // Pack parameters - copy full tensor structures and op_params
    ggml_et_rope_params params;
    params.src0 = *కణుపు->src[0];                       // F32 input tensor
    params.src1 = *కణుపు->src[1];                       // I32 position tensor
    ఒకవేళ (కణుపు->src[2]) {
        params.src2 = *కణుపు->src[2];                   // F32 frequency factors (optional)
    } లేకపోతే {
        memset(&params.src2, 0, పరిమాణం(params.src2));  // Zero if not provided
    }
    params.dst = *కణుపు;                                // F32 output tensor

    params.rope_params.n_past     = ((స్థిరరాశి int32_t *) కణుపు->op_params)[0];
    params.rope_params.n_dims     = ((స్థిరరాశి int32_t *) కణుపు->op_params)[1];
    params.rope_params.mode       = ((స్థిరరాశి int32_t *) కణుపు->op_params)[2];
    params.rope_params.n_ctx      = ((స్థిరరాశి int32_t *) కణుపు->op_params)[3];
    params.rope_params.n_ctx_orig = ((స్థిరరాశి int32_t *) కణుపు->op_params)[4];
    జ్ఞాపకం_నకలు(&params.rope_params.freq_పునాది, (స్థిరరాశి int32_t *) కణుపు->op_params + 5, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&params.rope_params.freq_scale, (స్థిరరాశి int32_t *) కణుపు->op_params + 6, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&params.rope_params.ext_గుణకం, (స్థిరరాశి int32_t *) కణుపు->op_params + 7, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&params.rope_params.attn_గుణకం, (స్థిరరాశి int32_t *) కణుపు->op_params + 8, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&params.rope_params.beta_fast, (స్థిరరాశి int32_t *) కణుపు->op_params + 9, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&params.rope_params.beta_slow, (స్థిరరాశి int32_t *) కణుపు->op_params + 10, పరిమాణం(దశాంశం));
    ఒకవేళ (params.rope_params.mode & GGML_ROPE_TYPE_MROPE) {
        జ్ఞాపకం_నకలు(params.rope_params.sections, (స్థిరరాశి int32_t *) కణుపు->op_params + 11, పరిమాణం(int32_t) * 4);
    } లేకపోతే {
        memset(params.rope_params.sections, 0, పరిమాణం(params.rope_params.sections));
    }

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    తార్కిక                    cpu_comparison_చురుకైన = అబద్ధం;
    ఒకవేళ (rope_cpu_compare_config.enabled) {
        GGML_LOG_DEBUG("ET: Initializing CPU comparison for ROPE operation\n");
        ఒకవేళ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, కణుపు, GGML_OP_ROPE)) {
            cpu_comparison_చురుకైన = నిజం;
        } లేకపోతే {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for ROPE operation\n");
        }
    }

    తార్కిక kernel_ఫలితం = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, పరిమాణం(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ఒకవేళ (cpu_comparison_చురుకైన) {
        ఒకవేళ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, కణుపు, &rope_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for ROPE operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_చివర_EXT("ROPE", kernel_name, కణుపు, "mode=0x%x|n_dims=%d|freq_base=%.2f|freq_scale=%.2f",
                    params.rope_params.mode, params.rope_params.n_dims, (ద్విగుణ) params.rope_params.freq_పునాది,
                    (ద్విగుణ) params.rope_params.freq_scale);
    తిరిగి kernel_ఫలితం;
}

తార్కిక ggml_et_op_rms_norm(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు) {
    ET_PERF_ప్రారంభం();

    ఒకవేళ (!dev_ctx || !కణుపు) {
        GGML_LOG_ERROR("ET: Invalid parameters for RMS_NORM operation\n");
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (!కణుపు->src[0]) {
        GGML_LOG_ERROR("ET: RMS_NORM operation missing required input\n");
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి అక్షరం * kernel_name;

    ఒకవేళ (కణుపు->type == GGML_TYPE_F32 && కణుపు->src[0]->type == GGML_TYPE_F32) {
        kernel_name = "rms_norm_f32";

    } లేకపోతే {
        GGML_LOG_ERROR("ET: RMS_NORM operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(కణుపు->type),
                       ggml_type_name(కణుపు->src[0]->type));
        తిరిగి అబద్ధం;
    }

    దశాంశం eps;
    జ్ఞాపకం_నకలు(&eps, కణుపు->op_params, పరిమాణం(దశాంశం));

    ggml_et_rms_norm_params params;
    params.src0 = *కణుపు->src[0];  // F32 input tensor
    params.dst  = *కణుపు;          // F32 output tensor
    params.eps  = eps;            // Epsilon parameter for numerical stability

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    తార్కిక                    cpu_comparison_చురుకైన = అబద్ధం;
    ఒకవేళ (rms_norm_cpu_compare_config.enabled) {
        ఒకవేళ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, కణుపు, GGML_OP_RMS_NORM)) {
            cpu_comparison_చురుకైన = నిజం;
        } లేకపోతే {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for RMS_NORM operation\n");
        }
    }

    తార్కిక kernel_ఫలితం = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, పరిమాణం(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ఒకవేళ (cpu_comparison_చురుకైన) {
        ఒకవేళ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, కణుపు, &rms_norm_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for RMS_NORM operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_చివర_EXT("RMS_NORM", kernel_name, కణుపు, "eps=%.6f", (ద్విగుణ) eps);
    తిరిగి kernel_ఫలితం;
}

తార్కిక ggml_et_op_norm(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు) {
    ET_PERF_ప్రారంభం();

    ఒకవేళ (!dev_ctx || !కణుపు) {
        GGML_LOG_ERROR("ET: Invalid parameters for NORM operation\n");
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (!కణుపు->src[0]) {
        GGML_LOG_ERROR("ET: NORM operation missing required input\n");
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి అక్షరం * kernel_name;

    ఒకవేళ (కణుపు->type == GGML_TYPE_F32 && కణుపు->src[0]->type == GGML_TYPE_F32) {
        kernel_name = "norm_f32";

    } లేకపోతే {
        GGML_LOG_ERROR("ET: NORM operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(కణుపు->type),
                       ggml_type_name(కణుపు->src[0]->type));
        తిరిగి అబద్ధం;
    }

    దశాంశం eps;
    జ్ఞాపకం_నకలు(&eps, కణుపు->op_params, పరిమాణం(దశాంశం));

    ggml_et_norm_params params;
    params.src0 = *కణుపు->src[0];  // F32 input tensor
    params.dst  = *కణుపు;          // F32 output tensor
    params.eps  = eps;            // Epsilon parameter for numerical stability

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    తార్కిక                    cpu_comparison_చురుకైన = అబద్ధం;
    ఒకవేళ (norm_cpu_compare_config.enabled) {
        ఒకవేళ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, కణుపు, GGML_OP_NORM)) {
            cpu_comparison_చురుకైన = నిజం;
        } లేకపోతే {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for NORM operation\n");
        }
    }

    తార్కిక kernel_ఫలితం = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, పరిమాణం(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ఒకవేళ (cpu_comparison_చురుకైన) {
        ఒకవేళ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, కణుపు, &norm_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for NORM operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_చివర_EXT("NORM", kernel_name, కణుపు, "eps=%.6f", (ద్విగుణ) eps);
    తిరిగి kernel_ఫలితం;
}

తార్కిక ggml_et_op_l2_norm(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు) {
    ET_PERF_ప్రారంభం();

    ఒకవేళ (!dev_ctx || !కణుపు) {
        GGML_LOG_ERROR("ET: Invalid parameters for L2_NORM operation\n");
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (!కణుపు->src[0]) {
        GGML_LOG_ERROR("ET: L2_NORM operation missing required input\n");
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి అక్షరం * kernel_name;

    ఒకవేళ (కణుపు->type == GGML_TYPE_F32 && కణుపు->src[0]->type == GGML_TYPE_F32) {
        kernel_name = "l2_norm_f32";

    } లేకపోతే {
        GGML_LOG_ERROR("ET: L2_NORM operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(కణుపు->type),
                       ggml_type_name(కణుపు->src[0]->type));
        తిరిగి అబద్ధం;
    }

    దశాంశం eps;
    జ్ఞాపకం_నకలు(&eps, కణుపు->op_params, పరిమాణం(దశాంశం));

    ggml_et_l2_norm_params params;
    params.src0 = *కణుపు->src[0];  // F32 input tensor
    params.dst  = *కణుపు;          // F32 output tensor
    params.eps  = eps;            // Epsilon parameter for numerical stability

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    తార్కిక                    cpu_comparison_చురుకైన = అబద్ధం;
    ఒకవేళ (l2_norm_cpu_compare_config.enabled) {
        ఒకవేళ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, కణుపు, GGML_OP_L2_NORM)) {
            cpu_comparison_చురుకైన = నిజం;
        } లేకపోతే {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for L2_NORM operation\n");
        }
    }

    తార్కిక kernel_ఫలితం = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, పరిమాణం(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ఒకవేళ (cpu_comparison_చురుకైన) {
        ఒకవేళ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, కణుపు, &l2_norm_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for L2_NORM operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_చివర_EXT("L2_NORM", kernel_name, కణుపు, "eps=%.6f", (ద్విగుణ) eps);
    తిరిగి kernel_ఫలితం;
}

తార్కిక ggml_et_op_సమూహం_norm(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు) {
    ET_PERF_ప్రారంభం();

    ఒకవేళ (!dev_ctx || !కణుపు) {
        GGML_LOG_ERROR("ET: Invalid parameters for GROUP_NORM operation\n");
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (!కణుపు->src[0]) {
        GGML_LOG_ERROR("ET: GROUP_NORM operation missing required input\n");
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (కణుపు->type != GGML_TYPE_F32 || కణుపు->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: GROUP_NORM operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(కణుపు->type),
                       ggml_type_name(కణుపు->src[0]->type));
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి int32_t n_groups = ggml_get_op_params_i32(కణుపు, 0);
    దశాంశం         eps;
    జ్ఞాపకం_నకలు(&eps, (స్థిరరాశి దశాంశం *) కణుపు->op_params + 1, పరిమాణం(దశాంశం));

    ggml_et_సమూహం_norm_params params;
    params.src0     = *కణుపు->src[0];
    params.dst      = *కణుపు;
    params.n_groups = n_groups;
    params.eps      = eps;

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    తార్కిక                    cpu_comparison_చురుకైన = అబద్ధం;
    ఒకవేళ (సమూహం_norm_cpu_compare_config.enabled) {
        ఒకవేళ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, కణుపు, GGML_OP_సమూహం_NORM)) {
            cpu_comparison_చురుకైన = నిజం;
        } లేకపోతే {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for GROUP_NORM operation\n");
        }
    }

    తార్కిక kernel_ఫలితం = ggml_et_launch_kernel(dev_ctx, "group_norm_f32", &params, పరిమాణం(params), 0xFFFFFFFF);

    ఒకవేళ (cpu_comparison_చురుకైన) {
        ఒకవేళ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, కణుపు, &సమూహం_norm_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for GROUP_NORM operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_చివర_EXT("GROUP_NORM", "group_norm_f32", కణుపు, "eps=%.6f|n_groups=%d", (ద్విగుణ) eps, n_groups);
    తిరిగి kernel_ఫలితం;
}

తార్కిక ggml_et_op_im2col(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు) {
    ET_PERF_ప్రారంభం();

    ఒకవేళ (!dev_ctx || !కణుపు) {
        GGML_LOG_ERROR("ET: Invalid parameters for IM2COL operation\n");
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (!కణుపు->src[0] || !కణుపు->src[1]) {
        GGML_LOG_ERROR("ET: IM2COL operation missing required inputs\n");
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి తార్కిక supported_types =
        (కణుపు->type == GGML_TYPE_F32 && కణుపు->src[1]->type == GGML_TYPE_F32) ||
        (కణుపు->type == GGML_TYPE_F16 && (కణుపు->src[1]->type == GGML_TYPE_F16 || కణుపు->src[1]->type == GGML_TYPE_F32));

    ఒకవేళ (!supported_types) {
        GGML_LOG_ERROR("ET: IM2COL operation with unsupported types: dst=%s src1=%s\n", ggml_type_name(కణుపు->type),
                       ggml_type_name(కణుపు->src[1]->type));
        తిరిగి అబద్ధం;
    }

    ggml_et_im2col_params params;
    params.src0 = *కణుపు->src[0];
    params.src1 = *కణుపు->src[1];
    params.dst  = *కణుపు;

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    తార్కిక                    cpu_comparison_చురుకైన = అబద్ధం;
    ఒకవేళ (im2col_cpu_compare_config.enabled) {
        ఒకవేళ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, కణుపు, GGML_OP_IM2_నిలువు)) {
            cpu_comparison_చురుకైన = నిజం;
        } లేకపోతే {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for IM2COL operation\n");
        }
    }

    తార్కిక kernel_ఫలితం = ggml_et_launch_kernel(dev_ctx, "im2col", &params, పరిమాణం(params), 0xFFFFFFFF);

    ఒకవేళ (cpu_comparison_చురుకైన) {
        ఒకవేళ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, కణుపు, &im2col_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for IM2COL operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_చివర("IM2COL", "im2col", కణుపు);
    తిరిగి kernel_ఫలితం;
}

తార్కిక ggml_et_op_conv_2d(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు) {
    ET_PERF_ప్రారంభం();

    ఒకవేళ (!dev_ctx || !కణుపు) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (!కణుపు->src[0] || !కణుపు->src[1]) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (!కణుపు->దత్తాంశం || !కణుపు->src[0]->దత్తాంశం || !కణుపు->src[1]->దత్తాంశం) {
        తిరిగి అబద్ధం;
    }

    // Kernel constraints (mirror supports_op; recheck here as a guard).
    స్థిరరాశి ggml_tensor * flt = కణుపు->src[0];  // [Kw, Kh, Cin, Cout]
    స్థిరరాశి ggml_tensor * లోపల  = కణుపు->src[1];  // [W,  H,  Cin, N]
    ఒకవేళ (కణుపు->type != GGML_TYPE_F32 || flt->type != GGML_TYPE_F32 || లోపల->type != GGML_TYPE_F32) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి int32_t s0 = ggml_get_op_params_i32(కణుపు, 0);
    స్థిరరాశి int32_t s1 = ggml_get_op_params_i32(కణుపు, 1);
    స్థిరరాశి int32_t p0 = ggml_get_op_params_i32(కణుపు, 2);
    స్థిరరాశి int32_t p1 = ggml_get_op_params_i32(కణుపు, 3);
    స్థిరరాశి int32_t d0 = ggml_get_op_params_i32(కణుపు, 4);
    స్థిరరాశి int32_t d1 = ggml_get_op_params_i32(కణుపు, 5);

    ఒకవేళ (s0 < 1 || s1 < 1) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (d0 != 1 || d1 != 1) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (flt->ne[2] % 16 != 0 || flt->ne[3] % 16 != 0) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (లోపల->ne[3] != 1) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (కణుపు->ne[0] <= 0) {
        తిరిగి అబద్ధం;  // OW > 0 (any width OK; staging path handles non-16)
    }
    (శూన్య) p0;
    (శూన్య) p1;

    ggml_et_ద్విమాన_params params;
    params.src0 = *కణుపు->src[0];
    params.src1 = *కణుపు->src[1];
    params.dst  = *కణుపు;

    తార్కిక kernel_ఫలితం = ggml_et_launch_kernel(dev_ctx, "conv_2d_f32_me", &params, పరిమాణం(params), 0xFFFFFFFFu);

    ET_PERF_చివర("CONV_2D", "conv_2d_f32_me", కణుపు);
    తిరిగి kernel_ఫలితం;
}

తార్కిక ggml_et_op_softmax(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు) {
    ET_PERF_ప్రారంభం();

    ఒకవేళ (!dev_ctx || !కణుపు) {
        GGML_LOG_ERROR("ET: Invalid parameters for SOFTMAX operation\n");
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (!కణుపు->src[0]) {
        GGML_LOG_ERROR("ET: SOFTMAX operation missing required input\n");
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి అక్షరం * kernel_name;

    ఒకవేళ (కణుపు->type == GGML_TYPE_F32 && కణుపు->src[0]->type == GGML_TYPE_F32) {
        kernel_name = "softmax_f32";

    } లేకపోతే {
        GGML_LOG_ERROR("ET: SOFTMAX operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(కణుపు->type),
                       ggml_type_name(కణుపు->src[0]->type));
        తిరిగి అబద్ధం;
    }

    // Validate contiguity requirements
    ఒకవేళ (!ggml_is_contiguous(కణుపు)) {
        GGML_LOG_ERROR("ET: SOFTMAX operation requires contiguous destination tensor\n");
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (!ggml_is_contiguous(కణుపు->src[0])) {
        GGML_LOG_ERROR("ET: SOFTMAX operation requires contiguous source tensor\n");
        తిరిగి అబద్ధం;
    }

    // Check optional mask tensor
    ఒకవేళ (కణుపు->src[1]) {
        ఒకవేళ (కణుపు->src[1]->type != GGML_TYPE_F32) {
            GGML_LOG_ERROR("ET: SOFTMAX operation with unsupported mask type: %s (F32 required)\n",
                           ggml_type_name(కణుపు->src[1]->type));
            తిరిగి అబద్ధం;
        }
        ఒకవేళ (!ggml_is_contiguous(కణుపు->src[1])) {
            GGML_LOG_ERROR("ET: SOFTMAX operation requires contiguous mask tensor\n");
            తిరిగి అబద్ధం;
        }
    }

    // Check optional sinks tensor
    ఒకవేళ (కణుపు->src[2]) {
        ఒకవేళ (కణుపు->src[2]->type != GGML_TYPE_F32) {
            GGML_LOG_ERROR("ET: SOFTMAX operation with unsupported sinks type: %s (F32 required)\n",
                           ggml_type_name(కణుపు->src[2]->type));
            తిరిగి అబద్ధం;
        }
        ఒకవేళ (!ggml_is_contiguous(కణుపు->src[2])) {
            GGML_LOG_ERROR("ET: SOFTMAX operation requires contiguous sinks tensor\n");
            తిరిగి అబద్ధం;
        }
    }

    // Extract scale and max_bias from op_params
    దశాంశం scale    = 1.0f;
    దశాంశం max_bias = 0.0f;
    ఒకవేళ (కణుపు->op_params) {
        జ్ఞాపకం_నకలు(&scale, (స్థిరరాశి దశాంశం *) కణుపు->op_params + 0, పరిమాణం(దశాంశం));
        జ్ఞాపకం_నకలు(&max_bias, (స్థిరరాశి దశాంశం *) కణుపు->op_params + 1, పరిమాణం(దశాంశం));
    }

    ggml_et_softmax_params params;
    params.src0 = *కణుపు->src[0];                       // F32 input tensor
    ఒకవేళ (కణుపు->src[1]) {
        params.src1 = *కణుపు->src[1];                   // F32 mask tensor
    } లేకపోతే {
        memset(&params.src1, 0, పరిమాణం(params.src1));  // Zero if no mask
    }
    ఒకవేళ (కణుపు->src[2]) {
        params.src2 = *కణుపు->src[2];                   // F32 sinks tensor
    } లేకపోతే {
        memset(&params.src2, 0, పరిమాణం(params.src2));  // Zero if no sinks
    }
    params.dst      = *కణుపు;                           // F32 output tensor
    params.scale    = scale;                           // Scale factor
    params.max_bias = max_bias;                        // ALiBi bias

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    తార్కిక                    cpu_comparison_చురుకైన = అబద్ధం;
    ఒకవేళ (softmax_cpu_compare_config.enabled) {
        ఒకవేళ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, కణుపు, GGML_OP_SOFT_MAX)) {
            cpu_comparison_చురుకైన = నిజం;
        } లేకపోతే {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for SOFTMAX operation\n");
        }
    }

    తార్కిక kernel_ఫలితం = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, పరిమాణం(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ఒకవేళ (cpu_comparison_చురుకైన) {
        ఒకవేళ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, కణుపు, &softmax_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for SOFTMAX operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_చివర_EXT("SOFTMAX", kernel_name, కణుపు, "scale=%.6f|max_bias=%.6f|has_mask=%s", (ద్విగుణ) scale,
                    (ద్విగుణ) max_bias, కణుపు->src[1] ? "yes" : "no");
    తిరిగి kernel_ఫలితం;
}

తార్కిక ggml_et_op_flash_attn_ext(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు) {
    ET_PERF_ప్రారంభం();

    ఒకవేళ (!dev_ctx || !కణుపు) {
        GGML_LOG_ERROR("ET: Invalid parameters for FLASH_ATTN_EXT operation\n");
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (!కణుపు->src[0] || !కణుపు->src[1] || !కణుపు->src[2]) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT operation missing required inputs\n");
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (కణుపు->type != GGML_TYPE_F32 || కణుపు->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT requires F32 Q and dst, got dst=%s q=%s\n", ggml_type_name(కణుపు->type),
                       ggml_type_name(కణుపు->src[0]->type));
        తిరిగి అబద్ధం;
    }

    // K and V can be F16 or F32
    ఒకవేళ ((కణుపు->src[1]->type != GGML_TYPE_F32 && కణుపు->src[1]->type != GGML_TYPE_F16) ||
        (కణుపు->src[2]->type != GGML_TYPE_F32 && కణుపు->src[2]->type != GGML_TYPE_F16)) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT K/V must be F16 or F32, got k=%s v=%s\n", ggml_type_name(కణుపు->src[1]->type),
                       ggml_type_name(కణుపు->src[2]->type));
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (కణుపు->src[4] != శూన్యసూచిక) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT baseline kernel does not support sinks\n");
        తిరిగి అబద్ధం;
    }

    // Mask is optional; if present must be F16 or F32
    ఒకవేళ (కణుపు->src[3] != శూన్యసూచిక && కణుపు->src[3]->type != GGML_TYPE_F32 && కణుపు->src[3]->type != GGML_TYPE_F16) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT mask must be F16 or F32, got %s\n", ggml_type_name(కణుపు->src[3]->type));
        తిరిగి అబద్ధం;
    }

    // Q and dst must be row-contiguous F32
    ఒకవేళ (!ggml_is_contiguous_అడ్డువరుసలు(కణుపు) || !ggml_is_contiguous_అడ్డువరుసలు(కణుపు->src[0])) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT requires row-contiguous Q and dst\n");
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (కణుపు->nb[0] != పరిమాణం(దశాంశం) || కణుపు->src[0]->nb[0] != పరిమాణం(దశాంశం)) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT requires contiguous F32 rows for Q and dst\n");
        తిరిగి అబద్ధం;
    }

    // K/V must have element-sized stride in dim 0
    స్థిరరాశి size_t k_elem = కణుపు->src[1]->type == GGML_TYPE_F16 ? 2 : 4;
    స్థిరరాశి size_t v_elem = కణుపు->src[2]->type == GGML_TYPE_F16 ? 2 : 4;
    ఒకవేళ (కణుపు->src[1]->nb[0] != k_elem || కణుపు->src[2]->nb[0] != v_elem) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT K/V must have element-sized stride in dim 0\n");
        తిరిగి అబద్ధం;
    }

    దశాంశం scale         = 1.0f;
    దశాంశం max_bias      = 0.0f;
    దశాంశం logit_softcap = 0.0f;
    జ్ఞాపకం_నకలు(&scale, (స్థిరరాశి దశాంశం *) కణుపు->op_params + 0, పరిమాణం(scale));
    జ్ఞాపకం_నకలు(&max_bias, (స్థిరరాశి దశాంశం *) కణుపు->op_params + 1, పరిమాణం(max_bias));
    జ్ఞాపకం_నకలు(&logit_softcap, (స్థిరరాశి దశాంశం *) కణుపు->op_params + 2, పరిమాణం(logit_softcap));

    ఒకవేళ (max_bias != 0.0f || logit_softcap != 0.0f) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT baseline kernel does not support max_bias or logit_softcap\n");
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి ggml_prec prec = ggml_flash_attn_ext_get_prec(కణుపు);
    ఒకవేళ (prec != GGML_PREC_F32 && prec != GGML_PREC_DEFAULT) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT baseline kernel only supports F32 precision\n");
        తిరిగి అబద్ధం;
    }

    // dk must match between Q and K; dv must match between V and dst
    ఒకవేళ (కణుపు->src[0]->ne[0] != కణుపు->src[1]->ne[0]) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT dk mismatch: Q=%lld K=%lld\n", (దీర్ఘ దీర్ఘ) కణుపు->src[0]->ne[0],
                       (దీర్ఘ దీర్ఘ) కణుపు->src[1]->ne[0]);
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (కణుపు->src[2]->ne[0] != కణుపు->ne[0]) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT dv mismatch: V=%lld dst=%lld\n", (దీర్ఘ దీర్ఘ) కణుపు->src[2]->ne[0],
                       (దీర్ఘ దీర్ఘ) కణుపు->ne[0]);
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (కణుపు->src[2]->ne[0] > 512) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT dv=%lld exceeds maximum 512\n", (దీర్ఘ దీర్ఘ) కణుపు->src[2]->ne[0]);
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (కణుపు->src[0]->ne[0] > 512) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT dk=%lld exceeds maximum 512\n", (దీర్ఘ దీర్ఘ) కణుపు->src[0]->ne[0]);
        తిరిగి అబద్ధం;
    }

    // GQA: n_head_q must be a multiple of n_head_kv
    స్థిరరాశి int64_t nhq = కణుపు->src[0]->ne[2];
    స్థిరరాశి int64_t nhk = కణుపు->src[1]->ne[2];
    ఒకవేళ (nhq % nhk != 0) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT n_head_q (%lld) not divisible by n_head_kv (%lld)\n", (దీర్ఘ దీర్ఘ) nhq,
                       (దీర్ఘ దీర్ఘ) nhk);
        తిరిగి అబద్ధం;
    }

    // K and V must have matching sequence length, heads, and batch dims
    ఒకవేళ (కణుపు->src[1]->ne[1] != కణుపు->src[2]->ne[1] || కణుపు->src[1]->ne[2] != కణుపు->src[2]->ne[2] ||
        కణుపు->src[1]->ne[3] != కణుపు->src[2]->ne[3]) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT K/V shape mismatch\n");
        తిరిగి అబద్ధం;
    }

    // dst layout checks: [dv, nhq, nq, no]
    ఒకవేళ (కణుపు->src[0]->ne[1] != కణుపు->ne[2] || కణుపు->src[0]->ne[2] != కణుపు->ne[1] ||
        కణుపు->src[0]->ne[3] != కణుపు->ne[3]) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT dst shape mismatch\n");
        తిరిగి అబద్ధం;
    }

    // Batch dims: Q batch must match K batch
    ఒకవేళ (కణుపు->src[0]->ne[3] != కణుపు->src[1]->ne[3]) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT batch dimension mismatch\n");
        తిరిగి అబద్ధం;
    }

    ggml_et_flash_attn_ext_params params;
    memset(&params, 0, పరిమాణం(params));
    params.src0 = *కణుపు->src[0];
    params.src1 = *కణుపు->src[1];
    params.src2 = *కణుపు->src[2];
    ఒకవేళ (కణుపు->src[3] != శూన్యసూచిక) {
        params.ముసుగు     = *కణుపు->src[3];
        params.has_ముసుగు = 1;
    }
    params.dst   = *కణుపు;
    params.scale = scale;

    // Use matrix engine kernel when K/V are F16 and dk is a multiple of 32
    స్థిరరాశి అక్షరం * kernel_name;
    ఒకవేళ (కణుపు->src[1]->type == GGML_TYPE_F16 && కణుపు->src[2]->type == GGML_TYPE_F16 && (కణుపు->src[0]->ne[0] % 32) == 0) {
        kernel_name = "flash_attn_ext_f16_me";
    } లేకపోతే {
        kernel_name = "flash_attn_ext_f32";
    }

    స్థిరరాశి తార్కిక kernel_ఫలితం = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, పరిమాణం(params), 0xFFFFFFFF);

    ET_PERF_చివర_EXT("FLASH_ATTN_EXT", kernel_name, కణుపు, "scale=%.6f", (ద్విగుణ) scale);
    తిరిగి kernel_ఫలితం;
}

తార్కిక ggml_et_op_get_అడ్డువరుసలు(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు) {
    ET_PERF_ప్రారంభం();

    ఒకవేళ (!dev_ctx || !కణుపు) {
        GGML_LOG_ERROR("ET: Invalid parameters for GET_ROWS operation\n");
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (!కణుపు->src[0] || !కణుపు->src[1]) {
        GGML_LOG_ERROR("ET: GET_ROWS operation missing required inputs\n");
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి అక్షరం * kernel_name;

    ఒకవేళ (కణుపు->type == GGML_TYPE_F32 && కణుపు->src[1]->type == GGML_TYPE_I32 &&
        (కణుపు->src[0]->type == GGML_TYPE_F32 || కణుపు->src[0]->type == GGML_TYPE_F16 ||
         కణుపు->src[0]->type == GGML_TYPE_Q4_0 || కణుపు->src[0]->type == GGML_TYPE_Q8_0 ||
         కణుపు->src[0]->type == GGML_TYPE_Q4_K)) {
        kernel_name = "get_rows_f32";

    } లేకపోతే {
        GGML_LOG_ERROR("ET: GET_ROWS operation with unsupported types: dst=%s src0=%s src1=%s\n",
                       ggml_type_name(కణుపు->type), ggml_type_name(కణుపు->src[0]->type),
                       ggml_type_name(కణుపు->src[1]->type));
        తిరిగి అబద్ధం;
    }

    // Validate contiguity requirements
    ఒకవేళ (!ggml_is_contiguous(కణుపు)) {
        GGML_LOG_ERROR("ET: GET_ROWS operation requires contiguous destination tensor\n");
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (!ggml_is_contiguous(కణుపు->src[0])) {
        GGML_LOG_ERROR("ET: GET_ROWS operation requires contiguous data tensor\n");
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (!ggml_is_contiguous(కణుపు->src[1])) {
        GGML_LOG_ERROR("ET: GET_ROWS operation requires contiguous indices tensor\n");
        తిరిగి అబద్ధం;
    }

    // Validate dimension constraints from ggml implementation
    ఒకవేళ (కణుపు->src[0]->ne[2] != కణుపు->src[1]->ne[1] || కణుపు->src[1]->ne[3] != 1) {
        GGML_LOG_ERROR(
            "ET: GET_ROWS operation dimension constraint failed: src0.ne[2]=%lld != src1.ne[1]=%lld or src1.ne[3]=%lld "
            "!= 1\n",
            (దీర్ఘ దీర్ఘ) కణుపు->src[0]->ne[2], (దీర్ఘ దీర్ఘ) కణుపు->src[1]->ne[1], (దీర్ఘ దీర్ఘ) కణుపు->src[1]->ne[3]);
        తిరిగి అబద్ధం;
    }

    ggml_et_get_అడ్డువరుసలు_params params;
    params.src0 = *కణుపు->src[0];  // Data tensor (F32 or Q8_0)
    params.src1 = *కణుపు->src[1];  // Indices tensor (I32)
    params.dst  = *కణుపు;          // Output tensor (F32)

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    తార్కిక                    cpu_comparison_చురుకైన = అబద్ధం;
    ఒకవేళ (get_అడ్డువరుసలు_cpu_compare_config.enabled) {
        ఒకవేళ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, కణుపు, GGML_OP_GET_అడ్డువరుసలు)) {
            cpu_comparison_చురుకైన = నిజం;
        } లేకపోతే {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for GET_ROWS operation\n");
        }
    }

    తార్కిక kernel_ఫలితం = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, పరిమాణం(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ఒకవేళ (cpu_comparison_చురుకైన) {
        ఒకవేళ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, కణుపు, &get_అడ్డువరుసలు_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for GET_ROWS operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_చివర("GET_ROWS", kernel_name, కణుపు);
    తిరిగి kernel_ఫలితం;
}

తార్కిక ggml_et_op_cont(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు) {
    ET_PERF_ప్రారంభం();

    // Validate source tensor exists
    ఒకవేళ (!కణుపు->src[0]) {
        GGML_LOG_ERROR("ET: CONT operation missing source tensor\n");
        తిరిగి అబద్ధం;
    }

    // Validate types match (input and output must be same type)
    ఒకవేళ (కణుపు->type != కణుపు->src[0]->type) {
        GGML_LOG_ERROR("ET: CONT operation type mismatch: src=%s dst=%s\n", ggml_type_name(కణుపు->src[0]->type),
                       ggml_type_name(కణుపు->type));
        తిరిగి అబద్ధం;
    }

    // Validate supported types
    ఒకవేళ (కణుపు->type != GGML_TYPE_F32 && కణుపు->type != GGML_TYPE_F16) {
        GGML_LOG_ERROR("ET: CONT operation unsupported type: %s (only F32 and F16 supported)\n",
                       ggml_type_name(కణుపు->type));
        తిరిగి అబద్ధం;
    }

    // Validate contiguity - output must be contiguous, input can be non-contiguous
    ఒకవేళ (!ggml_is_contiguous(కణుపు)) {
        GGML_LOG_ERROR("ET: CONT operation requires contiguous output tensor\n");
        తిరిగి అబద్ధం;
    }

    // Select kernel based on type
    స్థిరరాశి అక్షరం * kernel_name;
    ఒకవేళ (కణుపు->type == GGML_TYPE_F32) {
        kernel_name = "cont_f32";
    } లేకపోతే ఒకవేళ (కణుపు->type == GGML_TYPE_F16) {
        kernel_name = "cont_f16";
    } లేకపోతే {
        GGML_LOG_ERROR("ET: CONT operation with unsupported type: %s\n", ggml_type_name(కణుపు->type));
        తిరిగి అబద్ధం;
    }

    ggml_et_cont_params params;
    params.src0 = *కణుపు->src[0];  // Input tensor (potentially non-contiguous)
    params.dst  = *కణుపు;          // Output tensor (contiguous)

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    తార్కిక                    cpu_comparison_చురుకైన = అబద్ధం;
    ఒకవేళ (cont_cpu_compare_config.enabled) {
        ఒకవేళ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, కణుపు, GGML_OP_CONT)) {
            cpu_comparison_చురుకైన = నిజం;
        } లేకపోతే {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for CONT operation\n");
        }
    }

    తార్కిక kernel_ఫలితం = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, పరిమాణం(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ఒకవేళ (cpu_comparison_చురుకైన) {
        ఒకవేళ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, కణుపు, &cont_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for CONT operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_చివర("CONT", kernel_name, కణుపు);
    తిరిగి kernel_ఫలితం;
}

తార్కిక ggml_et_op_cumsum(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు) {
    ET_PERF_ప్రారంభం();

    ఒకవేళ (!dev_ctx || !కణుపు || !కణుపు->src[0]) {
        GGML_LOG_ERROR("ET: Invalid parameters for CUMSUM operation\n");
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (కణుపు->type != GGML_TYPE_F32 || కణుపు->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: CUMSUM operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(కణుపు->type),
                       ggml_type_name(కణుపు->src[0]->type));
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి అక్షరం * kernel_name = "cumsum_f32";

    ggml_et_cumsum_params params;
    params.src0 = *కణుపు->src[0];
    params.dst  = *కణుపు;

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    తార్కిక                    cpu_comparison_చురుకైన = అబద్ధం;
    ఒకవేళ (cumsum_cpu_compare_config.enabled) {
        ఒకవేళ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, కణుపు, GGML_OP_CUMSUM)) {
            cpu_comparison_చురుకైన = నిజం;
        } లేకపోతే {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for CUMSUM operation\n");
        }
    }

    తార్కిక kernel_ఫలితం = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, పరిమాణం(params), 0xFFFFFFFF);

    ఒకవేళ (cpu_comparison_చురుకైన) {
        ఒకవేళ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, కణుపు, &cumsum_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for CUMSUM operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_చివర("CUMSUM", kernel_name, కణుపు);
    తిరిగి kernel_ఫలితం;
}

తార్కిక ggml_et_op_cpy(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు) {
    ET_PERF_ప్రారంభం();

    // CPY copies data from src[0] into the layout of dst (which matches src[1])
    // For same-type with contiguous dst, this is identical to CONT
    ఒకవేళ (!కణుపు->src[0]) {
        GGML_LOG_ERROR("ET: CPY operation missing source tensor\n");
        తిరిగి అబద్ధం;
    }

    // Scalar / zero-element special path: if any dimension is 0, nothing to copy
    స్థిరరాశి int64_t nelements = కణుపు->ne[0] * కణుపు->ne[1] * కణుపు->ne[2] * కణుపు->ne[3];
    ఒకవేళ (nelements == 0) {
        GGML_LOG_DEBUG("ET: CPY no-op (zero elements): ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]\n",
                       కణుపు->ne[0], కణుపు->ne[1], కణుపు->ne[2], కణుపు->ne[3]);
        ET_PERF_చివర("CPY", "noop", కణుపు);
        తిరిగి నిజం;
    }

    // Only F32 and F16 supported for dst
    ఒకవేళ (కణుపు->type != GGML_TYPE_F32 && కణుపు->type != GGML_TYPE_F16) {
        GGML_LOG_ERROR("ET: CPY unsupported dst type: %s\n", ggml_type_name(కణుపు->type));
        తిరిగి అబద్ధం;
    }

    // Select kernel based on src/dst type combination
    స్థిరరాశి అక్షరం * kernel_name;
    ఒకవేళ (కణుపు->src[0]->type == GGML_TYPE_F32 && కణుపు->type == GGML_TYPE_F32) {
        kernel_name = "cont_f32";
    } లేకపోతే ఒకవేళ (కణుపు->src[0]->type == GGML_TYPE_F16 && కణుపు->type == GGML_TYPE_F16) {
        kernel_name = "cont_f16";
    } లేకపోతే ఒకవేళ (కణుపు->src[0]->type == GGML_TYPE_F32 && కణుపు->type == GGML_TYPE_F16) {
        kernel_name = "cpy_f32_f16";
    } లేకపోతే {
        GGML_LOG_ERROR("ET: CPY unsupported type combination: src=%s dst=%s\n", ggml_type_name(కణుపు->src[0]->type),
                       ggml_type_name(కణుపు->type));
        తిరిగి అబద్ధం;
    }

    ggml_et_cont_params params;
    params.src0 = *కణుపు->src[0];
    params.dst  = *కణుపు;

    // CPU comparison for debugging
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    తార్కిక                    cpu_comparison_చురుకైన = అబద్ధం;
    ఒకవేళ (cont_cpu_compare_config.enabled) {
        ఒకవేళ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, కణుపు, GGML_OP_CPY)) {
            cpu_comparison_చురుకైన = నిజం;
        } లేకపోతే {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for CPY operation\n");
        }
    }

    తార్కిక kernel_ఫలితం = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, పరిమాణం(params), 0xFFFFFFFF);

    ఒకవేళ (cpu_comparison_చురుకైన) {
        ఒకవేళ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, కణుపు, &cont_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for CPY operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_చివర("CPY", kernel_name, కణుపు);
    తిరిగి kernel_ఫలితం;
}

తార్కిక ggml_et_op_concat(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు) {
    ET_PERF_ప్రారంభం();

    ఒకవేళ (!dev_ctx || !కణుపు) {
        GGML_LOG_ERROR("ET: Invalid parameters for CONCAT operation\n");
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (!కణుపు->src[0] || !కణుపు->src[1]) {
        GGML_LOG_ERROR("ET: CONCAT operation missing required inputs\n");
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి అక్షరం * kernel_name;

    ఒకవేళ (కణుపు->type == GGML_TYPE_F32 && కణుపు->src[0]->type == GGML_TYPE_F32 && కణుపు->src[1]->type == GGML_TYPE_F32) {
        kernel_name = "concat_f32";

    } లేకపోతే {
        GGML_LOG_ERROR("ET: CONCAT operation with unsupported types: dst=%s src0=%s src1=%s\n",
                       ggml_type_name(కణుపు->type), ggml_type_name(కణుపు->src[0]->type),
                       ggml_type_name(కణుపు->src[1]->type));
        తిరిగి అబద్ధం;
    }

    int32_t dim;
    జ్ఞాపకం_నకలు(&dim, కణుపు->op_params, పరిమాణం(int32_t));

    ggml_et_concat_params params;
    params.src0 = *కణుపు->src[0];
    params.src1 = *కణుపు->src[1];
    params.dst  = *కణుపు;
    params.dim  = dim;

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    తార్కిక                    cpu_comparison_చురుకైన = అబద్ధం;
    ఒకవేళ (concat_cpu_compare_config.enabled) {
        ఒకవేళ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, కణుపు, GGML_OP_CONCAT)) {
            cpu_comparison_చురుకైన = నిజం;
        } లేకపోతే {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for CONCAT operation\n");
        }
    }

    తార్కిక kernel_ఫలితం = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, పరిమాణం(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ఒకవేళ (cpu_comparison_చురుకైన) {
        ఒకవేళ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, కణుపు, &concat_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for CONCAT operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_చివర_EXT("CONCAT", kernel_name, కణుపు, "dim=%d", dim);
    తిరిగి kernel_ఫలితం;
}

తార్కిక ggml_et_op_repeat(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు) {
    ET_PERF_ప్రారంభం();

    ఒకవేళ (!dev_ctx || !కణుపు) {
        GGML_LOG_ERROR("ET: Invalid parameters for REPEAT operation\n");
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (!కణుపు->src[0]) {
        GGML_LOG_ERROR("ET: REPEAT operation missing required input\n");
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి అక్షరం * kernel_name;

    ఒకవేళ (కణుపు->type == GGML_TYPE_F32 && కణుపు->src[0]->type == GGML_TYPE_F32) {
        // No-op REPEAT (every repeat factor is 1): the output is just a copy
        // of the input. Route to cont_f32, whose contiguous fast path handles
        // arbitrary sizes (including those rejected by repeat_f32's gate,
        // e.g. ne[0]=1).
        ఒకవేళ (ggml_are_same_shape(కణుపు->src[0], కణుపు)) {
            kernel_name = "cont_f32";
        } లేకపోతే {
            kernel_name = "repeat_f32";
        }

    } లేకపోతే {
        GGML_LOG_ERROR("ET: REPEAT operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(కణుపు->type),
                       ggml_type_name(కణుపు->src[0]->type));
        తిరిగి అబద్ధం;
    }

    // ggml_et_cont_params and ggml_et_repeat_params have identical layouts
    // (just src0 + dst), so the same payload works for either kernel.
    ggml_et_repeat_params params;
    params.src0 = *కణుపు->src[0];
    params.dst  = *కణుపు;

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    తార్కిక                    cpu_comparison_చురుకైన = అబద్ధం;
    ఒకవేళ (repeat_cpu_compare_config.enabled) {
        ఒకవేళ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, కణుపు, GGML_OP_REPEAT)) {
            cpu_comparison_చురుకైన = నిజం;
        } లేకపోతే {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for REPEAT operation\n");
        }
    }

    తార్కిక kernel_ఫలితం = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, పరిమాణం(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ఒకవేళ (cpu_comparison_చురుకైన) {
        ఒకవేళ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, కణుపు, &repeat_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for REPEAT operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_చివర("REPEAT", kernel_name, కణుపు);
    తిరిగి kernel_ఫలితం;
}

తార్కిక ggml_et_op_ssm_conv(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు) {
    ET_PERF_ప్రారంభం();

    ఒకవేళ (!dev_ctx || !కణుపు || !కణుపు->src[0] || !కణుపు->src[1]) {
        GGML_LOG_ERROR("ET: Invalid parameters for SSM_CONV operation\n");
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (కణుపు->type != GGML_TYPE_F32 || కణుపు->src[0]->type != GGML_TYPE_F32 || కణుపు->src[1]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: SSM_CONV operation with unsupported types: dst=%s src0=%s src1=%s\n",
                       ggml_type_name(కణుపు->type), ggml_type_name(కణుపు->src[0]->type),
                       ggml_type_name(కణుపు->src[1]->type));
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి అక్షరం * kernel_name = "ssm_conv_f32";

    ggml_et_ssm_conv_params params;
    params.src0 = *కణుపు->src[0];
    params.src1 = *కణుపు->src[1];
    params.dst  = *కణుపు;

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    తార్కిక                    cpu_comparison_చురుకైన = అబద్ధం;
    ఒకవేళ (ssm_conv_cpu_compare_config.enabled) {
        ఒకవేళ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, కణుపు, GGML_OP_SSM_CONV)) {
            cpu_comparison_చురుకైన = నిజం;
        } లేకపోతే {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for SSM_CONV operation\n");
        }
    }

    తార్కిక kernel_ఫలితం = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, పరిమాణం(params), 0xFFFFFFFF);

    ఒకవేళ (cpu_comparison_చురుకైన) {
        ఒకవేళ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, కణుపు, &ssm_conv_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for SSM_CONV operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_చివర("SSM_CONV", kernel_name, కణుపు);
    తిరిగి kernel_ఫలితం;
}

తార్కిక ggml_et_op_ssm_పరిశీలన(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు) {
    ET_PERF_ప్రారంభం();

    ఒకవేళ (!dev_ctx || !కణుపు) {
        GGML_LOG_ERROR("ET: Invalid parameters for SSM_SCAN operation\n");
        తిరిగి అబద్ధం;
    }

    కొరకు (పూర్ణాంకం i = 0; i < 7; ++i) {
        ఒకవేళ (!కణుపు->src[i]) {
            GGML_LOG_ERROR("ET: SSM_SCAN missing required input %d\n", i);
            తిరిగి అబద్ధం;
        }
    }

    ఒకవేళ (కణుపు->type != GGML_TYPE_F32 || కణుపు->src[0]->type != GGML_TYPE_F32 || కణుపు->src[1]->type != GGML_TYPE_F32 ||
        కణుపు->src[2]->type != GGML_TYPE_F32 || కణుపు->src[3]->type != GGML_TYPE_F32 ||
        కణుపు->src[4]->type != GGML_TYPE_F32 || కణుపు->src[5]->type != GGML_TYPE_F32 ||
        కణుపు->src[6]->type != GGML_TYPE_I32) {
        GGML_LOG_ERROR("ET: SSM_SCAN operation with unsupported types\n");
        తిరిగి అబద్ధం;
    }

    ggml_et_ssm_పరిశీలన_params params;
    params.src0 = *కణుపు->src[0];
    params.src1 = *కణుపు->src[1];
    params.src2 = *కణుపు->src[2];
    params.src3 = *కణుపు->src[3];
    params.src4 = *కణుపు->src[4];
    params.src5 = *కణుపు->src[5];
    params.src6 = *కణుపు->src[6];
    params.dst  = *కణుపు;
    params.K    = ggml_get_op_params_i32(కణుపు, 0);

    తార్కిక kernel_ఫలితం = ggml_et_launch_kernel(dev_ctx, "ssm_scan_f32", &params, పరిమాణం(params), 0xFFFFFFFF);

    ET_PERF_చివర("SSM_SCAN", "ssm_scan_f32", కణుపు);
    తిరిగి kernel_ఫలితం;
}

తార్కిక ggml_et_op_rwkv_wkv6(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు) {
    ET_PERF_ప్రారంభం();

    ఒకవేళ (!dev_ctx || !కణుపు) {
        GGML_LOG_ERROR("ET: Invalid parameters for RWKV_WKV6 operation\n");
        తిరిగి అబద్ధం;
    }

    // Validate all 6 source tensors exist
    కొరకు (పూర్ణాంకం i = 0; i <= 5; i++) {
        ఒకవేళ (!కణుపు->src[i]) {
            GGML_LOG_ERROR("ET: RWKV_WKV6 operation missing src[%d]\n", i);
            తిరిగి అబద్ధం;
        }
    }

    ఒకవేళ (కణుపు->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: RWKV_WKV6 only supports F32, got %s\n", ggml_type_name(కణుపు->type));
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి అక్షరం * kernel_name = "rwkv_wkv6_f32";

    స్థిరరాశి int64_t S      = కణుపు->src[0]->ne[0];  // head_size
    స్థిరరాశి int64_t H      = కణుపు->src[0]->ne[1];  // num heads
    స్థిరరాశి int64_t T      = కణుపు->src[1]->ne[2];  // num tokens
    స్థిరరాశి int64_t n_seqs = కణుపు->src[5]->ne[1];  // num sequences
    స్థిరరాశి int64_t C      = S * H;

    ggml_et_rwkv_wkv6_params params;
    params.k        = (దశాంశం *) కణుపు->src[0]->దత్తాంశం;
    params.v        = (దశాంశం *) కణుపు->src[1]->దత్తాంశం;
    params.r        = (దశాంశం *) కణుపు->src[2]->దత్తాంశం;
    params.tf       = (దశాంశం *) కణుపు->src[3]->దత్తాంశం;
    params.td       = (దశాంశం *) కణుపు->src[4]->దత్తాంశం;
    params.state_లోపల = (దశాంశం *) కణుపు->src[5]->దత్తాంశం;
    params.dst      = (దశాంశం *) కణుపు->దత్తాంశం;
    params.C        = (int32_t) C;
    params.H        = (int32_t) H;
    params.S        = (int32_t) S;
    params.T        = (int32_t) T;
    params.n_seqs   = (int32_t) n_seqs;

    // Phase 1: Initialize CPU comparison context
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    తార్కిక                    cpu_comparison_చురుకైన = అబద్ధం;
    ఒకవేళ (rwkv_wkv6_cpu_compare_config.enabled) {
        ఒకవేళ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, కణుపు, GGML_OP_RWKV_WKV6)) {
            cpu_comparison_చురుకైన = నిజం;
        } లేకపోతే {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for RWKV_WKV6 operation\n");
        }
    }

    తార్కిక kernel_ఫలితం = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, పరిమాణం(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare
    ఒకవేళ (cpu_comparison_చురుకైన) {
        ఒకవేళ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, కణుపు, &rwkv_wkv6_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for RWKV_WKV6 operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_చివర_EXT("RWKV_WKV6", kernel_name, కణుపు, "S=%d H=%d T=%d n_seqs=%d", (పూర్ణాంకం) S, (పూర్ణాంకం) H, (పూర్ణాంకం) T,
                    (పూర్ణాంకం) n_seqs);
    తిరిగి kernel_ఫలితం;
}

తార్కిక ggml_et_op_rwkv_wkv7(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు) {
    ET_PERF_ప్రారంభం();

    ఒకవేళ (!dev_ctx || !కణుపు) {
        GGML_LOG_ERROR("ET: Invalid parameters for RWKV_WKV7 operation\n");
        తిరిగి అబద్ధం;
    }

    // Validate all 7 source tensors exist
    కొరకు (పూర్ణాంకం i = 0; i <= 6; i++) {
        ఒకవేళ (!కణుపు->src[i]) {
            GGML_LOG_ERROR("ET: RWKV_WKV7 operation missing src[%d]\n", i);
            తిరిగి అబద్ధం;
        }
    }

    ఒకవేళ (కణుపు->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: RWKV_WKV7 only supports F32, got %s\n", ggml_type_name(కణుపు->type));
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి అక్షరం * kernel_name = "rwkv_wkv7_f32";

    స్థిరరాశి int64_t S      = కణుపు->src[2]->ne[0];  // head_size
    స్థిరరాశి int64_t H      = కణుపు->src[2]->ne[1];  // num heads
    స్థిరరాశి int64_t T      = కణుపు->src[1]->ne[2];  // num tokens
    స్థిరరాశి int64_t n_seqs = కణుపు->src[6]->ne[1];  // num sequences
    స్థిరరాశి int64_t C      = S * H;

    ggml_et_rwkv_wkv7_params params;
    params.r        = (దశాంశం *) కణుపు->src[0]->దత్తాంశం;
    params.w        = (దశాంశం *) కణుపు->src[1]->దత్తాంశం;
    params.k        = (దశాంశం *) కణుపు->src[2]->దత్తాంశం;
    params.v        = (దశాంశం *) కణుపు->src[3]->దత్తాంశం;
    params.a        = (దశాంశం *) కణుపు->src[4]->దత్తాంశం;
    params.b        = (దశాంశం *) కణుపు->src[5]->దత్తాంశం;
    params.state_లోపల = (దశాంశం *) కణుపు->src[6]->దత్తాంశం;
    params.dst      = (దశాంశం *) కణుపు->దత్తాంశం;
    params.C        = (int32_t) C;
    params.H        = (int32_t) H;
    params.S        = (int32_t) S;
    params.T        = (int32_t) T;
    params.n_seqs   = (int32_t) n_seqs;

    // Phase 1: Initialize CPU comparison context
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    తార్కిక                    cpu_comparison_చురుకైన = అబద్ధం;
    ఒకవేళ (rwkv_wkv7_cpu_compare_config.enabled) {
        ఒకవేళ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, కణుపు, GGML_OP_RWKV_WKV7)) {
            cpu_comparison_చురుకైన = నిజం;
        } లేకపోతే {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for RWKV_WKV7 operation\n");
        }
    }

    తార్కిక kernel_ఫలితం = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, పరిమాణం(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare
    ఒకవేళ (cpu_comparison_చురుకైన) {
        ఒకవేళ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, కణుపు, &rwkv_wkv7_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for RWKV_WKV7 operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_చివర_EXT("RWKV_WKV7", kernel_name, కణుపు, "S=%d H=%d T=%d n_seqs=%d", (పూర్ణాంకం) S, (పూర్ణాంకం) H, (పూర్ణాంకం) T,
                    (పూర్ణాంకం) n_seqs);
    తిరిగి kernel_ఫలితం;
}

స్థిర ggml_et_cpu_compare_config gated_delta_net_cpu_compare_config = {
    /* .enabled = */ అబద్ధం,
    /* .use_cpu_result = */ అబద్ధం,
    /* .log_differences = */ నిజం,
    /* .tolerance = */ 1e-4f,
    /* .max_log_elements = */ 4096
};

తార్కిక ggml_et_op_gated_delta_net(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు) {
    ET_PERF_ప్రారంభం();

    ఒకవేళ (!dev_ctx || !కణుపు) {
        GGML_LOG_ERROR("ET: Invalid parameters for GATED_DELTA_NET operation\n");
        తిరిగి అబద్ధం;
    }

    // Validate all 6 source tensors exist
    కొరకు (పూర్ణాంకం i = 0; i <= 5; i++) {
        ఒకవేళ (!కణుపు->src[i]) {
            GGML_LOG_ERROR("ET: GATED_DELTA_NET operation missing src[%d]\n", i);
            తిరిగి అబద్ధం;
        }
    }

    ఒకవేళ (కణుపు->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: GATED_DELTA_NET only supports F32, got %s\n", ggml_type_name(కణుపు->type));
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి అక్షరం * kernel_name = "gated_delta_net_f32";

    స్థిరరాశి ggml_tensor * src_q     = కణుపు->src[0];
    స్థిరరాశి ggml_tensor * src_k     = కణుపు->src[1];
    స్థిరరాశి ggml_tensor * src_v     = కణుపు->src[2];
    స్థిరరాశి ggml_tensor * src_g     = కణుపు->src[3];
    స్థిరరాశి ggml_tensor * src_beta  = కణుపు->src[4];
    స్థిరరాశి ggml_tensor * src_state = కణుపు->src[5];

    స్థిరరాశి int64_t S_v      = src_v->ne[0];
    స్థిరరాశి int64_t H        = src_v->ne[1];
    స్థిరరాశి int64_t n_టోకెన్లు = src_v->ne[2];
    స్థిరరాశి int64_t n_seqs   = src_v->ne[3];
    స్థిరరాశి int64_t H_q      = src_q->ne[1];
    స్థిరరాశి int64_t H_k      = src_k->ne[1];
    స్థిరరాశి int64_t n_seqs_q = src_q->ne[3];
    స్థిరరాశి int64_t n_seqs_k = src_k->ne[3];

    ggml_et_gated_delta_net_params params;
    params.q        = *src_q;
    params.k        = *src_k;
    params.v        = *src_v;
    params.g        = *src_g;
    params.beta     = *src_beta;
    params.state_లోపల = *src_state;
    params.dst      = *కణుపు;
    params.S_v      = (int32_t) S_v;
    params.H        = (int32_t) H;
    params.H_q      = (int32_t) H_q;
    params.H_k      = (int32_t) H_k;
    params.n_టోకెన్లు = (int32_t) n_టోకెన్లు;
    params.n_seqs   = (int32_t) n_seqs;
    params.n_seqs_q = (int32_t) n_seqs_q;
    params.n_seqs_k = (int32_t) n_seqs_k;
    params.kda      = (src_g->ne[0] == S_v) ? 1 : 0;
    params.K        = ggml_get_op_params_i32(కణుపు, 0);
    params.scale    = 1.0f / sqrtf((దశాంశం) S_v);

    // CPU comparison for debugging
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    తార్కిక                    cpu_comparison_చురుకైన = అబద్ధం;
    ఒకవేళ (gated_delta_net_cpu_compare_config.enabled) {
        ఒకవేళ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, కణుపు, GGML_OP_GATED_DELTA_NET)) {
            cpu_comparison_చురుకైన = నిజం;
        } లేకపోతే {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for GATED_DELTA_NET operation\n");
        }
    }

    తార్కిక kernel_ఫలితం = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, పరిమాణం(params), 0xFFFFFFFF);

    ఒకవేళ (cpu_comparison_చురుకైన) {
        ఒకవేళ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, కణుపు, &gated_delta_net_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for GATED_DELTA_NET operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_చివర_EXT("GATED_DELTA_NET", kernel_name, కణుపు, "S_v=%d H=%d n_tokens=%d n_seqs=%d kda=%d", (పూర్ణాంకం) S_v,
                    (పూర్ణాంకం) H, (పూర్ణాంకం) n_టోకెన్లు, (పూర్ణాంకం) n_seqs, params.kda);
    తిరిగి kernel_ఫలితం;
}

తార్కిక ggml_et_op_సమితి_అడ్డువరుసలు(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు) {
    ET_PERF_ప్రారంభం();

    ఒకవేళ (!dev_ctx || !కణుపు) {
        GGML_LOG_ERROR("ET: Invalid parameters for SET_ROWS operation\n");
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (!కణుపు->src[0] || !కణుపు->src[1] || !కణుపు->src[2]) {
        GGML_LOG_ERROR(
            "ET: SET_ROWS operation missing required inputs (needs src[0]=base, src[1]=indices, src[2]=data)\n");
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి అక్షరం * kernel_name;

    // Support F32 data with I64 indices -> F32/F16 output (scatter operation)
    ఒకవేళ (కణుపు->src[0]->type == GGML_TYPE_F32 && కణుపు->src[1]->type == GGML_TYPE_I64 &&
        (కణుపు->type == GGML_TYPE_F32 || కణుపు->type == GGML_TYPE_F16)) {
        ఒకవేళ (కణుపు->type == GGML_TYPE_F32 || కణుపు->type == GGML_TYPE_F16) {
            kernel_name = "set_rows_f32";
        } లేకపోతే {
            GGML_LOG_ERROR("ET: SET_ROWS unsupported output type: %s\n", ggml_type_name(కణుపు->type));
            తిరిగి అబద్ధం;
        }

    } లేకపోతే {
        GGML_LOG_ERROR("ET: SET_ROWS operation with unsupported types: dst=%s src0=%s src1=%s\n",
                       ggml_type_name(కణుపు->type), ggml_type_name(కణుపు->src[0]->type),
                       ggml_type_name(కణుపు->src[1]->type));
        తిరిగి అబద్ధం;
    }

    // Validate contiguity requirements
    ఒకవేళ (!ggml_is_contiguous_అడ్డువరుసలు(కణుపు)) {
        GGML_LOG_ERROR("ET: SET_ROWS operation requires contiguous-rows destination tensor\n");
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (!ggml_is_contiguous_అడ్డువరుసలు(కణుపు->src[0])) {
        GGML_LOG_ERROR("ET: SET_ROWS operation requires contiguous-rows source tensor\n");
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (!ggml_is_contiguous(కణుపు->src[1])) {
        GGML_LOG_ERROR("ET: SET_ROWS operation requires contiguous indices tensor\n");
        తిరిగి అబద్ధం;
    }

    // Validate dimension constraints from ggml implementation
    ఒకవేళ (!(కణుపు->ne[0] == కణుపు->src[0]->ne[0] &&              // same number of columns
          కణుపు->ne[2] == కణుపు->src[0]->ne[2] &&              // same batch size
          కణుపు->ne[3] == కణుపు->src[0]->ne[3] &&              // same outer dimension
          కణుపు->src[0]->ne[1] == కణుపు->src[1]->ne[0] &&      // src rows = index count
          కణుపు->src[0]->ne[2] % కణుపు->src[1]->ne[1] == 0 &&  // batch constraint
          కణుపు->src[0]->ne[3] % కణుపు->src[1]->ne[2] == 0 &&  // outer constraint
          కణుపు->src[1]->ne[3] == 1)) {                       // indices constraint
        GGML_LOG_ERROR("ET: SET_ROWS operation dimension constraint failed\n");
        తిరిగి అబద్ధం;
    }

    ggml_et_సమితి_అడ్డువరుసలు_params params;
    params.src0 = *కణుపు->src[0];  // F32 source data tensor
    params.src1 = *కణుపు->src[1];  // I64 indices tensor
    params.dst  = *కణుపు;          // F32/F16 destination tensor

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    తార్కిక                    cpu_comparison_చురుకైన = అబద్ధం;
    ఒకవేళ (సమితి_అడ్డువరుసలు_cpu_compare_config.enabled) {
        ఒకవేళ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, కణుపు, GGML_OP_సమితి_అడ్డువరుసలు)) {
            cpu_comparison_చురుకైన = నిజం;
        } లేకపోతే {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for SET_ROWS operation\n");
        }
    }

    తార్కిక kernel_ఫలితం = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, పరిమాణం(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ఒకవేళ (cpu_comparison_చురుకైన) {
        ఒకవేళ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, కణుపు, &సమితి_అడ్డువరుసలు_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for SET_ROWS operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_చివర("SET_ROWS", kernel_name, కణుపు);
    తిరిగి kernel_ఫలితం;
}

తార్కిక ggml_et_op_fill(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు) {
    ET_PERF_ప్రారంభం();

    ggml_et_fill_params params;
    params.dst = *కణుపు;
    జ్ఞాపకం_నకలు(&params.c, కణుపు->op_params, పరిమాణం(దశాంశం));

    తార్కిక kernel_ఫలితం = ggml_et_launch_kernel(dev_ctx, "fill_f32", &params, పరిమాణం(params), 0xFFFFFFFF);

    ET_PERF_చివర("FILL", "fill_f32", కణుపు);
    తిరిగి kernel_ఫలితం;
}

తార్కిక ggml_et_op_diag(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు) {
    ET_PERF_ప్రారంభం();

    ఒకవేళ (!కణుపు->src[0]) {
        GGML_LOG_ERROR("ET: DIAG operation missing source tensor\n");
        తిరిగి అబద్ధం;
    }

    ggml_et_diag_params params;
    params.src0 = *కణుపు->src[0];
    params.dst  = *కణుపు;

    తార్కిక kernel_ఫలితం = ggml_et_launch_kernel(dev_ctx, "diag_f32", &params, పరిమాణం(params), 0xFFFFFFFF);

    ET_PERF_చివర("DIAG", "diag_f32", కణుపు);
    తిరిగి kernel_ఫలితం;
}

తార్కిక ggml_et_op_tri(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు) {
    ET_PERF_ప్రారంభం();

    ఒకవేళ (!కణుపు->src[0]) {
        GGML_LOG_ERROR("ET: TRI operation missing source tensor\n");
        తిరిగి అబద్ధం;
    }

    ggml_et_tri_params params;
    params.src0 = *కణుపు->src[0];
    params.dst  = *కణుపు;
    జ్ఞాపకం_నకలు(&params.tri_type, కణుపు->op_params, పరిమాణం(int32_t));

    తార్కిక kernel_ఫలితం = ggml_et_launch_kernel(dev_ctx, "tri_f32", &params, పరిమాణం(params), 0xFFFFFFFF);

    ET_PERF_చివర("TRI", "tri_f32", కణుపు);
    తిరిగి kernel_ఫలితం;
}

తార్కిక ggml_et_op_solve_tri(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు) {
    ET_PERF_ప్రారంభం();

    ఒకవేళ (!కణుపు->src[0] || !కణుపు->src[1]) {
        GGML_LOG_ERROR("ET: SOLVE_TRI operation missing source tensor(s)\n");
        తిరిగి అబద్ధం;
    }

    ggml_et_solve_tri_params params;
    params.src0 = *కణుపు->src[0];  // A (lower-triangular)
    params.src1 = *కణుపు->src[1];  // B (RHS)
    params.dst  = *కణుపు;          // X (solution)

    తార్కిక kernel_ఫలితం = ggml_et_launch_kernel(dev_ctx, "solve_tri_f32", &params, పరిమాణం(params), 0xFFFFFFFF);

    ET_PERF_చివర("SOLVE_TRI", "solve_tri_f32", కణుపు);
    తిరిగి kernel_ఫలితం;
}

తార్కిక ggml_et_op_సమితి(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు) {
    ET_PERF_ప్రారంభం();

    ఒకవేళ (!కణుపు->src[0] || !కణుపు->src[1]) {
        GGML_LOG_ERROR("ET: SET operation missing source tensor(s)\n");
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి తార్కిక   inplace = (తార్కిక) ((స్థిరరాశి int32_t *) కణుపు->op_params)[4];
    స్థిరరాశి size_t స్థానభ్రంశం  = ((స్థిరరాశి int32_t *) కణుపు->op_params)[3];
    స్థిరరాశి size_t nb1     = ((స్థిరరాశి int32_t *) కణుపు->op_params)[0];
    స్థిరరాశి size_t nb2     = ((స్థిరరాశి int32_t *) కణుపు->op_params)[1];
    స్థిరరాశి size_t nb3     = ((స్థిరరాశి int32_t *) కణుపు->op_params)[2];

    ఒకవేళ (!inplace) {
        GGML_LOG_ERROR("ET: SET only supports inplace (inplace=%d)\n", inplace);
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (కణుపు->type != GGML_TYPE_F32 || కణుపు->src[0]->type != GGML_TYPE_F32 || కణుపు->src[1]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: SET only supports F32 (dst=%s src0=%s src1=%s)\n", ggml_type_name(కణుపు->type),
                       ggml_type_name(కణుపు->src[0]->type), ggml_type_name(కణుపు->src[1]->type));
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (!ggml_are_same_shape(కణుపు, కణుపు->src[0])) {
        GGML_LOG_ERROR("ET: SET requires same-shape src0 and dst\n");
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (!ggml_is_contiguous(కణుపు) || !ggml_is_contiguous(కణుపు->src[0]) || !ggml_is_contiguous(కణుపు->src[1])) {
        GGML_LOG_ERROR("ET: SET requires contiguous dst, src0, and src1\n");
        తిరిగి అబద్ధం;
    }

    ggml_et_సమితి_params params;
    params.src1   = *కణుపు->src[1];
    params.dst    = *కణుపు;
    params.nb1    = (int32_t) nb1;
    params.nb2    = (int32_t) nb2;
    params.nb3    = (int32_t) nb3;
    params.స్థానభ్రంశం = (int32_t) స్థానభ్రంశం;

    తార్కిక kernel_ఫలితం = ggml_et_launch_kernel(dev_ctx, "set_f32", &params, పరిమాణం(params), 0xFFFFFFFF);

    ET_PERF_చివర("SET", "set_f32", కణుపు);
    తిరిగి kernel_ఫలితం;
}

తార్కిక ggml_et_op_pad(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి ggml_tensor * కణుపు) {
    ET_PERF_ప్రారంభం();

    ఒకవేళ (!కణుపు->src[0]) {
        GGML_LOG_ERROR("ET: PAD operation missing source tensor\n");
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (కణుపు->type != GGML_TYPE_F32 || కణుపు->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: PAD only supports F32 (src=%s dst=%s)\n", ggml_type_name(కణుపు->src[0]->type),
                       ggml_type_name(కణుపు->type));
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (!ggml_is_contiguous(కణుపు)) {
        GGML_LOG_ERROR("ET: PAD requires contiguous output tensor\n");
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (కణుపు->src[0]->nb[0] != పరిమాణం(దశాంశం)) {
        GGML_LOG_ERROR("ET: PAD requires element-contiguous src dim0 (nb[0]=%zu)\n", (size_t) కణుపు->src[0]->nb[0]);
        తిరిగి అబద్ధం;
    }

    // Extract padding parameters from op_params
    స్థిరరాశి int32_t * op_params = (స్థిరరాశి int32_t *) కణుపు->op_params;

    ggml_et_pad_params params;
    params.src0  = *కణుపు->src[0];
    params.dst   = *కణుపు;
    params.lp[0] = op_params[0];
    params.rp[0] = op_params[1];
    params.lp[1] = op_params[2];
    params.rp[1] = op_params[3];
    params.lp[2] = op_params[4];
    params.rp[2] = op_params[5];
    params.lp[3] = op_params[6];
    params.rp[3] = op_params[7];

    // v1: no dim0 padding
    ఒకవేళ (params.lp[0] != 0 || params.rp[0] != 0) {
        GGML_LOG_ERROR("ET: PAD dim0 padding not supported (lp0=%d rp0=%d)\n", params.lp[0], params.rp[0]);
        తిరిగి అబద్ధం;
    }

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    తార్కిక                    cpu_comparison_చురుకైన = అబద్ధం;
    ఒకవేళ (pad_cpu_compare_config.enabled) {
        ఒకవేళ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, కణుపు, GGML_OP_PAD)) {
            cpu_comparison_చురుకైన = నిజం;
        } లేకపోతే {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for PAD operation\n");
        }
    }

    తార్కిక kernel_ఫలితం = ggml_et_launch_kernel(dev_ctx, "pad_f32", &params, పరిమాణం(params), 0xFFFFFFFF);

    ఒకవేళ (cpu_comparison_చురుకైన) {
        ఒకవేళ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, కణుపు, &pad_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for PAD operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_చివర("PAD", "pad_f32", కణుపు);
    తిరిగి kernel_ఫలితం;
}
