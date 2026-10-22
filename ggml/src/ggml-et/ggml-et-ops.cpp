#ଅନ୍ତର୍ଭାଵ "ggml-et-ops.h"

#ଅନ୍ତର୍ଭାଵ "ggml-et-cpu-compare.h"
#ଅନ୍ତର୍ଭାଵ "ggml-et-kernels.h"
#ଅନ୍ତର୍ଭାଵ "ggml-impl.h"

#ଅନ୍ତର୍ଭାଵ <stdio.h>

#ଅନ୍ତର୍ଭାଵ <cstdint>

// CPU comparison configuration - can be enabled for debugging
ସ୍ଥିର ggml_et_cpu_compare_config rope_cpu_compare_config = {
    /* .enabled = */ ମିଥ୍ୟା,
    /* .use_cpu_result = */ ମିଥ୍ୟା,  // Replace ET result with CPU result
    /* .log_differences = */ ସତ୍ୟ,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

ସ୍ଥିର ggml_et_cpu_compare_config rms_norm_cpu_compare_config = {
    /* .enabled = */ ମିଥ୍ୟା,
    /* .use_cpu_result = */ ମିଥ୍ୟା,
    /* .log_differences = */ ସତ୍ୟ,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

ସ୍ଥିର ggml_et_cpu_compare_config norm_cpu_compare_config = {
    /* .enabled = */ ମିଥ୍ୟା,
    /* .use_cpu_result = */ ମିଥ୍ୟା,
    /* .log_differences = */ ସତ୍ୟ,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

ସ୍ଥିର ggml_et_cpu_compare_config l2_norm_cpu_compare_config = {
    /* .enabled = */ ମିଥ୍ୟା,
    /* .use_cpu_result = */ ମିଥ୍ୟା,
    /* .log_differences = */ ସତ୍ୟ,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

ସ୍ଥିର ggml_et_cpu_compare_config group_norm_cpu_compare_config = {
    /* .enabled = */ ମିଥ୍ୟା,
    /* .use_cpu_result = */ ମିଥ୍ୟା,
    /* .log_differences = */ ସତ୍ୟ,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

ସ୍ଥିର ggml_et_cpu_compare_config im2col_cpu_compare_config = {
    /* .enabled = */ ମିଥ୍ୟା,
    /* .use_cpu_result = */ ମିଥ୍ୟା,
    /* .log_differences = */ ସତ୍ୟ,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

ସ୍ଥିର ggml_et_cpu_compare_config unary_cpu_compare_config = {
    /* .enabled = */ ମିଥ୍ୟା,
    /* .use_cpu_result = */ ମିଥ୍ୟା,
    /* .log_differences = */ ସତ୍ୟ,
    /* .tolerance = */ 1e-4f,
    /* .max_log_elements = */ 4096
};

ସ୍ଥିର ggml_et_cpu_compare_config sum_rows_cpu_compare_config = {
    /* .enabled = */ ମିଥ୍ୟା,
    /* .use_cpu_result = */ ମିଥ୍ୟା,
    /* .log_differences = */ ସତ୍ୟ,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

ସ୍ଥିର ggml_et_cpu_compare_config clamp_cpu_compare_config = {
    /* .enabled = */ ମିଥ୍ୟା,
    /* .use_cpu_result = */ ମିଥ୍ୟା,
    /* .log_differences = */ ସତ୍ୟ,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

ସ୍ଥିର ggml_et_cpu_compare_config mean_cpu_compare_config = {
    /* .enabled = */ ମିଥ୍ୟା,
    /* .use_cpu_result = */ ମିଥ୍ୟା,
    /* .log_differences = */ ସତ୍ୟ,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

ସ୍ଥିର ggml_et_cpu_compare_config sqr_cpu_compare_config = {
    /* .enabled = */ ମିଥ୍ୟା,
    /* .use_cpu_result = */ ମିଥ୍ୟା,
    /* .log_differences = */ ସତ୍ୟ,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

ସ୍ଥିର ggml_et_cpu_compare_config elmap_cpu_compare_config = {
    /* .enabled = */ ମିଥ୍ୟା,
    /* .use_cpu_result = */ ମିଥ୍ୟା,
    /* .log_differences = */ ସତ୍ୟ,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

ସ୍ଥିର ggml_et_cpu_compare_config glu_cpu_compare_config = {
    /* .enabled = */ ମିଥ୍ୟା,
    /* .use_cpu_result = */ ମିଥ୍ୟା,
    /* .log_differences = */ ସତ୍ୟ,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

ସ୍ଥିର ggml_et_cpu_compare_config mul_mat_cpu_compare_config = {
    /* .enabled = */ ମିଥ୍ୟା,
    /* .use_cpu_result = */ ମିଥ୍ୟା,
    /* .log_differences = */ ସତ୍ୟ,
    /* .tolerance = */ 0.01,
    /* .max_log_elements = */ 4096
};

ସ୍ଥିର ggml_et_cpu_compare_config mul_mat_id_cpu_compare_config = {
    /* .enabled = */ ମିଥ୍ୟା,
    /* .use_cpu_result = */ ମିଥ୍ୟା,
    /* .log_differences = */ ସତ୍ୟ,
    /* .tolerance = */ 0.01,
    /* .max_log_elements = */ 4096
};

ସ୍ଥିର ggml_et_cpu_compare_config softmax_cpu_compare_config = {
    /* .enabled = */ ମିଥ୍ୟା,
    /* .use_cpu_result = */ ମିଥ୍ୟା,
    /* .log_differences = */ ସତ୍ୟ,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 1024
};

ସ୍ଥିର ggml_et_cpu_compare_config get_rows_cpu_compare_config = {
    /* .enabled = */ ମିଥ୍ୟା,
    /* .use_cpu_result = */ ମିଥ୍ୟା,
    /* .log_differences = */ ସତ୍ୟ,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 2048
};

ସ୍ଥିର ggml_et_cpu_compare_config pad_cpu_compare_config = {
    /* .enabled = */ ମିଥ୍ୟା,
    /* .use_cpu_result = */ ମିଥ୍ୟା,
    /* .log_differences = */ ସତ୍ୟ,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

ସ୍ଥିର ggml_et_cpu_compare_config cont_cpu_compare_config = {
    /* .enabled = */ ମିଥ୍ୟା,
    /* .use_cpu_result = */ ମିଥ୍ୟା,
    /* .log_differences = */ ସତ୍ୟ,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

ସ୍ଥିର ggml_et_cpu_compare_config concat_cpu_compare_config = {
    /* .enabled = */ ମିଥ୍ୟା,
    /* .use_cpu_result = */ ମିଥ୍ୟା,
    /* .log_differences = */ ସତ୍ୟ,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

ସ୍ଥିର ggml_et_cpu_compare_config cumsum_cpu_compare_config = {
    /* .enabled = */ ମିଥ୍ୟା,
    /* .use_cpu_result = */ ମିଥ୍ୟା,
    /* .log_differences = */ ସତ୍ୟ,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

ସ୍ଥିର ggml_et_cpu_compare_config repeat_cpu_compare_config = {
    /* .enabled = */ ମିଥ୍ୟା,
    /* .use_cpu_result = */ ମିଥ୍ୟା,
    /* .log_differences = */ ସତ୍ୟ,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

ସ୍ଥିର ggml_et_cpu_compare_config ssm_conv_cpu_compare_config = {
    /* .enabled = */ ମିଥ୍ୟା,
    /* .use_cpu_result = */ ମିଥ୍ୟା,
    /* .log_differences = */ ସତ୍ୟ,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

ସ୍ଥିର ggml_et_cpu_compare_config rwkv_wkv6_cpu_compare_config = {
    /* .enabled = */ ମିଥ୍ୟା,
    /* .use_cpu_result = */ ମିଥ୍ୟା,
    /* .log_differences = */ ସତ୍ୟ,
    /* .tolerance = */ 1e-4f,
    /* .max_log_elements = */ 4096
};

ସ୍ଥିର ggml_et_cpu_compare_config rwkv_wkv7_cpu_compare_config = {
    /* .enabled = */ ମିଥ୍ୟା,
    /* .use_cpu_result = */ ମିଥ୍ୟା,
    /* .log_differences = */ ସତ୍ୟ,
    /* .tolerance = */ 1e-4f,
    /* .max_log_elements = */ 4096
};

ସ୍ଥିର ggml_et_cpu_compare_config set_rows_cpu_compare_config = {
    /* .enabled = */ ମିଥ୍ୟା,
    /* .use_cpu_result = */ ମିଥ୍ୟା,
    /* .log_differences = */ ସତ୍ୟ,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 2048
};

ତାର୍କିକ ggml_et_op_rms_norm_mul(ggml_backend_et_device_context * dev_ctx,
                             ଅଚଳ ggml_tensor *              rms_norm_ଗଣ୍ଠି,
                             ଅଚଳ ggml_tensor *              mul_ଗଣ୍ଠି) {
    ET_PERF_START();

    ଯଦି (!dev_ctx || !rms_norm_ଗଣ୍ଠି || !mul_ଗଣ୍ଠି) {
        GGML_LOG_ERROR("ET: Invalid parameters for fused RMS_NORM_MUL operation\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (!rms_norm_ଗଣ୍ଠି->src[0]) {
        GGML_LOG_ERROR("ET: Fused RMS_NORM_MUL missing required input\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // Extract weights: the MUL operand that isn't the rms_norm output
    ଅଚଳ ggml_tensor * weights = (mul_ଗଣ୍ଠି->src[0] == rms_norm_ଗଣ୍ଠି) ? mul_ଗଣ୍ଠି->src[1] : mul_ଗଣ୍ଠି->src[0];

    ଯଦି (!weights) {
        GGML_LOG_ERROR("ET: Fused RMS_NORM_MUL missing weights tensor\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଦଶମିକ eps;
    memcpy(&eps, rms_norm_ଗଣ୍ଠି->op_params, ଆକାର(ଦଶମିକ));

    ggml_et_rms_norm_mul_params params;
    params.src0 = *rms_norm_ଗଣ୍ଠି->src[0];  // input to normalize
    params.src1 = *weights;                // normalization weights
    params.dst  = *mul_ଗଣ୍ଠି;               // final output
    params.eps  = eps;

    ତାର୍କିକ kernel_result = ggml_et_launch_kernel(dev_ctx, "rms_norm_mul_f32", &params, ଆକାର(params), 0xFFFFFFFF);

    ET_PERF_END_EXT("RMS_NORM_MUL", "rms_norm_mul_f32", mul_ଗଣ୍ଠି, "eps=%.6f", (ଦ୍ୱିଗୁଣ) eps);
    ଫେରନ୍ତୁ kernel_result;
}

ତାର୍କିକ ggml_et_op_scale(ggml_backend_et_device_context * dev_ctx, ଅଚଳ ggml_tensor * ଗଣ୍ଠି) {
    ET_PERF_START();

    ଯଦି (!dev_ctx || !ଗଣ୍ଠି) {
        GGML_LOG_ERROR("ET: Invalid parameters for SCALE operation\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (!ଗଣ୍ଠି->src[0]) {
        GGML_LOG_ERROR("ET: SCALE operation missing required input\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (ଗଣ୍ଠି->type != GGML_TYPE_F32 || ଗଣ୍ଠି->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: SCALE operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(ଗଣ୍ଠି->type),
                       ggml_type_name(ଗଣ୍ଠି->src[0]->type));
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଦଶମିକ scale, bias;
    memcpy(&scale, (ଅଚଳ ଦଶମିକ *) ଗଣ୍ଠି->op_params + 0, ଆକାର(ଦଶମିକ));
    memcpy(&bias, (ଅଚଳ ଦଶମିକ *) ଗଣ୍ଠି->op_params + 1, ଆକାର(ଦଶମିକ));

    ggml_et_scale_params params;
    params.src0  = *ଗଣ୍ଠି->src[0];
    params.dst   = *ଗଣ୍ଠି;
    params.scale = scale;
    params.bias  = bias;

    ତାର୍କିକ kernel_result = ggml_et_launch_kernel(dev_ctx, "scale_f32", &params, ଆକାର(params), 0xFFFFFFFF);

    ET_PERF_END_EXT("SCALE", "scale_f32", ଗଣ୍ଠି, "scale=%.6f|bias=%.6f", (ଦ୍ୱିଗୁଣ) scale, (ଦ୍ୱିଗୁଣ) bias);
    ଫେରନ୍ତୁ kernel_result;
}

ତାର୍କିକ ggml_et_op_sqr(ggml_backend_et_device_context * dev_ctx, ଅଚଳ ggml_tensor * ଗଣ୍ଠି) {
    ET_PERF_START();

    ଯଦି (!dev_ctx || !ଗଣ୍ଠି) {
        GGML_LOG_ERROR("ET: Invalid parameters for SQR operation\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (!ଗଣ୍ଠି->src[0]) {
        GGML_LOG_ERROR("ET: SQR operation missing required input\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (ଗଣ୍ଠି->type != GGML_TYPE_F32 || ଗଣ୍ଠି->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: SQR operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(ଗଣ୍ଠି->type),
                       ggml_type_name(ଗଣ୍ଠି->src[0]->type));
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ggml_et_sqr_params params;
    params.src0 = *ଗଣ୍ଠି->src[0];  // F32 input tensor
    params.dst  = *ଗଣ୍ଠି;          // F32 output tensor

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ତାର୍କିକ                    cpu_comparison_active = ମିଥ୍ୟା;
    ଯଦି (sqr_cpu_compare_config.enabled) {
        ଯଦି (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ଗଣ୍ଠି, GGML_OP_SQR)) {
            cpu_comparison_active = ସତ୍ୟ;
        } ଅନ୍ୟଥା {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for SQR operation\n");
        }
    }

    ତାର୍କିକ kernel_result = ggml_et_launch_kernel(dev_ctx, "sqr_f32", &params, ଆକାର(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ଯଦି (cpu_comparison_active) {
        ଯଦି (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ଗଣ୍ଠି, &sqr_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for SQR operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("SQR", "sqr_f32", ଗଣ୍ଠି);
    ଫେରନ୍ତୁ kernel_result;
}

ତାର୍କିକ ggml_et_op_sum_rows(ggml_backend_et_device_context * dev_ctx, ଅଚଳ ggml_tensor * ଗଣ୍ଠି) {
    ET_PERF_START();

    ଯଦି (!dev_ctx || !ଗଣ୍ଠି) {
        GGML_LOG_ERROR("ET: Invalid parameters for SUM_ROWS operation\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (!ଗଣ୍ଠି->src[0]) {
        GGML_LOG_ERROR("ET: SUM_ROWS operation missing required input\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (ଗଣ୍ଠି->type != GGML_TYPE_F32 || ଗଣ୍ଠି->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: SUM_ROWS operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(ଗଣ୍ଠି->type),
                       ggml_type_name(ଗଣ୍ଠି->src[0]->type));
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ggml_et_sum_rows_params params;
    params.src0 = *ଗଣ୍ଠି->src[0];
    params.dst  = *ଗଣ୍ଠି;

    // Phase 1: Initialize CPU comparison context
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ତାର୍କିକ                    cpu_comparison_active = ମିଥ୍ୟା;
    ଯଦି (sum_rows_cpu_compare_config.enabled) {
        ଯଦି (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ଗଣ୍ଠି, GGML_OP_SUM_ROWS)) {
            cpu_comparison_active = ସତ୍ୟ;
        } ଅନ୍ୟଥା {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for SUM_ROWS operation\n");
        }
    }

    ତାର୍କିକ kernel_result = ggml_et_launch_kernel(dev_ctx, "sum_rows_f32", &params, ଆକାର(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare
    ଯଦି (cpu_comparison_active) {
        ଯଦି (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ଗଣ୍ଠି, &sum_rows_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for SUM_ROWS operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("SUM_ROWS", "sum_rows_f32", ଗଣ୍ଠି);
    ଫେରନ୍ତୁ kernel_result;
}

ତାର୍କିକ ggml_et_op_mean(ggml_backend_et_device_context * dev_ctx, ଅଚଳ ggml_tensor * ଗଣ୍ଠି) {
    ET_PERF_START();

    ଯଦି (!dev_ctx || !ଗଣ୍ଠି) {
        GGML_LOG_ERROR("ET: Invalid parameters for MEAN operation\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (!ଗଣ୍ଠି->src[0]) {
        GGML_LOG_ERROR("ET: MEAN operation missing required input\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (ଗଣ୍ଠି->type != GGML_TYPE_F32 || ଗଣ୍ଠି->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: MEAN operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(ଗଣ୍ଠି->type),
                       ggml_type_name(ଗଣ୍ଠି->src[0]->type));
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ggml_et_mean_params params;
    params.src0 = *ଗଣ୍ଠି->src[0];
    params.dst  = *ଗଣ୍ଠି;

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ତାର୍କିକ                    cpu_comparison_active = ମିଥ୍ୟା;
    ଯଦି (mean_cpu_compare_config.enabled) {
        ଯଦି (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ଗଣ୍ଠି, GGML_OP_MEAN)) {
            cpu_comparison_active = ସତ୍ୟ;
        } ଅନ୍ୟଥା {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for MEAN operation\n");
        }
    }

    ତାର୍କିକ kernel_result = ggml_et_launch_kernel(dev_ctx, "mean_f32", &params, ଆକାର(params), 0xFFFFFFFF);

    ଯଦି (cpu_comparison_active) {
        ଯଦି (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ଗଣ୍ଠି, &mean_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for MEAN operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("MEAN", "mean_f32", ଗଣ୍ଠି);
    ଫେରନ୍ତୁ kernel_result;
}

ତାର୍କିକ ggml_et_op_clamp(ggml_backend_et_device_context * dev_ctx, ଅଚଳ ggml_tensor * ଗଣ୍ଠି) {
    ET_PERF_START();

    ଯଦି (!dev_ctx || !ଗଣ୍ଠି) {
        GGML_LOG_ERROR("ET: Invalid parameters for CLAMP operation\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (!ଗଣ୍ଠି->src[0]) {
        GGML_LOG_ERROR("ET: CLAMP operation missing required input\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (ଗଣ୍ଠି->type != GGML_TYPE_F32 || ଗଣ୍ଠି->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: CLAMP operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(ଗଣ୍ଠି->type),
                       ggml_type_name(ଗଣ୍ଠି->src[0]->type));
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ggml_et_clamp_params params;
    params.src0 = *ଗଣ୍ଠି->src[0];
    params.dst  = *ଗଣ୍ଠି;
    // op_params layout per ggml.c::ggml_clamp: { min, max } as floats
    memcpy(&params.min_val, (ଅଚଳ ଦଶମିକ *) ଗଣ୍ଠି->op_params + 0, ଆକାର(ଦଶମିକ));
    memcpy(&params.max_val, (ଅଚଳ ଦଶମିକ *) ଗଣ୍ଠି->op_params + 1, ଆକାର(ଦଶମିକ));

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ତାର୍କିକ                    cpu_comparison_active = ମିଥ୍ୟା;
    ଯଦି (clamp_cpu_compare_config.enabled) {
        ଯଦି (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ଗଣ୍ଠି, GGML_OP_CLAMP)) {
            cpu_comparison_active = ସତ୍ୟ;
        } ଅନ୍ୟଥା {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for CLAMP operation\n");
        }
    }

    ତାର୍କିକ kernel_result = ggml_et_launch_kernel(dev_ctx, "clamp_f32", &params, ଆକାର(params), 0xFFFFFFFF);

    ଯଦି (cpu_comparison_active) {
        ଯଦି (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ଗଣ୍ଠି, &clamp_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for CLAMP operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("CLAMP", "clamp_f32", ଗଣ୍ଠି);
    ଫେରନ୍ତୁ kernel_result;
}

ତାର୍କିକ ggml_et_op_unary(ggml_backend_et_device_context * dev_ctx, ଅଚଳ ggml_tensor * ଗଣ୍ଠି) {
    ET_PERF_START();

    ଯଦି (!dev_ctx || !ଗଣ୍ଠି) {
        GGML_LOG_ERROR("ET: Invalid parameters for UNARY operation\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (!ଗଣ୍ଠି->src[0]) {
        GGML_LOG_ERROR("ET: UNARY operation missing required input\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (ଗଣ୍ଠି->type != GGML_TYPE_F32 || ଗଣ୍ଠି->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: UNARY operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(ଗଣ୍ଠି->type),
                       ggml_type_name(ଗଣ୍ଠି->src[0]->type));
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ ggml_unary_op uop     = ggml_get_unary_op(ଗଣ୍ଠି);
    ଅଚଳ ଅକ୍ଷର *        op_name = ggml_unary_op_name(uop);

    ggml_et_unary_params params;
    params.src0     = *ଗଣ୍ଠି->src[0];  // F32 input tensor
    params.dst      = *ଗଣ୍ଠି;          // F32 output tensor
    params.unary_op = (int32_t) uop;

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ତାର୍କିକ                    cpu_comparison_active = ମିଥ୍ୟା;
    ଯଦି (unary_cpu_compare_config.enabled) {
        ଯଦି (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ଗଣ୍ଠି, GGML_OP_UNARY)) {
            cpu_comparison_active = ସତ୍ୟ;
        } ଅନ୍ୟଥା {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for UNARY/%s operation\n", op_name);
        }
    }

    ତାର୍କିକ kernel_result = ggml_et_launch_kernel(dev_ctx, "unary_f32", &params, ଆକାର(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ଯଦି (cpu_comparison_active) {
        ଯଦି (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ଗଣ୍ଠି, &unary_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for UNARY/%s operation\n", op_name);
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END_EXT("UNARY", "unary_f32", ଗଣ୍ଠି, "op=%s", op_name);
    ଫେରନ୍ତୁ kernel_result;
}

ତାର୍କିକ ggml_et_op_mul(ggml_backend_et_device_context * dev_ctx, ଅଚଳ ggml_tensor * ଗଣ୍ଠି) {
    // Delegate to generic element map operation
    ଫେରନ୍ତୁ ggml_et_op_elmap(dev_ctx, ଗଣ୍ଠି);
}

ତାର୍କିକ ggml_et_op_add(ggml_backend_et_device_context * dev_ctx, ଅଚଳ ggml_tensor * ଗଣ୍ଠି) {
    // Delegate to generic element map operation
    ଫେରନ୍ତୁ ggml_et_op_elmap(dev_ctx, ଗଣ୍ଠି);
}

ତାର୍କିକ ggml_et_op_sub(ggml_backend_et_device_context * dev_ctx, ଅଚଳ ggml_tensor * ଗଣ୍ଠି) {
    // Delegate to generic element map operation
    ଫେରନ୍ତୁ ggml_et_op_elmap(dev_ctx, ଗଣ୍ଠି);
}

ତାର୍କିକ ggml_et_op_elmap(ggml_backend_et_device_context * dev_ctx, ଅଚଳ ggml_tensor * ଗଣ୍ଠି) {
    ET_PERF_START();

    ଯଦି (!dev_ctx || !ଗଣ୍ଠି) {
        GGML_LOG_ERROR("ET: Invalid parameters for element map operation\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (!ଗଣ୍ଠି->src[0] || !ଗଣ୍ଠି->src[1]) {
        GGML_LOG_ERROR("ET: Element map operation missing required inputs\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (ଗଣ୍ଠି->type != GGML_TYPE_F32 || ଗଣ୍ଠି->src[0]->type != GGML_TYPE_F32 || ଗଣ୍ଠି->src[1]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: Element map operation with unsupported types: dst=%s src0=%s src1=%s\n",
                       ggml_type_name(ଗଣ୍ଠି->type), ggml_type_name(ଗଣ୍ଠି->src[0]->type),
                       ggml_type_name(ଗଣ୍ଠି->src[1]->type));
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ ଅକ୍ଷର * op_name = ggml_op_name(ଗଣ୍ଠି->op);

    ggml_et_elmap_params params;
    params.src0 = *ଗଣ୍ଠି->src[0];
    params.src1 = *ଗଣ୍ଠି->src[1];
    params.dst  = *ଗଣ୍ଠି;  // F32 output tensor (op type stored in dst.op)

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ତାର୍କିକ                    cpu_comparison_active = ମିଥ୍ୟା;
    ଯଦି (elmap_cpu_compare_config.enabled) {
        ଯଦି (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ଗଣ୍ଠି, ଗଣ୍ଠି->op)) {
            cpu_comparison_active = ସତ୍ୟ;
        } ଅନ୍ୟଥା {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for %s operation\n", op_name);
        }
    }

    // fprintf(stderr, "ET: el_map s0 [%ld, %ld, %ld, %ld] s1 [%ld, %ld, %ld, %ld]\n",
    //     node->src[0]->ne[0], node->src[0]->ne[1], node->src[0]->ne[2], node->src[0]->ne[3],
    //     node->src[1]->ne[0], node->src[1]->ne[1], node->src[1]->ne[2], node->src[1]->ne[3]);

    ତାର୍କିକ kernel_result = ggml_et_launch_kernel(dev_ctx, "el_map_f32", &params, ଆକାର(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ଯଦି (cpu_comparison_active) {
        ଯଦି (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ଗଣ୍ଠି, &elmap_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for %s operation\n", op_name);
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END(op_name, "el_map_f32", ଗଣ୍ଠି);
    ଫେରନ୍ତୁ kernel_result;
}

ତାର୍କିକ ggml_et_op_glu(ggml_backend_et_device_context * dev_ctx, ଅଚଳ ggml_tensor * ଗଣ୍ଠି) {
    ET_PERF_START();

    // Validate inputs
    ଯଦି (!dev_ctx || !ଗଣ୍ଠି) {
        GGML_LOG_ERROR("ET: Invalid parameters for GLU operation\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (!ଗଣ୍ଠି->src[0]) {
        GGML_LOG_ERROR("ET: GLU operation missing required input\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ ତାର୍କିକ is_split_mode = ଗଣ୍ଠି->src[1] != nullptr;

    // Only support F32 (as validated by supports_op)
    ଯଦି (ଗଣ୍ଠି->type != GGML_TYPE_F32 || ଗଣ୍ଠି->src[0]->type != GGML_TYPE_F32 ||
        (is_split_mode && ଗଣ୍ଠି->src[1]->type != GGML_TYPE_F32)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // Extract GLU operation parameters from op_params
    int32_t glu_op_type = ggml_get_op_params_i32(ଗଣ୍ଠି, 0);  // GLU variant (REGLU, GEGLU, SWIGLU, etc.)
    int32_t swapped     = ggml_get_op_params_i32(ଗଣ୍ଠି, 1);  // Whether gate/value are swapped

    // Supported variants
    ବାଛ (glu_op_type) {
        ସ୍ଥିତି GGML_GLU_OP_REGLU:
        ସ୍ଥିତି GGML_GLU_OP_GEGLU:
        ସ୍ଥିତି GGML_GLU_OP_SWIGLU:
        ସ୍ଥିତି GGML_GLU_OP_SWIGLU_OAI:
        ସ୍ଥିତି GGML_GLU_OP_SWIGLU_CLAMP:
        ସ୍ଥିତି GGML_GLU_OP_GEGLU_ERF:
        ସ୍ଥିତି GGML_GLU_OP_GEGLU_QUICK:
            ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            GGML_LOG_ERROR("ET: GLU operation with unsupported variant: %s\n",
                           ggml_glu_op_name((ggml_glu_op) glu_op_type));
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // Get GLU operation name for logging
    ଅଚଳ ଅକ୍ଷର * glu_op_name = ggml_glu_op_name((ggml_glu_op) glu_op_type);

    // Pack parameters. Single-tensor mode is encoded by zeroing src1.
    ggml_et_glu_params params = {};
    params.src0               = *ଗଣ୍ଠି->src[0];
    ଯଦି (is_split_mode) {
        params.src1 = *ଗଣ୍ଠି->src[1];
    }
    params.dst         = *ଗଣ୍ଠି;
    params.glu_op_type = glu_op_type;
    params.swapped     = swapped;
    params.alpha       = 0.0f;
    params.limit       = 0.0f;
    ଯଦି (glu_op_type == GGML_GLU_OP_SWIGLU_OAI) {
        params.alpha = ggml_get_op_params_f32(ଗଣ୍ଠି, 2);
    }
    ଯଦି (glu_op_type == GGML_GLU_OP_SWIGLU_OAI || glu_op_type == GGML_GLU_OP_SWIGLU_CLAMP) {
        params.limit = ggml_get_op_params_f32(ଗଣ୍ଠି, 3);
    }
    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ତାର୍କିକ                    cpu_comparison_active = ମିଥ୍ୟା;
    ଯଦି (glu_cpu_compare_config.enabled) {
        ଯଦି (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ଗଣ୍ଠି, GGML_OP_GLU)) {
            cpu_comparison_active = ସତ୍ୟ;
        } ଅନ୍ୟଥା {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for %s operation\n", glu_op_name);
        }
    }

    // Launch ET kernel
    ତାର୍କିକ kernel_result = ggml_et_launch_kernel(dev_ctx, "glu_f32", &params, ଆକାର(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ଯଦି (cpu_comparison_active) {
        ଯଦି (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ଗଣ୍ଠି, &glu_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for %s operation\n", glu_op_name);
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("GLU", "glu_f32", ଗଣ୍ଠି);
    ଫେରନ୍ତୁ kernel_result;
}

ତାର୍କିକ ggml_et_op_mul_mat(ggml_backend_et_device_context * dev_ctx,
                        ଅଚଳ ggml_tensor *              ଗଣ୍ଠି,
                        ଅଚଳ ggml_tensor *              add_ଗଣ୍ଠି) {
    ET_PERF_START();

    ଯଦି (!dev_ctx || !ଗଣ୍ଠି) {
        GGML_LOG_ERROR("ET: Invalid parameters for MUL_MAT operation\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (!ଗଣ୍ଠି->src[0] || !ଗଣ୍ଠି->src[1]) {
        GGML_LOG_ERROR("ET: MUL_MAT operation missing required inputs\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // Fused MM+ADD: when add_node is non-NULL the caller has already validated
    // (Q8_0 weights, F32 acts, exact-shape ADD with stride parity to dst) via
    // ggml_et_can_fuse({MUL_MAT, ADD}). The kernel writes dst = mm + bias and
    // the ADD's output replaces MM's as the actual dst.
    ଅଚଳ ggml_tensor * fused_dst   = add_ଗଣ୍ଠି ? add_ଗଣ୍ଠି : ଗଣ୍ଠି;
    ଅଚଳ ggml_tensor * bias_tensor = nullptr;
    ଯଦି (add_ଗଣ୍ଠି) {
        bias_tensor = (add_ଗଣ୍ଠି->src[0] == ଗଣ୍ଠି) ? add_ଗଣ୍ଠି->src[1] : add_ଗଣ୍ଠି->src[0];
    }

    ଅଚଳ ଅକ୍ଷର * kernel_name;
    ଅଚଳ ଅକ୍ଷର * src0_type_name;

    ଯଦି (ଗଣ୍ଠି->type == GGML_TYPE_F32 && ଗଣ୍ଠି->src[0]->type == GGML_TYPE_Q4_0 && ଗଣ୍ଠି->src[1]->type == GGML_TYPE_F32 &&
        ଗଣ୍ଠି->src[1]->ne[1] >= 53 &&      // N >= 53
        ଗଣ୍ଠି->src[0]->ne[1] % 16 == 0 &&  // M % TILE_M
        ଗଣ୍ଠି->src[0]->ne[0] % 32 == 0) {  // K % BLOCK_K (Q4_0 block)

        // Matrix engine for N >= 53; partial N (via n_cur-1) and errata padding are handled in-kernel.
        kernel_name    = "mul_mat_Q4_0_matrix_engine";
        src0_type_name = "Q4_0";

    } ଅନ୍ୟଥା ଯଦି (ଗଣ୍ଠି->type == GGML_TYPE_F32 && ଗଣ୍ଠି->src[0]->type == GGML_TYPE_Q4_0 &&
               ଗଣ୍ଠି->src[1]->type == GGML_TYPE_F32) {
        kernel_name    = "mul_mat_Q4_0";  // N < 53, or M % 16 != 0 or K % 32 != 0
        src0_type_name = "Q4_0";

    } ଅନ୍ୟଥା ଯଦି (ଗଣ୍ଠି->type == GGML_TYPE_F32 && ଗଣ୍ଠି->src[0]->type == GGML_TYPE_Q8_0 &&
               ଗଣ୍ଠି->src[1]->type == GGML_TYPE_F32) {
        kernel_name    = "mul_mat_Q8_0";
        src0_type_name = "Q8_0";

    } ଅନ୍ୟଥା ଯଦି (ଗଣ୍ଠି->type == GGML_TYPE_F32 && ଗଣ୍ଠି->src[0]->type == GGML_TYPE_F16 &&
               ଗଣ୍ଠି->src[1]->type == GGML_TYPE_F16 && ଗଣ୍ଠି->ne[0] % 16 == 0 && ଗଣ୍ଠି->src[0]->ne[0] % 16 == 0 &&
               ଗଣ୍ଠି->src[0]->ne[1] % 16 == 0 && ଗଣ୍ଠି->src[1]->ne[0] != 1) {
        kernel_name    = "mul_mat_f16_matrix_engine";
        src0_type_name = "F16";

    } ଅନ୍ୟଥା ଯଦି (ଗଣ୍ଠି->type == GGML_TYPE_F32 && ଗଣ୍ଠି->src[0]->type == GGML_TYPE_F16 &&
               (ଗଣ୍ଠି->src[1]->type == GGML_TYPE_F16 || ଗଣ୍ଠି->src[1]->type == GGML_TYPE_F32)) {
        kernel_name    = "mul_mat_f16";
        src0_type_name = "F16";

    } ଅନ୍ୟଥା ଯଦି (ଗଣ୍ଠି->type == GGML_TYPE_F32 && ଗଣ୍ଠି->src[0]->type == GGML_TYPE_F32 &&
               ଗଣ୍ଠି->src[1]->type == GGML_TYPE_F32 && ଗଣ୍ଠି->ne[0] % 16 == 0 && ଗଣ୍ଠି->src[0]->ne[0] % 16 == 0 &&
               ଗଣ୍ଠି->src[0]->ne[1] % 16 == 0 && ଗଣ୍ଠି->src[1]->ne[0] != 1) {  // GEMV is faster with the generic path

        kernel_name    = "mul_mat_f32_matrix_engine";
        src0_type_name = "F32";
    } ଅନ୍ୟଥା ଯଦି (ଗଣ୍ଠି->type == GGML_TYPE_F32 && ଗଣ୍ଠି->src[0]->type == GGML_TYPE_F32 &&
               (ଗଣ୍ଠି->src[1]->type == GGML_TYPE_F16 || ଗଣ୍ଠି->src[1]->type == GGML_TYPE_F32)) {
        kernel_name    = "mul_mat_f32";
        src0_type_name = "F32";
    } ଅନ୍ୟଥା {
        GGML_LOG_ERROR("ET: MUL_MAT operation with unsupported types: dst=%s src0=%s src1=%s\n",
                       ggml_type_name(ଗଣ୍ଠି->type), ggml_type_name(ଗଣ୍ଠି->src[0]->type),
                       ggml_type_name(ଗଣ୍ଠି->src[1]->type));
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ggml_et_ଦ୍ୱିମିକ_params params;
    params.src0 = *ଗଣ୍ଠି->src[0];  // weight matrix
    params.src1 = *ଗଣ୍ଠି->src[1];  // activation matrix
    params.dst  = *fused_dst;     // output (= add_node when fused, else node)

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ତାର୍କିକ                    cpu_comparison_active = ମିଥ୍ୟା;
    ଯଦି (mul_mat_cpu_compare_config.enabled) {
        ଯଦି (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, fused_dst, GGML_OP_MUL_MAT)) {
            cpu_comparison_active = ସତ୍ୟ;
        } ଅନ୍ୟଥା {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for MUL_MAT operation\n");
        }
    }

    ତାର୍କିକ kernel_result;
    ଯଦି (ଗଣ୍ଠି->src[0]->type == GGML_TYPE_Q8_0) {
        // Q8_0 kernel always takes the extended struct. bias.data is non-NULL
        // only on the fused path; otherwise the kernel skips the add entirely.
        ggml_et_mm_q8_params q8_params = {};
        q8_params.src0                 = params.src0;
        q8_params.src1                 = params.src1;
        q8_params.dst                  = params.dst;
        ଯଦି (bias_tensor) {
            q8_params.bias = *bias_tensor;
        }
        kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &q8_params, ଆକାର(q8_params), 0xFFFFFFFF);
    } ଅନ୍ୟଥା {
        // Non-Q8 MM kernels don't yet support fused-add; the graph fuse check
        // already rejects non-Q8 pairs, so add_node is always nullptr here.
        kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ଆକାର(params), 0xFFFFFFFF);
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
    ଯଦି (cpu_comparison_active) {
        ଯଦି (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, fused_dst, &mul_mat_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for MUL_MAT operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    {
        // Calculate actual FLOPs including batch/sequence dimensions
        // dst shape: [M, N, ne2, ne3] where M=ne[1], N=ne[0]
        int64_t m   = ଗଣ୍ଠି->ne[1];
        int64_t n   = ଗଣ୍ଠି->ne[0];
        int64_t k   = ଗଣ୍ଠି->src[0]->ne[0];
        int64_t ne2 = ଗଣ୍ଠି->ne[2];
        int64_t ne3 = ଗଣ୍ଠି->ne[3];

        // Total FLOPs = (batch_size) * M * N * (2*K - 1)
        // Each MxN matrix-matrix multiply does M*N*(2*K-1) FLOPs
        // Broadcasting is handled by repeating computation, so count actual operations
        int64_t batch_size  = ne2 * ne3;
        int64_t total_flops = batch_size * m * n * (2 * k - 1);

        ଅକ୍ଷର kernel_variant[64];
        snprintf(kernel_variant, ଆକାର(kernel_variant), "%s_%sx%s", kernel_name, src0_type_name,
                 ggml_type_name(ଗଣ୍ଠି->src[1]->type));
        ET_PERF_END_EXT("MUL_MAT", kernel_variant, ଗଣ୍ଠି, "flops=%" PRId64, total_flops);
    }
    ଫେରନ୍ତୁ kernel_result;
}

ତାର୍କିକ ggml_et_op_mul_mat_id(ggml_backend_et_device_context * dev_ctx, ଅଚଳ ggml_tensor * ଗଣ୍ଠି) {
    ET_PERF_START();
    ଯଦି (!dev_ctx || !ଗଣ୍ଠି) {
        GGML_LOG_ERROR("ET: Invalid parameters for MUL_MAT_ID operation\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (!ଗଣ୍ଠି->src[0] || !ଗଣ୍ଠି->src[1] || !ଗଣ୍ଠି->src[2]) {
        GGML_LOG_ERROR("ET: MUL_MAT_ID operation missing required inputs\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ ଅକ୍ଷର * kernel_name;
    ଅଚଳ ଅକ୍ଷର * src0_type_name;

    // Support Q8_0/Q4_0/F16/F32 x F32 -> F32 matrix multiplication with expert selection
    ଯଦି (ଗଣ୍ଠି->type == GGML_TYPE_F32 && ଗଣ୍ଠି->src[0]->type == GGML_TYPE_Q8_0 && ଗଣ୍ଠି->src[1]->type == GGML_TYPE_F32 &&
        ଗଣ୍ଠି->src[2]->type == GGML_TYPE_I32) {
        kernel_name    = "mul_mat_id_Q8_0";
        src0_type_name = "Q8_0";

    } ଅନ୍ୟଥା ଯଦି (ଗଣ୍ଠି->type == GGML_TYPE_F32 && ଗଣ୍ଠି->src[0]->type == GGML_TYPE_Q4_0 &&
               ଗଣ୍ଠି->src[1]->type == GGML_TYPE_F32 && ଗଣ୍ଠି->src[2]->type == GGML_TYPE_I32) {
        kernel_name    = "mul_mat_id_Q4_0";
        src0_type_name = "Q4_0";

    } ଅନ୍ୟଥା ଯଦି (ଗଣ୍ଠି->type == GGML_TYPE_F32 && ଗଣ୍ଠି->src[0]->type == GGML_TYPE_F16 &&
               ଗଣ୍ଠି->src[1]->type == GGML_TYPE_F32 && ଗଣ୍ଠି->src[2]->type == GGML_TYPE_I32) {
        kernel_name    = "mul_mat_id_f32";
        src0_type_name = "F16";

    } ଅନ୍ୟଥା ଯଦି (ଗଣ୍ଠି->type == GGML_TYPE_F32 && ଗଣ୍ଠି->src[0]->type == GGML_TYPE_F32 &&
               ଗଣ୍ଠି->src[1]->type == GGML_TYPE_F32 && ଗଣ୍ଠି->src[2]->type == GGML_TYPE_I32) {
        kernel_name    = "mul_mat_id_f32";
        src0_type_name = "F32";

    } ଅନ୍ୟଥା {
        GGML_LOG_ERROR("ET: MUL_MAT_ID operation with unsupported types: dst=%s src0=%s src1=%s src2=%s\n",
                       ggml_type_name(ଗଣ୍ଠି->type), ggml_type_name(ଗଣ୍ଠି->src[0]->type),
                       ggml_type_name(ଗଣ୍ଠି->src[1]->type), ggml_type_name(ଗଣ୍ଠି->src[2]->type));
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // Pack parameters - copy full tensor structures
    ggml_et_mul_mat_id_params params;
    params.src0 = *ଗଣ୍ଠି->src[0];  // Expert weight matrices (Q8_0/F16/F32)
    params.src1 = *ଗଣ୍ଠି->src[1];  // Activation matrix (F32)
    params.src2 = *ଗଣ୍ଠି->src[2];  // Expert indices (I32)
    params.dst  = *ଗଣ୍ଠି;          // Output matrix (F32)

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ତାର୍କିକ                    cpu_comparison_active = ମିଥ୍ୟା;
    ଯଦି (mul_mat_id_cpu_compare_config.enabled) {
        ଯଦି (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ଗଣ୍ଠି, GGML_OP_MUL_MAT_ID)) {
            cpu_comparison_active = ସତ୍ୟ;
        } ଅନ୍ୟଥା {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for MUL_MAT_ID operation\n");
        }
    }

    // Launch ET kernel
    ତାର୍କିକ kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ଆକାର(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ଯଦି (cpu_comparison_active) {
        ଯଦି (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ଗଣ୍ଠି, &mul_mat_id_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for MUL_MAT_ID operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    // Calculate FLOPs (approximate - similar to MUL_MAT but with expert routing overhead)
    // Each expert computation is similar to a MUL_MAT, but we only compute for selected experts
    int64_t K             = ଗଣ୍ଠି->src[0]->ne[0];
    int64_t M             = ଗଣ୍ଠି->src[0]->ne[1];
    int64_t n_expert_used = ଗଣ୍ଠି->src[2]->ne[0];
    int64_t batch         = ଗଣ୍ଠି->src[2]->ne[1];

    int64_t total_flops = batch * n_expert_used * M * (2 * K - 1);

    ଅକ୍ଷର kernel_variant[64];
    snprintf(kernel_variant, ଆକାର(kernel_variant), "%s_%sx%s", kernel_name, src0_type_name,
             ggml_type_name(ଗଣ୍ଠି->src[1]->type));
    ET_PERF_END_EXT("MUL_MAT_ID", kernel_variant, ଗଣ୍ଠି, "flops=%" PRId64 "|n_expert=%lld|n_expert_used=%lld",
                    total_flops, (long long) ଗଣ୍ଠି->src[0]->ne[2], (long long) n_expert_used);

    ଫେରନ୍ତୁ kernel_result;
}

ତାର୍କିକ ggml_et_op_rope(ggml_backend_et_device_context * dev_ctx, ଅଚଳ ggml_tensor * ଗଣ୍ଠି) {
    ET_PERF_START();

    ଯଦି (!dev_ctx || !ଗଣ୍ଠି) {
        GGML_LOG_ERROR("ET: Invalid parameters for ROPE operation\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (!ଗଣ୍ଠି->src[0] || !ଗଣ୍ଠି->src[1]) {
        GGML_LOG_ERROR("ET: ROPE operation missing required inputs\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ ଅକ୍ଷର * kernel_name;

    ଯଦି (ଗଣ୍ଠି->type == GGML_TYPE_F32 && ଗଣ୍ଠି->src[0]->type == GGML_TYPE_F32 && ଗଣ୍ଠି->src[1]->type == GGML_TYPE_I32) {
        kernel_name = "rope_f32";
    } ଅନ୍ୟଥା {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // Pack parameters - copy full tensor structures and op_params
    ggml_et_rope_params params;
    params.src0 = *ଗଣ୍ଠି->src[0];                       // F32 input tensor
    params.src1 = *ଗଣ୍ଠି->src[1];                       // I32 position tensor
    ଯଦି (ଗଣ୍ଠି->src[2]) {
        params.src2 = *ଗଣ୍ଠି->src[2];                   // F32 frequency factors (optional)
    } ଅନ୍ୟଥା {
        memset(&params.src2, 0, ଆକାର(params.src2));  // Zero if not provided
    }
    params.dst = *ଗଣ୍ଠି;                                // F32 output tensor

    params.rope_params.n_past     = ((ଅଚଳ int32_t *) ଗଣ୍ଠି->op_params)[0];
    params.rope_params.n_dims     = ((ଅଚଳ int32_t *) ଗଣ୍ଠି->op_params)[1];
    params.rope_params.mode       = ((ଅଚଳ int32_t *) ଗଣ୍ଠି->op_params)[2];
    params.rope_params.n_ctx      = ((ଅଚଳ int32_t *) ଗଣ୍ଠି->op_params)[3];
    params.rope_params.n_ctx_orig = ((ଅଚଳ int32_t *) ଗଣ୍ଠି->op_params)[4];
    memcpy(&params.rope_params.freq_base, (ଅଚଳ int32_t *) ଗଣ୍ଠି->op_params + 5, ଆକାର(ଦଶମିକ));
    memcpy(&params.rope_params.freq_scale, (ଅଚଳ int32_t *) ଗଣ୍ଠି->op_params + 6, ଆକାର(ଦଶମିକ));
    memcpy(&params.rope_params.ext_factor, (ଅଚଳ int32_t *) ଗଣ୍ଠି->op_params + 7, ଆକାର(ଦଶମିକ));
    memcpy(&params.rope_params.attn_factor, (ଅଚଳ int32_t *) ଗଣ୍ଠି->op_params + 8, ଆକାର(ଦଶମିକ));
    memcpy(&params.rope_params.beta_fast, (ଅଚଳ int32_t *) ଗଣ୍ଠି->op_params + 9, ଆକାର(ଦଶମିକ));
    memcpy(&params.rope_params.beta_slow, (ଅଚଳ int32_t *) ଗଣ୍ଠି->op_params + 10, ଆକାର(ଦଶମିକ));
    ଯଦି (params.rope_params.mode & GGML_ROPE_TYPE_MROPE) {
        memcpy(params.rope_params.sections, (ଅଚଳ int32_t *) ଗଣ୍ଠି->op_params + 11, ଆକାର(int32_t) * 4);
    } ଅନ୍ୟଥା {
        memset(params.rope_params.sections, 0, ଆକାର(params.rope_params.sections));
    }

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ତାର୍କିକ                    cpu_comparison_active = ମିଥ୍ୟା;
    ଯଦି (rope_cpu_compare_config.enabled) {
        GGML_LOG_DEBUG("ET: Initializing CPU comparison for ROPE operation\n");
        ଯଦି (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ଗଣ୍ଠି, GGML_OP_ROPE)) {
            cpu_comparison_active = ସତ୍ୟ;
        } ଅନ୍ୟଥା {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for ROPE operation\n");
        }
    }

    ତାର୍କିକ kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ଆକାର(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ଯଦି (cpu_comparison_active) {
        ଯଦି (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ଗଣ୍ଠି, &rope_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for ROPE operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END_EXT("ROPE", kernel_name, ଗଣ୍ଠି, "mode=0x%x|n_dims=%d|freq_base=%.2f|freq_scale=%.2f",
                    params.rope_params.mode, params.rope_params.n_dims, (ଦ୍ୱିଗୁଣ) params.rope_params.freq_base,
                    (ଦ୍ୱିଗୁଣ) params.rope_params.freq_scale);
    ଫେରନ୍ତୁ kernel_result;
}

ତାର୍କିକ ggml_et_op_rms_norm(ggml_backend_et_device_context * dev_ctx, ଅଚଳ ggml_tensor * ଗଣ୍ଠି) {
    ET_PERF_START();

    ଯଦି (!dev_ctx || !ଗଣ୍ଠି) {
        GGML_LOG_ERROR("ET: Invalid parameters for RMS_NORM operation\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (!ଗଣ୍ଠି->src[0]) {
        GGML_LOG_ERROR("ET: RMS_NORM operation missing required input\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ ଅକ୍ଷର * kernel_name;

    ଯଦି (ଗଣ୍ଠି->type == GGML_TYPE_F32 && ଗଣ୍ଠି->src[0]->type == GGML_TYPE_F32) {
        kernel_name = "rms_norm_f32";

    } ଅନ୍ୟଥା {
        GGML_LOG_ERROR("ET: RMS_NORM operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(ଗଣ୍ଠି->type),
                       ggml_type_name(ଗଣ୍ଠି->src[0]->type));
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଦଶମିକ eps;
    memcpy(&eps, ଗଣ୍ଠି->op_params, ଆକାର(ଦଶମିକ));

    ggml_et_rms_norm_params params;
    params.src0 = *ଗଣ୍ଠି->src[0];  // F32 input tensor
    params.dst  = *ଗଣ୍ଠି;          // F32 output tensor
    params.eps  = eps;            // Epsilon parameter for numerical stability

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ତାର୍କିକ                    cpu_comparison_active = ମିଥ୍ୟା;
    ଯଦି (rms_norm_cpu_compare_config.enabled) {
        ଯଦି (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ଗଣ୍ଠି, GGML_OP_RMS_NORM)) {
            cpu_comparison_active = ସତ୍ୟ;
        } ଅନ୍ୟଥା {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for RMS_NORM operation\n");
        }
    }

    ତାର୍କିକ kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ଆକାର(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ଯଦି (cpu_comparison_active) {
        ଯଦି (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ଗଣ୍ଠି, &rms_norm_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for RMS_NORM operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END_EXT("RMS_NORM", kernel_name, ଗଣ୍ଠି, "eps=%.6f", (ଦ୍ୱିଗୁଣ) eps);
    ଫେରନ୍ତୁ kernel_result;
}

ତାର୍କିକ ggml_et_op_norm(ggml_backend_et_device_context * dev_ctx, ଅଚଳ ggml_tensor * ଗଣ୍ଠି) {
    ET_PERF_START();

    ଯଦି (!dev_ctx || !ଗଣ୍ଠି) {
        GGML_LOG_ERROR("ET: Invalid parameters for NORM operation\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (!ଗଣ୍ଠି->src[0]) {
        GGML_LOG_ERROR("ET: NORM operation missing required input\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ ଅକ୍ଷର * kernel_name;

    ଯଦି (ଗଣ୍ଠି->type == GGML_TYPE_F32 && ଗଣ୍ଠି->src[0]->type == GGML_TYPE_F32) {
        kernel_name = "norm_f32";

    } ଅନ୍ୟଥା {
        GGML_LOG_ERROR("ET: NORM operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(ଗଣ୍ଠି->type),
                       ggml_type_name(ଗଣ୍ଠି->src[0]->type));
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଦଶମିକ eps;
    memcpy(&eps, ଗଣ୍ଠି->op_params, ଆକାର(ଦଶମିକ));

    ggml_et_norm_params params;
    params.src0 = *ଗଣ୍ଠି->src[0];  // F32 input tensor
    params.dst  = *ଗଣ୍ଠି;          // F32 output tensor
    params.eps  = eps;            // Epsilon parameter for numerical stability

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ତାର୍କିକ                    cpu_comparison_active = ମିଥ୍ୟା;
    ଯଦି (norm_cpu_compare_config.enabled) {
        ଯଦି (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ଗଣ୍ଠି, GGML_OP_NORM)) {
            cpu_comparison_active = ସତ୍ୟ;
        } ଅନ୍ୟଥା {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for NORM operation\n");
        }
    }

    ତାର୍କିକ kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ଆକାର(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ଯଦି (cpu_comparison_active) {
        ଯଦି (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ଗଣ୍ଠି, &norm_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for NORM operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END_EXT("NORM", kernel_name, ଗଣ୍ଠି, "eps=%.6f", (ଦ୍ୱିଗୁଣ) eps);
    ଫେରନ୍ତୁ kernel_result;
}

ତାର୍କିକ ggml_et_op_l2_norm(ggml_backend_et_device_context * dev_ctx, ଅଚଳ ggml_tensor * ଗଣ୍ଠି) {
    ET_PERF_START();

    ଯଦି (!dev_ctx || !ଗଣ୍ଠି) {
        GGML_LOG_ERROR("ET: Invalid parameters for L2_NORM operation\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (!ଗଣ୍ଠି->src[0]) {
        GGML_LOG_ERROR("ET: L2_NORM operation missing required input\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ ଅକ୍ଷର * kernel_name;

    ଯଦି (ଗଣ୍ଠି->type == GGML_TYPE_F32 && ଗଣ୍ଠି->src[0]->type == GGML_TYPE_F32) {
        kernel_name = "l2_norm_f32";

    } ଅନ୍ୟଥା {
        GGML_LOG_ERROR("ET: L2_NORM operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(ଗଣ୍ଠି->type),
                       ggml_type_name(ଗଣ୍ଠି->src[0]->type));
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଦଶମିକ eps;
    memcpy(&eps, ଗଣ୍ଠି->op_params, ଆକାର(ଦଶମିକ));

    ggml_et_l2_norm_params params;
    params.src0 = *ଗଣ୍ଠି->src[0];  // F32 input tensor
    params.dst  = *ଗଣ୍ଠି;          // F32 output tensor
    params.eps  = eps;            // Epsilon parameter for numerical stability

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ତାର୍କିକ                    cpu_comparison_active = ମିଥ୍ୟା;
    ଯଦି (l2_norm_cpu_compare_config.enabled) {
        ଯଦି (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ଗଣ୍ଠି, GGML_OP_L2_NORM)) {
            cpu_comparison_active = ସତ୍ୟ;
        } ଅନ୍ୟଥା {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for L2_NORM operation\n");
        }
    }

    ତାର୍କିକ kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ଆକାର(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ଯଦି (cpu_comparison_active) {
        ଯଦି (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ଗଣ୍ଠି, &l2_norm_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for L2_NORM operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END_EXT("L2_NORM", kernel_name, ଗଣ୍ଠି, "eps=%.6f", (ଦ୍ୱିଗୁଣ) eps);
    ଫେରନ୍ତୁ kernel_result;
}

ତାର୍କିକ ggml_et_op_group_norm(ggml_backend_et_device_context * dev_ctx, ଅଚଳ ggml_tensor * ଗଣ୍ଠି) {
    ET_PERF_START();

    ଯଦି (!dev_ctx || !ଗଣ୍ଠି) {
        GGML_LOG_ERROR("ET: Invalid parameters for GROUP_NORM operation\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (!ଗଣ୍ଠି->src[0]) {
        GGML_LOG_ERROR("ET: GROUP_NORM operation missing required input\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (ଗଣ୍ଠି->type != GGML_TYPE_F32 || ଗଣ୍ଠି->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: GROUP_NORM operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(ଗଣ୍ଠି->type),
                       ggml_type_name(ଗଣ୍ଠି->src[0]->type));
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ int32_t n_groups = ggml_get_op_params_i32(ଗଣ୍ଠି, 0);
    ଦଶମିକ         eps;
    memcpy(&eps, (ଅଚଳ ଦଶମିକ *) ଗଣ୍ଠି->op_params + 1, ଆକାର(ଦଶମିକ));

    ggml_et_group_norm_params params;
    params.src0     = *ଗଣ୍ଠି->src[0];
    params.dst      = *ଗଣ୍ଠି;
    params.n_groups = n_groups;
    params.eps      = eps;

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ତାର୍କିକ                    cpu_comparison_active = ମିଥ୍ୟା;
    ଯଦି (group_norm_cpu_compare_config.enabled) {
        ଯଦି (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ଗଣ୍ଠି, GGML_OP_GROUP_NORM)) {
            cpu_comparison_active = ସତ୍ୟ;
        } ଅନ୍ୟଥା {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for GROUP_NORM operation\n");
        }
    }

    ତାର୍କିକ kernel_result = ggml_et_launch_kernel(dev_ctx, "group_norm_f32", &params, ଆକାର(params), 0xFFFFFFFF);

    ଯଦି (cpu_comparison_active) {
        ଯଦି (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ଗଣ୍ଠି, &group_norm_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for GROUP_NORM operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END_EXT("GROUP_NORM", "group_norm_f32", ଗଣ୍ଠି, "eps=%.6f|n_groups=%d", (ଦ୍ୱିଗୁଣ) eps, n_groups);
    ଫେରନ୍ତୁ kernel_result;
}

ତାର୍କିକ ggml_et_op_im2col(ggml_backend_et_device_context * dev_ctx, ଅଚଳ ggml_tensor * ଗଣ୍ଠି) {
    ET_PERF_START();

    ଯଦି (!dev_ctx || !ଗଣ୍ଠି) {
        GGML_LOG_ERROR("ET: Invalid parameters for IM2COL operation\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (!ଗଣ୍ଠି->src[0] || !ଗଣ୍ଠି->src[1]) {
        GGML_LOG_ERROR("ET: IM2COL operation missing required inputs\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ ତାର୍କିକ supported_types =
        (ଗଣ୍ଠି->type == GGML_TYPE_F32 && ଗଣ୍ଠି->src[1]->type == GGML_TYPE_F32) ||
        (ଗଣ୍ଠି->type == GGML_TYPE_F16 && (ଗଣ୍ଠି->src[1]->type == GGML_TYPE_F16 || ଗଣ୍ଠି->src[1]->type == GGML_TYPE_F32));

    ଯଦି (!supported_types) {
        GGML_LOG_ERROR("ET: IM2COL operation with unsupported types: dst=%s src1=%s\n", ggml_type_name(ଗଣ୍ଠି->type),
                       ggml_type_name(ଗଣ୍ଠି->src[1]->type));
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ggml_et_im2col_params params;
    params.src0 = *ଗଣ୍ଠି->src[0];
    params.src1 = *ଗଣ୍ଠି->src[1];
    params.dst  = *ଗଣ୍ଠି;

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ତାର୍କିକ                    cpu_comparison_active = ମିଥ୍ୟା;
    ଯଦି (im2col_cpu_compare_config.enabled) {
        ଯଦି (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ଗଣ୍ଠି, GGML_OP_IM2COL)) {
            cpu_comparison_active = ସତ୍ୟ;
        } ଅନ୍ୟଥା {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for IM2COL operation\n");
        }
    }

    ତାର୍କିକ kernel_result = ggml_et_launch_kernel(dev_ctx, "im2col", &params, ଆକାର(params), 0xFFFFFFFF);

    ଯଦି (cpu_comparison_active) {
        ଯଦି (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ଗଣ୍ଠି, &im2col_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for IM2COL operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("IM2COL", "im2col", ଗଣ୍ଠି);
    ଫେରନ୍ତୁ kernel_result;
}

ତାର୍କିକ ggml_et_op_conv_2d(ggml_backend_et_device_context * dev_ctx, ଅଚଳ ggml_tensor * ଗଣ୍ଠି) {
    ET_PERF_START();

    ଯଦି (!dev_ctx || !ଗଣ୍ଠି) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (!ଗଣ୍ଠି->src[0] || !ଗଣ୍ଠି->src[1]) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (!ଗଣ୍ଠି->data || !ଗଣ୍ଠି->src[0]->data || !ଗଣ୍ଠି->src[1]->data) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // Kernel constraints (mirror supports_op; recheck here as a guard).
    ଅଚଳ ggml_tensor * flt = ଗଣ୍ଠି->src[0];  // [Kw, Kh, Cin, Cout]
    ଅଚଳ ggml_tensor * in  = ଗଣ୍ଠି->src[1];  // [W,  H,  Cin, N]
    ଯଦି (ଗଣ୍ଠି->type != GGML_TYPE_F32 || flt->type != GGML_TYPE_F32 || in->type != GGML_TYPE_F32) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ int32_t s0 = ggml_get_op_params_i32(ଗଣ୍ଠି, 0);
    ଅଚଳ int32_t s1 = ggml_get_op_params_i32(ଗଣ୍ଠି, 1);
    ଅଚଳ int32_t p0 = ggml_get_op_params_i32(ଗଣ୍ଠି, 2);
    ଅଚଳ int32_t p1 = ggml_get_op_params_i32(ଗଣ୍ଠି, 3);
    ଅଚଳ int32_t d0 = ggml_get_op_params_i32(ଗଣ୍ଠି, 4);
    ଅଚଳ int32_t d1 = ggml_get_op_params_i32(ଗଣ୍ଠି, 5);

    ଯଦି (s0 < 1 || s1 < 1) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (d0 != 1 || d1 != 1) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (flt->ne[2] % 16 != 0 || flt->ne[3] % 16 != 0) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (in->ne[3] != 1) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (ଗଣ୍ଠି->ne[0] <= 0) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;  // OW > 0 (any width OK; staging path handles non-16)
    }
    (ଶୂନ୍ୟ) p0;
    (ଶୂନ୍ୟ) p1;

    ggml_et_ଦ୍ୱିମିକ_params params;
    params.src0 = *ଗଣ୍ଠି->src[0];
    params.src1 = *ଗଣ୍ଠି->src[1];
    params.dst  = *ଗଣ୍ଠି;

    ତାର୍କିକ kernel_result = ggml_et_launch_kernel(dev_ctx, "conv_2d_f32_me", &params, ଆକାର(params), 0xFFFFFFFFu);

    ET_PERF_END("CONV_2D", "conv_2d_f32_me", ଗଣ୍ଠି);
    ଫେରନ୍ତୁ kernel_result;
}

ତାର୍କିକ ggml_et_op_softmax(ggml_backend_et_device_context * dev_ctx, ଅଚଳ ggml_tensor * ଗଣ୍ଠି) {
    ET_PERF_START();

    ଯଦି (!dev_ctx || !ଗଣ୍ଠି) {
        GGML_LOG_ERROR("ET: Invalid parameters for SOFTMAX operation\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (!ଗଣ୍ଠି->src[0]) {
        GGML_LOG_ERROR("ET: SOFTMAX operation missing required input\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ ଅକ୍ଷର * kernel_name;

    ଯଦି (ଗଣ୍ଠି->type == GGML_TYPE_F32 && ଗଣ୍ଠି->src[0]->type == GGML_TYPE_F32) {
        kernel_name = "softmax_f32";

    } ଅନ୍ୟଥା {
        GGML_LOG_ERROR("ET: SOFTMAX operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(ଗଣ୍ଠି->type),
                       ggml_type_name(ଗଣ୍ଠି->src[0]->type));
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // Validate contiguity requirements
    ଯଦି (!ggml_is_contiguous(ଗଣ୍ଠି)) {
        GGML_LOG_ERROR("ET: SOFTMAX operation requires contiguous destination tensor\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (!ggml_is_contiguous(ଗଣ୍ଠି->src[0])) {
        GGML_LOG_ERROR("ET: SOFTMAX operation requires contiguous source tensor\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // Check optional mask tensor
    ଯଦି (ଗଣ୍ଠି->src[1]) {
        ଯଦି (ଗଣ୍ଠି->src[1]->type != GGML_TYPE_F32) {
            GGML_LOG_ERROR("ET: SOFTMAX operation with unsupported mask type: %s (F32 required)\n",
                           ggml_type_name(ଗଣ୍ଠି->src[1]->type));
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଯଦି (!ggml_is_contiguous(ଗଣ୍ଠି->src[1])) {
            GGML_LOG_ERROR("ET: SOFTMAX operation requires contiguous mask tensor\n");
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    }

    // Check optional sinks tensor
    ଯଦି (ଗଣ୍ଠି->src[2]) {
        ଯଦି (ଗଣ୍ଠି->src[2]->type != GGML_TYPE_F32) {
            GGML_LOG_ERROR("ET: SOFTMAX operation with unsupported sinks type: %s (F32 required)\n",
                           ggml_type_name(ଗଣ୍ଠି->src[2]->type));
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଯଦି (!ggml_is_contiguous(ଗଣ୍ଠି->src[2])) {
            GGML_LOG_ERROR("ET: SOFTMAX operation requires contiguous sinks tensor\n");
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    }

    // Extract scale and max_bias from op_params
    ଦଶମିକ scale    = 1.0f;
    ଦଶମିକ max_bias = 0.0f;
    ଯଦି (ଗଣ୍ଠି->op_params) {
        memcpy(&scale, (ଅଚଳ ଦଶମିକ *) ଗଣ୍ଠି->op_params + 0, ଆକାର(ଦଶମିକ));
        memcpy(&max_bias, (ଅଚଳ ଦଶମିକ *) ଗଣ୍ଠି->op_params + 1, ଆକାର(ଦଶମିକ));
    }

    ggml_et_softmax_params params;
    params.src0 = *ଗଣ୍ଠି->src[0];                       // F32 input tensor
    ଯଦି (ଗଣ୍ଠି->src[1]) {
        params.src1 = *ଗଣ୍ଠି->src[1];                   // F32 mask tensor
    } ଅନ୍ୟଥା {
        memset(&params.src1, 0, ଆକାର(params.src1));  // Zero if no mask
    }
    ଯଦି (ଗଣ୍ଠି->src[2]) {
        params.src2 = *ଗଣ୍ଠି->src[2];                   // F32 sinks tensor
    } ଅନ୍ୟଥା {
        memset(&params.src2, 0, ଆକାର(params.src2));  // Zero if no sinks
    }
    params.dst      = *ଗଣ୍ଠି;                           // F32 output tensor
    params.scale    = scale;                           // Scale factor
    params.max_bias = max_bias;                        // ALiBi bias

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ତାର୍କିକ                    cpu_comparison_active = ମିଥ୍ୟା;
    ଯଦି (softmax_cpu_compare_config.enabled) {
        ଯଦି (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ଗଣ୍ଠି, GGML_OP_SOFT_MAX)) {
            cpu_comparison_active = ସତ୍ୟ;
        } ଅନ୍ୟଥା {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for SOFTMAX operation\n");
        }
    }

    ତାର୍କିକ kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ଆକାର(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ଯଦି (cpu_comparison_active) {
        ଯଦି (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ଗଣ୍ଠି, &softmax_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for SOFTMAX operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END_EXT("SOFTMAX", kernel_name, ଗଣ୍ଠି, "scale=%.6f|max_bias=%.6f|has_mask=%s", (ଦ୍ୱିଗୁଣ) scale,
                    (ଦ୍ୱିଗୁଣ) max_bias, ଗଣ୍ଠି->src[1] ? "yes" : "no");
    ଫେରନ୍ତୁ kernel_result;
}

ତାର୍କିକ ggml_et_op_flash_attn_ext(ggml_backend_et_device_context * dev_ctx, ଅଚଳ ggml_tensor * ଗଣ୍ଠି) {
    ET_PERF_START();

    ଯଦି (!dev_ctx || !ଗଣ୍ଠି) {
        GGML_LOG_ERROR("ET: Invalid parameters for FLASH_ATTN_EXT operation\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (!ଗଣ୍ଠି->src[0] || !ଗଣ୍ଠି->src[1] || !ଗଣ୍ଠି->src[2]) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT operation missing required inputs\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (ଗଣ୍ଠି->type != GGML_TYPE_F32 || ଗଣ୍ଠି->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT requires F32 Q and dst, got dst=%s q=%s\n", ggml_type_name(ଗଣ୍ଠି->type),
                       ggml_type_name(ଗଣ୍ଠି->src[0]->type));
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // K and V can be F16 or F32
    ଯଦି ((ଗଣ୍ଠି->src[1]->type != GGML_TYPE_F32 && ଗଣ୍ଠି->src[1]->type != GGML_TYPE_F16) ||
        (ଗଣ୍ଠି->src[2]->type != GGML_TYPE_F32 && ଗଣ୍ଠି->src[2]->type != GGML_TYPE_F16)) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT K/V must be F16 or F32, got k=%s v=%s\n", ggml_type_name(ଗଣ୍ଠି->src[1]->type),
                       ggml_type_name(ଗଣ୍ଠି->src[2]->type));
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (ଗଣ୍ଠି->src[4] != nullptr) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT baseline kernel does not support sinks\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // Mask is optional; if present must be F16 or F32
    ଯଦି (ଗଣ୍ଠି->src[3] != nullptr && ଗଣ୍ଠି->src[3]->type != GGML_TYPE_F32 && ଗଣ୍ଠି->src[3]->type != GGML_TYPE_F16) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT mask must be F16 or F32, got %s\n", ggml_type_name(ଗଣ୍ଠି->src[3]->type));
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // Q and dst must be row-contiguous F32
    ଯଦି (!ggml_is_contiguous_rows(ଗଣ୍ଠି) || !ggml_is_contiguous_rows(ଗଣ୍ଠି->src[0])) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT requires row-contiguous Q and dst\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (ଗଣ୍ଠି->nb[0] != ଆକାର(ଦଶମିକ) || ଗଣ୍ଠି->src[0]->nb[0] != ଆକାର(ଦଶମିକ)) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT requires contiguous F32 rows for Q and dst\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // K/V must have element-sized stride in dim 0
    ଅଚଳ size_t k_elem = ଗଣ୍ଠି->src[1]->type == GGML_TYPE_F16 ? 2 : 4;
    ଅଚଳ size_t v_elem = ଗଣ୍ଠି->src[2]->type == GGML_TYPE_F16 ? 2 : 4;
    ଯଦି (ଗଣ୍ଠି->src[1]->nb[0] != k_elem || ଗଣ୍ଠି->src[2]->nb[0] != v_elem) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT K/V must have element-sized stride in dim 0\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଦଶମିକ scale         = 1.0f;
    ଦଶମିକ max_bias      = 0.0f;
    ଦଶମିକ logit_softcap = 0.0f;
    memcpy(&scale, (ଅଚଳ ଦଶମିକ *) ଗଣ୍ଠି->op_params + 0, ଆକାର(scale));
    memcpy(&max_bias, (ଅଚଳ ଦଶମିକ *) ଗଣ୍ଠି->op_params + 1, ଆକାର(max_bias));
    memcpy(&logit_softcap, (ଅଚଳ ଦଶମିକ *) ଗଣ୍ଠି->op_params + 2, ଆକାର(logit_softcap));

    ଯଦି (max_bias != 0.0f || logit_softcap != 0.0f) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT baseline kernel does not support max_bias or logit_softcap\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ ggml_prec prec = ggml_flash_attn_ext_get_prec(ଗଣ୍ଠି);
    ଯଦି (prec != GGML_PREC_F32 && prec != GGML_PREC_DEFAULT) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT baseline kernel only supports F32 precision\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // dk must match between Q and K; dv must match between V and dst
    ଯଦି (ଗଣ୍ଠି->src[0]->ne[0] != ଗଣ୍ଠି->src[1]->ne[0]) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT dk mismatch: Q=%lld K=%lld\n", (long long) ଗଣ୍ଠି->src[0]->ne[0],
                       (long long) ଗଣ୍ଠି->src[1]->ne[0]);
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (ଗଣ୍ଠି->src[2]->ne[0] != ଗଣ୍ଠି->ne[0]) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT dv mismatch: V=%lld dst=%lld\n", (long long) ଗଣ୍ଠି->src[2]->ne[0],
                       (long long) ଗଣ୍ଠି->ne[0]);
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (ଗଣ୍ଠି->src[2]->ne[0] > 512) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT dv=%lld exceeds maximum 512\n", (long long) ଗଣ୍ଠି->src[2]->ne[0]);
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (ଗଣ୍ଠି->src[0]->ne[0] > 512) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT dk=%lld exceeds maximum 512\n", (long long) ଗଣ୍ଠି->src[0]->ne[0]);
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // GQA: n_head_q must be a multiple of n_head_kv
    ଅଚଳ int64_t nhq = ଗଣ୍ଠି->src[0]->ne[2];
    ଅଚଳ int64_t nhk = ଗଣ୍ଠି->src[1]->ne[2];
    ଯଦି (nhq % nhk != 0) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT n_head_q (%lld) not divisible by n_head_kv (%lld)\n", (long long) nhq,
                       (long long) nhk);
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // K and V must have matching sequence length, heads, and batch dims
    ଯଦି (ଗଣ୍ଠି->src[1]->ne[1] != ଗଣ୍ଠି->src[2]->ne[1] || ଗଣ୍ଠି->src[1]->ne[2] != ଗଣ୍ଠି->src[2]->ne[2] ||
        ଗଣ୍ଠି->src[1]->ne[3] != ଗଣ୍ଠି->src[2]->ne[3]) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT K/V shape mismatch\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // dst layout checks: [dv, nhq, nq, no]
    ଯଦି (ଗଣ୍ଠି->src[0]->ne[1] != ଗଣ୍ଠି->ne[2] || ଗଣ୍ଠି->src[0]->ne[2] != ଗଣ୍ଠି->ne[1] ||
        ଗଣ୍ଠି->src[0]->ne[3] != ଗଣ୍ଠି->ne[3]) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT dst shape mismatch\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // Batch dims: Q batch must match K batch
    ଯଦି (ଗଣ୍ଠି->src[0]->ne[3] != ଗଣ୍ଠି->src[1]->ne[3]) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT batch dimension mismatch\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ggml_et_flash_attn_ext_params params;
    memset(&params, 0, ଆକାର(params));
    params.src0 = *ଗଣ୍ଠି->src[0];
    params.src1 = *ଗଣ୍ଠି->src[1];
    params.src2 = *ଗଣ୍ଠି->src[2];
    ଯଦି (ଗଣ୍ଠି->src[3] != nullptr) {
        params.mask     = *ଗଣ୍ଠି->src[3];
        params.has_mask = 1;
    }
    params.dst   = *ଗଣ୍ଠି;
    params.scale = scale;

    // Use matrix engine kernel when K/V are F16 and dk is a multiple of 32
    ଅଚଳ ଅକ୍ଷର * kernel_name;
    ଯଦି (ଗଣ୍ଠି->src[1]->type == GGML_TYPE_F16 && ଗଣ୍ଠି->src[2]->type == GGML_TYPE_F16 && (ଗଣ୍ଠି->src[0]->ne[0] % 32) == 0) {
        kernel_name = "flash_attn_ext_f16_me";
    } ଅନ୍ୟଥା {
        kernel_name = "flash_attn_ext_f32";
    }

    ଅଚଳ ତାର୍କିକ kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ଆକାର(params), 0xFFFFFFFF);

    ET_PERF_END_EXT("FLASH_ATTN_EXT", kernel_name, ଗଣ୍ଠି, "scale=%.6f", (ଦ୍ୱିଗୁଣ) scale);
    ଫେରନ୍ତୁ kernel_result;
}

ତାର୍କିକ ggml_et_op_get_rows(ggml_backend_et_device_context * dev_ctx, ଅଚଳ ggml_tensor * ଗଣ୍ଠି) {
    ET_PERF_START();

    ଯଦି (!dev_ctx || !ଗଣ୍ଠି) {
        GGML_LOG_ERROR("ET: Invalid parameters for GET_ROWS operation\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (!ଗଣ୍ଠି->src[0] || !ଗଣ୍ଠି->src[1]) {
        GGML_LOG_ERROR("ET: GET_ROWS operation missing required inputs\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ ଅକ୍ଷର * kernel_name;

    ଯଦି (ଗଣ୍ଠି->type == GGML_TYPE_F32 && ଗଣ୍ଠି->src[1]->type == GGML_TYPE_I32 &&
        (ଗଣ୍ଠି->src[0]->type == GGML_TYPE_F32 || ଗଣ୍ଠି->src[0]->type == GGML_TYPE_F16 ||
         ଗଣ୍ଠି->src[0]->type == GGML_TYPE_Q4_0 || ଗଣ୍ଠି->src[0]->type == GGML_TYPE_Q8_0 ||
         ଗଣ୍ଠି->src[0]->type == GGML_TYPE_Q4_K)) {
        kernel_name = "get_rows_f32";

    } ଅନ୍ୟଥା {
        GGML_LOG_ERROR("ET: GET_ROWS operation with unsupported types: dst=%s src0=%s src1=%s\n",
                       ggml_type_name(ଗଣ୍ଠି->type), ggml_type_name(ଗଣ୍ଠି->src[0]->type),
                       ggml_type_name(ଗଣ୍ଠି->src[1]->type));
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // Validate contiguity requirements
    ଯଦି (!ggml_is_contiguous(ଗଣ୍ଠି)) {
        GGML_LOG_ERROR("ET: GET_ROWS operation requires contiguous destination tensor\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (!ggml_is_contiguous(ଗଣ୍ଠି->src[0])) {
        GGML_LOG_ERROR("ET: GET_ROWS operation requires contiguous data tensor\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (!ggml_is_contiguous(ଗଣ୍ଠି->src[1])) {
        GGML_LOG_ERROR("ET: GET_ROWS operation requires contiguous indices tensor\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // Validate dimension constraints from ggml implementation
    ଯଦି (ଗଣ୍ଠି->src[0]->ne[2] != ଗଣ୍ଠି->src[1]->ne[1] || ଗଣ୍ଠି->src[1]->ne[3] != 1) {
        GGML_LOG_ERROR(
            "ET: GET_ROWS operation dimension constraint failed: src0.ne[2]=%lld != src1.ne[1]=%lld or src1.ne[3]=%lld "
            "!= 1\n",
            (long long) ଗଣ୍ଠି->src[0]->ne[2], (long long) ଗଣ୍ଠି->src[1]->ne[1], (long long) ଗଣ୍ଠି->src[1]->ne[3]);
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ggml_et_get_rows_params params;
    params.src0 = *ଗଣ୍ଠି->src[0];  // Data tensor (F32 or Q8_0)
    params.src1 = *ଗଣ୍ଠି->src[1];  // Indices tensor (I32)
    params.dst  = *ଗଣ୍ଠି;          // Output tensor (F32)

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ତାର୍କିକ                    cpu_comparison_active = ମିଥ୍ୟା;
    ଯଦି (get_rows_cpu_compare_config.enabled) {
        ଯଦି (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ଗଣ୍ଠି, GGML_OP_GET_ROWS)) {
            cpu_comparison_active = ସତ୍ୟ;
        } ଅନ୍ୟଥା {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for GET_ROWS operation\n");
        }
    }

    ତାର୍କିକ kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ଆକାର(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ଯଦି (cpu_comparison_active) {
        ଯଦି (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ଗଣ୍ଠି, &get_rows_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for GET_ROWS operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("GET_ROWS", kernel_name, ଗଣ୍ଠି);
    ଫେରନ୍ତୁ kernel_result;
}

ତାର୍କିକ ggml_et_op_cont(ggml_backend_et_device_context * dev_ctx, ଅଚଳ ggml_tensor * ଗଣ୍ଠି) {
    ET_PERF_START();

    // Validate source tensor exists
    ଯଦି (!ଗଣ୍ଠି->src[0]) {
        GGML_LOG_ERROR("ET: CONT operation missing source tensor\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // Validate types match (input and output must be same type)
    ଯଦି (ଗଣ୍ଠି->type != ଗଣ୍ଠି->src[0]->type) {
        GGML_LOG_ERROR("ET: CONT operation type mismatch: src=%s dst=%s\n", ggml_type_name(ଗଣ୍ଠି->src[0]->type),
                       ggml_type_name(ଗଣ୍ଠି->type));
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // Validate supported types
    ଯଦି (ଗଣ୍ଠି->type != GGML_TYPE_F32 && ଗଣ୍ଠି->type != GGML_TYPE_F16) {
        GGML_LOG_ERROR("ET: CONT operation unsupported type: %s (only F32 and F16 supported)\n",
                       ggml_type_name(ଗଣ୍ଠି->type));
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // Validate contiguity - output must be contiguous, input can be non-contiguous
    ଯଦି (!ggml_is_contiguous(ଗଣ୍ଠି)) {
        GGML_LOG_ERROR("ET: CONT operation requires contiguous output tensor\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // Select kernel based on type
    ଅଚଳ ଅକ୍ଷର * kernel_name;
    ଯଦି (ଗଣ୍ଠି->type == GGML_TYPE_F32) {
        kernel_name = "cont_f32";
    } ଅନ୍ୟଥା ଯଦି (ଗଣ୍ଠି->type == GGML_TYPE_F16) {
        kernel_name = "cont_f16";
    } ଅନ୍ୟଥା {
        GGML_LOG_ERROR("ET: CONT operation with unsupported type: %s\n", ggml_type_name(ଗଣ୍ଠି->type));
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ggml_et_cont_params params;
    params.src0 = *ଗଣ୍ଠି->src[0];  // Input tensor (potentially non-contiguous)
    params.dst  = *ଗଣ୍ଠି;          // Output tensor (contiguous)

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ତାର୍କିକ                    cpu_comparison_active = ମିଥ୍ୟା;
    ଯଦି (cont_cpu_compare_config.enabled) {
        ଯଦି (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ଗଣ୍ଠି, GGML_OP_CONT)) {
            cpu_comparison_active = ସତ୍ୟ;
        } ଅନ୍ୟଥା {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for CONT operation\n");
        }
    }

    ତାର୍କିକ kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ଆକାର(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ଯଦି (cpu_comparison_active) {
        ଯଦି (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ଗଣ୍ଠି, &cont_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for CONT operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("CONT", kernel_name, ଗଣ୍ଠି);
    ଫେରନ୍ତୁ kernel_result;
}

ତାର୍କିକ ggml_et_op_cumsum(ggml_backend_et_device_context * dev_ctx, ଅଚଳ ggml_tensor * ଗଣ୍ଠି) {
    ET_PERF_START();

    ଯଦି (!dev_ctx || !ଗଣ୍ଠି || !ଗଣ୍ଠି->src[0]) {
        GGML_LOG_ERROR("ET: Invalid parameters for CUMSUM operation\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (ଗଣ୍ଠି->type != GGML_TYPE_F32 || ଗଣ୍ଠି->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: CUMSUM operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(ଗଣ୍ଠି->type),
                       ggml_type_name(ଗଣ୍ଠି->src[0]->type));
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ ଅକ୍ଷର * kernel_name = "cumsum_f32";

    ggml_et_cumsum_params params;
    params.src0 = *ଗଣ୍ଠି->src[0];
    params.dst  = *ଗଣ୍ଠି;

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ତାର୍କିକ                    cpu_comparison_active = ମିଥ୍ୟା;
    ଯଦି (cumsum_cpu_compare_config.enabled) {
        ଯଦି (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ଗଣ୍ଠି, GGML_OP_CUMSUM)) {
            cpu_comparison_active = ସତ୍ୟ;
        } ଅନ୍ୟଥା {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for CUMSUM operation\n");
        }
    }

    ତାର୍କିକ kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ଆକାର(params), 0xFFFFFFFF);

    ଯଦି (cpu_comparison_active) {
        ଯଦି (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ଗଣ୍ଠି, &cumsum_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for CUMSUM operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("CUMSUM", kernel_name, ଗଣ୍ଠି);
    ଫେରନ୍ତୁ kernel_result;
}

ତାର୍କିକ ggml_et_op_cpy(ggml_backend_et_device_context * dev_ctx, ଅଚଳ ggml_tensor * ଗଣ୍ଠି) {
    ET_PERF_START();

    // CPY copies data from src[0] into the layout of dst (which matches src[1])
    // For same-type with contiguous dst, this is identical to CONT
    ଯଦି (!ଗଣ୍ଠି->src[0]) {
        GGML_LOG_ERROR("ET: CPY operation missing source tensor\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // Scalar / zero-element special path: if any dimension is 0, nothing to copy
    ଅଚଳ int64_t nelements = ଗଣ୍ଠି->ne[0] * ଗଣ୍ଠି->ne[1] * ଗଣ୍ଠି->ne[2] * ଗଣ୍ଠି->ne[3];
    ଯଦି (nelements == 0) {
        GGML_LOG_DEBUG("ET: CPY no-op (zero elements): ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]\n",
                       ଗଣ୍ଠି->ne[0], ଗଣ୍ଠି->ne[1], ଗଣ୍ଠି->ne[2], ଗଣ୍ଠି->ne[3]);
        ET_PERF_END("CPY", "noop", ଗଣ୍ଠି);
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }

    // Only F32 and F16 supported for dst
    ଯଦି (ଗଣ୍ଠି->type != GGML_TYPE_F32 && ଗଣ୍ଠି->type != GGML_TYPE_F16) {
        GGML_LOG_ERROR("ET: CPY unsupported dst type: %s\n", ggml_type_name(ଗଣ୍ଠି->type));
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // Select kernel based on src/dst type combination
    ଅଚଳ ଅକ୍ଷର * kernel_name;
    ଯଦି (ଗଣ୍ଠି->src[0]->type == GGML_TYPE_F32 && ଗଣ୍ଠି->type == GGML_TYPE_F32) {
        kernel_name = "cont_f32";
    } ଅନ୍ୟଥା ଯଦି (ଗଣ୍ଠି->src[0]->type == GGML_TYPE_F16 && ଗଣ୍ଠି->type == GGML_TYPE_F16) {
        kernel_name = "cont_f16";
    } ଅନ୍ୟଥା ଯଦି (ଗଣ୍ଠି->src[0]->type == GGML_TYPE_F32 && ଗଣ୍ଠି->type == GGML_TYPE_F16) {
        kernel_name = "cpy_f32_f16";
    } ଅନ୍ୟଥା {
        GGML_LOG_ERROR("ET: CPY unsupported type combination: src=%s dst=%s\n", ggml_type_name(ଗଣ୍ଠି->src[0]->type),
                       ggml_type_name(ଗଣ୍ଠି->type));
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ggml_et_cont_params params;
    params.src0 = *ଗଣ୍ଠି->src[0];
    params.dst  = *ଗଣ୍ଠି;

    // CPU comparison for debugging
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ତାର୍କିକ                    cpu_comparison_active = ମିଥ୍ୟା;
    ଯଦି (cont_cpu_compare_config.enabled) {
        ଯଦି (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ଗଣ୍ଠି, GGML_OP_CPY)) {
            cpu_comparison_active = ସତ୍ୟ;
        } ଅନ୍ୟଥା {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for CPY operation\n");
        }
    }

    ତାର୍କିକ kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ଆକାର(params), 0xFFFFFFFF);

    ଯଦି (cpu_comparison_active) {
        ଯଦି (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ଗଣ୍ଠି, &cont_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for CPY operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("CPY", kernel_name, ଗଣ୍ଠି);
    ଫେରନ୍ତୁ kernel_result;
}

ତାର୍କିକ ggml_et_op_concat(ggml_backend_et_device_context * dev_ctx, ଅଚଳ ggml_tensor * ଗଣ୍ଠି) {
    ET_PERF_START();

    ଯଦି (!dev_ctx || !ଗଣ୍ଠି) {
        GGML_LOG_ERROR("ET: Invalid parameters for CONCAT operation\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (!ଗଣ୍ଠି->src[0] || !ଗଣ୍ଠି->src[1]) {
        GGML_LOG_ERROR("ET: CONCAT operation missing required inputs\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ ଅକ୍ଷର * kernel_name;

    ଯଦି (ଗଣ୍ଠି->type == GGML_TYPE_F32 && ଗଣ୍ଠି->src[0]->type == GGML_TYPE_F32 && ଗଣ୍ଠି->src[1]->type == GGML_TYPE_F32) {
        kernel_name = "concat_f32";

    } ଅନ୍ୟଥା {
        GGML_LOG_ERROR("ET: CONCAT operation with unsupported types: dst=%s src0=%s src1=%s\n",
                       ggml_type_name(ଗଣ୍ଠି->type), ggml_type_name(ଗଣ୍ଠି->src[0]->type),
                       ggml_type_name(ଗଣ୍ଠି->src[1]->type));
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    int32_t dim;
    memcpy(&dim, ଗଣ୍ଠି->op_params, ଆକାର(int32_t));

    ggml_et_concat_params params;
    params.src0 = *ଗଣ୍ଠି->src[0];
    params.src1 = *ଗଣ୍ଠି->src[1];
    params.dst  = *ଗଣ୍ଠି;
    params.dim  = dim;

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ତାର୍କିକ                    cpu_comparison_active = ମିଥ୍ୟା;
    ଯଦି (concat_cpu_compare_config.enabled) {
        ଯଦି (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ଗଣ୍ଠି, GGML_OP_CONCAT)) {
            cpu_comparison_active = ସତ୍ୟ;
        } ଅନ୍ୟଥା {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for CONCAT operation\n");
        }
    }

    ତାର୍କିକ kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ଆକାର(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ଯଦି (cpu_comparison_active) {
        ଯଦି (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ଗଣ୍ଠି, &concat_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for CONCAT operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END_EXT("CONCAT", kernel_name, ଗଣ୍ଠି, "dim=%d", dim);
    ଫେରନ୍ତୁ kernel_result;
}

ତାର୍କିକ ggml_et_op_repeat(ggml_backend_et_device_context * dev_ctx, ଅଚଳ ggml_tensor * ଗଣ୍ଠି) {
    ET_PERF_START();

    ଯଦି (!dev_ctx || !ଗଣ୍ଠି) {
        GGML_LOG_ERROR("ET: Invalid parameters for REPEAT operation\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (!ଗଣ୍ଠି->src[0]) {
        GGML_LOG_ERROR("ET: REPEAT operation missing required input\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ ଅକ୍ଷର * kernel_name;

    ଯଦି (ଗଣ୍ଠି->type == GGML_TYPE_F32 && ଗଣ୍ଠି->src[0]->type == GGML_TYPE_F32) {
        // No-op REPEAT (every repeat factor is 1): the output is just a copy
        // of the input. Route to cont_f32, whose contiguous fast path handles
        // arbitrary sizes (including those rejected by repeat_f32's gate,
        // e.g. ne[0]=1).
        ଯଦି (ggml_are_same_shape(ଗଣ୍ଠି->src[0], ଗଣ୍ଠି)) {
            kernel_name = "cont_f32";
        } ଅନ୍ୟଥା {
            kernel_name = "repeat_f32";
        }

    } ଅନ୍ୟଥା {
        GGML_LOG_ERROR("ET: REPEAT operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(ଗଣ୍ଠି->type),
                       ggml_type_name(ଗଣ୍ଠି->src[0]->type));
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // ggml_et_cont_params and ggml_et_repeat_params have identical layouts
    // (just src0 + dst), so the same payload works for either kernel.
    ggml_et_repeat_params params;
    params.src0 = *ଗଣ୍ଠି->src[0];
    params.dst  = *ଗଣ୍ଠି;

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ତାର୍କିକ                    cpu_comparison_active = ମିଥ୍ୟା;
    ଯଦି (repeat_cpu_compare_config.enabled) {
        ଯଦି (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ଗଣ୍ଠି, GGML_OP_REPEAT)) {
            cpu_comparison_active = ସତ୍ୟ;
        } ଅନ୍ୟଥା {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for REPEAT operation\n");
        }
    }

    ତାର୍କିକ kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ଆକାର(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ଯଦି (cpu_comparison_active) {
        ଯଦି (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ଗଣ୍ଠି, &repeat_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for REPEAT operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("REPEAT", kernel_name, ଗଣ୍ଠି);
    ଫେରନ୍ତୁ kernel_result;
}

ତାର୍କିକ ggml_et_op_ssm_conv(ggml_backend_et_device_context * dev_ctx, ଅଚଳ ggml_tensor * ଗଣ୍ଠି) {
    ET_PERF_START();

    ଯଦି (!dev_ctx || !ଗଣ୍ଠି || !ଗଣ୍ଠି->src[0] || !ଗଣ୍ଠି->src[1]) {
        GGML_LOG_ERROR("ET: Invalid parameters for SSM_CONV operation\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (ଗଣ୍ଠି->type != GGML_TYPE_F32 || ଗଣ୍ଠି->src[0]->type != GGML_TYPE_F32 || ଗଣ୍ଠି->src[1]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: SSM_CONV operation with unsupported types: dst=%s src0=%s src1=%s\n",
                       ggml_type_name(ଗଣ୍ଠି->type), ggml_type_name(ଗଣ୍ଠି->src[0]->type),
                       ggml_type_name(ଗଣ୍ଠି->src[1]->type));
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ ଅକ୍ଷର * kernel_name = "ssm_conv_f32";

    ggml_et_ssm_conv_params params;
    params.src0 = *ଗଣ୍ଠି->src[0];
    params.src1 = *ଗଣ୍ଠି->src[1];
    params.dst  = *ଗଣ୍ଠି;

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ତାର୍କିକ                    cpu_comparison_active = ମିଥ୍ୟା;
    ଯଦି (ssm_conv_cpu_compare_config.enabled) {
        ଯଦି (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ଗଣ୍ଠି, GGML_OP_SSM_CONV)) {
            cpu_comparison_active = ସତ୍ୟ;
        } ଅନ୍ୟଥା {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for SSM_CONV operation\n");
        }
    }

    ତାର୍କିକ kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ଆକାର(params), 0xFFFFFFFF);

    ଯଦି (cpu_comparison_active) {
        ଯଦି (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ଗଣ୍ଠି, &ssm_conv_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for SSM_CONV operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("SSM_CONV", kernel_name, ଗଣ୍ଠି);
    ଫେରନ୍ତୁ kernel_result;
}

ତାର୍କିକ ggml_et_op_ssm_scan(ggml_backend_et_device_context * dev_ctx, ଅଚଳ ggml_tensor * ଗଣ୍ଠି) {
    ET_PERF_START();

    ଯଦି (!dev_ctx || !ଗଣ୍ଠି) {
        GGML_LOG_ERROR("ET: Invalid parameters for SSM_SCAN operation\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < 7; ++i) {
        ଯଦି (!ଗଣ୍ଠି->src[i]) {
            GGML_LOG_ERROR("ET: SSM_SCAN missing required input %d\n", i);
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    }

    ଯଦି (ଗଣ୍ଠି->type != GGML_TYPE_F32 || ଗଣ୍ଠି->src[0]->type != GGML_TYPE_F32 || ଗଣ୍ଠି->src[1]->type != GGML_TYPE_F32 ||
        ଗଣ୍ଠି->src[2]->type != GGML_TYPE_F32 || ଗଣ୍ଠି->src[3]->type != GGML_TYPE_F32 ||
        ଗଣ୍ଠି->src[4]->type != GGML_TYPE_F32 || ଗଣ୍ଠି->src[5]->type != GGML_TYPE_F32 ||
        ଗଣ୍ଠି->src[6]->type != GGML_TYPE_I32) {
        GGML_LOG_ERROR("ET: SSM_SCAN operation with unsupported types\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ggml_et_ssm_scan_params params;
    params.src0 = *ଗଣ୍ଠି->src[0];
    params.src1 = *ଗଣ୍ଠି->src[1];
    params.src2 = *ଗଣ୍ଠି->src[2];
    params.src3 = *ଗଣ୍ଠି->src[3];
    params.src4 = *ଗଣ୍ଠି->src[4];
    params.src5 = *ଗଣ୍ଠି->src[5];
    params.src6 = *ଗଣ୍ଠି->src[6];
    params.dst  = *ଗଣ୍ଠି;
    params.K    = ggml_get_op_params_i32(ଗଣ୍ଠି, 0);

    ତାର୍କିକ kernel_result = ggml_et_launch_kernel(dev_ctx, "ssm_scan_f32", &params, ଆକାର(params), 0xFFFFFFFF);

    ET_PERF_END("SSM_SCAN", "ssm_scan_f32", ଗଣ୍ଠି);
    ଫେରନ୍ତୁ kernel_result;
}

ତାର୍କିକ ggml_et_op_rwkv_wkv6(ggml_backend_et_device_context * dev_ctx, ଅଚଳ ggml_tensor * ଗଣ୍ଠି) {
    ET_PERF_START();

    ଯଦି (!dev_ctx || !ଗଣ୍ଠି) {
        GGML_LOG_ERROR("ET: Invalid parameters for RWKV_WKV6 operation\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // Validate all 6 source tensors exist
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i <= 5; i++) {
        ଯଦି (!ଗଣ୍ଠି->src[i]) {
            GGML_LOG_ERROR("ET: RWKV_WKV6 operation missing src[%d]\n", i);
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    }

    ଯଦି (ଗଣ୍ଠି->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: RWKV_WKV6 only supports F32, got %s\n", ggml_type_name(ଗଣ୍ଠି->type));
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ ଅକ୍ଷର * kernel_name = "rwkv_wkv6_f32";

    ଅଚଳ int64_t S      = ଗଣ୍ଠି->src[0]->ne[0];  // head_size
    ଅଚଳ int64_t H      = ଗଣ୍ଠି->src[0]->ne[1];  // num heads
    ଅଚଳ int64_t T      = ଗଣ୍ଠି->src[1]->ne[2];  // num tokens
    ଅଚଳ int64_t n_seqs = ଗଣ୍ଠି->src[5]->ne[1];  // num sequences
    ଅଚଳ int64_t C      = S * H;

    ggml_et_rwkv_wkv6_params params;
    params.k        = (ଦଶମିକ *) ଗଣ୍ଠି->src[0]->data;
    params.v        = (ଦଶମିକ *) ଗଣ୍ଠି->src[1]->data;
    params.r        = (ଦଶମିକ *) ଗଣ୍ଠି->src[2]->data;
    params.tf       = (ଦଶମିକ *) ଗଣ୍ଠି->src[3]->data;
    params.td       = (ଦଶମିକ *) ଗଣ୍ଠି->src[4]->data;
    params.state_in = (ଦଶମିକ *) ଗଣ୍ଠି->src[5]->data;
    params.dst      = (ଦଶମିକ *) ଗଣ୍ଠି->data;
    params.C        = (int32_t) C;
    params.H        = (int32_t) H;
    params.S        = (int32_t) S;
    params.T        = (int32_t) T;
    params.n_seqs   = (int32_t) n_seqs;

    // Phase 1: Initialize CPU comparison context
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ତାର୍କିକ                    cpu_comparison_active = ମିଥ୍ୟା;
    ଯଦି (rwkv_wkv6_cpu_compare_config.enabled) {
        ଯଦି (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ଗଣ୍ଠି, GGML_OP_RWKV_WKV6)) {
            cpu_comparison_active = ସତ୍ୟ;
        } ଅନ୍ୟଥା {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for RWKV_WKV6 operation\n");
        }
    }

    ତାର୍କିକ kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ଆକାର(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare
    ଯଦି (cpu_comparison_active) {
        ଯଦି (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ଗଣ୍ଠି, &rwkv_wkv6_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for RWKV_WKV6 operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END_EXT("RWKV_WKV6", kernel_name, ଗଣ୍ଠି, "S=%d H=%d T=%d n_seqs=%d", (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) S, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) H, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) T,
                    (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) n_seqs);
    ଫେରନ୍ତୁ kernel_result;
}

ତାର୍କିକ ggml_et_op_rwkv_wkv7(ggml_backend_et_device_context * dev_ctx, ଅଚଳ ggml_tensor * ଗଣ୍ଠି) {
    ET_PERF_START();

    ଯଦି (!dev_ctx || !ଗଣ୍ଠି) {
        GGML_LOG_ERROR("ET: Invalid parameters for RWKV_WKV7 operation\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // Validate all 7 source tensors exist
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i <= 6; i++) {
        ଯଦି (!ଗଣ୍ଠି->src[i]) {
            GGML_LOG_ERROR("ET: RWKV_WKV7 operation missing src[%d]\n", i);
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    }

    ଯଦି (ଗଣ୍ଠି->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: RWKV_WKV7 only supports F32, got %s\n", ggml_type_name(ଗଣ୍ଠି->type));
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ ଅକ୍ଷର * kernel_name = "rwkv_wkv7_f32";

    ଅଚଳ int64_t S      = ଗଣ୍ଠି->src[2]->ne[0];  // head_size
    ଅଚଳ int64_t H      = ଗଣ୍ଠି->src[2]->ne[1];  // num heads
    ଅଚଳ int64_t T      = ଗଣ୍ଠି->src[1]->ne[2];  // num tokens
    ଅଚଳ int64_t n_seqs = ଗଣ୍ଠି->src[6]->ne[1];  // num sequences
    ଅଚଳ int64_t C      = S * H;

    ggml_et_rwkv_wkv7_params params;
    params.r        = (ଦଶମିକ *) ଗଣ୍ଠି->src[0]->data;
    params.w        = (ଦଶମିକ *) ଗଣ୍ଠି->src[1]->data;
    params.k        = (ଦଶମିକ *) ଗଣ୍ଠି->src[2]->data;
    params.v        = (ଦଶମିକ *) ଗଣ୍ଠି->src[3]->data;
    params.a        = (ଦଶମିକ *) ଗଣ୍ଠି->src[4]->data;
    params.b        = (ଦଶମିକ *) ଗଣ୍ଠି->src[5]->data;
    params.state_in = (ଦଶମିକ *) ଗଣ୍ଠି->src[6]->data;
    params.dst      = (ଦଶମିକ *) ଗଣ୍ଠି->data;
    params.C        = (int32_t) C;
    params.H        = (int32_t) H;
    params.S        = (int32_t) S;
    params.T        = (int32_t) T;
    params.n_seqs   = (int32_t) n_seqs;

    // Phase 1: Initialize CPU comparison context
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ତାର୍କିକ                    cpu_comparison_active = ମିଥ୍ୟା;
    ଯଦି (rwkv_wkv7_cpu_compare_config.enabled) {
        ଯଦି (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ଗଣ୍ଠି, GGML_OP_RWKV_WKV7)) {
            cpu_comparison_active = ସତ୍ୟ;
        } ଅନ୍ୟଥା {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for RWKV_WKV7 operation\n");
        }
    }

    ତାର୍କିକ kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ଆକାର(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare
    ଯଦି (cpu_comparison_active) {
        ଯଦି (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ଗଣ୍ଠି, &rwkv_wkv7_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for RWKV_WKV7 operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END_EXT("RWKV_WKV7", kernel_name, ଗଣ୍ଠି, "S=%d H=%d T=%d n_seqs=%d", (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) S, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) H, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) T,
                    (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) n_seqs);
    ଫେରନ୍ତୁ kernel_result;
}

ସ୍ଥିର ggml_et_cpu_compare_config gated_delta_net_cpu_compare_config = {
    /* .enabled = */ ମିଥ୍ୟା,
    /* .use_cpu_result = */ ମିଥ୍ୟା,
    /* .log_differences = */ ସତ୍ୟ,
    /* .tolerance = */ 1e-4f,
    /* .max_log_elements = */ 4096
};

ତାର୍କିକ ggml_et_op_gated_delta_net(ggml_backend_et_device_context * dev_ctx, ଅଚଳ ggml_tensor * ଗଣ୍ଠି) {
    ET_PERF_START();

    ଯଦି (!dev_ctx || !ଗଣ୍ଠି) {
        GGML_LOG_ERROR("ET: Invalid parameters for GATED_DELTA_NET operation\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // Validate all 6 source tensors exist
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i <= 5; i++) {
        ଯଦି (!ଗଣ୍ଠି->src[i]) {
            GGML_LOG_ERROR("ET: GATED_DELTA_NET operation missing src[%d]\n", i);
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    }

    ଯଦି (ଗଣ୍ଠି->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: GATED_DELTA_NET only supports F32, got %s\n", ggml_type_name(ଗଣ୍ଠି->type));
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ ଅକ୍ଷର * kernel_name = "gated_delta_net_f32";

    ଅଚଳ ggml_tensor * src_q     = ଗଣ୍ଠି->src[0];
    ଅଚଳ ggml_tensor * src_k     = ଗଣ୍ଠି->src[1];
    ଅଚଳ ggml_tensor * src_v     = ଗଣ୍ଠି->src[2];
    ଅଚଳ ggml_tensor * src_g     = ଗଣ୍ଠି->src[3];
    ଅଚଳ ggml_tensor * src_beta  = ଗଣ୍ଠି->src[4];
    ଅଚଳ ggml_tensor * src_state = ଗଣ୍ଠି->src[5];

    ଅଚଳ int64_t S_v      = src_v->ne[0];
    ଅଚଳ int64_t H        = src_v->ne[1];
    ଅଚଳ int64_t n_tokens = src_v->ne[2];
    ଅଚଳ int64_t n_seqs   = src_v->ne[3];
    ଅଚଳ int64_t H_q      = src_q->ne[1];
    ଅଚଳ int64_t H_k      = src_k->ne[1];
    ଅଚଳ int64_t n_seqs_q = src_q->ne[3];
    ଅଚଳ int64_t n_seqs_k = src_k->ne[3];

    ggml_et_gated_delta_net_params params;
    params.q        = *src_q;
    params.k        = *src_k;
    params.v        = *src_v;
    params.g        = *src_g;
    params.beta     = *src_beta;
    params.state_in = *src_state;
    params.dst      = *ଗଣ୍ଠି;
    params.S_v      = (int32_t) S_v;
    params.H        = (int32_t) H;
    params.H_q      = (int32_t) H_q;
    params.H_k      = (int32_t) H_k;
    params.n_tokens = (int32_t) n_tokens;
    params.n_seqs   = (int32_t) n_seqs;
    params.n_seqs_q = (int32_t) n_seqs_q;
    params.n_seqs_k = (int32_t) n_seqs_k;
    params.kda      = (src_g->ne[0] == S_v) ? 1 : 0;
    params.K        = ggml_get_op_params_i32(ଗଣ୍ଠି, 0);
    params.scale    = 1.0f / sqrtf((ଦଶମିକ) S_v);

    // CPU comparison for debugging
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ତାର୍କିକ                    cpu_comparison_active = ମିଥ୍ୟା;
    ଯଦି (gated_delta_net_cpu_compare_config.enabled) {
        ଯଦି (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ଗଣ୍ଠି, GGML_OP_GATED_DELTA_NET)) {
            cpu_comparison_active = ସତ୍ୟ;
        } ଅନ୍ୟଥା {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for GATED_DELTA_NET operation\n");
        }
    }

    ତାର୍କିକ kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ଆକାର(params), 0xFFFFFFFF);

    ଯଦି (cpu_comparison_active) {
        ଯଦି (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ଗଣ୍ଠି, &gated_delta_net_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for GATED_DELTA_NET operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END_EXT("GATED_DELTA_NET", kernel_name, ଗଣ୍ଠି, "S_v=%d H=%d n_tokens=%d n_seqs=%d kda=%d", (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) S_v,
                    (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) H, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) n_tokens, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) n_seqs, params.kda);
    ଫେରନ୍ତୁ kernel_result;
}

ତାର୍କିକ ggml_et_op_set_rows(ggml_backend_et_device_context * dev_ctx, ଅଚଳ ggml_tensor * ଗଣ୍ଠି) {
    ET_PERF_START();

    ଯଦି (!dev_ctx || !ଗଣ୍ଠି) {
        GGML_LOG_ERROR("ET: Invalid parameters for SET_ROWS operation\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (!ଗଣ୍ଠି->src[0] || !ଗଣ୍ଠି->src[1] || !ଗଣ୍ଠି->src[2]) {
        GGML_LOG_ERROR(
            "ET: SET_ROWS operation missing required inputs (needs src[0]=base, src[1]=indices, src[2]=data)\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ ଅକ୍ଷର * kernel_name;

    // Support F32 data with I64 indices -> F32/F16 output (scatter operation)
    ଯଦି (ଗଣ୍ଠି->src[0]->type == GGML_TYPE_F32 && ଗଣ୍ଠି->src[1]->type == GGML_TYPE_I64 &&
        (ଗଣ୍ଠି->type == GGML_TYPE_F32 || ଗଣ୍ଠି->type == GGML_TYPE_F16)) {
        ଯଦି (ଗଣ୍ଠି->type == GGML_TYPE_F32 || ଗଣ୍ଠି->type == GGML_TYPE_F16) {
            kernel_name = "set_rows_f32";
        } ଅନ୍ୟଥା {
            GGML_LOG_ERROR("ET: SET_ROWS unsupported output type: %s\n", ggml_type_name(ଗଣ୍ଠି->type));
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

    } ଅନ୍ୟଥା {
        GGML_LOG_ERROR("ET: SET_ROWS operation with unsupported types: dst=%s src0=%s src1=%s\n",
                       ggml_type_name(ଗଣ୍ଠି->type), ggml_type_name(ଗଣ୍ଠି->src[0]->type),
                       ggml_type_name(ଗଣ୍ଠି->src[1]->type));
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // Validate contiguity requirements
    ଯଦି (!ggml_is_contiguous_rows(ଗଣ୍ଠି)) {
        GGML_LOG_ERROR("ET: SET_ROWS operation requires contiguous-rows destination tensor\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (!ggml_is_contiguous_rows(ଗଣ୍ଠି->src[0])) {
        GGML_LOG_ERROR("ET: SET_ROWS operation requires contiguous-rows source tensor\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (!ggml_is_contiguous(ଗଣ୍ଠି->src[1])) {
        GGML_LOG_ERROR("ET: SET_ROWS operation requires contiguous indices tensor\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // Validate dimension constraints from ggml implementation
    ଯଦି (!(ଗଣ୍ଠି->ne[0] == ଗଣ୍ଠି->src[0]->ne[0] &&              // same number of columns
          ଗଣ୍ଠି->ne[2] == ଗଣ୍ଠି->src[0]->ne[2] &&              // same batch size
          ଗଣ୍ଠି->ne[3] == ଗଣ୍ଠି->src[0]->ne[3] &&              // same outer dimension
          ଗଣ୍ଠି->src[0]->ne[1] == ଗଣ୍ଠି->src[1]->ne[0] &&      // src rows = index count
          ଗଣ୍ଠି->src[0]->ne[2] % ଗଣ୍ଠି->src[1]->ne[1] == 0 &&  // batch constraint
          ଗଣ୍ଠି->src[0]->ne[3] % ଗଣ୍ଠି->src[1]->ne[2] == 0 &&  // outer constraint
          ଗଣ୍ଠି->src[1]->ne[3] == 1)) {                       // indices constraint
        GGML_LOG_ERROR("ET: SET_ROWS operation dimension constraint failed\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ggml_et_set_rows_params params;
    params.src0 = *ଗଣ୍ଠି->src[0];  // F32 source data tensor
    params.src1 = *ଗଣ୍ଠି->src[1];  // I64 indices tensor
    params.dst  = *ଗଣ୍ଠି;          // F32/F16 destination tensor

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ତାର୍କିକ                    cpu_comparison_active = ମିଥ୍ୟା;
    ଯଦି (set_rows_cpu_compare_config.enabled) {
        ଯଦି (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ଗଣ୍ଠି, GGML_OP_SET_ROWS)) {
            cpu_comparison_active = ସତ୍ୟ;
        } ଅନ୍ୟଥା {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for SET_ROWS operation\n");
        }
    }

    ତାର୍କିକ kernel_result = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ଆକାର(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ଯଦି (cpu_comparison_active) {
        ଯଦି (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ଗଣ୍ଠି, &set_rows_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for SET_ROWS operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("SET_ROWS", kernel_name, ଗଣ୍ଠି);
    ଫେରନ୍ତୁ kernel_result;
}

ତାର୍କିକ ggml_et_op_fill(ggml_backend_et_device_context * dev_ctx, ଅଚଳ ggml_tensor * ଗଣ୍ଠି) {
    ET_PERF_START();

    ggml_et_fill_params params;
    params.dst = *ଗଣ୍ଠି;
    memcpy(&params.c, ଗଣ୍ଠି->op_params, ଆକାର(ଦଶମିକ));

    ତାର୍କିକ kernel_result = ggml_et_launch_kernel(dev_ctx, "fill_f32", &params, ଆକାର(params), 0xFFFFFFFF);

    ET_PERF_END("FILL", "fill_f32", ଗଣ୍ଠି);
    ଫେରନ୍ତୁ kernel_result;
}

ତାର୍କିକ ggml_et_op_diag(ggml_backend_et_device_context * dev_ctx, ଅଚଳ ggml_tensor * ଗଣ୍ଠି) {
    ET_PERF_START();

    ଯଦି (!ଗଣ୍ଠି->src[0]) {
        GGML_LOG_ERROR("ET: DIAG operation missing source tensor\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ggml_et_diag_params params;
    params.src0 = *ଗଣ୍ଠି->src[0];
    params.dst  = *ଗଣ୍ଠି;

    ତାର୍କିକ kernel_result = ggml_et_launch_kernel(dev_ctx, "diag_f32", &params, ଆକାର(params), 0xFFFFFFFF);

    ET_PERF_END("DIAG", "diag_f32", ଗଣ୍ଠି);
    ଫେରନ୍ତୁ kernel_result;
}

ତାର୍କିକ ggml_et_op_tri(ggml_backend_et_device_context * dev_ctx, ଅଚଳ ggml_tensor * ଗଣ୍ଠି) {
    ET_PERF_START();

    ଯଦି (!ଗଣ୍ଠି->src[0]) {
        GGML_LOG_ERROR("ET: TRI operation missing source tensor\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ggml_et_tri_params params;
    params.src0 = *ଗଣ୍ଠି->src[0];
    params.dst  = *ଗଣ୍ଠି;
    memcpy(&params.tri_type, ଗଣ୍ଠି->op_params, ଆକାର(int32_t));

    ତାର୍କିକ kernel_result = ggml_et_launch_kernel(dev_ctx, "tri_f32", &params, ଆକାର(params), 0xFFFFFFFF);

    ET_PERF_END("TRI", "tri_f32", ଗଣ୍ଠି);
    ଫେରନ୍ତୁ kernel_result;
}

ତାର୍କିକ ggml_et_op_solve_tri(ggml_backend_et_device_context * dev_ctx, ଅଚଳ ggml_tensor * ଗଣ୍ଠି) {
    ET_PERF_START();

    ଯଦି (!ଗଣ୍ଠି->src[0] || !ଗଣ୍ଠି->src[1]) {
        GGML_LOG_ERROR("ET: SOLVE_TRI operation missing source tensor(s)\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ggml_et_solve_tri_params params;
    params.src0 = *ଗଣ୍ଠି->src[0];  // A (lower-triangular)
    params.src1 = *ଗଣ୍ଠି->src[1];  // B (RHS)
    params.dst  = *ଗଣ୍ଠି;          // X (solution)

    ତାର୍କିକ kernel_result = ggml_et_launch_kernel(dev_ctx, "solve_tri_f32", &params, ଆକାର(params), 0xFFFFFFFF);

    ET_PERF_END("SOLVE_TRI", "solve_tri_f32", ଗଣ୍ଠି);
    ଫେରନ୍ତୁ kernel_result;
}

ତାର୍କିକ ggml_et_op_set(ggml_backend_et_device_context * dev_ctx, ଅଚଳ ggml_tensor * ଗଣ୍ଠି) {
    ET_PERF_START();

    ଯଦି (!ଗଣ୍ଠି->src[0] || !ଗଣ୍ଠି->src[1]) {
        GGML_LOG_ERROR("ET: SET operation missing source tensor(s)\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ ତାର୍କିକ   inplace = (ତାର୍କିକ) ((ଅଚଳ int32_t *) ଗଣ୍ଠି->op_params)[4];
    ଅଚଳ size_t offset  = ((ଅଚଳ int32_t *) ଗଣ୍ଠି->op_params)[3];
    ଅଚଳ size_t nb1     = ((ଅଚଳ int32_t *) ଗଣ୍ଠି->op_params)[0];
    ଅଚଳ size_t nb2     = ((ଅଚଳ int32_t *) ଗଣ୍ଠି->op_params)[1];
    ଅଚଳ size_t nb3     = ((ଅଚଳ int32_t *) ଗଣ୍ଠି->op_params)[2];

    ଯଦି (!inplace) {
        GGML_LOG_ERROR("ET: SET only supports inplace (inplace=%d)\n", inplace);
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (ଗଣ୍ଠି->type != GGML_TYPE_F32 || ଗଣ୍ଠି->src[0]->type != GGML_TYPE_F32 || ଗଣ୍ଠି->src[1]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: SET only supports F32 (dst=%s src0=%s src1=%s)\n", ggml_type_name(ଗଣ୍ଠି->type),
                       ggml_type_name(ଗଣ୍ଠି->src[0]->type), ggml_type_name(ଗଣ୍ଠି->src[1]->type));
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (!ggml_are_same_shape(ଗଣ୍ଠି, ଗଣ୍ଠି->src[0])) {
        GGML_LOG_ERROR("ET: SET requires same-shape src0 and dst\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (!ggml_is_contiguous(ଗଣ୍ଠି) || !ggml_is_contiguous(ଗଣ୍ଠି->src[0]) || !ggml_is_contiguous(ଗଣ୍ଠି->src[1])) {
        GGML_LOG_ERROR("ET: SET requires contiguous dst, src0, and src1\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ggml_et_set_params params;
    params.src1   = *ଗଣ୍ଠି->src[1];
    params.dst    = *ଗଣ୍ଠି;
    params.nb1    = (int32_t) nb1;
    params.nb2    = (int32_t) nb2;
    params.nb3    = (int32_t) nb3;
    params.offset = (int32_t) offset;

    ତାର୍କିକ kernel_result = ggml_et_launch_kernel(dev_ctx, "set_f32", &params, ଆକାର(params), 0xFFFFFFFF);

    ET_PERF_END("SET", "set_f32", ଗଣ୍ଠି);
    ଫେରନ୍ତୁ kernel_result;
}

ତାର୍କିକ ggml_et_op_pad(ggml_backend_et_device_context * dev_ctx, ଅଚଳ ggml_tensor * ଗଣ୍ଠି) {
    ET_PERF_START();

    ଯଦି (!ଗଣ୍ଠି->src[0]) {
        GGML_LOG_ERROR("ET: PAD operation missing source tensor\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (ଗଣ୍ଠି->type != GGML_TYPE_F32 || ଗଣ୍ଠି->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: PAD only supports F32 (src=%s dst=%s)\n", ggml_type_name(ଗଣ୍ଠି->src[0]->type),
                       ggml_type_name(ଗଣ୍ଠି->type));
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (!ggml_is_contiguous(ଗଣ୍ଠି)) {
        GGML_LOG_ERROR("ET: PAD requires contiguous output tensor\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (ଗଣ୍ଠି->src[0]->nb[0] != ଆକାର(ଦଶମିକ)) {
        GGML_LOG_ERROR("ET: PAD requires element-contiguous src dim0 (nb[0]=%zu)\n", (size_t) ଗଣ୍ଠି->src[0]->nb[0]);
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // Extract padding parameters from op_params
    ଅଚଳ int32_t * op_params = (ଅଚଳ int32_t *) ଗଣ୍ଠି->op_params;

    ggml_et_pad_params params;
    params.src0  = *ଗଣ୍ଠି->src[0];
    params.dst   = *ଗଣ୍ଠି;
    params.lp[0] = op_params[0];
    params.rp[0] = op_params[1];
    params.lp[1] = op_params[2];
    params.rp[1] = op_params[3];
    params.lp[2] = op_params[4];
    params.rp[2] = op_params[5];
    params.lp[3] = op_params[6];
    params.rp[3] = op_params[7];

    // v1: no dim0 padding
    ଯଦି (params.lp[0] != 0 || params.rp[0] != 0) {
        GGML_LOG_ERROR("ET: PAD dim0 padding not supported (lp0=%d rp0=%d)\n", params.lp[0], params.rp[0]);
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ତାର୍କିକ                    cpu_comparison_active = ମିଥ୍ୟା;
    ଯଦି (pad_cpu_compare_config.enabled) {
        ଯଦି (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ଗଣ୍ଠି, GGML_OP_PAD)) {
            cpu_comparison_active = ସତ୍ୟ;
        } ଅନ୍ୟଥା {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for PAD operation\n");
        }
    }

    ତାର୍କିକ kernel_result = ggml_et_launch_kernel(dev_ctx, "pad_f32", &params, ଆକାର(params), 0xFFFFFFFF);

    ଯଦି (cpu_comparison_active) {
        ଯଦି (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ଗଣ୍ଠି, &pad_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for PAD operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_END("PAD", "pad_f32", ଗଣ୍ଠି);
    ଫେରନ୍ତୁ kernel_result;
}
