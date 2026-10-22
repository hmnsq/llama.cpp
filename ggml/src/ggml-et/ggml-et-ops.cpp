#ಅನ್ತರ್ಭಾವ "ggml-et-ops.h"

#ಅನ್ತರ್ಭಾವ "ggml-et-cpu-compare.h"
#ಅನ್ತರ್ಭಾವ "ggml-et-kernels.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"

#ಅನ್ತರ್ಭಾವ <stdio.h>

#ಅನ್ತರ್ಭಾವ <cstdint>

// CPU comparison configuration - can be enabled for debugging
ಸ್ಥಿರ ggml_et_cpu_compare_config rope_cpu_compare_config = {
    /* .enabled = */ ಸುಳ್ಳು,
    /* .use_cpu_result = */ ಸುಳ್ಳು,  // Replace ET result with CPU result
    /* .log_differences = */ ನಿಜ,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

ಸ್ಥಿರ ggml_et_cpu_compare_config rms_norm_cpu_compare_config = {
    /* .enabled = */ ಸುಳ್ಳು,
    /* .use_cpu_result = */ ಸುಳ್ಳು,
    /* .log_differences = */ ನಿಜ,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

ಸ್ಥಿರ ggml_et_cpu_compare_config norm_cpu_compare_config = {
    /* .enabled = */ ಸುಳ್ಳು,
    /* .use_cpu_result = */ ಸುಳ್ಳು,
    /* .log_differences = */ ನಿಜ,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

ಸ್ಥಿರ ggml_et_cpu_compare_config l2_norm_cpu_compare_config = {
    /* .enabled = */ ಸುಳ್ಳು,
    /* .use_cpu_result = */ ಸುಳ್ಳು,
    /* .log_differences = */ ನಿಜ,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

ಸ್ಥಿರ ggml_et_cpu_compare_config ಗುಂಪು_norm_cpu_compare_config = {
    /* .enabled = */ ಸುಳ್ಳು,
    /* .use_cpu_result = */ ಸುಳ್ಳು,
    /* .log_differences = */ ನಿಜ,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

ಸ್ಥಿರ ggml_et_cpu_compare_config im2col_cpu_compare_config = {
    /* .enabled = */ ಸುಳ್ಳು,
    /* .use_cpu_result = */ ಸುಳ್ಳು,
    /* .log_differences = */ ನಿಜ,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

ಸ್ಥಿರ ggml_et_cpu_compare_config unary_cpu_compare_config = {
    /* .enabled = */ ಸುಳ್ಳು,
    /* .use_cpu_result = */ ಸುಳ್ಳು,
    /* .log_differences = */ ನಿಜ,
    /* .tolerance = */ 1e-4f,
    /* .max_log_elements = */ 4096
};

ಸ್ಥಿರ ggml_et_cpu_compare_config ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_cpu_compare_config = {
    /* .enabled = */ ಸುಳ್ಳು,
    /* .use_cpu_result = */ ಸುಳ್ಳು,
    /* .log_differences = */ ನಿಜ,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

ಸ್ಥಿರ ggml_et_cpu_compare_config clamp_cpu_compare_config = {
    /* .enabled = */ ಸುಳ್ಳು,
    /* .use_cpu_result = */ ಸುಳ್ಳು,
    /* .log_differences = */ ನಿಜ,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

ಸ್ಥಿರ ggml_et_cpu_compare_config ಸರಾಸರಿ_cpu_compare_config = {
    /* .enabled = */ ಸುಳ್ಳು,
    /* .use_cpu_result = */ ಸುಳ್ಳು,
    /* .log_differences = */ ನಿಜ,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

ಸ್ಥಿರ ggml_et_cpu_compare_config sqr_cpu_compare_config = {
    /* .enabled = */ ಸುಳ್ಳು,
    /* .use_cpu_result = */ ಸುಳ್ಳು,
    /* .log_differences = */ ನಿಜ,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

ಸ್ಥಿರ ggml_et_cpu_compare_config elmap_cpu_compare_config = {
    /* .enabled = */ ಸುಳ್ಳು,
    /* .use_cpu_result = */ ಸುಳ್ಳು,
    /* .log_differences = */ ನಿಜ,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

ಸ್ಥಿರ ggml_et_cpu_compare_config glu_cpu_compare_config = {
    /* .enabled = */ ಸುಳ್ಳು,
    /* .use_cpu_result = */ ಸುಳ್ಳು,
    /* .log_differences = */ ನಿಜ,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 4096
};

ಸ್ಥಿರ ggml_et_cpu_compare_config mul_mat_cpu_compare_config = {
    /* .enabled = */ ಸುಳ್ಳು,
    /* .use_cpu_result = */ ಸುಳ್ಳು,
    /* .log_differences = */ ನಿಜ,
    /* .tolerance = */ 0.01,
    /* .max_log_elements = */ 4096
};

ಸ್ಥಿರ ggml_et_cpu_compare_config mul_mat_id_cpu_compare_config = {
    /* .enabled = */ ಸುಳ್ಳು,
    /* .use_cpu_result = */ ಸುಳ್ಳು,
    /* .log_differences = */ ನಿಜ,
    /* .tolerance = */ 0.01,
    /* .max_log_elements = */ 4096
};

ಸ್ಥಿರ ggml_et_cpu_compare_config softmax_cpu_compare_config = {
    /* .enabled = */ ಸುಳ್ಳು,
    /* .use_cpu_result = */ ಸುಳ್ಳು,
    /* .log_differences = */ ನಿಜ,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 1024
};

ಸ್ಥಿರ ggml_et_cpu_compare_config get_ಅಡ್ಡಸಾಲುಗಳು_cpu_compare_config = {
    /* .enabled = */ ಸುಳ್ಳು,
    /* .use_cpu_result = */ ಸುಳ್ಳು,
    /* .log_differences = */ ನಿಜ,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 2048
};

ಸ್ಥಿರ ggml_et_cpu_compare_config pad_cpu_compare_config = {
    /* .enabled = */ ಸುಳ್ಳು,
    /* .use_cpu_result = */ ಸುಳ್ಳು,
    /* .log_differences = */ ನಿಜ,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

ಸ್ಥಿರ ggml_et_cpu_compare_config cont_cpu_compare_config = {
    /* .enabled = */ ಸುಳ್ಳು,
    /* .use_cpu_result = */ ಸುಳ್ಳು,
    /* .log_differences = */ ನಿಜ,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

ಸ್ಥಿರ ggml_et_cpu_compare_config concat_cpu_compare_config = {
    /* .enabled = */ ಸುಳ್ಳು,
    /* .use_cpu_result = */ ಸುಳ್ಳು,
    /* .log_differences = */ ನಿಜ,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

ಸ್ಥಿರ ggml_et_cpu_compare_config cumsum_cpu_compare_config = {
    /* .enabled = */ ಸುಳ್ಳು,
    /* .use_cpu_result = */ ಸುಳ್ಳು,
    /* .log_differences = */ ನಿಜ,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

ಸ್ಥಿರ ggml_et_cpu_compare_config repeat_cpu_compare_config = {
    /* .enabled = */ ಸುಳ್ಳು,
    /* .use_cpu_result = */ ಸುಳ್ಳು,
    /* .log_differences = */ ನಿಜ,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

ಸ್ಥಿರ ggml_et_cpu_compare_config ssm_conv_cpu_compare_config = {
    /* .enabled = */ ಸುಳ್ಳು,
    /* .use_cpu_result = */ ಸುಳ್ಳು,
    /* .log_differences = */ ನಿಜ,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 4096
};

ಸ್ಥಿರ ggml_et_cpu_compare_config rwkv_wkv6_cpu_compare_config = {
    /* .enabled = */ ಸುಳ್ಳು,
    /* .use_cpu_result = */ ಸುಳ್ಳು,
    /* .log_differences = */ ನಿಜ,
    /* .tolerance = */ 1e-4f,
    /* .max_log_elements = */ 4096
};

ಸ್ಥಿರ ggml_et_cpu_compare_config rwkv_wkv7_cpu_compare_config = {
    /* .enabled = */ ಸುಳ್ಳು,
    /* .use_cpu_result = */ ಸುಳ್ಳು,
    /* .log_differences = */ ನಿಜ,
    /* .tolerance = */ 1e-4f,
    /* .max_log_elements = */ 4096
};

ಸ್ಥಿರ ggml_et_cpu_compare_config ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_cpu_compare_config = {
    /* .enabled = */ ಸುಳ್ಳು,
    /* .use_cpu_result = */ ಸುಳ್ಳು,
    /* .log_differences = */ ನಿಜ,
    /* .tolerance = */ 1e-6f,
    /* .max_log_elements = */ 2048
};

ತಾರ್ಕಿಕ ggml_et_op_rms_norm_mul(ggml_backend_et_device_context * dev_ctx,
                             ಸ್ಥಿರಾಂಕ ggml_tensor *              rms_norm_ಗಂಟು,
                             ಸ್ಥಿರಾಂಕ ggml_tensor *              mul_ಗಂಟು) {
    ET_PERF_ಆರಂಭ();

    ಆದರೆ (!dev_ctx || !rms_norm_ಗಂಟು || !mul_ಗಂಟು) {
        GGML_LOG_ERROR("ET: Invalid parameters for fused RMS_NORM_MUL operation\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (!rms_norm_ಗಂಟು->src[0]) {
        GGML_LOG_ERROR("ET: Fused RMS_NORM_MUL missing required input\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // Extract weights: the MUL operand that isn't the rms_norm output
    ಸ್ಥಿರಾಂಕ ggml_tensor * weights = (mul_ಗಂಟು->src[0] == rms_norm_ಗಂಟು) ? mul_ಗಂಟು->src[1] : mul_ಗಂಟು->src[0];

    ಆದರೆ (!weights) {
        GGML_LOG_ERROR("ET: Fused RMS_NORM_MUL missing weights tensor\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ದಶಮಾಂಶ eps;
    ಸ್ಮರಣೆ_ನಕಲು(&eps, rms_norm_ಗಂಟು->op_params, ಗಾತ್ರ(ದಶಮಾಂಶ));

    ggml_et_rms_norm_mul_params params;
    params.src0 = *rms_norm_ಗಂಟು->src[0];  // input to normalize
    params.src1 = *weights;                // normalization weights
    params.dst  = *mul_ಗಂಟು;               // final output
    params.eps  = eps;

    ತಾರ್ಕಿಕ kernel_ಫಲಿತಾಂಶ = ggml_et_launch_kernel(dev_ctx, "rms_norm_mul_f32", &params, ಗಾತ್ರ(params), 0xFFFFFFFF);

    ET_PERF_ಕೊನೆ_EXT("RMS_NORM_MUL", "rms_norm_mul_f32", mul_ಗಂಟು, "eps=%.6f", (ದ್ವಿಗುಣ) eps);
    ಹಿಂತಿರುಗಿ kernel_ಫಲಿತಾಂಶ;
}

ತಾರ್ಕಿಕ ggml_et_op_scale(ggml_backend_et_device_context * dev_ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು) {
    ET_PERF_ಆರಂಭ();

    ಆದರೆ (!dev_ctx || !ಗಂಟು) {
        GGML_LOG_ERROR("ET: Invalid parameters for SCALE operation\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (!ಗಂಟು->src[0]) {
        GGML_LOG_ERROR("ET: SCALE operation missing required input\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (ಗಂಟು->type != GGML_TYPE_F32 || ಗಂಟು->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: SCALE operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(ಗಂಟು->type),
                       ggml_type_name(ಗಂಟು->src[0]->type));
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ದಶಮಾಂಶ scale, bias;
    ಸ್ಮರಣೆ_ನಕಲು(&scale, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ಗಂಟು->op_params + 0, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&bias, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ಗಂಟು->op_params + 1, ಗಾತ್ರ(ದಶಮಾಂಶ));

    ggml_et_scale_params params;
    params.src0  = *ಗಂಟು->src[0];
    params.dst   = *ಗಂಟು;
    params.scale = scale;
    params.bias  = bias;

    ತಾರ್ಕಿಕ kernel_ಫಲಿತಾಂಶ = ggml_et_launch_kernel(dev_ctx, "scale_f32", &params, ಗಾತ್ರ(params), 0xFFFFFFFF);

    ET_PERF_ಕೊನೆ_EXT("SCALE", "scale_f32", ಗಂಟು, "scale=%.6f|bias=%.6f", (ದ್ವಿಗುಣ) scale, (ದ್ವಿಗುಣ) bias);
    ಹಿಂತಿರುಗಿ kernel_ಫಲಿತಾಂಶ;
}

ತಾರ್ಕಿಕ ggml_et_op_sqr(ggml_backend_et_device_context * dev_ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು) {
    ET_PERF_ಆರಂಭ();

    ಆದರೆ (!dev_ctx || !ಗಂಟು) {
        GGML_LOG_ERROR("ET: Invalid parameters for SQR operation\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (!ಗಂಟು->src[0]) {
        GGML_LOG_ERROR("ET: SQR operation missing required input\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (ಗಂಟು->type != GGML_TYPE_F32 || ಗಂಟು->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: SQR operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(ಗಂಟು->type),
                       ggml_type_name(ಗಂಟು->src[0]->type));
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ggml_et_sqr_params params;
    params.src0 = *ಗಂಟು->src[0];  // F32 input tensor
    params.dst  = *ಗಂಟು;          // F32 output tensor

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ತಾರ್ಕಿಕ                    cpu_comparison_ಸಕ್ರಿಯ = ಸುಳ್ಳು;
    ಆದರೆ (sqr_cpu_compare_config.enabled) {
        ಆದರೆ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ಗಂಟು, GGML_OP_SQR)) {
            cpu_comparison_ಸಕ್ರಿಯ = ನಿಜ;
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for SQR operation\n");
        }
    }

    ತಾರ್ಕಿಕ kernel_ಫಲಿತಾಂಶ = ggml_et_launch_kernel(dev_ctx, "sqr_f32", &params, ಗಾತ್ರ(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ಆದರೆ (cpu_comparison_ಸಕ್ರಿಯ) {
        ಆದರೆ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ಗಂಟು, &sqr_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for SQR operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_ಕೊನೆ("SQR", "sqr_f32", ಗಂಟು);
    ಹಿಂತಿರುಗಿ kernel_ಫಲಿತಾಂಶ;
}

ತಾರ್ಕಿಕ ggml_et_op_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು(ggml_backend_et_device_context * dev_ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು) {
    ET_PERF_ಆರಂಭ();

    ಆದರೆ (!dev_ctx || !ಗಂಟು) {
        GGML_LOG_ERROR("ET: Invalid parameters for SUM_ROWS operation\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (!ಗಂಟು->src[0]) {
        GGML_LOG_ERROR("ET: SUM_ROWS operation missing required input\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (ಗಂಟು->type != GGML_TYPE_F32 || ಗಂಟು->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: SUM_ROWS operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(ಗಂಟು->type),
                       ggml_type_name(ಗಂಟು->src[0]->type));
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ggml_et_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_params params;
    params.src0 = *ಗಂಟು->src[0];
    params.dst  = *ಗಂಟು;

    // Phase 1: Initialize CPU comparison context
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ತಾರ್ಕಿಕ                    cpu_comparison_ಸಕ್ರಿಯ = ಸುಳ್ಳು;
    ಆದರೆ (ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_cpu_compare_config.enabled) {
        ಆದರೆ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ಗಂಟು, GGML_OP_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು)) {
            cpu_comparison_ಸಕ್ರಿಯ = ನಿಜ;
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for SUM_ROWS operation\n");
        }
    }

    ತಾರ್ಕಿಕ kernel_ಫಲಿತಾಂಶ = ggml_et_launch_kernel(dev_ctx, "sum_rows_f32", &params, ಗಾತ್ರ(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare
    ಆದರೆ (cpu_comparison_ಸಕ್ರಿಯ) {
        ಆದರೆ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ಗಂಟು, &ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for SUM_ROWS operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_ಕೊನೆ("SUM_ROWS", "sum_rows_f32", ಗಂಟು);
    ಹಿಂತಿರುಗಿ kernel_ಫಲಿತಾಂಶ;
}

ತಾರ್ಕಿಕ ggml_et_op_ಸರಾಸರಿ(ggml_backend_et_device_context * dev_ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು) {
    ET_PERF_ಆರಂಭ();

    ಆದರೆ (!dev_ctx || !ಗಂಟು) {
        GGML_LOG_ERROR("ET: Invalid parameters for MEAN operation\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (!ಗಂಟು->src[0]) {
        GGML_LOG_ERROR("ET: MEAN operation missing required input\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (ಗಂಟು->type != GGML_TYPE_F32 || ಗಂಟು->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: MEAN operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(ಗಂಟು->type),
                       ggml_type_name(ಗಂಟು->src[0]->type));
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ggml_et_ಸರಾಸರಿ_params params;
    params.src0 = *ಗಂಟು->src[0];
    params.dst  = *ಗಂಟು;

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ತಾರ್ಕಿಕ                    cpu_comparison_ಸಕ್ರಿಯ = ಸುಳ್ಳು;
    ಆದರೆ (ಸರಾಸರಿ_cpu_compare_config.enabled) {
        ಆದರೆ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ಗಂಟು, GGML_OP_ಸರಾಸರಿ)) {
            cpu_comparison_ಸಕ್ರಿಯ = ನಿಜ;
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for MEAN operation\n");
        }
    }

    ತಾರ್ಕಿಕ kernel_ಫಲಿತಾಂಶ = ggml_et_launch_kernel(dev_ctx, "mean_f32", &params, ಗಾತ್ರ(params), 0xFFFFFFFF);

    ಆದರೆ (cpu_comparison_ಸಕ್ರಿಯ) {
        ಆದರೆ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ಗಂಟು, &ಸರಾಸರಿ_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for MEAN operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_ಕೊನೆ("MEAN", "mean_f32", ಗಂಟು);
    ಹಿಂತಿರುಗಿ kernel_ಫಲಿತಾಂಶ;
}

ತಾರ್ಕಿಕ ggml_et_op_clamp(ggml_backend_et_device_context * dev_ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು) {
    ET_PERF_ಆರಂಭ();

    ಆದರೆ (!dev_ctx || !ಗಂಟು) {
        GGML_LOG_ERROR("ET: Invalid parameters for CLAMP operation\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (!ಗಂಟು->src[0]) {
        GGML_LOG_ERROR("ET: CLAMP operation missing required input\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (ಗಂಟು->type != GGML_TYPE_F32 || ಗಂಟು->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: CLAMP operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(ಗಂಟು->type),
                       ggml_type_name(ಗಂಟು->src[0]->type));
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ggml_et_clamp_params params;
    params.src0 = *ಗಂಟು->src[0];
    params.dst  = *ಗಂಟು;
    // op_params layout per ggml.c::ggml_clamp: { min, max } as floats
    ಸ್ಮರಣೆ_ನಕಲು(&params.min_val, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ಗಂಟು->op_params + 0, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&params.max_val, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ಗಂಟು->op_params + 1, ಗಾತ್ರ(ದಶಮಾಂಶ));

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ತಾರ್ಕಿಕ                    cpu_comparison_ಸಕ್ರಿಯ = ಸುಳ್ಳು;
    ಆದರೆ (clamp_cpu_compare_config.enabled) {
        ಆದರೆ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ಗಂಟು, GGML_OP_CLAMP)) {
            cpu_comparison_ಸಕ್ರಿಯ = ನಿಜ;
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for CLAMP operation\n");
        }
    }

    ತಾರ್ಕಿಕ kernel_ಫಲಿತಾಂಶ = ggml_et_launch_kernel(dev_ctx, "clamp_f32", &params, ಗಾತ್ರ(params), 0xFFFFFFFF);

    ಆದರೆ (cpu_comparison_ಸಕ್ರಿಯ) {
        ಆದರೆ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ಗಂಟು, &clamp_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for CLAMP operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_ಕೊನೆ("CLAMP", "clamp_f32", ಗಂಟು);
    ಹಿಂತಿರುಗಿ kernel_ಫಲಿತಾಂಶ;
}

ತಾರ್ಕಿಕ ggml_et_op_unary(ggml_backend_et_device_context * dev_ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು) {
    ET_PERF_ಆರಂಭ();

    ಆದರೆ (!dev_ctx || !ಗಂಟು) {
        GGML_LOG_ERROR("ET: Invalid parameters for UNARY operation\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (!ಗಂಟು->src[0]) {
        GGML_LOG_ERROR("ET: UNARY operation missing required input\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (ಗಂಟು->type != GGML_TYPE_F32 || ಗಂಟು->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: UNARY operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(ಗಂಟು->type),
                       ggml_type_name(ಗಂಟು->src[0]->type));
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ ggml_unary_op uop     = ggml_get_unary_op(ಗಂಟು);
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *        op_name = ggml_unary_op_name(uop);

    ggml_et_unary_params params;
    params.src0     = *ಗಂಟು->src[0];  // F32 input tensor
    params.dst      = *ಗಂಟು;          // F32 output tensor
    params.unary_op = (int32_t) uop;

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ತಾರ್ಕಿಕ                    cpu_comparison_ಸಕ್ರಿಯ = ಸುಳ್ಳು;
    ಆದರೆ (unary_cpu_compare_config.enabled) {
        ಆದರೆ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ಗಂಟು, GGML_OP_UNARY)) {
            cpu_comparison_ಸಕ್ರಿಯ = ನಿಜ;
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for UNARY/%s operation\n", op_name);
        }
    }

    ತಾರ್ಕಿಕ kernel_ಫಲಿತಾಂಶ = ggml_et_launch_kernel(dev_ctx, "unary_f32", &params, ಗಾತ್ರ(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ಆದರೆ (cpu_comparison_ಸಕ್ರಿಯ) {
        ಆದರೆ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ಗಂಟು, &unary_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for UNARY/%s operation\n", op_name);
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_ಕೊನೆ_EXT("UNARY", "unary_f32", ಗಂಟು, "op=%s", op_name);
    ಹಿಂತಿರುಗಿ kernel_ಫಲಿತಾಂಶ;
}

ತಾರ್ಕಿಕ ggml_et_op_mul(ggml_backend_et_device_context * dev_ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು) {
    // Delegate to generic element map operation
    ಹಿಂತಿರುಗಿ ggml_et_op_elmap(dev_ctx, ಗಂಟು);
}

ತಾರ್ಕಿಕ ggml_et_op_ಕೂಡಿಸು(ggml_backend_et_device_context * dev_ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು) {
    // Delegate to generic element map operation
    ಹಿಂತಿರುಗಿ ggml_et_op_elmap(dev_ctx, ಗಂಟು);
}

ತಾರ್ಕಿಕ ggml_et_op_sub(ggml_backend_et_device_context * dev_ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು) {
    // Delegate to generic element map operation
    ಹಿಂತಿರುಗಿ ggml_et_op_elmap(dev_ctx, ಗಂಟು);
}

ತಾರ್ಕಿಕ ggml_et_op_elmap(ggml_backend_et_device_context * dev_ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು) {
    ET_PERF_ಆರಂಭ();

    ಆದರೆ (!dev_ctx || !ಗಂಟು) {
        GGML_LOG_ERROR("ET: Invalid parameters for element map operation\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (!ಗಂಟು->src[0] || !ಗಂಟು->src[1]) {
        GGML_LOG_ERROR("ET: Element map operation missing required inputs\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (ಗಂಟು->type != GGML_TYPE_F32 || ಗಂಟು->src[0]->type != GGML_TYPE_F32 || ಗಂಟು->src[1]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: Element map operation with unsupported types: dst=%s src0=%s src1=%s\n",
                       ggml_type_name(ಗಂಟು->type), ggml_type_name(ಗಂಟು->src[0]->type),
                       ggml_type_name(ಗಂಟು->src[1]->type));
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * op_name = ggml_op_name(ಗಂಟು->op);

    ggml_et_elmap_params params;
    params.src0 = *ಗಂಟು->src[0];
    params.src1 = *ಗಂಟು->src[1];
    params.dst  = *ಗಂಟು;  // F32 output tensor (op type stored in dst.op)

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ತಾರ್ಕಿಕ                    cpu_comparison_ಸಕ್ರಿಯ = ಸುಳ್ಳು;
    ಆದರೆ (elmap_cpu_compare_config.enabled) {
        ಆದರೆ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ಗಂಟು, ಗಂಟು->op)) {
            cpu_comparison_ಸಕ್ರಿಯ = ನಿಜ;
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for %s operation\n", op_name);
        }
    }

    // fprintf(stderr, "ET: el_map s0 [%ld, %ld, %ld, %ld] s1 [%ld, %ld, %ld, %ld]\n",
    //     node->src[0]->ne[0], node->src[0]->ne[1], node->src[0]->ne[2], node->src[0]->ne[3],
    //     node->src[1]->ne[0], node->src[1]->ne[1], node->src[1]->ne[2], node->src[1]->ne[3]);

    ತಾರ್ಕಿಕ kernel_ಫಲಿತಾಂಶ = ggml_et_launch_kernel(dev_ctx, "el_map_f32", &params, ಗಾತ್ರ(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ಆದರೆ (cpu_comparison_ಸಕ್ರಿಯ) {
        ಆದರೆ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ಗಂಟು, &elmap_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for %s operation\n", op_name);
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_ಕೊನೆ(op_name, "el_map_f32", ಗಂಟು);
    ಹಿಂತಿರುಗಿ kernel_ಫಲಿತಾಂಶ;
}

ತಾರ್ಕಿಕ ggml_et_op_glu(ggml_backend_et_device_context * dev_ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು) {
    ET_PERF_ಆರಂಭ();

    // Validate inputs
    ಆದರೆ (!dev_ctx || !ಗಂಟು) {
        GGML_LOG_ERROR("ET: Invalid parameters for GLU operation\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (!ಗಂಟು->src[0]) {
        GGML_LOG_ERROR("ET: GLU operation missing required input\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_ವಿಭಜಿಸು_ಬಹುಲಕ = ಗಂಟು->src[1] != ಶೂನ್ಯಸೂಚಕ;

    // Only support F32 (as validated by supports_op)
    ಆದರೆ (ಗಂಟು->type != GGML_TYPE_F32 || ಗಂಟು->src[0]->type != GGML_TYPE_F32 ||
        (is_ವಿಭಜಿಸು_ಬಹುಲಕ && ಗಂಟು->src[1]->type != GGML_TYPE_F32)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // Extract GLU operation parameters from op_params
    int32_t glu_op_type = ggml_get_op_params_i32(ಗಂಟು, 0);  // GLU variant (REGLU, GEGLU, SWIGLU, etc.)
    int32_t swapped     = ggml_get_op_params_i32(ಗಂಟು, 1);  // Whether gate/value are swapped

    // Supported variants
    ಆಯ್ಕೆ (glu_op_type) {
        ಸ್ಥಿತಿ GGML_GLU_OP_REGLU:
        ಸ್ಥಿತಿ GGML_GLU_OP_GEGLU:
        ಸ್ಥಿತಿ GGML_GLU_OP_SWIGLU:
        ಸ್ಥಿತಿ GGML_GLU_OP_SWIGLU_OAI:
        ಸ್ಥಿತಿ GGML_GLU_OP_SWIGLU_CLAMP:
        ಸ್ಥಿತಿ GGML_GLU_OP_GEGLU_ERF:
        ಸ್ಥಿತಿ GGML_GLU_OP_GEGLU_ಶೀಘ್ರ:
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            GGML_LOG_ERROR("ET: GLU operation with unsupported variant: %s\n",
                           ggml_glu_op_name((ggml_glu_op) glu_op_type));
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // Get GLU operation name for logging
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * glu_op_name = ggml_glu_op_name((ggml_glu_op) glu_op_type);

    // Pack parameters. Single-tensor mode is encoded by zeroing src1.
    ggml_et_glu_params params = {};
    params.src0               = *ಗಂಟು->src[0];
    ಆದರೆ (is_ವಿಭಜಿಸು_ಬಹುಲಕ) {
        params.src1 = *ಗಂಟು->src[1];
    }
    params.dst         = *ಗಂಟು;
    params.glu_op_type = glu_op_type;
    params.swapped     = swapped;
    params.alpha       = 0.0f;
    params.ಎಲ್ಲೆ       = 0.0f;
    ಆದರೆ (glu_op_type == GGML_GLU_OP_SWIGLU_OAI) {
        params.alpha = ggml_get_op_params_f32(ಗಂಟು, 2);
    }
    ಆದರೆ (glu_op_type == GGML_GLU_OP_SWIGLU_OAI || glu_op_type == GGML_GLU_OP_SWIGLU_CLAMP) {
        params.ಎಲ್ಲೆ = ggml_get_op_params_f32(ಗಂಟು, 3);
    }
    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ತಾರ್ಕಿಕ                    cpu_comparison_ಸಕ್ರಿಯ = ಸುಳ್ಳು;
    ಆದರೆ (glu_cpu_compare_config.enabled) {
        ಆದರೆ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ಗಂಟು, GGML_OP_GLU)) {
            cpu_comparison_ಸಕ್ರಿಯ = ನಿಜ;
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for %s operation\n", glu_op_name);
        }
    }

    // Launch ET kernel
    ತಾರ್ಕಿಕ kernel_ಫಲಿತಾಂಶ = ggml_et_launch_kernel(dev_ctx, "glu_f32", &params, ಗಾತ್ರ(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ಆದರೆ (cpu_comparison_ಸಕ್ರಿಯ) {
        ಆದರೆ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ಗಂಟು, &glu_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for %s operation\n", glu_op_name);
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_ಕೊನೆ("GLU", "glu_f32", ಗಂಟು);
    ಹಿಂತಿರುಗಿ kernel_ಫಲಿತಾಂಶ;
}

ತಾರ್ಕಿಕ ggml_et_op_mul_mat(ggml_backend_et_device_context * dev_ctx,
                        ಸ್ಥಿರಾಂಕ ggml_tensor *              ಗಂಟು,
                        ಸ್ಥಿರಾಂಕ ggml_tensor *              ಕೂಡಿಸು_ಗಂಟು) {
    ET_PERF_ಆರಂಭ();

    ಆದರೆ (!dev_ctx || !ಗಂಟು) {
        GGML_LOG_ERROR("ET: Invalid parameters for MUL_MAT operation\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (!ಗಂಟು->src[0] || !ಗಂಟು->src[1]) {
        GGML_LOG_ERROR("ET: MUL_MAT operation missing required inputs\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // Fused MM+ADD: when add_node is non-NULL the caller has already validated
    // (Q8_0 weights, F32 acts, exact-shape ADD with stride parity to dst) via
    // ggml_et_can_fuse({MUL_MAT, ADD}). The kernel writes dst = mm + bias and
    // the ADD's output replaces MM's as the actual dst.
    ಸ್ಥಿರಾಂಕ ggml_tensor * fused_dst   = ಕೂಡಿಸು_ಗಂಟು ? ಕೂಡಿಸು_ಗಂಟು : ಗಂಟು;
    ಸ್ಥಿರಾಂಕ ggml_tensor * bias_tensor = ಶೂನ್ಯಸೂಚಕ;
    ಆದರೆ (ಕೂಡಿಸು_ಗಂಟು) {
        bias_tensor = (ಕೂಡಿಸು_ಗಂಟು->src[0] == ಗಂಟು) ? ಕೂಡಿಸು_ಗಂಟು->src[1] : ಕೂಡಿಸು_ಗಂಟು->src[0];
    }

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * kernel_name;
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * src0_type_name;

    ಆದರೆ (ಗಂಟು->type == GGML_TYPE_F32 && ಗಂಟು->src[0]->type == GGML_TYPE_Q4_0 && ಗಂಟು->src[1]->type == GGML_TYPE_F32 &&
        ಗಂಟು->src[1]->ne[1] >= 53 &&      // N >= 53
        ಗಂಟು->src[0]->ne[1] % 16 == 0 &&  // M % TILE_M
        ಗಂಟು->src[0]->ne[0] % 32 == 0) {  // K % BLOCK_K (Q4_0 block)

        // Matrix engine for N >= 53; partial N (via n_cur-1) and errata padding are handled in-kernel.
        kernel_name    = "mul_mat_Q4_0_matrix_engine";
        src0_type_name = "Q4_0";

    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಗಂಟು->type == GGML_TYPE_F32 && ಗಂಟು->src[0]->type == GGML_TYPE_Q4_0 &&
               ಗಂಟು->src[1]->type == GGML_TYPE_F32) {
        kernel_name    = "mul_mat_Q4_0";  // N < 53, or M % 16 != 0 or K % 32 != 0
        src0_type_name = "Q4_0";

    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಗಂಟು->type == GGML_TYPE_F32 && ಗಂಟು->src[0]->type == GGML_TYPE_Q8_0 &&
               ಗಂಟು->src[1]->type == GGML_TYPE_F32) {
        kernel_name    = "mul_mat_Q8_0";
        src0_type_name = "Q8_0";

    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಗಂಟು->type == GGML_TYPE_F32 && ಗಂಟು->src[0]->type == GGML_TYPE_F16 &&
               ಗಂಟು->src[1]->type == GGML_TYPE_F16 && ಗಂಟು->ne[0] % 16 == 0 && ಗಂಟು->src[0]->ne[0] % 16 == 0 &&
               ಗಂಟು->src[0]->ne[1] % 16 == 0 && ಗಂಟು->src[1]->ne[0] != 1) {
        kernel_name    = "mul_mat_f16_matrix_engine";
        src0_type_name = "F16";

    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಗಂಟು->type == GGML_TYPE_F32 && ಗಂಟು->src[0]->type == GGML_TYPE_F16 &&
               (ಗಂಟು->src[1]->type == GGML_TYPE_F16 || ಗಂಟು->src[1]->type == GGML_TYPE_F32)) {
        kernel_name    = "mul_mat_f16";
        src0_type_name = "F16";

    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಗಂಟು->type == GGML_TYPE_F32 && ಗಂಟು->src[0]->type == GGML_TYPE_F32 &&
               ಗಂಟು->src[1]->type == GGML_TYPE_F32 && ಗಂಟು->ne[0] % 16 == 0 && ಗಂಟು->src[0]->ne[0] % 16 == 0 &&
               ಗಂಟು->src[0]->ne[1] % 16 == 0 && ಗಂಟು->src[1]->ne[0] != 1) {  // GEMV is faster with the generic path

        kernel_name    = "mul_mat_f32_matrix_engine";
        src0_type_name = "F32";
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಗಂಟು->type == GGML_TYPE_F32 && ಗಂಟು->src[0]->type == GGML_TYPE_F32 &&
               (ಗಂಟು->src[1]->type == GGML_TYPE_F16 || ಗಂಟು->src[1]->type == GGML_TYPE_F32)) {
        kernel_name    = "mul_mat_f32";
        src0_type_name = "F32";
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_LOG_ERROR("ET: MUL_MAT operation with unsupported types: dst=%s src0=%s src1=%s\n",
                       ggml_type_name(ಗಂಟು->type), ggml_type_name(ಗಂಟು->src[0]->type),
                       ggml_type_name(ಗಂಟು->src[1]->type));
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ggml_et_ದ್ವಿಮಾನ_params params;
    params.src0 = *ಗಂಟು->src[0];  // weight matrix
    params.src1 = *ಗಂಟು->src[1];  // activation matrix
    params.dst  = *fused_dst;     // output (= add_node when fused, else node)

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ತಾರ್ಕಿಕ                    cpu_comparison_ಸಕ್ರಿಯ = ಸುಳ್ಳು;
    ಆದರೆ (mul_mat_cpu_compare_config.enabled) {
        ಆದರೆ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, fused_dst, GGML_OP_MUL_MAT)) {
            cpu_comparison_ಸಕ್ರಿಯ = ನಿಜ;
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for MUL_MAT operation\n");
        }
    }

    ತಾರ್ಕಿಕ kernel_ಫಲಿತಾಂಶ;
    ಆದರೆ (ಗಂಟು->src[0]->type == GGML_TYPE_Q8_0) {
        // Q8_0 kernel always takes the extended struct. bias.data is non-NULL
        // only on the fused path; otherwise the kernel skips the add entirely.
        ggml_et_mm_q8_params q8_params = {};
        q8_params.src0                 = params.src0;
        q8_params.src1                 = params.src1;
        q8_params.dst                  = params.dst;
        ಆದರೆ (bias_tensor) {
            q8_params.bias = *bias_tensor;
        }
        kernel_ಫಲಿತಾಂಶ = ggml_et_launch_kernel(dev_ctx, kernel_name, &q8_params, ಗಾತ್ರ(q8_params), 0xFFFFFFFF);
    } ಇಲ್ಲದಿದ್ದರೆ {
        // Non-Q8 MM kernels don't yet support fused-add; the graph fuse check
        // already rejects non-Q8 pairs, so add_node is always nullptr here.
        kernel_ಫಲಿತಾಂಶ = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ಗಾತ್ರ(params), 0xFFFFFFFF);
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
    ಆದರೆ (cpu_comparison_ಸಕ್ರಿಯ) {
        ಆದರೆ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, fused_dst, &mul_mat_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for MUL_MAT operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    {
        // Calculate actual FLOPs including batch/sequence dimensions
        // dst shape: [M, N, ne2, ne3] where M=ne[1], N=ne[0]
        int64_t m   = ಗಂಟು->ne[1];
        int64_t n   = ಗಂಟು->ne[0];
        int64_t k   = ಗಂಟು->src[0]->ne[0];
        int64_t ne2 = ಗಂಟು->ne[2];
        int64_t ne3 = ಗಂಟು->ne[3];

        // Total FLOPs = (batch_size) * M * N * (2*K - 1)
        // Each MxN matrix-matrix multiply does M*N*(2*K-1) FLOPs
        // Broadcasting is handled by repeating computation, so count actual operations
        int64_t batch_ಗಾತ್ರ  = ne2 * ne3;
        int64_t ಒಟ್ಟು_flops = batch_ಗಾತ್ರ * m * n * (2 * k - 1);

        ಅಕ್ಷರ kernel_variant[64];
        snprintf(kernel_variant, ಗಾತ್ರ(kernel_variant), "%s_%sx%s", kernel_name, src0_type_name,
                 ggml_type_name(ಗಂಟು->src[1]->type));
        ET_PERF_ಕೊನೆ_EXT("MUL_MAT", kernel_variant, ಗಂಟು, "flops=%" PRId64, ಒಟ್ಟು_flops);
    }
    ಹಿಂತಿರುಗಿ kernel_ಫಲಿತಾಂಶ;
}

ತಾರ್ಕಿಕ ggml_et_op_mul_mat_id(ggml_backend_et_device_context * dev_ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು) {
    ET_PERF_ಆರಂಭ();
    ಆದರೆ (!dev_ctx || !ಗಂಟು) {
        GGML_LOG_ERROR("ET: Invalid parameters for MUL_MAT_ID operation\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (!ಗಂಟು->src[0] || !ಗಂಟು->src[1] || !ಗಂಟು->src[2]) {
        GGML_LOG_ERROR("ET: MUL_MAT_ID operation missing required inputs\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * kernel_name;
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * src0_type_name;

    // Support Q8_0/Q4_0/F16/F32 x F32 -> F32 matrix multiplication with expert selection
    ಆದರೆ (ಗಂಟು->type == GGML_TYPE_F32 && ಗಂಟು->src[0]->type == GGML_TYPE_Q8_0 && ಗಂಟು->src[1]->type == GGML_TYPE_F32 &&
        ಗಂಟು->src[2]->type == GGML_TYPE_I32) {
        kernel_name    = "mul_mat_id_Q8_0";
        src0_type_name = "Q8_0";

    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಗಂಟು->type == GGML_TYPE_F32 && ಗಂಟು->src[0]->type == GGML_TYPE_Q4_0 &&
               ಗಂಟು->src[1]->type == GGML_TYPE_F32 && ಗಂಟು->src[2]->type == GGML_TYPE_I32) {
        kernel_name    = "mul_mat_id_Q4_0";
        src0_type_name = "Q4_0";

    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಗಂಟು->type == GGML_TYPE_F32 && ಗಂಟು->src[0]->type == GGML_TYPE_F16 &&
               ಗಂಟು->src[1]->type == GGML_TYPE_F32 && ಗಂಟು->src[2]->type == GGML_TYPE_I32) {
        kernel_name    = "mul_mat_id_f32";
        src0_type_name = "F16";

    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಗಂಟು->type == GGML_TYPE_F32 && ಗಂಟು->src[0]->type == GGML_TYPE_F32 &&
               ಗಂಟು->src[1]->type == GGML_TYPE_F32 && ಗಂಟು->src[2]->type == GGML_TYPE_I32) {
        kernel_name    = "mul_mat_id_f32";
        src0_type_name = "F32";

    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_LOG_ERROR("ET: MUL_MAT_ID operation with unsupported types: dst=%s src0=%s src1=%s src2=%s\n",
                       ggml_type_name(ಗಂಟು->type), ggml_type_name(ಗಂಟು->src[0]->type),
                       ggml_type_name(ಗಂಟು->src[1]->type), ggml_type_name(ಗಂಟು->src[2]->type));
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // Pack parameters - copy full tensor structures
    ggml_et_mul_mat_id_params params;
    params.src0 = *ಗಂಟು->src[0];  // Expert weight matrices (Q8_0/F16/F32)
    params.src1 = *ಗಂಟು->src[1];  // Activation matrix (F32)
    params.src2 = *ಗಂಟು->src[2];  // Expert indices (I32)
    params.dst  = *ಗಂಟು;          // Output matrix (F32)

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ತಾರ್ಕಿಕ                    cpu_comparison_ಸಕ್ರಿಯ = ಸುಳ್ಳು;
    ಆದರೆ (mul_mat_id_cpu_compare_config.enabled) {
        ಆದರೆ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ಗಂಟು, GGML_OP_MUL_MAT_ID)) {
            cpu_comparison_ಸಕ್ರಿಯ = ನಿಜ;
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for MUL_MAT_ID operation\n");
        }
    }

    // Launch ET kernel
    ತಾರ್ಕಿಕ kernel_ಫಲಿತಾಂಶ = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ಗಾತ್ರ(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ಆದರೆ (cpu_comparison_ಸಕ್ರಿಯ) {
        ಆದರೆ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ಗಂಟು, &mul_mat_id_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for MUL_MAT_ID operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    // Calculate FLOPs (approximate - similar to MUL_MAT but with expert routing overhead)
    // Each expert computation is similar to a MUL_MAT, but we only compute for selected experts
    int64_t K             = ಗಂಟು->src[0]->ne[0];
    int64_t M             = ಗಂಟು->src[0]->ne[1];
    int64_t n_expert_ಬಳಸಿದ = ಗಂಟು->src[2]->ne[0];
    int64_t batch         = ಗಂಟು->src[2]->ne[1];

    int64_t ಒಟ್ಟು_flops = batch * n_expert_ಬಳಸಿದ * M * (2 * K - 1);

    ಅಕ್ಷರ kernel_variant[64];
    snprintf(kernel_variant, ಗಾತ್ರ(kernel_variant), "%s_%sx%s", kernel_name, src0_type_name,
             ggml_type_name(ಗಂಟು->src[1]->type));
    ET_PERF_ಕೊನೆ_EXT("MUL_MAT_ID", kernel_variant, ಗಂಟು, "flops=%" PRId64 "|n_expert=%lld|n_expert_used=%lld",
                    ಒಟ್ಟು_flops, (ದೀರ್ಘ ದೀರ್ಘ) ಗಂಟು->src[0]->ne[2], (ದೀರ್ಘ ದೀರ್ಘ) n_expert_ಬಳಸಿದ);

    ಹಿಂತಿರುಗಿ kernel_ಫಲಿತಾಂಶ;
}

ತಾರ್ಕಿಕ ggml_et_op_rope(ggml_backend_et_device_context * dev_ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು) {
    ET_PERF_ಆರಂಭ();

    ಆದರೆ (!dev_ctx || !ಗಂಟು) {
        GGML_LOG_ERROR("ET: Invalid parameters for ROPE operation\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (!ಗಂಟು->src[0] || !ಗಂಟು->src[1]) {
        GGML_LOG_ERROR("ET: ROPE operation missing required inputs\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * kernel_name;

    ಆದರೆ (ಗಂಟು->type == GGML_TYPE_F32 && ಗಂಟು->src[0]->type == GGML_TYPE_F32 && ಗಂಟು->src[1]->type == GGML_TYPE_I32) {
        kernel_name = "rope_f32";
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // Pack parameters - copy full tensor structures and op_params
    ggml_et_rope_params params;
    params.src0 = *ಗಂಟು->src[0];                       // F32 input tensor
    params.src1 = *ಗಂಟು->src[1];                       // I32 position tensor
    ಆದರೆ (ಗಂಟು->src[2]) {
        params.src2 = *ಗಂಟು->src[2];                   // F32 frequency factors (optional)
    } ಇಲ್ಲದಿದ್ದರೆ {
        memset(&params.src2, 0, ಗಾತ್ರ(params.src2));  // Zero if not provided
    }
    params.dst = *ಗಂಟು;                                // F32 output tensor

    params.rope_params.n_past     = ((ಸ್ಥಿರಾಂಕ int32_t *) ಗಂಟು->op_params)[0];
    params.rope_params.n_dims     = ((ಸ್ಥಿರಾಂಕ int32_t *) ಗಂಟು->op_params)[1];
    params.rope_params.mode       = ((ಸ್ಥಿರಾಂಕ int32_t *) ಗಂಟು->op_params)[2];
    params.rope_params.n_ctx      = ((ಸ್ಥಿರಾಂಕ int32_t *) ಗಂಟು->op_params)[3];
    params.rope_params.n_ctx_orig = ((ಸ್ಥಿರಾಂಕ int32_t *) ಗಂಟು->op_params)[4];
    ಸ್ಮರಣೆ_ನಕಲು(&params.rope_params.freq_ತಳ, (ಸ್ಥಿರಾಂಕ int32_t *) ಗಂಟು->op_params + 5, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&params.rope_params.freq_scale, (ಸ್ಥಿರಾಂಕ int32_t *) ಗಂಟು->op_params + 6, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&params.rope_params.ext_ಗುಣಕ, (ಸ್ಥಿರಾಂಕ int32_t *) ಗಂಟು->op_params + 7, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&params.rope_params.attn_ಗುಣಕ, (ಸ್ಥಿರಾಂಕ int32_t *) ಗಂಟು->op_params + 8, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&params.rope_params.beta_fast, (ಸ್ಥಿರಾಂಕ int32_t *) ಗಂಟು->op_params + 9, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&params.rope_params.beta_slow, (ಸ್ಥಿರಾಂಕ int32_t *) ಗಂಟು->op_params + 10, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಆದರೆ (params.rope_params.mode & GGML_ROPE_TYPE_MROPE) {
        ಸ್ಮರಣೆ_ನಕಲು(params.rope_params.sections, (ಸ್ಥಿರಾಂಕ int32_t *) ಗಂಟು->op_params + 11, ಗಾತ್ರ(int32_t) * 4);
    } ಇಲ್ಲದಿದ್ದರೆ {
        memset(params.rope_params.sections, 0, ಗಾತ್ರ(params.rope_params.sections));
    }

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ತಾರ್ಕಿಕ                    cpu_comparison_ಸಕ್ರಿಯ = ಸುಳ್ಳು;
    ಆದರೆ (rope_cpu_compare_config.enabled) {
        GGML_LOG_DEBUG("ET: Initializing CPU comparison for ROPE operation\n");
        ಆದರೆ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ಗಂಟು, GGML_OP_ROPE)) {
            cpu_comparison_ಸಕ್ರಿಯ = ನಿಜ;
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for ROPE operation\n");
        }
    }

    ತಾರ್ಕಿಕ kernel_ಫಲಿತಾಂಶ = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ಗಾತ್ರ(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ಆದರೆ (cpu_comparison_ಸಕ್ರಿಯ) {
        ಆದರೆ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ಗಂಟು, &rope_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for ROPE operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_ಕೊನೆ_EXT("ROPE", kernel_name, ಗಂಟು, "mode=0x%x|n_dims=%d|freq_base=%.2f|freq_scale=%.2f",
                    params.rope_params.mode, params.rope_params.n_dims, (ದ್ವಿಗುಣ) params.rope_params.freq_ತಳ,
                    (ದ್ವಿಗುಣ) params.rope_params.freq_scale);
    ಹಿಂತಿರುಗಿ kernel_ಫಲಿತಾಂಶ;
}

ತಾರ್ಕಿಕ ggml_et_op_rms_norm(ggml_backend_et_device_context * dev_ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು) {
    ET_PERF_ಆರಂಭ();

    ಆದರೆ (!dev_ctx || !ಗಂಟು) {
        GGML_LOG_ERROR("ET: Invalid parameters for RMS_NORM operation\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (!ಗಂಟು->src[0]) {
        GGML_LOG_ERROR("ET: RMS_NORM operation missing required input\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * kernel_name;

    ಆದರೆ (ಗಂಟು->type == GGML_TYPE_F32 && ಗಂಟು->src[0]->type == GGML_TYPE_F32) {
        kernel_name = "rms_norm_f32";

    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_LOG_ERROR("ET: RMS_NORM operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(ಗಂಟು->type),
                       ggml_type_name(ಗಂಟು->src[0]->type));
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ದಶಮಾಂಶ eps;
    ಸ್ಮರಣೆ_ನಕಲು(&eps, ಗಂಟು->op_params, ಗಾತ್ರ(ದಶಮಾಂಶ));

    ggml_et_rms_norm_params params;
    params.src0 = *ಗಂಟು->src[0];  // F32 input tensor
    params.dst  = *ಗಂಟು;          // F32 output tensor
    params.eps  = eps;            // Epsilon parameter for numerical stability

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ತಾರ್ಕಿಕ                    cpu_comparison_ಸಕ್ರಿಯ = ಸುಳ್ಳು;
    ಆದರೆ (rms_norm_cpu_compare_config.enabled) {
        ಆದರೆ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ಗಂಟು, GGML_OP_RMS_NORM)) {
            cpu_comparison_ಸಕ್ರಿಯ = ನಿಜ;
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for RMS_NORM operation\n");
        }
    }

    ತಾರ್ಕಿಕ kernel_ಫಲಿತಾಂಶ = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ಗಾತ್ರ(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ಆದರೆ (cpu_comparison_ಸಕ್ರಿಯ) {
        ಆದರೆ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ಗಂಟು, &rms_norm_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for RMS_NORM operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_ಕೊನೆ_EXT("RMS_NORM", kernel_name, ಗಂಟು, "eps=%.6f", (ದ್ವಿಗುಣ) eps);
    ಹಿಂತಿರುಗಿ kernel_ಫಲಿತಾಂಶ;
}

ತಾರ್ಕಿಕ ggml_et_op_norm(ggml_backend_et_device_context * dev_ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು) {
    ET_PERF_ಆರಂಭ();

    ಆದರೆ (!dev_ctx || !ಗಂಟು) {
        GGML_LOG_ERROR("ET: Invalid parameters for NORM operation\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (!ಗಂಟು->src[0]) {
        GGML_LOG_ERROR("ET: NORM operation missing required input\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * kernel_name;

    ಆದರೆ (ಗಂಟು->type == GGML_TYPE_F32 && ಗಂಟು->src[0]->type == GGML_TYPE_F32) {
        kernel_name = "norm_f32";

    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_LOG_ERROR("ET: NORM operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(ಗಂಟು->type),
                       ggml_type_name(ಗಂಟು->src[0]->type));
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ದಶಮಾಂಶ eps;
    ಸ್ಮರಣೆ_ನಕಲು(&eps, ಗಂಟು->op_params, ಗಾತ್ರ(ದಶಮಾಂಶ));

    ggml_et_norm_params params;
    params.src0 = *ಗಂಟು->src[0];  // F32 input tensor
    params.dst  = *ಗಂಟು;          // F32 output tensor
    params.eps  = eps;            // Epsilon parameter for numerical stability

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ತಾರ್ಕಿಕ                    cpu_comparison_ಸಕ್ರಿಯ = ಸುಳ್ಳು;
    ಆದರೆ (norm_cpu_compare_config.enabled) {
        ಆದರೆ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ಗಂಟು, GGML_OP_NORM)) {
            cpu_comparison_ಸಕ್ರಿಯ = ನಿಜ;
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for NORM operation\n");
        }
    }

    ತಾರ್ಕಿಕ kernel_ಫಲಿತಾಂಶ = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ಗಾತ್ರ(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ಆದರೆ (cpu_comparison_ಸಕ್ರಿಯ) {
        ಆದರೆ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ಗಂಟು, &norm_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for NORM operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_ಕೊನೆ_EXT("NORM", kernel_name, ಗಂಟು, "eps=%.6f", (ದ್ವಿಗುಣ) eps);
    ಹಿಂತಿರುಗಿ kernel_ಫಲಿತಾಂಶ;
}

ತಾರ್ಕಿಕ ggml_et_op_l2_norm(ggml_backend_et_device_context * dev_ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು) {
    ET_PERF_ಆರಂಭ();

    ಆದರೆ (!dev_ctx || !ಗಂಟು) {
        GGML_LOG_ERROR("ET: Invalid parameters for L2_NORM operation\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (!ಗಂಟು->src[0]) {
        GGML_LOG_ERROR("ET: L2_NORM operation missing required input\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * kernel_name;

    ಆದರೆ (ಗಂಟು->type == GGML_TYPE_F32 && ಗಂಟು->src[0]->type == GGML_TYPE_F32) {
        kernel_name = "l2_norm_f32";

    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_LOG_ERROR("ET: L2_NORM operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(ಗಂಟು->type),
                       ggml_type_name(ಗಂಟು->src[0]->type));
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ದಶಮಾಂಶ eps;
    ಸ್ಮರಣೆ_ನಕಲು(&eps, ಗಂಟು->op_params, ಗಾತ್ರ(ದಶಮಾಂಶ));

    ggml_et_l2_norm_params params;
    params.src0 = *ಗಂಟು->src[0];  // F32 input tensor
    params.dst  = *ಗಂಟು;          // F32 output tensor
    params.eps  = eps;            // Epsilon parameter for numerical stability

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ತಾರ್ಕಿಕ                    cpu_comparison_ಸಕ್ರಿಯ = ಸುಳ್ಳು;
    ಆದರೆ (l2_norm_cpu_compare_config.enabled) {
        ಆದರೆ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ಗಂಟು, GGML_OP_L2_NORM)) {
            cpu_comparison_ಸಕ್ರಿಯ = ನಿಜ;
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for L2_NORM operation\n");
        }
    }

    ತಾರ್ಕಿಕ kernel_ಫಲಿತಾಂಶ = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ಗಾತ್ರ(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ಆದರೆ (cpu_comparison_ಸಕ್ರಿಯ) {
        ಆದರೆ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ಗಂಟು, &l2_norm_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for L2_NORM operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_ಕೊನೆ_EXT("L2_NORM", kernel_name, ಗಂಟು, "eps=%.6f", (ದ್ವಿಗುಣ) eps);
    ಹಿಂತಿರುಗಿ kernel_ಫಲಿತಾಂಶ;
}

ತಾರ್ಕಿಕ ggml_et_op_ಗುಂಪು_norm(ggml_backend_et_device_context * dev_ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು) {
    ET_PERF_ಆರಂಭ();

    ಆದರೆ (!dev_ctx || !ಗಂಟು) {
        GGML_LOG_ERROR("ET: Invalid parameters for GROUP_NORM operation\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (!ಗಂಟು->src[0]) {
        GGML_LOG_ERROR("ET: GROUP_NORM operation missing required input\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (ಗಂಟು->type != GGML_TYPE_F32 || ಗಂಟು->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: GROUP_NORM operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(ಗಂಟು->type),
                       ggml_type_name(ಗಂಟು->src[0]->type));
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ int32_t n_groups = ggml_get_op_params_i32(ಗಂಟು, 0);
    ದಶಮಾಂಶ         eps;
    ಸ್ಮರಣೆ_ನಕಲು(&eps, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ಗಂಟು->op_params + 1, ಗಾತ್ರ(ದಶಮಾಂಶ));

    ggml_et_ಗುಂಪು_norm_params params;
    params.src0     = *ಗಂಟು->src[0];
    params.dst      = *ಗಂಟು;
    params.n_groups = n_groups;
    params.eps      = eps;

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ತಾರ್ಕಿಕ                    cpu_comparison_ಸಕ್ರಿಯ = ಸುಳ್ಳು;
    ಆದರೆ (ಗುಂಪು_norm_cpu_compare_config.enabled) {
        ಆದರೆ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ಗಂಟು, GGML_OP_ಗುಂಪು_NORM)) {
            cpu_comparison_ಸಕ್ರಿಯ = ನಿಜ;
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for GROUP_NORM operation\n");
        }
    }

    ತಾರ್ಕಿಕ kernel_ಫಲಿತಾಂಶ = ggml_et_launch_kernel(dev_ctx, "group_norm_f32", &params, ಗಾತ್ರ(params), 0xFFFFFFFF);

    ಆದರೆ (cpu_comparison_ಸಕ್ರಿಯ) {
        ಆದರೆ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ಗಂಟು, &ಗುಂಪು_norm_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for GROUP_NORM operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_ಕೊನೆ_EXT("GROUP_NORM", "group_norm_f32", ಗಂಟು, "eps=%.6f|n_groups=%d", (ದ್ವಿಗುಣ) eps, n_groups);
    ಹಿಂತಿರುಗಿ kernel_ಫಲಿತಾಂಶ;
}

ತಾರ್ಕಿಕ ggml_et_op_im2col(ggml_backend_et_device_context * dev_ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು) {
    ET_PERF_ಆರಂಭ();

    ಆದರೆ (!dev_ctx || !ಗಂಟು) {
        GGML_LOG_ERROR("ET: Invalid parameters for IM2COL operation\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (!ಗಂಟು->src[0] || !ಗಂಟು->src[1]) {
        GGML_LOG_ERROR("ET: IM2COL operation missing required inputs\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ supported_types =
        (ಗಂಟು->type == GGML_TYPE_F32 && ಗಂಟು->src[1]->type == GGML_TYPE_F32) ||
        (ಗಂಟು->type == GGML_TYPE_F16 && (ಗಂಟು->src[1]->type == GGML_TYPE_F16 || ಗಂಟು->src[1]->type == GGML_TYPE_F32));

    ಆದರೆ (!supported_types) {
        GGML_LOG_ERROR("ET: IM2COL operation with unsupported types: dst=%s src1=%s\n", ggml_type_name(ಗಂಟು->type),
                       ggml_type_name(ಗಂಟು->src[1]->type));
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ggml_et_im2col_params params;
    params.src0 = *ಗಂಟು->src[0];
    params.src1 = *ಗಂಟು->src[1];
    params.dst  = *ಗಂಟು;

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ತಾರ್ಕಿಕ                    cpu_comparison_ಸಕ್ರಿಯ = ಸುಳ್ಳು;
    ಆದರೆ (im2col_cpu_compare_config.enabled) {
        ಆದರೆ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ಗಂಟು, GGML_OP_IM2_ಲಂಬಸಾಲು)) {
            cpu_comparison_ಸಕ್ರಿಯ = ನಿಜ;
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for IM2COL operation\n");
        }
    }

    ತಾರ್ಕಿಕ kernel_ಫಲಿತಾಂಶ = ggml_et_launch_kernel(dev_ctx, "im2col", &params, ಗಾತ್ರ(params), 0xFFFFFFFF);

    ಆದರೆ (cpu_comparison_ಸಕ್ರಿಯ) {
        ಆದರೆ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ಗಂಟು, &im2col_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for IM2COL operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_ಕೊನೆ("IM2COL", "im2col", ಗಂಟು);
    ಹಿಂತಿರುಗಿ kernel_ಫಲಿತಾಂಶ;
}

ತಾರ್ಕಿಕ ggml_et_op_conv_2d(ggml_backend_et_device_context * dev_ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು) {
    ET_PERF_ಆರಂಭ();

    ಆದರೆ (!dev_ctx || !ಗಂಟು) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (!ಗಂಟು->src[0] || !ಗಂಟು->src[1]) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (!ಗಂಟು->ದತ್ತಾಂಶ || !ಗಂಟು->src[0]->ದತ್ತಾಂಶ || !ಗಂಟು->src[1]->ದತ್ತಾಂಶ) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // Kernel constraints (mirror supports_op; recheck here as a guard).
    ಸ್ಥಿರಾಂಕ ggml_tensor * flt = ಗಂಟು->src[0];  // [Kw, Kh, Cin, Cout]
    ಸ್ಥಿರಾಂಕ ggml_tensor * ಒಳ  = ಗಂಟು->src[1];  // [W,  H,  Cin, N]
    ಆದರೆ (ಗಂಟು->type != GGML_TYPE_F32 || flt->type != GGML_TYPE_F32 || ಒಳ->type != GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ int32_t s0 = ggml_get_op_params_i32(ಗಂಟು, 0);
    ಸ್ಥಿರಾಂಕ int32_t s1 = ggml_get_op_params_i32(ಗಂಟು, 1);
    ಸ್ಥಿರಾಂಕ int32_t p0 = ggml_get_op_params_i32(ಗಂಟು, 2);
    ಸ್ಥಿರಾಂಕ int32_t p1 = ggml_get_op_params_i32(ಗಂಟು, 3);
    ಸ್ಥಿರಾಂಕ int32_t d0 = ggml_get_op_params_i32(ಗಂಟು, 4);
    ಸ್ಥಿರಾಂಕ int32_t d1 = ggml_get_op_params_i32(ಗಂಟು, 5);

    ಆದರೆ (s0 < 1 || s1 < 1) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (d0 != 1 || d1 != 1) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (flt->ne[2] % 16 != 0 || flt->ne[3] % 16 != 0) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (ಒಳ->ne[3] != 1) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (ಗಂಟು->ne[0] <= 0) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;  // OW > 0 (any width OK; staging path handles non-16)
    }
    (ಶೂನ್ಯ) p0;
    (ಶೂನ್ಯ) p1;

    ggml_et_ದ್ವಿಮಾನ_params params;
    params.src0 = *ಗಂಟು->src[0];
    params.src1 = *ಗಂಟು->src[1];
    params.dst  = *ಗಂಟು;

    ತಾರ್ಕಿಕ kernel_ಫಲಿತಾಂಶ = ggml_et_launch_kernel(dev_ctx, "conv_2d_f32_me", &params, ಗಾತ್ರ(params), 0xFFFFFFFFu);

    ET_PERF_ಕೊನೆ("CONV_2D", "conv_2d_f32_me", ಗಂಟು);
    ಹಿಂತಿರುಗಿ kernel_ಫಲಿತಾಂಶ;
}

ತಾರ್ಕಿಕ ggml_et_op_softmax(ggml_backend_et_device_context * dev_ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು) {
    ET_PERF_ಆರಂಭ();

    ಆದರೆ (!dev_ctx || !ಗಂಟು) {
        GGML_LOG_ERROR("ET: Invalid parameters for SOFTMAX operation\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (!ಗಂಟು->src[0]) {
        GGML_LOG_ERROR("ET: SOFTMAX operation missing required input\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * kernel_name;

    ಆದರೆ (ಗಂಟು->type == GGML_TYPE_F32 && ಗಂಟು->src[0]->type == GGML_TYPE_F32) {
        kernel_name = "softmax_f32";

    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_LOG_ERROR("ET: SOFTMAX operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(ಗಂಟು->type),
                       ggml_type_name(ಗಂಟು->src[0]->type));
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // Validate contiguity requirements
    ಆದರೆ (!ggml_is_contiguous(ಗಂಟು)) {
        GGML_LOG_ERROR("ET: SOFTMAX operation requires contiguous destination tensor\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (!ggml_is_contiguous(ಗಂಟು->src[0])) {
        GGML_LOG_ERROR("ET: SOFTMAX operation requires contiguous source tensor\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // Check optional mask tensor
    ಆದರೆ (ಗಂಟು->src[1]) {
        ಆದರೆ (ಗಂಟು->src[1]->type != GGML_TYPE_F32) {
            GGML_LOG_ERROR("ET: SOFTMAX operation with unsupported mask type: %s (F32 required)\n",
                           ggml_type_name(ಗಂಟು->src[1]->type));
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಆದರೆ (!ggml_is_contiguous(ಗಂಟು->src[1])) {
            GGML_LOG_ERROR("ET: SOFTMAX operation requires contiguous mask tensor\n");
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }

    // Check optional sinks tensor
    ಆದರೆ (ಗಂಟು->src[2]) {
        ಆದರೆ (ಗಂಟು->src[2]->type != GGML_TYPE_F32) {
            GGML_LOG_ERROR("ET: SOFTMAX operation with unsupported sinks type: %s (F32 required)\n",
                           ggml_type_name(ಗಂಟು->src[2]->type));
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಆದರೆ (!ggml_is_contiguous(ಗಂಟು->src[2])) {
            GGML_LOG_ERROR("ET: SOFTMAX operation requires contiguous sinks tensor\n");
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }

    // Extract scale and max_bias from op_params
    ದಶಮಾಂಶ scale    = 1.0f;
    ದಶಮಾಂಶ max_bias = 0.0f;
    ಆದರೆ (ಗಂಟು->op_params) {
        ಸ್ಮರಣೆ_ನಕಲು(&scale, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ಗಂಟು->op_params + 0, ಗಾತ್ರ(ದಶಮಾಂಶ));
        ಸ್ಮರಣೆ_ನಕಲು(&max_bias, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ಗಂಟು->op_params + 1, ಗಾತ್ರ(ದಶಮಾಂಶ));
    }

    ggml_et_softmax_params params;
    params.src0 = *ಗಂಟು->src[0];                       // F32 input tensor
    ಆದರೆ (ಗಂಟು->src[1]) {
        params.src1 = *ಗಂಟು->src[1];                   // F32 mask tensor
    } ಇಲ್ಲದಿದ್ದರೆ {
        memset(&params.src1, 0, ಗಾತ್ರ(params.src1));  // Zero if no mask
    }
    ಆದರೆ (ಗಂಟು->src[2]) {
        params.src2 = *ಗಂಟು->src[2];                   // F32 sinks tensor
    } ಇಲ್ಲದಿದ್ದರೆ {
        memset(&params.src2, 0, ಗಾತ್ರ(params.src2));  // Zero if no sinks
    }
    params.dst      = *ಗಂಟು;                           // F32 output tensor
    params.scale    = scale;                           // Scale factor
    params.max_bias = max_bias;                        // ALiBi bias

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ತಾರ್ಕಿಕ                    cpu_comparison_ಸಕ್ರಿಯ = ಸುಳ್ಳು;
    ಆದರೆ (softmax_cpu_compare_config.enabled) {
        ಆದರೆ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ಗಂಟು, GGML_OP_SOFT_MAX)) {
            cpu_comparison_ಸಕ್ರಿಯ = ನಿಜ;
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for SOFTMAX operation\n");
        }
    }

    ತಾರ್ಕಿಕ kernel_ಫಲಿತಾಂಶ = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ಗಾತ್ರ(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ಆದರೆ (cpu_comparison_ಸಕ್ರಿಯ) {
        ಆದರೆ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ಗಂಟು, &softmax_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for SOFTMAX operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_ಕೊನೆ_EXT("SOFTMAX", kernel_name, ಗಂಟು, "scale=%.6f|max_bias=%.6f|has_mask=%s", (ದ್ವಿಗುಣ) scale,
                    (ದ್ವಿಗುಣ) max_bias, ಗಂಟು->src[1] ? "yes" : "no");
    ಹಿಂತಿರುಗಿ kernel_ಫಲಿತಾಂಶ;
}

ತಾರ್ಕಿಕ ggml_et_op_flash_attn_ext(ggml_backend_et_device_context * dev_ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು) {
    ET_PERF_ಆರಂಭ();

    ಆದರೆ (!dev_ctx || !ಗಂಟು) {
        GGML_LOG_ERROR("ET: Invalid parameters for FLASH_ATTN_EXT operation\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (!ಗಂಟು->src[0] || !ಗಂಟು->src[1] || !ಗಂಟು->src[2]) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT operation missing required inputs\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (ಗಂಟು->type != GGML_TYPE_F32 || ಗಂಟು->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT requires F32 Q and dst, got dst=%s q=%s\n", ggml_type_name(ಗಂಟು->type),
                       ggml_type_name(ಗಂಟು->src[0]->type));
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // K and V can be F16 or F32
    ಆದರೆ ((ಗಂಟು->src[1]->type != GGML_TYPE_F32 && ಗಂಟು->src[1]->type != GGML_TYPE_F16) ||
        (ಗಂಟು->src[2]->type != GGML_TYPE_F32 && ಗಂಟು->src[2]->type != GGML_TYPE_F16)) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT K/V must be F16 or F32, got k=%s v=%s\n", ggml_type_name(ಗಂಟು->src[1]->type),
                       ggml_type_name(ಗಂಟು->src[2]->type));
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (ಗಂಟು->src[4] != ಶೂನ್ಯಸೂಚಕ) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT baseline kernel does not support sinks\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // Mask is optional; if present must be F16 or F32
    ಆದರೆ (ಗಂಟು->src[3] != ಶೂನ್ಯಸೂಚಕ && ಗಂಟು->src[3]->type != GGML_TYPE_F32 && ಗಂಟು->src[3]->type != GGML_TYPE_F16) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT mask must be F16 or F32, got %s\n", ggml_type_name(ಗಂಟು->src[3]->type));
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // Q and dst must be row-contiguous F32
    ಆದರೆ (!ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(ಗಂಟು) || !ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(ಗಂಟು->src[0])) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT requires row-contiguous Q and dst\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (ಗಂಟು->nb[0] != ಗಾತ್ರ(ದಶಮಾಂಶ) || ಗಂಟು->src[0]->nb[0] != ಗಾತ್ರ(ದಶಮಾಂಶ)) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT requires contiguous F32 rows for Q and dst\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // K/V must have element-sized stride in dim 0
    ಸ್ಥಿರಾಂಕ size_t k_elem = ಗಂಟು->src[1]->type == GGML_TYPE_F16 ? 2 : 4;
    ಸ್ಥಿರಾಂಕ size_t v_elem = ಗಂಟು->src[2]->type == GGML_TYPE_F16 ? 2 : 4;
    ಆದರೆ (ಗಂಟು->src[1]->nb[0] != k_elem || ಗಂಟು->src[2]->nb[0] != v_elem) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT K/V must have element-sized stride in dim 0\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ದಶಮಾಂಶ scale         = 1.0f;
    ದಶಮಾಂಶ max_bias      = 0.0f;
    ದಶಮಾಂಶ logit_softcap = 0.0f;
    ಸ್ಮರಣೆ_ನಕಲು(&scale, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ಗಂಟು->op_params + 0, ಗಾತ್ರ(scale));
    ಸ್ಮರಣೆ_ನಕಲು(&max_bias, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ಗಂಟು->op_params + 1, ಗಾತ್ರ(max_bias));
    ಸ್ಮರಣೆ_ನಕಲು(&logit_softcap, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ಗಂಟು->op_params + 2, ಗಾತ್ರ(logit_softcap));

    ಆದರೆ (max_bias != 0.0f || logit_softcap != 0.0f) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT baseline kernel does not support max_bias or logit_softcap\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ ggml_prec prec = ggml_flash_attn_ext_get_prec(ಗಂಟು);
    ಆದರೆ (prec != GGML_PREC_F32 && prec != GGML_PREC_DEFAULT) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT baseline kernel only supports F32 precision\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // dk must match between Q and K; dv must match between V and dst
    ಆದರೆ (ಗಂಟು->src[0]->ne[0] != ಗಂಟು->src[1]->ne[0]) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT dk mismatch: Q=%lld K=%lld\n", (ದೀರ್ಘ ದೀರ್ಘ) ಗಂಟು->src[0]->ne[0],
                       (ದೀರ್ಘ ದೀರ್ಘ) ಗಂಟು->src[1]->ne[0]);
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (ಗಂಟು->src[2]->ne[0] != ಗಂಟು->ne[0]) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT dv mismatch: V=%lld dst=%lld\n", (ದೀರ್ಘ ದೀರ್ಘ) ಗಂಟು->src[2]->ne[0],
                       (ದೀರ್ಘ ದೀರ್ಘ) ಗಂಟು->ne[0]);
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (ಗಂಟು->src[2]->ne[0] > 512) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT dv=%lld exceeds maximum 512\n", (ದೀರ್ಘ ದೀರ್ಘ) ಗಂಟು->src[2]->ne[0]);
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (ಗಂಟು->src[0]->ne[0] > 512) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT dk=%lld exceeds maximum 512\n", (ದೀರ್ಘ ದೀರ್ಘ) ಗಂಟು->src[0]->ne[0]);
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // GQA: n_head_q must be a multiple of n_head_kv
    ಸ್ಥಿರಾಂಕ int64_t nhq = ಗಂಟು->src[0]->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t nhk = ಗಂಟು->src[1]->ne[2];
    ಆದರೆ (nhq % nhk != 0) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT n_head_q (%lld) not divisible by n_head_kv (%lld)\n", (ದೀರ್ಘ ದೀರ್ಘ) nhq,
                       (ದೀರ್ಘ ದೀರ್ಘ) nhk);
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // K and V must have matching sequence length, heads, and batch dims
    ಆದರೆ (ಗಂಟು->src[1]->ne[1] != ಗಂಟು->src[2]->ne[1] || ಗಂಟು->src[1]->ne[2] != ಗಂಟು->src[2]->ne[2] ||
        ಗಂಟು->src[1]->ne[3] != ಗಂಟು->src[2]->ne[3]) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT K/V shape mismatch\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // dst layout checks: [dv, nhq, nq, no]
    ಆದರೆ (ಗಂಟು->src[0]->ne[1] != ಗಂಟು->ne[2] || ಗಂಟು->src[0]->ne[2] != ಗಂಟು->ne[1] ||
        ಗಂಟು->src[0]->ne[3] != ಗಂಟು->ne[3]) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT dst shape mismatch\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // Batch dims: Q batch must match K batch
    ಆದರೆ (ಗಂಟು->src[0]->ne[3] != ಗಂಟು->src[1]->ne[3]) {
        GGML_LOG_ERROR("ET: FLASH_ATTN_EXT batch dimension mismatch\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ggml_et_flash_attn_ext_params params;
    memset(&params, 0, ಗಾತ್ರ(params));
    params.src0 = *ಗಂಟು->src[0];
    params.src1 = *ಗಂಟು->src[1];
    params.src2 = *ಗಂಟು->src[2];
    ಆದರೆ (ಗಂಟು->src[3] != ಶೂನ್ಯಸೂಚಕ) {
        params.ಮುಖವಾಡ     = *ಗಂಟು->src[3];
        params.has_ಮುಖವಾಡ = 1;
    }
    params.dst   = *ಗಂಟು;
    params.scale = scale;

    // Use matrix engine kernel when K/V are F16 and dk is a multiple of 32
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * kernel_name;
    ಆದರೆ (ಗಂಟು->src[1]->type == GGML_TYPE_F16 && ಗಂಟು->src[2]->type == GGML_TYPE_F16 && (ಗಂಟು->src[0]->ne[0] % 32) == 0) {
        kernel_name = "flash_attn_ext_f16_me";
    } ಇಲ್ಲದಿದ್ದರೆ {
        kernel_name = "flash_attn_ext_f32";
    }

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ kernel_ಫಲಿತಾಂಶ = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ಗಾತ್ರ(params), 0xFFFFFFFF);

    ET_PERF_ಕೊನೆ_EXT("FLASH_ATTN_EXT", kernel_name, ಗಂಟು, "scale=%.6f", (ದ್ವಿಗುಣ) scale);
    ಹಿಂತಿರುಗಿ kernel_ಫಲಿತಾಂಶ;
}

ತಾರ್ಕಿಕ ggml_et_op_get_ಅಡ್ಡಸಾಲುಗಳು(ggml_backend_et_device_context * dev_ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು) {
    ET_PERF_ಆರಂಭ();

    ಆದರೆ (!dev_ctx || !ಗಂಟು) {
        GGML_LOG_ERROR("ET: Invalid parameters for GET_ROWS operation\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (!ಗಂಟು->src[0] || !ಗಂಟು->src[1]) {
        GGML_LOG_ERROR("ET: GET_ROWS operation missing required inputs\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * kernel_name;

    ಆದರೆ (ಗಂಟು->type == GGML_TYPE_F32 && ಗಂಟು->src[1]->type == GGML_TYPE_I32 &&
        (ಗಂಟು->src[0]->type == GGML_TYPE_F32 || ಗಂಟು->src[0]->type == GGML_TYPE_F16 ||
         ಗಂಟು->src[0]->type == GGML_TYPE_Q4_0 || ಗಂಟು->src[0]->type == GGML_TYPE_Q8_0 ||
         ಗಂಟು->src[0]->type == GGML_TYPE_Q4_K)) {
        kernel_name = "get_rows_f32";

    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_LOG_ERROR("ET: GET_ROWS operation with unsupported types: dst=%s src0=%s src1=%s\n",
                       ggml_type_name(ಗಂಟು->type), ggml_type_name(ಗಂಟು->src[0]->type),
                       ggml_type_name(ಗಂಟು->src[1]->type));
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // Validate contiguity requirements
    ಆದರೆ (!ggml_is_contiguous(ಗಂಟು)) {
        GGML_LOG_ERROR("ET: GET_ROWS operation requires contiguous destination tensor\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (!ggml_is_contiguous(ಗಂಟು->src[0])) {
        GGML_LOG_ERROR("ET: GET_ROWS operation requires contiguous data tensor\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (!ggml_is_contiguous(ಗಂಟು->src[1])) {
        GGML_LOG_ERROR("ET: GET_ROWS operation requires contiguous indices tensor\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // Validate dimension constraints from ggml implementation
    ಆದರೆ (ಗಂಟು->src[0]->ne[2] != ಗಂಟು->src[1]->ne[1] || ಗಂಟು->src[1]->ne[3] != 1) {
        GGML_LOG_ERROR(
            "ET: GET_ROWS operation dimension constraint failed: src0.ne[2]=%lld != src1.ne[1]=%lld or src1.ne[3]=%lld "
            "!= 1\n",
            (ದೀರ್ಘ ದೀರ್ಘ) ಗಂಟು->src[0]->ne[2], (ದೀರ್ಘ ದೀರ್ಘ) ಗಂಟು->src[1]->ne[1], (ದೀರ್ಘ ದೀರ್ಘ) ಗಂಟು->src[1]->ne[3]);
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ggml_et_get_ಅಡ್ಡಸಾಲುಗಳು_params params;
    params.src0 = *ಗಂಟು->src[0];  // Data tensor (F32 or Q8_0)
    params.src1 = *ಗಂಟು->src[1];  // Indices tensor (I32)
    params.dst  = *ಗಂಟು;          // Output tensor (F32)

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ತಾರ್ಕಿಕ                    cpu_comparison_ಸಕ್ರಿಯ = ಸುಳ್ಳು;
    ಆದರೆ (get_ಅಡ್ಡಸಾಲುಗಳು_cpu_compare_config.enabled) {
        ಆದರೆ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ಗಂಟು, GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು)) {
            cpu_comparison_ಸಕ್ರಿಯ = ನಿಜ;
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for GET_ROWS operation\n");
        }
    }

    ತಾರ್ಕಿಕ kernel_ಫಲಿತಾಂಶ = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ಗಾತ್ರ(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ಆದರೆ (cpu_comparison_ಸಕ್ರಿಯ) {
        ಆದರೆ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ಗಂಟು, &get_ಅಡ್ಡಸಾಲುಗಳು_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for GET_ROWS operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_ಕೊನೆ("GET_ROWS", kernel_name, ಗಂಟು);
    ಹಿಂತಿರುಗಿ kernel_ಫಲಿತಾಂಶ;
}

ತಾರ್ಕಿಕ ggml_et_op_cont(ggml_backend_et_device_context * dev_ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು) {
    ET_PERF_ಆರಂಭ();

    // Validate source tensor exists
    ಆದರೆ (!ಗಂಟು->src[0]) {
        GGML_LOG_ERROR("ET: CONT operation missing source tensor\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // Validate types match (input and output must be same type)
    ಆದರೆ (ಗಂಟು->type != ಗಂಟು->src[0]->type) {
        GGML_LOG_ERROR("ET: CONT operation type mismatch: src=%s dst=%s\n", ggml_type_name(ಗಂಟು->src[0]->type),
                       ggml_type_name(ಗಂಟು->type));
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // Validate supported types
    ಆದರೆ (ಗಂಟು->type != GGML_TYPE_F32 && ಗಂಟು->type != GGML_TYPE_F16) {
        GGML_LOG_ERROR("ET: CONT operation unsupported type: %s (only F32 and F16 supported)\n",
                       ggml_type_name(ಗಂಟು->type));
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // Validate contiguity - output must be contiguous, input can be non-contiguous
    ಆದರೆ (!ggml_is_contiguous(ಗಂಟು)) {
        GGML_LOG_ERROR("ET: CONT operation requires contiguous output tensor\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // Select kernel based on type
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * kernel_name;
    ಆದರೆ (ಗಂಟು->type == GGML_TYPE_F32) {
        kernel_name = "cont_f32";
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಗಂಟು->type == GGML_TYPE_F16) {
        kernel_name = "cont_f16";
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_LOG_ERROR("ET: CONT operation with unsupported type: %s\n", ggml_type_name(ಗಂಟು->type));
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ggml_et_cont_params params;
    params.src0 = *ಗಂಟು->src[0];  // Input tensor (potentially non-contiguous)
    params.dst  = *ಗಂಟು;          // Output tensor (contiguous)

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ತಾರ್ಕಿಕ                    cpu_comparison_ಸಕ್ರಿಯ = ಸುಳ್ಳು;
    ಆದರೆ (cont_cpu_compare_config.enabled) {
        ಆದರೆ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ಗಂಟು, GGML_OP_CONT)) {
            cpu_comparison_ಸಕ್ರಿಯ = ನಿಜ;
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for CONT operation\n");
        }
    }

    ತಾರ್ಕಿಕ kernel_ಫಲಿತಾಂಶ = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ಗಾತ್ರ(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ಆದರೆ (cpu_comparison_ಸಕ್ರಿಯ) {
        ಆದರೆ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ಗಂಟು, &cont_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for CONT operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_ಕೊನೆ("CONT", kernel_name, ಗಂಟು);
    ಹಿಂತಿರುಗಿ kernel_ಫಲಿತಾಂಶ;
}

ತಾರ್ಕಿಕ ggml_et_op_cumsum(ggml_backend_et_device_context * dev_ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು) {
    ET_PERF_ಆರಂಭ();

    ಆದರೆ (!dev_ctx || !ಗಂಟು || !ಗಂಟು->src[0]) {
        GGML_LOG_ERROR("ET: Invalid parameters for CUMSUM operation\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (ಗಂಟು->type != GGML_TYPE_F32 || ಗಂಟು->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: CUMSUM operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(ಗಂಟು->type),
                       ggml_type_name(ಗಂಟು->src[0]->type));
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * kernel_name = "cumsum_f32";

    ggml_et_cumsum_params params;
    params.src0 = *ಗಂಟು->src[0];
    params.dst  = *ಗಂಟು;

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ತಾರ್ಕಿಕ                    cpu_comparison_ಸಕ್ರಿಯ = ಸುಳ್ಳು;
    ಆದರೆ (cumsum_cpu_compare_config.enabled) {
        ಆದರೆ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ಗಂಟು, GGML_OP_CUMSUM)) {
            cpu_comparison_ಸಕ್ರಿಯ = ನಿಜ;
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for CUMSUM operation\n");
        }
    }

    ತಾರ್ಕಿಕ kernel_ಫಲಿತಾಂಶ = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ಗಾತ್ರ(params), 0xFFFFFFFF);

    ಆದರೆ (cpu_comparison_ಸಕ್ರಿಯ) {
        ಆದರೆ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ಗಂಟು, &cumsum_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for CUMSUM operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_ಕೊನೆ("CUMSUM", kernel_name, ಗಂಟು);
    ಹಿಂತಿರುಗಿ kernel_ಫಲಿತಾಂಶ;
}

ತಾರ್ಕಿಕ ggml_et_op_cpy(ggml_backend_et_device_context * dev_ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು) {
    ET_PERF_ಆರಂಭ();

    // CPY copies data from src[0] into the layout of dst (which matches src[1])
    // For same-type with contiguous dst, this is identical to CONT
    ಆದರೆ (!ಗಂಟು->src[0]) {
        GGML_LOG_ERROR("ET: CPY operation missing source tensor\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // Scalar / zero-element special path: if any dimension is 0, nothing to copy
    ಸ್ಥಿರಾಂಕ int64_t nelements = ಗಂಟು->ne[0] * ಗಂಟು->ne[1] * ಗಂಟು->ne[2] * ಗಂಟು->ne[3];
    ಆದರೆ (nelements == 0) {
        GGML_LOG_DEBUG("ET: CPY no-op (zero elements): ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]\n",
                       ಗಂಟು->ne[0], ಗಂಟು->ne[1], ಗಂಟು->ne[2], ಗಂಟು->ne[3]);
        ET_PERF_ಕೊನೆ("CPY", "noop", ಗಂಟು);
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    // Only F32 and F16 supported for dst
    ಆದರೆ (ಗಂಟು->type != GGML_TYPE_F32 && ಗಂಟು->type != GGML_TYPE_F16) {
        GGML_LOG_ERROR("ET: CPY unsupported dst type: %s\n", ggml_type_name(ಗಂಟು->type));
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // Select kernel based on src/dst type combination
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * kernel_name;
    ಆದರೆ (ಗಂಟು->src[0]->type == GGML_TYPE_F32 && ಗಂಟು->type == GGML_TYPE_F32) {
        kernel_name = "cont_f32";
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಗಂಟು->src[0]->type == GGML_TYPE_F16 && ಗಂಟು->type == GGML_TYPE_F16) {
        kernel_name = "cont_f16";
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಗಂಟು->src[0]->type == GGML_TYPE_F32 && ಗಂಟು->type == GGML_TYPE_F16) {
        kernel_name = "cpy_f32_f16";
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_LOG_ERROR("ET: CPY unsupported type combination: src=%s dst=%s\n", ggml_type_name(ಗಂಟು->src[0]->type),
                       ggml_type_name(ಗಂಟು->type));
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ggml_et_cont_params params;
    params.src0 = *ಗಂಟು->src[0];
    params.dst  = *ಗಂಟು;

    // CPU comparison for debugging
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ತಾರ್ಕಿಕ                    cpu_comparison_ಸಕ್ರಿಯ = ಸುಳ್ಳು;
    ಆದರೆ (cont_cpu_compare_config.enabled) {
        ಆದರೆ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ಗಂಟು, GGML_OP_CPY)) {
            cpu_comparison_ಸಕ್ರಿಯ = ನಿಜ;
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for CPY operation\n");
        }
    }

    ತಾರ್ಕಿಕ kernel_ಫಲಿತಾಂಶ = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ಗಾತ್ರ(params), 0xFFFFFFFF);

    ಆದರೆ (cpu_comparison_ಸಕ್ರಿಯ) {
        ಆದರೆ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ಗಂಟು, &cont_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for CPY operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_ಕೊನೆ("CPY", kernel_name, ಗಂಟು);
    ಹಿಂತಿರುಗಿ kernel_ಫಲಿತಾಂಶ;
}

ತಾರ್ಕಿಕ ggml_et_op_concat(ggml_backend_et_device_context * dev_ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು) {
    ET_PERF_ಆರಂಭ();

    ಆದರೆ (!dev_ctx || !ಗಂಟು) {
        GGML_LOG_ERROR("ET: Invalid parameters for CONCAT operation\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (!ಗಂಟು->src[0] || !ಗಂಟು->src[1]) {
        GGML_LOG_ERROR("ET: CONCAT operation missing required inputs\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * kernel_name;

    ಆದರೆ (ಗಂಟು->type == GGML_TYPE_F32 && ಗಂಟು->src[0]->type == GGML_TYPE_F32 && ಗಂಟು->src[1]->type == GGML_TYPE_F32) {
        kernel_name = "concat_f32";

    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_LOG_ERROR("ET: CONCAT operation with unsupported types: dst=%s src0=%s src1=%s\n",
                       ggml_type_name(ಗಂಟು->type), ggml_type_name(ಗಂಟು->src[0]->type),
                       ggml_type_name(ಗಂಟು->src[1]->type));
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    int32_t dim;
    ಸ್ಮರಣೆ_ನಕಲು(&dim, ಗಂಟು->op_params, ಗಾತ್ರ(int32_t));

    ggml_et_concat_params params;
    params.src0 = *ಗಂಟು->src[0];
    params.src1 = *ಗಂಟು->src[1];
    params.dst  = *ಗಂಟು;
    params.dim  = dim;

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ತಾರ್ಕಿಕ                    cpu_comparison_ಸಕ್ರಿಯ = ಸುಳ್ಳು;
    ಆದರೆ (concat_cpu_compare_config.enabled) {
        ಆದರೆ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ಗಂಟು, GGML_OP_CONCAT)) {
            cpu_comparison_ಸಕ್ರಿಯ = ನಿಜ;
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for CONCAT operation\n");
        }
    }

    ತಾರ್ಕಿಕ kernel_ಫಲಿತಾಂಶ = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ಗಾತ್ರ(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ಆದರೆ (cpu_comparison_ಸಕ್ರಿಯ) {
        ಆದರೆ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ಗಂಟು, &concat_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for CONCAT operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_ಕೊನೆ_EXT("CONCAT", kernel_name, ಗಂಟು, "dim=%d", dim);
    ಹಿಂತಿರುಗಿ kernel_ಫಲಿತಾಂಶ;
}

ತಾರ್ಕಿಕ ggml_et_op_repeat(ggml_backend_et_device_context * dev_ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು) {
    ET_PERF_ಆರಂಭ();

    ಆದರೆ (!dev_ctx || !ಗಂಟು) {
        GGML_LOG_ERROR("ET: Invalid parameters for REPEAT operation\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (!ಗಂಟು->src[0]) {
        GGML_LOG_ERROR("ET: REPEAT operation missing required input\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * kernel_name;

    ಆದರೆ (ಗಂಟು->type == GGML_TYPE_F32 && ಗಂಟು->src[0]->type == GGML_TYPE_F32) {
        // No-op REPEAT (every repeat factor is 1): the output is just a copy
        // of the input. Route to cont_f32, whose contiguous fast path handles
        // arbitrary sizes (including those rejected by repeat_f32's gate,
        // e.g. ne[0]=1).
        ಆದರೆ (ggml_are_same_shape(ಗಂಟು->src[0], ಗಂಟು)) {
            kernel_name = "cont_f32";
        } ಇಲ್ಲದಿದ್ದರೆ {
            kernel_name = "repeat_f32";
        }

    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_LOG_ERROR("ET: REPEAT operation with unsupported types: dst=%s src0=%s\n", ggml_type_name(ಗಂಟು->type),
                       ggml_type_name(ಗಂಟು->src[0]->type));
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // ggml_et_cont_params and ggml_et_repeat_params have identical layouts
    // (just src0 + dst), so the same payload works for either kernel.
    ggml_et_repeat_params params;
    params.src0 = *ಗಂಟು->src[0];
    params.dst  = *ಗಂಟು;

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ತಾರ್ಕಿಕ                    cpu_comparison_ಸಕ್ರಿಯ = ಸುಳ್ಳು;
    ಆದರೆ (repeat_cpu_compare_config.enabled) {
        ಆದರೆ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ಗಂಟು, GGML_OP_REPEAT)) {
            cpu_comparison_ಸಕ್ರಿಯ = ನಿಜ;
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for REPEAT operation\n");
        }
    }

    ತಾರ್ಕಿಕ kernel_ಫಲಿತಾಂಶ = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ಗಾತ್ರ(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ಆದರೆ (cpu_comparison_ಸಕ್ರಿಯ) {
        ಆದರೆ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ಗಂಟು, &repeat_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for REPEAT operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_ಕೊನೆ("REPEAT", kernel_name, ಗಂಟು);
    ಹಿಂತಿರುಗಿ kernel_ಫಲಿತಾಂಶ;
}

ತಾರ್ಕಿಕ ggml_et_op_ssm_conv(ggml_backend_et_device_context * dev_ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು) {
    ET_PERF_ಆರಂಭ();

    ಆದರೆ (!dev_ctx || !ಗಂಟು || !ಗಂಟು->src[0] || !ಗಂಟು->src[1]) {
        GGML_LOG_ERROR("ET: Invalid parameters for SSM_CONV operation\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (ಗಂಟು->type != GGML_TYPE_F32 || ಗಂಟು->src[0]->type != GGML_TYPE_F32 || ಗಂಟು->src[1]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: SSM_CONV operation with unsupported types: dst=%s src0=%s src1=%s\n",
                       ggml_type_name(ಗಂಟು->type), ggml_type_name(ಗಂಟು->src[0]->type),
                       ggml_type_name(ಗಂಟು->src[1]->type));
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * kernel_name = "ssm_conv_f32";

    ggml_et_ssm_conv_params params;
    params.src0 = *ಗಂಟು->src[0];
    params.src1 = *ಗಂಟು->src[1];
    params.dst  = *ಗಂಟು;

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ತಾರ್ಕಿಕ                    cpu_comparison_ಸಕ್ರಿಯ = ಸುಳ್ಳು;
    ಆದರೆ (ssm_conv_cpu_compare_config.enabled) {
        ಆದರೆ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ಗಂಟು, GGML_OP_SSM_CONV)) {
            cpu_comparison_ಸಕ್ರಿಯ = ನಿಜ;
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for SSM_CONV operation\n");
        }
    }

    ತಾರ್ಕಿಕ kernel_ಫಲಿತಾಂಶ = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ಗಾತ್ರ(params), 0xFFFFFFFF);

    ಆದರೆ (cpu_comparison_ಸಕ್ರಿಯ) {
        ಆದರೆ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ಗಂಟು, &ssm_conv_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for SSM_CONV operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_ಕೊನೆ("SSM_CONV", kernel_name, ಗಂಟು);
    ಹಿಂತಿರುಗಿ kernel_ಫಲಿತಾಂಶ;
}

ತಾರ್ಕಿಕ ggml_et_op_ssm_ಪರಿಶೀಲನೆ(ggml_backend_et_device_context * dev_ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು) {
    ET_PERF_ಆರಂಭ();

    ಆದರೆ (!dev_ctx || !ಗಂಟು) {
        GGML_LOG_ERROR("ET: Invalid parameters for SSM_SCAN operation\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 7; ++i) {
        ಆದರೆ (!ಗಂಟು->src[i]) {
            GGML_LOG_ERROR("ET: SSM_SCAN missing required input %d\n", i);
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }

    ಆದರೆ (ಗಂಟು->type != GGML_TYPE_F32 || ಗಂಟು->src[0]->type != GGML_TYPE_F32 || ಗಂಟು->src[1]->type != GGML_TYPE_F32 ||
        ಗಂಟು->src[2]->type != GGML_TYPE_F32 || ಗಂಟು->src[3]->type != GGML_TYPE_F32 ||
        ಗಂಟು->src[4]->type != GGML_TYPE_F32 || ಗಂಟು->src[5]->type != GGML_TYPE_F32 ||
        ಗಂಟು->src[6]->type != GGML_TYPE_I32) {
        GGML_LOG_ERROR("ET: SSM_SCAN operation with unsupported types\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ggml_et_ssm_ಪರಿಶೀಲನೆ_params params;
    params.src0 = *ಗಂಟು->src[0];
    params.src1 = *ಗಂಟು->src[1];
    params.src2 = *ಗಂಟು->src[2];
    params.src3 = *ಗಂಟು->src[3];
    params.src4 = *ಗಂಟು->src[4];
    params.src5 = *ಗಂಟು->src[5];
    params.src6 = *ಗಂಟು->src[6];
    params.dst  = *ಗಂಟು;
    params.K    = ggml_get_op_params_i32(ಗಂಟು, 0);

    ತಾರ್ಕಿಕ kernel_ಫಲಿತಾಂಶ = ggml_et_launch_kernel(dev_ctx, "ssm_scan_f32", &params, ಗಾತ್ರ(params), 0xFFFFFFFF);

    ET_PERF_ಕೊನೆ("SSM_SCAN", "ssm_scan_f32", ಗಂಟು);
    ಹಿಂತಿರುಗಿ kernel_ಫಲಿತಾಂಶ;
}

ತಾರ್ಕಿಕ ggml_et_op_rwkv_wkv6(ggml_backend_et_device_context * dev_ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು) {
    ET_PERF_ಆರಂಭ();

    ಆದರೆ (!dev_ctx || !ಗಂಟು) {
        GGML_LOG_ERROR("ET: Invalid parameters for RWKV_WKV6 operation\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // Validate all 6 source tensors exist
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i <= 5; i++) {
        ಆದರೆ (!ಗಂಟು->src[i]) {
            GGML_LOG_ERROR("ET: RWKV_WKV6 operation missing src[%d]\n", i);
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }

    ಆದರೆ (ಗಂಟು->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: RWKV_WKV6 only supports F32, got %s\n", ggml_type_name(ಗಂಟು->type));
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * kernel_name = "rwkv_wkv6_f32";

    ಸ್ಥಿರಾಂಕ int64_t S      = ಗಂಟು->src[0]->ne[0];  // head_size
    ಸ್ಥಿರಾಂಕ int64_t H      = ಗಂಟು->src[0]->ne[1];  // num heads
    ಸ್ಥಿರಾಂಕ int64_t T      = ಗಂಟು->src[1]->ne[2];  // num tokens
    ಸ್ಥಿರಾಂಕ int64_t n_seqs = ಗಂಟು->src[5]->ne[1];  // num sequences
    ಸ್ಥಿರಾಂಕ int64_t C      = S * H;

    ggml_et_rwkv_wkv6_params params;
    params.k        = (ದಶಮಾಂಶ *) ಗಂಟು->src[0]->ದತ್ತಾಂಶ;
    params.v        = (ದಶಮಾಂಶ *) ಗಂಟು->src[1]->ದತ್ತಾಂಶ;
    params.r        = (ದಶಮಾಂಶ *) ಗಂಟು->src[2]->ದತ್ತಾಂಶ;
    params.tf       = (ದಶಮಾಂಶ *) ಗಂಟು->src[3]->ದತ್ತಾಂಶ;
    params.td       = (ದಶಮಾಂಶ *) ಗಂಟು->src[4]->ದತ್ತಾಂಶ;
    params.state_ಒಳ = (ದಶಮಾಂಶ *) ಗಂಟು->src[5]->ದತ್ತಾಂಶ;
    params.dst      = (ದಶಮಾಂಶ *) ಗಂಟು->ದತ್ತಾಂಶ;
    params.C        = (int32_t) C;
    params.H        = (int32_t) H;
    params.S        = (int32_t) S;
    params.T        = (int32_t) T;
    params.n_seqs   = (int32_t) n_seqs;

    // Phase 1: Initialize CPU comparison context
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ತಾರ್ಕಿಕ                    cpu_comparison_ಸಕ್ರಿಯ = ಸುಳ್ಳು;
    ಆದರೆ (rwkv_wkv6_cpu_compare_config.enabled) {
        ಆದರೆ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ಗಂಟು, GGML_OP_RWKV_WKV6)) {
            cpu_comparison_ಸಕ್ರಿಯ = ನಿಜ;
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for RWKV_WKV6 operation\n");
        }
    }

    ತಾರ್ಕಿಕ kernel_ಫಲಿತಾಂಶ = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ಗಾತ್ರ(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare
    ಆದರೆ (cpu_comparison_ಸಕ್ರಿಯ) {
        ಆದರೆ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ಗಂಟು, &rwkv_wkv6_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for RWKV_WKV6 operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_ಕೊನೆ_EXT("RWKV_WKV6", kernel_name, ಗಂಟು, "S=%d H=%d T=%d n_seqs=%d", (ಪೂರ್ಣಾಂಕ) S, (ಪೂರ್ಣಾಂಕ) H, (ಪೂರ್ಣಾಂಕ) T,
                    (ಪೂರ್ಣಾಂಕ) n_seqs);
    ಹಿಂತಿರುಗಿ kernel_ಫಲಿತಾಂಶ;
}

ತಾರ್ಕಿಕ ggml_et_op_rwkv_wkv7(ggml_backend_et_device_context * dev_ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು) {
    ET_PERF_ಆರಂಭ();

    ಆದರೆ (!dev_ctx || !ಗಂಟು) {
        GGML_LOG_ERROR("ET: Invalid parameters for RWKV_WKV7 operation\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // Validate all 7 source tensors exist
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i <= 6; i++) {
        ಆದರೆ (!ಗಂಟು->src[i]) {
            GGML_LOG_ERROR("ET: RWKV_WKV7 operation missing src[%d]\n", i);
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }

    ಆದರೆ (ಗಂಟು->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: RWKV_WKV7 only supports F32, got %s\n", ggml_type_name(ಗಂಟು->type));
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * kernel_name = "rwkv_wkv7_f32";

    ಸ್ಥಿರಾಂಕ int64_t S      = ಗಂಟು->src[2]->ne[0];  // head_size
    ಸ್ಥಿರಾಂಕ int64_t H      = ಗಂಟು->src[2]->ne[1];  // num heads
    ಸ್ಥಿರಾಂಕ int64_t T      = ಗಂಟು->src[1]->ne[2];  // num tokens
    ಸ್ಥಿರಾಂಕ int64_t n_seqs = ಗಂಟು->src[6]->ne[1];  // num sequences
    ಸ್ಥಿರಾಂಕ int64_t C      = S * H;

    ggml_et_rwkv_wkv7_params params;
    params.r        = (ದಶಮಾಂಶ *) ಗಂಟು->src[0]->ದತ್ತಾಂಶ;
    params.w        = (ದಶಮಾಂಶ *) ಗಂಟು->src[1]->ದತ್ತಾಂಶ;
    params.k        = (ದಶಮಾಂಶ *) ಗಂಟು->src[2]->ದತ್ತಾಂಶ;
    params.v        = (ದಶಮಾಂಶ *) ಗಂಟು->src[3]->ದತ್ತಾಂಶ;
    params.a        = (ದಶಮಾಂಶ *) ಗಂಟು->src[4]->ದತ್ತಾಂಶ;
    params.b        = (ದಶಮಾಂಶ *) ಗಂಟು->src[5]->ದತ್ತಾಂಶ;
    params.state_ಒಳ = (ದಶಮಾಂಶ *) ಗಂಟು->src[6]->ದತ್ತಾಂಶ;
    params.dst      = (ದಶಮಾಂಶ *) ಗಂಟು->ದತ್ತಾಂಶ;
    params.C        = (int32_t) C;
    params.H        = (int32_t) H;
    params.S        = (int32_t) S;
    params.T        = (int32_t) T;
    params.n_seqs   = (int32_t) n_seqs;

    // Phase 1: Initialize CPU comparison context
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ತಾರ್ಕಿಕ                    cpu_comparison_ಸಕ್ರಿಯ = ಸುಳ್ಳು;
    ಆದರೆ (rwkv_wkv7_cpu_compare_config.enabled) {
        ಆದರೆ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ಗಂಟು, GGML_OP_RWKV_WKV7)) {
            cpu_comparison_ಸಕ್ರಿಯ = ನಿಜ;
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for RWKV_WKV7 operation\n");
        }
    }

    ತಾರ್ಕಿಕ kernel_ಫಲಿತಾಂಶ = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ಗಾತ್ರ(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare
    ಆದರೆ (cpu_comparison_ಸಕ್ರಿಯ) {
        ಆದರೆ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ಗಂಟು, &rwkv_wkv7_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for RWKV_WKV7 operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_ಕೊನೆ_EXT("RWKV_WKV7", kernel_name, ಗಂಟು, "S=%d H=%d T=%d n_seqs=%d", (ಪೂರ್ಣಾಂಕ) S, (ಪೂರ್ಣಾಂಕ) H, (ಪೂರ್ಣಾಂಕ) T,
                    (ಪೂರ್ಣಾಂಕ) n_seqs);
    ಹಿಂತಿರುಗಿ kernel_ಫಲಿತಾಂಶ;
}

ಸ್ಥಿರ ggml_et_cpu_compare_config gated_delta_net_cpu_compare_config = {
    /* .enabled = */ ಸುಳ್ಳು,
    /* .use_cpu_result = */ ಸುಳ್ಳು,
    /* .log_differences = */ ನಿಜ,
    /* .tolerance = */ 1e-4f,
    /* .max_log_elements = */ 4096
};

ತಾರ್ಕಿಕ ggml_et_op_gated_delta_net(ggml_backend_et_device_context * dev_ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು) {
    ET_PERF_ಆರಂಭ();

    ಆದರೆ (!dev_ctx || !ಗಂಟು) {
        GGML_LOG_ERROR("ET: Invalid parameters for GATED_DELTA_NET operation\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // Validate all 6 source tensors exist
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i <= 5; i++) {
        ಆದರೆ (!ಗಂಟು->src[i]) {
            GGML_LOG_ERROR("ET: GATED_DELTA_NET operation missing src[%d]\n", i);
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }

    ಆದರೆ (ಗಂಟು->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: GATED_DELTA_NET only supports F32, got %s\n", ggml_type_name(ಗಂಟು->type));
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * kernel_name = "gated_delta_net_f32";

    ಸ್ಥಿರಾಂಕ ggml_tensor * src_q     = ಗಂಟು->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src_k     = ಗಂಟು->src[1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src_v     = ಗಂಟು->src[2];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src_g     = ಗಂಟು->src[3];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src_beta  = ಗಂಟು->src[4];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src_state = ಗಂಟು->src[5];

    ಸ್ಥಿರಾಂಕ int64_t S_v      = src_v->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t H        = src_v->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t n_ಟೋಕನುಗಳು = src_v->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t n_seqs   = src_v->ne[3];
    ಸ್ಥಿರಾಂಕ int64_t H_q      = src_q->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t H_k      = src_k->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t n_seqs_q = src_q->ne[3];
    ಸ್ಥಿರಾಂಕ int64_t n_seqs_k = src_k->ne[3];

    ggml_et_gated_delta_net_params params;
    params.q        = *src_q;
    params.k        = *src_k;
    params.v        = *src_v;
    params.g        = *src_g;
    params.beta     = *src_beta;
    params.state_ಒಳ = *src_state;
    params.dst      = *ಗಂಟು;
    params.S_v      = (int32_t) S_v;
    params.H        = (int32_t) H;
    params.H_q      = (int32_t) H_q;
    params.H_k      = (int32_t) H_k;
    params.n_ಟೋಕನುಗಳು = (int32_t) n_ಟೋಕನುಗಳು;
    params.n_seqs   = (int32_t) n_seqs;
    params.n_seqs_q = (int32_t) n_seqs_q;
    params.n_seqs_k = (int32_t) n_seqs_k;
    params.kda      = (src_g->ne[0] == S_v) ? 1 : 0;
    params.K        = ggml_get_op_params_i32(ಗಂಟು, 0);
    params.scale    = 1.0f / sqrtf((ದಶಮಾಂಶ) S_v);

    // CPU comparison for debugging
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ತಾರ್ಕಿಕ                    cpu_comparison_ಸಕ್ರಿಯ = ಸುಳ್ಳು;
    ಆದರೆ (gated_delta_net_cpu_compare_config.enabled) {
        ಆದರೆ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ಗಂಟು, GGML_OP_GATED_DELTA_NET)) {
            cpu_comparison_ಸಕ್ರಿಯ = ನಿಜ;
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for GATED_DELTA_NET operation\n");
        }
    }

    ತಾರ್ಕಿಕ kernel_ಫಲಿತಾಂಶ = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ಗಾತ್ರ(params), 0xFFFFFFFF);

    ಆದರೆ (cpu_comparison_ಸಕ್ರಿಯ) {
        ಆದರೆ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ಗಂಟು, &gated_delta_net_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for GATED_DELTA_NET operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_ಕೊನೆ_EXT("GATED_DELTA_NET", kernel_name, ಗಂಟು, "S_v=%d H=%d n_tokens=%d n_seqs=%d kda=%d", (ಪೂರ್ಣಾಂಕ) S_v,
                    (ಪೂರ್ಣಾಂಕ) H, (ಪೂರ್ಣಾಂಕ) n_ಟೋಕನುಗಳು, (ಪೂರ್ಣಾಂಕ) n_seqs, params.kda);
    ಹಿಂತಿರುಗಿ kernel_ಫಲಿತಾಂಶ;
}

ತಾರ್ಕಿಕ ggml_et_op_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು(ggml_backend_et_device_context * dev_ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು) {
    ET_PERF_ಆರಂಭ();

    ಆದರೆ (!dev_ctx || !ಗಂಟು) {
        GGML_LOG_ERROR("ET: Invalid parameters for SET_ROWS operation\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (!ಗಂಟು->src[0] || !ಗಂಟು->src[1] || !ಗಂಟು->src[2]) {
        GGML_LOG_ERROR(
            "ET: SET_ROWS operation missing required inputs (needs src[0]=base, src[1]=indices, src[2]=data)\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * kernel_name;

    // Support F32 data with I64 indices -> F32/F16 output (scatter operation)
    ಆದರೆ (ಗಂಟು->src[0]->type == GGML_TYPE_F32 && ಗಂಟು->src[1]->type == GGML_TYPE_I64 &&
        (ಗಂಟು->type == GGML_TYPE_F32 || ಗಂಟು->type == GGML_TYPE_F16)) {
        ಆದರೆ (ಗಂಟು->type == GGML_TYPE_F32 || ಗಂಟು->type == GGML_TYPE_F16) {
            kernel_name = "set_rows_f32";
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_LOG_ERROR("ET: SET_ROWS unsupported output type: %s\n", ggml_type_name(ಗಂಟು->type));
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_LOG_ERROR("ET: SET_ROWS operation with unsupported types: dst=%s src0=%s src1=%s\n",
                       ggml_type_name(ಗಂಟು->type), ggml_type_name(ಗಂಟು->src[0]->type),
                       ggml_type_name(ಗಂಟು->src[1]->type));
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // Validate contiguity requirements
    ಆದರೆ (!ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(ಗಂಟು)) {
        GGML_LOG_ERROR("ET: SET_ROWS operation requires contiguous-rows destination tensor\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (!ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(ಗಂಟು->src[0])) {
        GGML_LOG_ERROR("ET: SET_ROWS operation requires contiguous-rows source tensor\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (!ggml_is_contiguous(ಗಂಟು->src[1])) {
        GGML_LOG_ERROR("ET: SET_ROWS operation requires contiguous indices tensor\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // Validate dimension constraints from ggml implementation
    ಆದರೆ (!(ಗಂಟು->ne[0] == ಗಂಟು->src[0]->ne[0] &&              // same number of columns
          ಗಂಟು->ne[2] == ಗಂಟು->src[0]->ne[2] &&              // same batch size
          ಗಂಟು->ne[3] == ಗಂಟು->src[0]->ne[3] &&              // same outer dimension
          ಗಂಟು->src[0]->ne[1] == ಗಂಟು->src[1]->ne[0] &&      // src rows = index count
          ಗಂಟು->src[0]->ne[2] % ಗಂಟು->src[1]->ne[1] == 0 &&  // batch constraint
          ಗಂಟು->src[0]->ne[3] % ಗಂಟು->src[1]->ne[2] == 0 &&  // outer constraint
          ಗಂಟು->src[1]->ne[3] == 1)) {                       // indices constraint
        GGML_LOG_ERROR("ET: SET_ROWS operation dimension constraint failed\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ggml_et_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_params params;
    params.src0 = *ಗಂಟು->src[0];  // F32 source data tensor
    params.src1 = *ಗಂಟು->src[1];  // I64 indices tensor
    params.dst  = *ಗಂಟು;          // F32/F16 destination tensor

    // Phase 1: Initialize CPU comparison context and copy source buffers (before ET kernel)
    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ತಾರ್ಕಿಕ                    cpu_comparison_ಸಕ್ರಿಯ = ಸುಳ್ಳು;
    ಆದರೆ (ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_cpu_compare_config.enabled) {
        ಆದರೆ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ಗಂಟು, GGML_OP_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು)) {
            cpu_comparison_ಸಕ್ರಿಯ = ನಿಜ;
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for SET_ROWS operation\n");
        }
    }

    ತಾರ್ಕಿಕ kernel_ಫಲಿತಾಂಶ = ggml_et_launch_kernel(dev_ctx, kernel_name, &params, ಗಾತ್ರ(params), 0xFFFFFFFF);

    // Phase 2: Execute CPU computation and compare with ET result (after ET kernel)
    ಆದರೆ (cpu_comparison_ಸಕ್ರಿಯ) {
        ಆದರೆ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ಗಂಟು, &ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for SET_ROWS operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_ಕೊನೆ("SET_ROWS", kernel_name, ಗಂಟು);
    ಹಿಂತಿರುಗಿ kernel_ಫಲಿತಾಂಶ;
}

ತಾರ್ಕಿಕ ggml_et_op_fill(ggml_backend_et_device_context * dev_ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು) {
    ET_PERF_ಆರಂಭ();

    ggml_et_fill_params params;
    params.dst = *ಗಂಟು;
    ಸ್ಮರಣೆ_ನಕಲು(&params.c, ಗಂಟು->op_params, ಗಾತ್ರ(ದಶಮಾಂಶ));

    ತಾರ್ಕಿಕ kernel_ಫಲಿತಾಂಶ = ggml_et_launch_kernel(dev_ctx, "fill_f32", &params, ಗಾತ್ರ(params), 0xFFFFFFFF);

    ET_PERF_ಕೊನೆ("FILL", "fill_f32", ಗಂಟು);
    ಹಿಂತಿರುಗಿ kernel_ಫಲಿತಾಂಶ;
}

ತಾರ್ಕಿಕ ggml_et_op_diag(ggml_backend_et_device_context * dev_ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು) {
    ET_PERF_ಆರಂಭ();

    ಆದರೆ (!ಗಂಟು->src[0]) {
        GGML_LOG_ERROR("ET: DIAG operation missing source tensor\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ggml_et_diag_params params;
    params.src0 = *ಗಂಟು->src[0];
    params.dst  = *ಗಂಟು;

    ತಾರ್ಕಿಕ kernel_ಫಲಿತಾಂಶ = ggml_et_launch_kernel(dev_ctx, "diag_f32", &params, ಗಾತ್ರ(params), 0xFFFFFFFF);

    ET_PERF_ಕೊನೆ("DIAG", "diag_f32", ಗಂಟು);
    ಹಿಂತಿರುಗಿ kernel_ಫಲಿತಾಂಶ;
}

ತಾರ್ಕಿಕ ggml_et_op_tri(ggml_backend_et_device_context * dev_ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು) {
    ET_PERF_ಆರಂಭ();

    ಆದರೆ (!ಗಂಟು->src[0]) {
        GGML_LOG_ERROR("ET: TRI operation missing source tensor\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ggml_et_tri_params params;
    params.src0 = *ಗಂಟು->src[0];
    params.dst  = *ಗಂಟು;
    ಸ್ಮರಣೆ_ನಕಲು(&params.tri_type, ಗಂಟು->op_params, ಗಾತ್ರ(int32_t));

    ತಾರ್ಕಿಕ kernel_ಫಲಿತಾಂಶ = ggml_et_launch_kernel(dev_ctx, "tri_f32", &params, ಗಾತ್ರ(params), 0xFFFFFFFF);

    ET_PERF_ಕೊನೆ("TRI", "tri_f32", ಗಂಟು);
    ಹಿಂತಿರುಗಿ kernel_ಫಲಿತಾಂಶ;
}

ತಾರ್ಕಿಕ ggml_et_op_solve_tri(ggml_backend_et_device_context * dev_ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು) {
    ET_PERF_ಆರಂಭ();

    ಆದರೆ (!ಗಂಟು->src[0] || !ಗಂಟು->src[1]) {
        GGML_LOG_ERROR("ET: SOLVE_TRI operation missing source tensor(s)\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ggml_et_solve_tri_params params;
    params.src0 = *ಗಂಟು->src[0];  // A (lower-triangular)
    params.src1 = *ಗಂಟು->src[1];  // B (RHS)
    params.dst  = *ಗಂಟು;          // X (solution)

    ತಾರ್ಕಿಕ kernel_ಫಲಿತಾಂಶ = ggml_et_launch_kernel(dev_ctx, "solve_tri_f32", &params, ಗಾತ್ರ(params), 0xFFFFFFFF);

    ET_PERF_ಕೊನೆ("SOLVE_TRI", "solve_tri_f32", ಗಂಟು);
    ಹಿಂತಿರುಗಿ kernel_ಫಲಿತಾಂಶ;
}

ತಾರ್ಕಿಕ ggml_et_op_ಗಣ(ggml_backend_et_device_context * dev_ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು) {
    ET_PERF_ಆರಂಭ();

    ಆದರೆ (!ಗಂಟು->src[0] || !ಗಂಟು->src[1]) {
        GGML_LOG_ERROR("ET: SET operation missing source tensor(s)\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ   inplace = (ತಾರ್ಕಿಕ) ((ಸ್ಥಿರಾಂಕ int32_t *) ಗಂಟು->op_params)[4];
    ಸ್ಥಿರಾಂಕ size_t ಸ್ಥಾನಾಂತರ  = ((ಸ್ಥಿರಾಂಕ int32_t *) ಗಂಟು->op_params)[3];
    ಸ್ಥಿರಾಂಕ size_t nb1     = ((ಸ್ಥಿರಾಂಕ int32_t *) ಗಂಟು->op_params)[0];
    ಸ್ಥಿರಾಂಕ size_t nb2     = ((ಸ್ಥಿರಾಂಕ int32_t *) ಗಂಟು->op_params)[1];
    ಸ್ಥಿರಾಂಕ size_t nb3     = ((ಸ್ಥಿರಾಂಕ int32_t *) ಗಂಟು->op_params)[2];

    ಆದರೆ (!inplace) {
        GGML_LOG_ERROR("ET: SET only supports inplace (inplace=%d)\n", inplace);
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (ಗಂಟು->type != GGML_TYPE_F32 || ಗಂಟು->src[0]->type != GGML_TYPE_F32 || ಗಂಟು->src[1]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: SET only supports F32 (dst=%s src0=%s src1=%s)\n", ggml_type_name(ಗಂಟು->type),
                       ggml_type_name(ಗಂಟು->src[0]->type), ggml_type_name(ಗಂಟು->src[1]->type));
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (!ggml_are_same_shape(ಗಂಟು, ಗಂಟು->src[0])) {
        GGML_LOG_ERROR("ET: SET requires same-shape src0 and dst\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (!ggml_is_contiguous(ಗಂಟು) || !ggml_is_contiguous(ಗಂಟು->src[0]) || !ggml_is_contiguous(ಗಂಟು->src[1])) {
        GGML_LOG_ERROR("ET: SET requires contiguous dst, src0, and src1\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ggml_et_ಗಣ_params params;
    params.src1   = *ಗಂಟು->src[1];
    params.dst    = *ಗಂಟು;
    params.nb1    = (int32_t) nb1;
    params.nb2    = (int32_t) nb2;
    params.nb3    = (int32_t) nb3;
    params.ಸ್ಥಾನಾಂತರ = (int32_t) ಸ್ಥಾನಾಂತರ;

    ತಾರ್ಕಿಕ kernel_ಫಲಿತಾಂಶ = ggml_et_launch_kernel(dev_ctx, "set_f32", &params, ಗಾತ್ರ(params), 0xFFFFFFFF);

    ET_PERF_ಕೊನೆ("SET", "set_f32", ಗಂಟು);
    ಹಿಂತಿರುಗಿ kernel_ಫಲಿತಾಂಶ;
}

ತಾರ್ಕಿಕ ggml_et_op_pad(ggml_backend_et_device_context * dev_ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು) {
    ET_PERF_ಆರಂಭ();

    ಆದರೆ (!ಗಂಟು->src[0]) {
        GGML_LOG_ERROR("ET: PAD operation missing source tensor\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (ಗಂಟು->type != GGML_TYPE_F32 || ಗಂಟು->src[0]->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("ET: PAD only supports F32 (src=%s dst=%s)\n", ggml_type_name(ಗಂಟು->src[0]->type),
                       ggml_type_name(ಗಂಟು->type));
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (!ggml_is_contiguous(ಗಂಟು)) {
        GGML_LOG_ERROR("ET: PAD requires contiguous output tensor\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (ಗಂಟು->src[0]->nb[0] != ಗಾತ್ರ(ದಶಮಾಂಶ)) {
        GGML_LOG_ERROR("ET: PAD requires element-contiguous src dim0 (nb[0]=%zu)\n", (size_t) ಗಂಟು->src[0]->nb[0]);
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // Extract padding parameters from op_params
    ಸ್ಥಿರಾಂಕ int32_t * op_params = (ಸ್ಥಿರಾಂಕ int32_t *) ಗಂಟು->op_params;

    ggml_et_pad_params params;
    params.src0  = *ಗಂಟು->src[0];
    params.dst   = *ಗಂಟು;
    params.lp[0] = op_params[0];
    params.rp[0] = op_params[1];
    params.lp[1] = op_params[2];
    params.rp[1] = op_params[3];
    params.lp[2] = op_params[4];
    params.rp[2] = op_params[5];
    params.lp[3] = op_params[6];
    params.rp[3] = op_params[7];

    // v1: no dim0 padding
    ಆದರೆ (params.lp[0] != 0 || params.rp[0] != 0) {
        GGML_LOG_ERROR("ET: PAD dim0 padding not supported (lp0=%d rp0=%d)\n", params.lp[0], params.rp[0]);
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ggml_et_cpu_compare_ctx cpu_cmp_ctx;
    ತಾರ್ಕಿಕ                    cpu_comparison_ಸಕ್ರಿಯ = ಸುಳ್ಳು;
    ಆದರೆ (pad_cpu_compare_config.enabled) {
        ಆದರೆ (ggml_et_cpu_compare_init_pre(&cpu_cmp_ctx, ಗಂಟು, GGML_OP_PAD)) {
            cpu_comparison_ಸಕ್ರಿಯ = ನಿಜ;
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_LOG_WARN("ET: Failed to initialize CPU comparison for PAD operation\n");
        }
    }

    ತಾರ್ಕಿಕ kernel_ಫಲಿತಾಂಶ = ggml_et_launch_kernel(dev_ctx, "pad_f32", &params, ಗಾತ್ರ(params), 0xFFFFFFFF);

    ಆದರೆ (cpu_comparison_ಸಕ್ರಿಯ) {
        ಆದರೆ (!ggml_et_cpu_compare_compute_and_check(&cpu_cmp_ctx, ಗಂಟು, &pad_cpu_compare_config)) {
            GGML_LOG_WARN("ET: CPU comparison failed for PAD operation\n");
        }
        ggml_et_cpu_compare_free(&cpu_cmp_ctx);
    }

    ET_PERF_ಕೊನೆ("PAD", "pad_f32", ಗಂಟು);
    ಹಿಂತಿರುಗಿ kernel_ಫಲಿತಾಂಶ;
}
