#pragma once

#ಅನ್ತರ್ಭಾವ "ggml-cpu.h"
#ಅನ್ತರ್ಭಾವ "ggml-et-common.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"

// Configuration for CPU comparison
ಸಂರಚನಾ ggml_et_cpu_compare_config {
    bool   enabled;           // Whether to enable CPU comparison
    bool   use_cpu_result;    // Whether to replace ET result with CPU result
    bool   log_differences;   // Whether to log detailed element differences
    ಭಿನ್ನ  tolerance;         // Relative tolerance for comparison (default: 1e-5f)
    size_t max_log_elements;  // Maximum number of elements to log (default: 10)
};

// Default configuration
ಸ್ಥಿರ ಧ್ರುವ ggml_et_cpu_compare_config ggml_et_cpu_compare_default_config = {
    /* .enabled = */ false,
    /* .use_cpu_result = */ false,
    /* .log_differences = */ true,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 10
};

// CPU comparison context for a single operation
ಸಂರಚನಾ ggml_et_cpu_compare_ctx {
    ggml_backend_t cpu_backend;
    ggml_context * ggml_ctx;
    ggml_tensor *  cpu_src0;
    ggml_tensor *  cpu_src1;
    ggml_tensor *  cpu_src2;
    ggml_tensor *  cpu_dst;
    ggml_cgraph *  cpu_graph;
    ಶೂನ್ಯ *         cpu_src0_data;
    ಶೂನ್ಯ *         cpu_src1_data;
    ಶೂನ್ಯ *         cpu_src2_data;
    ಶೂನ್ಯ *         cpu_dst_data;
    ಶೂನ್ಯ *         et_dst_data;
    size_t         src0_size;
    size_t         src1_size;
    size_t         src2_size;
    size_t         dst_size;
};

// Phase 1: Initialize CPU comparison context and copy source buffers (call before ET kernel)
bool ggml_et_cpu_compare_init_pre(ggml_et_cpu_compare_ctx * ctx, ಧ್ರುವ ggml_tensor * ಗಂಟು, ggml_op op);

// Phase 2: Execute CPU computation and compare with ET result (call after ET kernel)
bool ggml_et_cpu_compare_compute_and_check(ggml_et_cpu_compare_ctx *          ctx,
                                           ಧ್ರುವ ggml_tensor *                ಗಂಟು,
                                           ಧ್ರುವ ggml_et_cpu_compare_config * config);

// Free CPU comparison context resources
ಶೂನ್ಯ ggml_et_cpu_compare_free(ggml_et_cpu_compare_ctx * ctx);
