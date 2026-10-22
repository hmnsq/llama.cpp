#pragma once

#অন্তর্ভাৱ "ggml-cpu.h"
#অন্তর্ভাৱ "ggml-et-common.h"
#অন্তর্ভাৱ "ggml-impl.h"

// Configuration for CPU comparison
সংরচনা ggml_et_cpu_compare_config {
    bool   enabled;           // Whether to enable CPU comparison
    bool   use_cpu_result;    // Whether to replace ET result with CPU result
    bool   log_differences;   // Whether to log detailed element differences
    ভিন্ন  tolerance;         // Relative tolerance for comparison (default: 1e-5f)
    size_t max_log_elements;  // Maximum number of elements to log (default: 10)
};

// Default configuration
স্থির ধ্রুৱ ggml_et_cpu_compare_config ggml_et_cpu_compare_default_config = {
    /* .enabled = */ false,
    /* .use_cpu_result = */ false,
    /* .log_differences = */ true,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 10
};

// CPU comparison context for a single operation
সংরচনা ggml_et_cpu_compare_ctx {
    ggml_backend_t cpu_backend;
    ggml_context * ggml_ctx;
    ggml_tensor *  cpu_src0;
    ggml_tensor *  cpu_src1;
    ggml_tensor *  cpu_src2;
    ggml_tensor *  cpu_dst;
    ggml_cgraph *  cpu_graph;
    শূন্য *         cpu_src0_data;
    শূন্য *         cpu_src1_data;
    শূন্য *         cpu_src2_data;
    শূন্য *         cpu_dst_data;
    শূন্য *         et_dst_data;
    size_t         src0_size;
    size_t         src1_size;
    size_t         src2_size;
    size_t         dst_size;
};

// Phase 1: Initialize CPU comparison context and copy source buffers (call before ET kernel)
bool ggml_et_cpu_compare_init_pre(ggml_et_cpu_compare_ctx * ctx, ধ্রুৱ ggml_tensor * গ্ৰন্থি, ggml_op op);

// Phase 2: Execute CPU computation and compare with ET result (call after ET kernel)
bool ggml_et_cpu_compare_compute_and_check(ggml_et_cpu_compare_ctx *          ctx,
                                           ধ্রুৱ ggml_tensor *                গ্ৰন্থি,
                                           ধ্রুৱ ggml_et_cpu_compare_config * config);

// Free CPU comparison context resources
শূন্য ggml_et_cpu_compare_free(ggml_et_cpu_compare_ctx * ctx);
