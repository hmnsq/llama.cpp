#ہدایت once

#شامل "ggml-cpu.h"
#شامل "ggml-et-common.h"
#شامل "ggml-impl.h"

// Configuration for CPU comparison
ساخت ggml_et_cpu_compare_config {
    bool   enabled;           // Whether to enable CPU comparison
    bool   use_cpu_نتیجہ;    // Whether to replace ET result with CPU result
    bool   log_differences;   // Whether to log detailed element differences
    اعشاری  tolerance;         // Relative tolerance for comparison (default: 1e-5f)
    size_t max_log_elements;  // Maximum number of elements to log (default: 10)
};

// Default configuration
ساکن ثابت ggml_et_cpu_compare_config ggml_et_cpu_compare_default_config = {
    /* .enabled = */ false,
    /* .use_cpu_result = */ false,
    /* .log_differences = */ true,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 10
};

// CPU comparison context for a single operation
ساخت ggml_et_cpu_compare_ctx {
    ggml_backend_t cpu_backend;
    ggml_context * ggml_ctx;
    ggml_tensor *  cpu_src0;
    ggml_tensor *  cpu_src1;
    ggml_tensor *  cpu_src2;
    ggml_tensor *  cpu_dst;
    ggml_cgraph *  cpu_گراف;
    خالی *         cpu_src0_ڈیٹا;
    خالی *         cpu_src1_ڈیٹا;
    خالی *         cpu_src2_ڈیٹا;
    خالی *         cpu_dst_ڈیٹا;
    خالی *         et_dst_ڈیٹا;
    size_t         src0_سائز;
    size_t         src1_سائز;
    size_t         src2_سائز;
    size_t         dst_سائز;
};

// Phase 1: Initialize CPU comparison context and copy source buffers (call before ET kernel)
bool ggml_et_cpu_compare_init_pre(ggml_et_cpu_compare_ctx * ctx, ثابت ggml_tensor * گرہ, ggml_op op);

// Phase 2: Execute CPU computation and compare with ET result (call after ET kernel)
bool ggml_et_cpu_compare_compute_and_check(ggml_et_cpu_compare_ctx *          ctx,
                                           ثابت ggml_tensor *                گرہ,
                                           ثابت ggml_et_cpu_compare_config * config);

// Free CPU comparison context resources
خالی ggml_et_cpu_compare_free(ggml_et_cpu_compare_ctx * ctx);
