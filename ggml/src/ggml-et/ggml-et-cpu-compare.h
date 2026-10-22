#ਹਦਾਇਤ once

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-cpu.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-et-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"

// Configuration for CPU comparison
ਬਣਤਰ ggml_et_cpu_compare_config {
    bool   enabled;           // Whether to enable CPU comparison
    bool   use_cpu_ਨਤੀਜਾ;    // Whether to replace ET result with CPU result
    bool   log_differences;   // Whether to log detailed element differences
    ਦਸ਼ਮਲਵ  tolerance;         // Relative tolerance for comparison (default: 1e-5f)
    size_t max_log_elements;  // Maximum number of elements to log (default: 10)
};

// Default configuration
ਸਥਿਰ ਅਚਲ ggml_et_cpu_compare_config ggml_et_cpu_compare_default_config = {
    /* .enabled = */ false,
    /* .use_cpu_result = */ false,
    /* .log_differences = */ true,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 10
};

// CPU comparison context for a single operation
ਬਣਤਰ ggml_et_cpu_compare_ctx {
    ggml_backend_t cpu_backend;
    ggml_context * ggml_ctx;
    ggml_tensor *  cpu_src0;
    ggml_tensor *  cpu_src1;
    ggml_tensor *  cpu_src2;
    ggml_tensor *  cpu_dst;
    ggml_cgraph *  cpu_ਗ੍ਰਾਫ;
    ਖਾਲੀ *         cpu_src0_ਅੰਕੜੇ;
    ਖਾਲੀ *         cpu_src1_ਅੰਕੜੇ;
    ਖਾਲੀ *         cpu_src2_ਅੰਕੜੇ;
    ਖਾਲੀ *         cpu_dst_ਅੰਕੜੇ;
    ਖਾਲੀ *         et_dst_ਅੰਕੜੇ;
    size_t         src0_ਮਾਪ;
    size_t         src1_ਮਾਪ;
    size_t         src2_ਮਾਪ;
    size_t         dst_ਮਾਪ;
};

// Phase 1: Initialize CPU comparison context and copy source buffers (call before ET kernel)
bool ggml_et_cpu_compare_init_pre(ggml_et_cpu_compare_ctx * ctx, ਅਚਲ ggml_tensor * ਗੰਢ, ggml_op op);

// Phase 2: Execute CPU computation and compare with ET result (call after ET kernel)
bool ggml_et_cpu_compare_compute_and_check(ggml_et_cpu_compare_ctx *          ctx,
                                           ਅਚਲ ggml_tensor *                ਗੰਢ,
                                           ਅਚਲ ggml_et_cpu_compare_config * config);

// Free CPU comparison context resources
ਖਾਲੀ ggml_et_cpu_compare_free(ggml_et_cpu_compare_ctx * ctx);
