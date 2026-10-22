#निर्देशः once

#अन्तर्भाव "ggml-cpu.h"
#अन्तर्भाव "ggml-et-common.h"
#अन्तर्भाव "ggml-impl.h"

// Configuration for CPU comparison
संरचना ggml_et_cpu_compare_config {
    bool   enabled;           // Whether to enable CPU comparison
    bool   use_cpu_परिणाम;    // Whether to replace ET result with CPU result
    bool   log_differences;   // Whether to log detailed element differences
    भिन्न  tolerance;         // Relative tolerance for comparison (default: 1e-5f)
    size_t max_log_elements;  // Maximum number of elements to log (default: 10)
};

// Default configuration
स्थिर ध्रुव ggml_et_cpu_compare_config ggml_et_cpu_compare_default_config = {
    /* .enabled = */ false,
    /* .use_cpu_result = */ false,
    /* .log_differences = */ true,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 10
};

// CPU comparison context for a single operation
संरचना ggml_et_cpu_compare_ctx {
    ggml_backend_t cpu_backend;
    ggml_context * ggml_ctx;
    ggml_tensor *  cpu_src0;
    ggml_tensor *  cpu_src1;
    ggml_tensor *  cpu_src2;
    ggml_tensor *  cpu_dst;
    ggml_cgraph *  cpu_graph;
    शून्य *         cpu_src0_दत्त;
    शून्य *         cpu_src1_दत्त;
    शून्य *         cpu_src2_दत्त;
    शून्य *         cpu_dst_दत्त;
    शून्य *         et_dst_दत्त;
    size_t         src0_आकार;
    size_t         src1_आकार;
    size_t         src2_आकार;
    size_t         dst_आकार;
};

// Phase 1: Initialize CPU comparison context and copy source buffers (call before ET kernel)
bool ggml_et_cpu_compare_init_pre(ggml_et_cpu_compare_ctx * ctx, ध्रुव ggml_tensor * ग्रन्थिः, ggml_op op);

// Phase 2: Execute CPU computation and compare with ET result (call after ET kernel)
bool ggml_et_cpu_compare_compute_and_check(ggml_et_cpu_compare_ctx *          ctx,
                                           ध्रुव ggml_tensor *                ग्रन्थिः,
                                           ध्रुव ggml_et_cpu_compare_config * config);

// Free CPU comparison context resources
शून्य ggml_et_cpu_compare_free(ggml_et_cpu_compare_ctx * ctx);
