#आशय once

#समावेश "ggml-cpu.h"
#समावेश "ggml-et-common.h"
#समावेश "ggml-impl.h"

// Configuration for CPU comparison
काष्ठा ggml_et_cpu_compare_config {
    bool   enabled;           // Whether to enable CPU comparison
    bool   use_cpu_परिणाम;    // Whether to replace ET result with CPU result
    bool   log_differences;   // Whether to log detailed element differences
    भग्न  tolerance;         // Relative tolerance for comparison (default: 1e-5f)
    माप_प्रकार max_log_elements;  // Maximum number of elements to log (default: 10)
};

// Default configuration
अटल स्थिर ggml_et_cpu_compare_config ggml_et_cpu_compare_default_config = {
    /* .enabled = */ false,
    /* .use_cpu_result = */ false,
    /* .log_differences = */ true,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 10
};

// CPU comparison context for a single operation
काष्ठा ggml_et_cpu_compare_ctx {
    ggml_backend_t cpu_backend;
    ggml_context * ggml_ctx;
    ggml_tensor *  cpu_src0;
    ggml_tensor *  cpu_src1;
    ggml_tensor *  cpu_src2;
    ggml_tensor *  cpu_dst;
    ggml_cgraph *  cpu_graph;
    व्योम *         cpu_src0_तथ्य;
    व्योम *         cpu_src1_तथ्य;
    व्योम *         cpu_src2_तथ्य;
    व्योम *         cpu_dst_तथ्य;
    व्योम *         et_dst_तथ्य;
    माप_प्रकार         src0_आकार;
    माप_प्रकार         src1_आकार;
    माप_प्रकार         src2_आकार;
    माप_प्रकार         dst_आकार;
};

// Phase 1: Initialize CPU comparison context and copy source buffers (call before ET kernel)
bool ggml_et_cpu_compare_init_pre(ggml_et_cpu_compare_ctx * ctx, स्थिर ggml_tensor * गाँठ, ggml_op op);

// Phase 2: Execute CPU computation and compare with ET result (call after ET kernel)
bool ggml_et_cpu_compare_compute_and_check(ggml_et_cpu_compare_ctx *          ctx,
                                           स्थिर ggml_tensor *                गाँठ,
                                           स्थिर ggml_et_cpu_compare_config * config);

// Free CPU comparison context resources
व्योम ggml_et_cpu_compare_free(ggml_et_cpu_compare_ctx * ctx);
