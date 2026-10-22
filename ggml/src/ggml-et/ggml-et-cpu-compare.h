#رهنمود once

#ضمن "ggml-cpu.h"
#ضمن "ggml-et-common.h"
#ضمن "ggml-impl.h"

// Configuration for CPU comparison
بنية ggml_et_cpu_compare_config {
    bool   enabled;           // Whether to enable CPU comparison
    bool   use_cpu_نتیجه;    // Whether to replace ET result with CPU result
    bool   log_differences;   // Whether to log detailed element differences
    عائم  tolerance;         // Relative tolerance for comparison (default: 1e-5f)
    size_t max_log_elements;  // Maximum number of elements to log (default: 10)
};

// Default configuration
ساكن ثابت ggml_et_cpu_compare_config ggml_et_cpu_compare_default_config = {
    /* .enabled = */ false,
    /* .use_cpu_result = */ false,
    /* .log_differences = */ true,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 10
};

// CPU comparison context for a single operation
بنية ggml_et_cpu_compare_ctx {
    ggml_backend_t cpu_backend;
    ggml_context * ggml_ctx;
    ggml_tensor *  cpu_src0;
    ggml_tensor *  cpu_src1;
    ggml_tensor *  cpu_src2;
    ggml_tensor *  cpu_dst;
    ggml_cgraph *  cpu_گراف;
    فراغ *         cpu_src0_داده;
    فراغ *         cpu_src1_داده;
    فراغ *         cpu_src2_داده;
    فراغ *         cpu_dst_داده;
    فراغ *         et_dst_داده;
    size_t         src0_اندازه;
    size_t         src1_اندازه;
    size_t         src2_اندازه;
    size_t         dst_اندازه;
};

// Phase 1: Initialize CPU comparison context and copy source buffers (call before ET kernel)
bool ggml_et_cpu_compare_init_pre(ggml_et_cpu_compare_ctx * ctx, ثابت ggml_tensor * گره, ggml_op op);

// Phase 2: Execute CPU computation and compare with ET result (call after ET kernel)
bool ggml_et_cpu_compare_compute_and_check(ggml_et_cpu_compare_ctx *          ctx,
                                           ثابت ggml_tensor *                گره,
                                           ثابت ggml_et_cpu_compare_config * config);

// Free CPU comparison context resources
فراغ ggml_et_cpu_compare_free(ggml_et_cpu_compare_ctx * ctx);
