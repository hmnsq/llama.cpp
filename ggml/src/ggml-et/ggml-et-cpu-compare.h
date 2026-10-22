#సూచన once

#అన్తర్భావ "ggml-cpu.h"
#అన్తర్భావ "ggml-et-common.h"
#అన్తర్భావ "ggml-impl.h"

// Configuration for CPU comparison
నిర్మాణం ggml_et_cpu_compare_config {
    bool   enabled;           // Whether to enable CPU comparison
    bool   use_cpu_ఫలితం;    // Whether to replace ET result with CPU result
    bool   log_differences;   // Whether to log detailed element differences
    దశాంశం  tolerance;         // Relative tolerance for comparison (default: 1e-5f)
    size_t max_log_elements;  // Maximum number of elements to log (default: 10)
};

// Default configuration
స్థిర స్థిరరాశి ggml_et_cpu_compare_config ggml_et_cpu_compare_default_config = {
    /* .enabled = */ false,
    /* .use_cpu_result = */ false,
    /* .log_differences = */ true,
    /* .tolerance = */ 1e-5f,
    /* .max_log_elements = */ 10
};

// CPU comparison context for a single operation
నిర్మాణం ggml_et_cpu_compare_ctx {
    ggml_backend_t cpu_backend;
    ggml_context * ggml_ctx;
    ggml_tensor *  cpu_src0;
    ggml_tensor *  cpu_src1;
    ggml_tensor *  cpu_src2;
    ggml_tensor *  cpu_dst;
    ggml_cgraph *  cpu_లేఖాచిత్రం;
    శూన్య *         cpu_src0_దత్తాంశం;
    శూన్య *         cpu_src1_దత్తాంశం;
    శూన్య *         cpu_src2_దత్తాంశం;
    శూన్య *         cpu_dst_దత్తాంశం;
    శూన్య *         et_dst_దత్తాంశం;
    size_t         src0_కొలత;
    size_t         src1_కొలత;
    size_t         src2_కొలత;
    size_t         dst_కొలత;
};

// Phase 1: Initialize CPU comparison context and copy source buffers (call before ET kernel)
bool ggml_et_cpu_compare_init_pre(ggml_et_cpu_compare_ctx * ctx, స్థిరరాశి ggml_tensor * కణుపు, ggml_op op);

// Phase 2: Execute CPU computation and compare with ET result (call after ET kernel)
bool ggml_et_cpu_compare_compute_and_check(ggml_et_cpu_compare_ctx *          ctx,
                                           స్థిరరాశి ggml_tensor *                కణుపు,
                                           స్థిరరాశి ggml_et_cpu_compare_config * config);

// Free CPU comparison context resources
శూన్య ggml_et_cpu_compare_free(ggml_et_cpu_compare_ctx * ctx);
