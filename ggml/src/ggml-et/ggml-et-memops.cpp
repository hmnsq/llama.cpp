#ضمن "ggml-et-memops.h"

#ضمن "ggml-et-kernels.h"
#ضمن "ggml-impl.h"

// Kernel parameter structure for memset operation
بنية memset_params {
    uint32_t op_type;  // GGML_ET_MEMOP_MEMSET
    uint32_t value;    // Value to set (extended to uint32_t for alignment)
    فراغ *   dst_ptr;  // Destination device pointer
    size_t   size;     // Number of bytes to set
};

منطقي ggml_et_memset(ggml_backend_et_device_context * dev_ctx, فراغ * dst_ptr, uint8_t value, size_t size) {
    ܐܢ (!dev_ctx || !dst_ptr || size == 0) {
        GGML_LOG_ERROR("ET: Invalid memset parameters\n");
        ܦܢܐ ܕܓܠܐ;
    }

    // Prepare kernel parameters
    memset_params params;
    params.op_type = GGML_ET_MEMOP_MEMSET;
    params.value   = value;
    params.dst_ptr = dst_ptr;
    params.size    = size;

    // Launch memops kernel (will lazy-load if not already loaded)
    منطقي success = ggml_et_launch_kernel(dev_ctx, "memops", &params, حجم(params));

    ܐܢ (!success) {
        GGML_LOG_ERROR("ET: memset kernel launch failed\n");
        ܦܢܐ ܕܓܠܐ;
    }

    ܦܢܐ ܫܪܪܐ;
}
