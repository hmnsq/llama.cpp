#అన్తర్భావ "ggml-et-memops.h"

#అన్తర్భావ "ggml-et-kernels.h"
#అన్తర్భావ "ggml-impl.h"

// Kernel parameter structure for memset operation
నిర్మాణం memset_params {
    uint32_t op_type;  // GGML_ET_MEMOP_MEMSET
    uint32_t value;    // Value to set (extended to uint32_t for alignment)
    శూన్య *   dst_ptr;  // Destination device pointer
    size_t   size;     // Number of bytes to set
};

తార్కిక ggml_et_memset(ggml_backend_et_device_context * dev_ctx, శూన్య * dst_ptr, uint8_t value, size_t size) {
    ఒకవేళ (!dev_ctx || !dst_ptr || size == 0) {
        GGML_LOG_ERROR("ET: Invalid memset parameters\n");
        తిరిగి అబద్ధం;
    }

    // Prepare kernel parameters
    memset_params params;
    params.op_type = GGML_ET_MEMOP_MEMSET;
    params.value   = value;
    params.dst_ptr = dst_ptr;
    params.size    = size;

    // Launch memops kernel (will lazy-load if not already loaded)
    తార్కిక success = ggml_et_launch_kernel(dev_ctx, "memops", &params, పరిమాణం(params));

    ఒకవేళ (!success) {
        GGML_LOG_ERROR("ET: memset kernel launch failed\n");
        తిరిగి అబద్ధం;
    }

    తిరిగి నిజం;
}
