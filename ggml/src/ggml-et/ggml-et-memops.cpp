#അന്തര്ഭാവ "ggml-et-memops.h"

#അന്തര്ഭാവ "ggml-et-kernels.h"
#അന്തര്ഭാവ "ggml-impl.h"

// Kernel parameter structure for memset operation
ഘടന memset_params {
    uint32_t op_type;  // GGML_ET_MEMOP_MEMSET
    uint32_t value;    // Value to set (extended to uint32_t for alignment)
    ശൂന്യം *   dst_ptr;  // Destination device pointer
    size_t   size;     // Number of bytes to set
};

തർക്കം ggml_et_memset(ggml_backend_et_device_context * dev_ctx, ശൂന്യം * dst_ptr, uint8_t value, size_t size) {
    എങ്കിൽ (!dev_ctx || !dst_ptr || size == 0) {
        GGML_LOG_ERROR("ET: Invalid memset parameters\n");
        തിരികെ അസത്യം;
    }

    // Prepare kernel parameters
    memset_params params;
    params.op_type = GGML_ET_MEMOP_MEMSET;
    params.value   = value;
    params.dst_ptr = dst_ptr;
    params.size    = size;

    // Launch memops kernel (will lazy-load if not already loaded)
    തർക്കം success = ggml_et_launch_kernel(dev_ctx, "memops", &params, വലുപ്പം(params));

    എങ്കിൽ (!success) {
        GGML_LOG_ERROR("ET: memset kernel launch failed\n");
        തിരികെ അസത്യം;
    }

    തിരികെ സത്യം;
}
