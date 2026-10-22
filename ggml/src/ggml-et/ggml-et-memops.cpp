#அந்தர்பாவ "ggml-et-memops.h"

#அந்தர்பாவ "ggml-et-kernels.h"
#அந்தர்பாவ "ggml-impl.h"

// Kernel parameter structure for memset operation
கட்டமைப்பு memset_params {
    uint32_t op_type;  // GGML_ET_MEMOP_MEMSET
    uint32_t value;    // Value to set (extended to uint32_t for alignment)
    வெறுமை *   dst_ptr;  // Destination device pointer
    size_t   size;     // Number of bytes to set
};

தருக்க ggml_et_memset(ggml_backend_et_device_context * dev_ctx, வெறுமை * dst_ptr, uint8_t value, size_t size) {
    எனில் (!dev_ctx || !dst_ptr || size == 0) {
        GGML_LOG_ERROR("ET: Invalid memset parameters\n");
        திரும்பு பொய்;
    }

    // Prepare kernel parameters
    memset_params params;
    params.op_type = GGML_ET_MEMOP_MEMSET;
    params.value   = value;
    params.dst_ptr = dst_ptr;
    params.size    = size;

    // Launch memops kernel (will lazy-load if not already loaded)
    தருக்க success = ggml_et_launch_kernel(dev_ctx, "memops", &params, அளவு(params));

    எனில் (!success) {
        GGML_LOG_ERROR("ET: memset kernel launch failed\n");
        திரும்பு பொய்;
    }

    திரும்பு உண்மை;
}
