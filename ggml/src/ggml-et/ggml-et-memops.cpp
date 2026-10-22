#شامل "ggml-et-memops.h"

#شامل "ggml-et-kernels.h"
#شامل "ggml-impl.h"

// Kernel parameter structure for memset operation
ساختار memset_params {
    uint32_t op_type;  // GGML_ET_MEMOP_MEMSET
    uint32_t value;    // Value to set (extended to uint32_t for alignment)
    تهی *   dst_ptr;  // Destination device pointer
    size_t   size;     // Number of bytes to set
};

منطقی ggml_et_memset(ggml_backend_et_device_context * dev_ctx, تهی * dst_ptr, uint8_t value, size_t size) {
    اگر (!dev_ctx || !dst_ptr || size == 0) {
        GGML_LOG_ERROR("ET: Invalid memset parameters\n");
        بازگشت نادرست;
    }

    // Prepare kernel parameters
    memset_params params;
    params.op_type = GGML_ET_MEMOP_MEMSET;
    params.value   = value;
    params.dst_ptr = dst_ptr;
    params.size    = size;

    // Launch memops kernel (will lazy-load if not already loaded)
    منطقی success = ggml_et_launch_kernel(dev_ctx, "memops", &params, اندازه(params));

    اگر (!success) {
        GGML_LOG_ERROR("ET: memset kernel launch failed\n");
        بازگشت نادرست;
    }

    بازگشت درست;
}
