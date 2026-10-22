#অন্তর্ভাব "ggml-et-memops.h"

#অন্তর্ভাব "ggml-et-kernels.h"
#অন্তর্ভাব "ggml-impl.h"

// Kernel parameter structure for memset operation
সংরচনা memset_params {
    uint32_t op_type;  // GGML_ET_MEMOP_MEMSET
    uint32_t value;    // Value to set (extended to uint32_t for alignment)
    শূন্য *   dst_ptr;  // Destination device pointer
    size_t   size;     // Number of bytes to set
};

তার্কিক ggml_et_memset(ggml_backend_et_device_context * dev_ctx, শূন্য * dst_ptr, uint8_t value, size_t size) {
    যদি (!dev_ctx || !dst_ptr || size == 0) {
        GGML_LOG_ERROR("ET: Invalid memset parameters\n");
        প্রত্যাগম অসত্য;
    }

    // Prepare kernel parameters
    memset_params params;
    params.op_type = GGML_ET_MEMOP_MEMSET;
    params.value   = value;
    params.dst_ptr = dst_ptr;
    params.size    = size;

    // Launch memops kernel (will lazy-load if not already loaded)
    তার্কিক success = ggml_et_launch_kernel(dev_ctx, "memops", &params, পরিমাণ(params));

    যদি (!success) {
        GGML_LOG_ERROR("ET: memset kernel launch failed\n");
        প্রত্যাগম অসত্য;
    }

    প্রত্যাগম সত্য;
}
