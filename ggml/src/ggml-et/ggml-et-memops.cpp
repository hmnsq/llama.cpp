#অন্তর্ভাব "ggml-et-memops.h"

#অন্তর্ভাব "ggml-et-kernels.h"
#অন্তর্ভাব "ggml-impl.h"

// Kernel parameter structure for memset operation
কাঠামো memset_params {
    uint32_t op_type;  // GGML_ET_MEMOP_MEMSET
    uint32_t value;    // Value to set (extended to uint32_t for alignment)
    শূন্য *   dst_ptr;  // Destination device pointer
    size_t   size;     // Number of bytes to set
};

যৌক্তিক ggml_et_memset(ggml_backend_et_device_context * dev_ctx, শূন্য * dst_ptr, uint8_t value, size_t size) {
    যদি (!dev_ctx || !dst_ptr || size == 0) {
        GGML_LOG_ERROR("ET: Invalid memset parameters\n");
        ফেরত মিথ্যা;
    }

    // Prepare kernel parameters
    memset_params params;
    params.op_type = GGML_ET_MEMOP_MEMSET;
    params.value   = value;
    params.dst_ptr = dst_ptr;
    params.size    = size;

    // Launch memops kernel (will lazy-load if not already loaded)
    যৌক্তিক success = ggml_et_launch_kernel(dev_ctx, "memops", &params, আকার(params));

    যদি (!success) {
        GGML_LOG_ERROR("ET: memset kernel launch failed\n");
        ফেরত মিথ্যা;
    }

    ফেরত সত্য;
}
