#अन्तर्भाव "ggml-et-memops.h"

#अन्तर्भाव "ggml-et-kernels.h"
#अन्तर्भाव "ggml-impl.h"

// Kernel parameter structure for memset operation
संरचना memset_params {
    uint32_t op_type;  // GGML_ET_MEMOP_MEMSET
    uint32_t value;    // Value to set (extended to uint32_t for alignment)
    शून्य *   dst_ptr;  // Destination device pointer
    माप_प्रकार   size;     // Number of bytes to set
};

तार्किक ggml_et_memset(ggml_backend_et_device_context * dev_ctx, शून्य * dst_ptr, uint8_t value, माप_प्रकार size) {
    जर (!dev_ctx || !dst_ptr || size == 0) {
        GGML_LOG_ERROR("ET: Invalid memset parameters\n");
        परत असत्य;
    }

    // Prepare kernel parameters
    memset_params params;
    params.op_type = GGML_ET_MEMOP_MEMSET;
    params.value   = value;
    params.dst_ptr = dst_ptr;
    params.size    = size;

    // Launch memops kernel (will lazy-load if not already loaded)
    तार्किक success = ggml_et_launch_kernel(dev_ctx, "memops", &params, आकार(params));

    जर (!success) {
        GGML_LOG_ERROR("ET: memset kernel launch failed\n");
        परत असत्य;
    }

    परत सत्य;
}
