#समावेश "ggml-et-memops.h"

#समावेश "ggml-et-kernels.h"
#समावेश "ggml-impl.h"

// Kernel parameter structure for memset operation
काष्ठा memset_params {
    uint32_t op_type;  // GGML_ET_MEMOP_MEMSET
    uint32_t value;    // Value to set (extended to uint32_t for alignment)
    व्योम *   dst_ptr;  // Destination device pointer
    माप_प्रकार   size;     // Number of bytes to set
};

bool ggml_et_memset(ggml_backend_et_device_context * dev_ctx, व्योम * dst_ptr, uint8_t value, माप_प्रकार size) {
    अगर (!dev_ctx || !dst_ptr || size == 0) {
        GGML_LOG_ERROR("ET: Invalid memset parameters\n");
        वापस false;
    }

    // Prepare kernel parameters
    memset_params params;
    params.op_type = GGML_ET_MEMOP_MEMSET;
    params.value   = value;
    params.dst_ptr = dst_ptr;
    params.size    = size;

    // Launch memops kernel (will lazy-load if not already loaded)
    bool success = ggml_et_launch_kernel(dev_ctx, "memops", &params, माप(params));

    अगर (!success) {
        GGML_LOG_ERROR("ET: memset kernel launch failed\n");
        वापस false;
    }

    वापस true;
}
