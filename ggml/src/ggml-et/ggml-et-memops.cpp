#અન્તર્ભાવ "ggml-et-memops.h"

#અન્તર્ભાવ "ggml-et-kernels.h"
#અન્તર્ભાવ "ggml-impl.h"

// Kernel parameter structure for memset operation
સંરચના memset_params {
    uint32_t op_type;  // GGML_ET_MEMOP_MEMSET
    uint32_t value;    // Value to set (extended to uint32_t for alignment)
    શૂન્ય *   dst_ptr;  // Destination device pointer
    size_t   size;     // Number of bytes to set
};

તાર્કિક ggml_et_memset(ggml_backend_et_device_context * dev_ctx, શૂન્ય * dst_ptr, uint8_t value, size_t size) {
    જો (!dev_ctx || !dst_ptr || size == 0) {
        GGML_LOG_ERROR("ET: Invalid memset parameters\n");
        પાછા ખોટું;
    }

    // Prepare kernel parameters
    memset_params params;
    params.op_type = GGML_ET_MEMOP_MEMSET;
    params.value   = value;
    params.dst_ptr = dst_ptr;
    params.size    = size;

    // Launch memops kernel (will lazy-load if not already loaded)
    તાર્કિક success = ggml_et_launch_kernel(dev_ctx, "memops", &params, કદ(params));

    જો (!success) {
        GGML_LOG_ERROR("ET: memset kernel launch failed\n");
        પાછા ખોટું;
    }

    પાછા સાચું;
}
