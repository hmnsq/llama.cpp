#ଅନ୍ତର୍ଭାଵ "ggml-et-memops.h"

#ଅନ୍ତର୍ଭାଵ "ggml-et-kernels.h"
#ଅନ୍ତର୍ଭାଵ "ggml-impl.h"

// Kernel parameter structure for memset operation
ସଂରଚନା memset_params {
    uint32_t op_type;  // GGML_ET_MEMOP_MEMSET
    uint32_t value;    // Value to set (extended to uint32_t for alignment)
    ଶୂନ୍ୟ *   dst_ptr;  // Destination device pointer
    size_t   size;     // Number of bytes to set
};

ତାର୍କିକ ggml_et_memset(ggml_backend_et_device_context * dev_ctx, ଶୂନ୍ୟ * dst_ptr, uint8_t value, size_t size) {
    ଯଦି (!dev_ctx || !dst_ptr || size == 0) {
        GGML_LOG_ERROR("ET: Invalid memset parameters\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // Prepare kernel parameters
    memset_params params;
    params.op_type = GGML_ET_MEMOP_MEMSET;
    params.value   = value;
    params.dst_ptr = dst_ptr;
    params.size    = size;

    // Launch memops kernel (will lazy-load if not already loaded)
    ତାର୍କିକ success = ggml_et_launch_kernel(dev_ctx, "memops", &params, ଆକାର(params));

    ଯଦି (!success) {
        GGML_LOG_ERROR("ET: memset kernel launch failed\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଫେରନ୍ତୁ ସତ୍ୟ;
}
