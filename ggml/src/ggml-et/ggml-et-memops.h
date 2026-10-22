#ہدایت once

#شامل "ggml-et-common.h"

#شامل <cstddef>
#شامل <cstdint>

// Memory operations using device kernel (memops.elf)
// Single kernel handles multiple operations via operation identifier

// Operation identifiers for memops kernel
شمار ggml_et_memop_type : uint32_t {
    GGML_ET_MEMOP_MEMSET = 0,
};

// Memset operation: fill device memory with a value
// Returns true on success, false on failure
bool ggml_et_memset(ggml_backend_et_device_context * dev_ctx, خالی * dst_ptr, uint8_t قدر, size_t سائز);
