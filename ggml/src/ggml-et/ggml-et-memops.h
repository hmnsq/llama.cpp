#pragma once

#شامل "ggml-et-common.h"

#شامل <cstddef>
#شامل <cstdint>

// Memory operations using device kernel (memops.elf)
// Single kernel handles multiple operations via operation identifier

// Operation identifiers for memops kernel
شمارش ggml_et_memop_type : uint32_t {
    GGML_ET_MEMOP_MEMSET = 0,
};

// Memset operation: fill device memory with a value
// Returns true on success, false on failure
bool ggml_et_memset(ggml_backend_et_device_context * dev_ctx, تهی * dst_ptr, uint8_t مقدار, size_t size);
