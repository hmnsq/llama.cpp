#pragma once

#அந்தர்பாவ "ggml-et-common.h"

#அந்தர்பாவ <cstddef>
#அந்தர்பாவ <cstdint>

// Memory operations using device kernel (memops.elf)
// Single kernel handles multiple operations via operation identifier

// Operation identifiers for memops kernel
எண்ணிக்கை ggml_et_memop_type : uint32_t {
    GGML_ET_MEMOP_MEMSET = 0,
};

// Memset operation: fill device memory with a value
// Returns true on success, false on failure
bool ggml_et_memset(ggml_backend_et_device_context * dev_ctx, வெறுமை * dst_ptr, uint8_t மதிப்பு, size_t size);
