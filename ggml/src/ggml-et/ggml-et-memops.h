#pragma once

#অন্তর্ভাৱ "ggml-et-common.h"

#অন্তর্ভাৱ <cstddef>
#অন্তর্ভাৱ <cstdint>

// Memory operations using device kernel (memops.elf)
// Single kernel handles multiple operations via operation identifier

// Operation identifiers for memops kernel
গণনা ggml_et_memop_type : uint32_t {
    GGML_ET_MEMOP_MEMSET = 0,
};

// Memset operation: fill device memory with a value
// Returns true on success, false on failure
bool ggml_et_memset(ggml_backend_et_device_context * dev_ctx, শূন্য * dst_ptr, uint8_t মান, size_t size);
