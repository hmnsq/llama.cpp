#pragma once

#ଅନ୍ତର୍ଭାଵ "ggml-et-common.h"

#ଅନ୍ତର୍ଭାଵ <cstddef>
#ଅନ୍ତର୍ଭାଵ <cstdint>

// Memory operations using device kernel (memops.elf)
// Single kernel handles multiple operations via operation identifier

// Operation identifiers for memops kernel
ଗଣନା ggml_et_memop_type : uint32_t {
    GGML_ET_MEMOP_MEMSET = 0,
};

// Memset operation: fill device memory with a value
// Returns true on success, false on failure
bool ggml_et_memset(ggml_backend_et_device_context * dev_ctx, ଶୂନ୍ୟ * dst_ptr, uint8_t ମୂଲ୍ୟ, size_t size);
