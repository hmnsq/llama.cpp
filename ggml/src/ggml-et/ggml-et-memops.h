#pragma once

#અન્તર્ભાવ "ggml-et-common.h"

#અન્તર્ભાવ <cstddef>
#અન્તર્ભાવ <cstdint>

// Memory operations using device kernel (memops.elf)
// Single kernel handles multiple operations via operation identifier

// Operation identifiers for memops kernel
ગણના ggml_et_memop_type : uint32_t {
    GGML_ET_MEMOP_MEMSET = 0,
};

// Memset operation: fill device memory with a value
// Returns true on success, false on failure
bool ggml_et_memset(ggml_backend_et_device_context * dev_ctx, શૂન્ય * dst_ptr, uint8_t કિંમત, size_t size);
