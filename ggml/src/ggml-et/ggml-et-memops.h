#ਹਦਾਇਤ once

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-et-common.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <cstddef>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>

// Memory operations using device kernel (memops.elf)
// Single kernel handles multiple operations via operation identifier

// Operation identifiers for memops kernel
ਗਿਣਤੀ ggml_et_memop_type : uint32_t {
    GGML_ET_MEMOP_MEMSET = 0,
};

// Memset operation: fill device memory with a value
// Returns true on success, false on failure
bool ggml_et_memset(ggml_backend_et_device_context * dev_ctx, ਖਾਲੀ * dst_ptr, uint8_t ਮੁੱਲ, size_t ਮਾਪ);
