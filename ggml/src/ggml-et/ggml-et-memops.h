#ಸೂಚನೆ once

#ಅನ್ತರ್ಭಾವ "ggml-et-common.h"

#ಅನ್ತರ್ಭಾವ <cstddef>
#ಅನ್ತರ್ಭಾವ <cstdint>

// Memory operations using device kernel (memops.elf)
// Single kernel handles multiple operations via operation identifier

// Operation identifiers for memops kernel
ಎಣಿಕೆ ggml_et_memop_type : uint32_t {
    GGML_ET_MEMOP_MEMSET = 0,
};

// Memset operation: fill device memory with a value
// Returns true on success, false on failure
bool ggml_et_memset(ggml_backend_et_device_context * dev_ctx, ಶೂನ್ಯ * dst_ptr, uint8_t ಮೌಲ್ಯ, size_t ಗಾತ್ರ_2);
