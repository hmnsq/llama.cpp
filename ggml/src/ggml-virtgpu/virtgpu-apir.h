#അന്തര്ഭാവ "backend/shared/apir_backend.h"
#അന്തര്ഭാവ "ggml-alloc.h"
#അന്തര്ഭാവ "ggml-impl.h"
#അന്തര്ഭാവ "ggml.h"
#അന്തര്ഭാവ "virtgpu-shm.h"
#അന്തര്ഭാവ "virtgpu-utils.h"

ഘടന apir_buffer_context_t {
    apir_buffer_host_handle_t host_handle;

    ഘടന virtgpu_shmem           shmem;
    apir_buffer_type_host_handle_t buft_host_handle;
};

#അന്തര്ഭാവ "virtgpu-forward.gen.h"
