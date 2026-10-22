#شامل "backend/shared/apir_backend.h"
#شامل "ggml-alloc.h"
#شامل "ggml-impl.h"
#شامل "ggml.h"
#شامل "virtgpu-shm.h"
#شامل "virtgpu-utils.h"

ساختار apir_buffer_context_t {
    apir_buffer_host_handle_t host_handle;

    ساختار virtgpu_shmem           shmem;
    apir_buffer_type_host_handle_t buft_host_handle;
};

#شامل "virtgpu-forward.gen.h"
