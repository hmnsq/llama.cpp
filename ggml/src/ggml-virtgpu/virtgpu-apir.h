#شامل "backend/shared/apir_backend.h"
#شامل "ggml-alloc.h"
#شامل "ggml-impl.h"
#شامل "ggml.h"
#شامل "virtgpu-shm.h"
#شامل "virtgpu-utils.h"

ساخت apir_بفر_context_t {
    apir_بفر_host_handle_t host_handle;

    ساخت virtgpu_shmem           shmem;
    apir_بفر_type_host_handle_t buft_host_handle;
};

#شامل "virtgpu-forward.gen.h"
