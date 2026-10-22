#ضمن "backend/shared/apir_backend.h"
#ضمن "ggml-alloc.h"
#ضمن "ggml-impl.h"
#ضمن "ggml.h"
#ضمن "virtgpu-shm.h"
#ضمن "virtgpu-utils.h"

بنية apir_بفر_context_t {
    apir_بفر_host_handle_t host_handle;

    بنية virtgpu_shmem           shmem;
    apir_بفر_type_host_handle_t buft_host_handle;
};

#ضمن "virtgpu-forward.gen.h"
