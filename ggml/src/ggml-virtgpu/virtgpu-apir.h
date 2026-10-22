#ਅਨ੍ਤਰ੍ਭਾਵ "backend/shared/apir_backend.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-alloc.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "virtgpu-shm.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "virtgpu-utils.h"

ਬਣਤਰ apir_ਬਫਰ_context_t {
    apir_ਬਫਰ_host_handle_t host_handle;

    ਬਣਤਰ virtgpu_shmem           shmem;
    apir_ਬਫਰ_type_host_handle_t buft_host_handle;
};

#ਅਨ੍ਤਰ੍ਭਾਵ "virtgpu-forward.gen.h"
