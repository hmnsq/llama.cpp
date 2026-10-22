#અન્તર્ભાવ "backend/shared/apir_backend.h"
#અન્તર્ભાવ "ggml-alloc.h"
#અન્તર્ભાવ "ggml-impl.h"
#અન્તર્ભાવ "ggml.h"
#અન્તર્ભાવ "virtgpu-shm.h"
#અન્તર્ભાવ "virtgpu-utils.h"

સંરચના apir_buffer_context_t {
    apir_buffer_host_handle_t host_handle;

    સંરચના virtgpu_shmem           shmem;
    apir_buffer_type_host_handle_t buft_host_handle;
};

#અન્તર્ભાવ "virtgpu-forward.gen.h"
