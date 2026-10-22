#அந்தர்பாவ "backend/shared/apir_backend.h"
#அந்தர்பாவ "ggml-alloc.h"
#அந்தர்பாவ "ggml-impl.h"
#அந்தர்பாவ "ggml.h"
#அந்தர்பாவ "virtgpu-shm.h"
#அந்தர்பாவ "virtgpu-utils.h"

கட்டமைப்பு apir_buffer_context_t {
    apir_buffer_host_handle_t host_handle;

    கட்டமைப்பு virtgpu_shmem           shmem;
    apir_buffer_type_host_handle_t buft_host_handle;
};

#அந்தர்பாவ "virtgpu-forward.gen.h"
