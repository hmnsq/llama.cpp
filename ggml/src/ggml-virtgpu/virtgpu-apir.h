#অন্তর্ভাব "backend/shared/apir_backend.h"
#অন্তর্ভাব "ggml-alloc.h"
#অন্তর্ভাব "ggml-impl.h"
#অন্তর্ভাব "ggml.h"
#অন্তর্ভাব "virtgpu-shm.h"
#অন্তর্ভাব "virtgpu-utils.h"

সংরচনা apir_buffer_context_t {
    apir_buffer_host_handle_t host_handle;

    সংরচনা virtgpu_shmem           shmem;
    apir_buffer_type_host_handle_t buft_host_handle;
};

#অন্তর্ভাব "virtgpu-forward.gen.h"
