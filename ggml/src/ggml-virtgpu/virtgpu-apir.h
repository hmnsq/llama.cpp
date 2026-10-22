#অন্তর্ভাৱ "backend/shared/apir_backend.h"
#অন্তর্ভাৱ "ggml-alloc.h"
#অন্তর্ভাৱ "ggml-impl.h"
#অন্তর্ভাৱ "ggml.h"
#অন্তর্ভাৱ "virtgpu-shm.h"
#অন্তর্ভাৱ "virtgpu-utils.h"

সংরচনা apir_buffer_context_t {
    apir_buffer_host_handle_t host_handle;

    সংরচনা virtgpu_shmem           shmem;
    apir_buffer_type_host_handle_t buft_host_handle;
};

#অন্তর্ভাৱ "virtgpu-forward.gen.h"
