#অন্তর্ভাব "backend/shared/apir_backend.h"
#অন্তর্ভাব "ggml-alloc.h"
#অন্তর্ভাব "ggml-impl.h"
#অন্তর্ভাব "ggml.h"
#অন্তর্ভাব "virtgpu-shm.h"
#অন্তর্ভাব "virtgpu-utils.h"

কাঠামো apir_বাফার_context_t {
    apir_বাফার_host_handle_t host_handle;

    কাঠামো virtgpu_shmem           shmem;
    apir_বাফার_type_host_handle_t buft_host_handle;
};

#অন্তর্ভাব "virtgpu-forward.gen.h"
