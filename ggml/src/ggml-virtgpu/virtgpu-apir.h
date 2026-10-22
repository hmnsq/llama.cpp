#अन्तर्भाव "backend/shared/apir_backend.h"
#अन्तर्भाव "ggml-alloc.h"
#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव "ggml.h"
#अन्तर्भाव "virtgpu-shm.h"
#अन्तर्भाव "virtgpu-utils.h"

संरचना apir_buffer_context_t {
    apir_buffer_host_handle_t host_handle;

    संरचना virtgpu_shmem           shmem;
    apir_buffer_type_host_handle_t buft_host_handle;
};

#अन्तर्भाव "virtgpu-forward.gen.h"
