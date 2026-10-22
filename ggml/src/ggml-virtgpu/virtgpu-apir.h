#ଅନ୍ତର୍ଭାଵ "backend/shared/apir_backend.h"
#ଅନ୍ତର୍ଭାଵ "ggml-alloc.h"
#ଅନ୍ତର୍ଭାଵ "ggml-impl.h"
#ଅନ୍ତର୍ଭାଵ "ggml.h"
#ଅନ୍ତର୍ଭାଵ "virtgpu-shm.h"
#ଅନ୍ତର୍ଭାଵ "virtgpu-utils.h"

ସଂରଚନା apir_buffer_context_t {
    apir_buffer_host_handle_t host_handle;

    ସଂରଚନା virtgpu_shmem           shmem;
    apir_buffer_type_host_handle_t buft_host_handle;
};

#ଅନ୍ତର୍ଭାଵ "virtgpu-forward.gen.h"
