#समावेश "backend/shared/apir_backend.h"
#समावेश "ggml-alloc.h"
#समावेश "ggml-impl.h"
#समावेश "ggml.h"
#समावेश "virtgpu-shm.h"
#समावेश "virtgpu-utils.h"

काष्ठा apir_बफर_context_t {
    apir_बफर_host_handle_t host_handle;

    काष्ठा virtgpu_shmem           shmem;
    apir_बफर_type_host_handle_t buft_host_handle;
};

#समावेश "virtgpu-forward.gen.h"
