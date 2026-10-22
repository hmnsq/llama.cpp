#ಅನ್ತರ್ಭಾವ "backend/shared/apir_backend.h"
#ಅನ್ತರ್ಭಾವ "ggml-alloc.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "virtgpu-shm.h"
#ಅನ್ತರ್ಭಾವ "virtgpu-utils.h"

ರಚನೆ apir_ಬಫರ್_context_t {
    apir_ಬಫರ್_host_handle_t host_handle;

    ರಚನೆ virtgpu_shmem           shmem;
    apir_ಬಫರ್_type_host_handle_t buft_host_handle;
};

#ಅನ್ತರ್ಭಾವ "virtgpu-forward.gen.h"
