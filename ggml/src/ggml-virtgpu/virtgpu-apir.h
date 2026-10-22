#అన్తర్భావ "backend/shared/apir_backend.h"
#అన్తర్భావ "ggml-alloc.h"
#అన్తర్భావ "ggml-impl.h"
#అన్తర్భావ "ggml.h"
#అన్తర్భావ "virtgpu-shm.h"
#అన్తర్భావ "virtgpu-utils.h"

నిర్మాణం apir_బఫరు_context_t {
    apir_బఫరు_host_handle_t host_handle;

    నిర్మాణం virtgpu_shmem           shmem;
    apir_బఫరు_type_host_handle_t buft_host_handle;
};

#అన్తర్భావ "virtgpu-forward.gen.h"
