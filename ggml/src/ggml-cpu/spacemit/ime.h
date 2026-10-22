#pragma once

#ಅನ್ತರ್ಭಾವ "ggml-alloc.h"

#ifdef __cplusplus
ಬಾಹ್ಯ "C" {
#endif

ggml_backend_buffer_type_t ggml_backend_cpu_riscv64_spacemit_buffer_type(ಶೂನ್ಯ);

ಶೂನ್ಯ ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity(ಪೂರ್ಣಾಙ್ಕ thread_n);

ಶೂನ್ಯ ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(ಪೂರ್ಣಾಙ್ಕ thread_n);

ಶೂನ್ಯ * ggml_backend_cpu_riscv64_spacemit_alloc_shared(size_t size, size_t alignment);

ಶೂನ್ಯ ggml_backend_cpu_riscv64_spacemit_free_shared(ಶೂನ್ಯ * ptr);

#ifdef __cplusplus
}
#endif
