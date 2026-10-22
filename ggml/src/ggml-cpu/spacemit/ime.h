#pragma once

#अन्तर्भाव "ggml-alloc.h"

#ifdef __cplusplus
बाह्य "C" {
#endif

ggml_backend_buffer_type_t ggml_backend_cpu_riscv64_spacemit_buffer_type(शून्य);

शून्य ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity(पूर्णाङ्क thread_n);

शून्य ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(पूर्णाङ्क thread_n);

शून्य * ggml_backend_cpu_riscv64_spacemit_alloc_shared(size_t size, size_t alignment);

शून्य ggml_backend_cpu_riscv64_spacemit_free_shared(शून्य * ptr);

#ifdef __cplusplus
}
#endif
