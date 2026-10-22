#pragma once

#شامل "ggml-alloc.h"

#ifdef __cplusplus
بیرونی "C" {
#endif

ggml_backend_buffer_type_t ggml_backend_cpu_riscv64_spacemit_buffer_type(تهی);

تهی ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity(صحیح thread_n);

تهی ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(صحیح thread_n);

تهی * ggml_backend_cpu_riscv64_spacemit_alloc_shared(size_t size, size_t alignment);

تهی ggml_backend_cpu_riscv64_spacemit_free_shared(تهی * ptr);

#ifdef __cplusplus
}
#endif
