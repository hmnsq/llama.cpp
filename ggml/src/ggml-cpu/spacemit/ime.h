#pragma once

#ضمن "ggml-alloc.h"

#ifdef __cplusplus
خارجي "C" {
#endif

ggml_backend_buffer_type_t ggml_backend_cpu_riscv64_spacemit_buffer_type(فراغ);

فراغ ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity(صحيح thread_n);

فراغ ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(صحيح thread_n);

فراغ * ggml_backend_cpu_riscv64_spacemit_alloc_shared(size_t size, size_t alignment);

فراغ ggml_backend_cpu_riscv64_spacemit_free_shared(فراغ * ptr);

#ifdef __cplusplus
}
#endif
