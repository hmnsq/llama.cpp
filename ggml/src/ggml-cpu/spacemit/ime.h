#pragma once

#അന്തര്ഭാവ "ggml-alloc.h"

#ifdef __cplusplus
ബാഹ്യ "C" {
#endif

ggml_backend_buffer_type_t ggml_backend_cpu_riscv64_spacemit_buffer_type(ശൂന്യം);

ശൂന്യം ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity(പൂർണ്ണസംഖ്യ thread_n);

ശൂന്യം ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(പൂർണ്ണസംഖ്യ thread_n);

ശൂന്യം * ggml_backend_cpu_riscv64_spacemit_alloc_shared(size_t size, size_t alignment);

ശൂന്യം ggml_backend_cpu_riscv64_spacemit_free_shared(ശൂന്യം * ptr);

#ifdef __cplusplus
}
#endif
