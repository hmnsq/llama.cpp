#pragma once

#અન્તર્ભાવ "ggml-alloc.h"

#ifdef __cplusplus
બાહ્ય "C" {
#endif

ggml_backend_buffer_type_t ggml_backend_cpu_riscv64_spacemit_buffer_type(શૂન્ય);

શૂન્ય ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity(પૂર્ણાંક thread_n);

શૂન્ય ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(પૂર્ણાંક thread_n);

શૂન્ય * ggml_backend_cpu_riscv64_spacemit_alloc_shared(size_t size, size_t alignment);

શૂન્ય ggml_backend_cpu_riscv64_spacemit_free_shared(શૂન્ય * ptr);

#ifdef __cplusplus
}
#endif
