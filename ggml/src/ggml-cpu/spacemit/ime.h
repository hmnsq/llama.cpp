#pragma once

#ଅନ୍ତର୍ଭାଵ "ggml-alloc.h"

#ifdef __cplusplus
ବାହ୍ୟ "C" {
#endif

ggml_backend_buffer_type_t ggml_backend_cpu_riscv64_spacemit_buffer_type(ଶୂନ୍ୟ);

ଶୂନ୍ୟ ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity(ପୂର୍ଣ୍ଣସଂଖ୍ୟା thread_n);

ଶୂନ୍ୟ ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(ପୂର୍ଣ୍ଣସଂଖ୍ୟା thread_n);

ଶୂନ୍ୟ * ggml_backend_cpu_riscv64_spacemit_alloc_shared(size_t size, size_t alignment);

ଶୂନ୍ୟ ggml_backend_cpu_riscv64_spacemit_free_shared(ଶୂନ୍ୟ * ptr);

#ifdef __cplusplus
}
#endif
