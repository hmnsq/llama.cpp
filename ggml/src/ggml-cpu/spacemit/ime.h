#pragma once

#অন্তর্ভাব "ggml-alloc.h"

#ifdef __cplusplus
বাহ্যিক "C" {
#endif

ggml_backend_বাফার_type_t ggml_backend_cpu_riscv64_spacemit_বাফার_type(শূন্য);

শূন্য ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity(পূর্ণসংখ্যা thread_n);

শূন্য ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(পূর্ণসংখ্যা thread_n);

শূন্য * ggml_backend_cpu_riscv64_spacemit_alloc_shared(size_t আকার_2, size_t alignment);

শূন্য ggml_backend_cpu_riscv64_spacemit_free_shared(শূন্য * ptr);

#ifdef __cplusplus
}
#endif
