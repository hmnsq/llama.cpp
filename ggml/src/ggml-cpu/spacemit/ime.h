#pragma once

#அந்தர்பாவ "ggml-alloc.h"

#ifdef __cplusplus
வெளி "C" {
#endif

ggml_backend_buffer_type_t ggml_backend_cpu_riscv64_spacemit_buffer_type(வெறுமை);

வெறுமை ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity(முழு thread_n);

வெறுமை ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(முழு thread_n);

வெறுமை * ggml_backend_cpu_riscv64_spacemit_alloc_shared(size_t size, size_t alignment);

வெறுமை ggml_backend_cpu_riscv64_spacemit_free_shared(வெறுமை * ptr);

#ifdef __cplusplus
}
#endif
