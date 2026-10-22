#لارښوونه once

#ضمن "ggml-alloc.h"

#که_تعریف_شوی __cplusplus
خارجي "C" {
#د_که_پای

ggml_backend_بفر_type_t ggml_backend_cpu_riscv64_spacemit_بفر_type(فراغ);

فراغ ggml_backend_cpu_riscv64_spacemit_سېټ_numa_thread_affinity(صحيح thread_n);

فراغ ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(صحيح thread_n);

فراغ * ggml_backend_cpu_riscv64_spacemit_alloc_shared(size_t اندازه, size_t alignment);

فراغ ggml_backend_cpu_riscv64_spacemit_free_shared(فراغ * ptr);

#که_تعریف_شوی __cplusplus
}
#د_که_پای
