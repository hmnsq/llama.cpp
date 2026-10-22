#رهنمود once

#ضمن "ggml-alloc.h"

#اگر_تعریف_شده __cplusplus
خارجي "C" {
#پایان_اگر

ggml_backend_میانگیر_type_t ggml_backend_cpu_riscv64_spacemit_میانگیر_type(فراغ);

فراغ ggml_backend_cpu_riscv64_spacemit_مجموعه_numa_thread_affinity(صحيح thread_n);

فراغ ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(صحيح thread_n);

فراغ * ggml_backend_cpu_riscv64_spacemit_alloc_shared(size_t اندازه, size_t alignment);

فراغ ggml_backend_cpu_riscv64_spacemit_free_shared(فراغ * ptr);

#اگر_تعریف_شده __cplusplus
}
#پایان_اگر
