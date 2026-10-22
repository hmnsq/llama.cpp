#సూచన once

#అన్తర్భావ "ggml-alloc.h"

#నిర్వచితమైతే __cplusplus
బాహ్య "C" {
#అయితే_ముగింపు

ggml_backend_బఫరు_type_t ggml_backend_cpu_riscv64_spacemit_బఫరు_type(శూన్య);

శూన్య ggml_backend_cpu_riscv64_spacemit_సమితి_numa_thread_affinity(పూర్ణాంకం thread_n);

శూన్య ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(పూర్ణాంకం thread_n);

శూన్య * ggml_backend_cpu_riscv64_spacemit_alloc_shared(size_t కొలత, size_t alignment);

శూన్య ggml_backend_cpu_riscv64_spacemit_free_shared(శూన్య * ptr);

#నిర్వచితమైతే __cplusplus
}
#అయితే_ముగింపు
