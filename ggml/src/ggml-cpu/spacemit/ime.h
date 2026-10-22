#निर्देशः once

#अन्तर्भाव "ggml-alloc.h"

#यदि_परिभाषितम् __cplusplus
बाह्य "C" {
#यदि_अन्तः

ggml_backend_बफर_type_t ggml_backend_cpu_riscv64_spacemit_बफर_type(शून्य);

शून्य ggml_backend_cpu_riscv64_spacemit_समुच्चयः_numa_thread_affinity(पूर्णाङ्क thread_n);

शून्य ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(पूर्णाङ्क thread_n);

शून्य * ggml_backend_cpu_riscv64_spacemit_alloc_shared(size_t आकार, size_t alignment);

शून्य ggml_backend_cpu_riscv64_spacemit_free_shared(शून्य * ptr);

#यदि_परिभाषितम् __cplusplus
}
#यदि_अन्तः
