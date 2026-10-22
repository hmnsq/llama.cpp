#आशय once

#समावेश "ggml-alloc.h"

#अगर_घोषित __cplusplus
बाह्य "C" {
#पूर्ण_अगर

ggml_backend_बफर_type_t ggml_backend_cpu_riscv64_spacemit_बफर_type(व्योम);

व्योम ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity(पूर्णांक thread_n);

व्योम ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(पूर्णांक thread_n);

व्योम * ggml_backend_cpu_riscv64_spacemit_alloc_shared(माप_प्रकार आकार, माप_प्रकार alignment);

व्योम ggml_backend_cpu_riscv64_spacemit_free_shared(व्योम * ptr);

#अगर_घोषित __cplusplus
}
#पूर्ण_अगर
