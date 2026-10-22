#अगर_अघोषित GGML_SYCL_MEMTRACE_HPP
#घोषणा GGML_SYCL_MEMTRACE_HPP

#समावेश <cstddef>

#घोषणा GGML_SYCL_MEMTRACE_TAG "[SYCL-MEMTRACE]"

क्रमागत ggml_sycl_mem_type {
    GGML_SYCL_MEM_बफर = 0,
    GGML_SYCL_MEM_POOL_LEG,
    GGML_SYCL_MEM_POOL_VMM,
    GGML_SYCL_MEM_ASYNC,
    GGML_SYCL_MEM_DIRECT,

    GGML_SYCL_MEM_TYPE_गिनती,
};

bool ggml_sycl_memtrace_enabled();

व्योम ggml_sycl_memtrace_add(ggml_sycl_mem_type type, स्थिर व्योम * ptr, माप_प्रकार bytes);
व्योम ggml_sycl_memtrace_del(स्थिर व्योम * ptr);

व्योम ggml_sycl_memtrace_report(स्थिर अक्षर * tag);
व्योम ggml_sycl_memtrace_report_device(स्थिर अक्षर * tag, पूर्णांक device, माप_प्रकार dev_free, माप_प्रकार dev_योग);
व्योम ggml_sycl_memtrace_fail(ggml_sycl_mem_type type, माप_प्रकार bytes);

#पूर्ण_अगर  // GGML_SYCL_MEMTRACE_HPP
