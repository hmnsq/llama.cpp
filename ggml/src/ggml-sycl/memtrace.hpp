#ifndef GGML_SYCL_MEMTRACE_HPP
#define GGML_SYCL_MEMTRACE_HPP

#अन्तर्भाव <cstddef>

#define GGML_SYCL_MEMTRACE_TAG "[SYCL-MEMTRACE]"

गणना ggml_sycl_mem_type {
    GGML_SYCL_MEM_BUFFER = 0,
    GGML_SYCL_MEM_POOL_LEG,
    GGML_SYCL_MEM_POOL_VMM,
    GGML_SYCL_MEM_ASYNC,
    GGML_SYCL_MEM_DIRECT,

    GGML_SYCL_MEM_TYPE_गिनती,
};

तार्किक ggml_sycl_memtrace_enabled();

शून्य ggml_sycl_memtrace_add(ggml_sycl_mem_type type, ध्रुव शून्य * ptr, size_t bytes);
शून्य ggml_sycl_memtrace_del(ध्रुव शून्य * ptr);

शून्य ggml_sycl_memtrace_report(ध्रुव वर्ण * tag);
शून्य ggml_sycl_memtrace_report_device(ध्रुव वर्ण * tag, पूर्णाङ्क device, size_t dev_free, size_t dev_total);
शून्य ggml_sycl_memtrace_fail(ggml_sycl_mem_type type, size_t bytes);

#endif  // GGML_SYCL_MEMTRACE_HPP
