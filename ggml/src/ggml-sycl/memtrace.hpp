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

    GGML_SYCL_MEM_TYPE_मोजणी,
};

तार्किक ggml_sycl_memtrace_enabled();

शून्य ggml_sycl_memtrace_add(ggml_sycl_mem_type type, स्थिरांक शून्य * ptr, माप_प्रकार bytes);
शून्य ggml_sycl_memtrace_del(स्थिरांक शून्य * ptr);

शून्य ggml_sycl_memtrace_report(स्थिरांक अक्षर * tag);
शून्य ggml_sycl_memtrace_report_device(स्थिरांक अक्षर * tag, पूर्णांक device, माप_प्रकार dev_free, माप_प्रकार dev_total);
शून्य ggml_sycl_memtrace_fail(ggml_sycl_mem_type type, माप_प्रकार bytes);

#endif  // GGML_SYCL_MEMTRACE_HPP
