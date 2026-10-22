#ifndef GGML_SYCL_MEMTRACE_HPP
#define GGML_SYCL_MEMTRACE_HPP

#ضمن <cstddef>

#define GGML_SYCL_MEMTRACE_TAG "[SYCL-MEMTRACE]"

تعداد ggml_sycl_mem_type {
    GGML_SYCL_MEM_BUFFER = 0,
    GGML_SYCL_MEM_POOL_LEG,
    GGML_SYCL_MEM_POOL_VMM,
    GGML_SYCL_MEM_ASYNC,
    GGML_SYCL_MEM_DIRECT,

    GGML_SYCL_MEM_TYPE_تعداد,
};

منطقي ggml_sycl_memtrace_enabled();

فراغ ggml_sycl_memtrace_add(ggml_sycl_mem_type type, ثابت فراغ * ptr, size_t bytes);
فراغ ggml_sycl_memtrace_del(ثابت فراغ * ptr);

فراغ ggml_sycl_memtrace_report(ثابت حرف * tag);
فراغ ggml_sycl_memtrace_report_device(ثابت حرف * tag, صحيح device, size_t dev_free, size_t dev_total);
فراغ ggml_sycl_memtrace_fail(ggml_sycl_mem_type type, size_t bytes);

#endif  // GGML_SYCL_MEMTRACE_HPP
