#ifndef GGML_SYCL_MEMTRACE_HPP
#define GGML_SYCL_MEMTRACE_HPP

#شامل <cstddef>

#define GGML_SYCL_MEMTRACE_TAG "[SYCL-MEMTRACE]"

شمارش ggml_sycl_mem_type {
    GGML_SYCL_MEM_BUFFER = 0,
    GGML_SYCL_MEM_POOL_LEG,
    GGML_SYCL_MEM_POOL_VMM,
    GGML_SYCL_MEM_ASYNC,
    GGML_SYCL_MEM_DIRECT,

    GGML_SYCL_MEM_TYPE_شمار,
};

منطقی ggml_sycl_memtrace_enabled();

تهی ggml_sycl_memtrace_add(ggml_sycl_mem_type type, ثابت تهی * ptr, size_t bytes);
تهی ggml_sycl_memtrace_del(ثابت تهی * ptr);

تهی ggml_sycl_memtrace_report(ثابت نویسه * tag);
تهی ggml_sycl_memtrace_report_device(ثابت نویسه * tag, صحیح device, size_t dev_free, size_t dev_total);
تهی ggml_sycl_memtrace_fail(ggml_sycl_mem_type type, size_t bytes);

#endif  // GGML_SYCL_MEMTRACE_HPP
