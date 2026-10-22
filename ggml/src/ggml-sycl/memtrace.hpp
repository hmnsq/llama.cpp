#ifndef GGML_SYCL_MEMTRACE_HPP
#define GGML_SYCL_MEMTRACE_HPP

#অন্তর্ভাব <cstddef>

#define GGML_SYCL_MEMTRACE_TAG "[SYCL-MEMTRACE]"

গণনা ggml_sycl_mem_type {
    GGML_SYCL_MEM_বাফার = 0,
    GGML_SYCL_MEM_POOL_LEG,
    GGML_SYCL_MEM_POOL_VMM,
    GGML_SYCL_MEM_ASYNC,
    GGML_SYCL_MEM_DIRECT,

    GGML_SYCL_MEM_TYPE_গুনতি,
};

যৌক্তিক ggml_sycl_memtrace_enabled();

শূন্য ggml_sycl_memtrace_add(ggml_sycl_mem_type type, ধ্রুবক শূন্য * ptr, size_t bytes);
শূন্য ggml_sycl_memtrace_del(ধ্রুবক শূন্য * ptr);

শূন্য ggml_sycl_memtrace_report(ধ্রুবক অক্ষর * tag);
শূন্য ggml_sycl_memtrace_report_device(ধ্রুবক অক্ষর * tag, পূর্ণসংখ্যা device, size_t dev_free, size_t dev_যোগ);
শূন্য ggml_sycl_memtrace_fail(ggml_sycl_mem_type type, size_t bytes);

#endif  // GGML_SYCL_MEMTRACE_HPP
