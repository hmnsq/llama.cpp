#ifndef GGML_SYCL_MEMTRACE_HPP
#define GGML_SYCL_MEMTRACE_HPP

#অন্তর্ভাব <cstddef>

#define GGML_SYCL_MEMTRACE_TAG "[SYCL-MEMTRACE]"

গণনা ggml_sycl_mem_type {
    GGML_SYCL_MEM_BUFFER = 0,
    GGML_SYCL_MEM_POOL_LEG,
    GGML_SYCL_MEM_POOL_VMM,
    GGML_SYCL_MEM_ASYNC,
    GGML_SYCL_MEM_DIRECT,

    GGML_SYCL_MEM_TYPE_গুনতি,
};

তার্কিক ggml_sycl_memtrace_enabled();

শূন্য ggml_sycl_memtrace_add(ggml_sycl_mem_type type, ধ্রুব শূন্য * ptr, size_t bytes);
শূন্য ggml_sycl_memtrace_del(ধ্রুব শূন্য * ptr);

শূন্য ggml_sycl_memtrace_report(ধ্রুব বর্ণ * tag);
শূন্য ggml_sycl_memtrace_report_device(ধ্রুব বর্ণ * tag, পূর্ণাঙ্ক device, size_t dev_free, size_t dev_total);
শূন্য ggml_sycl_memtrace_fail(ggml_sycl_mem_type type, size_t bytes);

#endif  // GGML_SYCL_MEMTRACE_HPP
