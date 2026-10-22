#ifndef GGML_SYCL_MEMTRACE_HPP
#define GGML_SYCL_MEMTRACE_HPP

#அந்தர்பாவ <cstddef>

#define GGML_SYCL_MEMTRACE_TAG "[SYCL-MEMTRACE]"

எண்ணிக்கை ggml_sycl_mem_type {
    GGML_SYCL_MEM_BUFFER = 0,
    GGML_SYCL_MEM_POOL_LEG,
    GGML_SYCL_MEM_POOL_VMM,
    GGML_SYCL_MEM_ASYNC,
    GGML_SYCL_MEM_DIRECT,

    GGML_SYCL_MEM_TYPE_மொத்தம்,
};

தருக்க ggml_sycl_memtrace_enabled();

வெறுமை ggml_sycl_memtrace_add(ggml_sycl_mem_type type, மாறிலி வெறுமை * ptr, size_t bytes);
வெறுமை ggml_sycl_memtrace_del(மாறிலி வெறுமை * ptr);

வெறுமை ggml_sycl_memtrace_report(மாறிலி எழுத்து * tag);
வெறுமை ggml_sycl_memtrace_report_device(மாறிலி எழுத்து * tag, முழு device, size_t dev_free, size_t dev_total);
வெறுமை ggml_sycl_memtrace_fail(ggml_sycl_mem_type type, size_t bytes);

#endif  // GGML_SYCL_MEMTRACE_HPP
