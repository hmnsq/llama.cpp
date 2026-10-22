#ifndef GGML_SYCL_MEMTRACE_HPP
#define GGML_SYCL_MEMTRACE_HPP

#ಅನ್ತರ್ಭಾವ <cstddef>

#define GGML_SYCL_MEMTRACE_TAG "[SYCL-MEMTRACE]"

ಗಣನಾ ggml_sycl_mem_type {
    GGML_SYCL_MEM_BUFFER = 0,
    GGML_SYCL_MEM_POOL_LEG,
    GGML_SYCL_MEM_POOL_VMM,
    GGML_SYCL_MEM_ASYNC,
    GGML_SYCL_MEM_DIRECT,

    GGML_SYCL_MEM_TYPE_ಎಣಿಕೆ,
};

ತಾರ್ಕಿಕ ggml_sycl_memtrace_enabled();

ಶೂನ್ಯ ggml_sycl_memtrace_add(ggml_sycl_mem_type type, ಧ್ರುವ ಶೂನ್ಯ * ptr, size_t bytes);
ಶೂನ್ಯ ggml_sycl_memtrace_del(ಧ್ರುವ ಶೂನ್ಯ * ptr);

ಶೂನ್ಯ ggml_sycl_memtrace_report(ಧ್ರುವ ವರ್ಣ * tag);
ಶೂನ್ಯ ggml_sycl_memtrace_report_device(ಧ್ರುವ ವರ್ಣ * tag, ಪೂರ್ಣಾಙ್ಕ device, size_t dev_free, size_t dev_total);
ಶೂನ್ಯ ggml_sycl_memtrace_fail(ggml_sycl_mem_type type, size_t bytes);

#endif  // GGML_SYCL_MEMTRACE_HPP
