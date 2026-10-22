#ifndef GGML_SYCL_MEMTRACE_HPP
#define GGML_SYCL_MEMTRACE_HPP

#ଅନ୍ତର୍ଭାଵ <cstddef>

#define GGML_SYCL_MEMTRACE_TAG "[SYCL-MEMTRACE]"

ଗଣନା ggml_sycl_mem_type {
    GGML_SYCL_MEM_BUFFER = 0,
    GGML_SYCL_MEM_POOL_LEG,
    GGML_SYCL_MEM_POOL_VMM,
    GGML_SYCL_MEM_ASYNC,
    GGML_SYCL_MEM_DIRECT,

    GGML_SYCL_MEM_TYPE_ଗଣନା,
};

ତାର୍କିକ ggml_sycl_memtrace_enabled();

ଶୂନ୍ୟ ggml_sycl_memtrace_add(ggml_sycl_mem_type type, ଅଚଳ ଶୂନ୍ୟ * ptr, size_t bytes);
ଶୂନ୍ୟ ggml_sycl_memtrace_del(ଅଚଳ ଶୂନ୍ୟ * ptr);

ଶୂନ୍ୟ ggml_sycl_memtrace_report(ଅଚଳ ଅକ୍ଷର * tag);
ଶୂନ୍ୟ ggml_sycl_memtrace_report_device(ଅଚଳ ଅକ୍ଷର * tag, ପୂର୍ଣ୍ଣସଂଖ୍ୟା device, size_t dev_free, size_t dev_total);
ଶୂନ୍ୟ ggml_sycl_memtrace_fail(ggml_sycl_mem_type type, size_t bytes);

#endif  // GGML_SYCL_MEMTRACE_HPP
