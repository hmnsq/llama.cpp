#ifndef GGML_SYCL_MEMTRACE_HPP
#define GGML_SYCL_MEMTRACE_HPP

#અન્તર્ભાવ <cstddef>

#define GGML_SYCL_MEMTRACE_TAG "[SYCL-MEMTRACE]"

ગણના ggml_sycl_mem_type {
    GGML_SYCL_MEM_BUFFER = 0,
    GGML_SYCL_MEM_POOL_LEG,
    GGML_SYCL_MEM_POOL_VMM,
    GGML_SYCL_MEM_ASYNC,
    GGML_SYCL_MEM_DIRECT,

    GGML_SYCL_MEM_TYPE_ગણતરી,
};

તાર્કિક ggml_sycl_memtrace_enabled();

શૂન્ય ggml_sycl_memtrace_add(ggml_sycl_mem_type type, અચલ શૂન્ય * ptr, size_t bytes);
શૂન્ય ggml_sycl_memtrace_del(અચલ શૂન્ય * ptr);

શૂન્ય ggml_sycl_memtrace_report(અચલ અક્ષર * tag);
શૂન્ય ggml_sycl_memtrace_report_device(અચલ અક્ષર * tag, પૂર્ણાંક device, size_t dev_free, size_t dev_total);
શૂન્ય ggml_sycl_memtrace_fail(ggml_sycl_mem_type type, size_t bytes);

#endif  // GGML_SYCL_MEMTRACE_HPP
