#ifndef GGML_SYCL_MEMTRACE_HPP
#define GGML_SYCL_MEMTRACE_HPP

#അന്തര്ഭാവ <cstddef>

#define GGML_SYCL_MEMTRACE_TAG "[SYCL-MEMTRACE]"

എണ്ണൽ ggml_sycl_mem_type {
    GGML_SYCL_MEM_BUFFER = 0,
    GGML_SYCL_MEM_POOL_LEG,
    GGML_SYCL_MEM_POOL_VMM,
    GGML_SYCL_MEM_ASYNC,
    GGML_SYCL_MEM_DIRECT,

    GGML_SYCL_MEM_TYPE_എണ്ണം,
};

തർക്കം ggml_sycl_memtrace_enabled();

ശൂന്യം ggml_sycl_memtrace_add(ggml_sycl_mem_type type, സ്ഥിരം ശൂന്യം * ptr, size_t bytes);
ശൂന്യം ggml_sycl_memtrace_del(സ്ഥിരം ശൂന്യം * ptr);

ശൂന്യം ggml_sycl_memtrace_report(സ്ഥിരം അക്ഷരം * tag);
ശൂന്യം ggml_sycl_memtrace_report_device(സ്ഥിരം അക്ഷരം * tag, പൂർണ്ണസംഖ്യ device, size_t dev_free, size_t dev_total);
ശൂന്യം ggml_sycl_memtrace_fail(ggml_sycl_mem_type type, size_t bytes);

#endif  // GGML_SYCL_MEMTRACE_HPP
