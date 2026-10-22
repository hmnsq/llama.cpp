#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_MEMTRACE_HPP
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SYCL_MEMTRACE_HPP

#ਅਨ੍ਤਰ੍ਭਾਵ <cstddef>

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SYCL_MEMTRACE_ਟੈਗ "[SYCL-MEMTRACE]"

ਗਿਣਤੀ ggml_sycl_mem_type {
    GGML_SYCL_MEM_ਬਫਰ = 0,
    GGML_SYCL_MEM_POOL_LEG,
    GGML_SYCL_MEM_POOL_VMM,
    GGML_SYCL_MEM_ASYNC,
    GGML_SYCL_MEM_ਸਿੱਧਾ,

    GGML_SYCL_MEM_TYPE_ਗਿਣਤੀ,
};

ਤਾਰਕਿਕ ggml_sycl_memtrace_enabled();

ਖਾਲੀ ggml_sycl_memtrace_ਜੋੜੋ(ggml_sycl_mem_type type, ਅਚਲ ਖਾਲੀ * ptr, size_t ਬਾਈਟਾਂ);
ਖਾਲੀ ggml_sycl_memtrace_del(ਅਚਲ ਖਾਲੀ * ptr);

ਖਾਲੀ ggml_sycl_memtrace_report(ਅਚਲ ਅੱਖਰ * ਟੈਗ);
ਖਾਲੀ ggml_sycl_memtrace_report_device(ਅਚਲ ਅੱਖਰ * ਟੈਗ, ਪੂਰਨ device, size_t dev_free, size_t dev_ਕੁੱਲ);
ਖਾਲੀ ggml_sycl_memtrace_fail(ggml_sycl_mem_type type, size_t ਬਾਈਟਾਂ);

#ਜੇ_ਅੰਤ  // GGML_SYCL_MEMTRACE_HPP
