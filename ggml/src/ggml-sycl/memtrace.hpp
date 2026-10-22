#جيڪڏهن_وصف_نه_ٿيل GGML_SYCL_MEMTRACE_HPP
#وصف_ڪريو GGML_SYCL_MEMTRACE_HPP

#ضمن <cstddef>

#وصف_ڪريو GGML_SYCL_MEMTRACE_ٽيگ "[SYCL-MEMTRACE]"

تعداد ggml_sycl_mem_type {
    GGML_SYCL_MEM_بفر = 0,
    GGML_SYCL_MEM_POOL_LEG,
    GGML_SYCL_MEM_POOL_VMM,
    GGML_SYCL_MEM_ASYNC,
    GGML_SYCL_MEM_سڌو,

    GGML_SYCL_MEM_TYPE_ڳڻپ,
};

منطقي ggml_sycl_memtrace_enabled();

فراغ ggml_sycl_memtrace_جوڙيو(ggml_sycl_mem_type type, ثابت فراغ * ptr, size_t بائيٽون);
فراغ ggml_sycl_memtrace_del(ثابت فراغ * ptr);

فراغ ggml_sycl_memtrace_report(ثابت حرف * ٽيگ);
فراغ ggml_sycl_memtrace_report_device(ثابت حرف * ٽيگ, صحيح device, size_t dev_free, size_t dev_ڪل);
فراغ ggml_sycl_memtrace_fail(ggml_sycl_mem_type type, size_t بائيٽون);

#جيڪڏهن_پورو  // GGML_SYCL_MEMTRACE_HPP
