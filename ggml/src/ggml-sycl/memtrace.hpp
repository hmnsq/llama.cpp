#که_نه_تعریف_شوی GGML_SYCL_MEMTRACE_HPP
#تعریف_کړه GGML_SYCL_MEMTRACE_HPP

#ضمن <cstddef>

#تعریف_کړه GGML_SYCL_MEMTRACE_نښان "[SYCL-MEMTRACE]"

تعداد ggml_sycl_mem_type {
    GGML_SYCL_MEM_بفر = 0,
    GGML_SYCL_MEM_POOL_LEG,
    GGML_SYCL_MEM_POOL_VMM,
    GGML_SYCL_MEM_ASYNC,
    GGML_SYCL_MEM_مستقیم,

    GGML_SYCL_MEM_TYPE_شمېر,
};

منطقي ggml_sycl_memtrace_enabled();

فراغ ggml_sycl_memtrace_جمع_کړه(ggml_sycl_mem_type type, ثابت فراغ * ptr, size_t بایټونه);
فراغ ggml_sycl_memtrace_del(ثابت فراغ * ptr);

فراغ ggml_sycl_memtrace_report(ثابت حرف * نښان);
فراغ ggml_sycl_memtrace_report_device(ثابت حرف * نښان, صحيح device, size_t dev_free, size_t dev_ټول);
فراغ ggml_sycl_memtrace_fail(ggml_sycl_mem_type type, size_t بایټونه);

#د_که_پای  // GGML_SYCL_MEMTRACE_HPP
