#اگر_تعریف_نشده GGML_SYCL_MEMTRACE_HPP
#تعریف_کن GGML_SYCL_MEMTRACE_HPP

#ضمن <cstddef>

#تعریف_کن GGML_SYCL_MEMTRACE_برچسب "[SYCL-MEMTRACE]"

تعداد ggml_sycl_mem_type {
    GGML_SYCL_MEM_میانگیر = 0,
    GGML_SYCL_MEM_POOL_LEG,
    GGML_SYCL_MEM_POOL_VMM,
    GGML_SYCL_MEM_ASYNC,
    GGML_SYCL_MEM_مستقیم,

    GGML_SYCL_MEM_TYPE_شمار,
};

منطقي ggml_sycl_memtrace_enabled();

فراغ ggml_sycl_memtrace_جمع_کن(ggml_sycl_mem_type type, ثابت فراغ * ptr, size_t بایتها);
فراغ ggml_sycl_memtrace_del(ثابت فراغ * ptr);

فراغ ggml_sycl_memtrace_report(ثابت حرف * برچسب);
فراغ ggml_sycl_memtrace_report_device(ثابت حرف * برچسب, صحيح device, size_t dev_free, size_t dev_مجموع);
فراغ ggml_sycl_memtrace_fail(ggml_sycl_mem_type type, size_t بایتها);

#پایان_اگر  // GGML_SYCL_MEMTRACE_HPP
