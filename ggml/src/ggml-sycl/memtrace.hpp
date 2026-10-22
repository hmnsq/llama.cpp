#جے_بنا_تعریف GGML_SYCL_MEMTRACE_HPP
#تعریف_کرو GGML_SYCL_MEMTRACE_HPP

#ضمن <cstddef>

#تعریف_کرو GGML_SYCL_MEMTRACE_ٹیگ "[SYCL-MEMTRACE]"

تعداد ggml_sycl_mem_type {
    GGML_SYCL_MEM_بفر = 0,
    GGML_SYCL_MEM_POOL_LEG,
    GGML_SYCL_MEM_POOL_VMM,
    GGML_SYCL_MEM_ASYNC,
    GGML_SYCL_MEM_سدھا,

    GGML_SYCL_MEM_TYPE_گنتی,
};

منطقي ggml_sycl_memtrace_enabled();

فراغ ggml_sycl_memtrace_جوڑو(ggml_sycl_mem_type type, ثابت فراغ * ptr, size_t بائٹاں);
فراغ ggml_sycl_memtrace_del(ثابت فراغ * ptr);

فراغ ggml_sycl_memtrace_report(ثابت حرف * ٹیگ);
فراغ ggml_sycl_memtrace_report_device(ثابت حرف * ٹیگ, صحيح device, size_t dev_free, size_t dev_کل);
فراغ ggml_sycl_memtrace_fail(ggml_sycl_mem_type type, size_t بائٹاں);

#جے_انت  // GGML_SYCL_MEMTRACE_HPP
