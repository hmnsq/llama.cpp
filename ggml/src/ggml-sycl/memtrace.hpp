#اگر_غیر_تعریف_شدہ GGML_SYCL_MEMTRACE_HPP
#تعریف_کرو GGML_SYCL_MEMTRACE_HPP

#شامل <cstddef>

#تعریف_کرو GGML_SYCL_MEMTRACE_لیبل "[SYCL-MEMTRACE]"

شمار ggml_sycl_mem_type {
    GGML_SYCL_MEM_بفر = 0,
    GGML_SYCL_MEM_POOL_LEG,
    GGML_SYCL_MEM_POOL_VMM,
    GGML_SYCL_MEM_ASYNC,
    GGML_SYCL_MEM_براہ_راست,

    GGML_SYCL_MEM_TYPE_گنتی,
};

منطقی ggml_sycl_memtrace_enabled();

خالی ggml_sycl_memtrace_جوڑو(ggml_sycl_mem_type type, ثابت خالی * ptr, size_t بائٹیں);
خالی ggml_sycl_memtrace_del(ثابت خالی * ptr);

خالی ggml_sycl_memtrace_report(ثابت حرف * لیبل);
خالی ggml_sycl_memtrace_report_device(ثابت حرف * لیبل, صحیح device, size_t dev_free, size_t dev_کل);
خالی ggml_sycl_memtrace_fail(ggml_sycl_mem_type type, size_t بائٹیں);

#اگر_ختم  // GGML_SYCL_MEMTRACE_HPP
