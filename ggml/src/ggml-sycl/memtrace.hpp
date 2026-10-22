#నిర్వచితం_కాకపోతే GGML_SYCL_MEMTRACE_HPP
#నిర్వచించు GGML_SYCL_MEMTRACE_HPP

#అన్తర్భావ <cstddef>

#నిర్వచించు GGML_SYCL_MEMTRACE_ట్యాగ్ "[SYCL-MEMTRACE]"

గణన ggml_sycl_mem_type {
    GGML_SYCL_MEM_బఫరు = 0,
    GGML_SYCL_MEM_POOL_LEG,
    GGML_SYCL_MEM_POOL_VMM,
    GGML_SYCL_MEM_ASYNC,
    GGML_SYCL_MEM_ప్రత్యక్ష,

    GGML_SYCL_MEM_TYPE_లెక్క,
};

తార్కిక ggml_sycl_memtrace_enabled();

శూన్య ggml_sycl_memtrace_కూడు(ggml_sycl_mem_type type, స్థిరరాశి శూన్య * ptr, size_t బైట్లు);
శూన్య ggml_sycl_memtrace_del(స్థిరరాశి శూన్య * ptr);

శూన్య ggml_sycl_memtrace_report(స్థిరరాశి అక్షరం * ట్యాగ్);
శూన్య ggml_sycl_memtrace_report_device(స్థిరరాశి అక్షరం * ట్యాగ్, పూర్ణాంకం device, size_t dev_free, size_t dev_మొత్తం);
శూన్య ggml_sycl_memtrace_fail(ggml_sycl_mem_type type, size_t బైట్లు);

#అయితే_ముగింపు  // GGML_SYCL_MEMTRACE_HPP
