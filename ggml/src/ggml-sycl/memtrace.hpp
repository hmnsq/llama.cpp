#यदि_अपरिभाषितम् GGML_SYCL_MEMTRACE_HPP
#परिभाषय GGML_SYCL_MEMTRACE_HPP

#अन्तर्भाव <cstddef>

#परिभाषय GGML_SYCL_MEMTRACE_चिह्नकम् "[SYCL-MEMTRACE]"

गणना ggml_sycl_mem_type {
    GGML_SYCL_MEM_बफर = 0,
    GGML_SYCL_MEM_POOL_LEG,
    GGML_SYCL_MEM_POOL_VMM,
    GGML_SYCL_MEM_ASYNC,
    GGML_SYCL_MEM_प्रत्यक्षम्,

    GGML_SYCL_MEM_TYPE_गणनम्,
};

तार्किक ggml_sycl_memtrace_enabled();

शून्य ggml_sycl_memtrace_सङ्कलय(ggml_sycl_mem_type type, ध्रुव शून्य * ptr, size_t अष्टकानि);
शून्य ggml_sycl_memtrace_del(ध्रुव शून्य * ptr);

शून्य ggml_sycl_memtrace_report(ध्रुव वर्ण * चिह्नकम्);
शून्य ggml_sycl_memtrace_report_device(ध्रुव वर्ण * चिह्नकम्, पूर्णाङ्क device, size_t dev_free, size_t dev_योग);
शून्य ggml_sycl_memtrace_fail(ggml_sycl_mem_type type, size_t अष्टकानि);

#यदि_अन्तः  // GGML_SYCL_MEMTRACE_HPP
