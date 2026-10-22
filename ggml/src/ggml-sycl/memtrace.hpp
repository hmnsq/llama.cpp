#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_SYCL_MEMTRACE_HPP
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SYCL_MEMTRACE_HPP

#ಅನ್ತರ್ಭಾವ <cstddef>

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SYCL_MEMTRACE_ಟ್ಯಾಗ್ "[SYCL-MEMTRACE]"

ಎಣಿಕೆ ggml_sycl_mem_type {
    GGML_SYCL_MEM_ಬಫರ್ = 0,
    GGML_SYCL_MEM_POOL_LEG,
    GGML_SYCL_MEM_POOL_VMM,
    GGML_SYCL_MEM_ASYNC,
    GGML_SYCL_MEM_ನೇರ,

    GGML_SYCL_MEM_TYPE_ಎಣಿಕೆ,
};

ತಾರ್ಕಿಕ ggml_sycl_memtrace_enabled();

ಶೂನ್ಯ ggml_sycl_memtrace_ಕೂಡಿಸು(ggml_sycl_mem_type type, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ptr, size_t ಬೈಟುಗಳು);
ಶೂನ್ಯ ggml_sycl_memtrace_del(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ptr);

ಶೂನ್ಯ ggml_sycl_memtrace_report(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಟ್ಯಾಗ್);
ಶೂನ್ಯ ggml_sycl_memtrace_report_device(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಟ್ಯಾಗ್, ಪೂರ್ಣಾಂಕ device, size_t dev_free, size_t dev_ಒಟ್ಟು);
ಶೂನ್ಯ ggml_sycl_memtrace_fail(ggml_sycl_mem_type type, size_t ಬೈಟುಗಳು);

#ಆದರೆ_ಅಂತ್ಯ  // GGML_SYCL_MEMTRACE_HPP
