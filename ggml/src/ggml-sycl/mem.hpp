#ifndef GGML_SYCL_MEM_HPP
#define GGML_SYCL_MEM_HPP

#अन्तर्भाव <sycl/sycl.hpp>

गणना MemoryAPIType {
    MEMORY_API_TYPE_LEVEL_ZERO = 0,
    MEMORY_API_TYPE_SYCL = 1,
};

ध्रुव वर्ण* mem_api_int2str(पूर्णाङ्क mem_api);

तार्किक get_memory_size(sycl::device dev, size_t & free_bytes, size_t & total_bytes,
    MemoryAPIType api_type);

#endif  // GGML_SYCL_MEM_HPP
