#ifndef GGML_SYCL_MEM_HPP
#define GGML_SYCL_MEM_HPP

#ಅನ್ತರ್ಭಾವ <sycl/sycl.hpp>

ಗಣನಾ MemoryAPIType {
    MEMORY_API_TYPE_LEVEL_ZERO = 0,
    MEMORY_API_TYPE_SYCL = 1,
};

ಧ್ರುವ ವರ್ಣ* mem_api_int2str(ಪೂರ್ಣಾಙ್ಕ mem_api);

ತಾರ್ಕಿಕ get_memory_size(sycl::device dev, size_t & free_bytes, size_t & total_bytes,
    MemoryAPIType api_type);

#endif  // GGML_SYCL_MEM_HPP
