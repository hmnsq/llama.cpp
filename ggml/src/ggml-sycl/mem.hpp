#ifndef GGML_SYCL_MEM_HPP
#define GGML_SYCL_MEM_HPP

#அந்தர்பாவ <sycl/sycl.hpp>

எண்ணிக்கை MemoryAPIType {
    MEMORY_API_TYPE_LEVEL_ZERO = 0,
    MEMORY_API_TYPE_SYCL = 1,
};

மாறிலி எழுத்து* mem_api_int2str(முழு mem_api);

தருக்க get_memory_size(sycl::device dev, size_t & free_bytes, size_t & total_bytes,
    MemoryAPIType api_type);

#endif  // GGML_SYCL_MEM_HPP
