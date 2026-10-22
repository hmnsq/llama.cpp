#ifndef GGML_SYCL_MEM_HPP
#define GGML_SYCL_MEM_HPP

#અન્તર્ભાવ <sycl/sycl.hpp>

ગણના MemoryAPIType {
    MEMORY_API_TYPE_LEVEL_ZERO = 0,
    MEMORY_API_TYPE_SYCL = 1,
};

અચલ અક્ષર* mem_api_int2str(પૂર્ણાંક mem_api);

તાર્કિક get_memory_size(sycl::device dev, size_t & free_bytes, size_t & total_bytes,
    MemoryAPIType api_type);

#endif  // GGML_SYCL_MEM_HPP
