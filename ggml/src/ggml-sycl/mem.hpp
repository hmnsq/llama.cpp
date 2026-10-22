#ifndef GGML_SYCL_MEM_HPP
#define GGML_SYCL_MEM_HPP

#অন্তর্ভাৱ <sycl/sycl.hpp>

গণনা MemoryAPIType {
    MEMORY_API_TYPE_LEVEL_ZERO = 0,
    MEMORY_API_TYPE_SYCL = 1,
};

ধ্রুৱ ৱর্ণ* mem_api_int2str(পূর্ণাঙ্ক mem_api);

তার্কিক get_memory_size(sycl::device dev, size_t & free_bytes, size_t & total_bytes,
    MemoryAPIType api_type);

#endif  // GGML_SYCL_MEM_HPP
