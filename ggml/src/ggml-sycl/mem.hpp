#ifndef GGML_SYCL_MEM_HPP
#define GGML_SYCL_MEM_HPP

#شامل <sycl/sycl.hpp>

شمارش MemoryAPIType {
    MEMORY_API_TYPE_LEVEL_ZERO = 0,
    MEMORY_API_TYPE_SYCL = 1,
};

ثابت نویسه* mem_api_int2str(صحیح mem_api);

منطقی get_memory_size(sycl::device dev, size_t & free_bytes, size_t & total_bytes,
    MemoryAPIType api_type);

#endif  // GGML_SYCL_MEM_HPP
