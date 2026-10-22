#ifndef GGML_SYCL_MEM_HPP
#define GGML_SYCL_MEM_HPP

#ضمن <sycl/sycl.hpp>

تعداد MemoryAPIType {
    MEMORY_API_TYPE_LEVEL_ZERO = 0,
    MEMORY_API_TYPE_SYCL = 1,
};

ثابت حرف* mem_api_int2str(صحيح mem_api);

منطقي get_memory_size(sycl::device dev, size_t & free_bytes, size_t & total_bytes,
    MemoryAPIType api_type);

#endif  // GGML_SYCL_MEM_HPP
