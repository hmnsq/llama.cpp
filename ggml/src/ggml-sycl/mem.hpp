#ifndef GGML_SYCL_MEM_HPP
#define GGML_SYCL_MEM_HPP

#include <sycl/sycl.hpp>

ga_nanaa MemoryAPIType {
    MEMORY_API_TYPE_LEVEL_ZERO = 0,
    MEMORY_API_TYPE_SYCL = 1,
};

const var_na* mem_api_int2str(පූර්ණ_සංඛ්‍යා mem_api);

bool get_memory_size(sycl::device dev, size_t & free_bytes, size_t & total_bytes,
    MemoryAPIType api_type);

#endif  // GGML_SYCL_MEM_HPP
