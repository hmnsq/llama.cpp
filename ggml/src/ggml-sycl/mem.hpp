#ifndef GGML_SYCL_MEM_HPP
#define GGML_SYCL_MEM_HPP

#অন্তর্ভাব <sycl/sycl.hpp>

গণনা MemoryAPIType {
    MEMORY_API_TYPE_LEVEL_ZERO = 0,
    MEMORY_API_TYPE_SYCL = 1,
};

ধ্রুবক অক্ষর* mem_api_int2str(পূর্ণসংখ্যা mem_api);

যৌক্তিক get_memory_আকার(sycl::device dev, size_t & free_bytes, size_t & যোগ_bytes,
    MemoryAPIType api_type);

#endif  // GGML_SYCL_MEM_HPP
