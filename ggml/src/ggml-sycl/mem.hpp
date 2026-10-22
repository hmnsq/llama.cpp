#ifndef GGML_SYCL_MEM_HPP
#define GGML_SYCL_MEM_HPP

#ଅନ୍ତର୍ଭାଵ <sycl/sycl.hpp>

ଗଣନା MemoryAPIType {
    MEMORY_API_TYPE_LEVEL_ZERO = 0,
    MEMORY_API_TYPE_SYCL = 1,
};

ଅଚଳ ଅକ୍ଷର* mem_api_int2str(ପୂର୍ଣ୍ଣସଂଖ୍ୟା mem_api);

ତାର୍କିକ get_memory_size(sycl::device dev, size_t & free_bytes, size_t & total_bytes,
    MemoryAPIType api_type);

#endif  // GGML_SYCL_MEM_HPP
