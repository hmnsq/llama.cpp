#ifndef GGML_SYCL_MEM_HPP
#define GGML_SYCL_MEM_HPP

#അന്തര്ഭാവ <sycl/sycl.hpp>

എണ്ണൽ MemoryAPIType {
    MEMORY_API_TYPE_LEVEL_ZERO = 0,
    MEMORY_API_TYPE_SYCL = 1,
};

സ്ഥിരം അക്ഷരം* mem_api_int2str(പൂർണ്ണസംഖ്യ mem_api);

തർക്കം get_memory_size(sycl::device dev, size_t & free_bytes, size_t & total_bytes,
    MemoryAPIType api_type);

#endif  // GGML_SYCL_MEM_HPP
