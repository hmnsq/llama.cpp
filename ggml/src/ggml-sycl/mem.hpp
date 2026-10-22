#अगर_अघोषित GGML_SYCL_MEM_HPP
#घोषणा GGML_SYCL_MEM_HPP

#समावेश <sycl/sycl.hpp>

क्रमागत MemoryAPIType {
    MEMORY_API_TYPE_LEVEL_ZERO = 0,
    MEMORY_API_TYPE_SYCL = 1,
};

स्थिर अक्षर* mem_api_int2str(पूर्णांक mem_api);

bool get_memory_आकार(sycl::device dev, माप_प्रकार & free_bytes, माप_प्रकार & योग_bytes,
    MemoryAPIType api_type);

#पूर्ण_अगर  // GGML_SYCL_MEM_HPP
