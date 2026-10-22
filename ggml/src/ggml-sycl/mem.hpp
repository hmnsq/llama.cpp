#جيڪڏهن_وصف_نه_ٿيل GGML_SYCL_MEM_HPP
#وصف_ڪريو GGML_SYCL_MEM_HPP

#ضمن <sycl/sycl.hpp>

تعداد يادگيري_APIType {
    يادگيري_API_TYPE_LEVEL_ZERO = 0,
    يادگيري_API_TYPE_SYCL = 1,
};

ثابت حرف* mem_api_int2str(صحيح mem_api);

منطقي get_يادگيري_سائيز(sycl::device dev, size_t & free_بائيٽون, size_t & ڪل_بائيٽون,
    يادگيري_APIType api_type);

#جيڪڏهن_پورو  // GGML_SYCL_MEM_HPP
