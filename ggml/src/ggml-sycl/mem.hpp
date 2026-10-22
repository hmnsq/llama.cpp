#که_نه_تعریف_شوی GGML_SYCL_MEM_HPP
#تعریف_کړه GGML_SYCL_MEM_HPP

#ضمن <sycl/sycl.hpp>

تعداد حافظه_APIType {
    حافظه_API_TYPE_LEVEL_ZERO = 0,
    حافظه_API_TYPE_SYCL = 1,
};

ثابت حرف* mem_api_int2str(صحيح mem_api);

منطقي get_حافظه_اندازه(sycl::device dev, size_t & free_بایټونه, size_t & ټول_بایټونه,
    حافظه_APIType api_type);

#د_که_پای  // GGML_SYCL_MEM_HPP
