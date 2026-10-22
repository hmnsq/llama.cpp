#اگر_تعریف_نشده GGML_SYCL_MEM_HPP
#تعریف_کن GGML_SYCL_MEM_HPP

#ضمن <sycl/sycl.hpp>

تعداد حافظه_APIType {
    حافظه_API_TYPE_LEVEL_ZERO = 0,
    حافظه_API_TYPE_SYCL = 1,
};

ثابت حرف* mem_api_int2str(صحيح mem_api);

منطقي get_حافظه_اندازه(sycl::device dev, size_t & free_بایتها, size_t & مجموع_بایتها,
    حافظه_APIType api_type);

#پایان_اگر  // GGML_SYCL_MEM_HPP
