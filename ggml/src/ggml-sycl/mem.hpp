#جے_بنا_تعریف GGML_SYCL_MEM_HPP
#تعریف_کرو GGML_SYCL_MEM_HPP

#ضمن <sycl/sycl.hpp>

تعداد حافظہ_APIType {
    حافظہ_API_TYPE_LEVEL_ZERO = 0,
    حافظہ_API_TYPE_SYCL = 1,
};

ثابت حرف* mem_api_int2str(صحيح mem_api);

منطقي get_حافظہ_ناپ(sycl::device dev, size_t & free_بائٹاں, size_t & کل_بائٹاں,
    حافظہ_APIType api_type);

#جے_انت  // GGML_SYCL_MEM_HPP
