#اگر_غیر_تعریف_شدہ GGML_SYCL_MEM_HPP
#تعریف_کرو GGML_SYCL_MEM_HPP

#شامل <sycl/sycl.hpp>

شمار حافظہ_APIType {
    حافظہ_API_TYPE_LEVEL_ZERO = 0,
    حافظہ_API_TYPE_SYCL = 1,
};

ثابت حرف* mem_api_int2str(صحیح mem_api);

منطقی get_حافظہ_سائز(sycl::device dev, size_t & free_بائٹیں, size_t & کل_بائٹیں,
    حافظہ_APIType api_type);

#اگر_ختم  // GGML_SYCL_MEM_HPP
