#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_MEM_HPP
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SYCL_MEM_HPP

#ਅਨ੍ਤਰ੍ਭਾਵ <sycl/sycl.hpp>

ਗਿਣਤੀ ਸਮ੍ਰਿਤੀ_APIType {
    ਸਮ੍ਰਿਤੀ_API_TYPE_LEVEL_ZERO = 0,
    ਸਮ੍ਰਿਤੀ_API_TYPE_SYCL = 1,
};

ਅਚਲ ਅੱਖਰ* mem_api_int2str(ਪੂਰਨ mem_api);

ਤਾਰਕਿਕ get_ਸਮ੍ਰਿਤੀ_ਮਾਪ(sycl::device dev, size_t & free_ਬਾਈਟਾਂ, size_t & ਕੁੱਲ_ਬਾਈਟਾਂ,
    ਸਮ੍ਰਿਤੀ_APIType api_type);

#ਜੇ_ਅੰਤ  // GGML_SYCL_MEM_HPP
