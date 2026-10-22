#నిర్వచితం_కాకపోతే GGML_SYCL_MEM_HPP
#నిర్వచించు GGML_SYCL_MEM_HPP

#అన్తర్భావ <sycl/sycl.hpp>

గణన జ్ఞాపకం_APIType {
    జ్ఞాపకం_API_TYPE_LEVEL_ZERO = 0,
    జ్ఞాపకం_API_TYPE_SYCL = 1,
};

స్థిరరాశి అక్షరం* mem_api_int2str(పూర్ణాంకం mem_api);

తార్కిక get_జ్ఞాపకం_కొలత(sycl::device dev, size_t & free_బైట్లు, size_t & మొత్తం_బైట్లు,
    జ్ఞాపకం_APIType api_type);

#అయితే_ముగింపు  // GGML_SYCL_MEM_HPP
