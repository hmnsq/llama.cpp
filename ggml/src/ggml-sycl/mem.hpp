#यदि_अपरिभाषितम् GGML_SYCL_MEM_HPP
#परिभाषय GGML_SYCL_MEM_HPP

#अन्तर्भाव <sycl/sycl.hpp>

गणना स्मृतिः_APIType {
    स्मृतिः_API_TYPE_LEVEL_ZERO = 0,
    स्मृतिः_API_TYPE_SYCL = 1,
};

ध्रुव वर्ण* mem_api_int2str(पूर्णाङ्क mem_api);

तार्किक get_स्मृतिः_आकार(sycl::device dev, size_t & free_अष्टकानि, size_t & योग_अष्टकानि,
    स्मृतिः_APIType api_type);

#यदि_अन्तः  // GGML_SYCL_MEM_HPP
