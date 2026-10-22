#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_SYCL_MEM_HPP
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SYCL_MEM_HPP

#ಅನ್ತರ್ಭಾವ <sycl/sycl.hpp>

ಎಣಿಕೆ ಸ್ಮರಣೆ_APIType {
    ಸ್ಮರಣೆ_API_TYPE_LEVEL_ZERO = 0,
    ಸ್ಮರಣೆ_API_TYPE_SYCL = 1,
};

ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* mem_api_int2str(ಪೂರ್ಣಾಂಕ mem_api);

ತಾರ್ಕಿಕ get_ಸ್ಮರಣೆ_ಗಾತ್ರ(sycl::device dev, size_t & free_ಬೈಟುಗಳು, size_t & ಒಟ್ಟು_ಬೈಟುಗಳು,
    ಸ್ಮರಣೆ_APIType api_type);

#ಆದರೆ_ಅಂತ್ಯ  // GGML_SYCL_MEM_HPP
