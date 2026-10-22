#यदि_अपरिभाषितम् GGML_SYCL_FATTN_SPARSE_HPP
#परिभाषय GGML_SYCL_FATTN_SPARSE_HPP

#अन्तर्भाव "common.hpp"

// Gather the K/V rows selected by a sparse mask and re-dispatch the dense
// kernels onto them. Returns false if the caller should use the dense path.
तार्किक ggml_sycl_flash_attn_ext_sparse(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#यदि_अन्तः // GGML_SYCL_FATTN_SPARSE_HPP
