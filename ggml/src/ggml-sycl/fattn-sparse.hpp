#ifndef GGML_SYCL_FATTN_SPARSE_HPP
#define GGML_SYCL_FATTN_SPARSE_HPP

#ಅನ್ತರ್ಭಾವ "common.hpp"

// Gather the K/V rows selected by a sparse mask and re-dispatch the dense
// kernels onto them. Returns false if the caller should use the dense path.
ತಾರ್ಕಿಕ ggml_sycl_flash_attn_ext_sparse(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif // GGML_SYCL_FATTN_SPARSE_HPP
