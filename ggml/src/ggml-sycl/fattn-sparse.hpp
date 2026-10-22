#اگر_غیر_تعریف_شدہ GGML_SYCL_FATTN_SPARSE_HPP
#تعریف_کرو GGML_SYCL_FATTN_SPARSE_HPP

#شامل "common.hpp"

// Gather the K/V rows selected by a sparse mask and re-dispatch the dense
// kernels onto them. Returns false if the caller should use the dense path.
منطقی ggml_sycl_flash_attn_ext_sparse(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#اگر_ختم // GGML_SYCL_FATTN_SPARSE_HPP
